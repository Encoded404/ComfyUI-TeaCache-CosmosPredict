"""TeaCache nodes for Anima/Cosmos-Predict2.

Two nodes:
  TeaCacheAnima — simple quality slider (0-100), auto-maps to threshold
  TeaCacheAnimaAdvanced — all 10 knobs exposed, defaults from presets

The presets are loaded from anima_presets.json in the same directory.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

from .tuning.config_types import TeacacheConfig


# ═══════════════════════════════════════════════════════════════════════════
#  Preset loading
# ═══════════════════════════════════════════════════════════════════════════

_PRESETS = None


def _load_presets() -> dict:
    global _PRESETS
    if _PRESETS is None:
        preset_path = Path(__file__).parent / "anima_presets.json"
        if preset_path.exists():
            with open(preset_path) as f:
                _PRESETS = json.load(f)
        else:
            _PRESETS = {}
    return _PRESETS


def _get_quality_zone(quality: float) -> dict:
    presets = _load_presets()
    for zone in presets.get("quality_zones", []):
        lo, hi = zone["quality_range"]
        if lo <= quality <= hi:
            return zone
    return None


def _quality_to_thresh(quality: float) -> Tuple[float, float, float, dict]:
    """Map quality 0-100 to (threshold, speedup, lpips, zone).

    Uses nearest-control-point interpolation from the quality zone.
    """
    zone = _get_quality_zone(quality)
    if zone is None:
        return 0.07, 1.14, 0.05, {}

    points = zone["control_points"]
    points.sort(key=lambda p: p["quality"])

    if quality <= points[0]["quality"]:
        p = points[0]
        return p["thresh"], p["speedup"], p["lpips"], zone
    if quality >= points[-1]["quality"]:
        p = points[-1]
        return p["thresh"], p["speedup"], p["lpips"], zone

    # Linear interpolation between nearest control points
    for i in range(len(points) - 1):
        lo, hi = points[i], points[i + 1]
        if lo["quality"] <= quality <= hi["quality"]:
            t = (quality - lo["quality"]) / (hi["quality"] - lo["quality"])
            thresh = lo["thresh"] + t * (hi["thresh"] - lo["thresh"])
            speedup = lo["speedup"] + t * (hi["speedup"] - lo["speedup"])
            lpips = lo["lpips"] + t * (hi["lpips"] - lo["lpips"])
            return thresh, speedup, lpips, zone

    # Fallback
    p = points[-1]
    return p["thresh"], p["speedup"], p["lpips"], zone


def _build_config(quality: float) -> TeacacheConfig:
    """Build a TeacacheConfig from presets at the given quality level."""
    presets = _load_presets()
    defaults = presets.get("defaults", {})
    thresh, speedup, lpips, zone = _quality_to_thresh(quality)

    # Start with defaults, then apply zone overrides
    cfg_dict = dict(defaults)
    cfg_dict.update(zone.get("config_overrides", {}))
    cfg_dict["rel_l1_thresh"] = round(thresh, 4)

    return TeacacheConfig.from_dict(cfg_dict)


# ═══════════════════════════════════════════════════════════════════════════
#  Apply helper (shared by both nodes)
# ═══════════════════════════════════════════════════════════════════════════

def _apply_teacache(model, cfg: TeacacheConfig):
    """Apply a TeaCache config as the model's _forward.

    Uses the same patching mechanism as nodes.py's TeaCache.apply_teacache,
    but targeted at the Anima/Cosmos MiniTrainDIT architecture.
    """
    import comfy.patcher_extension as patch
    from .tuning.forward import teacache_anima_forward

    new_model = model.clone()
    diffusion_model = new_model.get_model_object("diffusion_model")

    # Inject config into transformer_options
    to = new_model.model_options.setdefault("transformer_options", {})
    cfg.inject_into_transformer_options(to)

    # Patch the _forward method
    context = patch.multiple(
        diffusion_model,
        _forward=teacache_anima_forward.__get__(
            diffusion_model, diffusion_model.__class__
        ),
    )

    # Wrapper to track step index and enable/disable TeaCache
    def unet_wrapper(model_function, kwargs):
        input_x = kwargs["input"]
        timestep = kwargs["timestep"]
        c = kwargs["c"]
        c_to = c.setdefault("transformer_options", {})
        sigmas = c_to.get("sample_sigmas")

        if sigmas is not None:
            # Find current step index from sigma matching
            matched = (sigmas == timestep[0]).nonzero()
            if len(matched) > 0:
                step_idx = matched[0].item()
            else:
                step_idx = 0
                for i in range(len(sigmas) - 1):
                    if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                        step_idx = i
                        break
            c_to["current_percent"] = step_idx / max(len(sigmas) - 1, 1)
            c_to["enable_teacache"] = (
                cfg.start_percent <= c_to["current_percent"] <= cfg.end_percent
            )

        with context:
            return model_function(input_x, timestep, **c)

    new_model.set_model_unet_function_wrapper(unet_wrapper)
    return new_model


# ═══════════════════════════════════════════════════════════════════════════
#  Node A: Simple quality slider
# ═══════════════════════════════════════════════════════════════════════════

class TeaCacheAnima:
    """Simple TeaCache node with a quality slider for Anima/Cosmos.

    Connects after Load Diffusion Model / Load LoRA, before KSampler.
    The Quality slider controls the speed/quality tradeoff:
      0 = lossless (no visible change)
      50 = balanced (recommended, ~1.3x speedup)
      100 = maximum speed (~2x speedup, noticeable quality drop)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The Anima diffusion model from Load Diffusion Model."}),
                "steps": ("INT", {"default": 30, "min": 20, "max": 60, "step": 1,
                                   "tooltip": "Number of sampling steps in your KSampler."}),
                "quality": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0,
                                       "display": "slider",
                                       "tooltip": "0 = max quality (lossless), 50 = balanced, 100 = max speed"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_teacache_anima"
    CATEGORY = "TeaCache"

    def apply_teacache_anima(self, model, steps, quality):
        thresh, speedup, lpips, zone = _quality_to_thresh(quality)
        cfg = _build_config(quality)

        new_model = _apply_teacache(model, cfg)

        # Build info string for the node status
        label = zone.get("label", "Tuned config")
        info = (
            f"[{label}]  "
            f"thresh={thresh:.3f}  "
            f"est. {speedup:.1f}x  "
            f"LPIPS≈{lpips:.3f}"
        )

        # ComfyUI node info display
        if hasattr(new_model, "widget_values"):
            new_model.widget_values = info

        print(f"  TeaCacheAnima: quality={quality:.0f} → {info}")
        return (new_model,)


# ═══════════════════════════════════════════════════════════════════════════
#  Node B: Advanced — all knobs exposed
# ═══════════════════════════════════════════════════════════════════════════

class TeaCacheAnimaAdvanced:
    """Advanced TeaCache node with all tuning knobs exposed for Anima/Cosmos.

    Defaults are set to the winning config from the tuning pipeline
    (pooled_latent, carry_over, bell schedule).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The Anima diffusion model."}),
                "steps": ("INT", {"default": 30, "min": 20, "max": 60, "step": 1}),
                "source": (
                    ["t_emb", "first_block_shift", "pooled_latent"],
                    {"default": "pooled_latent",
                     "tooltip": "Which signal drives the cache decision."},
                ),
                "metric_type": (
                    ["mean_only", "mean_and_max", "mean_max_std", "weighted_sum"],
                    {"default": "mean_only",
                     "tooltip": "How to combine delta statistics into distance."},
                ),
                "signal_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1000.0, "step": 0.1,
                                             "tooltip": "Multiplier for numerical stability (t_emb needs 50-200)."}),
                "mapping_type": (
                    ["identity", "polynomial", "power_law", "softplus"],
                    {"default": "identity",
                     "tooltip": "How to map distance to predicted output change."},
                ),
                "rel_l1_thresh": ("FLOAT", {"default": 0.51, "min": 0.001, "max": 10.0, "step": 0.001,
                                              "tooltip": "Accumulation threshold. Higher = faster, lower = better quality."}),
                "accumulation": (
                    ["hard_reset", "carry_over", "leaky", "windowed"],
                    {"default": "carry_over",
                     "tooltip": "How per-step distance accumulates over time."},
                ),
                "step_schedule": (
                    ["constant", "cosine", "linear_ramp", "linear_decay", "bell"],
                    {"default": "bell",
                     "tooltip": "How threshold varies across sampling steps."},
                ),
                "residual": (
                    ["hard", "blended", "scaled"],
                    {"default": "hard",
                     "tooltip": "How cached residual is applied when skipping blocks."},
                ),
                "start_percent": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01,
                                              "tooltip": "Don't cache until this fraction of steps."}),
                "end_percent": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01,
                                            "tooltip": "Stop caching after this fraction of steps."}),
                "block_mode": (
                    ["all_or_nothing", "split_fraction", "split_groups"],
                    {"default": "all_or_nothing",
                     "tooltip": "Which blocks to cache (all_or_nothing is safest)."},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_teacache_anima_advanced"
    CATEGORY = "TeaCache"

    def apply_teacache_anima_advanced(
        self, model, steps, source, metric_type, signal_scale,
        mapping_type, rel_l1_thresh, accumulation, step_schedule,
        residual, start_percent, end_percent, block_mode,
    ):
        cfg = TeacacheConfig(
            source=source,
            metric_type=metric_type,
            signal_scale=signal_scale,
            mapping_type=mapping_type,
            coefficients=[],
            accumulation_type=accumulation,
            rel_l1_thresh=rel_l1_thresh,
            step_schedule=step_schedule,
            start_percent=start_percent,
            end_percent=end_percent,
            residual_strategy=residual,
            block_mode=block_mode,
        )
        new_model = _apply_teacache(model, cfg)
        print(f"  TeaCacheAnimaAdvanced: src={source} thresh={rel_l1_thresh} "
              f"acc={accumulation} schedule={step_schedule}")
        return (new_model,)
