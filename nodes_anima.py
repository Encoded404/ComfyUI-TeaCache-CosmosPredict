"""TeaCache nodes for Anima/Cosmos-Predict2.

Two nodes:
  TeaCacheAnima — simple quality slider (0-100), auto-maps to threshold
                   via error-anchored control points from Pareto frontier
  TeaCacheAnimaAdvanced — all 10 knobs exposed, defaults from presets

Presets are loaded from anima_presets.json.  The new format (auto-generated
by build_presets.py) stores error-anchored control points with full configs.
The old format (quality_zones with hand-crafted control_points) is still
supported as a fallback.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

from .tuning.config_types import TeacacheConfig


# ── Rough heuristic: LPIPS ≈ accumulated_error × scale ────────────────
# Calibrated empirically from a few validation runs.  The exact LPIPS
# value shown in the node is a display hint — users calibrate to their
# own eyes.  Override in the preset file via _lpips_scale.
_DEFAULT_LPIPS_SCALE = 6.0


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


def _is_new_format(presets: dict) -> bool:
    """Return True if presets use the new error-anchored control_points format."""
    return "control_points" in presets and "quality_zones" not in presets


# ═══════════════════════════════════════════════════════════════════════════
#  Quality curve: maps the 0-100 slider to error space
# ═══════════════════════════════════════════════════════════════════════════

def _apply_quality_curve(quality: float, lo: float, hi: float, presets: dict) -> float:
    """Map quality (0-100) to a target accumulated_error using the configured curve.

    Supported curves:
      "linear":   uniform spacing (legacy default)
      "power":    polynomial compression toward low error (p > 1 → more
                  granularity at the high-quality end)
      "exponential": geometric spacing (proportional steps)

    Configured via _quality_curve and _quality_power in the presets JSON.
    """
    curve = presets.get("_quality_curve", "linear")
    q = quality / 100.0

    if curve == "exponential":
        ratio = hi / lo if lo > 0 else 1.0
        return lo * (ratio ** q)

    if curve == "power":
        p = presets.get("_quality_power", 2.0)
        return lo + (hi - lo) * (q ** p)

    # linear (default / backward-compatible)
    return lo + q * (hi - lo)


# ═══════════════════════════════════════════════════════════════════════════
#  New format: error-anchored control point interpolation
# ═══════════════════════════════════════════════════════════════════════════

def _quality_to_config(quality: float, steps: int = 30) -> Tuple[TeacacheConfig, float, float]:
    """Map quality 0-100 to a full TeacacheConfig via error-anchored control points.

    quality=0  → lowest error tolerance → best quality (near-lossless)
    quality=100 → highest error tolerance → most speed (aggressive)

    Returns (config, estimated_speedup, lpips_estimate).
    """
    presets = _load_presets()
    if not _is_new_format(presets):
        return _quality_to_config_legacy(quality)

    points = presets["control_points"]
    preset_steps = presets.get("_steps", steps)
    lpips_scale = presets.get("_lpips_scale", _DEFAULT_LPIPS_SCALE)
    error_range = presets.get("_error_range", [points[0]["error"], points[-1]["error"]])

    points.sort(key=lambda p: p["error"])
    lo_err, hi_err = error_range

    target_error = _apply_quality_curve(quality, lo_err, hi_err, presets)

    # Find bracketing control points
    if target_error <= points[0]["error"]:
        lo = hi = points[0]
        t = 0.0
    elif target_error >= points[-1]["error"]:
        lo = hi = points[-1]
        t = 1.0
    else:
        for i in range(len(points) - 1):
            if points[i]["error"] <= target_error <= points[i + 1]["error"]:
                lo, hi = points[i], points[i + 1]
                span = hi["error"] - lo["error"]
                t = (target_error - lo["error"]) / span if span > 0 else 0.0
                break
        else:
            lo = hi = points[-1]
            t = 1.0

    # Threshold: linear interpolation between bracketing points
    lo_thresh = lo["config"]["rel_l1_thresh"]
    hi_thresh = hi["config"]["rel_l1_thresh"]
    thresh = lo_thresh + t * (hi_thresh - lo_thresh)

    # Step-count scaling: fewer steps → larger per-step deltas → higher threshold
    if preset_steps > 0 and steps > 0 and steps != preset_steps:
        thresh *= preset_steps / steps

    # Discrete params: snap to nearest control point
    base_config = hi["config"] if t > 0.5 else lo["config"]
    cfg = TeacacheConfig.from_dict(base_config)
    cfg.rel_l1_thresh = round(max(thresh, 0.001), 4)

    # Display hints
    speedup = lo["speedup"] + t * (hi["speedup"] - lo["speedup"])
    lpips_est = target_error * lpips_scale

    return cfg, speedup, lpips_est


# ═══════════════════════════════════════════════════════════════════════════
#  Legacy format: quality_zones with hand-crafted control points
# ═══════════════════════════════════════════════════════════════════════════

def _quality_to_config_legacy(quality: float) -> Tuple[TeacacheConfig, float, float]:
    """Fallback for old anima_presets.json format (quality_zones)."""
    presets = _load_presets()
    zone = None
    for z in presets.get("quality_zones", []):
        lo, hi = z["quality_range"]
        if lo <= quality <= hi:
            zone = z
            break

    if zone is None:
        cfg = TeacacheConfig()
        return cfg, 1.0, 0.05

    points = sorted(zone["control_points"], key=lambda p: p["quality"])

    if quality <= points[0]["quality"]:
        p = points[0]
    elif quality >= points[-1]["quality"]:
        p = points[-1]
    else:
        for i in range(len(points) - 1):
            a, b = points[i], points[i + 1]
            if a["quality"] <= quality <= b["quality"]:
                t = (quality - a["quality"]) / (b["quality"] - a["quality"])
                thresh = a["thresh"] + t * (b["thresh"] - a["thresh"])
                speedup = a["speedup"] + t * (b["speedup"] - a["speedup"])
                lpips = a["lpips"] + t * (b["lpips"] - a["lpips"])
                p = {"thresh": thresh, "speedup": speedup, "lpips": lpips}
                break
        else:
            p = points[-1]

    defaults = presets.get("defaults", {})
    cfg_dict = dict(defaults)
    cfg_dict.update(zone.get("config_overrides", {}))
    cfg_dict["rel_l1_thresh"] = round(p["thresh"], 4)

    return TeacacheConfig.from_dict(cfg_dict), p["speedup"], p["lpips"]


# ═══════════════════════════════════════════════════════════════════════════
#  Apply helper (shared by both nodes)
# ═══════════════════════════════════════════════════════════════════════════

def _apply_teacache(model, cfg: TeacacheConfig):
    """Apply a TeaCache config as the model's _forward.

    Patches MiniTrainDIT._forward within a context manager, so the
    original forward is restored when TeaCache is not active.
    """
    from unittest.mock import patch
    from .tuning.forward import teacache_anima_forward

    new_model = model.clone()
    diffusion_model = new_model.get_model_object("diffusion_model")

    # Inject config into transformer_options
    to = new_model.model_options.setdefault("transformer_options", {})
    cfg.inject_into_transformer_options(to)

    # Context manager for _forward patching (restores original on exit)
    context = patch.object(
        diffusion_model,
        '_forward',
        teacache_anima_forward.__get__(diffusion_model, diffusion_model.__class__),
    )

    # Wrapper to track step index and enable/disable TeaCache
    def unet_wrapper(model_function, kwargs):
        input_x = kwargs["input"]
        timestep = kwargs["timestep"]
        c = kwargs["c"]
        c_to = c.setdefault("transformer_options", {})
        sigmas = c_to.get("sample_sigmas")

        teacache_enabled = False
        if sigmas is not None:
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
            teacache_enabled = (
                cfg.start_percent <= c_to["current_percent"] <= cfg.end_percent
            )
            c_to["enable_teacache"] = teacache_enabled

            # Reset state at start of each generation
            if step_idx == 0 and hasattr(diffusion_model, 'teacache_state'):
                delattr(diffusion_model, 'teacache_state')

        if teacache_enabled:
            with context:
                return model_function(input_x, timestep, **c)
        else:
            return model_function(input_x, timestep, **c)

    new_model.set_model_unet_function_wrapper(unet_wrapper)
    return new_model


# ═══════════════════════════════════════════════════════════════════════════
#  Node A: Simple quality slider
# ═══════════════════════════════════════════════════════════════════════════

class TeaCacheAnima:
    """Simple TeaCache node with a quality slider for Anima/Cosmos.

    Connects after Load Diffusion Model / Load LoRA, before KSampler.
    The Quality slider maps to accumulated error tolerance via
    auto-generated control points from the Pareto frontier:

      quality=0 → min error tolerance → near-lossless (LPIPS≈0.01)
      quality=50 → balanced (~1.3x speedup)
      quality=100 → max error tolerance → max speed (~2x speedup)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The Anima diffusion model from Load Diffusion Model."}),
                "steps": ("INT", {"default": 30, "min": 10, "max": 100, "step": 1,
                                    "tooltip": "Number of sampling steps in your KSampler. Threshold auto-scales."}),
                "quality": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0,
                                        "display": "slider",
                                        "tooltip": "0 = max quality (lossless), 50 = balanced, 100 = max speed"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_teacache_anima"
    CATEGORY = "TeaCache"

    def apply_teacache_anima(self, model, steps, quality):
        cfg, speedup, lpips_est = _quality_to_config(quality, steps=steps)

        new_model = _apply_teacache(model, cfg)

        info = (
            f"err≈{_quality_to_error(quality):.3f}  "
            f"thresh={cfg.rel_l1_thresh:.3f}  "
            f"est. {speedup:.1f}x  "
            f"LPIPS≈{lpips_est:.3f}  "
            f"src={cfg.source}"
        )

        if hasattr(new_model, "widget_values"):
            new_model.widget_values = info

        print(f"  TeaCacheAnima: quality={quality:.0f} steps={steps} → {info}")
        return (new_model,)


def _quality_to_error(quality: float) -> float:
    """Helper: map quality slider to target accumulated_error (for display)."""
    presets = _load_presets()
    if not _is_new_format(presets):
        # Legacy: approximate from first quality zone
        for z in presets.get("quality_zones", []):
            pts = sorted(z["control_points"], key=lambda p: p["quality"])
            if pts:
                lo = pts[0]
                hi = pts[-1]
                t = quality / 100.0
                lpips = lo["lpips"] + t * (hi["lpips"] - lo["lpips"])
                return lpips / _DEFAULT_LPIPS_SCALE
        return 0.05 * quality / 100.0

    points = presets["control_points"]
    error_range = presets.get("_error_range", [points[0]["error"], points[-1]["error"]])
    lo_err, hi_err = error_range
    return _apply_quality_curve(quality, lo_err, hi_err, presets)


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
                "steps": ("INT", {"default": 30, "min": 10, "max": 100, "step": 1}),
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
