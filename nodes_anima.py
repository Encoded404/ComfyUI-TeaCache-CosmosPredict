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

import torch

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

def _quality_to_config(quality: float) -> Tuple[TeacacheConfig, float, float, int]:
    """Map quality 0-100 to a full TeacacheConfig via error-anchored control points.

    quality=0  → lowest error tolerance → best quality (near-lossless)
    quality=100 → highest error tolerance → most speed (aggressive)

    Returns (config, estimated_speedup, lpips_estimate, preset_steps).
    """
    presets = _load_presets()
    if not _is_new_format(presets):
        return _quality_to_config_legacy(quality)

    points = presets["control_points"]
    preset_steps = presets.get("_steps", 30)
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

    # Threshold: two-phase midpoint-aware interpolation.
    #
    # When consecutive control points use the same signal config the
    # interpolation is a simple lerp between their base thresholds.
    # When they use *different* configs (e.g. crossing from pooled_latent
    # to t_emb), the threshold is interpolated separately on each side of
    # the midpoint, using that side's pre-computed midpoint threshold.
    # At the midpoint the config snaps from one source to the other.
    mid_data_available = (
        lo.get("high_mid") is not None and hi.get("low_mid") is not None
    )

    if mid_data_available:
        mid_err = (lo["error"] + hi["error"]) / 2

        if target_error <= mid_err:
            # Lower half: lo's config, lerp between lo base and lo high_mid
            base = lo
            lo_thresh = base["config"]["rel_l1_thresh"]
            lo_mid = base["high_mid"]
            seg = lo_mid["error"] - base["error"]
            t_seg = (target_error - base["error"]) / seg if seg > 0 else 0
            t_seg = max(0.0, min(1.0, t_seg))
            thresh = lo_thresh + t_seg * (lo_mid["thresh"] - lo_thresh)
            speedup = base["speedup"] + t_seg * (lo_mid["speedup"] - base["speedup"])
        else:
            # Upper half: hi's config, lerp between hi low_mid and hi base
            base = hi
            hi_thresh = base["config"]["rel_l1_thresh"]
            hi_mid = base["low_mid"]
            seg = base["error"] - hi_mid["error"]
            t_seg = (target_error - hi_mid["error"]) / seg if seg > 0 else 0
            t_seg = max(0.0, min(1.0, t_seg))
            thresh = hi_mid["thresh"] + t_seg * (hi_thresh - hi_mid["thresh"])
            speedup = hi_mid["speedup"] + t_seg * (base["speedup"] - hi_mid["speedup"])

        base_config = base["config"]
        cfg = TeacacheConfig.from_dict(base_config)
        cfg.rel_l1_thresh = round(max(thresh, 0.001), 4)
        lpips_est = target_error * lpips_scale

    else:
        # Legacy / presets without midpoint data — snap to nearest anchor
        base_config = hi["config"] if t > 0.5 else lo["config"]
        thresh = base_config.get("rel_l1_thresh", 0.07)
        cfg = TeacacheConfig.from_dict(base_config)
        cfg.rel_l1_thresh = round(max(thresh, 0.001), 4)
        nearest = hi if t > 0.5 else lo
        speedup = nearest["speedup"]
        lpips_est = target_error * lpips_scale

    return cfg, speedup, lpips_est, preset_steps


# ═══════════════════════════════════════════════════════════════════════════
#  Legacy format: quality_zones with hand-crafted control points
# ═══════════════════════════════════════════════════════════════════════════

def _quality_to_config_legacy(quality: float) -> Tuple[TeacacheConfig, float, float, int]:
    """Fallback for old anima_presets.json format (quality_zones)."""
    presets = _load_presets()
    preset_steps = presets.get("_steps", 30)
    zone = None
    for z in presets.get("quality_zones", []):
        lo, hi = z["quality_range"]
        if lo <= quality <= hi:
            zone = z
            break

    if zone is None:
        cfg = TeacacheConfig()
        return cfg, 1.0, 0.05, preset_steps

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

    return TeacacheConfig.from_dict(cfg_dict), p["speedup"], p["lpips"], preset_steps


# ═══════════════════════════════════════════════════════════════════════════
#  Apply helper (shared by both nodes)
# ═══════════════════════════════════════════════════════════════════════════

def _apply_teacache(model, cfg: TeacacheConfig, preset_steps: int = 30):
    """Apply a TeaCache config as the model's _forward.

    Patches MiniTrainDIT._forward within a context manager, so the
    original forward is restored when TeaCache is not active.
    """
    from unittest.mock import patch
    from .tuning.forward import teacache_anima_forward, step_schedule_multiplier

    new_model = model.clone()
    diffusion_model = new_model.get_model_object("diffusion_model")

    # Pre-acquire cross-attention dimension for context=None normalization
    _crossattn_dim = getattr(
        diffusion_model.blocks[0].cross_attn, "context_dim",
        getattr(diffusion_model, "crossattn_emb_channels", 1024),
    )

    # Inject config + step baseline into transformer_options
    to = new_model.model_options.setdefault("transformer_options", {})
    cfg.inject_into_transformer_options(to)
    to["preset_steps"] = preset_steps
    print(f"  [TeaCache Debug] _apply_teacache: tc_residual_strategy={to.get('tc_residual_strategy')!r}  tc_residual_params={to.get('tc_residual_params')!r}")

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

        # ── Fix 1: Normalize context=None → zero-tensor for torch.compile ──
        # The outer MiniTrainDIT.forward has context: torch.Tensor (non-optional).
        # When CFG unconditional passes None, dynamo recompiles every call.
        # A zero-token tensor preserves the None → self-attention fallback in
        # Attention.compute_qkv because 0-token k/v produce zero output.
        if c.get("c_crossattn") is None:
            c["c_crossattn"] = input_x.new_zeros(1, 0, _crossattn_dim)

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
            step_frac = step_idx / max(len(sigmas) - 1, 1)

            # ── Fix 2: Precompute step-schedule multiplier as 0-d tensor ──
            # Passing raw floats through transformer_options causes dynamo
            # to specialize on each distinct value → recompilation every step.
            # A 0-d tensor has stable shape/dtype so dynamo doesn't recompile.
            c_to["tc_current_percent"] = torch.tensor(step_frac)
            c_to["tc_threshold_mult"] = torch.tensor(
                step_schedule_multiplier(step_frac, cfg.step_schedule)
            )
            teacache_enabled = (
                cfg.start_percent <= step_frac <= cfg.end_percent
            )
            c_to["enable_teacache"] = teacache_enabled

            # Reset state at start of each generation
            if step_idx == 0 and hasattr(diffusion_model, 'teacache_state'):
                delattr(diffusion_model, 'teacache_state')

        # Always apply the _forward patch so torch.compile sees a stable
        # function identity.  teacache_anima_forward handles the
        # enable_teacache=False case efficiently by running all blocks.
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
    The Quality slider maps to accumulated error tolerance via
    auto-generated control points from the Pareto frontier:

      quality=0 → min error tolerance → near-lossless (LPIPS≈0.01)
      quality=50 → balanced (~1.3x speedup)
      quality=100 → max error tolerance → max speed (~2x speedup)

    A collapsible override section (toggle in the node) reveals optional
    parameter overrides.  When a dropdown's value changes, relevant sub-
    widgets appear automatically (e.g. residual_blend when residual_strategy
    is "blended").  The override state survives workflow save and reload.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The Anima diffusion model from Load Diffusion Model."}),
                "quality": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0,
                                        "display": "slider",
                                        "tooltip": "0 = max quality (lossless), 50 = balanced, 100 = max speed"}),
            },
            "optional": {
                "residual_strategy": (
                    ["auto", "hard", "blended", "scaled"],
                    {"default": "auto",
                     "tooltip": "Override preset's residual strategy."},
                ),
                "residual_blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                                "tooltip": "Blend factor for 'blended' strategy (unused; actual blend = 1 - confidence)."}),
                "residual_scale": ("FLOAT", {"default": 0.8, "min": 0.01, "max": 1.0, "step": 0.01,
                                                "tooltip": "Scale factor for 'scaled' strategy."}),
                "block_mode": (
                    ["auto", "all_or_nothing", "split_fraction", "split_groups"],
                    {"default": "auto",
                     "tooltip": "Override preset's block mode."},
                ),
                "accumulation_type": (
                    ["auto", "hard_reset", "carry_over", "leaky", "windowed"],
                    {"default": "auto",
                     "tooltip": "Override preset's accumulation type."},
                ),
                "step_schedule": (
                    ["auto", "constant", "cosine", "linear_ramp", "linear_decay", "bell"],
                    {"default": "auto",
                     "tooltip": "Override preset's step schedule."},
                ),
                "leak_factor": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 0.999, "step": 0.001,
                                             "tooltip": "Leak decay factor for 'leaky' accumulation."}),
                "window_size": ("INT", {"default": 5, "min": 2, "max": 50, "step": 1,
                                           "tooltip": "Window size for 'windowed' accumulation."}),
                "always_fraction": ("FLOAT", {"default": 0.36, "min": 0.01, "max": 0.99, "step": 0.01,
                                                 "tooltip": "Fraction of always-run blocks for 'split_fraction' mode."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_teacache_anima"
    CATEGORY = "TeaCache"

    def apply_teacache_anima(self, model, quality, **kwargs):
        cfg, speedup, lpips_est, preset_steps = _quality_to_config(quality)

        overrides = []
        if kwargs.get("residual_strategy", "auto") != "auto":
            cfg.residual_strategy = kwargs["residual_strategy"]
            overrides.append(f"res={kwargs['residual_strategy']}")
        if kwargs.get("block_mode", "auto") != "auto":
            cfg.block_mode = kwargs["block_mode"]
            overrides.append(f"blk={kwargs['block_mode']}")
        if kwargs.get("accumulation_type", "auto") != "auto":
            cfg.accumulation_type = kwargs["accumulation_type"]
            overrides.append(f"acc={kwargs['accumulation_type']}")
        if kwargs.get("step_schedule", "auto") != "auto":
            cfg.step_schedule = kwargs["step_schedule"]
            overrides.append(f"sched={kwargs['step_schedule']}")

        # Conditional sub-parameters
        if "residual_blend" in kwargs:
            cfg.residual_params = {"scale": kwargs["residual_blend"]}
            overrides.append(f"blend={kwargs['residual_blend']}")
        if "residual_scale" in kwargs:
            cfg.residual_params = {"scale": kwargs["residual_scale"]}
            overrides.append(f"rscale={kwargs['residual_scale']}")
        if "leak_factor" in kwargs:
            cfg.accumulation_params = {"leak_factor": kwargs["leak_factor"]}
            overrides.append(f"leak={kwargs['leak_factor']}")
        if "window_size" in kwargs:
            cfg.accumulation_params = {"window_size": int(kwargs["window_size"])}
            overrides.append(f"win={kwargs['window_size']}")
        if "always_fraction" in kwargs:
            cfg.block_params["always_fraction"] = kwargs["always_fraction"]
            overrides.append(f"frac={kwargs['always_fraction']}")

        print(f"  [TeaCache Debug] apply_teacache_anima: residual_strategy={cfg.residual_strategy!r}  residual_params={cfg.residual_params}  rel_l1_thresh={cfg.rel_l1_thresh}  source={cfg.source!r}")

        new_model = _apply_teacache(model, cfg, preset_steps=preset_steps)

        info = (
            f"err≈{_quality_to_error(quality):.3f}  "
            f"thresh={cfg.rel_l1_thresh:.3f}  "
            f"est. {speedup:.1f}x  "
            f"LPIPS≈{lpips_est:.3f}  "
            f"src={cfg.source}"
        )
        if overrides:
            info += "  {" + ", ".join(overrides) + "}"

        if hasattr(new_model, "widget_values"):
            new_model.widget_values = info

        print(f"  TeaCacheAnima: quality={quality:.0f} → {info}")
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
        self, model, source, metric_type, signal_scale,
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
