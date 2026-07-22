#!/usr/bin/env python3
"""Build anima_presets.json from the Pareto frontier produced by Phase 2.

Reads pareto_frontier.json (output of tuning/optimize.py) and writes an
error-anchored anima_presets.json for the TeaCacheAnima quality slider.

Each control point stores a full TeacacheConfig — different points on the
slider can use entirely different sources, accumulation strategies, and
step schedules.  The slider interpolates threshold linearly between
bracketing points and snaps discrete params to the nearest anchor.

The error range (accumulated out_rel) is publisher-controlled:
  error_min → quality=0  ("max quality", near-lossless)
  error_max → quality=100 ("max speed", aggressive caching)

LPIPS display in the node is a rough heuristic:  lpips ≈ error × 6.

Usage:
    python -m tuning.build_presets --pareto outputs/optimization/pareto_frontier.json
    python -m tuning.build_presets --pareto pareto.json --points 12 \\
        --error-min 0.005 --error-max 0.12 --steps 30
    python -m tuning.build_presets --pareto pareto.json \\
        --error-min 0.02 --error-max 0.12 \\
        --quality-curve power --quality-power 2.0
"""

import argparse
import json
from pathlib import Path


def _make_control_point(p):
    """Build a control point dict from a Pareto frontier entry."""
    cfg = p["config"]
    slim_config = {
        "source": cfg.get("source", "pooled_latent"),
        "metric_type": cfg.get("metric_type", "mean_only"),
        "metric_weights": cfg.get("metric_weights", {}),
        "signal_scale": cfg.get("signal_scale", 1.0),
        "mapping_type": cfg.get("mapping_type", "identity"),
        "coefficients": cfg.get("coefficients", []),
        "mapping_params": cfg.get("mapping_params", {}),
        "accumulation_type": cfg.get("accumulation_type", "hard_reset"),
        "accumulation_params": cfg.get("accumulation_params", {}),
        "rel_l1_thresh": cfg.get("rel_l1_thresh", 0.07),
        "step_schedule": cfg.get("step_schedule", "constant"),
        "start_percent": cfg.get("start_percent", 0.05),
        "end_percent": cfg.get("end_percent", 0.95),
        "block_mode": cfg.get("block_mode", "all_or_nothing"),
        "block_params": cfg.get("block_params", {}),
        "residual_strategy": cfg.get("residual_strategy", "hard"),
        "residual_params": cfg.get("residual_params", {}),
        "cross_feed_enabled": cfg.get("cross_feed_enabled", False),
        "cross_feed_strength": cfg.get("cross_feed_strength", 0.5),
        "cosim_threshold": cfg.get("cosim_threshold", 0.95),
        "block_level": cfg.get("block_level", "unified"),
        "block_level_config_scope": cfg.get("block_level_config_scope", ["*"]),
    }
    return {
        "error": round(p["accumulated_error"], 6),
        "config": slim_config,
        "speedup": round(p["estimated_speedup"], 3),
    }


def _collect_all_in_range(pareto, lo, hi):
    """Return a control point for every Pareto point within [lo, hi]."""
    return [_make_control_point(p) for p in pareto if lo <= p["accumulated_error"] <= hi]


def _sample_control_points(pareto, lo, hi, num_points,
                           quality_curve="linear", quality_power=2.0):
    """Sample *num_points* control points using 2-nearest-neighbor with greedy resolution.

    For each target error (evenly spaced in normalized-quality space), finds the
    top 2 closest Pareto points.  Candidates sorted by proximity (best-justified
    first) then greedily assigned: first choice if free, second choice if taken,
    forced first if both taken.  Final dedup via *error* key.
    """
    targets = [_apply_quality_curve(i / (num_points - 1), lo, hi,
                                    curve=quality_curve, power=quality_power)
               for i in range(num_points)]

    # Top-2 nearest neighbor per target
    candidates = []
    for t in targets:
        idx_sorted = sorted(range(len(pareto)),
                            key=lambda i: abs(pareto[i]["accumulated_error"] - t))
        fst, snd = idx_sorted[0], idx_sorted[1] if len(idx_sorted) > 1 else idx_sorted[0]
        candidates.append({
            "first_pidx":  fst,
            "first_dist":  abs(pareto[fst]["accumulated_error"] - t),
            "second_pidx": snd,
            "second_dist": abs(pareto[snd]["accumulated_error"] - t),
        })

    # Greedy assign: best-justified targets first
    candidates.sort(key=lambda c: (c["first_dist"], c["first_pidx"]))
    assigned = set()
    pts = []
    for c in candidates:
        if c["first_pidx"] not in assigned:
            chosen = c["first_pidx"]
        elif c["second_pidx"] not in assigned:
            chosen = c["second_pidx"]
        else:
            chosen = c["first_pidx"]
        assigned.add(chosen)
        pts.append(pareto[chosen])

    # Deduplicate by error
    seen = {}
    for p in pts:
        cp = _make_control_point(p)
        seen[cp["error"]] = cp
    return sorted(seen.values(), key=lambda cp: cp["error"])


def _resolve_error_range(pareto, error_min, error_max):
    """Determine min/max error bounds from inputs or data."""
    p_min = pareto[0]["accumulated_error"]
    p_max = pareto[-1]["accumulated_error"]

    lo = error_min if error_min is not None else p_min
    hi = error_max if error_max is not None else p_max

    # Clamp to data range so every point has a real config to sample
    lo = max(lo, p_min)
    hi = min(hi, p_max)

    if lo >= hi:
        lo, hi = p_min, p_max

    return lo, hi


def _apply_quality_curve(frac: float, lo: float, hi: float,
                         curve: str = "linear", power: float = 2.0) -> float:
    """Map a 0-1 fraction to an error value using the configured curve.

    Supported curves:
      "linear":      uniform spacing (legacy default)
      "power":       polynomial compression toward low error (p > 1 → finer
                     granularity at high-quality end)
      "exponential": geometric spacing (proportional steps)
    """
    if curve == "exponential":
        ratio = hi / lo if lo > 0 else 1.0
        return lo * (ratio ** frac)

    if curve == "power":
        return lo + (hi - lo) * (frac ** power)

    # linear (default / backward-compatible)
    return lo + frac * (hi - lo)


def build_presets(
    pareto_path,
    num_points=None,
    error_min=None,
    error_max=None,
    preset_steps=None,
    lpips_scale=6.0,
    quality_curve="linear",
    quality_power=2.0,
):
    with open(pareto_path) as f:
        pareto = json.load(f)

    if not pareto:
        raise ValueError("Pareto frontier is empty")

    pareto.sort(key=lambda r: r["accumulated_error"])
    lo, hi = _resolve_error_range(pareto, error_min, error_max)

    if num_points is not None:
        control_points = _sample_control_points(
            pareto, lo, hi, num_points,
            quality_curve=quality_curve, quality_power=quality_power,
        )
    else:
        control_points = _collect_all_in_range(pareto, lo, hi)

    presets = {
        "_description": (
            "Auto-generated TeaCache presets for Anima/Cosmos-Predict2. "
            "Each control point is a full configuration from the Pareto frontier. "
            "The slider interpolates threshold between neighboring points; "
            "discrete params snap to the nearer anchor."
        ),
        "_error_range": [round(lo, 6), round(hi, 6)],
        "_lpips_scale": lpips_scale,
        "control_points": control_points,
    }

    if quality_curve != "linear":
        presets["_quality_curve"] = quality_curve
        if quality_curve == "power":
            presets["_quality_power"] = quality_power

    if preset_steps is not None:
        presets["_steps"] = preset_steps

    return presets


def main():
    parser = argparse.ArgumentParser(
        description="Build TeaCache presets from Pareto frontier"
    )
    parser.add_argument(
        "--pareto", required=True,
        help="Path to pareto_frontier.json (Phase 2 output)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path (default: <project_root>/anima_presets.json)",
    )
    parser.add_argument(
        "--points", type=int, default=None,
        help="Limit to N control points (default: include every Pareto point in range)",
    )
    parser.add_argument(
        "--error-min", type=float, default=None,
        help="Override minimum accumulated_error bound (quality=0)",
    )
    parser.add_argument(
        "--error-max", type=float, default=None,
        help="Override maximum accumulated_error bound (quality=100)",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Step count the presets were calibrated for (stored as metadata)",
    )
    parser.add_argument(
        "--lpips-scale", type=float, default=6.0,
        help="Multiplier for error→LPIPS display hint (default: 6.0)",
    )
    parser.add_argument(
        "--quality-curve", default="linear",
        choices=["linear", "power", "exponential"],
        help="Curve for mapping quality slider to error (default: linear)",
    )
    parser.add_argument(
        "--quality-power", type=float, default=2.0,
        help="Exponent for power quality curve (default: 2.0)",
    )
    args = parser.parse_args()

    presets = build_presets(
        args.pareto,
        num_points=args.points,
        error_min=args.error_min,
        error_max=args.error_max,
        preset_steps=args.steps,
        lpips_scale=args.lpips_scale,
        quality_curve=args.quality_curve,
        quality_power=args.quality_power,
    )

    out_path = (
        Path(args.output) if args.output
        else Path(__file__).parent.parent / "anima_presets.json"
    )
    with open(out_path, "w") as f:
        json.dump(presets, f, indent=2)

    n = len(presets["control_points"])
    lo, hi = presets["_error_range"]
    print(f"Wrote {n} control points → {out_path}")
    print(f"  Error range: {lo:.6f} → {hi:.6f}")
    print(f"  LPIPS display: {lo * args.lpips_scale:.4f} → {hi * args.lpips_scale:.4f}")
    print(f"  {'─' * 78}")
    print(f"  {'#':>3}  {'error':>10}  {'thresh':>8}  "
          f"{'speedup':>7}  {'lpips~':>7}  {'source':<20}  {'acc':<12}")
    print(f"  {'─' * 78}")
    for i, cp in enumerate(presets["control_points"]):
        cfg = cp["config"]
        lpips_est = round(cp["error"] * args.lpips_scale, 3)
        print(f"  {i:>3}  {cp['error']:>10.5f}  {cfg['rel_l1_thresh']:>8.4f}  "
              f"{cp['speedup']:>6.2f}x  {lpips_est:>7.3f}  "
              f"{cfg['source']:<20}  {cfg['accumulation_type']:<12}")


if __name__ == "__main__":
    main()
