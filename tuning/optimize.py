#!/usr/bin/env python3
"""Phase 2: Offline configuration search.

Loads calibration data recorded in Phase 1 and simulates TeaCache decisions
for thousands of candidate configurations without touching the GPU.

For each configuration, it replays the accumulator logic on the recorded
sequence of delta stats, measuring:
  - skip_rate:      fraction of steps where should_calc = False
  - accumulated_error: sum of out_rel for skipped steps (quality proxy)
  - score:          combined metric (speedup × quality)

The output is a Pareto frontier + ranked list of optimal configurations.

Usage:
    python -m tuning.optimize --data outputs/20260711-120000/calibration_data.jsonl
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config_types import (
    CalibrationEntry, TeacacheConfig, OptimizationResult, TuningConfig,
)
from .forward import (
    compute_distance, apply_mapping, accumulate_distance,
    step_schedule_multiplier,
)


def load_calibration_data(data_path: str) -> List[CalibrationEntry]:
    entries = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(CalibrationEntry.from_dict(json.loads(line)))
    print(f"[load] {len(entries)} calibration entries from {data_path}")

    # Filter to valid entries (have output change data)
    valid = [e for e in entries if e.out_rel > 0 or e.res_rel > 0]
    print(f"[load] {len(valid)} valid entries (with ground truth output change)")
    return valid


def get_source_stats(entry: CalibrationEntry, source: str) -> dict:
    """Extract delta stats for a specific source signal."""
    if source == "t_emb" and entry.t_emb is not None:
        return {
            "mean": entry.t_emb.mean, "max": entry.t_emb.max,
            "std": entry.t_emb.std, "p95": entry.t_emb.p95,
            "median": entry.t_emb.median, "min": entry.t_emb.min,
            "denom": entry.t_emb.denom,
        }
    if source == "first_block_shift" and entry.shift is not None:
        return {
            "mean": entry.shift.mean, "max": entry.shift.max,
            "std": entry.shift.std, "p95": entry.shift.p95,
            "median": entry.shift.median, "min": entry.shift.min,
            "denom": entry.shift.denom,
        }
    if source == "pooled_latent" and entry.latent is not None:
        return {
            "mean": entry.latent.mean, "max": entry.latent.max,
            "std": entry.latent.std, "p95": entry.latent.p95,
            "median": entry.latent.median, "min": entry.latent.min,
            "denom": entry.latent.denom,
        }
    return None


def simulate_config(
    entries: List[CalibrationEntry],
    cfg: TeacacheConfig,
) -> Tuple[float, float, float, float]:
    """Simulate one TeacacheConfig on recorded data.

    Returns (skip_rate, accumulated_error, estimated_speedup).

    The simulation replays the accumulator logic on sequences of entries
    grouped by (prompt_id, seed, cond). For each group, it steps through
    entries in order, simulating the distance → mapping → accumulation →
    decision pipeline.

    skip_rate: fraction of entries where should_calc = False
    accumulated_error: sum of out_rel for skipped entries (proxy for quality loss)
    estimated_speedup: 1.0 / (1.0 - skip_rate * block_cost_ratio)
      where block_cost_ratio ≈ 0.85 (blocks are ~85% of forward compute)
    """
    BLOCK_COST_RATIO = 0.85

    # Group entries by (prompt_id, seed, cond)
    groups: Dict[Tuple[int, int, int, int], List[CalibrationEntry]] = {}
    for e in entries:
        key = (e.prompt_id, e.seed, e.cond, e.total_steps)
        groups.setdefault(key, []).append(e)

    # Sort each group by step
    for key in groups:
        groups[key].sort(key=lambda e: e.step)

    total_entries = 0
    skip_count = 0
    total_error = 0.0

    for key, group in groups.items():
        accumulated = 0.0
        total_entries += len(group)

        for entry in group:
            # Get stats for the chosen source
            stats = get_source_stats(entry, cfg.source)
            if stats is None:
                continue

            # Knob 2+3: Distance metric
            distance = compute_distance(stats, cfg.metric_type, cfg.metric_weights)
            if cfg.signal_scale != 1.0:
                distance *= cfg.signal_scale

            # Knob 4: Mapping
            predicted = apply_mapping(
                distance, cfg.mapping_type, cfg.coefficients, cfg.mapping_params
            )

            # Knob 7: Step schedule
            mult = step_schedule_multiplier(
                entry.step_fraction, cfg.step_schedule
            )
            effective_thresh = cfg.rel_l1_thresh * mult

            # Knob 5: Accumulation
            new_acc, should_calc = accumulate_distance(
                accumulated, predicted, effective_thresh,
                cfg.accumulation_type, cfg.accumulation_params,
            )
            accumulated = new_acc

            if not should_calc:
                skip_count += 1
                # Quality penalty: use out_rel as proxy for quality loss
                total_error += entry.out_rel if entry.out_rel > 0 else entry.res_rel

    skip_rate = skip_count / max(total_entries, 1)
    avg_error = total_error / max(total_entries, 1)
    # estimated_speedup: higher skip rate + LOW error = good
    quality_proxy = 1.0 / (1.0 + avg_error)
    speedup = 1.0 / (1.0 - skip_rate * BLOCK_COST_RATIO)

    return skip_rate, avg_error, speedup, quality_proxy


def fit_polynomial_coefficients(
    entries: List[CalibrationEntry],
    cfg: TeacacheConfig,
    degree: int = 4,
    quiet: bool = False,
) -> List[float]:
    """Fit polynomial coefficients for a given configuration (source, metric, scale).

    Uses all valid entries to build (rel_l1_scaled, out_rel) pairs and fit
    numpy.polyfit of the requested degree.
    """
    xs = []
    ys = []

    for entry in entries:
        stats = get_source_stats(entry, cfg.source)
        if stats is None:
            continue

        distance = compute_distance(stats, cfg.metric_type, cfg.metric_weights)
        if cfg.signal_scale != 1.0:
            distance *= cfg.signal_scale

        y_val = entry.out_rel if entry.out_rel > 0 else entry.res_rel
        if y_val <= 0:
            continue

        xs.append(distance)
        ys.append(y_val)

    if len(xs) < 50:
        # Not enough data for polyfit; return identity coefficients
        return [0.0, 0.0, 0.0, 1.0, 0.0]

    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)

    if mask.sum() < 50:
        return [0.0, 0.0, 0.0, 1.0, 0.0]

    coeffs = np.polyfit(xs[mask], ys[mask], deg=degree).tolist()
    predicted = np.polyval(coeffs, xs[mask])
    rmse = float(np.sqrt(np.mean((ys[mask] - predicted) ** 2)))

    if not quiet:
        print(f"  [fit] n={int(mask.sum())}  range=[{xs[mask].min():.5f}, {xs[mask].max():.5f}]  "
              f"RMSE={rmse:.4f}  max|coeff|={max(abs(c) for c in coeffs):.3e}")

    return coeffs


def generate_candidate_configs(tcfg: TuningConfig) -> List[TeacacheConfig]:
    """Generate all candidate configurations to test."""
    opt = tcfg.optimization
    configs = []

    for source in opt["sources"]:
        for metric_type in opt["metric_types"]:
            for metric_weights_scenario in opt["metric_weights_scenarios"]:
                # Skip incompatible combinations
                if metric_type == "mean_only" and metric_weights_scenario != {"mean": 1.0}:
                    continue
                if metric_type == "mean_and_max" and set(metric_weights_scenario.keys()) != {"mean", "max"}:
                    continue

                for mapping_type in opt["mapping_types"]:
                    # Determine mapping params to sweep
                    mapping_params_list = [{}]
                    mapping_type_key = opt.get("mapping_params_scenarios", {}).get(mapping_type, [])
                    if mapping_type_key:
                        mapping_params_list = mapping_type_key
                    elif mapping_type == "polynomial":
                        mapping_params_list = [{}]  # no extra params, coefficients from fit

                    for mapping_params in mapping_params_list:
                        for accum_type in opt["accumulation_types"]:
                            for accum_params_dict in opt["accumulation_params"]:
                                if accum_type == "hard_reset" and accum_params_dict != {}:
                                    continue
                                if accum_type == "carry_over" and accum_params_dict != {}:
                                    continue
                                if accum_type == "leaky" and "leak_factor" not in accum_params_dict:
                                    continue
                                if accum_type == "windowed" and "window_size" not in accum_params_dict:
                                    continue

                                for schedule in opt["step_schedules"]:
                                    scales = opt["signal_scales"].get(source, [1.0])
                                    for scale in scales:
                                        for residual_strat in opt["residual_strategies"]:
                                            res_params_list = [{}]
                                            if residual_strat in ("blended", "scaled"):
                                                res_params_list = opt.get("residual_params_scenarios", [{}])

                                            for res_params in res_params_list:
                                                for block_mode in opt.get("block_modes", ["all_or_nothing"]):
                                                    block_params_list = [{}]
                                                    block_scenarios = opt.get("block_params_scenarios", {})
                                                    if block_mode in block_scenarios:
                                                        block_params_list = block_scenarios[block_mode]

                                                    for block_params in block_params_list:
                                                        cfg = TeacacheConfig(
                                                            source=source,
                                                            metric_type=metric_type,
                                                            metric_weights=metric_weights_scenario,
                                                            signal_scale=scale,
                                                            mapping_type=mapping_type,
                                                            coefficients=[],
                                                            mapping_params=mapping_params,
                                                            accumulation_type=accum_type,
                                                            accumulation_params=accum_params_dict,
                                                            step_schedule=schedule,
                                                            start_percent=0.05,
                                                            end_percent=0.95,
                                                            residual_strategy=residual_strat,
                                                            residual_params=res_params,
                                                            block_mode=block_mode,
                                                            block_params=block_params,
                                                        )
                                                        configs.append(cfg)

    print(f"[candidates] Generated {len(configs)} candidate configurations")
    return configs


def optimize(configs: List[TeacacheConfig],
             entries: List[CalibrationEntry],
             tcfg: TuningConfig) -> List[OptimizationResult]:
    """Simulate all candidate configs, compute scores, build Pareto frontier.

    If cross_validate is True in config, splits entries by prompt_id:
    fits polynomial on train set, evaluates on holdout set.
    """
    import random
    import time as time_mod
    opt = tcfg.optimization
    results: List[OptimizationResult] = []
    t0 = time_mod.time()

    # Cross-validation split
    do_cv = opt.get("cross_validate", False)
    cv_fraction = opt.get("cv_holdout_fraction", 0.2)
    train_entries = entries
    holdout_entries = entries
    if do_cv:
        prompt_ids = sorted(set(e.prompt_id for e in entries))
        rng = random.Random(42)
        rng.shuffle(prompt_ids)
        n_holdout = max(1, int(len(prompt_ids) * cv_fraction))
        holdout_ids = set(prompt_ids[:n_holdout])
        train_entries = [e for e in entries if e.prompt_id not in holdout_ids]
        holdout_entries = [e for e in entries if e.prompt_id in holdout_ids]
        print(f"  CV: {len(train_entries)} train / {len(holdout_entries)} holdout "
              f"(split by prompt, {cv_fraction:.0%} holdout)")

    total = len(configs)
    last_log = 0

    for i, cfg in enumerate(configs):
        if cfg.mapping_type == "polynomial":
            cfg.coefficients = fit_polynomial_coefficients(
                train_entries, cfg, degree=opt.get("poly_degree", 4)
            )

        if do_cv:
            skip_rate, avg_error, speedup, quality = simulate_config(holdout_entries, cfg)
        else:
            skip_rate, avg_error, speedup, quality = simulate_config(entries, cfg)

        score = speedup * quality

        results.append(OptimizationResult(
            config=cfg,
            skip_rate=skip_rate,
            estimated_speedup=speedup,
            accumulated_error=avg_error,
            score=score,
        ))

        # Progress with ETA
        elapsed = time_mod.time() - t0
        do_log = (i + 1) % 50 == 0 or i == 0 or (i + 1) == total
        if do_log and (i + 1) != last_log:
            last_log = i + 1
            eta = elapsed / (i + 1) * (total - i - 1) if i > 0 else 0
            print(f"  [{i+1:>5d}/{total}] "
                  f"{(i+1)/total*100:5.1f}%  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                  f"last: src={cfg.source} {cfg.metric_type} {cfg.mapping_type} "
                  f"skip={skip_rate:.1%} sp={speedup:.2f}x score={score:.3f}")

    elapsed = time_mod.time() - t0
    print(f"  Complete: {elapsed:.1f}s ({elapsed/total*1000:.1f} ms per config)")

    results.sort(key=lambda r: r.score, reverse=True)

    # Build Pareto frontier — exclude configs with no caching effect
    pareto = []
    for r in results:
        if r.skip_rate < 0.01:
            continue
        dominated = False
        for p in pareto:
            if (p.estimated_speedup >= r.estimated_speedup
                    and p.accumulated_error <= r.accumulated_error):
                if (p.estimated_speedup > r.estimated_speedup
                        or p.accumulated_error < r.accumulated_error):
                    dominated = True
                    break
        if not dominated:
            pareto.append(r)

    print(f"\n[pareto] {len(pareto)} Pareto-optimal configurations "
          f"(out of {len(results)} total, {len(configs)} candidates)")
    print(f"[time]   {elapsed:.1f}s total, {elapsed/total*1000:.1f}ms/config, "
          f"{elapsed/len(configs)*1000:.1f}ms/candidate")

    return results, pareto


def main():
    parser = argparse.ArgumentParser(description="TeaCache Offline Optimizer")
    parser.add_argument("--data", required=True,
                        help="Path to calibration_data.jsonl from Phase 1")
    parser.add_argument("--config", default=None,
                        help="Path to config.json (default: tuning/config.json)")
    parser.add_argument("--output", default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    if args.config is None:
        args.config = str(Path(__file__).parent / "config.json")
    tcfg = TuningConfig.load(args.config)

    out_dir = Path(args.output or tcfg.output_dir) / "optimization"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  TeaCache Offline Optimizer — Phase 2")
    print("=" * 60)
    print(f"  Data:    {args.data}")
    print(f"  Output:  {out_dir}")
    print("=" * 60)

    # Load data
    entries = load_calibration_data(args.data)
    if len(entries) < 100:
        print(f"ERROR: Only {len(entries)} valid entries. Need more calibration data.")
        sys.exit(1)

    # Generate candidates
    configs = generate_candidate_configs(tcfg)

    # Optimize
    results, pareto = optimize(configs, entries, tcfg)

    # Save results
    (out_dir / "all_results.json").write_text(
        json.dumps([r.to_dict() for r in results[:500]], indent=2)
    )
    (out_dir / "pareto_frontier.json").write_text(
        json.dumps([r.to_dict() for r in pareto], indent=2)
    )

    # Print top 10
    print(f"\n  Top 10 configurations:")
    print(f"  {'─' * 60}")
    for i, r in enumerate(results[:10]):
        c = r.config
        print(f"  {i+1:>2}. src={c.source:<20} metric={c.metric_type:<15} "
              f"map={c.mapping_type:<12} acc={c.accumulation_type:<12} "
              f"scale={c.signal_scale:>6.0f}  "
              f"skip={r.skip_rate:.1%}  speedup={r.estimated_speedup:.2f}x  "
              f"error={r.accumulated_error:.4f}  score={r.score:.3f}")

    # Print Pareto in order of speedup
    pareto_sorted = sorted(pareto, key=lambda r: r.estimated_speedup)
    print(f"\n  Pareto frontier ({len(pareto)} configs):")
    print(f"  {'─' * 60}")
    for r in pareto_sorted:
        c = r.config
        print(f"  speedup={r.estimated_speedup:.2f}x  error={r.accumulated_error:.4f}  "
              f"src={c.source} metric={c.metric_type} map={c.mapping_type} "
              f"acc={c.accumulation_type} scale={c.signal_scale:.0f}")

    print(f"\n  Results saved to: {out_dir}")

    # Print the winning coefficients
    if results:
        winner = results[0]
        c = winner.config
        if c.coefficients:
            cs = ", ".join(f"{v:.10e}" for v in c.coefficients)
            print(f"\n  Winning coefficients for SUPPORTED_MODELS_COEFFICIENTS:")
            print(f'    "{c.source}_{c.metric_type}": [{cs}],')
            print(f"  Winner config: {winner.config.to_dict()}")


if __name__ == "__main__":
    main()
