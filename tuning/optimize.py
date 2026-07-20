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

Configurations that differ only in block_mode, residual_strategy, or other
cosmetic fields produce identical simulation results.  These are deduplicated
by signal-space signature before simulation and replicated afterwards,
giving a ~49× reduction in simulation work for the default search space.

Usage:
    python -m tuning.optimize --data outputs/20260711-120000/calibration_data.jsonl
"""

# ── Prevent BLAS thread contention in multiprocessing workers ───────
# Must be set BEFORE importing numpy (or torch, which imports numpy)
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import math
import sys
import time as time_mod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config_types import (
    CalibrationEntry, TeacacheConfig, OptimizationResult, TuningConfig,
)
from .sim_data import SimData
from .sim_runner import simulate_config as _simulate_config_sd
from .sim_runner import _get_source_stats as _get_group_stats
from .sim_runner import _compute_distance, _block_fraction

# ═══════════════════════════════════════════════════════════════════════════
#  Quality scoring functions
# ═══════════════════════════════════════════════════════════════════════════

def compute_quality_score(error: float, scoring: dict) -> float:
    """Convert simulated accumulated error to a quality score (0-1).

    scoring types:
      "linear":              1/(1+error) — mild penalty, no params
      "exponential":         exp(-error/target) — strong after target
      "gaussian":            exp(-0.5*(error/target)^2) — very strong after
      "step":                1.0 if error < target else 0.0 — hard cutoff
      "power":               1/(1+(error/target)^power) — configurable
      "thresholded_power":   1/(1+(max(0,error-target)/target)^power)
                               Zero penalty below target, then power-law ramp.
                               This is what you want when target means "errors
                               up to this value should cost near-nothing."

    target interpretation by type:
      exponential:       target → quality=0.37 (e^-1)
      gaussian:          target → quality=0.61 (e^-0.5)
      step:              target → hard quality=0 boundary
      power:             target → quality=0.50
      thresholded_power: target → quality=1.00 (zero penalty below)
    """
    stype = scoring.get("type", "thresholded_power")
    target = max(scoring.get("target", 0.05), 1e-8)

    if stype == "linear":
        return 1.0 / (1.0 + error)

    if stype == "exponential":
        return math.exp(-error / target)

    if stype == "gaussian":
        x = error / target
        return math.exp(-0.5 * x * x)

    if stype == "step":
        return 1.0 if error < target else 0.0

    if stype == "power":
        p = scoring.get("power", 2.0)
        x = error / target
        return 1.0 / (1.0 + x ** p)

    if stype == "thresholded_power":
        p = scoring.get("power", 3.0)
        tq = scoring.get("target_quality", 1.0)
        if tq >= 1.0:
            effective_target = target
        else:
            # Shift so quality = target_quality at error = target
            ratio = max((1.0 - tq) / tq, 1e-12)
            effective_target = target / (1.0 + ratio ** (1.0 / p))
        excess = max(0.0, error - effective_target) / effective_target
        return 1.0 / (1.0 + excess ** p)

    # fallback
    return 1.0 / (1.0 + error)


# ═══════════════════════════════════════════════════════════════════════════
#  Signal-space deduplication
# ═══════════════════════════════════════════════════════════════════════════

def _signal_signature(cfg: TeacacheConfig) -> tuple:
    """Hashable key for the simulation-relevant subset of a config.

    Fields like block_mode, residual_strategy, cross_feed do NOT affect
    the skip/error simulation — only speedup estimation (which we recompute
    per cosmetic variant using _block_fraction).

    Coefficients are NOT included because they are derived from the polyfit
    (keyed by source + metric_type + weights + scale) and populated later.
    """
    return (
        cfg.source,
        cfg.metric_type,
        tuple(sorted(cfg.metric_weights.items())),
        cfg.signal_scale,
        cfg.mapping_type,
        tuple(sorted(cfg.mapping_params.items())),
        cfg.accumulation_type,
        tuple(sorted(cfg.accumulation_params.items())),
        cfg.step_schedule,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Polynomial fit (SimData-backed)
# ═══════════════════════════════════════════════════════════════════════════

def _poly_fit_key(cfg: TeacacheConfig) -> tuple:
    """Return a hashable key for the polynomial fit that this config needs."""
    return (
        cfg.source,
        cfg.metric_type,
        tuple(sorted(cfg.metric_weights.items())),
        cfg.signal_scale,
    )


def fit_polynomial_coefficients(
    sim_data: SimData,
    cfg: TeacacheConfig,
    degree: int = 4,
    quiet: bool = False,
) -> List[float]:
    """Fit polynomial coefficients using precomputed SimData arrays."""
    xs_list, ys_list = [], []

    for group in sim_data.groups:
        stats = _get_group_stats(group, cfg.source)
        if stats is None:
            continue

        dist = _compute_distance(stats, cfg)
        if cfg.signal_scale != 1.0:
            dist *= cfg.signal_scale

        y = np.where(group.out_rel > 0, group.out_rel, group.res_rel)
        mask = np.isfinite(dist) & np.isfinite(y) & (dist > 0) & (y > 0)
        if mask.any():
            xs_list.append(dist[mask])
            ys_list.append(y[mask])

    if not xs_list:
        return [0.0, 0.0, 0.0, 1.0, 0.0]

    xs = np.concatenate(xs_list)
    ys = np.concatenate(ys_list)

    if len(xs) < 50:
        return [0.0, 0.0, 0.0, 1.0, 0.0]

    coeffs = np.polyfit(xs, ys, deg=degree).tolist()
    predicted = np.polyval(coeffs, xs)
    rmse = float(np.sqrt(np.mean((ys - predicted) ** 2)))

    if not quiet:
        print(f"  [fit] n={len(xs)}  range=[{xs.min():.5f}, {xs.max():.5f}]  "
              f"RMSE={rmse:.4f}  max|coeff|={max(abs(c) for c in coeffs):.3e}")

    return coeffs


def precompute_polyfits(
    configs: List[TeacacheConfig],
    sim_data: SimData,
    poly_degree: int = 4,
) -> Dict[tuple, list]:
    """Fit polynomial coefficients for every unique (source, metric, scale) combo.

    Only configs with mapping_type="polynomial" are fitted. Identity,
    power_law, and softplus configs don't need coefficients.
    """
    unique_keys = set()
    for cfg in configs:
        if cfg.mapping_type == "polynomial":
            unique_keys.add(_poly_fit_key(cfg))

    cache = {}
    for key in sorted(unique_keys):
        dummy = TeacacheConfig(
            source=key[0],
            metric_type=key[1],
            metric_weights=dict(key[2]),
            signal_scale=key[3],
            mapping_type="polynomial",
        )
        coeffs = fit_polynomial_coefficients(sim_data, dummy, degree=poly_degree, quiet=True)
        cache[key] = coeffs
    return cache


# ═══════════════════════════════════════════════════════════════════════════
#  Multiprocessing worker (module-level for pickling)
# ═══════════════════════════════════════════════════════════════════════════

_worker_sim_data: Optional[SimData] = None
_worker_poly_cache: Dict = {}
_worker_opt: dict = {}


def _init_worker(sim_data: SimData, poly_cache: dict, opt: dict):
    """Called once per worker process to share read-only simulation data."""
    global _worker_sim_data, _worker_poly_cache, _worker_opt
    _worker_sim_data = sim_data
    _worker_poly_cache = poly_cache
    _worker_opt = opt


def _best_threshold(
    sim_data: SimData,
    cfg: TeacacheConfig,
    thresholds: list,
    scoring_config: dict,
) -> tuple:
    """Sweep thresholds for one config.  Returns (skip, err, sp, quality, best_t)."""
    best_sc = -1.0
    best = (0.0, 0.0, 1.0, 1.0, thresholds[0])

    for t in thresholds:
        skip, err, sp, _ = _simulate_config_sd(sim_data, cfg, t)
        quality = compute_quality_score(err, scoring_config)
        sc = sp * quality
        if sc > best_sc:
            best_sc = sc
            best = (skip, err, sp, quality, t)

    return best


def _process_config(idx_and_cfg: tuple) -> tuple:
    """Process a single config in a worker process. Sweeps candidate thresholds.

    Returns (idx, skip, err, sp, quality, best_thresh).
    """
    idx, cfg = idx_and_cfg
    global _worker_sim_data, _worker_poly_cache, _worker_opt

    if cfg.mapping_type == "polynomial":
        key = _poly_fit_key(cfg)
        cfg.coefficients = _worker_poly_cache.get(key, [])

    thresholds = _worker_opt.get("candidate_thresholds", [0.07])
    scoring_config = _worker_opt.get("quality_scoring",
                                      {"type": "thresholded_power", "target": 0.05, "power": 3.0})

    skip, err, sp, quality, best_t = _best_threshold(
        _worker_sim_data, cfg, thresholds, scoring_config,
    )

    return idx, skip, err, sp, quality, best_t


# ═══════════════════════════════════════════════════════════════════════════
#  Pareto sweep worker (module-level for pickling, reuses _worker_sim_data)
# ═══════════════════════════════════════════════════════════════════════════

_worker_sweep_values: list = []
_worker_scoring: dict = {}


def _init_sweep_worker(sim_data: SimData, sweep_values: list, scoring_config: dict):
    """Called once per worker process to share read-only sweep data."""
    global _worker_sim_data, _worker_sweep_values, _worker_scoring
    _worker_sim_data = sim_data
    _worker_sweep_values = sweep_values
    _worker_scoring = scoring_config


def _sweep_pareto_config(idx_and_cfg: tuple) -> tuple:
    """Sweep all thresholds for one Pareto config.
    Returns (idx, skip, err, sp, best_t)."""
    idx, cfg = idx_and_cfg
    global _worker_sim_data, _worker_sweep_values, _worker_scoring

    skip, err, sp, _quality, best_t = _best_threshold(
        _worker_sim_data, cfg, _worker_sweep_values, _worker_scoring,
    )
    return idx, skip, err, sp, best_t


# ═══════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_calibration_data(data_path: str) -> List[CalibrationEntry]:
    entries = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(CalibrationEntry.from_dict(json.loads(line)))
    print(f"[load] {len(entries)} calibration entries from {data_path}")

    valid = [e for e in entries if e.out_rel > 0 or e.res_rel > 0]
    print(f"[load] {len(valid)} valid entries (with ground truth output change)")
    return valid


# ═══════════════════════════════════════════════════════════════════════════
#  Candidate generation  (unchanged from prior version)
# ═══════════════════════════════════════════════════════════════════════════

def generate_candidate_configs(tcfg: TuningConfig,
                                entries: Optional[List[CalibrationEntry]] = None,
                                ) -> List[TeacacheConfig]:
    """Generate all candidate configurations to test.

    If entries is provided and auto_scale_target is set in config,
    computes data-driven scale factors that push the average distance
    for each source toward the target value. These are added as extra
    scale candidates alongside the explicit scale list.
    """
    import random

    opt = tcfg.optimization
    configs = []

    # ── Auto-scale (data-driven) ───────────────────────────────────────
    auto_target = opt.get("auto_scale_target", None)
    auto_scales: Dict[str, list] = {}
    if auto_target is not None and entries:
        for source in ["t_emb", "first_block_shift", "pooled_latent"]:
            distances = []
            for e in entries:
                attr = {"t_emb": e.t_emb, "first_block_shift": e.shift,
                        "pooled_latent": e.latent}.get(source)
                if attr is None:
                    continue
                d = attr.mean
                if d > 0:
                    distances.append(d)
            if distances:
                avg_dist = sum(distances) / len(distances)
                auto_scale = round(auto_target / avg_dist, 1)
                auto_scales[source] = [auto_scale]
                print(f"  [auto_scale] {source}: avg dist={avg_dist:.5f}  "
                      f"\u2192  scale={auto_scale:.1f}  (target={auto_target})")
            else:
                auto_scales[source] = []

    # ── Generate candidates ────────────────────────────────────────────

    for source in opt["sources"]:
        for metric_type in opt["metric_types"]:
            for metric_weights_scenario in opt["metric_weights_scenarios"]:
                if metric_type == "mean_only" and metric_weights_scenario != {"mean": 1.0}:
                    continue
                if metric_type == "mean_and_max" and set(metric_weights_scenario.keys()) != {"mean", "max"}:
                    continue

                for mapping_type in opt["mapping_types"]:
                    mapping_params_list = [{}]
                    mapping_type_key = opt.get("mapping_params_scenarios", {}).get(mapping_type, [])
                    if mapping_type_key:
                        mapping_params_list = mapping_type_key
                    elif mapping_type == "polynomial":
                        mapping_params_list = [{}]

                    for mapping_params in mapping_params_list:
                        if source == "pooled_latent":
                            pl_mode = opt.get("pooled_latent_mode", "mean")
                            if "pooled_latent_mode" not in mapping_params:
                                mapping_params = dict(mapping_params, pooled_latent_mode=pl_mode)
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
                                    if source in auto_scales and auto_scales[source]:
                                        extra = [s for s in auto_scales[source] if s not in scales]
                                        scales = list(scales) + extra
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

    # ── Cap total candidates if configured ────────────────────────────
    max_cap = opt.get("max_candidates", 0)
    if max_cap > 0 and len(configs) > max_cap:
        rng = random.Random(42)
        rng.shuffle(configs)
        configs = configs[:max_cap]
        print(f"  [cap] Sampled {len(configs)}/{max_cap} candidates "
              f"(full pool had {len(configs)})")

    print(f"[candidates] Generated {len(configs)} candidate configurations")
    return configs


# ═══════════════════════════════════════════════════════════════════════════
#  Main optimization routine
# ═══════════════════════════════════════════════════════════════════════════

def optimize(configs: List[TeacacheConfig],
             entries: List[CalibrationEntry],
             tcfg: TuningConfig) -> Tuple[List[OptimizationResult], List[OptimizationResult]]:
    """Simulate all candidate configs, compute scores, build Pareto frontier.

    Uses signal-space deduplication: configs that differ only in block_mode /
    residual_strategy produce identical simulation results and are grouped
    together, giving a ~49× reduction for the default search space.

    Scoring is consistent across all phases: candidate threshold selection,
    final score computation, and Pareto sweeping all use the same
    compute_quality_score function.
    """
    import random
    opt = tcfg.optimization
    t0 = time_mod.time()

    # ── 1. Build SimData (once) ────────────────────────────────────────
    sim_data = SimData.from_entries(entries)

    # ── 2. CV split ────────────────────────────────────────────────────
    do_cv = opt.get("cross_validate", False)
    cv_fraction = opt.get("cv_holdout_fraction", 0.2)
    if do_cv:
        prompt_ids = sorted(set(e.prompt_id for e in entries))
        rng = random.Random(42)
        rng.shuffle(prompt_ids)
        n_holdout = max(1, int(len(prompt_ids) * cv_fraction))
        holdout_ids = set(prompt_ids[:n_holdout])
        train_sd = sim_data.filter_by_prompt_ids(set(prompt_ids) - holdout_ids)
        holdout_sd = sim_data.filter_by_prompt_ids(holdout_ids)
        print(f"  CV: {train_sd.n_entries} train / {holdout_sd.n_entries} holdout "
              f"(split by prompt, {cv_fraction:.0%} holdout)")
    else:
        train_sd = holdout_sd = sim_data

    # ── 3. Deduplicate by signal-space ─────────────────────────────────
    signal_groups: Dict[tuple, List[TeacacheConfig]] = {}
    for cfg in configs:
        sig = _signal_signature(cfg)
        signal_groups.setdefault(sig, []).append(cfg)

    unique_signal_configs = [group[0] for group in signal_groups.values()]
    ratio = len(configs) / max(len(unique_signal_configs), 1)
    print(f"  Dedup: {len(configs)} → {len(unique_signal_configs)} "
          f"unique signal-space configs ({ratio:.1f}× reduction)")

    # ── 4. Polynomial fits ─────────────────────────────────────────────
    t_pre = time_mod.time()
    poly_cache = precompute_polyfits(
        unique_signal_configs, train_sd,
        poly_degree=opt.get("poly_degree", 4),
    )
    n_fits = len(poly_cache)
    t_pre_elapsed = time_mod.time() - t_pre
    if n_fits:
        print(f"  [precompute] {n_fits} unique polynomial fits "
              f"in {t_pre_elapsed:.1f}s "
              f"({t_pre_elapsed/max(n_fits,1)*1000:.0f}ms each)")

    # ── 5. Simulate unique signal configs ───────────────────────────────
    thresholds = opt.get("candidate_thresholds", [0.07])
    scoring_config = opt.get("quality_scoring",
                              {"type": "thresholded_power", "target": 0.05, "power": 3.0})
    print(f"  Candidate thresholds: {thresholds}")
    print(f"  Quality scoring:      {scoring_config['type']} "
          f"(target={scoring_config.get('target', 0.05)})")

    total = len(unique_signal_configs)
    iter_count = total * holdout_sd.n_entries
    use_parallel = iter_count > 10_000_000

    # Simulate unique configs → collect (sig, OptimizationResult) pairs
    sim_results: Dict[tuple, OptimizationResult] = {}

    if use_parallel:
        n_workers = min(os.cpu_count() or 4, 16)
        chunksz = max(50, min(5000, total // (n_workers * 25)))
        print(f"  [parallel] {n_workers} workers × {total} configs, "
              f"chunksize={chunksz} ({iter_count//1_000_000}M entry-iterations, "
              f"~{total/(n_workers*chunksz):.0f} rounds)")

        indexed = list(enumerate(unique_signal_configs))
        results_list = [None] * total
        done_count = 0
        last_log = 0

        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(holdout_sd, poly_cache, opt),
        ) as pool:
            for idx, skip, err, sp, quality, best_thresh in pool.imap_unordered(
                _process_config, indexed, chunksize=chunksz
            ):
                cfg = unique_signal_configs[idx]
                cfg.rel_l1_thresh = best_thresh
                score = sp * quality
                results_list[idx] = OptimizationResult(
                    config=cfg, skip_rate=skip, estimated_speedup=sp,
                    accumulated_error=err, score=score,
                )
                done_count += 1
                elapsed = time_mod.time() - t0
                do_log = done_count % 50 == 0 or done_count == 1 or done_count == total
                if do_log and done_count != last_log:
                    last_log = done_count
                    eta = elapsed / done_count * (total - done_count) if done_count > 0 else 0
                    print(f"\r  [{done_count:>5d}/{total}] "
                          f"{done_count/total*100:5.1f}%  "
                          f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                          f"[{n_workers} workers]  "
                          f"sp={sp:.2f}x  score={score:.3f}",
                          end="", flush=True)

        for i, r in enumerate(results_list):
            cfg = unique_signal_configs[i]
            sig = _signal_signature(cfg)
            sim_results[sig] = r
    else:
        for i, cfg in enumerate(unique_signal_configs):
            if cfg.mapping_type == "polynomial":
                key = _poly_fit_key(cfg)
                cfg.coefficients = poly_cache.get(key, [])

            best_skip, best_err, best_sp, best_quality, best_thresh = _best_threshold(
                holdout_sd, cfg, thresholds, scoring_config,
            )

            cfg.rel_l1_thresh = best_thresh
            score = best_sp * best_quality
            sim_results[_signal_signature(cfg)] = OptimizationResult(
                config=cfg, skip_rate=best_skip, estimated_speedup=best_sp,
                accumulated_error=best_err, score=score,
            )

            elapsed = time_mod.time() - t0
            do_log = (i + 1) % 50 == 0 or i == 0 or (i + 1) == total
            if do_log:
                eta = elapsed / (i + 1) * (total - i - 1) if i > 0 else 0
                print(f"\r  [{i+1:>5d}/{total}] "
                      f"{(i+1)/total*100:5.1f}%  "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                      f"last: src={cfg.source} {cfg.metric_type} {cfg.mapping_type} "
                      f"skip={best_skip:.1%} sp={best_sp:.2f}x score={score:.3f}",
                      end="", flush=True)

    # ── 6. Replicate results to full config space ──────────────────────
    all_results: List[OptimizationResult] = []
    for sig, group in signal_groups.items():
        base = sim_results[sig]
        bc = base.config  # template — signal fields, don't mutate
        for full_cfg in group:
            # Clone by explicit field copy — avoids deepcopy on 862K configs
            result_cfg = TeacacheConfig(
                source=bc.source,
                metric_type=bc.metric_type,
                metric_weights=dict(bc.metric_weights),
                signal_scale=bc.signal_scale,
                mapping_type=bc.mapping_type,
                coefficients=list(bc.coefficients),
                mapping_params=dict(bc.mapping_params),
                accumulation_type=bc.accumulation_type,
                accumulation_params=dict(bc.accumulation_params),
                step_schedule=bc.step_schedule,
                rel_l1_thresh=bc.rel_l1_thresh,
                start_percent=bc.start_percent,
                end_percent=bc.end_percent,
                block_mode=full_cfg.block_mode,
                block_params=full_cfg.block_params,
                residual_strategy=full_cfg.residual_strategy,
                residual_params=full_cfg.residual_params,
                cross_feed_enabled=full_cfg.cross_feed_enabled,
                cross_feed_strength=full_cfg.cross_feed_strength,
            )

            bf = _block_fraction(result_cfg)
            sp = (1.0 / (1.0 - base.skip_rate * bf)
                  if bf > 0 and base.skip_rate * bf < 1.0 else 1.0)
            quality = compute_quality_score(base.accumulated_error, scoring_config)
            score = sp * quality

            all_results.append(OptimizationResult(
                config=result_cfg,
                skip_rate=base.skip_rate,
                estimated_speedup=sp,
                accumulated_error=base.accumulated_error,
                score=score,
            ))

    # ── 7. Sort + Pareto frontier ──────────────────────────────────────
    elapsed = time_mod.time() - t0
    mode = "parallel" if use_parallel else "serial"
    print()  # newline after progress bar
    print(f"  Complete ({mode}): {elapsed:.1f}s "
          f"({elapsed/len(configs)*1000:.1f}ms per full config, "
          f"{elapsed/len(unique_signal_configs)*1000:.1f}ms per unique)")

    all_results.sort(key=lambda r: r.score, reverse=True)

    p_start = time_mod.time()
    candidates_for_pareto = [r for r in all_results if r.skip_rate >= 0.01]
    candidates_for_pareto.sort(
        key=lambda r: (-r.estimated_speedup, -r.accumulated_error)
    )
    pareto = []
    best_quality = float("inf")
    for r in candidates_for_pareto:
        if r.accumulated_error < best_quality:
            pareto.append(r)
            best_quality = r.accumulated_error

    p_elapsed = time_mod.time() - p_start
    print(f"\n[pareto] {len(pareto)} Pareto-optimal configurations "
          f"(from {len(candidates_for_pareto)} with skip>=1% / "
          f"{len(all_results)} total, {len(configs)} candidates) "
          f"in {p_elapsed:.1f}s")
    print(f"[time]   {elapsed:.1f}s total, "
          f"{elapsed/len(configs)*1000:.1f}ms/full-config")

    # ── 8. Pareto threshold sweep ──────────────────────────────────────
    # Refine each Pareto winner's threshold.  Floor at the smallest
    # candidate threshold so the sweep cannot rediscover the "do nothing"
    # trivial solution (skip_rate=0, error=0, score=1.0).
    min_cand = min(opt.get("candidate_thresholds", [0.07]))
    pareto_range = opt.get("pareto_threshold_range", [min_cand, 10.0])
    pareto_range[0] = max(pareto_range[0], min_cand)
    pareto_count = opt.get("pareto_threshold_count", 500)

    t_sweep_start = time_mod.time()
    sweep_values = np.geomspace(*pareto_range, num=pareto_count).tolist()
    n_ps = len(pareto)

    print(f"\n  Pareto sweep: {pareto_count} thresholds "
          f"[{pareto_range[0]:.3f}..{pareto_range[1]:.1f}] × {n_ps} configs")

    use_parallel_ps = n_ps * pareto_count > 5000

    if use_parallel_ps:
        n_workers = min(os.cpu_count() or 4, 16)
        indexed = [(i, r.config) for i, r in enumerate(pareto)]
        done = 0
        best_overall_sp = 1.0

        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=n_workers,
            initializer=_init_sweep_worker,
            initargs=(holdout_sd, sweep_values, scoring_config),
        ) as pool:
            for idx, skip, err, sp, best_t in pool.imap_unordered(
                _sweep_pareto_config, indexed, chunksize=1
            ):
                r = pareto[idx]
                r.config.rel_l1_thresh = best_t
                r.skip_rate = skip
                r.accumulated_error = err
                r.estimated_speedup = sp
                r.score = sp * compute_quality_score(err, scoring_config)

                done += 1
                if sp > best_overall_sp:
                    best_overall_sp = sp
                elapsed = time_mod.time() - t_sweep_start
                eta = elapsed / done * (n_ps - done) if done > 0 else 0
                print(f"\r  [{done:>3d}/{n_ps}] "
                      f"{done/n_ps*100:5.1f}%  "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                      end="", flush=True)
                print(f"\n{' ' * 22}best: {best_overall_sp:.2f}x                           ",
                      end="", flush=True)
                print(f"\033[F", end="")  # cursor back up for next update
        print()
    else:
        for i, r in enumerate(pareto):
            best_skip, best_err, best_sp, _quality, best_t = _best_threshold(
                holdout_sd, r.config, sweep_values, scoring_config,
            )
            r.config.rel_l1_thresh = best_t
            r.skip_rate = best_skip
            r.accumulated_error = best_err
            r.estimated_speedup = best_sp
            r.score = best_sp * compute_quality_score(best_err, scoring_config)

            elapsed = time_mod.time() - t_sweep_start
            eta = elapsed / (i + 1) * (n_ps - i - 1) if i > 0 else 0
            print(f"\r  [{i+1:>3d}/{n_ps}] "
                  f"{(i+1)/n_ps*100:5.1f}%  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                  f"last={best_sp:.2f}x",
                  end="", flush=True)
        print()

    t_sweep_elapsed = time_mod.time() - t_sweep_start
    print(f"  Pareto sweep complete in {t_sweep_elapsed:.1f}s\n")

    return all_results, pareto


# ═══════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

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
    configs = generate_candidate_configs(tcfg, entries)

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
               f"scale={c.signal_scale:>6.4g}  "
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
               f"acc={c.accumulation_type} scale={c.signal_scale:.4g}")

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
