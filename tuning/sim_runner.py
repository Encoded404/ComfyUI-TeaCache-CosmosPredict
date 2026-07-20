"""Simulator orchestrator — stateless; all data lives in SimData.

Provides vectorized distance computation, mapping, and step-schedule lookups
on top of the precomputed GroupData arrays.  The sequential accumulation is
delegated to sim_engine.simulate_group (Numba-optional).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config_types import TeacacheConfig
from .sim_data import GroupData, SimData
from .sim_engine import simulate_group


def _get_source_stats(group: GroupData, source: str) -> Optional[dict]:
    """Return stats dict for the requested source, or None if unavailable."""
    mapping = {
        "t_emb":              group.t_emb_stats,
        "first_block_shift":  group.shift_stats,
        "pooled_latent":      group.latent_stats,
    }
    return mapping.get(source)


def _compute_distance(stats: dict, cfg: TeacacheConfig) -> np.ndarray:
    """Vectorized distance computation.  Returns (G,) float64."""
    if cfg.metric_type == "mean_only":
        return stats["mean"].copy()
    if cfg.metric_type == "mean_and_max":
        wm = cfg.metric_weights.get("mean", 0.7)
        wx = cfg.metric_weights.get("max", 0.3)
        return wm * stats["mean"] + wx * stats["max"]
    # mean_max_std (or weighted_sum as fallback)
    wm = cfg.metric_weights.get("mean", 0.5)
    wx = cfg.metric_weights.get("max", 0.3)
    ws = cfg.metric_weights.get("std", 0.2)
    return wm * stats["mean"] + wx * stats["max"] + ws * stats["std"]


def _apply_mapping(dist: np.ndarray, cfg: TeacacheConfig) -> np.ndarray:
    """Vectorized mapping function.  Returns (G,) float64."""
    if cfg.mapping_type == "identity":
        return dist

    if cfg.mapping_type == "polynomial":
        if not cfg.coefficients:
            return dist
        return np.polyval(np.array(cfg.coefficients, dtype=np.float64), dist)

    if cfg.mapping_type == "power_law":
        k = float(cfg.mapping_params.get("k", 1.0))
        alpha = float(cfg.mapping_params.get("alpha", 0.5))
        return k * np.power(dist + 1e-10, alpha)

    if cfg.mapping_type == "softplus":
        k = float(cfg.mapping_params.get("k", 10.0))
        offset = float(cfg.mapping_params.get("offset", 0.05))
        x = k * (dist - offset)
        result = x.copy()
        small = x < 20.0
        result[small] = np.log1p(np.exp(x[small]))
        return result

    return dist


def _block_fraction(cfg: TeacacheConfig, default: float = 0.85) -> float:
    """Fraction of compute cost that can be cached, based on block_mode."""
    if cfg.block_mode == "all_or_nothing":
        return default

    if cfg.block_mode == "split_fraction":
        always = float(cfg.block_params.get("always_fraction", 0.36))
        return max(0.0, (1.0 - always) * default)

    if cfg.block_mode == "split_groups":
        always_groups = set(cfg.block_params.get("always_groups", [0, 1, 2]))
        cache_groups = set(cfg.block_params.get("cache_groups", []))
        n_cache = len(cache_groups)
        n_total = len(always_groups | cache_groups)
        return (n_cache / n_total) * default if n_total > 0 else 0.0

    return default


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════

def simulate_config(
    sim_data: SimData,
    cfg: TeacacheConfig,
    threshold: float,
) -> Tuple[float, float, float, float]:
    """Simulate one TeacacheConfig for a single threshold.

    Returns (skip_rate, avg_error, estimated_speedup, quality_proxy).
    """
    total_skip = 0
    total_error = 0.0
    total_steps = 0

    for group in sim_data.groups:
        stats = _get_source_stats(group, cfg.source)
        if stats is None:
            continue

        # ── Vectorized: distance → signal scale → mapping ──
        dist = _compute_distance(stats, cfg)
        if cfg.signal_scale != 1.0:
            dist *= cfg.signal_scale
        predicted = _apply_mapping(dist, cfg)

        # ── Vectorized: effective thresholds ──
        step_mult = group.step_mult.get(cfg.step_schedule, group.step_mult["constant"])
        thresholds = threshold * step_mult * cfg.signal_scale

        # ── Precompute merged penalty array (out_rel > 0 ? out_rel : res_rel) ──
        penalties = np.where(group.out_rel > 0, group.out_rel, group.res_rel)

        # ── Sequential: accumulation (Numba or Python, dispatched once) ──
        skip, err = simulate_group(
            predicted, thresholds, penalties,
            cfg.accumulation_type, cfg.accumulation_params,
        )

        total_skip += skip
        total_error += err
        total_steps += group.n_steps

    if total_steps == 0:
        return 0.0, 0.0, 1.0, 1.0

    skip_rate = total_skip / total_steps
    avg_error = total_error / total_steps
    quality_proxy = 1.0 / (1.0 + avg_error)
    bf = _block_fraction(cfg)
    speedup = (1.0 / (1.0 - skip_rate * bf)
               if bf > 0 and skip_rate * bf < 1.0 else 1.0)

    return skip_rate, avg_error, speedup, quality_proxy
