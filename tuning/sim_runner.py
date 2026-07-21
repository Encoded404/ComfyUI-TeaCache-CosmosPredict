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


def _compute_block_cosim_means(
    groups: tuple,
) -> Optional[np.ndarray]:
    """Compute per-block mean cosine similarity across all groups.

    Returns (n_blocks,) float64 array of averages, or None if no per-block data.
    """
    all_means = []
    n_blocks = None
    for g in groups:
        if g.block_cos_sim is not None:
            means = np.nanmean(g.block_cos_sim, axis=1)  # (n_blocks,)
            all_means.append(means)
            if n_blocks is None:
                n_blocks = g.block_cos_sim.shape[0]
    if not all_means:
        return None
    stacked = np.stack(all_means, axis=-1)  # (n_blocks, n_groups)
    return np.nanmean(stacked, axis=-1)  # (n_blocks,)


def _inject_data_driven_block_params(
    cfg: TeacacheConfig,
    block_cosim_means: np.ndarray,  # (n_blocks,) precomputed mean cos_sim per block
) -> None:
    """Inject learned block_params from per-block cosine similarity means.

    For split_groups: partitions blocks into 3 groups and classifies each as
    always-run or cacheable based on cosim_threshold. Stores always_groups
    and cache_groups (group indices) in cfg.block_params.

    For dynamic (unified): computes per-block sensitivity multipliers
    normalized so 1.0 = average sensitivity.
    """
    n_blocks = len(block_cosim_means)

    if cfg.block_mode == "split_groups":
        split1 = max(n_blocks // 3, 1)
        split2 = max(2 * n_blocks // 3, split1 + 1)
        boundaries = [(0, split1), (split1, split2), (split2, n_blocks)]
        boundaries = [(s, e) for s, e in boundaries if e > s]

        always_groups, cache_groups = [], []
        for gi, (s, e) in enumerate(boundaries):
            group_mean = float(np.nanmean(block_cosim_means[s:e]))
            if group_mean > cfg.cosim_threshold:
                cache_groups.append(gi)
            else:
                always_groups.append(gi)

        cfg.block_params["always_groups"] = always_groups
        cfg.block_params["cache_groups"] = cache_groups

    elif cfg.block_mode == "dynamic":
        avg_sensitivity = float(np.nanmean(1.0 - block_cosim_means))
        if avg_sensitivity > 1e-8:
            cfg.block_params["sensitivity"] = ((1.0 - block_cosim_means) / avg_sensitivity).tolist()


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
    dyn_bf_weighted = 0.0
    dyn_skip_total = 0

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
        skip, err, mask = simulate_group(
            predicted, thresholds, penalties,
            cfg.accumulation_type, cfg.accumulation_params,
        )

        total_skip += skip
        total_error += err
        total_steps += group.n_steps

        if cfg.block_mode == "dynamic":
            dyn_bf = _compute_dynamic_block_fraction(
                cfg, group.block_cos_sim, mask,
            )
            dyn_bf_weighted += dyn_bf * skip
            dyn_skip_total += skip

    if total_steps == 0:
        return 0.0, 0.0, 1.0, 1.0

    skip_rate = total_skip / total_steps
    avg_error = total_error / total_steps
    quality_proxy = 1.0 / (1.0 + avg_error)

    if cfg.block_mode == "dynamic" and dyn_skip_total > 0:
        bf = dyn_bf_weighted / dyn_skip_total
    else:
        bf = _block_fraction(cfg)

    speedup = (1.0 / (1.0 - skip_rate * bf)
               if bf > 0 and skip_rate * bf < 1.0 else 1.0)

    return skip_rate, avg_error, speedup, quality_proxy


def _compute_dynamic_block_fraction(
    cfg: TeacacheConfig,
    block_cos_sim: Optional[np.ndarray],  # (n_blocks, n_steps) or None
    skip_mask: np.ndarray,                # bool (n_steps,)
    default: float = 0.85,
) -> float:
    """Compute block fraction using per-block cosine similarity at each skip step.

    On steps where the global accumulator says 'skip', blocks whose recorded
    cosine similarity exceeds cfg.cosim_threshold are considered cache-safe.
    The returned fraction represents the average compute saved per skip step,
    scaled by *default* (the fraction of total model compute in the blocks).
    """
    if block_cos_sim is None:
        return default

    n_blocks = block_cos_sim.shape[0]
    n_steps = min(len(skip_mask), block_cos_sim.shape[1])
    threshold = cfg.cosim_threshold

    total_cached_blocks = 0.0
    skip_count = 0
    for t in range(n_steps):
        if skip_mask[t]:
            skip_count += 1
            cached = 0
            for bi in range(n_blocks):
                cs = block_cos_sim[bi, t]
                if not np.isnan(cs) and cs > threshold:
                    cached += 1
            total_cached_blocks += cached / max(n_blocks, 1)

    if skip_count == 0:
        return 0.0
    return (total_cached_blocks / skip_count) * default
