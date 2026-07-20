"""Immutable precomputed simulation data from calibration entries.

Converts List[CalibrationEntry] (Array-of-Structs) into per-group Struct-of-Arrays
for zero-allocation simulation.  Grouping and sorting happen once at construction
time — simulate_config never repeats this work.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config_types import CalibrationEntry

_STAT_FIELDS = ("mean", "max", "std", "p95", "median", "min", "denom")


@dataclass(frozen=True, slots=True)
class GroupData:
    """Immutable precomputed stats for one (prompt_id, seed, cond, total_steps) group.

    All arrays are float64 with shape (n_steps,).  Entries are ordered by step
    index so sequential accumulation exactly replays the calibration timeline.
    """

    n_steps: int
    prompt_id: int
    seed: int
    cond: int  # 0 or 1 for cond / uncond CFG slot

    step_fraction: np.ndarray  # (n,)  ∈ [0, 1]
    out_rel: np.ndarray        # (n,)  ground-truth output change (quality penalty)
    res_rel: np.ndarray        # (n,)  fallback penalty when out_rel ≤ 0

    # Per-source delta statistics — None if this source wasn't recorded.
    t_emb_stats: Optional[Dict[str, np.ndarray]]   # {"mean": (n,), "max": (n,), ...}
    shift_stats: Optional[Dict[str, np.ndarray]]
    latent_stats: Optional[Dict[str, np.ndarray]]

    # Precomputed threshold multipliers for all 5 schedule types
    step_mult: Dict[str, np.ndarray]  # {"constant": (n,), "cosine": (n,), ...}


@dataclass(frozen=True, slots=True)
class SimData:
    """Immutable collection of all groups.  Constructed once at optimize() startup
    and shared read-only across multiprocessing workers via pickle."""

    groups: Tuple[GroupData, ...]
    n_entries: int

    @classmethod
    def from_entries(cls, entries: List[CalibrationEntry]) -> "SimData":
        """Group, sort, and precompute numpy arrays from calibration entries.

        Groups are formed by (prompt_id, seed, cond, total_steps).  Each group is
        sorted by step index so that sequential accumulation replays the timeline.
        """
        # 1. Group entries
        bucket: Dict[Tuple[int, int, int, int], List[CalibrationEntry]] = defaultdict(list)
        for e in entries:
            key = (e.prompt_id, e.seed, e.cond, e.total_steps)
            bucket[key].append(e)

        groups: List[GroupData] = []
        total = 0

        for (prompt_id, seed, cond, total_steps), g_entries in bucket.items():
            # Sort by step
            g_entries.sort(key=lambda e: e.step)
            n = len(g_entries)
            total += n

            # Extract flat arrays
            step_frac = np.empty(n, dtype=np.float64)
            out_rel = np.empty(n, dtype=np.float64)
            res_rel = np.empty(n, dtype=np.float64)

            t_emb = _extract_stats(g_entries, "t_emb")
            shift = _extract_stats(g_entries, "shift")
            latent = _extract_stats(g_entries, "latent")

            for i, e in enumerate(g_entries):
                step_frac[i] = e.step_fraction
                out_rel[i] = e.out_rel
                res_rel[i] = e.res_rel

            # Precompute step schedule multipliers
            step_mult = _compute_schedule_mults(step_frac)

            groups.append(GroupData(
                n_steps=n,
                prompt_id=prompt_id,
                seed=seed,
                cond=cond,
                step_fraction=step_frac,
                out_rel=out_rel,
                res_rel=res_rel,
                t_emb_stats=t_emb,
                shift_stats=shift,
                latent_stats=latent,
                step_mult=step_mult,
            ))

        return cls(groups=tuple(groups), n_entries=total)

    def filter_by_prompt_ids(self, keep_ids: set) -> "SimData":
        """Return new SimData containing only groups whose prompt_id is in keep_ids."""
        kept = tuple(g for g in self.groups if g.prompt_id in keep_ids)
        return SimData(groups=kept, n_entries=sum(g.n_steps for g in kept))


# ═══════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_stats(
    entries: List[CalibrationEntry], attr: str
) -> Optional[Dict[str, np.ndarray]]:
    """Extract DeltaStats fields for *attr* (e.g. 't_emb', 'shift', 'latent').

    Returns None if no entry in the group has this source available.
    """
    first = getattr(entries[0], attr)
    if first is None:
        # Check if any entry has it (mixed groups shouldn't happen but be safe)
        for e in entries:
            if getattr(e, attr) is not None:
                first = getattr(e, attr)
                break
        else:
            return None

    n = len(entries)
    result: Dict[str, np.ndarray] = {}
    for field in _STAT_FIELDS:
        arr = np.empty(n, dtype=np.float64)
        for i, e in enumerate(entries):
            src = getattr(e, attr)
            arr[i] = getattr(src, field, np.nan) if src is not None else np.nan
        result[field] = arr
    return result


def _compute_schedule_mults(step_frac: np.ndarray) -> Dict[str, np.ndarray]:
    """Precompute step-schedule multiplier arrays for all 5 schedule types."""
    return {
        "constant":     np.ones_like(step_frac, dtype=np.float64),
        "cosine":       np.cos(step_frac * np.pi / 2.0).astype(np.float64),
        "linear_ramp":  (0.5 + 0.5 * step_frac).astype(np.float64),
        "linear_decay": (2.0 - step_frac).astype(np.float64),
        "bell":         np.sin(step_frac * np.pi).astype(np.float64),
    }
