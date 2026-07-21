"""TeaCache simulation kernels — Numba-accelerated when available.

Four accumulation strategies, each with a pure-Python fallback and a Numba @njit
variant.  The module-level dispatch selects the right backend at import time.
All functions produce bit-identical results (strict IEEE 754 float64, no fastmath).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from numba import njit  # type: ignore[import-untyped]
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# ═══════════════════════════════════════════════════════════════════════════
#  Pure-Python fallbacks (identical algorithm, ~50× slower than Numba)
# ═══════════════════════════════════════════════════════════════════════════

def _py_hard_reset(
    predicted: np.ndarray, thresholds: np.ndarray, penalties: np.ndarray,
) -> Tuple[int, float, np.ndarray]:
    n = len(predicted)
    skip, err, acc = 0, 0.0, 0.0
    mask = np.empty(n, dtype=bool)
    for i in range(n):
        acc += predicted[i]
        if acc >= thresholds[i]:
            acc = 0.0
            mask[i] = False
        else:
            skip += 1
            err += penalties[i]
            mask[i] = True
    return skip, err, mask


def _py_carry_over(
    predicted: np.ndarray, thresholds: np.ndarray, penalties: np.ndarray,
) -> Tuple[int, float, np.ndarray]:
    n = len(predicted)
    skip, err, acc = 0, 0.0, 0.0
    mask = np.empty(n, dtype=bool)
    for i in range(n):
        acc += predicted[i]
        if acc >= thresholds[i]:
            acc -= thresholds[i]
            mask[i] = False
        else:
            skip += 1
            err += penalties[i]
            mask[i] = True
    return skip, err, mask


def _py_leaky(
    predicted: np.ndarray, thresholds: np.ndarray, penalties: np.ndarray,
    leak_factor: float,
) -> Tuple[int, float, np.ndarray]:
    n = len(predicted)
    skip, err, acc = 0, 0.0, 0.0
    mask = np.empty(n, dtype=bool)
    for i in range(n):
        acc = acc * leak_factor + predicted[i]
        if acc >= thresholds[i]:
            acc = 0.0
            mask[i] = False
        else:
            skip += 1
            err += penalties[i]
            mask[i] = True
    return skip, err, mask


def _py_windowed(
    predicted: np.ndarray, thresholds: np.ndarray, penalties: np.ndarray,
    window_size: int,
) -> Tuple[int, float, np.ndarray]:
    n = len(predicted)
    window = np.zeros(window_size, dtype=np.float64)
    w_idx, w_count, w_sum = 0, 0, 0.0
    min_w = max(2, window_size // 2)
    skip, err = 0, 0.0
    mask = np.empty(n, dtype=bool)

    for i in range(n):
        if w_count == window_size:
            w_sum -= window[w_idx]
        else:
            w_count += 1
        window[w_idx] = predicted[i]
        w_sum += predicted[i]
        w_idx = (w_idx + 1) % window_size

        avg = w_sum / w_count
        if avg >= thresholds[i] and w_count >= min_w:
            w_sum = 0.0
            w_count = 0
            window[:] = 0.0
            mask[i] = False
        else:
            skip += 1
            err += penalties[i]
            mask[i] = True
    return skip, err, mask


# ═══════════════════════════════════════════════════════════════════════════
#  Numba-compiled variants
# ═══════════════════════════════════════════════════════════════════════════

if _HAS_NUMBA:

    @njit(fastmath=False, cache=True)
    def _nb_hard_reset(predicted, thresholds, penalties):  # pragma: no cover
        n = len(predicted)
        skip, err, acc = 0, 0.0, 0.0
        mask = np.empty(n, dtype=np.bool_)
        for i in range(n):
            acc += predicted[i]
            if acc >= thresholds[i]:
                acc = 0.0
                mask[i] = False
            else:
                skip += 1
                err += penalties[i]
                mask[i] = True
        return skip, err, mask

    @njit(fastmath=False, cache=True)
    def _nb_carry_over(predicted, thresholds, penalties):  # pragma: no cover
        n = len(predicted)
        skip, err, acc = 0, 0.0, 0.0
        mask = np.empty(n, dtype=np.bool_)
        for i in range(n):
            acc += predicted[i]
            if acc >= thresholds[i]:
                acc -= thresholds[i]
                mask[i] = False
            else:
                skip += 1
                err += penalties[i]
                mask[i] = True
        return skip, err, mask

    @njit(fastmath=False, cache=True)
    def _nb_leaky(predicted, thresholds, penalties, leak_factor):  # pragma: no cover
        n = len(predicted)
        skip, err, acc = 0, 0.0, 0.0
        mask = np.empty(n, dtype=np.bool_)
        for i in range(n):
            acc = acc * leak_factor + predicted[i]
            if acc >= thresholds[i]:
                acc = 0.0
                mask[i] = False
            else:
                skip += 1
                err += penalties[i]
                mask[i] = True
        return skip, err, mask

    @njit(fastmath=False, cache=True)
    def _nb_windowed(predicted, thresholds, penalties, window_size):  # pragma: no cover
        n = len(predicted)
        window = np.zeros(window_size, dtype=np.float64)
        w_idx, w_count, w_sum = 0, 0, 0.0
        min_w = max(2, window_size // 2)
        skip, err = 0, 0.0
        mask = np.empty(n, dtype=np.bool_)
        for i in range(n):
            if w_count == window_size:
                w_sum -= window[w_idx]
            else:
                w_count += 1
            window[w_idx] = predicted[i]
            w_sum += predicted[i]
            w_idx = (w_idx + 1) % window_size
            avg = w_sum / w_count
            if avg >= thresholds[i] and w_count >= min_w:
                w_sum = 0.0
                w_count = 0
                window[:] = 0.0
                mask[i] = False
            else:
                skip += 1
                err += penalties[i]
                mask[i] = True
        return skip, err, mask

    _DISPATCH = {
        "hard_reset": _nb_hard_reset,
        "carry_over": _nb_carry_over,
        "leaky":      _nb_leaky,
        "windowed":   _nb_windowed,
    }
else:
    _DISPATCH = {
        "hard_reset": _py_hard_reset,
        "carry_over": _py_carry_over,
        "leaky":      _py_leaky,
        "windowed":   _py_windowed,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════

def simulate_group(
    predicted: np.ndarray,   # (G,) float64 — already scaled + mapped
    thresholds: np.ndarray,   # (G,) float64 — effective_thresh per step
    penalties: np.ndarray,    # (G,) float64 — merged out_rel / res_rel
    accum_type: str,
    accum_params: dict,
) -> Tuple[int, float, np.ndarray]:
    """Simulate one group's accumulation.  Returns (skip_count, total_error, skip_mask)."""
    fn = _DISPATCH[accum_type]
    if accum_type == "leaky":
        return fn(predicted, thresholds, penalties,
                  float(accum_params.get("leak_factor", 0.9)))
    if accum_type == "windowed":
        return fn(predicted, thresholds, penalties,
                  int(accum_params.get("window_size", 5)))
    return fn(predicted, thresholds, penalties)
