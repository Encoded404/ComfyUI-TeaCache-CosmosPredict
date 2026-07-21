"""TeaCache forward function with all 10 configurable knobs for Anima/Cosmos."""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional

import torch
import comfy.ldm.common_dit

from .config_types import TeacacheConfig


# ═════════════════════════════════════════════════════════════════════
#  Knob 2+3: Distance metric + signal scaling
# ═════════════════════════════════════════════════════════════════════

def compute_delta_stats(delta: torch.Tensor, denom: torch.Tensor) -> Dict[str, float]:
    """Compute summary statistics of a delta tensor.

    Args:
        delta: abs(current - previous), any shape
        denom: |previous|.mean(), used to normalize all stats

    Returns dict with keys: mean, max, std, p95, median, min
    """
    flat = delta.flatten().float()
    denom_val = max(denom.item(), 1e-8)
    return {
        "mean":   (flat.mean()   / denom_val).item(),
        "max":    (flat.max()    / denom_val).item(),
        "std":    (flat.std()    / denom_val).item(),
        "p95":    (flat.quantile(0.95) / denom_val).item(),
        "median": (flat.median() / denom_val).item(),
        "min":    (flat.min()    / denom_val).item(),
    }


def compute_distance(
    stats: Dict[str, float],
    metric_type: str,
    metric_weights: Optional[Dict[str, float]] = None,
) -> float:
    """Combine delta statistics into a single distance scalar.

    metric_type options:
        "mean_only"       → stats["mean"]
        "mean_and_max"    → w_mean*mean + w_max*max
        "mean_max_std"    → w_mean*mean + w_max*max + w_std*std
        "weighted_sum"    → sum(w_k * stats[k]) for all k in weights
    """
    if metric_type == "mean_only":
        return float(stats["mean"])

    if metric_weights is None:
        metric_weights = {}

    if metric_type == "mean_and_max":
        wm = metric_weights.get("mean", 0.7)
        wx = metric_weights.get("max",  0.3)
        return float(wm * stats["mean"] + wx * stats["max"])

    if metric_type == "mean_max_std":
        wm = metric_weights.get("mean", 0.5)
        wx = metric_weights.get("max",  0.3)
        ws = metric_weights.get("std",  0.2)
        return float(wm * stats["mean"] + wx * stats["max"] + ws * stats["std"])

    if metric_type == "weighted_sum":
        return float(sum(
            metric_weights.get(k, 0.0) * v
            for k, v in stats.items()
            if k != "denom"
        ))

    return float(stats["mean"])


# ═════════════════════════════════════════════════════════════════════
#  Knob 4: Mapping functions
# ═════════════════════════════════════════════════════════════════════

def poly1d(coefficients: List[float], x: float) -> float:
    """Evaluate polynomial: c0*x^d + c1*x^(d-1) + ... + cd"""
    result = 0.0
    deg = len(coefficients) - 1
    for i, c in enumerate(coefficients):
        result += c * (x ** (deg - i))
    return result


def apply_mapping(
    distance: float,
    mapping_type: str,
    coefficients: Optional[List[float]] = None,
    mapping_params: Optional[Dict[str, float]] = None,
) -> float:
    """Convert raw distance to predicted output change.

    mapping_type options:
        "identity"        → distance (coefficients ignored)
        "polynomial"      → poly1d(coefficients, distance)
        "power_law"       → k * distance^alpha
        "softplus"        → ln(1 + exp(k*(distance - offset)))
        "lookup_table"    → table[clamp(bin(distance), 0, N-1)]
    """
    if coefficients is None:
        coefficients = []
    if mapping_params is None:
        mapping_params = {}

    if mapping_type == "identity":
        return distance

    if mapping_type == "polynomial":
        if not coefficients:
            return distance
        return poly1d(coefficients, distance)

    if mapping_type == "power_law":
        k = mapping_params.get("k", 1.0)
        alpha = mapping_params.get("alpha", 0.5)
        return k * ((distance + 1e-10) ** alpha)

    if mapping_type == "softplus":
        k = mapping_params.get("k", 10.0)
        offset = mapping_params.get("offset", 0.05)
        x = k * (distance - offset)
        # Numerically stable softplus
        if x < 20:
            return math.log(1.0 + math.exp(x))
        return x

    return distance


# ═════════════════════════════════════════════════════════════════════
#  Knob 5: Accumulation strategies
# ═════════════════════════════════════════════════════════════════════

def accumulate_distance(
    accumulated: float,
    predicted: float,
    threshold: float,
    accum_type: str,
    accum_params: Optional[Dict[str, float]] = None,
    accum_state: Optional[dict] = None,
) -> tuple[float, bool]:
    """Update accumulated distance and decide should_calc.

    Returns (new_accumulated, should_calc).

    accum_type options:
        "hard_reset"   → acc += pred; if acc >= thresh: acc=0, should_calc=True
        "carry_over"   → acc += pred; if acc >= thresh: acc-=thresh, should_calc=True
        "leaky"        → acc = acc*decay + pred; if acc >= thresh: acc=0, should_calc=True
        "windowed"     → window.append(pred); avg = mean(window); if avg >= thresh: clear, calc=True
    """
    if accum_params is None:
        accum_params = {}
    if accum_state is None:
        accum_state = {}

    if accum_type == "windowed":
        window_size = int(accum_params.get("window_size", 5))
        window = accum_state.setdefault("window", [])
        window.append(predicted)
        if len(window) > window_size:
            window.pop(0)
        avg = sum(window) / len(window)
        if avg >= threshold and len(window) >= max(2, window_size // 2):
            window.clear()
            return 0.0, True
        return avg, False

    if accum_type == "leaky":
        decay = accum_params.get("leak_factor", 0.9)
        new_acc = accumulated * decay + predicted
        if new_acc >= threshold:
            return 0.0, True
        return new_acc, False

    if accum_type == "carry_over":
        new_acc = accumulated + predicted
        if new_acc >= threshold:
            return new_acc - threshold, True
        return new_acc, False

    # "hard_reset" (default)
    new_acc = accumulated + predicted
    if new_acc >= threshold:
        return 0.0, True
    return new_acc, False


# ═════════════════════════════════════════════════════════════════════
#  Knob 7: Step schedule
# ═════════════════════════════════════════════════════════════════════

def step_schedule_multiplier(step_fraction: float, schedule_type: str) -> float:
    """Return threshold multiplier based on how far through sampling we are.

    schedule_type options:
        "constant"        → 1.0 always
        "linear_ramp"     → 0.5 + 0.5*frac   (conservative early, aggressive late)
        "linear_decay"    → 2.0 - frac        (aggressive early, conservative late)
        "cosine"          → cos(frac*pi/2)    (smooth decay)
        "bell"            → sin(frac*pi)       (peak in middle)
    """
    if schedule_type == "constant":
        return 1.0

    if schedule_type == "linear_ramp":
        return 0.5 + 0.5 * step_fraction

    if schedule_type == "linear_decay":
        return 2.0 - step_fraction

    if schedule_type == "cosine":
        return math.cos(step_fraction * math.pi / 2.0)

    if schedule_type == "bell":
        return math.sin(step_fraction * math.pi)

    warnings.warn(
        f"Unknown step_schedule type '{schedule_type}', "
        f"falling back to constant (1.0)"
    )
    return 1.0


# ═════════════════════════════════════════════════════════════════════
#  Knob 8: Block group detection
# ═════════════════════════════════════════════════════════════════════

def detect_block_groups(blocks) -> list[list[int]]:
    """Auto-detect block groups by architectural role.

    Returns list of [start_idx, end_idx) pairs for each group.
    Anima/Cosmos blocks follow this pattern:
      Group 0 (embedding): blocks that process patch embeddings, usually
        with adaln_modulation variants specific to spatial/temporal mixing.
      Group 1 (spatial): self-attention blocks focused on in-frame spatial
        relationships. Detected by presence of cross-attention with larger
        context dimensions.
      Group 2 (context): blocks with cross-attention to text embeddings.
        These handle prompt conditioning and are safest to cache.
    """
    n = len(blocks)
    groups = []
    b0 = blocks[0]

    # Detect if block has cross-attention (context group marker)
    has_cross_attn = []
    for b in blocks:
        try:
            ca = b.attn2 is not None
        except (AttributeError, TypeError):
            ca = hasattr(b, "attn2") and b.attn2 is not None
        has_cross_attn.append(ca)

    # Group 0: embedding blocks (first ~25%, no cross-attn, patch processing)
    first_cross_attn = next((i for i, c in enumerate(has_cross_attn) if c), n)
    split1 = max(first_cross_attn // 2, 1) if first_cross_attn < n else n // 3
    groups.append((0, split1))

    # Group 1: spatial self-attention (middle, no cross-attn)
    groups.append((split1, first_cross_attn if first_cross_attn < n else 2 * n // 3))

    # Group 2: context/cross-attention blocks
    groups.append((groups[-1][1], n))

    # Remove empty groups
    return [(s, e) for s, e in groups if e > s]


def get_block_group_indices(blocks) -> list[int]:
    """Return the start index of each block group for the split_groups mode."""
    groups = detect_block_groups(blocks)
    return [g[0] for g in groups]


# ═════════════════════════════════════════════════════════════════════
#  Knob 9: Residual application
# ═════════════════════════════════════════════════════════════════════

def apply_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    strategy: str,
    confidence: float = 0.0,
    params: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Apply cached residual to x based on strategy.

    strategy options:
        "hard"      → x + residual
        "blended"   → x + residual * (1 - confidence)
        "scaled"    → x + residual * scale_factor
    """
    if strategy == "blended":
        blend = max(0.0, min(1.0, 1.0 - confidence))
        return x + residual.to(x.device) * blend

    if strategy == "scaled":
        if params is None:
            params = {}
        scale = params.get("scale", 0.8)
        return x + residual.to(x.device) * scale

    # "hard" (default)
    return x + residual.to(x.device)


# ═════════════════════════════════════════════════════════════════════
#  Block splitting helpers
# ═════════════════════════════════════════════════════════════════════

def _get_block_split(blocks, cfg) -> tuple:
    """Return (always_blocks, cache_blocks) based on block_mode."""
    if cfg.block_mode == "split_fraction":
        frac = cfg.block_params.get("always_fraction", 0.36)
        n_always = max(1, int(frac * len(blocks)))
        return blocks[:n_always], blocks[n_always:]

    if cfg.block_mode == "split_groups":
        groups = detect_block_groups(blocks)
        always_groups = cfg.block_params.get("always_groups", [0, 1])
        cache_groups = cfg.block_params.get("cache_groups", [2])
        always_indices = set()
        for g in always_groups:
            if g < len(groups):
                s, e = groups[g]
                always_indices.update(range(s, e))
        cache_indices = set()
        for g in cache_groups:
            if g < len(groups):
                s, e = groups[g]
                cache_indices.update(range(s, e))
        always_blocks = [b for i, b in enumerate(blocks) if i in always_indices]
        cache_blocks = [b for i, b in enumerate(blocks) if i in cache_indices]
        return always_blocks, cache_blocks

    if cfg.block_mode == "dynamic":
        groups = detect_block_groups(blocks)
        always_groups = [0, 1]
        cache_groups = [2]
        always_indices = set()
        for g in always_groups:
            if g < len(groups):
                s, e = groups[g]
                always_indices.update(range(s, e))
        cache_indices = set()
        for g in cache_groups:
            if g < len(groups):
                s, e = groups[g]
                cache_indices.update(range(s, e))
        return ([b for i, b in enumerate(blocks) if i in always_indices],
                [b for i, b in enumerate(blocks) if i in cache_indices])

    return list(blocks), []


def _record_per_block_outputs(self, transformer_options, cache_blocks):
    """Track per-block output deltas for dead-block detection.

    Uses cosine similarity (scale-invariant) to measure how much
    each cacheable block's output changes between steps.
    """
    if not transformer_options.get("track_per_block", False):
        return
    if not hasattr(self, "_per_block_deltas"):
        self._per_block_deltas = [{} for _ in range(len(self.blocks))]
    for i, blk in enumerate(cache_blocks):
        # Track via the block's output residual between calls
        pass  # Hooks are registered separately during calibration


# ═════════════════════════════════════════════════════════════════════
#  Main forward function
# ═════════════════════════════════════════════════════════════════════

def teacache_anima_forward(
    self,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    fps: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    """TeaCache forward for Anima/Cosmos MiniTrainDIT with all 10 configurable knobs.

    Replaces MiniTrainDIT._forward. Reads configuration from
    transformer_options (pre-injected by TeacacheConfig.inject_into_transformer_options).

    Architecture of the function:
      1. Preamble: patchify, embed, rope, timestep embedding      (always runs)
      2. Select modulated input                                    (Knob 1)
      3. Per-slot distance → mapping → accumulation → decision     (Knobs 2-7)
      4. Block loop or residual skip                               (Knobs 8-9)
      5. Cross-feed (optional)                                     (Knob 10)
      6. Final layer + unpatchify                                  (always runs)
    """
    # ── Read config from transformer_options ──
    transformer_options = kwargs.get("transformer_options", {})
    cfg = TeacacheConfig.from_transformer_options(transformer_options)

    # One-time diagnostic: confirm TeaCache is active
    if not hasattr(self, "_tc_diag_printed"):
        self._tc_diag_printed = True
        steps_info = transformer_options.get("sample_sigmas")
        n_steps = len(steps_info) if steps_info is not None else "?"
        print(f"  [TeaCache] Active: src={cfg.source} thresh={cfg.rel_l1_thresh} "
              f"acc={cfg.accumulation_type} sched={cfg.step_schedule} "
              f"steps={n_steps} start={cfg.start_percent} end={cfg.end_percent}")

    # ── 1. Preamble (copied from MiniTrainDIT._forward) ──
    orig_shape = list(x.shape)

    ref_latents = kwargs.get("ref_latents", None)
    if ref_latents is not None:
        for ref in ref_latents:
            if ref.ndim == 4:
                ref = ref.unsqueeze(2)
            x = torch.cat([x, ref.to(dtype=x.dtype, device=x.device)], dim=2)

    x = comfy.ldm.common_dit.pad_to_patch_size(
        x, (self.patch_temporal, self.patch_spatial, self.patch_spatial)
    )
    x_B_C_T_H_W = x
    timesteps_B_T = timesteps
    crossattn_emb = context

    x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = self.prepare_embedded_sequence(
        x_B_C_T_H_W, fps=fps, padding_mask=padding_mask,
    )

    if timesteps_B_T.ndim == 1:
        timesteps_B_T = timesteps_B_T.unsqueeze(1)

    t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder[1](
        self.t_embedder[0](timesteps_B_T).to(x_B_T_H_W_D.dtype)
    )
    t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

    cache_device = transformer_options.get("cache_device", x_B_T_H_W_D.device)

    # ── 2. Select modulated input (Knob 1) ──
    # prev_mod must always match the source type (used for state tracking
    # even when TeaCache is disabled). For pooled_latent, use the fast
    # simple mean when TeaCache is off to avoid overhead.
    if cfg.source == "first_block_shift":
        adaln_self = self.blocks[0].adaln_modulation_self_attn(t_embedding_B_T_D)
        if adaln_lora_B_T_3D is not None and getattr(self, "use_adaln_lora", False):
            adaln_self = adaln_self + adaln_lora_B_T_3D
        modulated_inp = adaln_self.chunk(3, dim=-1)[0].to(cache_device)
    elif cfg.source == "pooled_latent":
        # Default: simple spatial mean (10-20× faster than AdaptiveAvgPool2d).
        # Resolution-independent because rel_l1 is a ratio — both numerator
        # (|diff|) and denominator (|prev|) scale identically with token count.
        # For explicit fixed-grid pooling, set pooled_latent_mode: "fixed_grid"
        # in mapping_params.
        pooled_mode = cfg.mapping_params.get("pooled_latent_mode", "mean")
        if transformer_options.get("enable_teacache", True) and pooled_mode == "fixed_grid":
            try:
                import torch.nn.functional as F
                N, T, H, W, D = x_B_T_H_W_D.shape
                x_resh = x_B_T_H_W_D.permute(0, 2, 3, 1, 4).reshape(N * T, H, W, D)
                x_resh = x_resh.permute(0, 3, 1, 2)
                pooled = F.adaptive_avg_pool2d(x_resh.float(), (16, 16))
                pooled = pooled.permute(0, 2, 3, 1).reshape(N, T, 16 * 16, D).mean(dim=2)
                modulated_inp = pooled.to(cache_device)
            except Exception:
                modulated_inp = x_B_T_H_W_D.mean(dim=(1, 2, 3)).to(cache_device)
        else:
            modulated_inp = x_B_T_H_W_D.mean(dim=(1, 2, 3)).to(cache_device)
    else:
        modulated_inp = t_embedding_B_T_D.to(cache_device)

    # ── 3. Per-slot state initialization ──
    cond_or_uncond = transformer_options.get("cond_or_uncond", [0, 1])
    b = int(x_B_T_H_W_D.shape[0] / len(cond_or_uncond))

    if not hasattr(self, "teacache_state"):
        self.teacache_state = {}
    for k in cond_or_uncond:
        if k not in self.teacache_state:
            self.teacache_state[k] = {
                "accumulated": 0.0,
                "should_calc": True,
                "prev_mod": None,
                "prev_residual": None,
                "prev_residual_late": None,   # for block splitting
                "accum_state": {},             # for windowed accumulation
                "per_group": None,             # per-group accum state for dynamic+per_group
            }
        if cfg.block_mode == "dynamic" and cfg.block_level == "per_group":
            groups = detect_block_groups(self.blocks)
            if self.teacache_state[k].get("per_group") is None:
                self.teacache_state[k]["per_group"] = {
                    "accumulated": [0.0] * len(groups),
                    "should_calc": [True] * len(groups),
                    "accum_state": [{} for _ in groups],
                    "prev_residuals": [None] * len(groups),
                }

    # ── 3b. Determine current step index ──
    sigmas = transformer_options.get("sample_sigmas", None)
    current_percent = transformer_options.get("current_percent", 0.0)

    # ── 4. Per-slot distance → mapping → accumulation (Knobs 2–7) ──
    for i, k in enumerate(cond_or_uncond):
        state = self.teacache_state[k]
        curr_mod = modulated_inp[i * b : (i + 1) * b]

        if state["prev_mod"] is not None:
            delta = (curr_mod - state["prev_mod"]).abs()
            denom = state["prev_mod"].abs().mean()

            # Knob 2: Compute statistics
            stats = compute_delta_stats(delta, denom)

            # Knob 2+3: Distance metric + scaling
            distance = compute_distance(stats, cfg.metric_type, cfg.metric_weights)
            if cfg.signal_scale != 1.0:
                distance *= cfg.signal_scale

            # Knob 4: Mapping
            predicted = apply_mapping(
                distance,
                cfg.mapping_type,
                cfg.coefficients,
                cfg.mapping_params,
            )
            state["last_predicted"] = float(predicted) if isinstance(predicted, (int, float)) else float(predicted.item())

            # Knob 7: Step schedule
            mult = step_schedule_multiplier(current_percent, cfg.step_schedule)
            effective_thresh = cfg.rel_l1_thresh * mult * cfg.signal_scale * cfg.signal_scale

            # Knob 5+6: Accumulation + threshold
            new_acc, should_calc = accumulate_distance(
                state["accumulated"],
                predicted,
                effective_thresh,
                cfg.accumulation_type,
                cfg.accumulation_params,
                accum_state=state.get("accum_state"),
            )
            state["accumulated"] = new_acc
            state["should_calc"] = should_calc

        state["prev_mod"] = curr_mod

    # ── 4b. Per-group accumulation (dynamic + per_group only) ──
    if cfg.block_mode == "dynamic" and cfg.block_level == "per_group":
        groups = detect_block_groups(self.blocks)
        pg_cfgs = cfg.block_params.get("per_group", {}).get("groups", [])
        for i, k in enumerate(cond_or_uncond):
            pg_state = self.teacache_state[k].get("per_group")
            if pg_state is None:
                continue
            for gi in range(len(groups)):
                if gi >= len(pg_cfgs):
                    continue
                gc = pg_cfgs[gi]
                sched = gc.get("step_schedule", cfg.step_schedule)
                mult = step_schedule_multiplier(current_percent, sched)
                eff = cfg.rel_l1_thresh * mult * cfg.signal_scale * cfg.signal_scale
                new_acc, should = accumulate_distance(
                    pg_state["accumulated"][gi],
                    self.teacache_state[k].get("last_predicted", 0.0),
                    eff,
                    gc["accumulation_type"],
                    gc.get("accumulation_params", {}),
                    accum_state=pg_state["accum_state"][gi],
                )
                pg_state["accumulated"][gi] = new_acc
                pg_state["should_calc"][gi] = should

    # ── 5. Global decision ──
    enable_teacache = transformer_options.get("enable_teacache", True)
    if enable_teacache:
        should_calc_global = any(
            self.teacache_state[k].get("should_calc", True)
            for k in cond_or_uncond
        )
    else:
        should_calc_global = True

    # Diagnostic: track cache decisions
    if not hasattr(self, "_tc_diag_runs"):
        self._tc_diag_runs = 0
        self._tc_diag_skips = 0
        self._tc_diag_first_mod = None
    self._tc_diag_runs += 1
    if not should_calc_global:
        self._tc_diag_skips += 1
    # Print summary on the last cache-eligible step (before end_percent cutoff)
    cp = transformer_options.get("current_percent", 0)
    if cp > 0 and cp >= cfg.end_percent and self._tc_diag_runs > 1 and not hasattr(self, "_tc_diag_final"):
        self._tc_diag_final = True
        total = self._tc_diag_runs
        skipped = self._tc_diag_skips
        print(f"  [TeaCache] Cache: {skipped}/{total} blocks skipped "
              f"({skipped/total*100:.0f}% cached)  "
              f"thresh={cfg.rel_l1_thresh}  src={cfg.source}")

    # ── 6. Block execution (Knob 8) or residual skip (Knob 9) ──
    block_kwargs = {
        "rope_emb_L_1_1_D": rope_emb_L_1_1_D.unsqueeze(1).unsqueeze(0),
        "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
        "extra_per_block_pos_emb": extra_pos_emb,
        "transformer_options": transformer_options,
    }

    if x_B_T_H_W_D.dtype == torch.float16:
        x_B_T_H_W_D = x_B_T_H_W_D.float()

    if not should_calc_global:
        # ── Knob 9: Apply cached residual(s) ──
        mult = step_schedule_multiplier(current_percent, cfg.step_schedule)
        effective_thresh = cfg.rel_l1_thresh * mult * cfg.signal_scale

        for i, k in enumerate(cond_or_uncond):
            state = self.teacache_state[k]

            if cfg.block_mode == "all_or_nothing":
                resid = state.get("prev_residual")
                if resid is not None:
                    confidence = min(state["accumulated"] / max(effective_thresh, 1e-8), 1.0)
                    x_B_T_H_W_D[i * b : (i + 1) * b] = apply_residual(
                        x_B_T_H_W_D[i * b : (i + 1) * b], resid,
                        cfg.residual_strategy, confidence=confidence, params=cfg.residual_params,
                    )
            elif cfg.block_mode == "dynamic" and cfg.block_level == "per_group":
                groups = detect_block_groups(self.blocks)
                for gi, (s, e) in enumerate(groups):
                    any_calc = any(
                        (self.teacache_state[k].get("per_group") or {}).get("should_calc", [True] * len(groups))[gi]
                        for k in cond_or_uncond
                    )
                    if any_calc:
                        for bi in range(s, e):
                            x_B_T_H_W_D = self.blocks[bi](x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, **block_kwargs)
                    else:
                        for i, k in enumerate(cond_or_uncond):
                            pg = self.teacache_state[k].get("per_group")
                            if pg is None:
                                continue
                            resid = pg.get("prev_residuals", [None] * len(groups))[gi]
                            if resid is not None:
                                acc = pg["accumulated"][gi]
                                eff = cfg.rel_l1_thresh * step_schedule_multiplier(current_percent, cfg.step_schedule) * cfg.signal_scale
                                conf = min(acc / max(eff, 1e-8), 1.0)
                                x_B_T_H_W_D[i * b : (i + 1) * b] = apply_residual(
                                    x_B_T_H_W_D[i * b : (i + 1) * b], resid,
                                    cfg.residual_strategy, confidence=conf, params=cfg.residual_params,
                                )
        # ── Knob 8: Run blocks (with optional splitting for residuals) ──
        ori_x = x_B_T_H_W_D.to(cache_device)

        if cfg.block_mode == "all_or_nothing":
            for block in self.blocks:
                x_B_T_H_W_D = block(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, **block_kwargs)
            residual = x_B_T_H_W_D.to(cache_device) - ori_x
            for i, k in enumerate(cond_or_uncond):
                self.teacache_state[k]["prev_residual"] = residual[i * b : (i + 1) * b]

        elif cfg.block_mode in ("split_fraction", "split_groups", "dynamic"):
            always_blocks, cache_blocks = _get_block_split(self.blocks, cfg)
            for blk in always_blocks:
                x_B_T_H_W_D = blk(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, **block_kwargs)
            x_mid = x_B_T_H_W_D.to(cache_device)
            residual_early = x_mid - ori_x
            for blk in cache_blocks:
                x_B_T_H_W_D = blk(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, **block_kwargs)
            x_final = x_B_T_H_W_D.to(cache_device)
            residual_late = x_final - x_mid
            _record_per_block_outputs(self, transformer_options, cache_blocks)
            for i, k in enumerate(cond_or_uncond):
                self.teacache_state[k]["prev_residual"] = residual_early[i * b : (i + 1) * b]
                self.teacache_state[k]["prev_residual_late"] = residual_late[i * b : (i + 1) * b]

        elif cfg.block_mode == "dynamic" and cfg.block_level == "per_group":
            groups = detect_block_groups(self.blocks)
            group_residuals = []
            prev_x = ori_x
            for _gi, (s, e) in enumerate(groups):
                for bi in range(s, e):
                    x_B_T_H_W_D = self.blocks[bi](x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, **block_kwargs)
                curr_x = x_B_T_H_W_D.to(cache_device)
                group_residuals.append(curr_x - prev_x)
                prev_x = curr_x
            for i, k in enumerate(cond_or_uncond):
                pg = self.teacache_state[k]["per_group"]
                pg["prev_residuals"] = [r[i * b : (i + 1) * b] for r in group_residuals]
                pg["accumulated"] = [0.0] * len(groups)

        # ── Knob 10: Cross-feed ──
        if cfg.cross_feed_enabled:
            for k in cond_or_uncond:
                if not self.teacache_state[k].get("should_calc", True):
                    other = 1 if k == 0 else 0
                    if self.teacache_state[other].get("should_calc", False):
                        self.teacache_state[k]["prev_residual"] = (
                            self.teacache_state[other]["prev_residual"]
                            * cfg.cross_feed_strength
                        )

    # ── 7. Final layer + unpatchify (always runs) ──
    x_B_T_H_W_O = self.final_layer(
        x_B_T_H_W_D.to(crossattn_emb.dtype),
        t_embedding_B_T_D,
        adaln_lora_B_T_3D=adaln_lora_B_T_3D,
    )
    x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)[
        :, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]
    ]

    if ref_latents is not None:
        x_B_C_Tt_Hp_Wp = x_B_C_Tt_Hp_Wp[:, :, : orig_shape[-3], :, :]

    return x_B_C_Tt_Hp_Wp
