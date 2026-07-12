"""TeaCache forward function with all 10 configurable knobs for Anima/Cosmos."""

from __future__ import annotations

import math
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
) -> tuple[float, bool]:
    """Update accumulated distance and decide should_calc.

    Returns (new_accumulated, should_calc).

    accum_type options:
        "hard_reset"   → acc += pred; if acc >= thresh: acc=0, should_calc=True
        "carry_over"   → acc += pred; if acc >= thresh: acc-=thresh, should_calc=True
        "leaky"        → acc = acc*decay + pred; if acc >= thresh: acc=0, should_calc=True
        "windowed"     → window.append(pred); if mean >= thresh: should_calc=True
    """
    if accum_params is None:
        accum_params = {}

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

    return 1.0


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
    if cfg.source == "first_block_shift":
        adaln_self = self.blocks[0].adaln_modulation_self_attn(t_embedding_B_T_D)
        if adaln_lora_B_T_3D is not None and getattr(self, "use_adaln_lora", False):
            adaln_self = adaln_self + adaln_lora_B_T_3D
        modulated_inp = adaln_self.chunk(3, dim=-1)[0].to(cache_device)
    elif cfg.source == "pooled_latent":
        modulated_inp = x_B_T_H_W_D.mean(dim=(1, 2)).to(cache_device)
    else:
        modulated_inp = t_embedding_B_T_D.to(cache_device)

    # ── 3. Per-slot state initialization ──
    cond_or_uncond = transformer_options.get("cond_or_uncond", [0, 1])
    b = int(x_B_T_H_W_D.shape[0] / len(cond_or_uncond))

    if not hasattr(self, "teacache_state"):
        self.teacache_state = {
            k: {
                "accumulated": 0.0,
                "should_calc": True,
                "prev_mod": None,
                "prev_residual": None,
            }
            for k in cond_or_uncond
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

            # Knob 7: Step schedule
            mult = step_schedule_multiplier(current_percent, cfg.step_schedule)
            effective_thresh = cfg.rel_l1_thresh * mult

            # Knob 5+6: Accumulation + threshold
            new_acc, should_calc = accumulate_distance(
                state["accumulated"],
                predicted,
                effective_thresh,
                cfg.accumulation_type,
                cfg.accumulation_params,
            )
            state["accumulated"] = new_acc
            state["should_calc"] = should_calc

        state["prev_mod"] = curr_mod

    # ── 5. Global decision ──
    enable_teacache = transformer_options.get("enable_teacache", True)
    if enable_teacache:
        should_calc_global = any(
            self.teacache_state[k].get("should_calc", True)
            for k in cond_or_uncond
        )
    else:
        should_calc_global = True

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
        # ── Knob 9: Apply residual ──
        mult = step_schedule_multiplier(current_percent, cfg.step_schedule)
        effective_thresh = cfg.rel_l1_thresh * mult

        for i, k in enumerate(cond_or_uncond):
            state = self.teacache_state[k]
            resid = state.get("prev_residual")
            if resid is not None:
                confidence = min(
                    state["accumulated"] / max(effective_thresh, 1e-8), 1.0
                )
                x_B_T_H_W_D[i * b : (i + 1) * b] = apply_residual(
                    x_B_T_H_W_D[i * b : (i + 1) * b],
                    resid,
                    cfg.residual_strategy,
                    confidence=confidence,
                    params=cfg.residual_params,
                )
    else:
        # ── Knob 8: Run blocks ──
        ori_x = x_B_T_H_W_D.to(cache_device)

        if cfg.block_mode == "all_or_nothing":
            for block in self.blocks:
                x_B_T_H_W_D = block(
                    x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, **block_kwargs
                )

        residual = x_B_T_H_W_D.to(cache_device) - ori_x
        for i, k in enumerate(cond_or_uncond):
            self.teacache_state[k]["prev_residual"] = residual[i * b : (i + 1) * b]

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
