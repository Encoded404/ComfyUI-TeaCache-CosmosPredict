"""Calibration recorder: patches MiniTrainDIT._forward to record per-step statistics.

The recorded data captures delta statistics for ALL source signals simultaneously
plus ground truth output changes. This enables the offline optimizer (optimize.py)
to simulate any combination of knobs without touching the GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import comfy.ldm.common_dit

from .config_types import CalibrationEntry, DeltaStats, TeacacheConfig


def _make_delta_stats(delta: torch.Tensor, denom: torch.Tensor) -> DeltaStats:
    """Compute DeltaStats from a delta tensor."""
    flat = delta.flatten().float()
    d = max(denom.item(), 1e-8)
    return DeltaStats(
        mean   = (flat.mean()   / d).item(),
        max    = (flat.max()    / d).item(),
        std    = (flat.std()    / d).item(),
        p95    = (flat.quantile(0.95) / d).item(),
        median = (flat.median() / d).item(),
        min    = (flat.min()    / d).item(),
        denom  = d,
    )


def make_calibration_forward(source_hint: str = "all"):
    """Create a calibration-patched _forward() that records stats without caching.

    This forward ALWAYS runs all blocks but records:
      - Delta stats for t_emb, first_block_shift, and pooled_latent sources
      - Ground truth output change (out_rel) at each step
      - Residual change (res_rel) for validation

    Data is appended to self.calibration_log as CalibrationEntry dicts.
    """

    def _forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # ── Preamble (same as MiniTrainDIT._forward) ──
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

        t_emb, adaln_lora = self.t_embedder[1](
            self.t_embedder[0](timesteps_B_T).to(x_B_T_H_W_D.dtype)
        )
        t_emb = self.t_embedding_norm(t_emb)

        # Compute first_block_shift for recording
        shift_raw = self.blocks[0].adaln_modulation_self_attn(t_emb)
        if adaln_lora is not None and getattr(self, "use_adaln_lora", False):
            shift_raw = shift_raw + adaln_lora
        shift_val = shift_raw.chunk(3, dim=-1)[0]  # (B, D) — just the shift

        # Pooled latent — mean over all spatial dims (T=1, H, W)
        latent_pooled = x_B_T_H_W_D.mean(dim=(1, 2, 3))  # (B, D)

        to = kwargs.get("transformer_options", {})
        cache_device = to.get("cache_device", x_B_T_H_W_D.device)

        # ── Get step metadata ──
        cond_or_uncond = to.get("cond_or_uncond", [0, 1])
        b = int(x_B_T_H_W_D.shape[0] / len(cond_or_uncond))

        current_step = to.get("calibration_step", 0)
        total_steps  = to.get("calibration_total_steps", 30)
        step_fraction = current_step / max(total_steps - 1, 1)

        # ── Initialize calibration state ──
        # Always init both cond slots [0, 1] since CFG passes them
        # separately (cond_or_uncond can be [0] or [1] on different calls).
        if not hasattr(self, "_calib_state"):
            self._calib_state = {}
        for i, k in enumerate(cond_or_uncond):
            if k not in self._calib_state:
                self._calib_state[k] = {
                    "prev_t_emb": None,
                    "prev_shift": None,
                    "prev_latent": None,
                    "prev_out": None,
                    "prev_residual": None,
                }
        if not hasattr(self, "calibration_log"):
            self.calibration_log = []

        # ── Record delta stats for all sources ──
        for i, k in enumerate(cond_or_uncond):
            state = self._calib_state[k]
            c_t = t_emb.detach()[i * b : (i + 1) * b]
            c_s = shift_val.detach()[i * b : (i + 1) * b]
            c_l = latent_pooled.detach()[i * b : (i + 1) * b]

            if state["prev_t_emb"] is not None:
                t_emb_delta = (c_t - state["prev_t_emb"]).abs()
                t_emb_denom = state["prev_t_emb"].abs().mean()
                shift_delta = (c_s - state["prev_shift"]).abs()
                shift_denom = state["prev_shift"].abs().mean()
                latent_delta = (c_l - state["prev_latent"]).abs()
                latent_denom = state["prev_latent"].abs().mean()

                entry = CalibrationEntry(
                    step=current_step,
                    step_fraction=round(step_fraction, 6),
                    prompt_id=to.get("calibration_prompt_id", 0),
                    seed=to.get("calibration_seed", 0),
                    cond=int(k),
                    total_steps=total_steps,
                    t_emb=_make_delta_stats(t_emb_delta, t_emb_denom),
                    shift=_make_delta_stats(shift_delta, shift_denom),
                    latent=_make_delta_stats(latent_delta, latent_denom),
                )
                self.calibration_log.append(entry)

            state["prev_t_emb"]   = c_t
            state["prev_shift"]   = c_s
            state["prev_latent"]  = c_l

        # ── ALWAYS run all blocks (no caching during calibration) ──
        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D.unsqueeze(1).unsqueeze(0),
            "adaln_lora_B_T_3D": adaln_lora,
            "extra_per_block_pos_emb": extra_pos_emb,
            "transformer_options": to,
        }
        if x_B_T_H_W_D.dtype == torch.float16:
            x_B_T_H_W_D = x_B_T_H_W_D.float()

        ori_x = x_B_T_H_W_D.to(cache_device)

        # ── Per-block output tracking (if enabled) ──
        track_per_block = getattr(self, "_calib_track_per_block", False)
        if track_per_block:
            if not hasattr(self, "_calib_block_prevs"):
                self._calib_block_prevs = [None] * len(self.blocks)
                self._calib_block_currs = [None] * len(self.blocks)
                self._calib_block_deltas = []  # flat list of (bi, cos_sim) for this step
            self._calib_block_currs = [None] * len(self.blocks)

        for bi, block in enumerate(self.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D, t_emb, crossattn_emb, **block_kwargs,
            )
            if track_per_block:
                block_out = x_B_T_H_W_D.detach().to(cache_device)
                self._calib_block_currs[bi] = block_out

        # ── Per-block cosine similarity (deferred outside the block loop) ──
        if track_per_block:
            for bi in range(len(self.blocks)):
                curr = self._calib_block_currs[bi]
                prev = self._calib_block_prevs[bi]
                if prev is not None and curr is not None:
                    a = prev.flatten()
                    b = curr.flatten()
                    cos_sim = float((a @ b) / (a.norm() * b.norm() + 1e-8))
                    self._calib_block_deltas.append((bi, cos_sim))
                self._calib_block_prevs[bi] = curr.clone() if curr is not None else None

        residual = (x_B_T_H_W_D.to(cache_device) - ori_x).detach()

        # ── Record ground truth output/residual changes ──
        for i, k in enumerate(cond_or_uncond):
            state = self._calib_state[k]
            curr_out = x_B_T_H_W_D[i * b : (i + 1) * b].detach()
            curr_res = residual[i * b : (i + 1) * b]

            if state["prev_out"] is not None:
                out_delta = (curr_out - state["prev_out"]).abs()
                out_denom = state["prev_out"].abs().mean()
                out_rel_val = (out_delta.mean() / max(out_denom.item(), 1e-8)).item()
                out_rel_max_val = (out_delta.max() / max(out_denom.item(), 1e-8)).item()
                out_rel_std_val = (out_delta.std() / max(out_denom.item(), 1e-8)).item()

                # Attach to the most recent entry for this cond slot
                for entry in reversed(self.calibration_log):
                    if entry.cond == int(k):
                        entry.out_rel     = out_rel_val
                        entry.out_rel_max = out_rel_max_val
                        entry.out_rel_std = out_rel_std_val
                        break

            if state["prev_residual"] is not None:
                res_delta = (curr_res - state["prev_residual"]).abs()
                res_denom = state["prev_residual"].abs().mean()
                res_rel_val = (res_delta.mean() / max(res_denom.item(), 1e-8)).item()
                for entry in reversed(self.calibration_log):
                    if entry.cond == int(k):
                        entry.res_rel = res_rel_val
                        break

            state["prev_out"]      = curr_out
            state["prev_residual"] = curr_res

        # ── Per-block cos_sim attachment ──
        if track_per_block and self._calib_block_deltas:
            cos_sim_map = dict(self._calib_block_deltas)
            for entry in reversed(self.calibration_log):
                if entry.step == current_step:
                    entry.block_cos_sims = dict(cos_sim_map)
                if entry.step < current_step:
                    break
            self._calib_block_deltas = []  # clear for next step

        # ── Final layer + unpatchify ──
        x_out = self.final_layer(
            x_B_T_H_W_D.to(crossattn_emb.dtype),
            t_emb,
            adaln_lora_B_T_3D=adaln_lora,
        )
        result = self.unpatchify(x_out)[
            :, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]
        ]

        if ref_latents is not None:
            result = result[:, :, : orig_shape[-3], :, :]

        return result

    return _forward
