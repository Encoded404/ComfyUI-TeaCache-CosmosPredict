#!/usr/bin/env python3
"""Phase 4: Latent corrector training (plan Task 6, v3).

Trains the ``CorrectorUNet2D`` on i.i.d. ``(x_t, v_MA, v_true, t, lag, slot)``
pairs from a refiner recording run (``outputs/<ts>/refiner_data``). Supports:

- single-pass training (K=1) or on-policy K>1 deep supervision (``--multipass``)
  with K curriculum (1 → refine_passes_max), masked-K batching and stop-grad
  between passes (plan 6f);
- optimizers: Sophia (Gauss-Newton-Bartlett, deep-dive §1.2), AdamW,
  AdEMAMix, Schedule-Free AdamW (plan 6g);
- EMA with ramp (deep-dive §2.3), fp16 autocast + GradScaler (deep-dive §2.2);
- eval loop with the K-robustness curve (K=1,2,3), per-lag rel-MSE slices and
  the plan's gates (did-it-learn on the ladder-only K=1 error vs the probe's
  linear ceiling, per-lag coverage, K-robustness gap) — plan 6h;
- stale-only training: the synthesized d=0 anchor pairs (v_MA = v_true) are
  excluded from the training set — the corrector post-processes skip-step
  (stale) velocities only — and kept in the eval set as a diagnostic
  (per_lag[0], anchors_only);
- checkpointing: EMA .safetensors per eval + best-by-eval (ladder-only K=1)
  + full-state .pt resume with config-drift warning (plan 6i);
- live reporting: tqdm progress bar (EMA loss, lr, K, it/s, remaining),
  per-50-step durable log lines, per-eval timing reports (train/eval/hessian
  phases; the ETA projects eval + checkpoint overhead), VRAM peak, and a
  JSONL metrics file (``train_metrics.jsonl`` next to the checkpoints;
  ``--metrics`` relocates, ``--no-progress`` disables the bar);
- pre-train corpus diagnostics (probe Task 3b stats — SVD rank, affine
  ceiling, staleness curve, Δ_MA distribution, t-gap cancellation, v_true
  step correlation; from ``refiner_probe_report.json`` or recomputed from a
  pair subsample) and per-eval v_t error recovery tables vs the K=0 TeaCache
  base (abs err, ×base, % recovered), also mirrored into the metrics JSONL.

Prototype upgrades (v2 training):

- lag conditioning: the corrector receives the skip age (steps since the
  last full run) as an adaLN-conditioning input (``--lag-cond``, default
  on) — the deployment lag counter lives in ``teacache_state["lag"]``;
- resolution conditioning: the bottleneck grid dims are embedded into the
  conditioning (``--res-cond``, default on) — resolution independence;
- per-lag loss normalization (``--lag-loss-normalize``, default on): each
  sample's rel-MSE is scaled by mean_base_err(lag)/base_err(lag) so no lag
  hogs the gradient by difficulty;
- fixed eval set: the eval sampler's batches are built once at startup and
  replayed at every eval (identical K=0 base across evals), and the EMA
  model is what gets evaluated / best-selected (it is what ships);
- resolution curriculum ramp: stage 1 → full mix interpolates over
  ``stage2_ramp_fraction`` of the run instead of a hard switch, and
  ``resolution_weights`` upweight the 1024² buckets;
- de-bursted training batches: the epoch's generation runs are merged
  round-robin in windows (``--deburst-windows``, default on) so consecutive
  batches come from different generations while the active window stays in
  the generation LRU.

Every flag defaults to the ``refiner_training`` config section
(config.json); CLI flags override config (calibrate.py precedence pattern).
The effective post-override config is snapshotted into checkpoint metadata.

Usage:
    python -m tuning.train_corrector --data outputs/<ts>/refiner_data
    python -m tuning.train_corrector --data outputs/<ts>/refiner_data \
        --multipass 0 --model-size 5m --max-steps 2000 --no-progress 1
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

from .config_types import TuningConfig
from . import refiner_data
from .corrector import (CorrectorConfig, CorrectorUNet2D, FULL_STEP_FLOP_512,
                        estimate_corrector_flops, save_corrector)
from .corrector_dataset import (CorrectorBatchSampler, CorrectorDataset,
                                FixedBatchListSampler, augment_batch,
                                collect_fit_maps, collate_corrector)
from .utils import MetricsLog, TrainTimer, format_duration
from .utils import (affine_oob_eval, affine_oob_json, affine_shape_area,
                    affine_shape_label, split_affine_per_pair)
from .utils import (delta_distribution, per_channel_affine_ceiling,
                    pooled_feature_ceiling, step_correlations, staleness_curve,
                    svd_rank_95)

from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # progress bars are optional (requirements: tqdm)
    tqdm = None

# ── Losses (plan 6f) ──────────────────────────────────────────────────


def per_sample_rel_mse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample rel-MSE: ‖pred−target‖²/(‖target‖²+ε), (B,)."""
    num = (pred - target).float().square().mean(dim=(1, 2, 3))
    den = target.float().square().mean(dim=(1, 2, 3)) + eps
    return num / den


def mse_loss(pred, target, eps: float = 1e-8):
    return (pred - target).float().square().mean(dim=(1, 2, 3))


def mse_l1_loss(pred, target, eps: float = 1e-8):
    """0.5·MSE + 0.5·L1 (per sample)."""
    d = (pred - target).float()
    return 0.5 * d.square().mean(dim=(1, 2, 3)) + 0.5 * d.abs().mean(dim=(1, 2, 3))


def charbonnier_loss(pred, target, eps: float = 1e-3):
    return ((pred - target).float().square() + eps ** 2).sqrt().mean(dim=(1, 2, 3))


LOSS_FNS = {
    "rel_mse": per_sample_rel_mse,
    "mse": mse_loss,
    "mse_l1": mse_l1_loss,
    "charbonnier": charbonnier_loss,
}


# ── Sophia (deep-dive §1.2, Gauss-Newton-Bartlett) ────────────────────


class SophiaG(torch.optim.Optimizer):
    """Diagonal-Hessian-preconditioned, clipped SGD (arXiv:2305.14342).

    Implements the plan's decided configuration: Gauss-Newton-Bartlett
    estimator every ``hessian_every`` steps, in fp32 outside autocast,
    element-wise clip ρ (deep-dive §1.2 code).
    """

    def __init__(self, params, lr=4e-4, betas=(0.965, 0.99), rho=0.04,
                 eps=1e-12, weight_decay=0.0, hessian_every=10):
        defaults = dict(lr=lr, betas=betas, rho=rho, eps=eps,
                        weight_decay=weight_decay, hessian_every=hessian_every)
        super().__init__(params, defaults)
        self._hessian_steps = 0
        self._hessian_pending = False

    def _state(self, p):
        state = self.state[p]
        if "m" not in state:
            state["m"] = torch.zeros_like(p)
            state["h"] = torch.zeros_like(p)
            state["h_est"] = torch.zeros_like(p)
        elif "h_est" not in state:
            # state restored from a pre-h_est checkpoint
            state["h_est"] = torch.zeros_like(p)
        return state

    def update_hessian(self, loss: torch.Tensor) -> None:
        """GNB diagonal-Hessian estimate on a fp32 loss with a live graph.

        Cost: one extra forward + backward every ``hessian_every`` steps
        (amortized ~10–15%). Must be called before ``step``. Parameters not
        used in the graph (e.g. the prompt layers on an all-uncond 0-token
        batch) are skipped — their Hessian is undefined that step and their
        stored h stays stale until a batch that exercises them.

        This is a double-backward estimate (gradient of the gradient), so the
        loss graph must be built without the fused SDPA backends (flash /
        memory-efficient): they implement no second-order derivative. Callers
        should build ``loss`` under
        ``torch.nn.attention.sdpa_kernel(SDPBackend.MATH)`` so attention runs
        as bmm/softmax/matmul, which are double-backwardable.
        """
        self._hessian_steps += 1
        params = [p for g in self.param_groups for p in g["params"] if p.requires_grad]
        if not params:
            return
        grads = torch.autograd.grad(loss, params, create_graph=True,
                                    retain_graph=True, allow_unused=True)
        used = [(p, g) for p, g in zip(params, grads) if g is not None]
        if not used:
            return
        us = [torch.randn_like(g) for _, g in used]
        used_params = [p for p, _ in used]
        gu = torch.autograd.grad(
            [(g * u).sum() for (_, g), u in zip(used, us)], used_params,
            retain_graph=True,
        )
        with torch.no_grad():
            for p, u, gu_i in zip(used_params, us, gu):
                self._state(p)["h_est"].copy_((u * gu_i).detach().clamp_min(0))
        self._hessian_pending = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        hessian_this_step = self._hessian_pending
        self._hessian_pending = False
        for group in self.param_groups:
            b1, b2 = group["betas"]
            rho, eps, lr = group["rho"], group["eps"], group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self._state(p)
                m, h = state["m"], state["h"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                if hessian_this_step:
                    h.mul_(b2).add_(state["h_est"], alpha=1 - b2)
                update = (m / h.clamp_min(eps)).clamp(-rho, rho)
                p.add_(update, alpha=-lr)
                if wd:
                    p.mul_(1 - lr * wd)
        return loss


# ── EMA (deep-dive §2.3) ──────────────────────────────────────────────


def _model_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """state_dict of the real module (torch.compile wraps as _orig_mod)."""
    inner = getattr(model, "_orig_mod", model)
    return inner.state_dict()


def init_ema(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone().float() for k, v in _model_state(model).items()}


def ema_update(model: nn.Module, ema: Dict[str, torch.Tensor], step: int,
               decay: float = 0.9999, ramp_steps: int = 2000) -> None:
    ramp = min(1.0, (1 + step) / max(ramp_steps, 1))
    decay_eff = 1.0 - (1.0 - decay) * ramp
    with torch.no_grad():
        for k, v in _model_state(model).items():
            if v.is_floating_point():
                ema[k].mul_(decay_eff).add_(v.float(), alpha=1 - decay_eff)
            else:
                ema[k].copy_(v)


# ── Config resolution (plan 1b/6i: config defaults → CLI overrides) ────


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the latent corrector")
    parser.add_argument("--data", required=True, help="Path to a refiner_data dir")
    parser.add_argument("--config", default=None,
                        help="Path to config.json (default: tuning/config.json)")
    parser.add_argument("--multipass", type=int, default=None,
                        help="K>1 on-policy training (config: refiner_training.multipass)")
    parser.add_argument("--lag-cond", type=int, default=None,
                        help="Lag (skip-age) conditioning (config: lag_cond)")
    parser.add_argument("--res-cond", type=int, default=None,
                        help="Resolution conditioning (config: res_cond)")
    parser.add_argument("--lag-loss-normalize", type=int, default=None,
                        help="Per-lag loss normalization (config: lag_loss_normalize)")
    parser.add_argument("--deburst-windows", type=int, default=None,
                        help="Windowed round-robin de-burst batching "
                             "(config: deburst_windows)")
    parser.add_argument("--deburst-window-runs", type=int, default=None,
                        help="Runs per de-burst window (config: deburst_window_runs)")
    parser.add_argument("--eval-min-batches-per-shape", type=int, default=None,
                        help="Eval budget floor per shape bucket "
                             "(config: eval_min_batches_per_shape)")
    parser.add_argument("--refine-passes-max", type=int, default=None,
                        help="K ceiling 1-4 (config: refine_passes_max)")
    parser.add_argument("--k-curriculum", type=int, default=None,
                        help="Anneal K_max 1→refine_passes_max (config: k_curriculum)")
    parser.add_argument("--stop-grad", type=int, default=None,
                        help="Detach between passes (config: stop_grad)")
    parser.add_argument("--model-size", type=str, default=None,
                        help="Corrector size target: '5M', '20m', '1.5B', 'tiny' "
                             "(default: refiner_training.model_size)")
    parser.add_argument("--depth", type=str, default=None,
                        help="DiT bottleneck blocks: 'auto' (width first, depth "
                             "past the family ceiling) or 1-8 "
                             "(default: refiner_training.depth)")
    parser.add_argument("--optimizer", type=str, default=None,
                        choices=["adamw", "sophia", "ademamix", "schedulefree"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None, help="Sophia clip")
    parser.add_argument("--hessian-every", type=int, default=None, help="Sophia k")
    parser.add_argument("--wd", type=float, default=None, help="weight decay")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--loss", type=str, default=None,
                        choices=["rel_mse", "mse", "mse_l1", "charbonnier"])
    parser.add_argument("--lag-weights", type=str, default=None,
                        help="Per-lag resampling weights, e.g. '1,1,1,1,1' "
                             "(record_lags order)")
    parser.add_argument("--normalization", type=str, default=None,
                        choices=["none", "perchannel"])
    parser.add_argument("--normalization-samples", type=int, default=None,
                        help="Pairs subsampled for normalization/ε stats "
                             "(config: normalization_samples)")
    parser.add_argument("--resolution-curriculum", type=int, default=None,
                        help="Stage-1 (e.g. 64x64-only) resolution curriculum "
                             "(config: resolution_curriculum.enabled)")
    parser.add_argument("--accumulate-big-batches", type=int, default=None,
                        help="Gradient-accumulate ÷4 resolution buckets up to a "
                             "full batch (config: accumulate_big_batches)")
    parser.add_argument("--scale-aug", type=int, default=None,
                        help="Round-trip 0.75× random-scale augmentation "
                             "(config: scale_aug)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Initial eval interval (config: eval_every)")
    parser.add_argument("--eval-schedule-growth", type=float, default=None,
                        help="Geometric eval-interval growth factor per eval "
                             "(1.0 = constant; config: eval_schedule_growth)")
    parser.add_argument("--eval-interval-cap", type=int, default=None,
                        help="Max eval interval after growth "
                             "(config: eval_interval_cap)")
    parser.add_argument("--eval-max-batches", type=int, default=None,
                        help="Cap on eval batches per eval (0 = full eval set; "
                             "config: eval_max_batches)")
    parser.add_argument("--eval-full-k-every", type=int, default=None,
                        help="Every N-th eval runs the full K ladder (1 = "
                             "always; config: eval_full_k_every)")
    parser.add_argument("--compile", type=int, default=None,
                        help="torch.compile the corrector")
    parser.add_argument("--channels-last", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cache-size", type=int, default=None,
                        help="Decompressed-generation LRU cache size, in "
                             "generations (config: refiner_training.cache_size)")
    parser.add_argument("--recovery-fit-batches", type=int, default=None,
                        help="batch tensors per shape for the OOD affine fit "
                             "(config: refiner_training.recovery_fit_batches, default 128)")
    parser.add_argument("--recovery-eval-batches", type=int, default=None,
                        help="batch tensors per shape for the OOD affine score "
                             "(config: refiner_training.recovery_eval_batches, default 64)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader worker processes (config: "
                             "refiner_training.num_workers)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a full-state .pt checkpoint")
    parser.add_argument("--out", type=str, default=None,
                        help="Checkpoint output dir (default: <repo>/models)")
    parser.add_argument("--metrics", type=str, default=None,
                        help="JSONL metrics path (default: <out>/train_metrics.jsonl)")
    parser.add_argument("--no-progress", type=int, default=None,
                        help="Disable tqdm progress bars (default: auto TTY detect)")
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> dict:
    """Merge CLI over config.json ``refiner_training`` into the effective config."""
    if args.config is None:
        args.config = str(Path(__file__).parent / "config.json")
    tcfg = TuningConfig.load(args.config)
    rt = dict(tcfg.refiner_training or {})
    mapping = {
        "multipass": ("multipass", True),
        "lag_cond": ("lag_cond", True),
        "res_cond": ("res_cond", True),
        "lag_loss_normalize": ("lag_loss_normalize", True),
        "deburst_windows": ("deburst_windows", True),
        "deburst_window_runs": ("deburst_window_runs", 64),
        "eval_min_batches_per_shape": ("eval_min_batches_per_shape", 40),
        "refine_passes_max": ("refine_passes_max", 3),
        "k_curriculum": ("k_curriculum", True),
        "stop_grad": ("stop_grad", True),
        "model_size": ("model_size", "20m"),
        "depth": ("depth", "auto"),
        "optimizer": ("optimizer", "sophia"),
        "lr": ("lr", 4e-4),
        "rho": ("rho", 0.04),
        "hessian_every": ("hessian_every", 10),
        "wd": ("wd", 0.05),
        "batch_size": ("batch_size", 32),
        "ema_decay": ("ema_decay", 0.9999),
        "loss": ("loss", "rel_mse"),
        "lag_weights": ("lag_weights", [2.0, 1.5, 1.25, 1.0, 0.5, 0.25]),
        "normalization": ("normalization", "none"),
        "normalization_samples": ("normalization_samples", 128),
        "accumulate_big_batches": ("accumulate_big_batches", True),
        "scale_aug": ("scale_aug", True),
        "max_steps": ("max_steps", 60000),
        "eval_every": ("eval_every", 500),
        "eval_schedule_growth": ("eval_schedule_growth", 2.0),
        "eval_interval_cap": ("eval_interval_cap", 32000),
        "eval_max_batches": ("eval_max_batches", 512),
        "eval_full_k_every": ("eval_full_k_every", 5),
        "seed": ("seed", 42),
        "cache_size": ("cache_size", 128),
        "recovery_fit_batches": ("recovery_fit_batches", 128),
        "recovery_eval_batches": ("recovery_eval_batches", 64),
        "num_workers": ("num_workers", 0),
    }
    cfg = {}
    for flag, (key, default) in mapping.items():
        value = getattr(args, flag)
        if value is None:
            value = rt.get(key, default)
        cfg[key] = value
    if isinstance(cfg["lag_weights"], str):
        try:
            cfg["lag_weights"] = [float(x) for x in cfg["lag_weights"].split(",")]
        except ValueError:
            raise SystemExit(f"[train_corrector] invalid --lag-weights {cfg['lag_weights']!r}")
    cfg["multipass"] = bool(cfg["multipass"])
    cfg["lag_cond"] = bool(cfg["lag_cond"])
    cfg["res_cond"] = bool(cfg["res_cond"])
    cfg["lag_loss_normalize"] = bool(cfg["lag_loss_normalize"])
    cfg["deburst_windows"] = bool(cfg["deburst_windows"])
    cfg["deburst_window_runs"] = max(2, int(cfg["deburst_window_runs"]))
    cfg["eval_min_batches_per_shape"] = max(0, int(cfg["eval_min_batches_per_shape"]))
    cfg["k_curriculum"] = bool(cfg["k_curriculum"])
    cfg["stop_grad"] = bool(cfg["stop_grad"])
    cfg["accumulate_big_batches"] = bool(cfg["accumulate_big_batches"])
    cfg["scale_aug"] = bool(cfg["scale_aug"])
    cfg["compile"] = bool(args.compile)
    cfg["channels_last"] = bool(args.channels_last)
    cfg["refine_passes_max"] = max(1, min(int(cfg["refine_passes_max"]), 4))
    cfg["seed"] = int(cfg["seed"])
    cfg["depth"] = str(cfg["depth"]).strip().lower()
    if cfg["depth"] != "auto" and not (
            cfg["depth"].isdigit() and 1 <= int(cfg["depth"]) <= 8):
        raise SystemExit(f"[train_corrector] invalid --depth {cfg['depth']!r} "
                         f"(auto or 1-8)")
    cfg["normalization_samples"] = max(0, int(cfg["normalization_samples"]))
    if cfg["normalization"] == "perchannel" and cfg["normalization_samples"] < 1:
        raise SystemExit(f"[train_corrector] normalization=perchannel needs "
                         f"--normalization-samples >= 1, got "
                         f"{cfg['normalization_samples']}")
    rc = rt.get("resolution_curriculum") or {}
    if isinstance(rc, dict):
        cfg["resolution_curriculum"] = dict(rc)
    elif isinstance(rc, (bool, int)):
        cfg["resolution_curriculum"] = {"enabled": bool(rc)}
    else:
        cfg["resolution_curriculum"] = {}
    if args.resolution_curriculum is not None:
        cfg["resolution_curriculum"]["enabled"] = bool(args.resolution_curriculum)
    rc = cfg["resolution_curriculum"]
    rc.setdefault("enabled", False)
    rc.setdefault("stage1_fraction", 0.6)
    rc.setdefault("stage1_shapes", ["64x64"])
    rc.setdefault("stage2_ramp_fraction", 0.15)
    rc["enabled"] = bool(rc["enabled"])
    rc["stage1_fraction"] = max(0.0, min(float(rc["stage1_fraction"]), 1.0))
    rc["stage2_ramp_fraction"] = max(0.0, min(float(rc["stage2_ramp_fraction"]), 1.0))
    stage1_areas = []
    for spec in rc.get("stage1_shapes") or ["64x64"]:
        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            w, h = int(spec[0]), int(spec[1])
        else:
            w, h = refiner_data.parse_resolution(str(spec))
        stage1_areas.append(w * h)
    rc["stage1_areas"] = sorted(set(stage1_areas))
    # Per-shape bucket upweights ("HxW" → multiplier) for the training
    # sampler. Kept as string keys in the config dict (JSON-serializable for
    # the checkpoint snapshot); converted to (h, w) tuples at sampler build.
    resolution_weights = {}
    for key, value in (rc.get("resolution_weights") or {}).items():
        try:
            h, w = (int(x) for x in str(key).split("x"))
        except ValueError:
            raise SystemExit(f"[train_corrector] invalid resolution_weights key "
                             f"{key!r} (expected 'HxW')")
        resolution_weights[str(key)] = max(0.0, float(value))
    rc["resolution_weights"] = resolution_weights
    cfg["cache_size"] = max(1, int(cfg["cache_size"]))
    cfg["num_workers"] = max(0, int(cfg["num_workers"]))
    cfg["eval_every"] = max(int(cfg["eval_every"]), 1)
    cfg["eval_schedule_growth"] = max(float(cfg["eval_schedule_growth"]), 1.0)
    cfg["eval_interval_cap"] = max(int(cfg["eval_interval_cap"]), 0)
    cfg["eval_max_batches"] = max(int(cfg["eval_max_batches"]), 0)
    cfg["eval_full_k_every"] = max(int(cfg["eval_full_k_every"]), 1)
    cfg["out"] = args.out or str(Path(__file__).resolve().parent.parent / "models")
    cfg["data"] = args.data
    cfg["resume"] = args.resume
    cfg["metrics"] = args.metrics or str(Path(cfg["out"]) / "train_metrics.jsonl")
    cfg["no_progress"] = bool(args.no_progress) if args.no_progress is not None else False
    return cfg


def build_eval_plan(cfg: dict) -> List[int]:
    """Eval step numbers for the run: geometric schedule with eval points at
    ``eval_every × eval_schedule_growth``^k (500, 1000, 2000, 4000, ...),
    spacing capped at ``eval_interval_cap`` once the geometric step exceeds
    it, and never beyond ``max_steps``.

    Pure function of the config, so resume re-derives the same plan from the
    checkpoint step — no schedule state to persist. ``growth <= 1.0`` keeps a
    constant interval (legacy behavior).
    """
    max_steps = int(cfg["max_steps"])
    first = max(int(cfg.get("eval_every", 500)), 1)
    growth = float(cfg.get("eval_schedule_growth", 1.0))
    cap = max(int(cfg.get("eval_interval_cap", 0)), 0)
    plan: List[int] = []
    step = 0
    k = 0
    while True:
        if growth > 1.0:
            nxt_geom = int(first * growth ** k)
            if cap > 0 and nxt_geom > cap:
                step += cap
            else:
                step = nxt_geom
                k += 1
        else:
            step += first
        if step > max_steps:
            break
        plan.append(step)
    return plan


# ── Training step (plan 6f) ───────────────────────────────────────────


def compute_train_loss(model, batch, Ks: torch.Tensor, loss_fn, eps: float,
                       stop_grad: bool, multipass: bool,
                       k_max: int,
                       lag_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Masked-K on-policy deep-supervised loss (plan 6f, deep-dive §5.4/§6.1).

    Pass i supervises the model's own i-th pass output against v_true with
    weight w_i = i / Σ_{j=1}^{K_s} j per sample; pass inputs are detached
    between passes when ``stop_grad``.

    ``lag_scale`` (optional, indexed by lag value) multiplies each sample's
    loss so per-lag gradient mass is not dominated by the hardest lags.
    """
    x, v, vt = batch["x_t"], batch["v_ma"].clone(), batch["v_true"]
    prompt, pmask, t = batch["prompt"], batch["prompt_mask"], batch["t_frac"]
    lag = batch["lag"]
    total = torch.zeros((), device=x.device, dtype=torch.float32)
    n_active = torch.zeros((), device=x.device)
    for i in range(1, k_max + 1):
        mask = Ks >= i
        if not mask.any():
            continue
        dv = model(x[mask], v[mask], prompt[mask], t[mask], pmask[mask],
                   lag=lag[mask].float())
        v_new = v[mask] + dv
        per_sample = loss_fn(v_new, vt[mask], eps)
        if lag_scale is not None:
            per_sample = per_sample * lag_scale[lag[mask]]
        # deep-supervision weight: i / Σ_{j=1}^{K_s} j per sample
        k_s = Ks[mask].float()
        w = i / (k_s * (k_s + 1) / 2)
        total = total + (w * per_sample).mean()
        n_active = n_active + 1
        if stop_grad:
            v[mask] = v_new.detach()
        else:
            v[mask] = v_new
    if n_active == 0:
        return torch.zeros((), device=x.device, requires_grad=True)
    return total


def estimate_per_lag_base_rel(ds, n_per_lag: int = 128, seed: int = 42,
                              min_samples: int = 8) -> Dict[int, float]:
    """Mean base (K=0) rel-MSE per lag from a seeded subsample of train pairs.

    Used to normalize the per-lag loss contribution (``lag_loss_normalize``):
    w_lag = mean(e)/e_lag, so lags whose base error is large (hard lags) stop
    hogging the gradient. Loads via the dataset's generation LRU (a few
    decodes total). Lags with fewer than ``min_samples`` valid loads are
    skipped.
    """
    lags_present = sorted({lag for _, _, _, lag in ds.pairs if lag != 0})
    by_lag: Dict[int, List[float]] = {}
    idxs = list(range(len(ds.pairs)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    for i in idxs:
        lag = ds.pairs[i][3]
        if lag == 0 or len(by_lag.get(lag, [])) >= n_per_lag:
            continue
        if len(by_lag) >= len(lags_present) and all(
                len(v) >= n_per_lag for v in by_lag.values()):
            break
        try:
            _, v_ma, v_true, _ = ds._load_pair(i)
        except KeyError:  # ring-availability drift
            continue
        e = per_sample_rel_mse(v_ma.unsqueeze(0), v_true.unsqueeze(0),
                               ds.rel_mse_eps).item()
        by_lag.setdefault(lag, []).append(e)
    out = {}
    for lag, es in by_lag.items():
        if len(es) >= min_samples:
            out[lag] = sum(es) / len(es)
    return out


# ── Eval (plan 6h) ────────────────────────────────────────────────────

# Eval step-region slices: thirds of the schedule, mirroring the probe's
# staleness regions (PROBE_GUIDE §7). region = min(R-1, int(R * t_frac)).
EVAL_T_REGIONS = 3
EVAL_T_REGION_NAMES = {0: "early", 1: "mid", 2: "late"}


def eval_model(model, eval_loader, lags: List[int], ks: List[int],
               device, eps: float, show_progress: bool = False,
               eval_map_batches: int = 64) -> Tuple[Dict, Dict, Dict, Dict]:
    """Per-lag, per-K rel-MSE (‖v̂−v_true‖₂/‖v_true‖₂) and cosine, plus the
    same metrics grouped per spatial shape ((k, (h, w))) for the per-resolution
    eval report (resolution-independence check).

    Also slices every metric per schedule third ((k, region, …) with
    region = early/mid/late by t_frac, mirroring the probe's staleness
    regions) into ``by_t`` — "pooled" (all pairs), "lags" (per ladder lag,
    excluding the d=0 anchor only when reporting), "shapes" (per latent
    shape) — so where the corrector helps is visible instead of pooled away.

    Also collects the eval (v_ma, v_true) maps per latent shape (with per-pair
    d=0-anchor masks, capped at ``eval_map_batches`` batch tensors per shape)
    for the OOD affine recovery row — the affine is fit on TRAIN pairs by the
    caller and scored here, never on its own eval slice.
    """
    model.eval()
    acc = {(k, lag): [0.0, 0.0, 0] for k in ks for lag in lags}
    acc_shape: Dict[Tuple[int, Tuple[int, int]], List[float]] = {}
    acc_t: Dict[Tuple[int, int], List[float]] = {}
    acc_t_lag: Dict[Tuple[int, int, int], List[float]] = {}
    acc_t_shape: Dict[Tuple[int, Tuple[int, int], int], List[float]] = {}
    eval_maps: Dict[Tuple[int, int], dict] = {}
    batches = eval_loader
    if show_progress and tqdm is not None:
        batches = tqdm(eval_loader, desc="eval", unit="batch", leave=False,
                       dynamic_ncols=True, mininterval=0.5)
    with torch.no_grad():
        for batch in batches:
            x = batch["x_t"].to(device)
            v0 = batch["v_ma"].to(device)
            vt = batch["v_true"].to(device)
            prompt = batch["prompt"].to(device)
            pmask = batch["prompt_mask"].to(device)
            t = batch["t_frac"].to(device)
            lags_b = batch["lag"].tolist()
            hw = (int(x.shape[-2]), int(x.shape[-1]))
            em = eval_maps.setdefault(hw, {"vm": [], "vt": [], "anchor": []})
            if len(em["vm"]) < eval_map_batches:
                em["vm"].append(batch["v_ma"].float().cpu())
                em["vt"].append(batch["v_true"].float().cpu())
                em["anchor"].extend([lag == 0 for lag in lags_b])
            # fp16 autocast, same precision path as the training step; the
            # metrics are computed on .float() tensors below, so rel-MSE and
            # cosine are unaffected by the autocast dtype.
            with torch.autocast(device.type, dtype=torch.float16):
                t_fracs = t.tolist()
                for k in ks:
                    v = v0
                    for _ in range(k):
                        v = v + model(x, v, prompt, t, pmask,
                                      lag=batch["lag"].to(device).float())
                    err = (v - vt).float().flatten(1).norm(dim=1)
                    den = vt.float().flatten(1).norm(dim=1) + 1e-8
                    rel = (err / den).tolist()
                    cos = F.cosine_similarity(v.float().flatten(1),
                                              vt.float().flatten(1), dim=1).tolist()
                    for j, lag in enumerate(lags_b):
                        region = min(EVAL_T_REGIONS - 1,
                                     int(EVAL_T_REGIONS * t_fracs[j]))
                        if lag in lags:
                            a = acc[(k, lag)]
                            a[0] += rel[j]
                            a[1] += cos[j]
                            a[2] += 1
                        a = acc_t.setdefault((k, region), [0.0, 0.0, 0.0])
                        a[0] += rel[j]
                        a[1] += cos[j]
                        a[2] += 1
                        if lag in lags:
                            a = acc_t_lag.setdefault((k, region, lag),
                                                     [0.0, 0.0, 0.0])
                            a[0] += rel[j]
                            a[1] += cos[j]
                            a[2] += 1
                        a = acc_t_shape.setdefault((k, region, hw),
                                                   [0.0, 0.0, 0.0])
                        a[0] += rel[j]
                        a[1] += cos[j]
                        a[2] += 1
                    a = acc_shape.setdefault((k, hw), [0.0, 0.0, 0.0])
                    a[0] += sum(rel)
                    a[1] += sum(cos)
                    a[2] += len(rel)
    out = {}
    for (k, lag), (s_rel, s_cos, n) in acc.items():
        if n:
            out[(k, lag)] = {"rel_mse": s_rel / n, "cosine": s_cos / n, "n": n}
    by_shape = {}
    for (k, hw), (s_rel, s_cos, n) in acc_shape.items():
        if n:
            by_shape[(k, hw)] = {"rel_mse": s_rel / n, "cosine": s_cos / n, "n": n}
    by_t = {"pooled": {}, "lags": {}, "shapes": {}}
    for (k, region), (s_rel, s_cos, n) in acc_t.items():
        if n:
            by_t["pooled"][(region, k)] = {"rel_mse": s_rel / n,
                                           "cosine": s_cos / n, "n": n}
    for (k, region, lag), (s_rel, s_cos, n) in acc_t_lag.items():
        if n:
            by_t["lags"][(region, k, lag)] = {"rel_mse": s_rel / n,
                                              "cosine": s_cos / n, "n": n}
    for (k, region, hw), (s_rel, s_cos, n) in acc_t_shape.items():
        if n:
            by_t["shapes"][(region, k, hw)] = {"rel_mse": s_rel / n,
                                               "cosine": s_cos / n, "n": n}
    model.train()
    return out, by_shape, by_t, eval_maps


def load_linear_ceiling(data_dir: Path) -> Optional[float]:
    """Probe's per-channel affine ceiling (1−R²) for the did-it-learn gate.

    The probe report is per-resolution; the gate ceiling is taken from the
    report's dominant stratum (``decision_gate.stratum``), falling back to the
    max across strata, and legacy flat reports are still accepted.
    """
    report = data_dir.parent / "refiner_probe_report.json"
    if not report.exists():
        return None
    try:
        r = json.loads(report.read_text())
        pca = r["learnability"]["predictability_ceiling"]["per_channel_affine"]
        if isinstance(pca, dict):
            stratum = (r.get("decision_gate") or {}).get("stratum")
            if stratum and stratum in pca:
                return float(pca[stratum])
            return float(max(pca.values()))
        return float(pca)
    except Exception:
        return None


# ── Recovery table: each method's v_t error vs the TeaCache base ──────


def recovery_rows(results: Dict[Tuple[int, int], dict],
                  ks: List[int]) -> List[dict]:
    """Per-K rel-MSE pooled per-pair (n-weighted over lags), plus ladder-only
    and d=0-anchor-only splits and the pair count (K=0 is the base)."""
    rows = []
    for k in ks:
        items = [(lag, r["rel_mse"], r["n"]) for (kk, lag), r in results.items()
                 if kk == k]
        if not items:
            rows.append({"k": k, "rel_mse": float("nan"),
                         "ladder_only": float("nan"), "anchors_only": float("nan"),
                         "n_pairs": 0})
            continue

        def pooled(sub):
            tot = sum(m * n for _, m, n in sub)
            n = sum(n for _, _, n in sub)
            return tot / n if n else float("nan")

        rows.append({"k": k,
                     "rel_mse": pooled(items),
                     "ladder_only": pooled([it for it in items if it[0] != 0]),
                     "anchors_only": pooled([it for it in items if it[0] == 0]),
                     "n_pairs": sum(n for _, _, n in items)})
    return rows


def recovery_table_lines(rows: List[dict],
                         affine: Optional[dict] = None) -> List[str]:
    """Human-readable recovery table: abs err, ×base and % recovered per K.

    ``affine`` is an :func:`~tuning.utils.affine_oob_eval` result (plus a
    "split" key with ladder/anchor means) for the OOD per-channel affine row.
    """
    base = next((r["rel_mse"] for r in rows if r["k"] == 0), None)
    base_ladder = next((r["ladder_only"] for r in rows if r["k"] == 0), float("nan"))
    lines = [
        "  v_t error recovery vs TeaCache base "
        "(pooled rel-MSE ‖v̂−v_true‖₂/‖v_true‖₂)",
        "  " + "─" * 62,
        f"  {'method':<24}{'abs err':>10}{'× base':>9}{'recovered':>12}",
        "  " + "─" * 62,
    ]
    for r in rows:
        name = "TeaCache base (K=0)" if r["k"] == 0 else f"Corrector K={r['k']}"
        err = r["rel_mse"]
        if base is None or base <= 0 or err != err:
            lines.append(f"  {name:<24}{err:>10.5f}")
            continue
        ratio = err / base
        rec = None if r["k"] == 0 else 1.0 - ratio
        rec_str = "—" if rec is None else f"{100 * rec:>10.1f}%"
        lines.append(f"  {name:<24}{err:>10.5f}{ratio:>9.3f}{rec_str:>12}")
        if r["ladder_only"] == r["ladder_only"]:
            lad_ratio = lad_rec = None
            if base_ladder == base_ladder and base_ladder > 0:
                lad_ratio = r["ladder_only"] / base_ladder
                lad_rec = None if r["k"] == 0 else 1.0 - lad_ratio
            lad_rat_str = "—" if lad_ratio is None else f"{lad_ratio:>9.3f}"
            lad_rec_str = "—" if lad_rec is None else f"{100 * lad_rec:>10.1f}%"
            lines.append(f"    ladder only          {r['ladder_only']:>10.5f}"
                         f"{lad_rat_str}{lad_rec_str:>12}")
        if r["anchors_only"] == r["anchors_only"]:
            lines.append(f"    d=0 anchors only     {r['anchors_only']:>10.5f}")
    if affine is not None:
        a_all = affine.get("overall")
        if a_all is None:
            lines.append(f"  {'Per-channel affine (OOD)':<24}{'—':>10}")
        else:
            ratio = a_all / base if base and base > 0 else None
            rec = None if ratio is None else 1.0 - ratio
            rat_str = "—" if ratio is None else f"{ratio:>9.3f}"
            rec_str = "—" if rec is None else f"{100 * rec:>10.1f}%"
            lines.append(f"  {'Per-channel affine (OOD)':<24}{a_all:>10.5f}"
                         f"{rat_str}{rec_str:>12}")
            lad = (affine.get("split") or {}).get("ladder")
            anc = (affine.get("split") or {}).get("anchor")
            if lad is not None:
                lad_ratio = lad_rec = None
                if base_ladder == base_ladder and base_ladder > 0:
                    lad_ratio = lad / base_ladder
                    lad_rec = 1.0 - lad_ratio
                lad_rat_str = "—" if lad_ratio is None else f"{lad_ratio:>9.3f}"
                lad_rec_str = "—" if lad_rec is None else f"{100 * lad_rec:>10.1f}%"
                lines.append(f"    ladder only          {lad:>10.5f}"
                             f"{lad_rat_str}{lad_rec_str:>12}")
            if anc is not None:
                lines.append(f"    d=0 anchors only     {anc:>10.5f}")
    lines.append("  " + "─" * 62)
    lines.append(f"  {'Oracle (v_true)':<24}{'0.00000':>10}{'0.000':>9}{'100.0%':>12}")
    by_shape = (affine or {}).get("by_shape") or {}
    if by_shape:
        lines.append("  affine per stratum (fit = train pairs, scored = eval pairs):")
        for shape, d in sorted(by_shape.items(), key=lambda kv: affine_shape_area(kv[0])):
            if d.get("rel_mse") is None:
                lines.append(f"    {affine_shape_label(shape)}:  no train fit pairs — row skipped")
                continue
            lines.append(f"    {affine_shape_label(shape)}:  rel {d['rel_mse']:.4f}  "
                         f"(n={d['n_pairs']}, fit {d['fit_n_batches']} batches)  "
                         f"|a−1|≤{d['max_abs_a_minus_1']:.4f}  |b|≤{d['max_abs_b']:.4f}")
    return lines


# ── Corpus diagnostics (probe Task 3b stats at pre-train time) ─────────


def corpus_diagnostics(data_dir: Path, ds, n_samples: int = 384) -> dict:
    """Probe learnability stats for pre-train sanity.

    Prefers the probe's ``refiner_probe_report.json`` (written next to the data
    dir); falls back to recomputing the cheap stats from a random subsample of
    dataset pairs (random-access loads, no full generation decodes).
    """
    report = Path(data_dir).parent / "refiner_probe_report.json"
    if report.exists():
        try:
            return _diagnostics_from_report(report)
        except Exception:
            pass
    return _recompute_diagnostics(ds, n_samples)


def _diagnostics_from_report(report_path: Path) -> dict:
    r = json.loads(report_path.read_text())
    learn = r.get("learnability") or {}
    diag = {"source": "probe report"}
    svd = learn.get("svd") or {}
    diag["svd"] = {k: v.get("rank_95") for k, v in svd.items()}
    pca = learn.get("predictability_ceiling") or {}
    diag["affine_ceiling"] = pca.get("per_channel_affine") or {}
    diag["pooled_feature_ceiling"] = pca.get("pooled_feature")
    diag["staleness_curve"] = learn.get("staleness_curve") or {}
    diag["delta_distribution"] = learn.get("delta_distribution") or {}
    diag["v_true_step_correlation"] = learn.get("v_true_step_correlation") or {}
    diag["t_gap_cancellation"] = r.get("t_gap_cancellation") or {}
    gate = r.get("decision_gate") or {}
    diag["decision_gate"] = {"proceed": gate.get("proceed"),
                             "stratum": gate.get("stratum")}
    return diag


def _recompute_diagnostics(ds, n_samples: int = 384) -> dict:
    """Recompute the cheap learnability stats from a dataset subsample."""
    data_dir = Path(ds.data_dir)
    entries = ds.entries
    pairs = ds.pairs
    idxs = list(range(len(pairs)))
    ds.rng.shuffle(idxs)
    delta_by_lag: Dict[int, List[Tuple[int, torch.Tensor]]] = {}
    delta_by_shape: Dict[Tuple[int, int], List[torch.Tensor]] = {}
    affine: Dict[Tuple[int, int], Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
    gap_delta: List[float] = []
    loaded = 0
    for i in idxs:
        if loaded >= n_samples:
            break
        gi, slot, t, lag = pairs[i]
        if lag == 0:  # synthesized anchor (v_MA == v_true) — no staleness signal
            continue
        entry = entries[gi]
        try:
            x, v, vt = refiner_data.load_pair_tensors(
                data_dir / entry["bin"], (t, slot, lag))
        except KeyError:
            continue
        loaded += 1
        d = vt - v
        shape = (int(d.shape[-2]), int(d.shape[-1]))
        region = min(2, 3 * t // max(int(entry.get("num_steps", 1)), 1))
        if len(delta_by_lag.setdefault(lag, [])) < 48:
            delta_by_lag[lag].append((region, d))
        if lag == 1 and len(delta_by_shape.setdefault(shape, [])) < 128:
            delta_by_shape[shape].append(d)
        avm, avt = affine.setdefault(shape, ([], []))
        if len(avm) < 256:
            avm.append(v)
            avt.append(vt)
        if len(gap_delta) < 512:
            gap_delta.append(d.abs().mean().item())
    if not delta_by_shape and not delta_by_lag:
        raise ValueError("no lag>0 pairs available for diagnostics")
    diag = {"source": "recomputed", "pairs_loaded": loaded}
    diag["svd"] = {f"{s[0]}x{s[1]}": svd_rank_95(delta_by_shape[s])["rank_95"]
                   for s in sorted(delta_by_shape)}
    diag["affine_ceiling"] = {f"{s[0]}x{s[1]}":
                              round(per_channel_affine_ceiling(*affine[s]), 4)
                              for s in sorted(affine)}
    diag["staleness_curve"] = staleness_curve(delta_by_lag)
    diag["delta_distribution"] = {
        f"{s[0]}x{s[1]}": delta_distribution(delta_by_shape[s])
        for s in sorted(delta_by_shape)}
    step_stats = _step_stats_from_pairs(data_dir, entries)
    gap_true = step_stats.pop("_gap_true", None)
    diag["v_true_step_correlation"] = step_stats
    gap_delta_m = round(float(sum(gap_delta)) / len(gap_delta), 6) if gap_delta else None
    diag["t_gap_cancellation"] = {
        "mean_abs_v_true_step_delta": gap_true,
        "mean_abs_delta_ma": gap_delta_m,
        "ratio": round(gap_delta_m / max(gap_true, 1e-9), 3)
        if gap_delta_m is not None and gap_true else None,
    }
    return diag


def _step_stats_from_pairs(data_dir: Path, entries: list,
                           max_gens: int = 2, max_steps: int = 48) -> dict:
    """v_true step-to-step cosine per region + the t-gap step delta.

    Uses the d=0 anchor (always present) of the first few generations so no
    lag-dependent pair index is needed.
    """
    seq_by_shape: Dict[Tuple[int, int], List[torch.Tensor]] = {}
    gap_true: List[float] = []
    for entry in entries[:max_gens]:
        slots = entry.get("slots") or []
        if not slots:
            continue
        n_steps = int(entry.get("num_steps", 1))
        prev = {}
        for t in range(min(n_steps, max_steps)):
            try:
                _, _, vt = refiner_data.load_pair_tensors(
                    data_dir / entry["bin"], (t, slots[0], 0))
            except KeyError:
                continue
            shape = (int(vt.shape[-2]), int(vt.shape[-1]))
            seq_by_shape.setdefault(shape, []).append(vt)
            if t > 0 and shape in prev:
                gap_true.append((vt.float() - prev[shape].float()).abs().mean().item())
            prev[shape] = vt
    out = step_correlations(seq_by_shape)
    out["_gap_true"] = round(float(sum(gap_true)) / len(gap_true), 6) if gap_true else None
    return out


def print_corpus_diagnostics(diag: dict) -> None:
    """Print the diagnostics block (pre-train; plain print — no bar yet)."""
    print(f"  ── Corpus diagnostics ({diag.get('source', '?')}) ──")
    svd = diag.get("svd") or {}
    if svd:
        print("  SVD rank(95%) Δ_MA:    " + "  ".join(
            f"{k}={v}" for k, v in sorted(svd.items())))
    pca = diag.get("affine_ceiling") or {}
    if pca:
        print("  Affine ceiling (1−R²):  " + "  ".join(
            f"{k}={v:.4f}" for k, v in sorted(pca.items())))
    pf = diag.get("pooled_feature_ceiling")
    if pf is not None:
        print(f"  Pooled-feature ceiling: {pf:.4f}")
    tg = diag.get("t_gap_cancellation") or {}
    if tg.get("ratio") is not None:
        print(f"  t-gap cancellation:   |Δ_MA|={tg['mean_abs_delta_ma']} vs "
              f"|v_true(t)−v_true(t−1)|={tg['mean_abs_v_true_step_delta']} "
              f"→ ratio {tg['ratio']}")
    stal = diag.get("staleness_curve") or {}
    if stal:
        print(f"  Staleness monotone:   {stal.get('per_region_monotone')}")
        for r in range(3):
            region = stal.get(f"region_{r}")
            if region:
                print(f"      staleness region {r}: " + "  ".join(
                    f"d{lag}={v:.4f}" for lag, v in sorted(region.items())))
    dd = diag.get("delta_distribution") or {}
    if dd:
        print("  Δ_MA distribution:      " + "  ".join(
            f"{k}: μ={v.get('mean')} σ={v.get('std')} "
            f"sparsity={v.get('sparsity')}" for k, v in sorted(dd.items())))
    sc = diag.get("v_true_step_correlation") or {}
    parts = [f"{k}={v}" for k, v in sorted(sc.items())
             if k != "_gap_true" and v is not None]
    if parts:
        print("  v_true step corr:     " + "  ".join(parts))
    gate = diag.get("decision_gate") or {}
    if gate.get("proceed") is not None:
        print(f"  Decision gate ({gate.get('stratum')}): "
              f"{'PROCEED' if gate['proceed'] else 'RECONSIDER'}")


# ── Checkpointing (plan 6i) ───────────────────────────────────────────


def save_full_state(path, model, ema, opt, scaler, scheduler, step, cfg, best,
                    best_step: Optional[int] = None, wall_s: Optional[float] = None):
    torch.save({
        "model": model.state_dict(),
        "ema": ema,
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "best": best,
        "best_step": best_step,
        "wall_s": wall_s,
        "config_snapshot": cfg,
    }, path)


def config_drift(snapshot: dict, cfg: dict) -> List[str]:
    keys = ["model_size", "depth", "optimizer", "lr", "batch_size", "loss",
            "max_steps", "refine_passes_max", "multipass",
            "resolution_curriculum", "accumulate_big_batches", "scale_aug",
            "recovery_fit_batches", "recovery_eval_batches"]
    drift = []
    for k in keys:
        if snapshot.get(k) != cfg.get(k):
            drift.append(f"  {k}: checkpoint={snapshot.get(k)!r} current={cfg.get(k)!r}")
    return drift


# ── Main ──────────────────────────────────────────────────────────────


def main(argv=None):
    args = parse_args(argv)
    cfg = resolve_config(args)
    data_dir = Path(cfg["data"])
    if not (data_dir / "manifest.json").exists():
        raise SystemExit(f"[train_corrector] no manifest.json in {data_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    size = cfg["model_size"]

    torch.manual_seed(cfg["seed"])

    # ── Datasets ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  Latent Corrector Training")
    print("=" * 60)
    print(f"  Data:           {data_dir}")
    print(f"  Size:           {size} (depth {cfg['depth']})   optimizer: {cfg['optimizer']}  lr={cfg['lr']}")
    print(f"  Multipass:      {cfg['multipass']} (K_max={1 if not cfg['multipass'] else cfg['refine_passes_max']}, "
          f"curriculum={cfg['k_curriculum']}, stop_grad={cfg['stop_grad']})")
    print(f"  Conditioning:   lag={cfg['lag_cond']}  res={cfg['res_cond']}  "
          f"lag-loss-norm={cfg['lag_loss_normalize']}  de-burst={cfg['deburst_windows']}")
    print(f"  Loss:           {cfg['loss']}   batch={cfg['batch_size']}  "
          f"max_steps={cfg['max_steps']}")
    print(f"  Cache:          {cfg['cache_size']} generations   "
          f"loader workers: {cfg['num_workers']}")
    print(f"  Device:         {device}")

    print("  Preparing dataset (pair index + normalization stats)...")
    manifest = refiner_data.load_manifest(data_dir)
    ds = CorrectorDataset(
        data_dir,
        seed=cfg["seed"],
        lag_weights=cfg["lag_weights"],
        compute_normalization=(cfg["normalization"] == "perchannel"),
        normalization_samples=cfg["normalization_samples"],
        show_progress=not cfg["no_progress"],
        manifest=manifest,
        gen_cache_size=cfg["cache_size"],
        synthesize_anchors=False,
    )
    if not len(ds):
        raise SystemExit("[train_corrector] no training pairs (all generations eval?)")
    print(f"  Training pairs: {len(ds)} (stale-only, no d=0 anchors)   eval pairs: "
          f"{len(CorrectorDataset(data_dir, only_eval=True, normalization_samples=0, manifest=manifest))}")
    try:
        print_corpus_diagnostics(corpus_diagnostics(data_dir, ds))
    except Exception as e:
        print(f"  ⚠ corpus diagnostics unavailable: {e}")

    # Resolution curriculum (stage 1 = cheap stratum only, then a ramped mix
    # to full): the stage-1 sampler filters the pair buckets by latent area,
    # so its batches are ~4× cheaper while the model builds scale-invariant
    # features; the ramp interpolates stage1 → full mix over
    # stage2_ramp_fraction of the run (no hard switch — hard switches let the
    # 64×64-learned field degrade the 1024² strata it never trained on).
    curriculum = cfg["resolution_curriculum"]
    # Tuple-keyed view for the sampler (bucket keys are (h, w) tuples).
    res_weights = {}
    for key, value in (curriculum.get("resolution_weights") or {}).items():
        h, w = (int(x) for x in key.split("x"))
        res_weights[(h, w)] = value
    sampler = CorrectorBatchSampler(
        ds, batch_size=cfg["batch_size"], seed=cfg["seed"],
        bucket_weights=res_weights or None,
        deburst_windows=cfg["deburst_windows"],
        deburst_window_runs=cfg["deburst_window_runs"])
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_corrector,
                        num_workers=cfg["num_workers"],
                        pin_memory=cfg["num_workers"] > 0)
    steps_per_epoch = max(len(sampler), 1)
    print(f"  Batches/epoch:  {steps_per_epoch}   "
          f"(~{cfg['max_steps'] / steps_per_epoch:.1f} epochs over {cfg['max_steps']} steps)")

    stage1_loader = None
    stage1_until = 0
    ramp_start = 0
    ramp_end = 0
    if curriculum.get("enabled"):
        s1_sampler = CorrectorBatchSampler(
            ds, batch_size=cfg["batch_size"], seed=cfg["seed"] + 7,
            include_areas=curriculum["stage1_areas"],
            bucket_weights=res_weights or None,
            deburst_windows=cfg["deburst_windows"],
            deburst_window_runs=cfg["deburst_window_runs"])
        if s1_sampler.buckets:
            stage1_loader = DataLoader(ds, batch_sampler=s1_sampler,
                                       collate_fn=collate_corrector,
                                       num_workers=cfg["num_workers"],
                                       pin_memory=cfg["num_workers"] > 0)
            stage1_until = int(cfg["max_steps"] * curriculum["stage1_fraction"])
            ramp_steps = int(cfg["max_steps"] * curriculum["stage2_ramp_fraction"])
            ramp_start = stage1_until
            ramp_end = min(ramp_start + ramp_steps, cfg["max_steps"])
            print(f"  Curriculum:     stage 1 (steps ≤ {ramp_start}) → ramp over "
                  f"{ramp_end - ramp_start} steps → full mix; areas "
                  f"{curriculum['stage1_areas']}")
        else:
            print(f"  ⚠ resolution curriculum enabled but no pairs at "
                  f"{curriculum['stage1_areas']} — training full mix only")
    print(f"  Scale aug:      {cfg['scale_aug']} (0.75× round-trip)   "
          f"accumulate ÷4 buckets: {cfg['accumulate_big_batches']}")
    eval_loader = None
    eval_ds = None
    try:
        eval_ds = CorrectorDataset(data_dir, only_eval=True, seed=cfg["seed"],
                                   normalization_samples=0, manifest=manifest)
        eval_max_batches = cfg["eval_max_batches"] or None
        eval_sampler = CorrectorBatchSampler(eval_ds, batch_size=16,
                                             seed=cfg["seed"] + 1,
                                             max_batches=eval_max_batches,
                                             min_batches_per_shape=cfg[
                                                 "eval_min_batches_per_shape"])
        # Fixed eval set: the batch index list is built once and replayed at
        # every eval, so the K=0 base and the per-shape slices are identical
        # across evals (the sampler's epoch counter would otherwise resample
        # each evaluation, making the recovery curve a moving target).
        eval_batches = list(eval_sampler)
        eval_loader = DataLoader(eval_ds,
                                 batch_sampler=FixedBatchListSampler(eval_batches),
                                 collate_fn=collate_corrector,
                                 num_workers=cfg["num_workers"],
                                 pin_memory=cfg["num_workers"] > 0)
    except ValueError as e:
        print(f"  ⚠ no eval generations — eval loop and gates skipped ({e})")
    # Fit maps for the OOD affine recovery row: drawn from the TRAIN pairs
    # only (never the eval set), balanced per stratum across every generation
    # (see collect_fit_maps), so the affine faces the same distribution
    # contract as the corrector (fit=train, scored=eval).
    eval_shapes = set(eval_ds.pair_shapes()) if eval_ds is not None else set()
    lags = sorted(set(lag for _, _, _, lag in ds.pairs))
    if 0 not in lags:
        lags = [0] + lags

    # ── Model ─────────────────────────────────────────────────────────
    try:
        ccfg = CorrectorConfig.for_size(size, depth=cfg["depth"],
                                        lag_cond=cfg["lag_cond"],
                                        res_cond=cfg["res_cond"])
    except ValueError as e:
        raise SystemExit(f"[train_corrector] {e}")
    if cfg["normalization"] == "perchannel":
        ccfg.normalization = "perchannel"
        ccfg.normalization_stats = ds.normalization_stats
    model = CorrectorUNet2D(ccfg)
    step = 0
    ema = init_ema(model)
    best = None
    best_step: Optional[int] = None
    opt = None
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    scheduler = None
    sd = None

    # ── Resume (plan 6i): model state loads BEFORE the optimizer is built, so
    # a pre-conditioning checkpoint (no lag/res embedders) can downgrade the
    # model's config first and the optimizer's parameter groups stay in sync.
    if cfg["resume"]:
        sd = torch.load(cfg["resume"], map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(sd["model"], strict=False)
        if missing:
            cond_keys = {k for k in missing
                         if "lag_embedder" in k or "res_embedder" in k}
            if set(missing) == cond_keys and model.cfg.lag_cond:
                # Pre-conditioning checkpoint: rebuild the model without the
                # embedder modules so the optimizer's parameter groups (built
                # below from this model) match the checkpoint's.
                print("  ⚠ resume: pre-conditioning checkpoint — embedders "
                      "dropped, conditioning disabled")
                model.cfg.lag_cond = False
                model.cfg.res_cond = False
                model = CorrectorUNet2D(model.cfg)
                missing2, unexpected2 = model.load_state_dict(
                    sd["model"], strict=False)
                if missing2:
                    raise RuntimeError(f"[train_corrector] resume: missing keys "
                                       f"{sorted(missing2)}")
                unexpected = unexpected2
            else:
                raise RuntimeError(f"[train_corrector] resume: missing keys "
                                   f"{sorted(missing)}")
        if unexpected:
            print(f"  ⚠ resume: unexpected keys {sorted(unexpected)} ignored")
        ema = {k: v.to(device) for k, v in sd["ema"].items()}
        step = int(sd["step"])
        best = sd.get("best")
        best_step = sd.get("best_step")
        snapshot = sd.get("config_snapshot") or {}
        drift = config_drift(snapshot, cfg)
        if drift:
            print("  ⚠ config drift vs checkpoint snapshot (resume):")
            print("\n".join(drift))
        else:
            print(f"  Resumed from {cfg['resume']} at step {step}")
        if sd.get("wall_s") is not None:
            print(f"  Previously elapsed: {format_duration(sd['wall_s'])}")

    if cfg["optimizer"] == "sophia":
        opt = SophiaG(model.parameters(), lr=cfg["lr"], rho=cfg["rho"],
                      hessian_every=cfg["hessian_every"], weight_decay=cfg["wd"])
    elif cfg["optimizer"] == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95),
                                eps=1e-8, weight_decay=cfg["wd"])
    elif cfg["optimizer"] == "ademamix":
        import pytorch_optimizer as po
        opt = po.AdEMAMix(model.parameters(), lr=cfg["lr"],
                          betas=(0.9, 0.999, 0.9999), alpha=5.0,
                          weight_decay=cfg["wd"], weight_decouple=True)
    elif cfg["optimizer"] == "schedulefree":
        import pytorch_optimizer as po
        opt = po.ScheduleFreeAdamW(model.parameters(), lr=cfg["lr"],
                                   weight_decay=cfg["wd"], warmup_steps=max(1, cfg["max_steps"] // 20))
        opt.train()
    else:
        raise SystemExit(f"[train_corrector] unknown optimizer {cfg['optimizer']!r}")

    if cfg["optimizer"] != "schedulefree":
        warmup = max(1, int(cfg["max_steps"] * 0.05))
        lin = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01,
                                                total_iters=warmup)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                         T_max=max(cfg["max_steps"] - warmup, 1))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, [lin, cos], milestones=[warmup])

    if cfg["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)
    ema = {k: v.to(device) for k, v in ema.items()}
    print(f"  Params:         target {ccfg.target_params/1e6:.2f}M → "
          f"{model.num_params()/1e6:.2f}M achieved (depth {ccfg.num_blocks}, "
          f"bottleneck {ccfg.bottleneck_dim}, {ccfg.heads} heads)")
    flops = estimate_corrector_flops(ccfg)
    print(f"  FLOPs/pass @512²: {flops/1e9:.1f} GFLOP "
          f"({flops/FULL_STEP_FLOP_512:.2%} of a full model step)")

    # Optimizer state loads after construction (the parameter groups were
    # built from the reconciled model above).
    if sd is not None:
        opt.load_state_dict(sd["optimizer"])
        scaler.load_state_dict(sd["scaler"])
        if scheduler is not None and sd.get("scheduler"):
            scheduler.load_state_dict(sd["scheduler"])
    ema_model = CorrectorUNet2D(ccfg).to(device)
    if cfg["compile"]:
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile enabled — first steps include compile latency")

    loss_fn = LOSS_FNS[cfg["loss"]]
    eps = ds.rel_mse_eps
    # Per-lag loss normalization (R5): w_lag = mean(e)/e_lag so each lag
    # contributes ~equal gradient mass regardless of difficulty. Estimated
    # once from a seeded train-pair subsample; lags without estimates get 1.0.
    lag_scale_t: Optional[torch.Tensor] = None
    if cfg["lag_loss_normalize"]:
        base_err = estimate_per_lag_base_rel(ds, seed=cfg["seed"] + 3)
        if base_err:
            lags_sorted = sorted(base_err)
            es = torch.tensor([base_err[l] for l in lags_sorted],
                              dtype=torch.float32)
            w = (es.mean() / es.clamp_min(1e-6)).clamp(0.2, 5.0)
            w = w / w.mean()
            lag_scale_t = torch.ones(max(lags_sorted) + 1, dtype=torch.float32)
            for lag, wv in zip(lags_sorted, w.tolist()):
                lag_scale_t[lag] = wv
            lag_scale_t = lag_scale_t.to(device)
            print("  Lag loss norm:  " + "  ".join(
                f"lag{l}={wv:.2f}" for l, wv in zip(lags_sorted, w.tolist())))
        else:
            print("  ⚠ lag_loss_normalize enabled but no per-lag estimates "
                  "could be computed — training unweighted")
    k_max_final = cfg["refine_passes_max"] if cfg["multipass"] else 1
    # K=0 is the TeaCache base (v = v_MA, zero passes) — the recovery-table
    # baseline; K>=1 are the corrector passes. The eval ladder alternates
    # between the light [0,1] form (every eval) and the full form (every
    # eval_full_k_every-th eval) to keep the K-robustness check without the
    # 3× pass cost on every eval.
    ks_eval_full = [0] + [k for k in (1, 2, 3) if k <= k_max_final]
    ks_eval_light = [0] + ([1] if k_max_final >= 1 else [])
    eval_plan = build_eval_plan(cfg)
    next_eval = next((s for s in eval_plan if s > step), None)
    n_evals_done = sum(1 for s in eval_plan if s <= step)
    if eval_plan:
        print(f"  Eval plan:      {eval_plan}")
    else:
        print(f"  Eval plan:      none (max_steps {cfg['max_steps']} < eval_every "
              f"{cfg['eval_every']}) — training only")
    ceiling = load_linear_ceiling(data_dir)
    if ceiling is not None:
        print(f"  Linear ceiling: {ceiling:.4f} (1−R², probe) — did-it-learn target "
              f"(ladder-only K=1) ≤ {0.8 * ceiling:.4f}")
    else:
        print("  Linear ceiling: not found (no probe report) — did-it-learn gate skipped")
    gate_deadline = 6 * cfg["eval_every"]
    gate_fired = False

    if cfg["channels_last"]:
        def to_cl(b):
            return {k: (v.to(memory_format=torch.channels_last) if v.ndim == 4 else v)
                    for k, v in b.items()}
    else:
        def to_cl(b):
            return b

    aug_rng = random.Random(cfg["seed"] + 1000)
    active_loader = stage1_loader if stage1_loader is not None else loader
    iter_loader = iter(active_loader)
    t_start = time.time()
    timer = TrainTimer(window=100)
    metrics = MetricsLog(cfg["metrics"])
    loss_ema: Optional[float] = None
    min_loss = float("inf")
    last_eval_rows: List[dict] = []
    last_affine: Optional[dict] = None
    affine_fit_maps: Optional[dict] = None
    pbar = None
    if tqdm is not None and not cfg["no_progress"]:
        pbar = tqdm(total=cfg["max_steps"], desc="train", unit="step",
                    initial=step, dynamic_ncols=True, mininterval=0.5)

    def emit(line: str) -> None:
        """Print a log line above the progress bar (bar-safe)."""
        if pbar is not None:
            pbar.write(line)
        else:
            print(line)

    cur_rng = random.Random(cfg["seed"] + 2000)

    def pick_loader(step: int):
        """Curriculum schedule: stage-1 loader until ramp_start, then a
        probabilistic ramp to the full-mix loader over [ramp_start, ramp_end]."""
        if stage1_loader is None:
            return loader
        if step <= ramp_start:
            return stage1_loader
        if step >= ramp_end:
            return loader
        p = (step - ramp_start) / max(ramp_end - ramp_start, 1)
        return loader if cur_rng.random() < p else stage1_loader

    while step < cfg["max_steps"]:
        step += 1
        t_step0 = time.time()
        want = pick_loader(step)
        if want is not active_loader:
            active_loader = want
            iter_loader = iter(active_loader)
            if step == ramp_start + 1 and stage1_loader is not None:
                emit(f"  [curriculum] ramp to full resolution mix from step {step}")
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(active_loader)
            batch = next(iter_loader)
        batch = {k: v.to(device) for k, v in batch.items()}
        batch = to_cl(augment_batch(batch, aug_rng, scale_aug=cfg["scale_aug"]))

        k_max_cur = k_max_final
        if cfg["k_curriculum"] and cfg["multipass"]:
            k_max_cur = 1 + int(round((k_max_final - 1) * step / cfg["max_steps"]))

        if isinstance(opt, SophiaG) and step % cfg["hessian_every"] == 0:
            t_h0 = time.time()
            # update_hessian double-backpropagates (gradient of the gradient);
            # the fused SDPA backends (flash / memory-efficient) have no
            # second-order derivative, so build the graph on the math backend
            # (bmm/softmax/matmul — see SophiaG.update_hessian docstring).
            with torch.autocast("cuda", enabled=False), \
                    torch.nn.attention.sdpa_kernel(
                        torch.nn.attention.SDPBackend.MATH):
                dv = model(batch["x_t"].float(), batch["v_ma"].float(),
                           batch["prompt"].float(), batch["t_frac"].float(),
                           batch["prompt_mask"], lag=batch["lag"].float())
                per_s = per_sample_rel_mse(batch["v_ma"].float() + dv,
                                           batch["v_true"].float(), eps)
                if lag_scale_t is not None:
                    per_s = per_s * lag_scale_t[batch["lag"]]
                loss_h = per_s.mean()
            opt.update_hessian(loss_h)
            timer.add("hessian", time.time() - t_h0)

        # Gradient accumulation: ÷4 buckets (128² / 1024×512) are accumulated
        # up to a full batch so every optimizer step sees ~batch_size samples
        # (equal gradient-noise level across resolutions).
        opt.zero_grad(set_to_none=True)
        n_micro = max(1, cfg["batch_size"] // int(batch["x_t"].shape[0])) \
            if cfg["accumulate_big_batches"] else 1
        micro_losses: List[torch.Tensor] = []
        for m in range(n_micro):
            if m > 0:
                try:
                    batch = next(iter_loader)
                except StopIteration:
                    iter_loader = iter(active_loader)
                    batch = next(iter_loader)
                batch = {k: v.to(device) for k, v in batch.items()}
                batch = to_cl(augment_batch(batch, aug_rng, scale_aug=cfg["scale_aug"]))
            Ks = torch.randint(1, k_max_cur + 1, (batch["x_t"].shape[0],),
                               device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = compute_train_loss(model, batch, Ks, loss_fn, eps,
                                          cfg["stop_grad"], cfg["multipass"],
                                          k_max_cur, lag_scale=lag_scale_t)
            micro_losses.append(loss.detach())
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        ema_update(model, ema, step, decay=cfg["ema_decay"])
        timer.record_step(time.time() - t_step0)

        loss_v = float(sum(micro_losses) / max(len(micro_losses), 1))
        loss_ema = loss_v if loss_ema is None else 0.99 * loss_ema + 0.01 * loss_v
        min_loss = min(min_loss, loss_v)
        grad_norm_v = float(grad_norm) if grad_norm is not None else float("nan")

        if pbar is not None:
            pbar.set_postfix(
                loss=f"{loss_ema:.5f}",
                lr=f"{opt.param_groups[0]['lr']:.1e}",
                K=k_max_cur,
                it=f"{timer.steps_per_sec():.2f}/s",
                rem=format_duration(timer.remaining_seconds(
                    cfg["max_steps"] - step,
                    sum(1 for s in eval_plan if s > step))),
                refresh=False)
            pbar.update(1)

        if step % 50 == 0 or step == 1:
            wall = time.time() - t_start
            rem_s = timer.remaining_seconds(
                cfg["max_steps"] - step, sum(1 for s in eval_plan if s > step))
            line = (f"  [train] step {step:>6d}/{cfg['max_steps']}  "
                    f"loss={loss_v:.5f} (ema {loss_ema:.5f}, min {min_loss:.5f})  "
                    f"lr={opt.param_groups[0]['lr']:.2e}  "
                    f"it/s={timer.steps_per_sec():.2f}  "
                    f"elapsed={format_duration(wall)}  "
                    f"remaining~{format_duration(rem_s)}")
            emit(line)
            metrics.write({
                "type": "step", "step": step,
                "epoch": step / steps_per_epoch,
                "loss": loss_v, "loss_ema": loss_ema, "min_loss": min_loss,
                "grad_norm": grad_norm_v, "lr": opt.param_groups[0]["lr"],
                "k_max": k_max_cur, "wall_s": wall,
                "vram_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
            })

        if step == next_eval and eval_loader is not None:
            t_e0 = time.time()
            ks_eval = (ks_eval_full if n_evals_done % cfg["eval_full_k_every"] == 0
                       else ks_eval_light)
            # Eval the EMA model — it is the artifact that ships and the one
            # best-checkpoint selection is based on (the online weights are
            # only training machinery).
            ema_model.load_state_dict(ema)
            results, by_shape, by_t, eval_maps = eval_model(
                ema_model, eval_loader, lags, ks_eval, device,
                eps, show_progress=(pbar is not None),
                eval_map_batches=int(cfg.get("recovery_eval_batches", 64)))
            k1_pairs = [r["rel_mse"] for (kk, _), r in results.items() if kk == 1]
            k1 = sum(k1_pairs) / max(len(k1_pairs), 1) if k1_pairs else None
            affine = None
            if affine_fit_maps is None and eval_shapes:
                affine_fit_maps = collect_fit_maps(
                    ds, eval_shapes, seed=cfg["seed"] + 6,
                    cap_batches_per_shape=int(cfg.get("recovery_fit_batches", 128)),
                    show_progress=(pbar is not None), desc="affine-fit")
            if affine_fit_maps and eval_maps:
                aff = affine_oob_eval(affine_fit_maps, eval_maps)
                lad, anc = split_affine_per_pair(aff, eval_maps)
                aff["split"] = {"ladder": round(sum(lad) / len(lad), 5) if lad else None,
                                "anchor": round(sum(anc) / len(anc), 5) if anc else None}
                affine = aff
                del eval_maps  # release the eval-map tensors before training resumes
            row = "  [eval] step %d  " % step
            for k in ks_eval:
                avg = sum(r["rel_mse"] for (kk, _), r in results.items() if kk == k)
                n = sum(1 for (kk, _) in results if kk == k)
                row += f"K{k}={avg/max(n,1):.4f}  "
            emit(row)
            for lag in lags:
                parts = [f"K{k}={results.get((k, lag), {}).get('rel_mse', float('nan')):.4f}"
                         for k in ks_eval]
                emit(f"      lag {lag:>2d}: " + "  ".join(parts))
            # Per-lag recovery vs the K=0 base (K=0 per-lag is in `results`):
            # makes the lag-1 over-correction / lag-16 under-correction
            # visible instead of hiding it in the pooled number.
            per_lag_rec = {}
            for lag in lags:
                if lag == 0:
                    continue
                r0 = results.get((0, lag), {}).get("rel_mse")
                r1 = results.get((1, lag), {}).get("rel_mse")
                if r0 and r1 is not None and r1 == r1:
                    per_lag_rec[lag] = 1.0 - r1 / r0
            if per_lag_rec:
                emit("      recovery per lag: " + "  ".join(
                    f"lag{lag}={100 * rec:+.1f}%" for lag, rec in per_lag_rec.items()))
            shape_keys = sorted({hw for (_, hw) in by_shape}, key=lambda hw: hw[0] * hw[1])
            for hw in shape_keys:
                parts = []
                for k in ks_eval:
                    r = by_shape.get((k, hw))
                    parts.append(f"K{k}={r['rel_mse']:.4f}" if r is not None else f"K{k}=  -  ")
                emit(f"      {hw[0]}x{hw[1]}: " + "  ".join(parts))
            # Per-t step-region slices (early/mid/late thirds, probe §7):
            # ladder-only recovery per region plus the K=1 t×lag and t×shape
            # breakdowns (per-cell recovery % in parens) — where does the
            # corrector help, instead of the pooled average hiding it.
            t_regions = sorted({r for (r, _), _ in by_t["pooled"].items()})

            def _t_lad_mean(reg: int, k: int):
                tot = 0.0
                n = 0
                for (rr, kk, lag), st in by_t["lags"].items():
                    if rr == reg and kk == k and lag != 0:
                        tot += st["rel_mse"] * st["n"]
                        n += st["n"]
                return tot / n if n else None

            per_t_rec = {}
            for region in t_regions:
                r0 = _t_lad_mean(region, 0)
                r1 = _t_lad_mean(region, 1)
                if r0 and r1 is not None and r1 == r1:
                    per_t_rec[region] = 1.0 - r1 / r0
            if per_t_rec:
                emit("      recovery per t (ladder-only): " + "  ".join(
                    f"{EVAL_T_REGION_NAMES.get(r, r)}={100 * rec:+.1f}%"
                    for r, rec in sorted(per_t_rec.items())))
            for region in t_regions:
                parts = []
                for lag in lags:
                    if lag == 0:
                        continue
                    st1 = by_t["lags"].get((region, 1, lag))
                    if st1 is None:
                        continue
                    rec = None
                    st0 = by_t["lags"].get((region, 0, lag))
                    if st0 and st0["rel_mse"]:
                        rec = 1.0 - st1["rel_mse"] / st0["rel_mse"]
                    parts.append(f"d{lag}={st1['rel_mse']:.4f}"
                                 + (f"({100 * rec:+.1f}%)" if rec is not None else ""))
                if parts:
                    emit(f"      t {EVAL_T_REGION_NAMES.get(region, region)} × lag (K1): "
                         + "  ".join(parts))
            for region in t_regions:
                parts = []
                for hw in shape_keys:
                    st1 = by_t["shapes"].get((region, 1, hw))
                    if st1 is None:
                        continue
                    rec = None
                    st0 = by_t["shapes"].get((region, 0, hw))
                    if st0 and st0["rel_mse"]:
                        rec = 1.0 - st1["rel_mse"] / st0["rel_mse"]
                    parts.append(f"{hw[0]}x{hw[1]}={st1['rel_mse']:.4f}"
                                 + (f"({100 * rec:+.1f}%)" if rec is not None else ""))
                if parts:
                    emit(f"      t {EVAL_T_REGION_NAMES.get(region, region)} × shape (K1): "
                         + "  ".join(parts))
            rows = recovery_rows(results, ks_eval)
            last_eval_rows = rows
            last_affine = affine
            bl = next((r["ladder_only"] for r in rows if r["k"] == 0), None)
            # Ladder-only K=1 error (lag ≥ 1, excluding the synthesized d=0
            # anchors): the anchor-diluted pooled k1 lets a no-op corrector
            # pass trivially, so gates and best-checkpoint selection use this
            # split (plan 6h).
            k1_lad = next((r["ladder_only"] for r in rows if r["k"] == 1),
                          None)
            for line2 in recovery_table_lines(rows, affine):
                emit(line2)
            if ceiling is not None and not gate_fired and step >= gate_deadline:
                if (k1_lad is None or k1_lad != k1_lad
                        or k1_lad > 0.8 * ceiling):
                    emit("  ✗ BELOW LINEAR CEILING (ladder-only K=1) — "
                         "did-it-learn gate fired; stopping per plan 6h")
                    step = cfg["max_steps"]
                    gate_fired = True
            # K-robustness + per-lag coverage (report-only warnings; needs the
            # full ladder, so light [0,1] evals skip it). Both compare on the
            # ladder-only split: d=0 anchors score ≈0 for every K and would
            # compress the pooled gap / inflate the per-lag growth from lag 0.
            if ks_eval[-1] > 1:
                r1_lad = next((r["ladder_only"] for r in rows if r["k"] == 1),
                              None)
                r3_lad = next((r["ladder_only"] for r in rows
                               if r["k"] == max(ks_eval)), None)
                if (r1_lad is not None and r3_lad is not None
                        and r1_lad == r1_lad and r3_lad == r3_lad
                        and r3_lad > 0 and (r1_lad - r3_lad) / r3_lad > 0.2):
                    emit("  ⚠ K-robustness gap large (K=1 ≫ K=K_max, "
                         "ladder-only) — consider strengthening the K "
                         "curriculum (plan 6h)")
            per_lag_k1 = [(lag, results.get((1, lag), {}).get("rel_mse",
                                                             float("nan")))
                          for lag in lags if lag != 0]
            finite = [(lag, x) for lag, x in per_lag_k1 if x == x]
            if len(finite) >= 3 and finite[0][1] > 0:
                growth = (finite[-1][1] - finite[0][1]) / finite[0][1]
                if growth > 1.5:
                    emit(f"  ⚠ Per-lag rel-MSE grows {growth:.2f}× from lag "
                         f"{finite[0][0]} to lag {finite[-1][0]} — check the "
                         "per-lag coverage gate (plan 6h); extend record_lags "
                         "if superlinear")

            # Checkpointing (plan 6i)
            ema_model.load_state_dict(ema)
            cfg_meta = {k: v for k, v in cfg.items()
                        if k not in ("data", "resume", "out", "compile",
                                     "channels_last", "metrics", "no_progress")}
            cfg_meta["lag_cond"] = ccfg.lag_cond
            cfg_meta["res_cond"] = ccfg.res_cond
            if k1_lad is not None and k1_lad == k1_lad and (
                    best is None or k1_lad < best):
                best = k1_lad
                best_step = step
                save_corrector(ema_model, out_dir / f"corrector-{size}-best.safetensors",
                               ccfg, extra_metadata={"config_snapshot": cfg_meta,
                                                     "best_rel_mse_k1_ladder": k1_lad})
            save_corrector(ema_model, out_dir / f"corrector-{size}-{step}.safetensors",
                           ccfg, extra_metadata={"config_snapshot": cfg_meta})
            for old in sorted(out_dir.glob(f"corrector-{size}-*.safetensors")):
                if f"-{step}." in old.name or "-best." in old.name:
                    continue
                old.unlink(missing_ok=True)
            save_full_state(out_dir / f"corrector-{size}-train.pt", model, ema, opt,
                            scaler, scheduler, step, cfg, best, best_step=best_step,
                            wall_s=time.time() - t_start)
            emit(f"      saved corrector-{size}-{step}.safetensors"
                 + (f" (best ladder K1={best:.4f} @ {best_step})"
                    if best is not None else ""))
            timer.record_eval(time.time() - t_e0)
            n_evals_done += 1
            next_eval = (eval_plan[n_evals_done]
                         if n_evals_done < len(eval_plan) else None)
            for line2 in timer.summary_lines(time.time() - t_start, step,
                                             cfg["max_steps"], eval_plan):
                emit(line2)
            metrics.write({
                "type": "eval", "step": step, "k1": k1,
                "k_avg": {str(k): round(sum(r["rel_mse"] for (kk, _), r in results.items()
                                            if kk == k) / max(sum(1 for (kk, _) in results if kk == k), 1), 6)
                          for k in ks_eval},
                "per_lag": {str(lag): results.get((1, lag), {}).get("rel_mse")
                            for lag in lags},
                "recovery_per_lag": {str(lag): round(rec, 4)
                                     for lag, rec in per_lag_rec.items()},
                "per_shape": {f"{h}x{w}": {str(k): round(by_shape.get((k, (h, w)), {}).get("rel_mse", float("nan")), 6)
                                           for k in ks_eval}
                              for (h, w) in shape_keys},
                "per_t": {str(r): {str(k): round(st["rel_mse"], 6)
                                   for (rr, k), st in by_t["pooled"].items()
                                   if rr == r} for r in t_regions},
                "recovery_per_t": {str(r): round(rec, 4)
                                   for r, rec in per_t_rec.items()},
                "per_t_lag": {str(r): {str(lag): round(by_t["lags"][(r, 1, lag)]["rel_mse"], 6)
                                       for lag in lags
                                       if lag != 0 and (r, 1, lag) in by_t["lags"]}
                              for r in t_regions},
                "per_t_shape": {str(r): {f"{h}x{w}": round(by_t["shapes"][(r, 1, (h, w))]["rel_mse"], 6)
                                         for (h, w) in shape_keys
                                         if (r, 1, (h, w)) in by_t["shapes"]}
                                for r in t_regions},
                "recovery": {str(r["k"]): {"rel_mse": round(r["rel_mse"], 6),
                                           "ladder_only": round(r["ladder_only"], 6),
                                           "ladder_ratio_base": (round(r["ladder_only"] / bl, 4)
                                                                 if bl and r["ladder_only"] == r["ladder_only"] else None),
                                           "ladder_recovered": (round(1.0 - r["ladder_only"] / bl, 4)
                                                                if bl and r["k"] != 0 and r["ladder_only"] == r["ladder_only"] else None),
                                           "anchors_only": round(r["anchors_only"], 6),
                                           "n_pairs": r["n_pairs"]}
                             for r in rows},
                "affine_oob": affine_oob_json(affine) if affine is not None else None,
                "best_k1_ladder": best, "best_step": best_step,
                "gate_fired": gate_fired,
            })

    # ── Final (plan 6i) ────────────────────────────────────────────────
    ema_model.load_state_dict(ema)
    final = out_dir / f"corrector-{size}.safetensors"
    cfg_meta = {k: v for k, v in cfg.items()
                if k not in ("data", "resume", "out", "compile",
                             "channels_last", "metrics", "no_progress")}
    cfg_meta["lag_cond"] = ccfg.lag_cond
    cfg_meta["res_cond"] = ccfg.res_cond
    save_corrector(ema_model, final, ccfg,
                   extra_metadata={"config_snapshot": cfg_meta})
    if pbar is not None:
        pbar.n = step
        pbar.total = step
        pbar.refresh()
        pbar.close()
    wall = time.time() - t_start
    print("\n  ── Training complete ──")
    print(f"  Steps:        {step}/{cfg['max_steps']}")
    print(f"  Wall time:    {format_duration(wall)}  (train {format_duration(timer.phase_seconds('train'))}, "
          f"eval {format_duration(timer.phase_seconds('eval'))}, "
          f"hessian {format_duration(timer.phase_seconds('hessian'))})")
    print(f"  Throughput:   {timer.steps_per_sec():.2f} it/s  "
          f"(median step {timer.step_time() * 1000:.0f} ms)")
    if loss_ema is not None:
        print(f"  Loss:         final {loss_v:.5f}   min {min_loss:.5f}   "
              f"(ema {loss_ema:.5f})")
    if best is not None:
        print(f"  Best ladder K1: {best:.4f} @ step {best_step}")
    if last_eval_rows:
        for line2 in recovery_table_lines(last_eval_rows, last_affine):
            print(line2)
    print(f"  VRAM peak:    {torch.cuda.max_memory_allocated() / (1024 ** 3):.1f} GB")
    print(f"  Metrics:      {cfg['metrics']}")
    print(f"  Final checkpoint: {final}  (K_recommended=1)")
    metrics.close()


if __name__ == "__main__":
    main()
