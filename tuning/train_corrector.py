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
  the plan's gates (did-it-learn vs the probe's linear ceiling, per-lag
  coverage, K-robustness gap) — plan 6h;
- checkpointing: EMA .safetensors per eval + best-by-eval + full-state .pt
  resume with config-drift warning (plan 6i);
- live reporting: tqdm progress bar (EMA loss, lr, K, it/s, remaining),
  per-50-step durable log lines, per-eval timing reports (train/eval/hessian
  phases; the ETA projects eval + checkpoint overhead), VRAM peak, and a
  JSONL metrics file (``train_metrics.jsonl`` next to the checkpoints;
  ``--metrics`` relocates, ``--no-progress`` disables the bar).

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
                                augment_batch, collate_corrector)
from .utils import MetricsLog, TrainTimer, format_duration

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
        return state

    def update_hessian(self, loss: torch.Tensor) -> None:
        """GNB diagonal-Hessian estimate on a fp32 loss with a live graph.

        Cost: one extra forward + backward every ``hessian_every`` steps
        (amortized ~10–15%). Must be called before ``step``. Parameters not
        used in the graph (e.g. the prompt layers on an all-uncond 0-token
        batch) are skipped — their Hessian is undefined that step and their
        stored h stays stale until a batch that exercises them.
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
                self._state(p)["h"].copy_((u * gu_i).detach().clamp_min(0))
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
                    h.mul_(b2).add_(state["h"], alpha=1 - b2)
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
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--compile", type=int, default=None,
                        help="torch.compile the corrector")
    parser.add_argument("--channels-last", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
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
        "batch_size": ("batch_size", 16),
        "ema_decay": ("ema_decay", 0.9999),
        "loss": ("loss", "rel_mse"),
        "lag_weights": ("lag_weights", [1, 1, 1, 1, 1]),
        "normalization": ("normalization", "none"),
        "normalization_samples": ("normalization_samples", 128),
        "accumulate_big_batches": ("accumulate_big_batches", True),
        "scale_aug": ("scale_aug", True),
        "max_steps": ("max_steps", 60000),
        "eval_every": ("eval_every", 500),
        "seed": ("seed", 42),
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
    rc["enabled"] = bool(rc["enabled"])
    rc["stage1_fraction"] = max(0.0, min(float(rc["stage1_fraction"]), 1.0))
    stage1_areas = []
    for spec in rc.get("stage1_shapes") or ["64x64"]:
        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            w, h = int(spec[0]), int(spec[1])
        else:
            w, h = refiner_data.parse_resolution(str(spec))
        stage1_areas.append(w * h)
    rc["stage1_areas"] = sorted(set(stage1_areas))
    cfg["out"] = args.out or str(Path(__file__).resolve().parent.parent / "models")
    cfg["data"] = args.data
    cfg["resume"] = args.resume
    cfg["metrics"] = args.metrics or str(Path(cfg["out"]) / "train_metrics.jsonl")
    cfg["no_progress"] = bool(args.no_progress) if args.no_progress is not None else False
    return cfg


# ── Training step (plan 6f) ───────────────────────────────────────────


def compute_train_loss(model, batch, Ks: torch.Tensor, loss_fn, eps: float,
                       stop_grad: bool, multipass: bool,
                       k_max: int) -> torch.Tensor:
    """Masked-K on-policy deep-supervised loss (plan 6f, deep-dive §5.4/§6.1).

    Pass i supervises the model's own i-th pass output against v_true with
    weight w_i = i / Σ_{j=1}^{K_s} j per sample; pass inputs are detached
    between passes when ``stop_grad``.
    """
    x, v, vt = batch["x_t"], batch["v_ma"].clone(), batch["v_true"]
    prompt, pmask, t = batch["prompt"], batch["prompt_mask"], batch["t_frac"]
    total = torch.zeros((), device=x.device, dtype=torch.float32)
    n_active = torch.zeros((), device=x.device)
    for i in range(1, k_max + 1):
        mask = Ks >= i
        if not mask.any():
            continue
        dv = model(x[mask], v[mask], prompt[mask], t[mask], pmask[mask])
        v_new = v[mask] + dv
        per_sample = loss_fn(v_new, vt[mask], eps)
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


# ── Eval (plan 6h) ────────────────────────────────────────────────────


def eval_model(model, eval_loader, lags: List[int], ks: List[int],
               device, eps: float, show_progress: bool = False) -> Tuple[Dict, Dict]:
    """Per-lag, per-K rel-MSE (‖v̂−v_true‖₂/‖v_true‖₂) and cosine, plus the
    same metrics grouped per spatial shape ((k, (h, w))) for the per-resolution
    eval report (resolution-independence check)."""
    model.eval()
    acc = {(k, lag): [0.0, 0.0, 0] for k in ks for lag in lags}
    acc_shape: Dict[Tuple[int, Tuple[int, int]], List[float]] = {}
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
            for k in ks:
                v = v0
                for _ in range(k):
                    v = v + model(x, v, prompt, t, pmask)
                err = (v - vt).float().flatten(1).norm(dim=1)
                den = vt.float().flatten(1).norm(dim=1) + 1e-8
                rel = (err / den).tolist()
                cos = F.cosine_similarity(v.float().flatten(1),
                                          vt.float().flatten(1), dim=1).tolist()
                for j, lag in enumerate(lags_b):
                    if lag in lags:
                        a = acc[(k, lag)]
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
    model.train()
    return out, by_shape


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
            "resolution_curriculum", "accumulate_big_batches", "scale_aug"]
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
    print(f"  Loss:           {cfg['loss']}   batch={cfg['batch_size']}  "
          f"max_steps={cfg['max_steps']}")
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
    )
    if not len(ds):
        raise SystemExit("[train_corrector] no training pairs (all generations eval?)")
    print(f"  Training pairs: {len(ds)}   eval pairs: "
          f"{len(CorrectorDataset(data_dir, only_eval=True, normalization_samples=0, manifest=manifest))}")

    sampler = CorrectorBatchSampler(ds, batch_size=cfg["batch_size"], seed=cfg["seed"])
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_corrector,
                        num_workers=0)
    steps_per_epoch = max(len(sampler), 1)
    print(f"  Batches/epoch:  {steps_per_epoch}   "
          f"(~{cfg['max_steps'] / steps_per_epoch:.1f} epochs over {cfg['max_steps']} steps)")

    # Resolution curriculum (stage 1 = cheap stratum only, then full mix):
    # the stage-1 sampler filters the pair buckets by latent area, so its
    # batches are ~4× cheaper while the model builds scale-invariant features.
    curriculum = cfg["resolution_curriculum"]
    stage1_loader = None
    stage1_until = 0
    if curriculum.get("enabled"):
        s1_sampler = CorrectorBatchSampler(
            ds, batch_size=cfg["batch_size"], seed=cfg["seed"] + 7,
            include_areas=curriculum["stage1_areas"])
        if s1_sampler.buckets:
            stage1_loader = DataLoader(ds, batch_sampler=s1_sampler,
                                       collate_fn=collate_corrector, num_workers=0)
            stage1_until = int(cfg["max_steps"] * curriculum["stage1_fraction"])
            print(f"  Curriculum:     stage 1 (steps ≤ {stage1_until}) → "
                  f"areas {curriculum['stage1_areas']}; then full mix")
        else:
            print(f"  ⚠ resolution curriculum enabled but no pairs at "
                  f"{curriculum['stage1_areas']} — training full mix only")
    print(f"  Scale aug:      {cfg['scale_aug']} (0.75× round-trip)   "
          f"accumulate ÷4 buckets: {cfg['accumulate_big_batches']}")
    eval_loader = None
    try:
        eval_ds = CorrectorDataset(data_dir, only_eval=True, seed=cfg["seed"],
                                   normalization_samples=0, manifest=manifest)
        eval_sampler = CorrectorBatchSampler(eval_ds, batch_size=16, seed=cfg["seed"] + 1)
        eval_loader = DataLoader(eval_ds, batch_sampler=eval_sampler,
                                 collate_fn=collate_corrector, num_workers=0)
    except ValueError as e:
        print(f"  ⚠ no eval generations — eval loop and gates skipped ({e})")
    lags = sorted(set(lag for _, _, _, lag in ds.pairs))
    if 0 not in lags:
        lags = [0] + lags

    # ── Model ─────────────────────────────────────────────────────────
    try:
        ccfg = CorrectorConfig.for_size(size, depth=cfg["depth"])
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

    # ── Resume (plan 6i) ──────────────────────────────────────────────
    if cfg["resume"]:
        sd = torch.load(cfg["resume"], map_location=device, weights_only=False)
        model.load_state_dict(sd["model"])
        ema = {k: v.to(device) for k, v in sd["ema"].items()}
        opt.load_state_dict(sd["optimizer"])
        scaler.load_state_dict(sd["scaler"])
        if scheduler is not None and sd.get("scheduler"):
            scheduler.load_state_dict(sd["scheduler"])
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
    ema_model = CorrectorUNet2D(ccfg).to(device)
    if cfg["compile"]:
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile enabled — first steps include compile latency")

    loss_fn = LOSS_FNS[cfg["loss"]]
    eps = ds.rel_mse_eps
    k_max_final = cfg["refine_passes_max"] if cfg["multipass"] else 1
    ks_eval = [k for k in (1, 2, 3) if k <= k_max_final]
    ceiling = load_linear_ceiling(data_dir)
    if ceiling is not None:
        print(f"  Linear ceiling: {ceiling:.4f} (1−R², probe) — did-it-learn target "
              f"≤ {0.8 * ceiling:.4f}")
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
    pbar = None
    if tqdm is not None and not cfg["no_progress"]:
        pbar = tqdm(total=cfg["max_steps"], desc="train", unit="step",
                    dynamic_ncols=True, mininterval=0.5)

    def emit(line: str) -> None:
        """Print a log line above the progress bar (bar-safe)."""
        if pbar is not None:
            pbar.write(line)
        else:
            print(line)

    while step < cfg["max_steps"]:
        step += 1
        t_step0 = time.time()
        if stage1_until and step == stage1_until + 1:
            active_loader = loader
            iter_loader = iter(loader)
            emit(f"  [curriculum] stage 2: full resolution mix from step {step}")
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
            with torch.autocast("cuda", enabled=False):
                dv = model(batch["x_t"].float(), batch["v_ma"].float(),
                           batch["prompt"].float(), batch["t_frac"].float(),
                           batch["prompt_mask"])
                loss_h = per_sample_rel_mse(batch["v_ma"].float() + dv,
                                            batch["v_true"].float(), eps).mean()
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
                                          cfg["stop_grad"], cfg["multipass"], k_max_cur)
            micro_losses.append(loss.detach())
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
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
                    cfg["max_steps"] - step, cfg["eval_every"])),
                refresh=False)
            pbar.update(1)

        if step % 50 == 0 or step == 1:
            wall = time.time() - t_start
            line = (f"  [train] step {step:>6d}/{cfg['max_steps']}  "
                    f"loss={loss_v:.5f} (ema {loss_ema:.5f}, min {min_loss:.5f})  "
                    f"lr={opt.param_groups[0]['lr']:.2e}  "
                    f"it/s={timer.steps_per_sec():.2f}  "
                    f"elapsed={format_duration(wall)}  "
                    f"remaining~{format_duration(timer.remaining_seconds(cfg['max_steps'] - step, cfg['eval_every']))}")
            emit(line)
            metrics.write({
                "type": "step", "step": step,
                "epoch": step / steps_per_epoch,
                "loss": loss_v, "loss_ema": loss_ema, "min_loss": min_loss,
                "grad_norm": grad_norm_v, "lr": opt.param_groups[0]["lr"],
                "k_max": k_max_cur, "wall_s": wall,
                "vram_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
            })

        if step % cfg["eval_every"] == 0 and eval_loader is not None:
            t_e0 = time.time()
            results, by_shape = eval_model(model, eval_loader, lags, ks_eval, device,
                                           eps, show_progress=(pbar is not None))
            k1_pairs = [r["rel_mse"] for (kk, _), r in results.items() if kk == 1]
            k1 = sum(k1_pairs) / max(len(k1_pairs), 1) if k1_pairs else None
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
            shape_keys = sorted({hw for (_, hw) in by_shape}, key=lambda hw: hw[0] * hw[1])
            for hw in shape_keys:
                parts = []
                for k in ks_eval:
                    r = by_shape.get((k, hw))
                    parts.append(f"K{k}={r['rel_mse']:.4f}" if r is not None else f"K{k}=  -  ")
                emit(f"      {hw[0]}x{hw[1]}: " + "  ".join(parts))
            if ceiling is not None and not gate_fired and step >= gate_deadline:
                if k1 is None or k1 > 0.8 * ceiling:
                    emit("  ✗ BELOW LINEAR CEILING — did-it-learn gate fired; "
                         "stopping per plan 6h")
                    step = cfg["max_steps"]
                    gate_fired = True
            # K-robustness + per-lag coverage (report-only warnings)
            if len(ks_eval) > 1:
                r1 = sum(v["rel_mse"] for (kk, _), v in results.items() if kk == 1)
                r3 = sum(v["rel_mse"] for (kk, _), v in results.items() if kk == max(ks_eval))
                nk = max(1, sum(1 for (kk, _) in results if kk == 1))
                if r3 > 0 and (r1 - r3) / r3 > 0.2:
                    emit("  ⚠ K-robustness gap large (K=1 ≫ K=K_max) — consider "
                         "strengthening the K curriculum (plan 6h)")
            per_lag_k1 = [results.get((1, lag), {}).get("rel_mse", float("nan"))
                          for lag in lags]
            finite = [x for x in per_lag_k1 if x == x]
            if len(finite) >= 3 and finite[0] > 0:
                growth = (finite[-1] - finite[0]) / finite[0]
                if growth > 1.5:
                    emit(f"  ⚠ Per-lag rel-MSE grows {growth:.2f}× from lag "
                         f"{lags[0]} to lag {lags[-1]} — check the per-lag "
                         "coverage gate (plan 6h); extend record_lags if "
                         "superlinear")

            # Checkpointing (plan 6i)
            ema_model.load_state_dict(ema)
            cfg_meta = {k: v for k, v in cfg.items()
                        if k not in ("data", "resume", "out", "compile",
                                     "channels_last", "metrics", "no_progress")}
            if k1 is not None and (best is None or k1 < best):
                best = k1
                best_step = step
                save_corrector(ema_model, out_dir / f"corrector-{size}-best.safetensors",
                               ccfg, extra_metadata={"config_snapshot": cfg_meta,
                                                     "best_rel_mse_k1": k1})
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
                 + (f" (best K1={best:.4f} @ {best_step})" if best is not None else ""))
            timer.add("eval", time.time() - t_e0)
            for line2 in timer.summary_lines(time.time() - t_start, step,
                                             cfg["max_steps"], cfg["eval_every"]):
                emit(line2)
            metrics.write({
                "type": "eval", "step": step, "k1": k1,
                "k_avg": {str(k): round(sum(r["rel_mse"] for (kk, _), r in results.items()
                                            if kk == k) / max(sum(1 for (kk, _) in results if kk == k), 1), 6)
                          for k in ks_eval},
                "per_lag": {str(lag): results.get((1, lag), {}).get("rel_mse")
                            for lag in lags},
                "per_shape": {f"{h}x{w}": {str(k): round(by_shape.get((k, (h, w)), {}).get("rel_mse", float("nan")), 6)
                                           for k in ks_eval}
                              for (h, w) in shape_keys},
                "best_k1": best, "best_step": best_step, "gate_fired": gate_fired,
            })

    # ── Final (plan 6i) ────────────────────────────────────────────────
    ema_model.load_state_dict(ema)
    final = out_dir / f"corrector-{size}.safetensors"
    cfg_meta = {k: v for k, v in cfg.items()
                if k not in ("data", "resume", "out", "compile",
                             "channels_last", "metrics", "no_progress")}
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
        print(f"  Best K1:      {best:.4f} @ step {best_step}")
    print(f"  VRAM peak:    {torch.cuda.max_memory_allocated() / (1024 ** 3):.1f} GB")
    print(f"  Metrics:      {cfg['metrics']}")
    print(f"  Final checkpoint: {final}  (K_recommended=1)")
    metrics.close()


if __name__ == "__main__":
    main()
