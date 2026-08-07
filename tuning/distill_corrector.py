#!/usr/bin/env python3
"""K=3 → K=1 (and cross-size) corrector distillation (plan Task 6l, deep-dive §4).

The teacher is the EMA-frozen K-pass corrector; the student (initialized from
the teacher where shapes match) learns to reproduce the teacher's K-pass
result in one pass:

- ``--mode static``: progressive distillation (arXiv:2202.00512) — targets are
  the teacher's K-pass rollouts from the recorded ``v_MA`` inputs;
- ``--mode gkd`` (default): on-policy distillation (arXiv:2306.13649) — the
  student first generates its own state from ``v_MA`` (detached), the teacher
  is evaluated on that state, and the student matches it.

Emits ``corrector-{size}-turbo.safetensors`` (K_recommended=1). The cross-size
ladder (50M → 20M → 5M) uses the same mechanism.

Usage:
    python -m tuning.distill_corrector --teacher models/corrector-20m.safetensors \
        --data outputs/<ts>/refiner_data --student-size 5m --mode gkd
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config_types import TuningConfig
from .corrector import CorrectorConfig, CorrectorUNet2D, load_corrector, save_corrector
from .corrector_dataset import (CorrectorBatchSampler, CorrectorDataset,
                                augment_batch, collate_corrector)
from .train_corrector import ema_update, init_ema

from torch.utils.data import DataLoader


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill a K-pass corrector into K=1")
    parser.add_argument("--teacher", required=True, help="Teacher .safetensors")
    parser.add_argument("--data", required=True, help="refiner_data dir (i.i.d. pairs)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--student-size", type=str, default="5m",
                        help="Student size target: '5M', '20m', '1.5B', 'tiny'")
    parser.add_argument("--student-depth", type=str, default="auto",
                        help="Student DiT blocks: 'auto' or 1-8")
    parser.add_argument("--mode", type=str, default="gkd", choices=["static", "gkd"])
    parser.add_argument("--on-policy", type=int, default=1,
                        help="GKD: student's own state as the teacher input")
    parser.add_argument("--teacher-passes", type=int, default=3,
                        help="Teacher K to distill down to 1")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--compile", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Pull defaults from the refiner_training config section for shared knobs.
    if args.config is None:
        args.config = str(Path(__file__).parent / "config.json")
    tcfg = TuningConfig.load(args.config)
    rt = dict(tcfg.refiner_training or {})
    batch_size = args.batch_size if args.batch_size is not None else rt.get("batch_size", 16)
    lr = args.lr if args.lr is not None else rt.get("lr", 4e-4)
    max_steps = args.max_steps if args.max_steps is not None else rt.get("max_steps", 60000) // 3
    eval_every = args.eval_every if args.eval_every is not None else rt.get("eval_every", 500)
    ema_decay = args.ema_decay if args.ema_decay is not None else rt.get("ema_decay", 0.9999)
    seed = args.seed if args.seed is not None else rt.get("seed", 42)
    out = args.out or str(Path(__file__).resolve().parent.parent / "models")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    teacher = load_corrector(args.teacher).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print("=" * 60)
    print("  Corrector Distillation (K={} → 1, mode={})".format(
        args.teacher_passes, args.mode))
    print(f"  Teacher:        {args.teacher} ({teacher.cfg.size})")
    print(f"  Student size:   {args.student_size}   data: {args.data}")
    print(f"  Batch:          {batch_size}   lr: {lr}   steps: {max_steps}")

    depth = str(args.student_depth).strip().lower()
    if depth != "auto" and not (depth.isdigit() and 1 <= int(depth) <= 8):
        raise SystemExit(f"[distill_corrector] invalid --student-depth "
                         f"{args.student_depth!r} (auto or 1-8)")
    try:
        ccfg = CorrectorConfig.for_size(args.student_size, depth=depth)
    except ValueError as e:
        raise SystemExit(f"[distill_corrector] {e}")
    ccfg.normalization = teacher.cfg.normalization
    ccfg.normalization_stats = teacher.cfg.normalization_stats
    student = CorrectorUNet2D(ccfg)
    # Warm start from the teacher where shapes match (deep-dive §4.1).
    tsd = teacher.state_dict()
    ssd = student.state_dict()
    loaded = 0
    for k in ssd:
        if k in tsd and tsd[k].shape == ssd[k].shape:
            ssd[k] = tsd[k].clone()
            loaded += 1
    student.load_state_dict(ssd)
    print(f"  Warm-start keys: {loaded}/{len(ssd)} (same-size: all; cross-size: subset)")
    student = student.to(device)

    ds = CorrectorDataset(Path(args.data), seed=seed, synthesize_anchors=False)
    sampler = CorrectorBatchSampler(ds, batch_size=batch_size, seed=seed)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_corrector,
                        num_workers=0)
    ema = init_ema(student)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    if args.compile:
        student = torch.compile(student, mode="reduce-overhead")
    aug_rng = random.Random(seed + 7)

    def teacher_rollout(x, v, prompt, t, pmask):
        v_t = v
        for _ in range(args.teacher_passes):
            v_t = v_t + teacher(x, v_t, prompt, t, pmask)
        return v_t

    it = iter(loader)
    t0 = time.time()
    for step in range(1, max_steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}
        batch = augment_batch(batch, aug_rng)
        x, v0 = batch["x_t"], batch["v_ma"]
        prompt, pmask, t = batch["prompt"], batch["prompt_mask"], batch["t_frac"]

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            v_s = v0
            if args.mode == "gkd" and args.on_policy:
                with torch.no_grad():
                    v_s = (v_s + student(x, v_s, prompt, t, pmask)).detach()
            with torch.no_grad():
                target = teacher_rollout(x, v_s, prompt, t, pmask)
            pred = v_s + student(x, v_s, prompt, t, pmask)
            loss = F.mse_loss(pred.float(), target.float())
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        ema_update(student, ema, step, decay=ema_decay)

        if step % max(eval_every, 1) == 0:
            ema_model = CorrectorUNet2D(ccfg).to(device)
            ema_model.load_state_dict(ema)
            save_corrector(ema_model, Path(out) / f"corrector-{args.student_size}-turbo-{step}.safetensors",
                           ccfg)
            print(f"  [distill] step {step}/{max_steps}  loss={loss.item():.5f}  "
                  f"elapsed={time.time()-t0:.0f}s  saved step checkpoint")

    ema_model = CorrectorUNet2D(ccfg)
    ema_model.load_state_dict(ema)
    final = Path(out) / f"corrector-{args.student_size}-turbo.safetensors"
    save_corrector(ema_model, final, ccfg,
                   extra_metadata={"config_snapshot": {
                       "teacher": args.teacher, "mode": args.mode,
                       "teacher_passes": args.teacher_passes,
                       "student_size": args.student_size}})
    print(f"\n  Turbo checkpoint: {final}  (K_recommended=1)")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
