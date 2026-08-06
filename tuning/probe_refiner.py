#!/usr/bin/env python3
"""Refiner probe pipeline (plan Task 3).

Records 3–5 generations with refiner latent capture (or analyzes an existing
``refiner_data`` dir) and measures:

- 3a codec benchmarks (raw bf16 / blosc2 bitshuffle+zstd / zfpy reversible /
  fpzip / int8 per-channel / zfpy fixed-accuracy) per tensor type, plus the
  key **t-gap cancellation** validation (|Δ_MA| vs |v_true(t)−v_true(t−1)|);
- 3b learnability stats (Δ_MA SVD rank, per-channel affine share, v_true
  step correlation, predictability ceiling, staleness curve per step region,
  Δ_MA distribution, K-gate pass-improvement stats);
- 3c decision gate (rank ≤ 8, ceiling ≤ 60%, staleness per-region monotone);
- 3d Day-1 experiment: small MLP on pooled features, then a tiny 2D UNet —
  verdict: the full corrector is worth building iff the small model beats the
  linear ceiling by ≥25% rel-MSE; plus the **lag-readability** check (Δ̂ vs lag
  smooth/monotone).

Writes ``refiner_probe_report.json`` next to the data dir (consumed by
``train_corrector``'s did-it-learn gate) and prints a console table.

Usage:
    python -m tuning.probe_refiner --comfy-dir /path/to/ComfyUI   # record + analyze
    python -m tuning.probe_refiner --data outputs/<ts>/refiner_data  # analyze only
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config_types import TuningConfig
from .corrector import CorrectorConfig, CorrectorUNet2D
from . import refiner_data
from .refiner_data import (CODEC_BLOSC2, CODEC_RAW, CODEC_ZFPY, _compress,
                           _decompress)
from .corrector_dataset import CorrectorDataset, CorrectorBatchSampler, collate_corrector
from .prompt_loader import load_prompt_config, select_prompts, GenerationPromptSampler, resolve_generation
from .utils import load_models, sample

from torch.utils.data import DataLoader

try:
    import zfpy
    _HAS_ZFPY = True
except ImportError:
    _HAS_ZFPY = False

try:
    import fpzip
    _HAS_FPZIP = True
except ImportError:
    _HAS_FPZIP = False


# ═══════════════════════════════════════════════════════════════════════
#  Recording (3–5 generations, plan Task 3)
# ═══════════════════════════════════════════════════════════════════════


def record_probe_generations(tcfg: TuningConfig, num_gens: int, seeds: List[int],
                             lags: List[int], out_dir: Path) -> Path:
    """Run *num_gens* generations with refiner capture into out_dir/refiner_data."""
    from .calibrate import patch_for_refiner
    from .prompt_loader import GenerationPromptSampler  # noqa: F811

    refiner_dir = out_dir / "refiner_data"
    refiner_dir.mkdir(parents=True, exist_ok=True)
    manifest = refiner_data.init_manifest({
        "record_lags": lags, "implicit_d0": True, "keep_all": True,
        "top_n": -1, "record_slots": "both", "mode": "only",
    })
    refiner_data.save_manifest(refiner_dir, manifest)

    unet, clip, vae = load_models(
        tcfg.comfy_dir, tcfg.model_name, tcfg.clip_name, tcfg.clip_type, tcfg.vae_name,
    )
    prompt_config = load_prompt_config(str(Path(__file__).parent / tcfg.calibration["prompts_file"]))
    entries = select_prompts(prompt_config, method=tcfg.calibration.get("prompt_selection", "from_top"),
                             count=num_gens, tag_filter=tcfg.calibration.get("prompt_tag_filter"))
    prompt_sampler = GenerationPromptSampler(prompt_config, None, {})
    steps = int(tcfg.sampling.get("default_steps", 30))
    width = int(tcfg.sampling.get("width", 512))
    height = int(tcfg.sampling.get("height", 512))
    sampler_name = tcfg.sampling.get("sampler", "er_sde")
    scheduler = tcfg.sampling.get("scheduler", "normal")
    cfg_val = float(tcfg.sampling.get("cfg", 5.0))

    rc = {"record_lags": lags, "record_slots": "both", "dtype": "bfloat16"}
    for i, pdata in enumerate(entries):
        seed = seeds[i % len(seeds)]
        full_prompt, neg_prompt, _ = resolve_generation(
            prompt_sampler, pdata.entry, i, seed, steps, width, height,
        )
        dm, original_fwd = patch_for_refiner(
            unet, steps, prompt_id=i, seed=seed, track_per_block=False,
            refiner_dir=str(refiner_dir), refiner_cfg=rc,
        )
        try:
            sample(unet, clip, vae, full_prompt, seed=seed, steps=steps, cfg=cfg_val,
                   sampler_name=sampler_name, scheduler=scheduler,
                   width=width, height=height, negative=neg_prompt)
        finally:
            from .calibrate import restore_model
            restore_model(dm, original_fwd, unet)
        refiner_data.finalize_refiner_generation(
            dm, refiner_dir, manifest,
            run_meta={"name": f"gen_{i:04d}_p{i:02d}_s{seed}",
                      "prompt_id": i, "seed": seed,
                      "sampler": sampler_name, "scheduler": scheduler, "cfg": cfg_val,
                      "width": width, "height": height, "steps": steps, "fps": None,
                      "prompt": full_prompt, "negative": neg_prompt, "artists": []},
            refiner_cfg=rc,
            eval_prompt_ids={len(entries) - 1},  # last gen = deterministic holdout
        )
        print(f"  [probe] recorded generation {i + 1}/{len(entries)}")
    refiner_data.save_manifest(refiner_dir, manifest)
    return refiner_dir


# ═══════════════════════════════════════════════════════════════════════
#  Tensor sampling helpers
# ═══════════════════════════════════════════════════════════════════════

def _collect_tensors(data_dir: Path, max_per_type: int = 12) -> dict:
    """Collect up to max_per_type tensors of each kind from the recorded gens."""
    out = {"x_t": [], "v_ma": [], "v_true": [], "delta": []}
    for entry in refiner_data.iter_generations(data_dir):
        gen = refiner_data.load_generation(data_dir / entry["bin"])
        for slot in gen.recorded_slots:
            for t in range(min(len(gen.v_true[slot]), 8)):
                if len(out["x_t"]) >= max_per_type:
                    return out
                x = gen.x[slot][t]
                vt = gen.v_true[slot][t]
                out["x_t"].append(x)
                out["v_true"].append(vt)
                step_ma = gen.v_ma[slot][t]
                for lag in sorted(step_ma):
                    if len(out["v_ma"]) >= max_per_type:
                        break
                    out["v_ma"].append(step_ma[lag])
                    out["delta"].append(vt - step_ma[lag])
    return out


# ═══════════════════════════════════════════════════════════════════════
#  3a — codec benchmarks
# ═══════════════════════════════════════════════════════════════════════

def _codec_bits_per_element(name: str, codec_id: int, tensors: List[torch.Tensor],
                            lossy=False) -> dict:
    n_bytes, n_el = 0, 0
    t_enc = t_dec = 0.0
    for t in tensors:
        t0 = time.time()
        blob = _compress(t, codec_id, _DTYPE_ID)
        t_enc += time.time() - t0
        t0 = time.time()
        back = _decompress(blob, codec_id, _DTYPE_ID, tuple(t.shape))
        t_dec += time.time() - t0
        if not lossy:
            assert torch.equal(back, t), f"{name} round-trip mismatch"
        n_bytes += len(blob)
        n_el += t.numel()
    mbps = (n_bytes / 1e6) / max(t_enc + t_dec, 1e-9)
    return {"codec": name, "bits_per_element": round(8 * n_bytes / max(n_el, 1), 3),
            "throughput_mbps": round(mbps, 1)}


_DTYPE_ID = refiner_data._DTYPE_IDS["bfloat16"]


def benchmark_codecs(tensors_by_type: dict) -> dict:
    """3a: bits/element + throughput per tensor type per codec; lossy metrics."""
    results = {}
    codecs = [("raw_bf16", CODEC_RAW, False),
              ("blosc2_bitshuffle_zstd9", CODEC_BLOSC2, False)]
    if _HAS_ZFPY:
        codecs.append(("zfpy_reversible", CODEC_ZFPY, False))
    for kind, tensors in tensors_by_type.items():
        if not tensors:
            continue
        rows = [_codec_bits_per_element(name, cid, tensors) for name, cid, _ in codecs]
        # fpzip (optional): per-tensor lossless 3D calls
        if _HAS_FPZIP:
            n_bytes, n_el, t0 = 0, 0, time.time()
            ok = True
            for t in tensors:
                arr = t.float().numpy()  # bf16 → fp32 upcast (like zfpy)
                blob = fpzip.compress(arr, precision=0)
                back = fpzip.decompress(blob)
                n_bytes += len(blob)
                n_el += t.numel()
                ok = ok and np.array_equal(arr, back)
            rows.append({"codec": "fpzip_lossless",
                         "bits_per_element": round(8 * n_bytes / max(n_el, 1), 3),
                         "throughput_mbps": round((n_bytes / 1e6) / max(time.time() - t0, 1e-9), 1),
                         "roundtrip_ok": ok})
        # int8 per-channel (lossy)
        n_bytes, n_el, se, sm, cs = 0, 0, 0.0, 0.0, 0.0
        for t in tensors:
            f = t.float()
            scale = f.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            q = (f / scale * 127).round().clamp(-127, 127).to(torch.int8)
            back = q.float() / 127 * scale
            n_bytes += q.numel() + f.shape[0] * 4
            n_el += f.numel()
            se += (f - back).square().mean().item() / f.square().mean().clamp_min(1e-8).item()
            sm += (f - back).abs().max().item()
            cs += F.cosine_similarity(f.flatten(1), back.flatten(1), dim=1).mean().item()
        n = max(len(tensors), 1)
        rows.append({"codec": "int8_perchannel",
                     "bits_per_element": round(8 * n_bytes / max(n_el, 1), 3),
                     "rel_mse": round(se / n, 5), "max_err": round(sm / n, 5),
                     "cosine": round(cs / n, 5)})
        # zfpy fixed-accuracy (lossy)
        if _HAS_ZFPY:
            for tol in (1e-3, 1e-2):
                n_bytes, n_el, se, sm, cs, t0 = 0, 0, 0.0, 0.0, 0.0, time.time()
                for t in tensors:
                    arr = t.float().numpy()
                    blob = zfpy.compress_numpy(arr, tolerance=tol)
                    back = torch.from_numpy(zfpy.decompress_numpy(blob))
                    n_bytes += len(blob)
                    n_el += arr.size
                    se += (arr - back.numpy()).mean() ** 2 / max(arr.var(), 1e-8)
                    sm += np.abs(arr - back.numpy()).max()
                    cs += F.cosine_similarity(torch.from_numpy(arr).flatten(1),
                                              back.flatten(1), dim=1).mean().item()
                rows.append({"codec": f"zfpy_fixed_{tol:.0e}",
                             "bits_per_element": round(8 * n_bytes / max(n_el, 1), 3),
                             "rel_mse": round(se / max(len(tensors), 1), 5),
                             "max_err": round(sm / max(len(tensors), 1), 5),
                             "cosine": round(cs / max(len(tensors), 1), 5)})
        results[kind] = rows
    return results


# ═══════════════════════════════════════════════════════════════════════
#  3b — learnability stats
# ═══════════════════════════════════════════════════════════════════════

def _flatten_maps(tensors: List[torch.Tensor]) -> Optional[np.ndarray]:
    if not tensors:
        return None
    return torch.stack([t.float() for t in tensors]).flatten(0, 1).numpy()


def svd_rank_95(delta_maps: List[torch.Tensor]) -> dict:
    """SVD spectrum of Δ_MA (H·W, 16); rank for 95% variance (plan 3b)."""
    m = torch.stack([d.float() for d in delta_maps])          # (N, C, H, W)
    m = m.permute(0, 2, 3, 1).reshape(-1, m.shape[1])          # (N·H·W, 16)
    m = m - m.mean(dim=0, keepdim=True)
    try:
        u, s, v = torch.svd(m)
    except Exception:
        return {"rank_95": None, "singular_values": None}
    var = (s * s).cumsum(0)
    var = var / var[-1].clamp_min(1e-12)
    rank95 = int((var <= 0.95).sum().item()) + 1
    return {"rank_95": rank95,
            "singular_values": [round(float(x), 4) for x in s[:16].tolist()]}


def per_channel_affine_ceiling(v_ma_maps: List[torch.Tensor],
                               v_true_maps: List[torch.Tensor]) -> float:
    """Best per-channel affine v̂ = a_c·v_MA_c + b_c; 1 − R² (plan 3b)."""
    x = torch.stack([v.float().flatten(1) for v in v_ma_maps]).reshape(-1, 16)
    y = torch.stack([v.float().flatten(1) for v in v_true_maps]).reshape(-1, 16)
    x_aug = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)   # (N, 17)
    coefs, _, _, _ = torch.linalg.lstsq(x_aug, y)              # (17, 16)
    pred = x_aug @ coefs
    ss_res = ((y - pred) ** 2).sum(dim=0)
    ss_tot = ((y - y.mean(dim=0)) ** 2).sum(dim=0).clamp_min(1e-12)
    r2 = (1 - ss_res / ss_tot).mean().item()
    return max(0.0, 1.0 - r2)


def pooled_feature_ceiling(x_t_maps, v_ma_maps, v_true_maps) -> float:
    """Linear predictor from pooled features (per-channel mean/std + t) → 1−R²."""
    feats, y = [], []
    for xt, vm, vt in zip(x_t_maps, v_ma_maps, v_true_maps):
        f = torch.cat([xt.float().mean(dim=(1, 2)), xt.float().std(dim=(1, 2)),
                       vm.float().mean(dim=(1, 2)), vm.float().std(dim=(1, 2))])
        feats.append(f)
        y.append(vt.float().mean(dim=(1, 2)))
    X = torch.stack(feats)                                      # (N, 64)
    Y = torch.stack(y)                                          # (N, 16)
    X = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)
    coefs, _, _, _ = torch.linalg.lstsq(X, Y)
    pred = X @ coefs
    ss_res = ((Y - pred) ** 2).sum()
    ss_tot = ((Y - Y.mean(dim=0)) ** 2).sum().clamp_min(1e-12)
    return float(max(0.0, 1.0 - ss_res / ss_tot).item())


def step_correlations(v_true_by_slot: dict) -> dict:
    """Step-to-step correlation of v_true per region (early/mid/late)."""
    out = {}
    all_corrs = {"early": [], "mid": [], "late": []}
    for slot, vts in v_true_by_slot.items():
        n = len(vts)
        if n < 3:
            continue
        thirds = [("early", slice(0, n // 3)), ("mid", slice(n // 3, 2 * n // 3)),
                  ("late", slice(2 * n // 3, n))]
        for name, sl in thirds:
            for t in range(sl.start + 1, sl.stop):
                a = vts[t].float().flatten()
                b = vts[t - 1].float().flatten()
                all_corrs[name].append(F.cosine_similarity(a[None], b[None]).item())
    for name, corrs in all_corrs.items():
        out[name] = round(float(np.mean(corrs)), 4) if corrs else None
    return out


def staleness_curve(delta_by_lag: Dict[int, List[Tuple[int, torch.Tensor]]],
                    num_regions: int = 3) -> dict:
    """mean|Δ_MA| vs lag per step region (early/mid/late); monotonicity per region."""
    curve = {r: {} for r in range(num_regions)}
    for lag, items in delta_by_lag.items():
        for region, d in items:
            curve[region].setdefault(lag, []).append(d.abs().mean().item())
    out = {}
    monotone = True
    for r in range(num_regions):
        lags = sorted(curve[r])
        means = {lag: float(np.mean(curve[r][lag])) for lag in lags}
        out[f"region_{r}"] = means
        vals = [means[lag] for lag in lags]
        if len(vals) > 1:
            diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
            if any(d < -1e-9 for d in diffs):
                monotone = False
    out["per_region_monotone"] = monotone
    return out


def delta_distribution(delta_maps: List[torch.Tensor]) -> dict:
    d = torch.stack([x.float() for x in delta_maps])
    flat = d.flatten()
    return {
        "mean": round(float(flat.mean()), 6),
        "std": round(float(flat.std()), 6),
        "sparsity": round(float((flat.abs() < 1e-4).float().mean()), 4),
        "per_channel_std": [round(float(s), 5) for s in d.std(dim=(0, 2, 3)).tolist()],
    }


# ═══════════════════════════════════════════════════════════════════════
#  3d — Day-1 experiment
# ═══════════════════════════════════════════════════════════════════════

def day1_mlp(x_t_maps, v_ma_maps, v_true_maps, device, steps=400, lr=1e-3) -> float:
    """MLP on pooled features predicting per-channel mean Δ_MA (3d stage 1)."""
    feats, targets = [], []
    for xt, vm, vt in zip(x_t_maps, v_ma_maps, v_true_maps):
        feats.append(torch.cat([xt.float().mean(dim=(1, 2)), xt.float().std(dim=(1, 2)),
                                vm.float().mean(dim=(1, 2)), vm.float().std(dim=(1, 2))]))
        targets.append((vt - vm).float().mean(dim=(1, 2)))
    X = torch.stack(feats).to(device)
    Y = torch.stack(targets).to(device)
    n = X.shape[0]
    n_tr = max(1, int(n * 0.8))
    mlp = torch.nn.Sequential(
        torch.nn.Linear(64, 128), torch.nn.SiLU(),
        torch.nn.Linear(128, 16)).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    idx = torch.randperm(n, device=device)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(mlp(X[idx[:n_tr]]), Y[idx[:n_tr]])
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = mlp(X[idx[n_tr:]])
        err = (pred - Y[idx[n_tr:]]).square().mean().sqrt().item()
    return err


def day1_unet(data_dir: Path, device, max_steps: int = 600, batch_size: int = 8,
              seed: int = 42) -> dict:
    """Tiny 2D UNet on the real pairs (3d stage 2); eval vs the linear ceiling."""
    torch.manual_seed(seed)
    train_ds = CorrectorDataset(data_dir, seed=seed)
    train_loader = DataLoader(train_ds, batch_sampler=CorrectorBatchSampler(
        train_ds, batch_size=batch_size, seed=seed), collate_fn=collate_corrector,
        num_workers=0)
    try:
        eval_ds = CorrectorDataset(data_dir, only_eval=True, seed=seed)
    except ValueError:
        eval_ds = None

    ccfg = CorrectorConfig.for_size("tiny")
    model = CorrectorUNet2D(ccfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    it = iter(train_loader)
    for step in range(1, max_steps + 1):
        try:
            b = next(it)
        except StopIteration:
            it = iter(train_loader)
            b = next(it)
        b = {k: v.to(device) for k, v in b.items()}
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            dv = model(b["x_t"], b["v_ma"], b["prompt"], b["t_frac"], b["prompt_mask"])
            loss = ((b["v_ma"] + dv - b["v_true"]).float().square().mean(
                dim=(1, 2, 3)) / (b["v_true"].float().square().mean(dim=(1, 2, 3)) + 1e-8)).mean()
        loss.backward()
        opt.step()
    # eval
    def eval_pairs(ds):
        if ds is None:
            return None
        errs = []
        loader = DataLoader(ds, batch_sampler=CorrectorBatchSampler(ds, 16, seed + 1),
                            collate_fn=collate_corrector, num_workers=0)
        model.eval()
        with torch.no_grad():
            for b in loader:
                b = {k: v.to(device) for k, v in b.items()}
                v = b["v_ma"]
                for _ in range(1):
                    v = v + model(b["x_t"], v, b["prompt"], b["t_frac"], b["prompt_mask"])
                err = (v - b["v_true"]).float().flatten(1).norm(dim=1)
                den = b["v_true"].float().flatten(1).norm(dim=1) + 1e-8
                errs.extend((err / den).tolist())
        model.train()
        return float(np.mean(errs))

    # Lag-readability (plan 3d): Δ̂ vs lag for fixed (x_t, t) — smooth/monotone?
    lag_stats = _lag_readability(model, device, data_dir)

    return {"day1_unet_rel_mse": eval_pairs(eval_ds),
            "lag_readability": lag_stats}


def _lag_readability(model, device, data_dir) -> dict:
    """For fixed (x_t, t) steps: Δ̂(lag) must vary smoothly (ideally monotonically).

    Uses the trained Day-1 UNet on steps where the full ladder exists: the
    corrections for each lag share (x_t, prompt, t) and differ only in v_MA_d,
    so monotone ‖Δ̂_d‖ with d means staleness is decodable from v_MA's content
    (d is never a model input).
    """
    from scipy.stats import spearmanr
    mags_by_step, smoothness = [], []
    n = 0
    for entry in refiner_data.iter_generations(data_dir):
        gen = refiner_data.load_generation(data_dir / entry["bin"])
        for slot in gen.recorded_slots:
            for t in range(len(gen.v_true[slot])):
                step_ma = gen.v_ma[slot][t]
                lags = sorted(step_ma)
                if len(lags) < 3:
                    continue
                x = gen.x[slot][t].unsqueeze(0).to(device)
                prompt = gen.prompt.get(slot)
                if prompt is None or not prompt.numel():
                    prompt = torch.zeros(1, 0, 1024, device=device)
                else:
                    prompt = prompt.unsqueeze(0).to(device)
                t_frac = torch.tensor([gen.step_fractions.get(slot, [0.0] * len(gen.v_true[slot]))[t]],
                                      device=device)
                mags = []
                with torch.no_grad():
                    for lag in lags:
                        v = step_ma[lag].unsqueeze(0).to(device)
                        dv = model(x, v, prompt, t_frac)
                        mags.append(dv.abs().mean().item())
                mags_by_step.append(mags)
                diffs = [abs(mags[i + 1] - mags[i]) for i in range(len(mags) - 1)]
                scale = max(np.mean(mags), 1e-9)
                smoothness.append(np.mean(diffs) / scale)
                n += 1
                if n >= 24:
                    break
            if n >= 24:
                break
        if n >= 24:
            break
    if not mags_by_step:
        return {"note": "no steps with ≥3 lags found"}
    corrs = []
    for mags in mags_by_step:
        d = list(range(len(mags)))
        rho, _ = spearmanr(d, mags)
        if rho == rho:
            corrs.append(rho)
    return {
        "smoothness": round(float(np.mean(smoothness)), 4),
        "spearman_lag_vs_mag": round(float(np.mean(corrs)), 4) if corrs else None,
        "n_steps": n,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Report + gates
# ═══════════════════════════════════════════════════════════════════════

def main(argv=None):
    parser = argparse.ArgumentParser(description="Refiner probe (plan Task 3)")
    parser.add_argument("--comfy-dir", default=None, help="ComfyUI dir (recording)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None, help="Existing refiner_data dir")
    parser.add_argument("--prompts", type=int, default=3, help="Gens to record (3-5)")
    parser.add_argument("--seeds", default="34635345,53453634", help="Comma-separated")
    parser.add_argument("--record-lags", default="1,2,4,8,16")
    parser.add_argument("--day1-steps", type=int, default=600)
    parser.add_argument("--day1-batch", type=int, default=8)
    args = parser.parse_args(argv)

    if args.config is None:
        args.config = str(Path(__file__).parent / "config.json")
    tcfg = TuningConfig.load(args.config)

    if args.data:
        data_dir = Path(args.data)
    else:
        if not args.comfy_dir:
            raise SystemExit("provide --data (existing refiner_data) or --comfy-dir (record)")
        tcfg.comfy_dir = args.comfy_dir
        out_dir = Path(tcfg.output_dir) / time.strftime("%Y%m%d-%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        seeds = [int(s) for s in args.seeds.split(",")]
        lags = refiner_data.parse_lags(args.record_lags)
        data_dir = record_probe_generations(tcfg, max(2, min(args.prompts, 5)),
                                            seeds, lags, out_dir)
    report = {"data": str(data_dir), "lags": refiner_data.parse_lags(args.record_lags)}

    print("=" * 60)
    print("  Refiner Probe")
    print("=" * 60)
    print(f"  Data: {data_dir}")

    # ── Collect tensors ────────────────────────────────────────────────
    tensors = _collect_tensors(data_dir)
    n_gens = len(refiner_data.iter_generations(data_dir))
    print(f"  Generations: {n_gens}  tensors: "
          f"{ {k: len(v) for k, v in tensors.items()} }")
    if not tensors["delta"]:
        raise SystemExit("[probe] no recorded pairs found — nothing to analyze")

    # ── 3a codecs + t-gap cancellation ─────────────────────────────────
    codec_results = benchmark_codecs(tensors)
    report["codec"] = codec_results
    # t-gap cancellation: |Δ_MA| vs |v_true(t) − v_true(t−1)|
    gap_true, gap_delta = [], []
    for entry in refiner_data.iter_generations(data_dir):
        gen = refiner_data.load_generation(data_dir / entry["bin"])
        for slot in gen.recorded_slots:
            vts = gen.v_true[slot]
            for t in range(1, len(vts)):
                gap_true.append((vts[t].float() - vts[t - 1].float()).abs().mean().item())
                step_ma = gen.v_ma[slot][t]
                for lag in step_ma:
                    gap_delta.append((vts[t].float() - step_ma[lag].float()).abs().mean().item())
    tg = {"mean_abs_v_true_step_delta": round(float(np.mean(gap_true)), 6),
          "mean_abs_delta_ma": round(float(np.mean(gap_delta)), 6)}
    tg["ratio"] = round(tg["mean_abs_delta_ma"] / max(tg["mean_abs_v_true_step_delta"], 1e-9), 3)
    report["t_gap_cancellation"] = tg
    print(f"\n  t-gap cancellation: |Δ_MA|={tg['mean_abs_delta_ma']} vs "
          f"|v_true(t)−v_true(t−1)|={tg['mean_abs_v_true_step_delta']} "
          f"→ ratio {tg['ratio']} (<1 confirms the t-gap cancels)")

    # ── 3b learnability ────────────────────────────────────────────────
    deltas_by_lag: Dict[int, List[Tuple[int, torch.Tensor]]] = {}
    v_ma_all, v_true_all, x_t_all = [], [], []
    v_true_by_slot: Dict[int, List[torch.Tensor]] = {}
    for entry in refiner_data.iter_generations(data_dir):
        gen = refiner_data.load_generation(data_dir / entry["bin"])
        n_steps = max(int(entry.get("num_steps", 1)), 1)
        for slot in gen.recorded_slots:
            v_true_by_slot.setdefault(slot, [])
            for t in range(min(len(gen.v_true[slot]), 8)):
                vt = gen.v_true[slot][t]
                xt = gen.x[slot][t]
                v_true_by_slot[slot].append(vt)
                region = min(2, 3 * t // n_steps)
                for lag in sorted(gen.v_ma[slot][t]):
                    vm = gen.v_ma[slot][t][lag]
                    deltas_by_lag.setdefault(lag, []).append((region, vt - vm))
                    if len(v_ma_all) < 256:
                        v_ma_all.append(vm)
                        v_true_all.append(vt)
                        x_t_all.append(xt)
    learn = {}
    learn["svd"] = svd_rank_95([d for _, d in deltas_by_lag.get(1, [])][:128])
    learn["predictability_ceiling"] = {
        "per_channel_affine": round(per_channel_affine_ceiling(v_ma_all, v_true_all), 4),
        "pooled_feature": round(pooled_feature_ceiling(x_t_all, v_ma_all, v_true_all), 4),
    }
    learn["v_true_step_correlation"] = step_correlations(v_true_by_slot)
    learn["staleness_curve"] = staleness_curve(deltas_by_lag)
    learn["delta_distribution"] = delta_distribution([d for _, d in deltas_by_lag.get(1, [])])
    report["learnability"] = learn

    rank = learn["svd"]["rank_95"]
    ceiling = learn["predictability_ceiling"]["per_channel_affine"]
    monotone = learn["staleness_curve"].get("per_region_monotone", False)
    gate = {
        "effective_rank_le_8": bool(rank is not None and rank <= 8),
        "ceiling_le_60pct": bool(ceiling is not None and ceiling <= 0.60),
        "staleness_per_region_monotone": bool(monotone),
        "proceed": bool((rank is None or rank <= 8) and (ceiling is None or ceiling <= 0.60)
                        and monotone),
    }
    if not gate["proceed"]:
        gate["note"] = ("reconsider: larger model, or the token-residual "
                        "strategy (plan §4)")
    report["decision_gate"] = gate
    print(f"\n  SVD rank(95%): {rank}   affine ceiling (1−R²): {ceiling}   "
          f"staleness monotone: {monotone}")
    print(f"  Decision gate: {'PROCEED' if gate['proceed'] else 'RECONSIDER'}")

    # ── 3d Day-1 ───────────────────────────────────────────────────────
    print("\n  Day-1 experiment (MLP → tiny UNet, plan 3d)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_err = day1_mlp(x_t_all, v_ma_all, v_true_all, device)
    print(f"  Day-1 MLP (pooled features) mean-Δ error: {mlp_err:.5f}")
    day1 = day1_unet(data_dir, device, max_steps=args.day1_steps,
                     batch_size=args.day1_batch)
    day1["mlp_pooled_mean_error"] = round(float(mlp_err), 5)
    day1["linear_ceiling"] = ceiling
    if day1["day1_unet_rel_mse"] is not None:
        day1["beats_linear_ceiling_by_25pct"] = bool(
            day1["day1_unet_rel_mse"] <= 0.75 * max(ceiling, 1e-9))
        day1["verdict"] = ("build_full_corrector" if day1["beats_linear_ceiling_by_25pct"]
                           else "reconsider_before_full_training")
        print(f"  Day-1 UNet rel-MSE: {day1['day1_unet_rel_mse']:.4f} vs ceiling "
              f"{ceiling:.4f} → {day1['verdict']}")
    else:
        day1["verdict"] = "no_eval_pairs"
    report["day1"] = day1

    # ── Write report ───────────────────────────────────────────────────
    report_path = data_dir.parent / "refiner_probe_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
