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
``train_corrector``'s did-it-learn gate). The report is written progressively
(atomic tmp+replace) so a crash keeps the results computed so far. Live
reporting mirrors ``train_corrector``: tqdm bars for the recording, analysis
and Day-1 phases (``--no-progress`` disables), per-50-step Day-1 log lines, a
JSONL metrics stream (``probe_metrics.jsonl``; ``--metrics`` relocates), and a
final summary with per-phase timings and VRAM peak.

Usage:
    python -m tuning.probe_refiner --comfy-dir /path/to/ComfyUI   # record + analyze
    python -m tuning.probe_refiner --data outputs/<ts>/refiner_data  # analyze only
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from .utils import MetricsLog, TrainTimer, format_duration
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

try:
    from tqdm import tqdm
except ImportError:  # progress bars are optional (requirements: tqdm)
    tqdm = None


def _bar(show_progress: bool, total: Optional[int], desc: str, unit: str = "it"):
    """tqdm bar or None (tqdm missing or ``show_progress`` disabled)."""
    if not show_progress or tqdm is None:
        return None
    return tqdm(total=total, desc=desc, unit=unit, leave=False,
                dynamic_ncols=True, mininterval=0.5)


def emit(bar, line: str) -> None:
    """Print a log line above the active progress bar (bar-safe)."""
    if bar is not None:
        bar.write(line)
    else:
        print(line)


def _write_report(report: dict, data_dir: Path) -> Path:
    """Atomic (tmp+replace) write of the report — crash-safe, progressive."""
    report_path = Path(data_dir).parent / "refiner_probe_report.json"
    tmp = report_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, default=str))
    tmp.replace(report_path)
    return report_path


# ═══════════════════════════════════════════════════════════════════════
#  Recording (3–5 generations, plan Task 3)
# ═══════════════════════════════════════════════════════════════════════


def record_probe_generations(tcfg: TuningConfig, num_gens: int, seeds: List[int],
                             lags: List[int], out_dir: Path,
                             show_progress: bool = False) -> Path:
    """Run *num_gens* generations with refiner capture into out_dir/refiner_data.

    A failed generation is logged and skipped (the manifest keeps the ones
    that succeeded); the run aborts only if every generation fails.
    """
    from .calibrate import patch_for_refiner, restore_model
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
    pbar = _bar(show_progress, len(entries), "record", "gen")
    failures = 0
    for i, pdata in enumerate(entries):
        seed = seeds[i % len(seeds)]
        full_prompt, neg_prompt, _ = resolve_generation(
            prompt_sampler, pdata.entry, i, seed, steps, width, height,
        )
        dm, original_fwd = patch_for_refiner(
            unet, steps, prompt_id=i, seed=seed, track_per_block=False,
            refiner_dir=str(refiner_dir), refiner_cfg=rc,
        )
        t_gen = time.time()
        ok = True
        try:
            sample(unet, clip, vae, full_prompt, seed=seed, steps=steps, cfg=cfg_val,
                   sampler_name=sampler_name, scheduler=scheduler,
                   width=width, height=height, negative=neg_prompt)
        except Exception as e:
            ok = False
            failures += 1
            emit(pbar, f"  ⚠ [probe] generation {i + 1}/{len(entries)} failed: "
                       f"{type(e).__name__}: {e}")
        finally:
            restore_model(dm, original_fwd, unet)
        if not ok:
            if pbar is not None:
                pbar.update(1)
            continue
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
        emit(pbar, f"  [probe] recorded generation {i + 1}/{len(entries)} "
                   f"({format_duration(time.time() - t_gen)} on GPU)")
        if pbar is not None:
            pbar.set_postfix(prompt=i, seed=seed, refresh=False)
            pbar.update(1)
    if failures == len(entries):
        raise RuntimeError(f"[probe] all {len(entries)} generations failed to record")
    refiner_data.save_manifest(refiner_dir, manifest)
    if pbar is not None:
        pbar.close()
    if failures:
        print(f"  [probe] {failures}/{len(entries)} generations failed — "
              f"continuing with the recorded data")
    return refiner_dir


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
#  Analysis walk (parallel-capable, plan Task 3b)
# ═══════════════════════════════════════════════════════════════════════

_ANALYSIS_CAPS = {"tensor_cap": 12, "shape_cap": 256}


def _analyze_generation(entry: dict, data_dir: Path, caps: dict) -> dict:
    """Decompress one generation and compute its analysis contributions.

    Pure per-generation work with no shared state, so the walk can run over a
    thread pool (blosc2 and numpy release the GIL during decode). The capped
    collections (codec tensors, per-shape samples) are capped per generation
    and truncated again at merge, keeping the same caps as the sequential
    single walk (which generation fills the caps may differ in parallel mode).
    """
    gen = refiner_data.load_generation(data_dir / entry["bin"])
    n_steps = max(int(entry.get("num_steps", 1)), 1)
    p = {
        "gap_true": [], "gap_delta": [],
        "deltas_by_lag": {}, "deltas_by_lag_shape": {},
        "v_ma_by_shape": {}, "v_true_by_shape": {}, "x_t_by_shape": {},
        "v_true_corr_by_shape": {},
        "tensors": {"x_t": [], "v_true": [], "v_ma": [], "delta": []},
    }
    for slot in gen.recorded_slots:
        vts = gen.v_true[slot]
        # t-gap cancellation (all steps, all lags)
        for t in range(1, len(vts)):
            p["gap_true"].append(
                (vts[t].float() - vts[t - 1].float()).abs().mean().item())
            step_ma = gen.v_ma[slot][t]
            for lag in step_ma:
                p["gap_delta"].append(
                    (vts[t].float() - step_ma[lag].float()).abs().mean().item())
        # learnability + codec tensor collection (first 8 steps)
        for t in range(min(len(vts), 8)):
            vt = vts[t]
            xt = gen.x[slot][t]
            vshape = (int(vt.shape[-2]), int(vt.shape[-1]))
            p["v_true_corr_by_shape"].setdefault(vshape, []).append(vt)
            if len(p["tensors"]["x_t"]) < caps["tensor_cap"]:
                p["tensors"]["x_t"].append(xt)
                p["tensors"]["v_true"].append(vt)
            region = min(2, 3 * t // n_steps)
            for lag in sorted(gen.v_ma[slot][t]):
                vm = gen.v_ma[slot][t][lag]
                d = vt - vm
                shape = (int(d.shape[-2]), int(d.shape[-1]))
                p["deltas_by_lag"].setdefault(lag, []).append((region, d))
                p["deltas_by_lag_shape"].setdefault(lag, {}).setdefault(
                    shape, []).append((region, d))
                if len(p["tensors"]["v_ma"]) < caps["tensor_cap"]:
                    p["tensors"]["v_ma"].append(vm)
                    p["tensors"]["delta"].append(d)
                if len(p["v_ma_by_shape"].setdefault(shape, [])) < caps["shape_cap"]:
                    p["v_ma_by_shape"][shape].append(vm)
                    p["v_true_by_shape"].setdefault(shape, []).append(vt)
                    p["x_t_by_shape"].setdefault(shape, []).append(xt)
    return p


def _merge_analysis(acc: dict, p: dict, caps: dict) -> None:
    """Merge one generation's partial analysis into the accumulator."""
    acc["gap_true"].extend(p["gap_true"])
    acc["gap_delta"].extend(p["gap_delta"])
    for lag, items in p["deltas_by_lag"].items():
        acc["deltas_by_lag"].setdefault(lag, []).extend(items)
    for lag, shapes in p["deltas_by_lag_shape"].items():
        for shape, items in shapes.items():
            acc["deltas_by_lag_shape"].setdefault(lag, {}).setdefault(
                shape, []).extend(items)
    for shape, items in p["v_true_corr_by_shape"].items():
        acc["v_true_corr_by_shape"].setdefault(shape, []).extend(items)
    for shape, items in p["v_ma_by_shape"].items():
        acc["v_ma_by_shape"].setdefault(shape, []).extend(items)
    for shape, items in p["v_true_by_shape"].items():
        acc["v_true_by_shape"].setdefault(shape, []).extend(items)
    for shape, items in p["x_t_by_shape"].items():
        acc["x_t_by_shape"].setdefault(shape, []).extend(items)
    for k in ("x_t", "v_true", "v_ma", "delta"):
        acc["tensors"][k].extend(p["tensors"][k])
    # Global caps at merge (parallel workers over-collect per generation);
    # v_ma/v_true/x_t per-shape lists stay aligned (appended in lockstep).
    for k in ("x_t", "v_true", "v_ma", "delta"):
        acc["tensors"][k] = acc["tensors"][k][: caps["tensor_cap"]]
    for shape in list(acc["v_ma_by_shape"]):
        acc["v_ma_by_shape"][shape] = acc["v_ma_by_shape"][shape][: caps["shape_cap"]]
        acc["v_true_by_shape"][shape] = acc["v_true_by_shape"][shape][: caps["shape_cap"]]
        acc["x_t_by_shape"][shape] = acc["x_t_by_shape"][shape][: caps["shape_cap"]]


def run_analysis_walk(entries: List[dict], data_dir: Path, n_threads: int,
                      show_progress: bool = False) -> dict:
    """Decompress + analyze all generations, sequentially or over a thread pool.

    In parallel mode torch's CPU intra-op threads are pinned to 1 so the
    per-generation reductions (mean/cos_sim on small tensors) are bit-exact
    across runs and identical to the sequential walk; the decode speed comes
    from blosc2's own thread pool, which is unaffected. The same pinning
    applies to the sequential walk so report values are reproducible
    regardless of machine core count.
    """
    caps = _ANALYSIS_CAPS
    acc = {
        "gap_true": [], "gap_delta": [],
        "deltas_by_lag": {}, "deltas_by_lag_shape": {},
        "v_ma_by_shape": {}, "v_true_by_shape": {}, "x_t_by_shape": {},
        "v_true_corr_by_shape": {},
        "tensors": {"x_t": [], "v_true": [], "v_ma": [], "delta": []},
    }
    old_torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    bar = _bar(show_progress, len(entries), "analysis", "gen")
    try:
        if n_threads <= 1:
            for entry in entries:
                _merge_analysis(acc, _analyze_generation(entry, data_dir, caps), caps)
                if bar is not None:
                    bar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=n_threads) as ex:
                futures = [ex.submit(_analyze_generation, entry, data_dir, caps)
                           for entry in entries]
                for f in as_completed(futures):
                    _merge_analysis(acc, f.result(), caps)
                    if bar is not None:
                        bar.update(1)
    finally:
        torch.set_num_threads(old_torch_threads)
    if bar is not None:
        bar.close()
    return acc


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
              seed: int = 42, show_progress: bool = False,
              metrics: Optional[MetricsLog] = None,
              timer: Optional[TrainTimer] = None,
              manifest: Optional[dict] = None,
              cache_size: int = 64) -> dict:
    """Tiny 2D UNet on the real pairs (3d stage 2); eval vs the linear ceiling.

    Live reporting mirrors ``train_corrector``: tqdm bar with EMA loss, it/s
    and remaining, per-50-step log lines, and per-step JSONL rows.
    """
    torch.manual_seed(seed)
    train_ds = CorrectorDataset(data_dir, seed=seed, show_progress=show_progress,
                                normalization_samples=0, manifest=manifest,
                                gen_cache_size=cache_size)
    train_loader = DataLoader(train_ds, batch_sampler=CorrectorBatchSampler(
        train_ds, batch_size=batch_size, seed=seed), collate_fn=collate_corrector,
        num_workers=0)
    try:
        eval_ds = CorrectorDataset(data_dir, only_eval=True, seed=seed,
                                   show_progress=show_progress,
                                   normalization_samples=0, manifest=manifest,
                                   gen_cache_size=cache_size)
    except ValueError:
        eval_ds = None

    ccfg = CorrectorConfig.for_size("tiny")
    model = CorrectorUNet2D(ccfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    it = iter(train_loader)
    pbar = _bar(show_progress, max_steps, "day1-unet", "step")
    t_start = time.time()
    loss_ema: Optional[float] = None
    min_loss = float("inf")
    final_loss = float("nan")
    for step in range(1, max_steps + 1):
        t_step0 = time.time()
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
        loss_v = float(loss.item())
        loss_ema = loss_v if loss_ema is None else 0.99 * loss_ema + 0.01 * loss_v
        min_loss = min(min_loss, loss_v)
        final_loss = loss_v
        if timer is not None:
            timer.record_step(time.time() - t_step0)
        if pbar is not None:
            st = timer.step_time() if timer is not None else 0.0
            rem = st * (max_steps - step) if st > 0 else 0.0
            pbar.set_postfix(loss=f"{loss_ema:.5f}",
                             it=f"{timer.steps_per_sec() if timer is not None else float('nan'):.2f}/s",
                             rem=format_duration(rem), refresh=False)
            pbar.update(1)
        if step % 50 == 0 or step == 1:
            line = (f"  [day1] step {step:>4d}/{max_steps}  loss={loss_v:.5f} "
                    f"(ema {loss_ema:.5f}, min {min_loss:.5f})  "
                    f"it/s={timer.steps_per_sec() if timer is not None else float('nan'):.2f}  "
                    f"elapsed={format_duration(time.time() - t_start)}")
            emit(pbar, line)
            if metrics is not None:
                metrics.write({
                    "type": "day1_step", "step": step, "loss": loss_v,
                    "loss_ema": loss_ema, "min_loss": min_loss,
                    "wall_s": time.time() - t_start,
                    "vram_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
                })
    if pbar is not None:
        pbar.close()
    # eval
    def eval_pairs(ds):
        if ds is None:
            return None
        errs = []
        loader = DataLoader(ds, batch_sampler=CorrectorBatchSampler(ds, 16, seed + 1),
                            collate_fn=collate_corrector, num_workers=0)
        bar2 = _bar(show_progress, len(loader), "day1-eval", "batch")
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
                if bar2 is not None:
                    bar2.update(1)
        model.train()
        if bar2 is not None:
            bar2.close()
        return float(np.mean(errs))

    t_e0 = time.time()
    day1_eval = eval_pairs(eval_ds)
    if timer is not None:
        timer.add("eval", time.time() - t_e0)

    # Lag-readability (plan 3d): Δ̂ vs lag for fixed (x_t, t) — smooth/monotone?
    t_l0 = time.time()
    lag_stats = _lag_readability(model, device, data_dir,
                                 show_progress=show_progress, manifest=manifest)
    if timer is not None:
        timer.add("lag", time.time() - t_l0)

    return {"day1_unet_rel_mse": day1_eval,
            "lag_readability": lag_stats,
            "loss_final": round(final_loss, 6),
            "loss_min": round(min_loss, 6),
            "steps": max_steps,
            "wall_s": round(time.time() - t_start, 1)}


def _lag_readability(model, device, data_dir, show_progress: bool = False,
                     manifest: Optional[dict] = None) -> dict:
    """For fixed (x_t, t) steps: Δ̂(lag) must vary smoothly (ideally monotonically).

    Uses the trained Day-1 UNet on steps where the full ladder exists: the
    corrections for each lag share (x_t, prompt, t) and differ only in v_MA_d,
    so monotone ‖Δ̂_d‖ with d means staleness is decodable from v_MA's content
    (d is never a model input).
    """
    from scipy.stats import spearmanr
    mags_by_step, smoothness = [], []
    n = 0
    entries = refiner_data.iter_generations(data_dir, manifest=manifest)
    bar = _bar(show_progress, len(entries), "lag-readability", "gen")
    for entry in entries:
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
        if bar is not None:
            bar.update(1)
        if n >= 24:
            break
    if bar is not None:
        bar.close()
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
    parser.add_argument("--analysis-threads", type=int, default=None,
                        help="Threads for the generation analysis walk "
                             "(default: min(8, cpu count); 1 = sequential)")
    parser.add_argument("--no-progress", type=int, default=None,
                        help="Disable tqdm progress bars (default: auto TTY detect)")
    parser.add_argument("--metrics", default=None,
                        help="JSONL metrics path (default: <data_dir>/../probe_metrics.jsonl)")
    args = parser.parse_args(argv)

    show_progress = tqdm is not None and not bool(args.no_progress)
    timer = TrainTimer(window=100)
    t_start = time.time()

    if args.config is None:
        args.config = str(Path(__file__).parent / "config.json")
    tcfg = TuningConfig.load(args.config)

    if args.data:
        data_dir = Path(args.data)
        recording = None
    else:
        if not args.comfy_dir:
            raise SystemExit("provide --data (existing refiner_data) or --comfy-dir (record)")
        tcfg.comfy_dir = args.comfy_dir
        out_dir = Path(tcfg.output_dir) / time.strftime("%Y%m%d-%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        seeds = [int(s) for s in args.seeds.split(",")]
        lags = refiner_data.parse_lags(args.record_lags)
        t_rec0 = time.time()
        data_dir = record_probe_generations(tcfg, max(2, min(args.prompts, 5)),
                                            seeds, lags, out_dir,
                                            show_progress=show_progress)
        timer.add("record", time.time() - t_rec0)
        recording = {"gens_requested": max(2, min(args.prompts, 5))}
    report = {"data": str(data_dir), "lags": refiner_data.parse_lags(args.record_lags)}
    if recording is not None:
        report["recording"] = recording

    metrics = MetricsLog(args.metrics or str(Path(data_dir).parent / "probe_metrics.jsonl"))

    print("=" * 60)
    print("  Refiner Probe")
    print("=" * 60)
    print(f"  Data: {data_dir}")

    # ── Single walk: decode each generation once for codecs + t-gap + learnability ──
    # (parallel-capable: decode is threaded inside blosc2; the walk itself can
    # run over a thread pool — default min(8, cpu count), --analysis-threads)
    if args.analysis_threads is None:
        analysis_threads = min(8, os.cpu_count() or 1)
    else:
        analysis_threads = max(1, int(args.analysis_threads))
    print(f"  Loading generations and collecting tensors/stats "
          f"({analysis_threads} analysis thread(s))...")
    manifest = refiner_data.load_manifest(data_dir)
    entries = refiner_data.iter_generations(data_dir, manifest=manifest)
    t_an0 = time.time()
    acc = run_analysis_walk(entries, data_dir, analysis_threads,
                            show_progress=show_progress)
    tensors = acc["tensors"]
    gap_true, gap_delta = acc["gap_true"], acc["gap_delta"]
    deltas_by_lag: Dict[int, List[Tuple[int, torch.Tensor]]] = acc["deltas_by_lag"]
    deltas_by_lag_shape: Dict[int, Dict[Tuple[int, int], List[Tuple[int, torch.Tensor]]]] = acc["deltas_by_lag_shape"]
    v_ma_by_shape: Dict[Tuple[int, int], List[torch.Tensor]] = acc["v_ma_by_shape"]
    v_true_by_shape: Dict[Tuple[int, int], List[torch.Tensor]] = acc["v_true_by_shape"]
    x_t_by_shape: Dict[Tuple[int, int], List[torch.Tensor]] = acc["x_t_by_shape"]
    v_true_corr_by_shape: Dict[Tuple[int, int], List[torch.Tensor]] = acc["v_true_corr_by_shape"]
    timer.add("analysis", time.time() - t_an0)
    n_gens = len(entries)
    print(f"  Generations: {n_gens}  tensors: "
          f"{ {k: len(v) for k, v in tensors.items()} }")
    if not tensors["delta"]:
        raise SystemExit("[probe] no recorded pairs found — nothing to analyze")
    metrics.write({"type": "phase", "phase": "analysis", "gens": n_gens,
                   "wall_s": round(timer.phase_seconds("analysis"), 2)})

    # ── 3a codecs + t-gap cancellation ─────────────────────────────────
    t_c0 = time.time()
    codec_results = benchmark_codecs(tensors)
    timer.add("codec", time.time() - t_c0)
    report["codec"] = codec_results
    tg = {"mean_abs_v_true_step_delta": round(float(np.mean(gap_true)), 6),
          "mean_abs_delta_ma": round(float(np.mean(gap_delta)), 6)}
    tg["ratio"] = round(tg["mean_abs_delta_ma"] / max(tg["mean_abs_v_true_step_delta"], 1e-9), 3)
    report["t_gap_cancellation"] = tg
    print(f"\n  t-gap cancellation: |Δ_MA|={tg['mean_abs_delta_ma']} vs "
          f"|v_true(t)−v_true(t−1)|={tg['mean_abs_v_true_step_delta']} "
          f"→ ratio {tg['ratio']} (<1 confirms the t-gap cancels)")
    _write_report(report, data_dir)

    # ── 3b learnability (per resolution stratum — the corpus mixes latent
    # shapes, and SVD rank / affine ceiling are resolution-dependent) ─────
    learn = {}
    d1 = deltas_by_lag_shape.get(1, {})
    learn["svd"] = {f"{s[0]}x{s[1]}": svd_rank_95([d for _, d in d1.get(s, [])][:128])
                    for s in sorted(d1, key=lambda s: s[0] * s[1])}
    pooled_x = [t for s in sorted(x_t_by_shape) for t in x_t_by_shape[s]]
    pooled_ma = [t for s in sorted(v_ma_by_shape) for t in v_ma_by_shape[s]]
    pooled_vt = [t for s in sorted(v_true_by_shape) for t in v_true_by_shape[s]]
    learn["predictability_ceiling"] = {
        "per_channel_affine": {f"{s[0]}x{s[1]}":
                               round(per_channel_affine_ceiling(v_ma_by_shape[s], v_true_by_shape[s]), 4)
                               for s in sorted(v_ma_by_shape, key=lambda s: s[0] * s[1])},
        "pooled_feature": round(pooled_feature_ceiling(pooled_x, pooled_ma, pooled_vt), 4),
    }
    learn["v_true_step_correlation"] = step_correlations(v_true_corr_by_shape)
    learn["staleness_curve"] = staleness_curve(deltas_by_lag)
    learn["delta_distribution"] = {f"{s[0]}x{s[1]}": delta_distribution([d for _, d in d1.get(s, [])])
                                   for s in sorted(d1, key=lambda s: s[0] * s[1])}
    report["learnability"] = learn

    # Decision gate on the dominant (most lag-1 delta samples) stratum — the
    # pooled-mix numbers would average incomparable resolutions.
    dom = max(d1, key=lambda s: len(d1[s])) if d1 else None
    dom_key = f"{dom[0]}x{dom[1]}" if dom else None
    rank = learn["svd"].get(dom_key, {}).get("rank_95") if dom_key else None
    ceiling = learn["predictability_ceiling"]["per_channel_affine"].get(dom_key) \
        if dom_key else None
    monotone = learn["staleness_curve"].get("per_region_monotone", False)
    gate = {
        "effective_rank_le_8": bool(rank is not None and rank <= 8),
        "ceiling_le_60pct": bool(ceiling is not None and ceiling <= 0.60),
        "staleness_per_region_monotone": bool(monotone),
        "stratum": dom_key,
        "proceed": bool((rank is None or rank <= 8) and (ceiling is None or ceiling <= 0.60)
                        and monotone),
    }
    if not gate["proceed"]:
        gate["note"] = ("reconsider: larger model, or the token-residual "
                        "strategy (plan §4)")
    report["decision_gate"] = gate
    print(f"\n  SVD rank(95%): " + "  ".join(
        f"{k}={v.get('rank_95')}" for k, v in sorted(learn['svd'].items())))
    print(f"  Affine ceiling (1−R²): " + "  ".join(
        f"{k}={v:.4f}" for k, v in sorted(learn['predictability_ceiling']['per_channel_affine'].items())))
    print(f"  Staleness monotone: {monotone}")
    print(f"  Decision gate ({dom_key or 'no lag-1 deltas'}): "
          f"{'PROCEED' if gate['proceed'] else 'RECONSIDER'}")
    stal = learn["staleness_curve"]
    for r in range(3):
        region = stal.get(f"region_{r}")
        if region:
            print(f"      staleness region {r}: " + "  ".join(
                f"d{lag}={v:.4f}" for lag, v in sorted(region.items())))
    _write_report(report, data_dir)

    # ── 3d Day-1 ───────────────────────────────────────────────────────
    print("\n  Day-1 experiment (MLP → tiny UNet, plan 3d)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_err = day1_mlp(pooled_x, pooled_ma, pooled_vt, device)
    print(f"  Day-1 MLP (pooled features) mean-Δ error: {mlp_err:.5f}")
    day1_cache_size = int((tcfg.refiner_training or {}).get("cache_size", 64))
    day1 = day1_unet(data_dir, device, max_steps=args.day1_steps,
                     batch_size=args.day1_batch, show_progress=show_progress,
                     metrics=metrics, timer=timer, manifest=manifest,
                     cache_size=day1_cache_size)
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
    metrics.write({
        "type": "day1_eval", "rel_mse": day1.get("day1_unet_rel_mse"),
        "verdict": day1["verdict"], "wall_s": round(timer.phase_seconds("train"), 2),
    })

    # ── Write report + summary ─────────────────────────────────────────
    report_path = _write_report(report, data_dir)
    wall = time.time() - t_start
    phase_parts = []
    for name, label in (("record", "record"), ("analysis", "analysis"),
                        ("codec", "codec bench"), ("train", "day1 train"),
                        ("eval", "day1 eval"), ("lag", "lag-readability")):
        s = timer.phase_seconds(name)
        if s > 0:
            phase_parts.append(f"{label} {format_duration(s)}")
    print("\n  ── Probe complete ──")
    print(f"  Wall time:    {format_duration(wall)}"
          + (f"  ({' | '.join(phase_parts)})" if phase_parts else ""))
    print(f"  VRAM peak:    {torch.cuda.max_memory_allocated() / (1024 ** 3):.1f} GB")
    print(f"  Metrics:      {metrics.path}")
    print(f"  Report:       {report_path}")
    print("=" * 60)
    metrics.write({"type": "done", "wall_s": round(wall, 2)})
    metrics.close()


if __name__ == "__main__":
    main()
