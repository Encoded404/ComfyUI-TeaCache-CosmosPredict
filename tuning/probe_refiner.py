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
from .utils import (affine_oob_eval, affine_oob_json, affine_shape_area,
                    affine_shape_label, collect_affine_maps,
                    split_affine_per_pair)
from .utils import (delta_distribution, per_channel_affine_ceiling,
                    pooled_feature_ceiling, step_correlations, staleness_curve,
                    svd_rank_95)

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

    rc = {"record_lags": lags, "record_slots": "both", "dtype": "bfloat16",
          "clevel": int((tcfg.refiner or {}).get("clevel", 9))}
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
#  3b — learnability stats (shared: utils.py, reused by train_corrector)
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
#  Analysis walk (parallel-capable, plan Task 3b)
# ═══════════════════════════════════════════════════════════════════════

_ANALYSIS_CAPS = {"tensor_cap": 12, "shape_cap": 256}
_ANALYSIS_CACHE_DIR = ".probe_cache"
_ANALYSIS_X_STEPS = 8


def _gap_cache_path(cache_dir: Path, name: str) -> Path:
    return Path(cache_dir) / f"{name}.gaps.json"


def _load_gap_cache(cache_dir: Optional[Path], data_dir: Path,
                    entry: dict) -> Optional[dict]:
    """Cached per-generation t-gap lists, valid iff the .bin is unchanged.

    The t-gap cancellation stats (all steps × lags) are the only part of the
    walk that needs a full-generation decode; they are pure functions of the
    recorded tensors, so they can be cached per generation and the learnability
    stats re-decoded from just the first ``_ANALYSIS_X_STEPS`` steps. The cache
    is invalidated by the .bin size + mtime, so re-recorded generations are
    recomputed automatically. Float JSON round-trips are exact.
    """
    if cache_dir is None:
        return None
    p = _gap_cache_path(cache_dir, entry["name"])
    try:
        j = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    binp = data_dir / entry["bin"]
    try:
        st = binp.stat()
    except OSError:
        return None
    if j.get("size") != st.st_size or j.get("mtime") != st.st_mtime:
        return None
    gt, gd = j.get("gap_true"), j.get("gap_delta")
    if not isinstance(gt, list) or not isinstance(gd, list):
        return None
    return {"gap_true": gt, "gap_delta": gd}


def _save_gap_cache(cache_dir: Optional[Path], data_dir: Path,
                    entry: dict, gaps: dict) -> None:
    if cache_dir is None:
        return
    cache_dir = Path(cache_dir)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    binp = data_dir / entry["bin"]
    st = binp.stat()
    j = {"size": st.st_size, "mtime": st.st_mtime,
         "gap_true": gaps["gap_true"], "gap_delta": gaps["gap_delta"]}
    p = _gap_cache_path(cache_dir, entry["name"])
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(j))
    tmp.replace(p)


def _analyze_generation(entry: dict, data_dir: Path, caps: dict,
                        cache_dir: Optional[Path] = None,
                        use_cache: bool = True) -> Tuple[dict, bool]:
    """Decompress one generation and compute its analysis contributions.

    Pure per-generation work with no shared state, so the walk can run over a
    thread pool (blosc2 and numpy release the GIL during decode). The capped
    collections (codec tensors, per-shape samples) are capped per generation
    and truncated again at merge, keeping the same caps as the sequential
    single walk (which generation fills the caps may differ in parallel mode).

    With the analysis cache enabled: on a cache hit the t-gap lists are read
    from ``<data_dir>/.probe_cache`` and only the first 8 steps are decoded
    (``load_generation(max_steps=8, load_prompt=False)`` — the learnability
    stats need nothing else); on a miss the full generation is decoded,
    the t-gap lists computed and cached. Returns (partial, cache_hit).
    """
    cached = None
    if use_cache:
        cached = _load_gap_cache(cache_dir, data_dir, entry)
    if cached is not None:
        gen = refiner_data.load_generation(data_dir / entry["bin"],
                                           max_steps=_ANALYSIS_X_STEPS,
                                           load_prompt=False)
        gap_true, gap_delta = cached["gap_true"], cached["gap_delta"]
    else:
        gen = refiner_data.load_generation(data_dir / entry["bin"],
                                           decode_x_steps=_ANALYSIS_X_STEPS,
                                           load_prompt=False)
        n_steps = max(int(entry.get("num_steps", 1)), 1)
        gap_true, gap_delta = [], []
        for slot in gen.recorded_slots:
            vts = gen.v_true[slot]
            # t-gap cancellation (all steps, all lags)
            for t in range(1, len(vts)):
                gap_true.append(
                    (vts[t].float() - vts[t - 1].float()).abs().mean().item())
                step_ma = gen.v_ma[slot][t]
                for lag in step_ma:
                    gap_delta.append(
                        (vts[t].float() - step_ma[lag].float()).abs().mean().item())
        _save_gap_cache(cache_dir, data_dir, entry,
                        {"gap_true": gap_true, "gap_delta": gap_delta})
    n_steps = max(int(entry.get("num_steps", 1)), 1)
    p = {
        "gap_true": gap_true, "gap_delta": gap_delta,
        "deltas_by_lag": {}, "deltas_by_lag_shape": {},
        "v_ma_by_shape": {}, "v_true_by_shape": {}, "x_t_by_shape": {},
        "v_true_corr_by_shape": {},
        "tensors": {"x_t": [], "v_true": [], "v_ma": [], "delta": []},
    }
    for slot in gen.recorded_slots:
        vts = gen.v_true[slot]
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
    return p, cached is not None


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
                      show_progress: bool = False,
                      cache_dir: Optional[Path] = None,
                      use_cache: bool = True) -> Tuple[dict, dict]:
    """Decompress + analyze all generations, sequentially or over a thread pool.

    In parallel mode torch's CPU intra-op threads are pinned to 1 so the
    per-generation reductions (mean/cos_sim on small tensors) are bit-exact
    across runs and identical to the sequential walk; the decode speed comes
    from blosc2's own thread pool, which is unaffected. The same pinning
    applies to the sequential walk so report values are reproducible
    regardless of machine core count.

    With ``cache_dir`` set, per-generation t-gap stats are cached (see
    ``_analyze_generation``) and cache-hit generations only decode the first
    ``_ANALYSIS_X_STEPS`` steps. Returns (accumulator, {"hits": n, "misses": m}).
    """
    caps = _ANALYSIS_CAPS
    acc = {
        "gap_true": [], "gap_delta": [],
        "deltas_by_lag": {}, "deltas_by_lag_shape": {},
        "v_ma_by_shape": {}, "v_true_by_shape": {}, "x_t_by_shape": {},
        "v_true_corr_by_shape": {},
        "tensors": {"x_t": [], "v_true": [], "v_ma": [], "delta": []},
    }
    stats = {"hits": 0, "misses": 0}
    old_torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    bar = _bar(show_progress, len(entries), "analysis", "gen")
    try:
        if n_threads <= 1:
            for entry in entries:
                p, hit = _analyze_generation(entry, data_dir, caps, cache_dir, use_cache)
                _merge_analysis(acc, p, caps)
                stats["hits" if hit else "misses"] += 1
                if bar is not None:
                    bar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=n_threads) as ex:
                futures = [ex.submit(_analyze_generation, entry, data_dir, caps,
                                     cache_dir, use_cache)
                           for entry in entries]
                for f in as_completed(futures):
                    p, hit = f.result()
                    _merge_analysis(acc, p, caps)
                    stats["hits" if hit else "misses"] += 1
                    if bar is not None:
                        bar.update(1)
    finally:
        torch.set_num_threads(old_torch_threads)
    if bar is not None:
        bar.close()
    return acc, stats


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
              cache_size: int = 64,
              recovery_fit_batches: int = 128,
              recovery_eval_batches: int = 64) -> dict:
    """Tiny 2D UNet on the real pairs (3d stage 2); eval vs the linear ceiling.

    ``recovery_fit_batches`` / ``recovery_eval_batches`` cap how many batch
    tensors per latent shape are held in RAM for the OOD affine recovery row
    (fit side / scored eval side; config: ``refiner_training``).

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
    def eval_pairs(ds, fit_maps, eval_map_batches):
        if ds is None:
            return None
        errs, base_errs = [], []
        split = {"base": {"ladder": [], "anchor": []},
                 "model": {"ladder": [], "anchor": []}}
        eval_maps = {}
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
                rel = (err / den).tolist()
                base = (b["v_ma"] - b["v_true"]).float().flatten(1).norm(dim=1)
                base_rel = (base / den).tolist()
                anchors = (b["lag"].cpu() == 0).tolist()
                errs.extend(rel)
                base_errs.extend(base_rel)
                # Eval maps for the OOD affine row are capped per shape (the
                # base/model per-pair stats above stay on ALL pairs — floats
                # only). Keeping the tensors unbounded is what blew past RAM.
                vm = b["v_ma"].float().cpu()
                vt = b["v_true"].float().cpu()
                shape = (int(vm.shape[-2]), int(vm.shape[-1]))
                em = eval_maps.setdefault(shape, {"vm": [], "vt": [], "anchor": []})
                if len(em["vm"]) < eval_map_batches:
                    em["vm"].append(vm)
                    em["vt"].append(vt)
                    em["anchor"].extend(anchors)
                for r, br, is_a in zip(rel, base_rel, anchors):
                    (split["base"]["anchor"] if is_a else split["base"]["ladder"]).append(br)
                    (split["model"]["anchor"] if is_a else split["model"]["ladder"]).append(r)
                if bar2 is not None:
                    bar2.update(1)
        model.train()
        if bar2 is not None:
            bar2.close()
        # OOD affine: fit per stratum on TRAIN maps, score per stratum on the
        # eval maps (the UNet is also evaluated only on eval pairs — the same
        # distribution contract). d=0 anchors are split out because identity
        # scores exactly 0 there and any |a−1|/|b| is pure loss.
        aff = affine_oob_eval(fit_maps, eval_maps)
        aff_lad, aff_anc = split_affine_per_pair(aff, eval_maps)
        split["affine"] = {"ladder": aff_lad, "anchor": aff_anc}
        return {"rel_mse": float(np.mean(errs)) if errs else None,
                "base_rel_mse": float(np.mean(base_errs)) if base_errs else None,
                "split": split,
                "affine": aff,
                "n_pairs": len(errs)}

    t_e0 = time.time()
    ev = None
    if eval_ds is not None:
        # Fit maps only for the strata the eval set actually uses, so the
        # fit loader is a short pass (per-shape cap + visited-batch cap).
        eval_shapes = set(eval_ds.pair_shapes())
        fit_sampler = CorrectorBatchSampler(
            train_ds, 16, seed + 5,
            include_areas=sorted(h * w for h, w in eval_shapes))
        fit_loader = DataLoader(train_ds, batch_sampler=fit_sampler,
                                collate_fn=collate_corrector, num_workers=0)
        fit_maps = collect_affine_maps(fit_loader,
                                       cap_batches_per_shape=recovery_fit_batches,
                                       known_shapes=eval_shapes)
        ev = eval_pairs(eval_ds, fit_maps, recovery_eval_batches)
        del fit_maps
    day1_eval = ev["rel_mse"] if ev is not None else None
    if timer is not None:
        timer.add("eval", time.time() - t_e0)

    split_json = None
    affine_json = None
    base_rel_mse = None
    n_pairs = 0
    if ev is not None:
        base_rel_mse = ev["base_rel_mse"]
        n_pairs = ev["n_pairs"]
        split_json = {"base": {k: round(float(np.mean(v)), 5) if v else None
                               for k, v in ev["split"]["base"].items()},
                      "model": {k: round(float(np.mean(v)), 5) if v else None
                                for k, v in ev["split"]["model"].items()},
                      "affine": {k: round(float(np.mean(v)), 5) if v else None
                                 for k, v in ev["split"]["affine"].items()}}
        affine_json = affine_oob_json(ev["affine"])
        del ev  # release the eval-map tensors before the lag-readability pass

    # Lag-readability (plan 3d): Δ̂ vs lag for fixed (x_t, t) — smooth/monotone?
    t_l0 = time.time()
    lag_stats = _lag_readability(model, device, data_dir,
                                 show_progress=show_progress, manifest=manifest)
    if timer is not None:
        timer.add("lag", time.time() - t_l0)

    return {"day1_unet_rel_mse": day1_eval,
            "base_rel_mse": base_rel_mse,
            "recovery_split": split_json,
            "affine_oob": affine_json,
            "eval_pairs": n_pairs,
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
    parser.add_argument("--no-analysis-cache", type=int, default=None,
                        help="Disable the per-generation t-gap analysis cache "
                             "(default: enabled; re-runs decode only the "
                             "first 8 steps per generation)")
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
    # run over a thread pool — default min(8, cpu count), --analysis-threads).
    # Per-generation t-gap stats are cached in <data_dir>/.probe_cache so
    # re-runs only decode the first 8 steps (--no-analysis-cache disables).
    if args.analysis_threads is None:
        analysis_threads = min(8, os.cpu_count() or 1)
    else:
        analysis_threads = max(1, int(args.analysis_threads))
    cache_dir = None
    use_cache = not bool(args.no_analysis_cache)
    if use_cache:
        cache_dir = data_dir / _ANALYSIS_CACHE_DIR
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"  [probe] ⚠ analysis cache unavailable: {e} — continuing "
                  "without caching")
            cache_dir = None
    print(f"  Loading generations and collecting tensors/stats "
          f"({analysis_threads} analysis thread(s))...")
    manifest = refiner_data.load_manifest(data_dir)
    entries = refiner_data.iter_generations(data_dir, manifest=manifest)
    if entries:
        e0 = entries[0]
        legacy = "" if "clevel" in e0 else " (legacy default)"
        print(f"  Codec/level:    {e0.get('codec')} / clevel "
              f"{e0.get('clevel', 9)}{legacy}")
    t_an0 = time.time()
    acc, walk_stats = run_analysis_walk(entries, data_dir, analysis_threads,
                                        show_progress=show_progress,
                                        cache_dir=cache_dir, use_cache=use_cache)
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
    if cache_dir is not None:
        print(f"  Analysis cache: {walk_stats['hits']}/{n_gens} hit "
              f"({walk_stats['misses']} decoded fully)")
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
                     cache_size=day1_cache_size,
                     recovery_fit_batches=int((tcfg.refiner_training or {}).get("recovery_fit_batches", 128)),
                     recovery_eval_batches=int((tcfg.refiner_training or {}).get("recovery_eval_batches", 64)))
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
        "base_rel_mse": day1.get("base_rel_mse"),
        "affine_rel_mse": (day1.get("affine_oob") or {}).get("overall"),
        "verdict": day1["verdict"], "wall_s": round(timer.phase_seconds("train"), 2),
    })

    # ── Recovery table: each method's v_t error vs the TeaCache base ─────
    base_rel = day1.get("base_rel_mse")
    if base_rel is not None and base_rel > 0:
        split = day1.get("recovery_split") or {}
        aff = day1.get("affine_oob") or {}
        b_lad = (split.get("base") or {}).get("ladder")
        b_anc = (split.get("base") or {}).get("anchor")
        m_lad = (split.get("model") or {}).get("ladder")
        m_anc = (split.get("model") or {}).get("anchor")
        a_lad = (split.get("affine") or {}).get("ladder")
        a_anc = (split.get("affine") or {}).get("anchor")
        print("\n  v_t error recovery vs TeaCache base "
              "(rel-L2 ‖v̂−v_true‖₂/‖v_true‖₂, eval pairs)")
        print("  " + "─" * 62)
        print(f"  {'method':<24}{'abs err':>10}{'× base':>9}{'recovered':>12}")
        print("  " + "─" * 62)
        rows = [
            {"name": "TeaCache base (v_MA)", "err": base_rel, "ladder": b_lad,
             "anchor": b_anc, "is_base": True},
            {"name": "Per-channel affine (OOD)", "err": aff.get("overall"),
             "ladder": a_lad, "anchor": a_anc},
            {"name": "Day-1 tiny UNet", "err": day1.get("day1_unet_rel_mse"),
             "ladder": m_lad, "anchor": m_anc},
            {"name": "Oracle (v_true)", "err": 0.0, "ladder": None, "anchor": None,
             "oracle": True},
        ]
        recovery = {}
        for r in rows:
            name, err = r["name"], r["err"]
            if err is None:
                print(f"  {name:<24}{'—':>10}")
                recovery[name] = None
                continue
            ratio = 0.0 if r.get("oracle") else err / base_rel
            rec = 1.0 if r.get("oracle") else (None if r.get("is_base") else 1.0 - ratio)
            recovery[name] = {"abs_err": round(err, 5),
                              "ratio_base": round(ratio, 4),
                              "recovered": round(rec, 4) if rec is not None else None,
                              "ladder_only": round(r["ladder"], 5) if r["ladder"] is not None else None,
                              "anchors_only": round(r["anchor"], 5) if r["anchor"] is not None else None}
            rec_str = "—" if rec is None else f"{100 * rec:>10.1f}%"
            print(f"  {name:<24}{err:>10.5f}{ratio:>9.3f}{rec_str:>12}")
            if r["ladder"] is not None:
                print(f"    ladder only          {r['ladder']:>10.5f}")
            if r["anchor"] is not None:
                print(f"    d=0 anchors only     {r['anchor']:>10.5f}")
        print("  " + "─" * 62)
        by_shape = aff.get("by_shape") or {}
        if by_shape:
            print("  affine per stratum (fit = train pairs, scored = eval pairs):")
            for shape, d in sorted(by_shape.items(), key=lambda kv: affine_shape_area(kv[0])):
                if d.get("rel_mse") is None:
                    print(f"    {affine_shape_label(shape)}:  no train fit pairs — row skipped")
                    continue
                print(f"    {affine_shape_label(shape)}:  rel {d['rel_mse']:.4f}  "
                      f"(n={d['n_pairs']}, fit {d['fit_n_batches']} batches)  "
                      f"|a−1|≤{d['max_abs_a_minus_1']:.4f}  |b|≤{d['max_abs_b']:.4f}")
        day1["recovery"] = recovery
        report["day1"] = day1
        metrics.write({"type": "recovery", "base_rel_mse": round(base_rel, 5),
                       "methods": recovery, "affine_oob": aff})

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
