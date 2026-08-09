#!/usr/bin/env python3
"""Phase 3: End-to-end validation of top configurations — BASE TeaCache side.

This module verifies the TeaCache configurations themselves (skip rates,
wall-clock speedup, image quality vs precomputed baselines). The latent
CORRECTOR (Mode B′) has its own verification suite — ``tuning/verify_refiner.py``
(deployed-weighted recovery, per-channel uniformity, spectral grain tests,
OOD prompts, trust-map A/B, harness self-tests) — run that for refiner-side
gates. This is the verifier split: base TeaCache here, refiner there.

Loads Pareto-optimal configurations from Phase 2, selects them by uniform
sampling across the error range (not just the knee), and measures real
wall-clock speedup + quality metrics against precomputed baselines.

Baselines are cached per (resolution, steps, prompt, seed) so every config
reuses the same baseline — ~45% reduction in total generations.

Supports multiple resolutions and step counts to verify generalization.

Usage:
    python -m tuning.validate --comfy-dir /path/to/ComfyUI \
        --pareto outputs/optimization/pareto_frontier.json \
        [--num-error-samples 8] [--quick] [--thorough] [--extra-sweep]
    python -m tuning.verify_refiner --comfy-dir /path/to/ComfyUI \
        --corrector models/corrector-30m-96000.safetensors   # refiner side
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from .config_types import (
    TeacacheConfig, OptimizationResult, ValidationResult, TuningConfig,
)
from .utils import (
    load_models, sample, get_diffusion_model,
    compute_quality_metrics, img_to_tensor, QualityMetrics,
    print_metrics_legend, print_schedule_estimate, print_speed_summary,
    RunSpec,
)
from .forward import teacache_anima_forward
from .prompt_loader import load_prompt_config, select_prompts, GenerationPromptSampler, resolve_generation
from .artist_tags import load_pool_for_config


# ═══════════════════════════════════════════════════════════════════════════
#  Config selection from Pareto frontier
# ═══════════════════════════════════════════════════════════════════════════

def select_validation_configs(
    pareto_results: List[OptimizationResult],
    num_error_samples: int,
    closest_per_sample: int,
    include_score_top: int,
) -> List[TeacacheConfig]:
    """Select configs uniformly across the Pareto frontier error range.

    Samples N points uniformly in accumulated_error space, takes the K
    closest Pareto configs per sample point, and includes the top M by
    score (knee configs).  Deduplicates by signal signature.
    """
    if not pareto_results:
        return []

    selected_cfgs: List[TeacacheConfig] = []
    seen: set = set()

    def _sig(r: OptimizationResult) -> tuple:
        c = r.config
        return (
            c.source, c.metric_type, c.mapping_type,
            round(c.rel_l1_thresh, 8), c.accumulation_type,
            c.step_schedule, c.block_mode,
            tuple(sorted(c.accumulation_params.items())),
            tuple(sorted(c.mapping_params.items())),
        )

    def _add(r: OptimizationResult) -> None:
        s = _sig(r)
        if s not in seen:
            seen.add(s)
            selected_cfgs.append(r.config)

    # ── Error-uniform samples ──────────────────────────────────────────
    pareto_by_error = sorted(pareto_results, key=lambda r: r.accumulated_error)
    min_err = pareto_by_error[0].accumulated_error
    max_err = pareto_by_error[-1].accumulated_error

    if max_err - min_err < 1e-12:
        by_score = sorted(pareto_results, key=lambda r: r.score, reverse=True)
        for r in by_score[:max(num_error_samples, include_score_top)]:
            _add(r)
        return selected_cfgs

    target_errors = np.linspace(min_err, max_err, num=num_error_samples)

    for target in target_errors:
        distances = sorted(
            [(abs(r.accumulated_error - target), r) for r in pareto_results],
            key=lambda x: x[0],
        )
        for _, r in distances[:closest_per_sample]:
            _add(r)

    # ── Top M by score (knee) ──────────────────────────────────────────
    by_score = sorted(pareto_results, key=lambda r: r.score, reverse=True)
    for r in by_score[:include_score_top]:
        _add(r)

    return selected_cfgs


# ═══════════════════════════════════════════════════════════════════════════
#  Baseline precomputation
# ═══════════════════════════════════════════════════════════════════════════

def precompute_baselines(
    unet, clip, vae,
    prompts: List[dict],
    seeds: List[int],
    resolutions: List[Tuple[int, int]],
    step_counts: List[int],
    tcfg: TuningConfig,
    sampler: GenerationPromptSampler,
) -> Dict[tuple, dict]:
    """Generate baselines once per (resolution, steps, prompt, seed) combo.

    Returns dict keyed by (width, height, steps, prompt_idx, seed), each
    value being {"image": PIL.Image, "time": float, "prompt": str,
    "negative": str, "artists": list}. The prompt for each key is drawn
    deterministically (per-generation variants + artist tags), so the
    teacache run under the same key uses the identical prompt.
    """
    baselines: Dict[tuple, dict] = {}
    total = len(resolutions) * len(step_counts) * len(prompts) * len(seeds)
    done = 0

    print(f"\n  Precomputing {total} baselines...")

    for w, h in resolutions:
        for steps in step_counts:
            for pi, pdata in enumerate(prompts):
                for seed in seeds:
                    key = (w, h, steps, pi, seed)
                    full_prompt, neg_prompt, artists = resolve_generation(
                        sampler, pdata["entry"], pi, seed, steps, w, h,
                    )
                    torch.cuda.empty_cache()
                    t0 = time.time()
                    img = sample(
                        unet, clip, vae, full_prompt,
                        seed=seed, steps=steps,
                        cfg=tcfg.sampling["cfg"],
                        sampler_name=tcfg.sampling["sampler"],
                        scheduler=tcfg.sampling["scheduler"],
                        width=w, height=h,
                        negative=neg_prompt,
                    )
                    dt = time.time() - t0
                    baselines[key] = {
                        "image": img, "time": dt,
                        "prompt": full_prompt, "negative": neg_prompt,
                        "artists": artists,
                    }
                    done += 1
                    artist_short = artists[0][:20] if artists else "none"
                    print(f"  [{done:>3d}/{total}] {w}x{h} s={steps:>2d} "
                          f"p={pi} seed={seed} art={artist_short}  {dt:.1f}s")

    return baselines


# ═══════════════════════════════════════════════════════════════════════════
#  Prompt loading
# ═══════════════════════════════════════════════════════════════════════════

def load_validation_prompts(
    tcfg: TuningConfig,
    num_prompts: int = None,
    num_seeds: int = None,
) -> Tuple[list, list, GenerationPromptSampler]:
    """Load selected prompt entries for validation.

    Prefix/negative variants and artist tags are NOT resolved here — they are
    drawn deterministically per generation (keyed by prompt/seed/steps/
    resolution), so every baseline and its teacache run share the same prompt
    string for the same key.

    Returns (prompts_list, seeds_list, sampler) where prompts_list contains
    dicts with an 'entry' key.
    """
    cfg = tcfg.validation
    prompt_config = load_prompt_config(
        str(Path(__file__).parent / cfg["prompts_file"])
    )
    count = num_prompts if num_prompts is not None else cfg.get("num_prompts", 2)
    entries = select_prompts(
        prompt_config,
        method=cfg.get("prompt_selection", "from_top"),
        count=count,
        tag_filter=cfg.get("prompt_tag_filter"),
    )
    prompts = [{"entry": e} for e in entries]

    seeds = list(cfg.get("seeds", [34635345]))
    if num_seeds is not None:
        seeds = seeds[:num_seeds]

    artist_pool = load_pool_for_config(tcfg)
    sampler = GenerationPromptSampler(prompt_config, artist_pool, tcfg.artist_tags)
    return prompts, seeds, sampler


# ═══════════════════════════════════════════════════════════════════════════
#  TeaCache helpers
# ═══════════════════════════════════════════════════════════════════════════

def _cleanup_patch(dm, unet):
    """Remove TeaCache patch and config from model."""
    to = unet.model_options.get("transformer_options", {})
    for k in list(to.keys()):
        if k.startswith("tc_"):
            del to[k]
    to.pop("enable_teacache", None)
    to.pop("rel_l1_thresh", None)
    to.pop("coefficients", None)
    to.pop("current_percent", None)
    unet.model_options.pop("model_function_wrapper", None)
    if hasattr(dm, "teacache_state"):
        delattr(dm, "teacache_state")


def run_single_teacache(
    unet, clip, vae,
    prompt: str, negative: str,
    cfg: TeacacheConfig,
    seed: int, steps: int,
    sampler: str, scheduler: str,
    width: int, height: int,
    cfg_val: float = 5.0,
    corrector: str = None,
    corrector_passes: int = 1,
    corrector_trust: float = 1.0,
    return_latent: bool = False,
) -> tuple:
    """Run one TeaCache generation. Returns
    (image_or_latent, wall_time_sec, total_steps, cached_steps, drift, slot_stats).

    total_steps is the number of denoising steps where TeaCache was active.
    cached_steps is the number of those steps where caching was used.
    drift is the skip-run length distribution (max run + histogram) tracked
    in the wrapper — no forward state needed (plan Task 7).
    slot_stats is the per-slot correction report from Mode B′ (diagnostic).
    """
    dm = get_diffusion_model(unet)
    if hasattr(dm, "teacache_state"):
        delattr(dm, "teacache_state")

    # Reset diagnostic counters for this generation
    dm._tc_diag_runs = 0
    dm._tc_diag_skips = 0

    original = dm._forward
    dm._forward = teacache_anima_forward.__get__(dm, dm.__class__)

    to = unet.model_options.setdefault("transformer_options", {})
    cfg.inject_into_transformer_options(to)
    to["enable_teacache"] = True

    # ── Mode B′ corrector (plan Task 7) ────────────────────────────────
    if corrector:
        from .corrector import prepare_corrector, CorrectorUNet2D, CorrectorConfig
        if corrector == "__zero_init__":
            # Sanity gate: fresh zero-init corrector ⇒ Mode B′ ≡ Mode A
            corr = CorrectorUNet2D(CorrectorConfig.for_size("tiny"))
            corr = corr.to(next(dm.parameters()).device).eval()
        else:
            corr = prepare_corrector(corrector, next(dm.parameters()).device)
        dm._tc_corrector = corr
        dm._tc_collect_stats = True
        dm._tc_corr_slot_stats = {}
        cfg.correction_mode = "latent_denoiser"
        cfg.refine_passes = max(1, int(corrector_passes))
        cfg.corrector_trust = float(corrector_trust)

    drift = {"max_skip_run": 0, "histogram": {}, "n_steps": 0}
    run_len = 0

    def tc_wrapper(model_function, kwargs):
        nonlocal run_len
        c = kwargs["c"]
        timestep = kwargs["timestep"]
        c_to = c.setdefault("transformer_options", {})
        sigmas = c_to.get("sample_sigmas")
        if sigmas is not None:
            matched = (sigmas == timestep[0]).nonzero()
            if len(matched) > 0:
                step_idx = matched[0].item()
            else:
                step_idx = 0
                for i in range(len(sigmas) - 1):
                    if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                        step_idx = i
                        break
            c_to["current_percent"] = step_idx / max(len(sigmas) - 1, 1)
            c_to["enable_teacache"] = (
                cfg.start_percent <= c_to["current_percent"] <= cfg.end_percent
            )
        out = model_function(kwargs["input"], timestep, **c)
        # Drift exposure (plan Task 7): skip-run lengths from the state the
        # forward already maintains — no forward changes needed.
        st = getattr(dm, "teacache_state", None)
        skipped = False
        if st:
            skipped = not any(s.get("should_calc", True) for s in st.values())
        if skipped:
            run_len += 1
        else:
            if run_len > 0:
                drift["histogram"][run_len] = drift["histogram"].get(run_len, 0) + 1
                drift["max_skip_run"] = max(drift["max_skip_run"], run_len)
            run_len = 0
        drift["n_steps"] += 1
        return out

    try:
        unet.set_model_unet_function_wrapper(tc_wrapper)
        t0 = time.time()
        img = sample(
            unet, clip, vae, prompt,
            seed=seed, steps=steps,
            width=width, height=height,
            cfg=cfg_val,
            sampler_name=sampler, scheduler=scheduler,
            negative=negative,
            return_latent=return_latent,
        )
        dt = time.time() - t0
        total_steps = getattr(dm, "_tc_diag_runs", 0)
        cached_steps = getattr(dm, "_tc_diag_skips", 0)
        if run_len > 0:
            drift["histogram"][run_len] = drift["histogram"].get(run_len, 0) + 1
            drift["max_skip_run"] = max(drift["max_skip_run"], run_len)
        slot_stats = dict(getattr(dm, "_tc_corr_slot_stats", {}))
    finally:
        dm._forward = original
        _cleanup_patch(dm, unet)
        if hasattr(dm, "_tc_corrector"):
            delattr(dm, "_tc_corrector")
        if hasattr(dm, "_tc_collect_stats"):
            delattr(dm, "_tc_collect_stats")

    return img, dt, total_steps, cached_steps, drift, slot_stats

# ═══════════════════════════════════════════════════════════════════════════
#  Validation core
# ═══════════════════════════════════════════════════════════════════════════

def validate_config(
    cfg: TeacacheConfig,
    unet, clip, vae,
    prompts: list,
    seeds: List[int],
    tcfg: TuningConfig,
    qm: QualityMetrics,
    baselines: Dict[tuple, dict],
    width: int,
    height: int,
    steps: int,
    prompt_sampler: GenerationPromptSampler,
    quiet: bool = False,
    corrector: str = None,
    corrector_passes: int = 1,
    corrector_trust: float = 1.0,
) -> tuple:
    """End-to-end validation of one configuration at a specific
    resolution and step count.

    Uses precomputed baselines keyed by
    (width, height, steps, prompt_idx, seed). The prompt for each key is
    re-drawn via the sampler, which must match the draw used when the
    baseline was generated.

    Returns (ValidationResult, drift_report, slot_stats) — drift_report is
    the skip-run length distribution (Mode A or B′), slot_stats the per-slot
    correction magnitudes (Mode B′ only; empty otherwise).
    """
    metric_names = qm.metric_names()
    all_scores: Dict[str, List[float]] = {k: [] for k in metric_names}
    all_speedups: List[float] = []
    all_times: List[float] = []
    all_skip_rates: List[float] = []
    sampler = tcfg.sampling["sampler"]
    scheduler = tcfg.sampling["scheduler"]
    cfg_val = tcfg.sampling["cfg"]
    drift_agg = {"max_skip_run": 0, "histogram": {}}
    slot_agg = {}

    for pi, pdata in enumerate(prompts):
        for seed in seeds:
            key = (width, height, steps, pi, seed)
            bl = baselines.get(key)
            if bl is None:
                print(f"  WARNING missing baseline for key={key}, skipping")
                continue

            full_prompt, neg_prompt, _ = resolve_generation(
                prompt_sampler, pdata["entry"], pi, seed, steps, width, height,
            )
            img_base = bl["image"]
            t_base = bl["time"]

            img_tc, t_tc, total_diag_steps, cached_diag_steps, drift, slot_stats = \
                run_single_teacache(
                    unet, clip, vae,
                    full_prompt, neg_prompt,
                    cfg,
                    seed=seed, steps=steps,
                    sampler=sampler, scheduler=scheduler,
                    width=width, height=height,
                    cfg_val=cfg_val,
                    corrector=corrector,
                    corrector_passes=corrector_passes,
                    corrector_trust=corrector_trust,
                )
            drift_agg["max_skip_run"] = max(drift_agg["max_skip_run"],
                                            drift.get("max_skip_run", 0))
            for k, v in drift.get("histogram", {}).items():
                drift_agg["histogram"][k] = drift_agg["histogram"].get(k, 0) + v
            for k, (s_dv, s_blend, n) in slot_stats.items():
                a = slot_agg.setdefault(k, [0.0, 0.0, 0])
                a[0] += s_dv
                a[1] += s_blend
                a[2] += n

            speedup = t_base / max(t_tc, 0.001)
            all_speedups.append(speedup)
            all_times.append(t_tc)

            if total_diag_steps > 0:
                skip_rate = cached_diag_steps / total_diag_steps
            else:
                skip_rate = 0.0
            all_skip_rates.append(round(skip_rate, 4))

            scores = qm.measure(img_tc, img_base)
            for k in metric_names:
                all_scores[k].append(scores.get(k, float("nan")))

            if not quiet:
                lpips = scores.get("lpips_alex", float("inf"))
                print(f"  p={pi} s={seed} thresh={cfg.rel_l1_thresh} "
                      f"speedup={speedup:.2f}x  LPIPS={lpips:.4f}  "
                      f"skip={skip_rate:.1%} ({cached_diag_steps}/{total_diag_steps})")

    n = len(all_speedups)
    if n == 0:
        return (
            ValidationResult(
                config=cfg, mean_speedup=1.0,
                mean_psnr=float("inf"), mean_ssim=1.0, mean_lpips=0.0,
                mean_time_sec=0.0, n_samples=0, mean_metrics={},
                actual_skip_rate=0.0, skip_rates=[],
            ),
            drift_agg,
            {k: [round(a[0] / max(a[2], 1), 6), round(a[1] / max(a[2], 1), 6)]
             for k, a in slot_agg.items()},
        )

    mean_metrics = {}
    for k in metric_names:
        vals = [v for v in all_scores[k] if v == v and v != float("inf")]
        mean_metrics[k] = round(sum(vals) / max(len(vals), 1), 4) if vals else float("nan")

    mean_skip = round(sum(all_skip_rates) / len(all_skip_rates), 4) if all_skip_rates else 0.0

    result = ValidationResult(
        config=cfg,
        mean_speedup=round(sum(all_speedups) / n, 3),
        mean_psnr=round(mean_metrics.get("psnr", float("inf")), 2),
        mean_ssim=round(mean_metrics.get("ssim", 1.0), 4),
        mean_lpips=round(mean_metrics.get("lpips_alex", 0.0), 4),
        mean_time_sec=round(sum(all_times) / n, 2),
        n_samples=n,
        mean_metrics=mean_metrics,
        actual_skip_rate=mean_skip,
        skip_rates=all_skip_rates,
    )
    return (result, drift_agg,
            {k: [round(a[0] / max(a[2], 1), 6), round(a[1] / max(a[2], 1), 6)]
             for k, a in slot_agg.items()})


def sweep_thresholds(
    base_cfg: TeacacheConfig,
    unet, clip, vae,
    prompts: list,
    seeds: List[int],
    tcfg: TuningConfig,
    qm: QualityMetrics,
    thresh_values: List[float],
    baselines: Dict[tuple, dict],
    width: int,
    height: int,
    steps: int,
    prompt_sampler: GenerationPromptSampler,
) -> List[ValidationResult]:
    """Sweep threshold values for one base config at one resolution/steps."""
    results = []
    for thresh in thresh_values:
        cfg = TeacacheConfig.from_dict(base_cfg.to_dict())
        cfg.rel_l1_thresh = thresh
        print(f"\n  Threshold sweep: thresh={thresh}  ({width}x{height} @ {steps}s)")
        result, _drift, _slots = validate_config(
            cfg, unet, clip, vae, prompts, seeds, tcfg, qm,
            baselines, width, height, steps, prompt_sampler,
        )
        results.append(result)
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Mode B′ gates (plan Task 7)
# ═══════════════════════════════════════════════════════════════════════════

def run_sanity_gate(cfg: TeacacheConfig, unet, clip, vae, prompts, seeds, tcfg,
                    width, height, steps, prompt_sampler) -> dict:
    """Zero-init Mode B′ must reproduce Mode A EXACTLY (bitwise, plan Task 7).

    Compares the final sampler latents of a Mode A run and a Mode B′ run with
    a fresh zero-init corrector, under deterministic cuDNN settings.
    """
    import torch.backends.cudnn as cudnn
    old_det, old_bench = cudnn.deterministic, cudnn.benchmark
    cudnn.deterministic, cudnn.benchmark = True, False
    try:
        lat_a, lat_b, ok = None, None, False
        for pi, pdata in enumerate(prompts[:1]):
            for seed in seeds[:1]:
                full_prompt, neg_prompt, _ = resolve_generation(
                    prompt_sampler, pdata["entry"], pi, seed, steps, width, height,
                )
                sampler_name = tcfg.sampling["sampler"]
                scheduler = tcfg.sampling["scheduler"]
                cfg_val = tcfg.sampling["cfg"]
                lat_a, _t, _ts, _cs, _d, _s = run_single_teacache(
                    unet, clip, vae, full_prompt, neg_prompt,
                    TeacacheConfig.from_dict(cfg.to_dict()),
                    seed=seed, steps=steps, sampler=sampler_name, scheduler=scheduler,
                    width=width, height=height, cfg_val=cfg_val, return_latent=True,
                )
                cfg_b = TeacacheConfig.from_dict(cfg.to_dict())
                lat_b, _t, _ts, _cs, _d, _s = run_single_teacache(
                    unet, clip, vae, full_prompt, neg_prompt, cfg_b,
                    seed=seed, steps=steps, sampler=sampler_name, scheduler=scheduler,
                    width=width, height=height, cfg_val=cfg_val, return_latent=True,
                    corrector="__zero_init__",
                )
        if lat_a is not None and lat_b is not None:
            lat_a = lat_a[0] if isinstance(lat_a, tuple) else lat_a
            lat_b = lat_b[0] if isinstance(lat_b, tuple) else lat_b
            ok = torch.equal(lat_a, lat_b)
        return {"exact_equality": ok, "shape": list(lat_a.shape) if lat_a is not None else None}
    finally:
        cudnn.deterministic, cudnn.benchmark = old_det, old_bench


def shipping_gate(ab_rows: List[dict]) -> dict:
    """Plan Task 7 acceptance: Mode B′ ≥ Mode A LPIPS in ≥70% of (resolution,
    steps) cells AND ≥5% better on the aggressive half of the threshold curve."""
    from collections import defaultdict
    cells = defaultdict(lambda: {"better": 0, "worse": 0, "samples": []})
    for row in ab_rows:
        key = (row["resolution"][0], row["resolution"][1], row["steps"])
        lp_b = row["lpips_b"]
        lp_a = row["lpips_a"]
        if lp_a != lp_a or lp_b != lp_b:
            continue
        cells[key]["samples"].append((lp_a, lp_b))
        if lp_b <= lp_a:
            cells[key]["better"] += 1
        else:
            cells[key]["worse"] += 1
    cell_pass = 0
    cell_details = {}
    for key, c in sorted(cells.items()):
        n = c["better"] + c["worse"]
        frac = c["better"] / max(n, 1)
        cell_details[str(key)] = {"better": c["better"], "worse": c["worse"],
                                  "frac_better": round(frac, 3)}
        if frac >= 0.5:
            cell_pass += 1
    cells_70 = cell_pass / max(len(cells), 1) >= 0.7
    # Aggressive half: configs above the median threshold
    threshes = sorted(r["thresh"] for r in ab_rows)
    median_t = threshes[len(threshes) // 2] if threshes else 0.0
    agg = [r for r in ab_rows if r["thresh"] >= median_t and r["lpips_a"] == r["lpips_a"]
           and r["lpips_b"] == r["lpips_b"]]
    agg_improvement = None
    if agg:
        ma = sum(r["lpips_a"] for r in agg) / len(agg)
        mb = sum(r["lpips_b"] for r in agg) / len(agg)
        agg_improvement = (ma - mb) / max(ma, 1e-9)
    aggressive_5 = agg_improvement is not None and agg_improvement >= 0.05
    passed = cells_70 and aggressive_5
    return {
        "cells_70pct_better_or_equal": cells_70,
        "cell_details": cell_details,
        "aggressive_half_improvement": (round(agg_improvement, 4)
                                        if agg_improvement is not None else None),
        "aggressive_half_5pct": aggressive_5,
        "passed": passed,
        "note": ("ship Mode B′ as default" if passed
                 else "keep Mode B′ opt-in experimental"),
    }

# ═══════════════════════════════════════════════════════════════════════════
#  CLI argument parsing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _parse_resolutions(raw: str) -> List[Tuple[int, int]]:
    """Parse '512x512,1024x1024,1024x512' into [(512,512), ...]."""
    if not raw:
        return []
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            w, h = part.split("x")
            out.append((int(w), int(h)))
        except ValueError:
            print(f"  WARNING invalid resolution '{part}', expected WxH; skipping")
    return out


def _parse_int_list(raw: str) -> List[int]:
    """Parse '20,30,40' into [20, 30, 40]."""
    if not raw:
        return []
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            print(f"  WARNING invalid int '{part}', skipping")
    return out


def _parse_float_list(raw: str) -> List[float]:
    """Parse '0.01,0.05,0.10' into [0.01, 0.05, 0.10]."""
    if not raw:
        return []
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            print(f"  WARNING invalid float '{part}', skipping")
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Result formatting helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_flat_results(
    all_results: List[tuple],
) -> List[dict]:
    """Convert (result, width, height, steps[, mode]) tuples to JSON dicts."""
    out = []
    for item in all_results:
        result, w, h, steps = item[:4]
        mode = item[4] if len(item) > 4 else "A"
        d = result.to_dict()
        d["resolution"] = [w, h]
        d["steps"] = steps
        d["mode"] = mode
        out.append(d)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TeaCache Validator")
    parser.add_argument("--comfy-dir", required=True,
                        help="Path to ComfyUI installation")
    parser.add_argument("--pareto", required=True,
                        help="Path to pareto_frontier.json from Phase 2")
    parser.add_argument("--config", default=None,
                        help="Path to config.json")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3],
                        help="Metric tier (1=fastest, 3=most comprehensive)")

    # ── Presets ────────────────────────────────────────────────────────
    parser.add_argument("--quick", action="store_true",
                        help="1 prompt, 1 seed, 512^2+1024^2, 30 steps, N=6")
    parser.add_argument("--thorough", action="store_true",
                        help="2 prompts, 2 seeds, all resolutions/steps, N=12")

    # ── Config overrides ───────────────────────────────────────────────
    parser.add_argument("--num-error-samples", type=int, default=None,
                        help="Uniform error sample count across Pareto (default: 8)")
    parser.add_argument("--error-samples-span", type=int, default=None,
                        help="K-closest configs per error sample (default: 1)")
    parser.add_argument("--include-score-top", type=int, default=None,
                        help="Also top-M by score (knee, default: 2)")
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--resolutions", type=str, default=None,
                        help="Comma-separated WxH pairs, e.g. '512x512,1024x1024'")
    parser.add_argument("--step-counts", type=str, default=None,
                        help="Comma-separated step counts, e.g. '20,30,40'")
    parser.add_argument("--extra-sweep", action="store_true",
                        help="Run threshold sweeps at key error points")
    parser.add_argument("--extra-sweep-errors", type=str, default=None,
                        help="Comma-separated error targets, e.g. '0.01,0.05,0.10'")

    # ── Mode B′ latent corrector (plan Task 7) ─────────────────────────
    parser.add_argument("--corrector", default=None,
                        help="Path to a corrector .safetensors. When set, every "
                             "validated config also runs in Mode B′ and an A/B "
                             "report (Mode A vs Mode B′), per-slot quality, "
                             "drift exposure, the sanity gate and the shipping "
                             "gate are produced.")
    parser.add_argument("--corrector-passes", type=int, default=1,
                        help="K for Mode B′ validation runs (default 1)")
    parser.add_argument("--corrector-trust", type=float, default=1.0,
                        help="Correction blend factor for Mode B′ runs (default 1.0)")

    args = parser.parse_args()

    # ── Load config ────────────────────────────────────────────────────
    if args.config is None:
        args.config = str(Path(__file__).parent / "config.json")
    tcfg = TuningConfig.load(args.config)

    out_dir = Path(tcfg.output_dir) / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve parameters (config defaults → preset overrides → CLI overrides) ──
    vcfg = tcfg.validation

    num_error_samples = vcfg.get("num_error_samples", 8)
    error_samples_span = vcfg.get("error_samples_span", 1)
    include_score_top = vcfg.get("include_score_top", 2)
    num_prompts = vcfg.get("num_prompts", 2)
    num_seeds = 1  # default 1 seed
    resolutions = [tuple(r) for r in vcfg.get("resolutions", [[512, 512], [1024, 1024], [1024, 512]])]
    step_counts = list(vcfg.get("step_counts", [20, 30, 40]))
    do_extra_sweep = args.extra_sweep
    extra_sweep_errors = list(vcfg.get("extra_threshold_sweep_errors", [0.01, 0.03, 0.05, 0.10]))

    # ── Presets ────────────────────────────────────────────────────────
    if args.quick:
        num_error_samples = 6
        num_prompts = 1
        num_seeds = 1
        resolutions = [(512, 512), (1024, 1024)]
        step_counts = [30]
    elif args.thorough:
        num_error_samples = 12
        num_prompts = 2
        num_seeds = 2
        # keep full resolutions and step_counts from config

    # ── CLI overrides ──────────────────────────────────────────────────
    if args.num_error_samples is not None:
        num_error_samples = args.num_error_samples
    if args.error_samples_span is not None:
        error_samples_span = args.error_samples_span
    if args.include_score_top is not None:
        include_score_top = args.include_score_top
    if args.num_prompts is not None:
        num_prompts = args.num_prompts
    if args.num_seeds is not None:
        num_seeds = args.num_seeds
    if args.resolutions is not None:
        resolutions = _parse_resolutions(args.resolutions) or resolutions
    if args.step_counts is not None:
        step_counts = _parse_int_list(args.step_counts) or step_counts
    if args.extra_sweep_errors is not None:
        extra_sweep_errors = _parse_float_list(args.extra_sweep_errors) or extra_sweep_errors

    # ── Header ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("  TeaCache Validator - Phase 3")
    print("=" * 60)
    print(f"  Pareto:          {args.pareto}")
    print(f"  Tier:            {args.tier}")
    print(f"  Error samples:   {num_error_samples}  (span={error_samples_span})")
    print(f"  Score top incl:  {include_score_top}")
    print(f"  Prompts:         {num_prompts}  Seeds: {num_seeds}")
    print(f"  Resolutions:     {resolutions}")
    print(f"  Step counts:     {step_counts}")
    if do_extra_sweep:
        print(f"  Extra sweep at:  {extra_sweep_errors}")
    if args.corrector:
        print(f"  Corrector:       {args.corrector}  (Mode B′ A/B, K={args.corrector_passes}, "
              f"trust={args.corrector_trust})")
    print(f"  Output:          {out_dir}")
    total_combo = (len(resolutions) * len(step_counts) * num_prompts * num_seeds)
    print(f"  Baseline images: {total_combo}")
    print("=" * 60)

    # ── Metrics ────────────────────────────────────────────────────────
    qm = QualityMetrics(tier=args.tier)
    if not qm.available:
        print("\n  ERROR: pyiqa not found. Install: pip install -r tuning/requirements.txt")
        sys.exit(1)
    metric_names = qm.metric_names()
    print(f"  Metrics:  {', '.join(metric_names)}")

    # ── Load Pareto frontier ───────────────────────────────────────────
    pareto_data = json.loads(Path(args.pareto).read_text())
    pareto_results = [OptimizationResult.from_dict(d) for d in pareto_data]
    print(f"\n[load] {len(pareto_results)} Pareto-optimal configs")

    # ── Select configs ─────────────────────────────────────────────────
    selected_cfgs = select_validation_configs(
        pareto_results, num_error_samples, error_samples_span, include_score_top,
    )
    print(f"[select] {len(selected_cfgs)} configs to validate\n")

    if not selected_cfgs:
        print("  No configs selected — exiting.")
        return

    # ── Load models ────────────────────────────────────────────────────
    unet, clip, vae = load_models(
        tcfg.comfy_dir, tcfg.model_name,
        tcfg.clip_name, tcfg.clip_type, tcfg.vae_name,
    )

    # ── Load prompts ───────────────────────────────────────────────────
    prompts, seeds, prompt_sampler = load_validation_prompts(tcfg, num_prompts, num_seeds)
    if prompt_sampler.artist_pool is not None:
        acfg = tcfg.artist_tags
        print(f"[prompts] {len(prompts)} prompts x {len(seeds)} seeds "
              f"(artist tags ON: {len(prompt_sampler.artist_pool.artists)} tags, "
              f"mode={acfg.get('weight_mode', 'relative')}, "
              f"max_tags={acfg.get('max_tags', 1)})\n")
    else:
        print(f"[prompts] {len(prompts)} prompts x {len(seeds)} seeds "
              f"(artist tags OFF)\n")

    # ── Precompute baselines ───────────────────────────────────────────
    total_per_combo = len(prompts) * len(seeds)
    total_baseline_images = len(resolutions) * len(step_counts) * total_per_combo

    total_bsl_iterations = 0
    for w, h in resolutions:
        for s in step_counts:
            total_bsl_iterations += s * total_per_combo

    # Exact schedules (deterministic run order) for the pre-run estimates.
    baseline_schedule = [
        RunSpec(width=w, height=h, steps=s,
                sampler=tcfg.sampling["sampler"],
                scheduler=tcfg.sampling["scheduler"],
                cfg=tcfg.sampling["cfg"])
        for w, h in resolutions for s in step_counts for _ in range(total_per_combo)
    ]
    val_schedule = [
        RunSpec(width=w, height=h, steps=s,
                sampler=tcfg.sampling["sampler"],
                scheduler=tcfg.sampling["scheduler"],
                cfg=tcfg.sampling["cfg"])
        for _cfg in selected_cfgs for w, h in resolutions for s in step_counts
        for _ in range(total_per_combo)
    ]

    print_schedule_estimate(
        label="Baseline generation schedule",
        total_generations=total_baseline_images,
        extra_lines=[
            f"Resolutions:    {resolutions}",
            f"Step counts:    {step_counts}",
            f"Prompts x seeds: {len(prompts)} x {len(seeds)} = {total_per_combo}",
        ],
        schedule=baseline_schedule,
    )

    bsl_start = time.time()
    baselines = precompute_baselines(
        unet, clip, vae, prompts, seeds, resolutions, step_counts, tcfg,
        prompt_sampler,
    )
    bsl_elapsed = time.time() - bsl_start

    # ── Validate each (config × resolution × steps) ────────────────────
    all_results: List[tuple] = []  # (ValidationResult, width, height, steps)
    total_validations = len(selected_cfgs) * len(resolutions) * len(step_counts)
    total_val_images = total_validations * total_per_combo

    val_iterations_per_combo = 0
    for w, h in resolutions:
        for s in step_counts:
            val_iterations_per_combo += s * total_per_combo
    total_val_iterations = val_iterations_per_combo * len(selected_cfgs)

    print_schedule_estimate(
        label="Validation schedule",
        total_generations=total_val_images,
        extra_lines=[
            f"Configs:        {len(selected_cfgs)}",
            f"Resolutions:    {resolutions}",
            f"Step counts:    {step_counts}",
            f"Prompts x seeds: {len(prompts)} x {len(seeds)} = {total_per_combo}",
        ],
        schedule=val_schedule,
    )

    val_start = time.time()
    done = 0
    ab_rows: List[dict] = []
    drift_reports: List[dict] = []

    for cfg in selected_cfgs:
        for w, h in resolutions:
            for steps in step_counts:
                done += 1
                print(f"\n[{done}/{total_validations}] "
                      f"{cfg.source} {cfg.metric_type} {cfg.mapping_type} "
                      f"{w}x{h} @ {steps}s  thresh={cfg.rel_l1_thresh}")

                result, drift, _slots = validate_config(
                    cfg, unet, clip, vae, prompts, seeds, tcfg, qm,
                    baselines, w, h, steps, prompt_sampler,
                )
                all_results.append((result, w, h, steps, "A"))
                drift_reports.append({"mode": "A", "thresh": cfg.rel_l1_thresh,
                                      "resolution": [w, h], "steps": steps,
                                      **drift})
                if args.corrector:
                    print(f"  ── Mode B′ (corrector) ──")
                    result_b, drift_b, slot_stats = validate_config(
                        cfg, unet, clip, vae, prompts, seeds, tcfg, qm,
                        baselines, w, h, steps, prompt_sampler,
                        corrector=args.corrector,
                        corrector_passes=args.corrector_passes,
                        corrector_trust=args.corrector_trust,
                    )
                    all_results.append((result_b, w, h, steps, "B"))
                    drift_reports.append({"mode": "B", "thresh": cfg.rel_l1_thresh,
                                          "resolution": [w, h], "steps": steps,
                                          **drift_b})
                    ab_rows.append({
                        "mode": "B",
                        "thresh": cfg.rel_l1_thresh,
                        "resolution": [w, h], "steps": steps,
                        "lpips_a": result.mean_lpips,
                        "lpips_b": result_b.mean_lpips,
                        "skip_a": result.actual_skip_rate,
                        "skip_b": result_b.actual_skip_rate,
                        "speedup_a": result.mean_speedup,
                        "speedup_b": result_b.mean_speedup,
                        "slot_stats": slot_stats,
                    })

    val_elapsed = time.time() - val_start

    # ── Mode B′ sanity gate + shipping gate (plan Task 7) ──────────────
    gate_report = {}
    if args.corrector:
        print("\n  ── Sanity gate: zero-init Mode B′ ≡ Mode A ──")
        first = selected_cfgs[0]
        sw, sh = resolutions[0]
        ssteps = step_counts[0]
        sanity = run_sanity_gate(first, unet, clip, vae, prompts, seeds, tcfg,
                                 sw, sh, ssteps, prompt_sampler)
        print(f"  exact equality: {sanity['exact_equality']}")
        print("\n  ── Shipping gate (plan Task 7) ──")
        ship = shipping_gate(ab_rows)
        for k, v in ship.items():
            if k != "cell_details":
                print(f"  {k}: {v}")
        gate_report = {"sanity": sanity, "shipping": ship}

    sweep_gens = 0
    sweep_iters = 0

    # ── Extra threshold sweeps (at default resolution) ─────────────────
    if do_extra_sweep and extra_sweep_errors:
        sweep_threshes = vcfg.get("extra_threshold_sweep", [
            0.02, 0.04, 0.06, 0.07, 0.08, 0.10, 0.15,
            0.20, 0.30, 0.50, 0.70, 1.0, 2.0, 5.0,
        ])
        # Sweep at first resolution x first step count
        sw, sh = resolutions[0]
        ssteps = step_counts[0]
        pareto_by_error = sorted(pareto_results,
                                 key=lambda r: r.accumulated_error)

        sweep_start = time.time()
        for target_err in extra_sweep_errors:
            nearest = min(pareto_by_error,
                          key=lambda r: abs(r.accumulated_error - target_err))
            print(f"\n  Extra sweep at error~{target_err:.4f} "
                  f"(closest err={nearest.accumulated_error:.4f}, "
                  f"speedup~{nearest.estimated_speedup:.2f}x)")
            sweep_results = sweep_thresholds(
                nearest.config, unet, clip, vae,
                prompts, seeds, tcfg, qm,
                sweep_threshes,
                baselines, sw, sh, ssteps, prompt_sampler,
            )
            for r in sweep_results:
                all_results.append((r, sw, sh, ssteps))
            sweep_gens += len(sweep_threshes) * total_per_combo
            sweep_iters += len(sweep_threshes) * ssteps * total_per_combo
        sweep_elapsed = time.time() - sweep_start
        val_elapsed += sweep_elapsed
        total_val_images += sweep_gens
        total_val_iterations += sweep_iters

    # ── Save results ───────────────────────────────────────────────────
    flat = _build_flat_results(all_results)
    (out_dir / "validation_results.json").write_text(
        json.dumps(flat, indent=2)
    )
    if drift_reports:
        (out_dir / "validation_drift.json").write_text(
            json.dumps(drift_reports, indent=2)
        )
    if gate_report:
        (out_dir / "validation_corrector_gates.json").write_text(
            json.dumps(gate_report, indent=2)
        )

    # ── Final speed summary across all GPU work ────────────────────────
    total_wall = bsl_elapsed + val_elapsed
    total_images = total_baseline_images + total_val_images
    total_iterations = total_bsl_iterations + total_val_iterations

    print_speed_summary(
        label="Validation run complete",
        total_generations=total_images,
        total_iterations=total_iterations,
        wall_seconds=total_wall,
    )
    print(f"  Baseline phase:  {int(bsl_elapsed // 60)}m {int(bsl_elapsed % 60)}s  ({total_baseline_images} images)")
    print(f"  Validation phase:{int(val_elapsed // 60)}m {int(val_elapsed % 60)}s  ({total_val_images} images)")
    if sweep_gens:
        print(f"    incl. sweeps:  {sweep_gens} sweep images")
    print(f"  {'─' * 56}")
    print(f"  Results saved to: {out_dir}/validation_results.json")

    # ── Print summary tables grouped by (resolution, steps) ────────────
    print(f"\n{'=' * 60}")
    print(f"  Validation Results")
    print(f"{'=' * 60}")
    print_metrics_legend()

    short_metrics = {
        "psnr": "PSNR", "ssim": "SSIM", "lpips_alex": "LPIPSa",
        "lpips_vgg": "LPIPSv", "dists": "DISTS", "ms_ssim": "MSSIM",
        "fsim": "FSIM", "vif": "VIF", "gmsd": "GMSD",
        "nlpd": "NLPD", "pieapp": "PieAPP", "vsi": "VSI",
    }
    display_names = [short_metrics.get(n, n[:6]) for n in metric_names]

    # Group by (resolution, steps)
    from collections import defaultdict
    groups: Dict[tuple, List[ValidationResult]] = defaultdict(list)
    for item in all_results:
        result, w, h, steps = item[:4]
        groups[(w, h, steps)].append(result)

    for (w, h, steps), group_results in sorted(groups.items()):
        n_configs = len(group_results)
        primary = [r for r in group_results
                   if not hasattr(r.config, '_is_sweep')]
        print(f"\n  {w}x{h} @ {steps} steps  ({n_configs} results)")
        header = (f"  {'#':>3} | {'source':<20} | {'speed':>6} | {'skip':>6} | "
                  + " | ".join(f"{d:>7}" for d in display_names))
        print(header)
        print(f"  {'-' * len(header)}")

        for i, r in enumerate(primary):
            c = r.config
            vals = " | ".join(
                f"{r.mean_metrics.get(n, float('nan')):>7.4f}"
                for n in metric_names
            )
            print(f"  {i+1:>3} | {c.source:<20} | {r.mean_speedup:>5.2f}x | "
                  f"{r.actual_skip_rate:>5.1%} | {vals}")

        # Best by LPIPS < 0.05
        good = [r for r in group_results if not (r.mean_lpips >= 0.05)]
        if good:
            best = max(good, key=lambda r: r.mean_speedup)
            print(f"  * Recommended (LPIPS<0.05): "
                  f"thresh={best.config.rel_l1_thresh} "
                  f"speedup={best.mean_speedup}x "
                  f"LPIPS={best.mean_lpips}")
        else:
            best = min(group_results,
                       key=lambda r: r.mean_lpips if r.mean_lpips != 0 else float('inf'))
            print(f"  * Best quality (no LPIPS<0.05): "
                  f"thresh={best.config.rel_l1_thresh} "
                  f"speedup={best.mean_speedup}x "
                  f"LPIPS={best.mean_lpips}")

    # ── Mode B′ A/B report (plan Task 7) ───────────────────────────────
    if ab_rows:
        from collections import defaultdict
        print(f"\n{'=' * 60}")
        print(f"  Mode A vs Mode B′ — A/B report")
        print(f"{'=' * 60}")
        ab_groups = defaultdict(list)
        for row in ab_rows:
            ab_groups[(tuple(row["resolution"]), row["steps"])].append(row)
        for (res, st), rows in sorted(ab_groups.items()):
            print(f"\n  {res[0]}x{res[1]} @ {st} steps  ({len(rows)} configs)")
            print(f"  {'thresh':>7} | {'LPIPS A':>8} | {'LPIPS B':>8} | {'ΔLPIPS':>8} | "
                  f"{'skip A':>7} | {'skip B':>7} | {'slot stats'}")
            for r in sorted(rows, key=lambda x: x["thresh"]):
                d = r["lpips_b"] - r["lpips_a"]
                slots = ", ".join(f"{k}: Δv̂={v[0]:.5f}" for k, v in sorted(r["slot_stats"].items()))
                print(f"  {r['thresh']:>7.4f} | {r['lpips_a']:>8.4f} | {r['lpips_b']:>8.4f} | "
                      f"{d:>+8.4f} | {r['skip_a']:>6.1%} | {r['skip_b']:>6.1%} | {slots}")

    print(f"\n  Results saved to: {out_dir}/validation_results.json")


if __name__ == "__main__":
    main()
