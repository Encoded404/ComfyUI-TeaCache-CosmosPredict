#!/usr/bin/env python3
"""Phase 3: End-to-end validation of top configurations.

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
)
from .forward import teacache_anima_forward
from .prompt_loader import load_prompt_config, select_prompts, resolve_prompt


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
) -> Dict[tuple, dict]:
    """Generate baselines once per (resolution, steps, prompt, seed) combo.

    Returns dict keyed by (width, height, steps, prompt_idx, seed), each
    value being {"image": PIL.Image, "time": float}.
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
                    torch.cuda.empty_cache()
                    t0 = time.time()
                    img = sample(
                        unet, clip, vae, pdata["prompt"],
                        seed=seed, steps=steps,
                        cfg=tcfg.sampling["cfg"],
                        sampler_name=tcfg.sampling["sampler"],
                        scheduler=tcfg.sampling["scheduler"],
                        width=w, height=h,
                        negative=pdata["negative"],
                    )
                    dt = time.time() - t0
                    baselines[key] = {"image": img, "time": dt}
                    done += 1
                    print(f"  [{done:>3d}/{total}] {w}x{h} s={steps:>2d} "
                          f"p={pi} seed={seed}  {dt:.1f}s")

    return baselines


# ═══════════════════════════════════════════════════════════════════════════
#  Prompt loading
# ═══════════════════════════════════════════════════════════════════════════

def load_validation_prompts(
    tcfg: TuningConfig,
    num_prompts: int = None,
    num_seeds: int = None,
) -> Tuple[list, list]:
    """Load and resolve prompts for validation.

    Returns (prompts_list, seeds_list) where prompts_list contains dicts
    with 'prompt', 'negative', and 'entry' keys.
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
    resolved = []
    for i, entry in enumerate(entries):
        full, neg = resolve_prompt(
            prompt_config, entry,
            prefix_variant_idx=i % max(len(prompt_config.prefix_variants), 1),
            negative_variant_idx=i % max(len(prompt_config.negative_variants), 1),
        )
        resolved.append({"prompt": full, "negative": neg, "entry": entry})

    seeds = list(cfg.get("seeds", [34635345]))
    if num_seeds is not None:
        seeds = seeds[:num_seeds]

    return resolved, seeds


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
) -> tuple:
    """Run one TeaCache generation. Returns (image, wall_time_sec, total_steps, cached_steps).

    total_steps is the number of denoising steps where TeaCache was active.
    cached_steps is the number of those steps where caching was used.
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

    def tc_wrapper(model_function, kwargs):
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
        return model_function(kwargs["input"], timestep, **c)

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
        )
        dt = time.time() - t0
        total_steps = getattr(dm, "_tc_diag_runs", 0)
        cached_steps = getattr(dm, "_tc_diag_skips", 0)
    finally:
        dm._forward = original
        _cleanup_patch(dm, unet)

    return img, dt, total_steps, cached_steps

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
    quiet: bool = False,
) -> ValidationResult:
    """End-to-end validation of one configuration at a specific
    resolution and step count.

    Uses precomputed baselines keyed by
    (width, height, steps, prompt_idx, seed).
    """
    metric_names = qm.metric_names()
    all_scores: Dict[str, List[float]] = {k: [] for k in metric_names}
    all_speedups: List[float] = []
    all_times: List[float] = []
    all_skip_rates: List[float] = []
    sampler = tcfg.sampling["sampler"]
    scheduler = tcfg.sampling["scheduler"]
    cfg_val = tcfg.sampling["cfg"]

    for pi, pdata in enumerate(prompts):
        for seed in seeds:
            key = (width, height, steps, pi, seed)
            bl = baselines.get(key)
            if bl is None:
                print(f"  WARNING missing baseline for key={key}, skipping")
                continue

            img_base = bl["image"]
            t_base = bl["time"]

            img_tc, t_tc, total_diag_steps, cached_diag_steps = run_single_teacache(
                unet, clip, vae,
                pdata["prompt"], pdata["negative"],
                cfg,
                seed=seed, steps=steps,
                sampler=sampler, scheduler=scheduler,
                width=width, height=height,
                cfg_val=cfg_val,
            )

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
        return ValidationResult(
            config=cfg, mean_speedup=1.0,
            mean_psnr=float("inf"), mean_ssim=1.0, mean_lpips=0.0,
            mean_time_sec=0.0, n_samples=0, mean_metrics={},
            actual_skip_rate=0.0, skip_rates=[],
        )

    mean_metrics = {}
    for k in metric_names:
        vals = [v for v in all_scores[k] if v == v and v != float("inf")]
        mean_metrics[k] = round(sum(vals) / max(len(vals), 1), 4) if vals else float("nan")

    mean_skip = round(sum(all_skip_rates) / len(all_skip_rates), 4) if all_skip_rates else 0.0

    return ValidationResult(
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
) -> List[ValidationResult]:
    """Sweep threshold values for one base config at one resolution/steps."""
    results = []
    for thresh in thresh_values:
        cfg = TeacacheConfig.from_dict(base_cfg.to_dict())
        cfg.rel_l1_thresh = thresh
        print(f"\n  Threshold sweep: thresh={thresh}  ({width}x{height} @ {steps}s)")
        result = validate_config(
            cfg, unet, clip, vae, prompts, seeds, tcfg, qm,
            baselines, width, height, steps,
        )
        results.append(result)
    return results

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
    """Convert (result, width, height, steps) tuples to JSON-serializable dicts."""
    out = []
    for result, w, h, steps in all_results:
        d = result.to_dict()
        d["resolution"] = [w, h]
        d["steps"] = steps
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
    prompts, seeds = load_validation_prompts(tcfg, num_prompts, num_seeds)
    print(f"[prompts] {len(prompts)} prompts x {len(seeds)} seeds\n")

    # ── Precompute baselines ───────────────────────────────────────────
    total_per_combo = len(prompts) * len(seeds)
    total_baseline_images = len(resolutions) * len(step_counts) * total_per_combo

    total_bsl_iterations = 0
    total_bsl_pixel_steps = 0.0
    for w, h in resolutions:
        for s in step_counts:
            total_bsl_iterations += s * total_per_combo
            total_bsl_pixel_steps += (w * h) * s * total_per_combo
    avg_bsl_steps = total_bsl_iterations / total_baseline_images if total_baseline_images > 0 else 1.0
    est_res = int((total_bsl_pixel_steps / max(total_bsl_iterations, 1)) ** 0.5)

    print_schedule_estimate(
        label="Baseline generation schedule",
        total_generations=total_baseline_images,
        avg_steps=avg_bsl_steps,
        width=est_res,
        height=est_res,
        extra_lines=[
            f"Resolutions:    {resolutions}",
            f"Step counts:    {step_counts}",
            f"Prompts x seeds: {len(prompts)} x {len(seeds)} = {total_per_combo}",
        ],
    )

    bsl_start = time.time()
    baselines = precompute_baselines(
        unet, clip, vae, prompts, seeds, resolutions, step_counts, tcfg,
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
        avg_steps=avg_bsl_steps,
        width=est_res,
        height=est_res,
        extra_lines=[
            f"Configs:        {len(selected_cfgs)}",
            f"Resolutions:    {resolutions}",
            f"Step counts:    {step_counts}",
            f"Prompts x seeds: {len(prompts)} x {len(seeds)} = {total_per_combo}",
        ],
    )

    val_start = time.time()
    done = 0

    for cfg in selected_cfgs:
        for w, h in resolutions:
            for steps in step_counts:
                done += 1
                print(f"\n[{done}/{total_validations}] "
                      f"{cfg.source} {cfg.metric_type} {cfg.mapping_type} "
                      f"{w}x{h} @ {steps}s  thresh={cfg.rel_l1_thresh}")

                result = validate_config(
                    cfg, unet, clip, vae, prompts, seeds, tcfg, qm,
                    baselines, w, h, steps,
                )
                all_results.append((result, w, h, steps))

    val_elapsed = time.time() - val_start
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
                baselines, sw, sh, ssteps,
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
    for result, w, h, steps in all_results:
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

    print(f"\n  Results saved to: {out_dir}/validation_results.json")


if __name__ == "__main__":
    main()
