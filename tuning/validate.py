#!/usr/bin/env python3
"""Phase 3: End-to-end validation of top configurations.

Loads the top-K configurations from Phase 2, runs them through the actual
TeaCache forward function, and measures real wall-clock speedup + quality
metrics (PSNR/SSIM/LPIPS) against a baseline.

This is the only phase that requires the GPU and is run last.

Usage:
    python -m tuning.validate --comfy-dir /path/to/ComfyUI \
        --pareto outputs/optimization/pareto_frontier.json \
        [--top-k 10] [--extra-sweep]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import torch

from .config_types import (
    TeacacheConfig, OptimizationResult, ValidationResult, TuningConfig,
)
from .utils import (
    load_models, sample, get_diffusion_model,
    compute_quality_metrics, img_to_tensor, QualityMetrics,
    print_metrics_legend,
)
from .forward import teacache_anima_forward


from .prompt_loader import load_prompt_config, select_prompts, resolve_prompt


def load_validation_prompts(tcfg: TuningConfig):
    """Load and resolve prompts for validation based on config settings."""
    cfg = tcfg.validation
    prompt_config = load_prompt_config(
        str(Path(__file__).parent / cfg["prompts_file"])
    )
    entries = select_prompts(
        prompt_config,
        method=cfg.get("prompt_selection", "from_top"),
        count=cfg["num_prompts"],
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
    return resolved


def patch_with_config(unet, cfg: TeacacheConfig):
    """Apply a TeaCache config as the model's _forward."""
    dm = get_diffusion_model(unet)

    # Reset cache state
    if hasattr(dm, "teacache_state"):
        delattr(dm, "teacache_state")

    # Replace _forward
    dm._forward = teacache_anima_forward.__get__(dm, dm.__class__)

    # Inject config into transformer_options
    to = unet.model_options.setdefault("transformer_options", {})
    cfg.inject_into_transformer_options(to)
    to["enable_teacache"] = True

    return dm


def cleanup_patch(dm, unet):
    """Remove TeaCache patch and config."""
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
    """Run one TeaCache generation. Returns (image, wall_time_sec).

    Handles patching, step-tracking wrapper, sampling, and cleanup.
    Used by both validate_config() and the smoke test.
    """
    dm = get_diffusion_model(unet)
    if hasattr(dm, "teacache_state"):
        delattr(dm, "teacache_state")

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
    finally:
        cleanup_patch(dm, unet)

    return img, dt


def validate_config(
    cfg: TeacacheConfig,
    unet, clip, vae,
    prompts: list,
    seeds: List[int],
    tcfg: TuningConfig,
    qm: QualityMetrics,
) -> ValidationResult:
    """End-to-end validation of one configuration.

    prompts: list of dicts with 'prompt' and 'negative' keys
    """
    metric_names = qm.metric_names()
    all_scores: dict[str, list[float]] = {k: [] for k in metric_names}
    all_speedups: list[float] = []
    all_times: list[float] = []

    for pi, pdata in enumerate(prompts):
        for seed in seeds:
            sp = f"p={pi} s={seed} thresh={cfg.rel_l1_thresh}"

            # Baseline (no patching)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            t0 = time.time()
            img_base = sample(
                unet, clip, vae, pdata["prompt"],
                seed=seed, steps=tcfg.sampling.get("default_steps", 30),
                cfg=tcfg.sampling["cfg"],
                sampler_name=tcfg.sampling["sampler"],
                scheduler=tcfg.sampling["scheduler"],
                width=tcfg.sampling["width"],
                height=tcfg.sampling["height"],
                negative=pdata["negative"],
            )
            t_base = time.time() - t0

            # TeaCache run (shared function — same code path as smoke test)
            img_tc, t_tc = run_single_teacache(
                unet, clip, vae,
                pdata["prompt"], pdata["negative"],
                cfg,
                seed=seed, steps=tcfg.sampling.get("default_steps", 30),
                sampler=tcfg.sampling["sampler"], scheduler=tcfg.sampling["scheduler"],
                width=tcfg.sampling["width"], height=tcfg.sampling["height"],
                cfg_val=tcfg.sampling["cfg"],
            )

            speedup = t_base / max(t_tc, 0.001)
            all_speedups.append(speedup)
            all_times.append(t_tc)

            # Compute all quality metrics
            scores = qm.measure(img_tc, img_base)
            for k in metric_names:
                all_scores[k].append(scores.get(k, float("nan")))

            lpips = scores.get("lpips_alex", float("inf"))
            print(f"  p={pi} s={seed} thresh={cfg.rel_l1_thresh} "
                  f"speedup={speedup:.2f}x  LPIPS={lpips:.4f}")

    # Aggregate
    n = len(all_speedups)
    mean_metrics = {}
    for k in metric_names:
        vals = [v for v in all_scores[k] if v != float("inf") and v == v]
        mean_metrics[k] = round(sum(vals) / max(len(vals), 1), 4) if vals else float("nan")

    return ValidationResult(
        config=cfg,
        mean_speedup=round(sum(all_speedups) / n, 3),
        mean_psnr=round(mean_metrics.get("psnr", float("inf")), 2),
        mean_ssim=round(mean_metrics.get("ssim", 1.0), 4),
        mean_lpips=round(mean_metrics.get("lpips_alex", 0.0), 4),
        mean_time_sec=round(sum(all_times) / n, 2),
        n_samples=n,
        mean_metrics=mean_metrics,
    )


def sweep_thresholds(
    base_cfg: TeacacheConfig,
    unet, clip, vae,
    prompts: list,
    seeds: List[int],
    tcfg: TuningConfig,
    qm: QualityMetrics,
    thresh_values: List[float],
) -> List[ValidationResult]:
    """Sweep threshold values for a given base config."""
    results = []
    for thresh in thresh_values:
        cfg = TeacacheConfig.from_dict(base_cfg.to_dict())
        cfg.rel_l1_thresh = thresh
        print(f"\n  Threshold sweep: thresh={thresh}")
        result = validate_config(cfg, unet, clip, vae, prompts, seeds, tcfg, qm)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="TeaCache Validator")
    parser.add_argument("--comfy-dir", required=True,
                        help="Path to ComfyUI installation")
    parser.add_argument("--pareto", required=True,
                        help="Path to pareto_frontier.json from Phase 2")
    parser.add_argument("--config", default=None,
                        help="Path to config.json")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Validate top K configurations (default: 10)")
    parser.add_argument("--extra-sweep", action="store_true",
                        help="Also run threshold sweeps for top configs")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3],
                        help="Metric tier (1=fastest, 3=most comprehensive)")
    args = parser.parse_args()

    if args.config is None:
        args.config = str(Path(__file__).parent / "config.json")
    tcfg = TuningConfig.load(args.config)

    out_dir = Path(tcfg.output_dir) / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  TeaCache Validator — Phase 3")
    print("=" * 60)
    print(f"  Pareto:  {args.pareto}")
    print(f"  Top-K:   {args.top_k}")
    print(f"  Tiers:   {args.tier}")
    print(f"  Output:  {out_dir}")

    # Create shared metrics instance before validation runs
    qm = QualityMetrics(tier=args.tier)
    if not qm.available:
        print("  ⚠ pyiqa not found. Install: pip install -r tuning/requirements.txt")
        sys.exit(1)
    metric_names = qm.metric_names()
    print(f"  Metrics:  {', '.join(metric_names)}")
    print("=" * 60)

    # Load Pareto frontier
    pareto_data = json.loads(Path(args.pareto).read_text())
    pareto_configs = [OptimizationResult.from_dict(d) for d in pareto_data]
    print(f"[load] {len(pareto_configs)} Pareto-optimal configs")

    # Load models
    unet, clip, vae = load_models(
        tcfg.comfy_dir, tcfg.model_name,
        tcfg.clip_name, tcfg.clip_type, tcfg.vae_name,
    )

    # Load prompts
    prompts = load_validation_prompts(tcfg)
    seeds = tcfg.validation["seeds"]

    # Select top-K diverse configs
    # Pick from different points on the Pareto frontier
    sorted_by_speed = sorted(pareto_configs, key=lambda r: r.estimated_speedup)
    selected = []

    # Pick configs evenly distributed along the frontier
    n = min(args.top_k, len(sorted_by_speed))
    indices = [int(i * (len(sorted_by_speed) - 1) / max(n - 1, 1)) for i in range(n)]
    selected = [sorted_by_speed[i] for i in sorted(set(indices))]

    print(f"\nValidating {len(selected)} configurations...")

    all_results = []
    for idx, opt_result in enumerate(selected):
        cfg = opt_result.config
        print(f"\n  [{idx+1}/{len(selected)}] {cfg.source} {cfg.metric_type} "
              f"{cfg.mapping_type} (sim speedup={opt_result.estimated_speedup:.2f}x)")

        result = validate_config(cfg, unet, clip, vae, prompts, seeds, tcfg, qm)
        all_results.append(result)

        # Extra threshold sweep for top-3
        if args.extra_sweep and idx < 3:
            extra_threshes = tcfg.validation.get("extra_threshold_sweep", [
                0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.0])
            sweep_results = sweep_thresholds(
                cfg, unet, clip, vae, prompts[:4], seeds[:2],
                tcfg, qm, extra_threshes,
            )
            all_results.extend(sweep_results)

    # Save and print
    (out_dir / "validation_results.json").write_text(
        json.dumps([r.to_dict() for r in all_results], indent=2)
    )

    print(f"\n{'=' * 60}")
    print(f"  Validation Results")
    print(f"{'=' * 60}")

    # Print the metric legend for context
    print_metrics_legend()

    # Build dynamic header from actual metric names
    short_metrics = {
        "psnr": "PSNR", "ssim": "SSIM", "lpips_alex": "LPIPSa",
        "lpips_vgg": "LPIPSv", "dists": "DISTS", "ms_ssim": "MSSIM",
        "fsim": "FSIM", "vif": "VIF", "gmsd": "GMSD",
        "nlpd": "NLPD", "pieapp": "PieAPP", "vsi": "VSI",
    }
    display_names = [short_metrics.get(n, n[:6]) for n in metric_names]

    header = (f"  {'#':>3} | {'source':<20} | {'speed':>6} | "
              + " | ".join(f"{d:>7}" for d in display_names))
    print(header)
    print(f"  {'─' * len(header)}")

    primary = [r for r in all_results if not hasattr(r.config, '_is_sweep')]
    for i, r in enumerate(primary):
        c = r.config
        vals = " | ".join(
            f"{r.mean_metrics.get(n, float('nan')):>7.4f}"
            for n in metric_names
        )
        print(f"  {i+1:>3} | {c.source:<20} | {r.mean_speedup:>5.2f}x | {vals}")

    # Find best by LPIPS < 0.05 and max speedup
    good = [r for r in all_results if r.mean_lpips < 0.05]
    if good:
        best = max(good, key=lambda r: r.mean_speedup)
        extra = ""
        if best.mean_metrics:
            extra = "  " + "  ".join(
                f"{short_metrics.get(n, n)}={best.mean_metrics.get(n, '?'):.4f}"
                for n in ["dists", "ms_ssim", "fsim", "vif", "gmsd"]
                if n in best.mean_metrics
            )
        print(f"\n  ★ Recommended (LPIPS<0.05): thresh={best.config.rel_l1_thresh} "
              f"speedup={best.mean_speedup}x LPIPS={best.mean_lpips}"
              f"{extra}")
    else:
        best = min(all_results, key=lambda r: r.mean_lpips)
        print(f"\n  ★ Best quality (no LPIPS<0.05): thresh={best.config.rel_l1_thresh} "
              f"speedup={best.mean_speedup}x LPIPS={best.mean_lpips}")

    print(f"\n  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
