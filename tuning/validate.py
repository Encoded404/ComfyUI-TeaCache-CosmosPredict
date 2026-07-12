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
    compute_quality_metrics, img_to_tensor,
)
from .forward import teacache_anima_forward


def load_prompts(prompts_file: str, num_prompts: int) -> list:
    path = Path(__file__).parent / prompts_file
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    return lines[:num_prompts]


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
    if hasattr(dm, "teacache_state"):
        delattr(dm, "teacache_state")


def validate_config(
    cfg: TeacacheConfig,
    unet, clip, vae,
    prompts: List[str],
    seeds: List[int],
    tcfg: TuningConfig,
) -> ValidationResult:
    """End-to-end validation of one configuration."""
    results = []

    for pi, prompt in enumerate(prompts):
        for seed in seeds:
            print(f"    p={pi} s={seed} thresh={cfg.rel_l1_thresh}  ", end="")

            # Baseline (no patching)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            t0 = time.time()
            img_base = sample(
                unet, clip, vae, prompt,
                seed=seed, steps=tcfg.sampling.get("default_steps", 30),
                cfg=tcfg.sampling["cfg"],
                sampler_name=tcfg.sampling["sampler"],
                scheduler=tcfg.sampling["scheduler"],
                width=tcfg.sampling["width"],
                height=tcfg.sampling["height"],
                negative=tcfg.calibration.get("negative_prompt", ""),
            )
            t_base = time.time() - t0

            # TeaCache run
            dm = patch_with_config(unet, cfg)

            try:
                torch.cuda.empty_cache()
                t0 = time.time()
                img_tc = sample(
                    unet, clip, vae, prompt,
                    seed=seed, steps=tcfg.sampling.get("default_steps", 30),
                    cfg=tcfg.sampling["cfg"],
                    sampler_name=tcfg.sampling["sampler"],
                    scheduler=tcfg.sampling["scheduler"],
                    width=tcfg.sampling["width"],
                    height=tcfg.sampling["height"],
                    negative=tcfg.calibration.get("negative_prompt", ""),
                )
                t_tc = time.time() - t0
            finally:
                cleanup_patch(dm, unet)

            speedup = t_base / max(t_tc, 0.001)
            psnr, ssim, lpips = compute_quality_metrics(img_tc, img_base)

            results.append({
                "time_base": t_base,
                "time_tc": t_tc,
                "speedup": speedup,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips,
            })

            print(f"speedup={speedup:.2f}x  LPIPS={lpips:.4f}")

    # Aggregate
    n = len(results)
    mean_speedup  = sum(r["speedup"] for r in results) / n
    mean_time     = sum(r["time_tc"] for r in results) / n
    mean_psnr     = sum(r["psnr"] for r in results if r["psnr"] != float("inf")) / max(
        sum(1 for r in results if r["psnr"] != float("inf")), 1
    )
    mean_ssim     = sum(r["ssim"]  for r in results) / n
    mean_lpips    = sum(r["lpips"] for r in results) / n

    return ValidationResult(
        config=cfg,
        mean_speedup=round(mean_speedup, 3),
        mean_psnr=round(mean_psnr, 2),
        mean_ssim=round(mean_ssim, 4),
        mean_lpips=round(mean_lpips, 4),
        mean_time_sec=round(mean_time, 2),
        n_samples=n,
    )


def sweep_thresholds(
    base_cfg: TeacacheConfig,
    unet, clip, vae,
    prompts: List[str],
    seeds: List[int],
    tcfg: TuningConfig,
    thresh_values: List[float],
) -> List[ValidationResult]:
    """Sweep threshold values for a given base config."""
    results = []
    for thresh in thresh_values:
        cfg = TeacacheConfig.from_dict(base_cfg.to_dict())
        cfg.rel_l1_thresh = thresh
        print(f"\n  Threshold sweep: thresh={thresh}")
        result = validate_config(cfg, unet, clip, vae, prompts, seeds, tcfg)
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
    print(f"  Output:  {out_dir}")
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
    prompts = load_prompts(
        tcfg.validation["prompts_file"],
        tcfg.validation["num_prompts"],
    )
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

        result = validate_config(cfg, unet, clip, vae, prompts, seeds, tcfg)
        all_results.append(result)

        # Extra threshold sweep for top-3
        if args.extra_sweep and idx < 3:
            extra_threshes = tcfg.validation.get("extra_threshold_sweep", [
                0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.0])
            sweep_results = sweep_thresholds(
                cfg, unet, clip, vae, prompts[:4], seeds[:2],
                tcfg, extra_threshes,
            )
            all_results.extend(sweep_results)

    # Save and print
    (out_dir / "validation_results.json").write_text(
        json.dumps([r.to_dict() for r in all_results], indent=2)
    )

    print(f"\n{'=' * 60}")
    print(f"  Validation Results")
    print(f"{'=' * 60}")
    header = (f"  {'#':>3} | {'source':<20} | {'metric':<15} | {'speedup':>8} | "
              f"{'PSNR':>8} | {'SSIM':>6} | {'LPIPS':>6}")
    print(header)
    print(f"  {'─' * len(header)}")

    primary = [r for r in all_results if not hasattr(r.config, '_is_sweep')]
    for i, r in enumerate(primary):
        c = r.config
        print(f"  {i+1:>3} | {c.source:<20} | {c.metric_type:<15} | "
              f"{r.mean_speedup:>7.2f}x | {r.mean_psnr:>8.1f} | "
              f"{r.mean_ssim:>6.4f} | {r.mean_lpips:>6.4f}")

    # Find best by LPIPS < 0.05 and max speedup
    good = [r for r in all_results if r.mean_lpips < 0.05]
    if good:
        best = max(good, key=lambda r: r.mean_speedup)
        print(f"\n  ★ Recommended (LPIPS<0.05): thresh={best.config.rel_l1_thresh} "
              f"speedup={best.mean_speedup}x LPIPS={best.mean_lpips} "
              f"({best.config.source} {best.config.metric_type})")
    else:
        best = min(all_results, key=lambda r: r.mean_lpips)
        print(f"\n  ★ Best quality (no LPIPS<0.05): thresh={best.config.rel_l1_thresh} "
              f"speedup={best.mean_speedup}x LPIPS={best.mean_lpips}")

    print(f"\n  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
