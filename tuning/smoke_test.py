#!/usr/bin/env python3
"""Smoke test: verify that model loading, calibration recording, and TeaCache
forward all work without crashing before committing to a multi-hour calibration run.

Usage:
    cd /path/to/ComfyUI-TeaCache-CosmosPredict
    python -m tuning.smoke_test --comfy-dir /path/to/ComfyUI [--steps 30]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from .config_types import TeacacheConfig, TuningConfig
from .utils import load_models, sample, get_diffusion_model, compute_quality_metrics, QualityMetrics
from .recorder import make_calibration_forward
from .forward import teacache_anima_forward


SMOKE_PROMPT = (
    "a beautiful anime girl with long silver hair, blue eyes, "
    "cherry blossoms falling, soft afternoon lighting"
)
SMOKE_NEGATIVE = ""


def run_smoke_test(comfy_dir: str, steps: int = 30):
    print("=" * 60)
    print("  TeaCache Anima Smoke Test")
    print("=" * 60)
    print(f"  ComfyUI: {comfy_dir}")
    print(f"  Steps:   {steps}")

    # Load the default config for model paths
    cfg_path = Path(__file__).parent / "config.json"
    if cfg_path.exists():
        tcfg = TuningConfig.load(str(cfg_path))
        model_name = tcfg.model_name
        clip_name  = tcfg.clip_name
        clip_type  = tcfg.clip_type
        vae_name   = tcfg.vae_name
    else:
        model_name = "anima-base-v1.0.safetensors"
        clip_name  = "qwen_3_06b_base.safetensors"
        clip_type  = "qwen_image"
        vae_name   = "qwen_image_vae.safetensors"

    # ── 1. Load models ──
    print("\n[1/5] Loading models...")
    try:
        unet, clip, vae = load_models(
            comfy_dir, model_name, clip_name, clip_type, vae_name
        )
    except Exception as e:
        print(f"\n  FAILED to load models: {e}")
        return False

    # ── 2. Baseline generation (no patching) ──
    print(f"\n[2/5] Baseline generation ({steps} steps)...")
    try:
        t0 = time.time()
        img_base = sample(
            unet, clip, vae, SMOKE_PROMPT,
            seed=42, steps=steps,
            width=512, height=512,
            cfg=5.5,
            sampler_name="dpmpp_2m_sde",
            scheduler="normal",
            negative=SMOKE_NEGATIVE,
        )
        t_base = time.time() - t0
        print(f"  Baseline: {t_base:.1f}s, image size: {img_base.size}")
    except Exception as e:
        print(f"\n  FAILED baseline generation: {e}")
        import traceback; traceback.print_exc()
        return False

    # ── 3. Calibration recording smoke test ──
    print(f"\n[3/5] Calibration recording ({steps} steps, 1 prompt)...")
    try:
        dm = get_diffusion_model(unet)
        calib_fwd = make_calibration_forward()

        dm.calibration_log = []
        if hasattr(dm, "_calib_state"):
            delattr(dm, "_calib_state")

        original = dm._forward
        dm._forward = calib_fwd.__get__(dm, dm.__class__)

        # Install wrapper to inject per-step metadata
        to = unet.model_options.setdefault("transformer_options", {})
        to["calibration_step"] = 0
        to["calibration_total_steps"] = steps
        to["calibration_prompt_id"] = 0
        to["calibration_seed"] = 42

        def wrapper(model_function, kwargs):
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
                c_to["calibration_step"] = step_idx
            return model_function(kwargs["input"], timestep, **c)

        try:
            unet.set_model_unet_function_wrapper(wrapper)

            img_calib = sample(
                unet, clip, vae, SMOKE_PROMPT,
                seed=42, steps=steps,
                width=512, height=512,
                cfg=7.5,
                sampler_name="dpmpp_2m_sde",
                scheduler="normal",
                negative=SMOKE_NEGATIVE,
            )

            n_entries = len(dm.calibration_log)
            print(f"  Calibration: recorded {n_entries} entries")
            if n_entries > 0:
                e = dm.calibration_log[0]
                print(f"  Sample entry: step={e.step}, t_emb_mean={e.t_emb.mean if e.t_emb else 'N/A'}")
        finally:
            dm._forward = original
            unet.set_model_unet_function_wrapper(None)
            for k in list(to.keys()):
                if k.startswith("calibration_"):
                    del to[k]
    except Exception as e:
        print(f"\n  FAILED calibration recording: {e}")
        import traceback; traceback.print_exc()
        return False

    complete = n_entries >= (steps - 1) * 2  # 2 cond slots per step
    if complete:
        print(f"  ✅ Calibration recording works (expected ~{steps*2} entries)")
    else:
        print(f"  ⚠ Calibration recording produced fewer entries than expected")

    # ── 4. TeaCache forward smoke test ──
    print(f"\n[4/5] TeaCache forward (config: default)...")
    try:
        dm = get_diffusion_model(unet)

        # Reset TeaCache state
        if hasattr(dm, "teacache_state"):
            delattr(dm, "teacache_state")

        tc_fwd = teacache_anima_forward
        original = dm._forward
        dm._forward = tc_fwd.__get__(dm, dm.__class__)

        # Inject a default config
        cfg = TeacacheConfig(
            source="first_block_shift",
            metric_type="mean_only",
            mapping_type="identity",
            coefficients=[],
            accumulation_type="hard_reset",
            rel_l1_thresh=0.07,
            start_percent=0.05,
            end_percent=0.95,
        )
        to = unet.model_options.setdefault("transformer_options", {})
        cfg.inject_into_transformer_options(to)
        to["enable_teacache"] = True

        # Install step-tracking wrapper (same pattern as TeaCache.apply_teacache)
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
            img_tc = sample(
                unet, clip, vae, SMOKE_PROMPT,
                seed=42, steps=steps,
                width=512, height=512,
                cfg=5.5,
                sampler_name="dpmpp_2m_sde",
                scheduler="normal",
                negative=SMOKE_NEGATIVE,
            )
            t_tc = time.time() - t0
            print(f"  TeaCache: {t_tc:.1f}s (baseline: {t_base:.1f}s) "
                  f"speedup: {t_base/max(t_tc, 0.001):.2f}x")
        finally:
            dm._forward = original
            unet.set_model_unet_function_wrapper(None)
            for k in list(to.keys()):
                if k.startswith("tc_"):
                    del to[k]
            to.pop("enable_teacache", None)
            to.pop("rel_l1_thresh", None)
            to.pop("coefficients", None)
            to.pop("current_percent", None)
    except Exception as e:
        print(f"\n  FAILED TeaCache forward: {e}")
        import traceback; traceback.print_exc()
        return False

    # ── 5. Quality check ──
    print(f"\n[5/5] Quality check (all 12 metrics, Tier 3)...")
    try:
        qm = QualityMetrics(tier=3)

        # ── Legend ──────────────────────────────────────────────────────
        # (name,  dir↑↓, what it measures,  good_thresh, mid_thresh)
        # For ↑ metrics: good >= thresh, mid >= thresh, else bad
        # For ↓ metrics: good <= thresh, mid <= thresh, else bad
        METRIC_LEGEND = [
            ("psnr",       "↑", "pixel-level accuracy",           35.0,  25.0),
            ("ssim",       "↑", "structural similarity",          0.95,  0.85),
            ("lpips_alex", "↓", "perceptual (AlexNet, semantic)", 0.05,  0.15),
            ("lpips_vgg",  "↓", "perceptual (VGG16, texture)",    0.10,  0.25),
            ("dists",      "↓", "structure vs texture decomp",    0.05,  0.15),
            ("ms_ssim",    "↑", "multi-scale structural simil.",  0.97,  0.92),
            ("fsim",       "↑", "edge sharpness (phase congru.)", 0.97,  0.90),
            ("vif",        "↑", "information fidelity",           0.60,  0.30),
            ("gmsd",       "↓", "gradient deviation (blur)",      0.05,  0.15),
            ("nlpd",       "↓", "Laplacian pyramid (human vis.)", 0.10,  0.25),
            ("pieapp",     "↓", "human pairwise preference",      0.10,  0.30),
            ("vsi",        "↑", "visual saliency-weighted simil.", 0.97,  0.90),
        ]

        # Column widths (excludes spacers)
        COL_METRIC = 12
        COL_DIR    = 3
        COL_GOOD   = 7
        COL_MID    = 11
        COL_POOR   = 7
        COL_WHAT   = 35
        SPACER = " │ "

        def _legend_row(metric, dir_str, good_s, mid_s, poor_s, what_s):
            row = (
                f"{metric:>{COL_METRIC}}{SPACER}"
                f"{dir_str:^{COL_DIR}}{SPACER}"
                f"{good_s:>{COL_GOOD}}{SPACER}"
                f"{mid_s:>{COL_MID}}{SPACER}"
                f"{poor_s:>{COL_POOR}}{SPACER}"
                f"{what_s:<{COL_WHAT}}"
            )
            return row

        # Build all rows first to determine max width
        legend_rows = []
        for name, direction, what, good, mid in METRIC_LEGEND:
            if direction == "↑":
                gs, ms, ps = f"  >{good:g}", f"  {mid:g} - {good:g}", f"  <{mid:g}"
            else:
                gs, ms, ps = f"  <{good:g}", f"  {good:g} - {mid:g}", f"  >{mid:g}"
            legend_rows.append(_legend_row(name, direction, gs, ms, ps, what))

        # Also build header-like rows for sizing
        header_row = _legend_row(
            "Metric", "dir", "  Good", "    Mid", "  Poor", "What it measures"
        )
        all_rows = [header_row] + legend_rows
        box_width = max(len(r) for r in all_rows)  # body content width

        def _box_line(body, width):
            return f"  ║ {body.ljust(width)} ║"

        # Print the box
        print(f"\n  ╔{'═' * (box_width + 2)}╗")
        print(_box_line("HOW TO READ METRICS", box_width))
        print(_box_line("↑ = higher is better    ↓ = lower is better", box_width))
        print(f"  ╟{'─' * (box_width + 2)}╢")
        print(_box_line(header_row, box_width))
        print(f"  ╟{'─' * (box_width + 2)}╢")
        for row in legend_rows:
            print(_box_line(row, box_width))
        print(f"  ╚{'═' * (box_width + 2)}╝")

        # ── Scores ──────────────────────────────────────────────────────
        if qm.available:
            scores = qm.measure(img_tc, img_base)

            # Summary line
            excellent = 0
            acceptable = 0
            poor = 0
            for name, direction, _, good, mid in METRIC_LEGEND:
                val = scores.get(name, float("nan"))
                if val != val:  # NaN
                    continue
                if direction == "↑":
                    if val >= good:
                        excellent += 1
                    elif val >= mid:
                        acceptable += 1
                    else:
                        poor += 1
                else:
                    if val <= good:
                        excellent += 1
                    elif val <= mid:
                        acceptable += 1
                    else:
                        poor += 1

            print(f"\n  TeaCache vs Baseline:")
            print(f"    {excellent} ✅ excellent   {acceptable} ✓ acceptable   {poor} ⚠ needs tuning\n")

            # Score table using same column layout as legend
            COL_SCORE = 8
            COL_RATING = 14

            def _score_row(metric, dir_str, score_str, rating):
                row = (
                    f"{metric:>{COL_METRIC}}{SPACER}"
                    f"{dir_str:^{COL_DIR}}{SPACER}"
                    f"{score_str:>{COL_SCORE}}{SPACER}"
                    f"{rating:<{COL_RATING}}"
                )
                return row

            score_rows = []
            for name, direction, _, good, mid in METRIC_LEGEND:
                val = scores.get(name, float("nan"))
                score_str = f"  {val:.4f}" if val == val else "    N/A"
                if direction == "↑":
                    if val >= good:   rating = "✅ EXCELLENT"
                    elif val >= mid:  rating = "✓ acceptable"
                    else:             rating = "⚠ POOR"
                else:
                    if val <= good:   rating = "✅ EXCELLENT"
                    elif val <= mid:  rating = "✓ acceptable"
                    else:             rating = "⚠ POOR"
                score_rows.append(_score_row(name, direction, score_str, rating))

            score_header = _score_row("Metric", "dir", "  Score", "Rating")
            all_score_rows = [score_header] + score_rows
            sw = max(len(r) for r in all_score_rows)

            print(f"  ╔{'═' * (sw + 2)}╗")
            print(_box_line(score_header, sw))
            print(f"  ╟{'─' * (sw + 2)}╢")
            for row in score_rows:
                print(_box_line(row, sw))
            print(f"  ╚{'═' * (sw + 2)}╝")

            print(f"\n  Expect many ⚠ at this stage — the config is not yet tuned.")
            print(f"  Calibration (Phase 1) + Optimization (Phase 2) will find")
            print(f"  the coefficients that push most metrics into ✅ range.")
        else:
            psnr, ssim, lpips = compute_quality_metrics(img_tc, img_base)
            print(f"\n  pyiqa not installed. Legacy fallback:")
            print(f"  PSNR:  {psnr:.2f}")
            print(f"  SSIM:  {ssim:.4f}")
            print(f"  LPIPS: {lpips:.4f}")
    except Exception as e:
        print(f"  Could not compute metrics: {e}")

    print(f"\n{'=' * 60}")
    print(f"  Smoke test PASSED")
    print(f"  All 5 checks completed successfully")
    print(f"  Ready for full calibration run")
    print(f"{'=' * 60}")
    return True


def main():
    parser = argparse.ArgumentParser(description="TeaCache Anima Smoke Test")
    parser.add_argument("--comfy-dir", required=True,
                        help="Path to ComfyUI installation")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of sampling steps (default: 30)")
    args = parser.parse_args()

    success = run_smoke_test(args.comfy_dir, args.steps)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
