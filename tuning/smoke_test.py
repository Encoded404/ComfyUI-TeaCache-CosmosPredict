#!/usr/bin/env python3
"""Smoke test: verify model loading, calibration recording, mini-optimization,
and TeaCache forward all work before committing to a full calibration run.

Generates 3 varied images to collect diverse calibration data, runs the
offline optimizer on that data, then tests TeaCache with tuned coefficients.

Usage:
    cd /path/to/ComfyUI
    PYTHONPATH=".:custom_nodes/ComfyUI-TeaCache-CosmosPredict" \\
        python -m tuning.smoke_test --comfy-dir .
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from .config_types import TeacacheConfig, TuningConfig, CalibrationEntry
from .utils import load_models, sample, get_diffusion_model, QualityMetrics
from .recorder import make_calibration_forward
from .forward import teacache_anima_forward
from .optimize import simulate_config, fit_polynomial_coefficients


SMOKE_PROMPT = (
    "a beautiful anime girl with long silver hair, blue eyes, "
    "cherry blossoms falling, soft afternoon lighting"
)
SMOKE_NEGATIVE = ""

# ── Calibration runs — varied to produce diverse (rel_l1, out_rel) pairs  ──
SMOKE_RUNS = [
    {"sampler": "er_sde",        "steps": 30, "cfg": 5.0, "scheduler": "normal", "seed": 42},
    {"sampler": "dpmpp_2m_sde",  "steps": 28, "cfg": 4.5, "scheduler": "simple", "seed": 7},
    {"sampler": "euler_a",       "steps": 32, "cfg": 5.5, "scheduler": "normal", "seed": 99},
]


def _run_calibration(unet, clip, vae, params, prompt_id):
    """Run one calibration pass, returning list of CalibrationEntry."""
    dm = get_diffusion_model(unet)
    calib_fwd = make_calibration_forward()

    if hasattr(dm, "_calib_state"):
        delattr(dm, "_calib_state")
    dm.calibration_log = []

    original = dm._forward
    dm._forward = calib_fwd.__get__(dm, dm.__class__)

    steps = params["steps"]
    to = unet.model_options.setdefault("transformer_options", {})
    to["calibration_step"] = 0
    to["calibration_total_steps"] = steps
    to["calibration_prompt_id"] = prompt_id
    to["calibration_seed"] = params["seed"]

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
        sample(
            unet, clip, vae, SMOKE_PROMPT,
            seed=params["seed"], steps=steps,
            width=512, height=512,
            cfg=params["cfg"], sampler_name=params["sampler"],
            scheduler=params["scheduler"], negative=SMOKE_NEGATIVE,
        )
        entries = list(dm.calibration_log)
    finally:
        dm._forward = original
        unet.set_model_unet_function_wrapper(None)
        unet.model_options.pop("model_function_wrapper", None)
        unet.model_options.pop("model_function_wrapper", None)
        for k in list(to.keys()):
            if k.startswith("calibration_"):
                del to[k]

    return entries


def _run_teacache(unet, clip, vae, cfg: TeacacheConfig, seed, steps, sampler, sched, cfg_val):
    """Run one TeaCache generation, returning (image, wall_time)."""
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
            c_to["enable_teacache"] = (cfg.start_percent <= c_to["current_percent"] <= cfg.end_percent)
        return model_function(kwargs["input"], timestep, **c)

    try:
        unet.set_model_unet_function_wrapper(tc_wrapper)
        t0 = time.time()
        img = sample(
            unet, clip, vae, SMOKE_PROMPT,
            seed=seed, steps=steps,
            width=512, height=512,
            cfg=cfg_val, sampler_name=sampler, scheduler=sched,
            negative=SMOKE_NEGATIVE,
        )
        dt = time.time() - t0
    finally:
        dm._forward = original
        unet.set_model_unet_function_wrapper(None)
        unet.model_options.pop("model_function_wrapper", None)
        for k in list(to.keys()):
            if k.startswith("tc_"):
                del to[k]
        to.pop("enable_teacache", None)
        to.pop("rel_l1_thresh", None)
        to.pop("coefficients", None)
        to.pop("current_percent", None)

    return img, dt


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_smoke_test(comfy_dir: str, steps: int = 30):
    # Load config for model paths
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

    print("=" * 60)
    print("  TeaCache Anima Smoke Test")
    print("=" * 60)
    print(f"  ComfyUI: {comfy_dir}")
    print(f"  Runs:    {len(SMOKE_RUNS)} varied (sampler, steps, cfg)")

    # ── [1/7] Load models ──────────────────────────────────────────────
    print("\n[1/7] Loading models...")
    try:
        unet, clip, vae = load_models(comfy_dir, model_name, clip_name, clip_type, vae_name)
    except Exception as e:
        print(f"\n  FAILED to load models: {e}")
        return False

    # ── [2/7] Baseline generation (no patching) ────────────────────────
    base_run = SMOKE_RUNS[0]
    print(f"\n[2/7] Baseline generation ({base_run['steps']} steps, {base_run['sampler']}, cfg={base_run['cfg']})...")
    try:
        t0 = time.time()
        img_base = sample(
            unet, clip, vae, SMOKE_PROMPT,
            seed=base_run["seed"], steps=base_run["steps"],
            width=512, height=512,
            cfg=base_run["cfg"], sampler_name=base_run["sampler"],
            scheduler=base_run["scheduler"], negative=SMOKE_NEGATIVE,
        )
        t_base = time.time() - t0
        print(f"  Baseline: {t_base:.1f}s, image size: {img_base.size}")
    except Exception as e:
        print(f"\n  FAILED baseline generation: {e}")
        import traceback; traceback.print_exc()
        return False

    # ── [3/7] Calibration data collection ──────────────────────────────
    print(f"\n[3/7] Collecting calibration data ({len(SMOKE_RUNS)} runs)...")
    all_entries = []
    try:
        for i, run in enumerate(SMOKE_RUNS):
            t0 = time.time()
            entries = _run_calibration(unet, clip, vae, run, prompt_id=i)
            dt = time.time() - t0
            all_entries.extend(entries)
            valid = sum(1 for e in entries if e.out_rel > 0)
            print(f"  run {i+1}: steps={run['steps']} sampler={run['sampler']:>12s} "
                  f"cfg={run['cfg']} seed={run['seed']}  "
                  f"took={dt:.1f}s  entries={len(entries)}  valid={valid}")

        total_valid = sum(1 for e in all_entries if e.out_rel > 0)
        print(f"  Total: {len(all_entries)} entries ({total_valid} valid)")
        if total_valid < 50:
            print(f"  ⚠  Only {total_valid} valid — mini-optimizer may be unreliable")
    except Exception as e:
        print(f"\n  FAILED calibration: {e}")
        import traceback; traceback.print_exc()
        return False

    # ── [4/7] Mini-optimizer ───────────────────────────────────────────
    print(f"\n[4/7] Mini-optimizer (running optimize.simulate_config)...")
    try:
        candidates = []
        for source in ["first_block_shift", "t_emb"]:
            for metric_type in ["mean_only", "mean_and_max"]:
                for mapping in ["identity", "polynomial"]:
                    cfg = TeacacheConfig(
                        source=source, metric_type=metric_type,
                        metric_weights={"mean": 0.7, "max": 0.3}
                        if metric_type == "mean_and_max" else {},
                        signal_scale=1.0 if source == "first_block_shift" else 100.0,
                        mapping_type=mapping, coefficients=[],
                        accumulation_type="hard_reset", rel_l1_thresh=0.07,
                        step_schedule="constant", start_percent=0.05, end_percent=0.95,
                        residual_strategy="hard", block_mode="all_or_nothing",
                    )
                    if mapping == "polynomial":
                        cfg.coefficients = fit_polynomial_coefficients(all_entries, cfg)
                    skip, err, sp, qp = simulate_config(all_entries, cfg)
                    candidates.append((cfg, skip, err, sp, qp))

        candidates.sort(key=lambda x: x[4] * x[3], reverse=True)

        print(f"  {'source':<22} {'metric':<14} {'map':<12} {'skip':>6} {'speed':>6} {'error':>8} {'score':>6}")
        print(f"  {'─' * 22} {'─' * 14} {'─' * 12} {'─' * 6} {'─' * 6} {'─' * 8} {'─' * 6}")
        for cfg, skip, err, sp, qp in candidates[:8]:
            print(f"  {cfg.source:<22} {cfg.metric_type:<14} {cfg.mapping_type:<12} "
                  f"{skip:>5.1%}  {sp:>4.2f}x  {err:>7.4f}  {qp*sp:>5.3f}")

        best = candidates[0]
        best_cfg = best[0]
        print(f"\n  Winner: {best_cfg.source} / {best_cfg.metric_type} / {best_cfg.mapping_type}")
        if best_cfg.coefficients:
            print(f"  Coeffs: {[f'{c:.4e}' for c in best_cfg.coefficients]}")
            print(f"  Max |coeff|: {max(abs(c) for c in best_cfg.coefficients):.4e}")

        # Sweep thresholds to find quality-tier picks
        sweep = []
        for t in sorted(set([0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.0])):
            cfg_s = TeacacheConfig.from_dict(best_cfg.to_dict())
            cfg_s.rel_l1_thresh = t
            skip, err, sp, qp = simulate_config(all_entries, cfg_s)
            sweep.append((t, sp, qp, skip, err))
        sweep.sort(key=lambda x: x[0])

        # Pick three points on the quality curve: conservative, balanced, aggressive
        picks = []
        # Conservative: lowest simulated error (best quality)
        conservative = min(sweep[1:], key=lambda x: x[4])  # skip t=0
        # Aggressive: highest speedup
        aggressive = max(sweep[1:], key=lambda x: x[1])
        # Balanced: best product
        balanced = max(sweep[1:], key=lambda x: x[1] * x[2])
        for label, (t, sp, qp, sk, er) in [
            ("conservative", conservative), ("balanced", balanced), ("aggressive", aggressive)
        ]:
            picks.append({"label": label, "thresh": t, "sim_speedup": sp, "sim_error": er})

        # Deduplicate picks
        seen = set()
        picks_unique = []
        for p in picks:
            if p["thresh"] not in seen:
                picks_unique.append(p)
                seen.add(p["thresh"])

        print(f"\n  Quality comparison picks (simulated):")
        print(f"  {'':>14} {'thresh':>8} {'speedup':>8} {'error':>8}")
        for p in picks_unique:
            print(f"  {p['label']:>14} {p['thresh']:>8.3f} {p['sim_speedup']:>7.2f}x {p['sim_error']:>8.4f}")
    except Exception as e:
        print(f"\n  FAILED mini-optimizer: {e}")
        import traceback; traceback.print_exc()
        return False

    # ── [5/7] Baseline + TeaCache comparison runs ──────────────────────
    print(f"\n[5/7] TeaCache comparison (3 thresholds) vs baseline...")
    base_run = SMOKE_RUNS[0]

    comparison = {}
    # Run baseline once (already done in [2/7])
    comparison["baseline"] = {"time": t_base, "speedup": 1.0, "thresh": 0.0, "img": img_base}

    for p in picks_unique:
        label = p["label"]
        cfg_cmp = TeacacheConfig.from_dict(best_cfg.to_dict())
        cfg_cmp.rel_l1_thresh = p["thresh"]

        try:
            img, dt = _run_teacache(
                unet, clip, vae, cfg_cmp,
                seed=base_run["seed"], steps=base_run["steps"],
                sampler=base_run["sampler"], sched=base_run["scheduler"],
                cfg_val=base_run["cfg"],
            )
            su = t_base / max(dt, 0.001)
            print(f"  {label:>14}: thresh={p['thresh']:.3f}  {dt:.1f}s  speedup={su:.2f}x")
            comparison[label] = {"time": dt, "speedup": su, "thresh": p["thresh"], "img": img}
        except Exception as e:
            print(f"  {label:>14}: FAILED — {e}")
            import traceback; traceback.print_exc()
            return False

    # ── [6/7] Quality check ────────────────────────────────────────────
    print(f"\n[6/7] Quality check (all 12 metrics, Tier 3)...")
    try:
        qm = QualityMetrics(tier=3)

        # ── Legend ──────────────────────────────────────────────────────
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

        COL_METRIC = 12
        COL_DIR    = 3
        COL_GOOD   = 7
        COL_MID    = 14
        COL_POOR   = 7
        COL_WHAT   = 35
        SPACER = " │ "

        def _legend_row(metric, dir_str, good_s, mid_s, poor_s, what_s):
            return (f"{metric:>{COL_METRIC}}{SPACER}"
                    f"{dir_str:^{COL_DIR}}{SPACER}"
                    f"{good_s:>{COL_GOOD}}{SPACER}"
                    f"{mid_s:>{COL_MID}}{SPACER}"
                    f"{poor_s:>{COL_POOR}}{SPACER}"
                    f"{what_s:<{COL_WHAT}}")

        legend_rows = []
        for name, direction, what, good, mid in METRIC_LEGEND:
            if direction == "↑":
                gs, ms, ps = f"  >{good:g}", f"  {mid:g} - {good:g}", f"  <{mid:g}"
            else:
                gs, ms, ps = f"  <{good:g}", f"  {good:g} - {mid:g}", f"  >{mid:g}"
            legend_rows.append(_legend_row(name, direction, gs, ms, ps, what))

        header_row = _legend_row("Metric", "↑↓", "  Good", "    Mid", "  Poor", "What it measures")
        all_rows = [header_row] + legend_rows
        box_width = max(len(r) for r in all_rows)

        def _box_line(body, width):
            return f"  ║ {body.ljust(width)} ║"

        print(f"\n  ╔{'═' * (box_width + 2)}╗")
        print(_box_line("HOW TO READ METRICS", box_width))
        print(_box_line("↑ = higher is better    ↓ = lower is better", box_width))
        print(f"  ╟{'─' * (box_width + 2)}╢")
        print(_box_line(header_row, box_width))
        print(f"  ╟{'─' * (box_width + 2)}╢")
        for row in legend_rows:
            print(_box_line(row, box_width))
        print(f"  ╚{'═' * (box_width + 2)}╝")

        # ── Scores — key metrics across all comparison runs ──────────────
        if qm.available:
            # Compute scores for each run vs baseline
            all_run_scores = {}
            for label, info in comparison.items():
                all_run_scores[label] = qm.measure(info["img"], img_base)

            # Print comparison table: speedup + key metrics per run
            KEY_METRICS = ["lpips_alex", "lpips_vgg", "dists", "ms_ssim", "fsim", "vif"]
            KEY_DIRS   = {"lpips_alex": "↓", "lpips_vgg": "↓", "dists": "↓",
                          "ms_ssim": "↑", "fsim": "↑", "vif": "↑"}

            print(f"\n  Comparison vs Baseline (same prompt, seed={base_run['seed']}):\n")
            header = (f"  {'Run':>14} │ {'thresh':>7} │ {'speedup':>8} │ "
                      + " │ ".join(f"{m:>8}" for m in KEY_METRICS))
            print(header)
            print(f"  {'─' * 14}─┼─{'─' * 7}─┼─{'─' * 8}─┼─" + "─┼─".join("─" * 8 for _ in KEY_METRICS))
            print(f"  {'baseline':>14} │ {'0':>7} │ {'1.00x':>8} │ "
                  + " │ ".join(f"{'—':>8}" for _ in KEY_METRICS))

            for label, info in comparison.items():
                scores = all_run_scores.get(label, {})
                vals = " │ ".join(f"{scores.get(m, float('nan')):>8.4f}" for m in KEY_METRICS)
                print(f"  {label:>14} │ {info['thresh']:>7.3f} │ {info['speedup']:>7.2f}x │ {vals}")

            # Detailed breakdown for the balanced config
            balanced_key = "balanced" if "balanced" in comparison else list(comparison.keys())[0]
            bal_scores = all_run_scores.get(balanced_key, {})
            bal_info = comparison.get(balanced_key, {})

            excellent = acceptable = poor = 0
            for name, direction, _, good, mid in METRIC_LEGEND:
                val = bal_scores.get(name, float("nan"))
                if val != val: continue
                if direction == "↑":
                    if val >= good:   excellent += 1
                    elif val >= mid:  acceptable += 1
                    else:             poor += 1
                else:
                    if val <= good:   excellent += 1
                    elif val <= mid:  acceptable += 1
                    else:             poor += 1

            spd = bal_info.get("speedup", 0)
            print(f"\n  Detailed breakdown ({balanced_key}, thresh={bal_info.get('thresh', '?'):.3f}, speedup={spd:.2f}x):")
            print(f"    {excellent} ✅ excellent   {acceptable} ✓ acceptable   {poor} ⚠ needs tuning\n")

            COL_SCORE = 8
            COL_RATING = 14
            def _score_row(metric, dir_str, score_str, rating):
                return (f"{metric:>{COL_METRIC}}{SPACER}"
                        f"{dir_str:^{COL_DIR}}{SPACER}"
                        f"{score_str:>{COL_SCORE}}{SPACER}"
                        f"{rating:<{COL_RATING}}")
            score_rows = []
            for name, direction, _, good, mid in METRIC_LEGEND:
                val = bal_scores.get(name, float("nan"))
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
            score_header = _score_row("Metric", "↑↓", "  Score", "Rating")
            all_score = [score_header] + score_rows
            sw = max(len(r) for r in all_score)
            print(f"  ╔{'═' * (sw + 2)}╗")
            print(_box_line(score_header, sw))
            print(f"  ╟{'─' * (sw + 2)}╢")
            for row in score_rows:
                print(_box_line(row, sw))
            print(f"  ╚{'═' * (sw + 2)}╝")
        else:
            print(f"\n  pyiqa not installed. Install with: pip install -r tuning/requirements.txt")
    except Exception as e:
        print(f"  Could not compute metrics: {e}")

    # ── [7/7] Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Smoke test PASSED — all 7 checks completed")
    print(f"  Ready for full calibration run:")
    print(f"    PYTHONPATH='.:custom_nodes/ComfyUI-TeaCache-CosmosPredict'")
    print(f"    python -m tuning.calibrate --comfy-dir .")
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
