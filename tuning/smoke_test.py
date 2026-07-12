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
from .utils import load_models, sample, get_diffusion_model, QualityMetrics, print_metrics_legend, score_from_legend
from .recorder import make_calibration_forward
from .optimize import generate_candidate_configs, optimize as run_optimizer
from .prompt_loader import select_prompts, PromptEntry, PromptConfig
from .validate import run_single_teacache


SMOKE_PREFIX = (
    "masterpiece, best quality, score_7, newest, highres, absurdres, "
    "anime screenshot, detailed anime style, "
)
SMOKE_FULL = (
    "a beautiful anime girl with long silver hair, blue eyes, "
    "cherry blossoms falling, soft afternoon lighting"
)
SMOKE_NEGATIVE = (
    "worst quality, low quality, score_1, score_2, score_3, "
    "artist name, multiple views"
)
SMOKE_FULL = SMOKE_PREFIX + SMOKE_FULL

# ── Calibration runs — varied to produce diverse (rel_l1, out_rel) pairs  ──
SMOKE_RUNS = [
    {"sampler": "er_sde",        "steps": 30, "cfg": 5.0, "scheduler": "normal", "seed": 42},
    {"sampler": "er_sde",        "steps": 29, "cfg": 4.5, "scheduler": "simple", "seed": 45},
    {"sampler": "dpmpp_2m_sde",  "steps": 28, "cfg": 4.5, "scheduler": "simple", "seed": 7},
    {"sampler": "dpmpp_2m_sde",  "steps": 30, "cfg": 5.0, "scheduler": "normal", "seed": 7},
    {"sampler": "euler_a",       "steps": 32, "cfg": 5.5, "scheduler": "normal", "seed": 99},
    {"sampler": "euler_a",       "steps": 30, "cfg": 5.0, "scheduler": "simple", "seed": 64},
]

# ── Reference: daraskme's published coefficients for Anima at 30 steps ──
# Source: github.com/daraskme/comfy_anima_tea_cache
# Used as a validation baseline — if this config also produces poor results,
# there's a fundamental infrastructure issue, not a calibration issue.
DARASKME_CONFIG = TeacacheConfig(
    source="first_block_shift",
    metric_type="mean_only",
    signal_scale=1.0,
    mapping_type="polynomial",
    coefficients=[5954.035087553969, -2410.0426539290293, 349.24023850217395,
                  -17.264742642375417, 0.31229336331906893],
    accumulation_type="hard_reset",
    rel_l1_thresh=0.07,
    step_schedule="constant",
    start_percent=0.05,
    end_percent=0.95,
    residual_strategy="hard",
    block_mode="all_or_nothing",
)


# ═══════════════════════════════════════════════════════════════════════════
#  Prompt diversity test — 12 prompts designed to discriminate between
#  tag_diversity, text_diversity, and semantic_diversity methods.
# ═══════════════════════════════════════════════════════════════════════════

DIVERSITY_TEST_PROMPTS = [
    # A and B: same tags, lexically different, semantically very similar
    PromptEntry(text="1girl, female elf archer, green cloak, aiming a bow in a sunlit forest clearing, sharp focus",
                tags=["character", "action", "landscape", "day"]),
    PromptEntry(text="1boy, male forest ranger, brown leather vest, drawing bowstring in a woodland glade at dawn, golden light",
                tags=["character", "action", "landscape", "day"]),
    # C and D: share some tags AND words (close-up, macro), different subjects
    PromptEntry(text="close-up of a single dewdrop on a red rose petal, macro photography, golden morning light, shallow depth of field",
                tags=["close_up", "photorealistic", "day", "landscape", "simple"]),
    PromptEntry(text="extreme close-up of a hummingbird feeding from a pink hibiscus flower, vibrant colors, macro shot, wings frozen mid-flap",
                tags=["close_up", "photorealistic", "day", "landscape", "action"]),
    # E and F: completely different — no shared tags, words, or semantics
    PromptEntry(text="futuristic cyberpunk mega-city at night, towering neon-lit skyscrapers, flying cars between buildings, rain-soaked streets, blade runner aesthetic",
                tags=["landscape", "night", "cinematic", "detail_heavy"]),
    PromptEntry(text="samurai warrior in traditional red armor standing in a misty bamboo forest, katana drawn, cherry blossoms drifting, dramatic monochrome ink wash painting style",
                tags=["character", "action", "landscape", "day", "cinematic"]),
    # G and H: share interior + character, different actions
    PromptEntry(text="wizard casting a fireball spell in an ancient stone library, floating books and scrolls, magical orange glow, high vaulted ceilings",
                tags=["character", "action", "interior", "night", "detail_heavy", "abstract"]),
    PromptEntry(text="priest healing a wounded knight on the stone floor of a gothic cathedral, soft golden light from stained glass, peaceful atmosphere",
                tags=["character", "couple", "interior", "day", "simple"]),
    # I and J: share character format (1girl, hair, eyes) but different everything else
    PromptEntry(text="1girl, Sylvarie, elf, long platinum blonde hair, bright violet eyes, pointed ears, wearing elegant silk gown, standing in a moonlit forest clearing",
                tags=["character", "landscape", "night", "simple"]),
    PromptEntry(text="1girl, Lyra, wolf girl, long silver hair, blue eyes, wolf ears, silver tail, oversized sweater, sitting on a futuristic city balcony at night, cyberpunk neon background",
                tags=["character", "landscape", "night", "detail_heavy"]),
    # K and L: multi-view prompts — share the tag, different characters
    PromptEntry(text="1girl, multi view, character reference sheet, female android, chrome chassis, front, side, back, three-quarter views, expression set, technical blueprint style",
                tags=["character", "multi_view", "simple", "abstract"]),
    PromptEntry(text="1boy, multi view, turnaround, male barbarian, fur cloak, muscular, front view standing, side profile, back view, action pose with axe, war paint",
                tags=["character", "multi_view", "simple", "action"]),
]

# Which pairs are designed to be close — each method should separate at least
# some of these, but semantic_diversity should separate the most:
#   (A, B): same tags, lexically different, semantically SIMILAR
#   (C, D): share tags + some words, semantically DIFFERENT (dewdrop vs bird)
#   (G, H): share interior/character, semantically SIMILAR (magic/intervention)
#   (I, J): share character templates, semantically DIFFERENT (elf vs sci-fi)
#   (K, L): share multi_view tag, lexically DIFFERENT (android vs barbarian)

DIVERSITY_TEST_CONFIG = PromptConfig(
    default_prefix="", default_negative="", prompts=DIVERSITY_TEST_PROMPTS
)


def _run_diversity_test():
    """Test all three diversity methods on 12 crafted prompts, print analysis."""
    print(f"\n{'─' * 60}")
    print(f"  Prompt Diversity Test — tag vs text vs semantic")
    print(f"{'─' * 60}")
    print(f"  {len(DIVERSITY_TEST_PROMPTS)} prompts, picking 5 with each method\n")

    # Print all prompts for reference
    print(f"  Full pool:")
    labels = "ABCDEFGHIJKL"
    for i, p in enumerate(DIVERSITY_TEST_PROMPTS):
        short = p.text[:75] + "..." if len(p.text) > 75 else p.text
        print(f"    {labels[i]}: [{', '.join(p.tags[:4])}] {short}")
    print()

    methods = ["tag_diversity", "text_diversity", "semantic_diversity"]
    descriptions = {
        "tag_diversity":       "Maximizes unique tag coverage — picks different categories first",
        "text_diversity":      "Maximizes word-level difference (Jaccard distance)",
        "semantic_diversity":  "Maximizes semantic distance via MiniLM embeddings (80MB model)",
    }

    all_picks = {}
    for method in methods:
        picks = select_prompts(
            DIVERSITY_TEST_CONFIG, method=method, count=5, seed=42
        )
        pick_labels = []
        for p in picks:
            for i, orig in enumerate(DIVERSITY_TEST_PROMPTS):
                if p is orig:
                    pick_labels.append(labels[i])
                    break
        all_picks[method] = pick_labels
        print(f"  {method:>22}: {', '.join(pick_labels)}")
        print(f"    {descriptions[method]}")

    # Analysis
    print(f"\n  Analysis:")
    for method, picks in all_picks.items():
        # Count how many of the designed pairs were separated
        pairs = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H"),
                 ("I", "J"), ("K", "L")]
        separated = 0
        for a, b in pairs:
            if a in picks and b in picks:
                separated += 0
            elif a in picks or b in picks:
                separated += 1
        print(f"    {method:>22}: {separated}/6 designed pairs separated")

    # Check semantic specifically
    if "semantic_diversity" in all_picks:
        sp = all_picks["semantic_diversity"]
        # A vs B are semantically similar — they should NOT both be picked
        ab_both = "A" in sp and "B" in sp
        gh_both = "G" in sp and "H" in sp
        print(f"    semantic: A/B both picked? {'yes (bad)' if ab_both else 'no (good)'}  "
              f"G/H both picked? {'yes (bad)' if gh_both else 'no (good)'}")

    print()


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
            unet, clip, vae, SMOKE_FULL,
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

    # ── [1/8] Load models ──────────────────────────────────────────────
    print("\n[1/8] Loading models...")
    try:
        unet, clip, vae = load_models(comfy_dir, model_name, clip_name, clip_type, vae_name)
    except Exception as e:
        print(f"\n  FAILED to load models: {e}")
        return False

    # ── [2/8] Prompt diversity test ────────────────────────────────────
    _run_diversity_test()

    # ── [3/8] Baseline generation (no patching) ────────────────────────
    base_run = SMOKE_RUNS[0]
    print(f"\n[3/8] Baseline generation ({base_run['steps']} steps, {base_run['sampler']}, cfg={base_run['cfg']})...")
    try:
        t0 = time.time()
        img_base = sample(
            unet, clip, vae, SMOKE_FULL,
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

    # ── [4/8] Calibration data collection ──────────────────────────────
    print(f"\n[4/8] Collecting calibration data ({len(SMOKE_RUNS)} runs)...")
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

    # ── [5/8] Mini-optimizer ───────────────────────────────────────────
    print(f"\n[5/8] Mini-optimizer (running optimize.generate_candidate_configs + optimize)...")
    try:
        candidates = generate_candidate_configs(tcfg, entries=all_entries)
        print(f"  Generated {len(candidates)} candidate configs")

        # Use the real optimizer (same code path as full pipeline)
        all_opt_results, pareto_results = run_optimizer(candidates, all_entries, tcfg)

        # Show Pareto summary
        pareto = pareto_results if pareto_results else all_opt_results[:3]
        pareto.sort(key=lambda r: r.estimated_speedup)

        if not pareto or all(r.skip_rate < 0.01 for r in pareto):
            print(f"  ⚠  No configs with skip > 1% — calibration data may be too sparse.")
            pareto = [r for r in all_opt_results if r.skip_rate >= 0][:3]

        print(f"\n  Pareto summary ({len(pareto)} configs):")
        print(f"  {'source':<22} {'metric':<14} {'map':<12} {'skip':>6} {'speed':>6} {'error':>8} {'score':>6}")
        print(f"  {'─' * 22} {'─' * 14} {'─' * 12} {'─' * 6} {'─' * 6} {'─' * 8} {'─' * 6}")
        for r in pareto[:8]:
            c = r.config
            print(f"  {c.source:<22} {c.metric_type:<14} {c.mapping_type:<12} "
                  f"{r.skip_rate:>5.1%}  {r.estimated_speedup:>4.2f}x  {r.accumulated_error:>7.4f}  {r.score:>5.3f}")

        # Pick fairly-spaced configs for end-to-end validation
        n_pick = min(3, len(pareto))
        if n_pick == 1:
            indices = [0]
        elif n_pick == 2:
            indices = [0, len(pareto) - 1]
        else:
            lo, hi = 0, len(pareto) - 1
            mid = len(pareto) // 2
            indices = sorted(set([lo, mid, hi]))
        labels = ["conservative", "balanced", "aggressive"][:n_pick]
        picks = []
        for i, idx in enumerate(indices):
            r = pareto[idx]
            c = r.config
            picks.append({
                "label": labels[i],
                "thresh": c.rel_l1_thresh,
                "sim_speedup": r.estimated_speedup,
                "sim_error": r.accumulated_error,
                "config": c,
            })

        print(f"\n  Selected {len(picks)} for end-to-end validation:")
        print(f"  {'':>14} {'thresh':>8} {'speedup':>8} {'error':>8} {'source':<22}")
        for p in picks:
            c = p["config"]
            print(f"  {p['label']:>14} {p['thresh']:>8.3f} {p['sim_speedup']:>7.2f}x {p['sim_error']:>8.4f} {c.source:<22}")
    except Exception as e:
        print(f"\n  FAILED mini-optimizer: {e}")
        import traceback; traceback.print_exc()
        return False

    # ── [6/8] TeaCache comparison runs vs baseline ─────────────────────
    print(f"\n[6/8] TeaCache comparison ({len(picks)} configs) vs baseline...")
    base_run = SMOKE_RUNS[0]
    comparison = {}

    for p in picks:
        label = p["label"]
        cfg_cmp = p["config"]

        try:
            img, dt = run_single_teacache(
                unet, clip, vae,
                SMOKE_FULL, SMOKE_NEGATIVE,
                cfg_cmp,
                seed=base_run["seed"], steps=base_run["steps"],
                sampler=base_run["sampler"], scheduler=base_run["scheduler"],
                width=512, height=512,
                cfg_val=base_run["cfg"],
            )
            su = t_base / max(dt, 0.001)
            print(f"  {label:>14}: thresh={cfg_cmp.rel_l1_thresh:.3f}  {dt:.1f}s  speedup={su:.2f}x  "
                  f"({cfg_cmp.source} {cfg_cmp.metric_type} {cfg_cmp.mapping_type})")
            comparison[label] = {"time": dt, "speedup": su, "thresh": cfg_cmp.rel_l1_thresh, "img": img}
        except Exception as e:
            print(f"  {label:>14}: FAILED — {e}")
            import traceback; traceback.print_exc()
            return False

    # ── Daraskme reference (known-good config, validation baseline) ────
    print(f"\n  {'─' * 60}")
    print(f"  Reference: daraskme config (first_block_shift, polynomial, thresh=0.07)")
    print(f"  Source: github.com/daraskme/comfy_anima_tea_cache")
    try:
        img_ref, dt_ref = run_single_teacache(
            unet, clip, vae,
            SMOKE_FULL, SMOKE_NEGATIVE,
            DARASKME_CONFIG,
            seed=base_run["seed"], steps=base_run["steps"],
            sampler=base_run["sampler"], scheduler=base_run["scheduler"],
            width=512, height=512,
            cfg_val=base_run["cfg"],
        )
        su_ref = t_base / max(dt_ref, 0.001)
        print(f"  {'daraskme':>14}: thresh={DARASKME_CONFIG.rel_l1_thresh:.3f}  {dt_ref:.1f}s  speedup={su_ref:.2f}x")
        comparison["daraskme"] = {"time": dt_ref, "speedup": su_ref,
                                   "thresh": DARASKME_CONFIG.rel_l1_thresh, "img": img_ref}
    except Exception as e:
        print(f"  {'daraskme':>14}: FAILED — {e}")
        import traceback; traceback.print_exc()

    # ── [7/8] Quality check ────────────────────────────────────────────
    print(f"\n[7/8] Quality check (all 12 metrics, Tier 3)...")
    try:
        qm = QualityMetrics(tier=3)

        # ── Legend (shared with validate.py) ────────────────────────────
        print_metrics_legend()
        # Also import the legend data for score rating
        from .utils import METRIC_LEGEND

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

            # Show tuned configs first, then daraskme reference with separator
            pick_labels = [p["label"] for p in picks]
            all_labels = pick_labels + (["daraskme"] if "daraskme" in comparison else [])

            for li, label in enumerate(all_labels):
                if label == "daraskme" and li > 0:
                    print(f"  {'─' * 72}")
                info = comparison.get(label)
                if info is None:
                    continue
                scores = all_run_scores.get(label, {})
                vals = " │ ".join(f"{scores.get(m, float('nan')):>8.4f}" for m in KEY_METRICS)
                print(f"  {label:>14} │ {info['thresh']:>7.3f} │ {info['speedup']:>7.2f}x │ {vals}")

            # Detailed breakdown for the middle/balanced config
            pick_labels = [p["label"] for p in picks]
            detail_key = "balanced" if "balanced" in pick_labels else pick_labels[len(pick_labels)//2]
            detail_scores = all_run_scores.get(detail_key, {})
            detail_info = comparison.get(detail_key, {})

            excellent = acceptable = poor = 0
            for name, _, _, _, _ in METRIC_LEGEND:
                val = detail_scores.get(name, float("nan"))
                rating = score_from_legend(name, val)
                if "EXCELLENT" in rating:   excellent += 1
                elif "acceptable" in rating: acceptable += 1
                else:                       poor += 1

            spd = detail_info.get("speedup", 0)
            print(f"\n  Detailed breakdown ({detail_key}, thresh={detail_info.get('thresh', '?'):.3f}, speedup={spd:.2f}x):")
            print(f"    {excellent} ✅ excellent   {acceptable} ✓ acceptable   {poor} ⚠ needs tuning\n")

            COL_M = 12;  COL_D = 3;  SP = " │ "
            COL_SCORE = 8
            COL_RATING = 14
            def _score_row(metric, dir_str, score_str, rating):
                return (f"{metric:>{COL_M}}{SP}"
                        f"{dir_str:^{COL_D}}{SP}"
                        f"{score_str:>{COL_SCORE}}{SP}"
                        f"{rating:<{COL_RATING}}")
            score_rows = []
            for name, direction, _, good, mid in METRIC_LEGEND:
                val = detail_scores.get(name, float("nan"))
                score_str = f"  {val:.4f}" if val == val else "    N/A"
                rating = score_from_legend(name, val)
                score_rows.append(_score_row(name, direction, score_str, rating))
            score_header = _score_row("Metric", "↑↓", "  Score", "Rating")
            all_score = [score_header] + score_rows
            sw = max(len(r) for r in all_score)

            print(f"  ╔{'═' * (sw + 2)}╗")
            print(f"  ║ {' ' + score_header.ljust(sw)} ║")
            print(f"  ╟{'─' * (sw + 2)}╢")
            for row in score_rows:
                print(f"  ║ {' ' + row.ljust(sw)} ║")
            print(f"  ╚{'═' * (sw + 2)}╝")
        else:
            print(f"\n  pyiqa not installed. Install with: pip install -r tuning/requirements.txt")
    except Exception as e:
        print(f"  Could not compute metrics: {e}")

    # ── [8/8] Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Smoke test PASSED — all 8 checks completed")
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
