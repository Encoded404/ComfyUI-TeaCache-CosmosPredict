#!/usr/bin/env python3
"""Phase 1: Calibration data recording.

Runs the Anima model with a calibration-patched forward function that records
delta statistics for ALL source signals (t_emb, first_block_shift, pooled_latent)
plus ground truth output changes at every step.

The recorded JSONL file enables the offline optimizer (optimize.py) to simulate
thousands of TeaCache configurations without touching the GPU.

Usage:
    cd /path/to/ComfyUI-TeaCache-CosmosPredict
    python -m tuning.calibrate --comfy-dir /path/to/ComfyUI [--prompts 12 --seeds 0,7,42]

Runtime estimate (A100-40GB):
    24 prompts × 4 seeds × 5 step variants = 480 generations
    ~12 seconds per generation → ~96 minutes total
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from .config_types import CalibrationEntry, TuningConfig
from .utils import load_models, sample, get_diffusion_model, estimate_calibration_time, detect_gpu
from .recorder import make_calibration_forward
from .prompt_loader import load_prompt_config, select_prompts, resolve_prompt


def load_calibration_prompts(tcfg: TuningConfig):
    """Load and resolve prompts for calibration based on config settings."""
    cfg = tcfg.calibration
    prompt_config = load_prompt_config(
        str(Path(__file__).parent / cfg["prompts_file"])
    )
    entries = select_prompts(
        prompt_config,
        method=cfg.get("prompt_selection", "from_top"),
        count=cfg["num_prompts"],
        tag_filter=cfg.get("prompt_tag_filter"),
    )
    # Resolve with cycling prefix/negative variants
    resolved = []
    for i, entry in enumerate(entries):
        full, neg = resolve_prompt(
            prompt_config, entry,
            prefix_variant_idx=i % max(len(prompt_config.prefix_variants), 1),
            negative_variant_idx=i % max(len(prompt_config.negative_variants), 1),
        )
        resolved.append({"prompt": full, "negative": neg, "entry": entry})
    return resolved


def patch_for_calibration(unet, steps: int, prompt_id: int, seed: int,
                          track_per_block: bool = False):
    """Patch the model's _forward with the calibration recorder and inject metadata.

    The recorder reads calibration_step and calibration_total_steps from
    transformer_options to tag each entry. We use a unet_wrapper_function
    (same pattern as TeaCache.apply_teacache) to track the step index.
    """
    diffusion_model = get_diffusion_model(unet)

    # Replace _forward with calibration version
    calib_fwd = make_calibration_forward()
    original_fwd = diffusion_model._forward
    diffusion_model._forward = calib_fwd.__get__(
        diffusion_model, diffusion_model.__class__
    )

    # Reset calibration state
    if hasattr(diffusion_model, "_calib_state"):
        # Also reset per-block tracking state if switching modes
        for attr in ("_calib_state", "_calib_block_prev", "_calib_block_deltas"):
            if hasattr(diffusion_model, attr):
                delattr(diffusion_model, attr)
    diffusion_model.calibration_log = []

    # Inject metadata into transformer_options
    to = unet.model_options.setdefault("transformer_options", {})
    to["calibration_step"] = 0
    to["calibration_total_steps"] = steps
    to["calibration_prompt_id"] = prompt_id
    to["calibration_seed"] = seed

    # Per-block tracking state lives on the model directly, not in
    # transformer_options, to avoid interfering with model-specific
    # WrapperExecutor chains (Cosmos Predict2, etc.).
    diffusion_model._calib_track_per_block = track_per_block

    # Add a wrapper to update step index
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

    unet.set_model_unet_function_wrapper(wrapper)

    return diffusion_model, original_fwd


def restore_model(diffusion_model, original_fwd, unet):
    """Restore original _forward and remove calibration metadata."""
    diffusion_model._forward = original_fwd
    unet.set_model_unet_function_wrapper(None)
    to = unet.model_options.get("transformer_options", {})
    for k in list(to.keys()):
        if k.startswith("calibration_"):
            del to[k]


def run_calibration(comfy_dir: str, config_path: str = None):
    # Load config
    if config_path is None:
        config_path = str(Path(__file__).parent / "config.json")
    tcfg = TuningConfig.load(config_path)

    # Setup output
    out_dir = Path(tcfg.output_dir) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(tcfg.__dict__ if hasattr(tcfg, '__dict__') else {}, indent=2, default=str)
    )

    print("=" * 60)
    print("  TeaCache Calibration — Phase 1")
    print("=" * 60)
    gpu_display, gpu_speed = detect_gpu()
    print(f"  GPU:            {gpu_display}  (×{gpu_speed:.1f} vs V100)")
    print(f"  ComfyUI:        {tcfg.comfy_dir}")
    print(f"  Model:          {tcfg.model_name}")
    print(f"  Steps:          {tcfg.sampling['step_variants']}")
    print(f"  Weights:        {tcfg.sampling['step_weights']}")
    print(f"  Resolution:     {tcfg.sampling['width']}×{tcfg.sampling['height']}")
    print(f"  Prompts:        {tcfg.calibration['num_prompts']}")
    print(f"  Seeds:          {tcfg.calibration['seeds']}")
    record_blocks = bool(tcfg.calibration.get("record_block_data", False))
    print(f"  Block data:     {'ON  (per-block deltas recorded)' if record_blocks else 'OFF'}")
    print(f"  Output:         {out_dir}")
    print("=" * 60)

    # Load models
    unet, clip, vae = load_models(
        tcfg.comfy_dir, tcfg.model_name,
        tcfg.clip_name, tcfg.clip_type, tcfg.vae_name,
    )
    prompts = load_calibration_prompts(tcfg)
    seeds = tcfg.calibration["seeds"]
    step_variants = tcfg.sampling["step_variants"]
    step_weights = tcfg.sampling["step_weights"]
    sampler_variants = tcfg.sampling.get("sampler_variants", [tcfg.sampling["sampler"]])
    scheduler_variants = tcfg.sampling.get("scheduler_variants", [tcfg.sampling["scheduler"]])
    cfg_variants = tcfg.sampling.get("cfg_variants", [tcfg.sampling["cfg"]])

    print(f"  Samplers:       {sampler_variants}")
    print(f"  Schedulers:     {scheduler_variants}")
    print(f"  CFGs:           {cfg_variants}")
    print(f"\n  Selected prompts ({len(prompts)}):")
    for i, pdata in enumerate(prompts):
        sampler = sampler_variants[i % len(sampler_variants)]
        cfg_val = cfg_variants[i % len(cfg_variants)]
        tags = [t for t in pdata["entry"].tags[:4]] if "entry" in pdata else []
        short = pdata["prompt"][:80].replace("\n", " ")
        print(f"    {i:>2}: [{sampler} cfg={cfg_val}]  [{', '.join(tags)}]  {short}...")

    total_runs = len(prompts) * len(seeds) * len(step_variants)
    # Estimate entries: each run produces ~ (steps - 1) × 2 cond slots
    avg_steps = sum(step_variants) / max(len(step_variants), 1)
    est_entries = int(total_runs * (avg_steps - 1) * 2)

    # Estimate time based on real GPU, step mix, and resolution
    w = tcfg.sampling["width"]
    h = tcfg.sampling["height"]
    est_seconds, gpu_name, gpu_factor = estimate_calibration_time(
        total_runs, step_variants, step_weights, w, h,
    )

    print(f"\n  {'─' * 56}")
    print(f"  Run schedule")
    print(f"  {'─' * 56}")
    print(f"  Permutation:  {len(prompts)} prompts × {len(seeds)} seeds × {len(step_variants)} step variants")
    print(f"                = {total_runs} total generations")
    print(f"  GPU:          {gpu_name}  (×{gpu_factor:.1f} vs V100)")
    print(f"  Resolution:   {w}×{h}")
    print(f"  Avg. steps:   {avg_steps:.1f} (weighted)")
    print(f"  Est. entries: ~{est_entries} ({int(est_entries/1000)}k) calibration data points")
    print(f"  Est. time:    ~{int(est_seconds // 60)}m {int(est_seconds % 60)}s")
    print(f"  Est. disk:    ~{est_entries * 300 // 1000}k kB  (JSONL)")
    print(f"  Output dir:   {out_dir}")
    print(f"  {'─' * 56}\n")
    print(f"  Press Ctrl+C to abort, or wait 3 seconds...")
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n  Aborted.")
        return

    all_entries: list[CalibrationEntry] = []
    run_idx = 0
    wall_start = time.time()

    data_file = out_dir / "calibration_data.jsonl"

    for pi, pdata in enumerate(prompts):
        # Cycle sampler / scheduler / cfg per prompt for variety
        cur_sampler  = sampler_variants[pi % len(sampler_variants)]
        cur_scheduler = scheduler_variants[pi % len(scheduler_variants)]
        cur_cfg       = cfg_variants[pi % len(cfg_variants)]

        for seed in seeds:
            for st in step_variants:
                steps = int(st)
                weight = step_weights[step_variants.index(st)]

                run_idx += 1
                t0 = time.time()

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                dm, original_fwd = patch_for_calibration(
                    unet, steps, prompt_id=pi, seed=seed,
                    track_per_block=record_blocks,
                )

                try:
                    img = sample(
                        unet, clip, vae, pdata["prompt"],
                        seed=seed, steps=steps,
                        cfg=cur_cfg,
                        sampler_name=cur_sampler,
                        scheduler=cur_scheduler,
                        width=tcfg.sampling["width"],
                        height=tcfg.sampling["height"],
                        negative=pdata["negative"],
                    )
                finally:
                    restore_model(dm, original_fwd, unet)

                dt = time.time() - t0
                run_entries = list(dm.calibration_log)

                # Tag entries with step variant info
                for e in run_entries:
                    e.total_steps = steps
                    e.step_fraction = e.step / max(steps - 1, 1)
                    e.sampler = cur_sampler
                    e.scheduler = cur_scheduler

                all_entries.extend(run_entries)

                # Save incrementally
                with data_file.open("a") as f:
                    for e in run_entries:
                        f.write(json.dumps(e.to_dict()) + "\n")

                valid = [e for e in run_entries if e.out_rel > 0]
                vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
                eta = (dt * (total_runs - run_idx)) / 60.0

                print(
                    f"[calib] {run_idx}/{total_runs}  "
                    f"p={pi} s={seed} steps={steps}  "
                    f"sampler={cur_sampler} cfg={cur_cfg}  "
                    f"took={dt:.1f}s  entries={len(run_entries)}  "
                    f"valid={len(valid)}  VRAM={vram:.1f}GB  ETA={eta:.0f}m"
                )

    # Summary
    wall_elapsed = time.time() - wall_start
    valid_all = [e for e in all_entries if e.out_rel > 0]

    it_per_sec = total_runs / wall_elapsed if wall_elapsed > 0 else 0.0
    if it_per_sec >= 1.0:
        speed_str = f"{it_per_sec:.1f} it/s"
    else:
        speed_str = f"{wall_elapsed / total_runs:.1f} s/it" if total_runs > 0 else "N/A"

    print(f"\n{'=' * 60}")
    print(f"  Calibration complete")
    print(f"  Total time:      {int(wall_elapsed // 60)}m {int(wall_elapsed % 60)}s")
    print(f"  Throughput:      {speed_str}  ({total_runs} runs)")
    print(f"  Total entries:   {len(all_entries)}")
    print(f"  Valid entries:   {len(valid_all)}  (with out_rel)")
    if record_blocks:
        block_entries = sum(1 for e in all_entries if e.block_cos_sims is not None)
        print(f"  Block entries:   {block_entries}  (per-block cos_sim data)")
    print(f"  Data saved to:   {data_file}")
    print(f"{'=' * 60}")

    return str(data_file)


def main():
    parser = argparse.ArgumentParser(description="TeaCache Calibration Recorder")
    parser.add_argument("--comfy-dir", required=True,
                        help="Path to ComfyUI installation")
    parser.add_argument("--config", default=None,
                        help="Path to config.json (default: tuning/config.json)")
    parser.add_argument("--prompts", type=int, default=None,
                        help="Override number of calibration prompts")
    parser.add_argument("--seeds", default=None,
                        help="Override seeds (comma-separated)")
    args = parser.parse_args()

    # Overrides
    if args.config is None:
        args.config = str(Path(__file__).parent / "config.json")

    tcfg = TuningConfig.load(args.config)
    if args.prompts is not None:
        tcfg.calibration["num_prompts"] = args.prompts
    if args.seeds is not None:
        tcfg.calibration["seeds"] = [int(s) for s in args.seeds.split(",")]

    data_file = run_calibration(args.comfy_dir, args.config)
    print(f"\nNext step: python -m tuning.optimize --data {data_file}")


if __name__ == "__main__":
    main()
