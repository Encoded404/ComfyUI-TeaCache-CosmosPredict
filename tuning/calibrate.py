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

from .config_types import CalibrationEntry, TuningConfig
from .utils import load_models, sample, get_diffusion_model
from .recorder import make_calibration_forward


def load_prompts(prompts_file: str, num_prompts: int) -> list:
    path = Path(__file__).parent / prompts_file
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    return lines[:num_prompts]


def patch_for_calibration(unet, steps: int, prompt_id: int, seed: int):
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
        delattr(diffusion_model, "_calib_state")
    diffusion_model.calibration_log = []

    # Inject metadata into transformer_options
    to = unet.model_options.setdefault("transformer_options", {})
    to["calibration_step"] = 0
    to["calibration_total_steps"] = steps
    to["calibration_prompt_id"] = prompt_id
    to["calibration_seed"] = seed

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
    print(f"  ComfyUI:        {tcfg.comfy_dir}")
    print(f"  Model:          {tcfg.model_name}")
    print(f"  Steps:          {tcfg.sampling['step_variants']}")
    print(f"  Weights:        {tcfg.sampling['step_weights']}")
    print(f"  Prompts:        {tcfg.calibration['num_prompts']}")
    print(f"  Seeds:          {tcfg.calibration['seeds']}")
    print(f"  Output:         {out_dir}")
    print("=" * 60)

    # Load models
    unet, clip, vae = load_models(
        tcfg.comfy_dir, tcfg.model_name,
        tcfg.clip_name, tcfg.clip_type, tcfg.vae_name,
    )
    prompts = load_prompts(
        tcfg.calibration["prompts_file"],
        tcfg.calibration["num_prompts"],
    )
    seeds = tcfg.calibration["seeds"]
    step_variants = tcfg.sampling["step_variants"]
    step_weights = tcfg.sampling["step_weights"]
    negative = tcfg.calibration.get("negative_prompt", "")

    total_runs = len(prompts) * len(seeds) * len(step_variants)
    all_entries: list[CalibrationEntry] = []
    run_idx = 0

    data_file = out_dir / "calibration_data.jsonl"

    for pi, prompt in enumerate(prompts):
        for seed in seeds:
            for st in step_variants:
                steps = int(st)
                weight = step_weights[step_variants.index(st)]
                # step_weights are informational for the optimizer
                # (it can weight simulation results by step count)

                run_idx += 1
                t0 = time.time()

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # Patch model for calibration
                dm, original_fwd = patch_for_calibration(
                    unet, steps, prompt_id=pi, seed=seed
                )

                try:
                    img = sample(
                        unet, clip, vae, prompt,
                        seed=seed, steps=steps,
                        cfg=tcfg.sampling["cfg"],
                        sampler_name=tcfg.sampling["sampler"],
                        scheduler=tcfg.sampling["scheduler"],
                        width=tcfg.sampling["width"],
                        height=tcfg.sampling["height"],
                        negative=negative,
                    )
                finally:
                    restore_model(dm, original_fwd, unet)

                dt = time.time() - t0
                run_entries = list(dm.calibration_log)

                # Tag entries with step variant info
                for e in run_entries:
                    e.total_steps = steps
                    e.step_fraction = e.step / max(steps - 1, 1)

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
                    f"took={dt:.1f}s  entries={len(run_entries)}  "
                    f"valid={len(valid)}  VRAM={vram:.1f}GB  ETA={eta:.0f}m"
                )

    # Summary
    valid_all = [e for e in all_entries if e.out_rel > 0]
    print(f"\n{'=' * 60}")
    print(f"  Calibration complete")
    print(f"  Total entries: {len(all_entries)}")
    print(f"  Valid entries (with out_rel): {len(valid_all)}")
    print(f"  Data saved to: {data_file}")
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
