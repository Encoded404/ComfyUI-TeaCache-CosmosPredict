#!/usr/bin/env python3
"""Phase 1: Calibration data recording.

Runs the Anima model with a calibration-patched forward function that records
delta statistics for ALL source signals (t_emb, first_block_shift, pooled_latent)
plus ground truth output changes at every step.

The recorded JSONL file enables the offline optimizer (optimize.py) to simulate
thousands of TeaCache configurations without touching the GPU.

With refiner recording enabled (--refiner-data both|only, plan Task 2), the same
runs additionally capture per-step latent tensors (x_t, v_t, prompt embeddings)
into outputs/<timestamp>/refiner_data/ for the latent-space refiner.

Usage:
    cd /path/to/ComfyUI-TeaCache-CosmosPredict
    python -m tuning.calibrate --comfy-dir /path/to/ComfyUI [--prompts 12 --seeds 0,7,42]
    python -m tuning.calibrate --comfy-dir /path/to/ComfyUI --refiner-data both

Runtime estimate (A100-40GB):
    24 prompts × 4 seeds × 5 step variants = 480 generations
    ~12 seconds per generation → ~96 minutes total
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

from .config_types import CalibrationEntry, TuningConfig
from .utils import load_models, sample, get_diffusion_model, detect_gpu, print_schedule_estimate, print_speed_summary
from .recorder import make_calibration_forward, make_refiner_forward
from . import refiner_data
from .prompt_loader import load_prompt_config, select_prompts, GenerationPromptSampler, resolve_generation
from .artist_tags import load_pool_for_config, print_artist_frequencies


def load_calibration_prompts(tcfg: TuningConfig):
    """Load and select prompts for calibration based on config settings.

    Prefix/negative variants and artist tags are NOT resolved here — they are
    drawn deterministically per generation (see GenerationPromptSampler), so
    each (prompt, seed, steps, resolution) gets its own variation.
    Returns (prompt_config, entries) where entries are {"entry": PromptEntry}.
    """
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
    return prompt_config, [{"entry": e} for e in entries]


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
        for attr in ("_calib_state", "_calib_block_prevs", "_calib_block_currs", "_calib_block_deltas"):
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


def patch_for_refiner(unet, steps: int, prompt_id: int, seed: int,
                      track_per_block: bool = False,
                      refiner_dir=None, refiner_cfg: dict = None):
    """Patch the model with the refiner recorder (calibration stats + latents).

    Mirrors patch_for_calibration (plan Task 2a): same transformer_options
    metadata injection + step-tracking wrapper, plus the latent capture state
    (_refiner_buf) and the record_slots/dtype settings from the refiner config.
    The wrapper additionally records the sigma/timestep per call.
    """
    diffusion_model = get_diffusion_model(unet)

    refiner_fwd = make_refiner_forward()
    original_fwd = diffusion_model._forward
    diffusion_model._forward = refiner_fwd.__get__(
        diffusion_model, diffusion_model.__class__
    )

    # Reset calibration + refiner state (per generation)
    for attr in ("_calib_state", "_calib_block_prevs", "_calib_block_currs",
                 "_calib_block_deltas", "_refiner_buf"):
        if hasattr(diffusion_model, attr):
            delattr(diffusion_model, attr)
    diffusion_model.calibration_log = []
    diffusion_model._refiner_buf = {
        "x": {}, "v": {}, "prompt": {}, "prompt_captured": set(),
        "timesteps": {}, "steps": {}, "step_fractions": {},
    }

    # Inject metadata into transformer_options
    to = unet.model_options.setdefault("transformer_options", {})
    to["calibration_step"] = 0
    to["calibration_total_steps"] = steps
    to["calibration_prompt_id"] = prompt_id
    to["calibration_seed"] = seed

    diffusion_model._calib_track_per_block = track_per_block
    diffusion_model._refiner_dir = refiner_dir

    rc = refiner_cfg or {}
    record_slots = rc.get("record_slots", "both")
    if record_slots not in ("both", "cond"):
        print(f"  [refiner] ⚠ invalid record_slots {record_slots!r} — using 'both'")
        record_slots = "both"
    diffusion_model._refiner_record_slots = record_slots

    dtype_name = rc.get("dtype", "bfloat16")
    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}.get(dtype_name)
    if dtype is None:
        print(f"  [refiner] ⚠ invalid dtype {dtype_name!r} — using bfloat16")
        dtype = torch.bfloat16
    diffusion_model._refiner_dtype = dtype

    # Add a wrapper to update step index + timestep
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

        try:
            t = timestep
            if isinstance(t, torch.Tensor):
                t = t.detach().float().cpu()
                t = float(t.reshape(-1)[0].item()) if t.numel() else 0.0
            c_to["calibration_timestep"] = float(t)
        except Exception:
            c_to["calibration_timestep"] = 0.0

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


# Seeded RNG for the deterministic resolution-mix assignment (Task 1):
# per generation index, the same draw happens on every run.
_RESOLUTION_RNG_SEED = 0


def _draw_resolution(rng, mix: dict):
    """Draw a resolution key ('WxH') by cumulative shares."""
    keys = list(mix.keys())
    shares = [mix[k] for k in keys]
    total = sum(shares)
    r = rng.random() * total
    acc = 0.0
    for k, s in zip(keys, shares):
        acc += s
        if r < acc:
            return k
    return keys[-1]


def run_calibration(comfy_dir: str, config_path: str = None, overrides: dict = None):
    # Load config
    if config_path is None:
        config_path = str(Path(__file__).parent / "config.json")
    tcfg = TuningConfig.load(config_path)

    # CLI/API overrides (num_prompts, seeds, sampling, refiner)
    overrides = overrides or {}
    if "num_prompts" in overrides:
        tcfg.calibration["num_prompts"] = overrides["num_prompts"]
    if "seeds" in overrides:
        tcfg.calibration["seeds"] = overrides["seeds"]
    if "sampling" in overrides:
        tcfg.sampling.update(overrides["sampling"])
    refiner_cfg = dict(getattr(tcfg, "refiner", {}) or {})
    refiner_cfg.update(overrides.get("refiner", {}) or {})

    # Resolve refiner mode: CLI wins; else config `enabled` (plan Task 1/2d)
    refiner_mode = refiner_cfg.get("mode", "")
    if refiner_mode not in ("both", "off", "only"):
        if "mode" in refiner_cfg:
            print(f"  ⚠ invalid refiner mode {refiner_mode!r} — falling back to config enabled flag")
        refiner_mode = "both" if refiner_cfg.get("enabled", False) else "off"
    refiner_cfg["mode"] = refiner_mode

    if refiner_mode != "off":
        record_slots = refiner_cfg.get("record_slots", "both")
        if record_slots not in ("both", "cond"):
            print(f"  ⚠ invalid refiner record_slots {record_slots!r} — using 'both'")
            refiner_cfg["record_slots"] = "both"

    # Resolution mix lives in the base sampling config (plan Task 1, revised):
    # it is a sampling property, not a refiner-recording one, so it applies to
    # every calibration generation regardless of refiner mode. Deterministic
    # assignment per generation index (seeded RNG, independent of
    # prompt/seed/step selection). Absent/empty → all generations at the base
    # sampling width/height (legacy behavior).
    mix_spec = tcfg.sampling.get("resolution_mix") or None
    res_mix = None
    if mix_spec:
        try:
            res_mix = refiner_data.parse_resolution_mix(mix_spec)
        except ValueError as e:
            raise SystemExit(f"[calibrate] invalid sampling.resolution_mix: {e}") from e

    # Setup output
    out_dir = Path(tcfg.output_dir) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(tcfg.__dict__ if hasattr(tcfg, '__dict__') else {}, indent=2, default=str)
    )

    # Calibration runs exactly the step counts in the sampling section;
    # optimization.step_count_variants never drives calibration runs.
    step_variants = tcfg.sampling["step_variants"]
    step_weights = tcfg.sampling["step_weights"]

    # Artist-tag pool + per-generation prompt sampler
    artist_pool = load_pool_for_config(tcfg)

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
    if artist_pool is not None:
        acfg = tcfg.artist_tags
        print(f"  Artist tags:    ON  ({len(artist_pool.artists)} tags, "
              f"mode={acfg.get('weight_mode', 'relative')}, "
              f"max_tags={acfg.get('max_tags', 1)}, seed={acfg.get('seed', 0)})")
    else:
        print(f"  Artist tags:    OFF")
    record_blocks = bool(tcfg.calibration.get("record_block_data", False))
    print(f"  Block data:     {'ON  (per-block deltas recorded)' if record_blocks else 'OFF'}")
    print(f"  Refiner data:   {refiner_mode}"
          + (f"  (slots={refiner_cfg.get('record_slots', 'both')}, "
             f"top_n={refiner_cfg.get('top_n', -1)})" if refiner_mode != "off" else ""))
    res_display = res_mix or f"{tcfg.sampling['width']}x{tcfg.sampling['height']} (fixed)"
    print(f"  Resolutions:    {res_display}")
    print(f"  Output:         {out_dir}")
    print("=" * 60)

    # Load models
    unet, clip, vae = load_models(
        tcfg.comfy_dir, tcfg.model_name,
        tcfg.clip_name, tcfg.clip_type, tcfg.vae_name,
    )
    prompt_config, prompts = load_calibration_prompts(tcfg)
    prompt_sampler = GenerationPromptSampler(prompt_config, artist_pool, tcfg.artist_tags)
    seeds = tcfg.calibration["seeds"]
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
        entry = pdata["entry"]
        tags = [t for t in entry.tags[:4]]
        short = entry.text[:80].replace("\n", " ")
        print(f"    {i:>2}: [{sampler} cfg={cfg_val}]  [{', '.join(tags)}]  {short}...")

    total_runs = len(prompts) * len(seeds) * len(step_variants)
    # Estimate entries: each run produces ~ (steps - 1) × 2 cond slots
    if step_weights and len(step_weights) == len(step_variants):
        ws = step_weights
    else:
        ws = [1.0 / len(step_variants)] * len(step_variants)
    avg_steps = sum(s * w for s, w in zip(step_variants, ws)) / sum(ws)
    est_entries = int(total_runs * (avg_steps - 1) * 2)

    w = tcfg.sampling["width"]
    h = tcfg.sampling["height"]
    permutation = f"{len(prompts)} prompts \u00d7 {len(seeds)} seeds \u00d7 {len(step_variants)} step variants = {total_runs} total generations"

    extra_lines = [
        f"Permutation:   {permutation}",
        f"Est. entries: ~{est_entries} ({int(est_entries/1000)}k) calibration data points",
        f"Est. disk:    ~{est_entries * 300 // 1000}k kB  (JSONL)",
        f"Output dir:   {out_dir}",
    ]
    if refiner_mode != "off":
        refiner_disk = refiner_data.estimate_refiner_disk_bytes(
            total_runs, avg_steps, res_mix or {f"{w}x{h}": 1.0},
            refiner_cfg.get("record_slots", "both"),
        )
        extra_lines.append(
            f"Refiner disk: ~{refiner_disk/1e6:.0f} MB  lossless "
            f"({refiner_cfg.get('record_slots', 'both')} slots, mix-averaged)"
        )
    print_schedule_estimate(
        "Calibration run schedule",
        total_generations=total_runs,
        avg_steps=avg_steps,
        width=w,
        height=h,
        extra_lines=extra_lines,
    )
    print(f"  Press Ctrl+C to abort, or wait 3 seconds...")
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n  Aborted.")
        return

    all_entries: list[CalibrationEntry] = []
    artist_draws: list[list[str]] = []
    run_idx = 0
    total_iterations = 0
    wall_start = time.time()

    data_file = out_dir / "calibration_data.jsonl"

    # ── Refiner recording setup (Task 2d) ──
    refiner_dir = out_dir / "refiner_data"
    refiner_manifest = None
    eval_prompt_ids = set()
    # Resolution assignments: deterministic per generation index (plan Task 1);
    # applies to all calibration generations whenever sampling.resolution_mix
    # is set, independent of refiner mode.
    res_assignments = []
    if res_mix is not None:
        rng = random.Random(_RESOLUTION_RNG_SEED)
        res_assignments = [_draw_resolution(rng, res_mix) for _ in range(total_runs)]
        print(f"  [calibrate] resolutions: "
              f"{ {k: res_assignments.count(k) for k in res_mix} }")
    if refiner_mode != "off":
        refiner_dir.mkdir(parents=True, exist_ok=True)
        refiner_manifest = refiner_data.init_manifest(refiner_cfg)
        refiner_data.save_manifest(refiner_dir, refiner_manifest)
        # Eval prompts: deterministically the LAST eval_prompts selected prompts;
        # their recordings always persist (bypass ranking) and are excluded from
        # training pairs (plan Task 1).
        eval_n = int(refiner_cfg.get("eval_prompts", 6))
        eval_prompt_ids = set(range(max(len(prompts) - eval_n, 0), len(prompts)))
        print(f"  [refiner] recording to {refiner_dir}")
        print(f"  [refiner] eval prompts: {sorted(eval_prompt_ids)} "
              f"(recorded, excluded from training pairs)")

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

                # Resolution mix: deterministic per generation index (plan
                # Task 1, lives in sampling config); absent → base resolution.
                if res_assignments:
                    res_key = res_assignments[run_idx - 1]
                    width, height = refiner_data.parse_resolution(res_key)
                else:
                    width, height = w, h

                # Per-generation prefix/negative variant + artist-tag draw
                full_prompt, neg_prompt, artists = resolve_generation(
                    prompt_sampler, pdata["entry"], pi, seed, steps, width, height,
                )
                artist_draws.append(artists)

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                track_blocks = record_blocks and refiner_mode != "only"
                if refiner_mode == "off":
                    dm, original_fwd = patch_for_calibration(
                        unet, steps, prompt_id=pi, seed=seed,
                        track_per_block=track_blocks,
                    )
                else:
                    dm, original_fwd = patch_for_refiner(
                        unet, steps, prompt_id=pi, seed=seed,
                        track_per_block=track_blocks,
                        refiner_dir=refiner_dir,
                        refiner_cfg=refiner_cfg,
                    )

                try:
                    img = sample(
                        unet, clip, vae, full_prompt,
                        seed=seed, steps=steps,
                        cfg=cur_cfg,
                        sampler_name=cur_sampler,
                        scheduler=cur_scheduler,
                        width=width,
                        height=height,
                        negative=neg_prompt,
                    )
                finally:
                    restore_model(dm, original_fwd, unet)

                dt = time.time() - t0
                total_iterations += steps
                run_entries = list(dm.calibration_log)

                # Tag entries with step variant info
                for e in run_entries:
                    e.total_steps = steps
                    e.step_fraction = e.step / max(steps - 1, 1)
                    e.sampler = cur_sampler
                    e.scheduler = cur_scheduler

                all_entries.extend(run_entries)

                # Save incrementally (skipped in refiner-only mode)
                if refiner_mode != "only":
                    with data_file.open("a") as f:
                        for e in run_entries:
                            f.write(json.dumps(e.to_dict()) + "\n")

                # Refiner generation-end flow: volatility score, top_n eviction,
                # write .bin + .prompt.bin, update manifest (plan 2a/2c)
                if refiner_mode != "off":
                    refiner_data.finalize_refiner_generation(
                        dm, refiner_dir, refiner_manifest,
                        run_meta={
                            "name": f"gen_{run_idx - 1:04d}_p{pi:02d}_s{seed}",
                            "prompt_id": pi,
                            "seed": seed,
                            "sampler": cur_sampler,
                            "scheduler": cur_scheduler,
                            "cfg": cur_cfg,
                            "width": width,
                            "height": height,
                            "steps": steps,
                            "fps": None,
                            "prompt": full_prompt,
                            "negative": neg_prompt,
                            "artists": artists,
                        },
                        refiner_cfg=refiner_cfg,
                        eval_prompt_ids=eval_prompt_ids,
                    )

                valid = [e for e in run_entries if e.out_rel > 0]
                vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
                eta = (dt * (total_runs - run_idx)) / 60.0

                artist_short = artists[0][:24] if artists else "none"
                print(
                    f"[calib] {run_idx}/{total_runs}  "
                    f"p={pi} s={seed} steps={steps}  "
                    f"sampler={cur_sampler} cfg={cur_cfg}  "
                    f"res={width}x{height}  art={artist_short}  "
                    f"took={dt:.1f}s  entries={len(run_entries)}  "
                    f"valid={len(valid)}  VRAM={vram:.1f}GB  ETA={eta:.0f}m"
                )

    # Summary
    wall_elapsed = time.time() - wall_start
    valid_all = [e for e in all_entries if e.out_rel > 0]

    print_speed_summary(
        label="Calibration complete",
        total_generations=total_runs,
        total_iterations=total_iterations,
        wall_seconds=wall_elapsed,
    )
    if artist_pool is not None and artist_draws:
        print_artist_frequencies(artist_draws, artist_pool, len(artist_draws))
    print(f"  Total entries:   {len(all_entries)}")
    print(f"  Valid entries:   {len(valid_all)}  (with out_rel)")
    if record_blocks:
        block_entries = sum(1 for e in all_entries if e.block_cos_sims is not None)
        print(f"  Block entries:   {block_entries}  (per-block cos_sim data)")
    if refiner_mode != "only":
        print(f"  Data saved to:   {data_file}")
    else:
        print(f"  JSONL:           skipped (refiner-only mode)")
    if refiner_mode != "off":
        n_gens = len(refiner_data.iter_generations(refiner_dir))
        print(f"  Refiner data:    {n_gens} generations → {refiner_dir}")
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
    parser.add_argument("--refiner-data", choices=["both", "off", "only"], default=None,
                        help="Refiner latent recording mode (default: config "
                             "refiner.enabled → 'both' / 'off')")
    parser.add_argument("--refiner-top-n", type=int, default=None,
                        help="Keep at most N most-volatile generations per "
                             "resolution (default: config; -1 = keep all)")
    parser.add_argument("--refiner-resolutions", default=None,
                        help="Override sampling.resolution_mix, e.g. "
                             "'512x512:0.8,1024x1024:0.2' (plain 'WxH' entries "
                             "get equal shares)")
    args = parser.parse_args()

    # Overrides
    if args.config is None:
        args.config = str(Path(__file__).parent / "config.json")

    overrides = {}
    if args.prompts is not None:
        overrides["num_prompts"] = args.prompts
    if args.seeds is not None:
        overrides["seeds"] = [int(s) for s in args.seeds.split(",")]
    if args.refiner_resolutions is not None:
        overrides["sampling"] = {"resolution_mix": args.refiner_resolutions}

    refiner_overrides = {}
    if args.refiner_data is not None:
        refiner_overrides["mode"] = args.refiner_data
    if args.refiner_top_n is not None:
        refiner_overrides["top_n"] = args.refiner_top_n
        refiner_overrides["keep_all"] = args.refiner_top_n < 0
    if refiner_overrides:
        overrides["refiner"] = refiner_overrides

    data_file = run_calibration(args.comfy_dir, args.config, overrides=overrides)
    print(f"\nNext step: python -m tuning.optimize --data {data_file}")


if __name__ == "__main__":
    main()
