# TeaCache Tuning Toolkit

Calibration and optimization pipeline for finding optimal TeaCache parameters
for the Anima (Cosmos-Predict2) model.

## Quick Start

```bash
# From the ComfyUI root:
cd /path/to/ComfyUI
PYTHONPATH=".:custom_nodes/ComfyUI-TeaCache-CosmosPredict"
python -m tuning.smoke_test --comfy-dir .
```

## Pipeline

1. **Smoke test** (`smoke_test.py`) — 8 checks: model loading, prompt diversity test, baseline generation, calibration collection, mini-optimizer, TeaCache comparison vs daraskme reference, and 12-metric quality assessment. ~6 min on V100 at 512².
2. **Calibration** (`calibrate.py`) — records per-step delta stats (all 3 sources simultaneously) plus ground-truth output changes. Optionally records per-block cosine similarity for dead-block detection. Shows run schedule with time/disk estimates before starting. 30 min–4 hours.
3. **Optimization** (`optimize.py`) — offline config search. Pre-computes mapping fits (polynomial, power_law, softplus), sweeps candidate thresholds, builds Pareto frontier, fine-tunes winner thresholds. Supports Numba acceleration, cross-validation, signal-space deduplication, and data-driven block param injection. Multi-core CPU, 3–30 min.
4. **Validation** (`validate.py`) — end-to-end quality metrics (PSNR, SSIM, LPIPS, DISTS, MS-SSIM, FSIM, VIF, GMSD, NLPD, PieAPP, VSI) across multiple resolutions (512², 1024², 1024×512) and step counts (20, 30, 40). Configs are selected by uniform error sampling (not just the knee) so the full speedup-vs-quality curve is validated. Baselines are cached per resolution/steps/prompt/seed — ~45% fewer total generations. Results grouped by resolution and step count, with a recommended config per group.
5. **Build presets** (`build_presets.py`) — generates the shipped `anima_presets.json` from the Pareto frontier. Samples evenly-spaced accumulated error levels, each with a full `TeacacheConfig` (source, metric, accumulation, schedule, etc. can differ between control points). The TeaCacheAnima quality slider interpolates threshold between bracketing points and snaps discrete params to the nearest anchor at runtime. Run once after optimization, ship the output with the plugin.

```bash
# Phase 1 — calibration (run on GPU)
python -m tuning.calibrate --comfy-dir .

# Phase 2 — optimization (run on CPU after Phase 1)
python -m tuning.optimize --data outputs/<timestamp>/calibration_data.jsonl

# Phase 3 — validation (run on GPU after Phase 2)
python -m tuning.validate --comfy-dir . \
    --pareto outputs/optimization/pareto_frontier.json \
    --tier 2 --extra-sweep

# Quick validation (3 min)
python -m tuning.validate --comfy-dir . \
    --pareto outputs/optimization/pareto_frontier.json --quick

# Phase 4 — build presets (run on CPU after Phase 2 or Phase 3)
python -m tuning.build_presets \
    --pareto outputs/optimization/pareto_frontier.json \
    --error-min 0.01 --error-max 0.10 --steps 30

# Restrict the error range to conservative→balanced
python -m tuning.build_presets \
    --pareto outputs/optimization/pareto_frontier.json \
    --error-min 0.001 --error-max 0.04 --points 10
```

## Architecture

```
calibrate.py  ──→  calibration_data.jsonl  ──→  optimize.py  ──→  pareto_frontier.json
            │              ↑                                          │
            │    sim_data.py · sim_engine.py · sim_runner.py          │
            │              ↑                                          │
validate.py  ←────────────────────────────────────────────────────────┘
    │                                    │
    └── validation_results.json          └── build_presets.py
                                            │
                                            └── anima_presets.json
                                                    │
                                            TeaCacheAnima node
                                           (auto-loaded at startup)
```

The smoke test runs all four phases on a tiny dataset to verify everything works.

---

## Latent-Space Refiner (Mode B′)

An optional **latent correction model** ("corrector") that post-processes TeaCache
skip-step velocities toward the true velocity: `v_final = v_MA + trust·Δv̂`.
Mode A (the shipped preset behavior) is byte-for-byte unchanged and remains the
default; with a zero-initialized corrector, Mode B′ reproduces Mode A exactly.

### Pipeline stages

```
calibrate.py --refiner-data both|only
    └── outputs/<ts>/refiner_data/         # per-step x_t / v_MA ladder / v_true (lossless blosc2)
probe_refiner.py --data ...                # codec + learnability + Day-1 verdict
train_corrector.py --data ...              # corrector-{size}.safetensors → models/
validate.py --corrector models/corrector-20m.safetensors   # A/B, sanity + shipping gates
distill_corrector.py --teacher ...         # K=3→K=1 turbo (optional)
```

### Recording (plan Task 2)

`--refiner-data {both,off,only}` on `calibrate.py` records, per step per CFG
slot, the raw latent `x_t`, the Mode-A skip reconstruction `v_MA(d)` for every
lag in the ladder (built from a per-slot residual ring buffer — bit-for-bit the
deployment skip construction), and the true velocity `v_true` (the d=0 anchor
is synthesized at load). The recorder runs the full model every step — no
TeaCache decisions — so the data is config-independent.

```bash
python -m tuning.calibrate --comfy-dir . --refiner-data only \
    --refiner-lags "1,2,4,8,16" --prompts 1 --seeds 34635345
```

| Flag | Meaning |
|---|---|
| `--refiner-data` | `both` = scalar JSONL + latent data, `only` = latent data only, `off` = today's behavior |
| `--refiner-top-n` | keep the N most volatile generations per resolution (`-1` = keep all) |
| `--refiner-resolutions` | override `sampling.resolution_mix` (e.g. `512x512:0.8,1024x1024:0.2`) |
| `--refiner-lags` | staleness ladder (default `1,2,4,8,16`) |

Disk (lossless blosc2): ≈ **27 MB per generation @512², both slots** (30 steps:
per step per slot `x_t` 107 KB + 5×`v_MA` 51 KB + `v_true` 51 KB + 2×1.3 MB
prompts). 140 generations ≈ 3.8 GB; 360 all-512² ≈ 9.9 GB. The ring buffer
lives on GPU: ~128 MB (bf16 model) or ~256 MB (fp16 model) @512² both slots.

### Probe (plan Task 3)

```bash
python -m tuning.probe_refiner --comfy-dir .          # record 3–5 gens + analyze
python -m tuning.probe_refiner --data outputs/<ts>/refiner_data   # analyze only
```

Writes `refiner_probe_report.json` next to the data dir: codec bits/element per
tensor type, the t-gap-cancellation ratio (`|Δ_MA|` vs `|v_true(t)−v_true(t−1)|`
— must be < 1), Δ_MA SVD rank, the linear predictability ceiling (1−R² of the
best per-channel affine `v_MA → v_true`), the per-region staleness curve, and
the Day-1 experiment verdict (tiny UNet beats the linear ceiling by ≥25% →
build the full corrector). `train_corrector` reads the ceiling from this
report for its did-it-learn gate.

- **Progress & metrics**: tqdm bars for recording, analysis and Day-1 phases
  (`--no-progress` disables), per-50-step Day-1 log lines (EMA loss, it/s),
  and a final summary with per-phase timings and VRAM peak. Step + eval
  scalars stream to `probe_metrics.jsonl` (JSONL, `tail -f` friendly;
  `--metrics PATH` relocates).
- **Crash resilience**: the report is written progressively (atomic
  tmp+replace) after codec/learnability/gate results and again after the
  Day-1 experiment, so a crash keeps everything computed so far. Failed
  generations during recording are logged and skipped instead of aborting
  the run (all-fail still aborts).
- **Performance**: the analysis phase decodes each generation exactly once
  (single pass feeds the codec benchmark, t-gap cancellation and learnability
  stats instead of three full re-decodes).

### Training (plan Task 6)

```bash
python -m tuning.train_corrector --data outputs/<ts>/refiner_data \
    --multipass 1 --model-size 20m --depth auto --optimizer sophia --max-steps 60000
```

- **Precedence**: `refiner_training` in `tuning/config.json` provides every
  default; CLI flags override config; the effective post-override config is
  snapshotted into checkpoint metadata; `--resume` warns on config drift.
- **Size & depth**: `--model-size` is a *parameter target*, not a fixed
  architecture — `5M`, `20m`, `50M`, `1.5B` (K/M/B suffixes) or `tiny` (the
  Day-1 probe UNet, ~1.83M). The width ladder is solved to land within ±5%
  of the target while keeping the architecture profile intact (channels
  1:2:4, bottleneck = 4× the first stage, head_dim ≈ 32, mlp_ratio 4, fixed
  prompt/cond widths). `--depth {auto,1..8}` sets the DiT bottleneck block
  count: `auto` grows width first and only adds blocks past the canonical
  ceiling (bottleneck ≈ 640); a manual depth scales the widths down to
  compensate (`w ∝ 1/√depth` at fixed params). The default `20m` ≈ 20M
  params. The achieved count is printed at construction and stored in
  checkpoint metadata (`target_params`/`achieved_params`); warnings fire
  when the target can't be hit within tolerance, when it needs depth past
  the family ceiling, or when one pass approaches ~1% of a full model step
  @512² (the corrector stops being a cheap post-hook).
- **Multipass**: K≤3 on-policy deep supervision (K curriculum 1→3, masked-K
  batching, stop-grad between passes). Inference default K=1.
- **Optimizers**: `sophia` (Gauss-Newton-Bartlett, lr 4e-4, ρ=0.04, k=10 —
  implemented in `train_corrector.py` per the plan deep-dive), `adamw`,
  `ademamix`, `schedulefree`. fp16 autocast + GradScaler throughout.
- **Stale-only training**: the corrector post-processes skip-step (stale)
  v_MA only, so training pairs are the recorded ladder lags (1,2,4,8,16) —
  the synthesized d=0 anchors (v_MA = v_true) are **excluded** from training
  (off-manifold: the model is never queried on a fresh cache). The eval set
  keeps them as a diagnostic (`per_lag[0]`, `anchors_only` — a rise there is
  expected and harmless; never gated on). The did-it-learn gate and the
  best-checkpoint selection use the ladder-only K=1 error.
- **Eval every 500 steps**: per-lag rel-MSE slices at K=1,2,3, plus the gates —
  did-it-learn (ladder-only K=1 beats the probe ceiling by ≥20% within 6
  evals), K-robustness gap (ladder-only) and per-lag coverage warnings
  (growth measured from lag 1, not the d=0 anchors).
- **Data path & GPU utilization**: generation decode/collate runs in
  DataLoader worker processes (`refiner_training.num_workers`, default 4 —
  each worker partitions the generation LRU, so the effective cache is
  workers × `cache_size`) or, RAM-economically, in a single producer thread
  (`--prefetch-queue N` — pinned bounded queue, one shared LRU; 0 = main
  thread). Per-50-step log lines and `train_metrics.jsonl` step rows carry
  `data_ms` (fetch+H2D) vs `gpu_ms` (CUDA-event measured), and the final
  summary prints the data share of data+GPU — if data is a large fraction,
  raise `num_workers` or `--prefetch-queue`. Every eval also prints a
  **step-time breakdown since the last eval** (and writes it as the `timing`
  key of the eval row): per-phase GPU time (augment / forward / backward /
  Sophia hessian / optimizer / EMA — CUDA events around each phase), the
  fetch-wait vs H2D vs other CPU split, GPU-busy %, GPU time per resolution
  bucket, average K_max / micro-batches / hessian steps, the generation-LRU
  cache hit % (shared counters, worker-safe) with the mean miss-decode cost,
  and the average producer-queue depth at `--prefetch-queue` gets. This is
  the debugging view for "what is slowing training down" — a low hit % means
  `cache_size` too small for the de-burst window, a high fetch-wait with
  `--prefetch-queue` means the producer can't keep up, a high `other` means
  CPU-side bookkeeping (`.item()` syncs, scheduler, python overhead).
  Checkpoint saves (.safetensors
  + full-state .pt) snapshot on the main thread and write in background
  threads (tmp+replace), so eval-time disk I/O doesn't idle the GPU.
- **Progress & metrics**: live tqdm bar (EMA loss, lr, K, it/s, remaining),
  per-50-step log lines with ETA, and a per-eval timing report (train / eval /
  hessian phases; the ETA projects eval + checkpoint overhead). Step + eval
  scalars go to `train_metrics.jsonl` (JSONL) next to the checkpoints
  (`--no-progress` disables the bar; `--metrics PATH` relocates the log;
  `--resume` reports the previously elapsed wall time).
- **Outputs** (`models/`): `corrector-{size}-{step}.safetensors` per eval,
  `corrector-{size}-best.safetensors` (best ladder-only K=1),
  `corrector-{size}-train.pt` (resume state), final `corrector-{size}.safetensors`
  (EMA, fp16, K_recommended=1).

Distillation (optional, plan Task 6l):

```bash
python -m tuning.distill_corrector --teacher models/corrector-20m.safetensors \
    --data outputs/<ts>/refiner_data --student-size 5m --mode gkd
# → models/corrector-5m-turbo.safetensors (K_recommended=1)
```

### Grain-reduction training knobs

Two training-side knobs target the measured failure modes of the 30M model
(grain-like high-frequency residual error, and systematic per-channel
under-prediction of velocity energy). Shipped defaults (config.json) are the
"smooth, not overtaking" values — each knob alone stays well below the main
loss in influence:

| knob | default | meaning |
|---|---|---|
| `spectral_penalty` | `0.01` | spectral (grain) penalty weight λ |
| `spectral_mode` | `fraction` | `fraction` \| `absolute` normalization |
| `calibrate_gains` | `2048` | post-training gain calibration on N eval pairs |
| `gain_strength` | `0.5` | blend s of the calibrated gains at inference |

- **Spectral penalty** — `--spectral-penalty <λ>`: adds
  `λ · ‖HP(v̂ − v_true)‖²/‖v̂ − v_true‖²` per training sample (differentiable
  2×2 pooled high-pass). This is the exact metric the refiner verifier's
  spectral grain test measures — the 30M corrector's residual error carries
  3–17× the high-frequency share of the latent signal itself, which decodes
  as grain. The `fraction` mode is scale-invariant (it tilts the error's
  spectral shape without competing with the main loss on magnitude —
  "if you must be wrong, be wrong softly"); `absolute` also penalizes HF
  error energy relative to the target.
  **Scale guide** (measured on real pairs): the typical per-sample rel-MSE
  is ≈ 0.06 and the spectral fraction ≈ 0.64, so the penalty adds roughly
  `λ · 10×` percent to the main loss — λ=0.01 ≈ +10% (gentle tilt, the
  shipped default), λ=0.05 ≈ +50% (already substantial; pair it with the
  verifier's spectral + LPIPS/DISTS no-regression gates, since
  overcorrection reads as blur, not grain).
- **Per-channel gain calibration** — `--calibrate-gains <N>`: after
  training, calibrates 16 energy-matching scales g_c = ‖v_true_c‖/‖v̂_c‖ on
  N eval pairs (lag ≥ 1 only — the deployed skip-step regime; anchors are
  excluded because the corrector never runs on fresh steps), reports the K1
  rel-MSE with/without the gains per lag, and embeds the gains + strength
  into the saved checkpoint. At inference `refine()` applies
  `v̂′ = v̂ · (1 − s + s·g)` with `--gain-strength` (config `gain_strength`).
  The 30M model under-predicts per-channel energy by ~1–14% (gains ≈
  1.01–1.07); measured effect on rel-MSE is neutral (the loss already
  optimizes scale) — the value is perceptual/spectral, A/B-able via the
  verifier on calibrated vs uncalibrated checkpoints.
  **N guide**: the gains are energy ratios over a channel-wise sum — 2k
  pairs (the shipped default) is comfortably stable and costs a few seconds
  at training end; 256–512 is enough for a rough estimate, 8k+ gives
  diminishing returns. `--gain-strength` is the safety dial: `s=0` disables
  the calibration entirely (byte-identical to uncalibrated behavior),
  `s=1` fully applies the energy matching, `s=0.5` (default) applies half —
  meaningfully compensating the under-prediction while staying conservative
  against the calibration's own sampling noise and against over-amplifying
  noise in the low-energy channels.
  **Composition with trust**: the gains are applied inside `refine()` to the
  refined velocity **before** the deployment trust blend
  (`v_final = v_MA + trust·(v̂′ − v_MA)`), so at `trust < 1` the calibration
  is attenuated together with the correction — the two compose exactly only
  at trust = 1.

### Optimizer & convergence findings (why 96K steps, and how to need fewer)

The shipped 30M model trained for 96K steps (Sophia, lr 4e-4, warmup 5% +
cosine to 96K). Its own eval curve shows the two costs of that recipe: the
best ladder K1 was reached at **step 72K** (0.2816) and the final 96K eval
lands on it (0.2823) — **the last ~25% of the run bought nothing**, because
(a) cosine-to-96K had already annealed the LR to ~15% of peak by step 72K,
and (b) Sophia at 4e-4 moves conservatively. Both are fixable without any
architecture change.

#### Evidence: optimizer A/B on a matched-architecture task

A synthetic but *learnable* teacher task was built to compare optimizers on
this exact model family (fresh `CorrectorUNet2D` student per run, same
warmup/cosine machinery, same per-sample rel-MSE loss, same lag set, fp16
autocast; teacher = fixed random 1×1-conv correction scaled by (t, lag) +
tanh nonlinearity, so the task generalizes). Held-out rel-MSE:

| optimizer | ~500 steps | ~1000 steps | ~2000 steps | notes |
|---|---|---|---|---|
| sophia lr 4e-4 *(current)* | 0.622 | 0.616 | — | barely moves on this task |
| sophia lr 1e-3 | 0.56 | 0.542 | — | better, still slow |
| adamw lr 4e-4 | 0.409 | 0.353 | — | solid baseline |
| schedulefree lr 4e-4 | 0.490 | 0.433 | — | lags tuned cosine (as the paper predicts) |
| ademamix lr 4e-4 | 0.371 | 0.305 | — | ~1.2–1.5× over AdamW, stable |
| ademamix lr 1e-3 | 0.24 | 0.221 | nan | faster but diverges long-horizon |
| muon lr 0.02 adjusted | 0.104 | **0.066** | nan | fastest, unstable late |
| **muon lr 0.01 adjusted** | 0.13 | 0.062 | **0.0615** | fast **and** stable |
| muon lr 0.005 adjusted | 0.15 | ~0.07 | 0.0632 | same plateau, slightly slower start |

Takeaways, in order of confidence:

1. **Muon (`--optimizer muon --muon-lr 0.01`) is the step-efficiency winner**:
   ~5× better held-out error at 1000 steps than the current Sophia@4e-4
   recipe, stable to 2000 steps. This matches the published Muon study
   (arXiv 2509.24406, exactly the 30M–200M scale): "reaches the target loss
   with 48–52% of the training compute of AdamW", with extra data-efficiency
   at large batch sizes (pair with `--batch-size 64` and the existing
   `accumulate_big_batches`). Expect the 72K-best quality in roughly
   **25–35K steps (2.5–3× fewer)**.
2. **Muon is LR-sensitive**: 0.02 diverged (nan) at ~1500 steps in the A/B.
   The `use_adjusted_lr` rule (Kimi-K2 per-dimension scaling, on by default)
   is what makes 0.005–0.01 stable. Start at 0.01, drop to 0.005 if the
   beacon looks hot. The trainer wires Muon as two param groups (2D matrices
   → Muon with Newton–Schulz; biases/norms/1D → AdamW with decoupled decay).
3. **AdEMAMix (`--optimizer ademamix`, lr 4e-4) is the safe fallback**:
   consistently ahead of AdamW at every step and — per the paper
   (arXiv 2409.03137) — ~2× data-efficient with *slower forgetting*, which
   matters because the training corpus is small and replayed ~20×. Do **not**
   raise its lr to 1e-3 (diverged at 2000 steps in the A/B).
4. **Shorten the schedule**: with either optimizer, `--max-steps 40000–48000`
   recovers the wasted cosine tail. The 96K curve was flat after 72K because
   the LR was already in the noise floor; a 48K cosine spends the same budget
   while the model still has room to move. (WSD's river-valley analysis —
   arXiv 2410.05192 — explains exactly this plateau-and-cooldown shape.)
5. **Sophia at 4e-4 is conservative, not bad**: it trained the shipped model
   fine; the A/B just says its per-step efficiency at that LR is low. If you
   stay on Sophia, raise lr to ~1e-3 (2e-3 diverged) before concluding
   anything about the optimizer.

#### Recommended run recipe (faster convergence)

```bash
python -m tuning.train_corrector --data outputs/<ts>/refiner_data \
    --optimizer muon --muon-lr 0.01 \
    --max-steps 40000 --batch-size 64 \
    --beacon-every 500 --beacon-lr-patience 3 \
    --spectral-penalty 0.01 --calibrate-gains 2048 --gain-strength 0.5
```

- Muon converges fast but late runs can still go hot — the beacon is the
  stability watchdog: keep its frequency (every 500 steps, the default) and
  halve the patience (6 → 3) so an LR cut happens at ~1.5K steps of plateau
  instead of ~3K. The beacon reduces LR on plateaus; it is not the primary
  schedule (the cosine is).
- Keep `--eval-every 2000`-ish so the Muon run is watched closely; the
  geometric eval schedule can stay on top of it.
- If anything looks unstable, first drop `--muon-lr` to 0.005 (stability
  costs almost nothing at the 2000-step horizon in the A/B).

#### Not yet implemented (larger levers)

- **Deployed-weighted pair sampling**: weight training pairs by the measured
  `P(skip ∧ lag ∧ region)` from the verifier's skip logs (conservative
  control points skip at lag-1 only; max-error control points reach lag 24,
  beyond the trained ladder of 16 — and the current `lag_weights`
  [2.5, 1.5, 1.0, 0.5, 0.25, 0.125] — the 30M run's; the config.json default
  is [2.0, 1.5, 1.25, 1.0, 0.5, 0.25] — actually *down-weight* the hard lags).
  Making every step
  count toward the deployed metric is a task-level win on top of the
  optimizer gains.
- **DAgger-style on-policy refresh**: record new generations *with the
  corrector active in the loop* and train on the states it actually visits.
  Fixes the off-policy distribution shift (today's corpus was recorded with
  the base model only) and mines the model's own hard cases. Highest ceiling,
  needs an iteration loop around the existing recording infra.
- **Lag curriculum**: introduce lags 8/16 after ~50% of steps (classic
  easy→hard); pairs well with any of the optimizers above.

### Inference knobs (plan Task 5)

Both nodes (`TeaCacheAnima`, `TeaCacheAnimaAdvanced`) gain optional inputs:

- `corrector`: `off` | `latent_denoiser` (empty/missing `corrector_model`
  falls back to Mode A with a warning)
- `corrector_model`: dropdown of `.safetensors` checkpoints, populated from
  `ComfyUI/models/teacache_correctors/` (global) and the repo's `models/`
  directory (the default `--out` of `train_corrector.py`); legacy workflows
  with absolute paths still resolve
- `refine_passes`: K (1–4, default 1)
- `corrector_trust`: `v_final = v_MA + trust·(v̂ − v_MA)` (default 1.0)

The info string shows `mode=B(K=…)` vs `mode=A`. The corrector is compiled
standalone (`torch.compile`, "reduce-overhead", cached per path+device; set
`TEA_CACHE_NO_COMPILE=1` to disable). A warning is printed when the model has
patches/LoRA (weights were trained on anima-base-v1.0 outputs).

### A/B validation + gates (plan Task 7)

```bash
python -m tuning.validate --comfy-dir . --pareto outputs/optimization/pareto_frontier.json \
    --corrector models/corrector-20m.safetensors --corrector-passes 1
```

With `--corrector`, every validated config also runs in Mode B′; the report
contains the per-(resolution, steps) A/B table (LPIPS/skip-rate/speedup, per
config), per-slot correction magnitudes (surfaces CFG amplification of
uncond-slot errors), the skip-run length distribution (`validation_drift.json`),
the **sanity gate** (zero-init Mode B′ ≡ Mode A bitwise) and the **shipping
gate** (Mode B′ ≥ Mode A LPIPS in ≥70% of cells AND ≥5% better on the
aggressive half of the threshold curve). Only a passing shipping gate makes
Mode B′ a default for new users — until then it is an opt-in experimental toggle.

### Verifier split: base TeaCache vs refiner

`validate.py` is the **base TeaCache** verifier (skip rates, speedup, image
metrics vs baselines). The **refiner side** has its own suite:

```bash
python -m tuning.verify_refiner --comfy-dir /path/to/ComfyUI \
    --corrector models/corrector-30m-96000.safetensors \
    --config-errors 0.020,0.054 --out-dir refiner_verify
```

Per prompt (built-in in-distribution + OOD set — logo / flat illustration /
poster / line-art / low-poly, or `--prompts` JSON) × control point it runs
baseline / Mode A / Mode B′ (plus optional trust-map A/B and
`--sanity-zero-init`), and one full-model recording per prompt. It reports:

- **velocity level** (from the recording): per-(t-region, lag) K0/K1 rel-MSE
  ladder, per-channel recovery + uniformity (the "14 good, 2 bad" gate;
  flags channels < 50% recovery), d=0 anchor perturbation (diagnostic — the
  corrector never runs on fresh steps in deployment, but a future pipeline
  change could expose it), and the spectral grain test (HF share of the K1
  residual vs the velocity signal, per region).
- **deployed-weighted recovery**: the Mode-A skip log (exact per-step lag
  decisions, instrumented with both `current_percent` and `tc_current_percent`
  — an easy wiring bug that silently pins the hook to the early region) × the
  recording's per-(region, lag) errors → the expected per-skip-step error of
  A vs B′. This is the number that matches what users see; pooled ladder
  rel-MSE overstates it at aggressive control points. Also reports the max
  deployed lag vs the checkpoint's trained ladder (the 30M corpus ladder is
  [1, 2, 3, 4, 8, 16] while max-error deployment reaches lag 24; deployed
  lags between recorded ladder entries use the next recorded lag at or above
  them — error grows with lag — and beyond the deepest recorded lag they
  plateau there). The recording ladder defaults to the checkpoint's own
  trained ladder (read from its embedded `config_snapshot`); override with
  `--record-lags`.
- **pixel level**: full pyiqa suite A-vs-base and B′-vs-base, with a
  no-regression gate on LPIPS/DISTS/VIF vs Mode A, the final-latent spectral
  test, optional per-channel LPIPS ablation (`--perceptual-channel-ablation`),
  and baseline/A/B′ comparison PNGs (`--png`).
- **harness self-tests**: the corrector-alive probe (a zero-delta corrector
  fails it — the A/B latent-divergence check alone cannot detect a dead
  corrector once gain calibration is embedded, since the gains rescale the
  latent even for a zero-output model), the A/B latent-divergence wiring
  check, and `--sanity-zero-init` (a fresh zero-init corrector must reproduce
  Mode A byte-for-byte).

Known results for the shipped 30M checkpoint (512², 30 steps, cfg 5.5):
in-dist prompts pass the mid control point clean (≈76% deployed-weighted
recovery, no perceptual regressions) but regress LPIPS/DISTS at the max-error
control point; the OOD set regresses all 8 metrics at the conservative control
point — OOD style drift is a real gate, not noise.

---



## Phase 4: Build Presets

`build_presets.py` converts the Pareto frontier into a file the TeaCacheAnima node can read at ComfyUI startup.

### What it does

1. Loads `pareto_frontier.json` from Phase 2 (or Phase 3 if available).
2. Samples `N` evenly-spaced accumulated error levels across the frontier.
3. For each level, picks the nearest Pareto-optimal config and stores its full `TeacacheConfig` (source, metric, accumulation, schedule, block mode, etc.).
4. Writes `anima_presets.json` — the file the TeaCacheAnima quality slider reads.

### How the slider maps to error

The slider (quality 0–100) maps linearly to the accumulated error range defined by `--error-min` and `--error-max`:

```
quality=0   → error=error_min  → near-lossless (LPIPS display: ~error×lpips_scale)
quality=50  → error=midpoint   → balanced (~1.3× speedup)
quality=100 → error=error_max  → max speed (~2× speedup)
```

Between control points, **threshold** is linearly interpolated. **Discrete params** (source, accumulation type, step schedule, etc.) snap to the nearest control point halfway through each bracket. This means the config can switch from `pooled_latent` + `carry_over` at low error to `t_emb` + `leaky` at high error — whatever the Pareto frontier found optimal at each speedup level.

### Usage

```bash
python -m tuning.build_presets \
    --pareto outputs/optimization/pareto_frontier.json \
    --error-min 0.01 --error-max 0.10 \
    --steps 30 --points 8
```

| Flag | Default | Description |
|------|---------|-------------|
| `--pareto` | *(required)* | Path to `pareto_frontier.json` (Phase 2 output) |
| `--error-min` | from data | Minimum accumulated error (quality=0). Leftmost slider position. |
| `--error-max` | from data | Maximum accumulated error (quality=100). Rightmost slider position. |
| `--steps` | *(none)* | Step count the presets were calibrated for (stored as `_steps` metadata). Threshold auto-scales by `preset_steps / user_steps` at runtime. |
| `--points` | 8 | Number of control points to sample. More = smoother threshold curve. |
| `--lpips-scale` | 6.0 | Multiplier for `error → LPIPS` display hint in the node. Adjust based on validation results. |
| `--output` | `<project>/anima_presets.json` | Where to write the preset file. |

### Where the output goes

The default output path is `anima_presets.json` in the project root — the same directory as `nodes_anima.py`. This is where the node expects to find it at startup:

```
ComfyUI-TeaCache-CosmosPredict/
├── anima_presets.json   ← build_presets.py writes here by default
├── nodes_anima.py        ← TeaCacheAnima node reads it from here
├── tuning/
│   ├── build_presets.py  ← generates it
│   └── ...
```

After running `build_presets.py`, copy or symlink the output to the project root if you used `--output` elsewhere. The node auto-detects whether the file uses the new error-anchored format (`control_points` at top level) or the old hand-crafted format (`quality_zones`) and handles both.

### Preset file format (auto-generated)

```json
{
    "_description": "Auto-generated TeaCache presets...",
    "_steps": 30,
    "_error_range": [0.01, 0.10],
    "_lpips_scale": 6.0,
    "control_points": [
        {
            "error": 0.01,
            "config": {
                "source": "pooled_latent",
                "metric_type": "mean_only",
                "signal_scale": 1.0,
                "mapping_type": "identity",
                "accumulation_type": "carry_over",
                "rel_l1_thresh": 0.035,
                "step_schedule": "bell",
                "block_mode": "all_or_nothing",
                "residual_strategy": "hard"
            },
            "speedup": 1.02
        },
        {
            "error": 0.04,
            "config": {
                "source": "t_emb",
                "accumulation_type": "leaky",
                "rel_l1_thresh": 8.5,
                "step_schedule": "linear_ramp"
            },
            "speedup": 1.52
        }
    ]
}
```

Control points can have entirely different architectures — the Pareto frontier naturally surfaces the best config for each error regime. The slider interpolates threshold smoothly; the architecture switches at bracket midpoints.

---

## Configuration Reference

All settings are in `config.json`, loaded by `config_types.TuningConfig`.

### Top-level

| Field | Type | Description |
|-------|------|-------------|
| `comfy_dir` | string | Path to ComfyUI installation |
| `model_name` | string | UNet checkpoint filename |
| `clip_name` | string | CLIP model filename |
| `clip_type` | string | CLIP model type (e.g. `qwen_image`) |
| `vae_name` | string | VAE model filename |
| `output_dir` | string | Base output directory (default: `outputs`) |

### Sampling (`sampling`)

Controls image resolution, base sampling parameters, and calibration variety.

```json
"sampling": {
    "default_steps": 30,
    "step_variants": [25, 28, 30, 35, 40],
    "step_weights": [0.05, 0.10, 0.70, 0.10, 0.05],
    "cfg": 5.0,
    "sampler": "er_sde",
    "scheduler": "normal",
    "width": 512,
    "height": 512,
    "sampler_variants": ["er_sde", "dpmpp_2m_sde", "euler_a"],
    "scheduler_variants": ["normal", "simple"],
    "cfg_variants": [4.0, 4.5, 5.0, 5.5]
}
```

| Field | Description |
|-------|-------------|
| `default_steps` | Default step count (used by validation) |
| `step_variants` | Step counts tested during calibration |
| `step_weights` | Probability weights per step variant (same length as `step_variants`) |
| `cfg` | Default CFG scale |
| `sampler` / `scheduler` | Default sampler/scheduler (used by validation) |
| `width` / `height` | Image resolution — 512 is 4× faster than 1024² |
| `sampler_variants` | Samplers cycled per prompt for calibration diversity |
| `scheduler_variants` | Schedulers cycled per prompt |
| `cfg_variants` | CFG values cycled per prompt |

### Calibration (`calibration`)

```json
"calibration": {
    "prompts_file": "prompts/calibration.json",
    "prompt_selection": "semantic_diversity",
    "prompt_tag_filter": [],
    "num_prompts": 30,
    "seeds": [34635345, 53453634, 267454, 123],
    "negative_prompt": "",
    "record_block_data": true
}
```

| Field | Description |
|-------|-------------|
| `prompts_file` | Path to prompt JSON (relative to `tuning/`) |
| `prompt_selection` | Selection strategy (see Prompt Selection below) |
| `prompt_tag_filter` | Tag-based include/exclude filter (see Prompt Filtering below). `[]` = use all prompts |
| `num_prompts` | Number of prompts from the pool |
| `seeds` | Random seeds for reproducibility |
| `negative_prompt` | Fallback negative prompt (overridden by prompt-level `negative`) |
| `record_block_data` | When `true`, records per-block cosine similarity between steps. Required for `dynamic` block mode and `per_group` block level. Adds ~2-4 GB VRAM |

### Optimization (`optimization`)

The full search space for the offline optimizer. Every field controls which `TeacacheConfig` dimensions are swept.

```json
"optimization": {
    "sources": ["t_emb", "first_block_shift", "pooled_latent"],
    "pooled_latent_mode": "mean",
    "metric_types": ["mean_only", "mean_and_max", "mean_max_std"],
    "metric_weights_scenarios": [
        {"mean": 1.0},
        {"mean": 0.7, "max": 0.3},
        {"mean": 0.5, "max": 0.3, "std": 0.2}
    ],
    "mapping_types": ["identity", "polynomial", "power_law", "softplus"],
    "poly_degrees": [3, 4, 5],
    "accumulation_types": ["hard_reset", "carry_over", "leaky", "windowed"],
    "accumulation_params": [
        {},
        {"param": "leak_factor", "start": 0.50, "end": 0.99, "steps": 12, "spacing": "linear"},
        {"param": "leak_factor", "start": 0.991, "end": 0.999, "steps": 3, "spacing": "linear"},
        {"param": "window_size", "start": 2, "end": 15, "steps": 12, "spacing": "linear"}
    ],
    "residual_strategies": ["hard"],
    "step_schedules": ["constant", "cosine", "linear_ramp", "linear_decay", "bell"],
    "signal_scales": {
        "t_emb": [10, 50, 100, 200],
        "first_block_shift": [1.0],
        "pooled_latent": [1.0, 10, 50]
    },
    "auto_scale_target": 1.0,
    "block_modes": ["all_or_nothing", "split_fraction", "split_groups", "dynamic"],
    "block_levels": ["unified", "per_group"],
    "block_level_config_scope": ["accumulation_type", "step_schedule"],
    "cosim_thresholds": [
        {"param": "cosim_threshold", "start": 0.85, "end": 0.99, "steps": 5, "spacing": "linear"}
    ],
    "block_params_scenarios": {
        "split_fraction": [
            {"param": "always_fraction", "start": 0.05, "end": 0.70, "steps": 10, "spacing": "linear"}
        ]
    },
    "candidate_thresholds": [
        {"param": "thresh", "start": 0.005, "end": 0.50, "steps": 12, "spacing": "geom"}
    ],
    "pareto_threshold_range": [0.001, 10.0],
    "pareto_threshold_count": 500,
    "quality_scoring": {
        "type": "thresholded_power",
        "target": 0.05,
        "power": 8.0,
        "target_quality": 0.9
    },
    "cross_validate": true,
    "cv_holdout_fraction": 0.2,
    "max_candidates": 0
}
```

#### Source signals (`sources`)

| Value | Description |
|-------|-------------|
| `t_emb` | Timestep embedding — smallest signal, benefits from signal scaling |
| `first_block_shift` | AdaLN shift modulation from the first transformer block — best-performing signal on Anima |
| `pooled_latent` | Spatial mean of the patchified latent — resolution-independent ratio metric |

#### Pooled latent mode (`pooled_latent_mode`)

When `source` is `pooled_latent`, this controls how the latent is aggregated:

| Mode | Description | Speed |
|------|-------------|-------|
| `mean` (default) | `x.mean(dim=(1,2,3))` — single CUDA reduction | **10-20× faster** |
| `fixed_grid` | `AdaptiveAvgPool2d` to 16×16 grid — permute + pool2d + reshape | Slow |

**Why `mean` is resolution-independent**: The TeaCache distance metric is `rel_l1 = |curr - prev| / |prev|`. Both numerator and denominator are computed from the same pooled tensor, so the token count cancels out — the ratio is insensitive to resolution.

The `pooled_latent_mode` is automatically injected into each candidate's `mapping_params` by the optimizer (`optimize.py:617-620`).

#### Metric types (`metric_types`)

| Type | Formula | Description |
|------|---------|-------------|
| `mean_only` | `stats["mean"]` | Single scalar |
| `mean_and_max` | `w_mean·mean + w_max·max` | Balances average and outlier sensitivity |
| `mean_max_std` | `w_mean·mean + w_max·max + w_std·std` | Full distribution awareness |

#### Metric weights scenarios (`metric_weights_scenarios`)

Weight distributions swept for multi-component metrics. The optimizer automatically skips scenarios that don't match the metric type (e.g., won't test `{"mean": 0.7, "max": 0.3}` for `mean_only`).

#### Mapping types (`mapping_types`)

| Type | Formula | Parameters |
|------|---------|------------|
| `identity` | `distance` | None |
| `polynomial` | `poly1d(coefficients, distance)` | `poly_degree` (fitted from calibration data) |
| `power_law` | `k · (distance + ε)^α` | `k`, `α` (fitted via log-log OLS) |
| `softplus` | `ln(1 + exp(k·(distance - offset)))` | `k`, `offset` (fitted via `scipy.optimize.curve_fit` or grid search) |

`poly_degrees` (default: `[3, 4, 5]`) controls which degrees are tested for polynomial mapping. Each degree produces a separate candidate with its own fitted coefficients.

#### Accumulation types (`accumulation_types`)

| Type | Behavior | Config param |
|------|----------|-------------|
| `hard_reset` | `acc += pred`; if `acc ≥ thresh` → reset to 0, recalculate | — |
| `carry_over` | `acc += pred`; if `acc ≥ thresh` → subtract threshold, recalculate | — |
| `leaky` | `acc = acc·leak_factor + pred`; if `acc ≥ thresh` → reset to 0 | `leak_factor` |
| `windowed` | Rolling window; average triggers recalculation when window is ≥ half full | `window_size` |

#### Sweep spec format

`accumulation_params`, `cosim_thresholds`, `candidate_thresholds`, and `block_params_scenarios` all use a shared sweep specification:

```json
{
    "param": "parameter_name",
    "start": 0.50,
    "end": 0.99,
    "steps": 12,
    "spacing": "linear"
}
```

| Key | Description |
|-----|-------------|
| `param` | Name of the parameter being swept |
| `start` | Start value (inclusive) |
| `end` | End value (inclusive) |
| `steps` | Number of sweep points |
| `spacing` | `"linear"` or `"geom"` (geometric/log-spaced) |

An empty dict `{}` is a "no-param" concrete value. The optimizer automatically pairs valid param values with their associated type (e.g., `leak_factor` only with `leaky`).

#### Signal scales (`signal_scales`)

Per-source list of multipliers applied to raw distances before mapping. `t_emb` typically needs scaling (10-200×) since its delta values are very small; `first_block_shift` and `pooled_latent` are usually unscaled.

#### Auto scale target (`auto_scale_target`)

When set, the optimizer computes a data-driven scale factor for each source so the average distance approaches `auto_scale_target`. This adds an extra candidate scale alongside the explicit `signal_scales` list. Set to `null` to disable.

#### Block modes (`block_modes`)

| Mode | Description | Key params |
|------|-------------|------------|
| `all_or_nothing` | Cache all blocks or none — single residual | — |
| `split_fraction` | First N% blocks always run; rest are cacheable. Two residuals: early and late | `always_fraction` |
| `split_groups` | Blocks auto-partitioned into 3 groups by architectural role (embedding, spatial, context). `always_groups`/`cache_groups` decide which are always-run | `always_groups`, `cache_groups`, `cosim_threshold` |
| `dynamic` | Per-step dead-block detection via cosine similarity. Blocks with cos_sim above `cosim_threshold` are skipped | `cosim_threshold`, `sensitivity` |

When per-block cosine similarity data is available (`record_block_data: true`), `split_groups` and `dynamic` modes auto-classify block groups via data-driven thresholds. Without this data, `dynamic` mode is excluded from candidate generation.

#### Block levels (`block_levels`)

| Level | Description |
|-------|-------------|
| `unified` | Single accumulator for all blocks (standard) |
| `per_group` | Independent accumulators per block group. Each group gets its own accumulation type, params, and step schedule. Requires `dynamic` mode and `record_block_data: true` |

#### Block level config scope (`block_level_config_scope`)

Controls which parameters vary independently per group in `per_group` mode. List of param names (e.g., `["accumulation_type", "step_schedule"]`) or `["*"]` for all.

#### Cosine similarity thresholds (`cosim_thresholds`)

Swept for `split_groups` and `dynamic` block modes. Controls the similarity threshold above which a block or block group is considered cacheable.

#### Block params scenarios (`block_params_scenarios`)

Per-mode parameter sweeps. For example, `split_fraction` sweeps `always_fraction` from 0.05 to 0.70.

#### Candidate thresholds (`candidate_thresholds`)

Tested on **every** candidate in Phase 1. Default: 12 geometrically-spaced values from 0.005 to 0.50.

#### Pareto threshold range / count

Phase 2 fine sweep for Pareto winners only: `pareto_threshold_range` (log-spaced) × `pareto_threshold_count` (default: 500).

#### Quality scoring (`quality_scoring`)

Controls how simulated error maps to quality (0–1). The scoring function determines which configs the Pareto frontier favors.

| Type | Formula | Description |
|------|---------|-------------|
| `linear` | `1/(1+error)` | Mild penalty — speedup dominates |
| `exponential` | `exp(-error/target)` | error = target → quality = 0.37 |
| `gaussian` | `exp(-0.5·(error/target)²)` | error = target → quality = 0.61 |
| `step` | `1.0 if error < target else 0.0` | Hard cutoff |
| `power` | `1/(1+(error/target)^power)` | Configurable steepness |
| `thresholded_power` | `1/(1+(max(0,error-target)/target)^power)` | **Default** — zero penalty below target, power-law ramp above |

`thresholded_power` additional params:
- `power` (default: `8.0`) — steepness of penalty beyond target
- `target_quality` (default: `0.9`) — quality score at `error == target`. When < 1.0, the effective target is shifted so quality = target_quality at the configured target value

**Effect at example error levels (target=0.05, thresh_power with power=8, target_quality=0.9):**

| error | linear | exponential | gaussian | step | power(2) | thresh_power(8) |
|-------|--------|-------------|----------|------|----------|-----------------|
| 0.001 | 0.999 | 0.980 | 1.000 | 1.0 | 1.000 | 1.000 |
| 0.025 | 0.976 | 0.607 | 0.882 | 1.0 | 0.800 | 1.000 |
| 0.050 | 0.952 | 0.368 | 0.607 | 1.0 | 0.500 | 0.900 |
| 0.100 | 0.909 | 0.135 | 0.135 | 0.0 | 0.200 | 0.012 |

#### Cross-validation

When `cross_validate: true`, calibration data is split by `prompt_id`: a holdout fraction (`cv_holdout_fraction`, default: 0.2 / 20%) is reserved for evaluation. The training set is used for mapping fits and simulation. The holdout set gives honest error estimates not inflated by fitting to the same prompts.

#### Max candidates (`max_candidates`)

Cap on total configs. `0` = unlimited. When exceeded, configs are randomly sampled with seed 42 for a representative subset.

### Validation (`validation`)

Configurations are selected by **uniform error sampling** across the Pareto frontier — not just the knee. This validates the full speedup-vs-quality curve from conservative (near-lossless) to aggressive (fastest, some quality loss). Baselines are precomputed once per `(resolution, steps, prompt, seed)` and reused across all configs (~45% fewer generations).

Validation tests multiple resolutions and step counts to verify generalization beyond calibration settings (which are typically 512² at mixed step counts).

```json
"validation": {
    "prompts_file": "prompts/benchmark.json",
    "prompt_selection": "semantic_diversity",
    "prompt_tag_filter": [],
    "num_prompts": 2,
    "seeds": [34635345, 53453634, 267454, 123],
    "num_error_samples": 8,
    "error_samples_span": 1,
    "include_score_top": 2,
    "resolutions": [[512, 512], [1024, 1024], [1024, 512]],
    "step_counts": [20, 30, 40],
    "extra_threshold_sweep": [0.02, 0.04, 0.06, 0.07, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.0, 2.0, 5.0],
    "extra_threshold_sweep_errors": [0.01, 0.03, 0.05, 0.10]
}
```

| Field | Description |
|-------|-------------|
| `prompts_file` | Path to benchmark prompt JSON |
| `prompt_selection` | Selection strategy (same methods as calibration) |
| `prompt_tag_filter` | Tag-based filter (same format as calibration) |
| `num_prompts` | Number of benchmark prompts (default: 2) |
| `seeds` | Seed pool for validation runs (first N used based on config/CLI) |
| `num_error_samples` | N uniform points sampled across the Pareto accumulated_error range. Default: 8 |
| `error_samples_span` | K closest configs to each sample point. Default: 1 |
| `include_score_top` | Also validate top-M configs by score (knee). Default: 2 |
| `resolutions` | `[width, height]` pairs tested per config. Default: `[[512,512],[1024,1024],[1024,512]]` |
| `step_counts` | Step budgets tested per config. Default: `[20,30,40]` |
| `extra_threshold_sweep` | Threshold values swept when `--extra-sweep` is passed. Default: 14 values from 0.02 to 5.0 |
| `extra_threshold_sweep_errors` | Sweep the config nearest each error target. Default: `[0.01,0.03,0.05,0.10]` |

### Artist tags (`artist_tags`)

Anima uses artist tags (`@artist name`) heavily. The pool in `prompts/artists.json`
drives weighted per-generation injection: each generation (prompt × seed ×
steps × resolution) draws its own prefix/negative variant and artist tags, so
a run covers many more artists and variants without extra generations.

```json
"artist_tags": {
    "enabled": true,
    "pool_file": "prompts/artists.json",
    "weight_mode": "relative",
    "max_tags": 1,
    "seed": 42
}
```

| Field | Description |
|-------|-------------|
| `enabled` | Master switch. `false` (or missing pool file) → prompts unchanged |
| `pool_file` | Path to the artist pool JSON (relative to `tuning/`) |
| `weight_mode` | `relative`: weights are likelihoods — 3 is 3× as likely as 1. `static`: weights must sum to 1.0 (loader warns + auto-normalizes) |
| `max_tags` | Max non-empty artists injected per generation (drawn without replacement) |
| `seed` | RNG salt — combined with the generation key, keeps runs reproducible |

Pool format:

```json
{
  "artists": [
    { "tag": null,           "weight": 60 },   // null = "nothing": NO artist block
    { "tag": "@artgerm",     "weight": 3 },
    { "tag": "@anima \\(togashi\\)", "weight": 0.5 }
  ]
}
```

Tags are injected verbatim between the prefix and the prompt text and pass
straight through to the model — backslashes, unicode, slashes and escaped
parens are preserved (`\(` is a literal paren; `(text)` would be emphasis).
Entries with `weight <= 0` are never drawn but stay in the file. Tags
containing commas/newlines trigger a load warning. Baseline and TeaCache runs
in validation always use the identical prompt per generation key, so metrics
remain comparable.

#### CLI presets

| Flag | Effect |
|------|--------|
| `--quick` | 1 prompt, 1 seed, 512²+1024², 30 steps, N=6. ~3 min |
| `--thorough` | 2 prompts, 2 seeds, all resolutions/steps, N=12. ~78 min |

#### Generation budget examples

| Mode | Configs | Prompts | Seeds | Resolutions | Steps | Teacache | Baselines | Total | ~Time |
|------|---------|---------|-------|-------------|-------|----------|-----------|-------|-------|
| Quick | 6+2 | 1 | 1 | 2 | 1 | 16 | 4 | **20** | 3 min |
| Default | 8+2 | 2 | 1 | 3 | 3 | 90 | 18 | **108** | 13 min |
| Thorough | 12+2 | 2 | 2 | 3 | 3 | 252 | 36 | **288** | 34 min |

(Teacache = (num_error_samples * error_samples_span + include_score_top) × prompts × seeds × resolutions × steps)

---

## Prompt Selection & Filtering

### Prompt filtering

Control which prompts are used via tags (defined in `prompt_loader.py`):

```json
"prompt_tag_filter": []                 // Use ALL prompts
"prompt_tag_filter": ["-nsfw"]          // Exclude NSFW
"prompt_tag_filter": ["character", "-nsfw"]  // Character prompts, no NSFW
"prompt_tag_filter": ["landscape", "interior"]  // Only landscapes/interiors
```

Tags without `-` prefix are inclusion filters (at least one must match). Tags with `-` are exclusion filters.

**Tag reference:**

| Tag | Description |
|-----|------------|
| `character` | Focus on one or more characters, portrait or full-body |
| `couple` | Two characters interacting romantically or emotionally |
| `action` | Dynamic scene with movement, combat, or physical activity |
| `landscape` | Outdoor environment, nature, cityscape, or vista |
| `interior` | Indoor scene, room, building interior |
| `nsfw` | Explicit adult/erotic content |
| `abstract` | Non-representational art, patterns, surreal concepts |
| `multi_view` | Multiple views/angles of same subject (character sheet style) |
| `photorealistic` | Photography-like, realistic rather than illustrated |
| `detail_heavy` | Rich in detail: intricate backgrounds, textures, ornaments |
| `simple` | Minimal composition, clean background, few elements |
| `night` | Nighttime or low-light scene |
| `day` | Daytime or well-lit scene |
| `cinematic` | Film-like composition, dramatic lighting, wide shot |
| `close_up` | Close-up or extreme close-up, facial/emotional focus |

### Prompt selection methods

```json
"prompt_selection": "from_top"           // Take first N (deterministic)
"prompt_selection": "from_bottom"        // Take last N
"prompt_selection": "random"             // Random selection (seeded)
"prompt_selection": "text_diversity"     // Maximize word-level variety (Jaccard distance)
"prompt_selection": "tag_diversity"      // Maximize tag coverage
"prompt_selection": "semantic_diversity" // MiniLM embeddings (needs sentence-transformers)
"prompt_selection": "weighted_random"    // Weight by position (favor early entries)
```

The smoke test includes a 12-prompt diversity benchmark that discriminates between the three diversity methods.

---

## How the Optimizer Works

### Pre-computed mapping fits

The optimizer extracts unique `(source, metric_type, metric_weights, signal_scale, mapping_params)` combinations — typically ~60 from hundreds of thousands of candidates. Each is fitted once:

| Mapping type | Fitting method |
|-------------|----------------|
| `polynomial` | `numpy.polyfit` at target degree |
| `power_law` | Log-log OLS: `log(y) = log(k) + α·log(x)` |
| `softplus` | `scipy.optimize.curve_fit` (falls back to grid search) |
| `identity` | No fitting needed |

Results are stored in a dict keyed by signal signature. The simulation loop does a dict lookup instead of re-fitting for each candidate.

### Signal-space deduplication

Configs that differ only in `block_mode`, `residual_strategy`, or `cross_feed` produce **identical simulation results** — the accumulator logic doesn't change. The optimizer groups configs by their signal-space signature (`_signal_signature()` in `optimize.py:237-271`) and simulates each unique group once, then replicates results across cosmetic variants. This gives **~49× reduction** for the default search space.

### Multiprocessing and Numba

**Numba** (optional): Accumulation kernels (`hard_reset`, `carry_over`, `leaky`, `windowed`) are JIT-compiled with `@njit(fastmath=False, cache=True)`, giving ~50× speedup over pure-Python fallbacks. `sim_engine.py` dispatches automatically.

**Multiprocessing** (`spawn` context): When total work exceeds 10M entry-iterations, workers are spawned to avoid CUDA fork crashes. Each worker receives read-only copies of calibration data and mapping cache. `OMP_NUM_THREADS=1` is set before numpy import to prevent BLAS thread contention.

### Two-phase threshold sweep

**Phase 1 — Candidate**: Every unique signal config is simulated at all `candidate_thresholds` (default: 12 geometric values). The best-scoring threshold per config is kept.

**Phase 2 — Pareto**: After the frontier is built, each Pareto-optimal config is re-simulated across `pareto_threshold_count` (default: 500) log-spaced thresholds. The individually optimal threshold replaces the coarse candidate value.

### Cross-validation

When enabled, calibration data is split by `prompt_id`: a holdout fraction (default 20%) is reserved. The training set is used for mapping fits and simulation; the holdout set is used for quality scoring. This gives honest error estimates that aren't inflated by overfitting.

### Data-driven block params

When per-block cosine similarity data exists (`record_block_data: true`), the optimizer auto-computes block parameters:

- **`split_groups`**: Classifies each block group as always-run or cacheable based on `cosim_threshold` applied to the group's mean cosine similarity
- **`dynamic`**: Computes per-block sensitivity multipliers normalized so 1.0 = average sensitivity across all blocks. These feed into per-step block-fraction calculations during simulation

### Pareto frontier

Results filtered (skip_rate ≥ 1%), sorted by speedup descending, walked linearly. Each config is kept only if its error is strictly better than any previously-seen config at the same or higher speedup. O(n log n) via sorting + linear scan.

### Quality scoring

Score for each config is `speedup × quality_score`, where `quality_score` is computed by the configured scoring function (see Quality Scoring section).

---

## The 10-Knob TeaCache Forward

The forward function (`forward.py`) implements every TeaCache parameter:

| # | Knob | Values | Location |
|---|------|--------|----------|
| 1 | Signal source | `t_emb`, `first_block_shift`, `pooled_latent` | `forward.py:450-480` |
| 2 | Distance metric | `mean_only`, `mean_and_max`, `mean_max_std`, `weighted_sum` | `forward.py:40-78` |
| 3 | Signal scaling | Any float — per-source scale lists in config | `forward.py:527-528` |
| 4 | Mapping function | `identity`, `polynomial`, `power_law`, `softplus` | `forward.py:93-135` |
| 5 | Accumulation | `hard_reset`, `carry_over`, `leaky`, `windowed` | `forward.py:142-194` |
| 6 | Threshold | `rel_l1_thresh` (float) | `forward.py:541` |
| 7 | Step schedule | `constant`, `cosine`, `linear_ramp`, `linear_decay`, `bell` | `forward.py:201-230` |
| 8 | Block skipping | `all_or_nothing`, `split_fraction`, `split_groups`, `dynamic` | `forward.py:320-362` |
| 9 | Residual strategy | `hard`, `blended`, `scaled` | `forward.py:288-313` |
| 10 | Cross-feed | `enabled`/`disabled` + `strength` | `forward.py:715-723` |

### Block mode details

#### `all_or_nothing`
All blocks run (residual cached), or all blocks skipped (cached residual applied).

#### `split_fraction`
First `always_fraction × N` blocks always run; rest are cacheable. Stores two residuals: `prev_residual` (early, for always-blocks) and `prev_residual_late` (for cacheable blocks).

#### `split_groups`
Blocks auto-partitioned into 3 groups by architectural role: embedding, spatial, context. `always_groups` and `cache_groups` control which groups always run. Data-driven when per-block data exists.

#### `dynamic`
Per-step dead-block detection. At each step, blocks with cosine similarity above `cosim_threshold` are skipped. Two levels:
- **`unified`**: Single accumulator. Per-step block fraction is weighted by how many cacheable blocks pass the cosim threshold.
- **`per_group`**: Independent accumulators per block group. Each group has its own accumulation type, params, and step schedule. Requires `record_block_data: true`.

### Step schedule functions

| Schedule | Multiplier | Behavior |
|----------|-----------|----------|
| `constant` | `1.0` | Uniform threshold |
| `linear_ramp` | `0.5 + 0.5·frac` | Conservative early, aggressive late |
| `linear_decay` | `2.0 - frac` | Aggressive early, conservative late |
| `cosine` | `cos(frac·π/2)` | Smooth decay: aggressive early, conservative late |
| `bell` | `sin(frac·π)` | Peak in middle, conservative at start/end |

### Residual strategies

| Strategy | Formula | Description |
|----------|---------|-------------|
| `hard` | `x + residual` | Full residual, no scaling |
| `blended` | `x + residual·(1-confidence)` | Less residual when accumulator is close to threshold |
| `scaled` | `x + residual·scale` | Fixed fraction of residual |

---

## Validation Metrics

Quality is measured using 12 metrics across 3 tiers via `pyiqa`:

### Tier 1 (essential — always computed)
| Metric | Dir | What it measures |
|--------|-----|------------------|
| PSNR | ↑ | Pixel-level accuracy |
| SSIM | ↑ | Structural similarity |
| LPIPS (AlexNet) | ↓ | Perceptual — semantic |
| LPIPS (VGG16) | ↓ | Perceptual — texture-sensitive |
| DISTS | ↓ | Structure vs texture decomposition |
| MS-SSIM | ↑ | Multi-scale structural similarity |

### Tier 2 (moderate cost)
| Metric | Dir | What it measures |
|--------|-----|------------------|
| FSIM | ↑ | Edge sharpness via phase congruency |
| VIF | ↑ | Information fidelity — measures info loss |
| GMSD | ↓ | Gradient deviation — sensitive to blur |

### Tier 3 (expensive — human-preference)
| Metric | Dir | What it measures |
|--------|-----|------------------|
| NLPD | ↓ | Normalized Laplacian Pyramid distance |
| PieAPP | ↓ | Human pairwise preference (gold standard) |
| VSI | ↑ | Visual saliency-weighted similarity |

The shared `METRIC_LEGEND` in `utils.py` (lines 251-264) defines EXCELLENT/acceptable/POOR thresholds for every metric, used by both smoke test and validate.py.

---

## GPU Detection & Calibration Time Estimates

`utils.detect_gpu()` identifies the primary CUDA GPU and returns a speed factor relative to V100 (1.0). The database covers ~50 GPU models:

| GPU family | Speed factor | Notes |
|-----------|-------------|-------|
| H200 / H100 | ~4.0× | Enterprise Hopper |
| A100 / A6000 | ~2.3-2.4× | Ampere datacenter |
| RTX 5090 / 4090 | ~2.3× | Consumer Blackwell/Ada |
| V100 | 1.0× | **Baseline** (~12s / 30 steps at 512²) |
| L4 / T4 | ~0.4-0.95× | Inference-optimized |
| RTX 3060 | ~0.45× | Budget consumer |

The calibration time estimate accounts for GPU speed, weighted step mix, and image resolution (quadratic scaling from 512²).

---

## Smoke Test

The smoke test (`smoke_test.py`) runs 8 checks:

1. **Model loading** — UNet, CLIP, VAE via ComfyUI loaders
2. **Prompt diversity test** — compares tag_diversity, text_diversity, semantic_diversity on 12 crafted prompts
3. **Baseline generation** — reference image at default settings
4. **Calibration collection** — 6 varied runs (different sampler, steps, CFG)
5. **Mini-optimizer** — full optimizer pipeline, Pareto frontier, 3 configs (conservative, balanced, aggressive)
6. **TeaCache comparison** — 3 tuned configs vs baseline, actual speedup
7. **Daraskme reference** — validates against published config (first_block_shift + polynomial) as sanity check
8. **Quality check** — all 12 metrics at Tier 3 with EXCELLENT/acceptable/POOR ratings

---

## Performance Tuning

- Set `width: 512, height: 512` — 4× faster calibration than 1024²
- Reduce `num_prompts` and `seeds` for faster calibration
- Set `max_candidates: 5000` to cap the optimizer
- The optimizer uses **pre-computed polynomial fits** — ~60 fits replace 400K redundant calls
- **Signal-space deduplication** — ~49× reduction for default search space
- **Multiprocessing (spawn)** when work > 10M entry-iterations
- **Numba** provides ~50× acceleration for accumulation kernels (optional)
- Set `record_block_data: false` to reduce calibration VRAM by ~2-4 GB

## Requirements

```
pyiqa>=0.1.12               # Quality metrics (PSNR, SSIM, LPIPS, DISTS, FSIM, VIF, GMSD, NLPD, PieAPP, VSI)
sentence-transformers>=3.0  # Semantic prompt diversity (all-MiniLM-L6-v2, 80 MB, CPU only)
numba>=0.60                 # Optional JIT acceleration (degrades gracefully to Python)
numpy>=1.26                 # In ComfyUI
torch>=2.4                  # In ComfyUI
```

## Adding New Prompts

Prompts are stored in `prompts/calibration.json` and `prompts/benchmark.json`.

### Prompt file structure

```json
{
  "default_prefix": "masterpiece, best quality, ...",
  "default_negative": "worst quality, low quality, ...",
  "prefix_variants": [
    "masterpiece, best quality, score_7, newest, highres, ...",
    "masterpiece, best quality, score_7, absurdres, detailed illustration, ..."
  ],
  "negative_variants": [
    "worst quality, low quality, score_1, score_2, score_3, artist name, multiple views",
    "worst quality, low quality, score_1, score_2, artist name, watermark"
  ],
  "prompts": [
    {
      "text": "1girl, elf archer, forest background, sunlit clearing...",
      "prefix": null,
      "negative": null,
      "tags": ["character", "action", "landscape", "day"],
      "nsfw": false,
      "background_only": false
    }
  ]
}
```

| Field | Description |
|-------|-------------|
| `default_prefix` / `default_negative` | Applied to all prompts unless overridden |
| `prefix_variants` / `negative_variants` | Multiple variants cycled per prompt for data diversity |
| `text` | The prompt body (prefix is prepended automatically) |
| `prefix` / `negative` | Per-prompt overrides (`null` = use default) |
| `tags` | List of tag strings (see tag reference above) |
| `nsfw` | Explicit adult content flag |
| `background_only` | Prompt describes only background (no characters) |
