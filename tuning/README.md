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
4. **Validation** (`validate.py`) — end-to-end quality metrics (PSNR, SSIM, LPIPS, DISTS, MS-SSIM, FSIM, VIF, GMSD, NLPD, PieAPP, VSI). Shows comparison table and recommendation.

```bash
# Phase 1 — calibration (run on GPU)
python -m tuning.calibrate --comfy-dir .

# Phase 2 — optimization (run on CPU after Phase 1)
python -m tuning.optimize --data outputs/<timestamp>/calibration_data.jsonl

# Phase 3 — validation (run on GPU after Phase 2)
python -m tuning.validate --comfy-dir . \
    --pareto outputs/optimization/pareto_frontier.json \
    --top-k 10 --tier 2 --extra-sweep
```

## Architecture

```
calibrate.py  ──→  calibration_data.jsonl  ──→  optimize.py  ──→  pareto_frontier.json
            │              ↑                                          │
            │    sim_data.py · sim_engine.py · sim_runner.py          │
            │              ↑                                          │
validate.py  ←────────────────────────────────────────────────────────┘
    │
    └── validation_results.json
```

The smoke test runs all three phases on a tiny dataset to verify everything works.

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

```json
"validation": {
    "prompts_file": "prompts/benchmark.json",
    "prompt_selection": "semantic_diversity",
    "prompt_tag_filter": [],
    "num_prompts": 8,
    "seeds": [34635345, 53453634, 267454, 123],
    "top_k": 20,
    "extra_threshold_sweep": [0.02, 0.04, 0.06, 0.07, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.0, 2.0, 5.0]
}
```

| Field | Description |
|-------|-------------|
| `prompts_file` | Path to benchmark prompt JSON |
| `prompt_selection` | Selection strategy (same methods as calibration) |
| `prompt_tag_filter` | Tag-based filter (same format as calibration) |
| `num_prompts` | Number of benchmark prompts |
| `seeds` | Seeds for validation runs |
| `top_k` | Number of top configurations from Pareto frontier to validate |
| `extra_threshold_sweep` | Additional threshold values swept on top-3 configs (when `--extra-sweep` is passed). Tests the full quality-speedup curve |

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
