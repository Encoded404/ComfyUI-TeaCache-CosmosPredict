# TeaCache Tuning Toolkit

Calibration and optimization pipeline for finding optimal TeaCache parameters
for the Anima (Cosmos-Predict2) model.

## Quick Start

```bash
# On the V100, from the ComfyUI root:
cd /path/to/ComfyUI
PYTHONPATH=".:custom_nodes/ComfyUI-TeaCache-CosmosPredict"
python -m tuning.smoke_test --comfy-dir .
```

## Pipeline

1. **Smoke test** (`smoke_test.py`) — 8 quick checks including diversity test, mini-optimizer, daraskme reference. ~6 min on V100 at 512².
2. **Calibration** (`calibrate.py`) — records per-step (rel_l1, out_rel) pairs across diverse prompts, seeds, step counts, samplers, and CFGs. Shows run schedule with time/disk estimates before starting. 30 min–4 hours.
3. **Optimization** (`optimize.py`) — offline config search. Pre-computes polynomial fits, sweeps candidate thresholds, builds Pareto frontier, fine-tunes winner thresholds. Multi-core CPU, 3–30 min.
4. **Validation** (`validate.py`) — end-to-end quality metrics (PSNR, SSIM, LPIPS, DISTS, MS-SSIM, FSIM, VIF, GMSD). Shows comparison table and recommendation.

```bash
# Phase 1 — calibration (run on V100)
python -m tuning.calibrate --comfy-dir .

# Phase 2 — optimization (run on CPU after Phase 1)
python -m tuning.optimize --data outputs/<timestamp>/calibration_data.jsonl

# Phase 3 — validation (run on V100 after Phase 2)
python -m tuning.validate --comfy-dir . \
    --pareto outputs/optimization/pareto_frontier.json \
    --top-k 10 --tier 2 --extra-sweep
```

## Architecture

```
calibrate.py  ──→  calibration_data.jsonl  ──→  optimize.py  ──→  pareto_frontier.json
                                                                    │
validate.py  ←──────────────────────────────────────────────────────┘
    │
    └── validation_results.json
```

The smoke test runs all three phases on a tiny dataset to verify everything works.

---

## Configuration Reference

All settings are in `config.json`.

### Prompt filtering (`calibration.prompt_tag_filter`)

Control which prompts are used for calibration. Tags are defined in the prompt JSON files.

```json
"prompt_tag_filter": null              // Use ALL prompts
"prompt_tag_filter": ["-nsfw"]         // Exclude NSFW
"prompt_tag_filter": ["character", "-nsfw"]  // Character prompts, no NSFW
"prompt_tag_filter": ["landscape", "interior"]  // Only landscapes/interiors
```

**Tag reference** (defined in `prompt_loader.py`):

| Tag | Description |
|-----|------------|
| `character` | Focus on one or more characters |
| `couple` | Two characters interacting |
| `action` | Dynamic scene with movement |
| `landscape` | Outdoor environment |
| `interior` | Indoor scene |
| `nsfw` | Explicit adult content |
| `abstract` | Non-representational art |
| `multi_view` | Multiple views of same subject |
| `photorealistic` | Photography-like |
| `detail_heavy` | Intricate backgrounds |
| `simple` | Minimal composition |
| `night` | Nighttime lighting |
| `day` | Daytime lighting |
| `cinematic` | Film-like composition |
| `close_up` | Close-up shot |

### Prompt selection methods (`calibration.prompt_selection`)

```json
"prompt_selection": "from_top"          // Take first N (deterministic)
"prompt_selection": "text_diversity"    // Maximize word-level variety (Jaccard)
"prompt_selection": "tag_diversity"     // Maximize tag coverage
"prompt_selection": "semantic_diversity" // MiniLM embeddings (needs sentence-transformers)
"prompt_selection": "random"            // Random selection
```

### Calibration variety (`sampling`)

The calibrator cycles sampler, scheduler, and CFG per prompt for training data diversity:

```json
"sampler_variants": ["er_sde", "dpmpp_2m_sde", "euler_a"],
"scheduler_variants": ["normal", "simple"],
"cfg_variants": [4.0, 4.5, 5.0, 5.5]
```

Each calibration entry records which sampler/scheduler was used, enabling sampler-aware tuning in the future.

### Optimization search space (`optimization`)

**Dimensions swept**: source signals, distance metrics, mapping functions, accumulation strategies, step schedules, signal scales, residual strategies, block splitting modes.

```json
"sources": ["t_emb", "first_block_shift", "pooled_latent"],
"mapping_types": ["identity", "polynomial", "power_law", "softplus"],
"accumulation_types": ["hard_reset", "carry_over", "leaky", "windowed"],
"residual_strategies": ["hard", "blended", "scaled"],
"step_schedules": ["constant", "cosine", "linear_ramp", "linear_decay", "bell"],
"block_modes": ["all_or_nothing", "split_fraction", "split_groups"]
```

### Threshold sweeping (`optimization`)

Two-phase sweep gives precise optimal thresholds:

```json
"candidate_thresholds": [0.01, 0.07, 0.20],  // Tested on EVERY candidate
"pareto_threshold_range": [0.001, 10.0],      // Fine sweep for winners only
"pareto_threshold_count": 300                 // Log-spaced steps
```

### Quality scoring (`optimization.quality_scoring`)

Controls how simulated error is mapped to quality (0-1). The scoring function determines which configs the Pareto frontier favors.

```json
"quality_scoring": {
    "type": "exponential",   // Type (see below)
    "target": 0.05           // Error threshold where quality drops ~60%
}
```

| Type | Formula | Description |
|------|---------|-------------|
| `linear` | `1/(1+error)` | Mild penalty — speedup dominates at all ranges |
| `exponential` | `exp(-error/target)` | Strong discrimination — error beyond target is heavily penalized |
| `gaussian` | `exp(-0.5*(error/target)²)` | Very strong — even moderate errors are penalized |
| `step` | `1.0 if error < target else 0.0` | Hard cutoff — only configs below target qualify |
| `power` | `1/(1+(error/target)^power)` | Configurable — add `"power": 2.0` for steepness |

**Effect at different error levels (target=0.05)**:

| error | linear | exponential | gaussian | step | power(2) |
|-------|--------|-------------|----------|------|----------|
| 0.001 | 0.999 | 0.980 | 1.000 | 1.0 | 1.000 |
| 0.025 | 0.976 | 0.607 | 0.882 | 1.0 | 0.800 |
| 0.050 | 0.952 | 0.368 | 0.607 | 1.0 | 0.500 |
| 0.100 | 0.909 | 0.135 | 0.135 | 0.0 | 0.200 |

Default is `exponential` for a balanced tradeoff.

### Search space limits

```json
"max_candidates": 0      // 0 = unlimited. Set to e.g. 5000 for faster runs.
"auto_scale_target": 0.3 // Data-driven scale: scale = target / avg_distance
```

### Performance tuning

- Set `width: 512, height: 512` for calibration — 4x faster than 1024²
- Reduce `num_prompts` and `seeds` count for faster calibration
- Set `max_candidates: 5000` to cap the optimizer
- The optimizer uses **pre-computed polynomial fits** — ~60 unique fits replace 400K redundant polyfit calls
- The optimizer uses **multiprocessing (spawn)** when total work > 10M entry-iterations (~2s serial). Chunksize auto-tuned for ~30 IPC rounds.
- **Two-phase threshold sweep**: candidate phase runs 3 thresholds per config (0.01, 0.07, 0.20), Pareto phase fine-tunes winners across 300 values

### Adding new prompts

Prompts are stored in `prompts/calibration.json` (43 prompts) and `prompts/benchmark.json` (12 prompts). Each prompt needs a `text` field, `tags` list, and optional `nsfw` / `background_only` flags:

```json
{
  "text": "1girl, elf archer, forest background, sunlit clearing...",
  "prefix": null,
  "negative": null,
  "tags": ["character", "action", "landscape", "day"],
  "nsfw": false
}
```

Prefix/negative can be overridden per-prompt or use the global defaults from the file header.

---

## How the Optimizer Works

### Pre-computed polynomial fits

The optimizer extracts all unique `(source, metric_type, weights, scale)` combinations from the ~800K candidates — only ~60 are unique. Each is fitted once via `numpy.polyfit`, then stored in a lookup dict. The simulation loop just does a dict lookup instead of calling `numpy.polyfit` 400K times.

### Parallel execution

When `total_configs × entries > 10M`, the optimizer spawns workers using `multiprocessing.spawn()` (avoids CUDA fork crashes). Each worker gets a read-only copy of the calibration data and the polyfit cache. `OMP_NUM_THREADS=1` is set at module level before any numpy import to prevent BLAS thread contention. Progress shows worker count and ETA.

### Two-phase threshold sweep

**Phase 1 — candidate**: Every config is simulated at 3 thresholds (`candidate_thresholds`). The best-scoring threshold per config is kept. Costs 3× the per-config time but adds meaningful threshold data.

**Phase 2 — Pareto**: After the frontier is built from candidate results, each of the ~60-120 winning configs is simulated across 300 logarithmically-spaced thresholds. The individually optimal threshold replaces the coarse candidate value. Costs ~3s for 120 configs.

### Pareto frontier (O(n log n) skyline)

Results are filtered (skip_rate ≥ 1%), sorted by speedup descending, and walked linearly. Each config is kept only if its error is strictly better than any previously-seen config at the same or higher speedup. Down from O(n²) to O(n log n).

### Quality scoring

The score for each config is `speedup × quality_score` where `quality_score` is a configurable function of the simulated accumulated error. See the quality scoring section above for the 5 available types and their tradeoffs.

---

## The 10-Knob TeaCache Forward

The forward function (`forward.py`) implements every TeaCache parameter we search over:

| # | Knob | Values | Location |
|---|------|--------|----------|
| 1 | Signal source | t_emb, first_block_shift, pooled_latent | `forward.py:422-443` |

All fallthrough branches log warnings so missing options are immediately visible.

### Pooled latent mode (`pooled_latent_mode`)

When using `source: pooled_latent`, two pooling strategies are available:

| Mode | Description | Speed | Resolution independence |
|------|-------------|-------|------------------------|
| `mean` (default) | Simple `x.mean(dim=(1,2))` — single CUDA reduction | **10-20× faster** | Yes, via ratio cancellation |
| `fixed_grid` | AdaptiveAvgPool2d to 16×16 grid | Slow (permute + pool2d + reshape) | Explicitly resolution-normalized |

**Why `mean` is resolution-independent**: The TeaCache distance metric is `rel_l1 = |curr - prev| / |prev|`. Both numerator and denominator are computed from the same pooled tensor, so token count scales them identically — the ratio cancels out. Fixed-grid pooling was originally added assuming absolute magnitude mattered, but it doesn't for a ratio metric.

To use `fixed_grid`: set `"pooled_latent_mode": "fixed_grid"` in the config's `mapping_params_scenarios` for pooled_latent source configs. Only runs when TeaCache is active (not on start/end percent steps).
