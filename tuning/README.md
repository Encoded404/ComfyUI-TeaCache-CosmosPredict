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

1. **Smoke test** (`smoke_test.py`) — 8 quick checks, ~5 min
2. **Calibration** (`calibrate.py`) — records per-step data, 1-4 hours
3. **Optimization** (`optimize.py`) — offline config search, 1-30 min CPU
4. **Validation** (`validate.py`) — end-to-end quality metrics, 30 min

## Configuration

All settings are in `config.json`. Key sections:

### Prompt filtering (`config.json` → `calibration.prompt_tag_filter`)

Control which prompts are used for calibration. Tags are defined in the prompt JSON files.

```json
// Use ALL prompts:
"prompt_tag_filter": null

// Exclude NSFW content:
"prompt_tag_filter": ["-nsfw"]

// Only character prompts, exclude NSFW:
"prompt_tag_filter": ["character", "-nsfw"]

// Only landscape and interior scenes:
"prompt_tag_filter": ["landscape", "interior"]

// Exclude multiple tags:
"prompt_tag_filter": ["-nsfw", "-abstract", "-multi_view"]
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

### Prompt selection method (`config.json` → `calibration.prompt_selection`)

```json
"prompt_selection": "from_top"          // Take first N (deterministic)
"prompt_selection": "text_diversity"    // Maximize word-level variety
"prompt_selection": "tag_diversity"     // Maximize tag coverage
"prompt_selection": "semantic_diversity" // Maximize meaning variety (needs sentence-transformers)
"prompt_selection": "random"            // Random selection
```

### Search space limits

If optimization takes too long, cap the candidate count:

```json
"max_candidates": 5000   // Randomly sample 5000 configs (seeded, reproducible)
"max_candidates": 0      // Unlimited (may take hours with large config spaces)
```

### Performance tuning

- Set `width: 512, height: 512` for calibration — 4x faster than 1024²
- Reduce `num_prompts` for faster calibration
- Set `max_candidates: 5000` to cap the optimizer
- The `auto_scale_target` computes data-driven scale factors

### Adding new prompts

Prompts are stored in `prompts/calibration.json` and `prompts/benchmark.json`.
Each prompt needs a `text` field and a `tags` list:

```json
{
  "text": "1girl, elf archer, forest background, ...",
  "tags": ["character", "action", "landscape", "day"],
  "nsfw": false
}
```

## Architecture

```
calibrate.py  ──→  calibration_data.jsonl  ──→  optimize.py  ──→  pareto_frontier.json
                                                                    │
validate.py  ←──────────────────────────────────────────────────────┘
    │
    └── validation_results.json
```

The smoke test runs all three phases on a tiny dataset to verify everything works before committing to a multi-hour calibration run.
