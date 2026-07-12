"""Prompt loading, tag filtering, and selection strategies.

Supports a JSON prompt file format with multi-line prompts, tag
classification, prefix/negative templating, and multiple selection
methods (from_top, from_bottom, random, tag_diversity, tag_filter).

Prompt file format (calibration.json / benchmark.json):
{
  "default_prefix": "masterpiece, best quality, ...",
  "default_negative": "worst quality, low quality, ...",
  "prefix_variants": [
    "masterpiece, best quality, score_7, newest, highres, absurdres, anime screenshot, detailed anime style, ",
    "masterpiece, best quality, score_7, absurdres, detailed illustration, "
  ],
  "negative_variants": [
    "worst quality, low quality, score_1, score_2, score_3, artist name, multiple views",
    "worst quality, low quality, score_1, score_2, artist name, watermark"
  ],
  "prompts": [
    {
      "text": "1girl, Sylvarie, elf, pointed ears, long platinum blonde hair...",
      "prefix": null,
      "negative": null,
      "tags": ["character", "interior", "detail-heavy"],
      "nsfw": false,
      "background_only": false
    }
  ]
}
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


# ── Tag definitions ──────────────────────────────────────────────────
# Each tag describes a prompt's content type. The tag system enables
# filtering (include/exclude specific content types) and diversity
# sampling (ensure coverage across different visual domains).

TAG_DESCRIPTIONS: Dict[str, str] = {
    "character":      "Focus on one or more characters, portrait or full-body",
    "couple":         "Two characters interacting romantically or emotionally",
    "action":         "Dynamic scene with movement, combat, or physical activity",
    "landscape":      "Outdoor environment, nature, cityscape, or vista",
    "interior":       "Indoor scene, room, building interior",
    "nsfw":           "Explicit adult/erotic content",
    "abstract":       "Non-representational art, patterns, surreal concepts",
    "multi_view":     "Multiple views/angles of same subject (character sheet style)",
    "photorealistic": "Photography-like, realistic rather than illustrated",
    "detail_heavy":   "Rich in detail: intricate backgrounds, textures, ornaments",
    "simple":         "Minimal composition, clean background, few elements",
    "night":          "Nighttime or low-light scene",
    "day":            "Daytime or well-lit scene",
    "cinematic":      "Film-like composition, dramatic lighting, wide shot",
    "close_up":       "Close-up or extreme close-up, facial/emotional focus",
}

ALL_TAGS = set(TAG_DESCRIPTIONS.keys())


@dataclass
class PromptEntry:
    """One prompt with optional overrides and tag metadata."""
    text: str
    prefix: Optional[str] = None
    negative: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    nsfw: bool = False
    background_only: bool = False


@dataclass
class PromptConfig:
    """Loaded prompt file with defaults and selection config."""
    default_prefix: str = ""
    default_negative: str = ""
    prefix_variants: List[str] = field(default_factory=list)
    negative_variants: List[str] = field(default_factory=list)
    prompts: List[PromptEntry] = field(default_factory=list)


# ── Selection strategies ─────────────────────────────────────────────

def select_prompts(
    prompt_config: PromptConfig,
    method: str = "from_top",
    count: int = 24,
    tag_filter: Optional[List[str]] = None,
    seed: int = 42,
) -> List[PromptEntry]:
    """Select `count` prompts from the pool using the given strategy.

    Methods:
      from_top:        Take first N (deterministic, good for reproducibility)
      from_bottom:     Take last N
      random:          Random selection
      tag_diversity:   Greedy selection maximizing unique tag coverage
      tag_filter:      Only include prompts matching specific tags
                       (prefix tag with - to exclude, e.g. ["character", "-nsfw"])
      weighted_random: Weight by prompt position (favor early entries)
    """
    rng = random.Random(seed)

    # Apply tag filter
    pool = _apply_tag_filter(prompt_config.prompts, tag_filter)
    if not pool:
        raise ValueError(
            f"No prompts match tag filter {tag_filter}. "
            f"Available tags: {TAG_DESCRIPTIONS.keys()}"
        )

    if method == "from_top":
        return pool[:count]

    if method == "from_bottom":
        return pool[-count:] if len(pool) >= count else pool

    if method == "random":
        return rng.sample(pool, min(count, len(pool)))

    if method == "tag_diversity":
        return _select_tag_diversity(pool, count, rng)

    if method == "weighted_random":
        weights = [max(len(pool) - i, 1) for i in range(len(pool))]
        indices = rng.choices(range(len(pool)), weights=weights, k=min(count, len(pool)))
        return [pool[i] for i in indices]

    raise ValueError(f"Unknown selection method: {method}")


def resolve_prompt(prompt_config: PromptConfig, entry: PromptEntry,
                   prefix_variant_idx: int = 0,
                   negative_variant_idx: int = 0) -> tuple[str, str]:
    """Return (full_prompt, negative_prompt) with prefix/negative resolved."""
    prefix = entry.prefix if entry.prefix is not None else prompt_config.default_prefix
    negative = entry.negative if entry.negative is not None else prompt_config.default_negative

    # Apply variant if available
    if prompt_config.prefix_variants and prefix_variant_idx < len(prompt_config.prefix_variants):
        prefix = prompt_config.prefix_variants[prefix_variant_idx]
    if prompt_config.negative_variants and negative_variant_idx < len(prompt_config.negative_variants):
        negative = prompt_config.negative_variants[negative_variant_idx]

    return prefix + entry.text, negative


# ── Internal helpers ─────────────────────────────────────────────────

def _apply_tag_filter(
    prompts: List[PromptEntry],
    tag_filter: Optional[List[str]],
) -> List[PromptEntry]:
    """Filter prompts by tag list.

    Tags prefixed with - are exclusion tags.
    Tags without prefix are inclusion tags (at least one must match).
    """
    if not tag_filter:
        return list(prompts)

    include_tags = {t for t in tag_filter if not t.startswith("-")}
    exclude_tags = {t[1:] for t in tag_filter if t.startswith("-")}

    result = []
    for p in prompts:
        ptags = set(p.tags)
        if include_tags and not ptags & include_tags:
            continue
        if exclude_tags and ptags & exclude_tags:
            continue
        result.append(p)
    return result


def _select_tag_diversity(
    pool: List[PromptEntry], count: int, rng: random.Random
) -> List[PromptEntry]:
    """Greedy selection maximizing unique tag coverage.

    Starts with the prompt with the most unique tags, then iteratively
    adds prompts that contribute the most NEW tags to the selection.
    """
    count = min(count, len(pool))
    remaining = list(pool)
    selected = []
    covered_tags: Set[str] = set()

    for _ in range(count):
        best_idx = 0
        best_new = -1
        for i, p in enumerate(remaining):
            new_tags = set(p.tags) - covered_tags
            if len(new_tags) > best_new:
                best_new = len(new_tags)
                best_idx = i
        if best_new == 0:
            # No new tags available; pick random from remaining
            pick = remaining.pop(rng.randint(0, len(remaining) - 1))
            selected.append(pick)
            continue
        pick = remaining.pop(best_idx)
        selected.append(pick)
        covered_tags |= set(pick.tags)
    return selected


# ── File I/O ─────────────────────────────────────────────────────────

def load_prompt_config(filepath: str) -> PromptConfig:
    """Load a JSON prompt file."""
    with open(filepath) as f:
        data = json.load(f)

    prompts = [
        PromptEntry(
            text=p.get("text", ""),
            prefix=p.get("prefix"),
            negative=p.get("negative"),
            tags=p.get("tags", []),
            nsfw=p.get("nsfw", False),
            background_only=p.get("background_only", False),
        )
        for p in data.get("prompts", [])
    ]

    return PromptConfig(
        default_prefix=data.get("default_prefix", ""),
        default_negative=data.get("default_negative", ""),
        prefix_variants=data.get("prefix_variants", []),
        negative_variants=data.get("negative_variants", []),
        prompts=prompts,
    )


def list_available_tags(filepath: str) -> dict:
    """Scan a prompt file and report which tags exist and how many."""
    config = load_prompt_config(filepath)
    tag_counts = {}
    for p in config.prompts:
        for t in p.tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    return tag_counts
