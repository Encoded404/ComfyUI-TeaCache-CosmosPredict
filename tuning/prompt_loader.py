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
      from_top:          Take first N (deterministic, good for reproducibility)
      from_bottom:       Take last N
      random:            Random selection
      tag_diversity:     Greedy selection maximizing unique tag coverage
      text_diversity:    Greedy selection maximizing lexical (word-level) diversity
      tag_filter:        Only include prompts matching specific tags
                         (prefix tag with - to exclude, e.g. ["character", "-nsfw"])
      weighted_random:   Weight by prompt position (favor early entries)
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

    if method == "text_diversity":
        return _select_text_diversity(pool, count, rng)

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
            pick = remaining.pop(rng.randint(0, len(remaining) - 1))
            selected.append(pick)
            continue
        pick = remaining.pop(best_idx)
        selected.append(pick)
        covered_tags |= set(pick.tags)
    return selected


def _tokenize(text: str) -> Set[str]:
    """Tokenize prompt text into a set of lowercase word tokens.

    Strips weight annotations like (keyword:1.5) and punctuation,
    keeping only alphanumeric words of length >= 2.
    """
    import re
    # Remove weight annotations: (word:1.5) → word
    cleaned = re.sub(r'\(([^:)]+):[\d.]+\)', r'\1', text)
    # Remove parentheses and special chars
    cleaned = re.sub(r'[()\[\]{},]', ' ', cleaned)
    # Split and filter
    words = set()
    for w in cleaned.lower().split():
        w = w.strip()
        if len(w) >= 2 and w.isalpha():
            words.add(w)
    return words


def _select_text_diversity(
    pool: List[PromptEntry], count: int, rng: random.Random
) -> List[PromptEntry]:
    """Greedy selection maximizing lexical diversity across prompt text.

    Uses word-level Jaccard distance: picks prompts that share the
    fewest words with already-selected prompts. This favors prompts
    with different vocabulary — different subjects, settings, styles,
    and compositions without needing tags or an embedding model.

    Starts with a random prompt, then repeatedly picks the one with
    the highest average Jaccard distance to all selected prompts.
    """
    count = min(count, len(pool))
    if count <= 1:
        return [pool[rng.randint(0, len(pool) - 1)]]

    remaining = list(pool)
    # Start with the most word-dense prompt (to avoid picking a stub)
    tokenized = [_tokenize(p.text) for p in remaining]
    best_start = max(range(len(remaining)), key=lambda i: len(tokenized[i]))
    selected = [remaining.pop(best_start)]
    selected_tokens = [tokenized.pop(best_start)]

    for _ in range(count - 1):
        best_idx = 0
        best_score = -1.0
        for i in range(len(remaining)):
            p_words = tokenized[i]
            # Average Jaccard distance to all selected prompts
            total_dist = 0.0
            for sw in selected_tokens:
                intersection = len(p_words & sw)
                union = len(p_words | sw)
                total_dist += 1.0 - (intersection / max(union, 1))
            avg_dist = total_dist / max(len(selected_tokens), 1)
            if avg_dist > best_score:
                best_score = avg_dist
                best_idx = i

        pick = remaining.pop(best_idx)
        selected.append(pick)
        selected_tokens.append(tokenized.pop(best_idx))

    return selected


# ══════════════════════════════════════════════════════════════════════
#  SEMANTIC DIVERSITY (LLM-BASED) — investigation
# ══════════════════════════════════════════════════════════════════════
#
# Text-based Jaccard diversity only captures lexical overlap (shared
# words). Two prompts can be lexically different but semantically
# similar (e.g., "samurai in forest at dawn" vs "elf archer in woods
# at sunrise"). An embedding model would catch this.
#
# === Option A: Use ComfyUI's already-loaded CLIP model ===
#
# Anima loads qwen_image CLIP (~1.5GB on GPU). We could encode prompts
# through CLIP's text encoder and use cosine similarity between the
# pooled embeddings for diversity selection.
#
#   def _get_clip_embeddings(clip, texts) -> np.ndarray:
#       import numpy as np
#       embs = []
#       for t in texts:
#           tokens = clip.tokenize(t)
#           emb = clip.encode_from_tokens_scheduled(tokens)
#           embs.append(emb.flatten().cpu().numpy())
#       return np.vstack(embs)
#
#   def _select_semantic_diversity(pool, count, rng, clip):
#       texts = [p.text for p in pool]
#       embs = _get_clip_embeddings(clip, texts)
#       norms = np.linalg.norm(embs, axis=1)
#       ...
#
# Pros: zero new dependencies, CLIP already loaded
# Cons: GPU-heavy for many prompts, requires passing CLIP to loader,
#       CLIP embeddings optimized for image-text alignment, not
#       specifically for prompt-to-prompt similarity
#
# === Option B: Small sentence-transformers model (CPU) ===
#
#   pip install sentence-transformers
#   model = SentenceTransformer('all-MiniLM-L6-v2')  # 80 MB download
#   embeddings = model.encode(texts)  # 384-dim vectors
#
# Pros: trivial integration, 80 MB, runs on CPU, ~1000 prompts/sec,
#       specifically trained for semantic textual similarity
# Cons: adds sentence-transformers + torch dependency (~200 MB extra),
#       but torch is already installed via ComfyUI
#
# === Option C: FastText / GloVe word vectors (CPU) ===
#
# Average word vectors → sentence vector. ~100MB download, no
# sentence-transformers dependency. Lower quality than transformers
# but much smaller footprint.
#
#   pip install fasttext  # or use glove vectors
#   import fasttext.util
#   ft = fasttext.load_model('cc.en.300.bin')
#
# === Recommendation ===
#
# Option B (sentence-transformers/MiniLM) is the best tradeoff:
# - The model is tiny (80 MB) and downloads automatically on first use
# - CPU inference is fast enough for prompt selection (not in hot path)
# - The embeddings are specifically trained for semantic similarity
# - It would be a one-time dependency installed in smoke_test preamble
#
# Implementation sketch (add to prompt_loader.py):
#
#   _semantic_model = None
#
#   def _get_semantic_model():
#       global _semantic_model
#       if _semantic_model is None:
#           from sentence_transformers import SentenceTransformer
#           _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
#       return _semantic_model
#
#   def _select_semantic_diversity(pool, count, rng):
#       model = _get_semantic_model()
#       texts = [p.text for p in pool]
#       import numpy as np
#       embs = np.asarray(model.encode(texts, convert_to_numpy=True))
#       norms = np.linalg.norm(embs, axis=1, keepdims=True)
#       embs = embs / np.maximum(norms, 1e-8)
#
#       selected = [0]
#       for _ in range(count - 1):
#           best_idx = -1
#           best_dist = 2.0  # worst cosine dist
#           for i in range(len(pool)):
#               if i in selected: continue
#               min_sim = max(np.dot(embs[i], embs[s]) for s in selected)
#               if min_sim < best_dist:
#                   best_dist = min_sim
#                   best_idx = i
#           selected.append(best_idx)
#       return [pool[i] for i in selected]
#
# To enable: add "semantic_diversity" to the method dispatch in
# select_prompts() and add sentence-transformers to requirements.txt.


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
