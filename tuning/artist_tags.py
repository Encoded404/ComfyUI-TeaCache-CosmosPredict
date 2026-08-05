"""Weighted artist-tag pool: loading, selection, rendering, frequency stats.

Artist tags (e.g. "@artgerm") are injected between the prompt prefix and the
prompt text. Each pool entry has a *selection* weight — the probability of
being chosen for a generation — not a prompt-emphasis weight.

Two weight modes (chosen in config.json `artist_tags.weight_mode`):

  relative:  weights are relative likelihoods; a weight of 3 is 3x as likely
             as 1.  No normalization required — edit freely.
  static:    weights must already sum to 1.0 (they ARE the probabilities).
             Loader warns and auto-normalizes if the sum drifts.

The pool supports an empty sentinel: an entry with a null/empty tag is drawn
like any other tag — when selected, NO artist block is injected (prompt is
byte-identical to a no-artist run).

Tags pass through verbatim. Names may contain spaces, backslashes, escaped
parens (e.g. "@anima \\(togashi\\)"), unicode, slashes — never re-parsed or
altered. The only load-time warnings are for tags containing commas or
newlines, which would break the ", "-join rendering.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ArtistTag:
    """One pool entry. `tag=None` (or "") is the empty sentinel: no artists."""
    tag: Optional[str]
    weight: float


@dataclass
class ArtistPool:
    artists: List[ArtistTag] = field(default_factory=list)

    @property
    def active(self) -> List[ArtistTag]:
        """Entries eligible for drawing (weight > 0)."""
        return [t for t in self.artists if t.weight > 0]

    @property
    def has_empty(self) -> bool:
        return any(t.tag is None or t.tag == "" for t in self.active)

    def total_weight(self) -> float:
        return sum(t.weight for t in self.active)


# ── Loading ──────────────────────────────────────────────────────────

def load_artist_pool(path: str, weight_mode: str = "relative") -> ArtistPool:
    """Load a pool JSON file.

    Format:
        { "artists": [ {"tag": "@name", "weight": 1.0}, ... ] }

    Entries with weight <= 0 are excluded from drawing (kept for reference).
    weight_mode "static" requires the active weights to sum to 1.0; a
    deviation is warned about and auto-normalized.
    """
    with open(path) as f:
        data = json.load(f)

    artists = []
    for a in data.get("artists", []):
        tag = a.get("tag")
        weight = float(a.get("weight", 1.0))
        if weight <= 0:
            continue
        if tag is not None:
            if "," in tag or "\n" in tag or "\r" in tag:
                print(f"  ⚠ artist tag contains comma/newline: {tag!r} "
                      f"— will break \", \"-join rendering")
        artists.append(ArtistTag(tag=tag, weight=weight))

    pool = ArtistPool(artists=artists)

    if weight_mode == "static":
        total = pool.total_weight()
        if abs(total - 1.0) > 1e-3:
            print(f"  ⚠ artist pool {path}: weight_mode=static but "
                  f"Σweight={total:.4f} ≠ 1.0 — auto-normalizing")
            for t in pool.artists:
                t.weight /= total

    return pool


def load_pool_for_config(tcfg) -> Optional[ArtistPool]:
    """Build the artist pool from a TuningConfig. Returns None when artist
    tags are disabled or the pool file is missing (with a warning)."""
    acfg = getattr(tcfg, "artist_tags", None) or {}
    if not acfg.get("enabled", False):
        return None
    path = Path(__file__).parent / acfg.get("pool_file", "prompts/artists.json")
    if not path.exists():
        print(f"  ⚠ artist pool not found: {path} — artist tags disabled")
        return None
    return load_artist_pool(str(path), acfg.get("weight_mode", "relative"))


# ── Selection ────────────────────────────────────────────────────────

def _weighted_choice(pool: List[ArtistTag], rng: random.Random) -> ArtistTag:
    """One weighted draw from a list of entries (rng drives determinism)."""
    weights = [t.weight for t in pool]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for t, w in zip(pool, weights):
        acc += w
        if r < acc:
            return t
    return pool[-1]


def select_artists(pool: Optional[ArtistPool], rng: random.Random,
                   max_tags: int = 1) -> List[str]:
    """Weighted per-generation artist draw.

    Draw 1 includes the empty sentinel: selecting it yields [] (no artist
    block). Subsequent draws take non-empty tags WITHOUT replacement, with
    weights renormalized after each removal. Returns bare tag strings.
    """
    if pool is None or not pool.active:
        return []

    first = _weighted_choice(pool.active, rng)
    if not first.tag:
        return []

    chosen = [first.tag]
    remaining = [t for t in pool.active if t.tag and t.tag != first.tag]
    for _ in range(max(max_tags - 1, 0)):
        if not remaining:
            break
        pick = _weighted_choice(remaining, rng)
        chosen.append(pick.tag)
        remaining = [t for t in remaining if t.tag != pick.tag]
    return chosen


# ── Rendering ────────────────────────────────────────────────────────

def render_artist_block(tags: List[str]) -> str:
    """Render the injected block: 'a, b, ' (trailing separator, verbatim tags).

    Empty list → "" so the prompt is unchanged. The block is placed between
    the prefix and the prompt text by resolve_prompt().
    """
    if not tags:
        return ""
    return ", ".join(tags) + ", "


# ── Frequency stats ──────────────────────────────────────────────────

def realized_frequencies(draws: List[List[str]]) -> Dict[Optional[str], int]:
    """Tally per-tag draw counts across a run.

    Returns {tag: count} with key None counting "nothing" (empty) draws.
    """
    counts: Dict[Optional[str], int] = {}
    for d in draws:
        if not d:
            counts[None] = counts.get(None, 0) + 1
        else:
            for tag in d:
                counts[tag] = counts.get(tag, 0) + 1
    return counts


def print_artist_frequencies(draws: List[List[str]], pool: Optional[ArtistPool],
                             total_generations: int) -> None:
    """Print a realized-vs-configured draw distribution table."""
    counts = realized_frequencies(draws)
    total = float(max(sum(counts.values()), 1))

    pool_share: Dict[Optional[str], float] = {}
    if pool is not None and pool.active:
        tw = pool.total_weight()
        for t in pool.active:
            pool_share[t.tag] = t.weight / max(tw, 1e-9)

    rows = []
    for key, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        label = "nothing" if key is None else key
        share = count / total
        ps = pool_share.get(key)
        exp = f"  (pool {ps:.1%})" if ps is not None else ""
        rows.append(f"    {label:<32} {count:>5}  {share:>7.1%}{exp}")

    print(f"\n  Artist tag draw frequencies ({total_generations} generations):")
    print(f"  {'─' * 60}")
    print("\n".join(rows))
    print(f"  {'─' * 60}")
