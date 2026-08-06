"""Corrector training dataset over ``refiner_data`` (plan Task 6e, deep-dive §7).

Exposes **i.i.d. per-step pairs** ``(x_t, v_MA, Δ_MA, v_true, t_frac, lag, slot)``
from the recorded generations:

- eval-prompt generations are excluded from training indexing (they exist only
  for the eval loop; ``only_eval=True`` selects them exclusively);
- the d=0 anchor pair (``v_MA = v_true``, ``Δ_MA = 0``) is synthesized at load
  (``iter_pairs`` semantics — zero storage);
- per-lag resampling weights (``lag_weights``, in ``record_lags`` order) feed a
  weighted batch sampler; the d=0 pair takes weight 1.0;
- resolution-grouped batches (512² and 1024² separate; 1024² batches ÷4);
- prompt embeddings padded to the batch max token count with an attention mask;
  batches mixing prompt / 0-token uncond samples are handled inside the model
  via the mask (equivalent to the plan's two-group split, one compiled path).

Generations are decompressed lazily and held in a small LRU cache (the .bin
format stores compressed blobs, so whole-generation decompression is the
practical access pattern; ~60 MB per generation in RAM).
"""

from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, Sampler

from . import refiner_data

try:
    from tqdm import tqdm
except ImportError:  # progress bars are optional (requirements: tqdm)
    tqdm = None

MAX_CACHED_GENERATIONS = 8


def _lag_index(lag: int, lags: List[int]) -> int:
    """Position of *lag* in the recorded ladder; d=0 maps to -1 (no weight knob)."""
    try:
        return lags.index(lag)
    except ValueError:
        return -1


class CorrectorDataset(Dataset):
    """i.i.d. per-step (x_t, v_MA, v_true, t, lag, slot) pairs from refiner_data."""

    def __init__(self, data_dir, include_eval: bool = False, only_eval: bool = False,
                 seed: int = 42, lag_weights: Optional[List[float]] = None,
                 rel_mse_eps_scale: float = 1e-4,
                 normalization_samples: int = 128,
                 compute_normalization: bool = False,
                 show_progress: bool = True):
        data_dir = Path(data_dir)
        self.data_dir = data_dir
        self.seed = seed
        self.lag_weights = lag_weights or [1.0] * 5
        self.rel_mse_eps_scale = rel_mse_eps_scale
        self.rng = random.Random(seed)
        self._gen_cache: "OrderedDict[str, refiner_data.RefinerGeneration]" = OrderedDict()

        self.entries = refiner_data.iter_generations(data_dir)
        if only_eval:
            self.entries = [e for e in self.entries if e.get("eval")]
        elif not include_eval:
            self.entries = [e for e in self.entries if not e.get("eval")]
        if not self.entries:
            raise ValueError(
                f"CorrectorDataset: no {'eval ' if only_eval else 'training '}"
                f"generations in {data_dir}"
            )

        # Pair index: (gen_idx, slot, step, lag). A stored lag d exists at step
        # t iff d <= t (the recorder's ring has t entries before the current
        # append — refiner_data.py / recorder.py semantics); d=0 is always
        # synthesized.
        self.pairs: List[Tuple[int, int, int, int]] = []
        self.pair_weights: List[float] = []
        self.pair_shape: List[Tuple[int, int]] = []
        for gi, e in enumerate(self.entries):
            lags = sorted(int(d) for d in (e.get("lags") or []))
            shape = tuple(e.get("shape") or [16, 64, 64])[1:]
            for slot in (e.get("slots") or []):
                for t in range(int(e.get("num_steps", 0))):
                    for d in lags:
                        if d <= t:
                            self.pairs.append((gi, int(slot), t, d))
                            li = _lag_index(d, lags)
                            w = float(self.lag_weights[li]) if 0 <= li < len(self.lag_weights) else 1.0
                            self.pair_weights.append(w)
                            self.pair_shape.append(shape)
                    self.pairs.append((gi, int(slot), t, 0))
                    self.pair_weights.append(1.0)
                    self.pair_shape.append(shape)

        self.weights = torch.tensor(self.pair_weights, dtype=torch.float64)

        # Per-channel normalization stats (perchannel option, plan 6d):
        # mean/std over the 32-ch concat input (x_t ⊕ v_MA), from a subsample.
        self.normalization_stats: Optional[dict] = None
        self.rel_mse_eps: float = 1e-8
        if compute_normalization or normalization_samples > 0:
            self._compute_stats(normalization_samples, show_progress)

    # ── generation loading ─────────────────────────────────────────────

    def _load_gen(self, idx: int) -> "refiner_data.RefinerGeneration":
        name = self.entries[idx]["name"]
        if name in self._gen_cache:
            gen = self._gen_cache.pop(name)
            self._gen_cache[name] = gen  # LRU refresh
            return gen
        gen = refiner_data.load_generation(self.data_dir / self.entries[idx]["bin"])
        self._gen_cache[name] = gen
        while len(self._gen_cache) > MAX_CACHED_GENERATIONS:
            self._gen_cache.popitem(last=False)
        return gen

    def _compute_stats(self, n_samples: int, show_progress: bool = True):
        """Per-channel mean/std of the 32ch input (x_t ⊕ v_MA) and the rel-MSE ε
        floor, from one shared random subsample — one load per pair (plan 6d/6f)."""
        means, sqs, v_sq = [], [], []
        idxs = list(range(len(self.pairs)))
        self.rng.shuffle(idxs)
        pbar = None
        if show_progress and tqdm is not None and n_samples > 0:
            pbar = tqdm(total=min(n_samples, len(idxs)), desc="dataset stats",
                        unit="pair", leave=False, dynamic_ncols=True)
        n = 0
        for i in idxs:
            if n >= n_samples:
                break
            x, v, vt, _ = self._load_pair(i)
            z = torch.cat([x, v], dim=0).float()
            means.append(z.mean(dim=(1, 2)))
            sqs.append((z * z).mean(dim=(1, 2)))
            v_sq.append(vt.float().square().mean().item())
            n += 1
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()
        if n:
            mean = torch.stack(means).mean(dim=0)
            var = torch.stack(sqs).mean(dim=0) - mean * mean
            std = var.clamp_min(1e-6).sqrt()
            self.normalization_stats = {"mean": mean.tolist(), "std": std.tolist()}
            mean_v_sq = sum(v_sq) / len(v_sq)
            self.rel_mse_eps = max(mean_v_sq * self.rel_mse_eps_scale, 1e-8)

    def _load_pair(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        gi, slot, t, lag = self.pairs[i]
        gen = self._load_gen(gi)
        xs = gen.x[slot]
        vts = gen.v_true[slot]
        vmas = gen.v_ma[slot][t]
        x = xs[t]
        v_true = vts[t]
        if lag == 0:
            v_ma = v_true
        else:
            v_ma = vmas.get(lag)
            if v_ma is None:  # defensive: ring-availability drift → skip via caller
                raise KeyError(f"lag {lag} missing at step {t} in {gen.name}")
        fracs = gen.step_fractions.get(slot) or [0.0] * len(vts)
        meta = {
            "t_frac": float(fracs[t]) if t < len(fracs) else 0.0,
            "lag": lag,
            "slot": slot,
            "prompt": gen.prompt.get(slot),
        }
        return x, v_ma, v_true, meta

    # ── Dataset API ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict:
        x, v_ma, v_true, meta = self._load_pair(i)
        return {
            "x_t": x, "v_ma": v_ma, "v_true": v_true,
            "t_frac": meta["t_frac"], "lag": meta["lag"], "slot": meta["slot"],
            "prompt": meta["prompt"],
        }

    def pair_shapes(self) -> List[Tuple[int, int]]:
        return self.pair_shape


def collate_corrector(batch: List[dict]) -> dict:
    """Stack pairs; prompts padded to batch max N with an attention mask.

    Returns fp16 x_t / v_ma / v_true; prompt (B, N, 1024) + mask (B, N) bool
    (rows with no real tokens = the 0-token uncond form); t_frac/lag/slot as
    tensors. Augmentation is applied in the training step, not here (plan 7.4).
    """
    x_t = torch.stack([b["x_t"] for b in batch]).to(torch.float16)
    v_ma = torch.stack([b["v_ma"] for b in batch]).to(torch.float16)
    v_true = torch.stack([b["v_true"] for b in batch]).to(torch.float16)
    t_frac = torch.tensor([b["t_frac"] for b in batch], dtype=torch.float32)
    lag = torch.tensor([b["lag"] for b in batch], dtype=torch.long)
    slot = torch.tensor([b["slot"] for b in batch], dtype=torch.long)

    prompts = [b["prompt"] for b in batch]
    max_n = max((p.shape[0] for p in prompts if p is not None and p.numel()), default=0)
    if max_n:
        padded = torch.zeros(len(batch), max_n, prompts[0].shape[1], dtype=torch.float16)
        mask = torch.zeros(len(batch), max_n, dtype=torch.bool)
        for i, p in enumerate(prompts):
            if p is not None and p.numel():
                n = p.shape[0]
                padded[i, :n] = p[:n].to(torch.float16)
                mask[i, :n] = True
    else:
        padded = torch.zeros(len(batch), 0, 1024, dtype=torch.float16)
        mask = torch.zeros(len(batch), 0, dtype=torch.bool)

    return {
        "x_t": x_t, "v_ma": v_ma, "v_true": v_true,
        "prompt": padded, "prompt_mask": mask,
        "t_frac": t_frac, "lag": lag, "slot": slot,
    }


def augment_batch(batch: dict, rng: random.Random) -> dict:
    """Joint spatial flips + optional 90° rotations on GPU (plan 7.4 / 6f).

    Applies the same op to x_t, v_ma, v_true. No crops, no color.
    """
    b = batch["x_t"].shape[0]
    for i in range(b):
        if rng.random() < 0.5:
            batch["x_t"][i] = batch["x_t"][i].flip(-1)
            batch["v_ma"][i] = batch["v_ma"][i].flip(-1)
            batch["v_true"][i] = batch["v_true"][i].flip(-1)
        if rng.random() < 0.5:
            batch["x_t"][i] = batch["x_t"][i].flip(-2)
            batch["v_ma"][i] = batch["v_ma"][i].flip(-2)
            batch["v_true"][i] = batch["v_true"][i].flip(-2)
        if rng.random() < 0.25:
            k = rng.randint(1, 3)
            batch["x_t"][i] = torch.rot90(batch["x_t"][i], k, (-2, -1))
            batch["v_ma"][i] = torch.rot90(batch["v_ma"][i], k, (-2, -1))
            batch["v_true"][i] = torch.rot90(batch["v_true"][i], k, (-2, -1))
    return batch


class CorrectorBatchSampler(Sampler):
    """Resolution-grouped, weighted, deterministic batch sampler.

    - Pairs are bucketed by latent shape; 1024² batches use batch_size ÷ 4.
    - Within a bucket, pairs are drawn by weight (lag resampling) without
      replacement per epoch (``torch.multinomial``); buckets' batch lists are
      interleaved and shuffled (seeded; epoch counter advances each iter).
    """

    def __init__(self, dataset: CorrectorDataset, batch_size: int, seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self._epoch = 0
        self.buckets: Dict[Tuple[int, int], List[int]] = {}
        for i, (h, w) in enumerate(dataset.pair_shapes()):
            self.buckets.setdefault((h, w), []).append(i)

    def __len__(self) -> int:
        total = 0
        for idxs in self.buckets.values():
            bs = self._bucket_batch_size(idxs)
            total += (len(idxs) + bs - 1) // bs
        return total

    def _bucket_batch_size(self, idxs: List[int]) -> int:
        h, w = self.dataset.pair_shape[idxs[0]]
        return max(1, self.batch_size // 4) if h * w > 64 * 64 else self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        epoch = self._epoch
        self._epoch += 1
        gen = torch.Generator().manual_seed(self.seed + epoch)
        epoch_batches = []
        for idxs in self.buckets.values():
            bs = self._bucket_batch_size(idxs)
            w = self.dataset.weights[torch.tensor(idxs, dtype=torch.long)]
            order = torch.multinomial(w, num_samples=len(idxs), replacement=False,
                                      generator=gen).tolist()
            order = [idxs[j] for j in order]
            epoch_batches.extend(order[k:k + bs] for k in range(0, len(order), bs))
        rng.shuffle(epoch_batches)
        yield from epoch_batches
