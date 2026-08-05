"""Refiner latent-data storage (storage v1) — binary format, codecs, manifest.

The refiner recording pipeline captures, per generation, per step, per recorded
CFG slot, the 16-channel image latents ``x_t`` (pre-pad model input) and ``v_t``
(the post-unpatchify full-model output the sampler consumes), plus prompt
embeddings (once per generation). ``v_t`` is stored directly (no anchor+delta
chain) so reconstruction is exact by definition.

Layout (per calibration run):
    outputs/<timestamp>/refiner_data/
    ├── manifest.json
    ├── gen_0000_p03_s34635345.bin          # per-step x_t/v_t per slot
    ├── gen_0000_p03_s34635345.prompt.bin   # per-slot prompt embeddings
    └── ...

``.bin`` format (little-endian, struct-packed):
    header:   magic "TCREF" | format_version u8 | codec_id u8 | dtype_id u8
              | num_slots u8 | C u32 | H u32 | W u32 | num_steps u32 | metadata_len u32
    metadata: JSON bytes (per-generation metadata + per-slot step/timestep arrays)
    offsets:  num_steps × num_slots × 2 × u64  (x_t blob, then v_t blob, per step/slot)
    body:     compressed blobs, one per offset-table entry

``.prompt.bin`` format:
    header:   magic "TCPRT" | format_version u8 | codec_id u8 | dtype_id u8
              | D u32 | num_slots u8 | metadata_len u32
    metadata: JSON bytes (slot order, per-slot token counts)
    body:     per slot: N u32 | blob_len u64 | compressed (N, D) embedding blob

Codecs: 0 = raw bf16 (fallback), 1 = blosc2 bitshuffle+zstd level 9 on
bf16-as-int16 views (v1 default), 2 = zfpy reversible (bf16→fp32 upcast;
requires ``pip install zfpy``). Codec 1/2 fall back gracefully to raw when the
package is missing, so the calibration pipeline never hard-fails on compression.
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch

MAGIC_BIN = b"TCREF"
MAGIC_PROMPT = b"TCPRT"
FORMAT_VERSION = 1

CODEC_RAW = 0
CODEC_BLOSC2 = 1
CODEC_ZFPY = 2
CODEC_NAMES = {
    CODEC_RAW: "raw",
    CODEC_BLOSC2: "blosc2-bitshuffle-zstd",
    CODEC_ZFPY: "zfpy-reversible",
}

_DTYPE_IDS = {"bfloat16": 0, "float16": 1, "float32": 2}
_DTYPE_NAMES = {v: k for k, v in _DTYPE_IDS.items()}
_TORCH_DTYPES = {0: torch.bfloat16, 1: torch.float16, 2: torch.float32}

_HDR_BIN = struct.Struct("<5sBBBBIIIII")
_HDR_PROMPT = struct.Struct("<5sBBBIBI")
_ENTRY_HEAD = struct.Struct("<IQ")
_OFF = struct.Struct("<Q")


# ══════════════════════════════════════════════════════════════════════════
#  Codecs
# ══════════════════════════════════════════════════════════════════════════

_warned_codecs: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned_codecs:
        _warned_codecs.add(key)
        print(f"  [refiner] {msg}")


def _blosc2():
    try:
        import blosc2
        return blosc2
    except ImportError:
        return None


def _zfpy():
    try:
        import zfpy
        return zfpy
    except ImportError:
        return None


def _resolve_codec(codec_id: int) -> int:
    """Return the codec that can actually be used, falling back gracefully."""
    if codec_id == CODEC_BLOSC2 and _blosc2() is None:
        _warn_once(
            "blosc2-missing",
            "blosc2 not installed — storing refiner data raw (codec 0). "
            "pip install blosc2 for lossless compression.",
        )
        return CODEC_RAW
    if codec_id == CODEC_ZFPY:
        if _zfpy() is None:
            _warn_once(
                "zfpy-missing",
                "zfpy not installed — falling back to blosc2/raw for refiner data.",
            )
            return _resolve_codec(CODEC_BLOSC2)
    return codec_id


def _typesize(dtype_id: int) -> int:
    return 4 if dtype_id == _DTYPE_IDS["float32"] else 2


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    t = t.contiguous()
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16).numpy().tobytes()
    if t.dtype == torch.float32:
        return t.view(torch.int32).numpy().tobytes()
    return t.numpy().tobytes()


def _tensor_from_bytes(data: bytes, dtype_id: int,
                       shape: Tuple[int, ...]) -> torch.Tensor:
    dtype = _TORCH_DTYPES[dtype_id]
    if len(data) == 0:
        return torch.empty(tuple(shape), dtype=dtype)
    np_dtype = np.int16 if dtype in (torch.bfloat16, torch.float16) else np.int32
    arr = np.frombuffer(data, dtype=np_dtype).copy()
    return torch.from_numpy(arr).view(dtype).reshape(shape)


def _compress(t: torch.Tensor, codec_id: int, dtype_id: int) -> bytes:
    if t is None or t.numel() == 0:
        return b""
    raw = _tensor_to_bytes(t)
    if codec_id == CODEC_RAW:
        return raw
    if codec_id == CODEC_BLOSC2:
        b2 = _blosc2()
        if b2 is not None:
            return b2.compress2(
                raw, typesize=_typesize(dtype_id), clevel=9,
                filter=b2.Filter.BITSHUFFLE, codec=b2.Codec.ZSTD,
            )
    if codec_id == CODEC_ZFPY:
        z = _zfpy()
        if z is not None:
            arr = t.numpy() if t.dtype == torch.float32 else t.float().numpy()
            try:
                return z.compress_numpy(arr, reversible=True)
            except TypeError:
                # older zfpy API: no reversible flag; negative tolerance = lossless
                return z.compress_numpy(arr)
    return raw


def _decompress(data: bytes, codec_id: int, dtype_id: int,
                shape: Tuple[int, ...]) -> torch.Tensor:
    if len(data) == 0:
        return torch.empty(tuple(shape), dtype=_TORCH_DTYPES[dtype_id])
    if codec_id == CODEC_BLOSC2:
        b2 = _blosc2()
        if b2 is None:
            raise RuntimeError(
                "blosc2 is required to read codec-1 refiner data; pip install blosc2"
            )
        return _tensor_from_bytes(b2.decompress2(data), dtype_id, shape)
    if codec_id == CODEC_ZFPY:
        z = _zfpy()
        if z is None:
            raise RuntimeError(
                "zfpy is required to read codec-2 refiner data; pip install zfpy"
            )
        t = torch.from_numpy(z.decompress_numpy(data))
        t = t.to(_TORCH_DTYPES[dtype_id])
        return t.reshape(shape)
    return _tensor_from_bytes(data, dtype_id, shape)


# ══════════════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class RefinerRecording:
    """One generation's recorded tensors (mirrors the recorder's _refiner_buf).

    ``x``/``v`` are dicts of slot -> per-step tensors (C, H, W) in sampling
    order (step 0 = first denoising call). ``prompt`` is slot -> (N, D) bf16,
    captured once per generation. ``metadata`` carries prompt_id, seed, sampler,
    scheduler, cfg, width, height, steps, fps, eval, volatility.
    """
    x: Dict[int, List[torch.Tensor]]
    v: Dict[int, List[torch.Tensor]]
    prompt: Dict[int, Optional[torch.Tensor]]
    timesteps: Dict[int, List[float]]
    steps: Dict[int, List[int]]
    step_fractions: Dict[int, List[float]]
    metadata: dict = field(default_factory=dict)


@dataclass
class RefinerGeneration:
    """One generation loaded from disk (tensors returned as stored)."""
    name: str
    metadata: dict
    x: Dict[int, List[torch.Tensor]]
    v: Dict[int, List[torch.Tensor]]
    prompt: Dict[int, Optional[torch.Tensor]]
    timesteps: Dict[int, List[float]]
    steps: Dict[int, List[int]]
    step_fractions: Dict[int, List[float]]
    num_steps: int
    recorded_slots: List[int]
    prompt_id: int
    codec_id: int
    dtype_id: int
    shape: Tuple[int, int, int]


# ══════════════════════════════════════════════════════════════════════════
#  Writing
# ══════════════════════════════════════════════════════════════════════════


def write_generation(refiner_dir: Union[str, Path], recording: RefinerRecording,
                     name: str, codec_id: int = CODEC_BLOSC2
                     ) -> Tuple[Path, Path, dict]:
    """Write one generation's .bin + .prompt.bin.

    Compression happens once, in one batch at generation end (not per step
    inside the sampling loop). Returns (bin_path, prompt_path, manifest_entry).
    """
    refiner_dir = Path(refiner_dir)
    refiner_dir.mkdir(parents=True, exist_ok=True)
    codec_id = _resolve_codec(codec_id)

    slots = sorted(recording.x.keys())
    if not slots:
        raise ValueError(f"write_generation: no recorded slots for {name}")
    num_steps = len(recording.x[slots[0]])
    for s in slots:
        if len(recording.x[s]) != num_steps or len(recording.v[s]) != num_steps:
            raise ValueError(f"write_generation: inconsistent step counts in {name}")

    first = recording.x[slots[0]][0]
    dtype_name = str(first.dtype).replace("torch.", "")
    dtype_id = _DTYPE_IDS.get(dtype_name, _DTYPE_IDS["bfloat16"])
    c, h, w = int(first.shape[0]), int(first.shape[1]), int(first.shape[2])

    meta = dict(recording.metadata)
    meta.update({
        "slots": slots,
        "steps_per_slot": {str(s): recording.steps.get(s, []) for s in slots},
        "step_fractions_per_slot": {str(s): recording.step_fractions.get(s, []) for s in slots},
        "timesteps_per_slot": {str(s): recording.timesteps.get(s, []) for s in slots},
    })
    meta_bytes = json.dumps(meta, default=str).encode("utf-8")

    def blob(t: torch.Tensor) -> bytes:
        return _compress(t, codec_id, dtype_id)

    def build_blobs() -> List[bytes]:
        blobs = []
        for t in range(num_steps):
            for s in slots:
                blobs.append(blob(recording.x[s][t]))
                blobs.append(blob(recording.v[s][t]))
        return blobs

    blobs = build_blobs()

    header = _HDR_BIN.pack(MAGIC_BIN, FORMAT_VERSION, codec_id, dtype_id,
                           len(slots), c, h, w, num_steps, len(meta_bytes))
    off_count = num_steps * len(slots) * 2
    cursor = len(header) + len(meta_bytes) + off_count * _OFF.size
    offsets = []
    for b in blobs:
        offsets.append(cursor)
        cursor += len(b)

    bin_path = refiner_dir / f"{name}.bin"
    with bin_path.open("wb") as f:
        f.write(header)
        f.write(meta_bytes)
        for o in offsets:
            f.write(_OFF.pack(o))
        for b in blobs:
            f.write(b)

    # ── prompt file ──
    prompt_d = 0
    prompt_ns = []
    prompt_blobs = []
    for s in slots:
        p = recording.prompt.get(s)
        if p is not None and p.numel():
            prompt_d = int(p.shape[1])
        prompt_ns.append(int(p.shape[0]) if p is not None else 0)
        prompt_blobs.append(_compress(p, codec_id, dtype_id) if p is not None else b"")
    prompt_meta = {
        "slots": slots,
        "prompt_n": {str(s): n for s, n in zip(slots, prompt_ns)},
        "dtype": _DTYPE_NAMES[dtype_id],
    }
    prompt_meta_bytes = json.dumps(prompt_meta, default=str).encode("utf-8")
    header2 = _HDR_PROMPT.pack(MAGIC_PROMPT, FORMAT_VERSION, codec_id, dtype_id,
                               prompt_d, len(slots), len(prompt_meta_bytes))
    prompt_path = refiner_dir / f"{name}.prompt.bin"
    with prompt_path.open("wb") as f:
        f.write(header2)
        f.write(prompt_meta_bytes)
        for n, b in zip(prompt_ns, prompt_blobs):
            f.write(_ENTRY_HEAD.pack(n, len(b)))
            f.write(b)

    entry = {
        "name": name,
        "bin": bin_path.name,
        "prompt_bin": prompt_path.name,
        "codec_id": codec_id,
        "codec": CODEC_NAMES.get(codec_id, str(codec_id)),
        "dtype": _DTYPE_NAMES[dtype_id],
        "shape": [c, h, w],
        "slots": slots,
        "num_steps": num_steps,
        "bytes": bin_path.stat().st_size,
        "prompt_bytes": prompt_path.stat().st_size,
    }
    for k, v in recording.metadata.items():
        entry.setdefault(k, v)
    return bin_path, prompt_path, entry


# ══════════════════════════════════════════════════════════════════════════
#  Reading
# ══════════════════════════════════════════════════════════════════════════


def load_generation(bin_path: Union[str, Path]) -> RefinerGeneration:
    """Load one generation .bin (+ its .prompt.bin) — tensors as stored."""
    bin_path = Path(bin_path)
    data = bin_path.read_bytes()
    (magic, version, codec_id, dtype_id, num_slots,
     c, h, w, num_steps, meta_len) = _HDR_BIN.unpack_from(data, 0)
    if magic != MAGIC_BIN:
        raise ValueError(f"not a refiner .bin file: {bin_path}")
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported refiner format version {version} in {bin_path}")

    meta = json.loads(data[_HDR_BIN.size:_HDR_BIN.size + meta_len].decode("utf-8"))
    slots = [int(s) for s in meta.get("slots", [])]
    if len(slots) != num_slots:
        raise ValueError(f"slot count mismatch in {bin_path}: header {num_slots} vs metadata {len(slots)}")

    off_start = _HDR_BIN.size + meta_len
    off_count = num_steps * num_slots * 2
    offsets = [_OFF.unpack_from(data, off_start + 8 * i)[0] for i in range(off_count)]
    end = len(data)
    blobs = [data[offsets[i]: (offsets[i + 1] if i + 1 < off_count else end)]
             for i in range(off_count)]

    shape = (c, h, w)
    x, v = {}, {}
    idx = 0
    for t in range(num_steps):
        for s in slots:
            x.setdefault(s, []).append(_decompress(blobs[idx], codec_id, dtype_id, shape))
            v.setdefault(s, []).append(_decompress(blobs[idx + 1], codec_id, dtype_id, shape))
            idx += 2

    steps = {s: meta.get("steps_per_slot", {}).get(str(s), list(range(num_steps)))
             for s in slots}
    step_fractions = {s: meta.get("step_fractions_per_slot", {}).get(str(s), [])
                      for s in slots}
    timesteps = {s: meta.get("timesteps_per_slot", {}).get(str(s), []) for s in slots}
    prompt = load_prompt_file(bin_path.with_name(bin_path.stem + ".prompt.bin"))

    return RefinerGeneration(
        name=bin_path.stem,
        metadata=meta,
        x=x, v=v, prompt=prompt,
        timesteps=timesteps, steps=steps, step_fractions=step_fractions,
        num_steps=num_steps,
        recorded_slots=slots,
        prompt_id=int(meta.get("prompt_id", -1)),
        codec_id=codec_id, dtype_id=dtype_id,
        shape=shape,
    )


def load_prompt_file(path: Union[str, Path]) -> Dict[int, Optional[torch.Tensor]]:
    """Load a .prompt.bin file — slot -> (N, D) bf16 tensor (or empty (0, D))."""
    path = Path(path)
    if not path.exists():
        return {}
    data = path.read_bytes()
    (magic, version, codec_id, dtype_id, d, num_slots, meta_len) = \
        _HDR_PROMPT.unpack_from(data, 0)
    if magic != MAGIC_PROMPT:
        raise ValueError(f"not a refiner .prompt.bin file: {path}")
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported refiner format version {version} in {path}")
    meta = json.loads(data[_HDR_PROMPT.size:_HDR_PROMPT.size + meta_len].decode("utf-8"))
    slots = [int(s) for s in meta.get("slots", [])]
    prompt_n = {int(k): int(v) for k, v in meta.get("prompt_n", {}).items()}
    out: Dict[int, Optional[torch.Tensor]] = {}
    cursor = _HDR_PROMPT.size + meta_len
    for s in slots:
        n, blob_len = _ENTRY_HEAD.unpack_from(data, cursor)
        cursor += _ENTRY_HEAD.size
        blob = data[cursor:cursor + blob_len]
        cursor += blob_len
        out[s] = _decompress(blob, codec_id, dtype_id, (int(n), int(d)))
    return out


# ══════════════════════════════════════════════════════════════════════════
#  Manifest
# ══════════════════════════════════════════════════════════════════════════


def init_manifest(refiner_cfg: Optional[dict] = None) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "refiner_config": dict(refiner_cfg or {}),
        "generations": [],
    }


def load_manifest(refiner_dir: Union[str, Path]) -> dict:
    p = Path(refiner_dir) / "manifest.json"
    if not p.exists():
        return init_manifest()
    with open(p) as f:
        return json.load(f)


def save_manifest(refiner_dir: Union[str, Path], manifest: dict) -> None:
    refiner_dir = Path(refiner_dir)
    refiner_dir.mkdir(parents=True, exist_ok=True)
    tmp = refiner_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, default=str))
    tmp.replace(refiner_dir / "manifest.json")


def iter_generations(refiner_dir: Union[str, Path]) -> List[dict]:
    """List manifest entries for all recorded generations."""
    return load_manifest(refiner_dir).get("generations", [])


def iter_pairs(refiner_dir: Union[str, Path], include_eval: bool = False) -> Iterator[dict]:
    """Yield training pairs (x_t, v_prev, v_t, prompt, t_frac, slot) per step.

    Consecutive-step pairs per slot; eval-prompt generations are excluded by
    default (their recordings exist but are held out from training, plan 6e).
    """
    for entry in iter_generations(refiner_dir):
        if entry.get("eval") and not include_eval:
            continue
        gen = load_generation(Path(refiner_dir) / entry["bin"])
        for slot in gen.recorded_slots:
            xs, vs = gen.x[slot], gen.v[slot]
            fracs = gen.step_fractions.get(slot) or [0.0] * len(vs)
            for t in range(1, len(vs)):
                yield {
                    "x_t": xs[t],
                    "v_prev": vs[t - 1],
                    "v_t": vs[t],
                    "prompt": gen.prompt.get(slot),
                    "t_frac": fracs[t] if t < len(fracs) else 0.0,
                    "slot": slot,
                    "generation": gen,
                }


# ══════════════════════════════════════════════════════════════════════════
#  Volatility ranking + eviction (Task 2c)
# ══════════════════════════════════════════════════════════════════════════


def compute_volatility(v_per_slot: Dict[int, List[torch.Tensor]]) -> float:
    """score = mean over steps/slots of mean(|v_t − v_{t−1}|)."""
    scores = []
    for vs in v_per_slot.values():
        deltas = []
        for a, b in zip(vs[:-1], vs[1:]):
            deltas.append((b.float() - a.float()).abs().mean().item())
        if deltas:
            scores.append(float(np.mean(deltas)))
    return float(np.mean(scores)) if scores else 0.0


def evict_generations(manifest: dict, refiner_dir: Union[str, Path],
                      refiner_cfg: dict) -> int:
    """Apply the top_n cap: evict lowest-volatility generations per resolution.

    Ranking happens within-run only, per resolution stratum first (scores are
    resolution-dependent); eval generations always bypass ranking. Returns the
    number of evicted generations.
    """
    keep_all = bool(refiner_cfg.get("keep_all", True))
    top_n = refiner_cfg.get("top_n", -1)
    if keep_all or top_n is None or int(top_n) < 0:
        return 0
    top_n = int(top_n)
    refiner_dir = Path(refiner_dir)
    gens = manifest["generations"]
    strata: Dict[Tuple[int, int], List[dict]] = {}
    for e in gens:
        if e.get("eval"):
            continue
        strata.setdefault((int(e.get("width", 0)), int(e.get("height", 0))), []).append(e)
    removed = 0
    for entries in strata.values():
        while len(entries) > top_n:
            victim = min(entries, key=lambda e: e.get("volatility", 0.0))
            entries.remove(victim)
            gens.remove(victim)
            for fname in (victim.get("bin"), victim.get("prompt_bin")):
                if fname:
                    p = refiner_dir / fname
                    if p.exists():
                        p.unlink()
            removed += 1
    return removed


def finalize_refiner_generation(dm, refiner_dir: Union[str, Path], manifest: dict,
                                run_meta: dict, refiner_cfg: dict,
                                eval_prompt_ids=()):
    """Generation-end flow (plan 2a/2c): score, rank, evict, write, update manifest.

    ``dm`` is the patched diffusion model holding ``_refiner_buf``; ``run_meta``
    carries name/prompt_id/seed/sampler/scheduler/cfg/width/height/steps/fps.
    """
    buf = getattr(dm, "_refiner_buf", None)
    if not buf or not buf.get("x"):
        print(f"  [refiner] no recorded steps for {run_meta.get('name', '?')} — skipping")
        return None
    if not buf.get("v"):
        print(f"  [refiner] no v_t recorded for {run_meta.get('name', '?')} — skipping")
        return None

    metadata = dict(run_meta)
    metadata["eval"] = int(metadata.get("prompt_id", -1)) in set(eval_prompt_ids)
    metadata["volatility"] = compute_volatility(buf["v"])

    recording = RefinerRecording(
        x=buf["x"], v=buf["v"], prompt=buf.get("prompt", {}),
        timesteps=buf.get("timesteps", {}), steps=buf.get("steps", {}),
        step_fractions=buf.get("step_fractions", {}),
        metadata=metadata,
    )
    codec_id = int(refiner_cfg.get("codec", CODEC_BLOSC2))
    bin_path, prompt_path, entry = write_generation(
        Path(refiner_dir), recording, run_meta["name"], codec_id=codec_id,
    )
    manifest["generations"].append(entry)
    removed = evict_generations(manifest, Path(refiner_dir), refiner_cfg)
    save_manifest(Path(refiner_dir), manifest)
    if removed:
        print(f"  [refiner] evicted {removed} generation(s) (top_n={refiner_cfg.get('top_n')})")
    return entry


# ══════════════════════════════════════════════════════════════════════════
#  Resolution mix + disk estimate (Tasks 1 / 2d)
# ══════════════════════════════════════════════════════════════════════════


def parse_resolution(key: str) -> Tuple[int, int]:
    """Parse 'WxH' (e.g. '512x512') → (width, height)."""
    try:
        w, h = key.lower().split("x", 1)
        return int(w), int(h)
    except Exception as e:
        raise ValueError(
            f"invalid resolution {key!r} (expected 'WxH', e.g. '512x512')"
        ) from e


def parse_resolution_mix(spec) -> Dict[str, float]:
    """Normalize a resolution mix to {'WxH': share} with shares summing to 1.

    Accepts a dict (e.g. {"512x512": 0.8, "1024x1024": 0.2}) or a string
    (e.g. "512x512:0.8,1024x1024:0.2"; entries without ':share' get equal
    shares). Raises ValueError on malformed input.
    """
    if spec is None:
        return {"512x512": 1.0}
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if not parts:
            raise ValueError(f"empty resolution mix string: {spec!r}")
        items: Dict[str, Optional[float]] = {}
        for p in parts:
            if ":" in p:
                res, share = p.split(":", 1)
                items[res.strip().lower()] = float(share)
            else:
                items[p.strip().lower()] = None
        explicit = {k: v for k, v in items.items() if v is not None}
        implicit = [k for k, v in items.items() if v is None]
        if explicit and implicit:
            raise ValueError(
                f"mixed explicit/implicit shares in resolution mix {spec!r} "
                f"(use 'WxH:share' for all entries)"
            )
        if implicit:
            for k in implicit:
                items[k] = 1.0 / len(implicit)
        mix = {k: float(v) for k, v in items.items()}
    elif isinstance(spec, dict):
        mix = {str(k).lower(): float(v) for k, v in spec.items()}
    else:
        raise ValueError(
            f"resolution mix must be a dict or string, got {type(spec).__name__}"
        )
    if not mix:
        raise ValueError("resolution mix is empty")
    for key, share in mix.items():
        parse_resolution(key)
        if share <= 0:
            raise ValueError(f"resolution share must be > 0, got {share} for {key}")
    total = sum(mix.values())
    if total <= 0:
        raise ValueError(f"resolution mix shares must sum to > 0: {mix}")
    return {k: v / total for k, v in mix.items()}


def estimate_refiner_disk_bytes(total_generations: int, avg_steps: float,
                                resolution_mix: Optional[Dict[str, float]],
                                record_slots: str) -> float:
    """Lossless disk estimate for the schedule summary (Task 2d).

    Plan §4 figures @512², 30 steps: ~5.8 MB/gen cond-only, ~11.7 MB/gen both
    slots: per step x_t 128KB/1.2 + v_t 128KB/3 (lossless), prompt ~1.3MB/slot.
    Scales with resolution ratio and the actual step mix.
    """
    slots = 2 if record_slots == "both" else 1
    x_bytes = 128 * 1024 / 1.2
    v_bytes = 128 * 1024 / 3.0
    prompt_bytes = 1.3 * 1024 * 1024
    per_gen_512 = avg_steps * slots * (x_bytes + v_bytes) + slots * prompt_bytes
    total = 0.0
    for key, share in (resolution_mix or {"512x512": 1.0}).items():
        w, h = parse_resolution(key)
        total += share * total_generations * per_gen_512 * ((w * h) / (512.0 * 512.0))
    return total
