"""ComfyUI integration: model loading, sampling, metric computation.

IMPORTANT: These scripts must be run with ComfyUI's root directory as the
first entry in sys.path. Use the wrapper script or set PYTHONPATH.
See: run from ComfyUI root with python -m
    PYTHONPATH=".:custom_nodes/ComfyUI-TeaCache-CosmosPredict"
"""

import json
from collections import deque

import sys
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def setup_comfy_path(comfy_dir: str) -> None:
    """Ensure ComfyUI root is first in sys.path so 'import nodes' finds
    ComfyUI's nodes.py, not this addon's."""
    # Clear any stale cache in case the addon's nodes.py was imported earlier
    sys.modules.pop("nodes", None)
    if comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)


def load_models(comfy_dir: str,     model_name: str,
                clip_name: str,     clip_type: str,
                vae_name: str):
    """Load Anima UNet + CLIP + VAE via ComfyUI loaders."""
    setup_comfy_path(comfy_dir)

    import folder_paths
    import nodes

    mdir = str(Path(comfy_dir) / "models")
    folder_paths.add_model_folder_path("diffusion_models", mdir + "/diffusion_models")
    folder_paths.add_model_folder_path("text_encoders",    mdir + "/text_encoders")
    folder_paths.add_model_folder_path("vae",              mdir + "/vae")

    print(f"[load] UNet: {model_name}")
    unet = nodes.UNETLoader().load_unet(model_name, "default")[0]

    print(f"[load] CLIP: {clip_name} ({clip_type})")
    clip = nodes.CLIPLoader().load_clip(clip_name, clip_type, "default")[0]

    print(f"[load] VAE: {vae_name}")
    vae = nodes.VAELoader().load_vae(vae_name)[0]

    print("[load] All models ready")
    return unet, clip, vae


def sample(unet, clip, vae, prompt: str, *,
           seed: int = 42, steps: int = 30,
           cfg: float = 5.0,
           sampler_name: str = "er_sde",
           scheduler: str = "normal",
           width: int = 1024, height: int = 1024,
           negative: str = "",
           return_latent: bool = False):
    """Run a full sampling pass. Returns PIL.Image (or (latent, image) tuple)."""
    import nodes
    from PIL import Image

    pos = nodes.CLIPTextEncode().encode(clip, prompt)[0]
    neg = nodes.CLIPTextEncode().encode(clip, negative)[0]
    latent = nodes.EmptyLatentImage().generate(width, height, 1)[0]
    samples = nodes.KSampler().sample(
        unet, seed, steps, cfg, sampler_name, scheduler, pos, neg, latent, 1.0
    )[0]
    decoded = nodes.VAEDecode().decode(vae, samples)[0]
    arr = (decoded.detach().cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
    if arr.ndim == 4:
        arr = arr[0]
    img = Image.fromarray(arr)
    if return_latent:
        return samples, img
    return img


def img_to_tensor(img) -> torch.Tensor:
    """PIL Image -> (1, 3, H, W) float32 tensor on cuda."""
    from PIL import Image
    if isinstance(img, Image.Image):
        arr = np.asarray(img).astype("float32") / 255.0
    else:
        arr = img.astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to("cuda")


_PYIQA_AVAILABLE = False
_PYIQA_WARNED = False

# ── Metric definitions ────────────────────────────────────────────────
# Each tier adds progressively more expensive metrics.
# Tier 1: essential perceptual metrics (always computed)
# Tier 2: structure/texture/edge metrics (moderate cost)
# Tier 3: human-preference and specialized metrics (expensive)

_TIER1_METRICS = {
    "psnr":        "psnr",           # Pixel accuracy (higher=better)
    "ssim":        "ssim",           # Structural similarity (higher=better)
    "lpips_alex":  "lpips",          # Perceptual, AlexNet backbone (lower=better)
    "lpips_vgg":   "lpips-vgg",      # Perceptual, VGG16 backbone — catches texture drift better (lower=better)
    "dists":       "dists",          # Separates structure vs texture quality (lower=better)
    "ms_ssim":     "ms_ssim",        # Multi-scale SSIM, catches scale-specific artifacts (higher=better)
}

_TIER2_METRICS = {
    "fsim":        "fsim",           # Edge/sharpness via phase congruency (higher=better)
    "vif":         "vif",            # Information fidelity — measures info loss from caching (higher=better)
    "gmsd":        "gmsd",           # Gradient deviation — very sensitive to blur (lower=better)
}

_TIER3_METRICS = {
    "nlpd":        "nlpd",           # Normalized Laplacian Pyramid — human visual system model (lower=better)
    "pieapp":      "pieapp",         # Trained on human pairwise preference — gold standard (lower=better)
    "vsi":         "vsi",            # Visual saliency-weighted — penalizes degradation in important regions (higher=better)
}


class QualityMetrics:
    """Multi-metric image quality assessment via pyiqa.

    Metrics are lazily loaded by tier to minimize GPU memory.
    Each call to measure() returns a dict of named scores.

    Usage:
        qm = QualityMetrics(tier=1)
        scores = qm.measure(img_pred, img_ref)
    """

    _all_metric_names = (
        list(_TIER1_METRICS.keys())
        + list(_TIER2_METRICS.keys())
        + list(_TIER3_METRICS.keys())
    )

    def __init__(self, tier: int = 1):
        self.tier = tier
        self._pyiqa = None
        self._metrics: dict[str, object] = {}
        self._device = torch.device("cuda")
        self._loaded = False

    @property
    def available(self) -> bool:
        if self._loaded:
            return True
        try:
            import pyiqa
            self._pyiqa = pyiqa
            self._loaded = True
            return True
        except ImportError:
            return False

    def _warn_once(self):
        global _PYIQA_WARNED
        if not _PYIQA_WARNED:
            _PYIQA_WARNED = True
            print(
                "\n  ⚠ WARNING: pyiqa is not installed. "
                "Quality metrics are UNAVAILABLE.\n"
                "  Install with: pip install -r tuning/requirements.txt\n"
                "  All metric values below are placeholders — NOT real measurements.\n"
            )

    def _create_metric(self, friendly_name: str, pyiqa_name: str) -> None:
        if friendly_name in self._metrics or not self.available:
            return
        try:
            self._metrics[friendly_name] = self._pyiqa.create_metric(
                pyiqa_name, device=self._device
            )
        except Exception:
            self._metrics[friendly_name] = None  # mark as unavailable

    def measure(self, img_pred, img_ref) -> dict[str, float]:
        """Return dict of all applicable metric scores for one image pair."""
        if not self.available:
            self._warn_once()
            return {k: float("inf") for k in self._all_metric_names}

        t_pred = img_to_tensor(img_pred)
        t_ref = img_to_tensor(img_ref)

        # Lazy init all metrics on first call
        for friendly, pyiqa_name in _TIER1_METRICS.items():
            self._create_metric(friendly, pyiqa_name)
        if self.tier >= 2:
            for friendly, pyiqa_name in _TIER2_METRICS.items():
                self._create_metric(friendly, pyiqa_name)
        if self.tier >= 3:
            for friendly, pyiqa_name in _TIER3_METRICS.items():
                self._create_metric(friendly, pyiqa_name)

        scores = {}
        for name in self._all_metric_names:
            m = self._metrics.get(name)
            if m is not None:
                try:
                    scores[name] = float(m(t_pred, t_ref).item())
                except Exception:
                    scores[name] = float("nan")
            else:
                scores[name] = float("nan")

        return scores

    def metric_names(self) -> list[str]:
        """Return metric names active for current tier."""
        names = list(_TIER1_METRICS.keys())
        if self.tier >= 2:
            names += list(_TIER2_METRICS.keys())
        if self.tier >= 3:
            names += list(_TIER3_METRICS.keys())
        return names


# ── Legacy compatibility wrapper ──────────────────────────────────────

def compute_quality_metrics(
    img_pred, img_ref
) -> Tuple[float, float, float]:
    """Legacy wrapper — returns (PSNR, SSIM, LPIPS-alex)."""
    _global_qm = QualityMetrics(tier=1)
    scores = _global_qm.measure(img_pred, img_ref)
    return (
        scores.get("psnr", float("inf")),
        scores.get("ssim", 1.0),
        scores.get("lpips_alex", 0.0),
    )


def get_diffusion_model(unet):
    """Get the underlying MiniTrainDIT from a ComfyUI ModelPatcher."""
    return unet.get_model_object("diffusion_model")


def measure_vram() -> float:
    """Return peak VRAM usage in GB."""
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


# ═══════════════════════════════════════════════════════════════════════════
#  Shared metric legend (used by both validate.py and smoke_test.py)
# ═══════════════════════════════════════════════════════════════════════════

METRIC_LEGEND = [
    ("psnr",       "↑", "pixel-level accuracy",           35.0,  25.0),
    ("ssim",       "↑", "structural similarity",          0.95,  0.85),
    ("lpips_alex", "↓", "perceptual (AlexNet, semantic)", 0.05,  0.15),
    ("lpips_vgg",  "↓", "perceptual (VGG16, texture)",    0.10,  0.25),
    ("dists",      "↓", "structure vs texture decomp",    0.05,  0.15),
    ("ms_ssim",    "↑", "multi-scale structural simil.",  0.97,  0.92),
    ("fsim",       "↑", "edge sharpness (phase congru.)", 0.97,  0.90),
    ("vif",        "↑", "information fidelity",           0.60,  0.30),
    ("gmsd",       "↓", "gradient deviation (blur)",      0.05,  0.15),
    ("nlpd",       "↓", "Laplacian pyramid (human vis.)", 0.10,  0.25),
    ("pieapp",     "↓", "human pairwise preference",      0.10,  0.30),
    ("vsi",        "↑", "visual saliency-weighted simil.", 0.97,  0.90),
]


def print_metrics_legend():
    """Print the HOW TO READ METRICS legend box (shared by validate + smoke test)."""
    COL_METRIC = 12
    COL_DIR    = 3
    COL_GOOD   = 7
    COL_MID    = 14
    COL_POOR   = 7
    COL_WHAT   = 35
    SPACER = " │ "

    def _row(metric, dir_str, gs, ms, ps, what):
        return (f"{metric:>{COL_METRIC}}{SPACER}"
                f"{dir_str:^{COL_DIR}}{SPACER}"
                f"{gs:>{COL_GOOD}}{SPACER}"
                f"{ms:>{COL_MID}}{SPACER}"
                f"{ps:>{COL_POOR}}{SPACER}"
                f"{what:<{COL_WHAT}}")

    header = _row("Metric", "↑↓", "  Good", "    Mid", "  Poor", "What it measures")
    rows = [header]
    for name, direction, what, good, mid in METRIC_LEGEND:
        if direction == "↑":
            gs, ms, ps = f"  >{good:g}", f"  {mid:g} - {good:g}", f"  <{mid:g}"
        else:
            gs, ms, ps = f"  <{good:g}", f"  {good:g} - {mid:g}", f"  >{mid:g}"
        rows.append(_row(name, direction, gs, ms, ps, what))

    w = max(len(r) for r in rows)
    print(f"\n  ╔{'═' * (w + 2)}╗")
    print(f"  ║ {'HOW TO READ METRICS'.ljust(w)} ║")
    print(f"  ║ {'↑ = higher is better    ↓ = lower is better'.ljust(w)} ║")
    print(f"  ╟{'─' * (w + 2)}╢")
    print(f"  ║ {rows[0].ljust(w)} ║")
    print(f"  ╟{'─' * (w + 2)}╢")
    for row in rows[1:]:
        print(f"  ║ {row.ljust(w)} ║")
    print(f"  ╚{'═' * (w + 2)}╝")


# ═══════════════════════════════════════════════════════════════════════════
#  GPU detection + timing estimates
# ═══════════════════════════════════════════════════════════════════════════

# Baseline: V100 (125 FP16 TFLOPS) for Anima/Cosmos-Predict2, refitted from a
# 360-generation calibration run (2026-08-06, refiner recording ON):
#   512²:      15st 8.0s, 30st 17.4s, 45st 26.8s
#   1024²:     15st 25.4s, 30st 54.6s, 45st 84.8s
#   1024×512:  15st 12.2s, 30st 26.7s, 45st 40.5s
# A linear least-squares fit (fixed + per-pixel-step) over all nine buckets
# gives ~1.0 s fixed + ~0.46 s × pixel_ratio × steps; rounding the per-step
# term up to 0.50 keeps the schedule total within ~1% of the measured 2h 17m
# and errs slightly high at high resolution / low step counts.  Residual bias
# remains: per-step cost rises with step count (0.53 → 0.60 s/step at 512²)
# and superlinearly with pixels (attention), which two terms cannot capture.
_V100_SECONDS_PER_STEP_AT_512SQ = 0.53

# The refined timing model (ScheduleEstimator) splits that baseline into a
# per-generation fixed overhead (CLIP encode, VAE decode, patching, cache
# clears — mostly CPU/VRAM-bound) and a per-pixel-step compute cost, so
# resolution mixes and step mixes are accounted for exactly:
#   1.0 s fixed + 0.50 s × pixel_ratio × steps ≈ 16 s @ 512², 30 steps.
_V100_FIXED_OVERHEAD_SEC = 1.0
_V100_SECONDS_PER_PIXEL_STEP_AT_512SQ = 0.50

# Assumed NVMe bandwidth for refiner latent recording (lossless writes).
_REFINER_WRITE_BANDWIDTH_BYTES_PER_SEC = 1.0e9

# Sampler/scheduler/cfg factors (measured as dt vs the bucket mean) are only
# trusted once a dimension has this many samples AND deviates from 1.0 by at
# least this much (avoids noise).
_DIM_FACTOR_MIN_SAMPLES = 4
_DIM_FACTOR_MIN_DEVIATION = 0.05

# Bucket blend ramps from the hardware model to the measured mean over the
# first _BUCKET_BLEND_SAMPLES measurements of a (resolution, steps) bucket.
_BUCKET_BLEND_SAMPLES = 3

# Speed factors relative to V100 (1.0 = 12 s/30 steps at 512²).
#
# For GPUs ≤ ~350 TFLOPS: near-linear scaling (factor ≈ TFLOPS / 125).
# For GPUs > ~350 TFLOPS: diminishing returns — execution latency, memory
#   bandwidth, and CPU-GPU overhead dominate, so raw TFLOPS stop translating
#   1:1 into faster inference past a certain point.
#
# Factors are calibrated against real-world Anima/Cosmos DiT inference
# timings where available; otherwise estimated from TFLOPS data.
#
# Matched by case-insensitive substring search against torch.cuda.get_device_name().
# First match wins — order more-specific substrings before less-specific ones.
_GPU_SPEED_FACTORS: list[tuple[str, float]] = [
    # ── Datacenter / enterprise ──────────────────────────────────────
    ("b200",             5.5),    # 18000 TFLOPS — massive but latency-bound
    ("h200",             4.2),    # 1979 TFLOPS
    ("h100",             4.0),    # 1979 TFLOPS
    ("h800",             3.8),
    ("mi300x",           3.5),    # 1307 TFLOPS
    ("mi325x",           3.5),
    ("b100",             3.5),    # earlier Blackwell
    ("rtx pro 6000",     2.8),    # 1000 TFLOPS (Blackwell)
    ("rtx 6000 ada",     2.4),    # 364 TFLOPS
    ("a100",             2.3),    # 312 TFLOPS — well-characterised
    ("l40s",             2.4),    # 362 TFLOPS
    ("l40",              1.5),    # ~200 TFLOPS
    ("l4",               0.95),   # 121 TFLOPS
    ("a6000",            2.4),    # 364 TFLOPS (GA102)
    ("a5000",            1.4),
    ("a40",              1.8),    # lower-clocked A100 sibling
    ("a10",              1.0),
    ("a2",               0.7),
    ("t4",               0.4),    # 65 TFLOPS
    ("p100",             0.35),
    ("p40",              0.25),
    ("p4",               0.15),
    ("v100",             1.0),    # 125 TFLOPS — baseline

    # ── Consumer (RTX) — order by most-specific substrings first ─────
    ("rtx 5090",         2.3),    # 335 TFLOPS — plateaus despite high TFLOPS
    ("rtx 4090",         2.3),    # 330 TFLOPS (165 BF16, but DiT uses FP16)
    ("rtx 4080 super",   2.0),
    ("rtx 4080",         1.9),
    ("rtx 4070 ti super", 1.7),
    ("rtx 4070 ti",      1.5),
    ("rtx 4070 super",   1.4),
    ("rtx 4070",         1.3),
    ("rtx 4060 ti",      1.0),
    ("rtx 4060",         0.8),
    ("rtx 4050",         0.6),
    ("rtx 3090 ti",      1.15),
    ("rtx 3090",         1.1),    # 142 TFLOPS — near-linear with V100
    ("rtx 3080 ti",      0.95),
    ("rtx 3080",         0.80),
    ("rtx 3070 ti",      0.65),
    ("rtx 3070",         0.60),
    ("rtx 3060 ti",      0.50),
    ("rtx 3060",         0.45),
    ("rtx 3050",         0.30),
    ("rtx 2080 ti",      0.55),
    ("rtx 2080 super",   0.50),
    ("rtx 2080",         0.45),
    ("rtx 2070 super",   0.40),
    ("rtx 2070",         0.35),
    ("rtx 2060 super",   0.30),
    ("rtx 2060",         0.25),

    # ── AMD ──────────────────────────────────────────────────────────
    ("radeon ai pro r9700", 1.5),
    ("rx 9070 xt",       1.3),    # 194 TFLOPS
    ("rx 9070",          1.1),
    ("mi250x",           1.8),
    ("mi250",            1.5),
    ("mi210",            1.2),
    ("mi100",            1.0),
    ("mi50",             0.2),    # 26.5 TFLOPS
    ("radeon vii",       0.3),
    ("6900 xt",          0.25),
    ("6800 xt",          0.2),
]


def detect_gpu() -> tuple[str, float, bool]:
    """Detect the primary CUDA GPU and return (display_name, speed_factor, reliable).

    speed_factor is relative to a V100 (1.0).  Unknown GPUs are estimated
    from VRAM capacity as a rough heuristic (reliable=False — the factor is
    a guess, and pre-run estimates get a range).  Returns ("N/A", 1.0, False)
    when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return ("N/A", 1.0, False)

    name = torch.cuda.get_device_name(0) or "Unknown"
    name_lower = name.lower()

    for pattern, factor in sorted(_GPU_SPEED_FACTORS, key=lambda x: len(x[0]), reverse=True):
        if pattern in name_lower:
            return (name, factor, True)

    # Unknown GPU — guess from VRAM as a rough heuristic
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb >= 75:
            return (name, 3.0, False)
        if vram_gb >= 40:
            return (name, 2.0, False)
        if vram_gb >= 22:
            return (name, 1.5, False)
        if vram_gb >= 14:
            return (name, 1.0, False)
        return (name, 0.5, False)
    except Exception:
        return (name, 1.0, False)


class RunSpec(NamedTuple):
    """One scheduled generation: geometry + sampling dimensions.

    Only (width, height, steps) drive the hardware cost model; the
    sampler/scheduler/cfg fields feed the measured per-dimension factor
    analysis in ScheduleEstimator.
    """
    width: int
    height: int
    steps: int
    sampler: str = ""
    scheduler: str = ""
    cfg: float = 0.0


class ScheduleEstimator:
    """Combined hardware + measured ETA for a deterministic run schedule.

    The schedule (the exact list of RunSpecs, in execution order) is known
    before the first run, so "how many of each kind remain" is always an
    exact count.  Only the per-bucket wall times are unknown: they are
    estimated from the V100 hardware model until measured, then blended with
    the measured (resolution, steps) bucket means (weight n/3).  Sampler /
    scheduler / cfg factors correct the bucket mean once enough samples
    exist.
    """

    def __init__(self, schedule, gpu_name: str = "N/A", gpu_factor: float = 1.0,
                 fixed_overhead: float = _V100_FIXED_OVERHEAD_SEC,
                 per_pixel_step: float = _V100_SECONDS_PER_PIXEL_STEP_AT_512SQ):
        self.schedule = [s if isinstance(s, RunSpec) else RunSpec(*s) for s in schedule]
        self.gpu_name = gpu_name
        self.gpu_factor = gpu_factor if gpu_factor and gpu_factor > 0.1 else 1.0
        self.fixed_overhead = float(fixed_overhead)
        self.per_pixel_step = float(per_pixel_step)

        self._buckets: list[tuple] = []
        self._bucket_id: dict = {}
        for s in self.schedule:
            key = (s.width, s.height, s.steps)
            if key not in self._bucket_id:
                self._bucket_id[key] = len(self._buckets)
                self._buckets.append(key)
        self._run_bucket = [
            self._bucket_id[(s.width, s.height, s.steps)] for s in self.schedule
        ]

        self.total_runs = len(self.schedule)
        self.total_pixel_steps = sum((s.width * s.height) * s.steps for s in self.schedule)

        self._meas_sum = [0.0] * len(self._buckets)
        self._meas_n = [0] * len(self._buckets)
        self._step_sum: dict = {}
        self._step_n: dict = {}
        self._factor_sum: dict = {}   # (dim, value) -> [ratio_sum, n]
        self._recorded = 0

    # ── Hardware model ─────────────────────────────────────────────────

    def hardware_seconds(self, spec: RunSpec) -> float:
        """V100-baseline estimate for one run, scaled by the GPU factor."""
        px = (spec.width * spec.height) / (512.0 * 512.0)
        return (self.fixed_overhead + self.per_pixel_step * px * spec.steps) / self.gpu_factor

    # ── Measurement ────────────────────────────────────────────────────

    def bucket_blend(self, bid: int) -> float:
        """Blend hardware model → measured mean for a (res, steps) bucket."""
        n = self._meas_n[bid]
        if n == 0:
            return self.hardware_seconds(RunSpec(*self._buckets[bid]))
        mean = self._meas_sum[bid] / n
        w = min(1.0, n / float(_BUCKET_BLEND_SAMPLES))
        return w * mean + (1.0 - w) * self.hardware_seconds(RunSpec(*self._buckets[bid]))

    def record(self, spec: RunSpec, dt: float) -> None:
        """Record the measured wall time of one completed run."""
        spec = spec if isinstance(spec, RunSpec) else RunSpec(*spec)
        bid = self._bucket_id[(spec.width, spec.height, spec.steps)]
        n_before = self._meas_n[bid]
        mean_before = self._meas_sum[bid] / n_before if n_before else None

        self._meas_sum[bid] += dt
        self._meas_n[bid] += 1
        self._step_sum[spec.steps] = self._step_sum.get(spec.steps, 0.0) + dt
        self._step_n[spec.steps] = self._step_n.get(spec.steps, 0) + 1

        # Dimension factors are measured as dt vs the bucket's measured mean
        # (pure measurement, no hardware contamination).
        if mean_before is not None and mean_before > 0.01:
            for dim, val in (("sampler", spec.sampler),
                             ("scheduler", spec.scheduler),
                             ("cfg", spec.cfg)):
                if val is None or val == "" or val == 0.0:
                    continue
                st = self._factor_sum.setdefault((dim, val), [0.0, 0])
                st[0] += dt / mean_before
                st[1] += 1
        self._recorded += 1

    def _factor(self, dim: str, value) -> float:
        if value is None or value == "" or value == 0.0:
            return 1.0
        st = self._factor_sum.get((dim, value))
        if not st or st[1] < _DIM_FACTOR_MIN_SAMPLES:
            return 1.0
        f = st[0] / st[1]
        if abs(f - 1.0) < _DIM_FACTOR_MIN_DEVIATION:
            return 1.0
        return min(2.0, max(0.5, f))

    # ── Projection ─────────────────────────────────────────────────────

    def eta_seconds(self, completed: int) -> float:
        """Estimated wall time for runs [completed, total_runs)."""
        total = 0.0
        for i in range(completed, self.total_runs):
            s = self.schedule[i]
            eff = self.bucket_blend(self._run_bucket[i])
            eff *= self._factor("sampler", s.sampler)
            eff *= self._factor("scheduler", s.scheduler)
            eff *= self._factor("cfg", s.cfg)
            total += eff
        return total

    def remaining_bucket_counts(self, completed: int) -> dict:
        """Exact remaining-run counts per (w, h, steps) bucket."""
        counts: dict = {}
        for i in range(completed, self.total_runs):
            key = self._buckets[self._run_bucket[i]]
            counts[key] = counts.get(key, 0) + 1
        return counts

    def resolution_counts(self) -> dict:
        """Total schedule counts per 'WxH' resolution."""
        counts: dict = {}
        for s in self.schedule:
            key = f"{s.width}x{s.height}"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def step_counts(self) -> dict:
        """Total schedule counts per step count."""
        counts: dict = {}
        for s in self.schedule:
            counts[s.steps] = counts.get(s.steps, 0) + 1
        return counts

    def avg_pixel_ratio(self) -> float:
        """Pixel ratio vs 512² averaged over the exact schedule."""
        if not self.total_runs:
            return 1.0
        total_px = sum(s.width * s.height for s in self.schedule)
        return total_px / (self.total_runs * 512.0 * 512.0)

    # ── Reporting ──────────────────────────────────────────────────────

    def summary_lines(self, completed: int, elapsed: float) -> list[str]:
        """Lines for the periodic timing report (used by calibrate.py)."""
        W = 58
        pct = completed / self.total_runs * 100.0 if self.total_runs else 100.0
        eta = self.eta_seconds(completed)
        rem = self.remaining_bucket_counts(completed)
        fmt = format_duration

        head = f"── timing report ({completed} runs, {pct:.1f}%)"
        lines = [f"  {head} " + "─" * max(2, W - len(head) - 1)]
        lines.append(f"  {'elapsed':<12}{fmt(elapsed)}")
        lines.append(f"  {'remaining':<12}~{fmt(eta)}  ({self.total_runs - completed} gens)")

        parts = []
        for bid, key in enumerate(self._buckets):
            label = f"{key[0]}×{key[1]}/{key[2]}st"
            n = self._meas_n[bid]
            r = rem.get(key, 0)
            if not n and not r:
                continue
            if n:
                parts.append(f"{label}: {self._meas_sum[bid] / n:.1f}s (n={n}, rem {r})")
            else:
                parts.append(f"{label}: – (rem {r})")
        if parts:
            lines.append(f"  {'measured':<12}" + " | ".join(parts))

        parts = []
        for steps in sorted(self._step_n):
            rem_s = sum(1 for i in range(completed, self.total_runs)
                        if self.schedule[i].steps == steps)
            parts.append(f"{steps}st {self._step_sum[steps] / self._step_n[steps]:.1f}s "
                         f"(rem {rem_s})")
        if parts:
            lines.append(f"  {'steps':<12}" + " | ".join(parts))

        for dim, label in (("sampler", "sampler"), ("scheduler", "scheduler"),
                           ("cfg", "cfg")):
            parts = []
            for (d, val), st in sorted(self._factor_sum.items(), key=lambda kv: str(kv[0])):
                if d != dim:
                    continue
                f = st[0] / st[1]
                v = f"{val:g}" if isinstance(val, float) else str(val)
                applied = (st[1] >= _DIM_FACTOR_MIN_SAMPLES
                           and abs(f - 1.0) >= _DIM_FACTOR_MIN_DEVIATION)
                parts.append(f"{v} ×{f:.2f} (n={st[1]}){'*' if applied else ''}")
            if parts:
                lines.append(f"  {label:<12}" + " | ".join(parts))

        if rem:
            most = max(rem, key=rem.get)
            bid = self._bucket_id[most]
            hw = self.hardware_seconds(RunSpec(*most))
            w = min(1.0, self._meas_n[bid] / float(_BUCKET_BLEND_SAMPLES))
            lines.append(f"  {'hw baseline':<12}{hw:.1f}s @{most[0]}×{most[1]}/{most[2]}st "
                         f"(blend weight {w:.2f})")
        lines.append(f"  {'─' * W}")
        return lines


class MetricsLog:
    """JSONL sink for run scalars (step rows, eval rows, phase rows).

    One JSON object per line, flushed per write — ``tail -f`` friendly.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self._f = open(self.path, "a")

    def write(self, row: Dict) -> None:
        self._f.write(json.dumps(row, default=str) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def format_duration(seconds: float) -> str:
    """Format seconds as '45s', '4m 12s', or '1h 03m'."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m {int(seconds % 60):02d}s"
    return f"{minutes // 60}h {minutes % 60:02d}m"


# ── Learnability stats (probe Task 3b; reused by train_corrector) ─────


def svd_rank_95(delta_maps):
    """SVD spectrum of Δ_MA (H·W, 16); rank for 95% variance (plan 3b)."""
    m = torch.stack([d.float() for d in delta_maps])          # (N, C, H, W)
    m = m.permute(0, 2, 3, 1).reshape(-1, m.shape[1])          # (N·H·W, 16)
    m = m - m.mean(dim=0, keepdim=True)
    try:
        u, s, v = torch.svd(m)
    except Exception:
        return {"rank_95": None, "singular_values": None}
    var = (s * s).cumsum(0)
    var = var / var[-1].clamp_min(1e-12)
    rank95 = int((var <= 0.95).sum().item()) + 1
    return {"rank_95": rank95,
            "singular_values": [round(float(x), 4) for x in s[:16].tolist()]}


def per_channel_affine_ceiling(v_ma_maps, v_true_maps) -> float:
    """Best per-channel affine v̂ = a_c·v_MA_c + b_c; 1 − R² (plan 3b)."""
    x = torch.stack([v.float().flatten(1) for v in v_ma_maps]).reshape(-1, 16)
    y = torch.stack([v.float().flatten(1) for v in v_true_maps]).reshape(-1, 16)
    x_aug = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)   # (N, 17)
    coefs, _, _, _ = torch.linalg.lstsq(x_aug, y)              # (17, 16)
    pred = x_aug @ coefs
    ss_res = ((y - pred) ** 2).sum(dim=0)
    ss_tot = ((y - y.mean(dim=0)) ** 2).sum(dim=0).clamp_min(1e-12)
    r2 = (1 - ss_res / ss_tot).mean().item()
    return max(0.0, 1.0 - r2)


def pooled_feature_ceiling(x_t_maps, v_ma_maps, v_true_maps) -> float:
    """Linear predictor from pooled features (per-channel mean/std + t) → 1−R²."""
    feats, y = [], []
    for xt, vm, vt in zip(x_t_maps, v_ma_maps, v_true_maps):
        f = torch.cat([xt.float().mean(dim=(1, 2)), xt.float().std(dim=(1, 2)),
                       vm.float().mean(dim=(1, 2)), vm.float().std(dim=(1, 2))])
        feats.append(f)
        y.append(vt.float().mean(dim=(1, 2)))
    X = torch.stack(feats)                                      # (N, 64)
    Y = torch.stack(y)                                          # (N, 16)
    X = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)
    coefs, _, _, _ = torch.linalg.lstsq(X, Y)
    pred = X @ coefs
    ss_res = ((Y - pred) ** 2).sum()
    ss_tot = ((Y - Y.mean(dim=0)) ** 2).sum().clamp_min(1e-12)
    return float(max(0.0, 1.0 - ss_res / ss_tot).item())


def step_correlations(v_true_by_slot: dict) -> dict:
    """Step-to-step correlation of v_true per region (early/mid/late)."""
    out = {}
    all_corrs = {"early": [], "mid": [], "late": []}
    for slot, vts in v_true_by_slot.items():
        n = len(vts)
        if n < 3:
            continue
        thirds = [("early", slice(0, n // 3)), ("mid", slice(n // 3, 2 * n // 3)),
                  ("late", slice(2 * n // 3, n))]
        for name, sl in thirds:
            for t in range(sl.start + 1, sl.stop):
                a = vts[t].float().flatten()
                b = vts[t - 1].float().flatten()
                all_corrs[name].append(F.cosine_similarity(a[None], b[None]).item())
    for name, corrs in all_corrs.items():
        out[name] = round(float(np.mean(corrs)), 4) if corrs else None
    return out


def staleness_curve(delta_by_lag: dict, num_regions: int = 3) -> dict:
    """mean|Δ_MA| vs lag per step region (early/mid/late); monotonicity per region."""
    curve = {r: {} for r in range(num_regions)}
    for lag, items in delta_by_lag.items():
        for region, d in items:
            curve[region].setdefault(lag, []).append(d.abs().mean().item())
    out = {}
    monotone = True
    for r in range(num_regions):
        lags = sorted(curve[r])
        means = {lag: float(np.mean(curve[r][lag])) for lag in lags}
        out[f"region_{r}"] = means
        vals = [means[lag] for lag in lags]
        if len(vals) > 1:
            diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
            if any(d < -1e-9 for d in diffs):
                monotone = False
    out["per_region_monotone"] = monotone
    return out


def delta_distribution(delta_maps: list) -> dict:
    d = torch.stack([x.float() for x in delta_maps])
    flat = d.flatten()
    return {
        "mean": round(float(flat.mean()), 6),
        "std": round(float(flat.std()), 6),
        "sparsity": round(float((flat.abs() < 1e-4).float().mean()), 4),
        "per_channel_std": [round(float(s), 5) for s in d.std(dim=(0, 2, 3)).tolist()],
    }


# ── Recovery-table affine (OOD: fit on train pairs, score on eval pairs) ──


def fit_affine_coefs(vm_maps, vt_maps) -> torch.Tensor:
    """Best per-channel affine v̂ = a_c·v_MA_c + b_c from pooled pixels.

    ``vm_maps``/``vt_maps`` are lists of (B, C, H, W) (or (C, H, W)) tensors;
    the channel dim is read from ``shape[1]``. Returns a (C+1, C) coefficient
    matrix (last row = intercepts). Identity is in the class, so the fit can
    never be worse than "×1 + 0" *on this objective* (pooled per-pixel MSE).
    """
    c = vm_maps[0].shape[1]
    x = torch.cat([v.float() for v in vm_maps], 0).reshape(-1, c)
    y = torch.cat([v.float() for v in vt_maps], 0).reshape(-1, c)
    xa = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)   # (N, C+1)
    coefs, _, _, _ = torch.linalg.lstsq(xa, y)              # (C+1, C)
    return coefs


def affine_coef_diagnostics(coefs) -> dict:
    """How far a (C+1, C) affine map is from identity: |a_c − 1| and |b_c|.

    Near-zero values mean the fit learned nothing beyond identity (no affine
    structure to exploit); large values mean the fit slice is unrepresentative
    of the scored slice — a red flag for the recovery-table row.
    """
    c = coefs.shape[1]
    dev = coefs[:c, :].diag() - 1.0
    b = coefs[c, :]
    return {
        "max_abs_a_minus_1": round(float(dev.abs().max()), 6),
        "mean_abs_a_minus_1": round(float(dev.abs().mean()), 6),
        "max_abs_b": round(float(b.abs().max()), 6),
        "mean_abs_b": round(float(b.abs().mean()), 6),
    }


def affine_rel_mse_per_pair(coefs, vm, vt) -> List[float]:
    """Per-pair relative L2 ‖v̂−v_true‖₂/‖v_true‖₂ for an affine map.

    ``vm``/``vt`` are batched (B, C, H, W) tensors; returns one error per pair.
    """
    c = vm.shape[1]
    x = vm.reshape(-1, c).float()
    y = vt.reshape(-1, c).float()
    pred = torch.cat([x, torch.ones_like(x[:, :1])], dim=1) @ coefs
    pred = pred.reshape(vm.shape)
    err = (pred - vt.float()).flatten(1).norm(dim=1)
    den = vt.float().flatten(1).norm(dim=1) + 1e-8
    return (err / den).tolist()


def collect_affine_maps(loader, cap_batches_per_shape: int = 128,
                        cap_batches_total: int = 512,
                        known_shapes=None) -> dict:
    """Collect (v_ma, v_true) batch tensors per latent shape from a loader.

    Returns ``{shape: {"vm": [tensor...], "vt": [tensor...]}}`` — CPU float32,
    batched (B, C, H, W). Time and memory are bounded: at most
    ``cap_batches_per_shape`` batch tensors per shape (first-come) and at most
    ``cap_batches_total`` batches *visited* overall — the iteration always
    terminates, even when a single dominant shape fills the stream (which
    previously caused a silent full-corpus pass).

    ``known_shapes`` (iterable of (h, w) tuples) enables an early exit as soon
    as every known shape has reached its per-shape cap — pass the dataset's
    shapes when the loader is already restricted to those buckets
    (``CorrectorBatchSampler(include_areas=...)``).
    """
    by_shape = {}
    total = 0
    for b in loader:
        total += 1
        if total > cap_batches_total:
            break
        vm = b["v_ma"].float().cpu()
        vt = b["v_true"].float().cpu()
        shape = (int(vm.shape[-2]), int(vm.shape[-1]))
        entry = by_shape.setdefault(shape, {"vm": [], "vt": []})
        if len(entry["vm"]) >= cap_batches_per_shape:
            continue
        entry["vm"].append(vm)
        entry["vt"].append(vt)
        if known_shapes and all(
                len(by_shape.get(s, {"vm": []})["vm"]) >= cap_batches_per_shape
                for s in known_shapes):
            break
    return by_shape


def affine_oob_eval(fit_by_shape, eval_by_shape, cap_batches: int = 128) -> dict:
    """OOD per-channel affine: fit per stratum on train maps, score per
    stratum on eval maps with per-pair relative L2 (the recovery-table row).

    ``fit_by_shape``/``eval_by_shape``: {shape: {"vm": [...], "vt": [...]}} as
    produced by :func:`collect_affine_maps` (eval maps may carry an extra
    "anchor" key, ignored here). The fit uses the first ``cap_batches`` batch
    tensors per stratum; scoring covers every eval pair.

    Returns {"by_shape": {shape: {"per_pair": [float], "rel_mse": float,
             "n_pairs": int, "fit_n_batches": int, **affine_coef_diagnostics}},
             "overall": pooled mean over all eval pairs}.
    """
    out = {"by_shape": {}}
    pooled_sum, pooled_n = 0.0, 0
    for shape in sorted(eval_by_shape, key=lambda s: s[0] * s[1]):
        ef = eval_by_shape[shape]
        ff = fit_by_shape.get(shape, {"vm": [], "vt": []})
        if not ef["vm"]:
            continue
        if not ff["vm"]:
            out["by_shape"][shape] = {"per_pair": [], "rel_mse": None,
                                      "n_pairs": 0, "fit_n_batches": 0}
            continue
        coefs = fit_affine_coefs(ff["vm"][:cap_batches], ff["vt"][:cap_batches])
        per_pair = []
        for vm, vt in zip(ef["vm"], ef["vt"]):
            per_pair.extend(affine_rel_mse_per_pair(coefs, vm, vt))
        d = {"per_pair": per_pair,
             "rel_mse": round(float(np.mean(per_pair)), 5) if per_pair else None,
             "n_pairs": len(per_pair),
             "fit_n_batches": min(len(ff["vm"]), cap_batches),
             **affine_coef_diagnostics(coefs)}
        out["by_shape"][shape] = d
        if d["rel_mse"] is not None:
            pooled_sum += d["rel_mse"] * len(per_pair)
            pooled_n += len(per_pair)
    out["overall"] = round(pooled_sum / max(pooled_n, 1), 5) if pooled_n else None
    return out


def split_affine_per_pair(aff, eval_by_shape) -> Tuple[List[float], List[float]]:
    """Split an :func:`affine_oob_eval` result into (ladder, d=0 anchor) errors.

    The anchor masks must be stored per pair, in append order, under
    ``eval_by_shape[shape]["anchor"]`` (aligned with the "per_pair" lists).
    """
    ladder, anchor = [], []
    for shape, d in aff.get("by_shape", {}).items():
        masks = (eval_by_shape.get(shape) or {}).get("anchor", [])
        for err, is_a in zip(d.get("per_pair", []), masks):
            (anchor if is_a else ladder).append(err)
    return ladder, anchor


def affine_oob_json(aff) -> dict:
    """JSON-safe copy of an :func:`affine_oob_eval` result (drops per-pair lists)."""
    out = {"overall": aff.get("overall"), "by_shape": {}}
    for shape, d in (aff.get("by_shape") or {}).items():
        out["by_shape"][f"{shape[0]}x{shape[1]}"] = {
            k: v for k, v in d.items() if k != "per_pair"}
    return out


def affine_shape_area(shape) -> int:
    """Latent-area sort key for an affine by_shape key (tuple or 'WxH' str).

    ``affine_oob_json`` stringifies keys, so consumers must accept both forms.
    """
    if isinstance(shape, str):
        h, w = shape.split("x")
        return int(h) * int(w)
    return int(shape[0]) * int(shape[1])


def affine_shape_label(shape) -> str:
    """'WxH' label for an affine by_shape key (tuple or already-string)."""
    if isinstance(shape, str):
        return shape
    return f"{shape[0]}x{shape[1]}"


class TrainTimer:
    """Per-phase wall-time tracking + robust per-step rate for training loops.

    Mirrors the ScheduleEstimator pattern used by calibrate.py: train / eval /
    hessian / checkpoint phases are accumulated separately, and the projected
    remaining time includes the eval + checkpoint overhead at every
    ``eval_every`` boundary. The per-step rate is a median over a sliding
    window — robust to Sophia hessian steps and lazy-decompression cache
    misses (train_corrector.py, plan 6f/6h).
    """

    def __init__(self, window: int = 100):
        self._window = max(window, 1)
        self._step_times = deque(maxlen=self._window)
        self._phases: dict = {}
        self._n_steps = 0

    def record_step(self, dt: float) -> None:
        """One completed training step (data fetch → backward → EMA)."""
        self._n_steps += 1
        self._step_times.append(dt)
        self._phases["train"] = self._phases.get("train", 0.0) + dt

    def add(self, phase: str, dt: float) -> None:
        """Accumulate a non-train phase (hessian, eval, checkpoint, ...)."""
        self._phases[phase] = self._phases.get(phase, 0.0) + dt

    def step_time(self) -> float:
        """Median seconds/step over the window (0.0 before any steps)."""
        if not self._step_times:
            return 0.0
        ts = sorted(self._step_times)
        return ts[len(ts) // 2]

    def steps_per_sec(self) -> float:
        st = self.step_time()
        return 1.0 / st if st > 0 else 0.0

    def phase_seconds(self, phase: str) -> float:
        return self._phases.get(phase, 0.0)

    def remaining_seconds(self, steps_left: int, eval_every: int) -> float:
        """Projected wall time for ``steps_left`` steps, incl. eval phases."""
        st = self.step_time()
        if st <= 0 or steps_left <= 0:
            return 0.0
        total = st * steps_left
        n_evals = steps_left // max(eval_every, 1)
        per_eval = self._phases.get("eval", 0.0) / max(
            self._n_steps // max(eval_every, 1), 1)
        return total + n_evals * per_eval

    def summary_lines(self, elapsed: float, done: int, total: int,
                      eval_every: int) -> list[str]:
        """Periodic report block (mirrors ScheduleEstimator.summary_lines)."""
        W = 58
        pct = done / total * 100.0 if total else 100.0
        fmt = format_duration
        eta = self.remaining_seconds(total - done, eval_every)
        head = f"── timing report ({done} steps, {pct:.1f}%)"
        lines = [f"  {head} " + "─" * max(2, W - len(head) - 1)]
        lines.append(f"  {'elapsed':<12}{fmt(elapsed)}")
        lines.append(f"  {'remaining':<12}~{fmt(eta)}  ({total - done} steps)")
        st = self.step_time()
        if st > 0:
            lines.append(f"  {'it/s':<12}{self.steps_per_sec():.2f}  "
                         f"(median step {st * 1000:.0f} ms)")
        parts = []
        for phase, label in (("train", "train"), ("eval", "eval"),
                             ("hessian", "hessian"), ("checkpoint", "checkpoint")):
            s = self._phases.get(phase, 0.0)
            if s > 0:
                parts.append(f"{label} {fmt(s)}")
        if parts:
            lines.append(f"  {'phases':<12}" + "  |  ".join(parts))
        if eval_every > 0 and st > 0:
            to_next = (eval_every - done % eval_every) * st
            lines.append(f"  {'next eval':<12}~{fmt(to_next)}")
        return lines


def estimate_calibration_time(
    total_runs: int,
    step_variants: list,
    step_weights: list | None = None,
    width: int = 512,
    height: int = 512,
) -> tuple[float, str, float]:
    """Return (total_seconds, gpu_name, gpu_factor) for calibration planning.

    Accounts for the actual weighted mix of step counts, the image resolution,
    and the detected GPU's speed relative to V100.
    """
    gpu_name, gpu_factor, _ = detect_gpu()

    if step_weights and len(step_weights) == len(step_variants):
        ws = step_weights
    else:
        ws = [1.0 / len(step_variants)] * len(step_variants)

    avg_steps = sum(s * w for s, w in zip(step_variants, ws)) / sum(ws)
    pixel_ratio = (width * height) / (512.0 * 512.0)

    seconds = (total_runs * avg_steps * pixel_ratio *
               _V100_SECONDS_PER_STEP_AT_512SQ / max(gpu_factor, 0.1))

    return seconds, gpu_name, gpu_factor


def estimate_generation_time(
    total_generations: int,
    avg_steps: float,
    width: int = 512,
    height: int = 512,
) -> tuple[float, str, float]:
    """Return (total_seconds, gpu_name, gpu_factor) for a batch of generations.

    Generalised form of estimate_calibration_time — takes a simple avg_steps
    instead of step_variant/step_weight lists so it works for both calibration
    and validation phases.
    """
    gpu_name, gpu_factor, _ = detect_gpu()
    pixel_ratio = (width * height) / (512.0 * 512.0)
    seconds = (total_generations * avg_steps * pixel_ratio *
               _V100_SECONDS_PER_STEP_AT_512SQ / max(gpu_factor, 0.1))
    return seconds, gpu_name, gpu_factor


def compute_total_iterations(step_counts: dict[int, int]) -> int:
    """Sum steps × count for all planned generations.

    Args:
        step_counts: {num_steps: num_generations_at_that_step_count, ...}
    """
    return sum(steps * count for steps, count in step_counts.items())


def derive_step_anchors(variants: list, base: int) -> list:
    """Derive the step-count anchor list for step-multiplier measurement.

    Starts from the user-provided *variants*, sorts/dedupes, force-includes
    *base* (the calibrated default step count), and inserts the midpoint
    ``(a + b) // 2`` for each adjacent pair of user variants (skipped when
    no integer lies between them).

    Example: ``derive_step_anchors([3, 15, 45, 60], 30)`` →
    ``[3, 9, 15, 30, 45, 52, 60]``.
    """
    vs = sorted(set(variants))
    midpoints = []
    for a, b in zip(vs, vs[1:]):
        m = (a + b) // 2
        if a < m < b:
            midpoints.append(m)
    return sorted(set(vs) | set(midpoints) | {base})


def print_schedule_estimate(
    label: str,
    total_generations: int = 0,
    avg_steps: float = 0.0,
    width: int = 512,
    height: int = 512,
    extra_lines: list[str] | None = None,
    schedule: list[RunSpec] | None = None,
    refiner_disk_bytes: float = 0.0,
) -> float:
    """Print a pre-run estimate block and return estimated seconds.

    With *schedule* (exact list of RunSpecs), the estimate accounts for the
    per-run resolution and step mix, the per-generation fixed overhead, the
    refiner disk-write time (from *refiner_disk_bytes*), and shows a range
    when the GPU factor is a heuristic guess.  Without a schedule it falls
    back to the legacy avg-steps / base-resolution formula.

    Example output (schedule path):
      ──────────────────────────────────────────────────────────
      Calibration run schedule
      ──────────────────────────────────────────────────────────
      Generations:  360
      GPU:          NVIDIA A100  (×2.3 vs V100)
      Resolution:   512x512 75.0%, 1024x1024 15.0%, 1024x512 10.0%
                    (avg pixel ratio ×1.55, eff. ~637px)
      Steps:        15 steps ×60, 30 steps ×240, 45 steps ×60
      Est. compute: ~1h 01m (denoising)  + ~3m (fixed/gen)  + ~0.5m (refiner write)
      Est. time:    ~1h 04m
      Excludes:     model load/startup
      ──────────────────────────────────────────────────────────
    """
    gpu_name, gpu_factor, gpu_reliable = detect_gpu()
    w = 56
    print(f"\n  {'─' * w}")
    print(f"  {label}")
    print(f"  {'─' * w}")

    if schedule is not None:
        est = ScheduleEstimator(schedule, gpu_name=gpu_name, gpu_factor=gpu_factor)
        n = est.total_runs
        if n == 0:
            print(f"  Schedule is empty.")
            print(f"  {'─' * w}\n")
            return 0.0
        total = est.eta_seconds(0)
        compute = (est.total_pixel_steps * est.per_pixel_step
                   / (512.0 * 512.0) / est.gpu_factor)
        fixed = n * est.fixed_overhead / est.gpu_factor
        refiner = refiner_disk_bytes / _REFINER_WRITE_BANDWIDTH_BYTES_PER_SEC

        print(f"  Generations:  {n}")
        print(f"  GPU:          {gpu_name}  (×{gpu_factor:.1f} vs V100"
              f"{'' if gpu_reliable else ' — heuristic, uncertain'})")
        res_counts = est.resolution_counts()
        res_str = ", ".join(f"{k} {v / n * 100:.1f}%" for k, v in res_counts.items())
        print(f"  Resolution:   {res_str}")
        apr = est.avg_pixel_ratio()
        print(f"                (avg pixel ratio ×{apr:.2f}, "
              f"eff. ~{int(apr ** 0.5 * 512)}px)")
        st_str = ", ".join(f"{s} steps ×{c}" for s, c in sorted(est.step_counts().items()))
        print(f"  Steps:        {st_str}")
        print(f"  Est. compute: ~{format_duration(compute)} (denoising)  "
              f"+ ~{format_duration(fixed)} (fixed/gen)  "
              f"+ ~{format_duration(refiner)} (refiner write)")
        if gpu_reliable:
            print(f"  Est. time:    ~{format_duration(total)}")
        else:
            print(f"  Est. time:    ~{format_duration(total)}  "
                  f"(range {format_duration(total / 1.4)} – "
                  f"{format_duration(total / 0.7)})")
        print(f"  Excludes:     model load/startup")
    else:
        secs, gpu, factor = estimate_generation_time(
            total_generations, avg_steps, width, height,
        )
        print(f"  Generations:  {total_generations}")
        print(f"  Avg. steps:   {avg_steps:.1f}")
        print(f"  Resolution:   {width}×{height}")
        print(f"  GPU:          {gpu}  (×{factor:.1f} vs V100)")
        print(f"  Est. time:    ~{format_duration(secs)}")

    if extra_lines:
        for line in extra_lines:
            print(f"  {line}")
    print(f"  {'─' * w}\n")
    return total if schedule else secs


def print_speed_summary(
    label: str,
    total_generations: int,
    total_iterations: int,
    wall_seconds: float,
) -> None:
    """Print the final throughput report.

    Distinguishes generations/second (images) from actual denoising
    iterations/second (UNet forward calls).

    Example output:
      ============================================================
      Validation complete
      Total time:        14m 32s
      Images:            0.08 img/s  (24 images)
      Denoising steps:   2.5 it/s    (720 iterations)
      ============================================================
    """
    gen_per_sec = total_generations / wall_seconds if wall_seconds > 0 else 0.0
    it_per_sec = total_iterations / wall_seconds if wall_seconds > 0 else 0.0

    if gen_per_sec >= 1.0:
        gen_str = f"{gen_per_sec:.1f} img/s  ({total_generations} images)"
    else:
        sec_per_gen = wall_seconds / total_generations if total_generations > 0 else 0.0
        gen_str = f"{sec_per_gen:.1f} s/img  ({total_generations} images)"

    it_str = f"{it_per_sec:.1f} it/s  ({total_iterations} denoising steps)"

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Total time:        {int(wall_seconds // 60)}m {int(wall_seconds % 60)}s")
    print(f"  Throughput:        {gen_str}")
    print(f"  Throughput:        {it_str}")
    print(f"{'=' * 60}")


def score_from_legend(name: str, val: float) -> str:
    """Rate a metric value using the shared legend thresholds."""
    for n, direction, _, good, mid in METRIC_LEGEND:
        if n == name and val == val:  # val == val checks not NaN
            if direction == "↑":
                return "✅ EXCELLENT" if val >= good else "✓ acceptable" if val >= mid else "⚠ POOR"
            else:
                return "✅ EXCELLENT" if val <= good else "✓ acceptable" if val <= mid else "⚠ POOR"
    return "N/A"
