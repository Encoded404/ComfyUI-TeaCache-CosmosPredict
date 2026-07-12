"""ComfyUI integration: model loading, sampling, metric computation.

IMPORTANT: These scripts must be run with ComfyUI's root directory as the
first entry in sys.path. Use the wrapper script or set PYTHONPATH.
See: run from ComfyUI root with python -m
    PYTHONPATH=".:custom_nodes/ComfyUI-TeaCache-CosmosPredict"
"""

import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


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


def score_from_legend(name: str, val: float) -> str:
    """Rate a metric value using the shared legend thresholds."""
    for n, direction, _, good, mid in METRIC_LEGEND:
        if n == name and val == val:  # val == val checks not NaN
            if direction == "↑":
                return "✅ EXCELLENT" if val >= good else "✓ acceptable" if val >= mid else "⚠ POOR"
            else:
                return "✅ EXCELLENT" if val <= good else "✓ acceptable" if val <= mid else "⚠ POOR"
    return "N/A"
