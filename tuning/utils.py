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

def compute_quality_metrics(
    img_pred, img_ref
) -> Tuple[float, float, float]:
    """Return (PSNR, SSIM, LPIPS) for pred vs ref images.

    Requires: pip install -r tuning/requirements.txt
    If pyiqa is not installed, warns once and returns sentinel values.
    """
    global _PYIQA_AVAILABLE, _PYIQA_WARNED

    try:
        if not _PYIQA_AVAILABLE:
            import pyiqa
            _PYIQA_AVAILABLE = True
        device = torch.device("cuda")
        psnr_m  = pyiqa.create_metric("psnr",  device=device)
        ssim_m  = pyiqa.create_metric("ssim",  device=device)
        lpips_m = pyiqa.create_metric("lpips", device=device)

        t_pred = img_to_tensor(img_pred)
        t_ref  = img_to_tensor(img_ref)

        psnr  = float(psnr_m(t_pred, t_ref).item())
        ssim  = float(ssim_m(t_pred, t_ref).item())
        lpips = float(lpips_m(t_pred, t_ref).item())
        return psnr, ssim, lpips
    except ImportError:
        if not _PYIQA_WARNED:
            _PYIQA_WARNED = True
            print(
                "\n  ⚠ WARNING: pyiqa is not installed. "
                "Quality metrics (PSNR/SSIM/LPIPS) are UNAVAILABLE.\n"
                "  Install with: pip install -r tuning/requirements.txt\n"
                "  The values below are dummy placeholders — NOT real measurements.\n"
            )
        return float("inf"), 1.0, 0.0


def get_diffusion_model(unet):
    """Get the underlying MiniTrainDIT from a ComfyUI ModelPatcher."""
    return unet.get_model_object("diffusion_model")


def measure_vram() -> float:
    """Return peak VRAM usage in GB."""
    return torch.cuda.max_memory_allocated() / (1024 ** 3)
