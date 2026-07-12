"""ComfyUI integration: model loading, sampling, metric computation."""

import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def setup_comfy_path(comfy_dir: str) -> None:
    if comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)


def load_models(comfy_dir: str,     model_name: str,
                clip_name: str,     clip_type: str,
                vae_name: str):
    """Load Anima UNet + CLIP + VAE via ComfyUI node classes.

    Loads node classes by file path (same as sample()) to avoid
    this addon's nodes.py shadowing ComfyUI's.
    """
    setup_comfy_path(comfy_dir)

    import folder_paths
    _nodes = _load_comfy_nodes(comfy_dir)

    mdir = str(Path(comfy_dir) / "models")
    folder_paths.add_model_folder_path("diffusion_models", mdir + "/diffusion_models")
    folder_paths.add_model_folder_path("text_encoders",    mdir + "/text_encoders")
    folder_paths.add_model_folder_path("vae",              mdir + "/vae")

    print(f"[load] UNet: {model_name}")
    unet = _nodes.UNETLoader().load_unet(model_name, "default")[0]

    print(f"[load] CLIP: {clip_name} ({clip_type})")
    clip = _nodes.CLIPLoader().load_clip(clip_name, clip_type, "default")[0]

    print(f"[load] VAE: {vae_name}")
    vae = _nodes.VAELoader().load_vae(vae_name)[0]

    print("[load] All models ready")
    return unet, clip, vae


_comfy_nodes_cache = {}

def _load_comfy_nodes(comfy_dir_arg=None):
    """Load ComfyUI's nodes.py by file path, cached.

    Using 'import nodes' would find this addon's nodes.py instead of
    ComfyUI's. Loading by file path bypasses the module name conflict.
    """
    import importlib.util
    if comfy_dir_arg is None:
        dirs = [p for p in sys.path if p and p != "." and "comfy" in p.lower()]
        if not dirs:
            raise RuntimeError(
                "ComfyUI directory not found in sys.path. "
                "Run load_models() first or set PYTHONPATH."
            )
        comfy_dir_arg = dirs[0]
    if comfy_dir_arg not in _comfy_nodes_cache:
        spec = importlib.util.spec_from_file_location(
            "comfyui_nodes", str(Path(comfy_dir_arg) / "nodes.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _comfy_nodes_cache[comfy_dir_arg] = mod
    return _comfy_nodes_cache[comfy_dir_arg]


def sample(unet, clip, vae, prompt: str, *,
           seed: int = 42, steps: int = 30,
           cfg: float = 5.0,
           sampler_name: str = "er_sde",
           scheduler: str = "normal",
           width: int = 1024, height: int = 1024,
           negative: str = "",
           return_latent: bool = False):
    """Run a full sampling pass. Returns PIL.Image (or (latent, image) tuple).

    Loads ComfyUI node classes by file path to avoid the local nodes.py
    shadowing issue.
    """
    from PIL import Image

    _nodes = _load_comfy_nodes()
    pos = _nodes.CLIPTextEncode().encode(clip, prompt)[0]
    neg = _nodes.CLIPTextEncode().encode(clip, negative)[0]
    latent = _nodes.EmptyLatentImage().generate(width, height, 1)[0]
    samples = _nodes.KSampler().sample(
        unet, seed, steps, cfg, sampler_name, scheduler, pos, neg, latent, 1.0
    )[0]
    decoded = _nodes.VAEDecode().decode(vae, samples)[0]
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


def compute_quality_metrics(
    img_pred, img_ref
) -> Tuple[float, float, float]:
    """Return (PSNR, SSIM, LPIPS) for pred vs ref images.

    Requires: pip install pyiqa
    Returns (inf, 1.0, 0.0) if pyiqa not available.
    """
    try:
        import pyiqa
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
        return float("inf"), 1.0, 0.0


def get_diffusion_model(unet):
    """Get the underlying MiniTrainDIT from a ComfyUI ModelPatcher."""
    return unet.get_model_object("diffusion_model")


def measure_vram() -> float:
    """Return peak VRAM usage in GB."""
    return torch.cuda.max_memory_allocated() / (1024 ** 3)
