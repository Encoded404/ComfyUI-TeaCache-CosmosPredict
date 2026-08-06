"""Latent corrector model — ``CorrectorUNet2D`` (plan Task 4, deep-dive §3).

Small 2D UNet mapping ``(x_t ⊕ v_prev) → Δv̂`` on the 16-channel image latent
(T=1, 512²: 64×64 grid). Architecture (plan Task 4a/6c):

    Encoder: RepBlock(32→64) + stride-2 RepBlock(64→128) + stride-2 RepBlock(128→256)
    Bottleneck (H/4 × W/4 grid, dim 256, 8 heads):
      DiT 2D sincos pos-embed + 2× DiTBlock(256, 8, mlp_ratio=4)
      — 9-way adaLN (shift/scale/gate × self-attn, cross-attn(prompt), MLP);
        gates zero-init → blocks are identity at init
    Decoder: pixel-shuffle upsample + skip-concat + RepBlock (×2)
    Output: conv 3×3 64→16, weights AND bias zero-initialized
      ⇒ Δv̂ = 0 at init ⇒ Mode B′ ≡ Mode A exactly (Task 7 sanity gate)

RepBlocks are GroupNorm branch-sum blocks (deep-dive §3.1 variant (b)): the
fold at export is a pure conv sum (no BN stats) — ``fold_reparam()``.

``--model-size`` is a *parameter target* ("5M", "20m", "1.5B", "tiny"), and
``depth`` (auto or 1..8) sets the DiT block count; ``build_config`` solves
the width ladder to land on the target within ±5% while keeping the
architecture profile intact (see the solver section below).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Prompt embedding dim of the big model (Qwen3-0.6B via Qwen-Image,
# MiniTrainDIT crossattn_emb_channels — predict2.py default 1024).
PROMPT_DIM = 1024
IN_CHANNELS = 16
OUT_CHANNELS = 16
INPUT_CHANNELS = IN_CHANNELS * 2  # x_t ⊕ v_prev


# ═══════════════════════════════════════════════════════════════════════
#  Size targets & width/depth solver
# ═══════════════════════════════════════════════════════════════════════

# ``--model-size`` is a *parameter target*, not a fixed architecture: "5M",
# "20m", "1.5B" (K/M/B suffixes) or a named alias. The width ladder is solved
# to land on the target within ``DEFAULT_TOLERANCE`` while keeping the
# architecture profile intact (plan deep-dive §3.10, generalized):
#
#   channels = (bn/4, bn/2, bn);  bottleneck bn (multiple of 4, sincos assert);
#   heads = bn/head_dim with head_dim ≈ 32 (divisor in [24, 48] — SDPA
#           divisibility; attention params are head-count-independent);
#   mlp_ratio 4; cond_dim/prompt widths fixed; depth = DiT block count.
#
# Every layer is a linear map and every width is proportional to bn, so the
# parameter count is EXACTLY quadratic in bn for a fixed depth — the solver
# fits P(bn, d) = a·bn² + b·bn + c from three constructions and then works
# entirely on the closed form (deterministic, no per-candidate construction).

# Named aliases. "tiny" is the probe Day-1 UNet (plan Task 3d); its measured
# size is the point, and the solver reproduces it exactly (bn=112, depth 2).
SIZE_ALIASES: Dict[str, float] = {
    "tiny": 1_830_000.0,
}

BN_MIN = 64          # smallest bottleneck with a legal head_dim divisor
BN_MAX = 4096        # ~1.3–3.4B params at depth 2–8; beyond that nothing scales
HEAD_DIM_MIN, HEAD_DIM_MAX = 24, 48
AUTO_DEPTH_CANDIDATES = (2, 4, 6, 8)
BN_CEILING = 640     # 1.6× the canonical bottleneck 384 → auto adds depth past this
DEPTH_MIN, DEPTH_MAX = 1, 8
DEFAULT_TOLERANCE = 0.05

# Guardrail (plan §4b): warn when one corrector pass approaches ~1% of a full
# model step @512² (full step ≈ 3.4 TFLOP).
FULL_STEP_FLOP_512 = 3.4e12
FLOP_WARN_FRACTION = 0.01

_PARAM_TARGET_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([kmb]?)$", re.IGNORECASE)


def parse_param_target(spec: str) -> float:
    """Parse a size spec into a parameter-count target.

    ``"50M"`` → 50e6, ``"1.5B"`` → 1.5e9, ``"800k"`` → 8e5, ``"tiny"`` → the
    Day-1 alias. Unknown names raise ValueError.
    """
    key = str(spec).strip().lower()
    if key in SIZE_ALIASES:
        return SIZE_ALIASES[key]
    m = _PARAM_TARGET_RE.match(key)
    if m is None:
        raise ValueError(
            f"unknown corrector size {spec!r} — use a param target like "
            f"'5M'/'20m'/'1.5B' or {sorted(SIZE_ALIASES)}"
        )
    mult = {"k": 1e3, "m": 1e6, "b": 1e9}.get(m.group(2), 1.0)
    return float(m.group(1)) * mult


def _heads_for_bn(bn: int) -> int:
    """Head count for a bottleneck: divisor keeping head_dim closest to 32.

    head_dim must be in [HEAD_DIM_MIN, HEAD_DIM_MAX] and divide bn (SDPA
    requirement). The choice never affects the size solver (attention params
    are head-count-independent).
    """
    best = None
    for hd in range(HEAD_DIM_MIN, HEAD_DIM_MAX + 1):
        if bn % hd == 0 and (best is None or abs(hd - 32) < abs(best - 32)):
            best = hd
    assert best is not None, f"bn={bn} has no head_dim in [{HEAD_DIM_MIN}, {HEAD_DIM_MAX}]"
    return bn // best


_VALID_BNS: Optional[Tuple[int, ...]] = None


def _valid_bns() -> Tuple[int, ...]:
    """Bottleneck dims (multiples of 4) with a legal head_dim divisor, sorted."""
    global _VALID_BNS
    if _VALID_BNS is None:
        _VALID_BNS = tuple(
            bn for bn in range(BN_MIN, BN_MAX + 1, 4)
            if any(bn % hd == 0 for hd in range(HEAD_DIM_MIN, HEAD_DIM_MAX + 1))
        )
    return _VALID_BNS


def _cfg_for_dims(bn: int, depth: int, size: Optional[str] = None,
                  target: float = 0.0) -> CorrectorConfig:
    """Profile config for a bottleneck dim and block count (channels 1:2:4)."""
    return CorrectorConfig(
        size=size or f"{bn // 4}/{bn // 2}/{bn}",
        channels=(bn // 4, bn // 2, bn),
        bottleneck_dim=bn,
        heads=_heads_for_bn(bn),
        num_blocks=depth,
        target_params=target,
    )


_FIT_SAMPLES = (128, 256, 512)          # all legal head_dim=32 dims
_FIT_CACHE: Dict[int, Tuple[float, float, float]] = {}


def _fit_coeffs(depth: int) -> Tuple[float, float, float]:
    """Exact P(bn, depth) = a·bn² + b·bn + c, fitted from three constructions.

    Every width ∝ bn and every layer is linear ⇒ the count is exactly
    quadratic; three points determine it uniquely (head counts don't matter).
    """
    if depth in _FIT_CACHE:
        return _FIT_CACHE[depth]
    ys = [CorrectorUNet2D(_cfg_for_dims(bn, depth)).num_params() for bn in _FIT_SAMPLES]
    (b1, b2, b3), (y1, y2, y3) = _FIT_SAMPLES, ys
    a = ((y3 - y1) / (b3 - b1) - (y2 - y1) / (b2 - b1)) / (b3 - b2)
    b = (y2 - y1) / (b2 - b1) - a * (b1 + b2)
    c = y1 - a * b1 * b1 - b * b1
    _FIT_CACHE[depth] = (a, b, c)
    return (a, b, c)


def params_at(bn: int, depth: int) -> float:
    """Parameter count of the profile at (bn, depth) — exact closed form."""
    a, b, c = _fit_coeffs(depth)
    return a * bn * bn + b * bn + c


def _solve_bn(target: float, depth: int) -> Tuple[int, float]:
    """Closest legal bottleneck dim for (target, depth); returns (bn, achieved)."""
    best_bn, best_err = None, None
    for bn in _valid_bns():
        p = params_at(bn, depth)
        err = abs(p - target)
        if best_err is None or err < best_err:
            best_bn, best_err = bn, err
    return best_bn, params_at(best_bn, depth)


def build_config(target: float, depth="auto", tolerance: float = DEFAULT_TOLERANCE,
                 size: Optional[str] = None) -> CorrectorConfig:
    """Scale a corrector config to hit a parameter target.

    ``depth`` is ``"auto"`` (grow width first within the canonical family;
    add DiT blocks once the bottleneck exceeds BN_CEILING) or an int
    ``DEPTH_MIN..DEPTH_MAX`` (the widths scale down to compensate — at fixed
    params, w ∝ 1/sqrt(depth)). The closest achievable size is always
    returned; deviations beyond ``tolerance`` and guardrail violations
    (per-pass FLOPs approaching a full model step) print warnings.
    """
    label = size or f"{target / 1e6:.2f}M"
    if str(depth).strip().lower() == "auto":
        chosen = None          # (depth, bn, achieved); deepest viable as fallback
        for d in AUTO_DEPTH_CANDIDATES:
            bn, achieved = _solve_bn(target, d)
            if chosen is None or bn < chosen[1]:
                chosen = (d, bn, achieved)
            if bn <= BN_CEILING:
                break
        depth, bn, achieved = chosen
        if bn > BN_CEILING:
            print(f"  ⚠ {label}: target exceeds the canonical family ceiling "
                  f"(bottleneck ≤ {BN_CEILING}); using depth={depth} (deepest viable)")
    else:
        depth = int(depth)
        if not DEPTH_MIN <= depth <= DEPTH_MAX:
            raise ValueError(
                f"depth must be 'auto' or {DEPTH_MIN}..{DEPTH_MAX}, got {depth!r}"
            )
        lo = params_at(_valid_bns()[0], depth)
        hi = params_at(_valid_bns()[-1], depth)
        if not lo <= target <= hi:
            print(f"  ⚠ {label}: target outside depth={depth} reach "
                  f"({lo / 1e6:.1f}M..{hi / 1e6:.1f}M); clamping")
        bn, achieved = _solve_bn(target, depth)

    cfg = _cfg_for_dims(bn, depth, size=size, target=target)
    cfg.achieved_params = int(round(achieved))
    if abs(achieved - target) / target > tolerance:
        print(f"  ⚠ {label}: achieved {achieved / 1e6:.2f}M "
              f"(>{tolerance:.0%} off; closest on the discretized width ladder)")
    flops = estimate_corrector_flops(cfg)
    if flops > FLOP_WARN_FRACTION * FULL_STEP_FLOP_512:
        print(f"  ⚠ {label}: per-pass FLOPs @512² ≈ {flops / 1e9:.0f} GFLOP "
              f"({flops / FULL_STEP_FLOP_512:.1%} of a full model step) — the "
              f"corrector is no longer a cheap post-hook; consider a smaller target")
    return cfg


def estimate_corrector_flops(cfg: CorrectorConfig, img: int = 512) -> float:
    """Approximate per-pass FLOPs at a square resolution (order of magnitude).

    Used only for the size guardrail (plan §4b): conv layers at the UNet's
    spatial stages plus bottleneck attention/MLP at the (img/32)² token grid.
    """
    g = (img // 8) ** 2                     # latent pixels (4096 @512²)
    c0, c1, c2 = cfg.channels
    bn = cfg.bottleneck_dim
    conv = (
        2 * c0 * 32 * 9 * g                     # enc0: 32ch input @ g
        + 2 * c1 * c0 * 9 * g // 4              # enc1 @ g/4
        + 2 * c2 * c1 * 9 * g // 16             # enc2 @ g/16
        + 2 * c2 * (4 * c1) * 9 * g // 16       # up2 (pixel-shuffle ×4)
        + 2 * (2 * c1) * c1 * 9 * g // 4        # dec2
        + 2 * c1 * (4 * c0) * 9 * g // 4        # up1
        + 2 * (2 * c0) * c0 * 9 * g             # dec1
        + 2 * c0 * 16 * 9 * g                   # head
    )
    n = (img // 32) ** 2                    # bottleneck tokens (256 @512²)
    block = (28 * bn * bn + 5632 * bn) * n  # self+cross+mlp+adaLN, ×2 FMA
    return conv + cfg.num_blocks * block


@dataclass
class CorrectorConfig:
    """Full corrector configuration. Travels in the checkpoint metadata JSON."""
    arch: str = "unet2d"
    size: str = "20m"
    channels: Tuple[int, ...] = (64, 128, 256)
    bottleneck_dim: int = 256
    heads: int = 8
    num_blocks: int = 2
    cond_dim: int = 256
    mlp_ratio: float = 4.0
    prompt_dim: int = PROMPT_DIM
    in_channels: int = IN_CHANNELS
    out_channels: int = OUT_CHANNELS
    k_recommended: int = 1
    normalization: str = "none"          # "none" | "perchannel"
    normalization_stats: Optional[dict] = None   # {"mean": [...], "std": [...]} (32ch, input space)
    folded: bool = False                 # RepBlocks fused (export-time)
    target_params: float = 0.0           # requested size target (0 = fixed-dim config)
    achieved_params: int = 0             # measured at construction (solver closed form)

    @classmethod
    def for_size(cls, size: str = "20m", depth="auto",
                 tolerance: float = DEFAULT_TOLERANCE, **overrides) -> "CorrectorConfig":
        """Config for a size *target*: named alias or K/M/B param count.

        ``size`` is a target — ``"5M"``, ``"20m"``, ``"1.5B"``, ``"tiny"``
        (Day-1 alias) — and ``depth`` (``"auto"`` or ``DEPTH_MIN..DEPTH_MAX``)
        picks the DiT block count; the width ladder is solved to land on the
        target (see ``build_config``). The requested target is kept in
        ``target_params``, the achieved count in ``achieved_params``.
        """
        size = (size or "20m").strip().lower()
        target = parse_param_target(size)
        cfg = build_config(target, depth=depth, tolerance=tolerance, size=size)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> dict:
        return {
            "arch": self.arch, "size": self.size,
            "channels": list(self.channels),
            "bottleneck_dim": self.bottleneck_dim, "heads": self.heads,
            "num_blocks": self.num_blocks, "cond_dim": self.cond_dim,
            "mlp_ratio": self.mlp_ratio, "prompt_dim": self.prompt_dim,
            "in_channels": self.in_channels, "out_channels": self.out_channels,
            "k_recommended": self.k_recommended,
            "normalization": self.normalization,
            "normalization_stats": self.normalization_stats,
            "folded": self.folded,
            "target_params": self.target_params,
            "achieved_params": self.achieved_params,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CorrectorConfig":
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        cfg.channels = tuple(cfg.channels)
        return cfg


# ═══════════════════════════════════════════════════════════════════════
#  Positional + timestep embeddings (DiT reference, plan 6c)
# ═══════════════════════════════════════════════════════════════════════

def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """DiT's 1D sincos positional embedding (grid-size agnostic — any grid)."""
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid: Tuple[int, int],
                            device=None) -> torch.Tensor:
    """DiT's 2D sincos pos-embed for a (gh, gw) grid → (gh*gw, embed_dim)."""
    gh, gw = grid
    assert embed_dim % 4 == 0
    half = embed_dim // 2
    pos_h = torch.arange(gh, device=device, dtype=torch.float32)
    pos_w = torch.arange(gw, device=device, dtype=torch.float32)
    emb_h = get_1d_sincos_pos_embed_from_grid(half, pos_h)  # (gh, half)
    emb_w = get_1d_sincos_pos_embed_from_grid(half, pos_w)  # (gw, half)
    emb = torch.cat([emb_h[:, None, :].expand(gh, gw, half),
                     emb_w[None, :, :].expand(gh, gw, half)], dim=-1)
    return emb.reshape(gh * gw, embed_dim)


def timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: float = 10000.0) -> torch.Tensor:
    """DiT timestep embedding. t: (B,) → (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedder(nn.Module):
    """Linear(256→cond_dim) → SiLU → Linear(cond_dim→cond_dim)."""

    def __init__(self, cond_dim: int, embed_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, cond_dim, bias=True),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim, bias=True),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.mlp[0].in_features))


# ═══════════════════════════════════════════════════════════════════════
#  RepBlocks (deep-dive §3.1 variant (b) — GroupNorm branch-sum)
# ═══════════════════════════════════════════════════════════════════════

def _gn_groups(channels: int) -> int:
    """Largest divisor of *channels* ≤ 32 that divides it exactly."""
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class RepBlock(nn.Module):
    """y = SiLU(GN(conv3(x) + conv1(x))). Foldable: pure conv sum, no BN stats."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv3 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.conv1 = nn.Conv2d(cin, cout, 1, stride, 0, bias=False)
        self.gn = nn.GroupNorm(_gn_groups(cout), cout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv3(x)
        if self.conv1 is not None:
            y = y + self.conv1(x)
        return self.act(self.gn(y))

    def fuse(self):
        """In-place fold: merge conv1 into conv3 (exact — GN is after the sum)."""
        if self.conv1 is None:
            return
        w = self.conv3.weight.data + F.pad(self.conv1.weight.data, [1, 1, 1, 1])
        self.conv3.weight.data = w
        self.conv1 = None
        return self


# ═══════════════════════════════════════════════════════════════════════
#  Bottleneck (DiT blocks, 9-way adaLN, cross-attn to prompt)
# ═══════════════════════════════════════════════════════════════════════

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, D // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class CrossAttention(nn.Module):
    """Query = x; K/V = prompt (projected to cond_dim). 0-token-safe."""

    def __init__(self, dim: int, heads: int, kv_dim: int):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(kv_dim, 2 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor,
                prompt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """x: (B, N, D); prompt: (B, M, kv_dim); mask: (B, M) bool or None.

        Samples with no real tokens (0-token uncond form) get zero output —
        selected via ``torch.where`` so masked NaNs never propagate.
        """
        B, N, D = x.shape
        M = prompt.shape[1]
        if M == 0:
            return torch.zeros_like(x)
        q = self.q(x).reshape(B, N, self.heads, D // self.heads).transpose(1, 2)
        k, v = self.kv(prompt).chunk(2, dim=-1)
        k = k.reshape(B, M, self.heads, D // self.heads).transpose(1, 2)
        v = v.reshape(B, M, self.heads, D // self.heads).transpose(1, 2)
        has_prompt = torch.ones(B, dtype=torch.bool, device=x.device)
        if prompt_mask is not None:
            has_prompt = prompt_mask.any(dim=1)
            attn_mask = prompt_mask[:, None, None, :].expand(B, self.heads, N, M)
            attn_mask = attn_mask
        else:
            attn_mask = None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        if prompt_mask is not None and not has_prompt.all():
            out = torch.where(has_prompt[:, None, None], out, torch.zeros_like(out))
        return out


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """x = x + g1·attn(mod1(x)); x = x + g2·cross(mod2(x), prompt); x = x + g3·mlp(mod3(x)).

    9-way adaLN (shift/scale/gate × 3 submodules); all gates zero-init
    (plan 6c / deep-dive §3.2) ⇒ block is identity at init.
    """

    def __init__(self, dim: int, heads: int, cond_dim: int, mlp_ratio: float = 4.0,
                 kv_dim: int = PROMPT_DIM):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(dim, heads)
        self.cross = CrossAttention(dim, heads, kv_dim=kv_dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 9 * dim, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                prompt: torch.Tensor, prompt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        (s1, sc1, g1, s2, sc2, g2, s3, sc3, g3) = self.adaLN(c).chunk(9, dim=1)
        s1, sc1, g1 = s1.unsqueeze(1), sc1.unsqueeze(1), g1.unsqueeze(1)
        s2, sc2, g2 = s2.unsqueeze(1), sc2.unsqueeze(1), g2.unsqueeze(1)
        s3, sc3, g3 = s3.unsqueeze(1), sc3.unsqueeze(1), g3.unsqueeze(1)
        x = x + g1 * self.attn(modulate(self.norm1(x), s1, sc1))
        x = x + g2 * self.cross(modulate(self.norm1(x), s2, sc2), prompt, prompt_mask)
        x = x + g3 * self.mlp(modulate(self.norm2(x), s3, sc3))
        return x


# ═══════════════════════════════════════════════════════════════════════
#  Decoder + head
# ═══════════════════════════════════════════════════════════════════════

class UpBlock(nn.Module):
    """Pixel-shuffle 2× upsample (deep-dive §3.6): conv(cin→4·cout) + PixelShuffle(2)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout * 4, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(F.silu(self.conv(x)))


# ═══════════════════════════════════════════════════════════════════════
#  The model
# ═══════════════════════════════════════════════════════════════════════

class CorrectorUNet2D(nn.Module):
    """``(x_t ⊕ v_prev) → Δv̂`` on the (16, H, W) latent. Zero-init output head."""

    def __init__(self, cfg: CorrectorConfig):
        super().__init__()
        self.cfg = cfg
        c0, c1, c2 = cfg.channels
        prompt_proj_dim = cfg.prompt_dim

        # Encoder: 32ch input (x_t ⊕ v_prev)
        self.enc = nn.ModuleList([
            RepBlock(INPUT_CHANNELS, c0, stride=1),
            RepBlock(c0, c1, stride=2),
            RepBlock(c1, c2, stride=2),
        ])
        self.skip_dims = [c0, c1]

        # Bottleneck
        self.prompt_proj = nn.Linear(prompt_proj_dim, cfg.cond_dim, bias=True)
        self.t_embedder = TimestepEmbedder(cfg.cond_dim)
        self.blocks = nn.ModuleList([
            DiTBlock(cfg.bottleneck_dim, cfg.heads, cfg.cond_dim, cfg.mlp_ratio,
                     kv_dim=cfg.cond_dim)
            for _ in range(cfg.num_blocks)
        ])

        # Decoder: up + skip-concat + RepBlock
        self.up2 = UpBlock(cfg.bottleneck_dim, c1)           # → c1 @ H/2
        self.dec2 = RepBlock(c1 + c1, c1)                    # skip from enc1
        self.up1 = UpBlock(c1, c0)                           # → c0 @ H
        self.dec1 = RepBlock(c0 + c0, c0)                    # skip from enc0

        # Zero-init output head: Δv̂ = 0 at init ⇒ Mode B′ ≡ Mode A exactly
        self.head_gn = nn.GroupNorm(_gn_groups(c0), c0)
        self.head_conv = nn.Conv2d(c0, cfg.out_channels, 3, 1, 1)
        nn.init.zeros_(self.head_conv.weight)
        nn.init.zeros_(self.head_conv.bias)

        # Input normalization (perchannel option, plan 6d)
        stats = cfg.normalization_stats or {}
        self.register_buffer("_norm_mean", torch.tensor(stats.get("mean", []), dtype=torch.float32)
                             if stats.get("mean") else None)
        self.register_buffer("_norm_std", torch.tensor(stats.get("std", []), dtype=torch.float32)
                             if stats.get("std") else None)

        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    # ── forward ────────────────────────────────────────────────────────

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if self._norm_mean is None or self._norm_std is None:
            return x
        return (x - self._norm_mean.view(1, -1, 1, 1)) / self._norm_std.view(1, -1, 1, 1)

    def _unnormalize_output(self, out: torch.Tensor) -> torch.Tensor:
        if self._norm_std is None:
            return out
        v_std = self._norm_std[IN_CHANNELS:]
        return out * v_std.view(1, -1, 1, 1)

    def forward(self, x_t: torch.Tensor, v_prev: torch.Tensor,
                prompt: torch.Tensor, t: torch.Tensor,
                prompt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:
            x_t: (B, 16, H, W) raw pre-pad latent (current timestep)
            v_prev: (B, 16, H, W) timestep-aware Mode-A velocity (v_MA)
            prompt: (B, N, 1024) cross-attn embeddings (N=0 → identity)
            t: (B,) or (B, 1) step fraction
            prompt_mask: optional (B, N) bool — True = real token
        Returns Δv̂ (B, 16, H, W) in real velocity units.
        """
        if t.ndim == 2:
            t = t[:, 0]
        x = torch.cat([x_t, v_prev], dim=1).float()
        x = self._normalize_input(x)

        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < 2:
                skips.append(x)

        B, _, H, W = x.shape
        c = self.t_embedder(t.to(x.device))
        p = self.prompt_proj(prompt.float().to(x.device)) if prompt.numel() else None
        if p is not None and prompt_mask is not None:
            p = p * prompt_mask.unsqueeze(-1).to(p.dtype)

        b = x.flatten(2).transpose(1, 2)                       # (B, gh*gw, C)
        pos = get_2d_sincos_pos_embed(self.cfg.bottleneck_dim, (H, W), device=x.device)
        b = b + pos.unsqueeze(0)
        for block in self.blocks:
            b = block(b, c, p if p is not None else torch.zeros(B, 0, self.cfg.cond_dim,
                                                                 device=x.device),
                      prompt_mask)
        x = b.transpose(1, 2).reshape(B, self.cfg.bottleneck_dim, H, W)

        x = self.up2(x)                                        # c1 @ 2H
        x = torch.cat([x, skips[1]], dim=1)
        x = self.dec2(x)
        x = self.up1(x)                                        # c0 @ 4H
        x = torch.cat([x, skips[0]], dim=1)
        x = self.dec1(x)

        out = self.head_conv(F.silu(self.head_gn(x)))
        out = self._unnormalize_output(out)
        return out.to(v_prev.dtype)

    # ── K-pass refinement ─────────────────────────────────────────────

    def refine(self, x_t, v_prev, prompt, t, K: int, prompt_mask=None):
        """Weight-shared K-pass refinement: v ← v + model(x_t, v, prompt, t).

        Passes are trained on-policy with deep supervision (plan 4b/6f);
        at inference K is fixed per run (default 1). No stopping rules.
        """
        v = v_prev
        for _ in range(K):
            v = v + self(x_t, v, prompt, t, prompt_mask)
        return v

    # ── Export ─────────────────────────────────────────────────────────

    def fold_reparam(self) -> "CorrectorUNet2D":
        """Fuse every RepBlock's conv1 into conv3 (exact — deep-dive §3.1(b))."""
        for blk in self.enc:
            blk.fuse()
        for blk in (self.dec1, self.dec2):
            blk.fuse()
        self.cfg.folded = True
        return self

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════
#  Checkpoint IO (plan Task 4c)
# ═══════════════════════════════════════════════════════════════════════

def save_corrector(model: CorrectorUNet2D, path, cfg: Optional[CorrectorConfig] = None,
                   ema: Optional[dict] = None, extra_metadata: Optional[dict] = None,
                   dtype=torch.float16):
    """Save EMA (or online) weights as fp16 .safetensors + embedded JSON config."""
    import safetensors.torch as st
    path = Path(path)
    cfg = cfg or model.cfg
    if ema is not None:
        model = model.__class__(cfg)
        model.load_state_dict(ema)
    state = {k: v.detach().to(dtype).contiguous() for k, v in model.state_dict().items()}
    meta = {"config": json.dumps(cfg.to_dict(), default=str)}
    if extra_metadata:
        for k, v in extra_metadata.items():
            meta[str(k)] = json.dumps(v, default=str) if not isinstance(v, str) else v
    path.parent.mkdir(parents=True, exist_ok=True)
    st.save_file(state, str(path), metadata=meta)
    return path


_CORRECTOR_CACHE: Dict[Tuple[str, float], CorrectorUNet2D] = {}
_CORRECTOR_INF_CACHE: Dict[Tuple[str, str], nn.Module] = {}


def prepare_corrector(path, device) -> nn.Module:
    """Load + move + torch.compile the corrector for inference (plan 4a).

    Cached per (path, device). The shared load cache is deep-copied before
    moving so callers never mutate the cached CPU weights. Falls back to
    eager if torch.compile fails. Set ``TEA_CACHE_NO_COMPILE=1`` to skip
    compilation entirely (e.g. on platforms with flaky CUDA-graph capture).
    """
    import copy
    import os
    key = (str(path), str(device))
    if key in _CORRECTOR_INF_CACHE:
        return _CORRECTOR_INF_CACHE[key]
    corr = copy.deepcopy(load_corrector(path)).to(device).eval()
    if not os.environ.get("TEA_CACHE_NO_COMPILE"):
        try:
            corr = torch.compile(corr, mode="reduce-overhead")
        except Exception as e:
            print(f"  [TeaCache] ⚠ corrector compile failed ({e}); using eager")
    _CORRECTOR_INF_CACHE[key] = corr
    return corr


def load_corrector(path) -> CorrectorUNet2D:
    """Load a corrector checkpoint, cached by (path, mtime). Returns eval model."""
    import safetensors.torch as st
    path = str(Path(path).expanduser())
    mtime = Path(path).stat().st_mtime
    key = (path, mtime)
    if key in _CORRECTOR_CACHE:
        return _CORRECTOR_CACHE[key]
    if len(_CORRECTOR_CACHE) > 8:  # bound the cache
        _CORRECTOR_CACHE.clear()
    data = st.load_file(path)
    import safetensors
    with safetensors.safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
    cfg = CorrectorConfig.from_dict(json.loads(meta.get("config", "{}")))
    model = CorrectorUNet2D(cfg)
    if cfg.folded:
        model.fold_reparam()
    model.load_state_dict(data)
    model.eval()
    _CORRECTOR_CACHE[key] = model
    return model
