#!/usr/bin/env python3
"""Phase 3.5: Refiner verification (the refiner side of the verifier split).

``validate.py`` verifies base TeaCache (skip rates, image metrics vs
baseline); this module verifies the *latent corrector* on top of it, with
the comparisons motivated by the channel/spectral investigation of the 30M
corrector:

  per-prompt (in-distribution + OOD) × control point, it runs:
    baseline (full model), Mode A (TeaCache), Mode B′ (TeaCache + corrector),
    optionally Mode B′ with a per-(area, t-region) trust map, and (with
    ``--sanity-zero-init``) a fresh zero-init corrector B′ run used as a
    harness self-test (B′ ≡ A byte-for-byte catches silently-dead correctors).

  velocity-level (from one full-model recording per prompt):
    - per-(t-region, lag) K0/K1 rel-MSE → the ladder the deployment actually
      samples from
    - per-channel recovery / energy shares / gains (channel-uniformity gate:
      the "14 good 2 bad" regression test)
    - d=0 anchor perturbation (diagnostic: never fires in deployment, but a
      future pipeline change could expose it)
    - spectral grain test: high-frequency share of the K1 residual vs the
      velocity signal (the grain metric pooled rel-MSE cannot see)

  deployed-weighted recovery:
    the Mode-A skip log (exact per-step lag decisions) × the recording's
    per-(region, lag) errors → the expected per-skip-step error of Mode A
    vs Mode B′. This is the number that matches what users actually see;
    plain pooled rel-MSE over the ladder overstates it at aggressive
    control points.

  pixel-level (decode + pyiqa suite):
    - A vs baseline and B′ vs baseline on the full metric tiers
    - perceptual no-regression flags: B′ must not regress LPIPS/DISTS/VIF
      vs Mode A at the same control point
    - final-latent spectral test (HF share of the residual vs the signal)
    - optional per-channel perceptual ablation (16 channel-masked decodes)

  lag-depth coverage: max deployed lag from the skip logs vs the ladder the
  checkpoint was trained on (from its embedded config_snapshot) — the 30M
  model was trained to lag 16 while max-error deployment reaches lag 24.

Usage:
    python -m tuning.verify_refiner --comfy-dir /path/to/ComfyUI \
        --corrector models/corrector-30m-96000.safetensors \
        --config-errors 0.020,0.054 --seed 1 --out-dir refiner_verify

ComfyUI must be recent enough for the Qwen3 text encoder (the docker image
works). Model files are resolved via ``comfy_dir/models`` (symlink your
ComfyData/models there if needed).
"""

import argparse
import bisect
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tuning.corrector import high_freq_fraction

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ══════════════════════════════════════════════════════════════════════
#  Built-in prompt sets
# ══════════════════════════════════════════════════════════════════════

IN_DIST_PROMPTS = [
    ("1girl, archer, detailed hair, forest background, sunlit clearing, "
     "masterpiece, best quality"),
    ("1boy, knight, ornate armor, dramatic lighting, city street at dusk, "
     "highly detailed, masterpiece"),
]

# OOD: styles far from the training corpus (detailed character art). The 30M
# corrector measurably degrades these at conservative control points — the
# verifier treats them as a separate gate, not as noise.
OOD_PROMPTS = [
    "a minimal geometric logo design, flat vector style, bold shapes, white background, corporate branding, no text",
    "a simple flat illustration of a house, hard shading, minimal background, plain colors",
    "a bold vintage movie poster, limited color palette, strong typography, grainy print texture",
    "a clean line-art tattoo design, single black ink color, white background, ornamental swirls",
    "an isometric game asset render, low poly, soft studio lighting, solid pastel background",
]


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════


def _samples(x):
    """Unwrap ComfyUI node returns (dict / tensor / list) → tensor."""
    if isinstance(x, dict):
        return _samples(x.get("samples"))
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return _samples(x[0])
    return x


def to01(x: torch.Tensor) -> torch.Tensor:
    """VAE-decoded image (any layout) → (B, 3, H, W) float [0, 1]."""
    x = x.float().cpu()
    if x.dim() == 5:
        x = x.squeeze(2)
    if x.dim() == 4 and x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2)
    elif x.dim() == 4 and x.shape[1] == 3:
        pass
    elif x.dim() == 3:
        x = x.unsqueeze(0)
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
    return (x.clamp(-1, 1) + 1) / 2


def region_of(f: float) -> str:
    if f < 0.34:
        return "early"
    if f < 0.67:
        return "mid"
    return "late"


def step_fraction(step_idx: int, n_sigmas: int) -> float:
    return step_idx / max(n_sigmas - 1, 1)


# ══════════════════════════════════════════════════════════════════════
#  Verifier
# ══════════════════════════════════════════════════════════════════════


class RefinerVerifier:
    def __init__(self, args):
        self.args = args
        self.comfy_dir = args.comfy_dir
        self.out_dir = Path(args.out_dir)
        self.data_dir = self.out_dir / "refiner_data"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TEA_CACHE_NO_COMPILE", "1")

        # ── ComfyUI path + model loading ──────────────────────────────
        from tuning.utils import load_models, setup_comfy_path
        setup_comfy_path(self.comfy_dir)
        sys.path.insert(0, self.comfy_dir)
        self.nodes = __import__("nodes")
        self.unet, self.clip, self.vae = load_models(
            self.comfy_dir, args.model_name, args.clip_name,
            args.clip_type, args.vae_name)

        from tuning.corrector import load_corrector, prepare_corrector
        self.load_corrector = load_corrector
        self.prepare_corrector = prepare_corrector
        self.corr_cfg = load_corrector(args.corrector).cfg
        self.trained_lags = self._read_trained_lags(args.corrector)

        from tuning.config_types import TeacacheConfig
        self.TeacacheConfig = TeacacheConfig
        from tuning.forward import teacache_anima_forward
        self.teacache_anima_forward = teacache_anima_forward
        from tuning.utils import get_diffusion_model
        self.get_diffusion_model = get_diffusion_model

        # ── Control points from the shipped presets ───────────────────
        presets_path = Path(__file__).parent.parent / "anima_presets.json"
        presets = json.load(open(presets_path))
        cps = presets["control_points"]
        self.control_points = []
        for err_str in str(args.config_errors).split(","):
            err_str = err_str.strip()
            if not err_str:
                continue
            target = float(err_str)
            cp = min(cps, key=lambda c: abs(c["error"] - target))
            self.control_points.append((cp["error"], cp["config"]))
        self.cps = cps

        # ── Metrics ───────────────────────────────────────────────────
        self.metrics = None
        if not args.no_image_metrics:
            import pyiqa
            names = ["psnr", "ssim", "lpips", "dists", "gmsd", "fsim",
                     "vif", "ms_ssim"]
            self.metrics = {n: pyiqa.create_metric(n, device="cuda")
                            for n in names}
            self.lower_better = {"lpips", "dists", "gmsd"}

    def _read_trained_lags(self, corrector_path: str) -> Optional[List[int]]:
        try:
            import safetensors
            with safetensors.safe_open(corrector_path, framework="pt") as f:
                meta = f.metadata() or {}
            snap = json.loads(meta.get("config_snapshot", "{}"))
            # New checkpoints carry the actual ladder (train_corrector
            # persists it as trained_lags); older ones fall back to the
            # lag_weights-count heuristic below.
            for key in ("trained_lags", "record_lags", "lags"):
                lags = snap.get(key)
                if lags:
                    return [int(l) for l in lags]
            # The legacy training config stores per-lag weights, not the
            # ladder itself. The shipped 30M corpus ladder is documented in
            # the README: [1, 2, 3, 4, 8, 16] (6 weights); the older default
            # is the doubling sequence [1, 2, 4, 8, 16] (5 weights).
            n = len(snap.get("lag_weights") or [])
            if n == 6:
                return [1, 2, 3, 4, 8, 16]
            if n == 5:
                return [1, 2, 4, 8, 16]
            if n > 6:
                base = [1, 2, 3, 4, 8, 16]
                return base + [base[-1] * 2 ** k for k in range(1, n - 5)]
            return None
        except Exception:
            return None

    # ── sampling primitives ───────────────────────────────────────────

    def _encode(self, text: str):
        return self.nodes.CLIPTextEncode().encode(self.clip, text)[0]

    def _sample(self, prompt: str, seed: int, steps: int, w: int, h: int):
        pos = self._encode(prompt)
        neg = self._encode("")
        lat = self.nodes.EmptyLatentImage().generate(w, h, 1)[0]
        samples = self.nodes.KSampler().sample(
            self.unet, seed, steps, self.args.cfg, self.args.sampler,
            self.args.scheduler, pos, neg, lat, 1.0)[0]
        return _samples(samples).detach().float().cpu()

    def _decode(self, lat: torch.Tensor) -> torch.Tensor:
        img = self.nodes.VAEDecode().decode(
            self.vae, {"samples": lat.to(self.vae.device)})[0]
        return img.detach().float().cpu()

    def _reset_dm(self, dm):
        for attr in ("teacache_state", "_refiner_buf", "_calib_state",
                     "_tc_corrector", "_tc_collect_stats",
                     "_tc_corr_slot_stats", "_tc_diag_runs", "_tc_diag_skips",
                     "_tc_diag_printed", "_tc_diag_final", "_refiner_dir",
                     "_refiner_record_slots", "_refiner_dtype",
                     "_refiner_lags"):
            if hasattr(dm, attr):
                delattr(dm, attr)
        if hasattr(dm, "calibration_log"):
            dm.calibration_log = []

    # ── run modes ─────────────────────────────────────────────────────

    def run_baseline(self, prompt: str, seed: int, steps: int, w: int, h: int):
        dm = self.get_diffusion_model(self.unet)
        self._reset_dm(dm)
        to = self.unet.model_options.setdefault("transformer_options", {})
        to["enable_teacache"] = False
        lat = self._sample(prompt, seed, steps, w, h)
        return lat

    def run_teacache(self, prompt: str, seed: int, steps: int, w: int, h: int,
                     cfg_dict: dict, corrector_path: Optional[str] = None,
                     trust: float = 1.0,
                     trust_map: Optional[dict] = None,
                     zero_init: bool = False):
        """One TeaCache run (Mode A, B′, B′+trust-map, or zero-init sanity).

        Returns (latent, skip_log). The wrapper sets BOTH ``current_percent``
        and ``tc_current_percent`` (the key the forward actually reads — an
        easy wiring bug that silently keeps the hook in the early region).
        """
        from tuning.corrector import CorrectorConfig, CorrectorUNet2D
        dm = self.get_diffusion_model(self.unet)
        self._reset_dm(dm)
        cfg = self.TeacacheConfig.from_dict(cfg_dict)
        if corrector_path is not None:
            cfg.correction_mode = "latent_denoiser"
            cfg.refine_passes = 1
            cfg.corrector_trust = trust
            if trust_map:
                cfg.corrector_trust_map = trust_map
            if zero_init:
                corr = CorrectorUNet2D(CorrectorConfig.for_size("tiny"))
                corr = corr.to(next(dm.parameters()).device).eval()
                dm._tc_corrector = corr
            else:
                dm._tc_corrector = self.prepare_corrector(
                    corrector_path, next(dm.parameters()).device)
        original = dm._forward
        dm._forward = self.teacache_anima_forward.__get__(dm, dm.__class__)
        to = self.unet.model_options.setdefault("transformer_options", {})
        cfg.inject_into_transformer_options(to)
        to["enable_teacache"] = True

        skip_log = []
        start = float(cfg.start_percent)
        end = float(cfg.end_percent)

        def tc_wrapper(model_function, kwargs):
            c = kwargs["c"]
            timestep = kwargs["timestep"]
            c_to = c.setdefault("transformer_options", {})
            sigmas = c_to.get("sample_sigmas")
            if sigmas is not None:
                matched = (sigmas == timestep[0]).nonzero()
                if len(matched) > 0:
                    step_idx = matched[0].item()
                else:
                    step_idx = 0
                    for i in range(len(sigmas) - 1):
                        if (sigmas[i] - timestep[0]) * \
                                (sigmas[i + 1] - timestep[0]) <= 0:
                            step_idx = i
                            break
                frac = step_idx / max(len(sigmas) - 1, 1)
                c_to["current_percent"] = frac
                c_to["tc_current_percent"] = torch.tensor(frac)
                c_to["enable_teacache"] = start <= frac <= end
            out = model_function(kwargs["input"], timestep, **c)
            st = getattr(dm, "teacache_state", None)
            if st:
                skip_log.append({
                    "step": step_idx,
                    "skipped": not any(s.get("should_calc", True)
                                       for s in st.values()),
                    "lags": [int(s.get("lag", 0)) for s in st.values()],
                })
            return out

        self.unet.set_model_unet_function_wrapper(tc_wrapper)
        try:
            lat = self._sample(prompt, seed, steps, w, h)
        finally:
            dm._forward = original
            self.unet.model_options.pop("model_function_wrapper", None)
        return lat, skip_log

    # ── recording (full model every step, ladder capture) ─────────────

    def run_recording(self, prompt: str, seed: int, steps: int, w: int, h: int,
                      name: str, prompt_id: int, record_lags: List[int]):
        from tuning.calibrate import patch_for_refiner
        from tuning import refiner_data
        dm, original = patch_for_refiner(
            self.unet, steps, prompt_id, seed, track_per_block=False,
            refiner_dir=str(self.data_dir),
            refiner_cfg={"record_slots": "both", "record_lags": record_lags,
                         "dtype": "bfloat16", "clevel": 5, "top_n": -1})
        try:
            lat = self._sample(prompt, seed, steps, w, h)
        finally:
            dm._forward = original
            self.unet.model_options.pop("model_function_wrapper", None)
        manifest = refiner_data.init_manifest({})
        run_meta = {"name": name, "prompt_id": prompt_id, "seed": seed,
                    "sampler": self.args.sampler,
                    "scheduler": self.args.scheduler, "cfg": self.args.cfg,
                    "width": w, "height": h, "steps": steps}
        entry = refiner_data.finalize_refiner_generation(
            dm, self.data_dir, manifest, run_meta,
            {"codec": 1, "clevel": 5, "record_lags": record_lags,
             "top_n": -1})
        if entry is None:
            raise RuntimeError(
                "recording produced no refiner data — generation too short "
                f"for record_lags {record_lags}?")
        return entry, lat

    # ── velocity-level analysis on a recording ────────────────────────

    def analyze_recording(self, bin_path: str, prompt_path: str,
                          corrector_path: str, steps: int):
        """Per-(region, lag) K0/K1, per-channel stats, anchor, spectral."""
        from tuning import refiner_data
        gen = refiner_data.load_generation(bin_path)
        prompts = refiner_data.load_prompt_file(prompt_path)
        # Deep-copied: the shared load_corrector cache must never be moved to
        # CUDA by analysis (mirrors prepare_corrector's copy-before-move).
        corr = copy.deepcopy(self.load_corrector(corrector_path)).cuda().eval()
        device = "cuda"

        E0: Dict[Tuple[str, int], float] = {}
        E1: Dict[Tuple[str, int], float] = {}
        V: Dict[Tuple[str, int], float] = {}
        ch_E0 = torch.zeros(16, device=device)
        ch_E1 = torch.zeros(16, device=device)
        ch_V = torch.zeros(16, device=device)
        ch_T = torch.zeros(16, device=device)      # Σ‖v̂‖²
        anchor_e = {r: 0.0 for r in ("early", "mid", "late")}
        anchor_n = {r: 0 for r in ("early", "mid", "late")}
        spec_e = {r: [0.0, 0] for r in ("early", "mid", "late")}
        spec_v = {r: [0.0, 0] for r in ("early", "mid", "late")}
        n_pairs = 0
        n_lag0 = 0
        for slot in gen.recorded_slots:
            fracs = gen.step_fractions[slot] or \
                [i / max(steps - 1, 1) for i in range(len(gen.v_true[slot]))]
            prompt = prompts.get(slot)
            if prompt is not None and prompt.numel() == 0:
                prompt = None
            for t in range(len(gen.v_true[slot])):
                reg = region_of(fracs[t])
                x_t = gen.x[slot][t].cuda()
                vt = gen.v_true[slot][t].cuda()
                ch_V += (vt ** 2).sum(dim=(1, 2))
                # d=0 anchor diagnostic
                p = (prompt.cuda()[None] if prompt is not None
                     else torch.zeros(1, 0, self.corr_cfg.prompt_dim,
                                      device=device))
                t_frac = torch.tensor([fracs[t]], device=device)
                with torch.no_grad():
                    dv0 = corr.forward(x_t[None], vt[None], p, t_frac,
                                       lag=torch.tensor([0.0], device=device))
                anchor_e[reg] += float((dv0 ** 2).sum() / (vt ** 2).sum())
                anchor_n[reg] += 1
                for d, vma in gen.v_ma[slot][t].items():
                    key = (reg, d)
                    vm = vma.cuda()
                    V[key] = V.get(key, 0) + float((vt ** 2).sum())
                    E0[key] = E0.get(key, 0) + \
                        float(((vm - vt) ** 2).sum())
                    # K1 goes through refine() (K=1), not forward: refine is
                    # what deployment consumes, so an embedded gain
                    # calibration is measured here exactly as it applies at
                    # inference. The d=0 anchor above stays on forward (a
                    # raw-model diagnostic; the corrector never runs at d=0).
                    with torch.no_grad():
                        v_hat = corr.refine(x_t[None], vm[None], p, t_frac, 1,
                                            lag=torch.tensor(
                                                [float(d)], device=device))
                    v_hat = v_hat[0]
                    e = v_hat - vt
                    E1[key] = E1.get(key, 0) + float((e ** 2).sum())
                    ch_E0 += ((vm - vt) ** 2).sum(dim=(1, 2))
                    ch_E1 += (e ** 2).sum(dim=(1, 2))
                    ch_T += (v_hat ** 2).sum(dim=(1, 2))
                    # spectral: HF share of the K1 residual vs the signal
                    spec_e[reg][0] += high_freq_fraction(e.unsqueeze(0))
                    spec_v[reg][0] += high_freq_fraction(vt.unsqueeze(0))
                    spec_e[reg][1] += 1
                    spec_v[reg][1] += 1
                    n_pairs += 1

        rel = {k: (E0[k] / V[k], E1[k] / V[k]) for k in V if V[k] > 0}
        per_channel = {
            "rel_mse_K0": [float(x / ch_V[i]) for i, x in
                           enumerate(ch_E0.tolist())],
            "rel_mse_K1": [float(x / ch_V[i]) for i, x in
                           enumerate(ch_E1.tolist())],
            "energy_share": [float(x / ch_V.sum()) for x in ch_V.tolist()],
            "error_share_K1": [float(x / ch_E1.sum()) for x in
                               ch_E1.tolist()],
            "gain_T_over_V": [float(x / ch_V[i]) for i, x in
                              enumerate(ch_T.tolist())],
        }
        recovery = [1 - a / b for a, b in
                    zip(per_channel["rel_mse_K1"],
                        per_channel["rel_mse_K0"])]
        per_channel["recovery_pct"] = [r * 100 for r in recovery]
        per_channel["recovery_uniformity"] = {
            "mean": sum(recovery) / len(recovery),
            "std": (sum((r - sum(recovery) / len(recovery)) ** 2
                        for r in recovery) / len(recovery)) ** 0.5,
            "min": min(recovery), "max": max(recovery),
            "channels_below_50pct": [int(i) for i, r in
                                     enumerate(recovery) if r < 0.5],
        }

        lags_sorted = sorted({k[1] for k in rel})
        ladder = {}
        for reg in ("early", "mid", "late"):
            ladder[reg] = {
                str(d): {"K0": round(rel.get((reg, d), (float("nan"),
                        float("nan")))[0], 5),
                         "K1": round(rel.get((reg, d), (float("nan"),
                        float("nan")))[1], 5)}
                for d in lags_sorted
            }
        pooled_lags = {}
        for d in lags_sorted:
            sV = sum(V.get((reg, d), 0) for reg in ("early", "mid", "late"))
            sE0 = sum(E0.get((reg, d), 0) for reg in ("early", "mid", "late"))
            sE1 = sum(E1.get((reg, d), 0) for reg in ("early", "mid", "late"))
            if sV > 0:
                pooled_lags[str(d)] = {"K0": sE0 / sV, "K1": sE1 / sV}
        pooled = {"K0": sum(E0.values()) / sum(V.values()),
                  "K1": sum(E1.values()) / sum(V.values())}
        return {
            "n_pairs": n_pairs,
            "ladder": ladder,
            "pooled_lags": {k: {kk: round(vv, 5) for kk, vv in v.items()}
                            for k, v in pooled_lags.items()},
            "pooled": {k: round(v, 5) for k, v in pooled.items()},
            "pooled_recovery_pct": round(
                100 * (1 - pooled["K1"] / pooled["K0"]), 2),
            "per_channel": per_channel,
            "anchor_perturbation": {
                r: round(anchor_e[r] / max(anchor_n[r], 1), 5)
                for r in ("early", "mid", "late")},
            "spectral_hf": {
                r: {"residual_K1": round(spec_e[r][0] / max(spec_e[r][1], 1),
                                         4),
                    "signal": round(spec_v[r][0] / max(spec_v[r][1], 1), 4)}
                for r in ("early", "mid", "late")},
        }

    # ── deployed-weighted recovery ────────────────────────────────────

    def deployed_weighted(self, skip_log: list, rec_analysis: dict,
                          steps: int):
        """Expected per-buffer-skip rel-MSE from the Mode-A schedule × ladder.

        Lags beyond the recorded ladder use the deepest recorded lag's error
        as a plateau (deployment reaches lag 24; the ladder stops at 16).
        Lags BETWEEN recorded ladder entries use the next recorded lag at or
        above them — error grows with lag, so this is the conservative,
        monotone-consistent choice (the old code fell back to the deepest lag
        for any missing cell, inflating mid-ladder estimates).
        """
        lags = sorted({int(k) for k in rec_analysis["pooled_lags"]})
        if not lags:
            return None
        # per-region ladder cells (NaN-guarded), sorted per region for the
        # gap resolution below.
        region_rel = {}
        for reg in ("early", "mid", "late"):
            for d in lags:
                cell = rec_analysis["ladder"][reg].get(str(d))
                if cell and cell["K0"] == cell["K0"]:
                    region_rel[(reg, d)] = (cell["K0"], cell["K1"])
        region_cells = {reg: sorted(d for d in lags if (reg, d) in region_rel)
                        for reg in ("early", "mid", "late")}
        sumA = sumB = n = 0
        max_lag = 0
        lag_hist = {}
        for e in skip_log:
            if not e["skipped"]:
                continue
            reg = region_of(step_fraction(e["step"], steps))
            cells = region_cells[reg]
            if not cells:
                continue
            for lag in e["lags"]:
                max_lag = max(max_lag, lag)
                lag_hist[lag] = lag_hist.get(lag, 0) + 1
                d = cells[min(bisect.bisect_left(cells, lag), len(cells) - 1)]
                cell = region_rel[(reg, d)]
                if cell[0] == cell[0]:
                    sumA += cell[0]
                    sumB += cell[1]
                    n += 1
        if n == 0:
            return None
        meanA, meanB = sumA / n, sumB / n
        return {
            "n_buffer_skips": n,
            "max_deployed_lag": max_lag,
            "lag_histogram": {str(k): v for k, v in
                              sorted(lag_hist.items())},
            "modeA_per_skip": round(meanA, 5),
            "modeB_per_skip": round(meanB, 5),
            "recovery_pct": round(100 * (1 - meanB / meanA), 2),
        }

    # ── image metrics ─────────────────────────────────────────────────

    def image_metrics(self, img_a: torch.Tensor, img_b: torch.Tensor):
        out = {}
        img_a, img_b = to01(img_a).cuda(), to01(img_b).cuda()
        for name, met in self.metrics.items():
            with torch.no_grad():
                if name in self.lower_better:
                    v = met(img_b, img_a)
                else:
                    v = met(img_a, img_b)
                out[name] = round(float(v.mean()), 5)
        return out

    # ── per-channel perceptual ablation ───────────────────────────────

    def per_channel_ablation(self, lat_base: torch.Tensor,
                             lat_b: torch.Tensor, vae_dev: str = "cuda"):
        """LPIPS damage of each channel of the B′ final-latent error alone."""
        import pyiqa
        lpips = pyiqa.create_metric("lpips", device="cuda")
        img0 = to01(self._decode(lat_base)).cuda()
        err = lat_b - lat_base
        per_ch = {}
        for c in range(lat_base.shape[1]):
            Lp = lat_base.clone()
            Lp[:, c, ...] += err[:, c, ...]
            imgp = to01(self._decode(Lp)).cuda()
            with torch.no_grad():
                per_ch[str(c)] = round(
                    float(lpips(imgp, img0).mean()), 5)
        return per_ch

    # ── top-level flow ────────────────────────────────────────────────

    def _corrector_alive(self, corr) -> bool:
        """A live corrector must produce a nonzero delta on a synthetic input.

        Unlike the A/B latent-divergence check, this is immune to an embedded
        gain calibration: a zero-output corrector still rescales the latent
        through the gains, so only a direct delta probe can detect it. The
        normalization lookup maps any input area to the nearest stratum, and
        the 0-token prompt is the documented identity form.
        """
        dev = next(corr.parameters()).device
        x = torch.randn(1, 16, 64, 64, device=dev)
        v = torch.randn_like(x) * 0.1
        t = torch.tensor([0.5], device=dev)
        lag = torch.tensor([2.0], device=dev)
        p = torch.zeros(1, 0, self.corr_cfg.prompt_dim, device=dev)
        with torch.no_grad():
            dv = corr.forward(x, v, p, t, lag=lag)
        return float(dv.abs().max()) > 1e-6

    def verify_prompt(self, prompt: str, seed: int, steps: int, w: int,
                      h: int, prompt_id: int, corrector_path: str,
                      record_lags: List[int], png: Optional[Path]):
        name = f"gen_{prompt_id:04d}"
        out = {"prompt": prompt, "seed": seed}
        t0 = time.time()

        # 0. dead-corrector probe (once per prompt, before any runs)
        dm0 = self.get_diffusion_model(self.unet)
        out["corrector_alive"] = self._corrector_alive(
            self.prepare_corrector(corrector_path,
                                   next(dm0.parameters()).device))

        # 1. baseline + recording (full model every step)
        lat_base = self.run_baseline(prompt, seed, steps, w, h)
        entry, lat_rec = self.run_recording(
            prompt, seed, steps, w, h, name, prompt_id, record_lags)
        rec = self.analyze_recording(
            f"{self.data_dir}/{entry['bin']}",
            f"{self.data_dir}/{entry['prompt_bin']}",
            corrector_path, steps)
        out["velocity_level"] = rec
        out["recording"] = entry["name"]

        # 2. per control point: A, B′, optional trust-map + zero-init
        cfgs = []
        lat_a_first = None
        for err, cfg_dict in self.control_points:
            cname = f"e{err:.3f}"
            cell = {"error": err}
            lat_a, skip_a = self.run_teacache(
                prompt, seed, steps, w, h, cfg_dict)
            if lat_a_first is None:
                lat_a_first = lat_a
            lat_b, skip_b = self.run_teacache(
                prompt, seed, steps, w, h, cfg_dict,
                corrector_path=corrector_path)
            cell["skip_rate_pct"] = round(
                100 * sum(1 for e in skip_a if e["skipped"]) /
                max(len(skip_a), 1), 1)

            # harness: wiring check — B′ output must differ from A. This is
            # NOT a dead-corrector test once gain calibration is embedded
            # (the gains rescale the latent even for a zero-output corrector);
            # that is what `out["corrector_alive"]` and the zero-init sanity
            # are for.
            div_ab = float((lat_a - lat_b).abs().max())
            cell["harness"] = {
                "max_latent_diff_A_vs_B": round(div_ab, 6),
                "ok": div_ab > 1e-4,
            }

            cell["deployed_weighted"] = self.deployed_weighted(
                skip_a, rec, steps)

            # pixel level
            if self.metrics is not None:
                img_base = self._decode(lat_base)
                img_a = self._decode(lat_a)
                img_b = self._decode(lat_b)
                m_a = self.image_metrics(img_base, img_a)
                m_b = self.image_metrics(img_base, img_b)
                cell["image_metrics"] = {"A_vs_base": m_a,
                                         "B_vs_base": m_b}
                regress = []
                for n in self.lower_better:
                    if m_b[n] > m_a[n]:
                        regress.append(n)
                for n in ("psnr", "ssim", "fsim", "vif", "ms_ssim"):
                    if m_b[n] < m_a[n]:
                        regress.append(n)
                cell["no_regression"] = {
                    "regressed_vs_A": regress,
                    "ok": not regress,
                }
                # final-latent spectral test
                cell["spectral_final_latent"] = {
                    "hf_signal": round(high_freq_fraction(
                        lat_base - lat_base.mean()), 4),
                    "hf_errA": round(high_freq_fraction(
                        (lat_a - lat_base).unsqueeze(0)), 4),
                    "hf_errB": round(high_freq_fraction(
                        (lat_b - lat_base).unsqueeze(0)), 4),
                }
                if self.args.perceptual_channel_ablation:
                    cell["per_channel_lpips"] = self.per_channel_ablation(
                        lat_base, lat_b)
                if png is not None:
                    self._save_png(png, name, cname,
                                   img_base, img_a, img_b)

            # trust-map variant (only when explicitly requested)
            if self.args.trust_map:
                lat_t, skip_t = self.run_teacache(
                    prompt, seed, steps, w, h, cfg_dict,
                    corrector_path=corrector_path,
                    trust_map=self.args.trust_map)
                cell["trust_map"] = {
                    "map": self.args.trust_map,
                    "max_latent_diff_vs_B": round(
                        float((lat_t - lat_b).abs().max()), 6),
                }
                if self.metrics is not None:
                    img_t = self._decode(lat_t)
                    cell["trust_map"]["image_metrics"] = \
                        self.image_metrics(img_base, img_t)

            cfgs.append(cell)

        # zero-init sanity (first config only): fresh corrector ≡ Mode A of
        # the SAME control point — comparing against a different control
        # point's A latent would fail the gate spuriously (different config →
        # different skip schedule → different latent).
        if self.args.sanity_zero_init and self.control_points \
                and lat_a_first is not None:
            err, cfg_dict = self.control_points[0]
            lat_z, _ = self.run_teacache(
                prompt, seed, steps, w, h, cfg_dict,
                corrector_path=corrector_path, zero_init=True)
            div = float((lat_z - lat_a_first).abs().max())
            out["zero_init_sanity"] = {
                "max_latent_diff_zeroinit_vs_A": round(div, 6),
                "ok": div < 1e-3,
            }

        out["configs"] = cfgs
        out["wall_s"] = round(time.time() - t0, 1)
        return out

    def _save_png(self, png_dir: Path, name: str, cname: str,
                  img_base, img_a, img_b):
        from PIL import Image, ImageDraw
        png_dir.mkdir(parents=True, exist_ok=True)

        def to_pil(x):
            a = ((to01(x).squeeze(0).permute(1, 2, 0).clamp(0, 1)) * 255) \
                .byte().numpy()
            return Image.fromarray(a)

        imgs = [to_pil(img_base), to_pil(img_a), to_pil(img_b)]
        w, h = imgs[0].size
        c = min(w, h, 420)
        box = ((w - c) // 2, (h - c) // 2, (w + c) // 2, (h + c) // 2)
        imgs = [im.crop(box) for im in imgs]
        canvas = Image.new("RGB", (c * 3 + 40, c + 30), "white")
        d = ImageDraw.Draw(canvas)
        for i, (im, lb) in enumerate(zip(imgs,
                                         ("baseline", "Mode A", "Mode B"))):
            canvas.paste(im, (i * (c + 20), 30))
            d.text((i * (c + 20) + 10, 10), lb, fill="black")
        canvas.save(png_dir / f"{name}_{cname}.png")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Refiner verification (validate.py is the base TeaCache "
                    "verifier; this is the corrector side of the split)")
    p.add_argument("--comfy-dir", required=True,
                   help="ComfyUI root (fresh checkout with qwen3 support; "
                        "models/ must resolve to your model files)")
    p.add_argument("--corrector", required=True,
                   help="Corrector .safetensors to verify")
    p.add_argument("--model-name", default="anima-base-v1.0.safetensors")
    p.add_argument("--clip-name", default="qwen_3_06b_base.safetensors")
    p.add_argument("--clip-type", default="qwen_image")
    p.add_argument("--vae-name", default="qwen_image_vae.safetensors")
    p.add_argument("--out-dir", default="refiner_verify",
                   help="Output dir (report.json + refiner_data recording)")
    p.add_argument("--config-errors", default="0.020,0.054",
                   help="Comma list of preset control-point errors to test")
    p.add_argument("--prompts", default=None,
                   help="JSON list of prompts; default = 2 in-dist + 5 OOD "
                        "built-ins")
    p.add_argument("--in-dist-only", action="store_true",
                   help="Skip the OOD prompt set")
    p.add_argument("--seed", type=int, default=1,
                   help="Sampling seed for all prompts")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--res", default="512x512")
    p.add_argument("--cfg", type=float, default=5.5)
    p.add_argument("--sampler", default="er_sde")
    p.add_argument("--scheduler", default="normal")
    p.add_argument("--record-lags", default=None,
                   help="Comma list of lags to record; default = the "
                        "checkpoint's trained ladder (from its "
                        "config_snapshot), else 1,2,4,8,16")
    p.add_argument("--trust-map", default=None,
                   help="Per-(area,region) trust multipliers as JSON, e.g. "
                        '{"64x64:late": 0.25}')
    p.add_argument("--sanity-zero-init", action="store_true",
                   help="Run a fresh zero-init corrector B′ and assert ≡ Mode A")
    p.add_argument("--perceptual-channel-ablation", action="store_true",
                   help="Per-channel LPIPS damage (16 decodes per run)")
    p.add_argument("--no-image-metrics", action="store_true",
                   help="Skip pyiqa decode metrics (velocity-level only)")
    p.add_argument("--png", default=None,
                   help="Optional dir for baseline/A/B comparison images")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    verifier = RefinerVerifier(args)
    w, h = (int(x) for x in args.res.lower().split("x"))
    record_lags = ([int(x) for x in args.record_lags.split(",")]
                   if args.record_lags else
                   (verifier.trained_lags or [1, 2, 4, 8, 16]))
    if verifier.trained_lags and record_lags != verifier.trained_lags:
        print(f"  ⚠ recording ladder {record_lags} ≠ trained ladder "
              f"{verifier.trained_lags} — deployed lags between recorded "
              f"entries are resolved from the next recorded lag")
    if args.prompts:
        prompts = json.loads(args.prompts)
    else:
        prompts = list(IN_DIST_PROMPTS)
        if not args.in_dist_only:
            prompts += OOD_PROMPTS
    png_dir = Path(args.png) if args.png else None
    report = {
        "checkpoint": args.corrector,
        "trained_lags": verifier.trained_lags,
        "control_points": [e for e, _ in verifier.control_points],
        "sanity_zero_init": args.sanity_zero_init,
        "prompts": {},
    }
    for pid, prompt in enumerate(prompts):
        print(f"\n===== prompt {pid}: {prompt[:60]}...")
        try:
            report["prompts"][f"p{pid}"] = verifier.verify_prompt(
                prompt, args.seed, args.steps, w, h, pid,
                args.corrector, record_lags, png_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            report["prompts"][f"p{pid}"] = {"error": str(e)}
    out_path = Path(args.out_dir) / "report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out_path, "w"), indent=1, default=str)
    print(f"\nreport: {out_path}")

    # ── summary table ─────────────────────────────────────────────────
    print("\n── refiner verification summary ──────────────────────────")
    print(f"checkpoint: {args.corrector}  trained_lags: {verifier.trained_lags}")
    for pk, pr in report["prompts"].items():
        if "error" in pr:
            print(f"{pk}: ERROR {pr['error']}")
            continue
        vl = pr.get("velocity_level", {})
        print(f"{pk}: pooled K0={vl.get('pooled', {}).get('K0')} "
              f"K1={vl.get('pooled', {}).get('K1')} "
              f"recovery={vl.get('pooled_recovery_pct')}% "
              f"corrector_alive={pr.get('corrector_alive')}")
        rc = vl.get("per_channel", {}).get("recovery_uniformity", {})
        if rc:
            print(f"     channel recovery: mean={rc['mean']:.3f} "
                  f"std={rc['std']:.3f} min={rc['min']:.3f} "
                  f"below50={rc['channels_below_50pct']}")
        for cell in pr.get("configs", []):
            dw = cell.get("deployed_weighted") or {}
            nr = cell.get("no_regression") or {}
            print(f"     e={cell['error']:.3f} skip={cell.get('skip_rate_pct')}% "
                  f"deployed-recovery={dw.get('recovery_pct')}% "
                  f"(A={dw.get('modeA_per_skip')} B={dw.get('modeB_per_skip')}) "
                  f"maxlag={dw.get('max_deployed_lag')} "
                  f"regress={nr.get('regressed_vs_A')} "
                  f"harness={cell.get('harness', {}).get('ok')}")


if __name__ == "__main__":
    main()
