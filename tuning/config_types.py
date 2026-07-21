"""Dataclasses for TeaCache configuration and calibration data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional
from pathlib import Path


@dataclass
class TeacacheConfig:
    """All 10 tuning knobs for the TeaCache forward function.

    This is the complete parameterization of the cache decision loop.
    When shipped to users, most fields are baked into presets; the
    Advanced node exposes them all as toggles/sliders.
    """

    # Knob 1: Signal source
    source: str = "first_block_shift"

    # Knob 2: Distance metric
    metric_type: str = "mean_only"
    metric_weights: Dict[str, float] = field(default_factory=dict)

    # Knob 3: Signal scaling (for sources with tiny deltas)
    signal_scale: float = 1.0

    # Knob 4: Mapping function
    mapping_type: str = "polynomial"
    coefficients: List[float] = field(default_factory=list)
    mapping_params: Dict[str, float] = field(default_factory=dict)

    # Knob 5: Accumulation
    accumulation_type: str = "hard_reset"
    accumulation_params: Dict[str, float] = field(default_factory=dict)

    # Knob 6: Threshold
    rel_l1_thresh: float = 0.07

    # Knob 7: Step schedule
    step_schedule: str = "constant"
    start_percent: float = 0.05
    end_percent: float = 0.95

    # Knob 8: Block skipping
    block_mode: str = "all_or_nothing"
    block_params: Dict = field(default_factory=dict)
    cosim_threshold: float = 0.95  # threshold for split_groups partition + dynamic mode
    block_level: str = "unified"   # "unified" | "per_group" (dynamic mode only)
    block_level_config_scope: List[str] = field(default_factory=lambda: ["*"])  # params varied at group level

    # Knob 9: Residual strategy
    residual_strategy: str = "hard"
    residual_params: Dict[str, float] = field(default_factory=dict)

    # Knob 10: Cross-feed
    cross_feed_enabled: bool = False
    cross_feed_strength: float = 0.5

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "metric_type": self.metric_type,
            "metric_weights": self.metric_weights,
            "signal_scale": self.signal_scale,
            "mapping_type": self.mapping_type,
            "coefficients": self.coefficients,
            "mapping_params": self.mapping_params,
            "accumulation_type": self.accumulation_type,
            "accumulation_params": self.accumulation_params,
            "rel_l1_thresh": self.rel_l1_thresh,
            "step_schedule": self.step_schedule,
            "start_percent": self.start_percent,
            "end_percent": self.end_percent,
            "block_mode": self.block_mode,
            "block_params": self.block_params,
            "residual_strategy": self.residual_strategy,
            "residual_params": self.residual_params,
            "cross_feed_enabled": self.cross_feed_enabled,
            "cross_feed_strength": self.cross_feed_strength,
            "cosim_threshold": self.cosim_threshold,
            "block_level": self.block_level,
            "block_level_config_scope": self.block_level_config_scope,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TeacacheConfig":
        return cls(
            source=d.get("source", "first_block_shift"),
            metric_type=d.get("metric_type", "mean_only"),
            metric_weights=d.get("metric_weights", {}),
            signal_scale=d.get("signal_scale", 1.0),
            mapping_type=d.get("mapping_type", "polynomial"),
            coefficients=d.get("coefficients", []),
            mapping_params=d.get("mapping_params", {}),
            accumulation_type=d.get("accumulation_type", "hard_reset"),
            accumulation_params=d.get("accumulation_params", {}),
            rel_l1_thresh=d.get("rel_l1_thresh", 0.07),
            step_schedule=d.get("step_schedule", "constant"),
            start_percent=d.get("start_percent", 0.05),
            end_percent=d.get("end_percent", 0.95),
            block_mode=d.get("block_mode", "all_or_nothing"),
            block_params=d.get("block_params", {}),
            residual_strategy=d.get("residual_strategy", "hard"),
            residual_params=d.get("residual_params", {}),
            cross_feed_enabled=d.get("cross_feed_enabled", False),
            cross_feed_strength=d.get("cross_feed_strength", 0.5),
            cosim_threshold=d.get("cosim_threshold", 0.95),
            block_level=d.get("block_level", "unified"),
            block_level_config_scope=d.get("block_level_config_scope", ["*"]),
        )

    @classmethod
    def from_transformer_options(cls, to: dict) -> "TeacacheConfig":
        """Read tea-related keys from transformer_options dict."""
        d = {}
        for key in [
            "source", "metric_type", "metric_weights", "signal_scale",
            "mapping_type", "coefficients", "mapping_params",
            "accumulation_type", "accumulation_params",
            "rel_l1_thresh", "step_schedule",
            "start_percent", "end_percent",
            "block_mode", "block_params",
            "residual_strategy", "residual_params",
            "cross_feed_enabled", "cross_feed_strength",
            "cosim_threshold", "block_level", "block_level_config_scope",
        ]:
            if f"tc_{key}" in to:
                d[key] = to[f"tc_{key}"]
            elif key in to:
                d[key] = to[key]
        return cls.from_dict(d)

    def inject_into_transformer_options(self, to: dict) -> None:
        """Write config into transformer_options dict for the forward function."""
        for key, val in self.to_dict().items():
            to[f"tc_{key}"] = val
        to["rel_l1_thresh"] = self.rel_l1_thresh
        to["coefficients"] = self.coefficients


@dataclass
class DeltaStats:
    """Summary statistics of a delta tensor for one source at one step."""
    mean: float = 0.0
    max: float = 0.0
    std: float = 0.0
    p95: float = 0.0
    median: float = 0.0
    min: float = 0.0
    denom: float = 1.0


@dataclass
class CalibrationEntry:
    """One recorded step during calibration, with stats for all sources."""
    step: int
    step_fraction: float
    prompt_id: int
    seed: int
    cond: int
    total_steps: int

    # Delta stats per source
    t_emb: Optional[DeltaStats] = None
    shift: Optional[DeltaStats] = None
    latent: Optional[DeltaStats] = None

    # Ground truth output change
    out_rel: float = 0.0
    out_rel_max: float = 0.0
    out_rel_std: float = 0.0
    res_rel: float = 0.0

    # Per-block cosine similarity (block_idx -> cos_sim) at this step.
    # Populated when track_per_block is enabled during calibration.
    block_cos_sims: Optional[Dict[int, float]] = None

    # Sampler/scheduler identity (for sampler-specific tuning)
    sampler: str = ""
    scheduler: str = ""

    def to_dict(self) -> dict:
        d = {
            "step": self.step, "step_fraction": self.step_fraction,
            "prompt_id": self.prompt_id, "seed": self.seed,
            "cond": self.cond, "total_steps": self.total_steps,
            "sampler": self.sampler, "scheduler": self.scheduler,
            "out_rel": self.out_rel,
            "out_rel_max": self.out_rel_max,
            "out_rel_std": self.out_rel_std,
            "res_rel": self.res_rel,
        }
        if self.block_cos_sims is not None:
            d["block_cos_sims"] = {str(k): v for k, v in self.block_cos_sims.items()}
        for src_name in ["t_emb", "shift", "latent"]:
            stats = getattr(self, src_name)
            if stats is not None:
                d[src_name] = {
                    "mean": stats.mean, "max": stats.max,
                    "std": stats.std, "p95": stats.p95,
                    "median": stats.median, "min": stats.min,
                    "denom": stats.denom,
                }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationEntry":
        def _parse_stats(key: str) -> Optional[DeltaStats]:
            if key not in d:
                return None
            s = d[key]
            return DeltaStats(
                mean=s["mean"], max=s["max"], std=s["std"],
                p95=s["p95"], median=s["median"], min=s["min"],
                denom=s["denom"],
            )
        def _parse_block_cos_sims(raw) -> Optional[Dict[int, float]]:
            if raw is None or not isinstance(raw, dict):
                return None
            return {int(k): float(v) for k, v in raw.items()}
        return cls(
            step=d["step"], step_fraction=d["step_fraction"],
            prompt_id=d["prompt_id"], seed=d["seed"],
            cond=d["cond"], total_steps=d["total_steps"],
            sampler=d.get("sampler", ""), scheduler=d.get("scheduler", ""),
            t_emb=_parse_stats("t_emb"),
            shift=_parse_stats("shift"),
            latent=_parse_stats("latent"),
            out_rel=d.get("out_rel", 0.0),
            out_rel_max=d.get("out_rel_max", 0.0),
            out_rel_std=d.get("out_rel_std", 0.0),
            res_rel=d.get("res_rel", 0.0),
            block_cos_sims=_parse_block_cos_sims(d.get("block_cos_sims")),
        )


@dataclass
class OptimizationResult:
    """One configuration's simulated performance."""
    config: TeacacheConfig
    skip_rate: float
    estimated_speedup: float
    accumulated_error: float
    score: float  # combined quality × speedup

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "skip_rate": self.skip_rate,
            "estimated_speedup": self.estimated_speedup,
            "accumulated_error": self.accumulated_error,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OptimizationResult":
        return cls(
            config=TeacacheConfig.from_dict(d["config"]),
            skip_rate=d["skip_rate"],
            estimated_speedup=d["estimated_speedup"],
            accumulated_error=d["accumulated_error"],
            score=d["score"],
        )


@dataclass
class ValidationResult:
    """End-to-end measurement of one configuration."""
    config: TeacacheConfig
    mean_speedup: float
    mean_psnr: float
    mean_ssim: float
    mean_lpips: float
    mean_time_sec: float
    n_samples: int
    mean_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "config": self.config.to_dict(),
            "mean_speedup": self.mean_speedup,
            "mean_psnr": self.mean_psnr,
            "mean_ssim": self.mean_ssim,
            "mean_lpips": self.mean_lpips,
            "mean_time_sec": self.mean_time_sec,
            "n_samples": self.n_samples,
        }
        if self.mean_metrics:
            d["mean_metrics"] = self.mean_metrics
        return d


@dataclass
class TuningConfig:
    """Top-level configuration loaded from config.json."""
    comfy_dir: str
    model_name: str
    clip_name: str
    clip_type: str
    vae_name: str
    sampling: dict
    calibration: dict
    optimization: dict
    validation: dict
    output_dir: str

    @classmethod
    def load(cls, path: str) -> "TuningConfig":
        with open(path) as f:
            d = json.load(f)
        return cls(
            comfy_dir=d["comfy_dir"],
            model_name=d["model_name"],
            clip_name=d["clip_name"],
            clip_type=d["clip_type"],
            vae_name=d["vae_name"],
            sampling=d["sampling"],
            calibration=d["calibration"],
            optimization=d["optimization"],
            validation=d["validation"],
            output_dir=d.get("output_dir", "outputs"),
        )
