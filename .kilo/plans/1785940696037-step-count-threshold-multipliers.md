# Measured Step-Count Threshold Multipliers (`step_count_variants` + auto midpoints)

## Problem

At runtime, `tuning/forward.py` scales the TeaCache threshold by the unvalidated linear rule `step_scale = preset_steps / n_steps` (30/16 = 1.875×). This inflates the threshold beyond anything the tuning sim ever measured, so at 16 steps even quality 0 skips ~70% of steps. Replace the linear rule with **measured** multipliers: per-config threshold ratios `T_N / T_base` simulated at a configurable set of step counts, with midpoints auto-inserted between adjacent variants (mirroring the existing `low_mid`/`high_mid` machinery).

## Design decisions

- **Config sections stay pure**: `sampling.step_variants` is the ONLY thing that drives calibration runs (e.g. `[15, 30, 45]` — low/base/high). `optimization.step_count_variants` (`[3, 15, 45, 60]`) is a pure **output spec**: the curve points the final presets contain, estimated from whatever data calibration collected. Optimization never feeds back into calibration.
- **Derived anchor list**: `variants ∪ midpoints ∪ {base}`, sorted unique, where midpoint of adjacent pair (a, b) is `(a + b) // 2` (skip if no integer between), and `base` = `sampling.default_steps` (30) is always forced in. E.g. `[3, 15, 45, 60]` → `[3, 9, 15, 30, 45, 52, 60]`. Helper `derive_step_anchors(variants, base)` in `tuning/utils.py`.
- **Measurement** (per Pareto entry / config, in `optimize.py`, which already has `sim_data` in process):
  1. `base_curve = _sweep_threshold_curve(sim_data.filter_by_step_count(base), cfg, sweep_values, scoring)`; `E = entry.accumulated_error`; `T_base = _find_closest_on_curve(base_curve, E)[0]`.
  2. For each anchor N ≠ base:
     - **Direct** if calibration data exists at exactly N: `sub = sim_data.filter_by_step_count(N)`; `curve = _sweep_threshold_curve(sub, cfg, sweep_values, scoring)`; `ratio_N = _find_closest_on_curve(curve, E)[0] / T_base`.
     - **Estimated** otherwise: `sub = sim_data.resample_to_step_count(N)` — recorded groups (S > N) are resampled to N steps by keeping the entries nearest the N-step fractions j/(N−1) and scaling per-step deltas / output changes by `(S−1)/(N−1)` (first-order exact for a smooth trajectory; the same assumption the mapping fits rest on). Sweep + ratio as above. If no group has S ≥ N (anchor beyond the largest recorded count), omit the anchor.
  3. Store `step_mults = {"base": base, "points": [[N, ratio_N], ...]}` on the entry (always includes `[base, 1.0]`).
- **Same-error anchoring**: ratios match the control point's anchored accumulated error, NOT skip rate (skip rate doesn't track quality across step counts). The schedule/source/mapping/accumulation are baked into the measurement automatically.
- **Storage**: per-Pareto-entry `step_mults` serialized into `pareto_frontier.json`; `build_presets.py` copies it onto each control point via the existing `_pareto_idx` link (same pattern as `_add_midpoints`; `_add_step_mults` reads the idx before `_add_midpoints` pops it). Control point gains `"step_mults": {"base": 30, "points": [[3, 1.35], ...]}`.
- **Runtime**: piecewise-linear interpolation over sorted `points`, **clamped at endpoints** (never extrapolate — extrapolation downward caused the 1.875× bug). Injected as 0-d tensor `tc_step_mult` (existing "Fix 2" dynamo pattern), replacing the `preset_steps / n_steps` formula. All 4 multiplication sites (forward.py) stay unchanged.
- **Fallbacks** (no table → multiplier 1.0, no compensation): legacy-format presets, `TeaCacheAnimaAdvanced`, control points whose Pareto entry lacks `step_mults`, and anchors with missing calibration data (anchor omitted; interp over remaining).

## Tasks (ordered)

1. **`tuning/config.json`** — `optimization.step_count_variants: [3, 15, 45, 60]` (output spec only). `sampling.step_variants` trimmed to `[15, 30, 45]` with weights `[0.15, 0.70, 0.15]` so the low/high anchors (15/45) are directly measured while the bulk of data stays at base 30.
2. **`tuning/utils.py`** — `derive_step_anchors(variants: list[int], base: int) -> list[int]`: sort, dedupe, force-include `base`, insert `(a+b)//2` midpoint for each adjacent pair.
3. **`tuning/sim_data.py`** — `total_steps: int` field on `GroupData` (populated from the bucket key); `SimData.filter_by_step_count(n)` (filter on the new field, not `n_steps`, to avoid off-by-one); `SimData.resample_to_step_count(n)` + `_resample_group` helper (entry selection by nearest fraction, `(S−1)/(N−1)` amplitude scaling of deltas/out_rel, fraction re-tag, schedule-mult recompute, cos_sim column selection; groups at exactly N pass through; empty when n < 2 or n exceeds all recorded counts).
4. **`tuning/config_types.py`** — `step_mults: Optional[dict] = None` on `OptimizationResult` + `to_dict`/`from_dict` serialization (JSON-safe: `points` as list-of-lists). `step_mults: Optional[dict] = None` field on `TeacacheConfig`, excluded from `to_dict`/`from_dict`/`inject_into_transformer_options` (metadata carried on the Python object only, never in the slim preset config).
5. **`tuning/optimize.py`** — after the Pareto sweep phase (main function, where `sim_data` is in scope): for each Pareto entry, compute `step_mults` per the measurement procedure above (direct-or-resample, omit anchors beyond the largest recorded count). Reuse `_sweep_threshold_curve` as-is and `_find_closest_on_curve` from `build_presets.py`. Log measured vs estimated anchors.
6. **`tuning/build_presets.py`** — `_add_step_mults(control_points, pareto)`, called alongside `_add_midpoints`: copy each Pareto entry's `step_mults` onto its control points via `_pareto_idx` (read, not popped — `_add_midpoints` pops it afterwards); entries without data → `"step_mults": None`. Honor the existing `--steps` arg: `_steps` must equal the measured `base`; store `base` from the table rather than assuming 30.
7. **`tuning/forward.py`** — `step_count_multiplier(n_steps: int, mults: Optional[dict]) -> float`: sorted piecewise lerp over `points`, clamp at endpoints, `1.0` when `mults` is None or has <2 points. Replace the `preset_steps / n_steps` ratio: `step_scale = transformer_options.get("tc_step_mult", torch.tensor(1.0)).item()`.
8. **`nodes_anima.py`** — `_quality_to_config`: attach the snapped base config's `step_mults` to the returned `cfg` (in the two-phase path, `base` is already lo/hi; in the fallback path, the same base used for the config). `_apply_teacache` `unet_wrapper`: where `tc_threshold_mult` is computed, also set `c_to["tc_step_mult"] = torch.tensor(step_count_multiplier(n_steps, cfg.step_mults))` with `n_steps = max(len(sigmas) - 1, 1)`. Advanced node passes `None` → 1.0.
9. **`tuning/calibrate.py`** — unchanged from original behavior: runs exactly `sampling.step_variants` / `step_weights`. (The earlier union-with-anchors extension was reverted — the optimizer now estimates anchors from the limited data instead.)
10. **Validation** — see below.

## Validation

- Unit checks: `derive_step_anchors([3, 15, 45, 60], 30) == [3, 9, 15, 30, 45, 52, 60]`; `step_count_multiplier` returns 1.0 at base, lerps between anchors, clamps below min / above max, 1.0 on None; `resample_to_step_count` scales deltas/out_rel by `(S−1)/(N−1)`, re-tags fractions, keeps cos_sim columns, empty for n > max recorded.
- Rebuild presets from an existing `pareto_frontier.json` (which lacks `step_mults`): control points carry `None` → runtime behavior identical to the step_scale=1.0 path — no regression.
- With fresh calibration data at `[15, 30, 45]`: rebuild presets, verify every control point has a table with `[base, 1.0]`, direct ratios at 15/30/45 (sane: 15-step ratio 1.0–1.5, 45-step ratio ~0.4–1.0), estimated ratios at 3/9 (log shows "estimated"), and no points above 45 (runtime clamps to the 45-step ratio at 60).
- `tuning/validate.py` at steps 15/30/45, quality 0/50: skip rate at quality 0 should drop from ~70% (16 steps) to near the 30-step calibrated ~15%; no NaN / no threshold explosion.
- `python -m tuning.smoke_test` passes.

## Risks / notes

- **Estimation approximation**: anchors without direct calibration data are estimated by trajectory resampling — first-order exact under the smooth-trajectory assumption. Accuracy degrades with the amplitude factor `k = (S−1)/(N−1)` (e.g. a 3-step estimate from 15-step data amplifies noise ~7×), so keep `sampling.step_variants` reasonably close to the desired table range. For exact ratios at a specific step count, add it to `sampling.step_variants`.
- **Range clamping**: anchors beyond the largest recorded step count are omitted; the runtime clamps to the top table point. E.g. with `step_variants [15, 30, 45]` and `step_count_variants [3, 15, 45, 60]`, a 60-step run uses the 45-step ratio.
- **`base` consistency**: `_steps` in the presets (set by `build_presets --steps`, overridden by the table's `base`) must equal `sampling.default_steps` used for measurement; verify and document.
- **Pre-existing sim/runtime mismatch** (out of scope): the sim never applies `start_percent`/`end_percent` masking while the runtime does. Ratios are measured on the same unmasked basis as the existing error anchoring, so they stay consistent with the anchors; a global masking fix would change the anchoring itself and is deliberately not part of this plan.
- **Knob overrides**: if a user overrides `step_schedule`/`accumulation_type` in the node, the stored ratios (measured with the preset's knobs) become approximate — accepted.
- **Data dependency**: shipping an updated `anima_presets.json` with real ratios requires re-running `calibrate.py` (model generations — user's data, not code). Until then the feature is code-complete with graceful 1.0 fallback.
- **Known pre-existing bug (not part of this plan)**: `optimize.py`'s `_signal_signature` includes `mapping_params`, which power_law/softplus fitting mutates after dedup → `KeyError` at the replication step. Blocks full `optimize` runs with the default `mapping_types`. Fix deferred until tested.
- **Section purity**: `step_count_variants` lives in `optimization` and is consumed only by `optimize.py`; `calibrate.py` and `build_presets.py` never read it (tables arrive via the pareto JSON).
