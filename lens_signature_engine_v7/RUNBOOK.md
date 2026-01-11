# Runbook (v7)

This runbook describes the end-to-end flow for STD registration, validation,
activation, and inspection using the v7 signature engine.

## 1) STD registration (LOW/MID/HIGH)
Register STD images into the model registry.

```bash
python scripts/register_std.py --sku SKU_TEMP --ink INK1 --mode LOW \
  --stds C:\X\Color_total\Color_meter\data\samples\INK1\A1.png \
  --cfg configs/default.json --models_root models --expected_ink_count 2

python scripts/register_std.py --sku SKU_TEMP --ink INK1 --mode MID \
  --stds C:\X\Color_total\Color_meter\data\samples\INK1\A2.png \
  --cfg configs/default.json --models_root models --expected_ink_count 2

python scripts/register_std.py --sku SKU_TEMP --ink INK1 --mode HIGH \
  --stds C:\X\Color_total\Color_meter\data\samples\INK1\A3.png \
  --cfg configs/default.json --models_root models --expected_ink_count 2
```

Notes:
- LOW/MID/HIGH labels are user-defined groups. They do not have to be ordered.
- Registration results in INCOMPLETE; activation is a separate step.
- expected_ink_count is stored in registry (optional).

## 2) STD registration validation (phase: STD_REGISTRATION)
Validate STD sets without producing NG/OK labels.

```bash
python scripts/run_signature_engine.py \
  --sku SKU_TEMP --ink INK1 \
  --models_root models \
  --tests C:\X\Color_total\Color_meter\data\samples\INK1\A1.png C:\X\Color_total\Color_meter\data\samples\INK1\A2.png C:\X\Color_total\Color_meter\data\samples\INK1\A3.png \
  --phase STD_REGISTRATION \
  --cfg configs/default.json \
  --out results\std_reg_ink1.json
```

Expected labels:
- `STD_ACCEPTABLE`
- `STD_RETAKE` (gate fail / separation low / within-mode unstable)
- `STD_UNSTABLE` (reserved for strict order checks when enabled)

Key fields:
- `decision.registration_summary`: separation + stability checks
- `decision.debug.signature_mode_scores`: diagnostic only (if enabled)

## 3) Activation (ACTIVE)
Activate only after STD_REGISTRATION passes.

Example API call:
```
POST /models/activate
{
  "sku": "SKU_TEMP",
  "ink": "INK1",
  "low_version": "v20251225_095913",
  "mid_version": "v20251225_095913",
  "high_version": "v20251225_095913",
  "approved_by": "operator_A",
  "approved_at": "2025-12-25T09:59:00+09:00",
  "reason": "temporary standard approved"
}
```

Policy:
- Activation is allowed only when STD_REGISTRATION == STD_ACCEPTABLE.
- Active pointer must be unique per SKU+INK (single active set).
- Rollback restores the previous active pointer.
- Activation writes a pattern baseline for NG_PATTERN (relative) checks.
- approved_by/reason fields are optional in both UI and API (defaults to "SYSTEM").
- Delete is admin-only (soft delete) for test cleanup; ACTIVE is cleared but files are preserved.

## 4) Inspection (phase: INSPECTION)
Run inspection for production samples.

```bash
python scripts/run_signature_engine.py \
  --sku SKU_TEMP --ink INK1 \
  --models_root models \
  --tests C:\X\Color_total\Color_meter\data\samples\INK1\A1.png \
  --cfg configs/default.json \
  --out results\inspect_ink1.json \
  --expected_ink_count 2
```

Expected labels:
- `OK`
- `RETAKE` (gate failure)
- `NG_PATTERN`
- `NG_COLOR`

Key fields:
- `decision.best_mode`: LOW/MID/HIGH
- `decision.mode_scores`: per-mode signature results
- `decision.anomaly.type`: defect explanation when NG_PATTERN

Notes:
- INSPECTION uses the same signature basis as STD, but gate/anomaly thresholds are absolute.
  Even STD images can return `RETAKE`/`NG_PATTERN` if capture quality or anomaly thresholds are violated.
 - When pattern baseline is required and missing, INSPECTION returns `RETAKE` with `PATTERN_BASELINE_NOT_FOUND`.
- v1.2-shadow: OK results append pattern features to `models\pattern_baselines\ok_logs\<SKU>\<INK>\OKF_<baseline>.jsonl`.
- v2.0-shadow: INSPECTION can emit `decision.diagnostics.v2_diagnostics` when expected_ink_count is provided.

## 5) Reasons (standardized)
Model/registry:
- `MODEL_NOT_FOUND`
- `MODEL_INCOMPLETE:<MODE>`
- `MODEL_LOAD_FAILED:<MODE>`
- `CFG_MISMATCH:<MODE>`

Gate:
- `CENTER_NOT_IN_FRAME`
- `BLUR_LOW`
- `ILLUMINATION_UNEVEN`

Signature:
- `SIGNATURE_CORR_LOW`
- `DELTAE_P95_HIGH`
- `DELTAE_MEAN_HIGH`
- `BAND_VIOLATION_HIGH`

Color direction (v1):
- `COLOR_SHIFT_YELLOW` / `COLOR_SHIFT_BLUE`
- `COLOR_SHIFT_RED` / `COLOR_SHIFT_GREEN`
- `COLOR_SHIFT_DARK` / `COLOR_SHIFT_LIGHT`

Pattern (v1):
- `PATTERN_DOT_COVERAGE_HIGH`
- `PATTERN_DOT_COVERAGE_LOW`
- `PATTERN_EDGE_BLUR`
- `PATTERN_DOT_SPREAD`

Registration:
- `MODE_SEPARATION_LOW`
- `MODE_VARIANCE_HIGH:<MODE>`

## 6) Outputs
- Results JSON: `results\*.json`
- Registry index: `models\index.json`
- Model meta: `models\<SKU>\<INK>\<MODE>\<version>\meta.json`

## 7) Config changes
Any config change affects `cfg_hash` and requires STD re-registration.

## 8) Planned roadmap (post v7 stabilization)
This section records future work items focused on inspection quality and operator experience.

Priority A: decision quality tuning (accuracy-first)
- Use RETAKE reason_codes Top 3 to drive improvements (capture guidance, mask buffer, etc.).
- Reclassify STD_REGISTRATION edge cases based on accumulated data (warnings vs needs review).
- Run geom_consistency as warning-only for 2 weeks, then decide hard trigger.

## 9) Color scope note (v1 vs v2)
Current v7 inspection uses a single overall color signature (no per-ink separation).
- STD_REGISTRATION does not output NG_COLOR; it only validates capture quality and stability.
- INSPECTION compares overall radial signature and can report overall ΔL/Δa/Δb direction.

If per-ink separation is added later (v2), STD models must also be re-built with per-ink references.

## 10) v1.1 policy lock (production-ready)
v1.1 enforces relative pattern checks against ACTIVE baseline.

- NG_PATTERN triggers only when pattern features exceed baseline band.
- Normal gradient/rotation variation is not treated as NG_PATTERN.
- INSPECTION records pattern baseline references for audit/repro.
