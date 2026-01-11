# Inspection Flow (v7)

This document summarizes the inspection pipeline and output schema used by
`run_signature_engine.py` in `INSPECTION` mode. It is intended for
integration and report formatting.

## Pipeline flow
1) Load STD models (registry or direct paths)
   - If models are missing or cfg hash mismatches, return RETAKE with reasons.
2) Gate (capture validity)
   - center offset, blur, illumination symmetry.
   - If failed and `gate.diagnostic_on_fail=false`, skip signature/anomaly.
3) Color signature (best mode)
   - Compute radial signature once.
   - Evaluate LOW/MID/HIGH; pick best_mode by fail_ratio, dE_p95, corr.
4) Pattern/anomaly (relative baseline)
   - Pattern features are compared against ACTIVE baseline.
   - NG_PATTERN triggers only when features exceed baseline band.
   - If NG_PATTERN, attach defect type (RING/SECTOR/BLOB/UNIFORMITY).
5) Decision
   - RETAKE / NG_PATTERN / NG_COLOR / OK.

## Output schema (top-level)
```
{
  "cfg": { ... },                 // runtime config snapshot
  "results": [ ... ],             // list of per-image decisions
  "multi_mode": true,
  "sku": "SKU001",
  "phase": "INSPECTION"
}
```

## Per-image decision schema (key fields)
```
{
  "path": "path/to/image.png",
  "decision": {
    "label": "OK | RETAKE | NG_COLOR | NG_PATTERN",
    "reasons": [ ... ],
    "reason_codes": [ ... ],
    "reason_messages": [ ... ],
    "phase": "INSPECTION",
    "gate": {
      "passed": true,
      "reasons": [ ... ],
      "scores": {
        "center_offset_ratio": 0.05,
        "blur_var": 1200.0,
        "illum_sym": 0.03
      }
    },
    "signature": { ... },         // best_mode signature result
    "anomaly": { ... },           // pattern/anomaly result
    "diagnostics": { ... },       // direction + dot metrics (v1)
    "diagnostics.v2_diagnostics": { ... }, // shadow ink segmentation (v2.0)
    "best_mode": "LOW | MID | HIGH",
    "mode_scores": { ... },       // per-mode signature results
    "mode_shift": { ... },        // sku state shift flag (if enabled)
    "artifacts": {                // UI visualization assets
      "images": [
        "/api/v7/results/<run_id>/<img_id>_overlay.png",
        "/api/v7/results/<run_id>/<img_id>_heatmap.png"
      ]
    },
    "debug": { ... }              // optional, includes heatmap, geom, etc.
  }
}
```

## Notes
- `signature` and `anomaly` can be null when gate fails and diagnostics are off.
- `mode_shift` is a flag only; it never overrides `label`.
- `phase` is fixed to `INSPECTION` for production evaluation.
- If pattern baseline is required and missing, INSPECTION returns RETAKE with `PATTERN_BASELINE_NOT_FOUND`.
- `result.artifacts.images` may include overlay/heatmap image paths for UI visualization.
- v2.0 shadow runs only when expected_ink_count is available (input or registry).
- v2.0 quality adds separation_margin (min ?Eab margin vs d0) and warnings:
  - INK_CLUSTER_TOO_SMALL
  - INK_CLUSTER_OVERLAP_HIGH
  - INK_SEPARATION_LOW_CONFIDENCE
- v2.0 `roi` includes both config and effective start (`r_start_config`, `r_start_effective`).
- v2.1 shadow adds `auto_estimation` (silhouette-based) and warnings:
  - AUTO_K_LOW_CONFIDENCE
  - INK_COUNT_MISMATCH_SUSPECTED
  These do not affect `label` and are for diagnostics only.
- v2.2 shadow adds palette colors in `v2_diagnostics.palette` and per-cluster `mean_rgb`/`mean_hex`.
- v2.3 shadow adds ink baseline matching and per-ink deltas in `v2_diagnostics.ink_match`.
