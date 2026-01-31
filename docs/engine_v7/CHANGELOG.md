# Changelog

## v7.1 (local)
- Add `LENS_STATE_DIR` override for SKU state storage (precedence: `--state_dir` > env > `state/`).
- Record segment band violations as reasons only (no hard trigger).
- Promote NG_PATTERN defect type to top-level `anomaly.type` fields.

## v7.2 (local)
- Add model registry helpers for index-based loading of LOW/MID/HIGH.
- Add STD registration script that writes model.npz/model.json/meta.json and updates index.json.
- Add registry-based run option using sku/ink + models_root.
- Add cfg hash mismatch check (returns RETAKE).
- Add `gate.diagnostic_on_fail` to compute signature/anomaly under gate failure with `debug.inference_valid=false`.

## v7.3 (local)
- Add `phase` flag for STD registration vs inspection (`INSPECTION` / `STD_REGISTRATION`).
- Add STD registration evaluation mode with labels `STD_ACCEPTABLE`, `STD_RETAKE`, `STD_UNSTABLE`.
- Add registration summary (order/separation/within-mode stats) to decision output.
- Add `registration.order_check_enabled` to disable strict LOW/MID/HIGH order checks.

## v7.4 (local)
- Add runbook and flow docs (inspection, registration, UI).
- Add results summarizer script.

## v7.5 (local)
- Registration separation uses pairwise min (LOW_MID/MID_HIGH/LOW_HIGH).
- Registration summary includes geom consistency + warnings for disabled thresholds.
- Results summarizer supports base_dir and uses filename labels by default.

## v7.6 (local)
- Add activation step to runbook.
- Add UI flow diagram (text).
- Expand model registry API activation response details.

## v7.7 (local)
- Add MVP role separation (operator vs approver) for v7 UI/API.

## v7.8 (local)
- Add smoke tests for activation/role/cfg scenarios.
- Add lightweight metrics report script.

## v7.9 (local)
- Add v1.2-shadow OK feature logging (INSPECTION OK only) to jsonl under pattern_baselines/ok_logs.

## v7.10 (local)
- Add v2.0 shadow ink diagnostics (expected_ink_count, sampling, k-means clusters, separation metrics).
- Apply v2.0 ROI effective start and split sampling/segmentation warnings; add deterministic sampling fallback seed.

## v7.11 (local)
- Add v2.1 shadow auto-k estimation (silhouette proxy), confidence scoring, and mismatch warnings.

## v7.12 (local)
- Add v2.2 shadow palette colors (mean_rgb/mean_hex) and palette summary in v2 diagnostics.

## v7.13 (local)
- Add v2.3 shadow ink baseline matching (LOW/MID/HIGH) and per-ink deltas.

## v7.14 (local)
- Add effective_density (area_ratio * alpha) with 3-tier alpha fallback (L1 radial/L2 zone/L3 global).
- Add transition-region weighted alpha sampling (boundary/gradient weights).
- Make median_theta aggregation robust to outliers in radial signatures.
- Stabilize label matching with Hungarian assignment.
- Expose effective_density/alpha metadata in single analysis ink outputs.

## v7.15 (local)
- Add intrinsic ink color inference from white/black pairs (Linear v2 + Beer-Lambert k).
- Record intrinsic fields per cluster and include background calibration metadata in single analysis outputs.
