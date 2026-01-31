# Lens Signature Engine (MVP package)

Human-like inspection engine for dot/gradient contact lens patterns.

## Decision flow
1) Gate: capture validity (center/blur/illumination)
2) Color Signature: radial Lab signature similarity vs STD
3) Pattern/Anomaly: angular non-uniformity + center blobs
4) Decision: OK / RETAKE / NG_COLOR / NG_PATTERN

## Install deps
- opencv-python
- numpy

## Run
From repository root:
```bash
python scripts/run_signature_engine.py \
  --std /path/to/std.png \
  --tests /path/to/test1.png /path/to/test2.png \
  --cfg configs/default.json \
  --out result.json
```

Inspection flow and output schema overview: `INSPECTION_FLOW.md`
STD registration flow and output schema overview: `STD_REGISTRATION_FLOW.md`
Runbook (registration → validation → inspection): `RUNBOOK.md`
AI review template: `AI_REVIEW_SUMMARY.md`
UI/registration design draft: `UI_REGISTRATION_DESIGN.md`
Model registry API draft: `MODEL_REGISTRY_API.md`
UI flow diagram (text): `UI_FLOW_DIAGRAM.md`

### STD registration mode (no NG_COLOR/OK)
Use `--phase STD_REGISTRATION` to validate STD sets without producing NG/OK labels.
```bash
python scripts/run_signature_engine.py \
  --sku SKU001 --ink INK_A \
  --models_root models \
  --tests std_low_1.png std_mid_1.png std_high_1.png \
  --phase STD_REGISTRATION \
  --cfg configs/default.json \
  --out std_registration.json
```

## Config
Edit `configs/default.json`:
- `signature.corr_min`, `signature.de_p95_max`: color/gradient tolerance
- `anomaly.angular_unif_max`, `anomaly.center_blob_max`: pattern/defect sensitivity
- `gate.*`: retake thresholds
- `gate.diagnostic_on_fail`: run signature/anomaly even when gate fails (debug only)
- `registration.*`: STD registration checks (order/separation/within-mode stability)
  - `registration.order_check_enabled`: disable order check when LOW/MID/HIGH are labels, not strict levels
  - separation uses pairwise minimum (LOW_MID, MID_HIGH, LOW_HIGH)


## v2 additions
- Multi-STD training (mean/std bands)
- Band-based signature decision (fail_ratio)
- Anomaly heatmap (downsampled polar residual)

### Train STD model
```bash
python scripts/train_std_model.py --stds std1.png std2.png --cfg configs/default.json --out_prefix models/SKU001_std
```

### Run using trained model
```bash
python scripts/run_signature_engine.py --std_model models/SKU001_std --tests test1.png test2.png --cfg configs/default.json --out result.json
```


## v3 additions
- Weighted radial segments for signature band decision (inner/mid/outer)
- Optional per-segment band violation cap


## v4 additions
- Segment-specific band_k (inner/mid/outer) via signature.segment_k
- Analyzer uses per-segment k array for band violation


## v5 additions
- Auto-suggest segment_k from STD set (scripts/suggest_segment_k.py)

### Suggest segment_k
```bash
python scripts/suggest_segment_k.py --stds std1.png std2.png std3.png --cfg configs/default.json --percentile 99.5
```
Then copy the printed `segment_k` into `configs/default.json`.


## v7 additions (sampling QC friendly)
- Multi-mode STD (LOW/MID/HIGH) evaluation with best_mode selection (A-plan shared thresholds)
- SKU-level mode shift flag (does not override decision) for low-volume sampling (1~3/day)
- NG_PATTERN defect type heuristic: RING / SECTOR / BLOB / UNIFORMITY
- State directory override via `LENS_STATE_DIR` (fallback: `state/`)
- `segment_fail_ratio_max` is recorded as a reason only (no hard trigger)
- NG_PATTERN defect type promoted to top-level `anomaly.type`
- `phase` flag supports STD registration labels and summary output

## Future (post STD refresh)
- Replace separation check with pairwise min separation (LOW_MID/MID_HIGH/LOW_HIGH).
- Require `registration.max_within_std` threshold (within-mode stability gate).
- Add geom consistency checks in registration (center/radius drift).
- Extend registration summary with pairwise separation + geom consistency fields.

## Model registry (index.json)
Recommended storage layout:
```
models/
  index.json
  SKU001/
    INK_A/
      LOW/
        v20251224_153000/
          model.npz
          model.json
          meta.json
```

`model.npz` + `model.json` are used by the engine loader. `meta.json` holds capture/approval info.

### Register STD (updates index.json)
```bash
python scripts/register_std.py --sku SKU001 --ink INK_A --mode LOW \
  --stds low1.png low2.png --cfg configs/default.json
```

### Multi-mode run
```bash
python scripts/run_signature_engine.py \
  --std_model_low  models/SKU001_low \
  --std_model_mid  models/SKU001_mid \
  --std_model_high models/SKU001_high \
  --sku SKU001 \
  --tests test1.png test2.png \
  --cfg configs/default.json \
  --out result.json
```

### Registry-based run (index.json)
```bash
python scripts/run_signature_engine.py \
  --sku SKU001 --ink INK_A \
  --models_root models \
  --tests test1.png test2.png \
  --cfg configs/default.json \
  --out result.json
```

### Summarize results
```bash
python scripts/summarize_results.py --inputs results --phase INSPECTION --out results/summary.json
```
`--base_dir` can be used to keep file labels relative to a root folder.

### Smoke tests
```bash
python scripts/smoke_tests.py
```

### Metrics report (lightweight)
```bash
python scripts/metrics_report.py --inputs results --out results/metrics.json
```
For ACTIVE change comparisons, pass the model index and a window size:
```bash
python scripts/metrics_report.py --inputs results --index models/index.json --window_size 20 --out results/metrics.json
```
v2 shadow metrics (diagnostics only):
```bash
python scripts/metrics_report.py --inputs results --v2_shadow --v2_window_size 20 --v2_group_by baseline --out results/v2_shadow_metrics.json
```

### Integrity check (artifact validation)
```bash
python scripts/integrity_check.py --index models/index.json --results_dir results --out results/integrity_report.json
```

### Housekeeping (retention + archive)
Dry-run by default; use `--apply` to move files.
```bash
python scripts/housekeeping.py --index models/index.json --out results/housekeeping_plan.json
python scripts/housekeeping.py --index models/index.json --apply --out results/housekeeping_applied.json
```
