# STD Registration Flow (v7)

This document summarizes the STD registration/validation pipeline and output
schema used by `run_signature_engine.py` in `STD_REGISTRATION` phase.

## Pipeline flow
1) Load STD models (registry or direct paths)
   - Must have LOW/MID/HIGH. Otherwise return `STD_RETAKE`.
2) Gate (capture validity)
   - center offset, blur, illumination symmetry.
   - If failed and `gate.diagnostic_on_fail=false`, skip signature/anomaly debug.
3) Registration checks (phase-specific)
   - order check is optional (`registration.order_check_enabled`).
   - separation check uses pairwise minimum (LOW_MID, MID_HIGH, LOW_HIGH).
   - within-mode stability check (optional threshold).
4) Decision
   - `STD_ACCEPTABLE`, `STD_RETAKE`, or `STD_UNSTABLE`.

## Output schema (top-level)
```
{
  "cfg": { ... },
  "results": [ ... ],
  "multi_mode": true,
  "sku": "SKU001",
  "phase": "STD_REGISTRATION"
}
```

## Per-image decision schema (key fields)
```
{
  "path": "path/to/image.png",
  "decision": {
    "label": "STD_ACCEPTABLE | STD_RETAKE | STD_UNSTABLE",
    "reasons": [ ... ],
    "reason_codes": [ ... ],
    "reason_messages": [ ... ],
    "phase": "STD_REGISTRATION",
    "gate": { ... },
    "registration_summary": {
      "order_enabled": false,
      "order_metric": "L_mean",
      "order_direction": "asc",
      "order_values": { ... },
      "order_ok": true,
      "sep_threshold": 1.0,
      "separation": { "LOW_MID": 1.2, "MID_HIGH": 2.3, "LOW_HIGH": 3.1 },
      "min_pairwise_separation": 1.2,
      "separation_ok": true,
      "within_mode_stability": { ... },
      "within_mode_threshold": null,
      "within_mode_ok": null,
      "geom_consistency": {
        "center_drift_px": 1.2,
        "radius_drift_ratio": 0.05,
        "passed": null
      },
      "warnings": [
        "WITHIN_MODE_THRESHOLD_DISABLED",
        "GEOM_THRESHOLD_DISABLED"
      ]
    },
    "debug": { ... }   // optional signature/anomaly debug
  }
}
```

## Notes
- STD registration is for baseline creation + diagnostics collection only.
- Allowlist (STD registration): v1 measurement, v2 diagnostics (shadow).
- Disallow: v3 summary/trend/ops_judgment in STD registration (or diagnostics-only flag).
- `signature` and `anomaly` are not used for label decisions in this phase.
- When `gate.diagnostic_on_fail=true`, signature/anomaly are computed and stored
  in `debug` only for diagnostics.

## Robust stats shadow policy
- Robust stats (median/MAD) must run in shadow first.
- Promote to primary only if:
  - false RETAKE/NG rate decreases by at least X%, and
  - miss rate does not increase, and
  - observation window >= N samples (or >= N days).

## ΔE00 policy
- ΔE00 is for reporting/health only, not for pass/fail.
