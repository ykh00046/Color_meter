# AI Review Summary (Template)

Use this template to send a compact context bundle to AI reviewers.

## 1) Purpose
- Evaluate v7 STD registration + inspection pipeline behavior.
- Confirm schema stability and operational flow correctness.

## 2) Engine context
- Phase separation: `STD_REGISTRATION` vs `INSPECTION`.
- LOW/MID/HIGH are user-defined groups (not ordered levels).
- `registration.order_check_enabled=false` in current config.

## 3) Attached files (recommended)
- `configs/default.json`
- `models/index.json`
- STD registration results (`results/std_reg_*.json`)
- Inspection results (`results/inspect_*.json`) if available
- Sample `meta_*.json` for registered models

## 4) Key questions for review
- Are registration checks sufficient without order constraints?
- Are separation and within-mode stability thresholds reasonable defaults?
- Is the output schema clear for UI/DB integration?
- Any risks in phase separation or mode selection logic?

## 5) Known limitations (current)
- STD samples are temporary and roughly captured.
- Registration thresholds are placeholders pending real STD sets.

## 6) Next decisions
- Finalize registration thresholds (separation, within-mode stability).
- Add geom consistency checks (center/radius drift).
