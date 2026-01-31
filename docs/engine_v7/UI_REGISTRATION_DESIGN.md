# UI/Registration Design (Draft)

This document outlines a minimal UI flow for STD registration and inspection.

## A) STD Registration UI
### Screens
1) STD Set List
   - Filters: SKU, ink, status (INCOMPLETE/ACTIVE), updated date.
   - Actions: View, Register (new), Promote (if approval flow added).

2) STD Register Form
   - Inputs (required unless noted):
     - SKU
     - Ink (default `INK_DEFAULT`)
     - Mode (LOW/MID/HIGH)
     - STD images (upload multiple)
     - Notes (optional)
     - Created by (optional)
   - Actions:
     - Register (writes model + updates index)
     - Validate (run STD_REGISTRATION phase)

3) STD Validation Result
   - Label: STD_ACCEPTABLE / STD_RETAKE / STD_UNSTABLE
   - Registration summary:
     - separation (LOW_MID, MID_HIGH)
     - within-mode stability
   - Gate scores and reasons
   - Optional debug panel (signature/anomaly if enabled)
   - Required display fields:
     - `decision.label`
     - `decision.reasons`
     - `decision.registration_summary`
     - `decision.gate`

4) Activation (separate screen)
   - Inputs (required unless noted):
     - SKU, Ink
     - Candidate versions (LOW/MID/HIGH)
     - Approved by
     - Reason (optional)
   - Actions:
     - Activate (updates index.json active pointers)
     - Rollback (to previous active)
   - Required display fields:
     - Candidate STD_REGISTRATION result (ACCEPTABLE/RETAKE/UNSTABLE)
     - Current active versions (if any)
     - Last activation metadata (approved_by/approved_at/reason)

### Output fields (core)
- `decision.label`
- `decision.registration_summary`
- `decision.gate`

## B) Inspection UI
### Screens
1) Inspection Run
   - Inputs (required unless noted):
     - SKU, Ink
     - Image upload (single or batch)
     - Phase fixed to INSPECTION
   - Actions:
     - Run inspection

2) Inspection Result
   - Label: OK / RETAKE / NG_COLOR / NG_PATTERN
   - Best mode + mode scores
   - Anomaly type (if NG_PATTERN)
   - Gate scores, reasons
   - Required display fields:
     - `decision.label`
     - `decision.reasons`
     - `decision.best_mode`
     - `decision.mode_scores` (compact view)
     - `decision.anomaly.type`
     - `decision.gate`

### Output fields (core)
- `decision.label`
- `decision.best_mode`
- `decision.mode_scores`
- `decision.anomaly.type`
- `decision.gate`

## C) Registry/Storage Expectations
- Active STD models are referenced from `models/index.json`.
- Version folders contain `model.npz`, `model.json`, `meta.json`.
- Config changes require STD re-registration (`cfg_hash`).

## D) Activation Policy (Required)
- Registration adds INCOMPLETE entries only.
- Activation is separate and requires STD_REGISTRATION = ACCEPTABLE.
- Rollback must restore the previous active pointer.
 - Operator: STD_REGISTRATION / INSPECTION only.
 - Approver: ACTIVATE / ROLLBACK only.
