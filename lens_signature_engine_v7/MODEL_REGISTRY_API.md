# Model Registry API (Draft)

This document defines the minimal API for managing `models/index.json`.
It is intended to wrap the existing CLI flows (`register_std.py`, `run_signature_engine.py`).

## 1) Register STD (creates version + updates index as INCOMPLETE)
`POST /models/register`

Request (multipart):
- `sku` (string, required)
- `ink` (string, required, default `INK_DEFAULT`)
- `mode` (LOW | MID | HIGH)
- `std_images[]` (files)
- `notes` (string, optional)
- `created_by` (string, optional)
- `expected_ink_count` (int, optional)

Response:
```
{
  "status": "INCOMPLETE",
  "saved_dir": "models/SKU001/INK_A/LOW/v20251225_093248",
  "index_path": "models/index.json"
}
```

Notes:
- This endpoint mirrors `scripts/register_std.py`.
- Registration never activates; activation is a separate step.

## 2) Validate STD (registration phase)
`POST /models/validate`

Request:
- `sku` (string)
- `ink` (string)
- `images[]` (files) or `paths[]` (string list)

Response:
- Same schema as `run_signature_engine.py --phase STD_REGISTRATION`.

## 3) Inspect (production)
`POST /inspect`

Request:
- `sku` (string)
- `ink` (string)
- `images[]` (files) or `paths[]` (string list)
- `phase` optional, default `INSPECTION`
- `expected_ink_count` (int, optional; overrides registry for v2 shadow)

Response:
- Same schema as `run_signature_engine.py --phase INSPECTION`.

## 4) Registry status
`GET /models/status?sku=SKU001&ink=INK_A`

Response:
```
{
  "sku": "SKU001",
  "ink": "INK_A",
  "status": "ACTIVE",
  "expected_ink_count": 2,
  "active": {
    "LOW": "models/SKU001/INK_A/LOW/v20251225_093248",
    "MID": "models/SKU001/INK_A/MID/v20251225_093249",
    "HIGH": "models/SKU001/INK_A/HIGH/v20251225_093250"
  }
}
```

## 5) Activate / rollback
Activation is a separate, audited step.

`POST /models/activate`
- `sku`, `ink`
- `low_version`, `mid_version`, `high_version` (or full paths)
- `approved_by`, `approved_at`, `reason`
- Requires `STD_REGISTRATION` = ACCEPTABLE for all three modes (policy).

Response:
```
{
  "status": "ACTIVE",
  "sku": "SKU001",
  "ink": "INK_A",
  "active": {
    "LOW": "models/SKU001/INK_A/LOW/v20251225_093248",
    "MID": "models/SKU001/INK_A/MID/v20251225_093249",
    "HIGH": "models/SKU001/INK_A/HIGH/v20251225_093250"
  },
  "approved_by": "operator_A",
  "approved_at": "2025-12-25T10:00:00+09:00",
  "reason": "temporary standard approved"
}
```

`POST /models/rollback`
- `sku`, `ink`, `mode`, `version`
 - Restores previous active pointer only (no new training).

## 6) Delete entry (admin only)
`POST /api/v7/delete_entry`

Request:
```
{
  "sku": "SKU001",
  "ink": "INK_DEFAULT",
  "deleted_by": "admin_user",
  "reason": "cleanup test data"
}
```

Notes:
- Admin only (`X-User-Role: admin`)
- Soft delete: entry status set to `DELETED`, active pointers cleared.

## 7) Error policy (recommended)
- `MODEL_NOT_FOUND`
- `MODEL_INCOMPLETE:<MODE>`
- `MODEL_LOAD_FAILED:<MODE>`
- `CFG_MISMATCH:<MODE>`

## 8) Activation policy (required)
- INSPECTION must return `RETAKE` when no ACTIVE model exists.
- Activation allowed only after `STD_REGISTRATION` returns `STD_ACCEPTABLE`.

## 9) Role policy (minimal)
- `X-User-Role: operator` for register/validate/inspect.
- `X-User-Role: approver` for activate/rollback.
- `X-User-Role: admin` for delete_entry.
