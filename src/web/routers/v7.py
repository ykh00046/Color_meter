"""
Lens Signature Engine v7 UI/API bridge (MVP)

Sub-routers assembled here. Individual route handlers live in:
  - v7_registration.py  (register_validate, status, entries, cleanup, candidates)
  - v7_activation.py    (activate, rollback)
  - v7_inspection.py    (test_run, inspect, analyze_single)
  - v7_metrics.py       (v2_metrics, trend_line, delete_entry)
  - v7_plate.py         (plate_gate, intrinsic_calibrate, intrinsic_simulate)
"""

from fastapi import APIRouter

from src.config.v7_paths import add_repo_root_to_sys_path, ensure_v7_dirs

ensure_v7_dirs()
add_repo_root_to_sys_path()

from .v7_activation import router as activation_router
from .v7_inspection import router as inspection_router
from .v7_metrics import router as metrics_router
from .v7_plate import router as plate_router
from .v7_production_colors import router as production_colors_router
from .v7_registration import router as registration_router

router = APIRouter(prefix="/api/v7", tags=["V7 MVP"])

router.include_router(registration_router, tags=["V7 Registration"])
router.include_router(activation_router, tags=["V7 Activation"])
router.include_router(inspection_router, tags=["V7 Inspection"])
router.include_router(metrics_router, tags=["V7 Metrics"])
router.include_router(plate_router, tags=["V7 Plate"])
router.include_router(production_colors_router, tags=["V7 Production Colors"])
