"""
V7 Activation & Governance sub-router.

Routes:
  POST /activate - Activate a model version
  POST /rollback - Rollback to previous active version
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from src.config.v7_paths import V7_MODELS
from src.engine_v7.core.anomaly.pattern_baseline import build_pattern_baseline, extract_pattern_features
from src.engine_v7.core.geometry.lens_geometry import detect_lens_circle
from src.engine_v7.core.measure.baselines.ink_baseline import build_ink_baseline
from src.engine_v7.core.model_registry import compute_cfg_hash
from src.engine_v7.core.pipeline.analyzer import _registration_summary
from src.engine_v7.core.signature.model_io import load_model
from src.engine_v7.core.utils import apply_white_balance

from .v7_helpers import (
    _active_versions,
    _find_entry,
    _load_cfg,
    _load_std_images_from_versions,
    _normalize_expected_ink_count,
    _read_index,
    _require_role,
    _safe_delete_path,
    logger,
)
from .v7_registration import _auto_tune_cfg_from_std, _load_approval_pack, _validate_pack_for_activate

router = APIRouter()


class ActivateRequest(BaseModel):
    sku: str
    ink: str
    low_version: str
    mid_version: str
    high_version: str
    approved_by: Optional[str] = "SYSTEM"
    reason: Optional[str] = ""
    validation_label: Optional[str] = ""
    approval_pack_path: Optional[str] = ""


class RollbackRequest(BaseModel):
    sku: str
    ink: str
    approved_by: Optional[str] = "SYSTEM"
    reason: Optional[str] = ""
    validation_label: Optional[str] = ""
    approval_pack_path: Optional[str] = ""


def _finalize_pack(
    pack: Dict,
    pack_path: Path,
    action: str,
    approved_by: str,
    active_snapshot_path: str,
) -> None:
    pack["final"] = {
        "status": "APPROVED" if action == "activate" else "ROLLED_BACK",
        "action": action.upper(),
        "approved_at": datetime.now().isoformat(),
        "approved_by": approved_by,
        "active_snapshot_path": active_snapshot_path,
    }
    pack_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_active_snapshot(entry: Dict, active: Dict[str, str]) -> Optional[Path]:
    if not active:
        return None
    snapshots_dir = V7_MODELS / "active_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_cfg(entry.get("sku", ""))
    std_models: Dict[str, Any] = {}
    std_load_error: Optional[str] = None
    for mode in ["LOW", "MID", "HIGH"]:
        rel = active.get(mode, "")
        if not rel:
            continue
        try:
            prefix = V7_MODELS / rel / "model"
            std_models[mode] = load_model(str(prefix))
        except Exception as exc:
            std_load_error = str(exc)
            logger.warning("Failed to load std model for %s: %s", mode, exc)

    tuned_meta = None
    try:
        cfg, tuned_meta = _auto_tune_cfg_from_std(cfg, std_models, entry, active)
    except Exception as exc:
        logger.warning("Auto-tune failed: %s", exc)

    snapshot = {
        "schema_version": "1.0",
        "sku": entry.get("sku", ""),
        "ink": entry.get("ink", ""),
        "active": active,
        "active_versions": _active_versions(active),
        "created_at": datetime.now().isoformat(),
        "cfg_hash": compute_cfg_hash(cfg),
        "config_snapshot_hash": compute_cfg_hash(cfg),
        "config_snapshot": cfg,
        "config_snapshot_auto_tune": tuned_meta,
        "registration_summary": None,
        "warnings": [],
    }
    try:
        if std_load_error:
            raise ValueError(std_load_error)
        if any(m not in std_models for m in ["LOW", "MID", "HIGH"]):
            raise ValueError("missing std model for summary")
        summary, _ = _registration_summary(std_models, cfg)
        snapshot["registration_summary"] = summary
        snapshot["warnings"] = summary.get("warnings", [])
    except Exception as exc:
        snapshot["snapshot_error"] = str(exc)
        logger.warning("Failed to build active snapshot: %s", exc)

    snap_name = f"{entry.get('sku', '')}_{entry.get('ink', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    snap_path = snapshots_dir / snap_name
    snapshot["path"] = str((snapshots_dir / snap_name).relative_to(V7_MODELS))
    snap_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return snap_path


def _write_pattern_baseline(entry: Dict, active: Dict[str, str]) -> Optional[Path]:
    if not active:
        return None
    cfg = _load_cfg(entry.get("sku", ""))
    images_by_mode = _load_std_images_from_versions(active)
    std_images = []
    for mode in ["LOW", "MID", "HIGH"]:
        std_images.extend(images_by_mode.get(mode, []))
    if not std_images:
        return None

    features_list = []
    for p in std_images:
        try:
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            wb_cfg = cfg.get("white_balance", {}) or {}
            if wb_cfg.get("enabled", False):
                geom = detect_lens_circle(bgr)
                bgr, _ = apply_white_balance(bgr, geom, wb_cfg)
            features_list.append(extract_pattern_features(bgr, cfg=cfg))
        except Exception:
            continue
    if not features_list:
        return None

    baseline = {
        "schema_version": "pattern_baseline.v1",
        "sku": entry.get("sku", ""),
        "ink": entry.get("ink", ""),
        "active": active,
        "active_versions": _active_versions(active),
        "generated_at": datetime.now().isoformat(),
        "source": {"std_images": images_by_mode},
        "features": build_pattern_baseline(features_list),
        "policy": {"margins": cfg.get("pattern_baseline", {}).get("margins", {})},
    }

    baselines_dir = V7_MODELS / "pattern_baselines" / entry.get("sku", "") / entry.get("ink", "")
    baselines_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"PB_{entry.get('sku', '')}_{entry.get('ink', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    baseline_path = baselines_dir / base_name
    baseline["path"] = str(baseline_path.relative_to(V7_MODELS).as_posix())
    baseline_path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    return baseline_path


def _write_ink_baseline(entry: Dict, active: Dict[str, str]) -> Optional[Path]:
    cfg = _load_cfg(entry.get("sku", ""))
    expected = entry.get("expected_ink_count")
    if expected is None:
        return None
    images_by_mode = _load_std_images_from_versions(active)
    baseline = build_ink_baseline(images_by_mode, cfg, expected_k=int(expected))
    if not baseline:
        return None

    baselines_dir = V7_MODELS / "ink_baselines" / entry.get("sku", "") / entry.get("ink", "")
    baselines_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"IB_{entry.get('sku', '')}_{entry.get('ink', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    baseline_path = baselines_dir / base_name
    baseline_doc = {
        "schema_version": "ink_baseline.v1",
        "sku": entry.get("sku", ""),
        "ink": entry.get("ink", ""),
        "expected_ink_count": int(expected),
        "active": active,
        "active_versions": _active_versions(active),
        "generated_at": datetime.now().isoformat(),
        "source": {"std_images": images_by_mode},
        "baseline": baseline,
    }
    baseline_doc["path"] = str(baseline_path.relative_to(V7_MODELS).as_posix())
    baseline_path.write_text(json.dumps(baseline_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return baseline_path


def _finalize_activation(
    sku: str,
    ink: str,
    versions: Dict[str, str],
    approved_by: str,
    reason: str,
    approval_pack_info: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Core logic to finalize model activation.
    Updates index, creates snapshot and baselines.
    """
    index = _read_index()
    entry = _find_entry(index, sku, ink)
    if not entry:
        raise ValueError(f"SKU/INK not found in index: {sku}/{ink}")

    active = {
        "LOW": f"{sku}/{ink}/LOW/{versions['LOW']}",
        "MID": f"{sku}/{ink}/MID/{versions['MID']}",
        "HIGH": f"{sku}/{ink}/HIGH/{versions['HIGH']}",
    }

    history = entry.get("active_history", [])
    if entry.get("active"):
        history.append(
            {
                "from": entry["active"],
                "to": active,
                "action": "activate",
                "actor": approved_by,
                "reason": reason or "",
                "ts": datetime.now().isoformat(),
            }
        )

    entry["active"] = active
    entry["status"] = "ACTIVE"
    entry["activated_by"] = approved_by
    entry["activated_at"] = datetime.now().isoformat()
    entry["activated_reason"] = reason or ""
    entry["active_history"] = history

    # 1. Write Data Artifacts
    snap_path = _write_active_snapshot(entry, active)
    if snap_path:
        entry["active_snapshot"] = str(snap_path.relative_to(V7_MODELS))

    baseline_path = _write_pattern_baseline(entry, active)
    if baseline_path:
        entry["pattern_baseline"] = str(baseline_path.relative_to(V7_MODELS))

    ink_baseline_path = _write_ink_baseline(entry, active)
    if ink_baseline_path:
        entry["ink_baseline"] = str(ink_baseline_path.relative_to(V7_MODELS))

    # 2. Update Approval Pack if provided
    pack_update_failed = False
    if approval_pack_info:
        try:
            _finalize_pack(
                approval_pack_info["pack"],
                approval_pack_info["path"],
                "activate",
                approved_by,
                entry.get("active_snapshot", ""),
            )
        except Exception:
            pack_update_failed = True

    # 3. Save Index
    (V7_MODELS / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "status": "ACTIVE",
        "sku": sku,
        "ink": ink,
        "active": active,
        "active_snapshot": entry.get("active_snapshot", ""),
        "pattern_baseline": entry.get("pattern_baseline", ""),
        "ink_baseline": entry.get("ink_baseline", ""),
        "pack_update_failed": pack_update_failed,
    }


@router.post("/activate")
async def activate_model(req: ActivateRequest, x_user_role: Optional[str] = Header(default="")):
    _require_role("approver", x_user_role)
    if req.validation_label and req.validation_label != "STD_ACCEPTABLE":
        raise HTTPException(status_code=400, detail="Activation requires STD_ACCEPTABLE")

    pack_info = None
    if req.approval_pack_path:
        try:
            pack, pack_path = _load_approval_pack(req.approval_pack_path)
            _validate_pack_for_activate(pack, req.sku, req.ink)
            pack_info = {"pack": pack, "path": pack_path}
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="PACK_NOT_FOUND")
        except ValueError as exc:
            if str(exc) == "approval pack status blocked":
                raise HTTPException(status_code=400, detail="PACK_BLOCKED")
            raise HTTPException(status_code=400, detail="PACK_MISMATCH")

    try:
        versions = {"LOW": req.low_version, "MID": req.mid_version, "HIGH": req.high_version}
        result = _finalize_activation(
            sku=req.sku,
            ink=req.ink,
            versions=versions,
            approved_by=req.approved_by,
            reason=req.reason,
            approval_pack_info=pack_info,
        )
        return result
    except ValueError as e:
        # Security: Log details, give generic message to client
        logger.warning(f"Activation value error: {e}")
        raise HTTPException(status_code=404, detail="Resource not found or invalid configuration")
    except Exception as e:
        logger.error(f"Activation failed: {e}")
        raise HTTPException(status_code=500, detail="INTERNAL_SERVER_ERROR")


@router.post("/rollback")
async def rollback_model(req: RollbackRequest, x_user_role: Optional[str] = Header(default="")):
    _require_role("approver", x_user_role)
    if req.validation_label and req.validation_label != "STD_ACCEPTABLE":
        raise HTTPException(status_code=400, detail="Rollback requires STD_ACCEPTABLE")

    pack = None
    pack_path = None
    pack_update_failed = False
    if req.approval_pack_path:
        try:
            pack, pack_path = _load_approval_pack(req.approval_pack_path)
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="PACK_NOT_FOUND")
        except ValueError:
            raise HTTPException(status_code=400, detail="PACK_MISMATCH")

    index = _read_index()
    entry = _find_entry(index, req.sku, req.ink)
    if not entry:
        raise HTTPException(status_code=404, detail="SKU/INK not found")

    history = entry.get("active_history", [])
    if not history:
        raise HTTPException(status_code=400, detail="No rollback history found")

    last = history[-1]
    prev_active = last.get("from") or last.get("active")
    if not prev_active:
        raise HTTPException(status_code=400, detail="Rollback target not available")

    current_active = entry.get("active")
    entry["active"] = prev_active
    entry["status"] = "ACTIVE"
    entry["activated_by"] = req.approved_by
    entry["activated_at"] = datetime.now().isoformat()
    entry["activated_reason"] = req.reason or ""
    history.append(
        {
            "from": current_active,
            "to": prev_active,
            "action": "rollback",
            "actor": req.approved_by,
            "reason": req.reason or "",
            "pattern_baseline_path": entry.get("pattern_baseline", ""),
            "approval_pack_path": str(pack_path) if pack_path else "",
            "approval_id": (pack or {}).get("approval_id", ""),
            "decision_status": (pack or {}).get("decision", {}).get("status", ""),
            "ts": entry["activated_at"],
        }
    )
    entry["active_history"] = history

    snap_path = _write_active_snapshot(entry, prev_active)
    if snap_path:
        entry["active_snapshot"] = str(snap_path.relative_to(V7_MODELS))

    baseline_path = _write_pattern_baseline(entry, prev_active)
    if baseline_path:
        entry["pattern_baseline"] = str(baseline_path.relative_to(V7_MODELS))
    ink_baseline_path = _write_ink_baseline(entry, prev_active)
    if ink_baseline_path:
        entry["ink_baseline"] = str(ink_baseline_path.relative_to(V7_MODELS))

    if pack and pack_path:
        try:
            _finalize_pack(pack, pack_path, "rollback", req.approved_by, entry.get("active_snapshot", ""))
        except Exception:
            pack_update_failed = True

    (V7_MODELS / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "status": "ACTIVE",
        "sku": req.sku,
        "ink": req.ink,
        "active": prev_active,
        "approved_by": req.approved_by,
        "approved_at": entry["activated_at"],
        "reason": req.reason or "",
        "active_snapshot": entry.get("active_snapshot", ""),
        "pattern_baseline": entry.get("pattern_baseline", ""),
        "ink_baseline": entry.get("ink_baseline", ""),
        "approval_pack_path": str(pack_path) if pack_path else "",
        "pack_update_failed": pack_update_failed,
    }
