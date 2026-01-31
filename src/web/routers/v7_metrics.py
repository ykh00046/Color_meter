"""
V7 Metrics, Trend, and Entry Deletion sub-router.

Routes:
  GET  /v2_metrics   - Retrieve v2 shadow metrics
  GET  /trend_line   - Get decision trend analysis
  POST /delete_entry - Delete SKU/INK entry
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from src.config.v7_paths import V7_MODELS, V7_RESULTS, V7_ROOT
from src.engine_v7.core.insight.trend import build_v3_trend

from .v7_helpers import (
    _find_entry,
    _load_recent_decisions_for_trend,
    _read_index,
    _require_role,
    _safe_delete_path,
    logger,
)

router = APIRouter()


class DeleteEntryRequest(BaseModel):
    sku: str
    ink: str
    deleted_by: Optional[str] = "SYSTEM"
    reason: Optional[str] = ""
    hard_delete: Optional[bool] = False


def _delete_approval_packs(sku: str, ink: str) -> List[str]:
    approvals_dir = V7_ROOT / "approvals" / "approval_packs"
    if not approvals_dir.exists():
        return []
    deleted: List[str] = []
    for path in approvals_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ctx = data.get("context", {})
            if ctx.get("sku") == sku and ctx.get("ink") == ink:
                path.unlink()
                deleted.append(str(path))
                continue
        except Exception:
            logger.warning("Failed to read approval pack: %s", path)
        if path.name.startswith(f"APPROVAL_{sku}_{ink}_"):
            try:
                path.unlink()
                deleted.append(str(path))
            except Exception:
                logger.warning("Failed to delete approval pack: %s", path)
    return deleted


def _delete_results_dirs(sku: str, ink: str) -> List[str]:
    if not V7_RESULTS.exists():
        return []
    deleted: List[str] = []
    for run_dir in V7_RESULTS.iterdir():
        if not run_dir.is_dir():
            continue
        candidates = [
            run_dir / "inspection.json",
            run_dir / "std_registration.json",
            run_dir / "validation.json",
        ]
        matched = False
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if data.get("sku") == sku and data.get("ink") == ink:
                    matched = True
                    break
            except Exception:
                continue
        if matched:
            try:
                shutil.rmtree(run_dir)
                deleted.append(str(run_dir))
            except Exception:
                logger.warning("Failed to delete results dir: %s", run_dir)
    return deleted


@router.get("/v2_metrics")
async def get_v2_metrics(sku: str, ink: str = "INK_DEFAULT"):
    metrics_path = V7_ROOT / "results" / "v2_shadow_metrics.json"
    if not metrics_path.exists():
        return {"status": "NOT_FOUND", "groups": []}

    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        groups = data.get("groups", [])

        match = next((g for g in groups if g.get("sku") == sku), None)
        if match:
            return {"status": "OK", "metrics": match}
        return {"status": "NO_MATCH", "metrics": None}
    except Exception as e:
        logger.error(f"Failed to read v2 metrics: {e}")
        return {"status": "ERROR", "message": str(e)}


@router.get("/trend_line")
async def get_trend_line(sku: str, ink: str = "INK_DEFAULT"):
    decisions = _load_recent_decisions_for_trend(sku, ink, window_requested=20)
    trend = build_v3_trend(decisions, window_requested=20)
    return {"status": "OK", "trend": trend, "trend_line": (trend or {}).get("trend_line", "")}


@router.post("/delete_entry")
async def delete_entry(req: DeleteEntryRequest, x_user_role: Optional[str] = Header(default="")):
    _require_role("admin", x_user_role)
    index = _read_index()
    items = index.get("items", [])
    entry = _find_entry(index, req.sku, req.ink)
    if not entry:
        raise HTTPException(status_code=404, detail="SKU/INK not found")

    deleted_paths: List[str] = []
    if req.hard_delete:
        for mode, rel_path in (entry.get("active") or {}).items():
            deleted = _safe_delete_path(rel_path)
            if deleted:
                deleted_paths.append(deleted)
        deleted = _safe_delete_path(entry.get("active_snapshot", ""))
        if deleted:
            deleted_paths.append(deleted)
        deleted = _safe_delete_path(entry.get("pattern_baseline", ""))
        if deleted:
            deleted_paths.append(deleted)
        deleted = _safe_delete_path(entry.get("ink_baseline", ""))
        if deleted:
            deleted_paths.append(deleted)
        deleted_paths.extend(_delete_approval_packs(req.sku, req.ink))
        deleted_paths.extend(_delete_results_dirs(req.sku, req.ink))

        index["items"] = [item for item in items if not (item.get("sku") == req.sku and item.get("ink") == req.ink)]
        (V7_MODELS / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "status": "DELETED",
            "sku": req.sku,
            "ink": req.ink,
            "hard_delete": True,
            "deleted_paths": deleted_paths,
            "removed_from_index": True,
        }

    entry["status"] = "DELETED"
    entry["active"] = {}
    entry["active_snapshot"] = ""
    entry["pattern_baseline"] = ""
    entry["ink_baseline"] = ""
    entry["deleted_by"] = req.deleted_by or "SYSTEM"
    entry["deleted_at"] = datetime.now().isoformat()
    entry["deleted_reason"] = req.reason or ""

    (V7_MODELS / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "DELETED",
        "sku": req.sku,
        "ink": req.ink,
        "deleted_by": entry["deleted_by"],
        "deleted_at": entry["deleted_at"],
        "reason": entry["deleted_reason"],
        "hard_delete": bool(req.hard_delete),
        "deleted_paths": deleted_paths,
    }
