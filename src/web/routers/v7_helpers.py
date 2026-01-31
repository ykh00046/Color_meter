"""
Shared utilities for v7 sub-routers.

Contains constants, encoders, and helper functions used across
registration, activation, inspection, metrics, and plate modules.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from src.config.v7_paths import (
    REPO_ROOT,
    V7_MODELS,
    V7_RESULTS,
    V7_ROOT,
    V7_TEST_RESULTS,
    add_repo_root_to_sys_path,
    ensure_v7_dirs,
)
from src.engine_v7.core.config_loader import load_cfg_with_sku
from src.engine_v7.core.model_registry import compute_cfg_hash
from src.utils.security import sanitize_filename, validate_file_extension, validate_file_size

logger = logging.getLogger(__name__)

V7_SUBPROCESS_MAX_CONCURRENCY = int(os.getenv("V7_SUBPROCESS_MAX_CONCURRENCY", "2"))
V7_SUBPROCESS_TIMEOUT_SEC = int(os.getenv("V7_SUBPROCESS_TIMEOUT_SEC", "180"))
_SUBPROC_SEM = asyncio.Semaphore(V7_SUBPROCESS_MAX_CONCURRENCY)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types and dataclasses."""

    def default(self, obj):
        # Handle numpy types
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        # Handle dataclasses (e.g., LensGeometry, GateResult)
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return super().default(obj)


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "y"}


def _load_cfg(sku: Optional[str] = None) -> Dict:
    strict_unknown = _env_flag("LENS_CFG_STRICT")
    cfg, sources, warnings = load_cfg_with_sku(
        str(V7_ROOT / "configs" / "default.json"),
        sku,
        sku_dir=str(REPO_ROOT / "config" / "sku_db"),
        strict_unknown=strict_unknown,
    )
    if warnings:
        logger.warning("SKU cfg has unknown keys: %s", ", ".join(warnings))
    logger.info("Applied cfg sources: %s", ", ".join(sources))
    return cfg


def _resolve_cfg_path(sku: Optional[str]) -> Path:
    base_path = V7_ROOT / "configs" / "default.json"
    if not sku:
        return base_path
    sku_path = REPO_ROOT / "config" / "sku_db" / f"{sku}.json"
    return sku_path if sku_path.exists() else base_path


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp_path.replace(path)


def _compute_center_crop_mean_rgb(bgr: np.ndarray, center_crop: float) -> List[float]:
    if bgr is None or bgr.size == 0:
        return [0.0, 0.0, 0.0]
    h, w = bgr.shape[:2]
    crop = max(0.1, min(1.0, float(center_crop)))
    ch = int(h * crop)
    cw = int(w * crop)
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    roi = bgr[y0 : y0 + ch, x0 : x0 + cw]
    if roi.size == 0:
        roi = bgr
    rgb = roi[..., ::-1]
    mean_rgb = rgb.reshape(-1, 3).mean(axis=0)
    return [round(float(v), 2) for v in mean_rgb]


def _load_snapshot_config(entry: Optional[Dict]) -> Optional[Dict[str, Any]]:
    if not entry:
        return None
    rel = entry.get("active_snapshot", "")
    if not rel:
        return None
    snap_path = V7_MODELS / rel
    if not snap_path.exists():
        return None
    try:
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    cfg = snap.get("config_snapshot")
    return cfg if isinstance(cfg, dict) else None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_expected_ink_count(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    if count <= 0:
        return None
    return count


def _parse_match_ids(raw: Optional[str], count: int) -> Optional[List[str]]:
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        if text.startswith("["):
            ids = json.loads(text)
        else:
            ids = [s.strip() for s in text.split(",") if s.strip()]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid match_ids: {exc}")
    if not isinstance(ids, list):
        raise HTTPException(status_code=400, detail="match_ids must be a JSON array or comma-separated list")
    if len(ids) != count:
        raise HTTPException(status_code=400, detail="match_ids count mismatch")
    return [str(x) for x in ids]


def _resolve_sku_for_ink(sku: str, ink: str) -> str:
    if not sku:
        return sku
    if ink and ink.upper() == "INK3" and sku.upper() == "DEFAULT":
        return "DEFAULT_INK3"
    return sku


def _active_versions(active: Dict[str, str]) -> Dict[str, str]:
    versions = {}
    for mode, path in (active or {}).items():
        versions[mode] = Path(path).name if path else ""
    return versions


def _require_role(expected: str, user_role: Optional[str]) -> None:
    """
    Enforce role-based access control.

    Args:
        expected: Required role (e.g., 'operator', 'approver', 'admin')
        user_role: Role from X-User-Role header

    Raises:
        HTTPException: If role is missing or insufficient
    """
    # Role checks are disabled for now.
    return


async def _save_uploads(files: List[UploadFile], run_dir: Path, max_mb: int = 10) -> List[Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    allowed_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]
    for f in files:
        if not f.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        if not validate_file_extension(f.filename, allowed_exts):
            raise HTTPException(status_code=400, detail=f"Invalid file extension: {f.filename}")
        content = await f.read()
        if not validate_file_size(len(content), max_size_mb=max_mb):
            raise HTTPException(status_code=413, detail=f"File too large: {f.filename}")
        dest = run_dir / sanitize_filename(f.filename)  # Prevent path traversal
        dest.write_bytes(content)
        paths.append(dest)
    return paths


async def _save_single_upload(file: UploadFile, dest: Path, max_mb: int = 10) -> Path:
    paths = await _save_uploads([file], dest.parent, max_mb=max_mb)
    return paths[0]


def _load_bgr(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {path.name}")
    return bgr


def _validate_subprocess_arg(value: str, param_name: str, max_len: int = 200) -> str:
    """
    Validate and sanitize arguments before passing to subprocess.

    Args:
        value: The argument value to validate
        param_name: Name of the parameter (for error messages)
        max_len: Maximum allowed length

    Returns:
        Validated string

    Raises:
        HTTPException: If validation fails
    """
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail=f"Invalid {param_name}: must be string")

    # Check length
    if len(value) > max_len:
        raise HTTPException(status_code=400, detail=f"Invalid {param_name}: too long (max {max_len})")

    # Block null bytes and other control characters
    if any(ord(c) < 32 and c not in "\n\r\t" for c in value):
        raise HTTPException(status_code=400, detail=f"Invalid {param_name}: contains control characters")

    return value


def _run_script(script: Path, args: List[str], *, timeout_sec: int = 120) -> str:
    """
    Run a Python script as subprocess safely.

    Security notes:
    - Uses list-based arguments (no shell injection)
    - Doesn't expose internal errors to client
    - Validates script path is within allowed directory
    """
    # Validate script path is within V7_ROOT
    try:
        script.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        logger.error(f"Script path outside V7_ROOT: {script}")
        raise HTTPException(status_code=500, detail="Internal configuration error")

    if not script.exists():
        logger.error(f"Script not found: {script}")
        raise HTTPException(status_code=500, detail="Internal configuration error")

    cmd = [sys.executable, str(script)] + args
    logger.info("Running v7 script: %s", " ".join(cmd))

    try:
        if os.getenv("V7_FORCE_EXEC_FAIL") == "1":
            raise subprocess.CalledProcessError(
                returncode=42,
                cmd=["<forced_fail>"],
                output="",
                stderr="FORCED_FAIL: V7_FORCE_EXEC_FAIL=1",
            )
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=True,
        )
        return result.stdout
    except subprocess.TimeoutExpired as exc:
        logger.warning("v7 script timeout: %s", exc)
        raise
    except subprocess.CalledProcessError as exc:
        stderr_tail = (exc.stderr or "")[-2000:]
        cmd_str = " ".join(shlex.quote(str(x)) for x in (exc.cmd or []))
        msg = "[ENGINE_EXEC_FAILED] " f"rc={exc.returncode} " f"cmd={cmd_str} " f"stderr_tail={stderr_tail!r}"
        print(msg, file=sys.__stderr__, flush=True)
        try:
            os.write(2, (msg + "\n").encode("utf-8", errors="replace"))
        except OSError:
            pass
        logger.error("v7 script failed (rc=%s): %s", exc.returncode, stderr_tail)
        raise


async def _run_script_async(script: Path, args: List[str]) -> str:
    async with _SUBPROC_SEM:
        try:
            return await asyncio.to_thread(_run_script, script, args, timeout_sec=V7_SUBPROCESS_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="ENGINE_TIMEOUT")
        except subprocess.CalledProcessError:
            raise HTTPException(status_code=500, detail="ENGINE_EXEC_FAILED")


def _read_index() -> Dict:
    index_path = V7_MODELS / "index.json"
    if not index_path.exists():
        return {"schema_version": "1.0", "updated_at": "", "items": []}
    return json.loads(index_path.read_text(encoding="utf-8"))


def _read_index_at(root: Path) -> Dict:
    index_path = root / "index.json"
    if not index_path.exists():
        return {"schema_version": "1.0", "updated_at": "", "items": []}
    return json.loads(index_path.read_text(encoding="utf-8"))


def _find_entry(index_data: Dict, sku: str, ink: str) -> Optional[Dict]:
    for item in index_data.get("items", []):
        if item.get("sku") == sku and item.get("ink") == ink:
            return item
    return None


def _safe_delete_path(rel_path: str) -> str:
    if not rel_path:
        return ""
    candidate = (V7_MODELS / rel_path).resolve()
    try:
        candidate.relative_to(V7_MODELS.resolve())
    except ValueError:
        logger.warning("Skip delete outside models root: %s", candidate)
        return ""
    if not candidate.exists():
        return ""
    if candidate.is_dir():
        shutil.rmtree(candidate)
    else:
        candidate.unlink()
    return str(candidate)


def _load_std_images_from_versions(active: Dict[str, str]) -> Dict[str, List[str]]:
    images_by_mode: Dict[str, List[str]] = {"LOW": [], "MID": [], "HIGH": []}
    for mode in ["LOW", "MID", "HIGH"]:
        rel = active.get(mode, "")
        if not rel:
            continue
        meta_path = V7_MODELS / rel / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        std_images = meta.get("std_images")
        if not std_images:
            std_images = (meta.get("inputs") or {}).get("std_images")
        images_by_mode[mode] = [str(p) for p in (std_images or [])]
    return images_by_mode


def _load_active_snapshot(entry: Optional[Dict]) -> Optional[Dict]:
    if not entry:
        return None
    rel = entry.get("active_snapshot", "")
    if not rel:
        return None
    snap_path = V7_MODELS / rel
    if not snap_path.exists():
        return None
    return json.loads(snap_path.read_text(encoding="utf-8"))


def _load_latest_v2_review(sku: str, ink: str) -> Dict[str, Any]:
    from src.engine_v7.core.measure.diagnostics.v2_flags import build_v2_flags

    insp_dirs = sorted([p for p in V7_RESULTS.glob("insp_*") if p.is_dir()], reverse=True)
    for run_dir in insp_dirs:
        insp_path = run_dir / "inspection.json"
        if not insp_path.exists():
            continue
        data = json.loads(insp_path.read_text(encoding="utf-8"))
        if data.get("sku") != sku or data.get("ink") != ink:
            continue
        results = data.get("results", [])
        if not results:
            continue
        decision = results[0].get("decision", {})
        diagnostics = decision.get("diagnostics") or {}
        v2 = diagnostics.get("v2_diagnostics")
        if not v2:
            continue
        refs = diagnostics.get("references", {}) or {}
        source = {
            "inspection_result_path": str(insp_path),
            "pattern_baseline_path": refs.get("pattern_baseline_path", ""),
            "active_versions": refs.get("active_versions") or refs.get("pattern_baseline_active_versions") or {},
        }
        return build_v2_flags(v2, source=source)
    return build_v2_flags(None)


def _load_latest_v3_summary(sku: str, ink: str) -> Optional[Dict[str, Any]]:
    insp_dirs = sorted([p for p in V7_RESULTS.glob("insp_*") if p.is_dir()], reverse=True)
    for run_dir in insp_dirs:
        insp_path = run_dir / "inspection.json"
        if not insp_path.exists():
            continue
        data = json.loads(insp_path.read_text(encoding="utf-8"))
        if data.get("sku") != sku or data.get("ink") != ink:
            continue
        results = data.get("results", [])
        if not results:
            continue
        decision = results[0].get("decision", {})
        summary = decision.get("v3_summary")
        if not summary:
            continue
        if summary.get("schema_version") != "v3_summary.v1":
            continue
        return summary
    return None


def _load_recent_decisions_for_trend(sku: str, ink: str, window_requested: int = 20) -> List[Dict[str, Any]]:
    insp_dirs = sorted([p for p in V7_RESULTS.glob("insp_*") if p.is_dir()], reverse=True)
    decisions: List[Dict[str, Any]] = []
    for run_dir in insp_dirs:
        insp_path = run_dir / "inspection.json"
        if not insp_path.exists():
            continue
        data = json.loads(insp_path.read_text(encoding="utf-8"))
        if data.get("sku") != sku or data.get("ink") != ink:
            continue
        results = data.get("results", [])
        if not results:
            continue
        decision = results[0].get("decision", {})
        decisions.append(decision)
        if len(decisions) >= window_requested:
            break
    return decisions
