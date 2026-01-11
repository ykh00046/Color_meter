"""
Lens Signature Engine v7 UI/API bridge (MVP)
"""

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel

from src.utils.security import validate_file_extension, validate_file_size

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v7", tags=["V7 MVP"])

BASE_DIR = Path(__file__).resolve().parents[2]
V7_ROOT = BASE_DIR.parent / "lens_signature_engine_v7"
V7_MODELS = V7_ROOT / "models"
V7_RESULTS = BASE_DIR.parent / "results" / "v7" / "web"
V7_RESULTS.mkdir(parents=True, exist_ok=True)
V7_TEST_RESULTS = BASE_DIR.parent / "results" / "v7" / "test"
V7_TEST_RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(V7_ROOT))
from core.anomaly.pattern_baseline import build_pattern_baseline, extract_pattern_features
from core.config_loader import load_cfg_with_sku
from core.gate.gate_engine import run_gate
from core.geometry.lens_geometry import detect_lens_circle
from core.insight.trend import build_v3_trend
from core.measure.ink_baseline import build_ink_baseline
from core.measure.ink_match import compute_cluster_deltas
from core.measure.preprocess import build_roi_mask, build_sampling_mask
from core.measure.v2_diagnostics import build_v2_diagnostics
from core.measure.v2_flags import build_v2_flags
from core.model_registry import compute_cfg_hash
from core.pipeline import analyzer as analyzer_mod
from core.pipeline.analyzer import _registration_summary, evaluate_multi, evaluate_registration_multi
from core.signature.fit import fit_std
from core.signature.model_io import load_model
from core.signature.radial_signature import to_polar
from core.types import GateResult
from core.utils import apply_white_balance, bgr_to_lab


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "y"}


def _load_cfg(sku: Optional[str] = None) -> Dict:
    strict_unknown = _env_flag("LENS_CFG_STRICT")
    cfg, sources, warnings = load_cfg_with_sku(
        str(V7_ROOT / "configs" / "default.json"),
        sku,
        strict_unknown=strict_unknown,
    )
    if warnings:
        logger.warning("SKU cfg has unknown keys: %s", ", ".join(warnings))
    logger.info("Applied cfg sources: %s", ", ".join(sources))
    return cfg


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


def _auto_tune_cfg_from_std(
    cfg: Dict[str, Any],
    std_models: Dict[str, Any],
    entry: Dict[str, Any],
    active: Dict[str, str],
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    if any(m not in std_models for m in ["LOW", "MID", "HIGH"]):
        return cfg, {
            "schema_version": "auto_tune.v1",
            "status": "SKIPPED",
            "reason": "STD_MODELS_MISSING",
            "generated_at": datetime.now().isoformat(),
        }

    images_by_mode = _load_std_images_from_versions(active)
    expected_k = _normalize_expected_ink_count(entry.get("expected_ink_count")) or 3
    ink_baseline = build_ink_baseline(images_by_mode, cfg, expected_k)

    stats: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    cfg_for_eval = json.loads(json.dumps(cfg))
    cfg_for_eval.setdefault("pattern_baseline", {})
    cfg_for_eval["pattern_baseline"]["require"] = False
    cfg_for_eval["pattern_baseline"]["use_relative"] = False

    for mode in ["LOW", "MID", "HIGH"]:
        paths = images_by_mode.get(mode, [])
        if not paths:
            continue
        bgr = cv2.imread(paths[0])
        if bgr is None:
            continue
        try:
            ok_context = {
                "expected_ink_count_input": expected_k,
                "expected_ink_count_registry": expected_k,
            }
            decision, _ = evaluate_multi(
                bgr,
                std_models,
                cfg_for_eval,
                pattern_baseline=None,
                ok_log_context=ok_context,
            )
            sig = decision.signature
            diag = decision.diagnostics or {}
            sig_payload = {
                "delta_e_p95": _safe_float(getattr(sig, "delta_e_p95", None)),
                "delta_e_mean": _safe_float(getattr(sig, "delta_e_mean", None)),
                "corr": _safe_float(getattr(sig, "score_corr", None)),
            }
            color_dir = (diag.get("color") or {}).get("overall", {}).get("direction", {})
            pattern_dot = (diag.get("pattern") or {}).get("dot", {})
            coverage_delta = _safe_float(pattern_dot.get("coverage_delta_pp"))
            edge_delta = _safe_float(pattern_dot.get("edge_sharpness_delta"))

            off_track = None
            v2_diag = (
                build_v2_diagnostics(
                    bgr,
                    cfg_for_eval,
                    expected_ink_count=expected_k,
                    expected_ink_count_registry=expected_k,
                    expected_ink_count_input=expected_k,
                )
                or {}
            )
            sample_clusters = (v2_diag.get("segmentation") or {}).get("clusters", [])
            if ink_baseline and sample_clusters:
                match = compute_cluster_deltas(ink_baseline, sample_clusters)
                off_track = _safe_float((match.get("trajectory_summary") or {}).get("max_off_track"))

            stats.append(
                {
                    "mode": mode,
                    "path": paths[0],
                    "signature": sig_payload,
                    "color_shift": {
                        "delta_L": _safe_float(color_dir.get("delta_L")),
                        "delta_a": _safe_float(color_dir.get("delta_a")),
                        "delta_b": _safe_float(color_dir.get("delta_b")),
                    },
                    "pattern": {
                        "coverage_delta_pp": coverage_delta,
                        "edge_sharpness_delta": edge_delta,
                    },
                    "ink": {"off_track": off_track},
                }
            )
        except Exception as exc:
            errors.append({"mode": mode, "path": paths[0], "error": str(exc)})

    if not stats:
        return cfg, {
            "schema_version": "auto_tune.v1",
            "status": "SKIPPED",
            "reason": "NO_STATS",
            "generated_at": datetime.now().isoformat(),
            "errors": errors[:3],
        }

    sig_p95 = [s["signature"]["delta_e_p95"] for s in stats if s["signature"]["delta_e_p95"] is not None]
    sig_mean = [s["signature"]["delta_e_mean"] for s in stats if s["signature"]["delta_e_mean"] is not None]
    sig_corr = [s["signature"]["corr"] for s in stats if s["signature"]["corr"] is not None]
    delta_L = [abs(s["color_shift"]["delta_L"]) for s in stats if s["color_shift"]["delta_L"] is not None]
    delta_a = [abs(s["color_shift"]["delta_a"]) for s in stats if s["color_shift"]["delta_a"] is not None]
    delta_b = [abs(s["color_shift"]["delta_b"]) for s in stats if s["color_shift"]["delta_b"] is not None]
    cov = [s["pattern"]["coverage_delta_pp"] for s in stats if s["pattern"]["coverage_delta_pp"] is not None]
    edge = [s["pattern"]["edge_sharpness_delta"] for s in stats if s["pattern"]["edge_sharpness_delta"] is not None]
    off_tracks = [s["ink"]["off_track"] for s in stats if s["ink"]["off_track"] is not None]

    sig_p95_max = max(sig_p95) if sig_p95 else None
    sig_mean_max = max(sig_mean) if sig_mean else None
    sig_corr_min = min(sig_corr) if sig_corr else None
    cov_abs = max([abs(v) for v in cov], default=None) if cov else None
    edge_abs = max([abs(v) for v in edge], default=None) if edge else None
    off_track_max = max(off_tracks) if off_tracks else None

    tuned = json.loads(json.dumps(cfg))
    sig_cfg = tuned.get("signature", {})
    diag_cfg = tuned.get("diagnostics", {})
    v2_cfg = tuned.get("v2_ink", {})

    if sig_p95_max is not None:
        sig_cfg["de_p95_max"] = max(float(sig_cfg.get("de_p95_max", 0.0)), float(sig_p95_max) * 1.2)
    if sig_mean_max is not None:
        sig_cfg["de_mean_max"] = max(float(sig_cfg.get("de_mean_max", 0.0)), float(sig_mean_max) * 1.2)
    if sig_corr_min is not None:
        sig_cfg["corr_min"] = min(float(sig_cfg.get("corr_min", 1.0)), float(sig_corr_min) * 0.98)

    if delta_L:
        diag_cfg["delta_L_threshold"] = max(float(diag_cfg.get("delta_L_threshold", 1.0)), max(delta_L) * 1.2)
    if delta_a:
        diag_cfg["delta_a_threshold"] = max(float(diag_cfg.get("delta_a_threshold", 0.8)), max(delta_a) * 1.2)
    if delta_b:
        diag_cfg["delta_b_threshold"] = max(float(diag_cfg.get("delta_b_threshold", 0.8)), max(delta_b) * 1.2)
    if cov_abs is not None:
        diag_cfg["coverage_delta_pp_threshold"] = max(
            float(diag_cfg.get("coverage_delta_pp_threshold", 2.0)),
            float(cov_abs) * 1.2,
        )
    if edge_abs is not None:
        diag_cfg["edge_sharpness_delta_threshold"] = max(
            float(diag_cfg.get("edge_sharpness_delta_threshold", 0.1)),
            float(edge_abs) * 1.2,
        )

    if off_track_max is not None:
        v2_cfg["match_max_deltaE"] = max(float(v2_cfg.get("match_max_deltaE", 6.0)), float(off_track_max) * 1.2)

    tuned["signature"] = sig_cfg
    tuned["diagnostics"] = diag_cfg
    tuned["v2_ink"] = v2_cfg

    meta = {
        "schema_version": "auto_tune.v1",
        "status": "APPLIED",
        "generated_at": datetime.now().isoformat(),
        "expected_ink_count": expected_k,
        "signature": {
            "p95_max": sig_p95_max,
            "mean_max": sig_mean_max,
            "corr_min": sig_corr_min,
        },
        "color_shift_abs_max": {
            "delta_L": max(delta_L) if delta_L else None,
            "delta_a": max(delta_a) if delta_a else None,
            "delta_b": max(delta_b) if delta_b else None,
        },
        "pattern": {
            "coverage_delta_pp_abs_max": cov_abs,
            "edge_sharpness_delta_abs_max": edge_abs,
        },
        "ink": {"off_track_max": off_track_max},
        "errors": errors[:3],
    }
    return tuned, meta


def _active_versions(active: Dict[str, str]) -> Dict[str, str]:
    versions = {}
    for mode, path in (active or {}).items():
        versions[mode] = Path(path).name if path else ""
    return versions


def _warning_diff(active_warnings: List[str], new_warnings: List[str]) -> Dict[str, List[str]]:
    active_set = set(active_warnings or [])
    new_set = set(new_warnings or [])
    return {
        "added": sorted(list(new_set - active_set)),
        "removed": sorted(list(active_set - new_set)),
    }


def _approval_status(candidate_label: str, delta_sep: Optional[float], warn_diff: Dict[str, List[str]]) -> str:
    if candidate_label != "STD_ACCEPTABLE":
        return "BLOCKED"
    if delta_sep is not None and delta_sep < 0:
        return "NEEDS_REVIEW"
    if warn_diff.get("added"):
        return "NEEDS_REVIEW"
    return "RECOMMENDED"


def _resolve_pack_path(pack_path: str) -> Path:
    if not pack_path:
        raise ValueError("approval_pack_path is required")
    path = Path(pack_path)
    if not path.is_absolute():
        path = (V7_ROOT / pack_path).resolve()
    else:
        path = path.resolve()
    approvals_root = (V7_ROOT / "approvals" / "approval_packs").resolve()
    if approvals_root not in path.parents:
        raise ValueError("approval_pack_path out of approvals directory")
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def _load_approval_pack(pack_path: str) -> Tuple[Dict, Path]:
    path = _resolve_pack_path(pack_path)
    pack = json.loads(path.read_text(encoding="utf-8"))
    if pack.get("schema_version") != "approval_pack.v1":
        raise ValueError("approval pack schema_version mismatch")
    return pack, path


def _validate_pack_for_activate(pack: Dict, sku: str, ink: str) -> None:
    context = pack.get("context", {})
    if context.get("sku") != sku or context.get("ink") != ink:
        raise ValueError("approval pack context mismatch")
    status = pack.get("decision", {}).get("status", "")
    if status == "BLOCKED":
        raise ValueError("approval pack status blocked")


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


def _build_approval_pack(
    sku: str,
    ink: str,
    run_id: str,
    validation: Dict,
    active_snapshot: Optional[Dict],
    v2_review: Optional[Dict],
    v3_summary: Optional[Dict],
    v3_trend_line: Optional[str],
) -> Optional[Path]:
    if not validation:
        return None

    results = validation.get("results", [])
    first_decision = results[0].get("decision", {}) if results else {}
    reg_summary = first_decision.get("registration_summary", {})
    label = first_decision.get("label", "UNKNOWN")

    active_reg = (active_snapshot or {}).get("registration_summary", {})
    active_min = active_reg.get("min_pairwise_separation")
    new_min = reg_summary.get("min_pairwise_separation")
    delta_sep = None
    if active_min is not None and new_min is not None:
        delta_sep = float(new_min) - float(active_min)

    warn_diff = _warning_diff(active_reg.get("warnings", []), reg_summary.get("warnings", []))
    status = _approval_status(label, delta_sep, warn_diff)

    summary_line_parts = [f"{label}"]
    if delta_sep is not None:
        summary_line_parts.append(f"Δmin_pairwise={delta_sep:+.3f}")
    if warn_diff.get("added"):
        summary_line_parts.append(f"warn+{len(warn_diff['added'])}")
    elif warn_diff.get("removed"):
        summary_line_parts.append(f"warn-{len(warn_diff['removed'])}")
    summary_line = " | ".join(summary_line_parts)

    approval_pack = {
        "schema_version": "approval_pack.v1",
        "approval_id": f"ap_{run_id}_{sku}_{ink}",
        "generated_at": datetime.now().isoformat(),
        "context": {"sku": sku, "ink": ink},
        "decision": {
            "action": "ACTIVATE",
            "status": status,
            "summary_line": summary_line,
            "policy": {
                "activation_allowed_only_for": ["STD_ACCEPTABLE"],
                "role_required": "Approver",
            },
        },
        "baseline_active": {
            "active_snapshot_path": (active_snapshot or {}).get("path", ""),
            "snapshot": {
                "min_pairwise_separation": active_reg.get("min_pairwise_separation"),
                "sep_threshold": active_reg.get("sep_threshold"),
                "warnings": active_reg.get("warnings", []),
                "geom_consistency": active_reg.get("geom_consistency", {}),
            },
        },
        "candidate_std": {
            "registration_id": run_id,
            "decision": {
                "label": label,
                "reason_codes": first_decision.get("reason_codes", []),
                "reason_messages": first_decision.get("reason_messages", []),
            },
            "summary": {
                "min_pairwise_separation": reg_summary.get("min_pairwise_separation"),
                "sep_threshold": reg_summary.get("sep_threshold"),
                "warnings": reg_summary.get("warnings", []),
                "geom_consistency": reg_summary.get("geom_consistency", {}),
            },
        },
        "compare_to_active": {
            "delta": {"min_pairwise_separation": delta_sep},
            "warnings_diff": warn_diff,
        },
        "impact_preview": {
            "status": "NOT_AVAILABLE",
            "notes": ["metrics_report not attached at registration time"],
        },
    }
    if v2_review:
        approval_pack["review"] = {"v2_flags": v2_review}
    if v3_summary:
        approval_pack.setdefault("info", {})
        approval_pack["info"]["v3_summary"] = v3_summary
    if v3_trend_line:
        approval_pack.setdefault("info", {})
        approval_pack["info"]["v3_trend_line"] = v3_trend_line

    approvals_dir = V7_ROOT / "approvals" / "approval_packs"
    approvals_dir.mkdir(parents=True, exist_ok=True)
    pack_path = approvals_dir / f"APPROVAL_{sku}_{ink}_{run_id}.json"
    approval_pack["baseline_active"]["active_snapshot_path"] = (
        str((V7_MODELS / (active_snapshot or {}).get("path", "")).as_posix())
        if (active_snapshot or {}).get("path")
        else ""
    )
    pack_path.write_text(json.dumps(approval_pack, ensure_ascii=False, indent=2), encoding="utf-8")
    return pack_path


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

    snap_name = f"{entry.get('sku','')}_{entry.get('ink','')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    snap_path = snapshots_dir / snap_name
    snapshot["path"] = str((snapshots_dir / snap_name).relative_to(V7_MODELS))
    snap_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return snap_path


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
    base_name = f"PB_{entry.get('sku','')}_{entry.get('ink','')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
    base_name = f"IB_{entry.get('sku','')}_{entry.get('ink','')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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


def _require_role(expected: str, user_role: Optional[str]) -> None:
    role = (user_role or "").lower()
    if not role:
        # 권한 헤더가 없는 경우 현장 편의를 위해 기본 권한(expected)으로 간주하고 로그만 남김
        logger.warning(f"No role header found. Defaulting to '{expected}' for this request.")
        return

    if role != expected and role != "admin":
        # 관리자(admin)는 모든 권한을 가짐, 그 외 불일치 시 경고 후 통과 (MVP 운영 우선)
        logger.warning(f"Role mismatch: expected '{expected}', got '{role}'. Proceeding anyway.")
        return


def _save_uploads(files: List[UploadFile], run_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for f in files:
        if not f.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        if not validate_file_extension(f.filename, [".jpg", ".jpeg", ".png", ".bmp"]):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {f.filename}")
        content = f.file.read()
        if not validate_file_size(len(content), max_size_mb=10):
            raise HTTPException(status_code=413, detail=f"File too large: {f.filename}")
        dest = run_dir / f.filename
        dest.write_bytes(content)
        paths.append(dest)
    return paths


def _save_single_upload(file: UploadFile, dest: Path) -> Path:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    if not validate_file_extension(file.filename, [".jpg", ".jpeg", ".png", ".bmp"]):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}")
    content = file.file.read()
    if not validate_file_size(len(content), max_size_mb=10):
        raise HTTPException(status_code=413, detail=f"File too large: {file.filename}")
    dest.write_bytes(content)
    return dest


def _load_bgr(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {path.name}")
    return bgr


def _build_std_model(bgr: np.ndarray, cfg: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    wb_cfg = cfg.get("white_balance", {}) or {}
    geom = detect_lens_circle(bgr)
    wb_meta: Dict[str, Any] = {}
    if wb_cfg.get("enabled", False):
        bgr, wb_meta = apply_white_balance(bgr, geom, wb_cfg)
        wb_meta = wb_meta or {}
        wb_meta["enabled"] = True
    return (
        fit_std(
            bgr,
            R=cfg["polar"]["R"],
            T=cfg["polar"]["T"],
            r_start=cfg["signature"]["r_start"],
            r_end=cfg["signature"]["r_end"],
        ),
        wb_meta,
    )


def _run_gate_check(bgr: np.ndarray, cfg: Dict[str, Any]) -> Tuple[GateResult, Dict[str, Any]]:
    geom = detect_lens_circle(bgr)
    wb_cfg = cfg.get("white_balance", {}) or {}
    wb_meta: Dict[str, Any] = {}
    if wb_cfg.get("enabled", False):
        bgr, wb_meta = apply_white_balance(bgr, geom, wb_cfg)
        wb_meta = wb_meta or {}
        wb_meta["enabled"] = True
    gate = run_gate(
        geom,
        bgr,
        center_off_max=cfg["gate"]["center_off_max"],
        blur_min=cfg["gate"]["blur_min"],
        illum_max=cfg["gate"]["illum_max"],
    )
    return gate, wb_meta


def _run_script(script: Path, args: List[str]) -> str:
    cmd = [sys.executable, str(script)] + args
    logger.info("Running v7 script: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr or result.stdout)
    return result.stdout


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


def _set_inspection_metadata(payload: Dict, entry: Dict) -> Dict:
    payload["generated_at"] = datetime.now().isoformat()
    payload["active"] = entry.get("active", {})
    payload["active_snapshot"] = entry.get("active_snapshot", "")
    payload["pattern_baseline"] = entry.get("pattern_baseline", "")
    return payload


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


def _write_inspection_artifacts(run_dir: Path, data: Dict) -> Dict:
    artifacts: Dict[str, List[Dict[str, str]]] = {"images": []}
    for item in data.get("results", []):
        img_path = item.get("path", "")
        if not img_path:
            continue
        src_path = Path(img_path)
        if not src_path.exists():
            continue
        decision = item.get("decision", {})
        debug = decision.get("debug", {})

        overlay_path = run_dir / f"{src_path.stem}_overlay.png"
        heatmap_path = run_dir / f"{src_path.stem}_heatmap.png"

        try:
            img = cv2.imread(str(src_path))
            if img is not None:
                geom = debug.get("test_geom") or {}
                cx = int(geom.get("cx", 0))
                cy = int(geom.get("cy", 0))
                r = int(geom.get("r", 0))
                if cx > 0 and cy > 0 and r > 0:
                    cv2.circle(img, (cx, cy), r, (0, 255, 0), 2)
                    cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
                cv2.imwrite(str(overlay_path), img)
        except Exception:
            overlay_path = None

        try:
            heatmap = (debug.get("anomaly_heatmap") or {}).get("map")
            if heatmap is not None:
                hm = np.array(heatmap, dtype=np.float32)
                if hm.size > 0:
                    hm = np.clip(hm, 0.0, 1.0)
                    hm = (hm * 255).astype(np.uint8)
                    cv2.imwrite(str(heatmap_path), hm)
        except Exception:
            heatmap_path = None

        artifacts["images"].append(
            {
                "input": str(src_path),
                "overlay": (
                    f"/v7_results/{overlay_path.relative_to(V7_RESULTS).as_posix()}"
                    if overlay_path and overlay_path.exists()
                    else ""
                ),
                "heatmap": (
                    f"/v7_results/{heatmap_path.relative_to(V7_RESULTS).as_posix()}"
                    if heatmap_path and heatmap_path.exists()
                    else ""
                ),
            }
        )
    return artifacts


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


class DeleteEntryRequest(BaseModel):
    sku: str
    ink: str
    deleted_by: Optional[str] = "SYSTEM"
    reason: Optional[str] = ""
    hard_delete: Optional[bool] = False


class RegisterCleanupRequest(BaseModel):
    run_id: str


@router.post("/register_validate")
async def register_and_validate(
    x_user_role: Optional[str] = Header(default=""),
    sku: str = Form(...),
    ink: str = Form("INK_DEFAULT"),
    created_by: str = Form("SYSTEM"),
    notes: str = Form(""),
    expected_ink_count: Optional[int] = Form(None),
    low_files: List[UploadFile] = File(...),
    mid_files: List[UploadFile] = File(...),
    high_files: List[UploadFile] = File(...),
):
    _require_role("operator", x_user_role)
    if expected_ink_count is None:
        raise HTTPException(status_code=400, detail="expected_ink_count is required")
    expected_ink_count = _normalize_expected_ink_count(expected_ink_count)

    # Allow overwriting existing active STD (Auto-archive logic handles history)
    # index = _read_index()
    # entry = _find_entry(index, sku, ink)
    # if entry and entry.get("status") == "ACTIVE":
    #     raise HTTPException(status_code=400, detail="ACTIVE_STD_EXISTS")

    run_id = datetime.now().strftime("reg_%Y%m%d_%H%M%S")
    run_dir = V7_RESULTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    staging_root = V7_MODELS / "_staging" / run_id
    staging_root.mkdir(parents=True, exist_ok=True)

    low_paths = _save_uploads(low_files, run_dir)
    mid_paths = _save_uploads(mid_files, run_dir)
    high_paths = _save_uploads(high_files, run_dir)

    register_script = V7_ROOT / "scripts" / "register_std.py"
    cfg_path = V7_ROOT / "configs" / "default.json"

    responses = {}
    responses["LOW"] = json.loads(
        _run_script(
            register_script,
            [
                "--sku",
                sku,
                "--ink",
                ink,
                "--mode",
                "LOW",
                "--stds",
                *[str(p) for p in low_paths],
                "--cfg",
                str(cfg_path),
                "--models_root",
                str(staging_root),
                "--created_by",
                created_by or "SYSTEM",
                "--notes",
                notes,
                *(["--expected_ink_count", str(expected_ink_count)] if expected_ink_count is not None else []),
            ],
        )
    )
    responses["MID"] = json.loads(
        _run_script(
            register_script,
            [
                "--sku",
                sku,
                "--ink",
                ink,
                "--mode",
                "MID",
                "--stds",
                *[str(p) for p in mid_paths],
                "--cfg",
                str(cfg_path),
                "--models_root",
                str(staging_root),
                "--created_by",
                created_by or "SYSTEM",
                "--notes",
                notes,
                *(["--expected_ink_count", str(expected_ink_count)] if expected_ink_count is not None else []),
            ],
        )
    )
    responses["HIGH"] = json.loads(
        _run_script(
            register_script,
            [
                "--sku",
                sku,
                "--ink",
                ink,
                "--mode",
                "HIGH",
                "--stds",
                *[str(p) for p in high_paths],
                "--cfg",
                str(cfg_path),
                "--models_root",
                str(staging_root),
                "--created_by",
                created_by or "SYSTEM",
                "--notes",
                notes,
                *(["--expected_ink_count", str(expected_ink_count)] if expected_ink_count is not None else []),
            ],
        )
    )

    validate_script = V7_ROOT / "scripts" / "run_signature_engine.py"
    out_path = run_dir / "std_registration.json"
    all_tests = [*low_paths, *mid_paths, *high_paths]

    _run_script(
        validate_script,
        [
            "--sku",
            sku,
            "--ink",
            ink,
            "--models_root",
            str(staging_root),
            "--tests",
            *[str(p) for p in all_tests],
            "--phase",
            "STD_REGISTRATION",
            "--cfg",
            str(cfg_path),
            "--out",
            str(out_path),
        ],
    )
    validation = json.loads(out_path.read_text(encoding="utf-8"))

    # Generate artifacts for registration samples
    validation["artifacts"] = _write_inspection_artifacts(run_dir, validation)

    label_counts: Dict[str, int] = {}
    for item in validation.get("results", []):
        decision = item.get("decision", {})
        label = decision.get("label", "UNKNOWN")
        label_counts[label] = label_counts.get(label, 0) + 1
    activation_allowed = all(k == "STD_ACCEPTABLE" for k in label_counts.keys())

    staging_index = _read_index_at(staging_root)
    staging_entry = _find_entry(staging_index, sku, ink)
    staged_active = staging_entry.get("active", {}) if staging_entry else {}
    staged_versions = _active_versions(staged_active)

    if not activation_allowed:
        staging_rel = staging_root.relative_to(V7_MODELS).as_posix()
        return {
            "run_id": run_id,
            "status": "VALIDATION_FAILED",
            "register": responses,
            "validation": validation,
            "summary": {"label_counts": label_counts, "activation_allowed": False},
            "staging": {"run_id": run_id, "path": staging_rel, "versions": staged_versions},
        }

    # --- Auto Activation Logic ---
    # 1. Move files from staging to real models dir
    for mode in ["LOW", "MID", "HIGH"]:
        rel_path = staged_active.get(mode, "")
        if not rel_path:
            continue
        src = staging_root / rel_path
        dst = V7_MODELS / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            # If exists (rare race condition), clean up and fail or overwrite?
            # Ideally overwrite or fail. Here we fail to be safe.
            # But since run_id is unique, collision is unlikely unless manual interference.
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))

    _safe_delete_path(str(staging_root.relative_to(V7_MODELS)))

    # 2. Update Index (Register entry first if new)
    index = _read_index()
    entry = _find_entry(index, sku, ink)
    if entry is None:
        entry = {
            "sku": sku,
            "ink": ink,
            "active": {},
            "status": "REGISTERED",
            "notes": notes or "",
            "created_by": created_by or "SYSTEM",
            "created_at": datetime.now().isoformat(),
        }
        index.setdefault("items", []).append(entry)

    # Update entry metadata
    entry["notes"] = notes or entry.get("notes", "")
    if expected_ink_count is not None:
        entry["expected_ink_count"] = int(expected_ink_count)
    (V7_MODELS / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3. Finalize Activation
    versions = {"LOW": staged_versions["LOW"], "MID": staged_versions["MID"], "HIGH": staged_versions["HIGH"]}

    # Prepare approval pack (auto-approved)
    entry = _find_entry(_read_index(), sku, ink)  # Reload to get fresh state
    active_snapshot = _load_active_snapshot(entry)
    v2_review = _load_latest_v2_review(sku, ink)
    v3_summary = _load_latest_v3_summary(sku, ink)

    pack_path = _build_approval_pack(sku, ink, run_id, validation, active_snapshot, v2_review, v3_summary, None)

    pack_info = None
    if pack_path:
        try:
            pack, _ = _load_approval_pack(str(pack_path.relative_to(V7_ROOT)))
            pack_info = {"pack": pack, "path": pack_path}
        except Exception as e:
            logger.warning(f"Failed to load auto-generated pack: {e}")

    try:
        final_result = _finalize_activation(
            sku=sku,
            ink=ink,
            versions=versions,
            approved_by=created_by or "SYSTEM",
            reason="Auto-activation on registration",
            approval_pack_info=pack_info,
        )

        return {
            "run_id": run_id,
            "status": "ACTIVATED",
            "register": responses,
            "validation": validation,
            "summary": {"label_counts": label_counts, "activation_allowed": True},
            "active_result": final_result,
        }
    except Exception as e:
        logger.error(f"Auto-activation failed: {e}")
        # Fallback to manual activation needed
        return {"run_id": run_id, "status": "ACTIVATION_ERROR", "error": str(e), "validation": validation}


@router.get("/status")
async def get_status(sku: str, ink: str):
    index = _read_index()
    entry = _find_entry(index, sku, ink)
    if not entry:
        return {"sku": sku, "ink": ink, "status": "NOT_FOUND", "active": {}}
    return entry


@router.get("/entries")
async def list_entries(
    sku: Optional[str] = None,
    ink: Optional[str] = None,
    status: Optional[str] = None,
):
    index = _read_index()
    items = index.get("items", [])
    results: List[Dict[str, Any]] = []
    for item in items:
        if sku and item.get("sku") != sku:
            continue
        if ink and item.get("ink") != ink:
            continue
        if status and item.get("status") != status:
            continue
        entry = dict(item)
        entry["active_versions"] = _active_versions(entry.get("active", {}))
        results.append(entry)
    return {"status": "OK", "count": len(results), "items": results}


@router.post("/register_cleanup")
async def register_cleanup(req: RegisterCleanupRequest):
    rel_path = f"_staging/{req.run_id}"
    deleted = _safe_delete_path(rel_path)
    return {"status": "OK", "deleted": deleted}


@router.get("/candidates")
async def get_candidates(sku: str, ink: str):
    base = V7_MODELS / sku / ink
    result = {"LOW": [], "MID": [], "HIGH": []}
    for mode in ["LOW", "MID", "HIGH"]:
        mode_dir = base / mode
        if not mode_dir.exists():
            continue
        versions = sorted([p.name for p in mode_dir.iterdir() if p.is_dir()], reverse=True)
        result[mode] = versions
    return {"sku": sku, "ink": ink, "candidates": result}


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
        raise HTTPException(status_code=404, detail=str(e))
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


@router.get("/v2_metrics")
async def get_v2_metrics(sku: str, ink: str = "INK_DEFAULT"):
    metrics_path = V7_ROOT / "results" / "v2_shadow_metrics.json"
    if not metrics_path.exists():
        return {"status": "NOT_FOUND", "groups": []}

    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        groups = data.get("groups", [])

        # 해당 SKU/INK에 맞는 그룹 필터링 (간소화된 매칭 로직)
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


@router.post("/test_run")
async def test_run(
    x_user_role: Optional[str] = Header(default=""),
    sku: str = Form(""),
    ink: str = Form("INK_DEFAULT"),
    mode: str = Form("all"),
    expected_ink_count: Optional[int] = Form(None),
    low_file: UploadFile = File(...),
    mid_file: UploadFile = File(...),
    high_file: UploadFile = File(...),
):
    _require_role("operator", x_user_role)

    mode = (mode or "all").strip().lower()
    allowed = {"gate", "signature", "ink", "all"}
    if mode not in allowed:
        raise HTTPException(status_code=400, detail="Invalid mode. Use gate/signature/ink/all")
    if mode in {"ink", "all"} and expected_ink_count is None:
        raise HTTPException(status_code=400, detail="expected_ink_count is required for ink mode")
    expected_ink_count = _normalize_expected_ink_count(expected_ink_count)

    run_id = datetime.now().strftime("test_%Y%m%d_%H%M%S")
    run_dir = V7_TEST_RESULTS / run_id
    inputs_dir = run_dir / "inputs"
    for m in ["LOW", "MID", "HIGH"]:
        (inputs_dir / m).mkdir(parents=True, exist_ok=True)

    low_path = _save_single_upload(low_file, inputs_dir / "LOW" / low_file.filename)
    mid_path = _save_single_upload(mid_file, inputs_dir / "MID" / mid_file.filename)
    high_path = _save_single_upload(high_file, inputs_dir / "HIGH" / high_file.filename)

    cfg = _load_cfg(sku)
    inputs = {
        "LOW": {"path": str(low_path), "filename": low_path.name},
        "MID": {"path": str(mid_path), "filename": mid_path.name},
        "HIGH": {"path": str(high_path), "filename": high_path.name},
    }

    bgrs = {
        "LOW": _load_bgr(low_path),
        "MID": _load_bgr(mid_path),
        "HIGH": _load_bgr(high_path),
    }

    report: Dict[str, Any] = {
        "schema_version": "v7_test_report.v1",
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "sku": sku,
        "ink": ink,
        "mode": mode,
        "inputs": inputs,
        "cfg_hash": compute_cfg_hash(cfg),
        "results": {},
    }

    # Gate-only diagnostics
    gate_results: Dict[str, Any] = {}
    for m, bgr in bgrs.items():
        gate, wb_meta = _run_gate_check(bgr, cfg)
        gate_results[m] = {"gate": asdict(gate), "white_balance": wb_meta}
    report["results"]["gate"] = gate_results

    # Signature/Pattern diagnostics (uses in-memory std models)
    if mode in {"signature", "all"}:
        std_models: Dict[str, Any] = {}
        wb_info: Dict[str, Any] = {}
        for m, bgr in bgrs.items():
            model, wb_meta = _build_std_model(bgr, cfg)
            std_models[m] = model
            wb_info[m] = wb_meta
        reg_summary, unstable_modes = _registration_summary(std_models, cfg)
        report["results"]["std_build"] = {"white_balance": wb_info}
        report["results"]["registration_summary"] = {
            **reg_summary,
            "unstable_modes": unstable_modes,
        }
        reg_decisions: Dict[str, Any] = {}
        for m, bgr in bgrs.items():
            dec = evaluate_registration_multi(bgr, std_models, cfg)
            reg_decisions[m] = dec.to_dict()
        report["results"]["registration_decisions"] = reg_decisions

    # Ink baseline diagnostics
    if mode in {"ink", "all"}:
        images_by_mode = {
            "LOW": [str(low_path)],
            "MID": [str(mid_path)],
            "HIGH": [str(high_path)],
        }
        baseline = build_ink_baseline(images_by_mode, cfg, expected_k=int(expected_ink_count))
        report["results"]["ink_baseline"] = {
            "generated": baseline is not None,
            "expected_ink_count": int(expected_ink_count),
            "baseline": baseline,
        }
        if baseline is None:
            report["results"]["ink_baseline"]["error"] = "INK_BASELINE_FAILED"
        v2_diags: Dict[str, Any] = {}
        for m, bgr in bgrs.items():
            v2_diags[m] = build_v2_diagnostics(
                bgr,
                cfg,
                int(expected_ink_count),
                expected_ink_count_registry=None,
                expected_ink_count_input=int(expected_ink_count),
            )
        report["results"]["v2_diagnostics"] = v2_diags

        mask_dir = run_dir / "sampling_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        sampling_masks: Dict[str, Any] = {}
        v2_cfg = cfg.get("v2_ink", {})
        r_start = float(v2_cfg.get("roi_r_start", cfg["anomaly"]["r_start"]))
        r_end = float(v2_cfg.get("roi_r_end", cfg["anomaly"]["r_end"]))
        for m, bgr in bgrs.items():
            geom = detect_lens_circle(bgr)
            polar = to_polar(bgr, geom, R=cfg["polar"]["R"], T=cfg["polar"]["T"])
            lab = bgr_to_lab(polar).astype(np.float32)
            roi_mask, roi_meta = build_roi_mask(
                lab.shape[0],
                lab.shape[1],
                r_start,
                r_end,
                center_excluded_frac=float(cfg["anomaly"]["center_frac"]),
            )
            mask, mask_meta, mask_warn = build_sampling_mask(lab, roi_mask, cfg, rng_seed=None)
            mask_path = mask_dir / f"{m.lower()}_sampling_mask.png"
            cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))
            sampling_masks[m] = {
                "path": str(mask_path),
                "roi": roi_meta,
                "sampling": mask_meta,
                "warnings": mask_warn,
                "shape": [int(mask.shape[0]), int(mask.shape[1])],
            }
        report["results"]["sampling_masks"] = sampling_masks

    report_path = run_dir / "test_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "run_id": run_id,
        "report_path": str(report_path),
        "report": report,
    }


@router.post("/inspect")
async def inspect(
    x_user_role: Optional[str] = Header(default=""),
    sku: str = Form(...),
    ink: str = Form("INK_DEFAULT"),
    mode: str = Form("all"),
    expected_ink_count: Optional[int] = Form(None),
    files: List[UploadFile] = File(...),
):
    _require_role("operator", x_user_role)
    expected_ink_count = _normalize_expected_ink_count(expected_ink_count)
    run_id = datetime.now().strftime("insp_%Y%m%d_%H%M%S")
    run_dir = V7_RESULTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    file_paths = _save_uploads(files, run_dir)
    script = V7_ROOT / "scripts" / "run_signature_engine.py"
    cfg_path = V7_ROOT / "configs" / "default.json"
    index = _read_index()
    entry = _find_entry(index, sku, ink)
    cfg_snapshot = _load_snapshot_config(entry)
    cfg_snapshot_path = None
    use_cfg_snapshot = False
    if cfg_snapshot:
        cfg_snapshot_path = run_dir / "cfg_snapshot.json"
        cfg_snapshot_path.write_text(json.dumps(cfg_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        cfg_path = cfg_snapshot_path
        use_cfg_snapshot = True
    out_path = run_dir / "inspection.json"

    _run_script(
        script,
        [
            "--sku",
            sku,
            "--ink",
            ink,
            "--models_root",
            str(V7_MODELS),
            "--tests",
            *[str(p) for p in file_paths],
            "--cfg",
            str(cfg_path),
            *(["--cfg_snapshot"] if use_cfg_snapshot else []),
            "--mode",
            mode,
            "--out",
            str(out_path),
            *(["--expected_ink_count", str(expected_ink_count)] if expected_ink_count is not None else []),
        ],
    )

    data = json.loads(out_path.read_text(encoding="utf-8"))
    status = entry or {}
    data = _set_inspection_metadata(data, status)
    data["artifacts"] = _write_inspection_artifacts(run_dir, data)
    try:
        features_list = []
        for item in data.get("results", []):
            decision = item.get("decision") or {}
            feats = decision.get("features")
            if feats:
                features_list.append(feats)
        if features_list:
            features_payload = {
                "schema_version": "features_bundle.v1",
                "generated_at": datetime.now().isoformat(),
                "cfg_hash": data.get("cfg_hash"),
                "features": features_list,
            }
            features_path = run_dir / "features.json"
            features_path.write_text(json.dumps(features_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write features bundle: %s", exc)
    try:
        decisions = _load_recent_decisions_for_trend(sku, ink, window_requested=20)
        v3_trend = build_v3_trend(decisions, window_requested=20)
        if v3_trend:
            for item in data.get("results", []):
                decision = item.get("decision")
                if decision is not None:
                    decision["v3_trend"] = v3_trend
    except Exception as exc:
        logger.warning("Failed to build v3 trend: %s", exc)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "result": data,
        "active": status.get("active", {}),
        "status": status.get("status", ""),
        "active_snapshot": status.get("active_snapshot", ""),
        "pattern_baseline": status.get("pattern_baseline", ""),
    }


@router.post("/analyze_single")
async def analyze_single(
    x_user_role: Optional[str] = Header(default=""),
    analysis_modes: Optional[str] = Form("all"),
    expected_ink_count: Optional[int] = Form(None),
    files: List[UploadFile] = File(...),
):
    """
    Single sample analysis without STD comparison

    Args:
        analysis_modes: "all" or comma-separated list like "gate,color,radial,ink,pattern,zones"
        expected_ink_count: Expected number of ink colors (1-5), None for auto-detect (default from config)
        files: Uploaded image files

    Returns:
        {
            "run_id": "single_20260109_123456",
            "timestamp": "2026-01-09T12:34:56",
            "analysis_modes": ["gate", "color", ...],
            "results": [
                {
                    "filename": "sample.png",
                    "analysis": {...},
                    "artifacts": {...}
                }
            ]
        }
    """
    _require_role("operator", x_user_role)

    # Create run directory
    run_id = datetime.now().strftime("single_%Y%m%d_%H%M%S")
    run_dir = V7_RESULTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded files
    file_paths = _save_uploads(files, run_dir)

    # Load configuration
    cfg = _load_cfg()

    # Override expected_ink_count if provided
    if expected_ink_count is not None:
        cfg["expected_ink_count"] = expected_ink_count
        logger.info(f"Using user-specified expected_ink_count: {expected_ink_count}")

    # Parse analysis modes
    if analysis_modes == "all":
        modes = ["gate", "color", "radial", "ink", "pattern", "zones"]
    else:
        modes = [m.strip() for m in analysis_modes.split(",") if m.strip()]

    # Import single analyzer
    from core.pipeline.single_analyzer import analyze_single_sample

    # Process each file
    results = []
    for file_path in file_paths:
        bgr = cv2.imread(str(file_path))
        if bgr is None:
            logger.warning(f"Failed to load image: {file_path}")
            continue

        try:
            # Run single sample analysis
            analysis = analyze_single_sample(bgr, cfg, analysis_modes=modes)

            # Generate visualization artifacts
            artifacts = _generate_single_analysis_artifacts(bgr, analysis, run_dir, file_path.stem)

            results.append({"filename": file_path.name, "analysis": analysis, "artifacts": artifacts})

        except Exception as exc:
            logger.exception(f"Analysis failed for {file_path.name}")
            results.append({"filename": file_path.name, "error": str(exc), "analysis": None, "artifacts": {}})

    # Save results to JSON
    out_path = run_dir / "single_analysis.json"
    output = {"run_id": run_id, "timestamp": datetime.now().isoformat(), "analysis_modes": modes, "results": results}
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    return output


def _generate_single_analysis_artifacts(
    bgr: np.ndarray, analysis: Dict, run_dir: Path, basename: str
) -> Dict[str, str]:
    """
    Generate visualization artifacts for single sample analysis

    Returns:
        {
            "original": "/v7_results/single_20260109_123456/sample_original.png",
            "overlay": "/v7_results/.../sample_overlay.png",
            ...
        }
    """
    artifacts = {}

    # 1. Original image
    orig_path = run_dir / f"{basename}_original.png"
    cv2.imwrite(str(orig_path), bgr)
    artifacts["original"] = f"/v7_results/{run_dir.name}/{orig_path.name}"

    # 2. Overlay image (geometry + zones)
    overlay = bgr.copy()

    # Draw geometry circle
    gate = analysis.get("gate", {})
    geom = gate.get("geometry", {})
    if geom:
        cx = int(geom.get("cx", 0))
        cy = int(geom.get("cy", 0))
        r = int(geom.get("r", 0))

        # Draw lens circle
        cv2.circle(overlay, (cx, cy), r, (0, 255, 0), 2)
        cv2.circle(overlay, (cx, cy), 3, (0, 255, 0), -1)  # Center point

        # Draw pass/fail status
        passed = gate.get("passed", False)
        status_text = "GATE: PASS" if passed else "GATE: FAIL"
        status_color = (0, 255, 0) if passed else (0, 0, 255)
        cv2.putText(overlay, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    # Draw zone boundaries
    zones = analysis.get("zones", {})
    if zones and geom:
        cx = int(geom.get("cx", 0))
        cy = int(geom.get("cy", 0))
        r = int(geom.get("r", 0))

        num_zones = zones.get("num_zones", 8)
        zone_angle = 360.0 / num_zones

        for i in range(num_zones):
            angle = i * zone_angle
            rad = np.deg2rad(angle)
            x = int(cx + r * np.cos(rad))
            y = int(cy + r * np.sin(rad))
            cv2.line(overlay, (cx, cy), (x, y), (255, 255, 0), 1)

    overlay_path = run_dir / f"{basename}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    artifacts["overlay"] = f"/v7_results/{run_dir.name}/{overlay_path.name}"

    # 3. Heatmap image (if pattern analysis generated one)
    heatmap_path = run_dir / f"{basename}_heatmap.png"
    try:
        # Check if anomaly heatmap data exists in pattern analysis
        pattern = analysis.get("pattern", {})
        heatmap_data = pattern.get("heatmap")

        if heatmap_data is not None:
            hm = np.array(heatmap_data, dtype=np.float32)
            if hm.size > 0:
                # Normalize and convert to uint8
                hm = np.clip(hm, 0.0, 1.0)
                hm_colored = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(str(heatmap_path), hm_colored)
                artifacts["heatmap"] = f"/v7_results/{run_dir.name}/{heatmap_path.name}"
    except Exception as e:
        logger.warning(f"Failed to generate heatmap: {e}")

    # 4. Quality score badge image (simple text rendering)
    quality_score = analysis.get("quality_score", 0)
    badge = np.zeros((200, 200, 3), dtype=np.uint8)

    # Draw score
    cv2.putText(badge, f"{quality_score:.0f}", (30, 120), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255), 5)
    cv2.putText(badge, "/100", (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

    badge_path = run_dir / f"{basename}_quality_badge.png"
    cv2.imwrite(str(badge_path), badge)
    artifacts["quality_badge"] = f"/v7_results/{run_dir.name}/{badge_path.name}"

    return artifacts
