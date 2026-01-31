"""
V7 Registration sub-router.

Routes:
  POST /register_validate - Register STD images and validate
  GET  /status            - Get SKU/INK status
  GET  /entries           - List all entries
  POST /register_cleanup  - Clean up staging directory
  GET  /candidates        - Get version candidates for SKU/INK
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel

from src.config.v7_paths import REPO_ROOT, V7_MODELS, V7_RESULTS, V7_ROOT
from src.engine_v7.core.anomaly.pattern_baseline import build_pattern_baseline, extract_pattern_features
from src.engine_v7.core.gate.gate_engine import run_gate
from src.engine_v7.core.geometry.lens_geometry import detect_lens_circle
from src.engine_v7.core.measure.baselines.ink_baseline import build_ink_baseline
from src.engine_v7.core.measure.diagnostics.v2_diagnostics import build_v2_diagnostics
from src.engine_v7.core.measure.matching.ink_match import compute_cluster_deltas
from src.engine_v7.core.model_registry import compute_cfg_hash
from src.engine_v7.core.pipeline.analyzer import _registration_summary, evaluate_multi
from src.engine_v7.core.signature.fit import fit_std
from src.engine_v7.core.signature.model_io import load_model
from src.engine_v7.core.types import GateResult
from src.engine_v7.core.utils import apply_white_balance

from .v7_helpers import (
    _active_versions,
    _find_entry,
    _load_active_snapshot,
    _load_cfg,
    _load_latest_v2_review,
    _load_latest_v3_summary,
    _load_std_images_from_versions,
    _normalize_expected_ink_count,
    _read_index,
    _read_index_at,
    _require_role,
    _run_script_async,
    _safe_delete_path,
    _safe_float,
    _save_uploads,
    logger,
)

router = APIRouter()


class RegisterCleanupRequest(BaseModel):
    run_id: str


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
        summary_line_parts.append(f"?min_pairwise={delta_sep:+.3f}")
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

    run_id = datetime.now().strftime("reg_%Y%m%d_%H%M%S")
    run_dir = V7_RESULTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    staging_root = V7_MODELS / "_staging" / run_id
    staging_root.mkdir(parents=True, exist_ok=True)

    low_paths = await _save_uploads(low_files, run_dir)
    mid_paths = await _save_uploads(mid_files, run_dir)
    high_paths = await _save_uploads(high_files, run_dir)

    register_script = V7_ROOT / "scripts" / "register_std.py"
    cfg_path = V7_ROOT / "configs" / "default.json"

    responses = {}
    responses["LOW"] = json.loads(
        await _run_script_async(
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
        await _run_script_async(
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
        await _run_script_async(
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

    validate_script = REPO_ROOT / "scripts" / "engine_v7" / "run_signature_engine.py"
    out_path = run_dir / "std_registration.json"
    all_tests = [*low_paths, *mid_paths, *high_paths]

    await _run_script_async(
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
        from .v7_activation import _finalize_activation

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
