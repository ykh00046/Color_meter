"""
V7 Inspection sub-router.

Routes:
  POST /test_run       - Run diagnostic test on LOW/MID/HIGH samples
  POST /inspect        - Run full inspection pipeline
  POST /analyze_single - Single sample analysis without STD comparison
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile

from src.config.v7_paths import REPO_ROOT, V7_MODELS, V7_RESULTS, V7_ROOT, V7_TEST_RESULTS
from src.engine_v7.core.geometry.lens_geometry import detect_lens_circle
from src.engine_v7.core.insight.trend import build_v3_trend
from src.engine_v7.core.measure.baselines.ink_baseline import build_ink_baseline
from src.engine_v7.core.measure.diagnostics.v2_diagnostics import build_v2_diagnostics
from src.engine_v7.core.measure.segmentation.preprocess import build_roi_mask, build_sampling_mask
from src.engine_v7.core.model_registry import compute_cfg_hash
from src.engine_v7.core.pipeline.analyzer import _registration_summary, evaluate_registration_multi
from src.engine_v7.core.signature.radial_signature import to_polar
from src.engine_v7.core.utils import apply_white_balance, bgr_to_lab

from .v7_helpers import (
    NumpyEncoder,
    _find_entry,
    _load_bgr,
    _load_cfg,
    _load_recent_decisions_for_trend,
    _load_snapshot_config,
    _normalize_expected_ink_count,
    _parse_match_ids,
    _read_index,
    _require_role,
    _resolve_sku_for_ink,
    _run_script_async,
    _save_single_upload,
    _save_uploads,
    logger,
)
from .v7_plate import _generate_plate_pair_artifacts
from .v7_registration import _build_std_model, _run_gate_check, _write_inspection_artifacts

router = APIRouter()


def _set_inspection_metadata(payload: Dict, entry: Dict) -> Dict:
    payload["generated_at"] = datetime.now().isoformat()
    payload["active"] = entry.get("active", {})
    payload["active_snapshot"] = entry.get("active_snapshot", "")
    payload["pattern_baseline"] = entry.get("pattern_baseline", "")
    return payload


def _generate_single_analysis_artifacts(
    bgr: np.ndarray, analysis: Dict, run_dir: Path, basename: str
) -> Dict[str, str]:
    """
    Generate visualization artifacts for single sample analysis.
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

    low_path = await _save_single_upload(low_file, inputs_dir / "LOW" / low_file.filename)
    mid_path = await _save_single_upload(mid_file, inputs_dir / "MID" / mid_file.filename)
    high_path = await _save_single_upload(high_file, inputs_dir / "HIGH" / high_file.filename)

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
    sku_used = _resolve_sku_for_ink(sku, ink)
    expected_ink_count = _normalize_expected_ink_count(expected_ink_count)
    run_id = datetime.now().strftime("insp_%Y%m%d_%H%M%S")
    run_dir = V7_RESULTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    file_paths = await _save_uploads(files, run_dir)
    script = REPO_ROOT / "scripts" / "engine_v7" / "run_signature_engine.py"
    cfg_path = V7_ROOT / "configs" / "default.json"
    index = _read_index()
    entry = _find_entry(index, sku_used, ink)
    cfg_snapshot = _load_snapshot_config(entry)
    cfg_snapshot_path = None
    use_cfg_snapshot = False
    if cfg_snapshot:
        cfg_snapshot_path = run_dir / "cfg_snapshot.json"
        cfg_snapshot_path.write_text(json.dumps(cfg_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        cfg_path = cfg_snapshot_path
        use_cfg_snapshot = True
    out_path = run_dir / "inspection.json"

    # Map UI mode 'pattern' to engine mode 'signature'
    engine_mode = "signature" if mode == "pattern" else mode

    await _run_script_async(
        script,
        [
            "--sku",
            sku_used,
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
            engine_mode,
            "--out",
            str(out_path),
            *(["--expected_ink_count", str(expected_ink_count)] if expected_ink_count is not None else []),
        ],
    )

    data = json.loads(out_path.read_text(encoding="utf-8"))
    status = entry or {}
    data = _set_inspection_metadata(data, status)
    if sku_used != sku:
        data["sku_override"] = {"input": sku, "used": sku_used, "ink": ink}
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
    match_ids: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    black_files: Optional[List[UploadFile]] = File(None),
):
    """
    Single sample analysis without STD comparison.
    """
    _require_role("operator", x_user_role)

    # Create run directory
    run_id = datetime.now().strftime("single_%Y%m%d_%H%M%S")
    run_dir = V7_RESULTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded files
    white_dir = run_dir / "white"
    white_dir.mkdir(parents=True, exist_ok=True)
    file_paths = await _save_uploads(files, white_dir)
    black_paths: List[Path] = []
    if black_files:
        black_dir = run_dir / "black"
        black_dir.mkdir(parents=True, exist_ok=True)
        black_paths = await _save_uploads(black_files, black_dir)
        if len(black_paths) != len(file_paths):
            raise HTTPException(status_code=400, detail="files and black_files count mismatch")

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

    if black_files and "plate" not in modes:
        modes.append("plate")

    # Import single analyzer
    from src.engine_v7.core.pipeline.single_analyzer import analyze_single_sample

    # Process each file
    results = []
    if black_paths:
        match_list = _parse_match_ids(match_ids, len(file_paths)) if match_ids else None
        for idx, (white_path, black_path) in enumerate(zip(file_paths, black_paths)):
            white_bgr = cv2.imread(str(white_path))
            black_bgr = cv2.imread(str(black_path))
            if white_bgr is None or black_bgr is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to decode image pair: {white_path.name}, {black_path.name}",
                )
            match_id = match_list[idx] if match_list else f"{white_path.stem}_{black_path.stem}"

            try:
                analysis = analyze_single_sample(
                    white_bgr,
                    cfg,
                    analysis_modes=modes,
                    black_bgr=black_bgr,
                    match_id=match_id,
                )
                artifacts = _generate_single_analysis_artifacts(white_bgr, analysis, run_dir, white_path.stem)
                plate_artifacts = _generate_plate_pair_artifacts(
                    white_bgr,
                    black_bgr,
                    run_dir,
                    white_path.stem,
                    cfg,
                )
                artifacts.update(plate_artifacts)
                results.append(
                    {
                        "filename": white_path.name,
                        "black_filename": black_path.name,
                        "match_id": match_id,
                        "analysis": analysis,
                        "artifacts": artifacts,
                    }
                )
            except Exception as exc:
                logger.exception(f"Analysis failed for {white_path.name}")
                results.append(
                    {
                        "filename": white_path.name,
                        "black_filename": black_path.name,
                        "match_id": match_id,
                        "error": str(exc),
                        "analysis": None,
                        "artifacts": {},
                    }
                )
    else:
        for file_path in file_paths:
            bgr = cv2.imread(str(file_path))
            if bgr is None:
                raise HTTPException(status_code=400, detail=f"Failed to decode image: {file_path.name}")

            try:
                # Run single sample analysis
                analysis = analyze_single_sample(bgr, cfg, analysis_modes=modes)

                # Generate visualization artifacts
                artifacts = _generate_single_analysis_artifacts(bgr, analysis, run_dir, file_path.stem)

                results.append({"filename": file_path.name, "analysis": analysis, "artifacts": artifacts})

            except Exception as exc:
                logger.exception(f"Analysis failed for {file_path.name}")
                results.append({"filename": file_path.name, "error": str(exc), "analysis": None, "artifacts": {}})

    # Remove internal fields (not JSON-serializable, for in-memory use only)
    for r in results:
        if r.get("analysis") and isinstance(r["analysis"], dict):
            # Remove fields starting with '_' (e.g., _masks, _geom_internal)
            keys_to_remove = [k for k in r["analysis"].keys() if k.startswith("_")]
            for k in keys_to_remove:
                del r["analysis"][k]
            # Also check nested 'plate' section
            if "plate" in r["analysis"] and isinstance(r["analysis"]["plate"], dict):
                plate_keys = [k for k in r["analysis"]["plate"].keys() if k.startswith("_")]
                for k in plate_keys:
                    del r["analysis"]["plate"][k]

    # Save results to JSON
    out_path = run_dir / "single_analysis.json"
    output = {"run_id": run_id, "timestamp": datetime.now().isoformat(), "analysis_modes": modes, "results": results}
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, cls=NumpyEncoder), encoding="utf-8")

    # Convert to JSON-serializable format for FastAPI response
    serializable_output = json.loads(json.dumps(output, cls=NumpyEncoder))
    return serializable_output
