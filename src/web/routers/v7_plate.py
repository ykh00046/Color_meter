"""
V7 Plate Gate & Calibration sub-router.

Routes:
  POST /plate_gate          - Extract plate gate masks
  POST /intrinsic_calibrate - Calibrate intrinsic color references
  POST /intrinsic_simulate  - Simulate lens appearance (Beer-Lambert)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel

from src.config.v7_paths import V7_ROOT
from src.utils.security import validate_file_extension, validate_file_size

from .v7_helpers import (
    _atomic_write_json,
    _compute_center_crop_mean_rgb,
    _load_cfg,
    _require_role,
    _resolve_cfg_path,
    logger,
)

router = APIRouter()


class PlateGateRequest(BaseModel):
    """Request model for plate gate extraction."""

    sku: Optional[str] = None


def _generate_plate_pair_artifacts(
    white_bgr: np.ndarray,
    black_bgr: np.ndarray,
    run_dir: Path,
    basename: str,
    cfg: Dict[str, Any],
    geom_hint: Optional[Any] = None,
) -> Dict[str, str]:
    from src.engine_v7.core.plate.plate_engine import compute_plate_artifacts

    artifacts: Dict[str, str] = {}
    try:
        plate_debug = compute_plate_artifacts(white_bgr, black_bgr, cfg, geom_hint=geom_hint)
        white_norm = plate_debug.get("white_norm")
        black_aligned = plate_debug.get("black_aligned")
        alpha = plate_debug.get("alpha")
        masks = plate_debug.get("masks", {})
        ink_masks = plate_debug.get("ink_masks", {})

        if white_norm is None or alpha is None or not masks:
            return artifacts

        alpha_img = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        alpha_color = cv2.applyColorMap(alpha_img, cv2.COLORMAP_JET)
        alpha_path = run_dir / f"{basename}_plate_alpha.png"
        cv2.imwrite(str(alpha_path), alpha_color)
        artifacts["plate_alpha"] = f"/v7_results/{run_dir.name}/{alpha_path.name}"

        base = white_norm.copy()
        overlay = base.copy()
        ring = masks.get("ring")
        dot = masks.get("dot")
        clear = masks.get("clear")

        if ring is not None:
            overlay[ring] = (0, 0, 255)
        if dot is not None:
            overlay[dot] = (0, 255, 0)
        if clear is not None:
            overlay[clear] = (255, 0, 0)

        blended = cv2.addWeighted(base, 0.65, overlay, 0.35, 0)

        overlay_path = run_dir / f"{basename}_plate_masks.png"
        cv2.imwrite(str(overlay_path), blended)
        artifacts["plate_masks"] = f"/v7_results/{run_dir.name}/{overlay_path.name}"

        white_bg = np.full_like(base, 255)
        black_bg = np.zeros_like(base)

        def _write_mask_composites(name: str, mask: np.ndarray) -> None:
            if mask is None:
                return
            comp_white = white_bg.copy()
            comp_black = black_bg.copy()
            comp_white[mask] = base[mask]
            comp_black[mask] = base[mask]

            white_path = run_dir / f"{basename}_plate_{name}_white.png"
            black_path = run_dir / f"{basename}_plate_{name}_black.png"
            cv2.imwrite(str(white_path), comp_white)
            cv2.imwrite(str(black_path), comp_black)
            artifacts[f"plate_{name}_white"] = f"/v7_results/{run_dir.name}/{white_path.name}"
            artifacts[f"plate_{name}_black"] = f"/v7_results/{run_dir.name}/{black_path.name}"

        _write_mask_composites("ring", ring)
        _write_mask_composites("dot", dot)

        if clear is not None:
            clear_mask = clear.astype(np.uint8) * 255
            clear_path = run_dir / f"{basename}_plate_clear_mask.png"
            cv2.imwrite(str(clear_path), clear_mask)
            artifacts["plate_clear_mask"] = f"/v7_results/{run_dir.name}/{clear_path.name}"

        if black_aligned is not None and ink_masks:

            def _render(src: np.ndarray, mask: np.ndarray, bg: Tuple[int, int, int]) -> np.ndarray:
                out = np.full_like(src, bg)
                out[mask] = src[mask]
                return out

            for ink_id, mask in ink_masks.items():
                if mask is None or not mask.any():
                    continue
                w_on_w = _render(white_norm, mask, (255, 255, 255))
                w_on_b = _render(white_norm, mask, (0, 0, 0))
                b_on_w = _render(black_aligned, mask, (255, 255, 255))
                b_on_b = _render(black_aligned, mask, (0, 0, 0))

                pairs = {
                    f"{ink_id}_from_white_on_white": w_on_w,
                    f"{ink_id}_from_white_on_black": w_on_b,
                    f"{ink_id}_from_black_on_white": b_on_w,
                    f"{ink_id}_from_black_on_black": b_on_b,
                }
                for key, img in pairs.items():
                    out_path = run_dir / f"{basename}_{key}.png"
                    cv2.imwrite(str(out_path), img)
                    artifacts[key] = f"/v7_results/{run_dir.name}/{out_path.name}"
    except Exception as exc:
        logger.warning("Failed to generate plate artifacts: %s", exc)

    return artifacts


def _polar_mask_to_base64(polar_mask: np.ndarray) -> str:
    """
    Convert polar boolean mask to Base64 PNG image for UI visualization.

    Args:
        polar_mask: (T, R) boolean array

    Returns:
        Base64 encoded PNG string
    """
    import base64

    if polar_mask is None or polar_mask.size == 0:
        return ""

    # Boolean mask to uint8 image (0 or 255)
    img_array = polar_mask.astype(np.uint8) * 255

    ok, buf = cv2.imencode(".png", img_array)
    if not ok:
        return ""

    img_base64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    return img_base64


@router.post("/plate_gate")
async def extract_plate_gate_api(
    x_user_role: Optional[str] = Header(default=""),
    sku: Optional[str] = Form(None),
    white_file: UploadFile = File(...),
    black_file: UploadFile = File(...),
    include_images: bool = Form(False),
):
    """
    Lightweight Plate Gate extraction for Hard Gate sampling visualization.

    This endpoint extracts ink masks from white/black backlight image pairs
    without running full plate analysis. Useful for:
    - Quick gate quality validation
    - Mask visualization in UI
    - Pre-inspection checks
    """
    _require_role("operator", x_user_role)

    # Validate extensions (must return True)
    if not validate_file_extension(white_file.filename, [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]):
        raise HTTPException(status_code=400, detail="Invalid white_file extension")
    if not validate_file_extension(black_file.filename, [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]):
        raise HTTPException(status_code=400, detail="Invalid black_file extension")

    # Read bytes then validate size
    white_bytes = await white_file.read()
    black_bytes = await black_file.read()

    if not validate_file_size(len(white_bytes), max_size_mb=50):
        raise HTTPException(status_code=413, detail="white_file too large")
    if not validate_file_size(len(black_bytes), max_size_mb=50):
        raise HTTPException(status_code=413, detail="black_file too large")

    white_arr = np.frombuffer(white_bytes, np.uint8)
    black_arr = np.frombuffer(black_bytes, np.uint8)

    white_bgr = cv2.imdecode(white_arr, cv2.IMREAD_COLOR)
    black_bgr = cv2.imdecode(black_arr, cv2.IMREAD_COLOR)

    MAX_DIM = 8192
    for name, img in [("white_file", white_bgr), ("black_file", black_bgr)]:
        if img is None:
            raise HTTPException(status_code=400, detail=f"Failed to decode {name}")
        h, w = img.shape[:2]
        if h > MAX_DIM or w > MAX_DIM:
            raise HTTPException(status_code=413, detail=f"{name} too large: {w}x{h}")

    # Load configuration
    cfg = _load_cfg(sku)

    # Import and run plate gate extraction
    try:
        from src.engine_v7.core.plate.plate_gate import extract_plate_gate
    except Exception:
        logger.exception("Plate gate import failed")
        raise HTTPException(status_code=500, detail="Gate module import failed")

    try:
        gate_result = extract_plate_gate(white_bgr, black_bgr, cfg, include_polar_masks=include_images)
    except Exception as e:
        logger.exception("Plate gate extraction failed")
        raise HTTPException(status_code=500, detail="Gate extraction failed")

    # Extract results
    gate_quality = gate_result.get("gate_quality", {})
    registration = gate_result.get("registration", {})
    geom = gate_result.get("geom")

    # Convert masks to Base64 images
    ink_mask_polar = gate_result.get("ink_mask_core_polar")
    valid_polar = gate_result.get("valid_polar")

    if include_images:
        ink_mask_image = _polar_mask_to_base64(ink_mask_polar) if ink_mask_polar is not None else ""
        valid_polar_image = _polar_mask_to_base64(valid_polar) if valid_polar is not None else ""
    else:
        ink_mask_image = ""
        valid_polar_image = ""

    usable = bool(gate_quality.get("usable", False))
    reason = gate_quality.get("reason")
    if not reason:
        reason = "OK" if usable else "UNKNOWN"

    # Build response
    response = {
        "schema_version": "plate_gate.v1",
        "usable": usable,
        "artifact_ratio": float(gate_quality.get("artifact_ratio", 0.0)),
        "reason": reason,
        "raw_ink_sum": gate_quality.get("raw_ink_sum"),
        "raw_ink_area_ratio": gate_quality.get("raw_ink_area_ratio"),
        "pupil_leak_ratio": gate_quality.get("pupil_leak_ratio"),
        "bg_leak_ratio": gate_quality.get("bg_leak_ratio"),
        "quality_ok": gate_quality.get("quality_ok"),
        "quality_warns": gate_quality.get("quality_warns"),
        "pair_edge_iou": gate_quality.get("pair_edge_iou"),
        "pair_lf_cos": gate_quality.get("pair_lf_cos"),
        "pair_ok": gate_quality.get("pair_ok"),
        "pair_ok_hard": gate_quality.get("pair_ok_hard"),
        "pair_ok_soft": gate_quality.get("pair_ok_soft"),
        "registration": {
            "method": registration.get("method", "unknown"),
            "swapped": bool(registration.get("swapped", False)),
        },
        "geom": (
            {
                "cx": float(geom.cx),
                "cy": float(geom.cy),
                "r": float(geom.r),
            }
            if geom
            else None
        ),
        "ink_mask_polar_image": ink_mask_image,
        "valid_polar_image": valid_polar_image,
        "mask_shape": list(ink_mask_polar.shape) if include_images and ink_mask_polar is not None else None,
        "debug": {
            "gate_result_keys": sorted(list(gate_result.keys())),
            "has_gate_quality": "gate_quality" in gate_result,
        },
    }

    return response


@router.post("/intrinsic_calibrate")
async def intrinsic_calibrate_api(
    x_user_role: Optional[str] = Header(default=""),
    sku: Optional[str] = Form(None),
    white_file: UploadFile = File(...),
    black_file: UploadFile = File(...),
    center_crop: float = Form(0.5),
    mode: str = Form("FIXED"),
    gamma: Optional[float] = Form(None),
):
    """
    Calibrate intrinsic color references from empty white/black plate images.
    """
    _require_role("operator", x_user_role)

    allowed_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]

    if not validate_file_extension(white_file.filename, allowed_exts):
        raise HTTPException(status_code=400, detail="Invalid white_file extension")
    if not validate_file_extension(black_file.filename, allowed_exts):
        raise HTTPException(status_code=400, detail="Invalid black_file extension")

    white_bytes = await white_file.read()
    black_bytes = await black_file.read()
    if not validate_file_size(len(white_bytes), max_size_mb=50):
        raise HTTPException(status_code=413, detail="White calibration image too large")
    if not validate_file_size(len(black_bytes), max_size_mb=50):
        raise HTTPException(status_code=413, detail="Black calibration image too large")
    white_arr = np.frombuffer(white_bytes, np.uint8)
    black_arr = np.frombuffer(black_bytes, np.uint8)
    white_bgr = cv2.imdecode(white_arr, cv2.IMREAD_COLOR)
    black_bgr = cv2.imdecode(black_arr, cv2.IMREAD_COLOR)
    if white_bgr is None or black_bgr is None:
        raise HTTPException(status_code=400, detail="Failed to decode calibration images")

    ref_white = _compute_center_crop_mean_rgb(white_bgr, center_crop)
    ref_black = _compute_center_crop_mean_rgb(black_bgr, center_crop)

    cfg_path = _resolve_cfg_path(sku)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    intrinsic_cfg = cfg.get("intrinsic_color", {})
    if not isinstance(intrinsic_cfg, dict):
        intrinsic_cfg = {}

    gamma_used = float(gamma) if gamma is not None else float(intrinsic_cfg.get("gamma", 2.2))

    intrinsic_cfg.update(
        {
            "mode": str(mode).upper(),
            "ref_white_srgb": ref_white,
            "ref_black_srgb": ref_black,
            "gamma": gamma_used,
            "updated_at": datetime.now().isoformat(),
        }
    )
    cfg["intrinsic_color"] = intrinsic_cfg
    _atomic_write_json(cfg_path, cfg)

    return {
        "config_path": str(cfg_path),
        "intrinsic_color": intrinsic_cfg,
        "center_crop": center_crop,
    }


@router.post("/intrinsic_simulate")
async def intrinsic_simulate_api(
    x_user_role: Optional[str] = Header(default=""),
    k_rgb: str = Form(...),
    bg_srgb: str = Form(...),
    thickness: float = Form(1.0),
    gamma: float = Form(2.2),
):
    """
    Simulate lens appearance using Beer-Lambert k values.
    """
    _require_role("operator", x_user_role)

    try:
        k_vals = json.loads(k_rgb)
        bg_vals = json.loads(bg_srgb)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON for k_rgb or bg_srgb")

    if not (isinstance(k_vals, list) and isinstance(bg_vals, list) and len(k_vals) == 3 and len(bg_vals) == 3):
        raise HTTPException(status_code=400, detail="k_rgb and bg_srgb must be length-3 arrays")

    from src.engine_v7.core.measure.metrics.intrinsic_color import simulate_physical

    simulated = simulate_physical(
        np.array(k_vals, dtype=np.float32),
        np.array(bg_vals, dtype=np.float32),
        gamma=float(gamma),
        thickness=float(thickness),
    )
    t_preview = np.exp(-np.array(k_vals, dtype=np.float32) * float(thickness))

    return {
        "simulated_srgb": simulated.tolist(),
        "thickness": float(thickness),
        "transmittance_preview": np.round(t_preview, 6).tolist(),
    }
