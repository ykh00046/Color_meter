"""
Lightweight Plate Gate Extraction Module

This module provides fast extraction of plate masks for Hard Gate sampling,
without the overhead of full plate analysis. This is split from plate_engine.py
for performance optimization.

Usage:
    from ..plate.plate_gate import extract_plate_gate

    gate_result = extract_plate_gate(white_bgr, black_bgr, cfg)
    plate_mask = gate_result["ink_mask_core_polar"]
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from ..config_norm import get_plate_cfg, get_polar_dims
from ..geometry.lens_geometry import detect_lens_circle
from ..signature.radial_signature import to_polar
from ..types import LensGeometry

# Import shared helpers
from ._helpers import mask_to_polar, mean_gray_center, mean_gray_outer, radial_mask, resize_to_square_centered

# ==============================================================================
# Main Gate Extraction Function
# ==============================================================================


def extract_plate_gate(
    white_bgr: np.ndarray,
    black_bgr: np.ndarray,
    cfg: Dict[str, Any],
    geom_hint: Optional[LensGeometry] = None,
    include_polar_masks: bool = True,
) -> Dict[str, Any]:
    """
    Fast extraction of plate masks for Hard Gate sampling.

    This is a lightweight version that extracts only the essential masks needed
    for Hard Gate sampling, without running full plate analysis.

    Args:
        white_bgr: White backlight image
        black_bgr: Black backlight image
        cfg: Configuration dictionary
        geom_hint: Optional pre-computed lens geometry

    Returns:
        {
            "ink_mask_core_polar": np.ndarray or None,  # (T, R) boolean mask
            "valid_polar": np.ndarray or None,          # (T, R) boolean valid mask
            "geom": LensGeometry,
            "gate_quality": {
                "usable": bool,
                "artifact_ratio": float,
                "reason": str (if not usable),
            },
            "registration": {
                "method": str,
                "swapped": bool,
            }
        }
    """
    # Handle missing inputs
    if black_bgr is None:
        return {
            "ink_mask_core_polar": None,
            "valid_polar": None,
            "geom": None,
            "gate_quality": {"usable": False, "reason": "no_black_image"},
            "registration": {},
        }

    plate_cfg = get_plate_cfg(cfg)
    out_size = int(plate_cfg.get("out_size", 512))

    def _pair_iou(valid_w: np.ndarray, valid_b: np.ndarray) -> float:
        vw = valid_w > 0
        vb = valid_b > 0
        inter = np.logical_and(vw, vb).sum()
        union = np.logical_or(vw, vb).sum()
        return float(inter / union) if union > 0 else 0.0

    def _circle_mask(size: int, cx: float, cy: float, r: float) -> np.ndarray:
        m = np.zeros((size, size), np.uint8)
        cv2.circle(m, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
        return m

    def _edge_map(bgr: np.ndarray, lens_mask: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_and(gray, gray, mask=lens_mask)
        gray = cv2.equalizeHist(gray)
        edges = cv2.Canny(gray, 40, 120)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, k, iterations=1)
        edges = cv2.bitwise_and(edges, edges, mask=lens_mask)
        return edges > 0

    def _env_bool(name: str) -> Optional[bool]:
        v = os.getenv(name)
        if v is None:
            return None
        return v.strip().lower() in ("1", "true", "yes", "y", "on")

    def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-6:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _lowfreq_vec(bgr: np.ndarray, lens_mask: np.ndarray, out_hw: int = 32) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        roi = cv2.bitwise_and(gray, gray, mask=lens_mask)
        roi = cv2.GaussianBlur(roi, (21, 21), 0)
        small = cv2.resize(roi, (out_hw, out_hw), interpolation=cv2.INTER_AREA)
        m_small = cv2.resize(lens_mask, (out_hw, out_hw), interpolation=cv2.INTER_NEAREST) > 0
        v = small[m_small].astype(np.float32)
        if v.size == 0:
            return np.zeros((out_hw * out_hw,), np.float32)
        mu = float(v.mean())
        sd = float(v.std())
        if sd < 1e-6:
            v = v - mu
        else:
            v = (v - mu) / sd
        full = np.zeros((out_hw, out_hw), np.float32)
        full[m_small] = v
        return full.ravel()

    # 1. Detect geometry
    if geom_hint is None:
        geom_w = detect_lens_circle(white_bgr)
        geom_b = None
        if _env_bool("V7_PLATE_GATE_GEOM_BOTH") or geom_w is None:
            geom_b = detect_lens_circle(black_bgr)
        else:
            geom_b = geom_w
    else:
        geom_w = geom_hint
        geom_b = geom_hint

    if geom_w is None and geom_b is not None:
        geom_w = geom_b
    if geom_b is None and geom_w is not None:
        geom_b = geom_w

    # Geometry detect failure guard (prevent 500)
    if geom_w is None or geom_b is None:
        return {
            "ink_mask_core_polar": None,
            "valid_polar": None,
            "geom": None,
            "gate_quality": {"usable": False, "reason": "geom_detect_failed"},
            "registration": {"method": "simple_brightness", "swapped": False},
        }

    # 2. Resize and normalize
    try:
        w_norm, g = resize_to_square_centered(white_bgr, geom_w, out_size=out_size)
        b_norm, _ = resize_to_square_centered(black_bgr, geom_b, out_size=out_size)
    except Exception:
        return {
            "ink_mask_core_polar": None,
            "valid_polar": None,
            "geom": None,
            "gate_quality": {"usable": False, "reason": "resize_failed"},
            "registration": {"method": "simple_brightness", "swapped": False},
        }

    # 3. Ensure correct order (white brighter than black)
    r_clear = float(plate_cfg.get("r_clear", 0.40))
    w_c = mean_gray_center(w_norm, g, r_clear)
    b_c = mean_gray_center(b_norm, g, r_clear)

    swapped = False
    if w_c < b_c:
        w_norm, b_norm = b_norm, w_norm
        swapped = True

    # 4. Compute alpha map (simplified - no spatial background)
    r_bg = float(plate_cfg.get("r_bg", 0.35))
    diff_min = float(plate_cfg.get("alpha_diff_min", 1.0))
    denom_min = float(plate_cfg.get("alpha_denom_min", 20.0))

    # Background estimation from outer region
    bg_mask = radial_mask(out_size, out_size, g, r_bg, 1.0)
    w_bg = w_norm.astype(np.float32)[bg_mask].mean(axis=0)
    b_bg = b_norm.astype(np.float32)[bg_mask].mean(axis=0)

    # Alpha computation
    w_f = w_norm.astype(np.float32)
    b_f = b_norm.astype(np.float32)
    diff = np.abs(w_f - w_bg - b_f + b_bg)
    diff = np.maximum(diff, diff_min)

    denom = w_f - b_f
    denom = np.maximum(denom, denom_min)

    alpha = diff / denom
    alpha = np.clip(alpha, 0.02, 0.98)
    alpha = np.mean(alpha, axis=2)  # Average across channels

    # 5. Create basic masks (simplified version)
    r_clear_actual = float(plate_cfg.get("r_clear", 0.40))
    pupil_mask = radial_mask(out_size, out_size, g, 0.0, r_clear_actual)

    # Simple threshold for ink mask
    alpha_threshold = float(plate_cfg.get("ink_t", plate_cfg.get("alpha_print_th", 0.25)))
    raw_ink = alpha > alpha_threshold
    ink_mask = raw_ink & (~pupil_mask)

    # 6. Convert to polar (optional)
    debug_pair_iou = bool(_env_bool("V7_PLATE_GATE_DEBUG_PAIR_IOU"))
    polar_R = polar_T = None
    ink_mask_core_polar = None
    valid_polar = None
    if include_polar_masks or debug_pair_iou:
        polar_R, polar_T = get_polar_dims(cfg)
        if include_polar_masks:
            ink_mask_core_polar = mask_to_polar(ink_mask, g, polar_R, polar_T)
        valid_polar = mask_to_polar(~pupil_mask, g, polar_R, polar_T)

    # 7. Quality metrics
    lens_mask = radial_mask(out_size, out_size, g, 0.0, 1.0)
    lens_area = int(lens_mask.sum())
    raw_ink_total = int(raw_ink.sum())
    raw_ink_in_lens = int((raw_ink & lens_mask).sum())
    raw_ink_area_ratio = float(raw_ink_in_lens / max(lens_area, 1))

    pupil_leak_ratio = float((raw_ink & pupil_mask).sum() / max(raw_ink_total, 1))
    bg_leak_ratio = float((raw_ink & (~lens_mask)).sum() / max(raw_ink_total, 1))
    artifact_ratio = pupil_leak_ratio
    bg_artifact_ratio = bg_leak_ratio

    pupil_leak_soft_max = float(plate_cfg.get("pupil_leak_soft_max", 0.125))
    pupil_leak_hard_max = float(plate_cfg.get("pupil_leak_hard_max", 0.135))
    bg_leak_soft_max = float(plate_cfg.get("bg_leak_soft_max", 0.425))
    bg_leak_hard_max = float(plate_cfg.get("bg_leak_hard_max", 0.450))
    raw_ink_area_soft_min = float(plate_cfg.get("raw_ink_area_soft_min", 0.75))
    raw_ink_area_hard_min = float(plate_cfg.get("raw_ink_area_hard_min", 0.70))

    quality_warnings = []
    if raw_ink_area_ratio < raw_ink_area_soft_min:
        quality_warnings.append("warn_ink_low")
    if bg_leak_ratio > bg_leak_soft_max:
        quality_warnings.append("warn_bg_leak")
    if pupil_leak_ratio > pupil_leak_soft_max:
        quality_warnings.append("warn_pupil_leak")
    quality_ok = len(quality_warnings) == 0
    pair_iou = None
    if debug_pair_iou and valid_polar is not None:
        w_pol = to_polar(w_norm, g, R=polar_R, T=polar_T)
        b_pol = to_polar(b_norm, g, R=polar_R, T=polar_T)
        w_gray = w_pol.mean(axis=2)
        b_gray = b_pol.mean(axis=2)
        pair_valid_min = float(plate_cfg.get("pair_valid_min", 1.0))
        valid_w = (w_gray > pair_valid_min) & valid_polar
        valid_b = (b_gray > pair_valid_min) & valid_polar
        pair_iou = _pair_iou(valid_w, valid_b)

    lens_mask = _circle_mask(out_size, g.cx, g.cy, g.r)
    ew = _edge_map(w_norm, lens_mask)
    eb = _edge_map(b_norm, lens_mask)
    inter = np.logical_and(ew, eb).sum()
    union = np.logical_or(ew, eb).sum()
    pair_edge_iou = float(inter / union) if union > 0 else 0.0

    lf_hw = int(plate_cfg.get("pair_lf_hw", 32))
    w_lf = _lowfreq_vec(w_norm, lens_mask, out_hw=lf_hw)
    b_lf = _lowfreq_vec(b_norm, lens_mask, out_hw=lf_hw)
    pair_lf_cos = _cos_sim(w_lf, b_lf)

    edge_hard_min = float(plate_cfg.get("pair_edge_iou_hard_min", 0.44))
    lf_min = float(plate_cfg.get("pair_lf_cos_min", 0.25))
    lf_max = float(plate_cfg.get("pair_lf_cos_max", 0.45))

    env_edge_hard = os.getenv("V7_PAIR_EDGE_IOU_HARD_MIN")
    env_lf_min = os.getenv("V7_PAIR_LF_COS_MIN")
    env_lf_max = os.getenv("V7_PAIR_LF_COS_MAX")
    if env_edge_hard is not None:
        edge_hard_min = float(env_edge_hard)
    if env_lf_min is not None:
        lf_min = float(env_lf_min)
    if env_lf_max is not None:
        lf_max = float(env_lf_max)

    pair_ok_hard = pair_edge_iou >= edge_hard_min
    pair_ok_soft = lf_min <= pair_lf_cos <= lf_max
    pair_ok = bool(pair_ok_hard and pair_ok_soft)

    env_enforce = _env_bool("V7_PAIR_ENFORCE")
    enforce_pair = env_enforce if env_enforce is not None else bool(plate_cfg.get("pair_enforce", False))

    result = {
        "ink_mask_core_polar": ink_mask_core_polar,
        "valid_polar": valid_polar,
        "geom": g,
        "gate_quality": {
            "usable": True,
            "raw_ink_sum": raw_ink_total,
            "raw_ink_area_ratio": raw_ink_area_ratio,
            "pupil_leak_ratio": pupil_leak_ratio,
            "bg_leak_ratio": bg_leak_ratio,
            "artifact_ratio": artifact_ratio,
            "bg_artifact_ratio": bg_artifact_ratio,
            "quality_ok": quality_ok,
            "quality_warns": quality_warnings,
            "pair_edge_iou": pair_edge_iou,
            "pair_lf_cos": float(pair_lf_cos),
            "pair_ok_hard": bool(pair_ok_hard),
            "pair_ok_soft": bool(pair_ok_soft),
            "pair_ok": pair_ok,
            "pair_lf_cos_max": float(lf_max),
        },
        "registration": {
            "method": "simple_brightness",
            "swapped": swapped,
        },
    }

    if enforce_pair and not pair_ok:
        result["gate_quality"]["usable"] = False
        result["gate_quality"]["reason"] = "pair_mismatch"

    quality_enforce = _env_bool("V7_QUALITY_ENFORCE")
    if quality_enforce is None:
        quality_enforce = bool(plate_cfg.get("quality_enforce", False))
    if quality_enforce and not result["gate_quality"].get("reason"):
        if raw_ink_area_ratio < raw_ink_area_hard_min:
            result["gate_quality"]["usable"] = False
            result["gate_quality"]["reason"] = "ink_signal_low"
        elif bg_leak_ratio > bg_leak_hard_max:
            result["gate_quality"]["usable"] = False
            result["gate_quality"]["reason"] = "bg_leak_high"
        elif pupil_leak_ratio > pupil_leak_hard_max:
            result["gate_quality"]["usable"] = False
            result["gate_quality"]["reason"] = "pupil_leak_high"

    if pair_iou is not None:
        result.setdefault("debug", {})["pair_iou"] = pair_iou

    return result
