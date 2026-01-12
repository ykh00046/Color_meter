from __future__ import annotations

import math
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..types import GateResult, LensGeometry
from ..utils import bgr_to_lab, laplacian_var

# ============================================================================
# Decision Builder Integration Guide
# ============================================================================
#
# Gate scores include normalized [0~1] scores for easy decision synthesis:
#   - center_score: 1.0 = perfect center, 0.0 = at threshold
#   - sharpness_score: 1.0 = at/exceeds minimum, 0.0 = zero
#   - illumination_score: 1.0 = perfect uniform, 0.0 = at threshold
#   - All scores: higher is better (consistent direction)
#
# Handling missing scores (None):
#   - Missing scores indicate partial data loss (not hard failure)
#   - decision_builder should skip None scores and renormalize weights
#
# Example integration:
#
#   gate_scores = run_gate(...).scores
#
#   # Extract normalized scores
#   items = []
#   if gate_scores.get("center_score") is not None:
#       items.append(("center", gate_scores["center_score"], 0.4))
#   if gate_scores.get("sharpness_score") is not None:
#       items.append(("sharpness", gate_scores["sharpness_score"], 0.3))
#   if gate_scores.get("illumination_score") is not None:
#       items.append(("illum", gate_scores["illumination_score"], 0.3))
#
#   # Weighted average with renormalization
#   if items:
#       w_sum = sum(w for _, _, w in items)
#       quality_score = sum(v * w for _, v, w in items) / w_sum
#   else:
#       # All measurements failed (valid=True but all scores=None)
#       # This indicates environmental/setup issues requiring human review
#       return "REVIEW"  # Let operator decide if retake or environment adjustment needed
#
#   # Apply thresholds
#   if quality_score < 0.5:
#       return "RETAKE"
#   elif quality_score < 0.7:
#       return "REVIEW"
#   else:
#       return "PASS"
#
# ============================================================================


def run_gate(
    test_geom: LensGeometry,
    bgr: np.ndarray,
    *,
    center_off_max: float,
    blur_min: float,
    illum_max: float,
    pixel_to_mm: Optional[float] = None,
    min_radius: float = 20.0,
) -> GateResult:
    """
    Gate checks whether the image is suitable for reliable comparison.
    NOTE: This is "in-frame" validity, not STD pixel coordinate alignment.

    Args:
        test_geom: Detected lens geometry
        bgr: Input image
        center_off_max: Max center offset ratio (offset / radius)
        blur_min: Min sharpness (Laplacian variance)
        illum_max: Max illumination asymmetry (std/mean of ROI quadrants)
        pixel_to_mm: Optional calibration factor (pixels to mm). If None, mm values not reported.
        min_radius: Min valid radius in pixels to avoid geometry detection failures

    Returns GateResult with:
        - passed: bool
        - reasons: List[str] (e.g., "BLUR_LOW", "CENTER_NOT_IN_FRAME")
        - scores: Dict with measured values
    """
    reasons: List[str] = []
    h, w = bgr.shape[:2]
    img_cx, img_cy = w / 2.0, h / 2.0

    # 1. Geometry validity check
    if test_geom.r < min_radius:
        reasons.append(f"GEOMETRY_INVALID (radius: {test_geom.r:.1f}px < {min_radius:.1f}px)")
        # Early return with schema-consistent scores
        return GateResult(
            passed=False,
            reasons=reasons,
            scores={
                # Keep same keys as normal case, but mark invalid
                "center_offset_ratio": None,
                "center_offset_pixels": None,
                "sharpness_laplacian_var": None,
                "illumination_asymmetry": None,
                # Normalized scores (for decision_builder)
                "center_score": None,
                "sharpness_score": None,
                "illumination_score": None,
                "_thresholds": {
                    "center_off_max": center_off_max,
                    "blur_min": blur_min,
                    "illum_max": illum_max,
                    "min_radius": min_radius,
                },
                "_meta": {
                    "valid": False,
                    "pixel_to_mm": pixel_to_mm,
                    "sharpness_roi": "edge",
                    "edge_pixel_count": None,
                },
                "_guidance": {"error": "렌즈 검출 실패. 이미지를 확인하고 재촬영하세요."},
            },
        )

    # 2. Calculate center offset
    center_off_pixels = math.hypot(test_geom.cx - img_cx, test_geom.cy - img_cy)
    center_off_ratio = center_off_pixels / test_geom.r
    center_off_mm = center_off_pixels * pixel_to_mm if pixel_to_mm else None

    # 3. Create ROI mask (circular region, 95% of detected radius)
    y_grid, x_grid = np.ogrid[:h, :w]
    roi_mask = ((x_grid - test_geom.cx) ** 2 + (y_grid - test_geom.cy) ** 2) <= (test_geom.r * 0.95) ** 2

    # Calculate dynamic thresholds based on ROI size
    roi_pixel_count = int(np.sum(roi_mask))
    roi_area_theoretical = math.pi * (test_geom.r * 0.95) ** 2
    min_roi_pixels = max(100, int(0.01 * roi_area_theoretical))  # 1% of ROI area
    min_quad_pixels = max(50, int(min_roi_pixels * 0.05))  # 5% of min_roi_pixels

    # Verify coordinate space consistency: geom.r and bgr must use same scale
    # This assertion protects against bugs from mismatched coordinate spaces
    assert test_geom.r > 0 and test_geom.r < max(
        h, w
    ), f"Geometry radius {test_geom.r} outside valid image bounds [{h}x{w}]"

    # 4. Blur (sharpness) - Pattern/Edge ROI only
    # Use annulus region (0.45R ~ 0.95R) and top gradient pixels to avoid flat regions
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Create annulus mask (inner=0.45R, outer=0.95R) to exclude flat center
    dist_sq = (x_grid - test_geom.cx) ** 2 + (y_grid - test_geom.cy) ** 2
    inner_r = test_geom.r * 0.45
    outer_r = test_geom.r * 0.95
    annulus_mask = (dist_sq >= inner_r**2) & (dist_sq <= outer_r**2)

    # Calculate gradient magnitude to find edge pixels
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # Select top 10% gradient pixels within annulus (pattern/edge regions)
    annulus_pixels = annulus_mask.sum()
    blur_valid = annulus_pixels >= min_roi_pixels
    edge_pixel_count = 0

    if blur_valid:
        grad_in_annulus = gradient_mag[annulus_mask]
        grad_threshold = np.percentile(grad_in_annulus, 90)  # Top 10%
        edge_mask = annulus_mask & (gradient_mag >= grad_threshold)

        # Calculate sharpness only on edge pixels (pattern regions)
        edge_pixel_count = edge_mask.sum()
        if edge_pixel_count >= 10:  # Min 10 pixels for variance calculation
            blur = float(laplacian[edge_mask].var())
        else:
            blur = None  # Not enough edge pixels
    else:
        blur = None

    # Note: ROI_TOO_SMALL is not a hard gate failure, just partial data loss
    # We mark it but keep valid=True, let decision_builder handle the missing score

    # 5. Illumination asymmetry - ROI quadrants
    lab = bgr_to_lab(bgr)
    L = lab[..., 0]
    cx, cy = int(test_geom.cx), int(test_geom.cy)
    cx = max(1, min(w - 2, cx))
    cy = max(1, min(h - 2, cy))

    # Quadrant means within ROI (with min pixel count check)
    q = []
    quad_pixel_counts = []

    for y_slice, x_slice in [
        (slice(None, cy), slice(None, cx)),  # Top-left
        (slice(None, cy), slice(cx, None)),  # Top-right
        (slice(cy, None), slice(None, cx)),  # Bottom-left
        (slice(cy, None), slice(cx, None)),  # Bottom-right
    ]:
        quad_mask = roi_mask[y_slice, x_slice]
        quad_count = int(np.sum(quad_mask))
        quad_pixel_counts.append(quad_count)

        if quad_count >= min_quad_pixels:
            q.append(float(np.mean(L[y_slice, x_slice][quad_mask])))

    # Calculate illum only if we have enough valid quadrants
    # Note: L channel is OpenCV 8-bit Lab (0-255), not CIE L* (0-100)
    illum = None
    illum_too_dark = False
    if len(q) >= 3:  # Need at least 3 quadrants
        q_mean = np.mean(q)
        # Check if image is too dark for reliable illumination measurement
        # Threshold: 5.0 in 0-255 scale (~2% of range)
        if q_mean < 5.0:
            # Very dark image - illumination measurement unreliable
            illum_too_dark = True
        else:
            # Protect against division issues with ratio-based epsilon
            # Use 5% of mean as epsilon, minimum 5.0
            eps = max(5.0, 0.05 * q_mean)
            illum = float(np.std(q) / (q_mean + eps))

    # 6. Check thresholds and generate reasons (ratio-based)
    if center_off_ratio > center_off_max:
        reason = f"CENTER_NOT_IN_FRAME (ratio: {center_off_ratio:.3f} > {center_off_max:.3f})"
        if center_off_mm is not None:
            reason += f" [≈{center_off_mm:.2f}mm]"
        reasons.append(reason)

    if blur is not None and blur < blur_min:
        reasons.append(f"BLUR_LOW (sharpness: {blur:.1f} < {blur_min:.1f})")

    if illum is not None and illum > illum_max:
        reasons.append(f"ILLUMINATION_UNEVEN (asymmetry: {illum:.3f} > {illum_max:.3f})")

    # 7. Calculate normalized scores [0~1] for decision_builder
    # Higher is better (consistent direction)
    def _clamp(val: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(max_val, val))

    # Center: 1.0 = perfect center, 0.0 = at threshold
    center_score = 1.0 - _clamp(center_off_ratio / center_off_max, 0.0, 1.0)

    # Sharpness: 1.0 = at/exceeds minimum, 0.0 = zero
    # Clamp to [0, 1] for consistent normalization
    sharpness_score = _clamp(blur / blur_min, 0.0, 1.0) if blur is not None else None

    # Illumination: 1.0 = perfect uniform, 0.0 = at threshold
    illumination_score = 1.0 - _clamp(illum / illum_max, 0.0, 1.0) if illum is not None else None

    # Build scores dict
    scores = {
        # Primary metrics (ratio-based for stability)
        "center_offset_ratio": round(float(center_off_ratio), 3),
        "center_offset_pixels": round(float(center_off_pixels), 2),
        "sharpness_laplacian_var": round(float(blur), 1) if blur is not None else None,
        "illumination_asymmetry": round(float(illum), 3) if illum is not None else None,
        # Normalized scores [0~1] for decision_builder (higher is better)
        "center_score": round(float(center_score), 3),
        "sharpness_score": round(float(sharpness_score), 3) if sharpness_score is not None else None,
        "illumination_score": round(float(illumination_score), 3) if illumination_score is not None else None,
        # Thresholds for reference
        "_thresholds": {
            "center_off_max": center_off_max,
            "blur_min": blur_min,
            "illum_max": illum_max,
            "min_radius": min_radius,
            "min_roi_pixels": min_roi_pixels,
            "min_quad_pixels": min_quad_pixels,
        },
        # Metadata
        "_meta": {
            "valid": True,
            "pixel_to_mm": pixel_to_mm,
            "roi_pixel_count": int(roi_pixel_count),
            "quad_pixel_counts": quad_pixel_counts,
            "geom_scale": "image_space",  # Confirms geom.r and bgr use same coordinate system
            "sharpness_roi": "edge",  # Annulus (0.45R~0.95R) + top 10% gradient pixels
            "edge_pixel_count": int(edge_pixel_count) if blur_valid and blur is not None else None,
        },
        # Action guidance
        "_guidance": _get_action_guidance(reasons, blur, blur_min, center_off_ratio, illum, center_off_mm),
    }

    # Optional: add mm if calibration provided
    if center_off_mm is not None:
        scores["center_offset_mm"] = round(float(center_off_mm), 2)

    return GateResult(
        passed=(len(reasons) == 0),
        reasons=reasons,
        scores=scores,
    )


def _get_action_guidance(
    reasons: List[str],
    blur: Optional[float],
    blur_min: float,
    center_off_ratio: float,
    illum: Optional[float],
    center_off_mm: Optional[float] = None,
) -> Dict[str, str]:
    """Generate actionable guidance for failed gate checks with priority handling

    Priority order (highest first, limit to 1-2 messages):
    1. GEOMETRY_INVALID (critical error)
    2. TOO_DARK (environmental issue)
    3. CENTER (positioning issue)
    4. SHARPNESS (focus issue)
    5. ILLUMINATION (lighting issue)
    6. DATA_SPARSE (partial measurement loss)

    Note: Data sparsity (ROI_TOO_SMALL, ILLUM_DATA_SPARSE, ILLUM_TOO_DARK) are not
    hard failures - they result in partial missing scores which decision_builder handles.
    """
    guidance = {}

    # Priority 1: GEOMETRY_INVALID (critical, return immediately)
    if any("GEOMETRY_INVALID" in r for r in reasons):
        guidance["error"] = "렌즈 검출 실패. 이미지를 확인하고 재촬영하세요."
        return guidance

    # Collect potential messages with priority
    messages = []

    # Priority 2: TOO_DARK (environmental)
    if illum is None and "TOO_DARK" in str(reasons):
        messages.append((2, "warning", "이미지가 너무 어두워 조명 측정 불가. 노출/광원을 확인하세요."))

    # Priority 3: CENTER (positioning)
    if any("CENTER_NOT_IN_FRAME" in r for r in reasons):
        msg = f"렌즈 중심이 프레임 밖입니다 (ratio: {center_off_ratio:.3f})"
        if center_off_mm is not None:
            msg += f" [≈{center_off_mm:.2f}mm]"
        msg += ". 렌즈를 화면 중앙으로 이동하세요."
        messages.append((3, "position", msg))

    # Priority 4: SHARPNESS (focus)
    if any("BLUR_LOW" in r for r in reasons) and blur is not None:
        msg = f"이미지가 흐립니다 (sharpness: {blur:.1f} < {blur_min:.1f}). 초점을 다시 맞추고 재촬영하세요."
        messages.append((4, "focus", msg))

    # Priority 5: ILLUMINATION (lighting)
    if any("ILLUMINATION_UNEVEN" in r for r in reasons) and illum is not None:
        msg = f"조명이 불균일합니다 (asymmetry: {illum:.3f}). 조명을 균일하게 조정하세요."
        messages.append((5, "lighting", msg))

    # Priority 6: DATA_SPARSE (partial measurement loss)
    if blur is None and not any("TOO_DARK" in str(r) for r in reasons):
        messages.append((6, "warning", "샤프니스 측정 불가. ROI 영역이 부족하거나 렌즈 검출 오류일 수 있습니다."))

    if illum is None and "TOO_DARK" not in str(reasons):
        messages.append((6, "warning", "조명 측정 데이터 부족. 렌즈 위치를 확인하세요."))

    # Sort by priority and take top 2
    messages.sort(key=lambda x: x[0])
    for priority, key, msg in messages[:2]:
        if key in guidance:
            # If key already exists, append (for multiple warnings)
            guidance[key] += f" {msg}"
        else:
            guidance[key] = msg

    if not guidance:
        guidance["status"] = "측정 품질 양호"

    return guidance
