"""
Bias Analyzer for ColorChecker Calibration

Analyzes color measurement bias by comparing measured values
against ColorChecker reference standards.

Decision Logic (Robust Statistics):
- Uses p95 and trimmed_mean instead of max and mean to prevent outlier-driven recaptures
- Coverage validation ensures minimum patches available before analysis
- Threshold policy centralized for easy calibration adjustments

Default Threshold Policy:
{
    "mean_de_hold": 3.0,              # Hold if trimmed_mean > 3.0
    "mean_de_recapture": 5.0,         # Recapture if trimmed_mean > 5.0
    "max_de_hold": 5.0,               # (legacy, not used in robust mode)
    "max_de_recapture": 8.0,          # (legacy, not used in robust mode)
    "p95_de_hold": 5.0,               # Hold if p95 > 5.0
    "p95_de_recapture": 8.0,          # Recapture if p95 > 8.0
    "min_patches": 18,                # Minimum valid patches required
    "min_grayscale_patches": 4,       # Minimum grayscale patches required
    "grade_thresholds": [2.0, 3.0, 4.0, 5.0]  # A/B/C/D/F boundaries
}

Usage:
    result = analyze_colorchecker_bias(bgr, patch_rois, threshold_policy=policy)
    if result["operator_summary"]["decision"] == "RECAPTURE":
        # Handle recapture
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..utils import bgr_to_lab_cie, cie2000_deltaE
from .colorchecker_reference import (
    get_all_patch_ids,
    get_chromatic_patch_ids,
    get_grayscale_patch_ids,
    get_patch_lab,
    get_patch_name,
)


def extract_patch_lab_cie(bgr: np.ndarray, roi: Dict[str, int]) -> Optional[List[float]]:
    """
    Extract mean Lab_CIE from a patch ROI with defensive clipping.

    Args:
        bgr: BGR image
        roi: {"x": ..., "y": ..., "w": ..., "h": ...}

    Returns:
        [L*, a*, b*] in CIE scale, or None if ROI is invalid
    """
    H, W = bgr.shape[:2]
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

    # Clip ROI to image bounds
    x0 = max(0, min(W, int(x)))
    y0 = max(0, min(H, int(y)))
    x1 = max(0, min(W, int(x + w)))
    y1 = max(0, min(H, int(y + h)))

    # Check minimum size after clipping
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None  # ROI too small or out of bounds

    # Sample center 60% of patch to avoid edges
    w_clipped = x1 - x0
    h_clipped = y1 - y0
    margin_w = int(w_clipped * 0.2)
    margin_h = int(h_clipped * 0.2)

    x_start = x0 + margin_w
    y_start = y0 + margin_h
    x_end = x1 - margin_w
    y_end = y1 - margin_h

    # Check minimum size after margin
    if x_end - x_start < 2 or y_end - y_start < 2:
        return None  # ROI too small after margin

    patch_bgr = bgr[y_start:y_end, x_start:x_end]

    if patch_bgr.size == 0:
        return None

    # Convert to Lab CIE
    patch_lab_cie = bgr_to_lab_cie(patch_bgr)

    # Mean of patch
    mean_lab = np.mean(patch_lab_cie.reshape(-1, 3), axis=0)

    return [float(mean_lab[0]), float(mean_lab[1]), float(mean_lab[2])]


def analyze_colorchecker_bias(
    bgr: np.ndarray, patch_rois: Dict[str, Dict[str, int]], threshold_policy: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze ColorChecker bias from measured image with robust statistics and coverage validation.

    Args:
        bgr: ColorChecker image (BGR)
        patch_rois: {"A1": {"x": ..., "y": ..., "w": ..., "h": ...}, ...}
        threshold_policy: Optional policy override for decision rules
                         {"mean_de_hold": 3.0, "mean_de_recapture": 5.0,
                          "max_de_hold": 5.0, "max_de_recapture": 8.0,
                          "min_patches": 18, "min_grayscale_patches": 4}

    Returns:
        {
            "operator_summary": {...},
            "engineer_kpi": {...},
            "patch_results": [...],
            "coverage": {...}
        }
    """
    # Default thresholds (centralized policy - matches module docstring)
    policy = threshold_policy or {}
    mean_de_hold = policy.get("mean_de_hold", 3.0)
    mean_de_recapture = policy.get("mean_de_recapture", 5.0)
    max_de_hold = policy.get("max_de_hold", 5.0)  # Legacy, for backward compatibility
    max_de_recapture = policy.get("max_de_recapture", 8.0)  # Legacy
    p95_de_hold = policy.get("p95_de_hold", 5.0)  # Robust threshold
    p95_de_recapture = policy.get("p95_de_recapture", 8.0)  # Robust threshold
    min_patches = policy.get("min_patches", 18)
    min_grayscale_patches = policy.get("min_grayscale_patches", 4)
    grade_thresholds = policy.get("grade_thresholds", [2.0, 3.0, 4.0, 5.0])

    patch_results = []
    delta_e_list = []
    bias_L_list = []
    bias_a_list = []
    bias_b_list = []

    # Separate tracking for grayscale and chromatic
    gray_ids = get_grayscale_patch_ids()
    chromatic_ids = get_chromatic_patch_ids()

    gray_delta_e_list = []
    gray_bias_L_list = []
    chromatic_delta_e_list = []
    chromatic_bias_list = []  # Combined a/b for chromatic

    valid_grayscale_count = 0
    invalid_count = 0

    for patch_id in get_all_patch_ids():
        if patch_id not in patch_rois:
            continue

        # Extract measured Lab with defensive clipping
        measured_lab = extract_patch_lab_cie(bgr, patch_rois[patch_id])
        std_lab = get_patch_lab(patch_id)
        patch_name = get_patch_name(patch_id)

        # Handle invalid ROI
        if measured_lab is None:
            patch_results.append(
                {
                    "patch_id": patch_id,
                    "patch_name": patch_name,
                    "valid": False,
                    "reason": "ROI_OUT_OF_BOUNDS or ROI_TOO_SMALL",
                }
            )
            invalid_count += 1
            continue

        # Calculate ΔE
        delta_e = cie2000_deltaE(np.array([measured_lab]), np.array([std_lab]))[0]

        # Calculate bias
        bias_L = measured_lab[0] - std_lab[0]
        bias_a = measured_lab[1] - std_lab[1]
        bias_b = measured_lab[2] - std_lab[2]

        patch_results.append(
            {
                "patch_id": patch_id,
                "patch_name": patch_name,
                "valid": True,
                "measured_lab": [round(measured_lab[0], 2), round(measured_lab[1], 2), round(measured_lab[2], 2)],
                "standard_lab": [round(std_lab[0], 2), round(std_lab[1], 2), round(std_lab[2], 2)],
                "delta_e": round(float(delta_e), 2),
                "bias_L": round(float(bias_L), 2),
                "bias_a": round(float(bias_a), 2),
                "bias_b": round(float(bias_b), 2),
            }
        )

        # Add to overall lists
        delta_e_list.append(float(delta_e))
        bias_L_list.append(float(bias_L))
        bias_a_list.append(float(bias_a))
        bias_b_list.append(float(bias_b))

        # Separate grayscale and chromatic
        if patch_id in gray_ids:
            gray_delta_e_list.append(float(delta_e))
            gray_bias_L_list.append(float(bias_L))
            valid_grayscale_count += 1
        else:
            chromatic_delta_e_list.append(float(delta_e))
            chromatic_bias_list.append((float(bias_a), float(bias_b)))

    # Sort valid patches by delta_e (worst first)
    valid_patches = [p for p in patch_results if p.get("valid", True)]
    valid_patches.sort(key=lambda x: x.get("delta_e", 0), reverse=True)

    # Coverage validation
    expected_patches = 24
    provided_patches = len(patch_rois)
    used_patches = len(delta_e_list)

    coverage = {
        "expected": expected_patches,
        "provided": provided_patches,
        "used": used_patches,
        "invalid": invalid_count,
        "grayscale_valid": valid_grayscale_count,
        "coverage_ratio": round(used_patches / expected_patches, 2) if expected_patches > 0 else 0.0,
    }

    # Check coverage requirements
    coverage_fail_reasons = []
    if used_patches < min_patches:
        coverage_fail_reasons.append(f"INSUFFICIENT_PATCHES (used: {used_patches} < required: {min_patches})")
    if valid_grayscale_count < min_grayscale_patches:
        coverage_fail_reasons.append(
            f"INSUFFICIENT_GRAYSCALE (used: {valid_grayscale_count} < required: {min_grayscale_patches})"
        )

    # Calculate overall statistics
    mean_delta_e = float(np.mean(delta_e_list)) if delta_e_list else 0.0
    max_delta_e = float(np.max(delta_e_list)) if delta_e_list else 0.0
    median_delta_e = float(np.median(delta_e_list)) if delta_e_list else 0.0
    p95_delta_e = float(np.percentile(delta_e_list, 95)) if delta_e_list else 0.0

    # Trimmed mean: exclude top/bottom 10%
    if len(delta_e_list) >= 10:
        trimmed = sorted(delta_e_list)
        n_trim = max(1, len(trimmed) // 10)
        trimmed_mean_delta_e = float(np.mean(trimmed[n_trim:-n_trim]))
    else:
        trimmed_mean_delta_e = mean_delta_e

    mean_bias_L = float(np.mean(bias_L_list)) if bias_L_list else 0.0
    mean_bias_a = float(np.mean(bias_a_list)) if bias_a_list else 0.0
    mean_bias_b = float(np.mean(bias_b_list)) if bias_b_list else 0.0

    # Grayscale-specific statistics (white balance / exposure issues)
    gray_stats = {}
    if gray_delta_e_list:
        gray_stats = {
            "mean_delta_e": round(float(np.mean(gray_delta_e_list)), 2),
            "mean_bias_L": round(float(np.mean(gray_bias_L_list)), 2),
            "std_bias_L": round(float(np.std(gray_bias_L_list)), 2),
        }

    # Chromatic-specific statistics (color gamut / lighting spectrum issues)
    chromatic_stats = {}
    if chromatic_delta_e_list:
        chromatic_bias_a = [b[0] for b in chromatic_bias_list]
        chromatic_bias_b = [b[1] for b in chromatic_bias_list]
        chromatic_stats = {
            "mean_delta_e": round(float(np.mean(chromatic_delta_e_list)), 2),
            "mean_bias_a": round(float(np.mean(chromatic_bias_a)), 2),
            "mean_bias_b": round(float(np.mean(chromatic_bias_b)), 2),
            "std_bias_a": round(float(np.std(chromatic_bias_a)), 2),
            "std_bias_b": round(float(np.std(chromatic_bias_b)), 2),
        }

    # Generate Operator Summary with robust statistics
    operator_summary = _generate_calibration_decision(
        mean_delta_e,
        max_delta_e,
        valid_patches,
        coverage_fail_reasons,
        robust_stats={
            "median_delta_e": median_delta_e,
            "p95_delta_e": p95_delta_e,
            "trimmed_mean_delta_e": trimmed_mean_delta_e,
        },
        thresholds={
            "mean_de_hold": mean_de_hold,
            "mean_de_recapture": mean_de_recapture,
            "max_de_hold": max_de_hold,
            "max_de_recapture": max_de_recapture,
            "p95_de_hold": p95_de_hold,
            "p95_de_recapture": p95_de_recapture,
            "grade_thresholds": grade_thresholds,
        },
    )

    # Generate Engineer KPI
    engineer_kpi = {
        "color_accuracy": {
            "mean_delta_e": round(mean_delta_e, 2),
            "median_delta_e": round(median_delta_e, 2),
            "p95_delta_e": round(p95_delta_e, 2),
            "trimmed_mean_delta_e": round(trimmed_mean_delta_e, 2),
            "max_delta_e": round(max_delta_e, 2),
            "patches_over_3": sum(1 for de in delta_e_list if de > 3.0),
            "patches_over_5": sum(1 for de in delta_e_list if de > 5.0),
        },
        "bias_overall": {
            "mean_L": round(mean_bias_L, 2),
            "mean_a": round(mean_bias_a, 2),
            "mean_b": round(mean_bias_b, 2),
            "std_L": round(float(np.std(bias_L_list)), 2) if bias_L_list else 0.0,
            "std_a": round(float(np.std(bias_a_list)), 2) if bias_a_list else 0.0,
            "std_b": round(float(np.std(bias_b_list)), 2) if bias_b_list else 0.0,
        },
        "bias_grayscale": gray_stats,
        "bias_chromatic": chromatic_stats,
        "worst_patches": valid_patches[:3],  # Top 3 worst
    }

    return {
        "operator_summary": operator_summary,
        "engineer_kpi": engineer_kpi,
        "patch_results": patch_results,
        "coverage": coverage,
    }


def _generate_calibration_decision(
    mean_delta_e: float,
    max_delta_e: float,
    patch_results: List[Dict],
    coverage_fail_reasons: List[str],
    robust_stats: Dict[str, float],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    """
    Generate operator-friendly calibration decision with robust statistics and configurable thresholds.

    Decision rules (using threshold_policy + robust statistics):
    - RECAPTURE: coverage fail OR p95 > p95_de_recapture OR trimmed_mean > mean_de_recapture
    - HOLD: p95 > p95_de_hold OR trimmed_mean > mean_de_hold
    - PASS: coverage OK AND all robust metrics within limits

    Robust statistics prevent single outlier patches from triggering excessive recaptures.
    """
    # Extract thresholds (no hardcoded defaults here - all from policy)
    mean_de_hold = thresholds["mean_de_hold"]
    mean_de_recapture = thresholds["mean_de_recapture"]
    max_de_hold = thresholds["max_de_hold"]
    max_de_recapture = thresholds["max_de_recapture"]

    # Optional robust thresholds (use mean thresholds if not specified)
    p95_de_hold = thresholds.get("p95_de_hold", max_de_hold)
    p95_de_recapture = thresholds.get("p95_de_recapture", max_de_recapture)

    # Extract robust stats
    median_delta_e = robust_stats.get("median_delta_e", mean_delta_e)
    p95_delta_e = robust_stats.get("p95_delta_e", max_delta_e)
    trimmed_mean_delta_e = robust_stats.get("trimmed_mean_delta_e", mean_delta_e)

    # Decision logic using robust statistics
    # Priority 1: Coverage failure
    if coverage_fail_reasons:
        decision = "RECAPTURE"
        severity = "HIGH"
    # Priority 2: Robust RECAPTURE check (p95 and trimmed_mean)
    elif p95_delta_e > p95_de_recapture or trimmed_mean_delta_e > mean_de_recapture:
        decision = "RECAPTURE"
        severity = "HIGH"
    # Priority 3: Robust HOLD check
    elif p95_delta_e > p95_de_hold or trimmed_mean_delta_e > mean_de_hold:
        decision = "HOLD"
        severity = "MEDIUM"
    else:
        decision = "PASS"
        severity = "LOW"

    # Quality grade (use trimmed_mean for stability)
    grade_thresholds = thresholds.get("grade_thresholds", [2.0, 3.0, 4.0, 5.0])
    if trimmed_mean_delta_e <= grade_thresholds[0]:
        grade = "A"
    elif trimmed_mean_delta_e <= grade_thresholds[1]:
        grade = "B"
    elif trimmed_mean_delta_e <= grade_thresholds[2]:
        grade = "C"
    elif trimmed_mean_delta_e <= grade_thresholds[3]:
        grade = "D"
    else:
        grade = "F"

    # Top 2 worst patches (valid only)
    top2_reasons = []
    for patch in patch_results[:2]:
        if patch.get("valid", True):
            top2_reasons.append(f"{patch['patch_name'].title()} (ΔE: {patch.get('delta_e', 0):.1f})")

    # Action guidance
    if coverage_fail_reasons:
        action = f"패치 커버리지 부족: {', '.join(coverage_fail_reasons)}. ColorChecker 배치와 ROI 검출을 확인하세요."
    elif decision == "RECAPTURE":
        action = "측정 환경을 재조정하고 재촬영하세요. 조명 밝기와 색온도를 확인하세요."
    elif decision == "HOLD":
        action = "색 정확도가 낮습니다. 엔지니어 검토 후 환경 조정이 필요합니다."
    else:
        action = "측정 환경 적합. 렌즈 측정을 시작하세요."

    return {
        "decision": decision,
        "decision_reason_top2": top2_reasons,
        "coverage_fail_reasons": coverage_fail_reasons,
        "action": action,
        "quality_grade": grade,
        "severity": severity,
        "mean_delta_e": round(mean_delta_e, 2),
        "max_delta_e": round(max_delta_e, 2),
        "p95_delta_e": round(p95_delta_e, 2),
        "trimmed_mean_delta_e": round(trimmed_mean_delta_e, 2),
        "median_delta_e": round(median_delta_e, 2),
        "_decision_basis": "p95 + trimmed_mean (robust statistics)",
    }
