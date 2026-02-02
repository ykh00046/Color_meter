"""
Alpha Verification Module

Extracted from alpha_density.py (P1-2, P2-4).
Compares registration-less alpha with plate-based alpha for verification,
with transition region detection and core-region metrics.

Usage:
    from .alpha_verification import (
        AlphaVerificationResult,
        verify_alpha_agreement,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


def _detect_alpha_transitions(
    alpha1: np.ndarray,
    alpha2: np.ndarray,
    gradient_threshold: float = 0.05,
    dilation_radius: int = 2,
) -> np.ndarray:
    """
    P2-4: Auto-detect transition regions from alpha gradient.

    Transition regions are areas where:
    1. Alpha values change rapidly (high gradient) - indicates color boundaries
    2. The two alpha maps disagree significantly - potential registration errors

    Only marks as transition if BOTH conditions suggest boundary/issue:
    - High local gradient (structural edge in alpha)
    - AND significant local disagreement between methods

    Args:
        alpha1: First alpha map (T, R)
        alpha2: Second alpha map (T, R)
        gradient_threshold: Threshold for normalized alpha gradient (default 0.05)
        dilation_radius: Radius to expand detected transitions

    Returns:
        Boolean mask (T, R) where True = transition region
    """
    import cv2

    # Compute gradient magnitude for both alpha maps
    def compute_gradient(alpha: np.ndarray) -> np.ndarray:
        # Use Sobel for gradient in both directions
        alpha_clean = np.nan_to_num(alpha, nan=0.5)
        grad_theta = cv2.Sobel(alpha_clean.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        grad_r = cv2.Sobel(alpha_clean.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        # Normalize by Sobel scale (approx 4 for ksize=3)
        return np.sqrt(grad_theta**2 + grad_r**2) / 4.0

    grad1 = compute_gradient(alpha1)
    grad2 = compute_gradient(alpha2)

    # Combined gradient (max of both maps)
    grad_combined = np.maximum(grad1, grad2)

    # High gradient threshold - use adaptive threshold based on gradient distribution
    # This prevents random uniform noise from triggering everywhere
    grad_p95 = np.nanpercentile(grad_combined, 95)
    effective_threshold = max(gradient_threshold, grad_p95 * 0.8)

    # High gradient mask - only strong edges
    high_gradient_mask = grad_combined > effective_threshold

    # Alpha disagreement mask - where methods disagree
    alpha_diff = np.abs(np.nan_to_num(alpha1, nan=0.5) - np.nan_to_num(alpha2, nan=0.5))
    high_diff_mask = alpha_diff > 0.10  # 10% difference threshold

    # P2-4 key insight: Mark as transition only where BOTH:
    # 1. High gradient (structural edge)
    # 2. High disagreement (methods disagree at edge)
    # This avoids marking random noise as transitions
    transition_mask = high_gradient_mask & high_diff_mask

    # Also mark very high disagreement even without gradient (registration failure)
    very_high_diff_mask = alpha_diff > 0.25  # 25% is too high to ignore
    transition_mask = transition_mask | very_high_diff_mask

    # Dilate to expand transition regions (boundary effects extend beyond edge)
    if dilation_radius > 0 and np.any(transition_mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))
        transition_mask = cv2.dilate(transition_mask.astype(np.uint8), kernel, iterations=1) > 0

    return transition_mask


@dataclass
class AlphaVerificationResult:
    """Result of comparing registration-less alpha with plate-based alpha."""

    agreement_score: float  # 0-1, higher = better agreement (full image)
    rmse: float  # Root mean square error (full image)
    correlation: float  # Pearson correlation coefficient (full image)
    radial_agreement: np.ndarray  # Per-radial-bin agreement scores
    summary: Dict[str, Any]
    warnings: List[str]
    passed: bool  # True if agreement is acceptable

    # P2-4: Core region metrics (excluding boundaries/transitions)
    core_agreement_score: float = 0.0  # Agreement in non-transition regions
    core_rmse: float = 0.0  # RMSE in non-transition regions
    core_correlation: float = 0.0  # Correlation in non-transition regions
    transition_ratio: float = 0.0  # Fraction of pixels in transition regions


def verify_alpha_agreement(
    registrationless_alpha: np.ndarray,
    plate_alpha: np.ndarray,
    *,
    transition_mask: Optional[np.ndarray] = None,
    rmse_threshold: float = 0.15,
    correlation_threshold: float = 0.7,
    agreement_threshold: float = 0.7,
    use_core_for_decision: bool = True,
    auto_detect_transitions: bool = True,
    gradient_threshold: float = 0.05,
) -> AlphaVerificationResult:
    """
    P1-2: Compare registration-less alpha with plate-based alpha for verification.
    P2-4: Improved agreement score excluding boundary/transition regions.

    This function verifies that the registration-less alpha computation produces
    results consistent with the plate-based registration approach. When they agree,
    it provides confidence in the registration-less method. When they disagree,
    it flags potential issues.

    The P2-4 enhancement computes separate metrics for "core" regions (excluding
    boundaries and transitions) since these regions naturally have higher variance
    between methods. The core metrics provide a more meaningful agreement measure.

    Args:
        registrationless_alpha: Alpha map from build_polar_alpha_registrationless (T, R)
        plate_alpha: Alpha map from plate_engine (T, R) - may need resampling
        transition_mask: Optional (T, R) bool mask where True = transition region.
                        If None and auto_detect_transitions=True, will be computed.
        rmse_threshold: Maximum acceptable RMSE for passing verification
        correlation_threshold: Minimum acceptable correlation for passing
        agreement_threshold: Minimum acceptable agreement score for passing
        use_core_for_decision: If True, pass/fail uses core metrics (recommended)
        auto_detect_transitions: If True, detect transitions from alpha gradient
        gradient_threshold: Alpha gradient threshold for auto-detecting transitions

    Returns:
        AlphaVerificationResult with comparison metrics and pass/fail status.
        Includes both full-image and core-region (non-transition) metrics.
    """
    warnings: List[str] = []

    # Ensure shapes match (resample if necessary)
    if registrationless_alpha.shape != plate_alpha.shape:
        import cv2

        target_shape = registrationless_alpha.shape
        plate_alpha_resampled = cv2.resize(
            plate_alpha.astype(np.float32),
            (target_shape[1], target_shape[0]),  # (width, height)
            interpolation=cv2.INTER_LINEAR,
        )
        warnings.append(f"ALPHA_SHAPE_MISMATCH: plate {plate_alpha.shape} -> {target_shape}")
    else:
        plate_alpha_resampled = plate_alpha

    # Create valid mask (both have valid values)
    valid_mask = (
        ~np.isnan(registrationless_alpha)
        & ~np.isnan(plate_alpha_resampled)
        & (registrationless_alpha >= 0.02)
        & (registrationless_alpha <= 0.98)
        & (plate_alpha_resampled >= 0.02)
        & (plate_alpha_resampled <= 0.98)
    )

    valid_ratio = float(np.sum(valid_mask)) / valid_mask.size
    if valid_ratio < 0.1:
        warnings.append(f"LOW_VALID_OVERLAP: only {valid_ratio:.1%} valid pixels")
        return AlphaVerificationResult(
            agreement_score=0.0,
            rmse=1.0,
            correlation=0.0,
            radial_agreement=np.zeros(registrationless_alpha.shape[1]),
            summary={
                "valid_ratio": valid_ratio,
                "valid_pixels": int(np.sum(valid_mask)),
                "total_pixels": valid_mask.size,
            },
            warnings=warnings,
            passed=False,
            # P2-4: Default core metrics for early return
            core_agreement_score=0.0,
            core_rmse=1.0,
            core_correlation=0.0,
            transition_ratio=0.0,
        )

    # Extract valid values
    reg_valid = registrationless_alpha[valid_mask]
    plate_valid = plate_alpha_resampled[valid_mask]

    # Compute RMSE
    diff = reg_valid - plate_valid
    rmse = float(np.sqrt(np.mean(diff**2)))

    # Compute correlation
    if np.std(reg_valid) > 1e-6 and np.std(plate_valid) > 1e-6:
        correlation = float(np.corrcoef(reg_valid, plate_valid)[0, 1])
    else:
        correlation = 0.0
        warnings.append("LOW_VARIANCE: one or both alpha maps have near-zero variance")

    # P2-4: Detect or use provided transition mask
    T, R = registrationless_alpha.shape
    if transition_mask is not None:
        # Ensure transition mask matches shape
        if transition_mask.shape != (T, R):
            import cv2

            transition_mask = cv2.resize(
                transition_mask.astype(np.uint8),
                (R, T),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
    elif auto_detect_transitions:
        # Auto-detect transitions from alpha gradient
        transition_mask = _detect_alpha_transitions(registrationless_alpha, plate_alpha_resampled, gradient_threshold)
    else:
        # No transition mask - treat all pixels as core
        transition_mask = np.zeros((T, R), dtype=bool)

    # Compute transition ratio
    transition_ratio = float(np.sum(transition_mask & valid_mask)) / max(1, np.sum(valid_mask))

    # P2-4: Compute core region metrics (excluding transitions)
    core_mask = valid_mask & ~transition_mask
    core_count = np.sum(core_mask)

    if core_count >= 100:  # Minimum pixels for meaningful core metrics
        reg_core = registrationless_alpha[core_mask]
        plate_core = plate_alpha_resampled[core_mask]
        core_diff = reg_core - plate_core

        core_rmse = float(np.sqrt(np.mean(core_diff**2)))

        if np.std(reg_core) > 1e-6 and np.std(plate_core) > 1e-6:
            core_correlation = float(np.corrcoef(reg_core, plate_core)[0, 1])
        else:
            core_correlation = 0.0

        # Core agreement: 1 - mean absolute difference
        core_agreement_score = max(0.0, 1.0 - float(np.mean(np.abs(core_diff))))
    else:
        # Insufficient core pixels - fall back to full metrics
        core_rmse = rmse
        core_correlation = correlation
        core_agreement_score = 0.0
        warnings.append(f"LOW_CORE_PIXELS: only {core_count} core pixels, using full metrics")

    # Compute per-radial-bin agreement — vectorized
    T, R = registrationless_alpha.shape
    col_valid = ~np.isnan(registrationless_alpha) & ~np.isnan(plate_alpha_resampled)  # (T, R)
    col_valid_count = col_valid.sum(axis=0)  # (R,)

    abs_diff = np.where(col_valid, np.abs(registrationless_alpha - plate_alpha_resampled), 0.0)
    mean_abs_diff = np.where(col_valid_count > 0, abs_diff.sum(axis=0) / np.maximum(col_valid_count, 1), 0.0)

    radial_agreement = np.where(
        col_valid_count >= 10,
        np.maximum(0.0, 1.0 - mean_abs_diff),
        np.nan,
    ).astype(np.float32)

    # Compute overall agreement score
    valid_radial = radial_agreement[~np.isnan(radial_agreement)]
    if len(valid_radial) > 0:
        agreement_score = float(np.mean(valid_radial))
    else:
        agreement_score = 0.0

    # P2-4: Determine pass/fail using core or full metrics
    if use_core_for_decision and core_count >= 100:
        # Use core region metrics for more meaningful comparison
        decision_rmse = core_rmse
        decision_corr = core_correlation
        decision_agree = core_agreement_score
        decision_source = "core"
    else:
        # Use full-image metrics
        decision_rmse = rmse
        decision_corr = correlation
        decision_agree = agreement_score
        decision_source = "full"

    passed = (
        decision_rmse <= rmse_threshold
        and decision_corr >= correlation_threshold
        and decision_agree >= agreement_threshold
    )

    if not passed:
        if decision_rmse > rmse_threshold:
            warnings.append(f"HIGH_RMSE_{decision_source.upper()}: {decision_rmse:.3f} > {rmse_threshold}")
        if decision_corr < correlation_threshold:
            warnings.append(f"LOW_CORRELATION_{decision_source.upper()}: {decision_corr:.3f} < {correlation_threshold}")
        if decision_agree < agreement_threshold:
            warnings.append(f"LOW_AGREEMENT_{decision_source.upper()}: {decision_agree:.3f} < {agreement_threshold}")

    summary = {
        "valid_ratio": round(valid_ratio, 4),
        "valid_pixels": int(np.sum(valid_mask)),
        "total_pixels": valid_mask.size,
        "registrationless_mean": round(float(np.nanmean(registrationless_alpha)), 4),
        "plate_mean": round(float(np.nanmean(plate_alpha_resampled)), 4),
        "mean_diff": round(float(np.mean(diff)), 4),
        "std_diff": round(float(np.std(diff)), 4),
        "radial_agreement_mean": round(float(np.nanmean(radial_agreement)), 4),
        "radial_agreement_min": round(float(np.nanmin(valid_radial)), 4) if len(valid_radial) > 0 else 0.0,
        "thresholds": {
            "rmse": rmse_threshold,
            "correlation": correlation_threshold,
            "agreement": agreement_threshold,
        },
        # P2-4: Core region metrics
        "core_valid_pixels": int(core_count),
        "core_ratio": round(1.0 - transition_ratio, 4),
        "transition_ratio": round(transition_ratio, 4),
        "decision_source": decision_source,
    }

    return AlphaVerificationResult(
        agreement_score=round(agreement_score, 4),
        rmse=round(rmse, 4),
        correlation=round(correlation, 4),
        radial_agreement=radial_agreement,
        summary=summary,
        warnings=warnings,
        passed=passed,
        # P2-4: Core region metrics
        core_agreement_score=round(core_agreement_score, 4),
        core_rmse=round(core_rmse, 4),
        core_correlation=round(core_correlation, 4),
        transition_ratio=round(transition_ratio, 4),
    )
