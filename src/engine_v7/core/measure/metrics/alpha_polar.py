"""
Registration-less Polar Alpha Computation Module

Extracted from alpha_density.py (P1-1).
Computes alpha maps in polar coordinates without requiring 2D registration,
using median_theta aggregation for rotation invariance.

Usage:
    from .alpha_polar import PolarAlphaResult, build_polar_alpha_registrationless
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

if TYPE_CHECKING:
    from src.engine_v7.core.geometry.lens_geometry import LensGeometry


@dataclass
class PolarAlphaResult:
    """Result of registration-less polar alpha computation."""

    # 2D polar alpha map (T, R) - for per-cluster analysis
    polar_alpha: np.ndarray

    # 1D radial profile (R,) - median across theta for rotation invariance
    radial_profile: np.ndarray
    radial_confidence: np.ndarray  # (R,) confidence per radial bin

    # Quality metrics
    quality: Dict[str, Any]

    # Metadata
    meta: Dict[str, Any]


def build_polar_alpha_registrationless(
    white_bgr: np.ndarray,
    black_bgr: np.ndarray,
    geom: "LensGeometry",
    polar_R: int = 200,
    polar_T: int = 360,
    *,
    alpha_clip_min: float = 0.02,
    alpha_clip_max: float = 0.98,
    bg_outer_frac: float = 0.05,  # Fraction of outer radius for background estimation
    min_samples_per_bin: int = 10,  # Minimum samples for valid radial bin
    moire_detection_enabled: bool = True,
    moire_threshold: float = 0.15,  # Std dev threshold for moire detection
) -> PolarAlphaResult:
    """
    Build alpha map in polar coordinates WITHOUT registration.

    P1-1 Implementation: Registration-less Polar Alpha

    Key insight: By computing alpha in polar coordinates and using median_theta
    aggregation, we achieve rotation invariance without needing 2D registration.
    This is robust to rotational misalignment between white/black images.

    Algorithm:
    1. Convert both white and black images to polar coordinates
    2. Compute per-pixel alpha using standard formula
    3. For each radial bin r, take median(alpha[theta, r]) across all theta
    4. This gives a 1D radial alpha profile that's rotation-invariant

    Args:
        white_bgr: White backlight image (BGR)
        black_bgr: Black backlight image (BGR)
        geom: Lens geometry (center + radius)
        polar_R: Radial resolution
        polar_T: Angular resolution
        alpha_clip_min: Minimum alpha value (prevents artifacts)
        alpha_clip_max: Maximum alpha value (prevents artifacts)
        bg_outer_frac: Fraction of outer radius for background estimation
        min_samples_per_bin: Minimum samples for valid radial bin
        moire_detection_enabled: Whether to detect moire patterns
        moire_threshold: Threshold for moire detection (angular std)

    Returns:
        PolarAlphaResult with 2D polar_alpha, 1D radial profile, and quality metrics
    """
    from ...signature.radial_signature import to_polar

    # Convert to polar coordinates
    polar_white = to_polar(white_bgr, geom, R=polar_R, T=polar_T).astype(np.float32)
    polar_black = to_polar(black_bgr, geom, R=polar_R, T=polar_T).astype(np.float32)

    # Background estimation from outer edge
    outer_start = int(polar_R * (1.0 - bg_outer_frac))
    w_bg = np.median(polar_white[:, outer_start:, :], axis=(0, 1))
    b_bg = np.median(polar_black[:, outer_start:, :], axis=(0, 1))

    # Compute alpha per pixel
    # Formula: alpha = |W - W_bg - B + B_bg| / |W - B|
    # This measures how much the white/black difference deviates from expected
    diff = np.abs(polar_white - w_bg - polar_black + b_bg)
    denom = np.abs(polar_white - polar_black)
    denom = np.maximum(denom, 20.0)  # Prevent division by zero

    alpha_per_channel = diff / denom
    alpha_2d = np.mean(alpha_per_channel, axis=2)  # Average across BGR channels

    # Count clipping before clip
    pre_clip_at_min = np.sum(alpha_2d < alpha_clip_min)
    pre_clip_at_max = np.sum(alpha_2d > alpha_clip_max)
    total_pixels = polar_T * polar_R

    # Apply clipping
    alpha_2d = np.clip(alpha_2d, alpha_clip_min, alpha_clip_max)

    # Quality metrics
    nan_count = int(np.sum(np.isnan(alpha_2d)))
    nan_ratio = nan_count / total_pixels if total_pixels > 0 else 0.0
    clip_count = int(pre_clip_at_min + pre_clip_at_max)
    clip_ratio = clip_count / total_pixels if total_pixels > 0 else 0.0

    # Compute 1D radial profile using median_theta (rotation-invariant) — vectorized
    radial_n_samples = np.sum(~np.isnan(alpha_2d), axis=0).astype(np.int32)  # (R,)
    radial_profile = np.nanmedian(alpha_2d, axis=0).astype(np.float32)  # (R,)
    radial_std = np.nanstd(alpha_2d, axis=0).astype(np.float32)  # (R,)

    # Mark insufficient-sample bins as NaN
    insufficient = radial_n_samples < min_samples_per_bin
    radial_profile[insufficient] = np.nan
    radial_std[insufficient] = np.nan

    # Confidence: sample_conf * consistency_conf (vectorized)
    sample_conf = np.minimum(1.0, radial_n_samples.astype(np.float32) / polar_T)
    consistency_conf = np.maximum(0.0, 1.0 - radial_std / 0.5)
    radial_confidence = np.where(insufficient, 0.0, sample_conf * consistency_conf).astype(np.float32)

    # Interpolate NaN values in radial profile
    valid_mask = ~np.isnan(radial_profile)
    if np.any(valid_mask) and not np.all(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        invalid_indices = np.where(~valid_mask)[0]
        radial_profile[invalid_indices] = np.interp(invalid_indices, valid_indices, radial_profile[valid_indices])

    # Moire detection: high angular variance indicates moire artifacts
    moire_detected = False
    moire_severity = 0.0
    if moire_detection_enabled:
        # Check middle radial region (0.3 to 0.7 of radius)
        r_start = int(polar_R * 0.3)
        r_end = int(polar_R * 0.7)
        mid_std = radial_std[r_start:r_end]
        mid_std = mid_std[~np.isnan(mid_std)]
        if len(mid_std) > 0:
            mean_angular_std = float(np.mean(mid_std))
            moire_severity = mean_angular_std
            moire_detected = mean_angular_std > moire_threshold

    # Compute overall quality score
    quality_factors = []
    # Factor 1: Low NaN ratio (weight 0.3)
    nan_quality = max(0.0, 1.0 - nan_ratio * 10)  # 10% NaN = 0 quality
    quality_factors.append(("nan", nan_quality, 0.3))

    # Factor 2: Low clip ratio (weight 0.3)
    clip_quality = max(0.0, 1.0 - clip_ratio * 3.33)  # 30% clip = 0 quality
    quality_factors.append(("clip", clip_quality, 0.3))

    # Factor 3: Low moire (weight 0.2)
    moire_quality = max(0.0, 1.0 - moire_severity / moire_threshold) if moire_detection_enabled else 1.0
    quality_factors.append(("moire", moire_quality, 0.2))

    # Factor 4: Confidence (weight 0.2)
    mean_confidence = float(np.mean(radial_confidence[radial_confidence > 0])) if np.any(radial_confidence > 0) else 0.0
    quality_factors.append(("confidence", mean_confidence, 0.2))

    overall_quality = sum(q * w for _, q, w in quality_factors)

    quality = {
        "overall": round(overall_quality, 3),
        "nan_ratio": round(nan_ratio, 4),
        "clip_ratio": round(clip_ratio, 4),
        "moire_detected": moire_detected,
        "moire_severity": round(moire_severity, 4),
        "mean_radial_confidence": round(mean_confidence, 3),
        "valid_radial_bins": int(np.sum(valid_mask)),
        "total_radial_bins": polar_R,
        "factors": {name: round(q, 3) for name, q, _ in quality_factors},
    }

    meta = {
        "method": "registrationless_polar_median_theta",
        "polar_R": polar_R,
        "polar_T": polar_T,
        "alpha_clip": [alpha_clip_min, alpha_clip_max],
        "bg_outer_frac": bg_outer_frac,
        "alpha_mean": round(float(np.nanmean(alpha_2d)), 4),
        "alpha_std": round(float(np.nanstd(alpha_2d)), 4),
        "radial_profile_mean": round(float(np.nanmean(radial_profile)), 4),
    }

    return PolarAlphaResult(
        polar_alpha=alpha_2d,
        radial_profile=radial_profile,
        radial_confidence=radial_confidence,
        quality=quality,
        meta=meta,
    )
