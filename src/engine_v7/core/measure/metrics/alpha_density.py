"""
Alpha-Based Effective Density Calculation Module

This module provides:
1. Alpha radial profile extraction (1D) with median_theta aggregation
2. 3-tier fallback system for alpha values (L1 -> L2 -> L3)
3. Effective density calculation: area_ratio * alpha

Key Features:
- Robust to moire patterns and alignment errors via median aggregation
- Automatic fallback when sample counts are insufficient
- Quality-weighted blending between measured and fallback values

Usage:
    from .alpha_density import compute_effective_density, AlphaDensityResult

    result = compute_effective_density(
        polar_alpha=alpha_map,  # (T, R) from plate_gate
        cluster_masks=masks,     # Dict[str, (T, R) bool]
        area_ratios=ratios,      # Dict[str, float]
        cfg=config,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from src.engine_v7.core.geometry.lens_geometry import LensGeometry

import numpy as np


class AlphaFallbackLevel(Enum):
    """Fallback hierarchy for alpha values."""

    L1_RADIAL = "L1_radial"  # alpha_i(r) - per-cluster radial profile
    L2_ZONE = "L2_zone"  # alpha_zone_i - per-cluster zone averages
    L2_PLATE_LITE = "L2_plate_lite"  # plate_lite per-ink alpha (between L2 and L3)
    L3_GLOBAL = "L3_global"  # alpha_global or default 1.0


@dataclass
class AlphaRadialProfile:
    """1D radial alpha profile for a cluster."""

    r_bins: np.ndarray  # (n_bins,) bin center positions [0, 1]
    alpha_values: np.ndarray  # (n_bins,) median alpha per bin
    confidence: np.ndarray  # (n_bins,) confidence per bin [0, 1]
    n_samples: np.ndarray  # (n_bins,) sample count per bin
    valid_bins: int  # number of bins with sufficient samples
    total_bins: int  # total number of bins
    quality: float  # overall profile quality [0, 1]


@dataclass
class AlphaZoneProfile:
    """Zone-based alpha profile (inner/mid/outer)."""

    inner: float  # r < 0.4
    mid: float  # 0.4 <= r < 0.7
    outer: float  # r >= 0.7
    inner_conf: float
    mid_conf: float
    outer_conf: float


@dataclass
class ClusterAlphaResult:
    """Alpha analysis result for a single cluster."""

    color_id: str
    area_ratio: float

    # L1: Radial profile
    radial_profile: Optional[AlphaRadialProfile]

    # L2: Zone profile
    zone_profile: Optional[AlphaZoneProfile]

    # L3: Global alpha
    alpha_global: float

    # Final effective density
    effective_density: float
    alpha_used: float
    fallback_level: AlphaFallbackLevel
    fallback_reason: Optional[str]


@dataclass
class AlphaDensityResult:
    """Complete alpha density analysis result."""

    clusters: Dict[str, ClusterAlphaResult]
    global_alpha: float
    global_alpha_std: float
    config_used: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


# ==============================================================================
# Quality Gate: evaluate alpha map validity before per-cluster fallback
# ==============================================================================


@dataclass
class AlphaGateDecision:
    """Result of alpha quality gate evaluation."""

    passed: bool
    reason: str
    metrics: Dict[str, Any]


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def evaluate_alpha_quality_gate(
    alpha_map: np.ndarray,
    *,
    alpha_clip_min: float = 0.02,
    alpha_clip_max: float = 0.98,
    nan_ratio_max: float = 0.10,
    moire_score: Optional[float] = None,
    moire_max: float = 0.20,
    valid_ratio_after_min: float = 0.40,
    valid_pixels_after_min: int = 30_000,
) -> AlphaGateDecision:
    """
    Quality Gate: decide whether per-cluster L1/L2 is trustworthy or force L3.

    Gate is based on **post-clip-exclusion validity**, not raw clip ratio.
    ``clip_ratio_raw`` is kept as a diagnostic metric only.

    Args:
        alpha_map: (T, R) float alpha map
        alpha_clip_min/max: clip boundaries used during alpha computation
        nan_ratio_max: max NaN ratio (among raw total pixels)
        moire_score: optional moire severity (0-1)
        moire_max: hard-cut for moire
        valid_ratio_after_min: min ratio of valid (non-NaN, non-clipped) pixels
        valid_pixels_after_min: min absolute count of valid pixels

    Returns:
        AlphaGateDecision with pass/fail, reason, and full diagnostic metrics.
    """
    total_pixels = alpha_map.size

    # Raw counts
    nan_mask = ~np.isfinite(alpha_map)
    clip_at_min = int(np.sum(alpha_map <= alpha_clip_min + 0.001))
    clip_at_max = int(np.sum(alpha_map >= alpha_clip_max - 0.001))
    raw_nan = int(np.sum(nan_mask))
    raw_clip = clip_at_min + clip_at_max

    # Valid = not NaN and not clipped
    valid_mask = ~nan_mask & (alpha_map > alpha_clip_min + 0.001) & (alpha_map < alpha_clip_max - 0.001)
    raw_good = int(valid_mask.sum())

    nan_ratio_raw = _safe_ratio(raw_nan, total_pixels)
    clip_ratio_raw = _safe_ratio(raw_clip, total_pixels)
    valid_ratio_after = _safe_ratio(raw_good, total_pixels)

    metrics: Dict[str, Any] = {
        "total_pixels": total_pixels,
        "raw_nan": raw_nan,
        "raw_clip": raw_clip,
        "raw_good": raw_good,
        "clip_at_min": clip_at_min,
        "clip_at_max": clip_at_max,
        "nan_ratio_raw": round(nan_ratio_raw, 4),
        "clip_ratio_raw": round(clip_ratio_raw, 4),
        "valid_ratio_after_clip": round(valid_ratio_after, 4),
        "valid_pixels_after_clip": raw_good,
        "moire_severity": round(float(moire_score), 4) if moire_score is not None else None,
    }

    # ── Hard fails ──
    if total_pixels == 0:
        return AlphaGateDecision(False, "EMPTY_ALPHA_MAP", metrics)

    if moire_score is not None and moire_score > moire_max:
        return AlphaGateDecision(False, "MOIRE_TOO_HIGH", metrics)

    if raw_good < valid_pixels_after_min:
        return AlphaGateDecision(False, "TOO_FEW_VALID_PIXELS_AFTER_CLIP", metrics)

    if valid_ratio_after < valid_ratio_after_min:
        return AlphaGateDecision(False, "VALID_RATIO_TOO_LOW_AFTER_CLIP", metrics)

    if nan_ratio_raw > nan_ratio_max:
        return AlphaGateDecision(False, "NAN_RATIO_TOO_HIGH", metrics)

    return AlphaGateDecision(True, "OK", metrics)


# ==============================================================================
# Configuration defaults
# ==============================================================================

DEFAULT_ALPHA_CONFIG = {
    # Radial binning
    "n_r_bins": 20,  # Number of radial bins
    "r_start": 0.15,  # Start radius (exclude center)
    "r_end": 0.95,  # End radius (exclude edge)
    # Sample gate thresholds
    "min_samples_per_bin": 30,  # Minimum samples for valid bin
    "min_samples_ratio": 0.005,  # Or 0.5% of theta samples
    "min_valid_bins_ratio": 0.5,  # Need 50% valid bins for L1
    # Confidence and fallback
    "confidence_threshold_l1": 0.6,  # Min confidence to use L1
    "confidence_threshold_l2": 0.4,  # Min confidence to use L2
    "lerp_blend_enabled": True,  # Use confidence-weighted blending
    # Zone boundaries
    "zone_inner_end": 0.40,
    "zone_mid_end": 0.70,
    # Smoothing (optional)
    "smoothing_enabled": False,
    "smoothing_window": 3,  # Rolling median window
    # Boundary handling
    "transition_weights_enabled": False,
    "transition_use_gradient": True,
    "transition_use_boundary": True,
    "transition_boundary_weight": 0.5,
    "transition_boundary_width": 3,
    "transition_config": None,
    # ── Clip-exclusion ──
    "clip_exclude_enabled": True,  # Exclude clipped pixels from alpha stats
    # ── Quality gate (valid-pixel based, evaluated by evaluate_alpha_quality_gate) ──
    "quality_gate": {
        "valid_ratio_after_min": 0.40,  # min valid pixel ratio after clip exclusion
        "valid_pixels_after_min": 30000,  # min valid pixel count (adjust for resolution)
        "nan_ratio_max": 0.10,  # max NaN ratio (raw)
        "moire_max": 0.20,  # max moire severity (hard-cut)
    },
    # ── Small-cluster early-exit (Priority 2) ──
    "min_pixels_for_l1": 20000,  # Skip L1 if mask has fewer pixels
    # ── plate_lite fallback (Priority 1) ──
    # Hierarchy: L1 > L2 > L2_plate_lite > L3
    # plate_lite alpha is used ONLY when L1 and L2 both fail
    "plate_lite_clamp_min": 0.05,  # Safety clamp lower bound
    "plate_lite_clamp_max": 0.98,  # Safety clamp upper bound
    # Runtime keys (injected by pipeline, not user-configurable):
    # "_plate_lite_alpha_candidates": {color_id: alpha_mean, ...}
    # "_plate_lite_inks": [{alpha_mean, area_ratio, ink_key}, ...]
    # "_moire_severity": float
    # "_verification_enabled": bool
    # "_verification_agreement": float
}


# ==============================================================================
# Core alpha computation functions
# ==============================================================================


def compute_alpha_radial_1d(
    polar_alpha: np.ndarray,
    mask: np.ndarray,
    *,
    weight_map: Optional[np.ndarray] = None,
    n_bins: int = 20,
    r_start: float = 0.15,
    r_end: float = 0.95,
    min_samples_per_bin: int = 30,
    min_samples_ratio: float = 0.005,
) -> AlphaRadialProfile:
    """
    Compute 1D radial alpha profile using median_theta aggregation.

    For each radial bin, computes the median alpha across all theta (angles),
    which is robust to moire patterns and localized artifacts.

    Args:
        polar_alpha: Alpha map in polar coordinates (T, R)
        mask: Cluster mask in polar coordinates (T, R), boolean
        n_bins: Number of radial bins
        r_start: Start radius (normalized, 0-1)
        r_end: End radius (normalized, 0-1)
        min_samples_per_bin: Absolute minimum samples for valid bin
        min_samples_ratio: Minimum ratio of theta samples for valid bin

    Returns:
        AlphaRadialProfile with median alpha values and confidence
    """
    T, R = polar_alpha.shape

    # Compute minimum samples threshold (max of absolute and ratio-based)
    min_samples = max(min_samples_per_bin, int(T * min_samples_ratio))

    # Build radial bin edges and centers
    bin_edges = np.linspace(r_start, r_end, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Radial coordinate for each column
    r_coords = np.arange(R, dtype=np.float32) / max(R - 1, 1)

    # Initialize output arrays
    alpha_values = np.full(n_bins, np.nan, dtype=np.float32)
    confidence = np.zeros(n_bins, dtype=np.float32)
    n_samples = np.zeros(n_bins, dtype=np.int32)

    valid_count = 0

    for i in range(n_bins):
        r_lo, r_hi = bin_edges[i], bin_edges[i + 1]

        # Find columns in this radial bin
        col_mask = (r_coords >= r_lo) & (r_coords < r_hi)
        col_indices = np.where(col_mask)[0]

        if len(col_indices) == 0:
            continue

        # Extract alpha and mask for these columns
        bin_alpha = polar_alpha[:, col_indices]
        bin_mask = mask[:, col_indices]
        bin_weights = None
        if weight_map is not None:
            bin_weights = weight_map[:, col_indices]

        # Get valid samples (where mask is True)
        valid_mask = bin_mask.astype(bool)
        samples = bin_alpha[valid_mask]
        weights = bin_weights[valid_mask] if bin_weights is not None else None

        n_samples[i] = len(samples)

        if n_samples[i] >= min_samples:
            # Filter out NaN values before computing median
            valid_samples = samples[~np.isnan(samples)]
            if len(valid_samples) >= min_samples:
                valid_weights = None
                if weights is not None:
                    valid_weights = weights[~np.isnan(samples)]

                # Compute median_theta (median across all theta for this r-bin)
                if valid_weights is not None:
                    alpha_values[i] = _weighted_median(valid_samples, valid_weights)
                else:
                    alpha_values[i] = np.median(valid_samples)

                # Confidence based on sample count and spread
                # Higher samples and lower std -> higher confidence
                effective_samples = float(np.sum(valid_weights)) if valid_weights is not None else len(valid_samples)
                sample_conf = min(1.0, effective_samples / (min_samples * 3))
                std_conf = 1.0 / (1.0 + np.std(valid_samples))
                confidence[i] = float(sample_conf * 0.6 + std_conf * 0.4)
                valid_count += 1
            else:
                # Too many NaN values - mark as NA
                confidence[i] = 0.0
        else:
            # Insufficient samples - mark as NA
            confidence[i] = 0.0

    # Forward-fill/backward-fill for isolated NA bins (interpolation)
    alpha_values = _interpolate_na_bins(alpha_values, confidence)

    # Calculate overall quality
    quality = valid_count / n_bins if n_bins > 0 else 0.0

    return AlphaRadialProfile(
        r_bins=bin_centers,
        alpha_values=alpha_values,
        confidence=confidence,
        n_samples=n_samples,
        valid_bins=valid_count,
        total_bins=n_bins,
        quality=quality,
    )


def _interpolate_na_bins(
    values: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    """
    Interpolate NA bins using nearest valid neighbors.

    Only interpolates isolated NA bins (1-2 consecutive).
    Larger gaps remain NA (will trigger zone fallback).
    """
    result = values.copy()
    n = len(values)

    for i in range(n):
        if np.isnan(result[i]) and confidence[i] == 0:
            # Find nearest valid neighbors
            left_idx = None
            right_idx = None

            for j in range(i - 1, -1, -1):
                if not np.isnan(result[j]) and confidence[j] > 0:
                    left_idx = j
                    break

            for j in range(i + 1, n):
                if not np.isnan(values[j]) and confidence[j] > 0:
                    right_idx = j
                    break

            # Only interpolate if gap is small (max 2 bins)
            if left_idx is not None and right_idx is not None:
                gap = right_idx - left_idx
                if gap <= 3:  # interpolate gaps of 1-2 bins
                    t = (i - left_idx) / gap
                    result[i] = values[left_idx] * (1 - t) + values[right_idx] * t
            elif left_idx is not None:
                # Edge case: extend from left
                result[i] = values[left_idx]
            elif right_idx is not None:
                # Edge case: extend from right
                result[i] = values[right_idx]

    return result


def compute_alpha_zone(
    polar_alpha: np.ndarray,
    mask: np.ndarray,
    *,
    weight_map: Optional[np.ndarray] = None,
    inner_end: float = 0.40,
    mid_end: float = 0.70,
    min_samples: int = 50,
) -> AlphaZoneProfile:
    """
    Compute zone-based alpha averages (inner/mid/outer).

    This is L2 fallback when L1 radial profile has insufficient data.

    Args:
        polar_alpha: Alpha map (T, R)
        mask: Cluster mask (T, R)
        inner_end: Boundary between inner and mid zones
        mid_end: Boundary between mid and outer zones
        min_samples: Minimum samples for valid zone

    Returns:
        AlphaZoneProfile with per-zone alpha values
    """
    T, R = polar_alpha.shape
    r_coords = np.arange(R, dtype=np.float32) / max(R - 1, 1)

    # Define zone masks
    inner_cols = r_coords < inner_end
    mid_cols = (r_coords >= inner_end) & (r_coords < mid_end)
    outer_cols = r_coords >= mid_end

    def _zone_stats(col_mask: np.ndarray) -> Tuple[float, float]:
        """Compute median and confidence for a zone."""
        col_indices = np.where(col_mask)[0]
        if len(col_indices) == 0:
            return 1.0, 0.0  # Default alpha=1, no confidence

        zone_alpha = polar_alpha[:, col_indices]
        zone_mask = mask[:, col_indices]
        zone_weights = None
        if weight_map is not None:
            zone_weights = weight_map[:, col_indices]

        valid = zone_mask.astype(bool)
        samples = zone_alpha[valid]
        weights = zone_weights[valid] if zone_weights is not None else None

        # Filter out NaN values
        valid_samples = samples[~np.isnan(samples)]
        valid_weights = None
        if weights is not None:
            valid_weights = weights[~np.isnan(samples)]

        if len(valid_samples) < min_samples:
            return 1.0, 0.0

        if valid_weights is not None:
            median_val = _weighted_median(valid_samples, valid_weights)
            effective_samples = float(np.sum(valid_weights))
        else:
            median_val = float(np.median(valid_samples))
            effective_samples = len(valid_samples)
        conf = min(1.0, effective_samples / (min_samples * 5))
        return median_val, conf

    inner_alpha, inner_conf = _zone_stats(inner_cols)
    mid_alpha, mid_conf = _zone_stats(mid_cols)
    outer_alpha, outer_conf = _zone_stats(outer_cols)

    return AlphaZoneProfile(
        inner=inner_alpha,
        mid=mid_alpha,
        outer=outer_alpha,
        inner_conf=inner_conf,
        mid_conf=mid_conf,
        outer_conf=outer_conf,
    )


def compute_alpha_global(
    polar_alpha: np.ndarray,
    mask: np.ndarray,
    weight_map: Optional[np.ndarray] = None,
) -> Tuple[float, float, int]:
    """
    Compute global alpha value for a cluster.

    This is L3 fallback - simplest and most robust but least precise.

    Args:
        polar_alpha: Alpha map (T, R)
        mask: Cluster mask (T, R)

    Returns:
        (alpha_median, alpha_std, n_samples)
    """
    valid = mask.astype(bool)
    samples = polar_alpha[valid]
    weights = weight_map[valid] if weight_map is not None else None

    # Filter out NaN values
    valid_samples = samples[~np.isnan(samples)]
    valid_weights = None
    if weights is not None:
        valid_weights = weights[~np.isnan(samples)]

    if len(valid_samples) == 0:
        return 1.0, 0.0, 0  # Default to fully opaque

    if valid_weights is not None:
        median_val = _weighted_median(valid_samples, valid_weights)
        std_val = float(np.std(valid_samples))
        sample_count = int(np.sum(valid_weights))
    else:
        median_val = float(np.median(valid_samples))
        std_val = float(np.std(valid_samples))
        sample_count = len(valid_samples)
    return median_val, std_val, sample_count


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted median, ignoring NaN values."""
    if len(values) == 0:
        return float("nan")
    valid_mask = ~np.isnan(values)
    values = values[valid_mask]
    weights = weights[valid_mask]
    if len(values) == 0:
        return float("nan")
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumsum = np.cumsum(sorted_weights)
    total = cumsum[-1]
    if total == 0:
        return float(np.median(values))
    median_pos = total / 2.0
    idx = int(np.searchsorted(cumsum, median_pos))
    idx = min(idx, len(sorted_values) - 1)
    return float(sorted_values[idx])


# ==============================================================================
# 3-tier fallback system
# ==============================================================================


def apply_alpha_fallback(
    radial_profile: AlphaRadialProfile,
    zone_profile: AlphaZoneProfile,
    global_alpha: float,
    *,
    confidence_threshold_l1: float = 0.6,
    confidence_threshold_l2: float = 0.4,
    min_valid_bins_ratio: float = 0.5,
    lerp_blend: bool = True,
    plate_lite_alpha: Optional[float] = None,
    plate_lite_clamp: Tuple[float, float] = (0.05, 0.98),
    min_pixels_for_l1: int = 0,
    mask_pixel_count: int = 0,
) -> Tuple[float, AlphaFallbackLevel, Optional[str]]:
    """
    Apply 3-tier fallback to get final alpha value.

    Fallback hierarchy:
    - L1: Use radial profile mean if quality >= threshold
    - L2: Use zone-weighted average if zone confidence >= threshold
    - L2.5: Use plate_lite per-ink alpha if available (between L2 and L3)
    - L3: Use global alpha (or default 1.0)

    When lerp_blend is True, values are blended based on confidence:
        final = measured * confidence + fallback * (1 - confidence)

    Args:
        radial_profile: L1 radial alpha profile
        zone_profile: L2 zone alpha profile
        global_alpha: L3 global alpha value
        confidence_threshold_l1: Min quality for L1
        confidence_threshold_l2: Min confidence for L2
        min_valid_bins_ratio: Min ratio of valid bins for L1
        lerp_blend: Enable confidence-weighted blending
        plate_lite_alpha: Optional plate_lite per-ink alpha_mean (L2.5 fallback)
        plate_lite_clamp: Clamp range for plate_lite alpha (safety)
        min_pixels_for_l1: Skip L1 evaluation if mask has fewer pixels
        mask_pixel_count: Actual pixel count in the cluster mask

    Returns:
        (alpha_value, fallback_level, fallback_reason)
    """
    # Priority 2: Skip L1 for very small clusters (pixel-starved)
    l1_skipped_reason = None
    if min_pixels_for_l1 > 0 and 0 < mask_pixel_count < min_pixels_for_l1:
        l1_skipped_reason = f"pixels={mask_pixel_count}<min={min_pixels_for_l1}"
    else:
        # Try L1: Radial profile
        if (
            radial_profile.quality >= min_valid_bins_ratio
            and np.nanmean(radial_profile.confidence) >= confidence_threshold_l1
        ):
            # Weighted mean of radial profile
            valid_mask = ~np.isnan(radial_profile.alpha_values)
            if np.any(valid_mask):
                weights = radial_profile.confidence[valid_mask]
                values = radial_profile.alpha_values[valid_mask]

                if weights.sum() > 0:
                    alpha_l1 = float(np.average(values, weights=weights))

                    if lerp_blend:
                        # Blend with L2 based on L1 confidence
                        l1_conf = radial_profile.quality
                        l2_alpha = _compute_zone_weighted_alpha(zone_profile)
                        alpha_final = alpha_l1 * l1_conf + l2_alpha * (1 - l1_conf)
                    else:
                        alpha_final = alpha_l1

                    return alpha_final, AlphaFallbackLevel.L1_RADIAL, None

        l1_skipped_reason = f"L1_quality={radial_profile.quality:.2f}<{min_valid_bins_ratio}"

    # Try L2: Zone profile — use weighted confidence of zones that have data
    zone_confs = []
    zone_vals = []
    for zv, zc in [
        (zone_profile.inner, zone_profile.inner_conf),
        (zone_profile.mid, zone_profile.mid_conf),
        (zone_profile.outer, zone_profile.outer_conf),
    ]:
        if zc > 0:
            zone_confs.append(zc)
            zone_vals.append(zv)

    if zone_confs:
        zone_conf_effective = float(np.mean(zone_confs))
    else:
        zone_conf_effective = 0.0

    if zone_conf_effective >= confidence_threshold_l2:
        alpha_l2 = _compute_zone_weighted_alpha(zone_profile)

        if lerp_blend:
            # Blend with L3 based on L2 confidence
            alpha_final = alpha_l2 * zone_conf_effective + global_alpha * (1 - zone_conf_effective)
        else:
            alpha_final = alpha_l2

        return alpha_final, AlphaFallbackLevel.L2_ZONE, l1_skipped_reason

    # Try L2.5: plate_lite per-ink alpha (if available)
    if plate_lite_alpha is not None:
        clamped = float(np.clip(plate_lite_alpha, plate_lite_clamp[0], plate_lite_clamp[1]))
        reason = f"{l1_skipped_reason}; L2_zone_conf={zone_conf_effective:.2f}<{confidence_threshold_l2}"
        return clamped, AlphaFallbackLevel.L2_PLATE_LITE, reason

    # L3: Global fallback
    reason = f"{l1_skipped_reason}; L2_conf={zone_conf_effective:.2f}<{confidence_threshold_l2}"
    return global_alpha, AlphaFallbackLevel.L3_GLOBAL, reason


def _compute_zone_weighted_alpha(zone_profile: AlphaZoneProfile) -> float:
    """Compute confidence-weighted average of zone alphas."""
    values = np.array([zone_profile.inner, zone_profile.mid, zone_profile.outer])
    confs = np.array([zone_profile.inner_conf, zone_profile.mid_conf, zone_profile.outer_conf])

    if confs.sum() == 0:
        return 1.0  # Default

    return float(np.average(values, weights=confs))


# ==============================================================================
# Main API: compute_effective_density
# ==============================================================================


def compute_effective_density(
    polar_alpha: Optional[np.ndarray],
    cluster_masks: Dict[str, np.ndarray],
    area_ratios: Dict[str, float],
    alpha_weight_map: Optional[np.ndarray] = None,
    polar_lab: Optional[np.ndarray] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> AlphaDensityResult:
    """
    Compute effective density for all clusters with 3-tier alpha fallback.

    effective_density_i = area_ratio_i * alpha_i

    Where alpha_i is determined by the fallback hierarchy:
    - L1: alpha_i(r) from radial profile (best, if enough samples)
    - L2: alpha_zone_i from zone averages (fallback if L1 fails)
    - L3: alpha_global or 1.0 (final fallback)

    Args:
        polar_alpha: Alpha map from plate_gate (T, R), or None if unavailable
        cluster_masks: Dict mapping color_id to boolean mask (T, R)
        area_ratios: Dict mapping color_id to area ratio [0, 1]
        cfg: Configuration dict (uses DEFAULT_ALPHA_CONFIG if None)

    Returns:
        AlphaDensityResult containing per-cluster effective densities
    """
    # Merge config with defaults
    config = {**DEFAULT_ALPHA_CONFIG, **(cfg or {})}
    warnings: List[str] = []

    # Handle missing alpha map
    if polar_alpha is None:
        warnings.append("ALPHA_MAP_UNAVAILABLE")
        # Return area_ratio as effective_density (alpha=1.0)
        clusters = {}
        for color_id, area_ratio in area_ratios.items():
            clusters[color_id] = ClusterAlphaResult(
                color_id=color_id,
                area_ratio=area_ratio,
                radial_profile=None,
                zone_profile=None,
                alpha_global=1.0,
                effective_density=area_ratio,
                alpha_used=1.0,
                fallback_level=AlphaFallbackLevel.L3_GLOBAL,
                fallback_reason="no_alpha_map",
            )
        return AlphaDensityResult(
            clusters=clusters,
            global_alpha=1.0,
            global_alpha_std=0.0,
            config_used=config,
            warnings=warnings,
        )

    T, R = polar_alpha.shape

    # ── Quality Gate: delegate to evaluate_alpha_quality_gate() ─────────
    alpha_clip_min = config.get("alpha_clip_min", 0.02)
    alpha_clip_max = config.get("alpha_clip_max", 0.98)
    moire_severity = float(config.get("_moire_severity", 0.0))

    qg_cfg = config.get("quality_gate", {})
    gate = evaluate_alpha_quality_gate(
        polar_alpha,
        alpha_clip_min=alpha_clip_min,
        alpha_clip_max=alpha_clip_max,
        nan_ratio_max=float(qg_cfg.get("nan_ratio_max", 0.10)),
        moire_score=moire_severity if moire_severity > 0 else None,
        moire_max=float(qg_cfg.get("moire_max", 0.20)),
        valid_ratio_after_min=float(qg_cfg.get("valid_ratio_after_min", 0.40)),
        valid_pixels_after_min=int(qg_cfg.get("valid_pixels_after_min", 30000)),
    )

    # plate_lite availability can soften the gate
    has_plate_lite = bool(config.get("_plate_lite_alpha_candidates") or config.get("_plate_lite_inks"))

    _quality_gate_debug = {
        **gate.metrics,
        "has_plate_lite": has_plate_lite,
        "gate_passed": gate.passed or has_plate_lite,
        "gate_reason": gate.reason,
    }

    if not gate.passed:
        if has_plate_lite:
            # plate_lite available → soft pass (plate_lite will serve as fallback)
            warnings.append(
                f"ALPHA_QUALITY_GATE_SOFT_PASS(plate_lite) "
                f"reason={gate.reason} "
                f"valid={gate.metrics.get('valid_ratio_after_clip', 0):.1%} "
                f"pixels={gate.metrics.get('valid_pixels_after_clip', 0)} "
                f"moire={moire_severity:.2f}"
            )
        else:
            reason = f"alpha_quality_fail: {gate.reason}"
            warnings.append(
                f"ALPHA_QUALITY_GATE_FAILED "
                f"reason={gate.reason} "
                f"valid={gate.metrics.get('valid_ratio_after_clip', 0):.1%} "
                f"pixels={gate.metrics.get('valid_pixels_after_clip', 0)} "
                f"clip_raw={gate.metrics.get('clip_ratio_raw', 0):.1%} "
                f"moire={moire_severity:.2f}"
            )
            clusters = {}
            for color_id, area_ratio in area_ratios.items():
                clusters[color_id] = ClusterAlphaResult(
                    color_id=color_id,
                    area_ratio=area_ratio,
                    radial_profile=None,
                    zone_profile=None,
                    alpha_global=1.0,
                    effective_density=area_ratio,
                    alpha_used=1.0,
                    fallback_level=AlphaFallbackLevel.L3_GLOBAL,
                    fallback_reason=reason,
                )
            return AlphaDensityResult(
                clusters=clusters,
                global_alpha=1.0,
                global_alpha_std=0.0,
                config_used={**config, "_quality_gate_debug": _quality_gate_debug},
                warnings=warnings,
            )

    if config.get("transition_weights_enabled") and alpha_weight_map is None and polar_lab is not None:
        from .transition_detector import TransitionConfig, create_alpha_weight_map

        trans_cfg_dict = config.get("transition_config") or {}
        trans_cfg = TransitionConfig(**trans_cfg_dict) if trans_cfg_dict else None
        alpha_weight_map, weight_meta = create_alpha_weight_map(
            polar_lab,
            cluster_masks,
            use_gradient_weights=bool(config.get("transition_use_gradient", True)),
            use_boundary_weights=bool(config.get("transition_use_boundary", True)),
            gradient_config=trans_cfg,
            boundary_width=int(config.get("transition_boundary_width", 3)),
            boundary_weight=float(config.get("transition_boundary_weight", 0.5)),
        )
        warnings.append(f"ALPHA_WEIGHT_MAP_APPLIED mean_weight={weight_meta.get('mean_weight', 1.0):.3f}")

    # Priority 3: Clip-exclusion mask — exclude pixels at clip boundaries
    # from radial/zone/global alpha computation for cleaner statistics
    total_pixels = T * R
    clip_exclude_enabled = bool(config.get("clip_exclude_enabled", True))
    if clip_exclude_enabled:
        valid_alpha_mask = (
            ~np.isnan(polar_alpha) & (polar_alpha > alpha_clip_min + 0.001) & (polar_alpha < alpha_clip_max - 0.001)
        )
        clean_ratio = float(valid_alpha_mask.sum()) / total_pixels if total_pixels > 0 else 0.0
        if clean_ratio < 0.3:
            # Too few valid pixels after clip exclusion, fall back to raw (no copy needed)
            polar_alpha_clean = polar_alpha
            warnings.append(f"CLIP_EXCLUDE_INSUFFICIENT valid={clean_ratio:.1%}, using raw alpha")
        else:
            # Only copy when we actually need to modify values
            polar_alpha_clean = polar_alpha.copy()
            polar_alpha_clean[~valid_alpha_mask] = np.nan
            excluded_pct = 1.0 - clean_ratio
            if excluded_pct > 0.05:
                warnings.append(f"CLIP_EXCLUDE_APPLIED excluded={excluded_pct:.1%}")
    else:
        polar_alpha_clean = polar_alpha

    # Compute global alpha (for L3 fallback)
    # Use all pixels with any cluster mask
    combined_mask = np.zeros((T, R), dtype=bool)
    for mask in cluster_masks.values():
        combined_mask |= mask.astype(bool)

    global_alpha, global_alpha_std, _ = compute_alpha_global(polar_alpha_clean, combined_mask, alpha_weight_map)

    # P2-3: Adjust confidence thresholds based on verification agreement
    # If verification shows disagreement, increase thresholds → more likely to fallback to L3
    verification_agreement = float(config.get("_verification_agreement", 1.0))
    verification_enabled = config.get("_verification_enabled", False)

    # Base thresholds from config
    base_conf_l1 = config["confidence_threshold_l1"]
    base_conf_l2 = config["confidence_threshold_l2"]

    if verification_enabled and verification_agreement < 1.0:
        # Scale thresholds inversely with agreement
        # Low agreement (0.5) → multiply threshold by 1.5
        # High agreement (0.9) → multiply threshold by 1.1
        # Perfect agreement (1.0) → no change
        threshold_multiplier = 1.0 + (1.0 - verification_agreement) * 0.5
        adjusted_conf_l1 = min(0.95, base_conf_l1 * threshold_multiplier)
        adjusted_conf_l2 = min(0.95, base_conf_l2 * threshold_multiplier)
        warnings.append(
            f"VERIFICATION_CONFIDENCE_ADJUSTMENT: agreement={verification_agreement:.3f}, "
            f"L1_th={base_conf_l1:.2f}->{adjusted_conf_l1:.2f}, "
            f"L2_th={base_conf_l2:.2f}->{adjusted_conf_l2:.2f}"
        )
    else:
        adjusted_conf_l1 = base_conf_l1
        adjusted_conf_l2 = base_conf_l2

    # Priority 1: Extract plate_lite per-ink alpha candidates
    plate_lite_candidates = config.get("_plate_lite_alpha_candidates", {})
    plate_lite_clamp = (
        float(config.get("plate_lite_clamp_min", 0.05)),
        float(config.get("plate_lite_clamp_max", 0.98)),
    )

    # Priority 2: min_pixels threshold for L1 early-exit
    min_pixels_for_l1 = int(config.get("min_pixels_for_l1", 20000))

    # Process each cluster
    clusters: Dict[str, ClusterAlphaResult] = {}

    for color_id, mask in cluster_masks.items():
        area_ratio = area_ratios.get(color_id, 0.0)
        mask_pixel_count = int(mask.sum())

        # L1: Radial profile (using clip-excluded alpha)
        radial_profile = compute_alpha_radial_1d(
            polar_alpha_clean,
            mask,
            weight_map=alpha_weight_map,
            n_bins=config["n_r_bins"],
            r_start=config["r_start"],
            r_end=config["r_end"],
            min_samples_per_bin=config["min_samples_per_bin"],
            min_samples_ratio=config["min_samples_ratio"],
        )

        # L2: Zone profile (using clip-excluded alpha)
        zone_profile = compute_alpha_zone(
            polar_alpha_clean,
            mask,
            weight_map=alpha_weight_map,
            inner_end=config["zone_inner_end"],
            mid_end=config["zone_mid_end"],
        )

        # L3: Global alpha (cluster-specific, using clip-excluded alpha)
        cluster_alpha, cluster_std, n_samples = compute_alpha_global(polar_alpha_clean, mask, alpha_weight_map)

        # Resolve plate_lite alpha for this cluster (if available)
        pl_alpha = plate_lite_candidates.get(color_id)

        # Apply fallback (P2-3: use adjusted thresholds based on verification)
        alpha_used, fallback_level, fallback_reason = apply_alpha_fallback(
            radial_profile,
            zone_profile,
            cluster_alpha if n_samples > 0 else global_alpha,
            confidence_threshold_l1=adjusted_conf_l1,
            confidence_threshold_l2=adjusted_conf_l2,
            min_valid_bins_ratio=config["min_valid_bins_ratio"],
            lerp_blend=config["lerp_blend_enabled"],
            plate_lite_alpha=pl_alpha,
            plate_lite_clamp=plate_lite_clamp,
            min_pixels_for_l1=min_pixels_for_l1,
            mask_pixel_count=mask_pixel_count,
        )

        # Compute effective density
        effective_density = area_ratio * alpha_used

        # Track warnings
        if fallback_level == AlphaFallbackLevel.L3_GLOBAL:
            warnings.append(f"ALPHA_L3_FALLBACK_{color_id}")
        elif fallback_level == AlphaFallbackLevel.L2_PLATE_LITE:
            warnings.append(f"ALPHA_L2_PLATE_LITE_{color_id} pl_alpha={pl_alpha:.3f}")
        elif fallback_level == AlphaFallbackLevel.L2_ZONE:
            warnings.append(f"ALPHA_L2_FALLBACK_{color_id}")

        clusters[color_id] = ClusterAlphaResult(
            color_id=color_id,
            area_ratio=area_ratio,
            radial_profile=radial_profile,
            zone_profile=zone_profile,
            alpha_global=cluster_alpha,
            effective_density=effective_density,
            alpha_used=alpha_used,
            fallback_level=fallback_level,
            fallback_reason=fallback_reason,
        )

    return AlphaDensityResult(
        clusters=clusters,
        global_alpha=global_alpha,
        global_alpha_std=global_alpha_std,
        config_used={**config, "_quality_gate_debug": _quality_gate_debug},
        warnings=warnings,
    )


# ==============================================================================
# Utility functions for integration
# ==============================================================================


def extract_effective_densities(result: AlphaDensityResult) -> Dict[str, float]:
    """
    Extract effective densities as simple dict.

    Args:
        result: AlphaDensityResult from compute_effective_density

    Returns:
        Dict mapping color_id to effective_density
    """
    return {color_id: cluster.effective_density for color_id, cluster in result.clusters.items()}


def extract_alpha_summary(result: AlphaDensityResult) -> Dict[str, Dict[str, Any]]:
    """
    Extract alpha summary for metadata export.

    Args:
        result: AlphaDensityResult from compute_effective_density

    Returns:
        Dict with per-cluster alpha summary suitable for JSON export
    """
    summary = {}
    for color_id, cluster in result.clusters.items():
        summary[color_id] = {
            "area_ratio": round(cluster.area_ratio, 4),
            "alpha_used": round(cluster.alpha_used, 4),
            "effective_density": round(cluster.effective_density, 4),
            "fallback_level": cluster.fallback_level.value,
            "fallback_reason": cluster.fallback_reason,
            "radial_quality": round(cluster.radial_profile.quality, 3) if cluster.radial_profile else None,
            "zone_confidences": (
                {
                    "inner": round(cluster.zone_profile.inner_conf, 3),
                    "mid": round(cluster.zone_profile.mid_conf, 3),
                    "outer": round(cluster.zone_profile.outer_conf, 3),
                }
                if cluster.zone_profile
                else None
            ),
        }
    return summary


def build_alpha_map_polar(
    white_bgr: np.ndarray,
    black_bgr: np.ndarray,
    geom: "LensGeometry",
    polar_R: int,
    polar_T: int,
    *,
    alpha_clip_min: float = 0.02,
    alpha_clip_max: float = 0.98,
) -> np.ndarray:
    """
    Build alpha map directly in polar coordinates.

    This is a simplified version of plate_gate alpha computation
    that outputs directly in polar coordinates.

    Args:
        white_bgr: White backlight image
        black_bgr: Black backlight image
        geom: Lens geometry
        polar_R: Radial dimension
        polar_T: Theta dimension
        alpha_clip_min: Minimum alpha value
        alpha_clip_max: Maximum alpha value

    Returns:
        Alpha map (T, R) in polar coordinates
    """
    from ...signature.radial_signature import to_polar

    # Convert to polar
    polar_white = to_polar(white_bgr, geom, R=polar_R, T=polar_T)
    polar_black = to_polar(black_bgr, geom, R=polar_R, T=polar_T)

    # Compute alpha (simplified)
    w = polar_white.astype(np.float32)
    b = polar_black.astype(np.float32)

    # Background estimation from outer edge (last 5% of radius)
    outer_start = int(polar_R * 0.95)
    w_bg = np.median(w[:, outer_start:, :], axis=(0, 1))
    b_bg = np.median(b[:, outer_start:, :], axis=(0, 1))

    # Alpha = |W - W_bg - B + B_bg| / |W - B|
    diff = np.abs(w - w_bg - b + b_bg)
    denom = np.abs(w - b)
    denom = np.maximum(denom, 20.0)  # Prevent division by zero

    alpha = diff / denom
    alpha = np.clip(alpha, alpha_clip_min, alpha_clip_max)
    alpha = np.mean(alpha, axis=2)  # Average across channels

    return alpha


# ==============================================================================
# P1-1: Registration-less Polar Alpha
# ==============================================================================


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

    Key insight: By computing alpha in polar coordinates and using median_θ
    aggregation, we achieve rotation invariance without needing 2D registration.
    This is robust to rotational misalignment between white/black images.

    Algorithm:
    1. Convert both white and black images to polar coordinates
    2. Compute per-pixel alpha using standard formula
    3. For each radial bin r, take median(alpha[θ, r]) across all θ
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
    # Formula: α = |W - W_bg - B + B_bg| / |W - B|
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

    # Compute 1D radial profile using median_θ (rotation-invariant) — vectorized
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


# P1-2: Alpha verification functions (compare registration-less with plate-based)


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
