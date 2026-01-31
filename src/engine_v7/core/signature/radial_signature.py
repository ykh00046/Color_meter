from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Literal, Tuple

import cv2
import numpy as np

from ..types import LensGeometry
from ..utils import to_cie_lab


class AggregationMethod(Enum):
    """Theta aggregation method for radial signature."""

    MEAN = "mean"
    MEDIAN = "median"


def to_polar(bgr: np.ndarray, geom: LensGeometry, *, R: int, T: int) -> np.ndarray:
    """
    cv2.warpPolar output: (T, R, 3)
    - rows: theta (angle)
    - cols: r (radius)
    """
    polar = cv2.warpPolar(
        bgr,
        (R, T),
        (geom.cx, geom.cy),
        geom.r,
        flags=cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS,
    )
    return polar


def build_radial_signature(
    polar_bgr: np.ndarray,
    *,
    r_start: float,
    r_end: float,
    aggregation: Literal["mean", "median"] = "median",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Build radial signature by aggregating theta (angle) axis.

    Args:
        polar_bgr: Polar BGR image (T, R, 3)
        r_start: Start radius (normalized 0-1)
        r_end: End radius (normalized 0-1)
        aggregation: "median" (default, robust) or "mean" (legacy)

    Returns:
        (central_curve, p95_curve, meta)
        - central_curve: median or mean across theta for each radius
        - p95_curve: 95th percentile across theta
    """
    lab = to_cie_lab(polar_bgr)
    T, R, _ = lab.shape
    r0 = int(R * r_start)
    r1 = int(R * r_end)
    r0 = max(0, min(R - 1, r0))
    r1 = max(r0 + 1, min(R, r1))

    lab_roi = lab[:, r0:r1, :]
    if lab_roi.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            {"T": T, "R": R, "r0": r0, "r1": r1, "aggregation": aggregation},
        )

    # Use median_theta for robustness against outliers/moire
    if aggregation == "median":
        central_curve = np.median(lab_roi, axis=0).astype(np.float32)  # (R', 3)
    else:
        central_curve = lab_roi.mean(axis=0).astype(np.float32)  # (R', 3)

    p95_curve = np.percentile(lab_roi, 95, axis=0).astype(np.float32)
    meta = {"T": T, "R": R, "r0": r0, "r1": r1, "aggregation": aggregation}
    return central_curve, p95_curve, meta


def build_radial_signature_masked(
    polar_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    r_start: float,
    r_end: float,
    aggregation: Literal["mean", "median"] = "median",
    min_samples_per_bin: int = 5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Build radial signature using only pixels where mask is True.

    Uses median_theta aggregation by default for robustness against
    moire patterns and localized artifacts.

    Args:
        polar_bgr: Polar BGR image (T, R, 3)
        mask: Boolean or uint8 mask (T, R)
        r_start: Start radius (normalized 0-1)
        r_end: End radius (normalized 0-1)
        aggregation: "median" (default, robust) or "mean" (legacy)
        min_samples_per_bin: Minimum samples for valid radial bin

    Returns:
        (central_curve, p95_curve, meta)
    """
    lab = to_cie_lab(polar_bgr)
    T, R, _ = lab.shape
    r0 = int(R * r_start)
    r1 = int(R * r_end)
    r0 = max(0, min(R - 1, r0))
    r1 = max(r0 + 1, min(R, r1))

    lab_roi = lab[:, r0:r1, :]
    mask_roi = mask[:, r0:r1]

    if mask_roi.ndim == 3:
        mask_roi = mask_roi[..., 0]  # Handle if mask has channels

    R_prime = r1 - r0
    # Initialize with NaN instead of 0 to distinguish empty bins
    central_curve = np.full((R_prime, 3), np.nan, dtype=np.float32)
    p95_curve = np.full((R_prime, 3), np.nan, dtype=np.float32)

    # Track statistics
    first_valid_r_idx = None
    valid_bin_count = 0
    sample_counts = np.zeros(R_prime, dtype=np.int32)

    # First pass: Fill valid columns using median_theta
    for r_idx in range(R_prime):
        col_mask = mask_roi[:, r_idx] > 0
        n_samples = np.sum(col_mask)
        sample_counts[r_idx] = n_samples

        if n_samples >= min_samples_per_bin:
            if first_valid_r_idx is None:
                first_valid_r_idx = r_idx
            pixels = lab_roi[col_mask, r_idx, :]

            # Use median for robustness
            if aggregation == "median":
                central_curve[r_idx] = np.median(pixels, axis=0)
            else:
                central_curve[r_idx] = pixels.mean(axis=0)

            p95_curve[r_idx] = np.percentile(pixels, 95, axis=0)
            valid_bin_count += 1
        else:
            # Insufficient samples - hold last valid value (will be interpolated)
            if r_idx > 0 and not np.isnan(central_curve[r_idx - 1, 0]):
                central_curve[r_idx] = central_curve[r_idx - 1]
                p95_curve[r_idx] = p95_curve[r_idx - 1]

    # Second pass: Forward-fill from first valid column
    if first_valid_r_idx is not None and first_valid_r_idx > 0:
        first_valid_central = central_curve[first_valid_r_idx]
        first_valid_p95 = p95_curve[first_valid_r_idx]
        for r_idx in range(first_valid_r_idx):
            central_curve[r_idx] = first_valid_central
            p95_curve[r_idx] = first_valid_p95

    meta = {
        "T": T,
        "R": R,
        "r0": r0,
        "r1": r1,
        "first_valid_r_idx": first_valid_r_idx,
        "aggregation": aggregation,
        "valid_bin_count": valid_bin_count,
        "total_bins": R_prime,
        "quality": valid_bin_count / R_prime if R_prime > 0 else 0.0,
        "sample_counts": sample_counts.tolist(),
    }
    return central_curve, p95_curve, meta


# Legacy alias for backward compatibility
def build_radial_signature_masked_legacy(
    polar_bgr: np.ndarray, mask: np.ndarray, *, r_start: float, r_end: float
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Legacy function using mean aggregation. Use build_radial_signature_masked instead."""
    return build_radial_signature_masked(polar_bgr, mask, r_start=r_start, r_end=r_end, aggregation="mean")
