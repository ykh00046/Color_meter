from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np

from ..types import LensGeometry
from ..utils import bgr_to_lab


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
    polar_bgr: np.ndarray, *, r_start: float, r_end: float
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    lab = bgr_to_lab(polar_bgr)
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
            {"T": T, "R": R, "r0": r0, "r1": r1},
        )

    mean_curve = lab_roi.mean(axis=0)  # (R',3)
    p95_curve = np.percentile(lab_roi, 95, axis=0).astype(np.float32)
    meta = {"T": T, "R": R, "r0": r0, "r1": r1}
    return mean_curve, p95_curve, meta


def build_radial_signature_masked(
    polar_bgr: np.ndarray, mask: np.ndarray, *, r_start: float, r_end: float
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Build radial signature using only pixels where mask is True.
    mask shape: (T, R) - boolean or uint8
    """
    lab = bgr_to_lab(polar_bgr)
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
    mean_curve = np.full((R_prime, 3), np.nan, dtype=np.float32)
    p95_curve = np.full((R_prime, 3), np.nan, dtype=np.float32)

    # Vectorized approach or per-column?
    # Since mask can be arbitrary, per-column is safest for "radial" statistics.
    # Radial signature = stats over theta (rows) for each radius (col).

    # Track first valid r_idx for metadata
    first_valid_r_idx = None

    # First pass: Fill valid columns
    for r_idx in range(R_prime):
        col_mask = mask_roi[:, r_idx] > 0
        if np.any(col_mask):
            if first_valid_r_idx is None:
                first_valid_r_idx = r_idx
            pixels = lab_roi[col_mask, r_idx, :]
            mean_curve[r_idx] = pixels.mean(axis=0)
            p95_curve[r_idx] = np.percentile(pixels, 95, axis=0)
        else:
            # If a radius has NO masked pixels (e.g. gap), hold last valid value
            if r_idx > 0 and not np.isnan(mean_curve[r_idx - 1, 0]):
                mean_curve[r_idx] = mean_curve[r_idx - 1]
                p95_curve[r_idx] = p95_curve[r_idx - 1]
            # else remains NaN (will be forward-filled below if possible)

    # Second pass: Forward-fill from first valid column (if r_idx==0 was empty)
    if first_valid_r_idx is not None and first_valid_r_idx > 0:
        # Fill 0 ~ first_valid_r_idx-1 with first valid value
        first_valid_mean = mean_curve[first_valid_r_idx]
        first_valid_p95 = p95_curve[first_valid_r_idx]
        for r_idx in range(first_valid_r_idx):
            mean_curve[r_idx] = first_valid_mean
            p95_curve[r_idx] = first_valid_p95

    meta = {
        "T": T,
        "R": R,
        "r0": r0,
        "r1": r1,
        "first_valid_r_idx": first_valid_r_idx,
    }
    return mean_curve, p95_curve, meta
