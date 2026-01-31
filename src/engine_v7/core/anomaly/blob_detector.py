from __future__ import annotations

import math
from typing import Any, Dict, List

import cv2
import numpy as np

from ..types import LensGeometry


def detect_center_blobs(
    bgr: np.ndarray,
    geom: LensGeometry,
    *,
    frac: float = 0.28,
    min_area: int = 25,
    annulus_inner: float = 0.40,
    annulus_outer: float = 0.98,
    overlap_threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    Detect dark blobs in the center area (smudges/marks).
    Uses blackhat + Otsu + connected components with annulus overlap filtering.

    Args:
        bgr: Input BGR image
        geom: Lens geometry
        frac: Center ROI size as fraction of radius (default: 0.28)
        min_area: Minimum blob area fallback (will be overridden by dynamic calculation)
        annulus_inner: Inner radius of pattern annulus (default: 0.40)
        annulus_outer: Outer radius of pattern annulus (default: 0.98)
        overlap_threshold: Max allowed overlap ratio with annulus (default: 0.05)

    Returns:
        Dict with blob count, rejected count, and metadata
    """
    h, w = bgr.shape[:2]

    # 1. Calculate dynamic min_area based on lens radius
    # min_area = max(80, int(0.00025 * π * R²))
    min_area_dynamic = max(80, int(0.00025 * math.pi * geom.r**2))
    min_area_used = max(min_area, min_area_dynamic)  # Use larger of config and dynamic

    # 2. Create annulus mask (pattern region) to filter out edge artifacts
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_sq = (x_grid - geom.cx) ** 2 + (y_grid - geom.cy) ** 2
    r_inner = geom.r * annulus_inner
    r_outer = geom.r * annulus_outer
    annulus_mask = (dist_sq >= r_inner**2) & (dist_sq <= r_outer**2)

    # 3. Dilate annulus mask to catch edge blobs that bleed into center
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    annulus_dilated = cv2.dilate(annulus_mask.astype(np.uint8), kernel).astype(bool)

    # 4. Extract center ROI
    r = geom.r * frac
    x0 = int(max(0, geom.cx - r))
    x1 = int(min(w, geom.cx + r))
    y0 = int(max(0, geom.cy - r))
    y1 = int(min(h, geom.cy + r))
    roi = bgr[y0:y1, x0:x1]

    # 5. Blackhat + Otsu thresholding
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    _, thr = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6. Connected components
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(thr, connectivity=8)

    # 7. Filter blobs with multiple criteria
    blobs: List[Dict[str, float]] = []
    overlap_rejected_count = 0
    size_rejected_count = 0

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])

        # Size filter
        if area < min_area_used:
            size_rejected_count += 1
            continue

        # Create blob mask in full image coordinates
        blob_mask_roi = labels == i
        blob_mask_full = np.zeros((h, w), dtype=bool)
        blob_mask_full[y0:y1, x0:x1] = blob_mask_roi

        # Annulus overlap filter
        overlap_pixels = (blob_mask_full & annulus_dilated).sum()
        overlap_ratio = overlap_pixels / area

        if overlap_ratio > overlap_threshold:
            overlap_rejected_count += 1
            continue

        # Blob passes all filters - calculate metadata
        cx, cy = centroids[i]

        # Calculate eccentricity from moments
        moments = cv2.moments(blob_mask_roi.astype(np.uint8))
        eccentricity = 0.0

        if moments["m00"] > 0:
            # Calculate second central moments
            mu20 = moments["mu20"] / moments["m00"]
            mu02 = moments["mu02"] / moments["m00"]
            mu11 = moments["mu11"] / moments["m00"]

            # Calculate eigenvalues to get major/minor axis
            common = np.sqrt(4 * mu11**2 + (mu20 - mu02) ** 2)
            lambda1 = (mu20 + mu02 + common) / 2
            lambda2 = (mu20 + mu02 - common) / 2

            if lambda1 > 1e-6:
                eccentricity = np.sqrt(1 - (lambda2 / lambda1))

        blobs.append(
            {
                "area": float(area),
                "cx": float(cx + x0),
                "cy": float(cy + y0),
                "eccentricity": round(float(eccentricity), 3),
                "overlap_ratio": round(float(overlap_ratio), 3),
            }
        )

    # 8. Calculate statistics
    total_area = sum(b["area"] for b in blobs)
    avg_area = total_area / len(blobs) if blobs else 0.0
    max_area = max((b["area"] for b in blobs), default=0.0)

    return {
        "blob_count": len(blobs),
        "total_area": float(total_area),
        "avg_area_per_blob": float(avg_area),
        "max_area": float(max_area),
        "blobs": blobs,
        "roi_box": [x0, y0, x1, y1],
        # Filtering metadata (for debugging/monitoring)
        "overlap_rejected_count": overlap_rejected_count,
        "size_rejected_count": size_rejected_count,
        "min_area_used": min_area_used,
        "r_center_used": float(r),
        "annulus_config": {
            "inner": annulus_inner,
            "outer": annulus_outer,
            "overlap_threshold": overlap_threshold,
        },
    }
