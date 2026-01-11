from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from ..types import LensGeometry


def detect_center_blobs(
    bgr: np.ndarray, geom: LensGeometry, *, frac: float = 0.28, min_area: int = 25
) -> Dict[str, Any]:
    """
    Detect dark blobs in the center area (smudges/marks).
    Uses blackhat + Otsu + connected components.
    """
    h, w = bgr.shape[:2]
    r = geom.r * frac
    x0 = int(max(0, geom.cx - r))
    x1 = int(min(w, geom.cx + r))
    y0 = int(max(0, geom.cy - r))
    y1 = int(min(h, geom.cy + r))
    roi = bgr[y0:y1, x0:x1]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    _, thr = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(thr, connectivity=8)
    blobs: List[Dict[str, float]] = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            cx, cy = centroids[i]

            # Calculate eccentricity from moments
            blob_mask = (labels == i).astype(np.uint8)
            moments = cv2.moments(blob_mask)
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
                }
            )

    total_area = sum(b["area"] for b in blobs)
    avg_area = total_area / len(blobs) if blobs else 0.0

    return {
        "blob_count": len(blobs),
        "total_area": float(total_area),
        "avg_area_per_blob": float(avg_area),
        "blobs": blobs,
        "roi_box": [x0, y0, x1, y1],
    }
