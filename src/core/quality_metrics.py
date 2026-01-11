import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_quality_metrics(
    image_bgr: np.ndarray,
    lens_detection: Optional[Any] = None,
    include_dot_stats: bool = True,
) -> Dict[str, Any]:
    if image_bgr is None or image_bgr.size == 0:
        return {}

    lens_mask = None
    if lens_detection is not None:
        lens_mask = _circle_mask(
            image_bgr.shape[:2],
            float(lens_detection.center_x),
            float(lens_detection.center_y),
            float(lens_detection.radius),
        )

    metrics = {
        "blur": {"score": compute_blur_score(image_bgr)},
        "histogram": compute_histograms(image_bgr, lens_mask=lens_mask),
    }

    if include_dot_stats:
        dot_stats = compute_dot_stats(image_bgr, lens_mask=lens_mask)
        if dot_stats is not None:
            metrics["dot_stats"] = dot_stats

    return metrics


def compute_blur_score(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_histograms(
    image_bgr: np.ndarray,
    lens_mask: Optional[np.ndarray] = None,
    bins: int = 32,
) -> Dict[str, Any]:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    mask = lens_mask
    if mask is not None and mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    return {
        "bins": bins,
        "lab": {
            "L": _calc_hist(lab, 0, bins, (0, 256), mask),
            "a": _calc_hist(lab, 1, bins, (0, 256), mask),
            "b": _calc_hist(lab, 2, bins, (0, 256), mask),
        },
        "hsv": {
            "H": _calc_hist(hsv, 0, bins, (0, 180), mask),
            "S": _calc_hist(hsv, 1, bins, (0, 256), mask),
            "V": _calc_hist(hsv, 2, bins, (0, 256), mask),
        },
    }


def compute_dot_stats(
    image_bgr: np.ndarray,
    lens_mask: Optional[np.ndarray] = None,
    min_area: int = 5,
) -> Optional[Dict[str, Any]]:
    if lens_mask is None:
        return None

    from src.core.zone_analyzer_2d import InkMaskConfig, build_ink_mask

    ink_mask = build_ink_mask(image_bgr, lens_mask, InkMaskConfig())
    ink_mask = cv2.bitwise_and(ink_mask, ink_mask, mask=lens_mask)

    lens_pixels = int(np.sum(lens_mask > 0))
    if lens_pixels <= 0:
        return None

    ink_pixels = int(np.sum(ink_mask > 0))
    dot_coverage = ink_pixels / lens_pixels

    contours, _ = cv2.findContours(ink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    areas = [a for a in areas if a >= min_area]

    if not areas:
        return {
            "dot_count": 0,
            "dot_coverage": float(dot_coverage),
            "dot_area_mean": 0.0,
            "dot_area_std": 0.0,
            "dot_area_min": 0.0,
            "dot_area_max": 0.0,
        }

    areas_arr = np.asarray(areas, dtype=np.float32)
    return {
        "dot_count": int(len(areas)),
        "dot_coverage": float(dot_coverage),
        "dot_area_mean": float(np.mean(areas_arr)),
        "dot_area_std": float(np.std(areas_arr)),
        "dot_area_min": float(np.min(areas_arr)),
        "dot_area_max": float(np.max(areas_arr)),
    }


def _calc_hist(
    image: np.ndarray,
    channel: int,
    bins: int,
    value_range: tuple,
    mask: Optional[np.ndarray],
) -> list:
    hist = cv2.calcHist([image], [channel], mask, [bins], value_range).astype(np.float32)
    total = float(np.sum(hist))
    if total > 0:
        hist /= total
    return hist.flatten().tolist()


def _circle_mask(
    shape: tuple,
    center_x: float,
    center_y: float,
    radius: float,
) -> np.ndarray:
    """Build a binary circle mask."""
    h, w = shape
    yy, xx = np.indices((h, w))
    rr = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    mask = (rr <= radius).astype(np.uint8) * 255
    return mask
