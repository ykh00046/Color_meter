from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

from ..types import LensGeometry


def detect_lens_circle(bgr: np.ndarray, cfg: Optional[Dict[str, Any]] = None) -> LensGeometry:
    """
    Detect outer lens circle using HoughCircles with multi-stage parameter sweep.

    Improvements for operational stability:
    1. Multi-stage Hough parameter sweep (strict → loose) for varying lighting/exposure
    2. Center penalty in candidate selection to avoid false positives (frame/light rings)
    3. Explicit fallback signal (source="fallback", confidence=0.0) for debugging
    4. Resolution-adaptive blur kernel

    Args:
        bgr: Input BGR image
        cfg: Optional config with custom parameters:
             - min_radius_ratio: default 0.30
             - max_radius_ratio: default 0.49
             - hough_param1: default 120 (Canny high threshold)
             - hough_param2_stages: default [35, 28, 22] (accumulator thresholds)

    Returns:
        LensGeometry with confidence and source fields:
        - confidence: 1.0 (hough success) or 0.0 (fallback)
        - source: "hough_strict" | "hough_medium" | "hough_loose" | "fallback"
    """
    cfg = cfg or {}
    h, w = bgr.shape[:2]
    img_size = min(h, w)

    # Extract config parameters
    min_radius_ratio = cfg.get("min_radius_ratio", 0.30)
    max_radius_ratio = cfg.get("max_radius_ratio", 0.49)
    hough_param1 = cfg.get("hough_param1", 120)
    hough_param2_stages = cfg.get("hough_param2_stages", [35, 28, 22])

    # Resolution-adaptive blur (scale with image size)
    kernel_size = max(5, min(15, (img_size // 100) * 2 + 1))  # 5~15, odd
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 2)

    # Multi-stage Hough parameter sweep: strict → loose
    stage_names = ["hough_strict", "hough_medium", "hough_loose"]
    for stage_idx, param2 in enumerate(hough_param2_stages):
        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=img_size // 4,
            param1=hough_param1,
            param2=param2,
            minRadius=int(img_size * min_radius_ratio),
            maxRadius=int(img_size * max_radius_ratio),
        )

        if circles is not None:
            # Select best candidate with center penalty
            best_circle = _select_best_circle(circles, w, h)
            if best_circle is not None:
                x, y, r = best_circle
                stage_name = stage_names[min(stage_idx, len(stage_names) - 1)]
                return LensGeometry(cx=float(x), cy=float(y), r=float(r), confidence=1.0, source=stage_name)

    # Fallback: explicit signal for debugging
    return LensGeometry(cx=w / 2.0, cy=h / 2.0, r=img_size * 0.42, confidence=0.0, source="fallback")


def _select_best_circle(circles: np.ndarray, img_w: int, img_h: int) -> Optional[np.ndarray]:
    """
    Select best circle candidate using radius + center penalty.

    Prevents selecting false positives (frame/light rings) by penalizing
    circles far from image center.

    Args:
        circles: (N, 3) array of (x, y, r) from HoughCircles
        img_w, img_h: Image dimensions

    Returns:
        Best (x, y, r) or None if all candidates rejected
    """
    circles = np.squeeze(circles, axis=0)
    if circles.ndim == 1:  # Single circle
        circles = circles[np.newaxis, :]

    img_cx, img_cy = img_w / 2.0, img_h / 2.0

    scores = []
    for x, y, r in circles:
        # Center penalty: distance from image center
        center_dist = np.hypot(x - img_cx, y - img_cy)
        center_penalty = center_dist / min(img_w, img_h)  # Normalized [0~0.7 typically]

        # Score: prioritize large radius but penalize off-center
        # Normalize radius to [0~1] range
        r_normalized = r / (min(img_w, img_h) * 0.5)
        score = r_normalized - center_penalty * 0.5  # Weight: 0.5 for center penalty

        scores.append(score)

    # Select highest score
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]

    # Reject if too off-center (score < 0 means penalty dominated)
    if best_score < 0:
        return None

    return circles[best_idx]
