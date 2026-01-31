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
            best_circle, confidence_info = _select_best_circle_with_confidence(circles, w, h, gray_blur)
            if best_circle is not None:
                x, y, r = best_circle
                stage_name = stage_names[min(stage_idx, len(stage_names) - 1)]
                return LensGeometry(
                    cx=float(x),
                    cy=float(y),
                    r=float(r),
                    confidence=1.0,
                    source=stage_name,
                    # P2-1: Separate confidence metrics
                    center_confidence=confidence_info["center_confidence"],
                    radius_confidence=confidence_info["radius_confidence"],
                    center_offset_ratio=confidence_info["center_offset_ratio"],
                )

    # Fallback: explicit signal for debugging
    return LensGeometry(
        cx=w / 2.0,
        cy=h / 2.0,
        r=img_size * 0.42,
        confidence=0.0,
        source="fallback",
        center_confidence=0.0,
        radius_confidence=0.0,
        center_offset_ratio=0.0,
    )


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


def _select_best_circle_with_confidence(circles: np.ndarray, img_w: int, img_h: int, gray_blur: np.ndarray) -> tuple:
    """
    P2-1: Select best circle with detailed confidence metrics.

    Computes separate confidence scores for center position and radius
    to enable better debugging when alpha computation fails.

    Args:
        circles: (N, 3) array of (x, y, r) from HoughCircles
        img_w, img_h: Image dimensions
        gray_blur: Blurred grayscale image for edge strength analysis

    Returns:
        Tuple of (best_circle, confidence_info) where confidence_info contains:
        - center_confidence: 0-1 score based on center offset
        - radius_confidence: 0-1 score based on edge strength on circle boundary
        - center_offset_ratio: Normalized distance from image center
    """
    from typing import Dict

    circles = np.squeeze(circles, axis=0)
    if circles.ndim == 1:  # Single circle
        circles = circles[np.newaxis, :]

    img_cx, img_cy = img_w / 2.0, img_h / 2.0
    img_size = min(img_w, img_h)

    best_circle = None
    best_score = -float("inf")
    best_confidence_info: Dict[str, float] = {
        "center_confidence": 0.0,
        "radius_confidence": 0.0,
        "center_offset_ratio": 1.0,
    }

    for x, y, r in circles:
        # Center offset analysis
        center_dist = np.hypot(x - img_cx, y - img_cy)
        center_offset_ratio = center_dist / img_size

        # Center confidence: 1.0 if perfectly centered, decreases with offset
        # Allow 5% offset with full confidence, then linear decrease
        center_confidence = max(0.0, min(1.0, 1.0 - (center_offset_ratio - 0.05) * 5))

        # Radius confidence: based on edge strength along the detected circle
        radius_confidence = _compute_radius_confidence(gray_blur, x, y, r)

        # Combined score for selection (same as before, but we keep confidence separate)
        r_normalized = r / (img_size * 0.5)
        center_penalty = center_offset_ratio
        score = r_normalized - center_penalty * 0.5

        if score > best_score:
            best_score = score
            best_circle = np.array([x, y, r])
            best_confidence_info = {
                "center_confidence": round(center_confidence, 3),
                "radius_confidence": round(radius_confidence, 3),
                "center_offset_ratio": round(center_offset_ratio, 4),
            }

    # Reject if too off-center
    if best_score < 0:
        return None, best_confidence_info

    return best_circle, best_confidence_info


def _compute_radius_confidence(gray_blur: np.ndarray, cx: float, cy: float, r: float) -> float:
    """
    P2-1: Compute radius confidence based on edge strength along circle boundary.

    Samples points along the detected circle boundary and measures
    gradient magnitude. Strong, consistent edges = high confidence.

    Args:
        gray_blur: Blurred grayscale image
        cx, cy: Circle center
        r: Circle radius

    Returns:
        Confidence score 0-1
    """
    h, w = gray_blur.shape[:2]

    # Compute gradient magnitude
    grad_x = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Sample points along circle boundary
    n_samples = 72  # Every 5 degrees
    edge_strengths = []

    for i in range(n_samples):
        theta = 2 * np.pi * i / n_samples
        px = int(cx + r * np.cos(theta))
        py = int(cy + r * np.sin(theta))

        # Check bounds
        if 0 <= px < w and 0 <= py < h:
            edge_strengths.append(grad_mag[py, px])

    if len(edge_strengths) < n_samples * 0.5:
        # Too many points outside image
        return 0.3

    edge_strengths = np.array(edge_strengths)

    # Confidence based on:
    # 1. Mean edge strength (higher is better)
    # 2. Consistency (lower std relative to mean is better)
    mean_strength = np.mean(edge_strengths)
    std_strength = np.std(edge_strengths)

    # Normalize mean strength (typical range 10-100 for good edges)
    strength_score = min(1.0, mean_strength / 50.0)

    # Consistency score (coefficient of variation)
    if mean_strength > 1.0:
        cv = std_strength / mean_strength
        consistency_score = max(0.0, 1.0 - cv)  # Lower CV = higher score
    else:
        consistency_score = 0.0

    # Combined confidence
    confidence = 0.6 * strength_score + 0.4 * consistency_score

    return round(max(0.0, min(1.0, confidence)), 3)
