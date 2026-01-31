"""
Profile Analysis Module

Advanced signal processing for radial profiles.
Includes smoothing, derivative calculation, and boundary detection.
Ported from src/analysis/profile_analyzer.py (converted to functions).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks, savgol_filter

logger = logging.getLogger(__name__)


def smooth_profile(data: np.ndarray, window_length: int = 5, polyorder: int = 2) -> np.ndarray:
    """
    Apply Savitzky-Golay filter for smoothing.

    Args:
        data: 1D numpy array (profile data)
        window_length: Filter window length (must be odd).
        polyorder: Polynomial order (typically 2 or 3).

    Returns:
        Smoothed data array
    """
    if len(data) < window_length:
        # logger.warning(
        #     f"Data length ({len(data)}) is smaller than window_length ({window_length}). Skipping smoothing."
        # )
        return data

    if window_length % 2 == 0:
        window_length += 1

    try:
        return savgol_filter(data, window_length, polyorder)
    except Exception as e:
        logger.error(f"Error during smoothing: {e}")
        return data


def compute_gradient(data: np.ndarray, x_coords: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute 1st derivative (Gradient).

    Args:
        data: 1D numpy array (y values)
        x_coords: 1D numpy array (x values, e.g., radius). If None, spacing is 1.

    Returns:
        Gradient array (dy/dx)
    """
    if x_coords is None:
        return np.gradient(data)
    return np.gradient(data, x_coords)


def compute_second_derivative(data: np.ndarray, x_coords: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute 2nd derivative (Gradient of Gradient).

    Args:
        data: 1D numpy array (y values, smoothed recommended)
        x_coords: 1D numpy array (x values)

    Returns:
        Second derivative array (d^2y/dx^2)
    """
    first_derivative = compute_gradient(data, x_coords)
    return compute_gradient(first_derivative, x_coords)


def detect_inflection_points(
    second_derivative: np.ndarray, x_coords: np.ndarray, threshold: float = 1e-4
) -> List[Dict]:
    """
    Detect Zero-crossing points of 2nd derivative (Inflection Points).

    Args:
        second_derivative: 2nd derivative data
        x_coords: x-axis coordinates (radius)
        threshold: Threshold to consider as zero

    Returns:
        List of dictionaries containing inflection point info
    """
    inflections = []

    # Detect sign change
    signs = np.sign(second_derivative)
    sign_changes = ((np.roll(signs, 1) - signs) != 0).astype(int)
    sign_changes[0] = 0

    indices = np.where(sign_changes == 1)[0]

    for idx in indices:
        prev_val = second_derivative[idx - 1]
        curr_val = second_derivative[idx]

        if abs(prev_val - curr_val) > threshold:
            inflections.append(
                {
                    "method": "inflection_point",
                    "index": int(idx),
                    "radius": float(x_coords[idx]),
                    "value": float(curr_val),
                    "confidence": 0.5,
                }
            )

    return inflections


def detect_peaks(
    data: np.ndarray,
    x_coords: np.ndarray,
    height: Optional[float] = None,
    prominence: Optional[float] = None,
    distance: Optional[int] = None,
) -> List[Dict]:
    """
    Detect peaks in data (usually gradient magnitude).

    Args:
        data: Data to analyze
        x_coords: x-axis coordinates
        height: Minimum peak height
        prominence: Peak prominence
        distance: Minimum distance between peaks

    Returns:
        List of peak info dictionaries
    """
    peaks, properties = find_peaks(data, height=height, prominence=prominence, distance=distance)

    results = []
    for i, peak_idx in enumerate(peaks):
        peak_info = {
            "method": "peak_gradient",
            "index": int(peak_idx),
            "radius": float(x_coords[peak_idx]),
            "value": float(data[peak_idx]),
            "properties": {k: v[i] for k, v in properties.items()} if properties else {},
            "confidence": 0.8,
        }
        results.append(peak_info)

    return results


def detect_print_boundaries(
    r_norm: np.ndarray,
    a_data: np.ndarray,
    b_data: np.ndarray,
    method: str = "chroma",
    chroma_threshold: float = 2.0,
) -> Tuple[float, float, float]:
    """
    Automatically detect r_inner and r_outer of the print area.

    Args:
        r_norm: Normalized radius array (0~1)
        a_data: a* channel data
        b_data: b* channel data
        method: 'chroma', 'gradient', or 'hybrid'
        chroma_threshold: Background noise threshold (default 2.0)

    Returns:
        (r_inner, r_outer, confidence)
    """
    # 1. Calculate Chroma
    chroma = np.sqrt(a_data**2 + b_data**2)

    # 2. Estimate background noise level (10th percentile)
    noise_level = np.percentile(chroma, 10)

    # 3. Detect colored region
    threshold = noise_level + chroma_threshold
    colored_mask = chroma > threshold

    if not np.any(colored_mask):
        # logger.warning("No colored area detected, using full range.")
        return (0.0, 1.0, 0.0)

    # 4. Find first/last colored indices
    colored_indices = np.where(colored_mask)[0]
    inner_idx = colored_indices[0]
    outer_idx = colored_indices[-1]

    r_inner = float(r_norm[inner_idx])
    r_outer = float(r_norm[outer_idx])

    # 5. Gradient-based refinement
    if method in ("gradient", "hybrid"):
        chroma_grad = np.abs(compute_gradient(chroma, r_norm))
        grad_smooth = smooth_profile(chroma_grad, window_length=5)

        # Inner refinement
        inner_search_start = max(0, inner_idx - 10)
        inner_search_end = min(len(r_norm), inner_idx + 10)
        inner_region_grad = grad_smooth[inner_search_start:inner_search_end]

        if len(inner_region_grad) > 0:
            inner_peak_idx = np.argmax(inner_region_grad) + inner_search_start
            r_inner_refined = float(r_norm[inner_peak_idx])
            if abs(r_inner_refined - r_inner) < 0.1:
                r_inner = r_inner_refined

        # Outer refinement
        outer_search_start = max(0, outer_idx - 10)
        outer_search_end = min(len(r_norm), outer_idx + 10)
        outer_region_grad = grad_smooth[outer_search_start:outer_search_end]

        if len(outer_region_grad) > 0:
            outer_peak_idx = np.argmax(outer_region_grad) + outer_search_start
            r_outer_refined = float(r_norm[outer_peak_idx])
            if abs(r_outer_refined - r_outer) < 0.1:
                r_outer = r_outer_refined

    # 6. Safety check
    confidence = 1.0

    if r_outer - r_inner < 0.2:
        confidence = 0.3
    if r_inner > 0.5:
        confidence *= 0.7
    if r_outer < 0.7:
        confidence *= 0.8

    return (r_inner, r_outer, confidence)


def analyze_profile(
    r_norm: np.ndarray,
    l_data: np.ndarray,
    a_data: np.ndarray,
    b_data: np.ndarray,
    smoothing_window: int = 5,
    gradient_threshold: float = 0.5,
) -> Dict:
    """
    Perform comprehensive analysis on L, a, b profiles.

    Args:
        r_norm: Normalized radius array
        l_data: L* profile
        a_data: a* profile
        b_data: b* profile
        smoothing_window: Window size for smoothing
        gradient_threshold: Threshold for gradient peak detection

    Returns:
        Dictionary containing smoothed profiles, derivatives, and boundary candidates.
    """
    # 1. Smoothing
    l_smooth = smooth_profile(l_data, window_length=smoothing_window)
    a_smooth = smooth_profile(a_data, window_length=smoothing_window)
    b_smooth = smooth_profile(b_data, window_length=smoothing_window)

    # 2. Gradients
    grad_l = compute_gradient(l_smooth, r_norm)
    grad_a = compute_gradient(a_smooth, r_norm)
    grad_b = compute_gradient(b_smooth, r_norm)

    grad_magnitude = np.sqrt(grad_l**2 + grad_a**2 + grad_b**2)

    # 3. Second Derivatives
    sec_l = compute_second_derivative(l_smooth, r_norm)

    # 4. Candidates
    candidates = []

    # A. Gradient Peaks
    peaks = detect_peaks(
        grad_magnitude,
        r_norm,
        height=gradient_threshold,
        distance=int(len(r_norm) * 0.05),
    )
    candidates.extend(peaks)

    # B. Inflection Points
    inflections = detect_inflection_points(sec_l, r_norm, threshold=0.01)
    candidates.extend(inflections)

    return {
        "profile": {
            "radius": r_norm.tolist(),
            "L_raw": l_data.tolist(),
            "a_raw": a_data.tolist(),
            "b_raw": b_data.tolist(),
            "L_smoothed": l_smooth.tolist(),
            "a_smoothed": a_smooth.tolist(),
            "b_smoothed": b_smooth.tolist(),
        },
        "derivatives": {
            "gradient_L": grad_l.tolist(),
            "gradient_a": grad_a.tolist(),
            "gradient_b": grad_b.tolist(),
            "gradient_magnitude": grad_magnitude.tolist(),
            "second_derivative_L": sec_l.tolist(),
        },
        "boundary_candidates": candidates,
    }
