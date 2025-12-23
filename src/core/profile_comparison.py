"""
Radial Profile Comparison Module (P1-2)

Implements structural similarity and correlation analysis between two radial profiles.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


def compare_radial_profiles(test_profile: Dict[str, Any], std_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare radial profiles between test and STD using correlation and structural similarity.

    Args:
        test_profile: Test sample radial profile dict (L, a, b lists)
        std_profile: STD radial profile dict (L, a, b lists)

    Returns:
        Profile comparison details including:
        - correlation: Pearson correlation for L, a, b
        - structural_similarity: SSIM for L, a, b
        - gradient_similarity: Gradient correlation for L, a, b
        - profile_score: Overall similarity score (0-100)
    """
    # Extract profile data
    test_L = np.array(test_profile.get("L", []))
    test_a = np.array(test_profile.get("a", []))
    test_b = np.array(test_profile.get("b", []))

    std_L = np.array(std_profile.get("L", []))
    std_a = np.array(std_profile.get("a", []))
    std_b = np.array(std_profile.get("b", []))

    if len(test_L) == 0 or len(std_L) == 0:
        return {
            "profile_score": 0.0,
            "message": "Empty profile data",
            "correlation": {"avg": 0.0},
            "structural_similarity": {"avg": 0.0},
            "gradient_similarity": {"avg": 0.0},
        }

    test_length = len(test_L)
    std_length = len(std_L)
    length_match = test_length == std_length

    # Handle length mismatch with interpolation
    if not length_match:
        # Interpolate shorter profile to match longer one
        if test_length < std_length:
            x_old = np.linspace(0, 1, test_length)
            x_new = np.linspace(0, 1, std_length)
            test_L = np.interp(x_new, x_old, test_L)
            test_a = np.interp(x_new, x_old, test_a)
            test_b = np.interp(x_new, x_old, test_b)
        else:
            x_old = np.linspace(0, 1, std_length)
            x_new = np.linspace(0, 1, test_length)
            std_L = np.interp(x_new, x_old, std_L)
            std_a = np.interp(x_new, x_old, std_a)
            std_b = np.interp(x_new, x_old, std_b)

    # 1. Pearson Correlation Coefficient
    corr_L = _safe_pearsonr(test_L, std_L)
    corr_a = _safe_pearsonr(test_a, std_a)
    corr_b = _safe_pearsonr(test_b, std_b)
    corr_avg = (corr_L + corr_a + corr_b) / 3.0

    # 2. Structural Similarity (1D SSIM approximation)
    ssim_L = _safe_ssim_1d(test_L, std_L)
    ssim_a = _safe_ssim_1d(test_a, std_a)
    ssim_b = _safe_ssim_1d(test_b, std_b)
    ssim_avg = (ssim_L + ssim_a + ssim_b) / 3.0

    # 3. Gradient Similarity
    grad_test_L = np.gradient(test_L)
    grad_test_a = np.gradient(test_a)
    grad_test_b = np.gradient(test_b)

    grad_std_L = np.gradient(std_L)
    grad_std_a = np.gradient(std_a)
    grad_std_b = np.gradient(std_b)

    grad_corr_L = _safe_pearsonr(grad_test_L, grad_std_L)
    grad_corr_a = _safe_pearsonr(grad_test_a, grad_std_a)
    grad_corr_b = _safe_pearsonr(grad_test_b, grad_std_b)
    grad_corr_avg = (grad_corr_L + grad_corr_a + grad_corr_b) / 3.0

    # 4. Calculate profile_score
    # Convert correlation/SSIM from [-1, 1] to [0, 100]
    corr_score = (corr_avg + 1) * 50.0  # -1→0, 0→50, 1→100
    ssim_score = (ssim_avg + 1) * 50.0
    grad_score = (grad_corr_avg + 1) * 50.0

    # Weighted average: correlation 50%, SSIM 30%, gradient 20%
    profile_score = corr_score * 0.5 + ssim_score * 0.3 + grad_score * 0.2
    profile_score = max(0.0, min(100.0, profile_score))

    # Generate message
    if corr_avg >= 0.9:
        message = f"Excellent correlation (r={corr_avg:.3f})"
    elif corr_avg >= 0.7:
        message = f"Good correlation (r={corr_avg:.3f})"
    elif corr_avg >= 0.5:
        message = f"Moderate correlation (r={corr_avg:.3f})"
    else:
        message = f"Low correlation (r={corr_avg:.3f})"

    if not length_match:
        message += f" [Length adjusted: {test_length}→{max(test_length, std_length)}]"

    return {
        "correlation": {
            "L": float(corr_L),
            "a": float(corr_a),
            "b": float(corr_b),
            "avg": float(corr_avg),
        },
        "structural_similarity": {
            "L": float(ssim_L),
            "a": float(ssim_a),
            "b": float(ssim_b),
            "avg": float(ssim_avg),
        },
        "gradient_similarity": {
            "L": float(grad_corr_L),
            "a": float(grad_corr_a),
            "b": float(grad_corr_b),
            "avg": float(grad_corr_avg),
        },
        "profile_score": float(profile_score),
        "length_match": length_match,
        "test_length": int(test_length),
        "std_length": int(std_length),
        "message": message,
    }


def _safe_pearsonr(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Calculate Pearson correlation with error handling"""
    if len(arr1) < 2 or len(arr2) < 2:
        return 0.0
    if np.std(arr1) == 0 or np.std(arr2) == 0:
        return 1.0 if np.allclose(arr1, arr2) else 0.0
    try:
        corr, _ = pearsonr(arr1, arr2)
        return float(corr) if not np.isnan(corr) else 0.0
    except Exception:
        return 0.0


def _safe_ssim_1d(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Calculate 1D SSIM-like metric"""
    if len(arr1) < 2 or len(arr2) < 2:
        return 0.0

    # Normalize to 0-1 range based on combined range
    min_val = min(arr1.min(), arr2.min())
    max_val = max(arr1.max(), arr2.max())
    range_val = max_val - min_val

    if range_val < 1e-8:
        return 1.0 if np.allclose(arr1, arr2) else 0.0

    arr1_norm = (arr1 - min_val) / range_val
    arr2_norm = (arr2 - min_val) / range_val

    # SSIM components: luminance, contrast, structure
    mu1 = np.mean(arr1_norm)
    mu2 = np.mean(arr2_norm)
    sigma1 = np.std(arr1_norm)
    sigma2 = np.std(arr2_norm)
    sigma12 = np.mean((arr1_norm - mu1) * (arr2_norm - mu2))

    C1 = 0.01
    C2 = 0.03

    luminance = (2 * mu1 * mu2 + C1) / (mu1**2 + mu2**2 + C1)
    contrast = (2 * sigma1 * sigma2 + C2) / (sigma1**2 + sigma2**2 + C2)

    denom = sigma1 * sigma2 + C2 / 2
    structure = (sigma12 + C2 / 2) / denom if abs(denom) > 1e-10 else 1.0

    ssim = luminance * contrast * structure
    return float(np.clip(ssim, -1.0, 1.0))
