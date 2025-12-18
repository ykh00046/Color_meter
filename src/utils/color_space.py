"""
Color Space Conversion Utilities

OpenCV Lab ↔ Standard Lab (CIE L*a*b*) conversion functions.
"""

from typing import Tuple, Union

import numpy as np


def opencv_lab_to_standard(
    L_cv: Union[float, np.ndarray], a_cv: Union[float, np.ndarray], b_cv: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert OpenCV Lab to Standard CIE L*a*b*

    OpenCV Lab ranges:
        L: 0~255 (multiply by 100/255 to get 0~100)
        a: 0~255 (subtract 128 to get -128~127)
        b: 0~255 (subtract 128 to get -128~127)

    Standard Lab ranges:
        L*: 0~100
        a*: -128~127 (approximately)
        b*: -128~127 (approximately)

    Args:
        L_cv: OpenCV L channel (0~255)
        a_cv: OpenCV a channel (0~255)
        b_cv: OpenCV b channel (0~255)

    Returns:
        (L_std, a_std, b_std): Standard Lab values

    Example:
        >>> # White color in OpenCV Lab
        >>> L_std, a_std, b_std = opencv_lab_to_standard(255, 128, 128)
        >>> print(f"L*={L_std:.1f}, a*={a_std:.1f}, b*={b_std:.1f}")
        L*=100.0, a*=0.0, b*=0.0
    """
    L_std = L_cv * (100.0 / 255.0)
    a_std = a_cv - 128.0
    b_std = b_cv - 128.0

    return L_std, a_std, b_std


def standard_lab_to_opencv(
    L_std: Union[float, np.ndarray], a_std: Union[float, np.ndarray], b_std: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert Standard CIE L*a*b* to OpenCV Lab

    Args:
        L_std: Standard L* (0~100)
        a_std: Standard a* (-128~127)
        b_std: Standard b* (-128~127)

    Returns:
        (L_cv, a_cv, b_cv): OpenCV Lab values (0~255)

    Example:
        >>> # White color in standard Lab
        >>> L_cv, a_cv, b_cv = standard_lab_to_opencv(100, 0, 0)
        >>> print(f"L={L_cv:.1f}, a={a_cv:.1f}, b={b_cv:.1f}")
        L=255.0, a=128.0, b=128.0
    """
    L_cv = L_std * (255.0 / 100.0)
    a_cv = a_std + 128.0
    b_cv = b_std + 128.0

    return L_cv, a_cv, b_cv


def validate_standard_lab(L: float, a: float, b: float, tolerance: float = 5.0) -> bool:
    """
    Validate if Lab values are in standard range

    Args:
        L: L* value
        a: a* value
        b: b* value
        tolerance: Allow slight out-of-range values (default: 5.0)

    Returns:
        True if values are valid

    Example:
        >>> validate_standard_lab(72.2, 9.3, -5.2)
        True
        >>> validate_standard_lab(182.5, 127.5, 137.5)  # OpenCV scale!
        False
    """
    if not (-tolerance <= L <= 100 + tolerance):
        return False
    if not (-128 - tolerance <= a <= 127 + tolerance):
        return False
    if not (-128 - tolerance <= b <= 127 + tolerance):
        return False
    return True


def detect_lab_scale(L: float, a: float, b: float) -> str:
    """
    Detect if Lab values are in OpenCV or Standard scale

    Args:
        L: L value
        a: a value
        b: b value

    Returns:
        "opencv" or "standard" or "unknown"

    Example:
        >>> detect_lab_scale(182.5, 127.5, 137.5)
        'opencv'
        >>> detect_lab_scale(72.2, 9.3, -5.2)
        'standard'
    """
    # Heuristic detection
    if L > 100 and 0 <= a <= 255 and 0 <= b <= 255:
        return "opencv"
    elif 0 <= L <= 100 and -128 <= a <= 127 and -128 <= b <= 127:
        return "standard"
    elif 100 < L <= 255:
        return "opencv"  # Likely OpenCV
    else:
        return "unknown"
