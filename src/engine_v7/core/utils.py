from __future__ import annotations

import math
import warnings
from typing import NewType, Tuple, Union

import cv2
import numpy as np

# ==============================================================================
# Phase 1: Lab Color Type System (V6 Adaptive Engine)
# ==============================================================================

# CIE Lab scale: L*: 0-100, a*: -128 to +127, b*: -128 to +127
# This NewType provides compile-time documentation and IDE hints
CIELabArray = NewType("CIELabArray", np.ndarray)

# OpenCV 8-bit Lab scale: L: 0-255, a: 0-255, b: 0-255
CV8LabArray = NewType("CV8LabArray", np.ndarray)


def to_cie_lab(
    img: np.ndarray,
    *,
    source: str = "auto",
    validate: bool = True,
) -> CIELabArray:
    """
    Unified BGR to CIE L*a*b* conversion with validation.

    This is the single entry point for Lab conversion in the engine.
    Use this function instead of cv2.cvtColor directly.

    Args:
        img: Input image
            - BGR uint8 (H, W, 3): Standard OpenCV BGR image
            - CV8 Lab float32: OpenCV Lab scale (L:0-255, a:0-255, b:0-255)
            - CIE Lab float32: Already CIE scale (L:0-100, a:-128~127, b:-128~127)
        source: Input format hint
            - "auto": Auto-detect from data range (default)
            - "bgr": Force treat as BGR
            - "cv8_lab": Force treat as OpenCV Lab
            - "cie_lab": Force treat as CIE Lab (passthrough)
        validate: If True and __debug__, check output range

    Returns:
        CIELabArray: CIE L*a*b* with proper scale
            L*: 0-100 (lightness)
            a*: -128 to +127 (green-red)
            b*: -128 to +127 (blue-yellow)

    Example:
        >>> lab = to_cie_lab(bgr_image)  # From BGR
        >>> lab = to_cie_lab(cv8_lab, source="cv8_lab")  # From CV8 Lab
    """
    img = np.asarray(img)

    if img.size == 0:
        return CIELabArray(np.zeros(img.shape, dtype=np.float32))

    # Determine source format
    if source == "cie_lab":
        result = img.astype(np.float32)
    elif source == "bgr":
        cv8 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        result = _cv8_to_cie(cv8)
    elif source == "cv8_lab":
        result = _cv8_to_cie(img.astype(np.float32))
    else:  # auto
        result = _auto_convert_to_cie(img)

    # Validation in debug mode
    if validate and __debug__:
        _validate_cie_lab(result)

    return CIELabArray(result)


def _cv8_to_cie(cv8_lab: np.ndarray) -> np.ndarray:
    """Internal: CV8 Lab to CIE Lab conversion."""
    out = cv8_lab.astype(np.float32)
    out[..., 0] = cv8_lab[..., 0] * (100.0 / 255.0)  # L: 0-255 ??0-100
    out[..., 1] = cv8_lab[..., 1] - 128.0  # a: 0-255 ??-128~+127
    out[..., 2] = cv8_lab[..., 2] - 128.0  # b: 0-255 ??-128~+127
    return out


def _auto_convert_to_cie(img: np.ndarray) -> np.ndarray:
    """Internal: Auto-detect format and convert to CIE Lab."""
    # Case 1: BGR uint8 image
    if img.dtype == np.uint8 and img.ndim >= 2 and img.shape[-1] == 3:
        cv8 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        return _cv8_to_cie(cv8)

    # Case 2: Already float, need to detect Lab scale
    img = img.astype(np.float32)

    # Heuristic: Check a/b channel range
    if img.ndim >= 2 and img.shape[-1] == 3:
        ab = img[..., 1:3]
        ab_min, ab_max = float(np.nanmin(ab)), float(np.nanmax(ab))
        ab_mean = float(np.nanmean(ab))

        # CV8 Lab: a/b in 0-255, centered around 128
        if ab_min >= -5.0 and ab_max <= 260.0 and ab_mean > 40.0:
            return _cv8_to_cie(img)

        # Partial CV8 (only L needs conversion): L > 100 but a/b already centered
        L = img[..., 0]
        if float(np.nanmax(L)) > 110.0:
            out = img.copy()
            out[..., 0] = L * (100.0 / 255.0)
            return out

    # Assume already CIE Lab
    return img


def _validate_cie_lab(lab: np.ndarray) -> None:
    """Internal: Validate CIE Lab range (debug mode only)."""
    if lab.size == 0:
        return

    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    # Allow some tolerance for numerical precision
    L_max = float(np.nanmax(L))
    a_min, a_max = float(np.nanmin(a)), float(np.nanmax(a))
    b_min, b_max = float(np.nanmin(b)), float(np.nanmax(b))

    issues = []
    if L_max > 105.0:
        issues.append(f"L* max={L_max:.1f} (expected <=100)")
    if a_max > 135.0 or a_min < -135.0:
        issues.append(f"a* range=[{a_min:.1f}, {a_max:.1f}] (expected -128~127)")
    if b_max > 135.0 or b_min < -135.0:
        issues.append(f"b* range=[{b_min:.1f}, {b_max:.1f}] (expected -128~127)")

    if issues:
        warnings.warn(
            f"[CIELabArray] Possible scale mismatch: {'; '.join(issues)}",
            stacklevel=3,
        )


# ==============================================================================
# Legacy Functions (Deprecated - use to_cie_lab instead)
# Legacy Functions (Deprecated - use to_cie_lab instead)
# ==============================================================================


def bgr_to_lab(bgr: np.ndarray) -> CV8LabArray:
    """
    Convert BGR to OpenCV 8-bit Lab (L:0-255, a:0-255, b:0-255).

    .. deprecated:: 7.5
        Returns CV8 Lab which is NOT CIE scale.
        Use `to_cie_lab(bgr)` for CIE L*a*b* output.
        This function will be removed in version 8.0.
    """
    warnings.warn(
        "bgr_to_lab() returns CV8 Lab (0-255 scale), not CIE Lab. "
        "Use to_cie_lab(bgr) for CIE L*a*b* instead. "
        "This function will be removed in v8.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return CV8LabArray(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32))


def lab_cv8_to_cie(lab_cv8: np.ndarray) -> CIELabArray:
    """
    Convert OpenCV 8-bit Lab to CIE L*a*b*.

    OpenCV: L:0-255, a:0-255, b:0-255
    CIE:    L:0-100, a:-128~+127, b:-128~+127

    Note:
        For new code, prefer `to_cie_lab(lab, source="cv8_lab")`.
    """
    lab_cie = lab_cv8.astype(np.float32)
    lab_cie[..., 0] = lab_cv8[..., 0] * (100.0 / 255.0)  # L: 0-255 ??0-100
    lab_cie[..., 1] = lab_cv8[..., 1] - 128.0  # a: 0-255 ??-128~+127
    lab_cie[..., 2] = lab_cv8[..., 2] - 128.0  # b: 0-255 ??-128~+127
    return CIELabArray(lab_cie)


def lab_cie_to_cv8(lab_cie: np.ndarray) -> CV8LabArray:
    """
    Convert CIE L*a*b* to OpenCV 8-bit Lab.

    CIE:    L:0-100, a:-128~+127, b:-128~+127
    OpenCV: L:0-255, a:0-255, b:0-255
    """
    lab_cv8 = lab_cie.astype(np.float32)
    lab_cv8[..., 0] = lab_cie[..., 0] * (255.0 / 100.0)  # L: 0-100 ??0-255
    lab_cv8[..., 1] = lab_cie[..., 1] + 128.0  # a: -128~+127 ??0-255
    lab_cv8[..., 2] = lab_cie[..., 2] + 128.0  # b: -128~+127 ??0-255
    return CV8LabArray(lab_cv8)


def cie76_deltaE(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    return np.linalg.norm(lab1 - lab2, axis=-1)


def cie2000_deltaE(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 color difference (kL=kC=kH=1)."""
    lab1 = np.asarray(lab1, dtype=np.float32)
    lab2 = np.asarray(lab2, dtype=np.float32)

    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.sqrt(a1 * a1 + b1 * b1)
    C2 = np.sqrt(a2 * a2 + b2 * b2)
    C_bar = (C1 + C2) / 2.0
    C_bar7 = C_bar**7
    G = 0.5 * (1.0 - np.sqrt(C_bar7 / (C_bar7 + 25.0**7 + 1e-12)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.sqrt(a1p * a1p + b1 * b1)
    C2p = np.sqrt(a2p * a2p + b2 * b2)

    h1p = (np.degrees(np.arctan2(b1, a1p)) + 360.0) % 360.0
    h2p = (np.degrees(np.arctan2(b2, a2p)) + 360.0) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dhp = np.where((C1p * C2p) == 0, 0.0, dhp)

    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    L_bar = (L1 + L2) / 2.0
    C_bar_p = (C1p + C2p) / 2.0

    h_sum = h1p + h2p
    h_diff = np.abs(h1p - h2p)
    h_bar = (h1p + h2p) / 2.0
    h_bar = np.where((C1p * C2p) == 0, h_sum, h_bar)
    h_bar = np.where(((C1p * C2p) != 0) & (h_diff > 180.0) & (h_sum < 360.0), (h_sum + 360.0) / 2.0, h_bar)
    h_bar = np.where(((C1p * C2p) != 0) & (h_diff > 180.0) & (h_sum >= 360.0), (h_sum - 360.0) / 2.0, h_bar)

    T = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar))
        + 0.32 * np.cos(np.radians(3.0 * h_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar - 63.0))
    )

    delta_theta = 30.0 * np.exp(-(((h_bar - 275.0) / 25.0) ** 2))
    Rc = 2.0 * np.sqrt((C_bar_p**7) / (C_bar_p**7 + 25.0**7 + 1e-12))

    Sl = 1.0 + (0.015 * (L_bar - 50.0) ** 2) / np.sqrt(20.0 + (L_bar - 50.0) ** 2)
    Sc = 1.0 + 0.045 * C_bar_p
    Sh = 1.0 + 0.015 * C_bar_p * T

    Rt = -np.sin(np.radians(2.0 * delta_theta)) * Rc

    dE = np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))
    return dE


def compute_white_reference(bgr: np.ndarray, geom, *, optical_zone_ratio: float, p_low: float, p_high: float):
    h, w = bgr.shape[:2]
    cx = int(round(geom.cx))
    cy = int(round(geom.cy))
    r = int(round(geom.r * float(optical_zone_ratio)))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)

    # Use CIE Lab for L* channel (0-100 scale)
    lab = to_cie_lab(bgr, validate=False)
    L = lab[..., 0]
    pixels_L = L[mask == 255]
    if pixels_L.size == 0:
        return None, {"reason": "no_pixels", "n_pixels": 0}

    low = float(np.percentile(pixels_L, p_low))
    high = float(np.percentile(pixels_L, p_high))

    sel = (mask == 255) & (L >= low) & (L <= high)
    if not np.any(sel):
        sel = mask == 255

    pixels_bgr = bgr[sel]
    if pixels_bgr.size == 0:
        return None, {"reason": "no_pixels", "n_pixels": int(pixels_L.size)}

    mean_bgr = pixels_bgr.mean(axis=0)
    meta = {
        "optical_zone_ratio": float(optical_zone_ratio),
        "p_low": float(p_low),
        "p_high": float(p_high),
        "n_pixels": int(pixels_L.size),
        "n_pixels_used": int(pixels_bgr.shape[0]),
        "l_low": low,
        "l_high": high,
    }
    return mean_bgr, meta


def apply_white_balance(bgr: np.ndarray, geom, wb_cfg: dict):
    optical_zone_ratio = float(wb_cfg.get("optical_zone_ratio", 0.30))
    p_low = float(wb_cfg.get("p_low", 50.0))
    p_high = float(wb_cfg.get("p_high", 99.0))

    mean_bgr, meta = compute_white_reference(
        bgr,
        geom,
        optical_zone_ratio=optical_zone_ratio,
        p_low=p_low,
        p_high=p_high,
    )
    if mean_bgr is None:
        meta = meta or {}
        meta["applied"] = False
        return bgr, meta

    target = float(np.mean(mean_bgr))
    gains = target / (mean_bgr + 1e-6)

    balanced = bgr.astype(np.float32) * gains.reshape(1, 1, 3)
    balanced = np.clip(balanced, 0.0, 255.0).astype(np.uint8)

    meta = meta or {}
    meta.update(
        {
            "applied": True,
            "target": target,
            "mean_bgr": [float(x) for x in mean_bgr],
            "gains": [float(x) for x in gains],
        }
    )
    return balanced, meta


def laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    if a.size != b.size or a.size < 10:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def delta_e_cie2000(
    lab1: Tuple[float, float, float] | np.ndarray,
    lab2: Tuple[float, float, float] | np.ndarray,
    kL: float = 1.0,
    kC: float = 1.0,
    kH: float = 1.0,
) -> float:
    """
    Calculate CIEDE2000 color difference (Scalar version).
    Ported from src.utils.color_delta for v7 engine.

    Args:
        lab1: First color (L, a, b)
        lab2: Second color (L, a, b)
        kL: Lightness weight (default 1.0)
        kC: Chroma weight (default 1.0)
        kH: Hue weight (default 1.0)

    Returns:
        float: Delta E 2000 value
    """
    # Extract LAB values
    if isinstance(lab1, (tuple, list)):
        L1, a1, b1 = lab1
    else:
        L1, a1, b1 = lab1[0], lab1[1], lab1[2]

    if isinstance(lab2, (tuple, list)):
        L2, a2, b2 = lab2
    else:
        L2, a2, b2 = lab2[0], lab2[1], lab2[2]

    # 1. Calculate C1, C2 (Chroma)
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)

    # 2. Calculate C_bar
    C_bar = (C1 + C2) / 2.0

    # 3. Calculate G
    G = 0.5 * (1 - np.sqrt(C_bar**7 / (C_bar**7 + 25**7)))

    # 4. Calculate a'1, a'2
    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2

    # 5. Calculate C'1, C'2
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)

    # 6. Calculate h'1, h'2 (hue angle in degrees)
    h1_prime = np.degrees(np.arctan2(b1, a1_prime))
    if h1_prime < 0:
        h1_prime += 360

    h2_prime = np.degrees(np.arctan2(b2, a2_prime))
    if h2_prime < 0:
        h2_prime += 360

    # 7. Calculate ?L', ?C', ?H'
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    # Calculate ?h'
    if C1_prime * C2_prime == 0:
        delta_h_prime = 0
    else:
        diff = h2_prime - h1_prime
        if abs(diff) <= 180:
            delta_h_prime = diff
        elif diff > 180:
            delta_h_prime = diff - 360
        else:
            delta_h_prime = diff + 360

    # Calculate ?H'
    delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime / 2.0))

    # 8. Calculate L_bar', C_bar', H_bar'
    L_bar_prime = (L1 + L2) / 2.0
    C_bar_prime = (C1_prime + C2_prime) / 2.0

    # Calculate H_bar'
    if C1_prime * C2_prime == 0:
        H_bar_prime = h1_prime + h2_prime
    else:
        sum_h = h1_prime + h2_prime
        abs_diff = abs(h1_prime - h2_prime)
        if abs_diff <= 180:
            H_bar_prime = sum_h / 2.0
        elif sum_h < 360:
            H_bar_prime = (sum_h + 360) / 2.0
        else:
            H_bar_prime = (sum_h - 360) / 2.0

    # 9. Calculate T
    T = (
        1.0
        - 0.17 * np.cos(np.radians(H_bar_prime - 30))
        + 0.24 * np.cos(np.radians(2 * H_bar_prime))
        + 0.32 * np.cos(np.radians(3 * H_bar_prime + 6))
        - 0.20 * np.cos(np.radians(4 * H_bar_prime - 63))
    )

    # 10. Calculate SL, SC, SH
    SL = 1 + ((0.015 * (L_bar_prime - 50) ** 2) / np.sqrt(20 + (L_bar_prime - 50) ** 2))
    SC = 1 + 0.045 * C_bar_prime
    SH = 1 + 0.015 * C_bar_prime * T

    # 11. Calculate RT (rotation term)
    delta_theta = 30 * np.exp(-(((H_bar_prime - 275) / 25) ** 2))
    RC = 2 * np.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
    RT = -np.sin(np.radians(2 * delta_theta)) * RC

    # 12. Calculate ?E2000
    delta_E = np.sqrt(
        (delta_L_prime / (kL * SL)) ** 2
        + (delta_C_prime / (kC * SC)) ** 2
        + (delta_H_prime / (kH * SH)) ** 2
        + RT * (delta_C_prime / (kC * SC)) * (delta_H_prime / (kH * SH))
    )

    return float(delta_E)


def sanitize_filename(filename: str) -> str:
    """
    Remove path traversal and unsafe characters from filename.

    Args:
        filename: Original filename (potentially unsafe)

    Returns:
        Safe filename with directory components removed and unsafe chars replaced

    Examples:
        >>> sanitize_filename('../../../etc/passwd')
        'etc_passwd'
        >>> sanitize_filename('test<>file.jpg')
        'test__file.jpg'
    """
    import re
    from pathlib import Path

    # Extract basename only (removes directory traversal)
    safe_name = Path(filename).name

    # Remove unsafe characters (keep alphanumericDOIT, period, dash, underscore)
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", safe_name)

    # Return 'unnamed_file' if result is empty
    return safe_name if safe_name else "unnamed_file"
