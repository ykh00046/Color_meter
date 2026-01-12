from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np


def bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR to OpenCV 8-bit Lab (L:0-255, a:0-255, b:0-255).

    WARNING: This is NOT CIE Lab scale!
    Use bgr_to_lab_cie() for proper CIE L*a*b* (L:0-100, a:-128~+127, b:-128~+127)
    """
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def bgr_to_lab_cie(bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR to CIE L*a*b* with proper scale.

    Returns:
        np.ndarray: Lab with CIE scale
            L*: 0-100 (lightness)
            a*: -128 to +127 (green to red)
            b*: -128 to +127 (blue to yellow)
    """
    lab_cv8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab_cv8_to_cie(lab_cv8)


def lab_cv8_to_cie(lab_cv8: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV 8-bit Lab to CIE L*a*b*.

    OpenCV: L:0-255, a:0-255, b:0-255
    CIE:    L:0-100, a:-128~+127, b:-128~+127
    """
    lab_cie = lab_cv8.copy()
    lab_cie[..., 0] = lab_cv8[..., 0] * (100.0 / 255.0)  # L: 0-255 → 0-100
    lab_cie[..., 1] = lab_cv8[..., 1] - 128.0  # a: 0-255 → -128~+127
    lab_cie[..., 2] = lab_cv8[..., 2] - 128.0  # b: 0-255 → -128~+127
    return lab_cie


def lab_cie_to_cv8(lab_cie: np.ndarray) -> np.ndarray:
    """
    Convert CIE L*a*b* to OpenCV 8-bit Lab.

    CIE:    L:0-100, a:-128~+127, b:-128~+127
    OpenCV: L:0-255, a:0-255, b:0-255
    """
    lab_cv8 = lab_cie.copy()
    lab_cv8[..., 0] = lab_cie[..., 0] * (255.0 / 100.0)  # L: 0-100 → 0-255
    lab_cv8[..., 1] = lab_cie[..., 1] + 128.0  # a: -128~+127 → 0-255
    lab_cv8[..., 2] = lab_cie[..., 2] + 128.0  # b: -128~+127 → 0-255
    return lab_cv8


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

    lab = bgr_to_lab(bgr)
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

    # 7. Calculate ΔL', ΔC', ΔH'
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    # Calculate Δh'
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

    # Calculate ΔH'
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

    # 12. Calculate ΔE2000
    delta_E = np.sqrt(
        (delta_L_prime / (kL * SL)) ** 2
        + (delta_C_prime / (kC * SC)) ** 2
        + (delta_H_prime / (kH * SH)) ** 2
        + RT * (delta_C_prime / (kC * SC)) * (delta_H_prime / (kH * SH))
    )

    return float(delta_E)
