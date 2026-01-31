"""
Plate Module Common Helpers

Shared utility functions used by both plate_engine.py and plate_gate.py.
Extracted to avoid code duplication and ensure consistency.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ..types import LensGeometry


def resize_to_square_centered(
    bgr: np.ndarray, geom: LensGeometry, out_size: int = 512
) -> Tuple[np.ndarray, LensGeometry]:
    """
    Resize and center image to square canvas.

    Args:
        bgr: Input BGR image
        geom: Lens geometry (center and radius)
        out_size: Output square size in pixels

    Returns:
        (warped_image, new_geometry) where new_geometry is centered
    """
    target_r = out_size * 0.45
    scale = target_r / max(float(geom.r), 1e-6)

    M = np.array(
        [
            [scale, 0.0, out_size / 2.0 - float(geom.cx) * scale],
            [0.0, scale, out_size / 2.0 - float(geom.cy) * scale],
        ],
        dtype=np.float32,
    )

    warped = cv2.warpAffine(
        bgr,
        M,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return warped, LensGeometry(cx=out_size / 2.0, cy=out_size / 2.0, r=target_r)


def mean_gray_center(bgr: np.ndarray, geom: LensGeometry, r_clear: float) -> float:
    """
    Calculate median gray value in center region.

    Args:
        bgr: Input BGR image
        geom: Lens geometry
        r_clear: Normalized radius of clear region (0.0-1.0)

    Returns:
        Median gray value in the center region
    """
    h, w = bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((xx - geom.cx) ** 2 + (yy - geom.cy) ** 2) / max(float(geom.r), 1e-6)
    mask = rr <= float(r_clear)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if not np.any(mask):
        return float(np.median(gray))
    return float(np.median(gray[mask]))


def mean_gray_outer(bgr: np.ndarray, geom: LensGeometry, r_outer: float = 0.98) -> float:
    """
    Calculate median gray value in outer region.

    Args:
        bgr: Input BGR image
        geom: Lens geometry
        r_outer: Normalized radius threshold for outer region (0.0-1.0)

    Returns:
        Median gray value in the outer region
    """
    h, w = bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((xx - geom.cx) ** 2 + (yy - geom.cy) ** 2) / max(float(geom.r), 1e-6)
    mask = rr >= float(r_outer)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if not np.any(mask):
        return float(np.median(gray))
    return float(np.median(gray[mask]))


def radial_mask(h: int, w: int, geom: LensGeometry, r0: float, r1: float) -> np.ndarray:
    """
    Create radial annulus mask.

    Args:
        h: Image height
        w: Image width
        geom: Lens geometry
        r0: Inner normalized radius
        r1: Outer normalized radius

    Returns:
        Boolean mask where True indicates pixels in the annulus
    """
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((xx - geom.cx) ** 2 + (yy - geom.cy) ** 2) / max(float(geom.r), 1e-6)
    return (rr >= r0) & (rr <= r1)


def mask_to_polar(mask: np.ndarray, geom: LensGeometry, R: int, T: int) -> np.ndarray:
    """
    Convert binary mask to polar coordinates.

    Uses INTER_NEAREST to prevent mask bleeding/thickening that occurs with
    linear interpolation. Threshold >127 ensures only pixels with >50% mask
    value are included, preventing soft edges from expanding the mask.

    Args:
        mask: Binary mask (H, W)
        geom: Lens geometry
        R: Radial dimension of output
        T: Angular dimension of output

    Returns:
        Polar mask (T, R) as boolean array
    """
    mask_u8 = mask.astype(np.uint8) * 255
    flags = cv2.WARP_POLAR_LINEAR | cv2.INTER_NEAREST
    polar = cv2.warpPolar(mask_u8, (R, T), (geom.cx, geom.cy), geom.r, flags)
    return polar > 127  # Only pixels with >50% mask value


def median_bgr_center(bgr: np.ndarray, geom: LensGeometry, r_clear: float) -> np.ndarray:
    """
    Calculate median BGR value in center region.

    Args:
        bgr: Input BGR image
        geom: Lens geometry
        r_clear: Normalized radius of clear region

    Returns:
        Median BGR value as float32 array
    """
    h, w = bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((xx - geom.cx) ** 2 + (yy - geom.cy) ** 2) / max(float(geom.r), 1e-6)
    mask = rr <= float(r_clear)
    if not np.any(mask):
        return np.median(bgr.reshape(-1, 3), axis=0).astype(np.float32)
    pixels = bgr[mask]
    return np.median(pixels, axis=0).astype(np.float32)
