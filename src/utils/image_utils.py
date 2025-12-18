"""
유틸: 이미지 배열 보조 함수 모음.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class ImageValidationError(ValueError):
    """이미지 유효성 오류"""


def _validate_image(image: np.ndarray, name: str = "image") -> None:
    if not isinstance(image, np.ndarray):
        raise ImageValidationError(f"{name} must be numpy.ndarray")
    if image.dtype != np.uint8:
        raise ImageValidationError(f"{name} must have dtype uint8")
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ImageValidationError(f"{name} must have 3 or 4 channels (H, W, C)")


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """BGR 이미지를 RGB로 변환."""
    _validate_image(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """RGB 이미지를 BGR로 변환."""
    _validate_image(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def resize_keep_aspect(
    image: np.ndarray,
    max_width: int,
    max_height: int,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    종횡비를 유지하면서 최대 가로/세로 크기 이내로 리사이즈.
    """
    _validate_image(image)
    h, w = image.shape[:2]
    if w <= max_width and h <= max_height:
        return image.copy()
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def draw_circle(
    image: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """이미지 복사본에 원을 그려 반환."""
    _validate_image(image)
    out = image.copy()
    cv2.circle(out, center, int(radius), color, thickness=thickness)
    return out
