"""
Background Masker Module

2단계 배경 마스킹:
1단계: 원형 경계 마스킹 (렌즈 반경 외곽 제거)
2단계: 휘도/채도 기반 배경 픽셀 정밀 제거 (테두리 반사, 케이스 등)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MaskConfig:
    """
    배경 마스킹 설정

    Attributes:
        use_luminance_filter: 휘도 기반 필터링 사용 여부
        use_saturation_filter: 채도 기반 필터링 사용 여부
        L_min: 최소 휘도 (Lab L, 0~255) - 너무 어두운 픽셀 제거
        L_max: 최대 휘도 (Lab L, 0~255) - 너무 밝은 픽셀 제거 (반사광)
        saturation_min: 최소 채도 - 무채색 배경 제거
        morphology_enabled: 형태학적 연산 사용 여부 (노이즈 제거)
        morph_kernel_size: 형태학적 연산 커널 크기
    """

    use_luminance_filter: bool = True
    use_saturation_filter: bool = True
    L_min: float = 20.0
    L_max: float = 240.0
    saturation_min: float = 5.0
    morphology_enabled: bool = True
    morph_kernel_size: int = 3


@dataclass
class MaskResult:
    """
    마스킹 결과

    Attributes:
        mask: 이진 마스크 (True=유효, False=배경)
        valid_pixel_ratio: 유효 픽셀 비율 (0.0~1.0)
        filtered_by_luminance: 휘도 필터링으로 제거된 픽셀 수
        filtered_by_saturation: 채도 필터링으로 제거된 픽셀 수
        morphology_applied: 형태학적 연산 적용 여부
    """

    mask: np.ndarray
    valid_pixel_ratio: float
    filtered_by_luminance: int
    filtered_by_saturation: int
    morphology_applied: bool


class BackgroundMaskerError(Exception):
    """BackgroundMasker 처리 중 발생하는 예외"""

    pass


class BackgroundMasker:
    """
    2단계 배경 마스킹

    1단계: 원형 경계 마스킹 (렌즈 반경 외곽)
    2단계: 휘도/채도 기반 정밀 마스킹 (테두리 반사, 케이스, 잡음)

    알고리즘:
    1. 원형 마스크 생성 (렌즈 중심, 반경)
    2. Lab 색공간에서 L(휘도) 임계값 필터링
    3. Lab → HSV 변환 후 채도(S) 임계값 필터링
    4. 형태학적 연산 (Opening/Closing)으로 노이즈 제거

    PHASE7 Advanced 알고리즘:
    1. ROI 외곽에서 배경색 샘플링
    2. Otsu 이진화 + 색상 거리 이중 마스킹
    3. 형태학적 정제
    """

    def __init__(self, config: MaskConfig = None):
        """
        BackgroundMasker 초기화

        Args:
            config: 마스킹 설정 (None이면 기본값 사용)
        """
        self.config = config or MaskConfig()
        logger.info(
            f"BackgroundMasker initialized: L=[{self.config.L_min}, {self.config.L_max}], "
            f"saturation>={self.config.saturation_min}"
        )

    def create_advanced_mask(
        self,
        image_bgr: np.ndarray,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        radius: Optional[float] = None,
    ) -> MaskResult:
        """
        PHASE7 고급 배경 마스킹: ROI 기반 배경 샘플링 + Otsu + 색상 거리

        더 강건한 배경/렌즈 분리를 위해 케이스, 그림자, 오염에 대응합니다.

        Args:
            image_bgr: BGR 이미지 (H × W × 3)
            center_x: 렌즈 중심 x (픽셀). None이면 이미지 중심 사용
            center_y: 렌즈 중심 y (픽셀). None이면 이미지 중심 사용
            radius: 렌즈 반경 (픽셀). None이면 이미지 크기 기반 추정

        Returns:
            MaskResult: 마스킹 결과

        Algorithm:
            Stage 1: ROI 밖에서 배경색 샘플링 (lens 정보 활용)
            Stage 2: Otsu + 색상 거리 이중 마스킹
            Stage 3: 형태학적 정제
        """
        if image_bgr is None or image_bgr.size == 0:
            raise BackgroundMaskerError("Input image is empty")

        if len(image_bgr.shape) != 3 or image_bgr.shape[2] != 3:
            raise BackgroundMaskerError(f"Expected BGR image (H×W×3), got {image_bgr.shape}")

        h, w = image_bgr.shape[:2]

        # 기본값 설정
        if center_x is None:
            center_x = w / 2
        if center_y is None:
            center_y = h / 2
        if radius is None:
            radius = min(h, w) / 2.5  # Conservative estimate

        # Stage 1: ROI 기반 배경 샘플링
        bg_color = self._sample_background_color(image_bgr, center_x, center_y, radius)

        logger.debug(f"Detected background color (BGR): {bg_color}")

        # Stage 2a: Otsu 이진화
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Stage 2b: 색상 거리 마스크
        color_dist = np.linalg.norm(image_bgr.astype(np.float32) - bg_color, axis=2)
        _, color_dist_8bit = cv2.threshold(color_dist.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # AND 결합 (둘 다 foreground로 판단된 픽셀만)
        combined_mask = cv2.bitwise_and(otsu_mask, color_dist_8bit.astype(np.uint8))

        # Stage 3: 형태학적 정제
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # 통계 계산
        final_mask_bool = combined_mask > 0
        total_pixels = h * w
        valid_pixels = int(np.sum(final_mask_bool))
        valid_ratio = valid_pixels / total_pixels if total_pixels > 0 else 0.0

        result = MaskResult(
            mask=final_mask_bool,
            valid_pixel_ratio=valid_ratio,
            filtered_by_luminance=0,  # Not used in advanced method
            filtered_by_saturation=0,  # Not used in advanced method
            morphology_applied=True,
        )

        logger.info(f"Advanced mask created: {valid_pixels}/{total_pixels} pixels " f"({valid_ratio*100:.1f}% valid)")

        return result

    def _sample_background_color(
        self, image_bgr: np.ndarray, center_x: float, center_y: float, radius: float
    ) -> np.ndarray:
        """
        ROI 밖에서 배경색 샘플링

        Args:
            image_bgr: BGR 이미지
            center_x: 렌즈 중심 x
            center_y: 렌즈 중심 y
            radius: 렌즈 반경

        Returns:
            배경색 (B, G, R) numpy array
        """
        h, w = image_bgr.shape[:2]

        # 렌즈 영역 마스크 생성 (여유 20%)
        lens_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(
            lens_mask,
            (int(center_x), int(center_y)),
            int(radius * 1.2),  # 20% margin
            255,
            -1,
        )

        # ROI 밖에서만 샘플링
        bg_samples = image_bgr[lens_mask == 0]

        if len(bg_samples) == 0:
            # Fallback: 네 모서리 샘플링
            logger.warning("No background pixels outside ROI, using corner sampling")
            corners = [
                image_bgr[0:10, 0:10],
                image_bgr[0:10, w - 10 : w],
                image_bgr[h - 10 : h, 0:10],
                image_bgr[h - 10 : h, w - 10 : w],
            ]
            bg_samples = np.vstack([c.reshape(-1, 3) for c in corners])

        # 배경색: 중앙값 사용 (outlier에 강함)
        bg_color = np.median(bg_samples, axis=0).astype(np.float32)

        return bg_color

    def create_mask(self, image_lab: np.ndarray, center_x: float, center_y: float, radius: float) -> MaskResult:
        """
        2단계 배경 마스크 생성

        Args:
            image_lab: Lab 색공간 이미지 (H × W × 3)
            center_x: 렌즈 중심 x 좌표 (픽셀)
            center_y: 렌즈 중심 y 좌표 (픽셀)
            radius: 렌즈 반경 (픽셀)

        Returns:
            MaskResult: 마스킹 결과

        Raises:
            BackgroundMaskerError: 이미지 오류 시

        Example:
            >>> masker = BackgroundMasker()
            >>> mask_result = masker.create_mask(image_lab, 400, 400, 350)
            >>> print(f"Valid pixels: {mask_result.valid_pixel_ratio*100:.1f}%")
        """
        # 입력 검증
        if image_lab is None or image_lab.size == 0:
            raise BackgroundMaskerError("Input image is empty")

        if len(image_lab.shape) != 3 or image_lab.shape[2] != 3:
            raise BackgroundMaskerError(f"Expected Lab image (H×W×3), got {image_lab.shape}")

        h, w = image_lab.shape[:2]

        # 1단계: 원형 마스크 생성
        circular_mask = self._create_circular_mask(h, w, center_x, center_y, radius)

        # 2단계: 휘도/채도 기반 필터링
        luminance_mask = np.ones((h, w), dtype=bool)
        saturation_mask = np.ones((h, w), dtype=bool)

        filtered_luminance = 0
        filtered_saturation = 0

        if self.config.use_luminance_filter:
            luminance_mask, filtered_luminance = self._filter_by_luminance(image_lab)

        if self.config.use_saturation_filter:
            saturation_mask, filtered_saturation = self._filter_by_saturation(image_lab)

        # 최종 마스크: 모든 조건 AND
        final_mask = circular_mask & luminance_mask & saturation_mask

        # 형태학적 연산 (노이즈 제거)
        morphology_applied = False
        if self.config.morphology_enabled:
            final_mask = self._apply_morphology(final_mask)
            morphology_applied = True

        # 통계 계산
        total_pixels = h * w
        valid_pixels = int(np.sum(final_mask))
        valid_ratio = valid_pixels / total_pixels if total_pixels > 0 else 0.0

        result = MaskResult(
            mask=final_mask,
            valid_pixel_ratio=valid_ratio,
            filtered_by_luminance=filtered_luminance,
            filtered_by_saturation=filtered_saturation,
            morphology_applied=morphology_applied,
        )

        logger.info(f"Mask created: {valid_pixels}/{total_pixels} pixels ({valid_ratio*100:.1f}% valid)")
        logger.debug(f"Filtered: L={filtered_luminance}, S={filtered_saturation}")

        return result

    def _create_circular_mask(
        self, height: int, width: int, center_x: float, center_y: float, radius: float
    ) -> np.ndarray:
        """
        원형 마스크 생성 (1단계)

        Args:
            height: 이미지 높이
            width: 이미지 너비
            center_x: 중심 x
            center_y: 중심 y
            radius: 반경

        Returns:
            이진 마스크 (bool)
        """
        y, x = np.ogrid[:height, :width]
        dx = x - center_x
        dy = y - center_y
        distance = np.sqrt(dx**2 + dy**2)

        mask = distance <= radius
        return mask

    def _filter_by_luminance(self, image_lab: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        휘도(L) 기반 필터링

        Args:
            image_lab: Lab 이미지

        Returns:
            (마스크, 제거된 픽셀 수)
        """
        L_channel = image_lab[:, :, 0]

        # L_min ~ L_max 범위 내 픽셀만 유지
        mask = (L_channel >= self.config.L_min) & (L_channel <= self.config.L_max)

        filtered_count = int(np.sum(~mask))

        return mask, filtered_count

    def _filter_by_saturation(self, image_lab: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        채도(Saturation) 기반 필터링

        Args:
            image_lab: Lab 이미지

        Returns:
            (마스크, 제거된 픽셀 수)
        """
        # Lab → BGR → HSV 변환
        # (주의: Lab이 uint8 타입이어야 함)
        try:
            image_bgr = cv2.cvtColor(image_lab, cv2.COLOR_Lab2BGR)
            image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            saturation = image_hsv[:, :, 1]

            # 채도가 saturation_min 이상인 픽셀만 유지
            mask = saturation >= self.config.saturation_min

            filtered_count = int(np.sum(~mask))

            return mask, filtered_count

        except Exception as e:
            logger.warning(f"Saturation filtering failed: {e}")
            # Fallback: 모든 픽셀 유지
            h, w = image_lab.shape[:2]
            return np.ones((h, w), dtype=bool), 0

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        형태학적 연산 적용 (노이즈 제거)

        Args:
            mask: 입력 마스크

        Returns:
            정제된 마스크
        """
        # uint8 변환 (OpenCV 형태학적 연산은 uint8 필요)
        mask_uint8 = mask.astype(np.uint8) * 255

        # 커널 생성
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.config.morph_kernel_size, self.config.morph_kernel_size)
        )

        # Opening (침식 후 팽창) - 작은 노이즈 제거
        mask_opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

        # Closing (팽창 후 침식) - 작은 구멍 메우기
        mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

        # bool 변환
        final_mask = mask_closed > 0

        return final_mask
