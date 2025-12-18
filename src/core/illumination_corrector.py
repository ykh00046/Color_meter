"""
Illumination Corrector Module

조명 편차 보정 (선택 사항)
불균일한 조명으로 인한 휘도 편차를 보정하여 색상 측정 정밀도 향상.

주의: 색상(a, b)은 보정하지 않고 휘도(L)만 보정하여 색상 왜곡 방지.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CorrectorConfig:
    """
    조명 보정 설정

    Attributes:
        enabled: 조명 보정 활성화 여부
        method: 보정 방법
            - 'polynomial': 다항식 피팅 (Vignetting 보정)
            - 'gaussian': 가우시안 스무딩 (Vignetting 보정)
            - 'gray_world': Gray World 알고리즘 (White Balance)
            - 'white_patch': White Patch 알고리즘 (White Balance)
            - 'auto': Gray World와 White Patch 중 자동 선택
        polynomial_degree: 다항식 차수 (method='polynomial' 시)
        target_luminance: 목표 휘도 (None이면 평균 휘도 사용)
        preserve_color: 색상 보존 (True=L만 보정, False=Lab 전체 보정)
        target_mean: Gray World 목표 평균 (0-255, method='gray_world' 시)
    """

    enabled: bool = False  # 기본값: 비활성화 (명시적 활성화 필요)
    method: str = "polynomial"  # 기본값: polynomial (backward compatibility)
    polynomial_degree: int = 2
    target_luminance: Optional[float] = None
    preserve_color: bool = True
    target_mean: float = 128.0  # Gray World 목표 평균 (method='gray_world' 사용 시)


@dataclass
class CorrectionResult:
    """
    조명 보정 결과

    Attributes:
        corrected_image: 보정된 Lab 이미지
        correction_applied: 보정이 적용되었는지 여부
        luminance_profile: 반경별 평균 휘도 프로파일 (보정 전)
        correction_factors: 반경별 보정 계수
        method: 사용된 보정 방법
        scaling_factors: RGB 채널 스케일링 팩터 (R, G, B) - White Balance 전용
        deviation_before: 보정 전 채널 편차
        deviation_after: 보정 후 채널 편차
    """

    corrected_image: np.ndarray
    correction_applied: bool
    luminance_profile: Optional[np.ndarray] = None
    correction_factors: Optional[np.ndarray] = None
    method: Optional[str] = None
    scaling_factors: Optional[Tuple[float, float, float]] = None
    deviation_before: Optional[float] = None
    deviation_after: Optional[float] = None


class IlluminationCorrectorError(Exception):
    """IlluminationCorrector 처리 중 발생하는 예외"""

    pass


class IlluminationCorrector:
    """
    조명 편차 보정기 (선택 사항)

    불균일한 조명으로 인한 휘도 변화를 감지하고 보정.
    주로 Vignetting(중심 밝고 외곽 어두운) 현상 보정.

    알고리즘:
    1. 극좌표 변환으로 반경별 평균 휘도 계산
    2. 다항식 또는 가우시안 피팅으로 조명 프로파일 추정
    3. 목표 휘도로 정규화
    4. L 채널만 보정 (a, b는 색상이므로 유지)

    주의:
    - 과도한 보정은 색상 왜곡 유발 가능
    - 실제 색상 그라데이션과 조명 불균일 구분 어려움
    - 기본값은 비활성화 (enabled=False)
    """

    def __init__(self, config: CorrectorConfig = None):
        """
        IlluminationCorrector 초기화

        Args:
            config: 조명 보정 설정 (None이면 기본값 사용, 기본값은 비활성화)
        """
        self.config = config or CorrectorConfig()
        logger.info(
            f"IlluminationCorrector initialized: enabled={self.config.enabled}, " f"method={self.config.method}"
        )

    def correct(
        self, image_lab: np.ndarray, center_x: float, center_y: float, radius: float, mask: Optional[np.ndarray] = None
    ) -> CorrectionResult:
        """
        조명 편차 보정 적용

        Args:
            image_lab: Lab 색공간 이미지 (H × W × 3)
            center_x: 렌즈 중심 x 좌표 (픽셀)
            center_y: 렌즈 중심 y 좌표 (픽셀)
            radius: 렌즈 반경 (픽셀)
            mask: 배경 마스크 (H × W, bool, True=유효 픽셀, None=마스크 없음)

        Returns:
            CorrectionResult: 보정 결과

        Raises:
            IlluminationCorrectorError: 이미지 오류 시

        Example:
            >>> corrector = IlluminationCorrector(CorrectorConfig(enabled=True))
            >>> result = corrector.correct(image_lab, 400, 400, 350)
            >>> corrected_image = result.corrected_image
        """
        # 입력 검증
        if image_lab is None or image_lab.size == 0:
            raise IlluminationCorrectorError("Input image is empty")

        if len(image_lab.shape) != 3 or image_lab.shape[2] != 3:
            raise IlluminationCorrectorError(f"Expected Lab image (H×W×3), got {image_lab.shape}")

        # 보정 비활성화 시 원본 반환
        if not self.config.enabled:
            logger.debug("Illumination correction disabled, returning original image")
            return CorrectionResult(corrected_image=image_lab, correction_applied=False)

        h, w = image_lab.shape[:2]

        # PHASE7 White Balance 메서드 (gray_world, white_patch, auto)
        if self.config.method in ("gray_world", "white_patch", "auto"):
            return self._correct_white_balance(image_lab, center_x, center_y, radius, mask)

        # 기존 Vignetting 보정 메서드 (polynomial, gaussian)
        # 1. 반경별 평균 휘도 계산
        luminance_profile = self._calculate_luminance_profile(image_lab, center_x, center_y, radius, mask)

        # 2. 조명 프로파일 추정
        if self.config.method == "polynomial":
            correction_factors = self._polynomial_correction(luminance_profile)
        elif self.config.method == "gaussian":
            correction_factors = self._gaussian_correction(luminance_profile)
        else:
            raise IlluminationCorrectorError(f"Unknown correction method: {self.config.method}")

        # 3. 보정 적용
        corrected_image = self._apply_correction(image_lab, center_x, center_y, radius, correction_factors)

        result = CorrectionResult(
            corrected_image=corrected_image,
            correction_applied=True,
            luminance_profile=luminance_profile,
            correction_factors=correction_factors,
            method=self.config.method,
        )

        logger.info(f"Illumination correction applied: method={self.config.method}")

        return result

    def _calculate_luminance_profile(
        self, image_lab: np.ndarray, center_x: float, center_y: float, radius: float, mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        반경별 평균 휘도 프로파일 계산

        Args:
            image_lab: Lab 이미지
            center_x: 중심 x
            center_y: 중심 y
            radius: 반경
            mask: 배경 마스크

        Returns:
            반경별 평균 L 값 (정규화된 반경 0~1)
        """
        h, w = image_lab.shape[:2]
        y, x = np.ogrid[:h, :w]

        # 극좌표 변환
        dx = x - center_x
        dy = y - center_y
        r = np.sqrt(dx**2 + dy**2)
        r_norm = r / radius

        # 반경별 평균 계산 (100개 구간)
        n_bins = 100
        r_bins = np.linspace(0.0, 1.0, n_bins)
        luminance_profile = np.zeros(n_bins)

        L_channel = image_lab[:, :, 0]

        for i in range(n_bins - 1):
            r_start = r_bins[i]
            r_end = r_bins[i + 1]

            # 마스크 생성
            ring_mask = (r_norm >= r_start) & (r_norm < r_end)
            if mask is not None:
                ring_mask = ring_mask & mask

            # 평균 계산
            if np.sum(ring_mask) > 0:
                luminance_profile[i] = np.mean(L_channel[ring_mask])
            else:
                luminance_profile[i] = 0.0

        # 마지막 구간 처리
        luminance_profile[-1] = luminance_profile[-2] if luminance_profile[-2] > 0 else 128.0

        return luminance_profile

    def _polynomial_correction(self, luminance_profile: np.ndarray) -> np.ndarray:
        """
        다항식 피팅 기반 보정 계수 계산

        Args:
            luminance_profile: 반경별 평균 휘도

        Returns:
            보정 계수 배열
        """
        # 목표 휘도 (평균 또는 설정값)
        if self.config.target_luminance is not None:
            target_L = self.config.target_luminance
        else:
            target_L = np.mean(luminance_profile[luminance_profile > 0])

        # 다항식 피팅
        x = np.linspace(0.0, 1.0, len(luminance_profile))
        valid_mask = luminance_profile > 0

        if np.sum(valid_mask) < self.config.polynomial_degree + 1:
            # 데이터 부족 → 보정 없음
            logger.warning("Insufficient data for polynomial fitting")
            return np.ones_like(luminance_profile)

        # 피팅
        poly_coeffs = np.polyfit(x[valid_mask], luminance_profile[valid_mask], self.config.polynomial_degree)
        fitted_luminance = np.polyval(poly_coeffs, x)

        # 보정 계수 = 목표 / 추정값
        correction_factors = np.divide(
            target_L, fitted_luminance, out=np.ones_like(fitted_luminance), where=fitted_luminance > 0
        )

        # 과도한 보정 방지 (0.5 ~ 2.0 범위로 제한)
        correction_factors = np.clip(correction_factors, 0.5, 2.0)

        return correction_factors

    def _gaussian_correction(self, luminance_profile: np.ndarray) -> np.ndarray:
        """
        가우시안 스무딩 기반 보정 계수 계산

        Args:
            luminance_profile: 반경별 평균 휘도

        Returns:
            보정 계수 배열
        """
        # 목표 휘도
        if self.config.target_luminance is not None:
            target_L = self.config.target_luminance
        else:
            target_L = np.mean(luminance_profile[luminance_profile > 0])

        # 가우시안 스무딩
        from scipy.ndimage import gaussian_filter1d

        smoothed_luminance = gaussian_filter1d(luminance_profile, sigma=5.0)

        # 보정 계수
        correction_factors = np.divide(
            target_L, smoothed_luminance, out=np.ones_like(smoothed_luminance), where=smoothed_luminance > 0
        )

        # 과도한 보정 방지
        correction_factors = np.clip(correction_factors, 0.5, 2.0)

        return correction_factors

    def _apply_correction(
        self, image_lab: np.ndarray, center_x: float, center_y: float, radius: float, correction_factors: np.ndarray
    ) -> np.ndarray:
        """
        보정 계수를 이미지에 적용

        Args:
            image_lab: Lab 이미지
            center_x: 중심 x
            center_y: 중심 y
            radius: 반경
            correction_factors: 보정 계수 배열 (길이 100)

        Returns:
            보정된 Lab 이미지
        """
        h, w = image_lab.shape[:2]
        y, x = np.ogrid[:h, :w]

        # 극좌표 변환
        dx = x - center_x
        dy = y - center_y
        r = np.sqrt(dx**2 + dy**2)
        r_norm = r / radius

        # 보정 계수 맵 생성 (100개 → 이미지 크기로 보간)
        n_bins = len(correction_factors)
        r_norm_clipped = np.clip(r_norm, 0.0, 0.999)
        bin_indices = (r_norm_clipped * (n_bins - 1)).astype(int)
        correction_map = correction_factors[bin_indices]

        # 보정 적용 (L 채널만)
        corrected_image = image_lab.copy()

        if self.config.preserve_color:
            # L만 보정 (색상 보존)
            corrected_image[:, :, 0] = np.clip(image_lab[:, :, 0] * correction_map, 0, 255).astype(np.uint8)
        else:
            # Lab 전체 보정 (권장하지 않음)
            for i in range(3):
                corrected_image[:, :, i] = np.clip(image_lab[:, :, i] * correction_map, 0, 255).astype(np.uint8)

        return corrected_image

    def _correct_white_balance(
        self, image_lab: np.ndarray, center_x: float, center_y: float, radius: float, mask: Optional[np.ndarray]
    ) -> CorrectionResult:
        """
        White Balance 보정 (PHASE7 Priority 4)
        Gray World / White Patch / Auto 메서드 사용

        Args:
            image_lab: Lab 이미지
            center_x: 중심 x
            center_y: 중심 y
            radius: 반경
            mask: 배경 마스크

        Returns:
            CorrectionResult: 보정 결과
        """
        # ROI 마스크 생성 (렌즈 영역만 사용)
        roi_mask = self._create_roi_mask(image_lab, center_x, center_y, radius)
        if mask is not None:
            roi_mask = roi_mask & mask

        # Lab → BGR 변환 (보정은 BGR에서 수행)
        image_bgr = cv2.cvtColor(image_lab, cv2.COLOR_Lab2BGR)

        # 원본 편차 계산
        deviation_before = self._calculate_channel_deviation(image_bgr, roi_mask)

        # 보정 알고리즘 선택 및 적용
        if self.config.method == "gray_world":
            corrected_bgr, scaling_factors = self._gray_world(image_bgr, roi_mask)
            method_used = "gray_world"
        elif self.config.method == "white_patch":
            corrected_bgr, scaling_factors = self._white_patch(image_bgr, roi_mask)
            method_used = "white_patch"
        elif self.config.method == "auto":
            corrected_bgr, scaling_factors, method_used = self._auto_select(image_bgr, roi_mask)
        else:
            raise IlluminationCorrectorError(f"Unknown white balance method: {self.config.method}")

        # 보정 후 편차 계산
        deviation_after = self._calculate_channel_deviation(corrected_bgr, roi_mask)

        # BGR → Lab 변환
        corrected_lab = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2Lab)

        logger.info(
            f"White balance correction applied: method={method_used}, "
            f"scaling_factors={scaling_factors}, "
            f"deviation: {deviation_before:.2f} → {deviation_after:.2f}"
        )

        return CorrectionResult(
            corrected_image=corrected_lab,
            correction_applied=True,
            method=method_used,
            scaling_factors=scaling_factors,
            deviation_before=deviation_before,
            deviation_after=deviation_after,
        )

    def _create_roi_mask(self, image: np.ndarray, center_x: float, center_y: float, radius: float) -> np.ndarray:
        """ROI 마스크 생성 (렌즈 영역만)"""
        h, w = image.shape[:2]
        y_coords, x_coords = np.ogrid[:h, :w]

        # 원형 마스크 생성
        dist_from_center = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        mask = dist_from_center <= radius

        logger.debug(
            f"ROI mask created: center=({center_x:.0f}, {center_y:.0f}), "
            f"radius={radius:.0f}, valid_pixels={np.sum(mask):,}"
        )

        return mask

    def _gray_world(
        self, image_bgr: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Gray World 알고리즘: 각 채널의 평균을 목표값으로 스케일

        Args:
            image_bgr: BGR 이미지
            mask: ROI 마스크 (None이면 전체 이미지 사용)

        Returns:
            tuple: (corrected_bgr, scaling_factors)
        """
        image_float = image_bgr.astype(np.float32)

        # 각 채널의 평균 계산
        if mask is not None:
            b_mean = np.mean(image_float[:, :, 0][mask])
            g_mean = np.mean(image_float[:, :, 1][mask])
            r_mean = np.mean(image_float[:, :, 2][mask])
        else:
            b_mean = np.mean(image_float[:, :, 0])
            g_mean = np.mean(image_float[:, :, 1])
            r_mean = np.mean(image_float[:, :, 2])

        # 스케일링 팩터 계산
        target = self.config.target_mean
        scale_b = target / b_mean if b_mean > 0 else 1.0
        scale_g = target / g_mean if g_mean > 0 else 1.0
        scale_r = target / r_mean if r_mean > 0 else 1.0

        # 각 채널에 스케일 적용
        corrected = image_float.copy()
        corrected[:, :, 0] *= scale_b
        corrected[:, :, 1] *= scale_g
        corrected[:, :, 2] *= scale_r

        # Clip to valid range
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        logger.debug(
            f"Gray World: means=({b_mean:.1f}, {g_mean:.1f}, {r_mean:.1f}), "
            f"scales=({scale_b:.3f}, {scale_g:.3f}, {scale_r:.3f})"
        )

        return corrected, (scale_r, scale_g, scale_b)

    def _white_patch(
        self, image_bgr: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        White Patch 알고리즘: 가장 밝은 픽셀을 흰색으로 스케일

        Args:
            image_bgr: BGR 이미지
            mask: ROI 마스크 (None이면 전체 이미지 사용)

        Returns:
            tuple: (corrected_bgr, scaling_factors)
        """
        image_float = image_bgr.astype(np.float32)

        # 각 채널의 최대값 계산
        if mask is not None:
            b_max = np.max(image_float[:, :, 0][mask])
            g_max = np.max(image_float[:, :, 1][mask])
            r_max = np.max(image_float[:, :, 2][mask])
        else:
            b_max = np.max(image_float[:, :, 0])
            g_max = np.max(image_float[:, :, 1])
            r_max = np.max(image_float[:, :, 2])

        # 스케일링 팩터 계산
        scale_b = 255.0 / b_max if b_max > 0 else 1.0
        scale_g = 255.0 / g_max if g_max > 0 else 1.0
        scale_r = 255.0 / r_max if r_max > 0 else 1.0

        # 각 채널에 스케일 적용
        corrected = image_float.copy()
        corrected[:, :, 0] *= scale_b
        corrected[:, :, 1] *= scale_g
        corrected[:, :, 2] *= scale_r

        # Clip to valid range
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        logger.debug(
            f"White Patch: maxs=({b_max:.1f}, {g_max:.1f}, {r_max:.1f}), "
            f"scales=({scale_b:.3f}, {scale_g:.3f}, {scale_r:.3f})"
        )

        return corrected, (scale_r, scale_g, scale_b)

    def _auto_select(
        self, image_bgr: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Tuple[float, float, float], str]:
        """
        Auto 선택: 두 방법 중 원본과 편차가 적은 방법 선택

        Args:
            image_bgr: BGR 이미지
            mask: ROI 마스크

        Returns:
            tuple: (corrected_bgr, scaling_factors, method_name)
        """
        # 원본 편차
        original_deviation = self._calculate_channel_deviation(image_bgr, mask)

        # Gray World 시도
        corrected_gw, scales_gw = self._gray_world(image_bgr, mask)
        deviation_gw = self._calculate_channel_deviation(corrected_gw, mask)

        # White Patch 시도
        corrected_wp, scales_wp = self._white_patch(image_bgr, mask)
        deviation_wp = self._calculate_channel_deviation(corrected_wp, mask)

        # 원본과의 차이가 적은 방법 선택
        diff_gw = abs(deviation_gw - original_deviation)
        diff_wp = abs(deviation_wp - original_deviation)

        if diff_gw <= diff_wp:
            logger.debug(f"Auto selected Gray World: deviation={deviation_gw:.2f} (diff={diff_gw:.2f})")
            return corrected_gw, scales_gw, "gray_world"
        else:
            logger.debug(f"Auto selected White Patch: deviation={deviation_wp:.2f} (diff={diff_wp:.2f})")
            return corrected_wp, scales_wp, "white_patch"

    def _calculate_channel_deviation(self, image_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        채널 간 편차 계산 (표준편차의 평균)

        Args:
            image_bgr: BGR 이미지
            mask: ROI 마스크

        Returns:
            float: 채널 간 편차 (낮을수록 균일)
        """
        if mask is not None:
            b_values = image_bgr[:, :, 0][mask]
            g_values = image_bgr[:, :, 1][mask]
            r_values = image_bgr[:, :, 2][mask]
        else:
            b_values = image_bgr[:, :, 0].flatten()
            g_values = image_bgr[:, :, 1].flatten()
            r_values = image_bgr[:, :, 2].flatten()

        # 각 채널의 표준편차
        std_b = np.std(b_values)
        std_g = np.std(g_values)
        std_r = np.std(r_values)

        # 표준편차의 평균
        deviation = (std_b + std_g + std_r) / 3.0

        return deviation
