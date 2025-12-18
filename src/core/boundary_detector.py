"""
Boundary Detector Module

RadialProfile의 표준편차 분석을 통해 유효 색상 영역(r_inner ~ r_outer)을 자동 검출.
중심부 잡음과 외곽 배경을 자동으로 제외하여 분석 정밀도를 향상.
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from src.core.radial_profiler import RadialProfile

logger = logging.getLogger(__name__)


@dataclass
class BoundaryConfig:
    """
    경계 검출 설정

    Attributes:
        std_threshold_multiplier: 표준편차 임계값 승수 (중앙값 × 승수)
        min_r_inner: 최소 r_inner (0.0~1.0)
        max_r_outer: 최대 r_outer (0.0~1.0)
        edge_detection_method: 'std' (표준편차 기반) 또는 'gradient' (기울기 기반)
        smoothing_window: 표준편차 프로파일 평활화 윈도우 크기
    """

    std_threshold_multiplier: float = 1.5
    min_r_inner: float = 0.05
    max_r_outer: float = 0.95
    edge_detection_method: str = "std"
    smoothing_window: int = 5


@dataclass
class BoundaryResult:
    """
    경계 검출 결과

    Attributes:
        r_inner: 검출된 내부 경계 (정규화된 반경)
        r_outer: 검출된 외부 경계 (정규화된 반경)
        confidence: 검출 신뢰도 (0.0~1.0)
        method: 사용된 검출 방법
        metadata: 추가 메타데이터 (디버깅용)
    """

    r_inner: float
    r_outer: float
    confidence: float
    method: str
    metadata: dict


class BoundaryDetectorError(Exception):
    """BoundaryDetector 처리 중 발생하는 예외"""

    pass


class BoundaryDetector:
    """
    유효 영역 경계 자동 검출기

    RadialProfile의 표준편차 분석을 통해:
    - r_inner: 중심부 고분산 영역(반사광) 제외
    - r_outer: 외곽 저분산 영역(배경) 제외

    알고리즘:
    1. 종합 표준편차 계산 (std_L, std_a, std_b의 RMS)
    2. 중앙값 기반 임계값 설정
    3. 중심부에서 외곽으로 스캔하여 r_inner 검출
    4. 외곽에서 중심으로 스캔하여 r_outer 검출
    """

    def __init__(self, config: BoundaryConfig = None):
        """
        BoundaryDetector 초기화

        Args:
            config: 경계 검출 설정 (None이면 기본값 사용)
        """
        self.config = config or BoundaryConfig()
        logger.info(f"BoundaryDetector initialized: method={self.config.edge_detection_method}")

    def detect_boundaries(self, profile: RadialProfile) -> BoundaryResult:
        """
        RadialProfile로부터 r_inner, r_outer 자동 검출

        Args:
            profile: RadialProfile 객체

        Returns:
            BoundaryResult: 검출된 경계 정보

        Raises:
            BoundaryDetectorError: 프로파일 데이터 오류 시

        Example:
            >>> detector = BoundaryDetector()
            >>> boundaries = detector.detect_boundaries(radial_profile)
            >>> print(f"Valid region: {boundaries.r_inner:.2f} ~ {boundaries.r_outer:.2f}")
        """
        # 입력 검증
        if profile is None:
            raise BoundaryDetectorError("RadialProfile is None")

        if len(profile.r_normalized) == 0:
            raise BoundaryDetectorError("RadialProfile is empty")

        # 종합 표준편차 계산 (RMS)
        std_combined = self._calculate_combined_std(profile)

        # 평활화 (노이즈 제거)
        if self.config.smoothing_window > 1:
            std_combined = self._smooth_profile(std_combined)

        # 경계 검출
        if self.config.edge_detection_method == "std":
            r_inner, r_outer, confidence = self._detect_by_std_threshold(profile.r_normalized, std_combined)
        elif self.config.edge_detection_method == "gradient":
            r_inner, r_outer, confidence = self._detect_by_gradient(profile.r_normalized, std_combined)
        else:
            raise BoundaryDetectorError(f"Unknown method: {self.config.edge_detection_method}")

        # 범위 제한 적용
        r_inner = max(r_inner, self.config.min_r_inner)
        r_outer = min(r_outer, self.config.max_r_outer)

        # 유효성 검증
        if r_inner >= r_outer:
            logger.warning(f"Invalid boundaries: r_inner={r_inner:.3f} >= r_outer={r_outer:.3f}")
            # Fallback to safe defaults
            r_inner = self.config.min_r_inner
            r_outer = self.config.max_r_outer
            confidence = 0.0

        result = BoundaryResult(
            r_inner=r_inner,
            r_outer=r_outer,
            confidence=confidence,
            method=self.config.edge_detection_method,
            metadata={
                "std_mean": float(np.mean(std_combined)),
                "std_max": float(np.max(std_combined)),
                "std_min": float(np.min(std_combined)),
            },
        )

        logger.info(
            f"Boundaries detected: r_inner={r_inner:.3f}, r_outer={r_outer:.3f}, " f"confidence={confidence:.2f}"
        )

        return result

    def _calculate_combined_std(self, profile: RadialProfile) -> np.ndarray:
        """
        종합 표준편차 계산 (RMS)

        Args:
            profile: RadialProfile 객체

        Returns:
            종합 표준편차 배열
        """
        # RMS (Root Mean Square) 계산
        std_combined = np.sqrt((profile.std_L**2 + profile.std_a**2 + profile.std_b**2) / 3.0)
        return std_combined

    def _smooth_profile(self, data: np.ndarray) -> np.ndarray:
        """
        프로파일 평활화 (이동 평균)

        Args:
            data: 입력 데이터

        Returns:
            평활화된 데이터
        """
        window = self.config.smoothing_window
        if len(data) < window:
            return data

        # 이동 평균 (mode='same'으로 길이 유지)
        kernel = np.ones(window) / window
        smoothed = np.convolve(data, kernel, mode="same")

        return smoothed

    def _detect_by_std_threshold(
        self, r_normalized: np.ndarray, std_combined: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        표준편차 임계값 기반 경계 검출

        Args:
            r_normalized: 정규화된 반경 배열
            std_combined: 종합 표준편차 배열

        Returns:
            (r_inner, r_outer, confidence)
        """
        # 임계값 계산 (중앙값 × 승수)
        median_std = np.median(std_combined)
        threshold = median_std * self.config.std_threshold_multiplier

        logger.debug(f"STD threshold: median={median_std:.2f}, threshold={threshold:.2f}")

        # r_inner 검출: 중심부에서 외곽으로 스캔
        # 고분산 영역(반사광)을 지나 안정된 영역 찾기
        r_inner = self.config.min_r_inner
        for i in range(len(std_combined)):
            if r_normalized[i] < self.config.min_r_inner:
                continue

            # 안정 영역 도달 (표준편차가 임계값 이하)
            if std_combined[i] < threshold:
                r_inner = r_normalized[i]
                break

        # r_outer 검출: 외곽에서 중심으로 스캔
        # 배경 영역(저분산 또는 급격한 증가)을 제외
        r_outer = self.config.max_r_outer
        for i in range(len(std_combined) - 1, -1, -1):
            if r_normalized[i] > self.config.max_r_outer:
                continue

            # 유효 영역 도달 (표준편차가 급격히 증가하지 않음)
            if i > 0 and std_combined[i] < threshold * 2.0:
                r_outer = r_normalized[i]
                break

        # 신뢰도 계산 (검출된 영역의 크기 비율)
        valid_range = r_outer - r_inner
        confidence = min(1.0, valid_range / 0.8)  # 80% 이상이면 신뢰도 1.0

        return r_inner, r_outer, confidence

    def _detect_by_gradient(self, r_normalized: np.ndarray, std_combined: np.ndarray) -> Tuple[float, float, float]:
        """
        기울기 기반 경계 검출

        Args:
            r_normalized: 정규화된 반경 배열
            std_combined: 종합 표준편차 배열

        Returns:
            (r_inner, r_outer, confidence)
        """
        # 표준편차의 기울기 계산
        gradient = np.gradient(std_combined)

        # r_inner: 중심부에서 큰 음의 기울기 찾기 (고분산 → 저분산 전환점)
        r_inner = self.config.min_r_inner
        for i in range(1, len(gradient)):
            if r_normalized[i] < self.config.min_r_inner:
                continue

            # 음의 기울기가 0에 가까워지는 지점
            if gradient[i] > -0.5 and std_combined[i] < np.median(std_combined):
                r_inner = r_normalized[i]
                break

        # r_outer: 외곽에서 양의 기울기 찾기 (저분산 → 고분산 전환점)
        r_outer = self.config.max_r_outer
        for i in range(len(gradient) - 1, 0, -1):
            if r_normalized[i] > self.config.max_r_outer:
                continue

            # 양의 기울기가 증가하기 전 지점
            if gradient[i] < 1.0 and std_combined[i] < np.median(std_combined) * 1.5:
                r_outer = r_normalized[i]
                break

        # 신뢰도 계산
        valid_range = r_outer - r_inner
        confidence = min(1.0, valid_range / 0.8)

        return r_inner, r_outer, confidence
