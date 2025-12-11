"""
Zone Segmenter Module

변곡점 기반 자동 Zone 분할 모듈.
RadialProfile의 a* 그래디언트 분석을 통해 잉크 영역을 자동 분할한다.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy.signal import find_peaks, savgol_filter
import logging

from src.core.radial_profiler import RadialProfile

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    """
    색상 영역 데이터 클래스.

    Attributes:
        name: Zone 이름 ('A', 'A-B', 'B', 'B-C', 'C' 등)
        r_start: 시작 반경 (정규화, 0.0~1.0)
        r_end: 종료 반경 (정규화, 0.0~1.0)
        mean_L: 평균 L* 값
        mean_a: 평균 a* 값
        mean_b: 평균 b* 값
        std_L: L* 표준편차
        std_a: a* 표준편차
        std_b: b* 표준편차
        zone_type: 'pure' (순수 영역) 또는 'mix' (혼합 영역)
    """
    name: str
    r_start: float
    r_end: float
    mean_L: float
    mean_a: float
    mean_b: float
    std_L: float
    std_a: float
    std_b: float
    zone_type: str  # 'pure' or 'mix'


@dataclass
class SegmenterConfig:
    """
    ZoneSegmenter 설정.

    Attributes:
        detection_method: 변곡점 검출 방법 ('gradient' 또는 'delta_e')
        min_zone_width: 최소 zone 폭 (정규화, 0.0~1.0)
        smoothing_window: 그래디언트 스무딩 윈도우 크기 (홀수)
        min_gradient: 최소 그래디언트 크기 (피크 검출 임계값)
    """
    detection_method: str = 'gradient'
    min_zone_width: float = 0.05  # 최소 5% 폭
    smoothing_window: int = 11    # Savitzky-Golay 윈도우
    min_gradient: float = 0.5     # 최소 그래디언트


class ZoneSegmentationError(Exception):
    """Zone 분할 실패 시 발생하는 예외"""
    pass


class ZoneSegmenter:
    """
    변곡점 기반 Zone 자동 분할 클래스.

    RadialProfile의 a* 프로파일에서 그래디언트 급변 지점을 검출하여
    잉크 영역을 자동으로 분할한다.
    """

    def __init__(self, config: SegmenterConfig = SegmenterConfig()):
        """
        ZoneSegmenter 초기화.

        Args:
            config: Segmenter 설정
        """
        self.config = config

    def segment(self, profile: RadialProfile) -> List[Zone]:
        """
        프로파일을 Zone으로 분할.

        Args:
            profile: 색상 프로파일 (RadialProfile 객체)

        Returns:
            Zone 리스트 (바깥쪽 → 안쪽 순서)

        Raises:
            ZoneSegmentationError: Zone 분할 실패 시
        """
        if profile is None:
            raise ValueError("Profile cannot be None")

        # 1. 변곡점 검출
        inflections = self._detect_inflection_points(profile)

        logger.debug(f"Detected {len(inflections)} inflection points: {inflections}")

        # 2. 경계점 정렬 (바깥 → 안쪽, 즉 큰 r → 작은 r)
        boundaries = sorted(inflections, reverse=True)

        # 시작/끝 추가 (전체 범위 포함)
        r_max = profile.r_normalized[-1]  # 가장 큰 r (바깥쪽)
        r_min = profile.r_normalized[0]   # 가장 작은 r (안쪽)
        boundaries = [r_max] + boundaries + [r_min]

        logger.debug(f"Zone boundaries: {boundaries}")

        # 3. 각 구간을 Zone으로 변환
        zones = []
        zone_labels = self._generate_zone_labels(len(boundaries) - 1)

        for i in range(len(boundaries) - 1):
            r_start = boundaries[i]
            r_end = boundaries[i + 1]

            # 해당 구간의 평균 색상 계산
            mask = (profile.r_normalized >= r_end) & (profile.r_normalized < r_start)

            if np.sum(mask) == 0:
                logger.warning(f"No data points in zone {zone_labels[i]} ({r_start:.3f} - {r_end:.3f})")
                continue

            mean_L = np.mean(profile.L[mask])
            mean_a = np.mean(profile.a[mask])
            mean_b = np.mean(profile.b[mask])
            std_L = np.std(profile.L[mask])
            std_a = np.std(profile.a[mask])
            std_b = np.std(profile.b[mask])

            zone = Zone(
                name=zone_labels[i],
                r_start=r_start,
                r_end=r_end,
                mean_L=mean_L,
                mean_a=mean_a,
                mean_b=mean_b,
                std_L=std_L,
                std_a=std_a,
                std_b=std_b,
                zone_type='pure' if '-' not in zone_labels[i] else 'mix'
            )
            zones.append(zone)

        if len(zones) == 0:
            raise ZoneSegmentationError("No zones created after segmentation")

        logger.info(f"Segmented into {len(zones)} zones: {[z.name for z in zones]}")

        return zones

    def _detect_inflection_points(self, profile: RadialProfile) -> List[float]:
        """
        변곡점 r 값 검출.

        a* 프로파일의 그래디언트 급변 지점을 찾아 반환.
        a* 값이 색상 변화를 가장 잘 나타냄 (빨강-초록 축).

        Args:
            profile: 색상 프로파일

        Returns:
            변곡점의 r 값 리스트
        """
        if self.config.detection_method == 'gradient':
            return self._detect_by_gradient(profile)
        elif self.config.detection_method == 'delta_e':
            return self._detect_by_delta_e(profile)
        else:
            raise ValueError(f"Unknown detection method: {self.config.detection_method}")

    def _detect_by_gradient(self, profile: RadialProfile) -> List[float]:
        """
        a* 그래디언트 기반 변곡점 검출.

        Args:
            profile: 색상 프로파일

        Returns:
            변곡점의 r 값 리스트
        """
        # a* 1차 미분 (그래디언트)
        gradient_a = np.gradient(profile.a)

        # 스무딩 (노이즈 제거)
        if len(gradient_a) >= self.config.smoothing_window:
            gradient_smooth = savgol_filter(
                gradient_a,
                window_length=self.config.smoothing_window,
                polyorder=2
            )
        else:
            # 데이터 포인트가 윈도우보다 작으면 스무딩 스킵
            gradient_smooth = gradient_a

        # 그래디언트 절댓값에서 피크 검출
        min_distance_points = int(self.config.min_zone_width * len(profile.r_normalized))

        peaks, properties = find_peaks(
            np.abs(gradient_smooth),
            height=self.config.min_gradient,
            distance=max(1, min_distance_points)  # 최소 1 픽셀
        )

        # 피크에 해당하는 r 값
        if len(peaks) > 0:
            inflection_r = profile.r_normalized[peaks]
            return inflection_r.tolist()
        else:
            # 변곡점이 없으면 빈 리스트 반환 (단일 zone)
            logger.warning("No inflection points detected")
            return []

    def _detect_by_delta_e(self, profile: RadialProfile) -> List[float]:
        """
        ΔE 기반 변곡점 검출.

        연속된 두 반경 간 ΔE를 계산하여 급변 지점을 찾음.

        Args:
            profile: 색상 프로파일

        Returns:
            변곡점의 r 값 리스트
        """
        from src.utils.color_delta import delta_e_cie2000

        r = profile.r_normalized
        L = profile.L
        a = profile.a
        b = profile.b

        # 각 r에서 다음 r까지의 ΔE 계산
        delta_e_profile = []
        for i in range(len(r) - 1):
            lab1 = (L[i], a[i], b[i])
            lab2 = (L[i+1], a[i+1], b[i+1])
            de = delta_e_cie2000(lab1, lab2)
            delta_e_profile.append(de)

        delta_e_profile = np.array(delta_e_profile)

        # ΔE 피크 검출
        min_distance_points = int(self.config.min_zone_width * len(r))

        peaks, _ = find_peaks(
            delta_e_profile,
            height=5.0,  # 최소 ΔE = 5.0 (시각적으로 구별 가능)
            distance=max(1, min_distance_points)
        )

        if len(peaks) > 0:
            inflection_r = r[peaks]
            return inflection_r.tolist()
        else:
            logger.warning("No inflection points detected by delta_e method")
            return []

    def _generate_zone_labels(self, n_zones: int) -> List[str]:
        """
        Zone 개수에 따라 레이블 생성.

        Args:
            n_zones: Zone 개수

        Returns:
            Zone 레이블 리스트

        Examples:
            5개 → ['A', 'A-B', 'B', 'B-C', 'C']
            3개 → ['A', 'B', 'C']
            7개 → ['A', 'A-B', 'B', 'B-C', 'C', 'C-D', 'D']
        """
        if n_zones == 5:
            return ['A', 'A-B', 'B', 'B-C', 'C']
        elif n_zones == 3:
            return ['A', 'B', 'C']
        elif n_zones == 7:
            # 4색 잉크인 경우
            return ['A', 'A-B', 'B', 'B-C', 'C', 'C-D', 'D']
        elif n_zones == 1:
            # 변곡점 없음 (단일 zone)
            return ['A']
        else:
            # 일반적: 번호로 레이블
            return [f'Zone{i+1}' for i in range(n_zones)]

    def evaluate_mix_zone(
        self,
        mix_zone: Zone,
        pure_zone_before: Zone,
        pure_zone_after: Zone
    ) -> dict:
        """
        혼합 영역의 색상이 두 순수 영역 사이에 있는지 검증.

        Args:
            mix_zone: 혼합 영역
            pure_zone_before: 이전 순수 영역
            pure_zone_after: 다음 순수 영역

        Returns:
            dict:
                - is_valid: 혼합이 정상인지 여부
                - distance_from_line: 이론 혼합선으로부터의 거리
                - blend_ratio: 혼합 비율 추정 (0~1)
        """
        # 두 순수 영역의 LAB 값
        lab1 = np.array([pure_zone_before.mean_L,
                         pure_zone_before.mean_a,
                         pure_zone_before.mean_b])
        lab2 = np.array([pure_zone_after.mean_L,
                         pure_zone_after.mean_a,
                         pure_zone_after.mean_b])

        # Mix zone의 LAB 값
        lab_mix = np.array([mix_zone.mean_L,
                            mix_zone.mean_a,
                            mix_zone.mean_b])

        # 직선 lab1 - lab2 위의 가장 가까운 점 찾기
        line_vec = lab2 - lab1
        mix_vec = lab_mix - lab1

        # 투영
        line_length_sq = np.dot(line_vec, line_vec)
        if line_length_sq < 1e-6:
            # 두 순수 영역이 거의 같은 색상
            return {
                'is_valid': True,
                'distance_from_line': 0.0,
                'blend_ratio': 0.5
            }

        t = np.dot(mix_vec, line_vec) / line_length_sq
        t = np.clip(t, 0, 1)  # 0~1 범위로 제한

        closest_point = lab1 + t * line_vec

        # 거리 계산
        distance = np.linalg.norm(lab_mix - closest_point)

        # 허용 거리: 두 순수 영역 표준편차의 평균
        avg_std = (
            (pure_zone_before.std_L + pure_zone_after.std_L +
             pure_zone_before.std_a + pure_zone_after.std_a +
             pure_zone_before.std_b + pure_zone_after.std_b) / 6.0
        )

        # 정상 판정: 거리가 3*표준편차 이내
        is_valid = distance <= (3.0 * avg_std)

        logger.debug(f"Mix zone {mix_zone.name}: distance={distance:.2f}, "
                     f"blend_ratio={t:.2f}, is_valid={is_valid}")

        return {
            'is_valid': is_valid,
            'distance_from_line': distance,
            'blend_ratio': t
        }
