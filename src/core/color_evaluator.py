"""
Color Evaluator Module

색상 평가 및 OK/NG 판정 모듈.
CIEDE2000 색차 계산을 통해 SKU 기준 대비 품질을 평가한다.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from src.utils.color_delta import (
    delta_e_cie1976,
    delta_e_cie1994,
    delta_e_cie2000
)

from src.core.zone_segmenter import Zone

logger = logging.getLogger(__name__)


@dataclass
class ZoneResult:
    """
    Zone별 평가 결과.

    Attributes:
        zone_name: Zone 이름
        measured_lab: 측정된 LAB 값 (L*, a*, b*)
        target_lab: 기준 LAB 값
        delta_e: 색차 (ΔE2000)
        threshold: 허용 색차
        is_ok: 이 zone의 판정 (True=OK, False=NG)
    """
    zone_name: str
    measured_lab: tuple  # (L*, a*, b*)
    target_lab: tuple    # (L*, a*, b*)
    delta_e: float
    threshold: float
    is_ok: bool


@dataclass
class InspectionResult:
    """
    검사 결과.

    Attributes:
        sku: SKU 코드
        timestamp: 검사 시간
        judgment: 최종 판정 ('OK' 또는 'NG')
        overall_delta_e: 전체 평균 ΔE
        zone_results: Zone별 결과 리스트
        ng_reasons: NG 이유 목록
        confidence: 판정 신뢰도 (0.0~1.0)
        lens_detection: 렌즈 검출 결과 (시각화용, optional)
        zones: Zone 리스트 (시각화용, optional)
        image: 원본 이미지 (시각화용, optional)
    """
    sku: str
    timestamp: datetime
    judgment: str  # 'OK' or 'NG'
    overall_delta_e: float
    zone_results: List[ZoneResult]
    ng_reasons: List[str]
    confidence: float
    lens_detection: Optional[Any] = None  # LensDetection
    zones: Optional[List[Zone]] = None
    image: Optional[Any] = None  # np.ndarray


class ColorEvaluationError(Exception):
    """색상 평가 실패 시 발생하는 예외"""
    pass


class ColorEvaluator:
    """
    색상 평가 및 OK/NG 판정 클래스.

    CIEDE2000 색차 계산을 통해 측정된 Zone의 LAB 값을
    SKU 기준 값과 비교하여 품질을 평가한다.
    """

    def __init__(self, sku_config: Optional[Dict[str, Any]] = None):
        """
        ColorEvaluator 초기화.

        Args:
            sku_config: SKU별 기준값 설정 (dict 형태)
                {
                    'sku_code': 'SKU_001',
                    'default_threshold': 3.0,
                    'zones': {
                        'A': {'L': 45.2, 'a': 15.8, 'b': -42.3, 'threshold': 4.2},
                        'B': {'L': 78.3, 'a': -5.6, 'b': 65.2, 'threshold': 3.8},
                        ...
                    }
                }
        """
        self.sku_config = sku_config or {}

    def evaluate(
        self,
        zones: List[Zone],
        sku: str,
        sku_config: Optional[Dict[str, Any]] = None
    ) -> InspectionResult:
        """
        Zone 리스트를 평가하여 OK/NG 판정.

        Args:
            zones: 측정된 Zone 리스트
            sku: 제품 SKU 코드
            sku_config: SKU별 기준값 (옵션, 없으면 self.sku_config 사용)

        Returns:
            InspectionResult: 검사 결과

        Raises:
            ColorEvaluationError: SKU 기준값 없음 또는 평가 실패 시
        """
        # SKU 기준값 로드
        config = sku_config or self.sku_config

        if not config or 'zones' not in config:
            raise ColorEvaluationError(f"SKU {sku}의 기준값이 등록되지 않음")

        zone_targets = config['zones']
        default_threshold = config.get('default_threshold', 3.0)

        # 각 Zone 평가
        zone_results = []
        ng_reasons = []
        delta_e_list = []

        for zone in zones:
            # 해당 zone의 기준값 찾기
            target = zone_targets.get(zone.name)

            if target is None:
                # 기준값 없으면 스킵 (경고 로그)
                logger.warning(f"Zone {zone.name}의 기준값 없음")
                continue

            # 측정값
            measured_lab = (zone.mean_L, zone.mean_a, zone.mean_b)
            target_lab = (target['L'], target['a'], target['b'])

            # ΔE 계산
            de = self.calculate_delta_e(measured_lab, target_lab)
            delta_e_list.append(de)

            # 허용치 비교
            threshold = target.get('delta_e_threshold', target.get('threshold', default_threshold))
            is_ok = de <= threshold

            zone_result = ZoneResult(
                zone_name=zone.name,
                measured_lab=measured_lab,
                target_lab=target_lab,
                delta_e=de,
                threshold=threshold,
                is_ok=is_ok
            )
            zone_results.append(zone_result)

            # NG 이유 수집
            if not is_ok:
                ng_reasons.append(
                    f"Zone {zone.name}: ΔE={de:.2f} > {threshold:.2f}"
                )

        # 전체 판정
        all_ok = all(zr.is_ok for zr in zone_results)
        judgment = 'OK' if all_ok else 'NG'

        # 전체 평균 ΔE
        overall_delta_e = np.mean(delta_e_list) if delta_e_list else 0.0

        # 신뢰도 계산 (간단한 방법: 1 - 평균(ΔE/threshold))
        confidence = self._calculate_confidence(zone_results)

        result = InspectionResult(
            sku=sku,
            timestamp=datetime.now(),
            judgment=judgment,
            overall_delta_e=overall_delta_e,
            zone_results=zone_results,
            ng_reasons=ng_reasons,
            confidence=confidence
        )

        logger.info(f"Evaluation result for {sku}: {judgment}, "
                    f"avg ΔE={overall_delta_e:.2f}, confidence={confidence:.2f}")

        return result

    def calculate_delta_e(
        self,
        lab1: tuple,
        lab2: tuple,
        method: str = 'cie2000'
    ) -> float:
        """
        색차(ΔE) 계산.

        Args:
            lab1: (L1, a1, b1)
            lab2: (L2, a2, b2)
            method: 'cie1976', 'cie1994', or 'cie2000' (기본값)

        Returns:
            ΔE 값

        Raises:
            ValueError: 잘못된 method 또는 LAB 값
        """
        try:
            if method == 'cie2000':
                return delta_e_cie2000(lab1, lab2)
            elif method == 'cie1994':
                return delta_e_cie1994(lab1, lab2)
            elif method == 'cie1976':
                return delta_e_cie1976(lab1, lab2)
            else:
                raise ValueError(f"Unknown delta_e method: {method}")

        except Exception as e:
            logger.error(f"Error calculating delta_e: {e}")
            raise ColorEvaluationError(f"Delta E calculation failed: {e}")

    def _calculate_confidence(self, zone_results: List[ZoneResult]) -> float:
        """
        판정 신뢰도 계산.

        Args:
            zone_results: Zone별 결과 리스트

        Returns:
            신뢰도 (0.0~1.0)
        """
        if not zone_results:
            return 0.0

        # 신뢰도 = 1 - 평균(ΔE / threshold)
        # ΔE가 threshold에 가까울수록 신뢰도 낮음
        ratios = []
        for zr in zone_results:
            if zr.threshold > 0:
                ratio = zr.delta_e / zr.threshold
                ratios.append(min(ratio, 1.0))  # 1.0 초과는 1.0으로 clamp

        if not ratios:
            return 0.0

        avg_ratio = np.mean(ratios)
        confidence = 1.0 - avg_ratio

        # 0.0~1.0 범위로 clamp
        return max(0.0, min(1.0, confidence))

    def evaluate_with_mix_check(
        self,
        zones: List[Zone],
        sku: str,
        sku_config: Optional[Dict[str, Any]] = None,
        check_mix_zones: bool = True
    ) -> InspectionResult:
        """
        혼합 영역 검증을 포함한 평가.

        Args:
            zones: 측정된 Zone 리스트
            sku: 제품 SKU 코드
            sku_config: SKU별 기준값
            check_mix_zones: 혼합 영역 검증 수행 여부

        Returns:
            InspectionResult: 검사 결과
        """
        # 기본 평가
        result = self.evaluate(zones, sku, sku_config)

        # 혼합 영역 검증 (옵션)
        if check_mix_zones:
            from src.core.zone_segmenter import ZoneSegmenter, SegmenterConfig

            segmenter = ZoneSegmenter(SegmenterConfig())

            # Mix zone 검증
            for i, zone in enumerate(zones):
                if zone.zone_type == 'mix' and i > 0 and i < len(zones) - 1:
                    # 이전/다음 순수 영역 찾기
                    prev_pure = zones[i - 1] if zones[i - 1].zone_type == 'pure' else None
                    next_pure = zones[i + 1] if zones[i + 1].zone_type == 'pure' else None

                    if prev_pure and next_pure:
                        mix_eval = segmenter.evaluate_mix_zone(zone, prev_pure, next_pure)

                        if not mix_eval['is_valid']:
                            # Mix zone 불량
                            result.ng_reasons.append(
                                f"Zone {zone.name} (Mix): 비정상 혼합 "
                                f"(distance={mix_eval['distance_from_line']:.2f})"
                            )
                            result.judgment = 'NG'

                            logger.warning(f"Mix zone {zone.name} validation failed")

        return result
