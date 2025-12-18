"""
Color Evaluator Module

색상 평가 및 OK/NG 판정 모듈.
CIEDE2000 색차 계산을 통해 SKU 기준 대비 품질을 평가한다.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.zone_segmenter import Zone
from src.utils.color_delta import delta_e_cie1976, delta_e_cie1994, delta_e_cie2000

logger = logging.getLogger(__name__)


def describe_color_shift(dL: float, da: float, db: float) -> str:
    """
    색상 변화를 사람이 이해하기 쉬운 말로 번역 (6개 카테고리)

    Args:
        dL: 측정값 - 기준값 (L*)
        da: 측정값 - 기준값 (a*)
        db: 측정값 - 기준값 (b*)

    Returns:
        "어두워짐", "황색화 증가" 등의 설명
    """
    abs_vals = [(abs(dL), "L", dL), (abs(da), "a", da), (abs(db), "b", db)]
    abs_vals.sort(reverse=True)  # 가장 큰 변화 2개

    descriptions = []
    for i in range(min(2, len(abs_vals))):
        mag, axis, val = abs_vals[i]
        if mag < 1.0:  # 변화 미미
            continue

        if axis == "L":
            descriptions.append(f"{'어두워짐' if val < 0 else '밝아짐'}(ΔL={val:+.1f})")
        elif axis == "a":
            descriptions.append(f"{'녹색화' if val < 0 else '붉어짐'}(Δa={val:+.1f})")
        elif axis == "b":
            descriptions.append(f"{'청색화' if val < 0 else '황색화'}(Δb={val:+.1f})")

    return ", ".join(descriptions) if descriptions else "변화 미미"


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
        pixel_count: Zone 평균 계산에 사용된 픽셀(프로파일 포인트) 수
        diff: 색상 변화 상세 (운영 UX) - dL, da, db, direction
        std_lab: 표준편차 (std_L, std_a, std_b) - PHASE7 Priority 6
        chroma_stats: Chroma 사분위수 (q25, median, q75, iqr) - PHASE7 Priority 6
        internal_uniformity: 내부 균일도 점수 (0~1) - PHASE7 Priority 6
        uniformity_grade: 균일도 등급 (Good/Medium/Poor) - PHASE7 Priority 6
    """

    zone_name: str
    measured_lab: tuple  # (L*, a*, b*)
    target_lab: tuple  # (L*, a*, b*)
    delta_e: float
    threshold: float
    is_ok: bool
    pixel_count: int = 0  # Zone 평균 계산에 사용된 픽셀 수
    diff: Optional[Dict[str, Any]] = None  # 색상 변화 상세 (운영 UX)
    std_lab: Optional[tuple] = None  # (std_L, std_a, std_b) - PHASE7 Priority 6
    chroma_stats: Optional[Dict[str, float]] = None  # {q25, median, q75, iqr} - PHASE7 Priority 6
    internal_uniformity: Optional[float] = None  # 0~1 - PHASE7 Priority 6
    uniformity_grade: Optional[str] = None  # Good/Medium/Poor - PHASE7 Priority 6


@dataclass
class InspectionResult:
    """
    검사 결과.

    Attributes:
        sku: SKU 코드
        timestamp: 검사 시간
        judgment: 최종 판정 ('OK', 'OK_WITH_WARNING', 'NG', 'RETAKE')
        overall_delta_e: 전체 평균 ΔE
        zone_results: Zone별 결과 리스트
        ng_reasons: NG 이유 목록
        confidence: 판정 신뢰도 (0.0~1.0)
        decision_trace: 판정 과정 추적 (운영 UX) - final, because, overrides
        next_actions: 권장 조치 목록 (운영 UX)
        retake_reasons: RETAKE 상세 사유 (운영 UX) - code, reason, actions, lever
        analysis_summary: 프로파일 데이터 핵심 요약 (운영 UX) - uniformity, boundary, coverage
        confidence_breakdown: Confidence 분해 (운영 UX) - 각 요소별 기여도
        risk_factors: 위험 요소 목록 (운영 UX) - category, severity, message
        diagnostics: 진단 정보 목록 (PHASE7 Priority 5) - 각 단계별 처리 결과
        warnings: 경고 목록 (PHASE7 Priority 5) - 잠재적 문제점
        suggestions: 제안 목록 (PHASE7 Priority 5) - 개선/해결 방법
        ink_analysis: 잉크 색 분석 결과 (M3) - zone_based, image_based
        radial_profile: Radial profile 데이터 (P1-2) - r_normalized, L, a, b, std 등
        lens_detection: 렌즈 검출 결과 (시각화용, optional)
        zones: Zone 리스트 (시각화용, optional)
        image: 원본 이미지 (시각화용, optional)
        ring_sector_cells: Ring × Sector 2D 분할 셀 리스트 (PHASE7, optional)
        uniformity_analysis: 자기 참조 균일성 분석 결과 (PHASE7, optional)
    """

    sku: str
    timestamp: datetime
    judgment: str  # 'OK', 'OK_WITH_WARNING', 'NG', 'RETAKE'
    overall_delta_e: float
    zone_results: List[ZoneResult]
    ng_reasons: List[str]
    confidence: float
    decision_trace: Optional[Dict[str, Any]] = None  # 판정 추적 (운영 UX)
    next_actions: Optional[List[str]] = None  # 권장 조치 (운영 UX)
    retake_reasons: Optional[List[Dict[str, Any]]] = None  # RETAKE 상세 (운영 UX)
    analysis_summary: Optional[Dict[str, Any]] = None  # 프로파일 요약 (운영 UX)
    confidence_breakdown: Optional[Dict[str, Any]] = None  # Confidence 분해 (운영 UX)
    risk_factors: Optional[List[Dict[str, Any]]] = None  # 위험 요소 (운영 UX)
    diagnostics: Optional[List[str]] = None  # PHASE7 Priority 5: 진단 정보
    warnings: Optional[List[str]] = None  # PHASE7 Priority 5: 경고
    suggestions: Optional[List[str]] = None  # PHASE7 Priority 5: 제안
    ink_analysis: Optional[Dict[str, Any]] = None  # 잉크 색 도출 (사용자 목표)
    radial_profile: Optional[Dict[str, Any]] = None  # P1-2: Radial profile data for comparison
    lens_detection: Optional[Any] = None  # LensDetection
    zones: Optional[List[Zone]] = None
    image: Optional[Any] = None  # np.ndarray
    ring_sector_cells: Optional[List[Any]] = None  # List[RingSectorCell] from angular_profiler
    uniformity_analysis: Optional[Dict[str, Any]] = None  # Uniformity analysis result


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

    def evaluate(self, zones: List[Zone], sku: str, sku_config: Optional[Dict[str, Any]] = None) -> InspectionResult:
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

        if not config or "zones" not in config:
            raise ColorEvaluationError(f"SKU {sku}의 기준값이 등록되지 않음")

        zone_targets = config["zones"]
        default_threshold = config.get("default_threshold", 3.0)

        # 각 Zone 평가
        zone_results = []
        ng_reasons = []
        delta_e_list = []
        confidence_penalty = 1.0  # 신뢰도 페널티 (AI 피드백 반영)

        # SKU에 정의되지 않은 zone 검출 (경고)
        detected_zone_names = set(zone.name for zone in zones)
        sku_zone_names = set(zone_targets.keys())
        unexpected_zones = detected_zone_names - sku_zone_names

        if unexpected_zones:
            logger.warning(f"Detected zones not in SKU {sku}: {unexpected_zones}")
            # NG 이유에 추가
            ng_reasons.append(f"Unexpected zones detected: {', '.join(sorted(unexpected_zones))}")

        for zone in zones:
            # 해당 zone의 기준값 찾기
            target = zone_targets.get(zone.name)

            if target is None:
                # 기준값 없으면 스킵 (경고 로그)
                logger.warning(f"Zone {zone.name}의 기준값 없음")
                continue

            # AI 피드백 반영: pixel_count 하한선 검증
            # Ring Sector의 pixel_count가 2,700~14,000 수준이므로
            # Zone도 최소 2000 픽셀 이상이어야 대표성 있음
            MIN_PIXEL_COUNT = 2000
            if zone.pixel_count < MIN_PIXEL_COUNT:
                logger.warning(
                    f"Zone {zone.name} has insufficient pixels ({zone.pixel_count} < {MIN_PIXEL_COUNT}). "
                    f"This zone may not be representative."
                )
                ng_reasons.append(f"Zone {zone.name}: insufficient pixels ({zone.pixel_count})")
                # 신뢰도 페널티 누적
                confidence_penalty *= 0.7

            # 측정값
            measured_lab = (zone.mean_L, zone.mean_a, zone.mean_b)
            target_lab = (target["L"], target["a"], target["b"])

            # ΔE 계산
            de = self.calculate_delta_e(measured_lab, target_lab)
            delta_e_list.append(de)

            # 허용치 비교
            threshold = target.get("delta_e_threshold", target.get("threshold", default_threshold))
            is_ok = de <= threshold

            # PHASE7 Priority 6: Zone 내부 균일도 통계 계산
            zone_stats = self._calculate_zone_statistics(zone)

            zone_result = ZoneResult(
                zone_name=zone.name,
                measured_lab=measured_lab,
                target_lab=target_lab,
                delta_e=de,
                threshold=threshold,
                is_ok=is_ok,
                pixel_count=zone.pixel_count,  # AI 피드백 반영: pixel_count 전달
                std_lab=zone_stats["std_lab"],  # PHASE7 Priority 6
                chroma_stats=zone_stats["chroma_stats"],  # PHASE7 Priority 6
                internal_uniformity=zone_stats["internal_uniformity"],  # PHASE7 Priority 6
                uniformity_grade=zone_stats["uniformity_grade"],  # PHASE7 Priority 6
            )
            zone_results.append(zone_result)

            # NG 이유 수집
            if not is_ok:
                ng_reasons.append(f"Zone {zone.name}: ΔE={de:.2f} > {threshold:.2f}")

        # 빈 zone_results 체크 (Critical Bug Fix)
        if not zone_results:
            logger.error(f"No zones matched with SKU {sku} configuration")
            return InspectionResult(
                sku=sku,
                timestamp=datetime.now(),
                judgment="NG",
                overall_delta_e=0.0,
                zone_results=[],
                ng_reasons=["No zones matched with SKU configuration - check expected_zones or SKU settings"],
                confidence=0.0,
            )

        # expected_zones 수량 검증 (Critical Check)
        expected_zones = config.get("params", {}).get("expected_zones")
        actual_count = len(zones)

        if expected_zones is not None:
            if actual_count != expected_zones:
                msg = f"Zone count mismatch: expected {expected_zones}, detected {actual_count}"
                logger.warning(msg)
                ng_reasons.append(msg)
                # Zone 개수 불일치는 즉시 NG 사유 (우선순위 높음)
                # 그러나 개별 Zone 평가 결과도 함께 보여주기 위해 여기서 바로 리턴하지 않음

        # 전체 판정
        # ng_reasons가 하나라도 있으면 NG
        all_ok = (not ng_reasons) and all(zr.is_ok for zr in zone_results)
        judgment = "OK" if all_ok else "NG"

        # 전체 평균 ΔE
        overall_delta_e = float(np.mean(delta_e_list)) if delta_e_list else 0.0

        # 신뢰도 계산 (간단한 방법: 1 - 평균(ΔE/threshold))
        confidence = self._calculate_confidence(zone_results)

        # AI 피드백 반영: pixel_count 부족 페널티 적용
        confidence *= confidence_penalty

        result = InspectionResult(
            sku=sku,
            timestamp=datetime.now(),
            judgment=judgment,
            overall_delta_e=overall_delta_e,
            zone_results=zone_results,
            ng_reasons=ng_reasons,
            confidence=confidence,
        )

        logger.info(
            f"Evaluation result for {sku}: {judgment}, " f"avg ΔE={overall_delta_e:.2f}, confidence={confidence:.2f}"
        )

        return result

    def calculate_delta_e(self, lab1: tuple, lab2: tuple, method: str = "cie2000") -> float:
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
            if method == "cie2000":
                return delta_e_cie2000(lab1, lab2)
            elif method == "cie1994":
                return delta_e_cie1994(lab1, lab2)
            elif method == "cie1976":
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

        avg_ratio = float(np.mean(ratios))
        confidence = 1.0 - avg_ratio

        # 0.0~1.0 범위로 clamp
        return float(max(0.0, min(1.0, confidence)))

    def _calculate_zone_statistics(self, zone: Zone) -> Dict[str, Any]:
        """
        Zone 내부 균일도 통계 계산 (PHASE7 Priority 6)

        Args:
            zone: Zone 객체

        Returns:
            dict: {
                'std_lab': (std_L, std_a, std_b),
                'chroma_stats': {
                    'mean': float,  # 평균 chroma (approximation)
                    'std': float,   # chroma 표준편차 (approximation)
                },
                'internal_uniformity': float,  # 0~1, 1=완벽히 균일
                'uniformity_grade': str  # 'Good', 'Medium', 'Poor'
            }
        """
        # 표준편차
        std_lab = (zone.std_L, zone.std_a, zone.std_b)

        # Chroma 통계 (근사값)
        # Chroma = sqrt(a^2 + b^2)
        # 평균 chroma 근사: sqrt(mean_a^2 + mean_b^2)
        mean_chroma = np.sqrt(zone.mean_a**2 + zone.mean_b**2)

        # Chroma의 표준편차 근사
        # std(chroma) ≈ sqrt(std_a^2 + std_b^2) (rough approximation)
        std_chroma = np.sqrt(zone.std_a**2 + zone.std_b**2)

        chroma_stats = {
            "mean": float(mean_chroma),
            "std": float(std_chroma),
        }

        # 내부 균일도 점수 (0~1)
        # 표준편차가 작을수록 균일
        # internal_std = 평균(std_L, std_a, std_b)
        internal_std = float(np.mean([zone.std_L, zone.std_a, zone.std_b]))

        # 정규화: std가 0이면 1.0 (완벽 균일), std가 20 이상이면 0.0
        uniformity_score = 1.0 - min(internal_std / 20.0, 1.0)

        # 등급 부여
        if internal_std < 5:
            grade = "Good"
        elif internal_std < 10:
            grade = "Medium"
        else:
            grade = "Poor"

        return {
            "std_lab": std_lab,
            "chroma_stats": chroma_stats,
            "internal_uniformity": uniformity_score,
            "uniformity_grade": grade,
        }

    def evaluate_with_mix_check(
        self, zones: List[Zone], sku: str, sku_config: Optional[Dict[str, Any]] = None, check_mix_zones: bool = True
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
            from src.core.zone_segmenter import SegmenterConfig, ZoneSegmenter

            segmenter = ZoneSegmenter(SegmenterConfig())

            # Mix zone 검증
            for i, zone in enumerate(zones):
                if zone.zone_type == "mix" and i > 0 and i < len(zones) - 1:
                    # 이전/다음 순수 영역 찾기
                    prev_pure = zones[i - 1] if zones[i - 1].zone_type == "pure" else None
                    next_pure = zones[i + 1] if zones[i + 1].zone_type == "pure" else None

                    if prev_pure and next_pure:
                        mix_eval = segmenter.evaluate_mix_zone(zone, prev_pure, next_pure)

                        if not mix_eval["is_valid"]:
                            # Mix zone 불량
                            result.ng_reasons.append(
                                f"Zone {zone.name} (Mix): 비정상 혼합 "
                                f"(distance={mix_eval['distance_from_line']:.2f})"
                            )
                            result.judgment = "NG"

                            logger.warning(f"Mix zone {zone.name} validation failed")

        return result
