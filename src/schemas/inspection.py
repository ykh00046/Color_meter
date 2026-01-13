"""
Shared Inspection Data Schemas

Core data structures for inspection results, zones, and evaluations.
Shared between legacy and v7 code to avoid duplication.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Zone:
    """색상 영역 데이터 (1D radial zone from legacy engine)"""

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
    pixel_count: int = 0  # 이 Zone에 포함된 프로파일 포인트 수


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
    measured_lab: Tuple[float, float, float]  # (L*, a*, b*)
    target_lab: Tuple[float, float, float]  # (L*, a*, b*)
    delta_e: float
    threshold: float
    is_ok: bool
    pixel_count: int = 0  # Zone 평균 계산에 사용된 픽셀 수
    diff: Optional[Dict[str, Any]] = None  # 색상 변화 상세 (운영 UX)
    std_lab: Optional[Tuple[float, float, float]] = None  # (std_L, std_a, std_b) - PHASE7 Priority 6
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
        metrics: Quality metrics (blur, histogram, dot_stats)
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
    metrics: Optional[Dict[str, Any]] = None  # Quality metrics: blur, histogram, dot_stats
