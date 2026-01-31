"""
Shared Inspection Data Schemas

Core data structures for inspection results and evaluations.
Shared between legacy and v7 code to avoid duplication.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.engine_v7.core.types import LensGeometry


@dataclass
class InspectionResult:
    """
    검사 결과.

    Attributes:
        sku: SKU 코드
        timestamp: 검사 시간
        judgment: 최종 판정 ('OK', 'OK_WITH_WARNING', 'NG', 'RETAKE')
        overall_delta_e: 전체 평균 ΔE
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
        ink_analysis: 잉크 색 분석 결과 (M3)
        radial_profile: Radial profile 데이터 (P1-2) - r_normalized, L, a, b, std 등
        lens_detection: 렌즈 검출 결과 (시각화용, optional)
        image: 원본 이미지 (시각화용, optional)
        ring_sector_cells: Ring × Sector 2D 분할 셀 리스트 (PHASE7, optional)
        uniformity_analysis: 자기 참조 균일성 분석 결과 (PHASE7, optional)
        metrics: Quality metrics (blur, histogram, dot_stats)
    """

    sku: str
    timestamp: datetime
    judgment: str  # 'OK', 'OK_WITH_WARNING', 'NG', 'RETAKE'
    overall_delta_e: float
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
    lens_detection: Optional[LensGeometry] = None
    image: Optional[Any] = None  # np.ndarray
    ring_sector_cells: Optional[List[Any]] = None  # List[RingSectorCell] from angular_profiler
    uniformity_analysis: Optional[Dict[str, Any]] = None  # Uniformity analysis result
    metrics: Optional[Dict[str, Any]] = None  # Quality metrics: blur, histogram, dot_stats
