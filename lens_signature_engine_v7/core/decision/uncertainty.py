"""
Uncertainty & Evidence Management

구조화된 불확실성/증거 체계로 의사결정 품질 향상
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UncertaintyFactor:
    """
    불확실성 요인

    의사결정을 어렵게 만드는 요소 (Gate 품질 저하, 샘플링 문제 등)
    """

    category: str  # "gate" | "sampling" | "segmentation" | "signature"
    code: str
    severity: str  # "low" | "medium" | "high" | "critical"
    impact: float  # 0~1 (불확실성 기여도)
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "code": self.code,
            "severity": self.severity,
            "impact": round(self.impact, 3),
            "message": self.message,
            "details": self.details,
        }


@dataclass
class Evidence:
    """
    의사결정 근거

    OK 또는 NG를 지지하는 증거
    """

    type: str  # "gate" | "signature" | "ink" | "spatial"
    code: str
    strength: str  # "weak" | "moderate" | "strong"
    weight: float  # 0~1 (의사결정 기여도)
    value: Any
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "code": self.code,
            "strength": self.strength,
            "weight": round(self.weight, 3),
            "value": self.value,
            "message": self.message,
        }


class UncertaintyBuilder:
    """불확실성 요인 수집기"""

    def __init__(self):
        self.factors: List[UncertaintyFactor] = []

    def add_gate_factor(self, code: str, severity: str, impact: float, message: str, **details) -> UncertaintyBuilder:
        """Gate 품질 불확실성 추가"""
        factor = UncertaintyFactor(
            category="gate", code=code, severity=severity, impact=impact, message=message, details=details
        )
        self.factors.append(factor)
        return self

    def add_sampling_factor(
        self, code: str, severity: str, impact: float, message: str, **details
    ) -> UncertaintyBuilder:
        """샘플링 품질 불확실성 추가"""
        factor = UncertaintyFactor(
            category="sampling", code=code, severity=severity, impact=impact, message=message, details=details
        )
        self.factors.append(factor)
        return self

    def add_segmentation_factor(
        self, code: str, severity: str, impact: float, message: str, **details
    ) -> UncertaintyBuilder:
        """색상 분리 불확실성 추가"""
        factor = UncertaintyFactor(
            category="segmentation", code=code, severity=severity, impact=impact, message=message, details=details
        )
        self.factors.append(factor)
        return self

    def add_signature_factor(
        self, code: str, severity: str, impact: float, message: str, **details
    ) -> UncertaintyBuilder:
        """프로파일 매칭 불확실성 추가"""
        factor = UncertaintyFactor(
            category="signature", code=code, severity=severity, impact=impact, message=message, details=details
        )
        self.factors.append(factor)
        return self

    def build(self) -> Dict[str, Any]:
        """
        최종 uncertainty 객체 생성

        Returns:
            {
                "level": str,  # "low" | "medium" | "high" | "critical"
                "total_score": float,
                "factors": List[dict]
            }
        """
        total_score = sum(f.impact for f in self.factors)
        level = _calculate_uncertainty_level(total_score)

        return {"level": level, "total_score": round(total_score, 3), "factors": [f.to_dict() for f in self.factors]}


class EvidenceBuilder:
    """증거 수집기"""

    def __init__(self):
        self.positive: List[Evidence] = []  # NG 지지
        self.negative: List[Evidence] = []  # OK 지지
        self.neutral: List[Evidence] = []  # 중립적 관찰

    def add_positive(
        self, type: str, code: str, strength: str, weight: float, value: Any, message: str
    ) -> EvidenceBuilder:
        """NG를 지지하는 증거 추가"""
        evidence = Evidence(type=type, code=code, strength=strength, weight=weight, value=value, message=message)
        self.positive.append(evidence)
        return self

    def add_negative(
        self, type: str, code: str, strength: str, weight: float, value: Any, message: str
    ) -> EvidenceBuilder:
        """OK를 지지하는 증거 추가"""
        evidence = Evidence(type=type, code=code, strength=strength, weight=weight, value=value, message=message)
        self.negative.append(evidence)
        return self

    def add_neutral(self, type: str, code: str, value: Any, message: str) -> EvidenceBuilder:
        """중립적 관찰 추가"""
        evidence = Evidence(type=type, code=code, strength="neutral", weight=0.0, value=value, message=message)
        self.neutral.append(evidence)
        return self

    def build(self) -> Dict[str, Any]:
        """
        최종 evidence 객체 생성

        Returns:
            {
                "positive": List[dict],
                "negative": List[dict],
                "neutral": List[dict]
            }
        """
        return {
            "positive": [e.to_dict() for e in self.positive],
            "negative": [e.to_dict() for e in self.negative],
            "neutral": [e.to_dict() for e in self.neutral],
        }


def _calculate_uncertainty_level(total_score: float) -> str:
    """
    불확실성 레벨 계산

    Args:
        total_score: 모든 factor의 impact 합

    Returns:
        "low" | "medium" | "high" | "critical"
    """
    if total_score >= 0.80:
        return "critical"
    elif total_score >= 0.50:
        return "high"
    elif total_score >= 0.20:
        return "medium"
    else:
        return "low"


def calculate_confidence(
    uncertainty_score: float, evidence_positive_weight: float, evidence_negative_weight: float
) -> float:
    """
    최종 확신도 계산

    Args:
        uncertainty_score: 0~1 (높을수록 불확실)
        evidence_positive_weight: NG 증거 가중치 합
        evidence_negative_weight: OK 증거 가중치 합

    Returns:
        confidence: 0~1 (1=매우 확신)
    """
    # 1. 불확실성이 높으면 확신도 낮음
    base_confidence = 1.0 - uncertainty_score

    # 2. 증거 균형도 계산
    total_evidence = evidence_positive_weight + evidence_negative_weight
    if total_evidence > 0:
        # 한쪽으로 치우칠수록 높음
        imbalance = abs(evidence_positive_weight - evidence_negative_weight) / total_evidence
        evidence_boost = imbalance * 0.30
    else:
        evidence_boost = 0.0

    # 3. 최종 확신도
    confidence = min(base_confidence + evidence_boost, 1.0)
    return max(confidence, 0.0)


def extract_reason_codes(uncertainty: Dict[str, Any], evidence: Dict[str, Any], result: str) -> List[str]:
    """
    Reason codes 추출 (간소화)

    Args:
        uncertainty: UncertaintyBuilder.build() 결과
        evidence: EvidenceBuilder.build() 결과
        result: Decision result

    Returns:
        reason_codes: ["RETAKE_REQUIRED", "INK_COUNT_MISMATCH", ...]
    """
    codes = []

    # 1. RETAKE 사유
    if result == "RETAKE":
        codes.append("RETAKE_REQUIRED")
        # Critical factors 추가
        for factor in uncertainty["factors"]:
            if factor["severity"] == "critical":
                codes.append(factor["code"])

    # 2. HOLD 사유
    elif result == "HOLD":
        codes.append("UNCERTAINTY_HIGH")
        codes.append("MANUAL_REVIEW_REQUIRED")

    # 3. NG_COLOR 사유
    elif result == "NG_COLOR":
        # Strong positive evidence 추가
        for ev in evidence["positive"]:
            if ev["strength"] == "strong":
                codes.append(ev["code"])

    # 중복 제거
    return list(dict.fromkeys(codes))


def create_uncertainty_summary(uncertainty: Dict[str, Any]) -> str:
    """
    불확실성 요약 문자열

    Args:
        uncertainty: UncertaintyBuilder.build() 결과

    Returns:
        summary: Human-readable summary
    """
    level = uncertainty["level"].upper()
    total = uncertainty["total_score"]
    count = len(uncertainty["factors"])

    if count == 0:
        return "No uncertainty factors"

    lines = [f"Uncertainty: {level} (score={total:.2f}, {count} factors)"]

    for factor in uncertainty["factors"]:
        lines.append(f"  [{factor['severity'].upper()}] {factor['code']}: {factor['message']}")

    return "\n".join(lines)


def create_evidence_summary(evidence: Dict[str, Any]) -> str:
    """
    증거 요약 문자열

    Args:
        evidence: EvidenceBuilder.build() 결과

    Returns:
        summary: Human-readable summary
    """
    pos_count = len(evidence["positive"])
    neg_count = len(evidence["negative"])
    neu_count = len(evidence["neutral"])

    lines = [f"Evidence: {pos_count} positive, {neg_count} negative, {neu_count} neutral"]

    if pos_count > 0:
        lines.append("  Positive (NG):")
        for ev in evidence["positive"]:
            lines.append(f"    [{ev['strength'].upper()}] {ev['code']}: {ev['message']}")

    if neg_count > 0:
        lines.append("  Negative (OK):")
        for ev in evidence["negative"]:
            lines.append(f"    [{ev['strength'].upper()}] {ev['code']}: {ev['message']}")

    return "\n".join(lines)
