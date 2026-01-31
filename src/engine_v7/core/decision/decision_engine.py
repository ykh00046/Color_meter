from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..types import AnomalyResult, GateResult, SignatureResult
from .uncertainty import calculate_confidence, extract_reason_codes


def decide(gate: GateResult, sig: SignatureResult | None, anom: AnomalyResult | None) -> Tuple[str, List[str]]:
    """
    Human-like decision rule (Legacy V1):
      1) Gate fail -> RETAKE
      2) Anomaly fail -> NG_PATTERN
      3) Signature fail -> NG_COLOR
      4) else OK
    """
    if not gate.passed:
        return "RETAKE", gate.reasons
    if anom is not None and not anom.passed:
        return "NG_PATTERN", anom.reasons
    if sig is not None and not sig.passed:
        return "NG_COLOR", sig.reasons
    return "OK", []


def calculate_decision_v2(
    uncertainty: Dict[str, Any],
    evidence: Dict[str, Any],
    gate_scores: Optional[Dict[str, float]] = None,
    should_retake_func: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Uncertainty & Evidence 기반 의사결정 (V2)

    Args:
        uncertainty: UncertaintyBuilder.build() 결과
        evidence: EvidenceBuilder.build() 결과
        gate_scores: Gate 품질 지표 (optional)
        should_retake_func: 재촬영 판단 함수 (optional)

    Returns:
        {
            "result": str,  # "OK" | "RETAKE" | "HOLD" | "NG_COLOR"
            "confidence": float,  # 0~1
            "uncertainty": dict,
            "evidence": dict,
            "reason_codes": List[str]
        }

    분기 로직:
    1. uncertainty.level == "critical" OR should_retake_func() → RETAKE
    2. uncertainty.level == "high" → HOLD
    3. positive_evidence >> negative_evidence → NG_COLOR
    4. else → OK
    """
    uncertainty_level = uncertainty["level"]
    uncertainty_score = uncertainty["total_score"]

    # Evidence 가중치 합산
    positive_weight = sum(e["weight"] for e in evidence["positive"])
    negative_weight = sum(e["weight"] for e in evidence["negative"])

    # 1. RETAKE 판단
    if uncertainty_level == "critical":
        result = "RETAKE"

    # 재촬영 함수가 제공된 경우 추가 체크
    elif should_retake_func is not None and should_retake_func(gate_scores or {}):
        result = "RETAKE"

    # 2. HOLD 판단 (높은 불확실성)
    elif uncertainty_level == "high":
        result = "HOLD"

    # 3. NG_COLOR 판단 (강한 불량 증거)
    elif positive_weight > negative_weight:
        # Positive 증거가 더 많으면 NG
        result = "NG_COLOR"

    # 4. OK 판단
    else:
        result = "OK"

    # Confidence 계산
    confidence = calculate_confidence(
        uncertainty_score=uncertainty_score,
        evidence_positive_weight=positive_weight,
        evidence_negative_weight=negative_weight,
    )

    # Reason codes 추출
    reason_codes = extract_reason_codes(uncertainty, evidence, result)

    return {
        "result": result,
        "confidence": round(confidence, 3),
        "uncertainty": uncertainty,
        "evidence": evidence,
        "reason_codes": reason_codes,
    }


def should_retake_from_threshold(threshold_result: Dict[str, Any]) -> bool:
    """
    Threshold policy 결과로부터 재촬영 여부 판단

    Args:
        threshold_result: get_adaptive_threshold() 결과

    Returns:
        True if retake recommended
    """
    # threshold_policy.should_retake()와 동일 로직
    quality_level = threshold_result.get("quality_level", "good")
    adjustment = threshold_result.get("adjustment", 0.0)

    if quality_level == "very_poor":
        return True

    if adjustment >= 0.10:
        return True

    return False
