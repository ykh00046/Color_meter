"""
Unit tests for uncertainty schema and decision engine v2
"""

import pytest

from core.decision.decision_engine import calculate_decision_v2, should_retake_from_threshold
from core.decision.uncertainty import (
    EvidenceBuilder,
    UncertaintyBuilder,
    calculate_confidence,
    create_evidence_summary,
    create_uncertainty_summary,
    extract_reason_codes,
)


class TestUncertaintyBuilder:
    """UncertaintyBuilder 클래스 테스트"""

    def test_empty_builder(self):
        """빈 builder"""
        builder = UncertaintyBuilder()
        result = builder.build()

        assert result["level"] == "low"
        assert result["total_score"] == 0.0
        assert len(result["factors"]) == 0

    def test_add_gate_factor(self):
        """Gate factor 추가"""
        builder = UncertaintyBuilder()
        builder.add_gate_factor(
            code="BLUR_TOO_LOW", severity="high", impact=0.35, message="Sharpness too low", sharpness_score=150.0
        )

        result = builder.build()

        assert result["level"] == "medium"  # 0.35 → medium
        assert result["total_score"] == 0.35
        assert len(result["factors"]) == 1

        factor = result["factors"][0]
        assert factor["category"] == "gate"
        assert factor["code"] == "BLUR_TOO_LOW"
        assert factor["severity"] == "high"
        assert factor["impact"] == 0.35
        assert factor["details"]["sharpness_score"] == 150.0

    def test_multiple_factors(self):
        """여러 factor 추가"""
        builder = UncertaintyBuilder()
        builder.add_gate_factor(code="BLUR_TOO_LOW", severity="high", impact=0.35, message="Blur")
        builder.add_sampling_factor(code="SAMPLING_SPARSE", severity="medium", impact=0.25, message="Sparse")

        result = builder.build()

        # 0.35 + 0.25 = 0.60 → high
        assert result["level"] == "high"
        assert result["total_score"] == 0.60
        assert len(result["factors"]) == 2

    def test_critical_level(self):
        """Critical level (≥0.80)"""
        builder = UncertaintyBuilder()
        builder.add_gate_factor(code="BLUR_CRITICAL", severity="critical", impact=0.80, message="Critical blur")

        result = builder.build()

        assert result["level"] == "critical"
        assert result["total_score"] == 0.80


class TestEvidenceBuilder:
    """EvidenceBuilder 클래스 테스트"""

    def test_empty_builder(self):
        """빈 builder"""
        builder = EvidenceBuilder()
        result = builder.build()

        assert len(result["positive"]) == 0
        assert len(result["negative"]) == 0
        assert len(result["neutral"]) == 0

    def test_add_positive_evidence(self):
        """Positive evidence 추가"""
        builder = EvidenceBuilder()
        builder.add_positive(
            type="ink", code="EXCESSIVE_INK_COUNT", strength="strong", weight=0.80, value=5, message="5 inks detected"
        )

        result = builder.build()

        assert len(result["positive"]) == 1
        ev = result["positive"][0]
        assert ev["type"] == "ink"
        assert ev["code"] == "EXCESSIVE_INK_COUNT"
        assert ev["strength"] == "strong"
        assert ev["weight"] == 0.80
        assert ev["value"] == 5

    def test_add_negative_evidence(self):
        """Negative evidence 추가"""
        builder = EvidenceBuilder()
        builder.add_negative(
            type="signature",
            code="SIGNATURE_MATCH_EXCELLENT",
            strength="strong",
            weight=0.75,
            value=0.95,
            message="Excellent match",
        )

        result = builder.build()

        assert len(result["negative"]) == 1
        ev = result["negative"][0]
        assert ev["code"] == "SIGNATURE_MATCH_EXCELLENT"

    def test_add_neutral_evidence(self):
        """Neutral evidence 추가"""
        builder = EvidenceBuilder()
        builder.add_neutral(type="sampling", code="AUTO_K_SUGGESTS_3", value=3, message="Auto-k suggests 3")

        result = builder.build()

        assert len(result["neutral"]) == 1
        ev = result["neutral"][0]
        assert ev["strength"] == "neutral"
        assert ev["weight"] == 0.0


class TestCalculateConfidence:
    """calculate_confidence 함수 테스트"""

    def test_low_uncertainty_balanced_evidence(self):
        """낮은 불확실성, 균형 증거 → 중간 확신"""
        confidence = calculate_confidence(
            uncertainty_score=0.10, evidence_positive_weight=0.50, evidence_negative_weight=0.50
        )

        # base_confidence = 1.0 - 0.10 = 0.90
        # imbalance = 0 → evidence_boost = 0
        # total = 0.90
        assert confidence == pytest.approx(0.90, abs=0.01)

    def test_low_uncertainty_strong_evidence(self):
        """낮은 불확실성, 강한 증거 → 높은 확신"""
        confidence = calculate_confidence(
            uncertainty_score=0.10, evidence_positive_weight=0.80, evidence_negative_weight=0.10
        )

        # base_confidence = 0.90
        # imbalance = 0.70 / 0.90 = 0.778
        # evidence_boost = 0.778 * 0.30 = 0.233
        # total = 0.90 + 0.233 = 1.133 → cap at 1.0
        assert confidence >= 0.95

    def test_high_uncertainty_weak_evidence(self):
        """높은 불확실성, 약한 증거 → 낮은 확신"""
        confidence = calculate_confidence(
            uncertainty_score=0.70, evidence_positive_weight=0.20, evidence_negative_weight=0.25
        )

        # base_confidence = 1.0 - 0.70 = 0.30
        # imbalance 낮음 → evidence_boost 작음
        assert confidence < 0.50


class TestExtractReasonCodes:
    """extract_reason_codes 함수 테스트"""

    def test_retake_codes(self):
        """RETAKE reason codes"""
        uncertainty = {
            "level": "critical",
            "factors": [
                {"code": "BLUR_CRITICAL", "severity": "critical"},
                {"code": "ILLUMINATION_SEVERE", "severity": "high"},
            ],
        }
        evidence = {"positive": [], "negative": [], "neutral": []}

        codes = extract_reason_codes(uncertainty, evidence, "RETAKE")

        assert "RETAKE_REQUIRED" in codes
        assert "BLUR_CRITICAL" in codes

    def test_hold_codes(self):
        """HOLD reason codes"""
        uncertainty = {"level": "high", "factors": []}
        evidence = {"positive": [], "negative": [], "neutral": []}

        codes = extract_reason_codes(uncertainty, evidence, "HOLD")

        assert "UNCERTAINTY_HIGH" in codes
        assert "MANUAL_REVIEW_REQUIRED" in codes

    def test_ng_color_codes(self):
        """NG_COLOR reason codes"""
        uncertainty = {"level": "low", "factors": []}
        evidence = {
            "positive": [
                {"code": "EXCESSIVE_INK_COUNT", "strength": "strong"},
                {"code": "HIGH_DELTAEE", "strength": "strong"},
            ],
            "negative": [],
            "neutral": [],
        }

        codes = extract_reason_codes(uncertainty, evidence, "NG_COLOR")

        assert "EXCESSIVE_INK_COUNT" in codes
        assert "HIGH_DELTAEE" in codes


class TestCalculateDecisionV2:
    """calculate_decision_v2 함수 테스트"""

    def test_ok_decision(self):
        """OK 판정"""
        uncertainty = (
            UncertaintyBuilder()
            .add_gate_factor(code="BLUR_SLIGHTLY_LOW", severity="low", impact=0.10, message="Minor blur")
            .build()
        )

        evidence = (
            EvidenceBuilder()
            .add_negative(
                type="ink", code="INK_COUNT_EXACT", strength="strong", weight=0.70, value=2, message="Exact count"
            )
            .build()
        )

        decision = calculate_decision_v2(uncertainty, evidence)

        assert decision["result"] == "OK"
        assert decision["confidence"] > 0.80

    def test_retake_decision_critical(self):
        """RETAKE 판정 (critical uncertainty)"""
        uncertainty = (
            UncertaintyBuilder()
            .add_gate_factor(code="BLUR_CRITICAL", severity="critical", impact=0.80, message="Critical blur")
            .build()
        )

        evidence = EvidenceBuilder().build()

        decision = calculate_decision_v2(uncertainty, evidence)

        assert decision["result"] == "RETAKE"
        assert "RETAKE_REQUIRED" in decision["reason_codes"]

    def test_hold_decision(self):
        """HOLD 판정 (high uncertainty)"""
        uncertainty = (
            UncertaintyBuilder()
            .add_sampling_factor("SAMPLING_SPARSE", "medium", 0.25, "Sparse")
            .add_signature_factor("SIGNATURE_AMBIGUOUS", "medium", 0.30, "Ambiguous")
            .build()
        )

        evidence = EvidenceBuilder().build()

        decision = calculate_decision_v2(uncertainty, evidence)

        # 0.25 + 0.30 = 0.55 → high
        assert decision["result"] == "HOLD"
        assert "UNCERTAINTY_HIGH" in decision["reason_codes"]

    def test_ng_color_decision(self):
        """NG_COLOR 판정 (strong positive evidence)"""
        uncertainty = UncertaintyBuilder().build()  # 낮은 불확실성

        evidence = (
            EvidenceBuilder()
            .add_positive("ink", "EXCESSIVE_INK_COUNT", "strong", 0.80, 5, "5 inks")
            .add_positive("signature", "HIGH_DELTAEE", "strong", 0.75, 0.25, "High ΔE")
            .build()
        )

        decision = calculate_decision_v2(uncertainty, evidence)

        assert decision["result"] == "NG_COLOR"
        assert decision["confidence"] > 0.70

    def test_with_should_retake_func(self):
        """should_retake_func 사용"""
        uncertainty = UncertaintyBuilder().build()
        evidence = EvidenceBuilder().build()

        # Gate 품질 매우 나쁨
        gate_scores = {"sharpness_score": 100.0, "illumination_asymmetry": 0.25, "center_offset_mm": 5.0}

        def custom_retake(scores):
            return scores.get("sharpness_score", 999) < 150

        decision = calculate_decision_v2(
            uncertainty, evidence, gate_scores=gate_scores, should_retake_func=custom_retake
        )

        assert decision["result"] == "RETAKE"


class TestShouldRetakeFromThreshold:
    """should_retake_from_threshold 함수 테스트"""

    def test_very_poor_quality(self):
        """very_poor → retake"""
        threshold_result = {"quality_level": "very_poor", "adjustment": 0.10}

        assert should_retake_from_threshold(threshold_result) is True

    def test_max_adjustment(self):
        """max adjustment → retake"""
        threshold_result = {"quality_level": "poor", "adjustment": 0.10}

        assert should_retake_from_threshold(threshold_result) is True

    def test_good_quality(self):
        """good → no retake"""
        threshold_result = {"quality_level": "good", "adjustment": 0.0}

        assert should_retake_from_threshold(threshold_result) is False


class TestSummaryFunctions:
    """요약 문자열 생성 테스트"""

    def test_uncertainty_summary(self):
        """불확실성 요약"""
        uncertainty = UncertaintyBuilder().add_gate_factor("BLUR_TOO_LOW", "high", 0.35, "Sharpness 150").build()

        summary = create_uncertainty_summary(uncertainty)

        assert "MEDIUM" in summary
        assert "0.35" in summary
        assert "BLUR_TOO_LOW" in summary

    def test_evidence_summary(self):
        """증거 요약"""
        evidence = (
            EvidenceBuilder()
            .add_positive("ink", "EXCESSIVE_INK_COUNT", "strong", 0.80, 5, "5 inks")
            .add_negative("signature", "SIGNATURE_MATCH", "moderate", 0.50, 0.88, "Good")
            .build()
        )

        summary = create_evidence_summary(evidence)

        assert "1 positive" in summary
        assert "1 negative" in summary
        assert "EXCESSIVE_INK_COUNT" in summary


class TestEndToEndScenarios:
    """종단간 시나리오 테스트"""

    def test_gradient_product_ok(self):
        """그라데이션 제품, 정상"""
        # 낮은 불확실성
        uncertainty = UncertaintyBuilder().build()

        # Strong OK 증거
        evidence = (
            EvidenceBuilder()
            .add_negative("ink", "INK_COUNT_EXACT", "strong", 0.70, 2, "2 inks")
            .add_negative("signature", "SIGNATURE_MATCH_EXCELLENT", "strong", 0.75, 0.93, "Excellent")
            .add_neutral("sampling", "GRADIENT_DETECTED", 1, "Gradient detected")
            .build()
        )

        decision = calculate_decision_v2(uncertainty, evidence)

        assert decision["result"] == "OK"
        assert decision["confidence"] > 0.85

    def test_poor_quality_retake(self):
        """품질 나쁨 → RETAKE"""
        # Critical 불확실성
        uncertainty = UncertaintyBuilder().add_gate_factor("BLUR_CRITICAL", "critical", 0.80, "Blur 100").build()

        evidence = EvidenceBuilder().build()

        decision = calculate_decision_v2(uncertainty, evidence)

        assert decision["result"] == "RETAKE"
        assert decision["confidence"] < 0.30

    def test_ambiguous_hold(self):
        """애매한 경우 → HOLD"""
        # High 불확실성
        uncertainty = (
            UncertaintyBuilder()
            .add_sampling_factor("SAMPLING_SPARSE", "medium", 0.25, "Sparse")
            .add_segmentation_factor("SEGMENTATION_OVERLAP", "medium", 0.30, "Overlap")
            .build()
        )

        # 약한 증거들
        evidence = (
            EvidenceBuilder()
            .add_positive("spatial", "ANGULAR_FRAGMENTATION", "moderate", 0.40, 0.38, "Fragmented")
            .add_negative("ink", "INK_COUNT_EXACT", "strong", 0.70, 2, "Exact")
            .build()
        )

        decision = calculate_decision_v2(uncertainty, evidence)

        # 0.25 + 0.30 = 0.55 → high → HOLD
        assert decision["result"] == "HOLD"

    def test_clear_ng(self):
        """명확한 불량 → NG_COLOR"""
        # 낮은 불확실성
        uncertainty = UncertaintyBuilder().add_gate_factor("BLUR_SLIGHTLY_LOW", "low", 0.10, "Blur 280").build()

        # Strong NG 증거
        evidence = (
            EvidenceBuilder()
            .add_positive("ink", "EXCESSIVE_INK_COUNT", "strong", 0.80, 6, "6 inks")
            .add_positive("signature", "HIGH_DELTAEE", "strong", 0.75, 0.30, "ΔE 0.30")
            .build()
        )

        decision = calculate_decision_v2(uncertainty, evidence)

        assert decision["result"] == "NG_COLOR"
        assert decision["confidence"] > 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
