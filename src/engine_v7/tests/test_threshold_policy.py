"""
Unit tests for threshold_policy.py
"""

import pytest
from core.measure.metrics.threshold_policy import (
    _calculate_blur_penalty,
    _calculate_illumination_penalty,
    _calculate_offset_penalty,
    _determine_quality_level,
    classify_inkness,
    get_adaptive_threshold,
    get_threshold_summary,
    should_retake,
    validate_threshold_policy,
)


class TestCalculatePenalties:
    """개별 penalty 계산 함수 테스트"""

    def test_blur_penalty_good(self):
        """Sharpness 좋음 (≥500)"""
        penalty = _calculate_blur_penalty(550.0)
        assert penalty == 0.0

    def test_blur_penalty_medium(self):
        """Sharpness 보통 (200~500)"""
        penalty = _calculate_blur_penalty(350.0)
        assert penalty == 0.03

    def test_blur_penalty_poor(self):
        """Sharpness 나쁨 (<200)"""
        penalty = _calculate_blur_penalty(150.0)
        assert penalty == 0.08

    def test_illumination_penalty_good(self):
        """조명 좋음 (≤0.05)"""
        penalty = _calculate_illumination_penalty(0.03)
        assert penalty == 0.0

    def test_illumination_penalty_medium(self):
        """조명 보통 (0.05~0.15)"""
        penalty = _calculate_illumination_penalty(0.10)
        assert penalty == 0.03

    def test_illumination_penalty_poor(self):
        """조명 나쁨 (>0.15)"""
        penalty = _calculate_illumination_penalty(0.20)
        assert penalty == 0.05

    def test_offset_penalty_good(self):
        """중심 정렬 좋음 (≤1.0)"""
        penalty = _calculate_offset_penalty(0.8)
        assert penalty == 0.0

    def test_offset_penalty_medium(self):
        """중심 정렬 보통 (1.0~3.0)"""
        penalty = _calculate_offset_penalty(2.0)
        assert penalty == 0.02

    def test_offset_penalty_poor(self):
        """중심 정렬 나쁨 (>3.0)"""
        penalty = _calculate_offset_penalty(4.0)
        assert penalty == 0.05


class TestDetermineQualityLevel:
    """품질 레벨 판단 테스트"""

    def test_quality_good(self):
        """모든 지표 좋음"""
        level = _determine_quality_level(sharpness=550.0, illumination=0.03, offset=0.8)
        assert level == "good"

    def test_quality_medium(self):
        """일부 지표 보통"""
        level = _determine_quality_level(sharpness=350.0, illumination=0.03, offset=0.8)  # medium
        assert level == "medium"

    def test_quality_poor(self):
        """1개 지표 나쁨"""
        level = _determine_quality_level(sharpness=150.0, illumination=0.03, offset=0.8)  # poor
        assert level == "poor"

    def test_quality_very_poor(self):
        """2개 이상 지표 나쁨"""
        level = _determine_quality_level(sharpness=150.0, illumination=0.20, offset=0.8)  # poor  # poor
        assert level == "very_poor"


class TestGetAdaptiveThreshold:
    """get_adaptive_threshold 함수 테스트"""

    def test_default_no_gate_scores(self):
        """기본 설정 (Gate 점수 없음)"""
        result = get_adaptive_threshold()

        assert result["ink_threshold"] == 0.70
        assert result["review_lower"] == 0.55
        assert result["gap_upper"] == 0.50
        assert result["adjustment"] == 0.0
        assert result["reason"] == "no_adjustment"
        assert result["quality_level"] == "good"

    def test_good_quality_no_adjustment(self):
        """좋은 품질 → adjustment 없음"""
        gate_scores = {"sharpness_score": 550.0, "illumination_asymmetry": 0.03, "center_offset_mm": 0.8}

        result = get_adaptive_threshold(gate_scores=gate_scores)

        assert result["adjustment"] == 0.0
        assert result["ink_threshold"] == 0.70
        assert result["quality_level"] == "good"

    def test_medium_quality_small_adjustment(self):
        """보통 품질 → 작은 adjustment"""
        gate_scores = {
            "sharpness_score": 350.0,  # medium (+0.03)
            "illumination_asymmetry": 0.10,  # medium (+0.03)
            "center_offset_mm": 2.0,  # medium (+0.02)
        }

        result = get_adaptive_threshold(gate_scores=gate_scores)

        # 0.03 + 0.03 + 0.02 = 0.08
        assert result["adjustment"] == 0.08
        assert result["ink_threshold"] == 0.78
        assert result["review_lower"] == 0.63
        assert result["gap_upper"] == 0.58
        assert result["quality_level"] == "medium"
        assert "low_sharpness" in result["reason"]
        assert "illumination_issue" in result["reason"]

    def test_poor_quality_large_adjustment(self):
        """나쁜 품질 → 큰 adjustment"""
        gate_scores = {
            "sharpness_score": 150.0,  # poor (+0.08)
            "illumination_asymmetry": 0.20,  # poor (+0.05)
            "center_offset_mm": 4.0,  # poor (+0.05)
        }

        result = get_adaptive_threshold(gate_scores=gate_scores)

        # 0.08 + 0.05 + 0.05 = 0.18 → cap at 0.10
        assert result["adjustment"] == 0.10
        assert result["ink_threshold"] == 0.80
        assert result["quality_level"] == "very_poor"

    def test_adjustment_disabled(self):
        """보정 비활성화"""
        gate_scores = {"sharpness_score": 150.0, "illumination_asymmetry": 0.20, "center_offset_mm": 4.0}

        result = get_adaptive_threshold(gate_scores=gate_scores, enable_adjustment=False)

        assert result["adjustment"] == 0.0
        assert result["ink_threshold"] == 0.70

    def test_custom_base_threshold(self):
        """커스텀 base threshold"""
        result = get_adaptive_threshold(base_threshold=0.75)

        assert result["ink_threshold"] == 0.75
        assert result["review_lower"] == 0.60
        assert result["gap_upper"] == 0.55

    def test_custom_review_window(self):
        """커스텀 review window"""
        result = get_adaptive_threshold(review_window=0.20)

        assert result["ink_threshold"] == 0.70
        assert result["review_lower"] == 0.50  # 0.70 - 0.20


class TestClassifyInkness:
    """classify_inkness 함수 테스트"""

    def test_classify_ink(self):
        """INK 구간"""
        thresholds = {"ink_threshold": 0.70, "review_lower": 0.55, "gap_upper": 0.50}

        assert classify_inkness(0.85, thresholds) == "ink"
        assert classify_inkness(0.70, thresholds) == "ink"  # 경계값

    def test_classify_review(self):
        """REVIEW 구간"""
        thresholds = {"ink_threshold": 0.70, "review_lower": 0.55, "gap_upper": 0.50}

        assert classify_inkness(0.65, thresholds) == "review"
        assert classify_inkness(0.55, thresholds) == "review"  # 경계값

    def test_classify_gap(self):
        """GAP 구간"""
        thresholds = {"ink_threshold": 0.70, "review_lower": 0.55, "gap_upper": 0.50}

        assert classify_inkness(0.45, thresholds) == "gap"
        assert classify_inkness(0.10, thresholds) == "gap"

    def test_classify_with_adjustment(self):
        """Adjustment 적용된 threshold"""
        thresholds = {"ink_threshold": 0.80, "review_lower": 0.65, "gap_upper": 0.60}  # +0.10 보정

        # 0.75는 원래 INK였지만, adjustment로 REVIEW가 됨
        assert classify_inkness(0.75, thresholds) == "review"


class TestValidateThresholdPolicy:
    """validate_threshold_policy 함수 테스트"""

    def test_valid_policy(self):
        """정상 policy"""
        thresholds = {"ink_threshold": 0.70, "review_lower": 0.55, "gap_upper": 0.50}

        valid = validate_threshold_policy(thresholds)
        assert valid is True

    def test_invalid_order(self):
        """순서 잘못됨"""
        thresholds = {"ink_threshold": 0.50, "review_lower": 0.60, "gap_upper": 0.45}  # ink보다 높음 (잘못됨)

        with pytest.raises(ValueError, match="Invalid threshold order"):
            validate_threshold_policy(thresholds)

    def test_out_of_range(self):
        """범위 벗어남"""
        thresholds = {"ink_threshold": 1.10, "review_lower": 0.95, "gap_upper": 0.90}  # >1.0

        with pytest.raises(ValueError, match="out of range"):
            validate_threshold_policy(thresholds)

    def test_narrow_review_window_warning(self):
        """REVIEW 구간 너무 좁음 (경고)"""
        thresholds = {"ink_threshold": 0.70, "review_lower": 0.68, "gap_upper": 0.63}  # 폭 0.02 (너무 좁음)

        with pytest.warns(UserWarning, match="REVIEW window too narrow"):
            validate_threshold_policy(thresholds)


class TestShouldRetake:
    """should_retake 함수 테스트"""

    def test_good_quality_no_retake(self):
        """좋은 품질 → 재촬영 불필요"""
        thresholds = {"quality_level": "good", "adjustment": 0.0}

        assert should_retake(thresholds) is False

    def test_medium_quality_no_retake(self):
        """보통 품질 → 재촬영 불필요"""
        thresholds = {"quality_level": "medium", "adjustment": 0.05}

        assert should_retake(thresholds) is False

    def test_very_poor_quality_retake(self):
        """매우 나쁜 품질 → 재촬영 필요"""
        thresholds = {"quality_level": "very_poor", "adjustment": 0.10}

        assert should_retake(thresholds) is True

    def test_max_adjustment_retake(self):
        """최대 adjustment → 재촬영 필요"""
        thresholds = {"quality_level": "poor", "adjustment": 0.10}

        assert should_retake(thresholds) is True


class TestGetThresholdSummary:
    """get_threshold_summary 함수 테스트"""

    def test_summary_format(self):
        """요약 문자열 포맷 검증"""
        thresholds = {
            "quality_level": "medium",
            "adjustment": 0.05,
            "reason": "low_sharpness(+0.03), illumination_issue(+0.03)",
            "ink_threshold": 0.75,
            "review_lower": 0.60,
            "gap_upper": 0.55,
            "gate_scores": {"sharpness_score": 350.0, "illumination_asymmetry": 0.10, "center_offset_mm": 2.0},
        }

        summary = get_threshold_summary(thresholds)

        # 주요 내용 포함 확인
        assert "MEDIUM" in summary
        assert "0.75" in summary  # ink_threshold
        assert "low_sharpness" in summary
        assert "350.0" in summary  # sharpness_score


class TestEndToEndScenarios:
    """종단간 시나리오 테스트"""

    def test_gradient_product_good_quality(self):
        """그라데이션 제품, 좋은 품질"""
        gate_scores = {"sharpness_score": 600.0, "illumination_asymmetry": 0.02, "center_offset_mm": 0.5}

        thresholds = get_adaptive_threshold(gate_scores=gate_scores)

        # Adjustment 없음
        assert thresholds["adjustment"] == 0.0
        assert thresholds["quality_level"] == "good"

        # 그라데이션 밝은 부분: 0.65 (REVIEW)
        assert classify_inkness(0.65, thresholds) == "review"

        # 그라데이션 어두운 부분: 0.82 (INK)
        assert classify_inkness(0.82, thresholds) == "ink"

        # 재촬영 불필요
        assert should_retake(thresholds) is False

    def test_reflection_case_poor_quality(self):
        """반사 케이스, 나쁜 품질"""
        gate_scores = {
            "sharpness_score": 180.0,  # poor
            "illumination_asymmetry": 0.18,  # poor
            "center_offset_mm": 1.5,  # medium
        }

        thresholds = get_adaptive_threshold(gate_scores=gate_scores)

        # 큰 adjustment
        assert thresholds["adjustment"] == 0.10  # capped
        assert thresholds["quality_level"] == "very_poor"
        assert thresholds["ink_threshold"] == 0.80

        # 반사 (원래 0.62 → INK였음)
        # Adjustment로 REVIEW로 강등됨
        assert classify_inkness(0.62, thresholds) == "gap"
        assert classify_inkness(0.75, thresholds) == "review"

        # 재촬영 권고
        assert should_retake(thresholds) is True

    def test_normal_product_medium_quality(self):
        """일반 제품, 보통 품질"""
        gate_scores = {
            "sharpness_score": 400.0,  # medium
            "illumination_asymmetry": 0.08,  # medium
            "center_offset_mm": 0.9,  # good
        }

        thresholds = get_adaptive_threshold(gate_scores=gate_scores)

        # 작은 adjustment
        assert thresholds["adjustment"] == 0.06  # 0.03 + 0.03
        assert thresholds["quality_level"] == "medium"
        assert thresholds["ink_threshold"] == 0.76

        # 잉크: 0.80 (INK)
        assert classify_inkness(0.80, thresholds) == "ink"

        # Gap: 0.40 (GAP)
        assert classify_inkness(0.40, thresholds) == "gap"

        # 재촬영 불필요
        assert should_retake(thresholds) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
