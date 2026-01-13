"""
Unit tests for ColorEvaluator module
"""

from datetime import datetime

import numpy as np
import pytest

from src.core.color_evaluator import ColorEvaluationError, ColorEvaluator
from src.schemas.inspection import InspectionResult, Zone, ZoneResult

# ================================================================
# Fixtures
# ================================================================


@pytest.fixture
def sample_sku_config():
    """샘플 SKU 기준값"""
    return {
        "sku_code": "SKU_001",
        "default_threshold": 3.0,
        "zones": {
            "A": {"L": 50.0, "a": 20.0, "b": 10.0, "threshold": 4.0},
            "B": {"L": 60.0, "a": 10.0, "b": 30.0, "threshold": 3.5},
            "C": {"L": 70.0, "a": 5.0, "b": 50.0, "threshold": 4.5},
        },
    }


@pytest.fixture
def good_zones():
    """정상 범위의 zone 리스트 (OK 예상)"""
    return [
        Zone(
            name="A",
            r_start=1.0,
            r_end=0.6,
            mean_L=50.5,
            mean_a=20.5,
            mean_b=10.5,  # 기준값에 가까움
            std_L=1.0,
            std_a=0.5,
            std_b=0.5,
            zone_type="pure",
            pixel_count=5000,  # PHASE7: Add pixel count to avoid warnings
        ),
        Zone(
            name="B",
            r_start=0.6,
            r_end=0.3,
            mean_L=60.2,
            mean_a=10.3,
            mean_b=30.1,  # 기준값에 가까움
            std_L=1.0,
            std_a=0.5,
            std_b=0.5,
            zone_type="pure",
            pixel_count=5000,  # PHASE7: Add pixel count to avoid warnings
        ),
        Zone(
            name="C",
            r_start=0.3,
            r_end=0.0,
            mean_L=70.1,
            mean_a=5.1,
            mean_b=50.2,  # 기준값에 가까움
            std_L=1.0,
            std_a=0.5,
            std_b=0.5,
            zone_type="pure",
            pixel_count=5000,  # PHASE7: Add pixel count to avoid warnings
        ),
    ]


@pytest.fixture
def bad_zones():
    """불량 zone 리스트 (NG 예상)"""
    return [
        Zone(
            name="A",
            r_start=1.0,
            r_end=0.6,
            mean_L=50.0,
            mean_a=20.0,
            mean_b=10.0,  # OK
            std_L=1.0,
            std_a=0.5,
            std_b=0.5,
            zone_type="pure",
            pixel_count=5000,  # PHASE7: Add pixel count to avoid warnings
        ),
        Zone(
            name="B",
            r_start=0.6,
            r_end=0.3,
            mean_L=70.0,
            mean_a=30.0,
            mean_b=10.0,  # 크게 벗어남 (NG)
            std_L=1.0,
            std_a=0.5,
            std_b=0.5,
            zone_type="pure",
            pixel_count=5000,  # PHASE7: Add pixel count to avoid warnings
        ),
        Zone(
            name="C",
            r_start=0.3,
            r_end=0.0,
            mean_L=70.0,
            mean_a=5.0,
            mean_b=50.0,  # OK
            std_L=1.0,
            std_a=0.5,
            std_b=0.5,
            zone_type="pure",
            pixel_count=5000,  # PHASE7: Add pixel count to avoid warnings
        ),
    ]


# ================================================================
# Test Cases
# ================================================================


def test_zone_result_creation():
    """ZoneResult 데이터클래스 생성"""
    zr = ZoneResult(
        zone_name="A",
        measured_lab=(50.0, 20.0, 10.0),
        target_lab=(50.5, 20.5, 10.5),
        delta_e=1.2,
        threshold=3.0,
        is_ok=True,
    )

    assert zr.zone_name == "A"
    assert zr.measured_lab == (50.0, 20.0, 10.0)
    assert zr.target_lab == (50.5, 20.5, 10.5)
    assert zr.delta_e == 1.2
    assert zr.threshold == 3.0
    assert zr.is_ok is True


def test_inspection_result_creation():
    """InspectionResult 데이터클래스 생성"""
    now = datetime.now()
    result = InspectionResult(
        sku="SKU_001",
        timestamp=now,
        judgment="OK",
        overall_delta_e=1.5,
        zone_results=[],
        ng_reasons=[],
        confidence=0.95,
    )

    assert result.sku == "SKU_001"
    assert result.judgment == "OK"
    assert result.overall_delta_e == 1.5
    assert result.confidence == 0.95


def test_color_evaluator_creation():
    """ColorEvaluator 객체 생성"""
    evaluator = ColorEvaluator()
    assert isinstance(evaluator, ColorEvaluator)


def test_color_evaluator_with_config(sample_sku_config):
    """SKU config와 함께 생성"""
    evaluator = ColorEvaluator(sku_config=sample_sku_config)
    assert evaluator.sku_config == sample_sku_config


def test_calculate_delta_e_cie2000():
    """ΔE2000 계산"""
    evaluator = ColorEvaluator()

    lab1 = (50.0, 20.0, 10.0)
    lab2 = (50.0, 20.0, 10.0)

    # 동일한 색상 → ΔE = 0
    de = evaluator.calculate_delta_e(lab1, lab2, method="cie2000")
    assert de == pytest.approx(0.0, abs=0.01)

    # 다른 색상
    lab3 = (60.0, 10.0, 30.0)
    de2 = evaluator.calculate_delta_e(lab1, lab3, method="cie2000")
    assert de2 > 0.0


def test_calculate_delta_e_cie1994():
    """ΔE1994 계산"""
    evaluator = ColorEvaluator()

    lab1 = (50.0, 20.0, 10.0)
    lab2 = (60.0, 10.0, 30.0)

    de = evaluator.calculate_delta_e(lab1, lab2, method="cie1994")
    assert de > 0.0


def test_calculate_delta_e_cie1976():
    """ΔE1976 계산"""
    evaluator = ColorEvaluator()

    lab1 = (50.0, 20.0, 10.0)
    lab2 = (60.0, 10.0, 30.0)

    de = evaluator.calculate_delta_e(lab1, lab2, method="cie1976")
    assert de > 0.0


def test_calculate_delta_e_invalid_method():
    """잘못된 method"""
    evaluator = ColorEvaluator()

    lab1 = (50.0, 20.0, 10.0)
    lab2 = (60.0, 10.0, 30.0)

    with pytest.raises(ColorEvaluationError):
        evaluator.calculate_delta_e(lab1, lab2, method="invalid")


def test_evaluate_ok_case(good_zones, sample_sku_config):
    """정상 제품 평가 (OK)"""
    evaluator = ColorEvaluator(sku_config=sample_sku_config)

    result = evaluator.evaluate(good_zones, "SKU_001")

    assert result.sku == "SKU_001"
    assert result.judgment == "OK"
    assert len(result.zone_results) == 3
    assert all(zr.is_ok for zr in result.zone_results)
    assert len(result.ng_reasons) == 0
    assert result.confidence > 0.0


def test_evaluate_ng_case(bad_zones, sample_sku_config):
    """불량 제품 평가 (NG)"""
    evaluator = ColorEvaluator(sku_config=sample_sku_config)

    result = evaluator.evaluate(bad_zones, "SKU_001")

    assert result.sku == "SKU_001"
    assert result.judgment == "NG"
    assert len(result.zone_results) == 3
    # Zone B가 NG
    assert not all(zr.is_ok for zr in result.zone_results)
    assert len(result.ng_reasons) > 0
    assert "Zone B" in result.ng_reasons[0]


def test_evaluate_missing_sku_config(good_zones):
    """SKU 기준값 없을 때 에러"""
    evaluator = ColorEvaluator()  # No config

    with pytest.raises(ColorEvaluationError, match="기준값이 등록되지 않음"):
        evaluator.evaluate(good_zones, "SKU_001")


def test_evaluate_zone_result_fields(good_zones, sample_sku_config):
    """Zone별 결과 필드 확인"""
    evaluator = ColorEvaluator(sku_config=sample_sku_config)

    result = evaluator.evaluate(good_zones, "SKU_001")

    # 첫 번째 zone 결과 확인
    zr = result.zone_results[0]
    assert zr.zone_name == "A"
    assert len(zr.measured_lab) == 3
    assert len(zr.target_lab) == 3
    assert zr.delta_e >= 0.0
    assert zr.threshold > 0.0
    assert isinstance(zr.is_ok, bool)


def test_evaluate_overall_delta_e(good_zones, sample_sku_config):
    """전체 평균 ΔE 계산"""
    evaluator = ColorEvaluator(sku_config=sample_sku_config)

    result = evaluator.evaluate(good_zones, "SKU_001")

    # 평균 ΔE는 모든 zone ΔE의 평균
    avg_de = np.mean([zr.delta_e for zr in result.zone_results])
    assert result.overall_delta_e == pytest.approx(avg_de, abs=0.01)


def test_evaluate_confidence_calculation(good_zones, bad_zones, sample_sku_config):
    """신뢰도 계산"""
    evaluator = ColorEvaluator(sku_config=sample_sku_config)

    result_ok = evaluator.evaluate(good_zones, "SKU_001")
    result_ng = evaluator.evaluate(bad_zones, "SKU_001")

    # OK 제품이 NG 제품보다 신뢰도 높음
    assert result_ok.confidence > result_ng.confidence

    # 신뢰도 범위: 0.0 ~ 1.0
    assert 0.0 <= result_ok.confidence <= 1.0
    assert 0.0 <= result_ng.confidence <= 1.0


def test_evaluate_with_mix_check(sample_sku_config):
    """Mix zone 검증 포함 평가"""
    zones_with_mix = [
        Zone(
            name="A",
            r_start=1.0,
            r_end=0.7,
            mean_L=50.0,
            mean_a=20.0,
            mean_b=10.0,
            std_L=1.0,
            std_a=0.5,
            std_b=0.5,
            zone_type="pure",
            pixel_count=5000,  # PHASE7: Add pixel count
        ),
        Zone(
            name="A-B",
            r_start=0.7,
            r_end=0.5,
            mean_L=55.0,
            mean_a=15.0,
            mean_b=20.0,  # A와 B의 중간
            std_L=1.0,
            std_a=0.5,
            std_b=0.5,
            zone_type="mix",
            pixel_count=5000,  # PHASE7: Add pixel count
        ),
        Zone(
            name="B",
            r_start=0.5,
            r_end=0.0,
            mean_L=60.0,
            mean_a=10.0,
            mean_b=30.0,
            std_L=1.0,
            std_a=0.5,
            std_b=0.5,
            zone_type="pure",
            pixel_count=5000,  # PHASE7: Add pixel count
        ),
    ]

    # SKU config에 A-B 추가
    config_with_mix = sample_sku_config.copy()
    config_with_mix["zones"]["A-B"] = {"L": 55.0, "a": 15.0, "b": 20.0, "threshold": 3.0}

    evaluator = ColorEvaluator(sku_config=config_with_mix)

    result = evaluator.evaluate_with_mix_check(zones_with_mix, "SKU_001", check_mix_zones=True)

    assert result.sku == "SKU_001"
    # Mix zone이 정상이면 OK
    assert result.judgment == "OK"


def test_evaluate_timestamp():
    """검사 시간 기록"""
    evaluator = ColorEvaluator()

    config = {"zones": {"A": {"L": 50.0, "a": 20.0, "b": 10.0, "threshold": 3.0}}}

    zone = Zone(
        name="A",
        r_start=1.0,
        r_end=0.0,
        mean_L=50.5,
        mean_a=20.5,
        mean_b=10.5,
        std_L=1.0,
        std_a=0.5,
        std_b=0.5,
        zone_type="pure",
    )

    before = datetime.now()
    result = evaluator.evaluate([zone], "SKU_001", sku_config=config)
    after = datetime.now()

    # 타임스탬프가 before와 after 사이
    assert before <= result.timestamp <= after


def test_evaluate_missing_zone_in_config(sample_sku_config):
    """SKU config에 없는 zone은 스킵"""
    zones_with_unknown = [
        Zone(
            name="A",
            r_start=1.0,
            r_end=0.5,
            mean_L=50.0,
            mean_a=20.0,
            mean_b=10.0,
            std_L=1.0,
            std_a=0.5,
            std_b=0.5,
            zone_type="pure",
        ),
        Zone(
            name="UNKNOWN",
            r_start=0.5,
            r_end=0.0,
            mean_L=100.0,
            mean_a=50.0,
            mean_b=50.0,
            std_L=1.0,
            std_a=0.5,
            std_b=0.5,
            zone_type="pure",
        ),
    ]

    evaluator = ColorEvaluator(sku_config=sample_sku_config)

    result = evaluator.evaluate(zones_with_unknown, "SKU_001")

    # Zone A만 평가됨 (UNKNOWN 스킵)
    assert len(result.zone_results) == 1
    assert result.zone_results[0].zone_name == "A"


def test_evaluate_custom_threshold():
    """Zone별 custom threshold"""
    config = {
        "default_threshold": 3.0,
        "zones": {"A": {"L": 50.0, "a": 20.0, "b": 10.0, "delta_e_threshold": 5.0}},  # 큰 허용치
    }

    # ΔE가 3.0 초과하지만 5.0 이하
    zone_marginal = Zone(
        name="A",
        r_start=1.0,
        r_end=0.0,
        mean_L=54.0,
        mean_a=24.0,
        mean_b=14.0,  # ΔE ~3.5 예상
        std_L=1.0,
        std_a=0.5,
        std_b=0.5,
        zone_type="pure",
        pixel_count=5000,  # PHASE7: Add pixel count
    )

    evaluator = ColorEvaluator(sku_config=config)
    result = evaluator.evaluate([zone_marginal], "SKU_001")

    # Custom threshold 덕분에 OK
    assert result.judgment == "OK"


def test_calculate_confidence_edge_cases():
    """신뢰도 계산 edge case"""
    evaluator = ColorEvaluator()

    # 빈 리스트
    confidence = evaluator._calculate_confidence([])
    assert confidence == 0.0

    # ΔE = 0 (완벽)
    zr_perfect = ZoneResult(
        zone_name="A",
        measured_lab=(50.0, 20.0, 10.0),
        target_lab=(50.0, 20.0, 10.0),
        delta_e=0.0,
        threshold=3.0,
        is_ok=True,
    )
    confidence = evaluator._calculate_confidence([zr_perfect])
    assert confidence == pytest.approx(1.0, abs=0.01)

    # ΔE = threshold (경계)
    zr_boundary = ZoneResult(
        zone_name="A",
        measured_lab=(50.0, 20.0, 10.0),
        target_lab=(53.0, 20.0, 10.0),
        delta_e=3.0,
        threshold=3.0,
        is_ok=False,
    )
    confidence = evaluator._calculate_confidence([zr_boundary])
    assert confidence == pytest.approx(0.0, abs=0.01)


def test_evaluate_empty_zones(sample_sku_config):
    """빈 zone 리스트일 때 NG 처리 테스트 (Critical Bug Fix)"""
    evaluator = ColorEvaluator(sku_config=sample_sku_config)

    # 빈 zone 리스트로 evaluate 호출
    result = evaluator.evaluate(zones=[], sku="SKU_001")  # 빈 리스트

    # 빈 zone_results는 NG로 판정되어야 함
    assert result.judgment == "NG"
    assert result.overall_delta_e == 0.0
    assert result.confidence == 0.0
    assert len(result.zone_results) == 0
    assert len(result.ng_reasons) == 1
    assert "No zones matched" in result.ng_reasons[0]


def test_evaluate_no_matching_zones(sample_sku_config):
    """SKU config와 매칭되는 zone이 없을 때 NG 처리"""
    evaluator = ColorEvaluator(sku_config=sample_sku_config)

    # SKU config에 없는 zone 이름만 제공
    zones_no_match = [
        Zone(
            name="X",
            r_start=1.0,
            r_end=0.5,
            mean_L=50.0,
            mean_a=20.0,
            mean_b=10.0,
            std_L=1.0,
            std_a=0.5,
            std_b=0.5,
            zone_type="pure",
        ),
        Zone(
            name="Y",
            r_start=0.5,
            r_end=0.0,
            mean_L=60.0,
            mean_a=10.0,
            mean_b=30.0,
            std_L=1.0,
            std_a=0.5,
            std_b=0.5,
            zone_type="pure",
        ),
    ]

    result = evaluator.evaluate(zones_no_match, "SKU_001")

    # 매칭되는 zone이 없으므로 NG
    assert result.judgment == "NG"
    assert result.confidence == 0.0
    assert "No zones matched" in result.ng_reasons[0]
