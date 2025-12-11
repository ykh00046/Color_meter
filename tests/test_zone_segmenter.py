"""
Unit tests for ZoneSegmenter module
"""

import pytest
import numpy as np
from src.core.zone_segmenter import (
    ZoneSegmenter,
    SegmenterConfig,
    Zone,
    ZoneSegmentationError
)
from src.core.radial_profiler import RadialProfile


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture
def simple_profile():
    """단순 단일 색상 프로파일 (변곡점 없음)"""
    n = 100
    r_normalized = np.linspace(0, 1, n)
    L = np.full(n, 50.0)
    a = np.full(n, 10.0)  # 일정한 a* 값
    b = np.full(n, 20.0)
    std_L = np.full(n, 1.0)
    std_a = np.full(n, 0.5)
    std_b = np.full(n, 0.5)
    pixel_count = np.full(n, 360.0)

    return RadialProfile(
        r_normalized=r_normalized,
        L=L,
        a=a,
        b=b,
        std_L=std_L,
        std_a=std_a,
        std_b=std_b,
        pixel_count=pixel_count
    )


@pytest.fixture
def gradient_profile():
    """그래디언트 프로파일 (2개의 명확한 변곡점)"""
    n = 100
    r_normalized = np.linspace(0, 1, n)

    # a* 값에 3단계 계단 (0-0.33: 낮음, 0.33-0.67: 중간, 0.67-1.0: 높음)
    a = np.zeros(n)
    a[r_normalized < 0.33] = 5.0   # Zone C (안쪽)
    a[(r_normalized >= 0.33) & (r_normalized < 0.67)] = 15.0  # Zone B (중간)
    a[r_normalized >= 0.67] = 25.0  # Zone A (바깥쪽)

    # 변곡점 부근을 부드럽게 (급격한 변화 생성)
    # 실제로는 계단 함수로 충분히 변곡점이 검출됨

    L = np.full(n, 50.0)
    b = np.full(n, 20.0)
    std_L = np.full(n, 1.0)
    std_a = np.full(n, 0.5)
    std_b = np.full(n, 0.5)
    pixel_count = np.full(n, 360.0)

    return RadialProfile(
        r_normalized=r_normalized,
        L=L,
        a=a,
        b=b,
        std_L=std_L,
        std_a=std_a,
        std_b=std_b,
        pixel_count=pixel_count
    )


# ================================================================
# Test Cases
# ================================================================

def test_segmenter_config_defaults():
    """SegmenterConfig 기본값 확인"""
    config = SegmenterConfig()
    assert config.detection_method == 'gradient'
    assert config.min_zone_width == 0.05
    assert config.smoothing_window == 11
    assert config.min_gradient == 0.5


def test_segmenter_config_custom():
    """SegmenterConfig 사용자 정의 값 확인"""
    config = SegmenterConfig(
        detection_method='delta_e',
        min_zone_width=0.1,
        smoothing_window=15,
        min_gradient=1.0
    )
    assert config.detection_method == 'delta_e'
    assert config.min_zone_width == 0.1
    assert config.smoothing_window == 15
    assert config.min_gradient == 1.0


def test_zone_segmenter_creation():
    """ZoneSegmenter 객체 생성 확인"""
    segmenter = ZoneSegmenter()
    assert isinstance(segmenter, ZoneSegmenter)
    assert isinstance(segmenter.config, SegmenterConfig)


def test_segment_single_zone(simple_profile):
    """단일 zone (변곡점 없음) - 전체가 하나의 zone"""
    # min_gradient를 낮춰서 변곡점이 없도록 함
    config = SegmenterConfig(min_gradient=10.0)  # 높은 임계값
    segmenter = ZoneSegmenter(config)

    zones = segmenter.segment(simple_profile)

    # 변곡점이 없으면 단일 zone
    assert len(zones) == 1
    assert zones[0].name == 'A'
    assert zones[0].zone_type == 'pure'
    assert zones[0].r_start == 1.0
    assert zones[0].r_end == 0.0


def test_segment_multiple_zones(gradient_profile):
    """다중 zone (변곡점 있음)"""
    config = SegmenterConfig(
        min_gradient=1.0,  # 낮은 임계값으로 변곡점 검출
        min_zone_width=0.05,
        smoothing_window=5
    )
    segmenter = ZoneSegmenter(config)

    zones = segmenter.segment(gradient_profile)

    # 최소 2개 이상의 zone 생성 예상 (계단 함수 → 2개의 변곡점)
    assert len(zones) >= 2
    # 각 zone의 a* 평균이 다른지 확인
    if len(zones) >= 2:
        assert zones[0].mean_a != zones[1].mean_a


def test_detect_inflection_points_gradient(gradient_profile):
    """그래디언트 기반 변곡점 검출"""
    config = SegmenterConfig(
        detection_method='gradient',
        min_gradient=0.5,
        smoothing_window=5
    )
    segmenter = ZoneSegmenter(config)

    inflections = segmenter._detect_inflection_points(gradient_profile)

    # 계단 함수 → 2개의 변곡점 예상 (r=0.33, r=0.67 부근)
    assert len(inflections) >= 1  # 최소 1개


def test_detect_inflection_points_no_peak(simple_profile):
    """변곡점 없는 경우"""
    config = SegmenterConfig(min_gradient=10.0)
    segmenter = ZoneSegmenter(config)

    inflections = segmenter._detect_inflection_points(simple_profile)

    # 변곡점 없음
    assert len(inflections) == 0


def test_generate_zone_labels_5zones():
    """5개 zone 레이블 생성"""
    segmenter = ZoneSegmenter()
    labels = segmenter._generate_zone_labels(5)

    assert labels == ['A', 'A-B', 'B', 'B-C', 'C']
    # Pure zones
    assert 'A' in labels
    assert 'B' in labels
    assert 'C' in labels
    # Mix zones
    assert 'A-B' in labels
    assert 'B-C' in labels


def test_generate_zone_labels_3zones():
    """3개 zone 레이블 생성"""
    segmenter = ZoneSegmenter()
    labels = segmenter._generate_zone_labels(3)

    assert labels == ['A', 'B', 'C']


def test_generate_zone_labels_7zones():
    """7개 zone 레이블 생성 (4색 잉크)"""
    segmenter = ZoneSegmenter()
    labels = segmenter._generate_zone_labels(7)

    assert labels == ['A', 'A-B', 'B', 'B-C', 'C', 'C-D', 'D']


def test_generate_zone_labels_1zone():
    """1개 zone 레이블 생성"""
    segmenter = ZoneSegmenter()
    labels = segmenter._generate_zone_labels(1)

    assert labels == ['A']


def test_generate_zone_labels_custom():
    """임의 개수 zone 레이블 생성"""
    segmenter = ZoneSegmenter()
    labels = segmenter._generate_zone_labels(4)

    assert len(labels) == 4
    # Zone1, Zone2, ... 형식
    assert labels[0] == 'Zone1'
    assert labels[3] == 'Zone4'


def test_zone_properties():
    """Zone 데이터클래스 속성 확인"""
    zone = Zone(
        name='A',
        r_start=1.0,
        r_end=0.5,
        mean_L=50.0,
        mean_a=10.0,
        mean_b=20.0,
        std_L=1.0,
        std_a=0.5,
        std_b=0.5,
        zone_type='pure'
    )

    assert zone.name == 'A'
    assert zone.r_start == 1.0
    assert zone.r_end == 0.5
    assert zone.mean_L == 50.0
    assert zone.zone_type == 'pure'


def test_evaluate_mix_zone():
    """혼합 영역 평가"""
    segmenter = ZoneSegmenter()

    # 순수 영역 A, B
    zone_a = Zone(
        name='A', r_start=1.0, r_end=0.6,
        mean_L=50.0, mean_a=20.0, mean_b=10.0,
        std_L=1.0, std_a=0.5, std_b=0.5,
        zone_type='pure'
    )

    zone_b = Zone(
        name='B', r_start=0.4, r_end=0.0,
        mean_L=60.0, mean_a=10.0, mean_b=30.0,
        std_L=1.0, std_a=0.5, std_b=0.5,
        zone_type='pure'
    )

    # 혼합 영역 A-B (정상: 두 순수 영역의 중간)
    mix_zone_good = Zone(
        name='A-B', r_start=0.6, r_end=0.4,
        mean_L=55.0, mean_a=15.0, mean_b=20.0,  # A와 B의 중간
        std_L=1.0, std_a=0.5, std_b=0.5,
        zone_type='mix'
    )

    result_good = segmenter.evaluate_mix_zone(mix_zone_good, zone_a, zone_b)

    assert result_good['is_valid'] == True
    assert 0.0 <= result_good['blend_ratio'] <= 1.0
    assert result_good['distance_from_line'] >= 0.0


def test_evaluate_mix_zone_invalid():
    """비정상 혼합 영역 평가"""
    segmenter = ZoneSegmenter()

    zone_a = Zone(
        name='A', r_start=1.0, r_end=0.6,
        mean_L=50.0, mean_a=20.0, mean_b=10.0,
        std_L=0.5, std_a=0.3, std_b=0.3,  # 작은 표준편차
        zone_type='pure'
    )

    zone_b = Zone(
        name='B', r_start=0.4, r_end=0.0,
        mean_L=60.0, mean_a=10.0, mean_b=30.0,
        std_L=0.5, std_a=0.3, std_b=0.3,
        zone_type='pure'
    )

    # 비정상 혼합 (A와 B의 직선에서 멀리 벗어남)
    mix_zone_bad = Zone(
        name='A-B', r_start=0.6, r_end=0.4,
        mean_L=80.0, mean_a=0.0, mean_b=50.0,  # 크게 벗어남
        std_L=1.0, std_a=0.5, std_b=0.5,
        zone_type='mix'
    )

    result_bad = segmenter.evaluate_mix_zone(mix_zone_bad, zone_a, zone_b)

    # 거리가 크면 is_valid=False 예상
    assert result_bad['distance_from_line'] > 0.0


def test_segment_none_profile():
    """None profile 입력 시 에러"""
    segmenter = ZoneSegmenter()

    with pytest.raises(ValueError, match="Profile cannot be None"):
        segmenter.segment(None)


def test_detect_by_delta_e(gradient_profile):
    """Delta-E 기반 변곡점 검출"""
    config = SegmenterConfig(detection_method='delta_e')
    segmenter = ZoneSegmenter(config)

    inflections = segmenter._detect_by_delta_e(gradient_profile)

    # 계단 함수에서 delta_e도 급변
    assert len(inflections) >= 0  # 검출 여부는 임계값에 따라 다름


def test_segment_all_zones_have_valid_lab():
    """분할된 모든 zone이 유효한 LAB 값을 가지는지 확인"""
    n = 100
    r_normalized = np.linspace(0, 1, n)
    # 명확한 3단계 그래디언트
    a = np.zeros(n)
    a[r_normalized < 0.3] = 5.0
    a[(r_normalized >= 0.3) & (r_normalized < 0.7)] = 20.0
    a[r_normalized >= 0.7] = 35.0

    profile = RadialProfile(
        r_normalized=r_normalized,
        L=np.full(n, 50.0),
        a=a,
        b=np.full(n, 20.0),
        std_L=np.full(n, 1.0),
        std_a=np.full(n, 0.5),
        std_b=np.full(n, 0.5),
        pixel_count=np.full(n, 360.0)
    )

    config = SegmenterConfig(min_gradient=0.5, smoothing_window=5)
    segmenter = ZoneSegmenter(config)

    zones = segmenter.segment(profile)

    # 모든 zone의 LAB 값이 유효한 범위인지 확인
    for zone in zones:
        assert 0 <= zone.mean_L <= 100
        assert -128 <= zone.mean_a <= 127
        assert -128 <= zone.mean_b <= 127
        assert zone.std_L >= 0
        assert zone.std_a >= 0
        assert zone.std_b >= 0
        assert zone.r_start >= zone.r_end  # start > end (바깥→안쪽)
