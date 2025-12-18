"""
Unit tests for BoundaryDetector module
"""

import numpy as np
import pytest

from src.core.boundary_detector import BoundaryConfig, BoundaryDetector, BoundaryDetectorError, BoundaryResult
from src.core.radial_profiler import RadialProfile

# ================================================================
# Fixtures
# ================================================================


@pytest.fixture
def normal_profile():
    """
    정상적인 RadialProfile (중심부 고분산, 중간 안정, 외곽 저분산)
    """
    n = 100
    r_normalized = np.linspace(0.0, 1.0, n)

    # 중심부 (0.0~0.1): 고분산 (반사광)
    # 중간 (0.1~0.9): 안정 영역
    # 외곽 (0.9~1.0): 저분산 (배경)

    std_L = np.ones(n) * 10.0
    std_L[:10] = 50.0  # 중심부 고분산
    std_L[90:] = 5.0  # 외곽 저분산

    std_a = np.ones(n) * 8.0
    std_a[:10] = 40.0
    std_a[90:] = 3.0

    std_b = np.ones(n) * 8.0
    std_b[:10] = 40.0
    std_b[90:] = 3.0

    L = np.linspace(50, 200, n)
    a = np.linspace(100, 150, n)
    b = np.linspace(100, 150, n)
    pixel_count = np.full(n, 360, dtype=int)

    return RadialProfile(
        r_normalized=r_normalized, L=L, a=a, b=b, std_L=std_L, std_a=std_a, std_b=std_b, pixel_count=pixel_count
    )


@pytest.fixture
def uniform_profile():
    """
    균일한 RadialProfile (경계 검출이 어려운 경우)
    """
    n = 100
    r_normalized = np.linspace(0.0, 1.0, n)

    # 전체 영역 균일
    std_L = np.ones(n) * 10.0
    std_a = np.ones(n) * 8.0
    std_b = np.ones(n) * 8.0

    L = np.ones(n) * 128.0
    a = np.ones(n) * 128.0
    b = np.ones(n) * 128.0
    pixel_count = np.full(n, 360, dtype=int)

    return RadialProfile(
        r_normalized=r_normalized, L=L, a=a, b=b, std_L=std_L, std_a=std_a, std_b=std_b, pixel_count=pixel_count
    )


# ================================================================
# Test Cases
# ================================================================


def test_boundary_config_defaults():
    """BoundaryConfig 기본값 확인"""
    config = BoundaryConfig()
    assert config.std_threshold_multiplier == 1.5
    assert config.min_r_inner == 0.05
    assert config.max_r_outer == 0.95
    assert config.edge_detection_method == "std"
    assert config.smoothing_window == 5


def test_boundary_config_custom():
    """BoundaryConfig 커스텀 설정"""
    config = BoundaryConfig(
        std_threshold_multiplier=2.0, min_r_inner=0.1, max_r_outer=0.9, edge_detection_method="gradient"
    )
    assert config.std_threshold_multiplier == 2.0
    assert config.min_r_inner == 0.1
    assert config.max_r_outer == 0.9
    assert config.edge_detection_method == "gradient"


def test_boundary_detector_creation():
    """BoundaryDetector 생성 확인"""
    detector = BoundaryDetector()
    assert detector.config.edge_detection_method == "std"

    custom_config = BoundaryConfig(edge_detection_method="gradient")
    detector_custom = BoundaryDetector(custom_config)
    assert detector_custom.config.edge_detection_method == "gradient"


def test_detect_boundaries_normal_profile(normal_profile):
    """정상 프로파일에서 경계 검출"""
    detector = BoundaryDetector()
    result = detector.detect_boundaries(normal_profile)

    # 결과 타입 확인
    assert isinstance(result, BoundaryResult)
    assert isinstance(result.r_inner, float)
    assert isinstance(result.r_outer, float)
    assert isinstance(result.confidence, float)

    # 경계 범위 확인
    assert 0.0 <= result.r_inner < result.r_outer <= 1.0

    # r_inner는 중심부 고분산 영역 이후 (약 0.1 근처)
    assert result.r_inner >= 0.05
    assert result.r_inner <= 0.2

    # r_outer는 외곽 저분산 영역 이전 (약 0.9 근처)
    assert result.r_outer >= 0.8
    assert result.r_outer <= 0.95

    # 신뢰도 확인
    assert 0.0 <= result.confidence <= 1.0


def test_detect_boundaries_uniform_profile(uniform_profile):
    """균일 프로파일에서 경계 검출 (fallback)"""
    detector = BoundaryDetector()
    result = detector.detect_boundaries(uniform_profile)

    # Fallback to default range
    assert result.r_inner >= 0.05
    assert result.r_outer <= 0.95
    assert result.r_inner < result.r_outer


def test_detect_boundaries_gradient_method(normal_profile):
    """기울기 기반 검출 방법"""
    config = BoundaryConfig(edge_detection_method="gradient")
    detector = BoundaryDetector(config)
    result = detector.detect_boundaries(normal_profile)

    assert result.method == "gradient"
    assert 0.0 <= result.r_inner < result.r_outer <= 1.0


def test_boundary_result_metadata(normal_profile):
    """BoundaryResult 메타데이터 확인"""
    detector = BoundaryDetector()
    result = detector.detect_boundaries(normal_profile)

    assert "std_mean" in result.metadata
    assert "std_max" in result.metadata
    assert "std_min" in result.metadata

    assert result.metadata["std_mean"] > 0.0
    assert result.metadata["std_max"] >= result.metadata["std_mean"]
    assert result.metadata["std_min"] <= result.metadata["std_mean"]


def test_min_max_constraints():
    """min_r_inner/max_r_outer 제약 확인"""
    config = BoundaryConfig(min_r_inner=0.2, max_r_outer=0.8)
    detector = BoundaryDetector(config)

    # 정상 프로파일 생성
    n = 100
    r_normalized = np.linspace(0.0, 1.0, n)
    std_L = np.ones(n) * 10.0
    std_a = np.ones(n) * 8.0
    std_b = np.ones(n) * 8.0

    profile = RadialProfile(
        r_normalized=r_normalized,
        L=np.ones(n) * 128.0,
        a=np.ones(n) * 128.0,
        b=np.ones(n) * 128.0,
        std_L=std_L,
        std_a=std_a,
        std_b=std_b,
        pixel_count=np.full(n, 360, dtype=int),
    )

    result = detector.detect_boundaries(profile)

    # 제약 조건 확인
    assert result.r_inner >= 0.2
    assert result.r_outer <= 0.8


def test_invalid_input_none():
    """None 입력 에러 처리"""
    detector = BoundaryDetector()

    with pytest.raises(BoundaryDetectorError, match="RadialProfile is None"):
        detector.detect_boundaries(None)


def test_invalid_input_empty_profile():
    """빈 프로파일 에러 처리"""
    detector = BoundaryDetector()

    empty_profile = RadialProfile(
        r_normalized=np.array([]),
        L=np.array([]),
        a=np.array([]),
        b=np.array([]),
        std_L=np.array([]),
        std_a=np.array([]),
        std_b=np.array([]),
        pixel_count=np.array([]),
    )

    with pytest.raises(BoundaryDetectorError, match="RadialProfile is empty"):
        detector.detect_boundaries(empty_profile)


def test_invalid_detection_method():
    """잘못된 검출 방법 에러 처리"""
    config = BoundaryConfig(edge_detection_method="invalid_method")
    detector = BoundaryDetector(config)

    n = 50
    profile = RadialProfile(
        r_normalized=np.linspace(0.0, 1.0, n),
        L=np.ones(n) * 128.0,
        a=np.ones(n) * 128.0,
        b=np.ones(n) * 128.0,
        std_L=np.ones(n) * 10.0,
        std_a=np.ones(n) * 8.0,
        std_b=np.ones(n) * 8.0,
        pixel_count=np.full(n, 360, dtype=int),
    )

    with pytest.raises(BoundaryDetectorError, match="Unknown method"):
        detector.detect_boundaries(profile)


def test_smoothing_effect():
    """평활화 효과 확인"""
    # 노이즈가 있는 프로파일 생성
    n = 100
    r_normalized = np.linspace(0.0, 1.0, n)

    np.random.seed(42)
    std_L = np.ones(n) * 10.0 + np.random.randn(n) * 5.0  # 노이즈 추가
    std_a = np.ones(n) * 8.0 + np.random.randn(n) * 3.0
    std_b = np.ones(n) * 8.0 + np.random.randn(n) * 3.0

    noisy_profile = RadialProfile(
        r_normalized=r_normalized,
        L=np.ones(n) * 128.0,
        a=np.ones(n) * 128.0,
        b=np.ones(n) * 128.0,
        std_L=std_L,
        std_a=std_a,
        std_b=std_b,
        pixel_count=np.full(n, 360, dtype=int),
    )

    # 평활화 없이
    config_no_smooth = BoundaryConfig(smoothing_window=1)
    detector_no_smooth = BoundaryDetector(config_no_smooth)
    result_no_smooth = detector_no_smooth.detect_boundaries(noisy_profile)

    # 평활화 있음
    config_smooth = BoundaryConfig(smoothing_window=11)
    detector_smooth = BoundaryDetector(config_smooth)
    result_smooth = detector_smooth.detect_boundaries(noisy_profile)

    # 평활화 버전이 더 안정적인 결과를 낼 것으로 예상
    assert result_smooth.confidence >= 0.0
    assert result_no_smooth.confidence >= 0.0


def test_combined_std_calculation(normal_profile):
    """종합 표준편차 계산 확인"""
    detector = BoundaryDetector()
    std_combined = detector._calculate_combined_std(normal_profile)

    # RMS 계산 확인
    expected = np.sqrt((normal_profile.std_L**2 + normal_profile.std_a**2 + normal_profile.std_b**2) / 3.0)

    np.testing.assert_array_almost_equal(std_combined, expected, decimal=5)
