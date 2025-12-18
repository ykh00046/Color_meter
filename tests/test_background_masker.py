"""
Unit tests for BackgroundMasker module
"""

import cv2
import numpy as np
import pytest

from src.core.background_masker import BackgroundMasker, BackgroundMaskerError, MaskConfig, MaskResult

# ================================================================
# Fixtures
# ================================================================


@pytest.fixture
def normal_image_lab():
    """
    정상적인 Lab 이미지 (300x300)
    - 중심: 유효 색상 (L=128, 채도 있음)
    - 외곽: 배경 (L=255, 무채색)
    """
    h, w = 300, 300
    image = np.zeros((h, w, 3), dtype=np.uint8)

    # 중심부: 유효 색상 (파란색)
    cv2.circle(image, (150, 150), 120, (255, 0, 0), -1)  # BGR

    # 외곽: 배경 (흰색, 무채색)
    cv2.circle(image, (150, 150), 150, (255, 255, 255), -1)
    cv2.circle(image, (150, 150), 120, (255, 0, 0), -1)

    # BGR → Lab 변환
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    return image_lab


@pytest.fixture
def bright_spots_image_lab():
    """
    밝은 반사점이 있는 이미지 (하이라이트)
    """
    h, w = 300, 300
    image = np.zeros((h, w, 3), dtype=np.uint8)

    # 기본 색상 (녹색)
    cv2.circle(image, (150, 150), 140, (0, 255, 0), -1)

    # 밝은 반사점 (흰색)
    cv2.circle(image, (150, 150), 30, (255, 255, 255), -1)

    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    return image_lab


# ================================================================
# Test Cases
# ================================================================


def test_mask_config_defaults():
    """MaskConfig 기본값 확인"""
    config = MaskConfig()
    assert config.use_luminance_filter is True
    assert config.use_saturation_filter is True
    assert config.L_min == 20.0
    assert config.L_max == 240.0
    assert config.saturation_min == 5.0
    assert config.morphology_enabled is True
    assert config.morph_kernel_size == 3


def test_mask_config_custom():
    """MaskConfig 커스텀 설정"""
    config = MaskConfig(L_min=30.0, L_max=230.0, saturation_min=10.0, morphology_enabled=False)
    assert config.L_min == 30.0
    assert config.L_max == 230.0
    assert config.saturation_min == 10.0
    assert config.morphology_enabled is False


def test_background_masker_creation():
    """BackgroundMasker 생성 확인"""
    masker = BackgroundMasker()
    assert masker.config.use_luminance_filter is True

    custom_config = MaskConfig(use_luminance_filter=False)
    masker_custom = BackgroundMasker(custom_config)
    assert masker_custom.config.use_luminance_filter is False


def test_create_mask_normal_image(normal_image_lab):
    """정상 이미지에서 마스크 생성"""
    masker = BackgroundMasker()
    result = masker.create_mask(image_lab=normal_image_lab, center_x=150.0, center_y=150.0, radius=150.0)

    # 결과 타입 확인
    assert isinstance(result, MaskResult)
    assert isinstance(result.mask, np.ndarray)
    assert result.mask.dtype == bool

    # 마스크 shape 확인
    assert result.mask.shape == (300, 300)

    # 유효 픽셀 비율 확인 (0~1)
    assert 0.0 <= result.valid_pixel_ratio <= 1.0

    # 일부 픽셀은 유효해야 함
    assert result.valid_pixel_ratio > 0.0


def test_mask_result_properties(normal_image_lab):
    """MaskResult 속성 확인"""
    masker = BackgroundMasker()
    result = masker.create_mask(image_lab=normal_image_lab, center_x=150.0, center_y=150.0, radius=150.0)

    # 필터링 통계
    assert isinstance(result.filtered_by_luminance, int)
    assert isinstance(result.filtered_by_saturation, int)
    assert isinstance(result.morphology_applied, bool)

    # 필터링된 픽셀 수는 0 이상
    assert result.filtered_by_luminance >= 0
    assert result.filtered_by_saturation >= 0


def test_luminance_filter_bright_spots(bright_spots_image_lab):
    """휘도 필터링으로 밝은 반사점 제거"""
    # 엄격한 L_max 설정
    config = MaskConfig(L_max=200.0, use_saturation_filter=False)
    masker = BackgroundMasker(config)

    result = masker.create_mask(image_lab=bright_spots_image_lab, center_x=150.0, center_y=150.0, radius=150.0)

    # 밝은 픽셀이 필터링되었어야 함
    assert result.filtered_by_luminance > 0


def test_saturation_filter():
    """채도 필터링으로 무채색 제거"""
    # 무채색 이미지 생성 (회색)
    h, w = 200, 200
    image_gray = np.full((h, w, 3), 128, dtype=np.uint8)  # BGR 회색
    image_lab = cv2.cvtColor(image_gray, cv2.COLOR_BGR2Lab)

    config = MaskConfig(use_luminance_filter=False, saturation_min=10.0)
    masker = BackgroundMasker(config)

    result = masker.create_mask(image_lab=image_lab, center_x=100.0, center_y=100.0, radius=100.0)

    # 무채색 픽셀이 대부분 필터링되어야 함
    assert result.filtered_by_saturation > 0
    assert result.valid_pixel_ratio < 0.5  # 절반 이하만 유효


def test_circular_mask_boundary():
    """원형 마스크 경계 확인"""
    # 균일한 유효 색상 이미지
    h, w = 200, 200
    image_bgr = np.full((h, w, 3), [100, 150, 100], dtype=np.uint8)
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)

    # 필터링 없이 원형 마스크만
    config = MaskConfig(use_luminance_filter=False, use_saturation_filter=False, morphology_enabled=False)
    masker = BackgroundMasker(config)

    result = masker.create_mask(image_lab=image_lab, center_x=100.0, center_y=100.0, radius=80.0)

    # 원 내부만 True
    # 대략적으로 π * 80^2 ≈ 20106 픽셀
    expected_pixels = np.pi * 80**2
    actual_pixels = np.sum(result.mask)

    # ±10% 허용
    assert 0.9 * expected_pixels <= actual_pixels <= 1.1 * expected_pixels


def test_morphology_effect():
    """형태학적 연산 효과 확인"""
    h, w = 200, 200
    image_bgr = np.full((h, w, 3), [100, 150, 100], dtype=np.uint8)
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)

    # 형태학적 연산 없음
    config_no_morph = MaskConfig(morphology_enabled=False)
    masker_no_morph = BackgroundMasker(config_no_morph)
    result_no_morph = masker_no_morph.create_mask(image_lab=image_lab, center_x=100.0, center_y=100.0, radius=90.0)

    # 형태학적 연산 있음
    config_morph = MaskConfig(morphology_enabled=True, morph_kernel_size=5)
    masker_morph = BackgroundMasker(config_morph)
    result_morph = masker_morph.create_mask(image_lab=image_lab, center_x=100.0, center_y=100.0, radius=90.0)

    assert result_morph.morphology_applied is True
    assert result_no_morph.morphology_applied is False


def test_invalid_input_empty_image():
    """빈 이미지 에러 처리"""
    masker = BackgroundMasker()

    with pytest.raises(BackgroundMaskerError, match="Input image is empty"):
        masker.create_mask(image_lab=np.array([]), center_x=100.0, center_y=100.0, radius=100.0)


def test_invalid_input_wrong_shape():
    """잘못된 이미지 shape 에러 처리"""
    masker = BackgroundMasker()

    # 2D 이미지
    image_2d = np.zeros((200, 200), dtype=np.uint8)

    with pytest.raises(BackgroundMaskerError, match="Expected Lab image"):
        masker.create_mask(image_lab=image_2d, center_x=100.0, center_y=100.0, radius=100.0)


def test_large_radius():
    """반경이 이미지보다 큰 경우"""
    h, w = 200, 200
    image_bgr = np.full((h, w, 3), [100, 150, 100], dtype=np.uint8)
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)

    masker = BackgroundMasker()

    # 반경이 이미지 크기보다 큼
    result = masker.create_mask(image_lab=image_lab, center_x=100.0, center_y=100.0, radius=300.0)

    # 전체 이미지가 원 내부
    # 필터링은 여전히 적용됨
    assert result.valid_pixel_ratio > 0.0


def test_combined_filters(normal_image_lab):
    """휘도 + 채도 필터 동시 적용"""
    config = MaskConfig(
        use_luminance_filter=True, use_saturation_filter=True, L_min=30.0, L_max=230.0, saturation_min=10.0
    )
    masker = BackgroundMasker(config)

    result = masker.create_mask(image_lab=normal_image_lab, center_x=150.0, center_y=150.0, radius=150.0)

    # 두 필터 모두 적용되어야 함
    total_filtered = result.filtered_by_luminance + result.filtered_by_saturation
    assert total_filtered > 0
