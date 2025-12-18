"""
Unit tests for IlluminationCorrector module
"""

import cv2
import numpy as np
import pytest

from src.core.illumination_corrector import (
    CorrectionResult,
    CorrectorConfig,
    IlluminationCorrector,
    IlluminationCorrectorError,
)

# ================================================================
# Fixtures
# ================================================================


@pytest.fixture
def vignetting_image_lab():
    """
    Vignetting 효과가 있는 이미지 (중심 밝고 외곽 어두움)
    """
    h, w = 300, 300
    y, x = np.ogrid[:h, :w]
    center_x, center_y = 150, 150

    # 거리 계산
    dx = x - center_x
    dy = y - center_y
    distance = np.sqrt(dx**2 + dy**2)
    max_distance = 150.0

    # Vignetting: 중심부 밝음 (L=200), 외곽 어두움 (L=100)
    L = (200 - (distance / max_distance) * 100).clip(0, 255).astype(np.uint8)
    a = np.full((h, w), 128, dtype=np.uint8)
    b = np.full((h, w), 128, dtype=np.uint8)

    image_lab = np.stack([L, a, b], axis=-1)
    return image_lab


@pytest.fixture
def uniform_image_lab():
    """
    균일한 이미지 (보정 불필요)
    """
    h, w = 300, 300
    image_lab = np.full((h, w, 3), [150, 128, 128], dtype=np.uint8)
    return image_lab


# ================================================================
# Test Cases
# ================================================================


def test_corrector_config_defaults():
    """CorrectorConfig 기본값 확인"""
    config = CorrectorConfig()
    assert config.enabled is False  # 기본값: 비활성화
    assert config.method == "polynomial"
    assert config.polynomial_degree == 2
    assert config.target_luminance is None
    assert config.preserve_color is True


def test_corrector_config_custom():
    """CorrectorConfig 커스텀 설정"""
    config = CorrectorConfig(enabled=True, method="gaussian", target_luminance=150.0)
    assert config.enabled is True
    assert config.method == "gaussian"
    assert config.target_luminance == 150.0


def test_illumination_corrector_creation():
    """IlluminationCorrector 생성 확인"""
    corrector = IlluminationCorrector()
    assert corrector.config.enabled is False

    custom_config = CorrectorConfig(enabled=True)
    corrector_custom = IlluminationCorrector(custom_config)
    assert corrector_custom.config.enabled is True


def test_correction_disabled_returns_original(vignetting_image_lab):
    """보정 비활성화 시 원본 반환"""
    corrector = IlluminationCorrector(CorrectorConfig(enabled=False))

    result = corrector.correct(image_lab=vignetting_image_lab, center_x=150.0, center_y=150.0, radius=150.0)

    # 결과 타입 확인
    assert isinstance(result, CorrectionResult)
    assert result.correction_applied is False

    # 원본과 동일
    np.testing.assert_array_equal(result.corrected_image, vignetting_image_lab)


def test_correction_enabled_polynomial(vignetting_image_lab):
    """다항식 보정 활성화"""
    corrector = IlluminationCorrector(CorrectorConfig(enabled=True, method="polynomial"))

    result = corrector.correct(image_lab=vignetting_image_lab, center_x=150.0, center_y=150.0, radius=150.0)

    # 보정 적용됨
    assert result.correction_applied is True
    assert result.method == "polynomial"

    # 보정 데이터 존재
    assert result.luminance_profile is not None
    assert result.correction_factors is not None

    # 이미지 형태 유지
    assert result.corrected_image.shape == vignetting_image_lab.shape


def test_correction_enabled_gaussian(vignetting_image_lab):
    """가우시안 보정 활성화"""
    corrector = IlluminationCorrector(CorrectorConfig(enabled=True, method="gaussian"))

    result = corrector.correct(image_lab=vignetting_image_lab, center_x=150.0, center_y=150.0, radius=150.0)

    # 보정 적용됨
    assert result.correction_applied is True
    assert result.method == "gaussian"


def test_preserve_color_only_l_changed(vignetting_image_lab):
    """색상 보존: L만 변경, a/b는 유지"""
    corrector = IlluminationCorrector(CorrectorConfig(enabled=True, preserve_color=True))

    result = corrector.correct(image_lab=vignetting_image_lab, center_x=150.0, center_y=150.0, radius=150.0)

    # a, b 채널은 변경되지 않음
    np.testing.assert_array_equal(result.corrected_image[:, :, 1], vignetting_image_lab[:, :, 1])
    np.testing.assert_array_equal(result.corrected_image[:, :, 2], vignetting_image_lab[:, :, 2])


def test_uniform_image_minimal_correction(uniform_image_lab):
    """균일한 이미지는 보정 최소화"""
    corrector = IlluminationCorrector(CorrectorConfig(enabled=True))

    result = corrector.correct(image_lab=uniform_image_lab, center_x=150.0, center_y=150.0, radius=150.0)

    # 보정 계수가 1.0 근처여야 함
    if result.correction_factors is not None:
        # 대부분의 보정 계수가 0.9~1.1 범위
        assert np.mean(np.abs(result.correction_factors - 1.0)) < 0.2


def test_invalid_input_empty_image():
    """빈 이미지 에러 처리"""
    corrector = IlluminationCorrector(CorrectorConfig(enabled=True))

    with pytest.raises(IlluminationCorrectorError, match="Input image is empty"):
        corrector.correct(image_lab=np.array([]), center_x=150.0, center_y=150.0, radius=150.0)


def test_invalid_input_wrong_shape():
    """잘못된 이미지 shape 에러 처리"""
    corrector = IlluminationCorrector(CorrectorConfig(enabled=True))

    # 2D 이미지
    image_2d = np.zeros((300, 300), dtype=np.uint8)

    with pytest.raises(IlluminationCorrectorError, match="Expected Lab image"):
        corrector.correct(image_lab=image_2d, center_x=150.0, center_y=150.0, radius=150.0)


def test_invalid_method():
    """잘못된 보정 방법 에러 처리"""
    config = CorrectorConfig(enabled=True, method="invalid_method")
    corrector = IlluminationCorrector(config)

    image_lab = np.full((200, 200, 3), [150, 128, 128], dtype=np.uint8)

    with pytest.raises(IlluminationCorrectorError, match="Unknown correction method"):
        corrector.correct(image_lab=image_lab, center_x=100.0, center_y=100.0, radius=100.0)


def test_correction_with_mask(vignetting_image_lab):
    """마스크 적용 보정"""
    # 원형 마스크 생성
    h, w = 300, 300
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - 150) ** 2 + (y - 150) ** 2)
    mask = distance <= 140

    corrector = IlluminationCorrector(CorrectorConfig(enabled=True))

    result = corrector.correct(image_lab=vignetting_image_lab, center_x=150.0, center_y=150.0, radius=150.0, mask=mask)

    assert result.correction_applied is True
