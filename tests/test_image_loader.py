import pytest
import numpy as np
import cv2
from pathlib import Path
import os
import shutil

from src.core.image_loader import ImageLoader, ImageConfig

# 테스트용 더미 이미지 생성 함수
@pytest.fixture(scope="module")
def dummy_image_path(tmp_path_factory):
    """테스트용 더미 이미지 파일을 생성하고 경로를 반환한다."""
    # tmp_path_factory를 사용하여 모듈 스코프의 임시 디렉토리를 생성
    test_dir = tmp_path_factory.mktemp("image_loader_test_images")
    img_path = test_dir / "test_image.png"
    
    # 50x50의 파란색 이미지 생성
    dummy_img = np.zeros((50, 50, 3), dtype=np.uint8)
    dummy_img[:, :, 0] = 255 # Blue channel
    cv2.imwrite(str(img_path), dummy_img)
    
    return img_path

@pytest.fixture(scope="module")
def color_gradient_image_path(tmp_path_factory):
    """화이트 밸런스 테스트를 위한 색상 그라디언트 이미지 생성"""
    test_dir = tmp_path_factory.mktemp("image_loader_gradient_images")
    img_path = test_dir / "gradient_image.png"

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        # 파란색에서 빨간색으로 그라디언트
        img[i, :, 0] = int(255 * (1 - i / 99)) # Blue (decreasing)
        img[i, :, 2] = int(255 * (i / 99))     # Red (increasing)
    cv2.imwrite(str(img_path), img)
    return img_path

@pytest.fixture(scope="module")
def green_tint_image_path(tmp_path_factory):
    """녹색 틴트가 있는 이미지 (화이트 밸런스 테스트용)"""
    test_dir = tmp_path_factory.mktemp("image_loader_tint_images")
    img_path = test_dir / "green_tint_image.png"
    
    img = np.full((50, 50, 3), 100, dtype=np.uint8) # 회색 배경
    img[:, :, 1] = 150 # 녹색 틴트 추가
    cv2.imwrite(str(img_path), img)
    return img_path


# ================================================================
# ImageLoader 테스트 시작
# ================================================================

# Test Case 1: 기본 ImageConfig 생성 확인
def test_image_config_defaults():
    config = ImageConfig()
    assert config.resolution_w == 1920
    assert config.denoise_enabled is True
    assert config.white_balance_method == "gray_world"

# Test Case 2: 사용자 정의 ImageConfig 생성 확인
def test_image_config_custom():
    config = ImageConfig(resolution_w=1024, denoise_enabled=False, white_balance_method="none")
    assert config.resolution_w == 1024
    assert config.denoise_enabled is False
    assert config.white_balance_method == "none"

# Test Case 3: ImageLoader 객체 생성 확인
def test_image_loader_creation():
    loader = ImageLoader()
    assert isinstance(loader, ImageLoader)
    assert isinstance(loader.config, ImageConfig)

# Test Case 4: 파일로부터 이미지 로드 성공
def test_load_from_file_success(dummy_image_path):
    loader = ImageLoader()
    image = loader.load_from_file(dummy_image_path)
    assert image is not None
    assert isinstance(image, np.ndarray)
    assert image.shape == (50, 50, 3)
    assert np.all(image[0, 0] == [255, 0, 0]) # BGR 순서로 파란색 확인

# Test Case 5: 존재하지 않는 파일 로드 실패
def test_load_from_file_fail():
    loader = ImageLoader()
    non_existent_path = Path("non_existent_image.png")
    image = loader.load_from_file(non_existent_path)
    assert image is None

# Test Case 6: _denoise_image (Gaussian)
def test_denoise_gaussian(dummy_image_path):
    config = ImageConfig(denoise_method="gaussian", denoise_kernel_size=3)
    loader = ImageLoader(config)
    image = loader.load_from_file(dummy_image_path)
    # Add noise to image so denoise filter has something to work with
    noisy_image = image.copy()
    noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
    noisy_image = np.clip(noisy_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    denoised_image = loader._denoise_image(noisy_image)
    assert denoised_image.shape == noisy_image.shape
    # 가우시안 블러가 적용되면 이미지 값이 약간 변함 (정확한 값 비교는 어려움)
    assert not np.all(denoised_image == noisy_image) 

# Test Case 7: _denoise_image (Bilateral)
def test_denoise_bilateral(dummy_image_path):
    config = ImageConfig(denoise_method="bilateral", denoise_kernel_size=5)
    loader = ImageLoader(config)
    image = loader.load_from_file(dummy_image_path)
    # Add noise to image so denoise filter has something to work with
    noisy_image = image.copy()
    noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
    noisy_image = np.clip(noisy_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    denoised_image = loader._denoise_image(noisy_image)
    assert denoised_image.shape == noisy_image.shape
    assert not np.all(denoised_image == noisy_image)

# Test Case 8: _denoise_image (Disabled)
def test_denoise_disabled(dummy_image_path):
    # Disable ALL preprocessing to ensure image remains unchanged
    config = ImageConfig(denoise_enabled=False, white_balance_enabled=False, auto_roi_detection=False)
    loader = ImageLoader(config)
    image = loader.load_from_file(dummy_image_path)
    processed_image = loader.preprocess(image) # preprocess 내부에서 denoise_enabled를 확인함
    assert np.all(processed_image == image) # 노이즈 제거 비활성화 시 원본과 동일해야 함

# Test Case 9: _gray_world_white_balance 기본 동작 (녹색 틴트 제거)
def test_gray_world_white_balance_basic(green_tint_image_path):
    loader = ImageLoader()
    image = loader.load_from_file(green_tint_image_path)
    
    original_mean_bgr = np.mean(image, axis=(0,1))
    # print(f"Original mean BGR: {original_mean_bgr}") # 기대: B~100, G~150, R~100
    
    balanced_image = loader._gray_world_white_balance(image)
    
    balanced_mean_bgr = np.mean(balanced_image, axis=(0,1))
    # print(f"Balanced mean BGR: {balanced_mean_bgr}") # 기대: B,G,R 값이 모두 비슷하게 나옴
    
    # 평균 R, G, B 값이 서로 가까워졌는지 확인 (회색에 가까워짐)
    assert np.isclose(balanced_mean_bgr[0], balanced_mean_bgr[1], atol=5) # B와 G
    assert np.isclose(balanced_mean_bgr[1], balanced_mean_bgr[2], atol=5) # G와 R
    assert np.all(balanced_image >= 0) and np.all(balanced_image <= 255) # 값 범위 확인

# Test Case 10: _gray_world_white_balance (Disabled)
def test_white_balance_disabled(dummy_image_path):
    config = ImageConfig(white_balance_enabled=False)
    loader = ImageLoader(config)
    image = loader.load_from_file(dummy_image_path)
    processed_image = loader.preprocess(image)
    assert np.all(processed_image == image)

# Test Case 11: _detect_roi_from_image 기본 동작 (파란색 이미지의 ROI 검출)
def test_detect_roi_basic(dummy_image_path):
    loader = ImageLoader()
    image = loader.load_from_file(dummy_image_path)
    x, y, w, h = loader._detect_roi_from_image(image)
    # 50x50 파란색 이미지이므로 전체 이미지가 ROI로 잡혀야 함 (마진 포함)
    # 가장 큰 컨투어가 이미지 전체일 것이므로 0,0,50,50
    # 마진 20% (10픽셀) 추가
    assert x == 0
    assert y == 0
    assert w == 50
    assert h == 50

# Test Case 12: _detect_roi_from_image (이미지 중앙에 작은 객체)
def test_detect_roi_small_object_center(tmp_path_factory):
    test_dir = tmp_path_factory.mktemp("image_loader_roi_images")
    img_path = test_dir / "small_object.png"
    
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60, :] = 255 # 중앙에 흰색 사각형
    cv2.imwrite(str(img_path), img)

    loader = ImageLoader()
    image = loader.load_from_file(img_path)
    x, y, w, h = loader._detect_roi_from_image(image)
    
    # 중앙 객체 (20x20)에 마진 20% (4픽셀) 적용
    assert x == 36
    assert y == 36
    assert w == 28 # 20 + 2*4
    assert h == 28 # 20 + 2*4

# Test Case 13: preprocess 전체 파이프라인 (자동 ROI + 노이즈 제거 + 화이트 밸런스)
def test_preprocess_full_pipeline(green_tint_image_path):
    config = ImageConfig(auto_roi_detection=True, denoise_enabled=True, white_balance_enabled=True)
    loader = ImageLoader(config)
    image = loader.load_from_file(green_tint_image_path)
    
    processed_image = loader.preprocess(image)
    
    assert processed_image is not None
    assert processed_image.shape == image.shape # ROI가 전체 이미지인 경우
    # 각 전처리 단계가 이미지 값을 변경했음을 확인
    assert not np.all(processed_image == image)

# Test Case 14: preprocess - ROI 설정 시 자동 ROI 무시
def test_preprocess_manual_roi(dummy_image_path):
    config = ImageConfig(roi=(10, 10, 20, 20), auto_roi_detection=False)
    loader = ImageLoader(config)
    image = loader.load_from_file(dummy_image_path)
    processed_image = loader.preprocess(image)
    assert processed_image.shape == (20, 20, 3) # 수동 설정된 ROI 크기

# Test Case 15: load_from_camera - 더미 카메라 객체 사용
def test_load_from_camera_success():
    class MockCamera:
        def capture_frame(self):
            return np.zeros((100, 100, 3), dtype=np.uint8)
    
    loader = ImageLoader()
    mock_camera = MockCamera()
    image = loader.load_from_camera(mock_camera)
    assert image is not None
    assert image.shape == (100, 100, 3)

# Test Case 16: load_from_camera - 카메라 캡처 실패 시 None 반환
def test_load_from_camera_fail():
    class MockCamera:
        def capture_frame(self):
            return None
    
    loader = ImageLoader()
    mock_camera = MockCamera()
    image = loader.load_from_camera(mock_camera)
    assert image is None

# Test Case 17: preprocess - 입력 이미지가 None인 경우 처리
def test_preprocess_none_image():
    loader = ImageLoader()
    processed_image = loader.preprocess(None)
    assert processed_image is None

# Test Case 18: _detect_roi_from_image - 빈 이미지 처리
def test_detect_roi_empty_image():
    loader = ImageLoader()
    empty_image = np.zeros((0, 0, 3), dtype=np.uint8)
    x, y, w, h = loader._detect_roi_from_image(empty_image)
    assert (x, y, w, h) == (0, 0, 0, 0)

# Test Case 19: _detect_roi_from_image - 복잡한 배경 (가장 큰 컨투어만 잡기)
def test_detect_roi_complex_background(tmp_path_factory):
    test_dir = tmp_path_factory.mktemp("image_loader_complex_roi")
    img_path = test_dir / "complex_roi.png"
    
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # 큰 객체
    cv2.circle(img, (100, 100), 50, (255, 255, 255), -1)
    # 작은 노이즈
    cv2.circle(img, (20, 20), 5, (255, 255, 255), -1)
    cv2.imwrite(str(img_path), img)

    loader = ImageLoader()
    image = loader.load_from_file(img_path)
    x, y, w, h = loader._detect_roi_from_image(image)
    
    # 큰 원에 대한 ROI (중심 100,100, 반경 50, 직경 100)
    # 마진 20% = 직경(bounding box width) * 0.2 = 100 * 0.2 = 20
    diameter = 50 * 2
    margin = int(diameter * 0.2)
    expected_x = 100 - 50 - margin  # center - radius - margin = 100 - 50 - 20 = 30
    expected_y = 100 - 50 - margin  # 30
    expected_w = diameter + 2 * margin  # 100 + 40 = 140
    expected_h = diameter + 2 * margin  # 140

    # Allow ±2 pixel tolerance due to morphology operations affecting contour boundaries
    assert abs(x - expected_x) <= 2
    assert abs(y - expected_y) <= 2
    assert abs(w - expected_w) <= 4  # Width/height can vary by 2*margin tolerance
    assert abs(h - expected_h) <= 4

# Test Case 20: _gray_world_white_balance - 흰색 이미지에 적용
def test_gray_world_white_balance_white_image(tmp_path_factory):
    test_dir = tmp_path_factory.mktemp("image_loader_white_wb")
    img_path = test_dir / "white_image.png"
    
    white_img = np.full((50, 50, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(img_path), white_img)

    loader = ImageLoader()
    image = loader.load_from_file(img_path)
    balanced_image = loader._gray_world_white_balance(image)
    
    # 흰색 이미지는 보정 후에도 흰색이어야 함
    assert np.all(balanced_image == 255)

# Test Case 21: _gray_world_white_balance - 검은색 이미지에 적용
def test_gray_world_white_balance_black_image(tmp_path_factory):
    test_dir = tmp_path_factory.mktemp("image_loader_black_wb")
    img_path = test_dir / "black_image.png"
    
    black_img = np.full((50, 50, 3), 0, dtype=np.uint8)
    cv2.imwrite(str(img_path), black_img)

    loader = ImageLoader()
    image = loader.load_from_file(img_path)
    balanced_image = loader._gray_world_white_balance(image)
    
    # 검은색 이미지는 보정 후에도 검은색이어야 함 (RGB 평균이 0이므로 스케일 팩터가 무한대가 될 수 있으나, np.mean이 0이면 보정 X)
    assert np.all(balanced_image == 0)

# Test Case 22: _denoise_image - 노이즈 제거 비활성화 시 설정 무시
def test_denoise_disabled_config_ignore(dummy_image_path):
    # Disable all preprocessing to ensure image remains unchanged
    config = ImageConfig(denoise_enabled=False, denoise_method="bilateral", denoise_kernel_size=9,
                         white_balance_enabled=False, auto_roi_detection=False)
    loader = ImageLoader(config)
    image = loader.load_from_file(dummy_image_path)
    processed_image = loader.preprocess(image)
    assert np.all(processed_image == image) # 노이즈 제거 비활성화 시 다른 설정값 무시하고 원본과 동일해야 함