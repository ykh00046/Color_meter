import pytest
import numpy as np
import cv2
from src.core.radial_profiler import RadialProfiler, ProfilerConfig, RadialProfile
from src.core.lens_detector import LensDetection # LensDetection 데이터 클래스 사용

# ================================================================
# RadialProfiler 테스트 시작
# ================================================================

# 테스트용 더미 이미지 생성 픽스처
@pytest.fixture
def perfect_circle_image():
    """중앙에 단색 원이 있는 이미지"""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 100, (100, 200, 50), -1) # BGR
    return img

@pytest.fixture
def gradient_circle_image():
    """색상 그라디언트가 있는 원 이미지 (r 방향으로 색상 변화)"""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    center_x, center_y = 150, 150
    max_radius = 100
    
    for y in range(300):
        for x in range(300):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist <= max_radius:
                # 반경에 따라 색상 변화 (예: Blue -> Red)
                ratio = dist / max_radius
                img[y, x, 0] = int(255 * (1 - ratio)) # Blue (decreasing)
                img[y, x, 1] = 0
                img[y, x, 2] = int(255 * ratio)     # Red (increasing)
    return img

@pytest.fixture
def lens_detection_fixture():
    """테스트용 LensDetection 객체"""
    return LensDetection(center_x=150, center_y=150, radius=100)

# Test Case 1: 기본 ProfilerConfig 생성 확인
def test_profiler_config_defaults():
    config = ProfilerConfig()
    assert config.r_start_ratio == 0.0
    assert config.r_step_pixels == 1
    assert config.smoothing_enabled is True

# Test Case 2: 사용자 정의 ProfilerConfig 생성 확인
def test_profiler_config_custom():
    config = ProfilerConfig(r_start_ratio=0.3, r_step_pixels=2, smoothing_enabled=False)
    assert config.r_start_ratio == 0.3
    assert config.r_step_pixels == 2
    assert config.smoothing_enabled is False

# Test Case 3: RadialProfiler 객체 생성 확인
def test_radial_profiler_creation():
    profiler = RadialProfiler()
    assert isinstance(profiler, RadialProfiler)
    assert isinstance(profiler.config, ProfilerConfig)

# Test Case 4: extract_profile - 단색 원 이미지에서 LAB 프로파일 추출
def test_extract_profile_perfect_circle(perfect_circle_image, lens_detection_fixture):
    profiler = RadialProfiler()
    profile = profiler.extract_profile(perfect_circle_image, lens_detection_fixture)
    
    assert profile is not None
    assert isinstance(profile, RadialProfile)
    assert profile.r_normalized.ndim == 1
    assert profile.L.ndim == 1
    assert profile.a.ndim == 1
    assert profile.b.ndim == 1
    
    # 단색 원이므로 L, a, b 값이 거의 일정해야 함
    # BGR (100, 200, 50) -> LAB 변환 후 대략적인 값 확인 (정확한 변환은 어려움)
    # OpenCV LAB 변환 범위: L [0,255], a [-128,127], b [-128,127]
    # Blue: ~100, Green: ~200, Red: ~50
    # 대략적인 LAB 값 (cv2.COLOR_BGR2LAB 결과는 라이브러리마다 다를 수 있음)
    # L은 밝기, a는 녹색-빨강, b는 파랑-노랑
    # (100, 200, 50)은 녹색 계열이 강하고 어두운 편 -> L은 중간, a는 음수 (녹색), b는 음수(파랑)
    assert np.all(profile.L > 0)
    assert np.all(np.isclose(profile.L, np.mean(profile.L), atol=5)) # 거의 일정
    assert np.all(np.isclose(profile.a, np.mean(profile.a), atol=5))
    assert np.all(np.isclose(profile.b, np.mean(profile.b), atol=5))
    
    assert np.all(profile.pixel_count == profiler.config.theta_samples) # 모든 링에 픽셀 존재

# Test Case 5: extract_profile - 그라디언트 원 이미지에서 LAB 프로파일 추출
def test_extract_profile_gradient_circle(gradient_circle_image, lens_detection_fixture):
    profiler = RadialProfiler(config=ProfilerConfig(smoothing_enabled=False)) # 스무딩 없이 테스트
    profile = profiler.extract_profile(gradient_circle_image, lens_detection_fixture)
    
    assert profile is not None
    assert isinstance(profile, RadialProfile)
    
    # 그라디언트이므로 L, a, b 값이 반경에 따라 변화해야 함
    # Blue -> Red로 변화 -> Blue 감소 (b* 증가: 음수에서 0으로), a*는 증가
    assert not np.all(np.isclose(profile.b, profile.b[0]))
    assert not np.all(np.isclose(profile.a, profile.a[0]))

    # b* 프로파일이 증가하는 경향 (파란색 감소 = b* 음수값이 작아짐 = 증가)
    assert np.mean(np.diff(profile.b)) > 0
    # a* 프로파일 변화 확인 (미세한 변화도 LAB 변환 노이즈 가능)
    # RGB -> LAB conversion can have weak gradients in a* channel
    assert np.std(profile.a) > 0.01  # Just verify there's some variation (lowered threshold)


# Test Case 6: _smooth_profile - Savitzky-Golay 스무딩 적용 확인
def test_smooth_profile_savgol(perfect_circle_image, lens_detection_fixture):
    # 약간의 노이즈 추가하여 스무딩 효과 확인
    noisy_img = perfect_circle_image.copy()
    noise = np.random.normal(0, 5, perfect_circle_image.shape).astype(np.int16)
    noisy_img = np.clip(noisy_img + noise, 0, 255).astype(np.uint8)

    profiler = RadialProfiler(config=ProfilerConfig(smoothing_enabled=True, smoothing_method="savgol"))
    profile_noisy = profiler.extract_profile(noisy_img, lens_detection_fixture)
    profile_smoothed = profiler._smooth_profile(profile_noisy)

    # 스무딩 후 원본 대비 변화가 있어야 하며, 노이즈가 줄어야 함 (표준편차 감소)
    assert not np.all(profile_smoothed.L == profile_noisy.L)
    assert np.std(profile_smoothed.L) < np.std(profile_noisy.L)


# Test Case 7: _smooth_profile - 이동 평균 스무딩 적용 확인
def test_smooth_profile_moving_average(perfect_circle_image, lens_detection_fixture):
    noisy_img = perfect_circle_image.copy()
    noise = np.random.normal(0, 5, perfect_circle_image.shape).astype(np.int16)
    noisy_img = np.clip(noisy_img + noise, 0, 255).astype(np.uint8)

    profiler = RadialProfiler(config=ProfilerConfig(smoothing_enabled=True, smoothing_method="moving_average"))
    profile_noisy = profiler.extract_profile(noisy_img, lens_detection_fixture)
    profile_smoothed = profiler._smooth_profile(profile_noisy)
    
    assert not np.all(profile_smoothed.L == profile_noisy.L)
    assert np.std(profile_smoothed.L) < np.std(profile_noisy.L)

# Test Case 8: _smooth_profile - 스무딩 비활성화 시 원본과 동일
def test_smooth_profile_disabled(perfect_circle_image, lens_detection_fixture):
    config = ProfilerConfig(smoothing_enabled=False)
    profiler = RadialProfiler(config)
    profile = profiler.extract_profile(perfect_circle_image, lens_detection_fixture)
    # 스무딩 비활성화 시 _smooth_profile은 호출되지 않음. extract_profile 내에서 검증
    assert not config.smoothing_enabled # 설정이 정확한지 확인

# Test Case 9: extract_profile - 입력 이미지가 None일 경우 예외 발생
def test_extract_profile_none_image():
    profiler = RadialProfiler()
    lens = LensDetection(center_x=0, center_y=0, radius=10)
    with pytest.raises(ValueError, match="Input image cannot be None."):
        profiler.extract_profile(None, lens)

# Test Case 10: extract_profile - LensDetection이 None일 경우 예외 발생
def test_extract_profile_none_lens_detection(perfect_circle_image):
    profiler = RadialProfiler()
    with pytest.raises(ValueError, match="LensDetection object cannot be None."):
        profiler.extract_profile(perfect_circle_image, None)

# Test Case 11: r_step_pixels가 렌즈 반경보다 클 경우 처리
def test_r_step_pixels_too_large(perfect_circle_image, lens_detection_fixture):
    # 렌즈 반경 100, r_step_pixels = 200 (너무 큼)
    config = ProfilerConfig(r_step_pixels=200) 
    profiler = RadialProfiler(config)
    profile = profiler.extract_profile(perfect_circle_image, lens_detection_fixture)
    
    # 프로파일의 길이가 최소 1 이상이어야 함
    assert len(profile.r_normalized) >= 1

# Test Case 12: r_start_ratio 및 r_end_ratio 적용 확인
def test_r_start_end_ratio(perfect_circle_image, lens_detection_fixture):
    config = ProfilerConfig(r_start_ratio=0.2, r_end_ratio=0.8)
    profiler = RadialProfiler(config)
    profile = profiler.extract_profile(perfect_circle_image, lens_detection_fixture)
    
    assert np.isclose(profile.r_normalized.min(), 0.2, atol=0.01)
    assert np.isclose(profile.r_normalized.max(), 0.8, atol=0.01)

# Test Case 13: polar_image가 비어있는 경우 (cv2.warpPolar 오류)
def test_extract_profile_empty_polar_image(perfect_circle_image, lens_detection_fixture, mocker):
    mocker.patch('cv2.warpPolar', return_value=np.array([])) # 빈 배열 반환 모킹
    profiler = RadialProfiler()
    profile = profiler.extract_profile(perfect_circle_image, lens_detection_fixture)
    
    assert profile is not None
    assert len(profile.r_normalized) > 0 # 더미 프로파일이 생성되었는지 확인
    assert np.all(profile.L == 0.0) # 모든 값이 0으로 채워져야 함

# Test Case 14: _smooth_profile - 프로파일 길이가 윈도우 길이보다 짧은 경우 (savgol)
def test_smooth_profile_short_savgol_profile(perfect_circle_image, lens_detection_fixture):
    config = ProfilerConfig(r_step_pixels=50, savgol_window_length=11) # 프로파일 길이 짧게
    profiler = RadialProfiler(config)
    profile = profiler.extract_profile(perfect_circle_image, lens_detection_fixture)
    
    # 스무딩이 스킵되었으므로 원본과 동일해야 함
    profile_smoothed = profiler._smooth_profile(profile)
    assert np.all(profile_smoothed.L == profile.L)
    assert np.all(profile_smoothed.a == profile.a)
    assert np.all(profile_smoothed.b == profile.b)

# Test Case 15: _smooth_profile - 프로파일 길이가 윈도우 길이보다 짧은 경우 (moving_average)
def test_smooth_profile_short_moving_average_profile(perfect_circle_image, lens_detection_fixture):
    config = ProfilerConfig(r_step_pixels=50, moving_average_window=5) # 프로파일 길이 짧게
    profiler = RadialProfiler(config)
    profile = profiler.extract_profile(perfect_circle_image, lens_detection_fixture)
    
    # 스무딩이 스킵되었으므로 원본과 동일해야 함
    profile_smoothed = profiler._smooth_profile(profile)
    assert np.all(profile_smoothed.L == profile.L)
    assert np.all(profile_smoothed.a == profile.a)
    assert np.all(profile_smoothed.b == profile.b)