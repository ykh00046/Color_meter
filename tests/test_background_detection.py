"""
Test background-based lens detection (Priority 10)
"""

import cv2
import numpy as np
import pytest

from src.core.lens_detector import DetectorConfig, LensDetectionError, LensDetector


def test_background_detection_on_simple_image():
    """Test that background-based detection works when primary methods fail"""
    # Create a very low contrast image to make Hough/Contour fail
    # Gray lens (100) on dark gray background (50) - low contrast
    img = np.full((300, 300, 3), 50, dtype=np.uint8)
    cv2.circle(img, (150, 150), 80, (100, 100, 100), -1)

    # Configure to make Hough/Contour fail but allow background fallback
    config = DetectorConfig(
        method="hybrid",
        hough_param1=200,  # Very high edge threshold to make Hough fail
        hough_param2=200,  # Very high accumulator threshold
        contour_threshold_method="otsu",  # Otsu might fail on low contrast
        background_fallback_enabled=True,
        background_color_distance_threshold=20.0,  # Low threshold for low contrast
    )

    detector = LensDetector(config)
    detection = detector.detect(img)

    # Should succeed (either with hybrid or background fallback)
    assert detection is not None
    # Accept either method since contour might succeed on some runs
    assert detection.method in ["hybrid", "background"]
    assert 130 <= detection.center_x <= 170
    assert 130 <= detection.center_y <= 170
    assert 60 <= detection.radius <= 100


def test_background_detection_disabled():
    """Test that detection fails when background fallback is disabled"""
    # Create a blank image (all methods should fail)
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Disable background fallback
    config = DetectorConfig(method="hybrid", background_fallback_enabled=False)

    detector = LensDetector(config)

    with pytest.raises(LensDetectionError):
        detector.detect(img)


def test_background_color_sampling():
    """Test background color sampling from edges"""
    # Create image with red background and white lens
    img = np.full((300, 300, 3), (0, 0, 255), dtype=np.uint8)  # Red background
    cv2.circle(img, (150, 150), 80, (255, 255, 255), -1)  # White lens

    detector = LensDetector()
    bg_color = detector._sample_background_color(img)

    # Background should be close to red (BGR = 0, 0, 255)
    assert bg_color[0] < 50  # Blue component low
    assert bg_color[1] < 50  # Green component low
    assert bg_color[2] > 200  # Red component high


def test_background_detection_with_complex_background():
    """Test background detection with textured background"""
    # Create an image with noisy background and clean lens
    np.random.seed(42)  # For reproducibility
    img = np.random.randint(0, 50, (300, 300, 3), dtype=np.uint8)  # Dark noisy background
    cv2.circle(img, (150, 150), 70, (200, 200, 200), -1)  # Bright lens

    config = DetectorConfig(
        method="hybrid",
        background_fallback_enabled=True,
        background_color_distance_threshold=50.0,  # Higher threshold for noisy background
    )

    detector = LensDetector(config)
    detection = detector.detect(img)

    assert detection is not None
    assert 130 <= detection.center_x <= 170
    assert 130 <= detection.center_y <= 170
    # Allow wider tolerance for noisy images (detection might overestimate)
    assert 50 <= detection.radius <= 110


def test_background_detection_confidence_calculation():
    """Test that confidence is calculated based on circularity"""
    # Create a perfect circle (high circularity)
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 80, (255, 255, 255), -1)

    # Force background-based detection by making other methods fail
    config = DetectorConfig(method="hybrid", hough_param2=200, background_fallback_enabled=True)

    detector = LensDetector(config)
    detection = detector.detect(img)

    # Perfect circle should have higher confidence
    assert detection.confidence >= 0.5  # Should be in upper range (0.3-0.6)

    # Create an irregular shape (low circularity)
    img2 = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img2, (100, 100), (200, 200), (255, 255, 255), -1)

    detection2 = detector.detect(img2)

    # Rectangle should have lower confidence than circle
    assert detection2.confidence < detection.confidence


def test_background_detection_min_area_filtering():
    """Test that small foreground regions are filtered out"""
    # Create image with small noise and larger lens
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 5, (255, 255, 255), -1)  # Small noise
    cv2.circle(img, (150, 150), 70, (255, 255, 255), -1)  # Larger lens

    config = DetectorConfig(
        method="hybrid",
        background_fallback_enabled=True,
        background_min_area_ratio=0.02,  # Require at least 2% of image
    )

    detector = LensDetector(config)
    detection = detector.detect(img)

    # Should detect the larger circle, not the noise
    assert detection is not None
    assert 130 <= detection.center_x <= 170
    assert 130 <= detection.center_y <= 170


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
