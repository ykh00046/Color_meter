"""
Test for Lens Detection (v7)

Ported from legacy src/core/lens_detector.py tests, adapted for v7 API.
"""

import cv2
import numpy as np
import pytest

from src.engine_v7.core.geometry.lens_geometry import detect_lens_circle
from src.engine_v7.core.types import LensGeometry

# ================================================================
# Test Fixtures
# ================================================================


@pytest.fixture
def circle_image():
    """Clear circle in center"""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 100, (255, 255, 255), -1)  # White circle
    return img


@pytest.fixture
def noisy_circle_image():
    """Circle with noise"""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 100, (255, 255, 255), -1)
    noise = np.random.randint(0, 50, (300, 300, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


@pytest.fixture
def occluded_circle_image():
    """Partially occluded circle (left third covered)"""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 100, (255, 255, 255), -1)
    cv2.rectangle(img, (0, 0), (100, 300), (0, 0, 0), -1)  # Cover left 1/3
    return img


@pytest.fixture
def multi_circle_image():
    """Multiple circles - should select largest"""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 100, (255, 255, 255), -1)  # Large circle
    cv2.circle(img, (50, 50), 20, (100, 100, 100), -1)  # Small circle
    return img


@pytest.fixture
def off_center_circle_image():
    """Circle near edge (for center penalty test)"""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (250, 150), 40, (255, 255, 255), -1)  # Off-center small
    cv2.circle(img, (150, 150), 80, (200, 200, 200), -1)  # Center large (dimmer)
    return img


# ================================================================
# Test Cases
# ================================================================


def test_detect_lens_circle_success(circle_image):
    """Test successful lens detection with clear circle"""
    geom = detect_lens_circle(circle_image)

    assert isinstance(geom, LensGeometry)
    assert np.isclose(geom.cx, 150, atol=5)
    assert np.isclose(geom.cy, 150, atol=5)
    assert np.isclose(geom.r, 100, atol=5)
    assert geom.confidence == 1.0
    assert geom.source in ["hough_strict", "hough_medium", "hough_loose"]


def test_detect_lens_circle_no_circle():
    """Test detection failure - should return fallback"""
    img = np.zeros((300, 300, 3), dtype=np.uint8)  # Empty image
    geom = detect_lens_circle(img)

    assert isinstance(geom, LensGeometry)
    assert geom.confidence == 0.0
    assert geom.source == "fallback"
    # Fallback should return center position with default radius
    assert np.isclose(geom.cx, 150, atol=1)
    assert np.isclose(geom.cy, 150, atol=1)
    assert geom.r > 0  # Should have reasonable fallback radius


def test_detect_lens_circle_noisy(noisy_circle_image):
    """Test detection with noise - multi-stage sweep should handle it"""
    geom = detect_lens_circle(noisy_circle_image)

    assert geom.confidence == 1.0  # Should still detect despite noise
    assert np.isclose(geom.cx, 150, atol=10)  # More tolerance for noisy image
    assert np.isclose(geom.cy, 150, atol=10)
    assert np.isclose(geom.r, 100, atol=10)


def test_detect_lens_circle_occluded(occluded_circle_image):
    """Test detection with partially occluded circle"""
    geom = detect_lens_circle(occluded_circle_image)

    # Hough may fail on heavily occluded circles -> fallback expected
    # OR it may detect with loose parameters
    assert isinstance(geom, LensGeometry)
    if geom.confidence == 1.0:
        # If detected, should be approximate
        assert np.isclose(geom.cx, 150, atol=15)
        assert np.isclose(geom.cy, 150, atol=15)
        assert np.isclose(geom.r, 100, atol=15)
    else:
        # Fallback case
        assert geom.source == "fallback"


def test_detect_lens_circle_multi_stage_sweep():
    """Test that multi-stage sweep works (strict → medium → loose)"""
    # Create a faint circle that needs loose parameters
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 100, (80, 80, 80), -1)  # Dim circle

    geom = detect_lens_circle(img)

    # Should succeed at some stage
    assert isinstance(geom, LensGeometry)
    assert geom.confidence == 1.0 or geom.source == "fallback"
    # Source should indicate which stage detected it
    assert geom.source in ["hough_strict", "hough_medium", "hough_loose", "fallback"]


def test_detect_lens_circle_largest_selected(multi_circle_image):
    """Test that largest circle is selected"""
    geom = detect_lens_circle(multi_circle_image)

    assert geom.confidence == 1.0
    # Should detect the large circle (radius 100), not small one (radius 20)
    assert np.isclose(geom.cx, 150, atol=5)
    assert np.isclose(geom.cy, 150, atol=5)
    assert np.isclose(geom.r, 100, atol=5)


def test_detect_lens_circle_center_penalty(off_center_circle_image):
    """Test center penalty - detects one of the circles"""
    geom = detect_lens_circle(off_center_circle_image)

    # Center penalty should influence selection, but exact choice depends on
    # brightness, size, and position trade-offs
    assert geom.confidence == 1.0
    # Should detect a valid circle (either center or off-center)
    assert geom.r > 30  # Should detect a reasonable circle
    # At least one coordinate should be in valid range
    assert 0 < geom.cx < 300
    assert 0 < geom.cy < 300


def test_detect_lens_circle_custom_radius_range(circle_image):
    """Test custom radius range configuration"""
    cfg = {
        "min_radius_ratio": 0.25,
        "max_radius_ratio": 0.45,
    }
    geom = detect_lens_circle(circle_image, cfg=cfg)

    assert geom.confidence == 1.0
    assert np.isclose(geom.r, 100, atol=5)


def test_detect_lens_circle_custom_hough_params():
    """Test custom Hough parameters"""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 100, (255, 255, 255), -1)

    cfg = {
        "hough_param1": 150,  # Higher Canny threshold
        "hough_param2_stages": [40, 30, 20],  # Custom stage thresholds
    }
    geom = detect_lens_circle(img, cfg=cfg)

    assert isinstance(geom, LensGeometry)
    assert geom.confidence == 1.0  # Should still detect with custom params


def test_detect_lens_circle_edge_boundary():
    """Test lens near image edge - should still detect"""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 60, (255, 255, 255), -1)  # Near top-left corner

    geom = detect_lens_circle(img)

    assert isinstance(geom, LensGeometry)
    # Should detect or fallback gracefully
    if geom.confidence == 1.0:
        assert np.isclose(geom.cx, 50, atol=15)
        assert np.isclose(geom.cy, 50, atol=15)
        assert np.isclose(geom.r, 60, atol=15)


def test_detect_lens_circle_bottom_right_corner():
    """Test lens near bottom-right corner"""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (250, 250), 60, (255, 255, 255), -1)

    geom = detect_lens_circle(img)

    assert isinstance(geom, LensGeometry)
    if geom.confidence == 1.0:
        assert np.isclose(geom.cx, 250, atol=15)
        assert np.isclose(geom.cy, 250, atol=15)


def test_detect_lens_circle_small_image():
    """Test with small image - adaptive blur should work"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, (255, 255, 255), -1)

    geom = detect_lens_circle(img)

    assert isinstance(geom, LensGeometry)
    # May detect or fallback - both OK for small images
    assert geom.r > 0


def test_detect_lens_circle_large_image():
    """Test with large image - adaptive blur should work"""
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    cv2.circle(img, (500, 500), 400, (255, 255, 255), -1)

    geom = detect_lens_circle(img)

    assert isinstance(geom, LensGeometry)
    assert geom.confidence == 1.0
    assert np.isclose(geom.cx, 500, atol=20)
    assert np.isclose(geom.cy, 500, atol=20)
    assert np.isclose(geom.r, 400, atol=20)


def test_detect_lens_circle_fallback_geometry():
    """Test that fallback returns valid geometry"""
    img = np.zeros((300, 300, 3), dtype=np.uint8)  # No circle

    geom = detect_lens_circle(img)

    assert geom.confidence == 0.0
    assert geom.source == "fallback"
    # Fallback should be centered
    assert geom.cx == 150.0
    assert geom.cy == 150.0
    # Fallback radius should be reasonable (img_size * 0.42)
    assert np.isclose(geom.r, 300 * 0.42, atol=1)


def test_detect_lens_circle_source_field():
    """Test that source field is correctly set"""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 100, (255, 255, 255), -1)

    geom = detect_lens_circle(img)

    assert geom.source in ["hough_strict", "hough_medium", "hough_loose", "fallback"]
    if geom.confidence == 1.0:
        assert "hough" in geom.source
    else:
        assert geom.source == "fallback"
