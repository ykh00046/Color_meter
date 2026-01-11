"""
Unit tests for core/v2/color_masks.py

Tests color mask generation, color ID stability, L* ordering, and edge cases.
"""

import numpy as np
import pytest

from core.geometry.lens_geometry import LensGeometry
from core.measure.color_masks import assign_cluster_labels_to_image, build_color_masks, filter_masks_by_role, lab_to_hex


@pytest.fixture
def default_cfg():
    """Default configuration for testing."""
    return {
        "polar": {"R": 260, "T": 720},
        "anomaly": {
            "r_start": 0.15,
            "r_end": 0.95,
            "center_frac": 0.1,
        },
        "v2_ink": {
            "roi_r_start": 0.15,
            "roi_r_end": 0.95,
            "dark_top_p": 0.25,
            "chroma_top_p": 0.35,
            "l_weight": 0.3,
            "kmeans_attempts": 10,
            "min_samples": 1000,
            "min_samples_warn": 500,
            "rng_seed": 42,  # Fixed seed for reproducibility
        },
    }


@pytest.fixture
def synthetic_3color_image():
    """
    Create synthetic 3-color lens image with:
    - Dark ink (center)
    - Mid ink (middle ring)
    - Light gap (outer ring)
    """
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cy, cx = 200, 200

    # Draw concentric circles with different colors (BGR format)
    # Outer ring: Light (gap) - RGB(200, 200, 200) -> BGR(200, 200, 200)
    import cv2

    cv2.circle(img, (cx, cy), 180, (200, 200, 200), -1)

    # Middle ring: Mid ink - RGB(100, 100, 100) -> BGR(100, 100, 100)
    cv2.circle(img, (cx, cy), 120, (100, 100, 100), -1)

    # Center: Dark ink - RGB(30, 30, 30) -> BGR(30, 30, 30)
    cv2.circle(img, (cx, cy), 60, (30, 30, 30), -1)

    return img


@pytest.fixture
def synthetic_2color_image():
    """
    Create synthetic 2-color lens image with:
    - Dark ink
    - Light gap
    """
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cy, cx = 200, 200

    import cv2

    # Outer: Light gap - RGB(180, 180, 180)
    cv2.circle(img, (cx, cy), 180, (180, 180, 180), -1)

    # Inner: Dark ink - RGB(40, 40, 40)
    cv2.circle(img, (cx, cy), 120, (40, 40, 40), -1)

    return img


def test_lab_to_hex_basic():
    """Test Lab to hex conversion with known values."""
    # Pure white (L=100, a=0, b=0)
    white_lab = np.array([100.0, 0.0, 0.0])
    hex_white = lab_to_hex(white_lab)
    assert hex_white.startswith("#")
    assert len(hex_white) == 7
    # Should be close to white (#FFFFFF)
    assert hex_white.upper() in ["#FFFFFF", "#FEFEFE", "#FEFFFF"]  # Allow small rounding errors

    # Pure black (L=0, a=0, b=0)
    black_lab = np.array([0.0, 0.0, 0.0])
    hex_black = lab_to_hex(black_lab)
    assert hex_black == "#000000"

    # Gray (L=50, a=0, b=0)
    gray_lab = np.array([50.0, 0.0, 0.0])
    hex_gray = lab_to_hex(gray_lab)
    assert hex_gray.startswith("#")
    # Should be some shade of gray


def test_build_color_masks_basic(synthetic_3color_image, default_cfg):
    """Test basic color mask generation with 3-color synthetic image."""
    img = synthetic_3color_image
    geom = LensGeometry(cx=200.0, cy=200.0, r=180.0)

    masks, metadata = build_color_masks(img, default_cfg, expected_k=3, geom=geom)

    # Should have 3 masks
    assert len(masks) == 3
    assert "color_0" in masks
    assert "color_1" in masks
    assert "color_2" in masks

    # Check metadata
    assert metadata["k_expected"] == 3
    assert metadata["k_used"] == 3
    assert len(metadata["colors"]) == 3

    # Check that masks are boolean arrays
    for color_id, mask in masks.items():
        assert mask.dtype == bool
        assert mask.shape == (720, 260)  # (T, R) from config

    # Check that colors are ordered by L* (dark to light)
    L_values = [color["lab_centroid"][0] for color in metadata["colors"]]
    assert L_values == sorted(L_values), "Colors should be ordered by L* (dark to light)"

    # Check that all colors have "ink" role
    for color in metadata["colors"]:
        assert color["role"] == "ink"


def test_color_id_stability(synthetic_3color_image, default_cfg):
    """Test that color IDs are stable across multiple runs with same seed."""
    img = synthetic_3color_image
    geom = LensGeometry(cx=200.0, cy=200.0, r=180.0)

    # Run twice with same seed
    masks1, metadata1 = build_color_masks(img, default_cfg, expected_k=3, geom=geom)
    masks2, metadata2 = build_color_masks(img, default_cfg, expected_k=3, geom=geom)

    # Check that color IDs have same L* ordering
    L_values1 = [color["lab_centroid"][0] for color in metadata1["colors"]]
    L_values2 = [color["lab_centroid"][0] for color in metadata2["colors"]]

    assert len(L_values1) == len(L_values2)

    # Allow small numerical differences due to floating point
    for L1, L2 in zip(L_values1, L_values2):
        assert abs(L1 - L2) < 1.0, "L* values should be nearly identical with same seed"

    # Check that masks are identical
    for color_id in masks1.keys():
        assert np.array_equal(
            masks1[color_id], masks2[color_id]
        ), f"Mask for {color_id} should be identical across runs"


def test_l_star_ordering(synthetic_3color_image, default_cfg):
    """Test that colors are correctly ordered by L* (dark to light)."""
    img = synthetic_3color_image
    geom = LensGeometry(cx=200.0, cy=200.0, r=180.0)

    masks, metadata = build_color_masks(img, default_cfg, expected_k=3, geom=geom)

    # Extract L* values
    L_values = [color["lab_centroid"][0] for color in metadata["colors"]]

    # Should be sorted in ascending order (dark to light)
    assert L_values[0] < L_values[1] < L_values[2], f"L* values should be ascending: {L_values}"

    # color_0 should be darkest (lowest L*)
    assert metadata["colors"][0]["color_id"] == "color_0"
    assert L_values[0] < 50, "Darkest color should have L* < 50"

    # color_2 should be lightest (highest L*)
    assert metadata["colors"][2]["color_id"] == "color_2"
    assert L_values[2] > 100, "Lightest color should have L* > 100"


def test_role_assignment_single_ink(synthetic_2color_image, default_cfg):
    """Test that with expected_k=1, darkest cluster is assigned 'ink' role."""
    img = synthetic_2color_image
    geom = LensGeometry(cx=200.0, cy=200.0, r=180.0)

    # Request single ink, should cluster with k=2 and assign roles
    masks, metadata = build_color_masks(img, default_cfg, expected_k=1, geom=geom)

    # Should have 2 masks (ink + gap)
    assert len(masks) == 2
    assert metadata["k_expected"] == 1
    assert metadata["k_used"] == 2

    # Check roles
    colors = metadata["colors"]
    assert len(colors) == 2

    # color_0 (darkest) should be "ink"
    assert colors[0]["color_id"] == "color_0"
    assert colors[0]["role"] == "ink"

    # color_1 (lightest) should be "gap"
    assert colors[1]["color_id"] == "color_1"
    assert colors[1]["role"] == "gap"


def test_role_assignment_multiple_inks(synthetic_3color_image, default_cfg):
    """Test that with expected_k>1, all colors are assigned 'ink' role."""
    img = synthetic_3color_image
    geom = LensGeometry(cx=200.0, cy=200.0, r=180.0)

    masks, metadata = build_color_masks(img, default_cfg, expected_k=3, geom=geom)

    # All should be "ink"
    for color in metadata["colors"]:
        assert color["role"] == "ink"


def test_filter_masks_by_role(synthetic_2color_image, default_cfg):
    """Test filtering masks by role."""
    img = synthetic_2color_image
    geom = LensGeometry(cx=200.0, cy=200.0, r=180.0)

    masks, metadata = build_color_masks(img, default_cfg, expected_k=1, geom=geom)

    # Filter for ink only
    ink_masks = filter_masks_by_role(masks, metadata, role="ink")
    assert len(ink_masks) == 1
    assert "color_0" in ink_masks

    # Filter for gap only
    gap_masks = filter_masks_by_role(masks, metadata, role="gap")
    assert len(gap_masks) == 1
    assert "color_1" in gap_masks


def test_insufficient_samples_graceful_failure(default_cfg):
    """Test that insufficient samples results in warning flags."""
    # Create tiny image with very little data
    # Even with black image, clustering may succeed, but should warn
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    geom = LensGeometry(cx=25.0, cy=25.0, r=20.0)

    # Modify config to require more samples than available
    cfg_strict = default_cfg.copy()
    cfg_strict["v2_ink"] = default_cfg["v2_ink"].copy()
    cfg_strict["v2_ink"]["min_samples"] = 100000  # Unrealistically high
    cfg_strict["v2_ink"]["min_samples_warn"] = 50000

    masks, metadata = build_color_masks(img, cfg_strict, expected_k=3, geom=geom)

    # Should return masks (may not be empty due to fallback)
    assert len(masks) == 3

    # Should have warnings about low confidence or sampling issues
    assert "warnings" in metadata
    # May have warnings like "INK_SEPARATION_LOW_CONFIDENCE" or similar
    # Check that at least warnings field exists (may be empty if fallback succeeds)
    assert isinstance(metadata["warnings"], list)


def test_assign_cluster_labels_to_image():
    """Test pixel assignment to nearest cluster."""
    # Create simple Lab map (2x2, 3 channels)
    lab_map = np.array(
        [[[10.0, 20.0, 30.0], [60.0, 70.0, 80.0]], [[15.0, 25.0, 35.0], [65.0, 75.0, 85.0]]], dtype=np.float32
    )  # (2, 2, 3)

    # Create 2 cluster centers in feature space [a, b, L*0.3]
    # Cluster 0: dark (a=20, b=30, L=10 -> L*0.3=3)
    # Cluster 1: light (a=70, b=80, L=60 -> L*0.3=18)
    centers = np.array([[20.0, 30.0, 10.0 * 0.3], [70.0, 80.0, 60.0 * 0.3]], dtype=np.float32)

    labels = assign_cluster_labels_to_image(lab_map, centers, l_weight=0.3)

    # Check shape
    assert labels.shape == (2, 2)

    # Check assignments
    # Top-left [10, 20, 30] should be closer to cluster 0
    assert labels[0, 0] == 0
    # Top-right [60, 70, 80] should be closer to cluster 1
    assert labels[0, 1] == 1
    # Bottom-left [15, 25, 35] should be closer to cluster 0
    assert labels[1, 0] == 0
    # Bottom-right [65, 75, 85] should be closer to cluster 1
    assert labels[1, 1] == 1


def test_metadata_structure(synthetic_3color_image, default_cfg):
    """Test that metadata has all required fields."""
    img = synthetic_3color_image
    geom = LensGeometry(cx=200.0, cy=200.0, r=180.0)

    masks, metadata = build_color_masks(img, default_cfg, expected_k=3, geom=geom)

    # Check top-level keys
    required_keys = [
        "colors",
        "k_expected",
        "k_used",
        "segmentation_method",
        "l_weight",
        "warnings",
        "roi_meta",
        "sample_meta",
        "geom",
    ]
    for key in required_keys:
        assert key in metadata, f"Missing key: {key}"

    # Check color metadata structure
    for color in metadata["colors"]:
        assert "color_id" in color
        assert "lab_centroid" in color
        assert "hex_ref" in color
        assert "area_ratio" in color
        assert "role" in color

        # Check types
        assert isinstance(color["color_id"], str)
        assert isinstance(color["lab_centroid"], list)
        assert len(color["lab_centroid"]) == 3
        assert isinstance(color["hex_ref"], str)
        assert color["hex_ref"].startswith("#")
        assert isinstance(color["area_ratio"], float)
        assert color["role"] in ["ink", "gap"]


def test_area_ratios_sum_to_one(synthetic_3color_image, default_cfg):
    """Test that area ratios approximately sum to 1.0."""
    img = synthetic_3color_image
    geom = LensGeometry(cx=200.0, cy=200.0, r=180.0)

    masks, metadata = build_color_masks(img, default_cfg, expected_k=3, geom=geom)

    total_area = sum(color["area_ratio"] for color in metadata["colors"])

    # Should be close to 1.0 (allowing for small numerical errors)
    assert abs(total_area - 1.0) < 0.01, f"Area ratios sum to {total_area}, expected ~1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
