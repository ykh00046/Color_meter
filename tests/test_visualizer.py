"""
Integration tests for InspectionVisualizer

Tests visualization functionality including:
- Zone overlay visualization
- Comparison charts
- Dashboard generation
- Configuration options
- Error handling
- File I/O
"""

from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.core.lens_detector import LensDetection
from src.schemas.inspection import InspectionResult, Zone, ZoneResult
from src.visualizer import InspectionVisualizer, VisualizerConfig


@pytest.fixture
def sample_lens_detection():
    """Create a sample lens detection for testing"""
    return LensDetection(center_x=512.0, center_y=512.0, radius=400.0, confidence=0.95, method="hough")


@pytest.fixture
def sample_zones():
    """Create sample zones for testing"""
    return [
        Zone(
            name="A",
            r_start=0.0,
            r_end=0.3,
            mean_L=45.2,
            mean_a=15.8,
            mean_b=-42.3,
            std_L=1.2,
            std_a=0.8,
            std_b=1.5,
            zone_type="pure",
        ),
        Zone(
            name="B",
            r_start=0.3,
            r_end=0.7,
            mean_L=78.3,
            mean_a=-5.6,
            mean_b=65.2,
            std_L=1.5,
            std_a=0.9,
            std_b=2.1,
            zone_type="pure",
        ),
        Zone(
            name="C",
            r_start=0.7,
            r_end=1.0,
            mean_L=42.7,
            mean_a=28.3,
            mean_b=18.5,
            std_L=1.3,
            std_a=1.1,
            std_b=1.4,
            zone_type="pure",
        ),
    ]


@pytest.fixture
def sample_zone_results():
    """Create sample zone results for testing"""
    return [
        ZoneResult(
            zone_name="A",
            measured_lab=(45.5, 16.2, -41.8),
            target_lab=(45.2, 15.8, -42.3),
            delta_e=0.8,
            threshold=4.2,
            is_ok=True,
        ),
        ZoneResult(
            zone_name="B",
            measured_lab=(78.8, -5.2, 66.1),
            target_lab=(78.3, -5.6, 65.2),
            delta_e=1.2,
            threshold=3.8,
            is_ok=True,
        ),
        ZoneResult(
            zone_name="C",
            measured_lab=(43.0, 28.8, 19.2),
            target_lab=(42.7, 28.3, 18.5),
            delta_e=0.9,
            threshold=3.5,
            is_ok=True,
        ),
    ]


@pytest.fixture
def sample_inspection_result(sample_zone_results):
    """Create a sample OK inspection result"""
    return InspectionResult(
        sku="TEST_SKU",
        timestamp=datetime(2025, 12, 11, 14, 30, 0),
        judgment="OK",
        overall_delta_e=0.97,
        zone_results=sample_zone_results,
        ng_reasons=[],
        confidence=0.92,
    )


@pytest.fixture
def sample_ng_inspection_result(sample_zone_results):
    """Create a sample NG inspection result"""
    # Modify one zone to be NG
    ng_zone = ZoneResult(
        zone_name="B",
        measured_lab=(85.0, -2.0, 70.0),
        target_lab=(78.3, -5.6, 65.2),
        delta_e=8.5,
        threshold=3.8,
        is_ok=False,
    )

    zone_results = sample_zone_results.copy()
    zone_results[1] = ng_zone

    return InspectionResult(
        sku="TEST_SKU",
        timestamp=datetime(2025, 12, 11, 14, 30, 0),
        judgment="NG",
        overall_delta_e=3.4,
        zone_results=zone_results,
        ng_reasons=["Zone B: ΔE=8.50 > 3.80"],
        confidence=0.45,
    )


@pytest.fixture
def sample_image():
    """Create a sample test image (1024x1024 BGR)"""
    # Create a simple gradient image for testing
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)

    # Create radial pattern
    center = (512, 512)
    for y in range(1024):
        for x in range(1024):
            dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            if dist < 400:
                # Inner: Blue
                if dist < 120:
                    img[y, x] = [200, 50, 50]  # BGR
                # Middle: Green
                elif dist < 280:
                    img[y, x] = [50, 200, 50]
                # Outer: Red
                else:
                    img[y, x] = [50, 50, 200]

    return img


@pytest.fixture
def visualizer():
    """Create a visualizer instance with default config"""
    return InspectionVisualizer(VisualizerConfig())


@pytest.fixture
def custom_visualizer():
    """Create a visualizer with custom config"""
    config = VisualizerConfig(
        zone_line_thickness=3, zone_color_ok=(0, 200, 0), zone_color_ng=(0, 0, 200), zone_label_font_scale=0.8
    )
    return InspectionVisualizer(config)


# ========== Test Zone Overlay Visualization ==========


def test_visualize_zone_overlay_ok(
    visualizer, sample_image, sample_lens_detection, sample_zones, sample_inspection_result
):
    """Test zone overlay visualization with OK result"""
    sample_inspection_result.lens_detection = sample_lens_detection
    sample_inspection_result.zones = sample_zones
    sample_inspection_result.image = sample_image

    result_img = visualizer.visualize_zone_overlay(
        sample_image, sample_lens_detection, sample_zones, sample_inspection_result, show_result=True
    )

    # Verify output is a valid image
    assert isinstance(result_img, np.ndarray)
    assert result_img.shape == sample_image.shape
    assert result_img.dtype == np.uint8

    # Verify image was modified (not identical to input)
    assert not np.array_equal(result_img, sample_image)


def test_visualize_zone_overlay_ng(
    visualizer, sample_image, sample_lens_detection, sample_zones, sample_ng_inspection_result
):
    """Test zone overlay visualization with NG result"""
    sample_ng_inspection_result.lens_detection = sample_lens_detection
    sample_ng_inspection_result.zones = sample_zones
    sample_ng_inspection_result.image = sample_image

    result_img = visualizer.visualize_zone_overlay(
        sample_image, sample_lens_detection, sample_zones, sample_ng_inspection_result, show_result=True
    )

    assert isinstance(result_img, np.ndarray)
    assert result_img.shape == sample_image.shape


def test_visualize_zone_overlay_no_result_banner(
    visualizer, sample_image, sample_lens_detection, sample_zones, sample_inspection_result
):
    """Test zone overlay without judgment banner"""
    result_img = visualizer.visualize_zone_overlay(
        sample_image, sample_lens_detection, sample_zones, sample_inspection_result, show_result=False
    )

    assert isinstance(result_img, np.ndarray)


def test_visualize_zone_overlay_custom_config(
    custom_visualizer, sample_image, sample_lens_detection, sample_zones, sample_inspection_result
):
    """Test zone overlay with custom configuration"""
    result_img = custom_visualizer.visualize_zone_overlay(
        sample_image, sample_lens_detection, sample_zones, sample_inspection_result, show_result=True
    )

    assert isinstance(result_img, np.ndarray)


# ========== Test Comparison Visualization ==========


def test_visualize_comparison_ok(visualizer, sample_zones, sample_inspection_result):
    """Test comparison chart with OK result"""
    fig = visualizer.visualize_comparison(sample_zones, sample_inspection_result)

    assert isinstance(fig, plt.Figure)

    # Check that figure has 2 subplots (LAB comparison + ΔE vs threshold)
    axes = fig.get_axes()
    assert len(axes) == 2

    plt.close(fig)


def test_visualize_comparison_ng(visualizer, sample_zones, sample_ng_inspection_result):
    """Test comparison chart with NG result"""
    fig = visualizer.visualize_comparison(sample_zones, sample_ng_inspection_result)

    assert isinstance(fig, plt.Figure)
    axes = fig.get_axes()
    assert len(axes) == 2

    plt.close(fig)


# ========== Test Dashboard Visualization ==========


def test_visualize_dashboard_mixed_results(visualizer, sample_inspection_result, sample_ng_inspection_result):
    """Test dashboard with mixed OK/NG results"""
    # Create multiple results with different SKUs
    results = [
        sample_inspection_result,
        sample_ng_inspection_result,
    ]

    # Add some more results with variations
    for i in range(3):
        result = InspectionResult(
            sku=f"SKU_{i:03d}",
            timestamp=datetime.now(),
            judgment="OK" if i % 2 == 0 else "NG",
            overall_delta_e=1.5 + i * 0.5,
            zone_results=sample_inspection_result.zone_results,
            ng_reasons=[],
            confidence=0.85 - i * 0.1,
        )
        results.append(result)

    fig = visualizer.visualize_dashboard(results)

    assert isinstance(fig, plt.Figure)

    # Dashboard should have 4 subplots
    axes = fig.get_axes()
    assert len(axes) >= 4  # At least 4 subplots (pie, box, heatmap, bar)

    plt.close(fig)


def test_visualize_dashboard_all_ok(visualizer, sample_inspection_result):
    """Test dashboard with all OK results"""
    results = [sample_inspection_result] * 5

    fig = visualizer.visualize_dashboard(results)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_visualize_dashboard_all_ng(visualizer, sample_ng_inspection_result):
    """Test dashboard with all NG results"""
    results = [sample_ng_inspection_result] * 5

    fig = visualizer.visualize_dashboard(results)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ========== Test File I/O ==========


def test_save_visualization_image_png(visualizer, sample_image, tmp_path):
    """Test saving image visualization to PNG"""
    output_path = tmp_path / "test_overlay.png"

    visualizer.save_visualization(sample_image, output_path)

    assert output_path.exists()

    # Verify we can load it back
    loaded = cv2.imread(str(output_path))
    assert loaded is not None
    assert loaded.shape == sample_image.shape


def test_save_visualization_figure_png(visualizer, sample_zones, sample_inspection_result, tmp_path):
    """Test saving figure visualization to PNG"""
    output_path = tmp_path / "test_comparison.png"

    fig = visualizer.visualize_comparison(sample_zones, sample_inspection_result)
    visualizer.save_visualization(fig, output_path)

    assert output_path.exists()
    plt.close(fig)


def test_save_visualization_figure_pdf(visualizer, sample_zones, sample_inspection_result, tmp_path):
    """Test saving figure visualization to PDF"""
    output_path = tmp_path / "test_comparison.pdf"

    fig = visualizer.visualize_comparison(sample_zones, sample_inspection_result)
    visualizer.save_visualization(fig, output_path, format="pdf")

    assert output_path.exists()
    plt.close(fig)


# ========== Test Figure to Array Conversion ==========


def test_figure_to_array(visualizer, sample_zones, sample_inspection_result):
    """Test converting matplotlib figure to numpy array"""
    fig = visualizer.visualize_comparison(sample_zones, sample_inspection_result)

    arr = visualizer.figure_to_array(fig)

    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8
    assert arr.ndim == 3
    assert arr.shape[2] == 3  # BGR

    plt.close(fig)


# ========== Test Error Handling ==========


def test_visualize_zone_overlay_empty_zones(visualizer, sample_image, sample_lens_detection, sample_inspection_result):
    """Test zone overlay with empty zones list"""
    # Should handle empty zones gracefully
    result_img = visualizer.visualize_zone_overlay(
        sample_image, sample_lens_detection, [], sample_inspection_result, show_result=True
    )

    assert isinstance(result_img, np.ndarray)


def test_visualize_comparison_single_zone(visualizer, sample_zones, sample_inspection_result):
    """Test comparison chart with single zone"""
    single_zone = [sample_zones[0]]
    single_zone_result = InspectionResult(
        sku="TEST_SKU",
        timestamp=datetime.now(),
        judgment="OK",
        overall_delta_e=0.8,
        zone_results=[sample_inspection_result.zone_results[0]],
        ng_reasons=[],
        confidence=0.9,
    )

    fig = visualizer.visualize_comparison(single_zone, single_zone_result)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_visualize_dashboard_single_result(visualizer, sample_inspection_result):
    """Test dashboard with single result"""
    results = [sample_inspection_result]

    fig = visualizer.visualize_dashboard(results)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ========== Integration Tests with Real Pipeline ==========


@pytest.mark.integration
def test_end_to_end_visualization_with_real_image():
    """Integration test with real VIS_TEST images"""
    from src.pipeline import InspectionPipeline
    from src.utils.file_io import read_json

    # Load VIS_TEST SKU config
    sku_config = read_json(Path("config/sku_db/VIS_TEST.json"))

    # Create pipeline
    pipeline = InspectionPipeline(sku_config)

    # Process a test image
    test_image = "data/raw_images/VIS_OK_001.jpg"
    if not Path(test_image).exists():
        pytest.skip(f"Test image not found: {test_image}")

    result = pipeline.process(test_image, "VIS_TEST")

    # Verify visualization data is populated
    assert result.lens_detection is not None
    assert result.zones is not None
    assert result.image is not None

    # Create visualizations
    visualizer = InspectionVisualizer(VisualizerConfig())

    # Zone overlay
    overlay = visualizer.visualize_zone_overlay(
        result.image, result.lens_detection, result.zones, result, show_result=True
    )
    assert isinstance(overlay, np.ndarray)

    # Comparison
    comparison = visualizer.visualize_comparison(result.zones, result)
    assert isinstance(comparison, plt.Figure)
    plt.close(comparison)


@pytest.mark.integration
def test_batch_dashboard_visualization():
    """Integration test for batch dashboard visualization"""
    from src.pipeline import InspectionPipeline
    from src.utils.file_io import read_json

    # Load VIS_TEST SKU config
    sku_config = read_json(Path("config/sku_db/VIS_TEST.json"))

    # Create pipeline
    pipeline = InspectionPipeline(sku_config)

    # Find VIS_*.jpg images
    vis_images = list(Path("data/raw_images").glob("VIS_*.jpg"))

    if len(vis_images) < 2:
        pytest.skip("Not enough VIS_*.jpg images for batch test")

    # Process batch (use first 6 images)
    results = []
    for img_path in vis_images[:6]:
        result = pipeline.process(str(img_path), "VIS_TEST")
        results.append(result)

    # Create dashboard
    visualizer = InspectionVisualizer(VisualizerConfig())
    dashboard = visualizer.visualize_dashboard(results)

    assert isinstance(dashboard, plt.Figure)
    plt.close(dashboard)
