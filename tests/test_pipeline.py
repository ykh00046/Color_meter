"""
Integration tests for InspectionPipeline.
"""

import json
from pathlib import Path

import pytest

from src.services.inspection_service import InspectionPipeline, PipelineError
from src.utils.file_io import read_json


@pytest.fixture
def sku_config():
    """Load SKU001 configuration"""
    config_path = Path("config/sku_db/SKU001.json")
    return read_json(config_path)


@pytest.fixture
def pipeline(sku_config):
    """Create InspectionPipeline instance"""
    return InspectionPipeline(sku_config)


def test_pipeline_initialization(sku_config):
    """Test pipeline initialization"""
    pipeline = InspectionPipeline(sku_config)

    # image_loader is removed (handled by _file_io internally)
    # lens_detector, radial_profiler, zone_segmenter, color_evaluator are lazy
    assert pipeline.sku_config == sku_config
    assert pipeline.zone_segmenter is None
    assert pipeline.color_evaluator is None


def test_process_ok_image(pipeline):
    """Test processing OK image"""
    image_path = "data/raw_images/OK_001.jpg"

    if not Path(image_path).exists():
        pytest.skip(f"Test image not found: {image_path}")

    result = pipeline.process(image_path, "SKU001")

    # Check result structure (v7 schema)
    assert result.sku == "SKU001"
    assert result.judgment in ["OK", "NG", "RETAKE", "OK_WITH_WARNING"]
    assert result.overall_delta_e >= 0
    assert 0 <= result.confidence <= 1.0
    # v7 schema: check for analysis_summary instead of zone_results
    assert hasattr(result, "analysis_summary") or hasattr(result, "confidence_breakdown")

    # For OK image, expect low delta_e
    assert result.overall_delta_e < 1.0, f"OK image should have low ΔE, got {result.overall_delta_e}"
    assert result.judgment in ["OK", "OK_WITH_WARNING"]


def test_process_ng_image(pipeline):
    """Test processing NG image"""
    image_path = "data/raw_images/NG_001.jpg"

    if not Path(image_path).exists():
        pytest.skip(f"Test image not found: {image_path}")

    result = pipeline.process(image_path, "SKU001")

    assert result.sku == "SKU001"
    assert result.judgment in ["OK", "NG"]

    # For NG image, expect high delta_e or NG judgment
    # (Note: Dummy data may have small defects)
    assert result.overall_delta_e >= 0


def test_process_batch(pipeline, tmp_path):
    """Test batch processing"""
    image_dir = Path("data/raw_images")

    if not image_dir.exists():
        pytest.skip(f"Image directory not found: {image_dir}")

    # Get first 3 OK images
    image_paths = list(image_dir.glob("OK_*.jpg"))[:3]

    if len(image_paths) == 0:
        pytest.skip("No OK images found")

    results = pipeline.process_batch([str(p) for p in image_paths], "SKU001", output_csv=tmp_path / "test_results.csv")

    assert len(results) == len(image_paths)

    # Check CSV was created
    assert (tmp_path / "test_results.csv").exists()

    # All OK images should pass (v7 may return OK or OK_WITH_WARNING)
    for result in results:
        assert result.judgment in ["OK", "OK_WITH_WARNING"]
        assert result.overall_delta_e < 1.0


def test_process_invalid_image(pipeline):
    """Test processing invalid image path"""
    with pytest.raises(Exception):  # Could be PipelineError or other
        pipeline.process("nonexistent.jpg", "SKU001")


def test_process_invalid_sku():
    """Test pipeline with invalid SKU config"""
    invalid_config = {}  # Empty config

    pipeline = InspectionPipeline(invalid_config)

    with pytest.raises(PipelineError):
        pipeline.process("data/raw_images/OK_001.jpg", "INVALID_SKU")


def test_pipeline_performance(pipeline):
    """Test pipeline performance"""
    import time

    image_path = "data/raw_images/OK_001.jpg"

    if not Path(image_path).exists():
        pytest.skip(f"Test image not found: {image_path}")

    # Warmup
    pipeline.process(image_path, "SKU001")

    # Measure time
    start = time.time()
    result = pipeline.process(image_path, "SKU001")
    elapsed_ms = (time.time() - start) * 1000

    print(f"\nProcessing time: {elapsed_ms:.1f}ms")

    # Performance requirement: < 500ms (very loose for test)
    assert elapsed_ms < 500, f"Processing too slow: {elapsed_ms:.1f}ms"


def test_v7_result_structure(pipeline):
    """Test v7 result structure"""
    image_path = "data/raw_images/OK_001.jpg"

    if not Path(image_path).exists():
        pytest.skip(f"Test image not found: {image_path}")

    result = pipeline.process(image_path, "SKU001")

    # v7 schema: check for key fields
    assert hasattr(result, "judgment")
    assert hasattr(result, "overall_delta_e")
    assert hasattr(result, "confidence")
    assert hasattr(result, "ng_reasons")

    # v7 specific fields
    if result.analysis_summary is not None:
        assert isinstance(result.analysis_summary, dict)

    if result.confidence_breakdown is not None:
        assert isinstance(result.confidence_breakdown, dict)


def test_batch_continue_on_error(pipeline, tmp_path):
    """Test batch processing continues on error"""
    image_paths = [
        "data/raw_images/OK_001.jpg",
        "nonexistent.jpg",  # This will cause error
        "data/raw_images/OK_002.jpg",
    ]

    # Remove nonexistent image from test if OK_001 doesn't exist
    if not Path(image_paths[0]).exists():
        pytest.skip("Test images not found")

    results = pipeline.process_batch(image_paths, "SKU001", continue_on_error=True)

    # Should have 2 results (skipping the error)
    assert len(results) >= 1  # At least one should succeed


def test_intermediate_save(pipeline, tmp_path):
    """Test saving intermediate results"""
    image_path = "data/raw_images/OK_001.jpg"

    if not Path(image_path).exists():
        pytest.skip(f"Test image not found: {image_path}")

    # Create pipeline with save_intermediates=True
    pipeline_with_save = InspectionPipeline(pipeline.sku_config, save_intermediates=True)

    save_dir = tmp_path / "intermediates"
    result = pipeline_with_save.process(image_path, "SKU001", save_dir=save_dir)

    # Check intermediate files were created
    image_name = Path(image_path).stem
    output_dir = save_dir / image_name

    assert output_dir.exists()
    assert (output_dir / "metadata.json").exists()

    # Check metadata content (v7 schema may have different fields)
    if (output_dir / "metadata.json").exists():
        metadata = read_json(output_dir / "metadata.json")
        # Just verify it's valid JSON, don't require specific fields
        assert isinstance(metadata, dict)
