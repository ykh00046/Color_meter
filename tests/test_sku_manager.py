"""
Integration tests for SKU Manager
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from src.sku_manager import (
    InsufficientSamplesError,
    InvalidSkuDataError,
    SkuAlreadyExistsError,
    SkuConfigManager,
    SkuNotFoundError,
)


@pytest.fixture
def temp_db():
    """Create temporary SKU database directory"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def manager(temp_db):
    """Create SKU manager with temporary database"""
    return SkuConfigManager(db_path=temp_db)


@pytest.fixture
def sample_sku_data():
    """Sample SKU data for testing"""
    from datetime import datetime

    return {
        "sku_code": "SKU999",
        "description": "Test SKU",
        "default_threshold": 3.5,
        "zones": {"A": {"L": 70.0, "a": -10.0, "b": -30.0, "threshold": 4.0, "description": "Test zone A"}},
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "baseline_samples": 5,
            "calibration_method": "manual",
            "author": "test",
        },
    }


def test_create_sku_success(manager, sample_sku_data):
    """Test successful SKU creation"""
    sku_data = manager.create_sku(
        sku_code=sample_sku_data["sku_code"],
        description=sample_sku_data["description"],
        default_threshold=sample_sku_data["default_threshold"],
        zones=sample_sku_data["zones"],
    )

    assert sku_data["sku_code"] == sample_sku_data["sku_code"]
    assert sku_data["description"] == sample_sku_data["description"]
    assert sku_data["zones"]["A"]["L"] == 70.0
    assert "metadata" in sku_data
    assert sku_data["metadata"]["calibration_method"] == "manual"


def test_create_sku_already_exists(manager, sample_sku_data):
    """Test creating SKU that already exists"""
    manager.create_sku(sku_code=sample_sku_data["sku_code"], description=sample_sku_data["description"])

    with pytest.raises(SkuAlreadyExistsError):
        manager.create_sku(sku_code=sample_sku_data["sku_code"], description="Duplicate")


def test_get_sku_success(manager, sample_sku_data):
    """Test successful SKU retrieval"""
    created = manager.create_sku(
        sku_code=sample_sku_data["sku_code"], description=sample_sku_data["description"], zones=sample_sku_data["zones"]
    )

    retrieved = manager.get_sku(sample_sku_data["sku_code"])

    assert retrieved["sku_code"] == created["sku_code"]
    assert retrieved["zones"] == created["zones"]


def test_get_sku_not_found(manager):
    """Test getting non-existent SKU"""
    with pytest.raises(SkuNotFoundError):
        manager.get_sku("SKU999")


def test_update_sku_success(manager, sample_sku_data):
    """Test successful SKU update"""
    manager.create_sku(
        sku_code=sample_sku_data["sku_code"], description=sample_sku_data["description"], zones=sample_sku_data["zones"]
    )

    updated = manager.update_sku(
        sample_sku_data["sku_code"], {"description": "Updated description", "default_threshold": 4.5}
    )

    assert updated["description"] == "Updated description"
    assert updated["default_threshold"] == 4.5


def test_delete_sku_success(manager, sample_sku_data):
    """Test successful SKU deletion"""
    manager.create_sku(sku_code=sample_sku_data["sku_code"], description=sample_sku_data["description"])

    result = manager.delete_sku(sample_sku_data["sku_code"])
    assert result == True

    with pytest.raises(SkuNotFoundError):
        manager.get_sku(sample_sku_data["sku_code"])


def test_list_all_skus(manager):
    """Test listing all SKUs"""
    # Create multiple SKUs
    for i in range(3):
        manager.create_sku(sku_code=f"SKU{i:03d}", description=f"Test SKU {i}")

    skus = manager.list_all_skus()

    assert len(skus) == 3
    assert all("sku_code" in sku for sku in skus)
    assert all("description" in sku for sku in skus)
    assert all("zones_count" in sku for sku in skus)


def test_generate_baseline_single_zone(manager):
    """Test baseline generation with single zone images"""
    # Use existing SKU001 OK images
    image_dir = Path("data/raw_images")

    if not image_dir.exists():
        pytest.skip("Test images not available")

    ok_images = sorted(image_dir.glob("SKU001_OK_*.jpg"))[:5]

    if len(ok_images) < 3:
        pytest.skip("Not enough test images")

    sku_data = manager.generate_baseline(
        sku_code="SKU901",
        ok_images=ok_images,
        description="Test baseline generation",
        default_threshold=3.5,
        threshold_method="mean_plus_2std",
    )

    assert sku_data["sku_code"] == "SKU901"
    assert len(sku_data["zones"]) >= 1
    assert sku_data["metadata"]["baseline_samples"] == len(ok_images)
    assert sku_data["metadata"]["calibration_method"] == "auto_generated"

    # Check zone data
    for zone_name, zone_config in sku_data["zones"].items():
        assert "L" in zone_config
        assert "a" in zone_config
        assert "b" in zone_config
        assert "threshold" in zone_config
        assert 0 <= zone_config["L"] <= 100


def test_generate_baseline_multi_zone(manager):
    """Test baseline generation with multi-zone images"""
    # Use SKU002 or SKU003 images if available
    image_dir = Path("data/raw_images")

    if not image_dir.exists():
        pytest.skip("Test images not available")

    ok_images = sorted(image_dir.glob("SKU002_OK_*.jpg"))[:5]

    if len(ok_images) < 3:
        pytest.skip("Not enough test images")

    sku_data = manager.generate_baseline(
        sku_code="SKU902", ok_images=ok_images, description="Multi-zone test", threshold_method="mean_plus_2std"
    )

    assert sku_data["sku_code"] == "SKU902"
    assert sku_data["metadata"]["baseline_samples"] == len(ok_images)


def test_generate_baseline_insufficient_samples(manager):
    """Test baseline generation with insufficient samples"""
    image_dir = Path("data/raw_images")

    if not image_dir.exists():
        pytest.skip("Test images not available")

    ok_images = sorted(image_dir.glob("SKU001_OK_*.jpg"))[:2]  # Only 2 images

    if len(ok_images) < 2:
        pytest.skip("Not enough test images")

    with pytest.raises(InsufficientSamplesError):
        manager.generate_baseline(sku_code="SKU903", ok_images=ok_images, description="Should fail")


def test_threshold_calculation_mean_plus_2std(manager):
    """Test threshold calculation with mean_plus_2std method"""
    image_dir = Path("data/raw_images")

    if not image_dir.exists():
        pytest.skip("Test images not available")

    ok_images = sorted(image_dir.glob("SKU001_OK_*.jpg"))[:5]

    if len(ok_images) < 3:
        pytest.skip("Not enough test images")

    sku_data = manager.generate_baseline(
        sku_code="SKU904", ok_images=ok_images, default_threshold=3.5, threshold_method="mean_plus_2std"
    )

    # Threshold should be >= default_threshold (due to std addition)
    for zone_config in sku_data["zones"].values():
        assert zone_config["threshold"] >= 3.5


def test_validate_sku_schema_valid(manager, sample_sku_data):
    """Test SKU schema validation with valid data"""
    # This should not raise any exception
    is_valid = manager._validate_sku(sample_sku_data)
    assert is_valid == True


def test_validate_sku_schema_invalid_lab(manager):
    """Test SKU schema validation with invalid LAB values"""
    invalid_sku = {
        "sku_code": "SKU999",
        "description": "Invalid SKU",
        "default_threshold": 3.5,
        "zones": {"A": {"L": 150, "a": 0, "b": 0, "threshold": 4.0}},  # Invalid: L > 100
        "metadata": {
            "created_at": "2025-12-11",
            "last_updated": "2025-12-11",
            "baseline_samples": 5,
            "calibration_method": "manual",
        },
    }

    with pytest.raises(InvalidSkuDataError):
        manager._validate_sku(invalid_sku)


def test_invalid_sku_code_format(manager):
    """Test creating SKU with invalid code format"""
    with pytest.raises(InvalidSkuDataError):
        manager.create_sku(sku_code="INVALID_CODE", description="Invalid code")  # Should be SKU[0-9]+


def test_multi_sku_batch_processing():
    """Test batch processing with multiple SKUs"""
    from src.pipeline import InspectionPipeline
    from src.utils.file_io import read_json

    image_dir = Path("data/raw_images")
    config_dir = Path("config/sku_db")

    if not image_dir.exists() or not config_dir.exists():
        pytest.skip("Test data not available")

    # Test SKU001
    sku001_images = sorted(image_dir.glob("SKU001_OK_*.jpg"))[:3]
    if len(sku001_images) >= 3:
        sku001_config = read_json(config_dir / "SKU001.json")
        pipeline001 = InspectionPipeline(sku001_config)

        results001 = pipeline001.process_batch([str(p) for p in sku001_images], "SKU001")

        assert len(results001) == len(sku001_images)
        assert all(r.sku == "SKU001" for r in results001)

    # Test SKU002 if available
    sku002_images = sorted(image_dir.glob("SKU002_OK_*.jpg"))[:3]
    if len(sku002_images) >= 3 and (config_dir / "SKU002.json").exists():
        sku002_config = read_json(config_dir / "SKU002.json")
        pipeline002 = InspectionPipeline(sku002_config)

        results002 = pipeline002.process_batch([str(p) for p in sku002_images], "SKU002")

        assert len(results002) == len(sku002_images)
        assert all(r.sku == "SKU002" for r in results002)


def test_cli_sku_list_command():
    """Test CLI sku list command"""
    import subprocess

    result = subprocess.run(["python", "-m", "src.main", "sku", "list"], capture_output=True, text=True)

    assert result.returncode == 0
    assert "SKU Code" in result.stdout or "No SKUs found" in result.stdout


def test_cli_sku_show_command():
    """Test CLI sku show command"""
    import subprocess

    # Test with existing SKU001
    result = subprocess.run(["python", "-m", "src.main", "sku", "show", "SKU001"], capture_output=True, text=True)

    if Path("config/sku_db/SKU001.json").exists():
        assert result.returncode == 0
        assert "SKU001" in result.stdout
    else:
        assert result.returncode != 0
