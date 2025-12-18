"""
Update existing STD and Test samples with radial_profile data

Re-analyzes existing samples to include radial_profile in analysis_result.
"""

import json
import logging
from pathlib import Path

import cv2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.lens_detector import DetectorConfig, LensDetector
from src.core.zone_analyzer_2d import analyze_lens_zones_2d
from src.models.std_models import STDModel, STDSample
from src.models.test_models import TestSample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = "sqlite:///color_meter.db"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def update_std_sample(db, sample_id: int):
    """Update STD sample with radial_profile"""
    sample = db.query(STDSample).filter(STDSample.id == sample_id).first()
    if not sample:
        logger.error(f"STD Sample {sample_id} not found")
        return False

    logger.info(f"Updating STD Sample {sample_id}...")
    logger.info(f"Image path: {sample.image_path}")

    # Load image
    img_path = Path(sample.image_path)
    if not img_path.exists():
        logger.error(f"Image not found: {img_path}")
        return False

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        logger.error(f"Failed to load image: {img_path}")
        return False

    # Get STD model and SKU config
    std_model = sample.std_model
    sku_code = std_model.sku_code

    # Load SKU config (simplified - use hardcoded for testing)
    sku_config = {
        "sku_code": sku_code,
        "zones": {
            "A": {"L": 30.0, "a": 0.0, "b": 0.0, "delta_e_threshold": 5.0},
            "B": {"L": 50.0, "a": 0.0, "b": 0.0, "delta_e_threshold": 5.0},
            "C": {"L": 70.0, "a": 0.0, "b": 0.0, "delta_e_threshold": 5.0},
        },
        "params": {"expected_zones": 3},
    }

    # Detect lens
    detector = LensDetector(DetectorConfig())
    lens_detection = detector.detect(img_bgr)

    if not lens_detection:
        logger.error("Lens detection failed")
        return False

    logger.info(
        f"Lens detected: center=({lens_detection.center_x:.1f}, "
        f"{lens_detection.center_y:.1f}), radius={lens_detection.radius:.1f}"
    )

    # Analyze with zone_analyzer_2d (includes radial_profile)
    result, debug_info = analyze_lens_zones_2d(img_bgr, lens_detection, sku_config)

    # Convert result to dict
    from dataclasses import asdict
    from datetime import datetime as dt

    result_dict = {}
    for field in result.__dataclass_fields__:
        value = getattr(result, field)
        if isinstance(value, dt):
            value = value.isoformat()
        elif hasattr(value, "__dataclass_fields__"):  # Nested dataclass
            value = asdict(value)
        elif isinstance(value, list):
            value = [asdict(item) if hasattr(item, "__dataclass_fields__") else item for item in value]
        result_dict[field] = value

    # Ensure radial_profile is included
    if "radial_profile" not in result_dict or result_dict["radial_profile"] is None:
        logger.error("radial_profile not generated!")
        return False

    logger.info(f"radial_profile generated: {len(result_dict['radial_profile']['L'])} points")

    # Update database
    sample.analysis_result = result_dict
    db.commit()
    logger.info(f"STD Sample {sample_id} updated successfully!")
    return True


def update_test_sample(db, sample_id: int):
    """Update Test sample with radial_profile"""
    sample = db.query(TestSample).filter(TestSample.id == sample_id).first()
    if not sample:
        logger.error(f"Test Sample {sample_id} not found")
        return False

    logger.info(f"Updating Test Sample {sample_id}...")
    logger.info(f"Image path: {sample.image_path}")

    # Load image
    img_path = Path(sample.image_path)
    if not img_path.exists():
        logger.error(f"Image not found: {img_path}")
        return False

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        logger.error(f"Failed to load image: {img_path}")
        return False

    # Load SKU config
    sku_code = sample.sku_code
    sku_config = {
        "sku_code": sku_code,
        "zones": {
            "A": {"L": 30.0, "a": 0.0, "b": 0.0, "delta_e_threshold": 5.0},
            "B": {"L": 50.0, "a": 0.0, "b": 0.0, "delta_e_threshold": 5.0},
            "C": {"L": 70.0, "a": 0.0, "b": 0.0, "delta_e_threshold": 5.0},
        },
        "params": {"expected_zones": 3},
    }

    # Detect lens
    detector = LensDetector(DetectorConfig())
    lens_detection = detector.detect(img_bgr)

    if not lens_detection:
        logger.error("Lens detection failed")
        return False

    logger.info(
        f"Lens detected: center=({lens_detection.center_x:.1f}, "
        f"{lens_detection.center_y:.1f}), radius={lens_detection.radius:.1f}"
    )

    # Analyze with zone_analyzer_2d (includes radial_profile)
    result, debug_info = analyze_lens_zones_2d(img_bgr, lens_detection, sku_config)

    # Convert result to dict
    from dataclasses import asdict
    from datetime import datetime as dt

    result_dict = {}
    for field in result.__dataclass_fields__:
        value = getattr(result, field)
        if isinstance(value, dt):
            value = value.isoformat()
        elif hasattr(value, "__dataclass_fields__"):  # Nested dataclass
            value = asdict(value)
        elif isinstance(value, list):
            value = [asdict(item) if hasattr(item, "__dataclass_fields__") else item for item in value]
        result_dict[field] = value

    # Ensure radial_profile is included
    if "radial_profile" not in result_dict or result_dict["radial_profile"] is None:
        logger.error("radial_profile not generated!")
        return False

    logger.info(f"radial_profile generated: {len(result_dict['radial_profile']['L'])} points")

    # Update database
    sample.analysis_result = result_dict
    db.commit()
    logger.info(f"Test Sample {sample_id} updated successfully!")
    return True


def main():
    """Main function"""
    db = SessionLocal()

    try:
        logger.info("=" * 80)
        logger.info("Updating samples with radial_profile")
        logger.info("=" * 80)

        # Update STD Sample 1
        if update_std_sample(db, 1):
            logger.info("✓ STD Sample 1 updated")
        else:
            logger.error("✗ STD Sample 1 update failed")

        logger.info("")

        # Update Test Sample 3
        if update_test_sample(db, 3):
            logger.info("✓ Test Sample 3 updated")
        else:
            logger.error("✗ Test Sample 3 update failed")

        logger.info("\n" + "=" * 80)
        logger.info("Update completed!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Update failed: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    main()
