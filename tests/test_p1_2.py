"""
P1-2 Radial Profile Comparison Test Script

Tests the end-to-end radial profile comparison feature.
"""

import json
import logging
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.models.std_models import STDModel, STDSample
from src.models.test_models import ComparisonResult, TestSample
from src.services.comparison_service import ComparisonService
from src.services.std_service import STDService
from src.services.test_service import TestService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = "sqlite:///color_meter.db"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def test_radial_profile_comparison():
    """Test radial profile comparison end-to-end"""
    db = SessionLocal()

    try:
        logger.info("=" * 80)
        logger.info("P1-2 Radial Profile Comparison Test")
        logger.info("=" * 80)

        # 1. Get existing STD sample
        std_sample = db.query(STDSample).filter(STDSample.id == 1).first()
        if not std_sample:
            logger.error("STD Sample ID=1 not found")
            return

        logger.info(f"STD Sample ID=1, STD Model ID={std_sample.std_model_id}")

        # Check if radial_profile exists
        std_analysis = std_sample.analysis_result
        has_radial_profile = "radial_profile" in std_analysis if std_analysis else False
        logger.info(f"STD has radial_profile: {has_radial_profile}")

        if has_radial_profile:
            profile_length = len(std_analysis["radial_profile"]["L"])
            logger.info(f"STD radial_profile length: {profile_length} points")
        else:
            logger.warning("STD sample does not have radial_profile - needs re-registration")

        # 2. Get existing Test sample
        test_sample = db.query(TestSample).filter(TestSample.id == 3).first()
        if not test_sample:
            logger.error("Test Sample ID=3 not found")
            return

        logger.info(f"Test Sample ID=3, SKU={test_sample.sku_code}")

        # Check if radial_profile exists
        test_analysis = test_sample.analysis_result
        has_test_profile = "radial_profile" in test_analysis if test_analysis else False
        logger.info(f"Test has radial_profile: {has_test_profile}")

        if has_test_profile:
            profile_length = len(test_analysis["radial_profile"]["L"])
            logger.info(f"Test radial_profile length: {profile_length} points")
        else:
            logger.warning("Test sample does not have radial_profile - needs re-registration")

        # 3. Run comparison
        logger.info("\n" + "=" * 80)
        logger.info("Running comparison...")
        logger.info("=" * 80)

        comparison_service = ComparisonService(db)
        result = comparison_service.compare(test_sample_id=3, std_model_id=1)

        logger.info(f"\nComparison Result ID: {result.id}")
        logger.info(f"Judgment: {result.judgment.value}")
        logger.info(f"Total Score: {result.total_score:.2f}")
        logger.info(f"Zone Score: {result.zone_score:.2f}")
        logger.info(f"Ink Score: {result.ink_score:.2f}")
        logger.info(f"Profile Score: {result.profile_score:.2f}")
        logger.info(f"Confidence Score: {result.confidence_score:.2f}")

        # 4. Check profile_details
        if result.profile_details:
            logger.info("\n" + "=" * 80)
            logger.info("Profile Details:")
            logger.info("=" * 80)
            logger.info(json.dumps(result.profile_details, indent=2))
        else:
            logger.warning("No profile_details in comparison result")

        logger.info("\n" + "=" * 80)
        logger.info("Test completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    test_radial_profile_comparison()
