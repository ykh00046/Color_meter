"""
Test P2: Worst-Case Metrics Implementation

End-to-end test for worst-case metrics calculation.
"""

import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.services.comparison_service import ComparisonService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = "sqlite:///color_meter.db"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def main():
    """Test P2 worst-case metrics"""
    db = SessionLocal()

    try:
        logger.info("=" * 80)
        logger.info("P2: Worst-Case Metrics Test")
        logger.info("=" * 80)

        # Initialize service
        service = ComparisonService(db)

        # Run comparison (test_sample_id=3, std_model_id=1)
        logger.info("\nRunning comparison with worst-case metrics...")
        result = service.compare(test_sample_id=3, std_model_id=1)

        # Display worst-case metrics
        logger.info("\n" + "=" * 80)
        logger.info("Worst-Case Metrics Results")
        logger.info("=" * 80)

        if result.worst_case_metrics:
            wc = result.worst_case_metrics

            # Percentile statistics
            logger.info("\n📊 Percentile Statistics:")
            logger.info(f"  Mean ΔE:    {wc['percentiles']['mean']:.2f}")
            logger.info(f"  Median ΔE:  {wc['percentiles']['median']:.2f}")
            logger.info(f"  P95 ΔE:     {wc['percentiles']['p95']:.2f}")
            logger.info(f"  P99 ΔE:     {wc['percentiles']['p99']:.2f}")
            logger.info(f"  Max ΔE:     {wc['percentiles']['max']:.2f}")
            logger.info(f"  Std Dev:    {wc['percentiles']['std']:.2f}")

            # Hotspot information
            logger.info(f"\n🔥 Hotspot Detection:")
            logger.info(f"  Total Hotspots: {wc['hotspot_count']}")
            logger.info(f"  Coverage Ratio: {wc['coverage_ratio']:.2%}")
            logger.info(f"  Worst Zone:     {wc['worst_zone']}")

            # Individual hotspots
            if wc["hotspots"]:
                logger.info(f"\n  Top {len(wc['hotspots'])} Hotspots:")
                for i, hotspot in enumerate(wc["hotspots"], 1):
                    logger.info(f"    {i}. [{hotspot['severity']}] Zone {hotspot['zone']}:")
                    logger.info(f"       Area: {hotspot['area']} pixels")
                    logger.info(f"       Centroid: ({hotspot['centroid'][0]:.1f}, {hotspot['centroid'][1]:.1f})")
                    logger.info(f"       Mean ΔE: {hotspot['mean_delta_e']:.2f}")
                    logger.info(f"       Max ΔE:  {hotspot['max_delta_e']:.2f}")
            else:
                logger.info("  No hotspots detected (excellent quality!)")

            # Overall assessment
            logger.info("\n" + "=" * 80)
            logger.info("Overall Comparison Results")
            logger.info("=" * 80)
            logger.info(f"Total Score:  {result.total_score:.1f}")
            logger.info(f"Zone Score:   {result.zone_score:.1f}")
            logger.info(f"Ink Score:    {result.ink_score:.1f}")
            logger.info(f"Profile Score: {result.profile_score:.1f}")
            logger.info(f"Judgment:     {result.judgment}")
            logger.info(f"Is Pass:      {result.is_pass}")

            logger.info("\n" + "=" * 80)
            logger.info("✓ P2 Test PASSED - Worst-case metrics calculated successfully!")
            logger.info("=" * 80)

        else:
            logger.error("❌ P2 Test FAILED - worst_case_metrics not found in result")
            return False

        return True

    except Exception as e:
        logger.error(f"❌ P2 Test FAILED: {e}", exc_info=True)
        return False
    finally:
        db.close()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
