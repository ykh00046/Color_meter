"""
Test Service Layer

Handles test sample registration and management.
Bridges InspectionPipeline and database layer.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.models.test_models import TestSample
from src.models.user_models import AuditLog
from src.pipeline import InspectionPipeline
from src.sku_manager import SkuConfigManager

logger = logging.getLogger(__name__)


class TestServiceError(Exception):
    """Test Service operation errors"""

    pass


class TestService:
    """
    Service for Test Sample management.

    Responsibilities:
    - Register test sample images and analyze them
    - Store test sample data in database
    - Retrieve and manage test samples
    - Audit logging for all operations
    """

    def __init__(self, db_session: Session, sku_manager: SkuConfigManager):
        """
        Initialize Test Service.

        Args:
            db_session: SQLAlchemy database session
            sku_manager: SKU configuration manager
        """
        self.db = db_session
        self.sku_manager = sku_manager
        logger.info("TestService initialized")

    def register_test_sample(
        self,
        sku_code: str,
        image_path: str,
        batch_number: Optional[str] = None,
        sample_id: Optional[str] = None,
        operator: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> TestSample:
        """
        Register a new test sample.

        Workflow:
        1. Validate image path
        2. Load SKU configuration
        3. Run InspectionPipeline on test image
        4. Create TestSample with analysis results
        5. Record AuditLog
        6. Commit to database

        Args:
            sku_code: SKU code (e.g., 'SKU001')
            image_path: Path to test image file
            batch_number: Optional batch number
            sample_id: Optional unique sample identifier
            operator: Optional operator name
            notes: Optional notes about this sample

        Returns:
            TestSample: Created test sample instance

        Raises:
            TestServiceError: If registration fails
        """
        try:
            logger.info(f"Registering test sample: sku={sku_code}, image={image_path}")

            # 1. Validate image path
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                raise TestServiceError(f"Image file not found: {image_path}")

            # Get file info
            file_size_bytes = image_path_obj.stat().st_size

            # 2. Load SKU configuration
            try:
                sku_config = self.sku_manager.get_sku(sku_code)
            except Exception as e:
                raise TestServiceError(f"Failed to load SKU config for {sku_code}: {e}")

            # 3. Run InspectionPipeline
            logger.info(f"Running InspectionPipeline for test image: {image_path}")
            try:
                pipeline = InspectionPipeline(sku_config=sku_config)
                inspection_result = pipeline.process(image_path=str(image_path), sku=sku_code)
            except Exception as e:
                raise TestServiceError(f"InspectionPipeline failed: {e}")

            # 4. Check lens detection
            lens_detected = inspection_result.lens_detection is not None
            lens_detection_score = None

            # Get image dimensions from lens_detection if available
            image_width = None
            image_height = None
            if inspection_result.lens_detection:
                # Get from image if available
                if inspection_result.image is not None:
                    image_height, image_width = inspection_result.image.shape[:2]

            # 5. Convert InspectionResult to JSON-serializable dict
            analysis_dict = self._inspection_result_to_dict(inspection_result)

            # 6. Create TestSample
            test_sample = TestSample(
                sku_code=sku_code,
                batch_number=batch_number,
                sample_id=sample_id,
                image_path=str(image_path),
                analysis_result=analysis_dict,
                lens_detected=lens_detected,
                lens_detection_score=lens_detection_score,
                created_at=datetime.utcnow(),
                operator=operator,
                notes=notes,
                file_size_bytes=file_size_bytes,
                image_width=image_width,
                image_height=image_height,
            )
            self.db.add(test_sample)
            self.db.flush()  # Get test_sample.id

            logger.info(f"Created TestSample: id={test_sample.id}")

            # 7. Record AuditLog
            audit_log = AuditLog(
                user_id=None,  # TODO: Get from authentication context
                action="TEST_REGISTER",
                target_type="TestSample",
                target_id=test_sample.id,
                success=True,
                changes={
                    "sku_code": sku_code,
                    "batch_number": batch_number,
                    "sample_id": sample_id,
                    "image_path": str(image_path),
                    "judgment": inspection_result.judgment,
                    "lens_detected": lens_detected,
                },
                created_at=datetime.utcnow(),
            )
            self.db.add(audit_log)

            # 8. Commit transaction
            self.db.commit()

            logger.info(f"Test sample registered successfully: id={test_sample.id}, sku={sku_code}")
            return test_sample

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error during test registration: {e}", exc_info=True)
            raise TestServiceError(f"Database error: {e}")
        except TestServiceError:
            self.db.rollback()
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error during test registration: {e}", exc_info=True)
            raise TestServiceError(f"Unexpected error: {e}")

    def get_test_sample(self, test_id: int) -> Optional[TestSample]:
        """
        Retrieve test sample by ID.

        Args:
            test_id: Test sample ID

        Returns:
            TestSample or None if not found
        """
        try:
            test_sample = self.db.query(TestSample).filter(TestSample.id == test_id).first()
            return test_sample
        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving test sample {test_id}: {e}")
            raise TestServiceError(f"Failed to retrieve test sample: {e}")

    def list_test_samples(
        self, sku_code: Optional[str] = None, batch_number: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[TestSample]:
        """
        List test samples with optional filtering.

        Args:
            sku_code: Filter by SKU code (optional)
            batch_number: Filter by batch number (optional)
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of TestSample instances
        """
        try:
            query = self.db.query(TestSample)

            if sku_code:
                query = query.filter(TestSample.sku_code == sku_code)

            if batch_number:
                query = query.filter(TestSample.batch_number == batch_number)

            # Order by created_at descending (newest first)
            query = query.order_by(TestSample.created_at.desc())

            # Apply pagination
            query = query.limit(limit).offset(offset)

            return query.all()

        except SQLAlchemyError as e:
            logger.error(f"Database error listing test samples: {e}")
            raise TestServiceError(f"Failed to list test samples: {e}")

    def _inspection_result_to_dict(self, result) -> Dict[str, Any]:
        """
        Convert InspectionResult to JSON-serializable dictionary.

        Args:
            result: InspectionResult instance

        Returns:
            Dictionary representation
        """
        # Extract key fields from InspectionResult
        return {
            "sku": result.sku,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None,
            "judgment": result.judgment,
            "overall_delta_e": result.overall_delta_e,
            "confidence": result.confidence,
            "zone_results": (
                [
                    {
                        "zone_name": zr.zone_name,
                        "measured_lab": zr.measured_lab,
                        "target_lab": zr.target_lab,
                        "delta_e": zr.delta_e,
                        "threshold": zr.threshold,
                        "is_ok": zr.is_ok,
                        "pixel_count": zr.pixel_count,
                        "diff": zr.diff,
                        "std_lab": zr.std_lab,
                        "chroma_stats": zr.chroma_stats,
                        "internal_uniformity": zr.internal_uniformity,
                        "uniformity_grade": zr.uniformity_grade,
                    }
                    for zr in result.zone_results
                ]
                if result.zone_results
                else []
            ),
            "ng_reasons": result.ng_reasons,
            "next_actions": result.next_actions,
            "decision_trace": result.decision_trace,
            "diagnostics": result.diagnostics,
            "warnings": result.warnings,
            "suggestions": result.suggestions,
            "retake_reasons": result.retake_reasons,
            "analysis_summary": result.analysis_summary,
            "confidence_breakdown": result.confidence_breakdown,
            "risk_factors": result.risk_factors,
            "ink_analysis": result.ink_analysis,
        }
