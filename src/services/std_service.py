"""
STD Service Layer

Handles STD (Standard) registration, analysis, and management.
Bridges InspectionPipeline and database layer.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.models.std_models import STDModel, STDSample
from src.models.user_models import AuditLog
from src.pipeline import InspectionPipeline
from src.sku_manager import SkuConfigManager

logger = logging.getLogger(__name__)


class STDServiceError(Exception):
    """STD Service operation errors"""

    pass


class STDService:
    """
    Service for STD management.

    Responsibilities:
    - Register STD images and analyze them
    - Store STD profiles in database
    - Retrieve and manage STD models
    - Audit logging for all operations
    """

    def __init__(self, db_session: Session, sku_manager: SkuConfigManager):
        """
        Initialize STD Service.

        Args:
            db_session: SQLAlchemy database session
            sku_manager: SKU configuration manager
        """
        self.db = db_session
        self.sku_manager = sku_manager
        logger.info("STDService initialized")

    def register_std(
        self,
        sku_code: str,
        image_path: str,
        version: str = "v1.0",
        notes: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> STDModel:
        """
        Register a new STD (Standard) sample.

        For MVP: Single sample mode (n_samples=1)
        For P2: Multi-sample mode (n_samples=5-10)

        Workflow:
        1. Load SKU configuration
        2. Run InspectionPipeline on STD image
        3. Create STDModel with n_samples=1
        4. Create STDSample with analysis results
        5. Record AuditLog
        6. Commit to database

        Args:
            sku_code: SKU code (e.g., 'SKU001')
            image_path: Path to STD image file
            version: STD version string (default: 'v1.0')
            notes: Optional notes about this STD
            user_id: Optional user ID for audit logging

        Returns:
            STDModel: Created STD model instance

        Raises:
            STDServiceError: If registration fails
        """
        try:
            logger.info(f"Registering STD: sku={sku_code}, image={image_path}, version={version}")

            # 1. Validate image path
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                raise STDServiceError(f"Image file not found: {image_path}")

            # 2. Load SKU configuration
            try:
                sku_config = self.sku_manager.get_sku(sku_code)
            except Exception as e:
                raise STDServiceError(f"Failed to load SKU config for {sku_code}: {e}")

            # 3. Run InspectionPipeline
            logger.info(f"Running InspectionPipeline for STD image: {image_path}")
            try:
                pipeline = InspectionPipeline(sku_config=sku_config)
                inspection_result = pipeline.process(image_path=str(image_path), sku=sku_code)
            except Exception as e:
                raise STDServiceError(f"InspectionPipeline failed: {e}")

            # 4. Check if analysis was successful
            # For STD, we expect OK or OK_WITH_WARNING judgment
            # NG or RETAKE STDs should be rejected
            if inspection_result.judgment in ["NG", "RETAKE"]:
                logger.warning(
                    f"STD image quality issue: judgment={inspection_result.judgment}, "
                    f"reasons={inspection_result.ng_reasons or inspection_result.retake_reasons}"
                )
                # Allow registration but log warning
                # Production system may want to reject NG/RETAKE STDs

            # 5. Deactivate previous active STD for this SKU+version (if exists)
            existing_std = (
                self.db.query(STDModel)
                .filter(STDModel.sku_code == sku_code, STDModel.version == version, STDModel.is_active == True)
                .first()
            )
            if existing_std:
                logger.info(f"Deactivating existing STD: id={existing_std.id}")
                existing_std.is_active = False

            # 6. Create STDModel (MVP: n_samples=1)
            std_model = STDModel(
                sku_code=sku_code,
                version=version,
                n_samples=1,  # MVP: single sample
                notes=notes,
                is_active=True,
                created_at=datetime.utcnow(),
            )
            self.db.add(std_model)
            self.db.flush()  # Get std_model.id

            logger.info(f"Created STDModel: id={std_model.id}")

            # 7. Create STDSample with analysis result
            # Convert InspectionResult to JSON-serializable dict
            analysis_dict = self._inspection_result_to_dict(inspection_result)

            std_sample = STDSample(
                std_model_id=std_model.id,
                sample_index=1,  # 1-based index (first and only sample in MVP)
                image_path=str(image_path),
                analysis_result=analysis_dict,
                created_at=datetime.utcnow(),
            )
            self.db.add(std_sample)

            logger.info(f"Created STDSample: sample_index=1")

            # 8. Record AuditLog
            audit_log = AuditLog(
                user_id=user_id,
                action="STD_REGISTER",
                target_type="STDModel",
                target_id=std_model.id,
                success=True,
                changes={
                    "sku_code": sku_code,
                    "version": version,
                    "image_path": str(image_path),
                    "judgment": inspection_result.judgment,
                    "n_samples": 1,
                },
                created_at=datetime.utcnow(),
            )
            self.db.add(audit_log)

            # 9. Commit transaction
            self.db.commit()

            logger.info(f"STD registered successfully: id={std_model.id}, sku={sku_code}")
            return std_model

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error during STD registration: {e}", exc_info=True)
            raise STDServiceError(f"Database error: {e}")
        except STDServiceError:
            self.db.rollback()
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error during STD registration: {e}", exc_info=True)
            raise STDServiceError(f"Unexpected error: {e}")

    def get_std_by_id(self, std_id: int) -> Optional[STDModel]:
        """
        Retrieve STD model by ID.

        Args:
            std_id: STD model ID

        Returns:
            STDModel or None if not found
        """
        try:
            std_model = self.db.query(STDModel).filter(STDModel.id == std_id).first()
            return std_model
        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving STD {std_id}: {e}")
            raise STDServiceError(f"Failed to retrieve STD: {e}")

    def list_stds(
        self, sku_code: Optional[str] = None, active_only: bool = True, limit: int = 100, offset: int = 0
    ) -> List[STDModel]:
        """
        List STD models with optional filtering.

        Args:
            sku_code: Filter by SKU code (optional)
            active_only: Only return active STDs (default: True)
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of STDModel instances
        """
        try:
            query = self.db.query(STDModel)

            if sku_code:
                query = query.filter(STDModel.sku_code == sku_code)

            if active_only:
                query = query.filter(STDModel.is_active == True)

            # Order by created_at descending (newest first)
            query = query.order_by(STDModel.created_at.desc())

            # Apply pagination
            query = query.limit(limit).offset(offset)

            return query.all()

        except SQLAlchemyError as e:
            logger.error(f"Database error listing STDs: {e}")
            raise STDServiceError(f"Failed to list STDs: {e}")

    def get_active_std_for_sku(self, sku_code: str, version: str = "v1.0") -> Optional[STDModel]:
        """
        Get the currently active STD for a SKU.

        Args:
            sku_code: SKU code
            version: STD version (default: 'v1.0')

        Returns:
            STDModel or None if not found
        """
        try:
            std_model = (
                self.db.query(STDModel)
                .filter(STDModel.sku_code == sku_code, STDModel.version == version, STDModel.is_active == True)
                .first()
            )
            return std_model
        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving active STD for {sku_code}: {e}")
            raise STDServiceError(f"Failed to retrieve active STD: {e}")

    def deactivate_std(self, std_id: int, user_id: Optional[int] = None) -> bool:
        """
        Deactivate a STD model (soft delete).

        Args:
            std_id: STD model ID
            user_id: User ID for audit logging (optional)

        Returns:
            True if successful, False if not found
        """
        try:
            std_model = self.db.query(STDModel).filter(STDModel.id == std_id).first()

            if not std_model:
                return False

            std_model.is_active = False

            # Record AuditLog
            audit_log = AuditLog(
                user_id=user_id,
                action="STD_DEACTIVATE",
                target_type="STDModel",
                target_id=std_id,
                success=True,
                changes={"sku_code": std_model.sku_code, "version": std_model.version},
                created_at=datetime.utcnow(),
            )
            self.db.add(audit_log)

            self.db.commit()
            logger.info(f"Deactivated STD: id={std_id}")
            return True

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error deactivating STD {std_id}: {e}")
            raise STDServiceError(f"Failed to deactivate STD: {e}")

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
