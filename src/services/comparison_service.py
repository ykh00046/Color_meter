"""
Comparison Service Layer

Handles comparison between test samples and STD models.
Implements zone-based comparison with conservative judgment logic.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.models.std_models import STDModel, STDSample
from src.models.test_models import ComparisonResult, JudgmentStatus, TestSample
from src.models.user_models import AuditLog

logger = logging.getLogger(__name__)


class ComparisonServiceError(Exception):
    """Comparison Service operation errors"""

    pass


class ComparisonService:
    """
    Service for STD vs Test comparison.

    Responsibilities:
    - Compare test samples against STD models
    - Calculate zone-based scores
    - Determine judgment (PASS/FAIL/RETAKE/MANUAL_REVIEW)
    - Extract top failure reasons
    - Store comparison results in database
    """

    def __init__(self, db_session: Session):
        """
        Initialize Comparison Service.

        Args:
            db_session: SQLAlchemy database session
        """
        self.db = db_session
        logger.info("ComparisonService initialized")

    def compare(self, test_sample_id: int, std_model_id: Optional[int] = None) -> ComparisonResult:
        """
        Compare test sample against STD model.

        Workflow:
        1. Load TestSample
        2. Find active STD (if not provided)
        3. Load STD sample data
        4. Compare zones
        5. Calculate scores
        6. Determine judgment
        7. Extract failure reasons
        8. Save ComparisonResult
        9. Return result

        Args:
            test_sample_id: Test sample ID
            std_model_id: STD model ID (None = auto-match by SKU)

        Returns:
            ComparisonResult instance

        Raises:
            ComparisonServiceError: If comparison fails
        """
        start_time = time.time()

        try:
            logger.info(f"Starting comparison: test_sample_id={test_sample_id}, std_model_id={std_model_id}")

            # 1. Load TestSample
            test_sample = self.db.query(TestSample).filter(TestSample.id == test_sample_id).first()
            if not test_sample:
                raise ComparisonServiceError(f"Test sample not found: id={test_sample_id}")

            # 2. Find active STD (if not provided)
            if std_model_id is None:
                std_model = (
                    self.db.query(STDModel)
                    .filter(STDModel.sku_code == test_sample.sku_code, STDModel.is_active == True)
                    .first()
                )
                if not std_model:
                    raise ComparisonServiceError(f"No active STD found for SKU: {test_sample.sku_code}")
                std_model_id = std_model.id
            else:
                std_model = self.db.query(STDModel).filter(STDModel.id == std_model_id).first()
                if not std_model:
                    raise ComparisonServiceError(f"STD model not found: id={std_model_id}")

            # 3. Load STD sample data (MVP: use first sample)
            std_sample = std_model.samples[0] if std_model.samples else None
            if not std_sample:
                raise ComparisonServiceError(f"STD has no samples: id={std_model_id}")

            std_analysis = std_sample.analysis_result

            # 4. Compare zones
            test_analysis = test_sample.analysis_result
            zone_details = self._compare_zones(
                test_zones=test_analysis.get("zone_results", []), std_zones=std_analysis.get("zone_results", [])
            )

            # 5. Calculate scores
            scores = self._calculate_scores(
                zone_details=zone_details, test_confidence=test_analysis.get("confidence", 0)
            )

            # 6. Determine judgment
            judgment = self._determine_judgment(
                total_score=scores["total_score"], zone_details=zone_details, lens_detected=test_sample.lens_detected
            )

            # 7. Extract failure reasons
            failure_reasons = self._extract_failure_reasons(zone_details=zone_details, judgment=judgment)

            # 8. Create ComparisonResult
            processing_time_ms = int((time.time() - start_time) * 1000)

            comparison_result = ComparisonResult(
                test_sample_id=test_sample_id,
                std_model_id=std_model_id,
                total_score=scores["total_score"],
                zone_score=scores["zone_score"],
                ink_score=0.0,  # MVP: not implemented yet
                confidence_score=scores["confidence_score"],
                judgment=judgment,
                top_failure_reasons=failure_reasons,
                defect_classifications=None,  # P1: not implemented
                zone_details=zone_details,
                ink_details=None,  # MVP: not implemented
                alignment_details=None,  # P1: not implemented
                worst_case_metrics=None,  # P2: not implemented
                created_at=datetime.utcnow(),
                processing_time_ms=processing_time_ms,
            )
            self.db.add(comparison_result)
            self.db.flush()  # Get comparison_result.id

            logger.info(f"Created ComparisonResult: id={comparison_result.id}, judgment={judgment.value}")

            # 9. Record AuditLog
            audit_log = AuditLog(
                user_id=None,  # TODO: Get from authentication context
                action="COMPARISON_RUN",
                target_type="ComparisonResult",
                target_id=comparison_result.id,
                success=True,
                changes={
                    "test_sample_id": test_sample_id,
                    "std_model_id": std_model_id,
                    "judgment": judgment.value,
                    "total_score": scores["total_score"],
                    "processing_time_ms": processing_time_ms,
                },
                created_at=datetime.utcnow(),
            )
            self.db.add(audit_log)

            # 10. Commit transaction
            self.db.commit()

            logger.info(
                f"Comparison completed: id={comparison_result.id}, "
                f"score={scores['total_score']:.1f}, judgment={judgment.value}"
            )
            return comparison_result

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error during comparison: {e}", exc_info=True)
            raise ComparisonServiceError(f"Database error: {e}")
        except ComparisonServiceError:
            self.db.rollback()
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error during comparison: {e}", exc_info=True)
            raise ComparisonServiceError(f"Unexpected error: {e}")

    def get_comparison_result(self, comparison_id: int) -> Optional[ComparisonResult]:
        """
        Retrieve comparison result by ID.

        Args:
            comparison_id: Comparison result ID

        Returns:
            ComparisonResult or None if not found
        """
        try:
            return self.db.query(ComparisonResult).filter(ComparisonResult.id == comparison_id).first()
        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving comparison {comparison_id}: {e}")
            raise ComparisonServiceError(f"Failed to retrieve comparison: {e}")

    def list_comparison_results(
        self,
        sku_code: Optional[str] = None,
        judgment: Optional[JudgmentStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ComparisonResult]:
        """
        List comparison results with optional filtering.

        Args:
            sku_code: Filter by SKU code (optional)
            judgment: Filter by judgment (optional)
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of ComparisonResult instances
        """
        try:
            query = self.db.query(ComparisonResult)

            # Join with TestSample for SKU filtering
            if sku_code:
                query = query.join(TestSample).filter(TestSample.sku_code == sku_code)

            if judgment:
                query = query.filter(ComparisonResult.judgment == judgment)

            # Order by created_at descending (newest first)
            query = query.order_by(ComparisonResult.created_at.desc())

            # Apply pagination
            query = query.limit(limit).offset(offset)

            return query.all()

        except SQLAlchemyError as e:
            logger.error(f"Database error listing comparison results: {e}")
            raise ComparisonServiceError(f"Failed to list comparison results: {e}")

    def _compare_zones(self, test_zones: List[Dict[str, Any]], std_zones: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare zones between test and STD.

        Uses name-based matching (not zip) to handle mismatches.

        Args:
            test_zones: Test zone results from analysis_result
            std_zones: STD zone results from analysis_result

        Returns:
            zone_details: {
                "A": {
                    "color_score": 85.3,
                    "total_score": 85.3,
                    "delta_e": 2.5,
                    "measured_lab": [72.2, 137.3, 122.8],
                    "target_lab": [70.5, 135.1, 120.2]
                },
                ...
            }
        """
        zone_details = {}

        # Create STD zone map (name-based)
        std_map = {z["zone_name"]: z for z in std_zones}

        for tz in test_zones:
            zone_name = tz.get("zone_name", "Unknown")
            sz = std_map.get(zone_name)

            if not sz:
                # STD에 해당 Zone이 없음 → Error
                zone_details[zone_name] = {
                    "error": "STD_ZONE_MISSING",
                    "color_score": 0.0,
                    "total_score": 0.0,
                    "delta_e": 999.9,
                    "measured_lab": tz.get("measured_lab"),
                    "target_lab": None,
                }
                logger.warning(f"Zone {zone_name} not found in STD")
                continue

            # Calculate color similarity (ΔE → 0-100 score)
            delta_e = tz.get("delta_e", 999.9)
            color_score = max(0.0, 100.0 - delta_e * 10.0)  # ΔE=10 → 0점

            # Total zone score (MVP: color_score만 사용)
            total_score = color_score

            zone_details[zone_name] = {
                "color_score": color_score,
                "total_score": total_score,
                "delta_e": delta_e,
                "measured_lab": tz.get("measured_lab"),
                "target_lab": sz.get("target_lab"),
            }

        return zone_details

    def _calculate_scores(self, zone_details: Dict[str, Any], test_confidence: float) -> Dict[str, float]:
        """
        Calculate total scores.

        Args:
            zone_details: Zone comparison details
            test_confidence: InspectionPipeline confidence (0.0-1.0)

        Returns:
            {
                "zone_score": 82.5,
                "total_score": 82.5,
                "confidence_score": 88.0
            }
        """
        # Zone score (평균)
        zone_scores = [z["total_score"] for z in zone_details.values()]
        zone_score = sum(zone_scores) / len(zone_scores) if zone_scores else 0.0

        # Total score (MVP: zone_score와 동일)
        total_score = zone_score

        # Confidence score (InspectionPipeline 신뢰도)
        confidence_score = test_confidence * 100.0 if test_confidence else 0.0

        return {"zone_score": zone_score, "total_score": total_score, "confidence_score": confidence_score}

    def _determine_judgment(
        self, total_score: float, zone_details: Dict[str, Any], lens_detected: bool
    ) -> JudgmentStatus:
        """
        Determine judgment based on scores and conditions.

        Conservative logic:
        - MANUAL_REVIEW: Lens not detected OR zone errors
        - PASS: total_score >= 80 AND all zones >= 70
        - FAIL: total_score < 55 OR any zone < 45
        - RETAKE: 55 <= total_score < 80

        Args:
            total_score: Total comparison score
            zone_details: Zone comparison details
            lens_detected: Whether lens was detected

        Returns:
            JudgmentStatus
        """
        # Check for manual review cases
        if not lens_detected:
            logger.info("MANUAL_REVIEW: Lens not detected")
            return JudgmentStatus.MANUAL_REVIEW

        # Check for zone errors
        for zone_name, details in zone_details.items():
            if details.get("error"):
                logger.info(f"MANUAL_REVIEW: Zone error in {zone_name}")
                return JudgmentStatus.MANUAL_REVIEW

        # Check zone thresholds
        zone_scores = [z["total_score"] for z in zone_details.values()]
        min_zone_score = min(zone_scores) if zone_scores else 0.0

        # PASS: 높은 기준 (안정적)
        if total_score >= 80.0 and min_zone_score >= 70.0:
            logger.info(f"PASS: total={total_score:.1f}, min_zone={min_zone_score:.1f}")
            return JudgmentStatus.PASS

        # FAIL: 보수적 기준 (초기 현장 안정화)
        elif total_score < 55.0 or min_zone_score < 45.0:
            logger.info(f"FAIL: total={total_score:.1f}, min_zone={min_zone_score:.1f}")
            return JudgmentStatus.FAIL

        # RETAKE: 중간 구간 (넓게 설정)
        else:
            logger.info(f"RETAKE: total={total_score:.1f}, min_zone={min_zone_score:.1f}")
            return JudgmentStatus.RETAKE

    def _extract_failure_reasons(self, zone_details: Dict[str, Any], judgment: JudgmentStatus) -> List[Dict[str, Any]]:
        """
        Extract top 3 failure reasons.

        Args:
            zone_details: Zone comparison details
            judgment: Judgment status

        Returns:
            List of failure reasons (max 3), ranked by severity
        """
        if judgment == JudgmentStatus.PASS:
            return []

        issues = []

        for zone_name, details in zone_details.items():
            # Zone error
            if details.get("error"):
                issues.append(
                    {
                        "category": "ZONE_ERROR",
                        "zone": zone_name,
                        "message": f"Zone {zone_name}: {details['error']}",
                        "severity": 100.0,
                        "score": 0.0,
                    }
                )
                continue

            # Color issues
            color_score = details.get("color_score", 0.0)
            if color_score < 70.0:
                severity = 100.0 - color_score
                delta_e = details.get("delta_e", 0.0)
                issues.append(
                    {
                        "category": "ZONE_COLOR",
                        "zone": zone_name,
                        "message": f"Zone {zone_name}: ΔE={delta_e:.1f} (score={color_score:.1f})",
                        "severity": severity,
                        "score": color_score,
                    }
                )

        # Sort by severity (descending)
        issues.sort(key=lambda x: x["severity"], reverse=True)

        # Add rank to top 3
        for i, issue in enumerate(issues[:3]):
            issue["rank"] = i + 1

        return issues[:3]
