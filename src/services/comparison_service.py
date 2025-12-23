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

            # 4.5. Compare inks (M3)
            ink_details = None
            if test_analysis.get("ink_analysis") and std_analysis.get("ink_analysis"):
                logger.info("Running ink comparison (M3)")
                ink_details = self._compare_inks(
                    test_ink_analysis=test_analysis["ink_analysis"], std_ink_analysis=std_analysis["ink_analysis"]
                )
                logger.info(
                    f"Ink comparison complete: ink_score={ink_details.get('ink_score', 0.0) if ink_details else 0.0}"
                )
            else:
                logger.warning("Ink analysis not available in test or STD sample")

            # 4.6. Compare radial profiles (P1-2)
            profile_details = None
            if test_analysis.get("radial_profile") and std_analysis.get("radial_profile"):
                logger.info("Running radial profile comparison (P1-2)")
                from src.core.profile_comparison import compare_radial_profiles

                profile_details = compare_radial_profiles(
                    test_profile=test_analysis["radial_profile"],
                    std_profile=std_analysis["radial_profile"],
                )
                profile_score_value = profile_details.get("profile_score", 0.0) if profile_details else 0.0
                logger.info(f"Profile comparison complete: profile_score={profile_score_value}")
            else:
                logger.warning("Radial profile not available in test or STD sample")

            # 4.7. Calculate worst-case metrics (P2)
            worst_case_metrics = None
            if test_analysis.get("zone_results") and std_analysis.get("zone_results"):
                logger.info("Calculating worst-case metrics (P2)")
                worst_case_metrics = self._calculate_worst_case_metrics(
                    test_analysis=test_analysis, std_analysis=std_analysis, zone_details=zone_details
                )
                if worst_case_metrics:
                    logger.info(
                        f"Worst-case metrics complete: p95={worst_case_metrics['percentiles']['p95']:.2f}, "
                        f"hotspots={worst_case_metrics['hotspot_count']}"
                    )
            else:
                logger.warning("Zone results not available for worst-case metrics")

            # 5. Calculate scores
            scores = self._calculate_scores(
                zone_details=zone_details,
                test_confidence=test_analysis.get("confidence", 0),
                ink_details=ink_details,
                profile_details=profile_details,
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
                ink_score=scores.get("ink_score", 0.0),  # M3: ink comparison
                profile_score=scores.get("profile_score", 0.0),  # P1-2: radial profile comparison
                confidence_score=scores["confidence_score"],
                judgment=judgment,
                top_failure_reasons=failure_reasons,
                defect_classifications=None,  # P1: not implemented
                zone_details=zone_details,
                ink_details=ink_details,  # M3: ink comparison details
                profile_details=profile_details,  # P1-2: radial profile comparison details
                alignment_details=None,  # P1: not implemented
                worst_case_metrics=worst_case_metrics,  # P2: worst-case metrics
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

    def _calculate_scores(
        self,
        zone_details: Dict[str, Any],
        test_confidence: float,
        ink_details: Optional[Dict[str, Any]] = None,
        profile_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Calculate total scores.

        M3 Update: Includes ink_score in total_score calculation.
        P1-2 Update: Includes profile_score in total_score calculation.

        Args:
            zone_details: Zone comparison details
            test_confidence: InspectionPipeline confidence (0.0-1.0)
            ink_details: Ink comparison details (M3, optional)
            profile_details: Radial profile comparison details (P1-2, optional)

        Returns:
            {
                "zone_score": 82.5,
                "ink_score": 88.0,  # M3
                "profile_score": 89.5,  # P1-2
                "total_score": 85.0,  # P1-2: weighted average
                "confidence_score": 88.0
            }
        """
        # Zone score (평균)
        zone_scores = [z["total_score"] for z in zone_details.values()]
        zone_score = sum(zone_scores) / len(zone_scores) if zone_scores else 0.0

        # Ink score (M3)
        ink_score = ink_details.get("ink_score", 0.0) if ink_details else 0.0

        # Profile score (P1-2)
        profile_score = profile_details.get("profile_score", 0.0) if profile_details else 0.0

        # Confidence score
        confidence_score = test_confidence * 100.0 if test_confidence else 0.0

        # Total score calculation with fallback logic
        if profile_details:
            # P1-2: Include profile_score
            # Weights: zone 35%, ink 25%, profile 25%, confidence 15%
            total_score = zone_score * 0.35 + ink_score * 0.25 + profile_score * 0.25 + confidence_score * 0.15
            logger.info("Using P1-2 total_score formula (zone 35%, ink 25%, profile 25%, confidence 15%)")
        elif ink_details and ink_details.get("ink_count_match"):
            # M3 fallback: Include ink_score but no profile
            # Weights: zone 50%, ink 30%, confidence 20%
            total_score = zone_score * 0.5 + ink_score * 0.3 + confidence_score * 0.2
            logger.info("Using M3 total_score formula (zone 50%, ink 30%, confidence 20%)")
        else:
            # M2 fallback: zone_score only (no ink or profile data)
            total_score = zone_score
            logger.info("Using M2 total_score formula (zone 100%)")

        return {
            "zone_score": zone_score,
            "ink_score": ink_score,
            "profile_score": profile_score,
            "total_score": total_score,
            "confidence_score": confidence_score,
        }

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

    def _calculate_delta_e(self, lab1: List[float], lab2: List[float]) -> float:
        """
        Calculate CIEDE76 color difference.

        Args:
            lab1: First LAB color [L, a, b]
            lab2: Second LAB color [L, a, b]

        Returns:
            Delta E value
        """
        import numpy as np

        return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2))))

    def _compare_inks(self, test_ink_analysis: Dict[str, Any], std_ink_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare ink analysis results between test and STD.

        Uses image-based GMM ink estimation for comparison.

        Args:
            test_ink_analysis: Test sample ink_analysis
            std_ink_analysis: STD sample ink_analysis

        Returns:
            Ink comparison details including:
            - ink_count_match: bool
            - ink_pairs: List of matched ink pairs with scores
            - ink_score: Overall ink similarity score (0-100)
        """
        # Extract image-based ink data
        test_image_based = test_ink_analysis.get("image_based", {})
        std_image_based = std_ink_analysis.get("image_based", {})

        test_count = test_image_based.get("ink_count", 0)
        std_count = std_image_based.get("ink_count", 0)

        test_inks = test_image_based.get("inks", [])
        std_inks = std_image_based.get("inks", [])

        logger.info(f"Comparing inks: TEST={test_count}, STD={std_count}")

        # Check ink count match
        if test_count != std_count:
            logger.warning(f"Ink count mismatch: TEST={test_count} != STD={std_count}")
            return {
                "ink_count_match": False,
                "test_ink_count": test_count,
                "std_ink_count": std_count,
                "ink_pairs": [],
                "avg_delta_e": 0.0,
                "max_delta_e": 0.0,
                "ink_score": 0.0,
                "message": f"Ink count mismatch: TEST={test_count}, STD={std_count}",
            }

        # No inks to compare
        if test_count == 0:
            logger.warning("No inks detected in both samples")
            return {
                "ink_count_match": True,
                "test_ink_count": 0,
                "std_ink_count": 0,
                "ink_pairs": [],
                "avg_delta_e": 0.0,
                "max_delta_e": 0.0,
                "ink_score": 100.0,  # No inks = perfect match
                "message": "No inks detected",
            }

        # Sort inks by weight (descending) for matching
        test_inks_sorted = sorted(test_inks, key=lambda x: x.get("weight", 0), reverse=True)
        std_inks_sorted = sorted(std_inks, key=lambda x: x.get("weight", 0), reverse=True)

        # Match inks by rank (weight-based pairing)
        ink_pairs = []
        color_scores = []
        weight_scores = []
        delta_es = []

        for rank, (test_ink, std_ink) in enumerate(zip(test_inks_sorted, std_inks_sorted), start=1):
            # Extract data
            test_lab = test_ink.get("lab", [0, 0, 0])
            std_lab = std_ink.get("lab", [0, 0, 0])
            test_weight = test_ink.get("weight", 0.0)
            std_weight = std_ink.get("weight", 0.0)

            # Calculate color difference
            delta_e = self._calculate_delta_e(test_lab, std_lab)
            delta_es.append(delta_e)

            # Color score: ΔE=0 → 100, ΔE=10 → 0
            color_score = max(0.0, 100.0 - delta_e * 10.0)
            color_scores.append(color_score)

            # Weight difference
            weight_diff = abs(test_weight - std_weight)

            # Weight score: diff=0 → 100, diff=0.3 → 0
            weight_score = max(0.0, 100.0 - weight_diff * 333.0)
            weight_scores.append(weight_score)

            # Pair score (weighted average)
            pair_score = color_score * 0.7 + weight_score * 0.3

            ink_pairs.append(
                {
                    "rank": rank,
                    "test_ink": {
                        "weight": float(test_weight),
                        "lab": [float(v) for v in test_lab],
                        "hex": test_ink.get("hex", "#000000"),
                    },
                    "std_ink": {
                        "weight": float(std_weight),
                        "lab": [float(v) for v in std_lab],
                        "hex": std_ink.get("hex", "#000000"),
                    },
                    "delta_e": float(delta_e),
                    "weight_diff": float(weight_diff),
                    "color_score": float(color_score),
                    "weight_score": float(weight_score),
                    "pair_score": float(pair_score),
                }
            )

        # Calculate overall ink score
        avg_color = sum(color_scores) / len(color_scores) if color_scores else 0.0
        avg_weight = sum(weight_scores) / len(weight_scores) if weight_scores else 0.0
        ink_score = avg_color * 0.7 + avg_weight * 0.3

        result = {
            "ink_count_match": True,
            "test_ink_count": test_count,
            "std_ink_count": std_count,
            "ink_pairs": ink_pairs,
            "avg_delta_e": float(sum(delta_es) / len(delta_es)) if delta_es else 0.0,
            "max_delta_e": float(max(delta_es)) if delta_es else 0.0,
            "ink_score": float(ink_score),
        }

        logger.info(
            f"Ink comparison complete: ink_score={ink_score:.1f}, "
            f"avg_ΔE={result['avg_delta_e']:.2f}, "
            f"count_match={result['ink_count_match']}"
        )

        return result

    def _calculate_worst_case_metrics(
        self, test_analysis: Dict[str, Any], std_analysis: Dict[str, Any], zone_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate worst-case metrics for comparison (P2).

        Args:
            test_analysis: Test sample analysis_result
            std_analysis: STD sample analysis_result
            zone_details: Zone comparison details

        Returns:
            Worst-case metrics including:
            - percentiles: Delta-E percentile statistics
            - hotspots: Detected high delta-E regions
            - worst_zone: Zone with highest mean delta-E
            - coverage_ratio: Ratio of area exceeding threshold
        """
        import cv2
        import numpy as np
        from scipy import ndimage

        logger.info("Calculating worst-case metrics (P2)")

        # Get zone results from test sample
        test_zones = test_analysis.get("zone_results", [])
        if not test_zones:
            logger.warning("No zone results available for worst-case analysis")
            return None

        # Collect all delta-E values from zones
        all_delta_e = []
        zone_mean_delta_e = {}

        for zone in test_zones:
            zone_name = zone.get("name", "Unknown")
            delta_e = zone.get("delta_e")
            if delta_e is not None:
                all_delta_e.append(delta_e)
                zone_mean_delta_e[zone_name] = delta_e

        if not all_delta_e:
            logger.warning("No delta-E values available")
            return None

        # 1. Calculate percentile metrics
        all_delta_e_array = np.array(all_delta_e)
        percentiles = {
            "mean": float(np.mean(all_delta_e_array)),
            "median": float(np.median(all_delta_e_array)),
            "p95": float(np.percentile(all_delta_e_array, 95)),
            "p99": float(np.percentile(all_delta_e_array, 99)),
            "max": float(np.max(all_delta_e_array)),
            "std": float(np.std(all_delta_e_array)),
        }

        # 2. Identify worst zone
        worst_zone = max(zone_mean_delta_e, key=zone_mean_delta_e.get) if zone_mean_delta_e else None

        # 3. Detect hotspots (zones with delta-E > threshold)
        hotspot_threshold = percentiles["p95"]  # Use p95 as hotspot threshold
        hotspots = []

        for zone in test_zones:
            zone_name = zone.get("name", "Unknown")
            delta_e = zone.get("delta_e")
            if delta_e is not None and delta_e > hotspot_threshold:
                # Determine severity
                if delta_e > percentiles["p99"]:
                    severity = "CRITICAL"
                elif delta_e > percentiles["p95"]:
                    severity = "HIGH"
                else:
                    severity = "MEDIUM"

                hotspots.append(
                    {
                        "area": zone.get("pixel_count", 0),
                        "centroid": [0.0, 0.0],  # Placeholder - would need actual position
                        "mean_delta_e": float(delta_e),
                        "max_delta_e": float(delta_e),  # Same as mean for zone-level
                        "zone": zone_name,
                        "severity": severity,
                    }
                )

        # Sort hotspots by mean_delta_e (descending) and take top 5
        hotspots_sorted = sorted(hotspots, key=lambda x: x["mean_delta_e"], reverse=True)[:5]

        # 4. Calculate coverage ratio (zones exceeding threshold / total zones)
        zones_exceeding = sum(1 for de in all_delta_e if de > hotspot_threshold)
        coverage_ratio = zones_exceeding / len(all_delta_e) if all_delta_e else 0.0

        result = {
            "percentiles": percentiles,
            "hotspots": hotspots_sorted,
            "hotspot_count": len(hotspots),
            "worst_zone": worst_zone,
            "coverage_ratio": float(coverage_ratio),
        }

        logger.info(
            f"Worst-case metrics complete: p95={percentiles['p95']:.2f}, "
            f"p99={percentiles['p99']:.2f}, hotspots={len(hotspots)}, worst_zone={worst_zone}"
        )

        return result
