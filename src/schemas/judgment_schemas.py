"""
Judgment Schemas

Pydantic models for judgment criteria and results.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JudgmentEnum(str, Enum):
    """Judgment status enumeration"""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    RETAKE = "RETAKE"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class JudgmentCriteria(BaseModel):
    """
    Judgment criteria configuration

    Used to determine PASS/WARNING/FAIL thresholds.
    These values should be configured in Week 1 workshop.
    """

    # Structure thresholds
    min_structure_score: float = Field(
        default=70.0, description="Minimum structure similarity score for PASS (0-100)", ge=0.0, le=100.0
    )
    min_profile_correlation: float = Field(
        default=0.85, description="Minimum radial profile correlation for PASS (0-1)", ge=0.0, le=1.0
    )
    max_boundary_difference_percent: float = Field(
        default=3.0, description="Maximum allowed boundary position difference (%, ±)", ge=0.0, le=20.0
    )

    # Color thresholds
    min_color_score: float = Field(
        default=70.0, description="Minimum color similarity score for PASS (0-100)", ge=0.0, le=100.0
    )
    max_mean_delta_e: float = Field(default=3.0, description="Maximum allowed mean ΔE for PASS", ge=0.0, le=20.0)
    max_p95_delta_e: float = Field(
        default=5.0, description="Maximum allowed 95th percentile ΔE for PASS", ge=0.0, le=30.0
    )

    # Overall score thresholds
    pass_score_threshold: float = Field(
        default=80.0, description="Total score >= this → PASS (0-100)", ge=0.0, le=100.0
    )
    warning_score_threshold: float = Field(
        default=60.0, description="Total score >= this → WARNING, else FAIL (0-100)", ge=0.0, le=100.0
    )

    # Confidence thresholds
    min_confidence_for_auto_judgment: float = Field(
        default=80.0, description="Minimum confidence for automatic judgment (0-100)", ge=0.0, le=100.0
    )

    class Config:
        json_schema_extra = {
            "example": {
                "min_structure_score": 70.0,
                "min_profile_correlation": 0.85,
                "max_boundary_difference_percent": 3.0,
                "min_color_score": 70.0,
                "max_mean_delta_e": 3.0,
                "max_p95_delta_e": 5.0,
                "pass_score_threshold": 80.0,
                "warning_score_threshold": 60.0,
                "min_confidence_for_auto_judgment": 80.0,
            }
        }


class ConfidenceScore(BaseModel):
    """
    Confidence score breakdown

    Confidence is SEPARATE from judgment.
    High confidence + FAIL = "Definitely bad"
    Low confidence + FAIL = "Need manual review"
    """

    confidence: float = Field(..., description="Overall confidence score (0-100)", ge=0.0, le=100.0)

    # Confidence factors (all 0-100)
    lens_detection_confidence: float = Field(..., description="Lens detection quality", ge=0.0, le=100.0)
    alignment_confidence: float = Field(..., description="Alignment quality", ge=0.0, le=100.0)
    data_completeness: float = Field(..., description="Data completeness (zones, profile, etc.)", ge=0.0, le=100.0)

    # Recommendation
    recommendation: str = Field(..., description="Confidence-based recommendation", example="HIGH")

    class Config:
        json_schema_extra = {
            "example": {
                "confidence": 92.0,
                "lens_detection_confidence": 98.0,
                "alignment_confidence": 95.0,
                "data_completeness": 85.0,
                "recommendation": "HIGH",
            }
        }


class JudgmentResult(BaseModel):
    """
    Complete judgment result

    Combines judgment, confidence, and metadata.
    """

    # Judgment
    judgment: JudgmentEnum = Field(..., description="Judgment status")
    total_score: float = Field(..., description="Total similarity score (0-100)", ge=0.0, le=100.0)
    structure_score: float = Field(..., description="Structure similarity score (0-100)", ge=0.0, le=100.0)
    color_score: float = Field(..., description="Color similarity score (0-100)", ge=0.0, le=100.0)

    # Confidence (SEPARATE from judgment)
    confidence: ConfidenceScore = Field(..., description="Confidence score breakdown")

    # Criteria used
    criteria: JudgmentCriteria = Field(..., description="Criteria used for judgment")

    # Explanation
    judgment_reason: str = Field(
        ..., description="Brief reason for judgment", example="Total score 65.3 < 80.0 (PASS threshold)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "judgment": "FAIL",
                "total_score": 65.3,
                "structure_score": 87.5,
                "color_score": 50.2,
                "confidence": {
                    "confidence": 92.0,
                    "lens_detection_confidence": 98.0,
                    "zone_segmentation_confidence": 90.0,
                    "alignment_confidence": 95.0,
                    "data_completeness": 85.0,
                    "recommendation": "HIGH",
                },
                "criteria": {"pass_score_threshold": 80.0, "warning_score_threshold": 60.0, "max_p95_delta_e": 5.0},
                "judgment_reason": (
                    "Total score 65.3 < 80.0 (PASS threshold). " "Zone B color score 50.2 (p95 ΔE=6.1 > 5.0)"
                ),
            }
        }
