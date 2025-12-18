"""
Comparison Pydantic Schemas

Request/Response models for Comparison API.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CompareRequest(BaseModel):
    """Comparison request"""

    test_sample_id: int = Field(..., description="Test sample ID", gt=0)
    std_model_id: Optional[int] = Field(None, description="STD model ID (auto-match if None)", gt=0)

    class Config:
        json_schema_extra = {"example": {"test_sample_id": 1, "std_model_id": None}}


class ScoresData(BaseModel):
    """Scores data"""

    total: float = Field(..., description="Total score (0-100)", ge=0, le=100)
    zone: float = Field(..., description="Zone score (0-100)", ge=0, le=100)
    ink: float = Field(..., description="Ink score (0-100)", ge=0, le=100)
    profile: float = Field(0.0, description="Radial profile score (0-100, P1-2)", ge=0, le=100)
    confidence: float = Field(..., description="Confidence score (0-100)", ge=0, le=100)


class FailureReason(BaseModel):
    """Failure reason"""

    rank: int = Field(..., description="Rank (1-3)", ge=1, le=3)
    category: str = Field(..., description="Category (ZONE_COLOR, ZONE_ERROR, etc.)")
    zone: Optional[str] = Field(None, description="Zone name")
    message: str = Field(..., description="Failure message")
    severity: float = Field(..., description="Severity (0-100)", ge=0, le=100)
    score: float = Field(..., description="Related score", ge=0, le=100)


# M3: Ink Comparison Schemas


class InkData(BaseModel):
    """Individual ink data"""

    weight: float = Field(..., description="Ink pixel ratio (0-1)")
    lab: List[float] = Field(..., description="LAB color [L, a, b]")
    hex: str = Field(..., description="HEX color code")


class InkPairData(BaseModel):
    """Ink pair comparison data"""

    rank: int = Field(..., description="Ink rank (1=primary, 2=secondary, etc.)", ge=1)
    test_ink: InkData = Field(..., description="Test sample ink")
    std_ink: InkData = Field(..., description="STD ink")
    delta_e: float = Field(..., description="Color difference (CIEDE76)", ge=0)
    weight_diff: float = Field(..., description="Weight difference (absolute)", ge=0)
    color_score: float = Field(..., description="Color similarity score (0-100)", ge=0, le=100)
    weight_score: float = Field(..., description="Weight similarity score (0-100)", ge=0, le=100)
    pair_score: float = Field(..., description="Overall pair score (0-100)", ge=0, le=100)


class InkDetailsData(BaseModel):
    """Ink comparison details (M3)"""

    ink_count_match: bool = Field(..., description="Whether ink counts match")
    test_ink_count: int = Field(..., description="Test sample ink count", ge=0)
    std_ink_count: int = Field(..., description="STD ink count", ge=0)
    ink_pairs: List[InkPairData] = Field(..., description="Ink pair comparisons")
    avg_delta_e: float = Field(..., description="Average ink ΔE", ge=0)
    max_delta_e: float = Field(..., description="Maximum ink ΔE", ge=0)
    ink_score: float = Field(..., description="Overall ink score (0-100)", ge=0, le=100)
    message: Optional[str] = Field(None, description="Message (e.g., mismatch reason)")


# P1-2: Radial Profile Comparison Schemas


class CorrelationData(BaseModel):
    """Correlation coefficients for radial profile comparison"""

    L: float = Field(..., description="L* channel correlation", ge=-1, le=1)
    a: float = Field(..., description="a* channel correlation", ge=-1, le=1)
    b: float = Field(..., description="b* channel correlation", ge=-1, le=1)
    avg: float = Field(..., description="Average correlation across channels", ge=-1, le=1)


class StructuralSimilarityData(BaseModel):
    """Structural similarity metrics for radial profile comparison"""

    L: float = Field(..., description="L* channel SSIM", ge=-1, le=1)
    a: float = Field(..., description="a* channel SSIM", ge=-1, le=1)
    b: float = Field(..., description="b* channel SSIM", ge=-1, le=1)
    avg: float = Field(..., description="Average SSIM across channels", ge=-1, le=1)


class ProfileDetailsData(BaseModel):
    """Radial profile comparison details (P1-2)"""

    correlation: CorrelationData = Field(..., description="Pearson correlation coefficients")
    structural_similarity: StructuralSimilarityData = Field(..., description="Structural similarity (SSIM)")
    gradient_similarity: CorrelationData = Field(..., description="Gradient correlation coefficients")
    profile_score: float = Field(..., description="Overall profile similarity score (0-100)", ge=0, le=100)
    length_match: bool = Field(..., description="Whether profile lengths match")
    test_length: int = Field(..., description="Test profile length (number of points)", ge=0)
    std_length: int = Field(..., description="STD profile length (number of points)", ge=0)
    message: Optional[str] = Field(None, description="Summary message")


class CompareResponse(BaseModel):
    """Comparison response"""

    id: int = Field(..., description="Comparison result ID")
    test_sample_id: int = Field(..., description="Test sample ID")
    std_model_id: int = Field(..., description="STD model ID")
    scores: ScoresData = Field(..., description="Comparison scores")
    judgment: str = Field(..., description="Judgment (PASS/FAIL/RETAKE/MANUAL_REVIEW)")
    is_pass: bool = Field(..., description="Whether judgment is PASS")
    needs_action: bool = Field(..., description="Whether needs operator action")
    top_failure_reasons: Optional[List[FailureReason]] = Field(None, description="Top 3 failure reasons")
    created_at: datetime = Field(..., description="Created timestamp")
    processing_time_ms: int = Field(..., description="Processing time (milliseconds)")
    message: str = Field(..., description="Summary message")

    class Config:
        from_attributes = True


class ComparisonListItem(BaseModel):
    """Comparison list item"""

    id: int = Field(..., description="Comparison result ID")
    test_sample_id: int = Field(..., description="Test sample ID")
    std_model_id: int = Field(..., description="STD model ID")
    total_score: float = Field(..., description="Total score (0-100)")
    judgment: str = Field(..., description="Judgment")
    is_pass: bool = Field(..., description="Whether judgment is PASS")
    created_at: datetime = Field(..., description="Created timestamp")

    class Config:
        from_attributes = True


class ComparisonListResponse(BaseModel):
    """Comparison list response"""

    total: int = Field(..., description="Total number of items")
    items: List[ComparisonListItem] = Field(..., description="List of comparison results")


class ComparisonDetailResponse(BaseModel):
    """Comparison detail response (full data)"""

    id: int = Field(..., description="Comparison result ID")
    test_sample_id: int = Field(..., description="Test sample ID")
    std_model_id: int = Field(..., description="STD model ID")
    scores: ScoresData = Field(..., description="Comparison scores")
    judgment: str = Field(..., description="Judgment")
    is_pass: bool = Field(..., description="Whether judgment is PASS")
    needs_action: bool = Field(..., description="Whether needs operator action")
    top_failure_reasons: Optional[List[Dict[str, Any]]] = Field(None, description="Top failure reasons")
    zone_details: Dict[str, Any] = Field(..., description="Zone-by-zone details")
    ink_details: Optional[InkDetailsData] = Field(None, description="Ink comparison details (M3)")
    profile_details: Optional[ProfileDetailsData] = Field(None, description="Radial profile comparison details (P1-2)")
    created_at: datetime = Field(..., description="Created timestamp")
    processing_time_ms: int = Field(..., description="Processing time (milliseconds)")

    class Config:
        from_attributes = True
