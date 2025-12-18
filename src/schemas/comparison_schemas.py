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
    ink: float = Field(..., description="Ink score (0-100, MVP: 0)", ge=0, le=100)
    confidence: float = Field(..., description="Confidence score (0-100)", ge=0, le=100)


class FailureReason(BaseModel):
    """Failure reason"""

    rank: int = Field(..., description="Rank (1-3)", ge=1, le=3)
    category: str = Field(..., description="Category (ZONE_COLOR, ZONE_ERROR, etc.)")
    zone: Optional[str] = Field(None, description="Zone name")
    message: str = Field(..., description="Failure message")
    severity: float = Field(..., description="Severity (0-100)", ge=0, le=100)
    score: float = Field(..., description="Related score", ge=0, le=100)


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
    ink_details: Optional[Dict[str, Any]] = Field(None, description="Ink comparison details (MVP: None)")
    created_at: datetime = Field(..., description="Created timestamp")
    processing_time_ms: int = Field(..., description="Processing time (milliseconds)")

    class Config:
        from_attributes = True
