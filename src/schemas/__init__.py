"""
API Schemas Package

Pydantic models for request/response validation in FastAPI.
"""

from .comparison_schemas import (
    CompareRequest,
    CompareResponse,
    ComparisonDetailResponse,
    ComparisonListItem,
    ComparisonListResponse,
    CorrelationData,
    FailureReason,
    InkData,
    InkDetailsData,
    InkPairData,
    ProfileDetailsData,
    ScoresData,
    StructuralSimilarityData,
)
from .std_schemas import STDDetailResponse, STDListResponse, STDProfileData, STDRegisterRequest, STDRegisterResponse
from .test_schemas import TestDetailResponse, TestListItem, TestListResponse, TestRegisterRequest, TestRegisterResponse

__all__ = [
    # STD Schemas
    "STDRegisterRequest",
    "STDRegisterResponse",
    "STDDetailResponse",
    "STDListResponse",
    "STDProfileData",
    # Test Schemas
    "TestRegisterRequest",
    "TestRegisterResponse",
    "TestListResponse",
    "TestListItem",
    "TestDetailResponse",
    # Comparison Schemas
    "CompareRequest",
    "CompareResponse",
    "ComparisonListResponse",
    "ComparisonListItem",
    "ComparisonDetailResponse",
    "ScoresData",
    "FailureReason",
    "InkData",
    "InkPairData",
    "InkDetailsData",
    "CorrelationData",
    "StructuralSimilarityData",
    "ProfileDetailsData",
]
