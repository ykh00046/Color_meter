"""
Comparison API Router

Endpoints for test sample vs STD comparison.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.models.database import get_session
from src.models.test_models import JudgmentStatus
from src.schemas.comparison_schemas import (
    CompareRequest,
    CompareResponse,
    ComparisonDetailResponse,
    ComparisonListItem,
    ComparisonListResponse,
    FailureReason,
    ScoresData,
)
from src.services.comparison_service import ComparisonService, ComparisonServiceError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/compare", tags=["Comparison"])


def get_comparison_service(db: Session = Depends(get_session)) -> ComparisonService:
    return ComparisonService(db_session=db)


@router.post("", response_model=CompareResponse, status_code=status.HTTP_201_CREATED)
def compare(
    request: CompareRequest, comparison_service: ComparisonService = Depends(get_comparison_service)
) -> CompareResponse:
    try:
        logger.info(f"Comparison request: test_sample_id={request.test_sample_id}")
        comparison_result = comparison_service.compare(
            test_sample_id=request.test_sample_id, std_model_id=request.std_model_id
        )
        scores = ScoresData(
            total=comparison_result.total_score,
            zone=comparison_result.zone_score,
            ink=comparison_result.ink_score,
            profile=comparison_result.profile_score,
            confidence=comparison_result.confidence_score,
        )
        failure_reasons = None
        if comparison_result.top_failure_reasons:
            failure_reasons = [FailureReason(**fr) for fr in comparison_result.top_failure_reasons]
        response = CompareResponse(
            id=comparison_result.id,
            test_sample_id=comparison_result.test_sample_id,
            std_model_id=comparison_result.std_model_id,
            scores=scores,
            judgment=comparison_result.judgment.value,
            is_pass=comparison_result.is_pass,
            needs_action=comparison_result.needs_action,
            top_failure_reasons=failure_reasons,
            created_at=comparison_result.created_at,
            processing_time_ms=comparison_result.processing_time_ms,
            message=(
                f"Comparison completed: {comparison_result.judgment.value} "
                f"(score={comparison_result.total_score:.1f}, profile={comparison_result.profile_score:.1f})"
            ),
        )
        return response
    except ComparisonServiceError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/list", response_model=ComparisonListResponse)
def list_comparisons(
    sku_code: Optional[str] = None,
    judgment: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    comparison_service: ComparisonService = Depends(get_comparison_service),
) -> ComparisonListResponse:
    try:
        limit = min(limit, 1000)
        judgment_status = JudgmentStatus(judgment) if judgment else None
        comparisons = comparison_service.list_comparison_results(
            sku_code=sku_code, judgment=judgment_status, limit=limit, offset=offset
        )
        items = [
            ComparisonListItem(
                id=c.id,
                test_sample_id=c.test_sample_id,
                std_model_id=c.std_model_id,
                total_score=c.total_score,
                judgment=c.judgment.value,
                is_pass=c.is_pass,
                created_at=c.created_at,
            )
            for c in comparisons
        ]
        return ComparisonListResponse(total=len(items), items=items)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{comparison_id}", response_model=ComparisonDetailResponse)
def get_comparison_detail(
    comparison_id: int, comparison_service: ComparisonService = Depends(get_comparison_service)
) -> ComparisonDetailResponse:
    try:
        comparison = comparison_service.get_comparison_result(comparison_id)
        if not comparison:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Comparison not found: id={comparison_id}"
            )
        scores = ScoresData(
            total=comparison.total_score,
            zone=comparison.zone_score,
            ink=comparison.ink_score,
            profile=comparison.profile_score,
            confidence=comparison.confidence_score,
        )
        response = ComparisonDetailResponse(
            id=comparison.id,
            test_sample_id=comparison.test_sample_id,
            std_model_id=comparison.std_model_id,
            scores=scores,
            judgment=comparison.judgment.value,
            is_pass=comparison.is_pass,
            needs_action=comparison.needs_action,
            top_failure_reasons=comparison.top_failure_reasons,
            defect_classifications=comparison.defect_classifications,
            zone_details=comparison.zone_details,
            ink_details=comparison.ink_details,
            profile_details=comparison.profile_details,
            alignment_details=comparison.alignment_details,
            worst_case_metrics=comparison.worst_case_metrics,
            created_at=comparison.created_at,
            processing_time_ms=comparison.processing_time_ms,
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
