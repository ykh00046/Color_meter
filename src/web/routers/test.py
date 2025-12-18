"""
Test Sample API Router

Endpoints for test sample registration, querying, and management.
"""

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.models.database import get_session
from src.schemas.test_schemas import (
    TestDetailResponse,
    TestListItem,
    TestListResponse,
    TestRegisterRequest,
    TestRegisterResponse,
)
from src.services.test_service import TestService, TestServiceError
from src.sku_manager import SkuConfigManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/test", tags=["Test Sample Management"])

_config_manager = None


def get_config_manager() -> SkuConfigManager:
    global _config_manager
    if _config_manager is None:
        config_dir = Path("config/sku_db")
        _config_manager = SkuConfigManager(db_path=config_dir)
    return _config_manager


def get_test_service(
    db: Session = Depends(get_session), sku_mgr: SkuConfigManager = Depends(get_config_manager)
) -> TestService:
    return TestService(db_session=db, sku_manager=sku_mgr)


@router.post("/register", response_model=TestRegisterResponse, status_code=status.HTTP_201_CREATED)
def register_test(
    request: TestRegisterRequest, test_service: TestService = Depends(get_test_service)
) -> TestRegisterResponse:
    try:
        logger.info(f"Test registration request: sku={request.sku_code}")
        test_sample = test_service.register_test_sample(
            sku_code=request.sku_code,
            image_path=request.image_path,
            batch_number=request.batch_number,
            sample_id=request.sample_id,
            operator=request.operator,
            notes=request.notes,
        )
        response = TestRegisterResponse(
            id=test_sample.id,
            sku_code=test_sample.sku_code,
            batch_number=test_sample.batch_number,
            sample_id=test_sample.sample_id,
            lens_detected=test_sample.lens_detected,
            lens_detection_score=test_sample.lens_detection_score,
            created_at=test_sample.created_at,
            operator=test_sample.operator,
            message=f"Test sample registered successfully: id={test_sample.id}",
        )
        return response
    except TestServiceError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/list", response_model=TestListResponse)
def list_tests(
    sku_code: Optional[str] = None,
    batch_number: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    test_service: TestService = Depends(get_test_service),
) -> TestListResponse:
    try:
        limit = min(limit, 1000)
        test_samples = test_service.list_test_samples(
            sku_code=sku_code, batch_number=batch_number, limit=limit, offset=offset
        )
        items = [
            TestListItem(
                id=t.id,
                sku_code=t.sku_code,
                batch_number=t.batch_number,
                sample_id=t.sample_id,
                lens_detected=t.lens_detected,
                created_at=t.created_at,
                operator=t.operator,
            )
            for t in test_samples
        ]
        return TestListResponse(total=len(items), items=items)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{test_id}", response_model=TestDetailResponse)
def get_test_detail(test_id: int, test_service: TestService = Depends(get_test_service)) -> TestDetailResponse:
    try:
        test_sample = test_service.get_test_sample(test_id)
        if not test_sample:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Test sample not found: id={test_id}")
        response = TestDetailResponse(
            id=test_sample.id,
            sku_code=test_sample.sku_code,
            batch_number=test_sample.batch_number,
            sample_id=test_sample.sample_id,
            image_path=test_sample.image_path,
            lens_detected=test_sample.lens_detected,
            lens_detection_score=test_sample.lens_detection_score,
            created_at=test_sample.created_at,
            operator=test_sample.operator,
            notes=test_sample.notes,
            file_size_bytes=test_sample.file_size_bytes,
            image_width=test_sample.image_width,
            image_height=test_sample.image_height,
            analysis_result=test_sample.analysis_result,
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
