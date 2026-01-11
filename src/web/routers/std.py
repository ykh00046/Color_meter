"""
STD (Standard) API Router

Endpoints for STD registration, querying, and management.
"""

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.models.database import get_session
from src.schemas.std_schemas import (
    STDDetailResponse,
    STDListItem,
    STDListResponse,
    STDRegisterRequest,
    STDRegisterResponse,
)
from src.services.std_service import STDService, STDServiceError
from src.sku_manager import SkuConfigManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/std", tags=["STD Management"])

# Global ConfigManager instance
_config_manager = None


def get_config_manager() -> SkuConfigManager:
    """Get or create SkuConfigManager instance (singleton)"""
    global _config_manager
    if _config_manager is None:
        config_dir = Path("config/sku_db")
        _config_manager = SkuConfigManager(db_path=config_dir)
    return _config_manager


def get_std_service(
    db: Session = Depends(get_session), sku_mgr: SkuConfigManager = Depends(get_config_manager)
) -> STDService:
    """Get STDService instance (dependency injection)"""
    return STDService(db_session=db, sku_manager=sku_mgr)


@router.post(
    "/register",
    response_model=STDRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new STD profile",
    description="""
    Register a new STD (Standard) profile for a SKU.

    Workflow:
    1. Validate image path
    2. Load SKU configuration
    3. Run InspectionPipeline on STD image
    4. Store profile in database (n_samples=1 for MVP)
    5. Return STD model ID and summary

    **Note**: Only one active STD per SKU+version.
    Previous active STD will be deactivated automatically.
    """,
)
def register_std(
    request: STDRegisterRequest, std_service: STDService = Depends(get_std_service)
) -> STDRegisterResponse:
    """
    Register new STD profile.

    Args:
        request: STD registration request
        std_service: STDService instance (injected)

    Returns:
        STDRegisterResponse with created STD model info

    Raises:
        HTTPException 400: Invalid request (image not found, invalid SKU)
        HTTPException 500: Registration failed (pipeline error, DB error)
    """
    try:
        logger.info(f"STD registration request: sku={request.sku_code}, version={request.version}")

        # Register STD using service
        std_model = std_service.register_std(
            sku_code=request.sku_code,
            image_path=request.image_path,
            version=request.version,
            notes=request.notes,
            user_id=None,  # TODO: Get from authentication context
        )

        # Get analysis result to extract summary
        std_sample = std_model.samples[0] if std_model.samples else None
        analysis_result = std_sample.analysis_result if std_sample else {}

        n_zones = len(analysis_result.get("zone_results", []))

        # Estimate profile length from zone_results if available
        profile_length = 500  # Default estimate
        # Could extract from analysis_result if stored

        # Build response
        response = STDRegisterResponse(
            id=std_model.id,
            sku_code=std_model.sku_code,
            version=std_model.version,
            created_at=std_model.created_at,
            is_active=std_model.is_active,
            n_zones=n_zones if n_zones > 0 else 1,
            profile_length=profile_length,
            message=f"STD registered successfully for {request.sku_code} {request.version}",
        )

        logger.info(f"STD registered: id={std_model.id}")
        return response

    except STDServiceError as e:
        logger.error(f"STD registration failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during STD registration: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/list",
    response_model=STDListResponse,
    summary="List STD profiles",
    description="""
    Retrieve list of STD profiles with optional filtering.

    Query Parameters:
    - sku_code: Filter by SKU (optional)
    - active_only: Show only active STDs (default: true)
    - limit: Max results (default: 100)
    - offset: Pagination offset (default: 0)
    """,
)
def list_stds(
    sku_code: Optional[str] = None,
    active_only: bool = True,
    limit: int = 100,
    offset: int = 0,
    std_service: STDService = Depends(get_std_service),
) -> STDListResponse:
    """
    List STD profiles.

    Args:
        sku_code: Filter by SKU (optional)
        active_only: Only active STDs (default: True)
        limit: Max results (max: 1000)
        offset: Pagination offset
        std_service: STDService instance (injected)

    Returns:
        STDListResponse with list of STD profiles
    """
    try:
        # Enforce limit max
        limit = min(limit, 1000)

        # Query STDs
        std_models = std_service.list_stds(sku_code=sku_code, active_only=active_only, limit=limit, offset=offset)

        # Convert to response items
        items = [
            STDListItem(
                id=std.id,
                sku_code=std.sku_code,
                version=std.version,
                created_at=std.created_at,
                is_active=std.is_active,
                is_approved=std.is_approved,
                approved_by=std.approved_by,
                approved_at=std.approved_at,
            )
            for std in std_models
        ]

        return STDListResponse(total=len(items), items=items)

    except STDServiceError as e:
        logger.error(f"Failed to list STDs: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error listing STDs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/{std_id}",
    response_model=STDDetailResponse,
    summary="Get STD details",
    description="Retrieve detailed information about a specific STD profile",
)
def get_std_detail(std_id: int, std_service: STDService = Depends(get_std_service)) -> STDDetailResponse:
    """
    Get STD profile details.

    Args:
        std_id: STD model ID
        std_service: STDService instance (injected)

    Returns:
        STDDetailResponse with full STD profile data

    Raises:
        HTTPException 404: STD not found
    """
    try:
        std_model = std_service.get_std_by_id(std_id)

        if not std_model:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"STD not found: id={std_id}")

        # Get first sample (MVP: only one sample)
        std_sample = std_model.samples[0] if std_model.samples else None

        if not std_sample:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"STD has no samples: id={std_id}"
            )

        # Extract analysis result
        analysis_result = std_sample.analysis_result

        # Build response using stored analysis_result
        from src.schemas.std_schemas import STDProfileData, ZoneBoundaryData, ZoneColorData

        zone_results = analysis_result.get("zone_results", [])
        if not zone_results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="STD analysis_result missing zone_results"
            )

        zone_colors = {}
        for zr in zone_results:
            zone_name = zr.get("zone_name") or "A"
            lab = zr.get("measured_lab") or zr.get("target_lab")
            if not lab:
                continue
            zone_colors[zone_name] = ZoneColorData(L=float(lab[0]), a=float(lab[1]), b=float(lab[2]))

        if not zone_colors:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="STD analysis_result missing zone color data"
            )

        radial_profile = analysis_result.get("radial_profile") or {}
        radial_profile_L = radial_profile.get("L")
        radial_profile_a = radial_profile.get("a")
        radial_profile_b = radial_profile.get("b")
        if not (radial_profile_L and radial_profile_a and radial_profile_b):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="STD analysis_result missing radial_profile data",
            )

        image_width = analysis_result.get("image_width")
        image_height = analysis_result.get("image_height")
        if not image_width or not image_height:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="STD analysis_result missing image dimensions"
            )

        zone_boundaries = ZoneBoundaryData(inner_to_A=None, A_to_B=None, B_to_C=None, C_outer=None)

        profile_data = STDProfileData(
            sku_code=std_model.sku_code,
            version=std_model.version,
            zone_colors=zone_colors,
            zone_boundaries=zone_boundaries,
            radial_profile_L=radial_profile_L,
            radial_profile_a=radial_profile_a,
            radial_profile_b=radial_profile_b,
            image_width=image_width,
            image_height=image_height,
            lens_detected=bool(analysis_result.get("lens_detected", True)),
            lens_detection_score=analysis_result.get("lens_detection_score"),
        )

        response = STDDetailResponse(
            id=std_model.id,
            sku_code=std_model.sku_code,
            version=std_model.version,
            created_at=std_model.created_at,
            is_active=std_model.is_active,
            is_approved=std_model.is_approved,
            approved_by=std_model.approved_by,
            approved_at=std_model.approved_at,
            notes=std_model.notes,
            profile_data=profile_data,
        )

        return response

    except HTTPException:
        raise
    except STDServiceError as e:
        logger.error(f"Failed to get STD details: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting STD details: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}"
        )


@router.delete(
    "/{std_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Deactivate STD profile",
    description="Deactivate (soft delete) an STD profile",
)
def deactivate_std(std_id: int, std_service: STDService = Depends(get_std_service)):
    """
    Deactivate STD profile (soft delete).

    Args:
        std_id: STD model ID
        std_service: STDService instance (injected)

    Returns:
        204 No Content on success

    Raises:
        HTTPException 404: STD not found
    """
    try:
        success = std_service.deactivate_std(std_id=std_id, user_id=None)  # TODO: Get from authentication context

        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"STD not found: id={std_id}")

        return None  # 204 No Content

    except HTTPException:
        raise
    except STDServiceError as e:
        logger.error(f"Failed to deactivate STD: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error deactivating STD: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}"
        )
