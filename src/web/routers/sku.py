"""
SKU Management API Router

Endpoints for SKU configuration, auto-detection, and management.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.core.ink_estimator import InkEstimator
from src.sku_manager import (
    InvalidSkuDataError,
    SkuAlreadyExistsError,
    SkuConfigManager,
    SkuNotFoundError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sku", tags=["SKU Management"])

# Global SKU manager instance
_sku_manager = None


def get_sku_manager() -> SkuConfigManager:
    """Get or create SkuConfigManager instance (singleton)"""
    global _sku_manager
    if _sku_manager is None:
        config_dir = Path("config/sku_db")
        _sku_manager = SkuConfigManager(db_path=config_dir)
    return _sku_manager


# === Schemas ===


class ZoneLabValues(BaseModel):
    """LAB color values for a zone"""

    L: float = Field(..., description="Lightness (0-100)", ge=0.0, le=100.0)
    a: float = Field(..., description="a* (-128 to 127)", ge=-128.0, le=127.0)
    b: float = Field(..., description="b* (-128 to 127)", ge=-128.0, le=127.0)
    threshold: float = Field(default=8.0, description="ΔE threshold", ge=0.0, le=50.0)
    description: Optional[str] = Field(None, description="Zone description")


class AutoDetectRequest(BaseModel):
    """Request to auto-detect ink configuration"""

    sku_code: str = Field(..., description="SKU code", min_length=3, max_length=50)
    image_path: str = Field(..., description="Path to representative image")
    chroma_thresh: float = Field(default=6.0, description="Chroma threshold for ink detection", ge=0.0, le=50.0)
    L_max: float = Field(default=98.0, description="Maximum L value (highlight removal)", ge=0.0, le=100.0)
    merge_de_thresh: float = Field(default=5.0, description="ΔE threshold for merging similar colors", ge=0.0, le=50.0)
    linearity_thresh: float = Field(default=3.0, description="Linearity threshold for mixing correction", ge=0.0, le=50.0)

    class Config:
        json_schema_extra = {
            "example": {
                "sku_code": "SKU001",
                "image_path": "C:/X/Color_total/Color_meter/data/raw_images/SKU001_OK_001.jpg",
                "chroma_thresh": 6.0,
                "L_max": 98.0,
                "merge_de_thresh": 5.0,
                "linearity_thresh": 3.0,
            }
        }


class DetectedInk(BaseModel):
    """Detected ink color"""

    L: float
    a: float
    b: float
    weight: float
    hex: str
    suggested_threshold: float


class SuggestedZoneConfig(BaseModel):
    """Suggested zone configuration based on detected inks"""

    zone_name: str
    lab_values: ZoneLabValues
    confidence: float  # Based on ink weight
    source_ink_index: int


class AutoDetectResponse(BaseModel):
    """Response from auto-detect endpoint"""

    sku_code: str
    detected_inks: List[DetectedInk]
    suggested_zones: Dict[str, ZoneLabValues]
    ink_count: int
    meta: Dict[str, Any]
    warnings: List[str]
    message: str


# === Endpoints ===


@router.post("/auto-detect-ink", response_model=AutoDetectResponse, status_code=status.HTTP_200_OK)
def auto_detect_ink_config(request: AutoDetectRequest) -> AutoDetectResponse:
    """
    Auto-detect ink configuration from a representative image.

    Workflow:
    1. Load representative image
    2. Run InkEstimator to detect ink colors
    3. Map detected inks to zone LAB values
    4. Suggest zone configuration
    5. Return for user approval

    Note: This endpoint does NOT save the configuration automatically.
    User must review and approve before saving via update_sku endpoint.
    """
    try:
        logger.info(f"Auto-detect request: sku={request.sku_code}, image={request.image_path}")

        # 1. Validate image path
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Image not found: {request.image_path}")

        # 2. Run InkEstimator
        estimator = InkEstimator()
        ink_result = estimator.estimate(
            image_path=str(image_path),
            k_max=3,
            chroma_thresh=request.chroma_thresh,
            L_max=request.L_max,
            merge_de_thresh=request.merge_de_thresh,
            linearity_thresh=request.linearity_thresh,
        )

        logger.info(f"InkEstimator result: ink_count={ink_result['ink_count']}")

        # 3. Check if inks were detected
        if ink_result["ink_count"] == 0:
            warnings = []
            if "warning" in ink_result:
                warnings.append(ink_result["warning"])
            warnings.append("No inks detected. Try adjusting chroma_thresh or L_max parameters.")

            return AutoDetectResponse(
                sku_code=request.sku_code,
                detected_inks=[],
                suggested_zones={},
                ink_count=0,
                meta=ink_result.get("meta", {}),
                warnings=warnings,
                message="No inks detected. Please adjust parameters or use a different image.",
            )

        # 4. Convert detected inks to suggested zones
        detected_inks = ink_result["inks"]
        suggested_zones = _map_inks_to_zones(detected_inks)

        # 5. Format detected inks for response
        formatted_inks = [
            DetectedInk(
                L=ink["lab"][0],
                a=ink["lab"][1],
                b=ink["lab"][2],
                weight=ink["weight"],
                hex=ink["hex"],
                suggested_threshold=_calculate_threshold_from_weight(ink["weight"]),
            )
            for ink in detected_inks
        ]

        # 6. Build warnings
        warnings = []
        if ink_result["ink_count"] == 1:
            warnings.append("Only 1 ink detected. This may indicate low color variation or incorrect parameters.")
        if ink_result["meta"].get("correction_applied", False):
            warnings.append("Mixing correction was applied. Middle tone was detected as a mixture of extremes.")

        return AutoDetectResponse(
            sku_code=request.sku_code,
            detected_inks=formatted_inks,
            suggested_zones=suggested_zones,
            ink_count=ink_result["ink_count"],
            meta=ink_result.get("meta", {}),
            warnings=warnings,
            message=f"Successfully detected {ink_result['ink_count']} ink(s). Review and approve the suggested zone configuration.",
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Auto-detect failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Auto-detect failed: {str(e)}")


def _map_inks_to_zones(inks: List[Dict[str, Any]]) -> Dict[str, ZoneLabValues]:
    """
    Map detected inks to zone configuration.

    Strategy:
    - Sort inks by L value (darkest to brightest)
    - Assign to zones A, B, C (A = darkest outer zone, C = brightest inner zone)
    - Calculate thresholds based on ink weights
    """
    # Sort by L value (darkest first)
    sorted_inks = sorted(inks, key=lambda x: x["lab"][0])

    zone_names = ["A", "B", "C"]
    suggested_zones = {}

    for i, ink in enumerate(sorted_inks):
        if i >= 3:  # Only support up to 3 zones
            break

        zone_name = zone_names[i]
        threshold = _calculate_threshold_from_weight(ink["weight"])

        # Generate description based on L value
        L_value = ink["lab"][0]
        if L_value < 40:
            tone_desc = "darkest"
        elif L_value < 70:
            tone_desc = "medium"
        else:
            tone_desc = "lightest"

        if i == 0:
            position_desc = "outer zone"
        elif i == 1:
            position_desc = "middle zone"
        else:
            position_desc = "inner zone"

        suggested_zones[zone_name] = ZoneLabValues(
            L=round(ink["lab"][0], 1),
            a=round(ink["lab"][1], 1),
            b=round(ink["lab"][2], 1),
            threshold=threshold,
            description=f"{position_desc.capitalize()} ({tone_desc}) - Auto-detected",
        )

    return suggested_zones


def _calculate_threshold_from_weight(weight: float) -> float:
    """
    Calculate suggested ΔE threshold based on ink weight.

    Logic:
    - Higher weight (dominant ink) → stricter threshold
    - Lower weight (minor ink) → looser threshold
    """
    if weight > 0.6:
        return 6.0  # Strict threshold for dominant ink
    elif weight > 0.3:
        return 8.0  # Medium threshold
    else:
        return 10.0  # Loose threshold for minor inks


# === Additional SKU Management Endpoints ===


class UpdateZonesRequest(BaseModel):
    """Request to update SKU zone configuration"""

    zones: Dict[str, ZoneLabValues] = Field(..., description="Zone configurations (A, B, C)")
    notes: Optional[str] = Field(None, description="Update notes")


class UpdateZonesResponse(BaseModel):
    """Response from update zones endpoint"""

    sku_code: str
    zones: Dict[str, Any]
    updated_at: str
    message: str


@router.put("/{sku_code}/zones", response_model=UpdateZonesResponse, status_code=status.HTTP_200_OK)
def update_sku_zones(sku_code: str, request: UpdateZonesRequest) -> UpdateZonesResponse:
    """
    Update SKU zone configuration.

    Use this endpoint to apply auto-detected zones or manually update zone values.
    """
    try:
        sku_manager = get_sku_manager()

        # Convert ZoneLabValues to dict format
        zones_dict = {}
        for zone_name, zone_values in request.zones.items():
            zones_dict[zone_name] = {
                "L": zone_values.L,
                "a": zone_values.a,
                "b": zone_values.b,
                "threshold": zone_values.threshold,
                "description": zone_values.description or "",
            }

        # Update SKU
        updates = {"zones": zones_dict}
        if request.notes:
            updates["metadata"] = {"notes": request.notes, "calibration_method": "auto-detected"}

        updated_sku = sku_manager.update_sku(sku_code, updates)

        return UpdateZonesResponse(
            sku_code=sku_code,
            zones=updated_sku["zones"],
            updated_at=updated_sku["metadata"]["last_updated"],
            message=f"SKU {sku_code} zones updated successfully",
        )

    except SkuNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except InvalidSkuDataError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Update zones failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Update failed: {str(e)}")


class GetSkuResponse(BaseModel):
    """Response from get SKU endpoint"""

    sku_code: str
    description: str
    zones: Dict[str, Any]
    metadata: Dict[str, Any]


@router.get("/{sku_code}", response_model=GetSkuResponse, status_code=status.HTTP_200_OK)
def get_sku(sku_code: str) -> GetSkuResponse:
    """Get SKU configuration"""
    try:
        sku_manager = get_sku_manager()
        sku_data = sku_manager.get_sku(sku_code)

        return GetSkuResponse(
            sku_code=sku_data["sku_code"],
            description=sku_data.get("description", ""),
            zones=sku_data.get("zones", {}),
            metadata=sku_data.get("metadata", {}),
        )

    except SkuNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Get SKU failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Get SKU failed: {str(e)}")


class ListSkusResponse(BaseModel):
    """Response from list SKUs endpoint"""

    skus: List[str]
    count: int


@router.get("/", response_model=ListSkusResponse, status_code=status.HTTP_200_OK)
def list_skus() -> ListSkusResponse:
    """List all available SKU codes"""
    try:
        sku_manager = get_sku_manager()
        skus = sku_manager.list_skus()

        return ListSkusResponse(skus=skus, count=len(skus))

    except Exception as e:
        logger.error(f"List SKUs failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"List SKUs failed: {str(e)}")
