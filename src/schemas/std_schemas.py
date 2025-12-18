"""
STD Schemas

Pydantic models for STD registration and management APIs.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class ZoneColorData(BaseModel):
    """Zone color data (Lab color space)"""

    L: float = Field(..., description="Lightness (0-100)")
    a: float = Field(..., description="a* (-128 to 127)")
    b: float = Field(..., description="b* (-128 to 127)")

    class Config:
        json_schema_extra = {"example": {"L": 85.5, "a": 3.2, "b": -5.1}}


class ZoneBoundaryData(BaseModel):
    """Zone boundary positions"""

    inner_to_A: Optional[float] = Field(None, description="Inner to Zone A transition (pixels)")
    A_to_B: Optional[float] = Field(None, description="Zone A to B transition (pixels)")
    B_to_C: Optional[float] = Field(None, description="Zone B to C transition (pixels)")
    C_outer: Optional[float] = Field(None, description="Zone C outer boundary (pixels)")

    class Config:
        json_schema_extra = {"example": {"inner_to_A": 120.5, "A_to_B": 245.8, "B_to_C": 380.2, "C_outer": 512.0}}


class STDProfileData(BaseModel):
    """STD Profile complete data structure"""

    sku_code: str = Field(..., description="SKU code")
    version: str = Field(default="v1.0", description="Version identifier")

    # Zone colors
    zone_colors: Dict[str, ZoneColorData] = Field(..., description="Zone colors (A, B, C)")

    # Zone boundaries
    zone_boundaries: ZoneBoundaryData = Field(..., description="Zone boundary positions")

    # Radial profile (L, a, b arrays)
    radial_profile_L: List[float] = Field(
        ..., description="Radial profile Lightness (500 points)", min_length=100, max_length=1000
    )
    radial_profile_a: List[float] = Field(
        ..., description="Radial profile a* (500 points)", min_length=100, max_length=1000
    )
    radial_profile_b: List[float] = Field(
        ..., description="Radial profile b* (500 points)", min_length=100, max_length=1000
    )

    # Ink analysis (optional)
    ink_colors: Optional[List[ZoneColorData]] = Field(None, description="Detected ink colors")
    n_inks: Optional[int] = Field(None, description="Number of inks detected", ge=1, le=5)

    # Metadata
    image_width: int = Field(..., description="Image width (pixels)", gt=0)
    image_height: int = Field(..., description="Image height (pixels)", gt=0)
    lens_detected: bool = Field(default=True, description="Lens detection success")
    lens_detection_score: Optional[float] = Field(None, description="Lens detection confidence (0-1)", ge=0.0, le=1.0)

    @validator("radial_profile_a", "radial_profile_b")
    def validate_profile_length(cls, v, values):
        """Ensure all radial profiles have same length"""
        if "radial_profile_L" in values:
            if len(v) != len(values["radial_profile_L"]):
                raise ValueError("All radial profiles must have same length")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "sku_code": "SKU001",
                "version": "v1.0",
                "zone_colors": {
                    "A": {"L": 85.5, "a": 3.2, "b": -5.1},
                    "B": {"L": 65.2, "a": 15.8, "b": -45.3},
                    "C": {"L": 75.1, "a": 2.5, "b": -8.2},
                },
                "zone_boundaries": {"inner_to_A": 120.5, "A_to_B": 245.8, "B_to_C": 380.2, "C_outer": 512.0},
                "radial_profile_L": [85.0] * 500,
                "radial_profile_a": [3.0] * 500,
                "radial_profile_b": [-5.0] * 500,
                "ink_colors": [{"L": 65.2, "a": 15.8, "b": -45.3}],
                "n_inks": 1,
                "image_width": 1024,
                "image_height": 1024,
                "lens_detected": True,
                "lens_detection_score": 0.98,
            }
        }


class STDRegisterRequest(BaseModel):
    """Request to register new STD profile"""

    sku_code: str = Field(..., description="SKU code", min_length=3, max_length=50)
    version: str = Field(default="v1.0", description="Version identifier", max_length=20)
    image_path: str = Field(..., description="Path to STD image file")
    notes: Optional[str] = Field(None, description="Optional notes", max_length=500)

    # Optional: provide pre-analyzed profile data
    profile_data: Optional[STDProfileData] = Field(None, description="Pre-analyzed profile data (if already analyzed)")

    class Config:
        json_schema_extra = {
            "example": {
                "sku_code": "SKU001",
                "version": "v1.0",
                "image_path": "/data/std/SKU001_std.png",
                "notes": "Initial STD registration for SKU001",
            }
        }


class STDRegisterResponse(BaseModel):
    """Response after STD registration"""

    id: int = Field(..., description="STD model database ID")
    sku_code: str = Field(..., description="SKU code")
    version: str = Field(..., description="Version identifier")
    created_at: datetime = Field(..., description="Creation timestamp")
    is_active: bool = Field(..., description="Active status")

    # Profile summary
    n_zones: int = Field(..., description="Number of zones detected", ge=1, le=5)
    profile_length: int = Field(..., description="Radial profile length (points)")

    message: str = Field(default="STD registered successfully", description="Success message")

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "sku_code": "SKU001",
                "version": "v1.0",
                "created_at": "2025-12-17T10:30:00Z",
                "is_active": True,
                "n_zones": 3,
                "profile_length": 500,
                "message": "STD registered successfully",
            }
        }


class STDListItem(BaseModel):
    """STD list item (summary)"""

    id: int
    sku_code: str
    version: str
    created_at: datetime
    is_active: bool
    is_approved: bool = Field(..., description="Approval status")
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class STDListResponse(BaseModel):
    """Response for STD list query"""

    total: int = Field(..., description="Total number of STD profiles")
    items: List[STDListItem] = Field(..., description="STD profile list")

    class Config:
        json_schema_extra = {
            "example": {
                "total": 2,
                "items": [
                    {
                        "id": 1,
                        "sku_code": "SKU001",
                        "version": "v1.0",
                        "created_at": "2025-12-17T10:30:00Z",
                        "is_active": True,
                        "is_approved": True,
                        "approved_by": "admin",
                        "approved_at": "2025-12-17T11:00:00Z",
                    },
                    {
                        "id": 2,
                        "sku_code": "SKU002",
                        "version": "v1.0",
                        "created_at": "2025-12-17T12:00:00Z",
                        "is_active": True,
                        "is_approved": False,
                        "approved_by": None,
                        "approved_at": None,
                    },
                ],
            }
        }


class STDDetailResponse(BaseModel):
    """Response for STD detail query"""

    id: int
    sku_code: str
    version: str
    created_at: datetime
    is_active: bool
    is_approved: bool
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    notes: Optional[str] = None

    # Full profile data
    profile_data: STDProfileData = Field(..., description="Complete STD profile data")

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "sku_code": "SKU001",
                "version": "v1.0",
                "created_at": "2025-12-17T10:30:00Z",
                "is_active": True,
                "is_approved": True,
                "approved_by": "admin",
                "approved_at": "2025-12-17T11:00:00Z",
                "notes": "Initial STD registration",
                "profile_data": {
                    "sku_code": "SKU001",
                    "version": "v1.0",
                    "zone_colors": {
                        "A": {"L": 85.5, "a": 3.2, "b": -5.1},
                        "B": {"L": 65.2, "a": 15.8, "b": -45.3},
                        "C": {"L": 75.1, "a": 2.5, "b": -8.2},
                    },
                    "zone_boundaries": {"A_to_B": 245.8, "B_to_C": 380.2, "C_outer": 512.0},
                    "radial_profile_L": [85.0] * 500,
                    "radial_profile_a": [3.0] * 500,
                    "radial_profile_b": [-5.0] * 500,
                    "n_inks": 1,
                    "image_width": 1024,
                    "image_height": 1024,
                    "lens_detected": True,
                    "lens_detection_score": 0.98,
                },
            }
        }
