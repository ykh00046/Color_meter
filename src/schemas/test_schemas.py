"""
Test Sample Pydantic Schemas

Request/Response models for Test Sample API.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TestRegisterRequest(BaseModel):
    """Test Sample registration request"""

    sku_code: str = Field(..., description="SKU code", min_length=1, max_length=50)
    image_path: str = Field(..., description="Path to test image file")
    batch_number: Optional[str] = Field(None, description="Batch number", max_length=50)
    sample_id: Optional[str] = Field(None, description="Unique sample identifier", max_length=100)
    operator: Optional[str] = Field(None, description="Operator name", max_length=100)
    notes: Optional[str] = Field(None, description="Notes about this sample", max_length=500)

    class Config:
        json_schema_extra = {
            "example": {
                "sku_code": "SKU001",
                "image_path": "data/raw_images/SKU001_OK_002.jpg",
                "batch_number": "B001",
                "sample_id": "SKU001-B001-001",
                "operator": "홍길동",
                "notes": "양산 샘플 검사",
            }
        }


class TestRegisterResponse(BaseModel):
    """Test Sample registration response"""

    id: int = Field(..., description="Test sample ID")
    sku_code: str = Field(..., description="SKU code")
    batch_number: Optional[str] = Field(None, description="Batch number")
    sample_id: Optional[str] = Field(None, description="Sample identifier")
    lens_detected: bool = Field(..., description="Lens detection success")
    lens_detection_score: Optional[float] = Field(None, description="Lens detection score (0-1)")
    created_at: datetime = Field(..., description="Created timestamp")
    operator: Optional[str] = Field(None, description="Operator name")
    message: str = Field(..., description="Success message")

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "sku_code": "SKU001",
                "batch_number": "B001",
                "sample_id": "SKU001-B001-001",
                "lens_detected": True,
                "lens_detection_score": 0.95,
                "created_at": "2025-12-18T10:30:00",
                "operator": "홍길동",
                "message": "Test sample registered successfully",
            }
        }


class TestListItem(BaseModel):
    """Test Sample list item"""

    id: int = Field(..., description="Test sample ID")
    sku_code: str = Field(..., description="SKU code")
    batch_number: Optional[str] = Field(None, description="Batch number")
    sample_id: Optional[str] = Field(None, description="Sample identifier")
    lens_detected: bool = Field(..., description="Lens detection success")
    created_at: datetime = Field(..., description="Created timestamp")
    operator: Optional[str] = Field(None, description="Operator name")

    class Config:
        from_attributes = True


class TestListResponse(BaseModel):
    """Test Sample list response"""

    total: int = Field(..., description="Total number of items")
    items: List[TestListItem] = Field(..., description="List of test samples")

    class Config:
        json_schema_extra = {
            "example": {
                "total": 2,
                "items": [
                    {
                        "id": 1,
                        "sku_code": "SKU001",
                        "batch_number": "B001",
                        "sample_id": "SKU001-B001-001",
                        "lens_detected": True,
                        "created_at": "2025-12-18T10:30:00",
                        "operator": "홍길동",
                    }
                ],
            }
        }


class TestDetailResponse(BaseModel):
    """Test Sample detail response"""

    id: int = Field(..., description="Test sample ID")
    sku_code: str = Field(..., description="SKU code")
    batch_number: Optional[str] = Field(None, description="Batch number")
    sample_id: Optional[str] = Field(None, description="Sample identifier")
    image_path: str = Field(..., description="Image file path")
    lens_detected: bool = Field(..., description="Lens detection success")
    lens_detection_score: Optional[float] = Field(None, description="Lens detection score (0-1)")
    created_at: datetime = Field(..., description="Created timestamp")
    operator: Optional[str] = Field(None, description="Operator name")
    notes: Optional[str] = Field(None, description="Notes")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes")
    image_width: Optional[int] = Field(None, description="Image width (pixels)")
    image_height: Optional[int] = Field(None, description="Image height (pixels)")
    analysis_result: Dict[str, Any] = Field(..., description="Full analysis result from pipeline")

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "sku_code": "SKU001",
                "batch_number": "B001",
                "sample_id": "SKU001-B001-001",
                "image_path": "data/raw_images/SKU001_OK_002.jpg",
                "lens_detected": True,
                "lens_detection_score": 0.95,
                "created_at": "2025-12-18T10:30:00",
                "operator": "홍길동",
                "notes": "양산 샘플 검사",
                "file_size_bytes": 2048000,
                "image_width": 1920,
                "image_height": 1080,
                "analysis_result": {"sku": "SKU001", "judgment": "OK", "overall_delta_e": 2.5},
            }
        }
