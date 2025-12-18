from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class ZoneResultSchema(BaseModel):
    zone_name: str
    delta_e: float
    threshold: float
    is_ok: bool


class InspectionResultSchema(BaseModel):
    sku: str
    timestamp: datetime
    judgment: str
    overall_delta_e: float
    confidence: float
    zone_results: List[ZoneResultSchema]
    ng_reasons: Optional[List[str]]


class InspectResponse(BaseModel):
    run_id: str
    image: str
    judgment: str
    overall_delta_e: float
    zones: int
    result_path: str
