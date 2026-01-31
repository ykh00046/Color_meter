from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class InspectionResultSchema(BaseModel):
    sku: str
    timestamp: datetime
    judgment: str
    overall_delta_e: float
    confidence: float
    ng_reasons: Optional[List[str]]


class InspectResponse(BaseModel):
    run_id: str
    image: str
    judgment: str
    overall_delta_e: float
    result_path: str
