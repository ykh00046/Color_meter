"""
Inspection History API Router

Endpoints for managing inspection history and querying past results.
"""

import csv
import io
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import and_, case, desc, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.models import InspectionHistory, JudgmentType
from src.models.database import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inspection", tags=["inspection"])


# ================================
# Query Builder
# ================================


class InspectionQueryBuilder:
    """
    Fluent query builder for InspectionHistory queries.

    Eliminates duplicate filter logic across multiple endpoints.
    """

    def __init__(self, db: Session):
        self.db = db
        self.query = db.query(InspectionHistory)

    def filter_sku(self, sku_code: Optional[str]) -> "InspectionQueryBuilder":
        if sku_code:
            self.query = self.query.filter(InspectionHistory.sku_code == sku_code)
        return self

    def filter_operator(self, operator: Optional[str]) -> "InspectionQueryBuilder":
        if operator:
            self.query = self.query.filter(InspectionHistory.operator == operator)
        return self

    def filter_batch(self, batch_number: Optional[str]) -> "InspectionQueryBuilder":
        if batch_number:
            self.query = self.query.filter(InspectionHistory.batch_number == batch_number)
        return self

    def filter_judgment(self, judgment: Optional[str]) -> "InspectionQueryBuilder":
        if judgment:
            try:
                judgment_enum = JudgmentType(judgment.upper())
                self.query = self.query.filter(InspectionHistory.judgment == judgment_enum)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid judgment value: {judgment}")
        return self

    def filter_delta_e_range(
        self, min_delta_e: Optional[float], max_delta_e: Optional[float]
    ) -> "InspectionQueryBuilder":
        if min_delta_e is not None:
            self.query = self.query.filter(InspectionHistory.overall_delta_e >= min_delta_e)
        if max_delta_e is not None:
            self.query = self.query.filter(InspectionHistory.overall_delta_e <= max_delta_e)
        return self

    def filter_date_range(
        self, start_date: Optional[datetime], end_date: Optional[datetime]
    ) -> "InspectionQueryBuilder":
        if start_date:
            self.query = self.query.filter(InspectionHistory.created_at >= start_date)
        if end_date:
            self.query = self.query.filter(InspectionHistory.created_at <= end_date)
        return self

    def filter_needs_action(self, needs_action_only: bool) -> "InspectionQueryBuilder":
        if needs_action_only:
            self.query = self.query.filter(InspectionHistory.judgment.in_([JudgmentType.NG, JudgmentType.RETAKE]))
        return self

    def filter_has_warnings(self, has_warnings: Optional[bool]) -> "InspectionQueryBuilder":
        if has_warnings is True:
            self.query = self.query.filter(InspectionHistory.has_warnings == 1)
        elif has_warnings is False:
            self.query = self.query.filter(InspectionHistory.has_warnings == 0)
        return self

    def order_by_created_desc(self) -> "InspectionQueryBuilder":
        self.query = self.query.order_by(desc(InspectionHistory.created_at))
        return self

    def paginate(self, skip: int, limit: int) -> "InspectionQueryBuilder":
        self.query = self.query.offset(skip).limit(limit)
        return self

    def count(self) -> int:
        return self.query.count()

    def all(self) -> List[InspectionHistory]:
        return self.query.all()

    def first(self) -> Optional[InspectionHistory]:
        return self.query.first()


def _calculate_pass_rate(pass_count: int, total: int) -> float:
    """Calculate pass rate with safe division."""
    return round((pass_count or 0) / total if total > 0 else 0.0, 4)


def _empty_stats_response(start_date: datetime, end_date: datetime, days: int = 7) -> Dict[str, Any]:
    """Return empty stats response for missing table or no data."""
    return {
        "total_inspections": 0,
        "judgment_counts": {},
        "pass_rate": 0.0,
        "avg_delta_e": 0.0,
        "avg_confidence": 0.0,
        "time_range": {"start": start_date.isoformat(), "end": end_date.isoformat(), "days": days},
    }


def _to_jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None
    from dataclasses import asdict, is_dataclass

    if is_dataclass(value):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


# ================================
# Helper Functions
# ================================


def save_inspection_to_history(
    db: Session,
    session_id: str,
    sku_code: str,
    image_filename: str,
    result: Any,
    image_path: Optional[str] = None,
    operator: Optional[str] = None,
    batch_number: Optional[str] = None,
    processing_time_ms: Optional[int] = None,
) -> InspectionHistory:
    """
    Save inspection result to database.

    Args:
        db: Database session
        session_id: Unique session ID (run_id)
        sku_code: SKU code
        image_filename: Original image filename
        result: InspectionResult object
        image_path: Full path to saved image (optional)
        operator: Operator name (optional)
        batch_number: Batch or lot identifier (optional)
        processing_time_ms: Processing time in milliseconds (optional)

    Returns:
        InspectionHistory object
    """
    # Extract judgment enum
    judgment_value = getattr(result, "judgment", "RETAKE")
    try:
        judgment = JudgmentType(judgment_value)
    except ValueError:
        logger.warning(f"Unknown judgment value: {judgment_value}, defaulting to RETAKE")
        judgment = JudgmentType.RETAKE

    # Extract core fields
    overall_delta_e = float(getattr(result, "overall_delta_e", 0.0))
    confidence = float(getattr(result, "confidence", 0.0))

    # Extract ng_reasons, retake_reasons, decision_trace, next_actions
    ng_reasons = _to_jsonable(getattr(result, "ng_reasons", []))
    retake_reasons = _to_jsonable(getattr(result, "retake_reasons", []))
    decision_trace = _to_jsonable(getattr(result, "decision_trace", None))
    next_actions = _to_jsonable(getattr(result, "next_actions", []))

    # Extract lens detection info
    lens_detection = getattr(result, "lens_detection", None)
    lens_detected = 1 if lens_detection is not None else 0
    lens_confidence = float(getattr(lens_detection, "confidence", 0.0)) if lens_detection else None

    # Build full analysis result JSON
    analysis_result = _to_jsonable(
        {
            "sku": sku_code,
            "timestamp": getattr(result, "timestamp", datetime.utcnow()).isoformat(),
            "judgment": judgment_value,
            "overall_delta_e": overall_delta_e,
            "confidence": confidence,
            "ng_reasons": ng_reasons,
            "retake_reasons": retake_reasons,
            "decision_trace": decision_trace,
            "next_actions": next_actions,
            "confidence_breakdown": getattr(result, "confidence_breakdown", None),
            "risk_factors": getattr(result, "risk_factors", []),
            "analysis_summary": getattr(result, "analysis_summary", None),
            "metrics": getattr(result, "metrics", None),
            "diagnostics": getattr(result, "diagnostics", []),
            "warnings": getattr(result, "warnings", []),
            "suggestions": getattr(result, "suggestions", []),
            "ink_analysis": getattr(result, "ink_analysis", None),
            "radial_profile": getattr(result, "radial_profile", None),
            "uniformity_analysis": getattr(result, "uniformity_analysis", None),
        }
    )

    # Analysis flags
    has_warnings = 1 if getattr(result, "warnings", []) else 0
    has_ink_analysis = 1 if getattr(result, "ink_analysis", None) else 0
    has_radial_profile = 1 if getattr(result, "radial_profile", None) else 0

    # Create record
    history = InspectionHistory(
        session_id=session_id,
        sku_code=sku_code,
        image_filename=image_filename,
        image_path=image_path,
        judgment=judgment,
        overall_delta_e=overall_delta_e,
        confidence=confidence,
        analysis_result=analysis_result,
        ng_reasons=ng_reasons if ng_reasons else None,
        retake_reasons=retake_reasons if retake_reasons else None,
        decision_trace=decision_trace,
        next_actions=next_actions if next_actions else None,
        lens_detected=lens_detected,
        lens_confidence=lens_confidence,
        created_at=datetime.utcnow(),
        operator=operator,
        batch_number=batch_number,
        notes=None,
        processing_time_ms=processing_time_ms,
        has_warnings=has_warnings,
        has_ink_analysis=has_ink_analysis,
        has_radial_profile=has_radial_profile,
    )

    db.add(history)
    db.commit()
    db.refresh(history)

    logger.info(f"Saved inspection history: session_id={session_id}, judgment={judgment_value}")

    return history


# ================================
# API Endpoints
# ================================


@router.get("/history")
def list_inspection_history(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of records to return"),
    sku_code: Optional[str] = Query(None, description="Filter by SKU code"),
    operator: Optional[str] = Query(None, description="Filter by operator"),
    batch_number: Optional[str] = Query(None, description="Filter by batch/lot number"),
    judgment: Optional[str] = Query(None, description="Filter by judgment (OK, OK_WITH_WARNING, NG, RETAKE)"),
    min_delta_e: Optional[float] = Query(None, ge=0, description="Minimum overall_delta_e"),
    max_delta_e: Optional[float] = Query(None, ge=0, description="Maximum overall_delta_e"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date (ISO 8601)"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date (ISO 8601)"),
    needs_action_only: bool = Query(False, description="Only show NG/RETAKE results"),
    has_warnings: Optional[bool] = Query(None, description="Filter by warning existence"),
    db: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    List inspection history with pagination and filtering.

    Returns:
        {
            "total": 123,
            "skip": 0,
            "limit": 50,
            "results": [...]
        }
    """
    builder = (
        InspectionQueryBuilder(db)
        .filter_sku(sku_code)
        .filter_operator(operator)
        .filter_batch(batch_number)
        .filter_judgment(judgment)
        .filter_delta_e_range(min_delta_e, max_delta_e)
        .filter_date_range(start_date, end_date)
        .filter_needs_action(needs_action_only)
        .filter_has_warnings(has_warnings)
    )

    total = builder.count()
    results = builder.order_by_created_desc().paginate(skip, limit).all()

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "results": [r.to_dict() for r in results],
    }


@router.get("/history/{history_id}")
def get_inspection_history(
    history_id: int,
    include_full_result: bool = Query(False, description="Include full analysis result"),
    db: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    Get single inspection history record by ID.

    Args:
        history_id: Inspection history ID
        include_full_result: If True, include full analysis_result JSON

    Returns:
        Inspection history record
    """
    history = db.query(InspectionHistory).filter(InspectionHistory.id == history_id).first()

    if not history:
        raise HTTPException(status_code=404, detail=f"Inspection history {history_id} not found")

    if include_full_result:
        return history.to_dict_full()
    else:
        return history.to_dict()


@router.get("/history/session/{session_id}")
def get_inspection_by_session(
    session_id: str,
    include_full_result: bool = Query(False, description="Include full analysis result"),
    db: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    Get inspection history by session ID (run_id).

    Args:
        session_id: Session ID (run_id from /inspect endpoint)
        include_full_result: If True, include full analysis_result JSON

    Returns:
        Inspection history record
    """
    history = db.query(InspectionHistory).filter(InspectionHistory.session_id == session_id).first()

    if not history:
        raise HTTPException(status_code=404, detail=f"Inspection session {session_id} not found")

    if include_full_result:
        return history.to_dict_full()
    else:
        return history.to_dict()


@router.delete("/history/{history_id}")
def delete_inspection_history(
    history_id: int,
    db: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    Delete inspection history record.

    Args:
        history_id: Inspection history ID

    Returns:
        {"message": "Deleted successfully"}
    """
    history = db.query(InspectionHistory).filter(InspectionHistory.id == history_id).first()

    if not history:
        raise HTTPException(status_code=404, detail=f"Inspection history {history_id} not found")

    db.delete(history)
    db.commit()

    logger.info(f"Deleted inspection history: id={history_id}")

    return {"message": f"Inspection history {history_id} deleted successfully"}


@router.get("/history/stats/summary")
def get_inspection_stats(
    sku_code: Optional[str] = Query(None, description="Filter by SKU code"),
    days: int = Query(7, ge=1, le=365, description="Number of days to include"),
    start_date: Optional[datetime] = Query(None, description="Start date (ISO 8601)"),
    end_date: Optional[datetime] = Query(None, description="End date (ISO 8601)"),
    db: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    Get inspection statistics summary.

    Returns:
        {
            "total_inspections": 123,
            "judgment_counts": {"OK": 100, "OK_WITH_WARNING": 10, "NG": 5, "RETAKE": 8},
            "pass_rate": 0.89,
            "avg_delta_e": 3.5,
            "avg_confidence": 0.85,
            "time_range": {"start": "...", "end": "..."}
        }
    """
    # Calculate date range
    end_date = end_date or datetime.utcnow()
    if start_date is None:
        start_date = end_date - timedelta(days=days)

    # Build base query
    query = db.query(InspectionHistory).filter(InspectionHistory.created_at >= start_date)

    if sku_code:
        query = query.filter(InspectionHistory.sku_code == sku_code)

    # Total inspections
    try:
        total_inspections = query.count()
    except SQLAlchemyError as exc:
        if "no such table: inspection_history" in str(exc):
            return {
                "total_inspections": 0,
                "judgment_counts": {},
                "pass_rate": 0.0,
                "avg_delta_e": 0.0,
                "avg_confidence": 0.0,
                "time_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            }
        raise

    if total_inspections == 0:
        return {
            "total_inspections": 0,
            "judgment_counts": {},
            "pass_rate": 0.0,
            "avg_delta_e": 0.0,
            "avg_confidence": 0.0,
            "time_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        }

    # Judgment counts
    try:
        judgment_counts_query = (
            query.with_entities(InspectionHistory.judgment, func.count(InspectionHistory.id).label("count"))
            .group_by(InspectionHistory.judgment)
            .all()
        )
    except SQLAlchemyError as exc:
        if "no such table: inspection_history" in str(exc):
            return {
                "total_inspections": 0,
                "judgment_counts": {},
                "pass_rate": 0.0,
                "avg_delta_e": 0.0,
                "avg_confidence": 0.0,
                "time_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            }
        raise

    judgment_counts = {j.value: count for j, count in judgment_counts_query}

    # Pass rate (OK + OK_WITH_WARNING)
    pass_count = judgment_counts.get("OK", 0) + judgment_counts.get("OK_WITH_WARNING", 0)
    pass_rate = pass_count / total_inspections if total_inspections > 0 else 0.0

    # Average delta_e and confidence
    try:
        avg_stats = query.with_entities(
            func.avg(InspectionHistory.overall_delta_e).label("avg_delta_e"),
            func.avg(InspectionHistory.confidence).label("avg_confidence"),
        ).first()
    except SQLAlchemyError as exc:
        if "no such table: inspection_history" in str(exc):
            return {
                "total_inspections": 0,
                "judgment_counts": {},
                "pass_rate": 0.0,
                "avg_delta_e": 0.0,
                "avg_confidence": 0.0,
                "time_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            }
        raise

    return {
        "total_inspections": total_inspections,
        "judgment_counts": judgment_counts,
        "pass_rate": round(pass_rate, 4),
        "avg_delta_e": round(float(avg_stats.avg_delta_e or 0.0), 2),
        "avg_confidence": round(float(avg_stats.avg_confidence or 0.0), 4),
        "time_range": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "days": days,
        },
    }


@router.get("/history/stats/by-sku")
def get_stats_by_sku(
    days: int = Query(7, ge=1, le=365, description="Number of days to include"),
    start_date: Optional[datetime] = Query(None, description="Start date (ISO 8601)"),
    end_date: Optional[datetime] = Query(None, description="End date (ISO 8601)"),
    db: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    Get inspection statistics grouped by SKU.

    Returns:
        {
            "SKU001": {
                "total": 50,
                "pass_rate": 0.92,
                "avg_delta_e": 3.2
            },
            "SKU002": {...}
        }
    """
    # Calculate date range
    end_date = end_date or datetime.utcnow()
    if start_date is None:
        start_date = end_date - timedelta(days=days)

    # Query by SKU
    try:
        results = (
            db.query(
                InspectionHistory.sku_code,
                func.count(InspectionHistory.id).label("total"),
                func.sum(
                    case(
                        (InspectionHistory.judgment.in_([JudgmentType.OK, JudgmentType.OK_WITH_WARNING]), 1),
                        else_=0,
                    )
                ).label("pass_count"),
                func.avg(InspectionHistory.overall_delta_e).label("avg_delta_e"),
                func.avg(InspectionHistory.confidence).label("avg_confidence"),
            )
            .filter(InspectionHistory.created_at >= start_date)
            .group_by(InspectionHistory.sku_code)
            .all()
        )
    except SQLAlchemyError as exc:
        if "no such table: inspection_history" in str(exc):
            return {
                "time_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "days": days,
                },
                "stats_by_sku": {},
            }
        raise

    stats_by_sku = {}
    for sku_code, total, pass_count, avg_delta_e, avg_confidence in results:
        pass_rate = (pass_count or 0) / total if total > 0 else 0.0
        stats_by_sku[sku_code] = {
            "total": total,
            "pass_count": pass_count or 0,
            "pass_rate": round(pass_rate, 4),
            "avg_delta_e": round(float(avg_delta_e or 0.0), 2),
            "avg_confidence": round(float(avg_confidence or 0.0), 4),
        }

    return {
        "time_range": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "days": days,
        },
        "stats_by_sku": stats_by_sku,
    }


@router.get("/history/stats/daily")
def get_daily_stats(
    sku_code: Optional[str] = Query(None, description="Filter by SKU code"),
    days: int = Query(30, ge=1, le=365, description="Number of days to include"),
    start_date: Optional[datetime] = Query(None, description="Start date (ISO 8601)"),
    end_date: Optional[datetime] = Query(None, description="End date (ISO 8601)"),
    db: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    Get daily inspection counts and pass rates for the last N days.

    Returns:
        {"series": [{"date": "2025-12-10", "total": 5, "pass": 4, "ng": 1, "retake": 0}, ...]}
    """
    end_dt = end_date.date() if end_date else datetime.utcnow().date()
    start_dt = start_date.date() if start_date else end_dt - timedelta(days=days - 1)

    try:
        query = db.query(
            func.date(InspectionHistory.created_at).label("date"),
            func.count(InspectionHistory.id).label("total"),
            func.sum(
                case(
                    (InspectionHistory.judgment.in_([JudgmentType.OK, JudgmentType.OK_WITH_WARNING]), 1),
                    else_=0,
                )
            ).label("pass_count"),
            func.sum(
                case(
                    (InspectionHistory.judgment == JudgmentType.NG, 1),
                    else_=0,
                )
            ).label("ng_count"),
            func.sum(
                case(
                    (InspectionHistory.judgment == JudgmentType.RETAKE, 1),
                    else_=0,
                )
            ).label("retake_count"),
        ).filter(
            func.date(InspectionHistory.created_at) >= start_dt,
            func.date(InspectionHistory.created_at) <= end_dt,
        )
    except SQLAlchemyError as exc:
        if "no such table: inspection_history" in str(exc):
            return {"start": start_dt.isoformat(), "end": end_dt.isoformat(), "series": []}
        raise

    if sku_code:
        query = query.filter(InspectionHistory.sku_code == sku_code)

    query = query.group_by(func.date(InspectionHistory.created_at)).order_by(func.date(InspectionHistory.created_at))

    series = []
    try:
        rows = query.all()
    except SQLAlchemyError as exc:
        if "no such table: inspection_history" in str(exc):
            return {"start": start_dt.isoformat(), "end": end_dt.isoformat(), "series": []}
        raise

    for row in rows:
        pass_rate = (row.pass_count or 0) / row.total if row.total else 0.0
        series.append(
            {
                "date": row.date.isoformat(),
                "total": row.total,
                "pass": row.pass_count or 0,
                "ng": row.ng_count or 0,
                "retake": row.retake_count or 0,
                "pass_rate": round(pass_rate, 4),
            }
        )

    return {"start": start_dt.isoformat(), "end": end_dt.isoformat(), "series": series}


@router.get("/history/stats/retake-reasons")
def get_retake_reason_stats(
    days: int = Query(30, ge=1, le=365, description="Number of days to include"),
    start_date: Optional[datetime] = Query(None, description="Start date (ISO 8601)"),
    end_date: Optional[datetime] = Query(None, description="End date (ISO 8601)"),
    sku_code: Optional[str] = Query(None, description="Filter by SKU code"),
    db: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    Aggregate RETAKE reasons by code.

    Returns:
        {"total": 10, "reasons": [{"code": "R1", "count": 4}, ...]}
    """
    end_dt = end_date or datetime.utcnow()
    start_dt = start_date or (end_dt - timedelta(days=days))

    query = db.query(InspectionHistory.retake_reasons).filter(
        InspectionHistory.created_at >= start_dt,
        InspectionHistory.created_at <= end_dt,
        InspectionHistory.judgment == JudgmentType.RETAKE,
    )

    if sku_code:
        query = query.filter(InspectionHistory.sku_code == sku_code)

    try:
        records = query.all()
    except SQLAlchemyError as exc:
        if "no such table: inspection_history" in str(exc):
            return {
                "total": 0,
                "reasons": [],
                "time_range": {"start": start_dt.isoformat(), "end": end_dt.isoformat(), "days": days},
            }
        raise

    reason_counts: Dict[str, int] = {}
    total = 0

    for (reasons,) in records:
        if not reasons:
            continue
        for item in reasons:
            code = item.get("code") or "UNKNOWN"
            reason_counts[code] = reason_counts.get(code, 0) + 1
            total += 1

    reasons = [{"code": code, "count": count} for code, count in sorted(reason_counts.items(), key=lambda x: x[0])]

    return {
        "total": total,
        "reasons": reasons,
        "time_range": {"start": start_dt.isoformat(), "end": end_dt.isoformat(), "days": days},
    }


@router.get("/history/export")
def export_inspection_history(
    sku_code: Optional[str] = Query(None, description="Filter by SKU code"),
    judgment: Optional[str] = Query(None, description="Filter by judgment"),
    operator: Optional[str] = Query(None, description="Filter by operator"),
    batch_number: Optional[str] = Query(None, description="Filter by batch/lot number"),
    start_date: Optional[datetime] = Query(None, description="Start date (ISO 8601)"),
    end_date: Optional[datetime] = Query(None, description="End date (ISO 8601)"),
    needs_action_only: bool = Query(False, description="Only export NG/RETAKE results"),
    db: Session = Depends(get_session),
) -> StreamingResponse:
    """Export inspection history as CSV."""
    rows = (
        InspectionQueryBuilder(db)
        .filter_sku(sku_code)
        .filter_operator(operator)
        .filter_batch(batch_number)
        .filter_judgment(judgment)
        .filter_date_range(start_date, end_date)
        .filter_needs_action(needs_action_only)
        .order_by_created_desc()
        .all()
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "id",
            "created_at",
            "session_id",
            "sku_code",
            "image_filename",
            "judgment",
            "overall_delta_e",
            "confidence",
            "operator",
            "batch_number",
            "processing_time_ms",
            "notes",
        ]
    )

    for row in rows:
        writer.writerow(
            [
                row.id,
                row.created_at.isoformat() if row.created_at else "",
                row.session_id,
                row.sku_code,
                row.image_filename,
                row.judgment.value if row.judgment else "",
                row.overall_delta_e,
                row.confidence,
                row.operator or "",
                row.batch_number or "",
                row.processing_time_ms or "",
                (row.notes or "").replace("\n", " "),
            ]
        )

    output.seek(0)
    filename = f"inspection_history_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers=headers)
