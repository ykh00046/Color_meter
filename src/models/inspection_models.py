"""
Inspection History Models

SQLAlchemy models for storing inspection results and history.
"""

import enum
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Column, DateTime, Enum, Float, Index, Integer, String, Text
from sqlalchemy.ext.hybrid import hybrid_property

from .database import Base


class JudgmentType(enum.Enum):
    """Judgment type enumeration for inspection results"""

    OK = "OK"
    OK_WITH_WARNING = "OK_WITH_WARNING"
    NG = "NG"
    RETAKE = "RETAKE"


class InspectionHistory(Base):
    """
    Inspection History

    Stores inspection results for single lens images.
    Each record represents one inspection operation.
    """

    __tablename__ = "inspection_history"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Session info
    session_id = Column(String(100), unique=True, nullable=False, index=True)  # run_id from web API

    # SKU and image info
    sku_code = Column(String(50), nullable=False, index=True)
    image_filename = Column(String(500), nullable=False)  # Original filename
    image_path = Column(String(1000))  # Full path to saved image (optional)

    # Inspection results (core fields for quick queries)
    judgment = Column(Enum(JudgmentType), nullable=False, index=True)
    overall_delta_e = Column(Float, nullable=False, index=True)
    confidence = Column(Float, nullable=False)  # 0.0 - 1.0

    # Full analysis result (JSON blob for detailed data)
    # Contains: decision_trace, next_actions, retake_reasons,
    #           confidence_breakdown, risk_factors, ink_analysis, radial_profile, etc.
    analysis_result = Column(JSON, nullable=False)

    # NG/RETAKE reasons (extracted for quick access)
    ng_reasons = Column(JSON)  # List[str]
    retake_reasons = Column(JSON)  # List[Dict] with code, reason, actions, lever

    # Decision trace (extracted for explainability)
    decision_trace = Column(JSON)  # Dict with final, because, overrides

    # Next actions (extracted for operator guidance)
    next_actions = Column(JSON)  # List[str]

    # Lens detection quality
    lens_detected = Column(Integer, nullable=False, default=1)  # 1=detected, 0=not detected
    lens_confidence = Column(Float)  # Lens detection confidence (0.0 - 1.0)

    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    operator = Column(String(100))  # Who performed the inspection
    batch_number = Column(String(100))  # Batch or lot identifier
    notes = Column(Text)  # Additional notes

    # Performance metrics
    processing_time_ms = Column(Integer)  # Processing time in milliseconds

    # Analysis flags (for quick filtering)
    has_warnings = Column(Integer, nullable=False, default=0)  # 1 if warnings exist
    has_ink_analysis = Column(Integer, nullable=False, default=0)  # 1 if ink analysis exists
    has_radial_profile = Column(Integer, nullable=False, default=0)  # 1 if radial profile exists

    # Indices for performance
    __table_args__ = (
        Index("idx_inspection_sku_judgment", "sku_code", "judgment"),
        Index("idx_inspection_created", "created_at"),
        Index("idx_inspection_delta_e", "overall_delta_e"),
        Index("idx_inspection_confidence", "confidence"),
        Index("idx_inspection_batch_number", "batch_number"),
    )

    def __repr__(self) -> str:
        return (
            f"<InspectionHistory(id={self.id}, session_id={self.session_id}, "
            f"sku={self.sku_code}, judgment={self.judgment.value if self.judgment else None})>"
        )

    @hybrid_property
    def is_ok(self) -> bool:
        """Check if inspection passed (OK or OK_WITH_WARNING)"""
        return self.judgment in (JudgmentType.OK, JudgmentType.OK_WITH_WARNING)

    @hybrid_property
    def needs_action(self) -> bool:
        """Check if inspection needs operator action (NG or RETAKE)"""
        return self.judgment in (JudgmentType.NG, JudgmentType.RETAKE)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "sku_code": self.sku_code,
            "image_filename": self.image_filename,
            "image_path": self.image_path,
            "judgment": self.judgment.value if self.judgment else None,
            "overall_delta_e": self.overall_delta_e,
            "confidence": self.confidence,
            "ng_reasons": self.ng_reasons,
            "retake_reasons": self.retake_reasons,
            "decision_trace": self.decision_trace,
            "next_actions": self.next_actions,
            "lens_detected": bool(self.lens_detected),
            "lens_confidence": self.lens_confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "operator": self.operator,
            "batch_number": self.batch_number,
            "notes": self.notes,
            "processing_time_ms": self.processing_time_ms,
            "has_warnings": bool(self.has_warnings),
            "has_ink_analysis": bool(self.has_ink_analysis),
            "has_radial_profile": bool(self.has_radial_profile),
            "is_ok": self.is_ok,
            "needs_action": self.needs_action,
        }

    def to_dict_full(self) -> Dict[str, Any]:
        """Convert to dictionary with full analysis result"""
        data = self.to_dict()
        data["analysis_result"] = self.analysis_result
        return data

    def get_summary(self) -> str:
        """Get human-readable summary"""
        judgment_str = self.judgment.value if self.judgment else "UNKNOWN"

        if self.judgment == JudgmentType.OK:
            return f"OK (ΔE={self.overall_delta_e:.2f}, confidence={self.confidence:.2%})"
        elif self.judgment == JudgmentType.OK_WITH_WARNING:
            return f"OK with warnings (ΔE={self.overall_delta_e:.2f}, confidence={self.confidence:.2%})"
        elif self.judgment == JudgmentType.NG:
            reasons = ", ".join(self.ng_reasons) if self.ng_reasons else "No specific reason"
            return f"NG: {reasons}"
        elif self.judgment == JudgmentType.RETAKE:
            if self.retake_reasons and len(self.retake_reasons) > 0:
                top_reason = self.retake_reasons[0]
                return f"RETAKE: {top_reason.get('reason', 'Unknown')}"
            return "RETAKE (No specific reason)"

        return judgment_str
