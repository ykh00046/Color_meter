"""
Test Models

SQLAlchemy models for test samples and comparison results.
"""

import enum
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Boolean, Column, DateTime, Enum, Float, ForeignKey, Index, Integer, String
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship

from .database import Base


class JudgmentStatus(enum.Enum):
    """Judgment status enumeration"""

    PASS = "PASS"
    FAIL = "FAIL"
    RETAKE = "RETAKE"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class TestSample(Base):
    """
    Test Sample

    Production sample to be compared against STD.
    """

    __tablename__ = "test_samples"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # SKU info
    sku_code = Column(String(50), nullable=False, index=True)

    # Sample metadata
    batch_number = Column(String(50), index=True)
    sample_id = Column(String(100), unique=True, index=True)  # Unique sample identifier

    # Image reference
    image_path = Column(String(500), nullable=False)

    # Analysis result (full JSON from pipeline)
    analysis_result = Column(JSON, nullable=False)

    # Lens detection quality
    lens_detected = Column(Boolean, nullable=False, default=False)
    lens_detection_score = Column(Float)  # 0.0 - 1.0

    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    operator = Column(String(100))  # Who took the sample
    notes = Column(String(500))

    # Image info
    file_size_bytes = Column(Integer)
    image_width = Column(Integer)
    image_height = Column(Integer)

    # Relationships
    comparison_results = relationship("ComparisonResult", back_populates="test_sample", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_test_sample_sku_batch", "sku_code", "batch_number"),
        Index("idx_test_sample_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<TestSample(id={self.id}, sku={self.sku_code}, sample_id={self.sample_id})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "sku_code": self.sku_code,
            "batch_number": self.batch_number,
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "lens_detected": self.lens_detected,
            "lens_detection_score": self.lens_detection_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "operator": self.operator,
            "notes": self.notes,
            "file_size_bytes": self.file_size_bytes,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "analysis_result": self.analysis_result,
        }


class ComparisonResult(Base):
    """
    Comparison Result

    Result of comparing a test sample against STD statistical model.
    Includes dual scoring (zone + ink), worst-case metrics, explainability.
    """

    __tablename__ = "comparison_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    test_sample_id = Column(Integer, ForeignKey("test_samples.id", ondelete="CASCADE"), nullable=False, index=True)
    std_model_id = Column(Integer, ForeignKey("std_models.id"), nullable=False, index=True)

    # Scores (0-100 scale)
    total_score = Column(Float, nullable=False, index=True)
    zone_score = Column(Float, nullable=False)  # Zone-based (structure)
    ink_score = Column(Float, nullable=False)  # Ink-based (color)
    confidence_score = Column(Float)  # Confidence (0-100)

    # Judgment
    judgment = Column(Enum(JudgmentStatus), nullable=False, index=True)

    # Explainability: Top 3 failure reasons
    top_failure_reasons = Column(JSON)
    # Example: [
    #   {"rank": 1, "category": "ZONE_COLOR", "message": "Zone B ΔE=8.5", "severity": 85},
    #   {"rank": 2, "category": "BOUNDARY", "message": "Zone A +12px shift", "severity": 61},
    #   ...
    # ]

    # Phenomenological classification
    defect_classifications = Column(JSON)
    # Example: [
    #   {"category": "COLOR_DEFECTS", "type": "UNDERDOSE", "zone": "B", "severity": 85},
    #   ...
    # ]

    # Detailed results (JSON)
    zone_details = Column(JSON)
    # Example: {
    #   "A": {"color": 92, "boundary": 88, "area": 95, "total": 91.7},
    #   "B": {"color": 65, "boundary": 72, "area": 88, "total": 75.0},
    #   "C": {"color": 94, "boundary": 91, "area": 96, "total": 93.7}
    # }

    ink_details = Column(JSON)
    # Example: {
    #   "count_penalty": 0,
    #   "matched_delta_e": [2.1, 3.4],
    #   "color_scores": [79, 66],
    #   "matching": [[0, 0], [1, 1]]
    # }

    alignment_details = Column(JSON)
    # Example: {
    #   "shift": 2.3,
    #   "correlation": 0.985,
    #   "rmse": 1.2,
    #   "ssim": 0.92
    # }

    worst_case_metrics = Column(JSON)
    # Example: {
    #   "percentiles": {"mean": 2.5, "p95": 7.8, "p99": 12.1, "max": 15.3},
    #   "hotspots": [
    #     {"area": 150, "centroid": [245, 320], "mean_delta_e": 9.2}
    #   ],
    #   "clusters": [...]
    # }

    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    processing_time_ms = Column(Integer)  # Processing time in milliseconds

    # Relationships
    test_sample = relationship("TestSample", back_populates="comparison_results")

    __table_args__ = (
        Index("idx_comparison_judgment", "judgment"),
        Index("idx_comparison_score", "total_score"),
        Index("idx_comparison_std", "std_model_id"),
        Index("idx_comparison_created", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<ComparisonResult(id={self.id}, test={self.test_sample_id}, std={self.std_model_id}, "
            f"judgment={self.judgment.value if self.judgment else None}, score={self.total_score})>"
        )

    @hybrid_property
    def is_pass(self) -> bool:
        """Check if result is PASS"""
        return self.judgment == JudgmentStatus.PASS

    @hybrid_property
    def needs_action(self) -> bool:
        """Check if result needs operator action (FAIL or MANUAL_REVIEW)"""
        return self.judgment in (JudgmentStatus.FAIL, JudgmentStatus.MANUAL_REVIEW)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "test_sample_id": self.test_sample_id,
            "std_model_id": self.std_model_id,
            "scores": {
                "total": self.total_score,
                "zone": self.zone_score,
                "ink": self.ink_score,
                "confidence": self.confidence_score,
            },
            "judgment": self.judgment.value if self.judgment else None,
            "is_pass": self.is_pass,
            "needs_action": self.needs_action,
            "top_failure_reasons": self.top_failure_reasons,
            "defect_classifications": self.defect_classifications,
            "zone_details": self.zone_details,
            "ink_details": self.ink_details,
            "alignment_details": self.alignment_details,
            "worst_case_metrics": self.worst_case_metrics,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processing_time_ms": self.processing_time_ms,
        }

    def get_failure_summary(self) -> str:
        """Get human-readable failure summary"""
        if self.judgment == JudgmentStatus.PASS:
            return "PASS"

        if not self.top_failure_reasons or len(self.top_failure_reasons) == 0:
            return f"{self.judgment.value} (No specific reason)"

        # Get top reason
        top = self.top_failure_reasons[0]
        return f"{self.judgment.value}: {top.get('message', 'Unknown')}"
