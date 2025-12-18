"""
STD (Standard) Models

SQLAlchemy models for STD statistical profiles.
Supports multiple samples per SKU for statistical modeling.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship

from .database import Base


class STDModel(Base):
    """
    STD Statistical Model

    Represents a statistical model built from multiple STD samples.
    Each STD model aggregates 5-10 samples to compute mean ± σ.
    """

    __tablename__ = "std_models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sku_code = Column(String(50), nullable=False, index=True)
    version = Column(String(20), nullable=False, default="v1.0")

    # Statistical metadata
    n_samples = Column(Integer, nullable=False)  # Number of samples (5-10)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    approved_by = Column(String(100))
    approved_at = Column(DateTime)

    # Status
    is_active = Column(Boolean, nullable=False, default=True, index=True)

    # Notes
    notes = Column(String(500))

    # Relationships
    samples = relationship("STDSample", back_populates="std_model", cascade="all, delete-orphan")
    statistics = relationship("STDStatistics", back_populates="std_model", cascade="all, delete-orphan")

    # Unique constraint: only one active version per SKU
    __table_args__ = (
        UniqueConstraint("sku_code", "version", name="uq_sku_version"),
        Index("idx_std_active", "sku_code", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<STDModel(id={self.id}, sku={self.sku_code}, version={self.version}, n_samples={self.n_samples})>"

    @hybrid_property
    def is_approved(self) -> bool:
        """Check if STD model is approved"""
        return self.approved_at is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "sku_code": self.sku_code,
            "version": self.version,
            "n_samples": self.n_samples,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "is_active": self.is_active,
            "is_approved": self.is_approved,
            "notes": self.notes,
        }


class STDSample(Base):
    """
    STD Sample

    Individual sample images that compose an STD statistical model.
    Each STD model has 5-10 samples.
    """

    __tablename__ = "std_samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    std_model_id = Column(Integer, ForeignKey("std_models.id", ondelete="CASCADE"), nullable=False, index=True)
    sample_index = Column(Integer, nullable=False)  # 1-based index (1, 2, 3, ...)

    # Image reference
    image_path = Column(String(500), nullable=False)

    # Analysis result (full JSON from pipeline)
    analysis_result = Column(JSON, nullable=False)

    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    file_size_bytes = Column(Integer)
    image_width = Column(Integer)
    image_height = Column(Integer)

    # Relationships
    std_model = relationship("STDModel", back_populates="samples")

    __table_args__ = (
        UniqueConstraint("std_model_id", "sample_index", name="uq_std_sample_index"),
        Index("idx_std_sample_model", "std_model_id", "sample_index"),
    )

    def __repr__(self) -> str:
        return f"<STDSample(id={self.id}, model_id={self.std_model_id}, index={self.sample_index})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "std_model_id": self.std_model_id,
            "sample_index": self.sample_index,
            "image_path": self.image_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "file_size_bytes": self.file_size_bytes,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "analysis_result": self.analysis_result,
        }


class STDStatistics(Base):
    """
    STD Statistics

    Statistical summaries (mean, std, percentiles) for each zone/feature.
    Separate rows for each zone to enable efficient querying.
    """

    __tablename__ = "std_statistics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    std_model_id = Column(Integer, ForeignKey("std_models.id", ondelete="CASCADE"), nullable=False, index=True)

    # Zone/Feature identifier
    # Examples: 'A', 'B', 'C', 'profile', 'boundary_AB', 'boundary_BC', 'ink_0', 'ink_1'
    zone_name = Column(String(20), nullable=False, index=True)

    # Color statistics (Lab color space)
    mean_L = Column(Float)
    std_L = Column(Float)
    mean_a = Column(Float)
    std_a = Column(Float)
    mean_b = Column(Float)
    std_b = Column(Float)

    # Structure statistics (positions, boundaries)
    mean_position = Column(Float)  # For boundaries: mean radius in pixels
    std_position = Column(Float)
    percentile_5 = Column(Float)  # 5th percentile
    percentile_95 = Column(Float)  # 95th percentile

    # Acceptance criteria (auto-derived from statistics)
    max_delta_e = Column(Float)  # Maximum allowed ΔE (e.g., 3σ)
    tolerance_position = Column(Float)  # Position tolerance (e.g., 2σ)

    # Detailed statistics (full covariance matrix, histograms, etc.)
    detailed_stats = Column(JSON)

    # Relationships
    std_model = relationship("STDModel", back_populates="statistics")

    __table_args__ = (
        UniqueConstraint("std_model_id", "zone_name", name="uq_std_stat_zone"),
        Index("idx_std_stat_model_zone", "std_model_id", "zone_name"),
        Index("idx_std_stat_lab", "mean_L", "mean_a", "mean_b"),
    )

    def __repr__(self) -> str:
        return f"<STDStatistics(id={self.id}, model_id={self.std_model_id}, zone={self.zone_name})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "std_model_id": self.std_model_id,
            "zone_name": self.zone_name,
            "color": (
                {
                    "mean_L": self.mean_L,
                    "std_L": self.std_L,
                    "mean_a": self.mean_a,
                    "std_a": self.std_a,
                    "mean_b": self.mean_b,
                    "std_b": self.std_b,
                }
                if self.mean_L is not None
                else None
            ),
            "structure": (
                {
                    "mean_position": self.mean_position,
                    "std_position": self.std_position,
                    "percentile_5": self.percentile_5,
                    "percentile_95": self.percentile_95,
                }
                if self.mean_position is not None
                else None
            ),
            "acceptance_criteria": {"max_delta_e": self.max_delta_e, "tolerance_position": self.tolerance_position},
            "detailed_stats": self.detailed_stats,
        }

    @hybrid_property
    def color_stats_available(self) -> bool:
        """Check if color statistics are available"""
        return self.mean_L is not None

    @hybrid_property
    def structure_stats_available(self) -> bool:
        """Check if structure statistics are available"""
        return self.mean_position is not None
