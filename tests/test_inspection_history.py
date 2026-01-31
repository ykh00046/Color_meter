"""
Tests for Inspection History System

Tests the inspection history database models and API endpoints.
"""

import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models import InspectionHistory, JudgmentType
from src.models.database import Base
from src.schemas.inspection import InspectionResult
from src.web.routers.inspection import save_inspection_to_history


@pytest.fixture
def test_db():
    """Create an in-memory SQLite database for testing"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def sample_inspection_result():
    """Create a sample InspectionResult for testing (v7 schema)"""
    result = InspectionResult(
        sku="SKU001",
        timestamp=datetime.utcnow(),
        judgment="OK",
        overall_delta_e=2.65,
        ng_reasons=[],
        confidence=0.95,
        decision_trace={"final": "OK", "because": "All criteria passed", "overrides": []},
        next_actions=[],
        retake_reasons=[],
        analysis_summary={
            "radial": {"delta_e_mean": 2.5, "delta_e_p95": 3.2},
            "gate": {"passed": True, "scores": {"basic": 0.95}},
        },
        confidence_breakdown={
            "gate_passed": True,
            "signature_score": 0.95,
            "label": "OK",
        },
    )

    return result


def test_save_inspection_to_history(test_db, sample_inspection_result):
    """Test saving inspection result to database"""
    history = save_inspection_to_history(
        db=test_db,
        session_id="test_session_001",
        sku_code="SKU001",
        image_filename="test_image.jpg",
        result=sample_inspection_result,
        image_path="/path/to/test_image.jpg",
        operator="test_operator",
        processing_time_ms=1500,
    )

    assert history.id is not None
    assert history.session_id == "test_session_001"
    assert history.sku_code == "SKU001"
    assert history.image_filename == "test_image.jpg"
    assert history.judgment == JudgmentType.OK
    assert history.overall_delta_e == 2.65
    assert history.confidence == 0.95
    assert history.operator == "test_operator"
    assert history.processing_time_ms == 1500
    assert history.lens_detected == 0  # No lens_detection in sample
    assert history.has_warnings == 0
    assert history.has_ink_analysis == 0
    # v7 schema: analysis_summary and confidence_breakdown are present
    # Note: analysis_result is already a dict property, not a JSON string
    assert history.analysis_result is not None


def test_inspection_history_to_dict(test_db, sample_inspection_result):
    """Test InspectionHistory.to_dict() method"""
    history = save_inspection_to_history(
        db=test_db,
        session_id="test_session_002",
        sku_code="SKU002",
        image_filename="test2.jpg",
        result=sample_inspection_result,
    )

    data = history.to_dict()

    assert data["session_id"] == "test_session_002"
    assert data["sku_code"] == "SKU002"
    assert data["judgment"] == "OK"
    assert data["overall_delta_e"] == 2.65
    assert data["confidence"] == 0.95
    assert data["is_ok"] is True
    assert data["needs_action"] is False
    assert "analysis_result" not in data  # to_dict() doesn't include full result


def test_inspection_history_to_dict_full(test_db, sample_inspection_result):
    """Test InspectionHistory.to_dict_full() method"""
    history = save_inspection_to_history(
        db=test_db,
        session_id="test_session_003",
        sku_code="SKU003",
        image_filename="test3.jpg",
        result=sample_inspection_result,
    )

    data = history.to_dict_full()

    assert "analysis_result" in data
    assert data["analysis_result"]["judgment"] == "OK"
    # v7 schema: check for analysis_summary instead of zone_results
    assert "analysis_summary" in data["analysis_result"]
    assert "confidence_breakdown" in data["analysis_result"]


def test_inspection_history_judgment_types(test_db):
    """Test different judgment types"""
    from dataclasses import dataclass

    @dataclass
    class MockResult:
        judgment: str
        overall_delta_e: float
        confidence: float
        ng_reasons: list
        decision_trace: dict
        next_actions: list
        retake_reasons: list

    # Test OK
    result_ok = MockResult(
        judgment="OK",
        overall_delta_e=2.0,
        confidence=0.95,
        ng_reasons=[],
        decision_trace={},
        next_actions=[],
        retake_reasons=[],
    )
    history_ok = save_inspection_to_history(
        db=test_db,
        session_id="test_ok",
        sku_code="SKU001",
        image_filename="ok.jpg",
        result=result_ok,
    )
    assert history_ok.judgment == JudgmentType.OK
    assert history_ok.is_ok is True
    assert history_ok.needs_action is False

    # Test NG
    result_ng = MockResult(
        judgment="NG",
        overall_delta_e=8.5,
        confidence=0.70,
        ng_reasons=["Zone B ΔE too high"],
        decision_trace={},
        next_actions=["Check lens quality"],
        retake_reasons=[],
    )
    history_ng = save_inspection_to_history(
        db=test_db,
        session_id="test_ng",
        sku_code="SKU002",
        image_filename="ng.jpg",
        result=result_ng,
    )
    assert history_ng.judgment == JudgmentType.NG
    assert history_ng.is_ok is False
    assert history_ng.needs_action is True
    assert history_ng.ng_reasons == ["Zone B ΔE too high"]

    # Test RETAKE
    result_retake = MockResult(
        judgment="RETAKE",
        overall_delta_e=3.0,
        confidence=0.45,
        ng_reasons=[],
        decision_trace={},
        next_actions=["Retake image"],
        retake_reasons=[{"code": "R1", "reason": "Low confidence"}],
    )
    history_retake = save_inspection_to_history(
        db=test_db,
        session_id="test_retake",
        sku_code="SKU003",
        image_filename="retake.jpg",
        result=result_retake,
    )
    assert history_retake.judgment == JudgmentType.RETAKE
    assert history_retake.is_ok is False
    assert history_retake.needs_action is True
    assert len(history_retake.retake_reasons) == 1


def test_query_by_sku(test_db, sample_inspection_result):
    """Test querying inspection history by SKU"""
    # Save multiple records with different SKUs
    for i in range(3):
        save_inspection_to_history(
            db=test_db,
            session_id=f"test_session_{i}",
            sku_code="SKU001",
            image_filename=f"test{i}.jpg",
            result=sample_inspection_result,
        )

    for i in range(2):
        save_inspection_to_history(
            db=test_db,
            session_id=f"test_session_other_{i}",
            sku_code="SKU002",
            image_filename=f"other{i}.jpg",
            result=sample_inspection_result,
        )

    # Query by SKU001
    results_sku001 = test_db.query(InspectionHistory).filter(InspectionHistory.sku_code == "SKU001").all()
    assert len(results_sku001) == 3

    # Query by SKU002
    results_sku002 = test_db.query(InspectionHistory).filter(InspectionHistory.sku_code == "SKU002").all()
    assert len(results_sku002) == 2


def test_query_by_judgment(test_db):
    """Test querying inspection history by judgment"""
    from dataclasses import dataclass

    @dataclass
    class MockResult:
        judgment: str
        overall_delta_e: float
        confidence: float
        ng_reasons: list
        decision_trace: dict
        next_actions: list
        retake_reasons: list

    # Save OK result
    result_ok = MockResult(
        judgment="OK",
        overall_delta_e=2.0,
        confidence=0.95,
        ng_reasons=[],
        decision_trace={},
        next_actions=[],
        retake_reasons=[],
    )
    save_inspection_to_history(test_db, "ok_1", "SKU001", "ok.jpg", result_ok)

    # Save NG result
    result_ng = MockResult(
        judgment="NG",
        overall_delta_e=8.5,
        confidence=0.70,
        ng_reasons=[],
        decision_trace={},
        next_actions=[],
        retake_reasons=[],
    )
    save_inspection_to_history(test_db, "ng_1", "SKU001", "ng.jpg", result_ng)

    # Query OK
    results_ok = test_db.query(InspectionHistory).filter(InspectionHistory.judgment == JudgmentType.OK).all()
    assert len(results_ok) == 1

    # Query NG
    results_ng = test_db.query(InspectionHistory).filter(InspectionHistory.judgment == JudgmentType.NG).all()
    assert len(results_ng) == 1


def test_get_summary(test_db):
    """Test InspectionHistory.get_summary() method"""
    from dataclasses import dataclass

    @dataclass
    class MockResult:
        judgment: str
        overall_delta_e: float
        confidence: float
        ng_reasons: list
        decision_trace: dict
        next_actions: list
        retake_reasons: list

    # OK
    result_ok = MockResult(
        judgment="OK",
        overall_delta_e=2.5,
        confidence=0.95,
        ng_reasons=[],
        decision_trace={},
        next_actions=[],
        retake_reasons=[],
    )
    history_ok = save_inspection_to_history(test_db, "test_ok", "SKU001", "ok.jpg", result_ok)
    summary = history_ok.get_summary()
    assert "OK" in summary
    assert "2.50" in summary
    assert "95" in summary  # Can be 95.0% or 95.00%

    # NG
    result_ng = MockResult(
        judgment="NG",
        overall_delta_e=8.5,
        confidence=0.70,
        ng_reasons=["Zone B color mismatch"],
        decision_trace={},
        next_actions=[],
        retake_reasons=[],
    )
    history_ng = save_inspection_to_history(test_db, "test_ng", "SKU002", "ng.jpg", result_ng)
    summary = history_ng.get_summary()
    assert "NG" in summary
    assert "Zone B color mismatch" in summary
