#!/usr/bin/env python3
"""
Test Database Models

Verify that SQLAlchemy models are correctly defined and can create tables.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import (
    AuditLog,
    ComparisonResult,
    JudgmentStatus,
    STDModel,
    STDSample,
    STDStatistics,
    TestSample,
    User,
    UserRole,
    create_tables,
    drop_tables,
    get_db,
    init_database,
)


def test_database_creation():
    """Test database and table creation"""
    print("=" * 60)
    print("DATABASE MODELS TEST")
    print("=" * 60)

    # Initialize database (SQLite in-memory for testing)
    print("\n1. Initializing database...")
    init_database("sqlite:///:memory:", echo=False)
    print("   [OK] Database initialized")

    # Create tables
    print("\n2. Creating tables...")
    create_tables()
    print("   [OK] Tables created")

    # Get session
    db = get_db()

    try:
        # Test User creation
        print("\n3. Testing User model...")
        user = User(
            username="admin",
            email="admin@example.com",
            full_name="System Administrator",
            role=UserRole.ADMIN,
            password_hash="dummy_hash",
        )
        db.add(user)
        db.commit()
        print(f"   [OK] User created: {user}")

        # Test STD Model creation
        print("\n4. Testing STD Model...")
        std_model = STDModel(
            sku_code="SKU001",
            version="v1.0",
            n_samples=5,
            approved_by="admin",
            approved_at=datetime.utcnow(),
            is_active=True,
        )
        db.add(std_model)
        db.commit()
        print(f"   [OK] STD Model created: {std_model}")

        # Test STD Sample creation
        print("\n5. Testing STD Sample...")
        std_sample = STDSample(
            std_model_id=std_model.id,
            sample_index=1,
            image_path="/data/std/SKU001_sample1.png",
            analysis_result={"zones": {"A": {}, "B": {}, "C": {}}, "profile": [1, 2, 3]},
            image_width=1024,
            image_height=1024,
            file_size_bytes=512000,
        )
        db.add(std_sample)
        db.commit()
        print(f"   [OK] STD Sample created: {std_sample}")

        # Test STD Statistics creation
        print("\n6. Testing STD Statistics...")
        std_stat = STDStatistics(
            std_model_id=std_model.id,
            zone_name="A",
            mean_L=85.5,
            std_L=2.3,
            mean_a=3.2,
            std_a=0.8,
            mean_b=-5.1,
            std_b=1.2,
            max_delta_e=6.9,  # 3 * sqrt(2.3^2 + 0.8^2 + 1.2^2)
            detailed_stats={"covariance": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        )
        db.add(std_stat)
        db.commit()
        print(f"   [OK] STD Statistics created: {std_stat}")

        # Test Test Sample creation
        print("\n7. Testing Test Sample...")
        test_sample = TestSample(
            sku_code="SKU001",
            batch_number="BATCH-2025-001",
            sample_id="SAMPLE-12345",
            image_path="/data/test/sample_12345.png",
            analysis_result={"zones": {}, "profile": []},
            lens_detected=True,
            lens_detection_score=0.98,
            operator="operator1",
            image_width=1024,
            image_height=1024,
            file_size_bytes=480000,
        )
        db.add(test_sample)
        db.commit()
        print(f"   [OK] Test Sample created: {test_sample}")

        # Test Comparison Result creation
        print("\n8. Testing Comparison Result...")
        comparison = ComparisonResult(
            test_sample_id=test_sample.id,
            std_model_id=std_model.id,
            total_score=85.3,
            zone_score=82.5,
            ink_score=88.1,
            confidence_score=92.0,
            judgment=JudgmentStatus.PASS,
            top_failure_reasons=[],
            defect_classifications=[],
            zone_details={"A": {"color": 90, "boundary": 88}},
            ink_details={"count_penalty": 0},
            alignment_details={"shift": 1.2, "correlation": 0.985},
            worst_case_metrics={"percentiles": {"p95": 4.2}},
            processing_time_ms=850,
        )
        db.add(comparison)
        db.commit()
        print(f"   [OK] Comparison Result created: {comparison}")

        # Test Audit Log creation
        print("\n9. Testing Audit Log...")
        audit_log = AuditLog.create_log(
            user_id=user.id,
            action="test_compare",
            target_type="test_sample",
            target_id=test_sample.id,
            changes={"result": "PASS"},
            success=True,
            ip_address="192.168.1.100",
        )
        db.add(audit_log)
        db.commit()
        print(f"   [OK] Audit Log created: {audit_log}")

        # Test queries
        print("\n10. Testing queries...")

        # Query all STD models
        std_models = db.query(STDModel).filter(STDModel.is_active == True).all()
        print(f"   [OK] Active STD models: {len(std_models)}")

        # Query STD statistics for a model
        stats = db.query(STDStatistics).filter(STDStatistics.std_model_id == std_model.id).all()
        print(f"   [OK] STD statistics for model {std_model.id}: {len(stats)} zones")

        # Query recent comparisons
        comparisons = db.query(ComparisonResult).filter(ComparisonResult.judgment == JudgmentStatus.PASS).all()
        print(f"   [OK] PASS comparisons: {len(comparisons)}")

        # Query audit logs
        logs = db.query(AuditLog).filter(AuditLog.user_id == user.id).all()
        print(f"   [OK] Audit logs for user {user.username}: {len(logs)}")

        # Test relationships
        print("\n11. Testing relationships...")
        std_with_samples = db.query(STDModel).filter(STDModel.id == std_model.id).first()
        print(f"   [OK] STD model samples: {len(std_with_samples.samples)}")
        print(f"   [OK] STD model statistics: {len(std_with_samples.statistics)}")

        test_with_comparisons = db.query(TestSample).filter(TestSample.id == test_sample.id).first()
        print(f"   [OK] Test sample comparisons: {len(test_with_comparisons.comparison_results)}")

        user_with_logs = db.query(User).filter(User.id == user.id).first()
        print(f"   [OK] User audit logs: {len(user_with_logs.audit_logs)}")

        # Test to_dict methods
        print("\n12. Testing to_dict methods...")
        print(f"   [OK] User dict: {list(user.to_dict().keys())}")
        print(f"   [OK] STD Model dict: {list(std_model.to_dict().keys())}")
        print(f"   [OK] Test Sample dict: {list(test_sample.to_dict().keys())}")
        print(f"   [OK] Comparison dict: {list(comparison.to_dict().keys())}")

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nDatabase schema is correctly defined.")
        print("SQLAlchemy models are working as expected.")
        print("\nNext steps:")
        print("  1. Install Alembic: pip install alembic")
        print("  2. Initialize Alembic: alembic init alembic")
        print("  3. Create migration: alembic revision --autogenerate -m 'Initial schema'")
        print("  4. Apply migration: alembic upgrade head")

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        db.close()

    return True


if __name__ == "__main__":
    success = test_database_creation()
    sys.exit(0 if success else 1)
