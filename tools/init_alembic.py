#!/usr/bin/env python3
"""
Initialize Alembic for Database Migrations

Step-by-step guide to set up Alembic migrations.
"""

import subprocess
import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("ALEMBIC INITIALIZATION GUIDE")
    print("=" * 60)

    project_root = Path(__file__).parent.parent

    print("\nThis script will guide you through Alembic setup.")
    print("\nSteps:")
    print("  1. Install Alembic")
    print("  2. Initialize Alembic")
    print("  3. Create initial migration")
    print("  4. Apply migration")

    # Step 1: Check if Alembic is installed
    print("\n" + "=" * 60)
    print("STEP 1: Check Alembic Installation")
    print("=" * 60)

    try:
        import alembic

        print(f"[OK] Alembic is already installed (version: {alembic.__version__})")
    except ImportError:
        print("[INFO] Alembic is not installed.")
        print("\nTo install:")
        print("  pip install alembic>=1.13.0")
        print("\nOr install all requirements:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    # Step 2: Initialize Alembic
    print("\n" + "=" * 60)
    print("STEP 2: Initialize Alembic")
    print("=" * 60)

    alembic_dir = project_root / "alembic"
    if alembic_dir.exists():
        print(f"[INFO] Alembic directory already exists: {alembic_dir}")
        response = input("Reinitialize? (y/n): ")
        if response.lower() != "y":
            print("[SKIP] Alembic initialization skipped")
        else:
            print("\nRemoving existing alembic directory...")
            import shutil

            shutil.rmtree(alembic_dir)
            print("[OK] Removed existing directory")
    else:
        response = "y"

    if response.lower() == "y":
        print("\nInitializing Alembic...")
        print("Command: alembic init alembic")

        result = subprocess.run(["alembic", "init", "alembic"], cwd=project_root, capture_output=True, text=True)

        if result.returncode == 0:
            print("[OK] Alembic initialized successfully")
            print("\nCreated:")
            print("  - alembic/")
            print("  - alembic.ini")
        else:
            print(f"[FAIL] Alembic initialization failed")
            print(result.stderr)
            sys.exit(1)

    # Step 3: Configure alembic.ini
    print("\n" + "=" * 60)
    print("STEP 3: Configure alembic.ini")
    print("=" * 60)

    alembic_ini = project_root / "alembic.ini"
    if alembic_ini.exists():
        print(f"[INFO] Found alembic.ini: {alembic_ini}")
        print("\nYou need to edit alembic.ini:")
        print("  1. Find line: sqlalchemy.url = driver://user:pass@localhost/dbname")
        print("  2. Replace with: sqlalchemy.url = sqlite:///./color_meter.db")
        print("  3. (Or use PostgreSQL URL if using Postgres)")
        print("\nExample:")
        print("  # SQLite (default)")
        print("  sqlalchemy.url = sqlite:///./color_meter.db")
        print("")
        print("  # PostgreSQL (production)")
        print("  sqlalchemy.url = postgresql://user:password@localhost/color_meter")

        print("\n[ACTION REQUIRED] Edit alembic.ini now")
        input("Press Enter when done...")

    # Step 4: Configure env.py
    print("\n" + "=" * 60)
    print("STEP 4: Configure alembic/env.py")
    print("=" * 60)

    env_py = project_root / "alembic" / "env.py"
    if env_py.exists():
        print(f"[INFO] Found env.py: {env_py}")
        print("\nYou need to edit alembic/env.py:")
        print("  1. Find line: target_metadata = None")
        print("  2. Replace with:")
        print("     ```python")
        print("     import sys")
        print("     from pathlib import Path")
        print("     ")
        print("     # Add project root to path")
        print("     project_root = Path(__file__).parent.parent")
        print("     sys.path.insert(0, str(project_root))")
        print("     ")
        print("     # Import Base from models")
        print("     from src.models.database import Base")
        print("     ")
        print("     target_metadata = Base.metadata")
        print("     ```")

        print("\n[ACTION REQUIRED] Edit alembic/env.py now")
        input("Press Enter when done...")

    # Step 5: Create initial migration
    print("\n" + "=" * 60)
    print("STEP 5: Create Initial Migration")
    print("=" * 60)

    print("\nCreating initial migration...")
    print("Command: alembic revision --autogenerate -m 'Initial schema'")

    response = input("\nRun now? (y/n): ")
    if response.lower() == "y":
        result = subprocess.run(
            ["alembic", "revision", "--autogenerate", "-m", "Initial schema"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("[OK] Migration created successfully")
            print(result.stdout)
        else:
            print(f"[FAIL] Migration creation failed")
            print(result.stderr)
            sys.exit(1)
    else:
        print("[SKIP] Migration creation skipped")
        print("\nTo create manually:")
        print("  alembic revision --autogenerate -m 'Initial schema'")

    # Step 6: Apply migration
    print("\n" + "=" * 60)
    print("STEP 6: Apply Migration")
    print("=" * 60)

    print("\nApplying migration to database...")
    print("Command: alembic upgrade head")

    response = input("\nRun now? (y/n): ")
    if response.lower() == "y":
        result = subprocess.run(["alembic", "upgrade", "head"], cwd=project_root, capture_output=True, text=True)

        if result.returncode == 0:
            print("[OK] Migration applied successfully")
            print(result.stdout)
        else:
            print(f"[FAIL] Migration failed")
            print(result.stderr)
            sys.exit(1)
    else:
        print("[SKIP] Migration application skipped")
        print("\nTo apply manually:")
        print("  alembic upgrade head")

    # Summary
    print("\n" + "=" * 60)
    print("ALEMBIC SETUP COMPLETE")
    print("=" * 60)

    print("\nNext steps:")
    print("  1. Verify database was created: color_meter.db")
    print("  2. Check tables: sqlite3 color_meter.db '.tables'")
    print("  3. Start implementing STDService (Week 2)")

    print("\nUseful Alembic commands:")
    print("  alembic current              - Show current revision")
    print("  alembic history              - Show migration history")
    print("  alembic upgrade head         - Apply all migrations")
    print("  alembic downgrade -1         - Rollback one migration")
    print("  alembic revision --autogenerate -m 'msg'  - Create new migration")


if __name__ == "__main__":
    main()
