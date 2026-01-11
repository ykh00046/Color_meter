#!/usr/bin/env python3
"""
Backward Compatibility Test

Ensures that:
1. Existing aggregate STD models can still be registered
2. Aggregate models load correctly with load_std_models_auto()
3. Inspection works with aggregate models
4. Index.json without color_mode field defaults to aggregate
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from core.model_registry import get_color_mode, load_std_models_auto
from core.pipeline.analyzer import evaluate


def create_simple_test_image(output_path: str):
    """Create a simple grayscale lens image."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cy, cx = 200, 200

    # Simple gray circle
    cv2.circle(img, (cx, cy), 180, (100, 100, 100), -1)

    cv2.imwrite(output_path, img)
    print(f"Created: {output_path}")


def test_backward_compat():
    print("=" * 70)
    print("Backward Compatibility Test")
    print("=" * 70)

    # Setup
    temp_dir = Path(tempfile.mkdtemp(prefix="backward_compat_"))
    models_root = temp_dir / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    print(f"\nTemp directory: {temp_dir}")

    try:
        # Load config
        cfg_path = Path("configs/default.json")
        if not cfg_path.exists():
            print(f"[FAIL] Config not found at {cfg_path}")
            return False

        with open(cfg_path) as f:
            cfg = json.load(f)

        sku = "COMPAT_SKU"
        ink = "INK_DEFAULT"

        # Step 1: Create synthetic STD images
        print("\n" + "=" * 70)
        print("Step 1: Create Synthetic STD Images (Aggregate Mode)")
        print("=" * 70)

        std_images = []
        for mode in ["LOW", "MID", "HIGH"]:
            img_path = temp_dir / f"std_{mode.lower()}.png"
            create_simple_test_image(str(img_path))
            std_images.append((mode, str(img_path)))

        # Step 2: Register using aggregate mode (traditional)
        print("\n" + "=" * 70)
        print("Step 2: Register Aggregate STDs (Traditional Mode)")
        print("=" * 70)

        from scripts.register_std import main as register_main

        for mode, img_path in std_images:
            print(f"\nRegistering {mode} mode (aggregate)...")

            sys.argv = [
                "register_std.py",
                "--sku",
                sku,
                "--ink",
                ink,
                "--mode",
                mode,
                "--stds",
                img_path,
                "--cfg",
                str(cfg_path),
                "--models_root",
                str(models_root),
                "--color_mode",
                "aggregate",  # Explicitly use aggregate mode
                "--created_by",
                "compat_test",
                "--notes",
                f"Compatibility test {mode}",
            ]

            try:
                register_main()
                print(f"[OK] {mode} registration completed")
            except SystemExit as e:
                if e.code == 0:
                    print(f"[OK] {mode} registration completed")
                else:
                    print(f"[FAIL] {mode} registration failed with code {e.code}")
                    return False

        # Step 3: Verify index.json structure
        print("\n" + "=" * 70)
        print("Step 3: Verify index.json (Aggregate Mode)")
        print("=" * 70)

        index_path = models_root / "index.json"
        if not index_path.exists():
            print(f"[FAIL] index.json not found")
            return False

        with open(index_path) as f:
            index_data = json.load(f)

        print(f"[OK] index.json exists")

        # Find entry
        entry = None
        for item in index_data.get("items", []):
            if item.get("sku") == sku and item.get("ink") == ink:
                entry = item
                break

        if not entry:
            print(f"[FAIL] Entry not found for {sku}/{ink}")
            return False

        print(f"[OK] Entry found: {sku}/{ink}")
        print(f"  color_mode: {entry.get('color_mode', 'NOT SET')}")
        print(f"  status: {entry.get('status')}")
        print(f"  active modes: {list(entry.get('active', {}).keys())}")

        # Verify aggregate structure
        if entry.get("color_mode") != "aggregate":
            print(f"[FAIL] Expected color_mode='aggregate', got '{entry.get('color_mode')}'")
            return False

        if entry.get("status") != "ACTIVE":
            print(f"[FAIL] Expected status='ACTIVE', got '{entry.get('status')}'")
            return False

        # Step 4: Test color_mode detection
        print("\n" + "=" * 70)
        print("Step 4: Test Color Mode Detection")
        print("=" * 70)

        detected_mode = get_color_mode(str(models_root), sku, ink)
        print(f"Detected color_mode: {detected_mode}")

        if detected_mode != "aggregate":
            print(f"[FAIL] Expected 'aggregate', got '{detected_mode}'")
            return False

        print(f"[OK] Color mode correctly detected as aggregate")

        # Step 5: Load models using auto-loader
        print("\n" + "=" * 70)
        print("Step 5: Load Models with load_std_models_auto()")
        print("=" * 70)

        models, color_mode, color_metadata, reasons = load_std_models_auto(str(models_root), sku, ink)

        if reasons:
            print(f"[FAIL] Failed to load models: {reasons}")
            return False

        print(f"[OK] Models loaded successfully")
        print(f"  color_mode: {color_mode}")
        print(f"  modes: {list(models.keys())}")
        print(f"  color_metadata: {color_metadata}")

        if color_mode != "aggregate":
            print(f"[FAIL] Expected color_mode='aggregate', got '{color_mode}'")
            return False

        if set(models.keys()) != {"LOW", "MID", "HIGH"}:
            print(f"[FAIL] Expected LOW/MID/HIGH modes, got {list(models.keys())}")
            return False

        if color_metadata is not None:
            print(f"[FAIL] Expected color_metadata=None for aggregate mode, got {color_metadata}")
            return False

        # Step 6: Run inspection with aggregate models
        print("\n" + "=" * 70)
        print("Step 6: Run Inspection (Aggregate Mode)")
        print("=" * 70)

        test_img_path = temp_dir / "test.png"
        create_simple_test_image(str(test_img_path))

        test_bgr = cv2.imread(str(test_img_path))
        if test_bgr is None:
            print(f"[FAIL] Failed to read test image")
            return False

        print(f"[OK] Test image loaded: {test_bgr.shape}")

        # Run traditional evaluation
        try:
            decision = evaluate(
                test_bgr,
                models,
                cfg,
                mode="all",
            )

            print(f"\n[OK] Inspection completed")
            print(f"  Label: {decision.label}")
            print(f"  Reasons: {decision.reasons}")
            print(f"  Best mode: {decision.best_mode}")
            print(f"  Gate passed: {decision.gate.passed if decision.gate else 'N/A'}")

            if decision.signature:
                print(
                    f"  Signature: score_corr={decision.signature.score_corr:.3f}, "
                    + f"deltaE_mean={decision.signature.delta_e_mean:.2f}"
                )

            # Accept OK or NG_COLOR as valid results
            if decision.label not in ["OK", "NG_COLOR", "NG_GATE", "NG_ANOMALY"]:
                print(f"\n[WARN] Unexpected label: {decision.label}")

            print(f"\n[OK][OK][OK] Backward compatibility test PASSED!")
            return True

        except Exception as e:
            print(f"\n[FAIL] Inspection failed with error: {e}")
            import traceback

            traceback.print_exc()
            return False

    finally:
        # Cleanup
        print(f"\n" + "=" * 70)
        print("Cleanup")
        print("=" * 70)
        print(f"Removing temp directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_legacy_index():
    """Test that index.json without color_mode field defaults to aggregate."""
    print("\n" + "=" * 70)
    print("Legacy Index Test (No color_mode field)")
    print("=" * 70)

    temp_dir = Path(tempfile.mkdtemp(prefix="legacy_index_"))
    models_root = temp_dir / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    try:
        # Create a legacy index.json without color_mode field
        legacy_index = {
            "schema_version": "std_registry_index.v1",
            "engine_version": "v7",
            "items": [
                {
                    "sku": "LEGACY_SKU",
                    "ink": "INK_DEFAULT",
                    "active": {
                        "LOW": "LEGACY_SKU/INK_DEFAULT/LOW/v20250101_120000",
                        "MID": "LEGACY_SKU/INK_DEFAULT/MID/v20250101_120000",
                        "HIGH": "LEGACY_SKU/INK_DEFAULT/HIGH/v20250101_120000",
                    },
                    "status": "ACTIVE",
                    # No color_mode field (legacy)
                }
            ],
        }

        index_path = models_root / "index.json"
        with open(index_path, "w") as f:
            json.dump(legacy_index, f, indent=2)

        print(f"Created legacy index.json (no color_mode field)")

        # Test color_mode detection
        detected_mode = get_color_mode(str(models_root), "LEGACY_SKU", "INK_DEFAULT")
        print(f"Detected color_mode: {detected_mode}")

        if detected_mode != "aggregate":
            print(f"[FAIL] Expected 'aggregate' for legacy index, got '{detected_mode}'")
            return False

        print(f"[OK] Legacy index correctly defaults to aggregate mode")
        return True

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 70)
    print("BACKWARD COMPATIBILITY TEST SUITE")
    print("=" * 70)

    success1 = test_backward_compat()
    success2 = test_legacy_index()

    if success1 and success2:
        print("\n" + "=" * 70)
        print("*** ALL BACKWARD COMPATIBILITY TESTS PASSED! ***")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("*** BACKWARD COMPATIBILITY TESTS FAILED ***")
        print("=" * 70)
        sys.exit(1)
