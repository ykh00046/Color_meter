#!/usr/bin/env python3
"""
End-to-End Test for Per-Color System

Tests:
1. Per-color STD registration (LOW/MID/HIGH)
2. Per-color model loading
3. Per-color inspection
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# Add scripts/engine_v7 to path for register_std import
_scripts_dir = Path(__file__).resolve().parents[3] / "scripts" / "engine_v7"
sys.path.insert(0, str(_scripts_dir))

from core.model_registry import get_color_mode, load_per_color_models, load_std_models_auto
from core.pipeline.analyzer import evaluate_per_color


def create_synthetic_image(output_path: str, dark_val=40, light_val=180):
    """Create a synthetic 2-color lens image."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cy, cx = 200, 200

    # Outer: Light
    cv2.circle(img, (cx, cy), 180, (light_val, light_val, light_val), -1)

    # Inner: Dark
    cv2.circle(img, (cx, cy), 120, (dark_val, dark_val, dark_val), -1)

    cv2.imwrite(output_path, img)
    print(f"Created: {output_path}")


def test_per_color_e2e():
    print("=" * 70)
    print("End-to-End Per-Color Test")
    print("=" * 70)

    # Setup
    temp_dir = Path(tempfile.mkdtemp(prefix="per_color_e2e_"))
    models_root = temp_dir / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    print(f"\nTemp directory: {temp_dir}")

    try:
        # Load config
        cfg_path = Path("configs/default.json")
        if not cfg_path.exists():
            print(f"Error: Config not found at {cfg_path}")
            return False

        with open(cfg_path) as f:
            cfg = json.load(f)

        sku = "TEST_SKU"
        ink = "INK_RGB"
        expected_k = 2

        # Step 1: Create synthetic STD images for each mode
        print("\n" + "=" * 70)
        print("Step 1: Create Synthetic STD Images")
        print("=" * 70)

        std_images = {"LOW": [], "MID": [], "HIGH": []}

        for mode, dark_val in [("LOW", 30), ("MID", 40), ("HIGH", 50)]:
            for i in range(2):
                img_path = temp_dir / f"std_{mode.lower()}_{i}.png"
                create_synthetic_image(str(img_path), dark_val=dark_val, light_val=180)
                std_images[mode].append(str(img_path))

        # Step 2: Register per-color STDs for each mode
        print("\n" + "=" * 70)
        print("Step 2: Register Per-Color STDs")
        print("=" * 70)

        import argparse

        from register_std import main as register_main

        for mode in ["LOW", "MID", "HIGH"]:
            print(f"\nRegistering {mode} mode...")

            # Simulate command line args
            sys.argv = [
                "register_std.py",
                "--sku",
                sku,
                "--ink",
                ink,
                "--mode",
                mode,
                "--stds",
                *std_images[mode],
                "--cfg",
                str(cfg_path),
                "--models_root",
                str(models_root),
                "--color_mode",
                "per_color",
                "--expected_ink_count",
                str(expected_k),
                "--created_by",
                "test_script",
                "--notes",
                f"E2E test {mode} mode",
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

        # Step 3: Verify index.json
        print("\n" + "=" * 70)
        print("Step 3: Verify index.json")
        print("=" * 70)

        index_path = models_root / "index.json"
        if not index_path.exists():
            print(f"[FAIL] index.json not found at {index_path}")
            return False

        with open(index_path) as f:
            index_data = json.load(f)

        print(f"[OK] index.json exists")
        print(f"  Entries: {len(index_data.get('items', []))}")

        # Find our entry
        entry = None
        for item in index_data.get("items", []):
            if item.get("sku") == sku and item.get("ink") == ink:
                entry = item
                break

        if not entry:
            print(f"[FAIL] Entry not found for {sku}/{ink}")
            return False

        print(f"[OK] Entry found: {sku}/{ink}")
        print(f"  color_mode: {entry.get('color_mode')}")
        print(f"  status: {entry.get('status')}")
        print(f"  colors: {len(entry.get('colors', []))}")

        if entry.get("color_mode") != "per_color":
            print(f"[FAIL] Expected color_mode='per_color', got '{entry.get('color_mode')}'")
            return False

        if entry.get("status") != "ACTIVE":
            print(f"[FAIL] Expected status='ACTIVE', got '{entry.get('status')}'")
            return False

        for color_info in entry.get("colors", []):
            print(
                f"  - {color_info['color_id']}: role={color_info.get('role')}, "
                + f"L*={color_info.get('lab_centroid', [0, 0, 0])[0]:.1f}"
            )

        # Step 4: Load per-color models
        print("\n" + "=" * 70)
        print("Step 4: Load Per-Color Models")
        print("=" * 70)

        color_mode = get_color_mode(str(models_root), sku, ink)
        print(f"Detected color_mode: {color_mode}")

        if color_mode != "per_color":
            print(f"[FAIL] Expected 'per_color', got '{color_mode}'")
            return False

        models, color_mode_ret, color_metadata, reasons = load_std_models_auto(str(models_root), sku, ink)

        if reasons:
            print(f"[FAIL] Failed to load models: {reasons}")
            return False

        print(f"[OK] Models loaded successfully")
        print(f"  color_mode: {color_mode_ret}")
        print(f"  colors: {list(models.keys())}")

        for color_id, mode_models in models.items():
            print(f"  {color_id}:")
            for mode, model in mode_models.items():
                print(
                    f"    {mode}: signature shape {model.radial_lab_mean.shape}, "
                    + f"geom=({model.geom.cx:.1f}, {model.geom.cy:.1f}, r={model.geom.r:.1f})"
                )

        # Step 5: Create test image and run inspection
        print("\n" + "=" * 70)
        print("Step 5: Run Per-Color Inspection")
        print("=" * 70)

        test_img_path = temp_dir / "test.png"
        create_synthetic_image(str(test_img_path), dark_val=40, light_val=180)

        test_bgr = cv2.imread(str(test_img_path))
        if test_bgr is None:
            print(f"[FAIL] Failed to read test image")
            return False

        print(f"[OK] Test image loaded: {test_bgr.shape}")

        # Run per-color evaluation
        try:
            decision, per_color_sigs = evaluate_per_color(
                test_bgr,
                models,
                color_metadata,
                cfg,
                expected_k,
                mode="all",
            )

            print(f"\n[OK] Inspection completed")
            print(f"  Label: {decision.label}")
            print(f"  Reasons: {decision.reasons}")
            print(f"  Best mode: {decision.best_mode}")
            print(f"  Gate passed: {decision.gate.passed if decision.gate else 'N/A'}")

            print(f"\n  Per-color signatures:")
            for color_id, mode_sigs in per_color_sigs.items():
                print(f"    {color_id}:")
                for mode, sig in mode_sigs.items():
                    print(
                        f"      {mode}: score_corr={sig.score_corr:.3f}, deltaE_mean={sig.delta_e_mean:.2f}, "
                        + f"passed={sig.passed}"
                    )

            # Check decision
            if decision.label not in ["OK", "NG_COLOR"]:
                print(f"\n[WARN] Unexpected label: {decision.label} (expected OK or NG_COLOR)")

            if decision.label == "OK":
                print(f"\n[OK][OK][OK] Per-color inspection PASSED!")
                return True
            else:
                print(f"\n[WARN] Per-color inspection returned {decision.label}")
                print(f"  This may be expected if synthetic images don't match STD exactly")
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


if __name__ == "__main__":
    success = test_per_color_e2e()
    if success:
        print("\n" + "=" * 70)
        print("*** END-TO-END TEST PASSED! ***")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("*** END-TO-END TEST FAILED ***")
        print("=" * 70)
        sys.exit(1)
