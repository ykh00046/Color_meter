import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, ".")

from src.engine_v7.api import load_config
from src.engine_v7.core.pipeline.single_analyzer import analyze_single_sample
from src.engine_v7.core.plate.plate_engine import analyze_plate_pair


def run_verification():
    print("=" * 70)
    print(" HARD GATE & ALIGNMENT VERIFICATION RUN")
    print("=" * 70)

    # 1. Load Images (White/Black Pair required for Plate Gate)
    white_path = Path("data/White_B.png")
    black_path = Path("data/Black_B.png")

    if not white_path.exists() or not black_path.exists():
        # Try fallback
        white_path = Path("data/samples/INK3/F1.png")
        black_path = Path("data/samples/INK3/F1_Black.png")  # Hypothetical
        if not white_path.exists():
            print("[FAIL] Test images not found.")
            return

    print(f"White Image: {white_path}")
    print(f"Black Image: {black_path}")

    img_white = cv2.imread(str(white_path))
    img_black = cv2.imread(str(black_path))

    # 2. Setup Config
    sku = "DEFAULT"
    cfg = load_config(sku)
    cfg["expected_ink_count"] = 3
    # Ensure Plate Gate thresholds allow passing
    if "v2_ink" not in cfg:
        cfg["v2_ink"] = {}
    cfg["v2_ink"]["plate_gate_min_samples"] = 500  # Lower threshold for test safety

    # 3. Run Analysis (3 times for consistency check)
    runs = 3
    results = []

    for i in range(runs):
        print(f"\n>>> RUN {i+1}/{runs}")

        # Explicitly requesting 'plate' and 'ink'
        res = analyze_single_sample(
            test_bgr=img_white, black_bgr=img_black, cfg=cfg, analysis_modes=["gate", "plate", "ink", "radial"]
        )

        results.append(res)

        # [Check A] Hard Gate Bypass
        ink_meta = res.get("ink", {}).get("_meta", {})
        sample_meta = ink_meta.get("sample_meta", {})
        rule = sample_meta.get("rule", "unknown")

        print(f"[Check A] Sampling Rule: {rule}")
        if "plate_gate_override" in rule:
            print("  [PASS] Heuristic Bypassed (Hard Gate Active)")
        else:
            print(f"  [FAIL] Hard Gate NOT active (Reason: {rule})")

        # [Check B] Coordinate Alignment
        # Calculate sums from Plate result and Ink input
        plate_masks = res.get("plate", {}).get("_masks", {})
        cart_core = plate_masks.get("ink_mask_core")

        if cart_core is not None:
            cart_sum = np.sum(cart_core)
            # Reconstruct polar sum from ink metadata (approx)
            polar_sum = sample_meta.get("n_pixels_used", 0)

            ratio = polar_sum / (cart_sum + 1e-6)
            print(f"[Check B] Alignment Stats:")
            print(f"  Cartesian Sum: {cart_sum}")
            print(f"  Polar Sum: {polar_sum}")
            print(f"  Ratio: {ratio:.2f}")

            if 0.5 < ratio < 1.5:
                print("  [PASS] Area preservation reasonable (considering warp distortion)")
            else:
                print("  [FAIL] Significant area mismatch (Check warp parameters)")
        else:
            print("  [FAIL] No ink_mask_core in plate results")

    # [Check C] Consistency
    print("\n" + "=" * 70)
    print(" CONSISTENCY SUMMARY")
    print("=" * 70)

    k_values = [r.get("ink", {}).get("k") for r in results]
    print(f"Ink Counts: {k_values}")

    if len(set(k_values)) == 1 and k_values[0] == 3:
        print("[PASS] Ink count consistent and matches expected.")
    else:
        print("[FAIL] Ink count unstable or mismatch.")

    # Check L* variance of first cluster
    l_values = [r.get("ink", {}).get("clusters", [])[0].get("centroid_lab_cie", [0])[0] for r in results]
    print(f"Cluster 0 L* values: {l_values}")
    l_range = max(l_values) - min(l_values)

    if l_range < 2.0:
        print(f"[PASS] L* very stable (Range: {l_range:.2f})")
    else:
        print(f"[FAIL] L* unstable (Range: {l_range:.2f})")


if __name__ == "__main__":
    run_verification()
