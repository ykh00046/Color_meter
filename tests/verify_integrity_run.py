import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to python path
sys.path.insert(0, ".")

from src.engine_v7.api import load_config

# Import Core Modules directly to bypass API wrappers for deep inspection
from src.engine_v7.core.pipeline.single_analyzer import analyze_single_sample


def run_verification():
    print("=" * 60)
    print(" INTEGRITY VERIFICATION RUN (Scale / Retry / Simulation)")
    print("=" * 60)

    # 1. Load Image
    img_path = Path("data/White_B.png")
    if not img_path.exists():
        # Fallback
        img_path = Path("data/samples/INK3/F1.png")
        if not img_path.exists():
            print("[FAIL] No test image found.")
            return

    print(f"Target Image: {img_path}")
    img_bgr = cv2.imread(str(img_path))

    # 2. Load Config & Force Retry Conditions
    # We want to force a condition where pass2 might be triggered or useful.
    # But analyze_single_sample uses default config.
    sku = "DEFAULT"
    v7_cfg = load_config(sku)

    # Force expected_ink_count
    expected_k = 3
    v7_cfg["expected_ink_count"] = expected_k

    # Enable retry explicitly
    if "v2_ink" not in v7_cfg:
        v7_cfg["v2_ink"] = {}
    v7_cfg["v2_ink"]["enable_retry"] = True
    v7_cfg["v2_ink"]["role_policy"] = "legacy"  # This will force switch to 'inkness' in pass2

    print(f"Config: expected_k={expected_k}, retry=True, role_policy=legacy")

    # 3. Run Analysis
    print("\nRunning Analysis...")
    try:
        # We need 'ink' and 'plate' (if available, but here single image)
        # Simulation requires 'ink' mode.
        result = analyze_single_sample(
            test_bgr=img_bgr, cfg=v7_cfg, analysis_modes=["gate", "ink", "radial"]  # Minimal modes
        )
    except Exception as e:
        print(f"[CRITICAL FAIL] Analysis crashed: {e}")
        import traceback

        traceback.print_exc()
        return

    # 4. Verification Steps
    ink_data = result.get("ink", {})
    sim_data = result.get("color_simulation", {})

    if not ink_data:
        print("[FAIL] No ink data returned.")
        return

    meta = ink_data.get("_meta", {})
    # Note: analyze_single_sample wraps metadata into result['ink'] directly or in _meta?
    # Actually analyze_single_sample returns _analyze_ink_segmentation result.
    # _analyze_ink_segmentation returns dict with k, clusters, confidence, _meta.
    # And _meta contains the raw metadata from build_color_masks.

    # Let's inspect clusters
    clusters = ink_data.get("clusters", [])

    print("\n--- [Check A] Scale & Metadata ---")
    scale_ok = True
    for idx, c in enumerate(clusters):
        lab_cv8 = c.get("centroid_lab_cv8") or c.get("centroid_lab")  # Legacy key
        lab_cie = c.get("centroid_lab_cie")

        print(f"Cluster {idx}:")
        print(f"  CV8: {lab_cv8}")
        print(f"  CIE: {lab_cie}")

        # Check L range
        if not (0 <= lab_cie[0] <= 100):
            print(f"  [FAIL] CIE L value {lab_cie[0]} out of range [0, 100]")
            scale_ok = False
        if not (0 <= lab_cv8[0] <= 255):
            print(f"  [FAIL] CV8 L value {lab_cv8[0]} out of range [0, 255]")
            scale_ok = False

    if scale_ok:
        print("[PASS] Lab scales are correct.")

    print("\n--- [Check B] 2-Pass Retry Logic ---")
    # Check if pass 2 was used
    pass_info = meta.get("segmentation_pass", "unknown")
    k_used = meta.get("k_used")
    warnings = meta.get("warnings", [])

    print(f"Segmentation Pass: {pass_info}")
    print(f"K Used: {k_used} (Expected: {expected_k})")
    print(f"Warnings: {warnings}")

    # Logic check:
    # If pass2_retry, k_used should be expected_k + 1 (technically segmentation_k)
    # But result['k'] usually reports the finalized count.
    # Let's check segmentation_k in meta.
    seg_k = meta.get("segmentation_k")
    print(f"Segmentation K Param: {seg_k}")

    detected_count = meta.get("detected_ink_like_count")
    print(f"Detected Ink Count: {detected_count}")

    if pass_info == "pass2_retry":
        if seg_k == expected_k + 1:
            print("[PASS] Pass 2 ran with K+1.")
        else:
            print(f"[FAIL] Pass 2 ran but K={seg_k} (expected {expected_k+1})")

        if "INK_ROLE_FORCED_TOPK" in warnings:
            print("[PASS] Top-K Forcing activated.")
            if detected_count == expected_k:
                print("[PASS] Final ink count matches expected count.")
            else:
                print(f"[FAIL] Count mismatch despite Top-K (Got {detected_count})")
        else:
            print("[INFO] Top-K Forcing not triggered (maybe count matched naturally).")
    else:
        print("[INFO] Pass 1 was sufficient. Retry logic check skipped for this sample.")

    print("\n--- [Check C] Simulation Integrity ---")
    sims = sim_data.get("simulations", [])
    if not sims:
        print("[FAIL] No simulation data generated.")
    else:
        roles_ok = True
        for s in sims:
            role = s.get("role")
            print(f"Simulated Ink ID {s.get('ink_id')}: Role={role}")
            if role not in ["ink", "primary"]:
                print(f"  [FAIL] Invalid role '{role}' in simulation!")
                roles_ok = False

        if roles_ok:
            print("[PASS] Only ink/primary roles included in simulation.")

    print("\n" + "=" * 60)
    print(" VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_verification()
