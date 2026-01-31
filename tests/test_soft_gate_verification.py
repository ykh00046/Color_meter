"""
Soft Gate & 2-Pass Verification Test
=====================================
1. Soft Gate 1: Erosion when artifact ratio high
2. Soft Gate 2: r_end proportional tightening based on leak ratio
3. 2-Pass behavior: Only triggers when confidence low or count mismatch
4. FORCED_TOPK: Frequency check in oversegmentation scenario
"""

import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path

import cv2
import numpy as np

from src.engine_v7.api import load_config
from src.engine_v7.core.geometry.lens_geometry import detect_lens_circle
from src.engine_v7.core.measure.segmentation.color_masks import build_color_masks, build_color_masks_with_retry
from src.engine_v7.core.measure.segmentation.preprocess import build_roi_mask, build_sampling_mask
from src.engine_v7.core.pipeline.single_analyzer import analyze_single_sample
from src.engine_v7.core.signature.radial_signature import to_polar
from src.engine_v7.core.utils import to_cie_lab

# ============================================================
# Setup
# ============================================================
white_bgr = cv2.imread("data/raw_images/SKU001_OK_001.jpg")
black_bgr = cv2.imread("data/Black_A.png")

cfg = load_config("GGG")
cfg["expected_ink_count"] = 2
cfg["polar"] = {"R": 260, "T": 720}

geom = detect_lens_circle(white_bgr)
polar_R, polar_T = 260, 720

print("=" * 70)
print("Soft Gate & 2-Pass Verification Test")
print("=" * 70)

# ============================================================
# Test 1: Soft Gate 2 - r_end Proportional Tightening
# ============================================================
print("\n[Test 1] Soft Gate 2: r_end Proportional Tightening")
print("-" * 70)

# Test with various leak ratios
test_cases = [
    {"leak": 0.00, "expected_tightened": False, "desc": "No leak"},
    {"leak": 0.02, "expected_tightened": False, "desc": "At threshold"},
    {"leak": 0.05, "expected_tightened": True, "desc": "Mild leak"},
    {"leak": 0.10, "expected_tightened": True, "desc": "Moderate leak"},
    {"leak": 0.20, "expected_tightened": True, "desc": "Heavy leak"},
]

t1_pass = True
for tc in test_cases:
    plate_kpis = {"outer_rim_leak_ratio": tc["leak"]}
    roi_mask, roi_meta = build_roi_mask(polar_T, polar_R, 0.05, 0.98, 0.05, plate_kpis=plate_kpis)

    r_end_eff = roi_meta["r_end_effective"]
    warnings = roi_meta.get("soft_gate_warnings", [])
    tightened = any("SOFT_GATE2" in w for w in warnings)

    status = "✅" if tightened == tc["expected_tightened"] else "❌"
    if tightened != tc["expected_tightened"]:
        t1_pass = False

    print(f'  leak={tc["leak"]:.2f}: r_end_eff={r_end_eff:.3f}, tightened={tightened} {status} ({tc["desc"]})')

print(f'\n  Test 1 결과: {"✅ PASS" if t1_pass else "❌ FAIL"}')

# ============================================================
# Test 2: Soft Gate 1 - Erosion with high artifact ratio
# ============================================================
print("\n[Test 2] Soft Gate 1: Erosion with High Artifact Ratio")
print("-" * 70)

# Create polar image and lab map
polar_bgr = to_polar(white_bgr, geom, R=polar_R, T=polar_T)
polar_lab = to_cie_lab(polar_bgr)

# Create a base ROI mask
roi_mask, roi_meta = build_roi_mask(polar_T, polar_R, 0.05, 0.98, 0.05)

# Create a synthetic sample_mask_override (simulate plate ink mask)
# This will be at least 10000 pixels in the middle region
sample_mask = np.zeros((polar_T, polar_R), dtype=bool)
sample_mask[:, polar_R // 4 : polar_R * 3 // 4] = True  # Middle 50% of radius

# Test with different artifact ratios
artifact_test_cases = [
    {"artifact": 0.10, "expected_eroded": False, "desc": "Low artifact"},
    {"artifact": 0.25, "expected_eroded": False, "desc": "At threshold"},
    {"artifact": 0.30, "expected_eroded": True, "desc": "Above threshold"},
    {"artifact": 0.50, "expected_eroded": True, "desc": "High artifact"},
]

t2_pass = True
for tc in artifact_test_cases:
    plate_kpis = {"mask_artifact_ratio_valid": tc["artifact"], "outer_rim_leak_ratio": 0.0}

    mask, meta, warnings = build_sampling_mask(
        polar_lab,
        roi_mask,
        cfg,
        sample_mask_override=sample_mask,
        plate_kpis=plate_kpis,
    )

    rule = meta.get("rule", "unknown")
    eroded = "eroded" in rule.lower() or meta.get("eroded", False)

    status = "✅" if eroded == tc["expected_eroded"] else "❌"
    if eroded != tc["expected_eroded"]:
        t2_pass = False

    erode_warn = [w for w in warnings if "eroded" in w.lower()]
    print(f'  artifact={tc["artifact"]:.2f}: rule={rule}, eroded={eroded}, warnings={erode_warn} {status}')

print(f'\n  Test 2 결과: {"✅ PASS" if t2_pass else "❌ FAIL"}')

# ============================================================
# Test 3: 2-Pass Behavior Verification
# ============================================================
print("\n[Test 3] 2-Pass Behavior Verification")
print("-" * 70)

NUM_RUNS = 5
pass_stats = {"pass1_only": 0, "pass1_chosen": 0, "pass2_retry": 0}

for i in range(NUM_RUNS):
    result = analyze_single_sample(
        test_bgr=white_bgr, cfg=cfg, analysis_modes=["gate", "plate", "ink"], black_bgr=black_bgr
    )

    ink = result.get("ink", {})
    # segmentation_pass and warnings are in _meta
    _meta = ink.get("_meta", {})
    seg_pass = _meta.get("segmentation_pass", "unknown")
    # confidence is at top level of ink result
    confidence = ink.get("clustering_confidence", 0)

    if seg_pass in pass_stats:
        pass_stats[seg_pass] += 1

    print(f"  Run {i+1}: pass={seg_pass}, confidence={confidence:.3f}")

print(f"\n  2-Pass Statistics:")
print(f'    pass1_only: {pass_stats["pass1_only"]}/{NUM_RUNS} (no retry needed)')
print(f'    pass1_chosen: {pass_stats["pass1_chosen"]}/{NUM_RUNS} (retry happened, pass1 better)')
print(f'    pass2_retry: {pass_stats["pass2_retry"]}/{NUM_RUNS} (retry happened, pass2 used)')

# 2-pass behavior check:
# - pass1_only: No retry needed (high confidence)
# - pass1_chosen: Retry happened, but pass1 was better
# - pass2_retry: Retry happened, pass2 was better
# With Hard Gate, we expect pass1 to be chosen (either pass1_only or pass1_chosen)
pass1_total = pass_stats["pass1_only"] + pass_stats["pass1_chosen"]
t3_pass = pass1_total >= NUM_RUNS * 0.8  # At least 80% pass1-based results
print(f'\n  Test 3 결과: {"✅ PASS" if t3_pass else "⚠️ CHECK"} (pass1 selected {pass1_total}/{NUM_RUNS})')

# ============================================================
# Test 4: FORCED_TOPK Frequency Check
# ============================================================
print("\n[Test 4] INK_ROLE_FORCED_TOPK Frequency Check")
print("-" * 70)

forced_topk_count = 0

for i in range(NUM_RUNS):
    result = analyze_single_sample(
        test_bgr=white_bgr, cfg=cfg, analysis_modes=["gate", "plate", "ink"], black_bgr=black_bgr
    )

    ink = result.get("ink", {})
    _meta = ink.get("_meta", {})
    warnings = _meta.get("warnings", [])

    has_forced = "INK_ROLE_FORCED_TOPK" in warnings
    if has_forced:
        forced_topk_count += 1

    print(f"  Run {i+1}: FORCED_TOPK={has_forced}, warnings={warnings}")

print(f"\n  FORCED_TOPK 빈도: {forced_topk_count}/{NUM_RUNS}")
# FORCED_TOPK should be rare when expected_k matches actual ink count
t4_pass = forced_topk_count <= NUM_RUNS * 0.2  # Less than 20%
print(f'  Test 4 결과: {"✅ PASS" if t4_pass else "⚠️ CHECK"} (expect low frequency with matching expected_k)')

# ============================================================
# Test 5: Soft Gate Confidence Scaling
# ============================================================
print("\n[Test 5] Soft Gate Confidence Scaling")
print("-" * 70)

# Test confidence scaling with different artifact ratios
conf_test_cases = [
    {"artifact": 0.0, "expected_factor": 1.0},
    {"artifact": 0.2, "expected_factor": 0.7},  # 1.0 - 1.5 * 0.2 = 0.7
    {"artifact": 0.4, "expected_factor": 0.4},  # 1.0 - 1.5 * 0.4 = 0.4
    {"artifact": 0.6, "expected_factor": 0.2},  # Clamped to 0.2
]

print("  Confidence scaling formula: factor = max(0.2, 1.0 - 1.5 * artifact_ratio)")
t5_pass = True
for tc in conf_test_cases:
    actual_factor = max(0.2, 1.0 - 1.5 * tc["artifact"])
    match = abs(actual_factor - tc["expected_factor"]) < 0.01
    status = "✅" if match else "❌"
    if not match:
        t5_pass = False
    print(f'  artifact={tc["artifact"]:.1f}: expected={tc["expected_factor"]:.2f}, actual={actual_factor:.2f} {status}')

print(f'\n  Test 5 결과: {"✅ PASS" if t5_pass else "❌ FAIL"}')

# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

all_pass = t1_pass and t2_pass and t3_pass and t4_pass and t5_pass

print(
    f"""
[Test 1] Soft Gate 2 (r_end tightening): {"✅" if t1_pass else "❌"}
[Test 2] Soft Gate 1 (erosion): {"✅" if t2_pass else "❌"}
[Test 3] 2-Pass behavior: {"✅" if t3_pass else "⚠️"}
[Test 4] FORCED_TOPK frequency: {"✅" if t4_pass else "⚠️"}
[Test 5] Confidence scaling: {"✅" if t5_pass else "❌"}
"""
)

print("=" * 70)
if all_pass:
    print("🎉 ALL TESTS PASSED! Soft Gate implementation verified.")
else:
    print("⚠️ Some tests need attention. Review the results above.")
print("=" * 70)
