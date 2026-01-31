"""
Plate-Segmentation Gate Verification Checklist
- A. Hard Gate Bypass Verification
- B. Coordinate Alignment (Cart->Polar) Verification
- C. Extraction Stability Verification (3 Runs)
"""

import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path

import cv2
import numpy as np

from src.engine_v7.api import load_config
from src.engine_v7.core.geometry.lens_geometry import detect_lens_circle
from src.engine_v7.core.pipeline.single_analyzer import analyze_single_sample
from src.engine_v7.core.plate.plate_engine import analyze_plate_pair
from src.engine_v7.core.signature.radial_signature import to_polar

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

print("=" * 60)
print("Plate-Segmentation Gate 검증 체크리스트")
print("=" * 60)

# ============================================================
# A. Hard Gate Bypass 검증
# ============================================================
print("\n[A] Hard Gate Bypass 검증")
print("-" * 40)

# Run plate analysis
plate_cfg = cfg.get("plate", {})
plate_result = analyze_plate_pair(
    white_bgr=white_bgr,
    black_bgr=black_bgr,
    cfg=plate_cfg,
    geom_hint=geom,
)

plate_masks = plate_result.get("_masks", {})
plate_ink_mask_polar = plate_masks.get("ink_mask_core_polar")

# Check A.1: sample_mask_override exists
a1_pass = plate_ink_mask_polar is not None
print(f'  A.1 sample_mask_override 존재: {"✅ PASS" if a1_pass else "❌ FAIL"}')
if a1_pass:
    print(f"      Shape: {plate_ink_mask_polar.shape}, Dtype: {plate_ink_mask_polar.dtype}")

# Check A.2: override_valid_px > threshold
override_sum = int(np.sum(plate_ink_mask_polar)) if a1_pass else 0
threshold = 1000
a2_pass = override_sum >= threshold
print(f'  A.2 override_valid_px >= {threshold}: {"✅ PASS" if a2_pass else "❌ FAIL"}')
print(f"      Sum: {override_sum} pixels")

# Run full analysis to check A.3
results = analyze_single_sample(
    test_bgr=white_bgr, cfg=cfg, analysis_modes=["gate", "plate", "ink"], black_bgr=black_bgr
)

ink = results.get("ink", {})
sample_meta = ink.get("sample_meta", {})

# Check A.3: Heuristic bypassed
sampling_rule = sample_meta.get("rule", "unknown")
a3_pass = sampling_rule == "plate_gate_override"
print(f'  A.3 Heuristic bypass (rule=plate_gate_override): {"✅ PASS" if a3_pass else "❌ FAIL"}')
print(f"      Sampling Rule: {sampling_rule}")
print(f'      plate_gate_applied: {sample_meta.get("plate_gate_applied", False)}')
print(f'      n_pixels_used: {sample_meta.get("n_pixels_used", 0)}')

# ============================================================
# B. 좌표 정합 (Cart → Polar) 검증
# ============================================================
print("\n[B] 좌표 정합 검증")
print("-" * 40)

# Get Cartesian mask from plate
cart_mask = plate_masks.get("ink_mask_core")  # Cartesian 512x512
cart_sum = int(np.sum(cart_mask)) if cart_mask is not None else 0
polar_sum = override_sum

# Check B.1: Cart vs Polar sum ratio
if cart_sum > 0:
    ratio = polar_sum / cart_sum * 100
    b1_pass = 50 <= ratio <= 150  # Allow 50-150% range
    print(f'  B.1 Cart/Polar sum 비율 (50-150% 범위): {"✅ PASS" if b1_pass else "❌ FAIL"}')
    print(f"      Cart sum: {cart_sum}, Polar sum: {polar_sum}")
    print(f"      Ratio: {ratio:.1f}%")
else:
    print(f"  B.1 Cart/Polar sum 비율: ⚠️ SKIP (cart_mask 없음)")
    b1_pass = False

# Check B.2: Polar mask r distribution (should be concentrated in dot region)
if a1_pass:
    # r distribution: check where the mask pixels are located
    r_indices = np.where(plate_ink_mask_polar)[1]  # shape is (T, R), so axis 1 is R
    if len(r_indices) > 0:
        r_mean = np.mean(r_indices) / polar_R  # Normalize to 0-1
        r_std = np.std(r_indices) / polar_R
        r_min = np.min(r_indices) / polar_R
        r_max = np.max(r_indices) / polar_R

        # Dot region should be roughly 0.3-0.7 range (not at outer edge)
        b2_pass = r_max < 0.85  # Should not extend to outer rim
        print(f'  B.2 Polar mask r 분포 (outer rim 배제): {"✅ PASS" if b2_pass else "❌ FAIL"}')
        print(f"      r_mean: {r_mean:.3f}, r_std: {r_std:.3f}")
        print(f"      r_range: [{r_min:.3f}, {r_max:.3f}]")
    else:
        print(f"  B.2 Polar mask r 분포: ❌ FAIL (mask empty)")
        b2_pass = False
else:
    b2_pass = False

# Check B.3: Create debug overlay image
debug_dir = Path("debug_output")
debug_dir.mkdir(exist_ok=True)

# Create polar image for overlay
polar_bgr = to_polar(white_bgr, geom, R=polar_R, T=polar_T)

# Create overlay: polar image with mask highlighted
if a1_pass:
    overlay = polar_bgr.copy()
    # Highlight mask in red
    overlay[plate_ink_mask_polar, 2] = 255  # Red channel where mask is True
    overlay[plate_ink_mask_polar, 0] = 0  # Blue channel
    overlay[plate_ink_mask_polar, 1] = 0  # Green channel
    # Blend
    blended = cv2.addWeighted(polar_bgr, 0.6, overlay, 0.4, 0)

    debug_path = debug_dir / "debug_sampling_mask_overlay.png"
    cv2.imwrite(str(debug_path), blended)
    print(f"  B.3 Debug overlay 저장: ✅ {debug_path}")
else:
    print(f"  B.3 Debug overlay: ⚠️ SKIP (mask 없음)")

# ============================================================
# C. 추출 결과 안정성 검증 (3회 Run)
# ============================================================
print("\n[C] 추출 결과 안정성 검증 (3회 Run)")
print("-" * 40)

k_results = []
L_centroids_all = []
gray_white_counts = []

for run_idx in range(3):
    run_result = analyze_single_sample(
        test_bgr=white_bgr, cfg=cfg, analysis_modes=["gate", "plate", "ink"], black_bgr=black_bgr
    )

    run_ink = run_result.get("ink", {})
    run_k = run_ink.get("k", 0)
    run_clusters = run_ink.get("clusters", [])

    k_results.append(run_k)

    # Extract L* centroids
    L_centroids = [c.get("centroid_lab_cie", [0])[0] for c in run_clusters]
    L_centroids_all.append(sorted(L_centroids))

    # Check for fake gray/white clusters (L* > 70 with role='ink')
    ink_clusters = [c for c in run_clusters if c.get("role") == "ink"]
    gray_white_count = sum(1 for c in ink_clusters if c.get("centroid_lab_cie", [0])[0] > 70)
    gray_white_counts.append(gray_white_count)

    print(f"  Run {run_idx+1}: k={run_k}, L*={[round(v, 1) for v in L_centroids]}, gray/white_ink={gray_white_count}")

# Check C.1: expected_k maintained
expected_k = cfg["expected_ink_count"]
c1_pass = all(k == expected_k for k in k_results)
print(f'\n  C.1 expected_k 유지 (3회 연속): {"✅ PASS" if c1_pass else "❌ FAIL"}')
print(f"      Results: {k_results}, Expected: {expected_k}")

# Check C.2: No fake gray/white clusters
c2_pass = all(gwc == 0 for gwc in gray_white_counts)
print(f'  C.2 가짜 gray/white 클러스터 없음: {"✅ PASS" if c2_pass else "❌ FAIL"}')
print(f"      gray/white counts per run: {gray_white_counts}")

# Check C.3: L* centroid stability
if len(L_centroids_all) >= 2 and all(len(lc) > 0 for lc in L_centroids_all):
    # Check max L* jump between runs
    max_L_jump = 0
    for i in range(len(L_centroids_all) - 1):
        for j in range(min(len(L_centroids_all[i]), len(L_centroids_all[i + 1]))):
            jump = abs(L_centroids_all[i][j] - L_centroids_all[i + 1][j])
            max_L_jump = max(max_L_jump, jump)

    c3_pass = max_L_jump < 15  # L* jump should be < 15
    print(f'  C.3 L* centroid 안정성 (jump < 15): {"✅ PASS" if c3_pass else "❌ FAIL"}')
    print(f"      Max L* jump: {max_L_jump:.2f}")
else:
    c3_pass = False
    print(f"  C.3 L* centroid 안정성: ⚠️ SKIP")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("검증 결과 요약")
print("=" * 60)

all_pass = a1_pass and a2_pass and a3_pass and b1_pass and b2_pass and c1_pass and c2_pass and c3_pass

print(f"\n[A] Hard Gate Bypass:")
print(f'    A.1 sample_mask_override 존재: {"✅" if a1_pass else "❌"}')
print(f'    A.2 override_valid_px >= {threshold}: {"✅" if a2_pass else "❌"}')
print(f'    A.3 Heuristic bypass: {"✅" if a3_pass else "❌"}')

print(f"\n[B] 좌표 정합:")
print(f'    B.1 Cart/Polar sum 비율: {"✅" if b1_pass else "❌"}')
print(f'    B.2 Polar mask r 분포: {"✅" if b2_pass else "❌"}')

print(f"\n[C] 추출 안정성:")
print(f'    C.1 expected_k 유지: {"✅" if c1_pass else "❌"}')
print(f'    C.2 gray/white 클러스터 없음: {"✅" if c2_pass else "❌"}')
print(f'    C.3 L* centroid 안정성: {"✅" if c3_pass else "❌"}')

print(f"\n" + "=" * 60)
if all_pass:
    print("🎉 전체 검증 통과! 다음 단계로 진행 가능합니다.")
else:
    print("⚠️ 일부 항목 실패. 위 결과를 확인하세요.")
print("=" * 60)
