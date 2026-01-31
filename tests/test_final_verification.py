"""
Hard Gate Final Verification - 5 Essential Checks
A. Hard Gate Application Rate (5 runs)
B. expected_k Stability
C. Gray/Reflection Fake Ink Removal
D. Light Ink Survival Rate
E. Simulation Input (role filter)
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

# ============================================================
# Setup
# ============================================================
white_bgr = cv2.imread("data/raw_images/SKU001_OK_001.jpg")
black_bgr = cv2.imread("data/Black_A.png")

cfg = load_config("GGG")
cfg["expected_ink_count"] = 2  # This sample has 2 inks
cfg["polar"] = {"R": 260, "T": 720}

geom = detect_lens_circle(white_bgr)

print("=" * 70)
print("Hard Gate Final Verification - 5 Essential Checks")
print("=" * 70)

# ============================================================
# Run 5 consecutive analyses
# ============================================================
NUM_RUNS = 5
all_results = []

print(f"\nRunning {NUM_RUNS} consecutive analyses...")
for i in range(NUM_RUNS):
    result = analyze_single_sample(
        test_bgr=white_bgr, cfg=cfg, analysis_modes=["gate", "plate", "ink"], black_bgr=black_bgr
    )
    all_results.append(result)
    print(f"  Run {i+1}/{NUM_RUNS} complete")

# ============================================================
# A. Hard Gate Application Rate
# ============================================================
print("\n" + "=" * 70)
print("[A] Hard Gate Application Rate (Reproducibility)")
print("-" * 70)

hard_gate_rules = []
n_pixels_list = []

for i, result in enumerate(all_results):
    ink = result.get("ink", {})
    sample_meta = ink.get("sample_meta", {})
    rule = sample_meta.get("rule", "unknown")
    n_pixels = sample_meta.get("n_pixels_used", 0)
    hard_gate_rules.append(rule)
    n_pixels_list.append(n_pixels)
    print(f"  Run {i+1}: rule={rule}, n_pixels={n_pixels}")

# Check A.1: All runs should have override_hard_gate
override_count = sum(1 for r in hard_gate_rules if r == "plate_gate_override")
a1_pass = override_count == NUM_RUNS
print(f'\n  A.1 Hard Gate 적용률: {override_count}/{NUM_RUNS} {"✅ PASS" if a1_pass else "❌ FAIL"}')

# Check A.2: n_pixels_used stability (within ±20% of mean)
if n_pixels_list and all(n > 0 for n in n_pixels_list):
    mean_pixels = np.mean(n_pixels_list)
    max_deviation = max(abs(n - mean_pixels) / mean_pixels * 100 for n in n_pixels_list)
    a2_pass = max_deviation <= 20
    print(f'  A.2 n_pixels 안정성 (±20%): Max deviation={max_deviation:.1f}% {"✅ PASS" if a2_pass else "❌ FAIL"}')
    print(f"      Mean: {mean_pixels:.0f}, Range: [{min(n_pixels_list)}, {max(n_pixels_list)}]")
else:
    a2_pass = False
    print(f"  A.2 n_pixels 안정성: ❌ FAIL (no valid pixels)")

# ============================================================
# B. expected_k Stability
# ============================================================
print("\n" + "=" * 70)
print("[B] expected_k Stability")
print("-" * 70)

expected_k = cfg["expected_ink_count"]
k_results = []
detected_ink_counts = []

for i, result in enumerate(all_results):
    ink = result.get("ink", {})
    k = ink.get("k", 0)
    clusters = ink.get("clusters", [])
    ink_clusters = [c for c in clusters if c.get("role") == "ink"]
    detected_ink_count = len(ink_clusters)

    k_results.append(k)
    detected_ink_counts.append(detected_ink_count)
    print(f"  Run {i+1}: k={k}, detected_ink_count={detected_ink_count}")

# Check B.1: k matches expected_k
b1_pass = all(k == expected_k for k in k_results)
print(f'\n  B.1 k == expected_k ({expected_k}): {k_results} {"✅ PASS" if b1_pass else "❌ FAIL"}')

# Check B.2: No fluctuation (always same)
b2_pass = len(set(k_results)) == 1
print(f'  B.2 k 변동 없음: {"✅ PASS" if b2_pass else "❌ FAIL"} (unique values: {set(k_results)})')

# ============================================================
# C. Gray/Reflection Fake Ink Removal
# ============================================================
print("\n" + "=" * 70)
print("[C] Gray/Reflection Fake Ink Removal")
print("-" * 70)

GRAY_L_THRESHOLD = 70  # L* > 70 is considered gray/white
fake_ink_runs = []

for i, result in enumerate(all_results):
    ink = result.get("ink", {})
    clusters = ink.get("clusters", [])

    # Check for fake gray/white clusters with role='ink'
    ink_clusters = [c for c in clusters if c.get("role") == "ink"]
    fake_inks = []
    for c in ink_clusters:
        L_star = c.get("centroid_lab_cie", [0])[0]
        if L_star > GRAY_L_THRESHOLD:
            fake_inks.append(
                {"L": round(L_star, 1), "hex": c.get("mean_hex", "N/A"), "area_ratio": c.get("area_ratio", 0)}
            )

    fake_ink_runs.append(len(fake_inks))
    if fake_inks:
        print(f"  Run {i+1}: ⚠️ {len(fake_inks)} fake ink(s) found: {fake_inks}")
    else:
        print(f"  Run {i+1}: ✅ No fake gray/white inks")

c_pass = all(count == 0 for count in fake_ink_runs)
print(f'\n  C. Gray/White 가짜 잉크 제거: {"✅ PASS" if c_pass else "❌ FAIL"}')
print(f"     Fake ink counts per run: {fake_ink_runs}")

# ============================================================
# D. Light Ink (Yellow/Beige) Survival Rate
# ============================================================
print("\n" + "=" * 70)
print("[D] Light Ink Survival Rate")
print("-" * 70)

# Track the lightest ink cluster across runs (excluding gray/white fakes)
lightest_ink_L_values = []
lightest_ink_areas = []

for i, result in enumerate(all_results):
    ink = result.get("ink", {})
    clusters = ink.get("clusters", [])

    # Get all ink clusters
    ink_clusters = [c for c in clusters if c.get("role") == "ink"]

    if ink_clusters:
        # Find the lightest ink (highest L* but still < gray threshold)
        valid_inks = [c for c in ink_clusters if c.get("centroid_lab_cie", [0])[0] <= GRAY_L_THRESHOLD]
        if valid_inks:
            lightest = max(valid_inks, key=lambda c: c.get("centroid_lab_cie", [0])[0])
            L_star = lightest.get("centroid_lab_cie", [0])[0]
            area = lightest.get("area_ratio", 0)
            lightest_ink_L_values.append(L_star)
            lightest_ink_areas.append(area)
            print(f"  Run {i+1}: Lightest ink L*={L_star:.1f}, area_ratio={area:.3f}")
        else:
            print(f"  Run {i+1}: No valid light inks found")
            lightest_ink_L_values.append(None)
            lightest_ink_areas.append(None)
    else:
        print(f"  Run {i+1}: No ink clusters found")
        lightest_ink_L_values.append(None)
        lightest_ink_areas.append(None)

# Check D.1: Light ink L* stability
valid_L = [v for v in lightest_ink_L_values if v is not None]
if len(valid_L) >= 2:
    L_range = max(valid_L) - min(valid_L)
    d1_pass = L_range < 15  # L* should not jump more than 15
    print(f'\n  D.1 Light ink L* 안정성 (range < 15): Range={L_range:.1f} {"✅ PASS" if d1_pass else "❌ FAIL"}')
else:
    d1_pass = False
    print(f"\n  D.1 Light ink L* 안정성: ❌ FAIL (insufficient data)")

# Check D.2: Light ink area_ratio > 0
valid_areas = [a for a in lightest_ink_areas if a is not None]
if valid_areas:
    d2_pass = all(a > 0 for a in valid_areas)
    print(f'  D.2 Light ink area_ratio > 0: {valid_areas} {"✅ PASS" if d2_pass else "❌ FAIL"}')
else:
    d2_pass = False
    print(f"  D.2 Light ink area_ratio > 0: ❌ FAIL (no data)")

# ============================================================
# E. Simulation Input (role filter)
# ============================================================
print("\n" + "=" * 70)
print("[E] Simulation Input (role filter)")
print("-" * 70)

simulation_checks = []

for i, result in enumerate(all_results):
    sim = result.get("color_simulation", {})
    simulations = sim.get("simulations", [])

    # Get all cluster roles from ink result
    ink_result = result.get("ink", {})
    all_clusters = ink_result.get("clusters", [])

    ink_role_count = sum(1 for c in all_clusters if c.get("role") == "ink")
    gap_role_count = sum(1 for c in all_clusters if c.get("role") == "gap")
    sim_ink_count = len(simulations)

    # Check: simulation should only include role='ink' clusters
    # Also verify all simulations have role='ink'
    all_sim_roles_ink = all(s.get("role") == "ink" for s in simulations)
    match = sim_ink_count == ink_role_count and all_sim_roles_ink

    simulation_checks.append(match)

    sim_roles = [s.get("role") for s in simulations]
    status = "✅" if match else "❌"
    print(
        f"  Run {i+1}: ink={ink_role_count}, gap={gap_role_count}, " f"sim={sim_ink_count}, roles={sim_roles} {status}"
    )

e_pass = all(simulation_checks)
print(f'\n  E. Simulation에 role="ink"만 포함: {"✅ PASS" if e_pass else "❌ FAIL"}')

# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 70)
print("FINAL VERIFICATION SUMMARY")
print("=" * 70)

all_pass = a1_pass and a2_pass and b1_pass and b2_pass and c_pass and d1_pass and d2_pass and e_pass

print(
    f"""
[A] Hard Gate Application Rate
    A.1 적용률 {override_count}/{NUM_RUNS}: {"✅" if a1_pass else "❌"}
    A.2 n_pixels 안정성 (±20%): {"✅" if a2_pass else "❌"}

[B] expected_k Stability
    B.1 k == expected_k: {"✅" if b1_pass else "❌"}
    B.2 k 변동 없음: {"✅" if b2_pass else "❌"}

[C] Gray/Reflection Removal
    가짜 잉크 제거: {"✅" if c_pass else "❌"}

[D] Light Ink Survival
    D.1 L* 안정성: {"✅" if d1_pass else "❌"}
    D.2 area_ratio > 0: {"✅" if d2_pass else "❌"}

[E] Simulation Role Filter
    role="ink" only: {"✅" if e_pass else "❌"}
"""
)

print("=" * 70)
if all_pass:
    print("🎉 ALL CHECKS PASSED! Hard Gate implementation is verified.")
else:
    print("⚠️ SOME CHECKS FAILED. Review the results above.")
print("=" * 70)
