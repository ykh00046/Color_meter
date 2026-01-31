"""
Performance Benchmark: Plate Engine vs Plate Gate

Compares execution time between:
1. Full plate analysis (plate_engine.py)
2. Fast gate extraction (plate_gate.py)

Usage:
    python scripts/benchmark_plate_gate.py --white path/to/white.jpg --black path/to/black.jpg
"""

import argparse

# Add repo root to path
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine_v7.core.config_loader import load_cfg_with_sku
from src.engine_v7.core.plate.plate_engine import analyze_plate_pair
from src.engine_v7.core.plate.plate_gate import extract_plate_gate


def benchmark(white_path: str, black_path: str, n_runs: int = 5):
    """Run performance benchmark."""

    # Load images
    white_bgr = cv2.imread(white_path)
    black_bgr = cv2.imread(black_path)

    if white_bgr is None or black_bgr is None:
        print("Error: Could not load images")
        return

    # Load config (use default SKU)
    cfg = load_cfg_with_sku()

    print(f"Running {n_runs} iterations...")
    print("=" * 60)

    # Benchmark: Full Analysis
    times_full = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        result_full = analyze_plate_pair(white_bgr, black_bgr, cfg)
        t_elapsed = time.perf_counter() - t0
        times_full.append(t_elapsed)
        print(f"Full Analysis #{i+1}: {t_elapsed:.3f}s")

    print("-" * 60)

    # Benchmark: Fast Gate
    times_gate = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        result_gate = extract_plate_gate(white_bgr, black_bgr, cfg)
        t_elapsed = time.perf_counter() - t0
        times_gate.append(t_elapsed)
        print(f"Fast Gate #{i+1}: {t_elapsed:.3f}s")

    print("=" * 60)

    # Calculate statistics
    avg_full = np.mean(times_full)
    avg_gate = np.mean(times_gate)
    speedup = avg_full / avg_gate

    print(f"\n📊 **Results (avg of {n_runs} runs)**:")
    print(f"  Full Analysis: {avg_full:.3f}s ± {np.std(times_full):.3f}s")
    print(f"  Fast Gate:     {avg_gate:.3f}s ± {np.std(times_gate):.3f}s")
    print(f"  Speedup:       {speedup:.2f}x faster")
    print(f"  Time Saved:    {(avg_full - avg_gate) * 1000:.0f}ms per sample")

    # Validate mask equivalence
    print(f"\n🔍 **Mask Validation**:")
    mask_full = result_full.get("_masks", {}).get("ink_mask_core_polar")
    mask_gate = result_gate.get("ink_mask_core_polar")

    if mask_full is not None and mask_gate is not None:
        # Calculate overlap ratio
        overlap = np.sum(mask_full & mask_gate) / max(np.sum(mask_full | mask_gate), 1)
        print(f"  Mask Overlap:  {overlap * 100:.1f}%")

        if overlap > 0.95:
            print(f"  ✅ Masks are highly similar (acceptable)")
        elif overlap > 0.85:
            print(f"  ⚠️  Masks have moderate differences")
        else:
            print(f"  ❌ Masks are significantly different (investigate)")
    else:
        print(f"  ⚠️  Could not compare masks (one or both missing)")

    # Success criteria
    print(f"\n✅ **Success Criteria**:")
    print(f"  Target: Gate < 50% of Full time")
    ratio = avg_gate / avg_full
    if ratio < 0.5:
        print(f"  Status: ✅ PASSED ({ratio*100:.1f}% of full time)")
    else:
        print(f"  Status: ❌ FAILED ({ratio*100:.1f}% of full time)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark plate gate performance")
    parser.add_argument("--white", required=True, help="Path to white backlight image")
    parser.add_argument("--black", required=True, help="Path to black backlight image")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")

    args = parser.parse_args()

    benchmark(args.white, args.black, args.runs)
