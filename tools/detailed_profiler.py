"""
Detailed Radial Profiler Performance Analysis

Radial profiling 내부 각 단계별 성능을 측정합니다.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.image_loader import ImageConfig, ImageLoader
from src.core.lens_detector import DetectorConfig, LensDetector
from src.core.radial_profiler import ProfilerConfig, RadialProfiler
from src.utils.file_io import read_json


def profile_radial_profiling_detailed():
    """Radial profiling 세부 단계별 측정"""

    # Load test image
    data_dir = project_root / "data" / "raw_images"
    test_image = data_dir / "VIS_NG_001.jpg"

    # Initialize
    loader = ImageLoader(ImageConfig())
    detector = LensDetector(DetectorConfig())
    profiler = RadialProfiler(ProfilerConfig())

    # Prepare image
    image = loader.load_from_file(test_image)
    processed = loader.preprocess(image)
    detection = detector.detect(processed)

    cx, cy, radius = detection.center_x, detection.center_y, detection.radius
    r_samples = int(radius / profiler.config.r_step_pixels)

    print("=" * 80)
    print("  Detailed Radial Profiling Analysis")
    print("=" * 80)
    print(f"Image: {test_image.name}")
    print(f"Radius: {radius:.1f}px, r_samples: {r_samples}, theta: {profiler.config.theta_samples}")
    print("=" * 80)

    # Step 1: warpPolar
    start = time.perf_counter()
    polar_image = cv2.warpPolar(
        processed,
        (r_samples, profiler.config.theta_samples),
        (cx, cy),
        radius,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS,
    )
    t1 = (time.perf_counter() - start) * 1000

    # Step 2: cvtColor
    start = time.perf_counter()
    polar_lab = cv2.cvtColor(polar_image, cv2.COLOR_BGR2LAB)
    t2 = (time.perf_counter() - start) * 1000

    # Step 3: mean calculations
    start = time.perf_counter()
    L_profile = polar_lab[:, :, 0].mean(axis=1)
    a_profile = polar_lab[:, :, 1].mean(axis=1)
    b_profile = polar_lab[:, :, 2].mean(axis=1)
    t3 = (time.perf_counter() - start) * 1000

    # Step 4: std calculations
    start = time.perf_counter()
    std_L = polar_lab[:, :, 0].std(axis=1)
    std_a = polar_lab[:, :, 1].std(axis=1)
    std_b = polar_lab[:, :, 2].std(axis=1)
    t4 = (time.perf_counter() - start) * 1000

    # Step 5: smoothing (simulate)
    from scipy.signal import savgol_filter

    start = time.perf_counter()
    if len(L_profile) >= 11:
        L_smooth = savgol_filter(L_profile, 11, 3)
        a_smooth = savgol_filter(a_profile, 11, 3)
        b_smooth = savgol_filter(b_profile, 11, 3)
    t5 = (time.perf_counter() - start) * 1000

    total = t1 + t2 + t3 + t4 + t5

    print(f"\n{'Step':<40} {'Time (ms)':<12} {'%':<8}")
    print("-" * 80)
    print(f"{'1. cv2.warpPolar (polar transform)':<40} {t1:>10.2f}   {t1/total*100:>6.1f}%")
    print(f"{'2. cv2.cvtColor (BGR->LAB)':<40} {t2:>10.2f}   {t2/total*100:>6.1f}%")
    print(f"{'3. NumPy mean (3 channels)':<40} {t3:>10.2f}   {t3/total*100:>6.1f}%")
    print(f"{'4. NumPy std (3 channels)':<40} {t4:>10.2f}   {t4/total*100:>6.1f}%")
    print(f"{'5. Savgol filter smoothing':<40} {t5:>10.2f}   {t5/total*100:>6.1f}%")
    print("-" * 80)
    print(f"{'TOTAL':<40} {total:>10.2f}   {'100.0%':>7}")
    print("=" * 80)

    # Test with different theta_samples
    print("\n" + "=" * 80)
    print("  Impact of theta_samples on Performance")
    print("=" * 80)

    for theta in [90, 180, 360, 720]:
        start = time.perf_counter()
        polar_test = cv2.warpPolar(
            processed, (r_samples, theta), (cx, cy), radius, cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS
        )
        polar_lab_test = cv2.cvtColor(polar_test, cv2.COLOR_BGR2LAB)
        L_test = polar_lab_test[:, :, 0].mean(axis=1)
        elapsed = (time.perf_counter() - start) * 1000
        total_sequential = t1 + t2 + t3
        speedup = total_sequential / elapsed if elapsed > 0 else 0

        print(
            f"theta={theta:>4}: {elapsed:>8.2f}ms  "
            f"(speedup: {total_sequential:>6.2f}ms / {elapsed:>6.2f}ms = {speedup:.2f}x)"
        )

    print("=" * 80)


if __name__ == "__main__":
    profile_radial_profiling_detailed()
