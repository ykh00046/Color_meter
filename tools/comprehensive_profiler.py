"""
Comprehensive Performance Profiler - 2025-12-16

최신 시스템 (2D Zone Analysis + InkEstimator) 기반 성능 측정
"""

import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import psutil

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2

from src.core.ink_estimator import InkEstimator
from src.core.lens_detector import DetectorConfig, LensDetector
from src.core.zone_analyzer_2d import InkMaskConfig, analyze_lens_zones_2d
from src.utils.file_io import read_json


@dataclass
class StepResult:
    """개별 단계 결과"""

    name: str
    time_ms: float
    memory_mb: float
    cpu_percent: float


class ComprehensiveProfiler:
    """포괄적 성능 프로파일러"""

    def __init__(self):
        self.process = psutil.Process()

    def profile_2d_analysis(self, image_path: Path, sku_config: dict) -> tuple[List[StepResult], float]:
        """2D Zone Analysis 전체 파이프라인 프로파일링"""
        results = []

        # Step 1: Image Loading
        gc.collect()
        mem_start = self.process.memory_info().rss / 1024 / 1024
        t_start = time.perf_counter()

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")

        t_elapsed = (time.perf_counter() - t_start) * 1000
        mem_end = self.process.memory_info().rss / 1024 / 1024
        results.append(
            StepResult(
                name="1. Image Loading",
                time_ms=t_elapsed,
                memory_mb=mem_end - mem_start,
                cpu_percent=self.process.cpu_percent(),
            )
        )

        # Step 2: Lens Detection
        gc.collect()
        mem_start = self.process.memory_info().rss / 1024 / 1024
        t_start = time.perf_counter()

        detector = LensDetector(DetectorConfig())
        lens_detection = detector.detect(img_bgr)

        t_elapsed = (time.perf_counter() - t_start) * 1000
        mem_end = self.process.memory_info().rss / 1024 / 1024
        results.append(
            StepResult(
                name="2. Lens Detection",
                time_ms=t_elapsed,
                memory_mb=mem_end - mem_start,
                cpu_percent=self.process.cpu_percent(),
            )
        )

        # Step 3: 2D Zone Analysis (Main Analysis)
        gc.collect()
        mem_start = self.process.memory_info().rss / 1024 / 1024
        t_start = time.perf_counter()

        result_2d, debug_info_2d = analyze_lens_zones_2d(
            img_bgr=img_bgr,
            lens_detection=lens_detection,
            sku_config=sku_config,
            ink_mask_config=InkMaskConfig(),
            save_debug=False,
        )

        t_elapsed = (time.perf_counter() - t_start) * 1000
        mem_end = self.process.memory_info().rss / 1024 / 1024
        results.append(
            StepResult(
                name="3. 2D Zone Analysis",
                time_ms=t_elapsed,
                memory_mb=mem_end - mem_start,
                cpu_percent=self.process.cpu_percent(),
            )
        )

        # Step 4: Image-Based Ink Estimation (if enabled)
        if sku_config.get("params", {}).get("enable_image_based_ink_analysis", True):
            gc.collect()
            mem_start = self.process.memory_info().rss / 1024 / 1024
            t_start = time.perf_counter()

            estimator = InkEstimator()
            roi = lens_detection.roi
            if roi:
                x, y, w, h = roi
                roi_image = img_bgr[y : y + h, x : x + w]
                ink_result = estimator.estimate_from_array(roi_image)

            t_elapsed = (time.perf_counter() - t_start) * 1000
            mem_end = self.process.memory_info().rss / 1024 / 1024
            results.append(
                StepResult(
                    name="4. Image-Based Ink Estimation",
                    time_ms=t_elapsed,
                    memory_mb=mem_end - mem_start,
                    cpu_percent=self.process.cpu_percent(),
                )
            )

        total_time = sum(r.time_ms for r in results)
        return results, total_time

    def profile_batch(self, image_paths: List[Path], sku_config: dict) -> Dict:
        """배치 처리 프로파일링"""
        gc.collect()
        mem_start = self.process.memory_info().rss / 1024 / 1024
        peak_mem = mem_start

        t_start = time.perf_counter()

        for img_path in image_paths:
            try:
                img_bgr = cv2.imread(str(img_path))
                detector = LensDetector(DetectorConfig())
                lens_detection = detector.detect(img_bgr)
                result_2d, _ = analyze_lens_zones_2d(
                    img_bgr=img_bgr,
                    lens_detection=lens_detection,
                    sku_config=sku_config,
                    save_debug=False,
                )

                current_mem = self.process.memory_info().rss / 1024 / 1024
                peak_mem = max(peak_mem, current_mem)
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

        t_elapsed = (time.perf_counter() - t_start) * 1000

        return {
            "total_time_ms": t_elapsed,
            "avg_time_ms": t_elapsed / len(image_paths) if image_paths else 0,
            "throughput_per_sec": len(image_paths) / (t_elapsed / 1000) if t_elapsed > 0 else 0,
            "peak_memory_mb": peak_mem - mem_start,
        }


def print_step_results(results: List[StepResult], total_time: float):
    """단계별 결과 출력"""
    print("\n" + "=" * 90)
    print("  Performance Profiling Results (Latest System)")
    print("=" * 90)
    print(f"\n{'Step':<40} {'Time (ms)':<12} {'%':<8} {'Mem (MB)':<12} {'CPU %':<8}")
    print("-" * 90)

    for r in results:
        pct = (r.time_ms / total_time * 100) if total_time > 0 else 0
        print(f"{r.name:<40} {r.time_ms:>10.2f}   {pct:>6.1f}%  {r.memory_mb:>10.2f}   {r.cpu_percent:>6.1f}%")

    print("-" * 90)
    print(f"{'TOTAL':<40} {total_time:>10.2f}   {'100.0%':>7}  {'-':>10}   {'-':>6}")
    print("=" * 90 + "\n")


def main():
    """메인 함수"""
    print("\n" + "=" * 90)
    print("  COMPREHENSIVE PERFORMANCE PROFILER - Contact Lens Inspection System")
    print("  Version: 2025-12-16 (2D Zone Analysis + InkEstimator)")
    print("=" * 90)

    # Load test images and SKU
    data_dir = project_root / "data" / "raw_images"
    test_images = sorted(data_dir.glob("*.jpg"))

    if not test_images:
        print("ERROR: No test images found in data/raw_images/")
        return 1

    # Load SKU config
    sku_config_path = project_root / "config" / "sku_db" / "SKU001.json"
    if not sku_config_path.exists():
        print(f"ERROR: SKU config not found: {sku_config_path}")
        return 1

    sku_config = read_json(sku_config_path)

    print(f"\nFound {len(test_images)} test images")
    print(f"SKU Config: {sku_config_path.name}")

    # Profile single image
    print("\n" + "=" * 90)
    print("  SINGLE IMAGE PROFILING")
    print("=" * 90)
    print(f"Image: {test_images[0].name}")

    profiler = ComprehensiveProfiler()
    results, total_time = profiler.profile_2d_analysis(test_images[0], sku_config)

    print_step_results(results, total_time)

    # Identify bottleneck
    bottleneck = max(results, key=lambda r: r.time_ms)
    print(
        f"BOTTLENECK: {bottleneck.name} ({bottleneck.time_ms:.2f}ms, "
        f"{bottleneck.time_ms / total_time * 100:.1f}% of total)"
    )

    # Batch profiling
    print("\n" + "=" * 90)
    print("  BATCH PROCESSING PROFILING")
    print("=" * 90)

    for batch_size in [1, 5, 10, 20]:
        if batch_size > len(test_images):
            # Repeat images to meet batch size
            batch = (test_images * ((batch_size // len(test_images)) + 1))[:batch_size]
        else:
            batch = test_images[:batch_size]

        print(f"\nBatch size: {batch_size} images")
        batch_result = profiler.profile_batch(batch, sku_config)

        print(f"  Total time:          {batch_result['total_time_ms']:>10.2f} ms")
        print(f"  Avg per image:       {batch_result['avg_time_ms']:>10.2f} ms")
        print(f"  Throughput:          {batch_result['throughput_per_sec']:>10.2f} images/sec")
        print(f"  Peak memory:         {batch_result['peak_memory_mb']:>10.2f} MB")

    # System info
    print("\n" + "=" * 90)
    print("  SYSTEM INFORMATION")
    print("=" * 90)
    import platform

    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  CPU: {psutil.cpu_count(logical=False)} cores ({psutil.cpu_count()} threads)")
    print(f"  RAM: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
    print(f"  Python: {platform.python_version()}")

    # Recommendations
    print("\n" + "=" * 90)
    print("  OPTIMIZATION RECOMMENDATIONS")
    print("=" * 90)

    if "2D Zone Analysis" in bottleneck.name:
        print("\n1. 2D Zone Analysis is the bottleneck:")
        print("   - Consider reducing zone_masks resolution")
        print("   - Optimize sector statistics calculation")
        print("   - Cache ink_mask between SKUs if applicable")

    if "Ink Estimation" in bottleneck.name:
        print("\n2. Ink Estimation optimization:")
        print("   - Reduce GMM max_components for faster BIC")
        print("   - Implement early termination for single-ink cases")
        print("   - Cache specular rejection masks")

    print("\n3. General optimizations:")
    print("   - Use NumPy vectorization for pixel operations")
    print("   - Implement multi-threading for batch processing")
    print("   - Consider GPU acceleration for cv2.warpPolar")

    print("\n4. Memory optimization:")
    print("   - Release debug_info when not needed")
    print("   - Use np.float32 instead of np.float64 where possible")
    print("   - Implement streaming for large batch jobs")

    print("\n" + "=" * 90)
    print("  Profiling Complete!")
    print("=" * 90 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
