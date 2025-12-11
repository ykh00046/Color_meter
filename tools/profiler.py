"""
Performance Profiler for Contact Lens Inspection Pipeline

파이프라인 각 단계별 성능을 측정하고 병목 구간을 식별합니다.
"""

import sys
import time
import gc
from pathlib import Path
from typing import Dict, List, Tuple
import psutil
import numpy as np
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.image_loader import ImageLoader, ImageConfig
from src.core.lens_detector import LensDetector, DetectorConfig
from src.core.radial_profiler import RadialProfiler, ProfilerConfig
from src.core.zone_segmenter import ZoneSegmenter, SegmenterConfig
from src.core.color_evaluator import ColorEvaluator
from src.pipeline import InspectionPipeline
from src.utils.file_io import read_json


@dataclass
class ProfilingResult:
    """프로파일링 결과"""
    step_name: str
    elapsed_ms: float
    memory_mb: float
    cpu_percent: float


class PerformanceProfiler:
    """성능 프로파일러"""

    def __init__(self):
        self.process = psutil.Process()

    def profile_single_image(
        self,
        image_path: Path,
        sku_config: dict
    ) -> Tuple[List[ProfilingResult], float]:
        """
        단일 이미지 처리 프로파일링.

        Returns:
            (step_results, total_time_ms)
        """
        results = []

        # Initialize modules
        image_loader = ImageLoader(ImageConfig())
        lens_detector = LensDetector(DetectorConfig())
        radial_profiler = RadialProfiler(ProfilerConfig())
        zone_segmenter = ZoneSegmenter(SegmenterConfig())
        color_evaluator = ColorEvaluator(sku_config)

        # Step 1: Image Loading
        gc.collect()
        start_mem = self.process.memory_info().rss / 1024 / 1024
        start_cpu = self.process.cpu_percent()
        start_time = time.perf_counter()

        image = image_loader.load_from_file(image_path)

        end_time = time.perf_counter()
        end_mem = self.process.memory_info().rss / 1024 / 1024
        end_cpu = self.process.cpu_percent()

        results.append(ProfilingResult(
            step_name='1. Image Loading',
            elapsed_ms=(end_time - start_time) * 1000,
            memory_mb=end_mem - start_mem,
            cpu_percent=end_cpu
        ))

        # Step 2: Preprocessing
        gc.collect()
        start_mem = self.process.memory_info().rss / 1024 / 1024
        start_time = time.perf_counter()

        processed = image_loader.preprocess(image)

        end_time = time.perf_counter()
        end_mem = self.process.memory_info().rss / 1024 / 1024

        results.append(ProfilingResult(
            step_name='2. Preprocessing',
            elapsed_ms=(end_time - start_time) * 1000,
            memory_mb=end_mem - start_mem,
            cpu_percent=self.process.cpu_percent()
        ))

        # Step 3: Lens Detection
        gc.collect()
        start_mem = self.process.memory_info().rss / 1024 / 1024
        start_time = time.perf_counter()

        detection = lens_detector.detect(processed)

        end_time = time.perf_counter()
        end_mem = self.process.memory_info().rss / 1024 / 1024

        results.append(ProfilingResult(
            step_name='3. Lens Detection',
            elapsed_ms=(end_time - start_time) * 1000,
            memory_mb=end_mem - start_mem,
            cpu_percent=self.process.cpu_percent()
        ))

        # Step 4: Radial Profiling
        gc.collect()
        start_mem = self.process.memory_info().rss / 1024 / 1024
        start_time = time.perf_counter()

        profile = radial_profiler.extract_profile(processed, detection)

        end_time = time.perf_counter()
        end_mem = self.process.memory_info().rss / 1024 / 1024

        results.append(ProfilingResult(
            step_name='4. Radial Profiling',
            elapsed_ms=(end_time - start_time) * 1000,
            memory_mb=end_mem - start_mem,
            cpu_percent=self.process.cpu_percent()
        ))

        # Step 5: Zone Segmentation
        gc.collect()
        start_mem = self.process.memory_info().rss / 1024 / 1024
        start_time = time.perf_counter()

        zones = zone_segmenter.segment(profile)

        end_time = time.perf_counter()
        end_mem = self.process.memory_info().rss / 1024 / 1024

        results.append(ProfilingResult(
            step_name='5. Zone Segmentation',
            elapsed_ms=(end_time - start_time) * 1000,
            memory_mb=end_mem - start_mem,
            cpu_percent=self.process.cpu_percent()
        ))

        # Step 6: Color Evaluation
        gc.collect()
        start_mem = self.process.memory_info().rss / 1024 / 1024
        start_time = time.perf_counter()

        sku = list(sku_config['zones'].keys())[0] if sku_config.get('zones') else 'TEST'
        inspection_result = color_evaluator.evaluate(zones, 'VIS_TEST', sku_config)

        end_time = time.perf_counter()
        end_mem = self.process.memory_info().rss / 1024 / 1024

        results.append(ProfilingResult(
            step_name='6. Color Evaluation',
            elapsed_ms=(end_time - start_time) * 1000,
            memory_mb=end_mem - start_mem,
            cpu_percent=self.process.cpu_percent()
        ))

        # Calculate total
        total_time = sum(r.elapsed_ms for r in results)

        return results, total_time

    def profile_batch(
        self,
        image_paths: List[Path],
        sku_config: dict
    ) -> Dict:
        """
        배치 처리 프로파일링.

        Returns:
            {
                'total_time_ms': float,
                'avg_time_per_image_ms': float,
                'images_per_second': float,
                'peak_memory_mb': float
            }
        """
        pipeline = InspectionPipeline(sku_config)

        gc.collect()
        start_mem = self.process.memory_info().rss / 1024 / 1024
        peak_mem = start_mem

        start_time = time.perf_counter()

        for img_path in image_paths:
            result = pipeline.process(str(img_path), 'VIS_TEST')

            current_mem = self.process.memory_info().rss / 1024 / 1024
            peak_mem = max(peak_mem, current_mem)

        end_time = time.perf_counter()

        total_time = (end_time - start_time) * 1000
        avg_time = total_time / len(image_paths)
        throughput = len(image_paths) / (total_time / 1000)

        return {
            'total_time_ms': total_time,
            'avg_time_per_image_ms': avg_time,
            'images_per_second': throughput,
            'peak_memory_mb': peak_mem - start_mem
        }


def print_results(results: List[ProfilingResult], total_time: float):
    """프로파일링 결과 출력"""
    print("\n" + "="*80)
    print("  Performance Profiling Results")
    print("="*80)

    print(f"\n{'Step':<30} {'Time (ms)':<12} {'%':<8} {'Mem (MB)':<12} {'CPU %':<8}")
    print("-"*80)

    for r in results:
        percentage = (r.elapsed_ms / total_time * 100) if total_time > 0 else 0
        print(f"{r.step_name:<30} {r.elapsed_ms:>10.2f}   {percentage:>6.1f}%  {r.memory_mb:>10.2f}   {r.cpu_percent:>6.1f}%")

    print("-"*80)
    print(f"{'TOTAL':<30} {total_time:>10.2f}   {'100.0%':>7}  {'-':>10}   {'-':>6}")
    print("="*80 + "\n")


def main():
    """메인 함수"""
    print("Performance Profiler for Contact Lens Inspection Pipeline")
    print("="*80)

    # Load SKU config
    sku_path = project_root / 'config' / 'sku_db' / 'VIS_TEST.json'
    sku_config = read_json(sku_path)

    # Find test images
    data_dir = project_root / 'data' / 'raw_images'
    vis_images = sorted(data_dir.glob('VIS_*.jpg'))

    if len(vis_images) < 1:
        print("Error: No VIS_*.jpg images found")
        return 1

    print(f"Found {len(vis_images)} VIS_*.jpg images")

    # Profile single image
    print("\n" + "="*80)
    print("  SINGLE IMAGE PROFILING")
    print("="*80)
    print(f"Image: {vis_images[0].name}")

    profiler = PerformanceProfiler()
    results, total_time = profiler.profile_single_image(vis_images[0], sku_config)

    print_results(results, total_time)

    # Identify bottleneck
    slowest = max(results, key=lambda r: r.elapsed_ms)
    print(f"BOTTLENECK: {slowest.step_name} ({slowest.elapsed_ms:.2f}ms, "
          f"{slowest.elapsed_ms/total_time*100:.1f}% of total time)")

    # Profile batch (different sizes)
    print("\n" + "="*80)
    print("  BATCH PROCESSING PROFILING")
    print("="*80)

    for batch_size in [1, 6, 10]:
        if batch_size > len(vis_images):
            # Repeat images to reach batch size
            batch_images = (vis_images * (batch_size // len(vis_images) + 1))[:batch_size]
        else:
            batch_images = vis_images[:batch_size]

        print(f"\nBatch size: {batch_size} images")
        batch_result = profiler.profile_batch(batch_images, sku_config)

        print(f"  Total time:        {batch_result['total_time_ms']:>10.2f} ms")
        print(f"  Avg per image:     {batch_result['avg_time_per_image_ms']:>10.2f} ms")
        print(f"  Throughput:        {batch_result['images_per_second']:>10.2f} images/sec")
        print(f"  Peak memory:       {batch_result['peak_memory_mb']:>10.2f} MB")

    print("\n" + "="*80)
    print("  OPTIMIZATION RECOMMENDATIONS")
    print("="*80)

    # Recommendations based on bottleneck
    if slowest.step_name == '4. Radial Profiling':
        print("\n1. Radial Profiling is the bottleneck:")
        print("   - Consider NumPy vectorization improvements")
        print("   - Cache polar transformation matrices")
        print("   - Reduce resolution for initial profiling")

    elif slowest.step_name == '3. Lens Detection':
        print("\n1. Lens Detection is the bottleneck:")
        print("   - Tune Hough circle parameters")
        print("   - Consider faster alternative (template matching)")
        print("   - Reduce search space based on image size")

    elif slowest.step_name == '1. Image Loading':
        print("\n1. Image Loading is the bottleneck:")
        print("   - Use parallel loading for batch processing")
        print("   - Consider image format optimization")
        print("   - Implement caching for repeated access")

    print("\n2. Memory optimization:")
    print("   - Release intermediate results when not needed")
    print("   - Use in-place operations where possible")
    print("   - Implement streaming for large batches")

    print("\n3. Parallelization opportunities:")
    print("   - Batch image loading (ThreadPoolExecutor)")
    print("   - Independent zone processing")
    print("   - Visualization generation")

    print("\n" + "="*80)
    print("Profiling complete!")
    print("="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
