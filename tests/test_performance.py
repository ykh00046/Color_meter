"""
Performance Regression Tests

성능 저하를 방지하기 위한 회귀 테스트.
"""

import time
from pathlib import Path

import pytest

from src.pipeline import InspectionPipeline
from src.utils.file_io import read_json


@pytest.fixture
def test_image():
    """Test image path"""
    img_path = Path("data/raw_images/VIS_OK_001.jpg")
    if not img_path.exists():
        pytest.skip(f"Test image not found: {img_path}")
    return str(img_path)


@pytest.fixture
def sku_config():
    """VIS_TEST SKU config"""
    return read_json(Path("config/sku_db/VIS_TEST.json"))


@pytest.fixture
def pipeline(sku_config):
    """Pipeline instance"""
    return InspectionPipeline(sku_config)


def test_single_image_performance(pipeline, test_image):
    """단일 이미지 처리가 200ms 이내에 완료되어야 함 (회귀 방지)"""
    start = time.perf_counter()
    result = pipeline.process(test_image, "VIS_TEST")
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert result is not None
    assert elapsed_ms < 200, f"Processing took {elapsed_ms:.2f}ms, expected <200ms"


def test_batch_processing_linear_scaling(pipeline):
    """배치 처리가 선형 시간 복잡도를 유지해야 함"""
    vis_images = list(Path("data/raw_images").glob("VIS_*.jpg"))

    if len(vis_images) < 3:
        pytest.skip("Not enough VIS_*.jpg images for batch test")

    # Test with 1 image
    start = time.perf_counter()
    results_1 = pipeline.process_batch([str(vis_images[0])], "VIS_TEST")
    time_1 = time.perf_counter() - start

    # Test with 3 images
    start = time.perf_counter()
    results_3 = pipeline.process_batch([str(img) for img in vis_images[:3]], "VIS_TEST")
    time_3 = time.perf_counter() - start

    assert len(results_1) == 1
    assert len(results_3) == 3

    # Time should scale roughly linearly (allow 50% overhead)
    expected_time_3 = time_1 * 3 * 1.5
    assert time_3 < expected_time_3, f"Batch 3 took {time_3:.2f}s, expected <{expected_time_3:.2f}s"


def test_radial_profiling_performance(pipeline, test_image, sku_config):
    """Radial profiling 단계가 전체 시간의 95% 미만이어야 함"""
    from src.core.image_loader import ImageConfig, ImageLoader
    from src.core.lens_detector import DetectorConfig, LensDetector
    from src.core.radial_profiler import ProfilerConfig, RadialProfiler

    loader = ImageLoader(ImageConfig())
    detector = LensDetector(DetectorConfig())
    profiler = RadialProfiler(ProfilerConfig())

    # Total pipeline time
    start = time.perf_counter()
    result = pipeline.process(test_image, "VIS_TEST")
    total_time = time.perf_counter() - start

    # Radial profiling time
    image = loader.load_from_file(Path(test_image))
    processed = loader.preprocess(image)
    detection = detector.detect(processed)

    start = time.perf_counter()
    profile = profiler.extract_profile(processed, detection)
    profiling_time = time.perf_counter() - start

    profiling_ratio = profiling_time / total_time

    assert profiling_ratio < 0.95, f"Radial profiling takes {profiling_ratio*100:.1f}% of total time (should be <95%)"


def test_memory_efficiency(pipeline):
    """메모리 사용량이 배치 크기에 무관하게 일정 수준을 유지해야 함"""
    import gc

    import psutil

    vis_images = list(Path("data/raw_images").glob("VIS_*.jpg"))

    if len(vis_images) < 6:
        pytest.skip("Not enough VIS_*.jpg images")

    process = psutil.Process()

    # Batch of 3
    gc.collect()
    mem_before_3 = process.memory_info().rss / 1024 / 1024
    results_3 = pipeline.process_batch([str(img) for img in vis_images[:3]], "VIS_TEST")
    mem_after_3 = process.memory_info().rss / 1024 / 1024
    mem_increase_3 = mem_after_3 - mem_before_3

    # Batch of 6
    gc.collect()
    mem_before_6 = process.memory_info().rss / 1024 / 1024
    results_6 = pipeline.process_batch([str(img) for img in vis_images[:6]], "VIS_TEST")
    mem_after_6 = process.memory_info().rss / 1024 / 1024
    mem_increase_6 = mem_after_6 - mem_before_6

    assert len(results_3) == 3
    assert len(results_6) == 6

    # Memory increase should not explode when batch size doubles.
    # Use a minimum baseline of 1MB to avoid divide-by-zero cases,
    # and cap allowed growth by both ratio and absolute ceiling.
    baseline = max(mem_increase_3, 1.0)
    allowed = min(baseline * 4.0, 12.0)  # ratio cap and absolute cap
    message = (
        f"Memory increased {mem_increase_6:.1f}MB for 6 images vs "
        f"{mem_increase_3:.1f}MB for 3 images (allowed <{allowed:.1f}MB)"
    )
    assert mem_increase_6 < allowed, message


@pytest.mark.skip(reason="Parallel processing has overhead for small batches")
def test_parallel_batch_processing(pipeline):
    """병렬 배치 처리 기능 테스트 (large batches only)"""
    vis_images = [str(img) for img in list(Path("data/raw_images").glob("VIS_*.jpg"))[:6]]

    if len(vis_images) < 6:
        pytest.skip("Not enough test images")

    # Sequential
    start = time.perf_counter()
    results_seq = pipeline.process_batch(vis_images, "VIS_TEST", parallel=False)
    time_seq = time.perf_counter() - start

    # Parallel
    start = time.perf_counter()
    results_par = pipeline.process_batch(vis_images, "VIS_TEST", parallel=True, max_workers=4)
    time_par = time.perf_counter() - start

    assert len(results_seq) == len(results_par) == 6

    # Just verify it works (small batches have parallel overhead)
    assert results_par is not None
