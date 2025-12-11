# Performance Analysis Report

**Date:** 2025-12-11
**Tool:** `tools/profiler.py`
**Test Images:** VIS_*.jpg (6 images)
**SKU:** VIS_TEST

---

## Executive Summary

파이프라인 성능 프로파일링 결과, **Radial Profiling이 전체 처리 시간의 88.5%를 차지**하는 압도적인 병목으로 확인되었습니다. 현재 단일 이미지 처리 시간은 약 **90ms**이며, 목표인 100ms 이내를 달성하고 있으나, Radial Profiling을 최적화하면 **50% 이상의 성능 개선**이 가능할 것으로 예상됩니다.

---

## 1. Single Image Profiling Results

### Test Image: VIS_NG_001.jpg

| Step | Time (ms) | Percentage | Memory (MB) | CPU % |
|------|-----------|------------|-------------|-------|
| 1. Image Loading | 1.19 | 1.3% | 1.42 | 0.0% |
| 2. Preprocessing | 4.50 | 5.0% | 1.84 | 111.7% |
| 3. Lens Detection | 3.06 | 3.4% | 2.39 | 71.9% |
| **4. Radial Profiling** | **79.50** | **88.5%** | **4.30** | **80.5%** |
| 5. Zone Segmentation | 1.29 | 1.4% | 0.06 | 161.4% |
| 6. Color Evaluation | 0.24 | 0.3% | 0.03 | 78.7% |
| **TOTAL** | **89.79** | **100.0%** | - | - |

### Key Findings

1. **Radial Profiling dominates processing time (88.5%)**
   - 극좌표 변환이 주요 병목
   - NumPy 연산이지만 최적화 여지 존재
   - 79.50ms → 목표 40ms 이하로 개선 가능

2. **Other steps are well optimized (<5ms each)**
   - Image Loading: 1.19ms (파일 I/O 효율적)
   - Preprocessing: 4.50ms (CLAHE 적용 포함)
   - Lens Detection: 3.06ms (Hough Circle 빠름)
   - Zone Segmentation: 1.29ms (그래디언트 분석)
   - Color Evaluation: 0.24ms (ΔE 계산)

3. **Memory usage is acceptable**
   - Total increase: ~10MB per image
   - Radial profiling adds 4.3MB (polar transformation)

---

## 2. Batch Processing Profiling

### Batch Performance

| Batch Size | Total Time (ms) | Avg Per Image (ms) | Throughput (img/s) | Peak Memory (MB) |
|------------|-----------------|--------------------|--------------------|------------------|
| 1 | 6.45 | 6.45 | 155.06 | 1.43 |
| 6 | 40.79 | 6.80 | 147.11 | 3.14 |
| 10 | 72.80 | 7.28 | 137.37 | 2.56 |

### Observations

1. **Linear time scaling**
   - Avg per image remains ~6-7ms
   - No significant overhead from batch processing
   - Throughput: ~140-155 images/second

2. **Memory efficiency**
   - Peak memory: 2-3MB for 6-10 images
   - No memory leak observed
   - Memory released between images

3. **CPU utilization**
   - Multi-core utilization observed (CPU% > 100%)
   - NumPy operations are parallelized automatically
   - Room for explicit parallelization

---

## 3. Bottleneck Analysis

### Primary Bottleneck: Radial Profiling (88.5%)

**Current Implementation (`src/core/radial_profiler.py`):**
```python
# Polar transformation for each pixel
for angle_idx in range(num_angles):
    angle = angles[angle_idx]
    for r_idx in range(num_radii):
        r = radii[r_idx]
        x = int(center_x + r * np.cos(angle))
        y = int(center_y + r * np.sin(angle))
        # Sample pixel...
```

**Issues:**
1. Nested loops over angles and radii
2. Per-pixel coordinate calculation
3. Repeated trigonometric function calls
4. No caching of transformation matrices

---

## 4. Optimization Recommendations

### Priority 1: Radial Profiling Optimization (Expected: 50-60% speedup)

#### 4.1. Pre-compute Polar Transformation Matrices

**Current:** Calculate coordinates for every image
**Optimized:** Calculate once, cache for same image size

```python
class RadialProfiler:
    def __init__(self):
        self._polar_cache = {}  # Cache by (width, height, radius)

    def _get_polar_transform(self, shape, center, radius):
        cache_key = (shape[0], shape[1], int(radius))
        if cache_key not in self._polar_cache:
            # Compute once
            self._polar_cache[cache_key] = self._compute_polar_coords(...)
        return self._polar_cache[cache_key]
```

**Expected improvement:** 30-40ms reduction → **~50ms total time**

---

#### 4.2. Vectorized NumPy Operations

**Current:** Python loops with per-pixel operations
**Optimized:** Full NumPy vectorization

```python
# Vectorized coordinate calculation
angles_grid, radii_grid = np.meshgrid(angles, radii, indexing='ij')
x_coords = center_x + radii_grid * np.cos(angles_grid)
y_coords = center_y + radii_grid * np.sin(angles_grid)

# Single cv2.remap call (GPU-accelerated if available)
polar_image = cv2.remap(image, x_coords, y_coords, cv2.INTER_LINEAR)
```

**Expected improvement:** 10-20ms additional reduction → **~40ms total time**

---

#### 4.3. Reduce Resolution for Initial Pass

**Current:** Full resolution polar transformation
**Optimized:** Downsampled initial pass, full resolution only where needed

```python
# Quick pass with reduced angle count
quick_profile = self._extract_profile_fast(image, detection, num_angles=180)

# Adaptive: full resolution only in regions of interest
if needs_detail:
    full_profile = self._extract_profile_full(image, detection, num_angles=360)
```

**Expected improvement:** 5-10ms reduction for simple cases

---

### Priority 2: Batch Processing Parallelization (Expected: 2-3x speedup for batches)

#### 4.4. Parallel Image Loading

```python
from concurrent.futures import ThreadPoolExecutor

def process_batch_parallel(self, image_paths, sku):
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Load images in parallel
        images = list(executor.map(self.image_loader.load_from_file, image_paths))

    # Process sequentially (CPU-bound, no benefit from threads)
    results = [self._process_single(img, sku) for img in images]
```

**Expected improvement:** Batch processing 10 images: 72ms → **30-40ms**

---

#### 4.5. Memory-Efficient Streaming

```python
def process_batch_streaming(self, image_paths, sku):
    for img_path in image_paths:
        result = self.process(img_path, sku)
        yield result

        # Release memory immediately
        del result
        gc.collect()
```

**Benefit:** Process unlimited batch sizes without memory issues

---

### Priority 3: Minor Optimizations (Expected: 5-10% speedup)

#### 4.6. In-place Operations

```python
# Before: Creates new array
processed = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# After: Reuse buffer where possible
cv2.cvtColor(image, cv2.COLOR_BGR2LAB, dst=image)
```

---

#### 4.7. Lazy Evaluation

```python
# Only compute visualization data if requested
if save_intermediates or visualize:
    result.image = image
    result.lens_detection = detection
else:
    result.image = None
    result.lens_detection = None
```

---

## 5. Performance Targets

### Current Performance
- Single image: **89.79ms**
- Batch 10 images: **72.80ms** (7.28ms avg)
- Throughput: **137-155 images/sec**
- Memory: **~10MB per image**

### Target Performance (After Optimization)

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Single image | 89.79ms | **<50ms** | **44% faster** |
| Radial profiling | 79.50ms | **<30ms** | **62% faster** |
| Batch 10 images | 72.80ms | **<40ms** | **45% faster** |
| Throughput | 137 img/s | **250+ img/s** | **82% faster** |
| Memory | ~10MB | **<8MB** | **20% less** |

---

## 6. Implementation Plan

### Phase 2 Tasks (60 minutes)

1. **Radial Profiling Optimization** (40 min)
   - Implement polar coordinate caching
   - Vectorize NumPy operations
   - Replace loops with cv2.remap
   - Test with VIS_TEST images

2. **Batch Processing Optimization** (15 min)
   - Add ThreadPoolExecutor for image loading
   - Implement streaming for large batches
   - Memory management improvements

3. **Performance Testing** (5 min)
   - Create `tests/test_performance.py`
   - Regression tests to prevent slowdowns
   - Benchmark suite for CI/CD

---

## 7. Risk Assessment

### Low Risk
- Caching polar transforms (no algorithm change)
- Parallel image loading (I/O only)
- Memory management (existing patterns)

### Medium Risk
- Vectorization changes (requires careful testing)
- cv2.remap usage (different interpolation)
- In-place operations (potential side effects)

### Mitigation
- Comprehensive unit tests
- Visual inspection of results
- A/B testing with existing SKUs
- Rollback plan if quality degrades

---

## 8. Expected Outcomes

After implementing Priority 1 & 2 optimizations:

1. **50-60% faster single image processing**
   - 89.79ms → ~40-50ms
   - Exceeds <100ms target by significant margin

2. **2-3x faster batch processing**
   - Better resource utilization
   - Higher throughput for production

3. **No quality degradation**
   - Same algorithm, better implementation
   - Extensive testing to verify

4. **Better scalability**
   - Memory-efficient streaming
   - Ready for production deployment

---

## 9. Conclusion

Radial Profiling의 최적화를 통해 **전체 처리 시간을 50% 이상 단축**할 수 있으며, 이는 프로덕션 환경에서의 처리량을 두 배 이상 향상시킬 것입니다.

캐싱, 벡터화, 병렬 처리 기법을 적용하면 현재 **137 images/sec → 250+ images/sec** 수준의 성능 달성이 가능할 것으로 예상됩니다.

---

**Report Generated:** 2025-12-11
**Tool:** `tools/profiler.py`
**Next Step:** Phase 2 - Performance Optimization Implementation
