# Lab Color Space Scale Fix

**Issue Date**: 2025-12-12
**Severity**: CRITICAL
**Status**: ✅ FIXED

---

## 🚨 Problem Summary

The system was mixing **OpenCV Lab scale (0~255)** and **Standard CIE Lab scale (L*: 0~100, a*: -128~127, b*: -128~127)**, causing:
- **Massive color difference (ΔE) calculation errors** (51.47 instead of 17.90)
- **Invalid L* values exceeding 100** (L*=182 instead of 71.6)
- **All NG judgments incorrect** due to wrong scale comparison

---

## 📊 User's Reported Data

From inspection result JSON:
```json
{
  "measured_lab": [182.56, 127.55, 137.50],  // OpenCV scale!
  "target_lab": [72.2, 137.3, 122.8],        // Mixed scale!
  "delta_e": 51.47,                           // WRONG!
  "judgment": "NG"
}
```

**Ring 0 (center) even worse:**
```json
{
  "mean_L": 253.97,  // Should be ~99.6 (almost white)
  "mean_a": 127.99,  // Should be ~0.0
  "mean_b": 128.99   // Should be ~1.0
}
```

---

## 🔍 Root Cause Analysis

### OpenCV Lab Color Space

OpenCV `cv2.cvtColor(img, cv2.COLOR_BGR2Lab)` returns:
- **L channel: 0~255** (scaled from standard 0~100)
- **a channel: 0~255** (offset from standard -128~127)
- **b channel: 0~255** (offset from standard -128~127)

**Conversion formula:**
```python
L_standard = L_opencv * (100.0 / 255.0)
a_standard = a_opencv - 128.0
b_standard = b_opencv - 128.0
```

### What Went Wrong

1. **`radial_profiler.py:52`** - Extracted OpenCV Lab, used directly without conversion
2. **`zone_segmenter.py:105`** - Calculated mean from OpenCV Lab values
3. **`angular_profiler.py:224`** - Extracted OpenCV Lab statistics
4. **`config/sku_db/SKU001.json`** - Mixed scale:
   - L*: Standard scale (72.2)
   - a*, b*: OpenCV scale (137.3, 122.8)

5. **Color difference calculation** - Compared mismatched scales:
   ```python
   # WRONG!
   ΔE = delta_e(
       (182.56, 127.55, 137.50),  # OpenCV
       (72.2, 137.3, 122.8)        # Mixed
   ) = 51.47
   ```

---

## ✅ Solution Implemented

### 1. Created Lab Conversion Utility (`src/utils/color_space.py`)

```python
def opencv_lab_to_standard(L_cv, a_cv, b_cv):
    """Convert OpenCV Lab (0~255) to Standard Lab"""
    L_std = L_cv * (100.0 / 255.0)
    a_std = a_cv - 128.0
    b_std = b_cv - 128.0
    return L_std, a_std, b_std
```

### 2. Updated All Lab Extraction Points

#### `src/core/radial_profiler.py:53-74`
```python
polar_lab = cv2.cvtColor(polar_image, cv2.COLOR_BGR2LAB)

# OpenCV Lab → Standard Lab 변환
from src.utils.color_space import opencv_lab_to_standard
L_std, a_std, b_std = opencv_lab_to_standard(
    polar_lab[:, :, 0],
    polar_lab[:, :, 1],
    polar_lab[:, :, 2]
)

# Now calculate statistics in standard scale
L_profile = L_std.mean(axis=1)
a_profile = a_std.mean(axis=1)
b_profile = b_std.mean(axis=1)
```

#### `src/core/angular_profiler.py:223-231`
```python
# LAB 통계 계산 - OpenCV Lab → Standard Lab 변환
from src.utils.color_space import opencv_lab_to_standard

L_vals_cv = image_lab[cell_mask, 0]
a_vals_cv = image_lab[cell_mask, 1]
b_vals_cv = image_lab[cell_mask, 2]

L_vals, a_vals, b_vals = opencv_lab_to_standard(L_vals_cv, a_vals_cv, b_vals_cv)

cell = RingSectorCell(
    mean_L=float(np.mean(L_vals)),  # Now in standard scale
    mean_a=float(np.mean(a_vals)),
    mean_b=float(np.mean(b_vals)),
    ...
)
```

### 3. Updated SKU Configuration (`config/sku_db/SKU001.json`)

**Before (mixed scale):**
```json
{
  "L": 72.2,   // Standard
  "a": 137.3,  // OpenCV
  "b": 122.8   // OpenCV
}
```

**After (all standard):**
```json
{
  "L": 72.2,
  "a": 9.3,    // 137.3 - 128 = 9.3
  "b": -5.2    // 122.8 - 128 = -5.2
}
```

---

## 📈 Impact: Before vs After

### Color Difference Calculation

**User's sample data:**
- Measured: L*=182.56, a*=127.55, b*=137.50 (OpenCV)
- Target: L*=72.2, a*=137.3, b*=122.8 (Mixed)

| Metric | Before (Wrong) | After (Correct) | Improvement |
|--------|----------------|-----------------|-------------|
| **Measured L*** | 182.56 ❌ | 71.6 ✅ | In range (0~100) |
| **Measured a*** | 127.55 ❌ | -0.5 ✅ | In range (-128~127) |
| **Measured b*** | 137.50 ❌ | 9.5 ✅ | In range (-128~127) |
| **Target a*** | 137.3 ❌ | 9.3 ✅ | Converted to standard |
| **Target b*** | 122.8 ❌ | -5.2 ✅ | Converted to standard |
| **ΔE (Zone A)** | 51.47 ❌ | **17.90** ✅ | **-65% error** |

### Test Results

```bash
$ pytest tests/test_color_space.py -v -s

Delta E (wrong): 51.47
Delta E (correct): 17.90
Measured (std): L*=71.6, a*=-0.5, b*=9.5
Target (std): L*=72.2, a*=9.3, b*=-5.2
```

**All 12 tests PASSED** ✅

---

## 🎯 Actual Color Difference Analysis

After fixing the scale issue, **the sample is still NG**, but for the **correct reason**:

### True Color Differences
```
L* difference: 71.6 - 72.2 = -0.6 (negligible)
a* difference: -0.5 - 9.3 = -9.8 (significant!)
b* difference: 9.5 - (-5.2) = 14.7 (significant!)
```

**ΔE = 17.90** indicates:
- **L* (brightness)**: Nearly perfect match
- **a* (green ↔ red)**: 9.8 units difference (noticeable)
- **b* (blue ↔ yellow)**: 14.7 units difference (very noticeable)

**Judgment**: Sample is **legitimately NG** due to color shift in a*/b* chromaticity.

---

## 📝 Files Modified

### New Files
1. **`src/utils/color_space.py`** (108 lines)
   - `opencv_lab_to_standard()`: OpenCV → Standard conversion
   - `standard_lab_to_opencv()`: Standard → OpenCV conversion
   - `validate_standard_lab()`: Validation utility
   - `detect_lab_scale()`: Auto-detect scale type

2. **`tests/test_color_space.py`** (187 lines)
   - 12 comprehensive tests
   - Delta E before/after comparison
   - Array conversion tests

3. **`docs/LAB_SCALE_FIX.md`** (this document)

### Modified Files
1. **`src/core/radial_profiler.py`** - Lines 53-74
   - Added OpenCV → Standard conversion after `cv2.cvtColor`
   - All profile statistics now in standard scale

2. **`src/core/angular_profiler.py`** - Lines 223-231
   - Added OpenCV → Standard conversion for Ring×Sector cells
   - All cell statistics now in standard scale

3. **`config/sku_db/SKU001.json`** - All zone targets
   - Converted a*, b* from OpenCV (0~255) to Standard (-128~127)
   - Zone A: a: 137.3→9.3, b: 122.8→-5.2
   - Zone B: a: 135.0→7.0, b: 125.0→-3.0
   - Zone C: a: 132.0→4.0, b: 128.0→0.0
   - Zone D: a: 130.0→2.0, b: 130.0→2.0

---

## 🧪 Verification

Run all tests:
```bash
# Color space conversion tests
pytest tests/test_color_space.py -v -s

# Full test suite (ensure no regressions)
pytest tests/test_radial_profiler.py -v
pytest tests/test_angular_profiler.py -v
pytest tests/test_color_evaluator.py -v
```

Expected results after fix:
- ✅ All Lab values in range: L* (0~100), a* (-128~127), b* (-128~127)
- ✅ Delta E calculations reasonable (<30, not >50)
- ✅ OK/NG judgments accurate
- ✅ Ring 0 center L* ≈ 99.6 (white background, not 254)

---

## 🔄 Migration Guide

### For Existing SKU Configurations

If you have existing SKU configurations with OpenCV scale a*, b* values:

```python
# Conversion script
from src.utils.color_space import opencv_lab_to_standard

# For each zone in SKU config:
_, a_std, b_std = opencv_lab_to_standard(0, a_opencv, b_opencv)
# Update config with a_std, b_std
```

**Example:**
```python
# Old SKU002.json (OpenCV scale)
"a": 145.0, "b": 118.0

# Convert
_, a_std, b_std = opencv_lab_to_standard(0, 145.0, 118.0)
# a_std = 17.0, b_std = -10.0

# New SKU002.json (Standard scale)
"a": 17.0, "b": -10.0
```

### For Baseline Generation

The `sku_manager.py:generate_baseline()` function already uses the fixed
`radial_profiler` and `zone_segmenter`, so newly generated baselines will
automatically be in standard Lab scale.

**No manual conversion needed** for new baselines.

---

## 📚 References

- **OpenCV Lab Color Space**: https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html
- **CIE L\*a\*b\* Color Space**: https://en.wikipedia.org/wiki/CIELAB_color_space
- **CIEDE2000 Formula**: Used for all ΔE calculations in `src/utils/color_delta.py`

---

## ✅ Checklist for Production

Before deploying this fix:

- [x] Create Lab conversion utility
- [x] Update radial_profiler
- [x] Update angular_profiler
- [x] Update zone_segmenter (inherits from radial_profiler, no change needed)
- [x] Update SKU001.json
- [x] Write comprehensive tests
- [x] Verify delta E calculations
- [ ] **TODO**: Update all existing SKU configs (SKU002+)
- [ ] **TODO**: Re-run baseline generation for production SKUs
- [ ] **TODO**: Verify web UI displays correct values
- [ ] **TODO**: Update user documentation with Lab scale info

---

**Issue Resolution**: This was a **critical scaling bug** that affected all color measurements. Now fixed with proper OpenCV → Standard Lab conversion throughout the pipeline.

**Impact**: All previously recorded NG judgments with ΔE > 40 should be re-evaluated with the corrected scale.
