# Plate-Lite Calibration Guide

> **Version**: 1.0
> **Last Updated**: 2026-01-19

---

## Overview

Plate-Lite uses a physics-based ink color restoration formula that requires accurate `paper_color` (background color) to extract pure ink colors. This guide covers calibration procedures and troubleshooting.

---

## Paper Color Sources

| Source | Description | When to Use |
|--------|-------------|-------------|
| `static` | Fixed value from config | Default, stable environments |
| `auto` | Extracted from clear zone | Dynamic lighting conditions |
| `calibration` | Pre-measured reference | High-precision requirements |

### Configuration

```json
{
  "plate_lite": {
    "paper_color": {
      "lab": [95.0, 0.0, 0.0],
      "source": "static"
    }
  }
}
```

---

## Calibration Procedure

### Method 1: Static Calibration (Recommended)

1. **Capture Reference Image**
   - Place white background only (no lens)
   - Use same lighting as production
   - Capture image

2. **Measure Lab Values**
   ```python
   import cv2
   import numpy as np
   from src.engine_v7.core.utils import to_cie_lab

   img = cv2.imread("white_background.png")
   lab = to_cie_lab(img)

   # Center region average
   h, w = lab.shape[:2]
   roi = lab[h//4:3*h//4, w//4:3*w//4]
   paper_lab = roi.mean(axis=(0, 1))
   print(f"Paper Lab: {paper_lab.tolist()}")
   ```

3. **Update Configuration**
   ```json
   {
     "plate_lite": {
       "paper_color": {
         "lab": [94.2, 0.5, -0.3],
         "source": "calibration"
       }
     }
   }
   ```

### Method 2: Auto Mode

Set `source: "auto"` to extract paper color from the clear zone automatically.

**Requirements**:
- Clear zone must exist in the lens image
- Clear zone should not be contaminated (dust, reflections)

**Fallback Behavior**:
- If clear mask is empty/invalid, falls back to static value
- Warning `paper_color_auto_fallback_static` is logged

---

## Warning Messages

| Warning | Cause | Action |
|---------|-------|--------|
| `possible_white_black_swap` | White image darker than black | Check image order |
| `alpha_mean_too_low` | Average alpha < 0.1 | Very thin ink or image issue |
| `alpha_mean_too_high` | Average alpha > 0.9 | Very thick ink or no lens |
| `paper_color_auto_fallback_static` | Auto mode failed | Check clear zone mask |
| `paper_color_calibration_missing_fallback_static` | Calibration value missing | Set `paper_color.lab` |

---

## Troubleshooting

### Issue: Ink colors appear washed out

**Cause**: `paper_color` is too dark (L* too low)

**Solution**:
- Measure actual background L* value
- Typical white paper: L* = 92-96

### Issue: Ink colors appear oversaturated

**Cause**: `paper_color` is too bright (L* too high) or alpha calculation error

**Solution**:
- Verify white/black image order
- Check for `possible_white_black_swap` warning

### Issue: Inconsistent results between scans

**Cause**: Lighting variation affecting background

**Solution**:
- Use `source: "calibration"` with fixed reference
- Or use `source: "auto"` with stable clear zone

---

## Recommended Calibration Schedule

| Trigger | Action |
|---------|--------|
| Initial setup | Full calibration |
| Lighting change | Re-calibrate |
| Monthly | Verify calibration |
| Inconsistent results | Re-calibrate |

---

## Quick Reference

### Default Values

```json
{
  "plate_lite": {
    "enabled": true,
    "override_plate": true,
    "blur_ksize": 5,
    "backlight": 255.0,
    "alpha_threshold": 0.1,
    "paper_color": {
      "lab": [95.0, 0.0, 0.0],
      "source": "static"
    }
  }
}
```

### Typical Paper Lab Values

| Paper Type | L* | a* | b* |
|------------|-----|-----|-----|
| White office paper | 94-96 | -1 to 1 | -2 to 2 |
| Coated white | 96-98 | 0 | 0 |
| Off-white/cream | 90-93 | 0-2 | 2-5 |

---

## Related Documents

- [PLATE_LITE_COLOR_EXTRACTION_PLAN.md](../planning/PLATE_LITE_COLOR_EXTRACTION_PLAN.md)
- [PLATE_LITE_AB_RESULT.md](../planning/PLATE_LITE_AB_RESULT.md)
