"""
Quick Validation Tests for Engine Stabilization

Validates:
- P0: Config Normalization
- P1: Security fixes
- Legacy: Deprecation warnings
- Direction A: Mask compositor basic functionality
"""

import sys

sys.path.insert(0, r"c:\X\Color_meter")

import warnings

import numpy as np

print("=" * 60)
print("Color Meter v7 - Quick Validation Tests")
print("=" * 60)

# Test 1: Config Normalization (P0)
print("\n[1/4] Testing Config Normalization...")
try:
    from src.engine_v7.core.config_norm import get_plate_cfg, get_polar_dims

    # Legacy config (flat)
    cfg_old = {"polar_R": 260, "polar_T": 720}
    R, T = get_polar_dims(cfg_old)
    assert R == 260 and T == 720, f"Flat config failed: got R={R}, T={T}"

    # New config (nested)
    cfg_new = {"polar": {"R": 260, "T": 720}}
    R, T = get_polar_dims(cfg_new)
    assert R == 260 and T == 720, f"Nested config failed: got R={R}, T={T}"

    # Default fallback
    cfg_empty = {}
    R, T = get_polar_dims(cfg_empty)
    assert R == 260 and T == 720, f"Default fallback failed: got R={R}, T={T}"

    print("  ✅ Config normalization: ALL PASSED")
    print(f"    - Flat config: R={R}, T={T}")
    print(f"    - Nested config: R={R}, T={T}")
    print(f"    - Default fallback: R={R}, T={T}")
except Exception as e:
    print(f"  ❌ Config normalization FAILED: {e}")

# Test 2: Security (sanitize_filename)
print("\n[2/4] Testing Security Fixes...")
try:
    from src.engine_v7.core.utils import sanitize_filename

    test_cases = [
        ("../../../etc/passwd", "etc_passwd"),
        ("test<>file.jpg", "test__file.jpg"),
        ("normal_file.png", "normal_file.png"),
        ("", "unnamed_file"),
    ]

    all_passed = True
    for inp, expected in test_cases:
        result = sanitize_filename(inp)
        if result != expected:
            print(f"  ❌ '{inp}' → '{result}' (expected '{expected}')")
            all_passed = False

    if all_passed:
        print("  ✅ Security (filename sanitization): ALL PASSED")
        for inp, expected in test_cases:
            print(f"    - '{inp}' → '{sanitize_filename(inp)}'")
except Exception as e:
    print(f"  ❌ Security tests FAILED: {e}")

# Test 3: Legacy Cleanup (deprecation warnings)
print("\n[3/4] Testing Deprecation Warnings...")
try:
    import cv2

    from src.engine_v7.core.utils import bgr_to_lab

    # Create dummy image
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = bgr_to_lab(test_img)

        if len(w) > 0:
            warning_msg = str(w[0].message)
            if "v8.0" in warning_msg:
                print("  ✅ Deprecation warnings: PASSED")
                print(f"    - Warning emitted: Yes")
                print(f"    - Contains 'v8.0': Yes")
                print(f"    - Message: {warning_msg[:80]}...")
            else:
                print(f"  ⚠️  Warning doesn't mention v8.0: {warning_msg}")
        else:
            print("  ❌ No deprecation warning emitted")
except Exception as e:
    print(f"  ❌ Deprecation test FAILED: {e}")

# Test 4: Direction A (Mask Compositor)
print("\n[4/4] Testing Mask Compositor...")
try:
    from src.engine_v7.core.simulation.mask_compositor import composite_from_masks

    # Create dummy data
    lab_map = np.random.rand(720, 260, 3) * 100  # Polar Lab map
    masks = {
        "color_0": np.random.rand(720, 260) > 0.7,
        "color_1": np.random.rand(720, 260) > 0.7,
        "color_2": np.random.rand(720, 260) > 0.8,
    }

    result = composite_from_masks(lab_map, masks, downsample=4)

    # Validate structure
    assert "composite_lab" in result, "Missing composite_lab"
    assert len(result["composite_lab"]) == 3, "composite_lab should have 3 values"
    assert "overlap" in result, "Missing overlap"
    assert "zone_contributions" in result, "Missing zone_contributions"
    assert "n_pixels_sampled" in result, "Missing n_pixels_sampled"

    print("  ✅ Mask Compositor: PASSED")
    print(
        f"    - Composite Lab: [{result['composite_lab'][0]:.1f}, "
        f"{result['composite_lab'][1]:.1f}, {result['composite_lab'][2]:.1f}]"
    )
    print(f"    - Pixels sampled: {result['n_pixels_sampled']}")
    print(f"    - Overlap ratio: {result['overlap']['ratio']:.2%}")
    print(f"    - Confidence: {result['confidence']:.3f}")

except Exception as e:
    print(f"  ❌ Mask compositor FAILED: {e}")

print("\n" + "=" * 60)
print("Validation Complete!")
print("=" * 60)
