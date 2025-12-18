"""
Color Space Conversion Tests

Tests for OpenCV Lab ↔ Standard Lab conversions.
"""

import numpy as np
import pytest

from src.utils.color_space import (
    detect_lab_scale,
    opencv_lab_to_standard,
    standard_lab_to_opencv,
    validate_standard_lab,
)


class TestLabConversion:
    """Test Lab color space conversions"""

    def test_opencv_to_standard_white(self):
        """White color: OpenCV (255, 128, 128) → Standard (100, 0, 0)"""
        L_std, a_std, b_std = opencv_lab_to_standard(255, 128, 128)

        assert abs(L_std - 100.0) < 0.01
        assert abs(a_std - 0.0) < 0.01
        assert abs(b_std - 0.0) < 0.01

    def test_opencv_to_standard_black(self):
        """Black color: OpenCV (0, 128, 128) → Standard (0, 0, 0)"""
        L_std, a_std, b_std = opencv_lab_to_standard(0, 128, 128)

        assert abs(L_std - 0.0) < 0.01
        assert abs(a_std - 0.0) < 0.01
        assert abs(b_std - 0.0) < 0.01

    def test_opencv_to_standard_measured_values(self):
        """Real measured values from JSON"""
        # From user's JSON: measured_lab = [182.56, 127.55, 137.50]
        L_std, a_std, b_std = opencv_lab_to_standard(182.56, 127.55, 137.50)

        # Expected:
        # L = 182.56 * (100/255) = 71.59
        # a = 127.55 - 128 = -0.45
        # b = 137.50 - 128 = 9.50

        assert abs(L_std - 71.59) < 0.1
        assert abs(a_std - (-0.45)) < 0.1
        assert abs(b_std - 9.50) < 0.1

    def test_opencv_to_standard_array(self):
        """Test with numpy arrays"""
        L_cv = np.array([0, 128, 255])
        a_cv = np.array([128, 128, 128])
        b_cv = np.array([128, 128, 128])

        L_std, a_std, b_std = opencv_lab_to_standard(L_cv, a_cv, b_cv)

        expected_L = np.array([0, 50.2, 100.0])
        expected_a = np.array([0, 0, 0])
        expected_b = np.array([0, 0, 0])

        np.testing.assert_allclose(L_std, expected_L, rtol=0.01)
        np.testing.assert_allclose(a_std, expected_a, rtol=0.01)
        np.testing.assert_allclose(b_std, expected_b, rtol=0.01)

    def test_standard_to_opencv_white(self):
        """White color: Standard (100, 0, 0) → OpenCV (255, 128, 128)"""
        L_cv, a_cv, b_cv = standard_lab_to_opencv(100, 0, 0)

        assert abs(L_cv - 255.0) < 0.01
        assert abs(a_cv - 128.0) < 0.01
        assert abs(b_cv - 128.0) < 0.01

    def test_roundtrip_conversion(self):
        """Test OpenCV → Standard → OpenCV roundtrip"""
        L_cv_orig = 182.5
        a_cv_orig = 127.5
        b_cv_orig = 137.5

        # OpenCV → Standard
        L_std, a_std, b_std = opencv_lab_to_standard(L_cv_orig, a_cv_orig, b_cv_orig)

        # Standard → OpenCV
        L_cv, a_cv, b_cv = standard_lab_to_opencv(L_std, a_std, b_std)

        # Should match original
        assert abs(L_cv - L_cv_orig) < 0.01
        assert abs(a_cv - a_cv_orig) < 0.01
        assert abs(b_cv - b_cv_orig) < 0.01

    def test_validate_standard_lab_valid(self):
        """Valid standard Lab values should pass"""
        assert validate_standard_lab(72.2, 9.3, -5.2) is True
        assert validate_standard_lab(50.0, 0.0, 0.0) is True
        assert validate_standard_lab(100.0, 127.0, -128.0) is True

    def test_validate_standard_lab_invalid(self):
        """Invalid standard Lab values should fail"""
        # L out of range
        assert validate_standard_lab(182.5, 0, 0) is False
        assert validate_standard_lab(-10, 0, 0) is False

        # a, b out of range
        assert validate_standard_lab(50, 200, 0) is False
        assert validate_standard_lab(50, 0, -200) is False

    def test_detect_lab_scale_opencv(self):
        """Detect OpenCV scale values"""
        assert detect_lab_scale(182.5, 127.5, 137.5) == "opencv"
        assert detect_lab_scale(253.97, 127.99, 128.99) == "opencv"

    def test_detect_lab_scale_standard(self):
        """Detect Standard scale values"""
        assert detect_lab_scale(72.2, 9.3, -5.2) == "standard"
        assert detect_lab_scale(50.0, 0.0, 0.0) == "standard"

    def test_sku_target_conversion(self):
        """Test SKU target value conversion (from user's issue)"""
        # Original SKU001.json values (OpenCV scale for a*, b*)
        L_sku = 72.2  # Already standard
        a_sku_cv = 137.3  # OpenCV scale
        b_sku_cv = 122.8  # OpenCV scale

        # Convert a*, b* to standard
        _, a_sku_std, b_sku_std = opencv_lab_to_standard(0, a_sku_cv, b_sku_cv)

        # Expected: a=137.3-128=9.3, b=122.8-128=-5.2
        assert abs(a_sku_std - 9.3) < 0.1
        assert abs(b_sku_std - (-5.2)) < 0.1

        # Now L, a, b are all in standard scale
        assert validate_standard_lab(L_sku, a_sku_std, b_sku_std) is True


class TestDeltaEConsistency:
    """Test that delta E calculations are consistent after conversion"""

    def test_delta_e_calculation_before_after(self):
        """Compare delta E before and after Lab scale fix"""
        from src.utils.color_delta import delta_e_cie2000

        # User's measured values (OpenCV scale)
        measured_L_cv = 182.56
        measured_a_cv = 127.55
        measured_b_cv = 137.50

        # User's target values (mixed scale: L=standard, a/b=OpenCV)
        target_L = 72.2
        target_a_cv = 137.3
        target_b_cv = 122.8

        # WRONG calculation (mixing scales)
        de_wrong = delta_e_cie2000((measured_L_cv, measured_a_cv, measured_b_cv), (target_L, target_a_cv, target_b_cv))

        # CORRECT calculation (both in standard scale)
        measured_L_std, measured_a_std, measured_b_std = opencv_lab_to_standard(
            measured_L_cv, measured_a_cv, measured_b_cv
        )
        _, target_a_std, target_b_std = opencv_lab_to_standard(0, target_a_cv, target_b_cv)

        de_correct = delta_e_cie2000(
            (measured_L_std, measured_a_std, measured_b_std), (target_L, target_a_std, target_b_std)
        )

        # Wrong delta E should be huge (51.47 in user's case)
        assert de_wrong > 40  # Way too large

        # Correct delta E should be reasonable (~18)
        # Note: ΔE=17.9 indicates actual color difference exists
        assert 15 < de_correct < 25  # Much more reasonable than 51

        print(f"Delta E (wrong): {de_wrong:.2f}")
        print(f"Delta E (correct): {de_correct:.2f}")
        print(f"Measured (std): L*={measured_L_std:.1f}, a*={measured_a_std:.1f}, b*={measured_b_std:.1f}")
        print(f"Target (std): L*={target_L:.1f}, a*={target_a_std:.1f}, b*={target_b_std:.1f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
