"""
Unit tests for radial_signature median_theta feature.

Tests P1 feature: median aggregation for robustness against outliers/moire.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from src.engine_v7.core.signature.radial_signature import (
    AggregationMethod,
    build_radial_signature,
    build_radial_signature_masked,
    to_polar,
)


class TestBuildRadialSignatureMedian:
    """Tests for median_theta in build_radial_signature."""

    def test_median_default(self):
        """Test that median is the default aggregation method."""
        T, R = 360, 200
        # Create simple polar image
        polar_bgr = np.full((T, R, 3), 128, dtype=np.uint8)

        central, p95, meta = build_radial_signature(polar_bgr, r_start=0.1, r_end=0.9)

        assert meta["aggregation"] == "median"

    def test_median_vs_mean_with_outliers(self):
        """Test that median is more robust to outliers than mean."""
        T, R = 360, 200
        # Create polar image with outliers
        polar_bgr = np.full((T, R, 3), 128, dtype=np.uint8)
        # Add outliers at first 10 rows (extreme values)
        polar_bgr[:10, :, :] = 255

        # Median should be robust
        median_curve, _, meta_median = build_radial_signature(polar_bgr, r_start=0.1, r_end=0.9, aggregation="median")

        # Mean will be affected
        mean_curve, _, meta_mean = build_radial_signature(polar_bgr, r_start=0.1, r_end=0.9, aggregation="mean")

        # Median should be ~50 (Lab L from 128 BGR)
        # Mean will be higher due to outliers
        assert np.mean(mean_curve[:, 0]) > np.mean(median_curve[:, 0])

    def test_mean_aggregation_option(self):
        """Test that mean aggregation can still be used."""
        T, R = 360, 200
        polar_bgr = np.full((T, R, 3), 128, dtype=np.uint8)

        central, p95, meta = build_radial_signature(polar_bgr, r_start=0.1, r_end=0.9, aggregation="mean")

        assert meta["aggregation"] == "mean"

    def test_uniform_image_same_result(self):
        """Test that median and mean give same result for uniform image."""
        T, R = 360, 200
        polar_bgr = np.full((T, R, 3), 128, dtype=np.uint8)

        median_curve, _, _ = build_radial_signature(polar_bgr, r_start=0.1, r_end=0.9, aggregation="median")
        mean_curve, _, _ = build_radial_signature(polar_bgr, r_start=0.1, r_end=0.9, aggregation="mean")

        # Should be identical for uniform image
        np.testing.assert_array_almost_equal(median_curve, mean_curve, decimal=1)


class TestBuildRadialSignatureMaskedMedian:
    """Tests for median_theta in build_radial_signature_masked."""

    def test_masked_median_default(self):
        """Test that masked version also uses median by default."""
        T, R = 360, 200
        polar_bgr = np.full((T, R, 3), 128, dtype=np.uint8)
        mask = np.ones((T, R), dtype=bool)

        central, p95, meta = build_radial_signature_masked(polar_bgr, mask, r_start=0.1, r_end=0.9)

        assert meta["aggregation"] == "median"

    def test_masked_with_sparse_mask(self):
        """Test masked signature with sparse mask."""
        T, R = 360, 200
        polar_bgr = np.full((T, R, 3), 128, dtype=np.uint8)
        # Sparse mask - only every 10th row
        mask = np.zeros((T, R), dtype=bool)
        mask[::10, :] = True

        central, p95, meta = build_radial_signature_masked(polar_bgr, mask, r_start=0.1, r_end=0.9)

        # Should still produce valid output
        assert not np.any(np.isnan(central))
        assert meta["valid_bin_count"] > 0

    def test_masked_min_samples_gate(self):
        """Test minimum samples per bin gate."""
        T, R = 360, 200
        polar_bgr = np.full((T, R, 3), 128, dtype=np.uint8)
        # Very sparse mask - only 2 rows
        mask = np.zeros((T, R), dtype=bool)
        mask[0, :] = True
        mask[180, :] = True

        central, p95, meta = build_radial_signature_masked(
            polar_bgr, mask, r_start=0.1, r_end=0.9, min_samples_per_bin=50  # Higher than available
        )

        # Quality should be low
        assert meta["quality"] < 0.5

    def test_masked_quality_metric(self):
        """Test quality metric in masked signature."""
        T, R = 360, 200
        polar_bgr = np.full((T, R, 3), 128, dtype=np.uint8)
        mask = np.ones((T, R), dtype=bool)

        central, p95, meta = build_radial_signature_masked(polar_bgr, mask, r_start=0.1, r_end=0.9)

        # With full mask, quality should be high
        assert meta["quality"] > 0.9
        assert "sample_counts" in meta
        assert meta["valid_bin_count"] == meta["total_bins"]

    def test_masked_outlier_robustness(self):
        """Test that masked median is robust to outliers."""
        T, R = 360, 200
        polar_bgr = np.full((T, R, 3), 128, dtype=np.uint8)
        # Add outliers
        polar_bgr[:5, :, :] = 255

        mask = np.ones((T, R), dtype=bool)

        median_curve, _, _ = build_radial_signature_masked(
            polar_bgr, mask, r_start=0.1, r_end=0.9, aggregation="median"
        )
        mean_curve, _, _ = build_radial_signature_masked(polar_bgr, mask, r_start=0.1, r_end=0.9, aggregation="mean")

        # Median should be less affected by outliers
        assert np.mean(mean_curve[:, 0]) > np.mean(median_curve[:, 0])


class TestLegacyCompatibility:
    """Tests for backward compatibility."""

    def test_legacy_function_exists(self):
        """Test that legacy function exists for backward compat."""
        from src.engine_v7.core.signature.radial_signature import build_radial_signature_masked_legacy

        T, R = 360, 200
        polar_bgr = np.full((T, R, 3), 128, dtype=np.uint8)
        mask = np.ones((T, R), dtype=bool)

        # Should work and use mean
        central, p95, meta = build_radial_signature_masked_legacy(polar_bgr, mask, r_start=0.1, r_end=0.9)

        assert meta["aggregation"] == "mean"


class TestAggregationMethod:
    """Tests for AggregationMethod enum."""

    def test_enum_values(self):
        """Test enum has correct values."""
        assert AggregationMethod.MEAN.value == "mean"
        assert AggregationMethod.MEDIAN.value == "median"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
