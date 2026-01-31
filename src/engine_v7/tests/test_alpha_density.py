"""
Unit tests for alpha_density module.

Tests the P0 features:
1. compute_alpha_radial_1d - median_theta aggregation
2. 3-tier alpha fallback (L1/L2/L3)
3. compute_effective_density - full integration
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from src.engine_v7.core.measure.metrics.alpha_density import (  # Main API; Core functions; Types; Utilities
    DEFAULT_ALPHA_CONFIG,
    AlphaDensityResult,
    AlphaFallbackLevel,
    AlphaRadialProfile,
    AlphaZoneProfile,
    ClusterAlphaResult,
    apply_alpha_fallback,
    compute_alpha_global,
    compute_alpha_radial_1d,
    compute_alpha_zone,
    compute_effective_density,
    extract_alpha_summary,
    extract_effective_densities,
)


class TestComputeAlphaRadial1D:
    """Tests for compute_alpha_radial_1d function."""

    def test_basic_radial_profile(self):
        """Test basic radial profile computation."""
        T, R = 360, 200
        # Create synthetic alpha map with radial gradient
        r_coords = np.arange(R) / (R - 1)
        alpha_map = np.tile(r_coords, (T, 1)).astype(np.float32)
        mask = np.ones((T, R), dtype=bool)

        profile = compute_alpha_radial_1d(
            alpha_map,
            mask,
            n_bins=10,
            r_start=0.1,
            r_end=0.9,
            min_samples_per_bin=10,
        )

        assert isinstance(profile, AlphaRadialProfile)
        assert len(profile.r_bins) == 10
        assert len(profile.alpha_values) == 10
        assert profile.valid_bins > 0
        assert profile.quality > 0.5

    def test_median_theta_aggregation(self):
        """Test that median_theta is used (robust to outliers)."""
        T, R = 360, 200
        # Create alpha map with outliers at specific angles
        alpha_map = np.full((T, R), 0.5, dtype=np.float32)
        # Add outliers at first 10 rows (angles)
        alpha_map[:10, :] = 0.99

        mask = np.ones((T, R), dtype=bool)

        profile = compute_alpha_radial_1d(
            alpha_map,
            mask,
            n_bins=10,
            min_samples_per_bin=10,
        )

        # Median should be ~0.5, not affected by outliers
        valid_values = profile.alpha_values[~np.isnan(profile.alpha_values)]
        assert np.all(valid_values < 0.6), "Median should be robust to outliers"

    def test_sparse_mask_fallback(self):
        """Test behavior with sparse mask (few valid pixels)."""
        T, R = 360, 200
        alpha_map = np.full((T, R), 0.5, dtype=np.float32)
        # Very sparse mask - only 2 pixels per bin (rows 0 and 180)
        mask = np.zeros((T, R), dtype=bool)
        mask[0, :] = True
        mask[180, :] = True  # Only 2 rows total

        profile = compute_alpha_radial_1d(
            alpha_map,
            mask,
            n_bins=10,
            min_samples_per_bin=50,  # Higher than available (2 rows * ~16 cols per bin = ~32)
        )

        # Quality should be low due to insufficient samples
        assert profile.quality < 0.5
        assert profile.valid_bins < profile.total_bins

    def test_empty_mask(self):
        """Test with completely empty mask."""
        T, R = 360, 200
        alpha_map = np.full((T, R), 0.5, dtype=np.float32)
        mask = np.zeros((T, R), dtype=bool)

        profile = compute_alpha_radial_1d(alpha_map, mask, n_bins=10)

        assert profile.valid_bins == 0
        assert profile.quality == 0.0


class TestComputeAlphaZone:
    """Tests for compute_alpha_zone function."""

    def test_zone_computation(self):
        """Test zone-based alpha computation."""
        T, R = 360, 200
        # Create alpha map with different values per zone
        alpha_map = np.zeros((T, R), dtype=np.float32)
        inner_end = int(R * 0.4)
        mid_end = int(R * 0.7)

        alpha_map[:, :inner_end] = 0.3  # Inner
        alpha_map[:, inner_end:mid_end] = 0.5  # Mid
        alpha_map[:, mid_end:] = 0.7  # Outer

        mask = np.ones((T, R), dtype=bool)

        zone_profile = compute_alpha_zone(
            alpha_map,
            mask,
            inner_end=0.4,
            mid_end=0.7,
        )

        assert isinstance(zone_profile, AlphaZoneProfile)
        assert abs(zone_profile.inner - 0.3) < 0.1
        assert abs(zone_profile.mid - 0.5) < 0.1
        assert abs(zone_profile.outer - 0.7) < 0.1

    def test_zone_confidence(self):
        """Test zone confidence calculation."""
        T, R = 360, 200
        alpha_map = np.full((T, R), 0.5, dtype=np.float32)
        mask = np.ones((T, R), dtype=bool)

        zone_profile = compute_alpha_zone(alpha_map, mask, min_samples=50)

        # With full mask, all zones should have high confidence
        assert zone_profile.inner_conf > 0.5
        assert zone_profile.mid_conf > 0.5
        assert zone_profile.outer_conf > 0.5


class TestComputeAlphaGlobal:
    """Tests for compute_alpha_global function."""

    def test_global_alpha(self):
        """Test global alpha computation."""
        T, R = 360, 200
        alpha_map = np.full((T, R), 0.6, dtype=np.float32)
        mask = np.ones((T, R), dtype=bool)

        alpha, std, n = compute_alpha_global(alpha_map, mask)

        assert abs(alpha - 0.6) < 0.01
        assert std < 0.01
        assert n == T * R

    def test_global_alpha_empty_mask(self):
        """Test global alpha with empty mask."""
        T, R = 360, 200
        alpha_map = np.full((T, R), 0.6, dtype=np.float32)
        mask = np.zeros((T, R), dtype=bool)

        alpha, std, n = compute_alpha_global(alpha_map, mask)

        assert alpha == 1.0  # Default
        assert n == 0


class TestApplyAlphaFallback:
    """Tests for 3-tier fallback system."""

    def test_l1_fallback_used(self):
        """Test that L1 (radial) is used when quality is high."""
        # Create high-quality radial profile
        radial = AlphaRadialProfile(
            r_bins=np.linspace(0.1, 0.9, 10),
            alpha_values=np.full(10, 0.5),
            confidence=np.full(10, 0.8),
            n_samples=np.full(10, 100),
            valid_bins=10,
            total_bins=10,
            quality=1.0,
        )
        zone = AlphaZoneProfile(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        global_alpha = 0.7

        alpha, level, reason = apply_alpha_fallback(
            radial,
            zone,
            global_alpha,
            confidence_threshold_l1=0.6,
            min_valid_bins_ratio=0.5,
        )

        assert level == AlphaFallbackLevel.L1_RADIAL
        assert reason is None
        assert abs(alpha - 0.5) < 0.1

    def test_l2_fallback_triggered(self):
        """Test L2 fallback when L1 quality is low."""
        # Create low-quality radial profile
        radial = AlphaRadialProfile(
            r_bins=np.linspace(0.1, 0.9, 10),
            alpha_values=np.full(10, np.nan),
            confidence=np.zeros(10),
            n_samples=np.zeros(10, dtype=np.int32),
            valid_bins=0,
            total_bins=10,
            quality=0.0,
        )
        zone = AlphaZoneProfile(0.4, 0.5, 0.6, 0.8, 0.8, 0.8)
        global_alpha = 0.7

        alpha, level, reason = apply_alpha_fallback(
            radial,
            zone,
            global_alpha,
            confidence_threshold_l2=0.4,
        )

        assert level == AlphaFallbackLevel.L2_ZONE
        assert "L1_quality" in reason
        assert 0.4 < alpha < 0.6

    def test_l3_fallback_triggered(self):
        """Test L3 fallback when both L1 and L2 fail."""
        # Create empty profiles
        radial = AlphaRadialProfile(
            r_bins=np.linspace(0.1, 0.9, 10),
            alpha_values=np.full(10, np.nan),
            confidence=np.zeros(10),
            n_samples=np.zeros(10, dtype=np.int32),
            valid_bins=0,
            total_bins=10,
            quality=0.0,
        )
        zone = AlphaZoneProfile(1.0, 1.0, 1.0, 0.0, 0.0, 0.0)  # No confidence
        global_alpha = 0.8

        alpha, level, reason = apply_alpha_fallback(
            radial,
            zone,
            global_alpha,
            confidence_threshold_l2=0.4,
        )

        assert level == AlphaFallbackLevel.L3_GLOBAL
        assert "L2_conf" in reason
        assert abs(alpha - 0.8) < 0.1


class TestComputeEffectiveDensity:
    """Tests for main compute_effective_density function."""

    def test_basic_effective_density(self):
        """Test basic effective density computation."""
        T, R = 360, 200
        polar_alpha = np.full((T, R), 0.5, dtype=np.float32)

        # Two clusters with different areas
        mask1 = np.zeros((T, R), dtype=bool)
        mask1[:, :100] = True  # 50% of area

        mask2 = np.zeros((T, R), dtype=bool)
        mask2[:, 100:] = True  # 50% of area

        cluster_masks = {"color_0": mask1, "color_1": mask2}
        area_ratios = {"color_0": 0.5, "color_1": 0.5}

        result = compute_effective_density(polar_alpha, cluster_masks, area_ratios)

        assert isinstance(result, AlphaDensityResult)
        assert "color_0" in result.clusters
        assert "color_1" in result.clusters

        # effective_density = area_ratio * alpha
        # With alpha=0.5 and area=0.5, effective_density should be ~0.25
        for cluster in result.clusters.values():
            assert 0.2 < cluster.effective_density < 0.3

    def test_no_alpha_map(self):
        """Test behavior when alpha map is None."""
        cluster_masks = {
            "color_0": np.ones((360, 200), dtype=bool),
        }
        area_ratios = {"color_0": 0.8}

        result = compute_effective_density(None, cluster_masks, area_ratios)

        assert "ALPHA_MAP_UNAVAILABLE" in result.warnings
        # Without alpha, effective_density = area_ratio
        assert result.clusters["color_0"].effective_density == 0.8
        assert result.clusters["color_0"].alpha_used == 1.0
        assert result.clusters["color_0"].fallback_level == AlphaFallbackLevel.L3_GLOBAL

    def test_different_cluster_alphas(self):
        """Test clusters with different alpha values."""
        T, R = 360, 200
        polar_alpha = np.zeros((T, R), dtype=np.float32)
        polar_alpha[:, :100] = 0.3  # Low alpha region
        polar_alpha[:, 100:] = 0.8  # High alpha region

        mask1 = np.zeros((T, R), dtype=bool)
        mask1[:, :100] = True

        mask2 = np.zeros((T, R), dtype=bool)
        mask2[:, 100:] = True

        cluster_masks = {"ink_0": mask1, "ink_1": mask2}
        area_ratios = {"ink_0": 0.5, "ink_1": 0.5}

        result = compute_effective_density(polar_alpha, cluster_masks, area_ratios)

        # ink_0 should have lower effective density (low alpha)
        # ink_1 should have higher effective density (high alpha)
        assert result.clusters["ink_0"].effective_density < result.clusters["ink_1"].effective_density


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_extract_effective_densities(self):
        """Test extract_effective_densities utility."""
        T, R = 360, 200
        polar_alpha = np.full((T, R), 0.6, dtype=np.float32)
        cluster_masks = {"c0": np.ones((T, R), dtype=bool)}
        area_ratios = {"c0": 0.7}

        result = compute_effective_density(polar_alpha, cluster_masks, area_ratios)
        densities = extract_effective_densities(result)

        assert isinstance(densities, dict)
        assert "c0" in densities
        assert isinstance(densities["c0"], float)

    def test_extract_alpha_summary(self):
        """Test extract_alpha_summary utility."""
        T, R = 360, 200
        polar_alpha = np.full((T, R), 0.6, dtype=np.float32)
        cluster_masks = {"c0": np.ones((T, R), dtype=bool)}
        area_ratios = {"c0": 0.7}

        result = compute_effective_density(polar_alpha, cluster_masks, area_ratios)
        summary = extract_alpha_summary(result)

        assert isinstance(summary, dict)
        assert "c0" in summary
        assert "area_ratio" in summary["c0"]
        assert "alpha_used" in summary["c0"]
        assert "effective_density" in summary["c0"]
        assert "fallback_level" in summary["c0"]


class TestIntegration:
    """Integration tests with color_masks module."""

    def test_compute_cluster_effective_densities(self):
        """Test integration with color_masks metadata."""
        from src.engine_v7.core.measure.segmentation.color_masks import compute_cluster_effective_densities

        T, R = 360, 200
        polar_alpha = np.full((T, R), 0.5, dtype=np.float32)

        # Simulate color_masks output
        color_masks = {
            "color_0": np.ones((T, R), dtype=bool),
        }
        metadata = {
            "colors": [
                {
                    "color_id": "color_0",
                    "area_ratio": 0.6,
                    "role": "ink",
                }
            ],
            "warnings": [],
        }

        updated_meta, alpha_summary = compute_cluster_effective_densities(color_masks, metadata, polar_alpha)

        # Check metadata was updated
        assert "effective_density" in updated_meta["colors"][0]
        assert "alpha_used" in updated_meta["colors"][0]
        assert "alpha_fallback_level" in updated_meta["colors"][0]
        assert "alpha_analysis" in updated_meta

        # Check summary
        assert "color_0" in alpha_summary


class TestEdgeCases:
    """Edge case tests."""

    def test_very_small_mask(self):
        """Test with very small cluster mask."""
        T, R = 360, 200
        polar_alpha = np.full((T, R), 0.5, dtype=np.float32)

        # Tiny mask (only 10 pixels)
        mask = np.zeros((T, R), dtype=bool)
        mask[0, :10] = True

        cluster_masks = {"tiny": mask}
        area_ratios = {"tiny": 0.001}

        result = compute_effective_density(polar_alpha, cluster_masks, area_ratios)

        # Should still produce a result (with fallback)
        assert "tiny" in result.clusters
        assert result.clusters["tiny"].effective_density >= 0

    def test_alpha_clipping(self):
        """Test that alpha values are properly bounded."""
        T, R = 360, 200
        # Create alpha map with extreme values
        polar_alpha = np.full((T, R), 1.5, dtype=np.float32)  # Above 1.0

        cluster_masks = {"c0": np.ones((T, R), dtype=bool)}
        area_ratios = {"c0": 0.5}

        result = compute_effective_density(polar_alpha, cluster_masks, area_ratios)

        # Alpha should be used as-is (no automatic clipping in this function)
        # But effective_density = area * alpha should still be computed
        assert result.clusters["c0"].effective_density > 0

    def test_nan_in_alpha_map(self):
        """Test handling of NaN values in alpha map."""
        T, R = 360, 200
        polar_alpha = np.full((T, R), 0.5, dtype=np.float32)
        polar_alpha[::2, :] = np.nan  # 50% NaN

        cluster_masks = {"c0": np.ones((T, R), dtype=bool)}
        area_ratios = {"c0": 0.5}

        result = compute_effective_density(polar_alpha, cluster_masks, area_ratios)

        # Should handle NaN gracefully (np.nanmedian behavior)
        assert not np.isnan(result.clusters["c0"].effective_density)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
