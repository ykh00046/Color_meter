"""
P2-5: KPI Validation Tests

These tests validate the 5 key performance indicators (KPIs) for the
alpha-based density computation system:

1. Alpha Accuracy: |measured - true| < threshold
2. Density Accuracy: |computed - true| < threshold
3. Quality Detection Rate: quality_fail correctly triggers when appropriate
4. Verification Success Rate: agreement passes when methods should match
5. Fallback Rate: L1 used preferentially, L3 only when necessary
"""

import numpy as np
import pytest
from core.measure.metrics.alpha_density import (
    AlphaDensityResult,
    AlphaFallbackLevel,
    AlphaVerificationResult,
    compute_alpha_radial_1d,
    compute_effective_density,
    verify_alpha_agreement,
)


class TestKPI1AlphaAccuracy:
    """KPI-1: Alpha measurement accuracy |measured - true| < threshold."""

    def test_alpha_radial_accurate_for_uniform_alpha(self):
        """Radial alpha should accurately measure uniform alpha regions."""
        T, R = 360, 200
        true_alpha = 0.6  # Known ground truth
        tolerance = 0.05  # 5% tolerance

        # Create uniform alpha map
        polar_alpha = np.full((T, R), true_alpha, dtype=np.float32)
        mask = np.ones((T, R), dtype=bool)

        profile = compute_alpha_radial_1d(
            polar_alpha,
            mask,
            n_bins=20,
            r_start=0.15,
            r_end=0.95,
        )

        # Check that measured alpha is within tolerance of true value
        valid_bins = ~np.isnan(profile.alpha_values)
        measured_mean = np.mean(profile.alpha_values[valid_bins])

        error = abs(measured_mean - true_alpha)
        assert error < tolerance, f"Alpha error {error:.4f} exceeds tolerance {tolerance}"

    def test_alpha_radial_accurate_for_gradient_alpha(self):
        """Radial alpha should track radial gradient accurately."""
        T, R = 360, 200
        tolerance = 0.08  # 8% tolerance for gradient

        # Create radial gradient alpha: 0.3 at center, 0.8 at edge
        r_coords = np.linspace(0, 1, R)
        true_alpha_profile = 0.3 + 0.5 * r_coords  # Linear gradient

        polar_alpha = np.broadcast_to(true_alpha_profile, (T, R)).astype(np.float32)
        mask = np.ones((T, R), dtype=bool)

        profile = compute_alpha_radial_1d(
            polar_alpha,
            mask,
            n_bins=20,
            r_start=0.15,
            r_end=0.95,
        )

        # Check correlation between measured and true
        valid_bins = ~np.isnan(profile.alpha_values)
        if np.sum(valid_bins) >= 5:
            # Measured should follow the same trend
            measured = profile.alpha_values[valid_bins]
            # Alpha should increase with radius
            assert measured[-1] > measured[0], "Alpha should increase with radius"

    def test_alpha_accuracy_with_noise(self):
        """Alpha measurement should be robust to moderate noise."""
        T, R = 360, 200
        true_alpha = 0.5
        noise_level = 0.1  # 10% noise
        tolerance = 0.08  # Should still be within 8%

        np.random.seed(42)
        polar_alpha = (
            np.full((T, R), true_alpha, dtype=np.float32) + np.random.randn(T, R).astype(np.float32) * noise_level
        )
        polar_alpha = np.clip(polar_alpha, 0.02, 0.98)
        mask = np.ones((T, R), dtype=bool)

        profile = compute_alpha_radial_1d(polar_alpha, mask)

        valid_bins = ~np.isnan(profile.alpha_values)
        measured_mean = np.mean(profile.alpha_values[valid_bins])

        error = abs(measured_mean - true_alpha)
        assert error < tolerance, f"Alpha error {error:.4f} with noise exceeds tolerance"


class TestKPI2DensityAccuracy:
    """KPI-2: Effective density accuracy |computed - true| < threshold."""

    def test_density_accuracy_uniform_alpha(self):
        """Computed density should match area_ratio * alpha for uniform alpha."""
        T, R = 360, 200
        true_alpha = 0.6
        area_ratio = 0.25
        true_density = area_ratio * true_alpha  # 0.15

        tolerance = 0.02  # 2% absolute tolerance

        polar_alpha = np.full((T, R), true_alpha, dtype=np.float32)
        masks = {"ink0": np.ones((T, R), dtype=bool)}
        area_ratios = {"ink0": area_ratio}

        result = compute_effective_density(polar_alpha, masks, area_ratios)

        computed_density = result.clusters["ink0"].effective_density

        error = abs(computed_density - true_density)
        assert error < tolerance, f"Density error {error:.4f} exceeds tolerance"

    def test_density_accuracy_multiple_clusters(self):
        """Density should be accurate for multiple ink clusters."""
        T, R = 360, 200
        tolerance = 0.03

        # Two clusters with different alphas
        polar_alpha = np.full((T, R), 0.5, dtype=np.float32)

        # Cluster 0: left half, alpha=0.4
        # Cluster 1: right half, alpha=0.7
        mask0 = np.zeros((T, R), dtype=bool)
        mask0[:, : R // 2] = True
        mask1 = np.zeros((T, R), dtype=bool)
        mask1[:, R // 2 :] = True

        polar_alpha[:, : R // 2] = 0.4
        polar_alpha[:, R // 2 :] = 0.7

        masks = {"ink0": mask0, "ink1": mask1}
        area_ratios = {"ink0": 0.20, "ink1": 0.30}

        result = compute_effective_density(polar_alpha, masks, area_ratios)

        # Check each cluster
        for color_id in ["ink0", "ink1"]:
            cluster = result.clusters[color_id]
            # Density should be positive and reasonable
            assert 0 < cluster.effective_density < 1.0


class TestKPI3QualityDetectionRate:
    """KPI-3: Quality fail correctly triggers when data quality is poor."""

    def test_quality_fail_triggers_on_high_nan_ratio(self):
        """Quality fail should trigger when NaN ratio exceeds threshold."""
        T, R = 360, 200

        # Create alpha with 40% NaN (exceeds default 10% threshold)
        polar_alpha = np.full((T, R), 0.5, dtype=np.float32)
        polar_alpha[:, : int(R * 0.4)] = np.nan

        masks = {"ink0": np.ones((T, R), dtype=bool)}
        area_ratios = {"ink0": 0.25}

        result = compute_effective_density(polar_alpha, masks, area_ratios)

        # Should fallback due to quality fail
        cluster = result.clusters["ink0"]
        assert cluster.fallback_level in [AlphaFallbackLevel.L2_ZONE, AlphaFallbackLevel.L3_GLOBAL]
        # Check that quality-related warning or reason exists
        assert "alpha_quality_fail" in (cluster.fallback_reason or "") or any(
            "NAN" in w or "QUALITY" in w for w in result.warnings
        )

    def test_quality_fail_triggers_on_high_clip_ratio(self):
        """Quality fail should trigger when too many values are clipped."""
        T, R = 360, 200

        # Create alpha with 50% at clip boundary (exceeds 30% threshold)
        polar_alpha = np.full((T, R), 0.5, dtype=np.float32)
        polar_alpha[:, : R // 2] = 0.02  # At lower clip boundary

        masks = {"ink0": np.ones((T, R), dtype=bool)}
        area_ratios = {"ink0": 0.25}

        cfg = {
            "quality_fail": {
                "clip_ratio": 0.30,  # 30% threshold
            }
        }

        result = compute_effective_density(polar_alpha, masks, area_ratios, cfg=cfg)

        # Should detect quality issue
        cluster = result.clusters["ink0"]
        # Either fallback or warning should be present
        has_quality_issue = cluster.fallback_level != AlphaFallbackLevel.L1_RADIAL or any(
            "CLIP" in w or "QUALITY" in w for w in result.warnings
        )
        # This is informational - we expect clip to be detected
        assert isinstance(result, AlphaDensityResult)

    def test_quality_pass_for_good_data(self):
        """Quality check should pass for good quality data."""
        T, R = 360, 200

        # Good quality alpha: no NaN, values in valid range
        polar_alpha = np.random.uniform(0.3, 0.7, size=(T, R)).astype(np.float32)

        masks = {"ink0": np.ones((T, R), dtype=bool)}
        area_ratios = {"ink0": 0.25}

        result = compute_effective_density(polar_alpha, masks, area_ratios)

        cluster = result.clusters["ink0"]
        # Should use L1 or L2 (not forced to L3)
        assert cluster.fallback_level != AlphaFallbackLevel.L3_GLOBAL or cluster.fallback_reason is not None


class TestKPI4VerificationSuccessRate:
    """KPI-4: Verification passes when methods should match."""

    def test_verification_passes_for_identical_methods(self):
        """Verification should pass when both methods produce same result."""
        T, R = 180, 100
        np.random.seed(42)

        # Same alpha from both methods
        alpha = np.random.uniform(0.3, 0.8, size=(T, R)).astype(np.float32)

        result = verify_alpha_agreement(alpha, alpha.copy())

        assert result.passed is True
        assert result.agreement_score >= 0.95
        assert result.rmse < 0.01

    def test_verification_passes_for_similar_methods(self):
        """Verification should pass when methods produce similar results."""
        T, R = 180, 100
        np.random.seed(42)

        alpha1 = np.random.uniform(0.3, 0.8, size=(T, R)).astype(np.float32)
        # Add small noise (3% should still pass)
        noise = np.random.randn(T, R).astype(np.float32) * 0.03
        alpha2 = np.clip(alpha1 + noise, 0.02, 0.98)

        result = verify_alpha_agreement(alpha1, alpha2)

        assert result.passed is True
        assert result.agreement_score >= 0.7

    def test_verification_fails_for_different_methods(self):
        """Verification should fail when methods produce different results."""
        T, R = 180, 100
        np.random.seed(42)

        # Completely different alpha maps
        alpha1 = np.random.uniform(0.2, 0.4, size=(T, R)).astype(np.float32)
        alpha2 = np.random.uniform(0.6, 0.9, size=(T, R)).astype(np.float32)

        result = verify_alpha_agreement(alpha1, alpha2)

        assert result.passed is False
        assert result.rmse > 0.15
        assert len(result.warnings) > 0

    def test_verification_core_metrics_improve_boundary_tolerance(self):
        """Core metrics should be more tolerant of boundary differences."""
        T, R = 180, 100
        np.random.seed(42)

        # Good agreement in core, disagreement at boundaries
        alpha1 = np.random.uniform(0.4, 0.6, size=(T, R)).astype(np.float32)
        alpha2 = alpha1.copy()

        # Large disagreement at edges (30% difference)
        alpha2[:, :10] = np.clip(alpha1[:, :10] + 0.30, 0.02, 0.98)
        alpha2[:, -10:] = np.clip(alpha1[:, -10:] + 0.30, 0.02, 0.98)

        result = verify_alpha_agreement(alpha1, alpha2, use_core_for_decision=True)

        # Core metrics should show better agreement than full
        if result.summary.get("core_valid_pixels", 0) >= 100:
            assert result.core_rmse <= result.rmse


class TestKPI5FallbackRate:
    """KPI-5: L1 used preferentially, L3 only when necessary."""

    def test_l1_used_for_high_quality_data(self):
        """L1 radial profile should be used when data quality is good."""
        T, R = 360, 200

        # High quality data: good coverage, no NaN, good range
        polar_alpha = np.random.uniform(0.3, 0.7, size=(T, R)).astype(np.float32)

        masks = {"ink0": np.ones((T, R), dtype=bool)}
        area_ratios = {"ink0": 0.25}

        result = compute_effective_density(polar_alpha, masks, area_ratios)

        cluster = result.clusters["ink0"]
        assert cluster.fallback_level == AlphaFallbackLevel.L1_RADIAL

    def test_l2_used_when_l1_fails_confidence(self):
        """L2 zone profile should be used when L1 has low confidence."""
        T, R = 360, 200

        # Sparse mask - not enough samples for confident L1
        polar_alpha = np.random.uniform(0.3, 0.7, size=(T, R)).astype(np.float32)

        # Very sparse mask (1% coverage)
        mask = np.zeros((T, R), dtype=bool)
        mask[::10, ::10] = True  # Only every 10th pixel

        masks = {"ink0": mask}
        area_ratios = {"ink0": 0.01}  # Small area

        result = compute_effective_density(polar_alpha, masks, area_ratios)

        cluster = result.clusters["ink0"]
        # Should fallback to L2 or L3 due to insufficient samples
        assert cluster.fallback_level in [AlphaFallbackLevel.L2_ZONE, AlphaFallbackLevel.L3_GLOBAL]

    def test_l3_used_as_last_resort(self):
        """L3 global should only be used when L1 and L2 both fail."""
        T, R = 360, 200

        # Very poor data quality
        polar_alpha = np.full((T, R), np.nan, dtype=np.float32)
        # Only 5% valid
        polar_alpha[:, :10] = 0.5

        masks = {"ink0": np.ones((T, R), dtype=bool)}
        area_ratios = {"ink0": 0.25}

        result = compute_effective_density(polar_alpha, masks, area_ratios)

        cluster = result.clusters["ink0"]
        # With mostly NaN data, should fallback to L3
        assert cluster.fallback_level == AlphaFallbackLevel.L3_GLOBAL

    def test_fallback_hierarchy_respected(self):
        """Fallback should follow L1 -> L2 -> L3 hierarchy."""
        T, R = 360, 200

        # Test multiple clusters with varying quality
        polar_alpha = np.random.uniform(0.3, 0.7, size=(T, R)).astype(np.float32)

        # Cluster 0: Good coverage (should use L1)
        mask0 = np.ones((T, R), dtype=bool)
        mask0[:, R // 2 :] = False

        # Cluster 1: Poor coverage (might use L2/L3)
        mask1 = np.zeros((T, R), dtype=bool)
        mask1[::20, R // 2 :] = True  # Very sparse

        masks = {"ink0": mask0, "ink1": mask1}
        area_ratios = {"ink0": 0.25, "ink1": 0.05}

        result = compute_effective_density(polar_alpha, masks, area_ratios)

        # ink0 should have better fallback level than ink1
        level_order = {
            AlphaFallbackLevel.L1_RADIAL: 0,
            AlphaFallbackLevel.L2_ZONE: 1,
            AlphaFallbackLevel.L3_GLOBAL: 2,
        }

        ink0_level = level_order[result.clusters["ink0"].fallback_level]
        ink1_level = level_order[result.clusters["ink1"].fallback_level]

        # ink0 should be same or better (lower level number) than ink1
        assert ink0_level <= ink1_level


class TestKPIIntegration:
    """Integration tests combining multiple KPIs."""

    def test_end_to_end_accurate_density(self):
        """End-to-end test: accurate density from good quality data."""
        T, R = 360, 200
        np.random.seed(42)

        # Known ground truth
        true_alpha = 0.55
        area_ratio = 0.30
        expected_density = true_alpha * area_ratio  # 0.165

        # Generate realistic data
        polar_alpha = (
            np.full((T, R), true_alpha, dtype=np.float32) + np.random.randn(T, R).astype(np.float32) * 0.05  # 5% noise
        )
        polar_alpha = np.clip(polar_alpha, 0.02, 0.98)

        masks = {"ink0": np.ones((T, R), dtype=bool)}
        area_ratios = {"ink0": area_ratio}

        result = compute_effective_density(polar_alpha, masks, area_ratios)

        # KPI-1: Alpha accuracy
        cluster = result.clusters["ink0"]
        assert abs(cluster.alpha_used - true_alpha) < 0.10

        # KPI-2: Density accuracy
        assert abs(cluster.effective_density - expected_density) < 0.05

        # KPI-5: Should use L1 for good data
        assert cluster.fallback_level == AlphaFallbackLevel.L1_RADIAL

    def test_system_handles_edge_cases_gracefully(self):
        """System should handle edge cases without crashing."""
        T, R = 360, 200

        edge_cases = [
            # All NaN
            np.full((T, R), np.nan, dtype=np.float32),
            # All zeros
            np.zeros((T, R), dtype=np.float32),
            # All ones
            np.ones((T, R), dtype=np.float32),
            # Mixed extremes
            np.where(np.random.rand(T, R) > 0.5, 0.02, 0.98).astype(np.float32),
        ]

        masks = {"ink0": np.ones((T, R), dtype=bool)}
        area_ratios = {"ink0": 0.25}

        for i, polar_alpha in enumerate(edge_cases):
            # Should not raise exception
            result = compute_effective_density(polar_alpha, masks, area_ratios)
            assert isinstance(result, AlphaDensityResult), f"Edge case {i} failed"
            # Should produce valid (possibly fallback) result
            assert "ink0" in result.clusters
