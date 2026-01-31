"""
Unit tests for transition_detector module.

Tests P2 feature: transition/boundary region detection and weighting.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from src.engine_v7.core.measure.metrics.transition_detector import (
    TransitionConfig,
    TransitionResult,
    apply_transition_weights_to_samples,
    compute_cluster_boundary_mask,
    compute_lab_gradient_polar,
    compute_transition_weights,
    create_alpha_weight_map,
    detect_transition_mask,
    weighted_median,
)


class TestTransitionConfig:
    """Tests for TransitionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        cfg = TransitionConfig()
        assert cfg.gradient_threshold == 10.0
        assert cfg.dilation_radius == 3
        assert cfg.erosion_radius == 1
        assert cfg.transition_weight == 0.3
        assert cfg.core_weight == 1.0
        assert cfg.smooth_weights is True
        assert cfg.L_weight == 1.0
        assert cfg.a_weight == 1.5
        assert cfg.b_weight == 1.5

    def test_custom_values(self):
        """Test custom configuration."""
        cfg = TransitionConfig(
            gradient_threshold=15.0,
            dilation_radius=5,
            transition_weight=0.5,
        )
        assert cfg.gradient_threshold == 15.0
        assert cfg.dilation_radius == 5
        assert cfg.transition_weight == 0.5


class TestComputeLabGradientPolar:
    """Tests for compute_lab_gradient_polar function."""

    def test_uniform_image_zero_gradient(self):
        """Test that uniform image has near-zero gradient."""
        T, R = 360, 200
        # Uniform Lab image
        polar_lab = np.full((T, R, 3), 50.0, dtype=np.float32)

        gradient = compute_lab_gradient_polar(polar_lab)

        assert gradient.shape == (T, R)
        # Gradient should be very small (near zero)
        assert np.mean(gradient) < 1.0

    def test_horizontal_edge_gradient(self):
        """Test gradient detection for horizontal edge (theta direction)."""
        T, R = 360, 200
        polar_lab = np.zeros((T, R, 3), dtype=np.float32)
        # Create horizontal edge at theta=180
        polar_lab[:180, :, 0] = 100.0  # L channel
        polar_lab[180:, :, 0] = 20.0

        gradient = compute_lab_gradient_polar(polar_lab)

        # Gradient should be high near theta=180
        edge_region = gradient[175:185, :]
        non_edge_region = gradient[:170, :]

        assert np.mean(edge_region) > np.mean(non_edge_region) * 2

    def test_radial_edge_gradient(self):
        """Test gradient detection for radial edge (r direction)."""
        T, R = 360, 200
        polar_lab = np.zeros((T, R, 3), dtype=np.float32)
        # Create radial edge at r=100
        polar_lab[:, :100, 0] = 80.0  # L channel
        polar_lab[:, 100:, 0] = 20.0

        gradient = compute_lab_gradient_polar(polar_lab)

        # Gradient should be high near r=100
        edge_region = gradient[:, 95:105]
        non_edge_region = gradient[:, :90]

        assert np.mean(edge_region) > np.mean(non_edge_region) * 2

    def test_channel_weights(self):
        """Test that channel weights affect gradient magnitude."""
        T, R = 360, 200

        # Create image with edge only in 'a' channel
        polar_lab = np.zeros((T, R, 3), dtype=np.float32)
        polar_lab[:180, :, 1] = 50.0  # a channel
        polar_lab[180:, :, 1] = -50.0

        # High a_weight
        grad_high = compute_lab_gradient_polar(polar_lab, a_weight=2.0)
        # Low a_weight
        grad_low = compute_lab_gradient_polar(polar_lab, a_weight=0.5)

        assert np.mean(grad_high) > np.mean(grad_low) * 1.5


class TestDetectTransitionMask:
    """Tests for detect_transition_mask function."""

    def test_low_gradient_no_transition(self):
        """Test that low gradient produces no transition regions."""
        T, R = 360, 200
        gradient = np.full((T, R), 5.0, dtype=np.float32)  # Below threshold

        mask = detect_transition_mask(gradient, threshold=10.0)

        assert mask.dtype == bool
        assert np.sum(mask) == 0  # No transitions

    def test_high_gradient_has_transitions(self):
        """Test that high gradient produces transition regions."""
        T, R = 360, 200
        gradient = np.full((T, R), 5.0, dtype=np.float32)
        # Add high gradient region
        gradient[175:185, :] = 20.0  # Above threshold

        mask = detect_transition_mask(gradient, threshold=10.0, dilation_radius=0, erosion_radius=0)

        # Should have transitions
        assert np.any(mask)
        # Transitions should be around the high gradient region
        assert np.sum(mask[175:185, :]) > 0

    def test_dilation_expands_transition(self):
        """Test that dilation expands transition regions."""
        T, R = 360, 200
        gradient = np.full((T, R), 5.0, dtype=np.float32)
        gradient[180, 100] = 50.0  # Single high point

        mask_no_dil = detect_transition_mask(gradient, threshold=10.0, dilation_radius=0, erosion_radius=0)
        mask_with_dil = detect_transition_mask(gradient, threshold=10.0, dilation_radius=5, erosion_radius=0)

        assert np.sum(mask_with_dil) > np.sum(mask_no_dil)

    def test_erosion_removes_noise(self):
        """Test that erosion removes small noise regions."""
        T, R = 360, 200
        gradient = np.full((T, R), 5.0, dtype=np.float32)
        # Scattered noise points
        gradient[::20, ::20] = 15.0

        mask_no_erosion = detect_transition_mask(gradient, threshold=10.0, dilation_radius=0, erosion_radius=0)
        mask_with_erosion = detect_transition_mask(gradient, threshold=10.0, dilation_radius=0, erosion_radius=2)

        # Erosion should remove isolated points
        assert np.sum(mask_with_erosion) <= np.sum(mask_no_erosion)


class TestComputeTransitionWeights:
    """Tests for compute_transition_weights function."""

    def test_uniform_image_high_weights(self):
        """Test that uniform image gets high (core) weights everywhere."""
        T, R = 360, 200
        polar_lab = np.full((T, R, 3), 50.0, dtype=np.float32)

        result = compute_transition_weights(polar_lab)

        assert isinstance(result, TransitionResult)
        assert result.weight_map.shape == (T, R)
        # Most weights should be near core_weight (1.0)
        assert np.mean(result.weight_map) > 0.8

    def test_edge_image_low_weights_at_edge(self):
        """Test that edge region gets low weights."""
        T, R = 360, 200
        polar_lab = np.zeros((T, R, 3), dtype=np.float32)
        polar_lab[:180, :, :] = 100.0
        polar_lab[180:, :, :] = 0.0

        # Disable erosion to preserve thin edge detection
        cfg = TransitionConfig(gradient_threshold=5.0, erosion_radius=0)
        result = compute_transition_weights(polar_lab, cfg=cfg)

        # Transition ratio should be non-zero
        assert result.transition_ratio > 0
        # Edge region should have lower weights
        edge_weights = result.weight_map[175:185, :]
        core_weights = result.weight_map[:170, :]
        assert np.mean(edge_weights) < np.mean(core_weights)

    def test_transition_statistics(self):
        """Test that statistics are computed correctly."""
        T, R = 360, 200
        polar_lab = np.zeros((T, R, 3), dtype=np.float32)
        polar_lab[:180, :, :] = 80.0
        polar_lab[180:, :, :] = 20.0

        result = compute_transition_weights(polar_lab)

        assert 0.0 <= result.transition_ratio <= 1.0
        assert result.mean_gradient > 0
        assert result.max_gradient > result.mean_gradient
        assert isinstance(result.config, TransitionConfig)

    def test_per_cluster_transition_ratios(self):
        """Test per-cluster transition ratio computation."""
        T, R = 360, 200
        polar_lab = np.zeros((T, R, 3), dtype=np.float32)
        polar_lab[:180, :, :] = 80.0
        polar_lab[180:, :, :] = 20.0

        # Create cluster masks
        cluster_masks = {
            "cluster1": np.zeros((T, R), dtype=np.uint8),
            "cluster2": np.zeros((T, R), dtype=np.uint8),
        }
        cluster_masks["cluster1"][:180, :] = 1  # Top half
        cluster_masks["cluster2"][180:, :] = 1  # Bottom half

        result = compute_transition_weights(polar_lab, cluster_masks)

        assert "cluster1" in result.cluster_transition_ratios
        assert "cluster2" in result.cluster_transition_ratios


class TestComputeClusterBoundaryMask:
    """Tests for compute_cluster_boundary_mask function."""

    def test_empty_masks_returns_empty(self):
        """Test empty input returns empty mask."""
        mask = compute_cluster_boundary_mask({})
        assert mask.shape == (1, 1)

    def test_single_cluster_no_boundary(self):
        """Test single cluster has minimal boundary."""
        T, R = 360, 200
        cluster_masks = {
            "cluster1": np.ones((T, R), dtype=np.uint8),
        }

        mask = compute_cluster_boundary_mask(cluster_masks, boundary_width=1)

        # Single cluster should have boundary only at edges
        assert mask.shape == (T, R)

    def test_two_clusters_has_boundary(self):
        """Test two adjacent clusters have boundary between them."""
        T, R = 360, 200
        cluster_masks = {
            "cluster1": np.zeros((T, R), dtype=np.uint8),
            "cluster2": np.zeros((T, R), dtype=np.uint8),
        }
        cluster_masks["cluster1"][:, :100] = 1  # Left half
        cluster_masks["cluster2"][:, 100:] = 1  # Right half

        mask = compute_cluster_boundary_mask(cluster_masks, boundary_width=5)

        # Boundary should exist near r=100
        assert np.any(mask)
        boundary_region = mask[:, 95:105]
        assert np.sum(boundary_region) > 0


class TestApplyTransitionWeightsToSamples:
    """Tests for apply_transition_weights_to_samples function."""

    def test_basic_application(self):
        """Test basic weight application."""
        samples = np.array([1.0, 2.0, 3.0, 4.0])
        coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        weight_map = np.array([[0.5, 1.0], [1.0, 0.3]])

        _, weights = apply_transition_weights_to_samples(samples, coords, weight_map)

        assert len(weights) == 4
        assert weights[0] == 0.5
        assert weights[1] == 1.0
        assert weights[2] == 1.0
        assert weights[3] == 0.3

    def test_out_of_bounds_coordinates(self):
        """Test handling of out-of-bounds coordinates."""
        samples = np.array([1.0, 2.0])
        coords = np.array([[0, 0], [100, 100]])  # Second is out of bounds
        weight_map = np.ones((10, 10))

        _, weights = apply_transition_weights_to_samples(samples, coords, weight_map)

        assert weights[0] == 1.0
        assert weights[1] == 1.0  # Default for out of bounds


class TestWeightedMedian:
    """Tests for weighted_median function."""

    def test_uniform_weights(self):
        """Test that uniform weights give regular median."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.ones(5)

        result = weighted_median(values, weights)

        assert result == 3.0  # Regular median

    def test_weighted_result(self):
        """Test that weights shift the median."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Heavy weight on low values
        weights = np.array([10.0, 10.0, 1.0, 1.0, 1.0])

        result = weighted_median(values, weights)

        # Should be shifted toward low values
        assert result < 3.0

    def test_empty_array(self):
        """Test empty array returns NaN."""
        values = np.array([])
        weights = np.array([])

        result = weighted_median(values, weights)

        assert np.isnan(result)

    def test_nan_values_filtered(self):
        """Test that NaN values are filtered out."""
        values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        result = weighted_median(values, weights)

        assert result == 3.0  # Median of [1, 3, 5]


class TestCreateAlphaWeightMap:
    """Tests for create_alpha_weight_map function."""

    def test_basic_creation(self):
        """Test basic weight map creation."""
        T, R = 360, 200
        polar_lab = np.full((T, R, 3), 50.0, dtype=np.float32)
        cluster_masks = {
            "cluster1": np.ones((T, R), dtype=np.uint8),
        }

        weight_map, meta = create_alpha_weight_map(polar_lab, cluster_masks)

        assert weight_map.shape == (T, R)
        assert "mean_weight" in meta
        assert "low_weight_ratio" in meta

    def test_gradient_only(self):
        """Test with gradient weights only."""
        T, R = 360, 200
        polar_lab = np.zeros((T, R, 3), dtype=np.float32)
        polar_lab[:180, :, :] = 80.0
        cluster_masks = {"cluster1": np.ones((T, R), dtype=np.uint8)}

        weight_map, meta = create_alpha_weight_map(
            polar_lab,
            cluster_masks,
            use_gradient_weights=True,
            use_boundary_weights=False,
        )

        assert meta["gradient_transition"] is not None
        assert meta["boundary_transition"] is None

    def test_boundary_only(self):
        """Test with boundary weights only."""
        T, R = 360, 200
        polar_lab = np.full((T, R, 3), 50.0, dtype=np.float32)
        cluster_masks = {
            "cluster1": np.zeros((T, R), dtype=np.uint8),
            "cluster2": np.zeros((T, R), dtype=np.uint8),
        }
        cluster_masks["cluster1"][:, :100] = 1
        cluster_masks["cluster2"][:, 100:] = 1

        weight_map, meta = create_alpha_weight_map(
            polar_lab,
            cluster_masks,
            use_gradient_weights=False,
            use_boundary_weights=True,
        )

        assert meta["gradient_transition"] is None
        assert meta["boundary_transition"] is not None

    def test_combined_weights(self):
        """Test combined gradient and boundary weights."""
        T, R = 360, 200
        polar_lab = np.zeros((T, R, 3), dtype=np.float32)
        polar_lab[:180, :, :] = 80.0
        cluster_masks = {
            "cluster1": np.zeros((T, R), dtype=np.uint8),
            "cluster2": np.zeros((T, R), dtype=np.uint8),
        }
        cluster_masks["cluster1"][:, :100] = 1
        cluster_masks["cluster2"][:, 100:] = 1

        weight_map, meta = create_alpha_weight_map(
            polar_lab,
            cluster_masks,
            use_gradient_weights=True,
            use_boundary_weights=True,
        )

        assert meta["gradient_transition"] is not None
        assert meta["boundary_transition"] is not None
        # Combined should have lower weights than either alone
        assert meta["low_weight_ratio"] >= 0


class TestIntegrationWithAlphaDensity:
    """Integration tests with alpha_density module."""

    def test_transition_weights_in_alpha_computation(self):
        """Test that transition weights integrate with alpha density."""
        from src.engine_v7.core.measure.metrics.alpha_density import AlphaFallbackLevel, compute_effective_density

        T, R = 360, 200
        # Create alpha map
        polar_alpha = np.full((T, R), 0.8, dtype=np.float32)

        # Create polar Lab with edge
        polar_lab = np.zeros((T, R, 3), dtype=np.float32)
        polar_lab[:180, :, :] = 80.0

        cluster_masks = {
            "ink1": np.ones((T, R), dtype=np.uint8),
        }
        area_ratios = {"ink1": 0.5}

        # With transition weights enabled
        result = compute_effective_density(
            polar_alpha,
            cluster_masks,
            area_ratios,
            polar_lab=polar_lab,
            cfg={"transition_weights_enabled": True},
        )

        assert "ink1" in result.clusters
        # Check that warnings mention weight map
        has_weight_warning = any("WEIGHT_MAP" in w for w in result.warnings)
        assert has_weight_warning

    def test_transition_weights_disabled_by_default(self):
        """Test that transition weights are disabled by default."""
        from src.engine_v7.core.measure.metrics.alpha_density import DEFAULT_ALPHA_CONFIG, compute_effective_density

        assert DEFAULT_ALPHA_CONFIG["transition_weights_enabled"] is False

        T, R = 360, 200
        polar_alpha = np.full((T, R), 0.8, dtype=np.float32)
        cluster_masks = {"ink1": np.ones((T, R), dtype=np.uint8)}
        area_ratios = {"ink1": 0.5}

        result = compute_effective_density(
            polar_alpha,
            cluster_masks,
            area_ratios,
        )

        # Should not have weight map warning
        has_weight_warning = any("WEIGHT_MAP" in w for w in result.warnings)
        assert not has_weight_warning


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
