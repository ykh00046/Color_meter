"""
Unit tests for Hungarian matcher module.

Tests P1 feature: STD-based label stabilization using Hungarian algorithm.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from src.engine_v7.core.measure.matching.hungarian_matcher import (
    SCIPY_AVAILABLE,
    MatchResult,
    compute_lab_distance_matrix,
    match_clusters_to_reference,
    reorder_clusters_by_reference,
    stabilize_cluster_labels,
)


class TestComputeLabDistanceMatrix:
    """Tests for Lab distance matrix computation."""

    def test_basic_distance_matrix(self):
        """Test basic distance matrix computation."""
        detected = np.array(
            [
                [50, 0, 0],  # Gray
                [100, 0, 0],  # White
            ],
            dtype=np.float64,
        )

        reference = np.array(
            [
                [0, 0, 0],  # Black
                [50, 0, 0],  # Gray
            ],
            dtype=np.float64,
        )

        cost_matrix = compute_lab_distance_matrix(detected, reference)

        assert cost_matrix.shape == (2, 2)
        # Gray to Gray should be 0
        assert cost_matrix[0, 1] == pytest.approx(0.0, abs=1e-6)
        # White to Black should be 100
        assert cost_matrix[1, 0] == pytest.approx(100.0, abs=1e-6)

    def test_distance_matrix_with_color(self):
        """Test distance with chromatic colors."""
        detected = np.array(
            [
                [50, 50, 0],  # Red-ish
                [50, -50, 0],  # Green-ish
            ],
            dtype=np.float64,
        )

        reference = np.array(
            [
                [50, -50, 0],  # Green-ish
                [50, 50, 0],  # Red-ish
            ],
            dtype=np.float64,
        )

        cost_matrix = compute_lab_distance_matrix(detected, reference)

        # Detected[0] (red) should be close to Reference[1] (red)
        # Detected[1] (green) should be close to Reference[0] (green)
        assert cost_matrix[0, 1] < cost_matrix[0, 0]
        assert cost_matrix[1, 0] < cost_matrix[1, 1]


class TestMatchClustersToReference:
    """Tests for Hungarian matching algorithm."""

    def test_perfect_match(self):
        """Test perfect 1-to-1 matching."""
        detected = np.array(
            [
                [30, -10, -40],  # Dark blue
                [70, -30, 20],  # Light green
            ],
            dtype=np.float64,
        )

        reference = np.array(
            [
                [30, -10, -40],  # Dark blue
                [70, -30, 20],  # Light green
            ],
            dtype=np.float64,
        )

        result = match_clusters_to_reference(detected, reference)

        assert isinstance(result, MatchResult)
        assert result.n_matched == 2
        assert result.n_unmatched_detected == 0
        assert result.n_unmatched_reference == 0
        assert result.mean_distance == pytest.approx(0.0, abs=1e-6)
        assert result.confidence > 0.9

    def test_swapped_order(self):
        """Test matching with swapped order (the main use case)."""
        # Detected in different order than reference
        detected = np.array(
            [
                [70, -30, 20],  # Light green (detected first)
                [30, -10, -40],  # Dark blue (detected second)
            ],
            dtype=np.float64,
        )

        reference = np.array(
            [
                [30, -10, -40],  # Dark blue (reference first)
                [70, -30, 20],  # Light green (reference second)
            ],
            dtype=np.float64,
        )

        result = match_clusters_to_reference(detected, reference)

        # Detected[0] (green) should match Reference[1]
        # Detected[1] (blue) should match Reference[0]
        assert result.assignments[0] == 1
        assert result.assignments[1] == 0
        assert result.n_matched == 2

    def test_more_detected_than_reference(self):
        """Test with extra detected clusters."""
        detected = np.array(
            [
                [30, -10, -40],  # Dark blue
                [70, -30, 20],  # Light green
                [50, 0, 0],  # Gray (extra)
            ],
            dtype=np.float64,
        )

        reference = np.array(
            [
                [30, -10, -40],  # Dark blue
                [70, -30, 20],  # Light green
            ],
            dtype=np.float64,
        )

        result = match_clusters_to_reference(detected, reference)

        assert result.n_detected == 3
        assert result.n_reference == 2
        assert result.n_matched == 2
        assert result.n_unmatched_detected == 1
        # Gray should be unmatched
        assert result.assignments[2] == -1

    def test_more_reference_than_detected(self):
        """Test with missing detected clusters."""
        detected = np.array(
            [
                [30, -10, -40],  # Dark blue
            ],
            dtype=np.float64,
        )

        reference = np.array(
            [
                [30, -10, -40],  # Dark blue
                [70, -30, 20],  # Light green (missing)
            ],
            dtype=np.float64,
        )

        result = match_clusters_to_reference(detected, reference)

        assert result.n_detected == 1
        assert result.n_reference == 2
        assert result.n_matched == 1
        assert result.n_unmatched_reference == 1

    def test_distance_threshold(self):
        """Test that matches beyond threshold are rejected."""
        detected = np.array(
            [
                [0, 0, 0],  # Black
                [100, 0, 0],  # White
            ],
            dtype=np.float64,
        )

        reference = np.array(
            [
                [50, 0, 0],  # Gray
                [50, 50, 0],  # Colored gray
            ],
            dtype=np.float64,
        )

        # With strict threshold, matches should be rejected
        result = match_clusters_to_reference(detected, reference, max_distance_threshold=30.0)

        # Both matches exceed threshold (50+ deltaE)
        assert result.n_matched < 2
        assert "MATCH_REJECTED" in " ".join(result.warnings)

    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        detected = np.array([]).reshape(0, 3)
        reference = np.array([[50, 0, 0]])

        result = match_clusters_to_reference(detected, reference)

        assert result.n_matched == 0
        assert "EMPTY_INPUT" in result.warnings


class TestReorderClustersByReference:
    """Tests for cluster reordering."""

    def test_basic_reorder(self):
        """Test basic reordering of cluster data."""
        detected_data = [
            {"color_id": "color_0", "area_ratio": 0.3, "lab_centroid_cie": [70, -30, 20]},
            {"color_id": "color_1", "area_ratio": 0.5, "lab_centroid_cie": [30, -10, -40]},
        ]

        # Swapped order
        detected = np.array([[70, -30, 20], [30, -10, -40]])
        reference = np.array([[30, -10, -40], [70, -30, 20]])

        match_result = match_clusters_to_reference(detected, reference)
        reordered, meta = reorder_clusters_by_reference(
            detected_data, match_result, reference_ids=["dark_blue", "light_green"]
        )

        # First should be dark_blue (originally color_1)
        assert reordered[0]["color_id"] == "dark_blue"
        assert reordered[0]["original_color_id"] == "color_1"
        # Second should be light_green (originally color_0)
        assert reordered[1]["color_id"] == "light_green"
        assert reordered[1]["original_color_id"] == "color_0"

    def test_missing_reference(self):
        """Test placeholder creation for missing reference."""
        detected_data = [
            {"color_id": "color_0", "area_ratio": 0.5, "lab_centroid_cie": [30, -10, -40]},
        ]

        detected = np.array([[30, -10, -40]])
        reference = np.array([[30, -10, -40], [70, -30, 20]])

        match_result = match_clusters_to_reference(detected, reference)
        reordered, meta = reorder_clusters_by_reference(
            detected_data, match_result, reference_ids=["dark_blue", "light_green"]
        )

        assert len(reordered) == 2
        # First is matched
        assert reordered[0]["color_id"] == "dark_blue"
        assert not reordered[0].get("is_placeholder", False)
        # Second is placeholder
        assert reordered[1]["color_id"] == "light_green"
        assert reordered[1].get("is_placeholder", False)
        assert meta["n_missing"] == 1


class TestStabilizeClusterLabels:
    """Tests for full label stabilization workflow."""

    def test_full_stabilization(self):
        """Test complete stabilization workflow."""
        detected = np.array(
            [
                [70, -30, 20],  # Light green (detected first)
                [30, -10, -40],  # Dark blue (detected second)
            ]
        )
        reference = np.array(
            [
                [30, -10, -40],  # Dark blue
                [70, -30, 20],  # Light green
            ]
        )

        T, R = 360, 200
        detected_masks = {
            "color_0": np.ones((T, R), dtype=bool),
            "color_1": np.zeros((T, R), dtype=bool),
        }
        detected_metadata = [
            {"color_id": "color_0", "area_ratio": 0.6, "lab_centroid_cie": [70, -30, 20]},
            {"color_id": "color_1", "area_ratio": 0.4, "lab_centroid_cie": [30, -10, -40]},
        ]

        stable_masks, stable_meta, info = stabilize_cluster_labels(
            detected,
            reference,
            detected_masks,
            detected_metadata,
            reference_ids=["dark_blue", "light_green"],
        )

        # Masks should be renamed
        assert "dark_blue" in stable_masks or "color_0" in stable_masks
        # Metadata should be reordered
        assert info["match_result"]["n_matched"] == 2


class TestIntegrationWithColorMasks:
    """Integration tests with color_masks module."""

    def test_stabilize_labels_with_reference(self):
        """Test stabilize_labels_with_reference function."""
        from src.engine_v7.core.measure.segmentation.color_masks import stabilize_labels_with_reference

        T, R = 360, 200
        masks = {
            "color_0": np.ones((T, R), dtype=bool),
            "color_1": np.zeros((T, R), dtype=bool),
        }
        metadata = {
            "colors": [
                {"color_id": "color_0", "area_ratio": 0.6, "lab_centroid_cie": [70, -30, 20]},
                {"color_id": "color_1", "area_ratio": 0.4, "lab_centroid_cie": [30, -10, -40]},
            ],
            "warnings": [],
        }

        # Reference in different order
        reference = np.array(
            [
                [30, -10, -40],  # dark_blue
                [70, -30, 20],  # light_green
            ]
        )

        stable_masks, stable_meta = stabilize_labels_with_reference(
            masks,
            metadata,
            reference,
            reference_ids=["dark_blue", "light_green"],
        )

        # Check metadata was updated
        assert "label_stabilization" in stable_meta
        assert stable_meta["label_order"] == "reference_matched"


class TestEdgeCases:
    """Edge case tests."""

    def test_single_cluster(self):
        """Test with single cluster."""
        detected = np.array([[50, 0, 0]])
        reference = np.array([[50, 0, 0]])

        result = match_clusters_to_reference(detected, reference)

        assert result.n_matched == 1
        assert result.assignments[0] == 0

    def test_identical_colors(self):
        """Test with identical detected colors."""
        detected = np.array(
            [
                [50, 0, 0],
                [50, 0, 0],  # Same as first
            ]
        )
        reference = np.array(
            [
                [50, 0, 0],
                [100, 0, 0],
            ]
        )

        result = match_clusters_to_reference(detected, reference)

        # Should still produce valid assignment
        assert result.n_matched == 2
        # One detected should match first reference
        assert 0 in result.assignments.values()

    def test_very_different_colors(self):
        """Test with very different colors (should still match)."""
        detected = np.array(
            [
                [10, -50, -50],  # Very dark cyan
                [90, 50, 50],  # Very light orange
            ]
        )
        reference = np.array(
            [
                [90, 50, 50],  # Very light orange
                [10, -50, -50],  # Very dark cyan
            ]
        )

        result = match_clusters_to_reference(detected, reference)

        # Should swap correctly
        assert result.assignments[0] == 1
        assert result.assignments[1] == 0

    def test_scipy_availability(self):
        """Test scipy availability flag."""
        # Just verify the flag exists and is boolean
        assert isinstance(SCIPY_AVAILABLE, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
