"""
Test suite for InkEstimator module (GMM-based ink analysis)

Tests cover:
- Pixel sampling and filtering
- GMM clustering with BIC optimization
- Mixing correction logic (Linearity Check)
- Edge cases (1, 2, 3 inks)
- Parameter sensitivity
"""

import numpy as np
import pytest

from src.core.ink_estimator import InkColor, InkEstimator


class TestInkEstimatorSampling:
    """Test pixel sampling and filtering logic"""

    def test_sample_ink_pixels_basic(self):
        """Test basic pixel sampling"""
        estimator = InkEstimator(random_seed=42)

        # Create synthetic image (100x100, 3 channels BGR)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[25:75, 25:75] = [100, 150, 200]  # Add colored region (BGR)

        samples, sampling_info = estimator.sample_ink_pixels(img, chroma_thresh=6.0, L_max=98.0)

        assert samples.shape[1] == 3  # L, a, b
        assert len(samples) > 0  # Should find some pixels

        # Verify Lab range
        assert np.all(samples[:, 0] >= 0) and np.all(samples[:, 0] <= 100)  # L in [0, 100]
        assert np.all(samples[:, 1] >= -128) and np.all(samples[:, 1] <= 127)  # a in [-128, 127]
        assert np.all(samples[:, 2] >= -128) and np.all(samples[:, 2] <= 127)  # b in [-128, 127]

        # Verify sampling_info dict
        assert isinstance(sampling_info, dict)
        assert "candidate_pixels" in sampling_info
        assert "sampled_pixels" in sampling_info
        assert sampling_info["candidate_pixels"] > 0
        assert sampling_info["sampled_pixels"] > 0

    def test_chroma_threshold_filtering(self):
        """Test chroma threshold filtering"""
        estimator = InkEstimator(random_seed=42)

        # Low chroma image (grayscale)
        img_gray = np.full((100, 100, 3), 128, dtype=np.uint8)
        samples_gray, _ = estimator.sample_ink_pixels(img_gray, chroma_thresh=10.0, L_dark_thresh=10.0)

        # High chroma image (colorful)
        img_color = np.zeros((100, 100, 3), dtype=np.uint8)
        img_color[:, :] = [100, 200, 50]  # Vivid green (BGR)
        samples_color, _ = estimator.sample_ink_pixels(img_color, chroma_thresh=10.0, L_dark_thresh=10.0)

        # Colorful image should have more samples than gray image
        assert len(samples_color) > len(samples_gray)

        # Gray image with high chroma threshold should have few or no samples
        # (since gray has low chroma and L > L_dark_thresh)
        assert len(samples_gray) < 100  # Should be very few

    def test_black_ink_preservation(self):
        """Test that dark pixels are preserved even with low chroma"""
        estimator = InkEstimator(random_seed=42)

        # Black region (low L, low chroma)
        img = np.full((100, 100, 3), 255, dtype=np.uint8)  # White background
        img[25:75, 25:75] = [10, 10, 10]  # Very dark black region (BGR)

        samples, _ = estimator.sample_ink_pixels(img, chroma_thresh=6.0, L_dark_thresh=45.0)

        assert len(samples) > 0  # Black pixels should be detected
        assert np.mean(samples[:, 0]) < 20  # Mean L should be very low

        # Verify black pixels were captured despite low chroma
        # (they pass the L <= L_dark_thresh condition)
        min_L = np.min(samples[:, 0])
        assert min_L < 10  # At least some very dark pixels


class TestInkEstimatorClustering:
    """Test GMM clustering and BIC optimization"""

    def test_select_k_clusters_single_ink(self):
        """Test k selection for single ink"""
        estimator = InkEstimator(random_seed=42)

        # Generate samples from single Gaussian (tight cluster)
        np.random.seed(42)
        samples = np.random.randn(1000, 3) * 3 + np.array([50, 10, -5])
        samples = samples.astype(np.float32)

        gmm, bic = estimator.select_k_clusters(samples, k_min=1, k_max=3)

        assert gmm is not None
        assert len(gmm.means_) in [1, 2, 3]  # Should select 1-3 clusters

        # For single tight cluster, k=1 should be preferred (lowest BIC)
        # But we allow k=2 due to noise
        assert len(gmm.means_) <= 2  # Should not choose k=3 for single cluster

        # BIC should be finite (not inf or nan)
        assert np.isfinite(bic) or bic == 0.0  # 0.0 for KMeans fallback

    def test_select_k_clusters_multiple_inks(self):
        """Test k selection for multiple inks"""
        estimator = InkEstimator(random_seed=42)

        # Generate samples from 3 well-separated Gaussians (3 inks)
        np.random.seed(42)
        samples1 = np.random.randn(300, 3) * 2 + np.array([30, 5, -10])  # Dark ink
        samples2 = np.random.randn(300, 3) * 2 + np.array([60, 15, 5])  # Mid ink
        samples3 = np.random.randn(300, 3) * 2 + np.array([85, 25, 15])  # Light ink
        samples = np.vstack([samples1, samples2, samples3]).astype(np.float32)

        gmm, bic = estimator.select_k_clusters(samples, k_min=1, k_max=3)

        assert gmm is not None
        # BIC should prefer 3 clusters (but might select 2 due to noise)
        assert len(gmm.means_) >= 2

        # Verify cluster centers are reasonably separated
        means = gmm.means_
        if len(means) >= 2:
            # Calculate min distance between clusters
            min_dist = float("inf")
            for i in range(len(means)):
                for j in range(i + 1, len(means)):
                    dist = np.linalg.norm(means[i] - means[j])
                    min_dist = min(min_dist, dist)
            # Clusters should be well-separated
            assert min_dist > 10  # At least 10 units apart in Lab space


class TestInkEstimatorMixingCorrection:
    """Test mixing correction logic (Linearity Check)"""

    def test_mixing_correction_applied(self):
        """Test that mixing correction triggers for collinear clusters"""
        estimator = InkEstimator(random_seed=42)

        # 3 collinear centers (Dark -> Mid -> Light on L axis)
        # Mid point is on the line between Dark and Light
        centers = np.array(
            [
                [30.0, 5.0, -5.0],  # Dark
                [55.0, 10.0, 0.0],  # Mid (mixing point - nearly collinear)
                [80.0, 15.0, 5.0],  # Light
            ],
            dtype=np.float32,
        )

        weights = np.array([0.33, 0.34, 0.33], dtype=np.float32)

        new_centers, new_weights, corrected = estimator.correct_ink_count_by_mixing(
            centers, weights, linearity_thresh=5.0
        )

        assert corrected is True  # Should detect mixing
        assert len(new_centers) == 2  # Should merge to 2 inks

        # Verify weights are redistributed
        assert len(new_weights) == 2
        assert np.isclose(np.sum(new_weights), 1.0)  # Total weight should be 1.0

        # Verify new centers are Dark and Light (Mid removed)
        assert new_centers.shape == (2, 3)
        # First center should be close to Dark
        assert np.linalg.norm(new_centers[0] - centers[0]) < 0.1
        # Second center should be close to Light
        assert np.linalg.norm(new_centers[1] - centers[2]) < 0.1

    def test_mixing_correction_not_applied(self):
        """Test that mixing correction does NOT trigger for non-collinear clusters"""
        estimator = InkEstimator(random_seed=42)

        # 3 non-collinear centers (triangle in Lab space)
        # Mid point is far from the line between Dark and Light
        centers = np.array(
            [
                [30.0, 5.0, -10.0],  # Dark
                [55.0, 25.0, 5.0],  # Mid - Far from line (large perpendicular distance)
                [80.0, 10.0, 15.0],  # Light
            ],
            dtype=np.float32,
        )

        weights = np.array([0.33, 0.34, 0.33], dtype=np.float32)

        new_centers, new_weights, corrected = estimator.correct_ink_count_by_mixing(
            centers, weights, linearity_thresh=5.0
        )

        assert corrected is False  # Should NOT detect mixing
        assert len(new_centers) == 3  # Should keep 3 inks

        # Verify centers and weights are unchanged
        assert np.allclose(new_centers, centers)
        assert np.allclose(new_weights, weights)


class TestInkEstimatorEdgeCases:
    """Test edge cases and error handling"""

    def test_insufficient_pixels(self):
        """Test handling of images with too few ink pixels"""
        estimator = InkEstimator(random_seed=42)

        # Nearly empty image (all white)
        img = np.full((50, 50, 3), 255, dtype=np.uint8)  # White background (BGR)

        result = estimator.estimate_from_array(img)

        # Should handle gracefully - either 0 inks or minimal detection
        assert result["ink_count"] >= 0
        assert isinstance(result["ink_count"], int)

        # If no pixels detected, inks array should be empty
        if result["ink_count"] == 0:
            assert len(result["inks"]) == 0

    def test_trimmed_mean_robustness(self):
        """Test that trimmed mean removes outliers"""
        estimator = InkEstimator(random_seed=42)

        # Array with outliers (20 samples total)
        arr = np.array(
            [
                [50, 10, 5],
                [52, 11, 6],
                [51, 10, 5],
                [49, 9, 4],
                [51, 10, 5],
                [50, 10, 5],
                [52, 11, 6],
                [51, 10, 5],
                [49, 9, 4],
                [50, 10, 5],
                # Outliers (will be trimmed)
                [200, 100, 50],  # High outlier
                [200, 100, 50],
                [5, -50, -30],  # Low outlier
                [5, -50, -30],
                [50, 10, 5],
                [51, 10, 5],
                [49, 9, 4],
                [50, 10, 5],
                [52, 11, 6],
                [51, 10, 5],
            ],
            dtype=np.float32,
        )

        trimmed = estimator.trimmed_mean(arr, trim_ratio=0.2)

        # Result should be close to main cluster (around 50, 10, 5)
        # trim_ratio=0.2 removes top/bottom 20% per channel
        assert 48 < trimmed[0] < 54  # L
        assert 8 < trimmed[1] < 12  # a
        assert 3 < trimmed[2] < 7  # b

        # Verify trimmed mean is more robust than regular mean
        regular_mean = np.mean(arr, axis=0)
        # Regular mean should be affected by outliers (L > 50)
        assert regular_mean[0] > trimmed[0] or regular_mean[0] < trimmed[0] - 2


class TestInkEstimatorIntegration:
    """Integration tests with real images"""

    def test_estimate_from_array_basic(self):
        """Test estimate_from_array with synthetic image"""
        import cv2

        estimator = InkEstimator(random_seed=42)

        # Create synthetic 2-ink image (Blue and Red circles)
        img = np.full((200, 200, 3), 255, dtype=np.uint8)  # White background

        # Blue circle (left)
        cv2.circle(img, (70, 100), 40, (200, 50, 50), -1)  # BGR: Blue-ish

        # Red circle (right)
        cv2.circle(img, (130, 100), 40, (50, 50, 200), -1)  # BGR: Red-ish

        result = estimator.estimate_from_array(img)

        # Verify result structure
        assert "ink_count" in result
        assert "inks" in result
        assert "meta" in result

        # Should detect 1-2 inks (depending on color separation)
        assert result["ink_count"] >= 1
        assert result["ink_count"] <= 3  # Max 3 with current GMM settings

        # Verify ink structure
        if result["ink_count"] > 0:
            ink0 = result["inks"][0]
            assert "lab" in ink0
            assert "rgb" in ink0
            assert "hex" in ink0
            assert "weight" in ink0

            # Verify Lab ranges (lab is a tuple)
            L, a, b = ink0["lab"]
            assert 0 <= L <= 100
            assert -128 <= a <= 127
            assert -128 <= b <= 127

            # Verify RGB ranges
            R, G, B = ink0["rgb"]
            assert 0 <= R <= 255
            assert 0 <= G <= 255
            assert 0 <= B <= 255

            # Verify hex format
            assert ink0["hex"].startswith("#")
            assert len(ink0["hex"]) == 7  # #RRGGBB

            # Verify weight
            assert 0.0 <= ink0["weight"] <= 1.0

        # Verify meta info
        assert "sample_count" in result["meta"]
        assert "sampling_config" in result["meta"]
        assert result["meta"]["sample_count"] > 0

    def test_estimate_from_real_image(self):
        """Test estimation on real lens image"""
        from pathlib import Path

        import cv2

        # Use one of the actual test images (path relative to project root)
        test_image_path = Path(__file__).parent.parent / "data" / "raw_images" / "SKU001_NG_001.jpg"

        if not test_image_path.exists():
            pytest.skip(f"Test image not found: {test_image_path}")

        estimator = InkEstimator(random_seed=42)
        img = cv2.imread(str(test_image_path))

        if img is None:
            pytest.skip(f"Failed to load image: {test_image_path}")

        result = estimator.estimate_from_array(img)

        # Should detect at least 1 ink
        assert result["ink_count"] >= 1
        assert len(result["inks"]) == result["ink_count"]

        # Verify all inks have valid properties
        total_weight = 0.0
        for ink in result["inks"]:
            assert "lab" in ink
            assert "rgb" in ink
            assert "hex" in ink
            assert "weight" in ink

            # Lab ranges (lab is a tuple)
            L, a, b = ink["lab"]
            assert 0 <= L <= 100
            assert -128 <= a <= 127
            assert -128 <= b <= 127

            # RGB ranges
            R, G, B = ink["rgb"]
            assert 0 <= R <= 255
            assert 0 <= G <= 255
            assert 0 <= B <= 255

            # Hex format
            assert ink["hex"].startswith("#")
            assert len(ink["hex"]) == 7

            # Weight
            assert 0.0 < ink["weight"] <= 1.0
            total_weight += ink["weight"]

        # Total weight should be approximately 1.0
        assert 0.95 <= total_weight <= 1.05

        # Verify meta info
        assert result["meta"]["sample_count"] > 100  # Should have reasonable sample size
        assert "sampling_config" in result["meta"]

    def test_estimate_parameter_sensitivity(self):
        """Test that different parameters produce different results"""
        import cv2

        # Create synthetic gradient image
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        for i in range(200):
            color_val = int(i * 255 / 200)
            img[i, :] = [color_val, color_val // 2, 255 - color_val]

        # Test with different chroma thresholds
        estimator1 = InkEstimator(random_seed=42)
        result1 = estimator1.estimate_from_array(img, chroma_thresh=5.0)

        estimator2 = InkEstimator(random_seed=42)
        result2 = estimator2.estimate_from_array(img, chroma_thresh=15.0)

        # Higher chroma threshold should result in fewer samples
        # (but ink_count may vary due to GMM)
        assert result2["meta"]["sample_count"] <= result1["meta"]["sample_count"]

        # Both should detect at least 1 ink
        assert result1["ink_count"] >= 1
        assert result2["ink_count"] >= 1


# TODO: Add parametrized tests for different parameter combinations
# TODO: Add tests for estimate() file path version
# TODO: Add performance benchmarks
