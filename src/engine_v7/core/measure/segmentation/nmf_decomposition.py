"""
NMF-based Color Decomposition for Ink Analysis

This module uses Non-negative Matrix Factorization (NMF) to decompose
mixed colors into their constituent ink components.

Key Concepts:
- NMF decomposes image V ≈ W × H where:
  - V: (pixels × 3) Lab color data
  - W: (pixels × k) ink composition ratios per pixel (soft segmentation)
  - H: (k × 3) learned ink colors

Benefits over clustering:
1. Soft segmentation: pixels can partially belong to multiple inks
2. Physically meaningful: non-negativity matches ink mixing model
3. Gradient handling: quantifies mixing ratios in transition zones
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# sklearn NMF import with fallback
try:
    from sklearn.decomposition import NMF

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class NMFResult:
    """Result of NMF color decomposition."""

    # Soft segmentation: (H, W, k) - ink composition ratio per pixel
    soft_masks: np.ndarray

    # Learned ink colors: (k, 3) in Lab CIE scale
    ink_colors_cie: np.ndarray

    # Learned ink colors: (k, 3) in Lab CV8 scale
    ink_colors_cv8: np.ndarray

    # Hard segmentation: (H, W) - dominant ink index per pixel
    hard_labels: np.ndarray

    # Reconstruction error (lower is better)
    reconstruction_error: float

    # Mixing analysis
    mixing_ratios: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    n_components: int = 0
    iterations: int = 0


def _lab_cv8_to_cie(lab_cv8: np.ndarray) -> np.ndarray:
    """Convert Lab from CV8 scale (0-255) to CIE scale."""
    lab_cie = np.zeros_like(lab_cv8, dtype=np.float32)
    lab_cie[..., 0] = lab_cv8[..., 0] * 100.0 / 255.0  # L: 0-255 -> 0-100
    lab_cie[..., 1] = lab_cv8[..., 1] - 128.0  # a: 0-255 -> -128~127
    lab_cie[..., 2] = lab_cv8[..., 2] - 128.0  # b: 0-255 -> -128~127
    return lab_cie


def _lab_cie_to_cv8(lab_cie: np.ndarray) -> np.ndarray:
    """Convert Lab from CIE scale to CV8 scale (0-255)."""
    lab_cv8 = np.zeros_like(lab_cie, dtype=np.float32)
    lab_cv8[..., 0] = lab_cie[..., 0] * 255.0 / 100.0  # L: 0-100 -> 0-255
    lab_cv8[..., 1] = lab_cie[..., 1] + 128.0  # a: -128~127 -> 0-255
    lab_cv8[..., 2] = lab_cie[..., 2] + 128.0  # b: -128~127 -> 0-255
    return np.clip(lab_cv8, 0, 255)


def nmf_decompose_colors(
    lab_map: np.ndarray,
    n_components: int,
    mask: Optional[np.ndarray] = None,
    max_iter: int = 300,
    init: str = "nndsvda",
    l_weight: float = 1.0,
    rng_seed: Optional[int] = None,
) -> NMFResult:
    """
    Decompose Lab image into ink components using NMF.

    Args:
        lab_map: (H, W, 3) Lab image in CV8 scale
        n_components: Number of ink components to extract
        mask: Optional (H, W) boolean mask for valid pixels
        max_iter: Maximum NMF iterations
        init: NMF initialization method ('nndsvda', 'random', 'nndsvd')
        l_weight: Weight for L channel in feature space (0.3-1.0)
        rng_seed: Random seed for reproducibility

    Returns:
        NMFResult with soft masks, ink colors, and mixing analysis
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn is required for NMF. Install with: pip install scikit-learn")

    img_H, img_W, _ = lab_map.shape
    img_H, img_W = int(img_H), int(img_W)  # Ensure Python int for np.zeros
    n_components = int(n_components)  # Ensure Python int

    # Convert to CIE scale for better numerical properties
    lab_cie = _lab_cv8_to_cie(lab_map.astype(np.float32))

    # Apply mask if provided
    if mask is not None:
        valid_pixels = lab_cie[mask]
        n_valid = valid_pixels.shape[0]
    else:
        valid_pixels = lab_cie.reshape(-1, 3)
        n_valid = img_H * img_W
        mask = np.ones((img_H, img_W), dtype=bool)

    if n_valid < n_components:
        # Not enough pixels
        return NMFResult(
            soft_masks=np.zeros((img_H, img_W, n_components), dtype=np.float32),
            ink_colors_cie=np.zeros((n_components, 3), dtype=np.float32),
            ink_colors_cv8=np.zeros((n_components, 3), dtype=np.float32),
            hard_labels=np.zeros((img_H, img_W), dtype=np.int32),
            reconstruction_error=float("inf"),
            n_components=int(n_components),
        )

    # Build feature matrix with L weighting
    # NMF requires non-negative values, so shift Lab to positive range
    features = np.zeros((n_valid, 3), dtype=np.float64)
    features[:, 0] = valid_pixels[:, 0] * l_weight  # L (already 0-100)
    features[:, 1] = valid_pixels[:, 1] + 128.0  # a: shift to 0-255
    features[:, 2] = valid_pixels[:, 2] + 128.0  # b: shift to 0-255

    # Ensure non-negative (clip any numerical errors)
    features = np.clip(features, 0, None)

    # Run NMF
    nmf = NMF(
        n_components=n_components,
        init=init,
        max_iter=max_iter,
        random_state=rng_seed if rng_seed is not None else 42,
        solver="cd",  # Coordinate descent (faster)
        beta_loss="frobenius",
    )

    try:
        W = nmf.fit_transform(features)  # (n_valid, k) - pixel compositions
        H_matrix = nmf.components_  # (k, 3) - ink colors in feature space

        # Normalize W to sum to 1 per pixel (proper mixing ratios)
        W_sum = W.sum(axis=1, keepdims=True)
        W_sum[W_sum == 0] = 1.0  # Avoid division by zero
        W_normalized = W / W_sum

        # Compute ink colors from weighted average of original pixels
        # (NMF basis vectors don't directly correspond to physical Lab values)
        ink_colors_cie = np.zeros((n_components, 3), dtype=np.float32)
        for i in range(n_components):
            weights = W_normalized[:, i : i + 1]  # (n_valid, 1)
            weight_sum = weights.sum()
            if weight_sum > 0:
                # Weighted average of original Lab CIE values
                weighted_lab = (valid_pixels * weights).sum(axis=0) / weight_sum
                ink_colors_cie[i] = weighted_lab

        # Convert to CV8
        ink_colors_cv8 = _lab_cie_to_cv8(ink_colors_cie)

        # Build soft mask image (img_H, img_W, k)
        soft_masks = np.zeros((img_H, img_W, n_components), dtype=np.float32)
        soft_masks[mask] = W_normalized

        # Hard segmentation (dominant ink per pixel)
        hard_labels = np.zeros((img_H, img_W), dtype=np.int32)
        hard_labels[mask] = np.argmax(W_normalized, axis=1)

        # Calculate reconstruction error
        reconstruction = W @ H_matrix
        reconstruction_error = float(np.mean((features - reconstruction) ** 2))

        # Analyze mixing
        mixing_ratios = _analyze_mixing(W_normalized, n_components)

        return NMFResult(
            soft_masks=soft_masks,
            ink_colors_cie=ink_colors_cie,
            ink_colors_cv8=ink_colors_cv8,
            hard_labels=hard_labels,
            reconstruction_error=reconstruction_error,
            mixing_ratios=mixing_ratios,
            n_components=n_components,
            iterations=nmf.n_iter_,
        )

    except Exception as e:
        logger.warning(f"NMF decomposition failed: {e}")
        return NMFResult(
            soft_masks=np.zeros((img_H, img_W, n_components), dtype=np.float32),
            ink_colors_cie=np.zeros((n_components, 3), dtype=np.float32),
            ink_colors_cv8=np.zeros((n_components, 3), dtype=np.float32),
            hard_labels=np.zeros((img_H, img_W), dtype=np.int32),
            reconstruction_error=float("inf"),
            n_components=n_components,
        )


def _analyze_mixing(W: np.ndarray, n_components: int) -> Dict[str, Any]:
    """
    Analyze mixing patterns from NMF weight matrix.

    Args:
        W: (n_pixels, k) normalized weight matrix
        n_components: Number of components

    Returns:
        Dict with mixing statistics
    """
    # Calculate dominance (how clearly each pixel belongs to one ink)
    max_weights = np.max(W, axis=1)
    dominance_mean = float(np.mean(max_weights))
    dominance_std = float(np.std(max_weights))

    # Find mixed pixels (no clear dominant ink)
    mixed_threshold = 0.7  # If max weight < 0.7, it's mixed
    mixed_ratio = float(np.mean(max_weights < mixed_threshold))

    # Calculate per-component statistics
    component_stats = []
    for i in range(n_components):
        comp_weights = W[:, i]
        dominant_pixels = np.sum(np.argmax(W, axis=1) == i)

        component_stats.append(
            {
                "component_id": i,
                "mean_weight": float(np.mean(comp_weights)),
                "max_weight": float(np.max(comp_weights)),
                "dominant_pixel_count": int(dominant_pixels),
                "dominant_pixel_ratio": float(dominant_pixels / len(W)),
            }
        )

    # Calculate pairwise mixing
    pairwise_mixing = []
    for i in range(n_components):
        for j in range(i + 1, n_components):
            # Find pixels where both components have significant weight
            both_significant = (W[:, i] > 0.2) & (W[:, j] > 0.2)
            mixing_count = int(np.sum(both_significant))

            if mixing_count > 0:
                pairwise_mixing.append(
                    {
                        "components": [i, j],
                        "mixing_pixel_count": mixing_count,
                        "mixing_ratio": float(mixing_count / len(W)),
                    }
                )

    return {
        "dominance_mean": dominance_mean,
        "dominance_std": dominance_std,
        "mixed_pixel_ratio": mixed_ratio,
        "component_stats": component_stats,
        "pairwise_mixing": pairwise_mixing,
    }


def soft_to_hard_masks(
    soft_masks: np.ndarray,
    threshold: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Convert soft NMF masks to hard binary masks.

    Args:
        soft_masks: (H, W, k) soft segmentation from NMF
        threshold: Minimum weight to include in mask (0 = argmax only)

    Returns:
        Dict mapping color_id to binary mask
    """
    H, W, k = soft_masks.shape
    masks = {}

    if threshold <= 0:
        # Simple argmax assignment
        hard_labels = np.argmax(soft_masks, axis=2)
        for i in range(k):
            masks[f"color_{i}"] = hard_labels == i
    else:
        # Threshold-based assignment (pixel can belong to multiple inks)
        for i in range(k):
            masks[f"color_{i}"] = soft_masks[:, :, i] >= threshold

    return masks


def estimate_ink_coverage(
    soft_masks: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Estimate ink coverage from soft NMF masks.

    Unlike hard segmentation, this accounts for partial coverage
    in mixed/gradient regions.

    Args:
        soft_masks: (H, W, k) soft segmentation
        roi_mask: Optional (H, W) mask for valid region

    Returns:
        List of coverage info per ink component
    """
    H, W, k = soft_masks.shape

    if roi_mask is not None:
        # Weight by ROI
        weights = roi_mask.astype(np.float32)
        total_weight = weights.sum()
    else:
        weights = np.ones((H, W), dtype=np.float32)
        total_weight = H * W

    coverage = []
    for i in range(k):
        # Weighted sum of soft mask
        soft_coverage = np.sum(soft_masks[:, :, i] * weights)

        # Hard coverage (dominant pixels only)
        hard_mask = np.argmax(soft_masks, axis=2) == i
        hard_coverage = np.sum(hard_mask * weights)

        coverage.append(
            {
                "component_id": i,
                "soft_coverage_ratio": float(soft_coverage / total_weight),
                "hard_coverage_ratio": float(hard_coverage / total_weight),
                "coverage_difference": float((soft_coverage - hard_coverage) / total_weight),
            }
        )

    return coverage


def nmf_refine_clusters(
    lab_map: np.ndarray,
    initial_labels: np.ndarray,
    n_components: int,
    mask: Optional[np.ndarray] = None,
    max_iter: int = 200,
    rng_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Refine initial clustering results using NMF.

    This combines the speed of K-Means/GMM initialization with
    the soft segmentation benefits of NMF.

    Args:
        lab_map: (H, W, 3) Lab image in CV8 scale
        initial_labels: (H, W) initial cluster labels from K-Means/GMM
        n_components: Number of components
        mask: Optional (H, W) boolean mask
        max_iter: NMF iterations
        rng_seed: Random seed

    Returns:
        Tuple of (refined_labels, soft_masks, metadata)
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn is required for NMF")

    H, W, _ = lab_map.shape

    # Use initial labels to compute initial ink colors (for NMF initialization)
    initial_colors = []
    for i in range(n_components):
        cluster_mask = initial_labels == i
        if np.any(cluster_mask):
            mean_color = np.mean(lab_map[cluster_mask], axis=0)
            initial_colors.append(mean_color)
        else:
            initial_colors.append(np.array([128, 128, 128]))  # Gray default

    initial_colors = np.array(initial_colors, dtype=np.float32)

    # Run NMF with custom initialization
    result = nmf_decompose_colors(
        lab_map,
        n_components=n_components,
        mask=mask,
        max_iter=max_iter,
        init="nndsvda",  # Use default init (custom H init not directly supported)
        rng_seed=rng_seed,
    )

    # Build metadata
    meta = {
        "initial_colors": initial_colors.tolist(),
        "nmf_colors": result.ink_colors_cv8.tolist(),
        "reconstruction_error": result.reconstruction_error,
        "mixing_ratios": result.mixing_ratios,
        "iterations": result.iterations,
    }

    return result.hard_labels, result.soft_masks, meta


def visualize_nmf_decomposition(
    soft_masks: np.ndarray,
    ink_colors_cv8: np.ndarray,
) -> np.ndarray:
    """
    Visualize NMF decomposition as a reconstructed color image.

    Args:
        soft_masks: (H, W, k) soft segmentation
        ink_colors_cv8: (k, 3) ink colors in CV8 scale

    Returns:
        (H, W, 3) reconstructed Lab image in CV8 scale
    """
    H, W, k = soft_masks.shape

    # Weighted sum of ink colors
    reconstructed = np.zeros((H, W, 3), dtype=np.float32)

    for i in range(k):
        # Add contribution of this ink
        weight = soft_masks[:, :, i : i + 1]  # (H, W, 1)
        color = ink_colors_cv8[i]  # (3,)
        reconstructed += weight * color

    return np.clip(reconstructed, 0, 255).astype(np.uint8)
