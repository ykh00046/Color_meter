"""
Transition Region Detection and Weighting Module

This module provides detection and handling of transition (boundary) regions
in polar coordinate space. Transition regions are areas where colors blend
or change rapidly, which can introduce noise in alpha measurements.

Key Features:
- Gradient-based transition detection in polar coordinates
- Configurable weight maps for transition exclusion/down-weighting
- Integration with alpha computation for cleaner measurements

Usage:
    from .transition_detector import (
        compute_transition_weights,
        TransitionConfig,
    )

    weights = compute_transition_weights(polar_lab, masks, cfg)
    # Use weights in alpha computation to down-weight boundary pixels
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TransitionConfig:
    """Configuration for transition region detection."""

    # Gradient threshold for transition detection
    gradient_threshold: float = 10.0  # Lab units

    # Morphological operations
    dilation_radius: int = 3  # Pixels to expand transition region
    erosion_radius: int = 1  # Pixels to shrink before dilation

    # Weight values
    transition_weight: float = 0.3  # Weight for transition pixels (0-1)
    core_weight: float = 1.0  # Weight for core (non-transition) pixels

    # Smoothing
    smooth_weights: bool = True
    smooth_sigma: float = 2.0

    # Per-channel weights for gradient magnitude
    L_weight: float = 1.0
    a_weight: float = 1.5  # Higher weight for chromatic changes
    b_weight: float = 1.5


@dataclass
class TransitionResult:
    """Result of transition detection."""

    # Weight map (T, R) with values [transition_weight, core_weight]
    weight_map: np.ndarray

    # Binary transition mask (T, R) - True = transition region
    transition_mask: np.ndarray

    # Gradient magnitude map (T, R)
    gradient_magnitude: np.ndarray

    # Statistics
    transition_ratio: float  # Fraction of pixels in transition
    mean_gradient: float
    max_gradient: float

    # Per-cluster transition ratios
    cluster_transition_ratios: Dict[str, float]

    # Config used
    config: TransitionConfig


def compute_lab_gradient_polar(
    polar_lab: np.ndarray,
    *,
    L_weight: float = 1.0,
    a_weight: float = 1.5,
    b_weight: float = 1.5,
) -> np.ndarray:
    """
    Compute gradient magnitude in polar Lab image.

    The gradient is computed along both theta (rows) and radius (cols)
    directions, with configurable channel weights.

    Args:
        polar_lab: Lab image in polar coordinates (T, R, 3)
        L_weight: Weight for L channel gradient
        a_weight: Weight for a channel gradient
        b_weight: Weight for b channel gradient

    Returns:
        Gradient magnitude map (T, R)
    """
    T, R, _ = polar_lab.shape

    # Compute gradients along theta (row) and radius (col) directions
    # Use Sobel for better noise handling
    grad_theta_L = cv2.Sobel(polar_lab[:, :, 0], cv2.CV_32F, 1, 0, ksize=3)
    grad_theta_a = cv2.Sobel(polar_lab[:, :, 1], cv2.CV_32F, 1, 0, ksize=3)
    grad_theta_b = cv2.Sobel(polar_lab[:, :, 2], cv2.CV_32F, 1, 0, ksize=3)

    grad_r_L = cv2.Sobel(polar_lab[:, :, 0], cv2.CV_32F, 0, 1, ksize=3)
    grad_r_a = cv2.Sobel(polar_lab[:, :, 1], cv2.CV_32F, 0, 1, ksize=3)
    grad_r_b = cv2.Sobel(polar_lab[:, :, 2], cv2.CV_32F, 0, 1, ksize=3)

    # Weighted gradient magnitude per channel
    grad_L = np.sqrt(grad_theta_L**2 + grad_r_L**2) * L_weight
    grad_a = np.sqrt(grad_theta_a**2 + grad_r_a**2) * a_weight
    grad_b = np.sqrt(grad_theta_b**2 + grad_r_b**2) * b_weight

    # Combined magnitude (Euclidean)
    gradient_magnitude = np.sqrt(grad_L**2 + grad_a**2 + grad_b**2)

    return gradient_magnitude.astype(np.float32)


def detect_transition_mask(
    gradient_magnitude: np.ndarray,
    threshold: float,
    *,
    dilation_radius: int = 3,
    erosion_radius: int = 1,
) -> np.ndarray:
    """
    Detect transition regions from gradient magnitude.

    Args:
        gradient_magnitude: Gradient magnitude map (T, R)
        threshold: Gradient threshold for transition detection
        dilation_radius: Radius to expand transition regions
        erosion_radius: Radius to erode before dilation (removes noise)

    Returns:
        Binary mask where True = transition region
    """
    # Threshold to get initial transition mask
    transition_mask = gradient_magnitude > threshold

    # Morphological operations to clean up and expand
    if erosion_radius > 0:
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius * 2 + 1, erosion_radius * 2 + 1))
        transition_mask = cv2.erode(transition_mask.astype(np.uint8), kernel_erode, iterations=1) > 0

    if dilation_radius > 0:
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))
        transition_mask = cv2.dilate(transition_mask.astype(np.uint8), kernel_dilate, iterations=1) > 0

    return transition_mask


def compute_transition_weights(
    polar_lab: np.ndarray,
    cluster_masks: Optional[Dict[str, np.ndarray]] = None,
    cfg: Optional[TransitionConfig] = None,
) -> TransitionResult:
    """
    Compute transition region weights for alpha computation.

    This function detects boundary/transition regions where colors blend
    and creates a weight map that down-weights these regions in alpha
    calculations for more stable measurements.

    Args:
        polar_lab: Lab image in polar coordinates (T, R, 3)
        cluster_masks: Optional dict of cluster masks for per-cluster stats
        cfg: TransitionConfig or None for defaults

    Returns:
        TransitionResult with weight map and statistics
    """
    if cfg is None:
        cfg = TransitionConfig()

    T, R, _ = polar_lab.shape

    # Compute gradient magnitude
    gradient_magnitude = compute_lab_gradient_polar(
        polar_lab,
        L_weight=cfg.L_weight,
        a_weight=cfg.a_weight,
        b_weight=cfg.b_weight,
    )

    # Detect transition regions
    transition_mask = detect_transition_mask(
        gradient_magnitude,
        cfg.gradient_threshold,
        dilation_radius=cfg.dilation_radius,
        erosion_radius=cfg.erosion_radius,
    )

    # Create weight map
    weight_map = np.full((T, R), cfg.core_weight, dtype=np.float32)
    weight_map[transition_mask] = cfg.transition_weight

    # Optional smoothing for gradual weight transitions
    if cfg.smooth_weights and cfg.smooth_sigma > 0:
        weight_map = cv2.GaussianBlur(
            weight_map,
            (0, 0),
            cfg.smooth_sigma,
        )
        # Re-clamp to valid range
        weight_map = np.clip(weight_map, cfg.transition_weight, cfg.core_weight)

    # Compute statistics
    transition_ratio = float(np.mean(transition_mask))
    mean_gradient = float(np.mean(gradient_magnitude))
    max_gradient = float(np.max(gradient_magnitude))

    # Per-cluster transition ratios
    cluster_transition_ratios: Dict[str, float] = {}
    if cluster_masks:
        for color_id, mask in cluster_masks.items():
            mask_bool = mask.astype(bool)
            if np.any(mask_bool):
                cluster_trans = transition_mask & mask_bool
                ratio = float(np.sum(cluster_trans) / np.sum(mask_bool))
                cluster_transition_ratios[color_id] = round(ratio, 4)
            else:
                cluster_transition_ratios[color_id] = 0.0

    return TransitionResult(
        weight_map=weight_map,
        transition_mask=transition_mask,
        gradient_magnitude=gradient_magnitude,
        transition_ratio=round(transition_ratio, 4),
        mean_gradient=round(mean_gradient, 2),
        max_gradient=round(max_gradient, 2),
        cluster_transition_ratios=cluster_transition_ratios,
        config=cfg,
    )


def compute_cluster_boundary_mask(
    cluster_masks: Dict[str, np.ndarray],
    *,
    boundary_width: int = 3,
) -> np.ndarray:
    """
    Compute boundary mask between adjacent clusters.

    This identifies pixels that are at the boundary between different
    clusters (ink colors), which often have mixed/uncertain values.

    Args:
        cluster_masks: Dict of cluster masks (T, R)
        boundary_width: Width of boundary region in pixels

    Returns:
        Binary mask where True = boundary between clusters
    """
    if not cluster_masks:
        return np.zeros((1, 1), dtype=bool)

    # Get dimensions from first mask
    first_mask = next(iter(cluster_masks.values()))
    T, R = first_mask.shape

    # Create combined label map
    label_map = np.zeros((T, R), dtype=np.int32)
    for i, (color_id, mask) in enumerate(cluster_masks.items()):
        label_map[mask.astype(bool)] = i + 1

    # Find boundaries using morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (boundary_width * 2 + 1, boundary_width * 2 + 1))

    # Dilate each cluster and find overlaps
    boundary_mask = np.zeros((T, R), dtype=bool)

    for i in range(1, len(cluster_masks) + 1):
        cluster_mask = (label_map == i).astype(np.uint8)
        dilated = cv2.dilate(cluster_mask, kernel, iterations=1)
        # Boundary = dilated region minus original
        boundary = (dilated > 0) & (cluster_mask == 0)
        boundary_mask |= boundary

    return boundary_mask


def apply_transition_weights_to_samples(
    samples: np.ndarray,
    sample_coords: np.ndarray,
    weight_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply transition weights to sample values.

    Args:
        samples: Sample values (N,) or (N, C)
        sample_coords: Sample coordinates (N, 2) as (theta_idx, r_idx)
        weight_map: Weight map (T, R)

    Returns:
        (weighted_samples, weights) where weights can be used for
        weighted median/mean calculations
    """
    N = len(samples)
    weights = np.ones(N, dtype=np.float32)

    for i in range(N):
        t_idx, r_idx = int(sample_coords[i, 0]), int(sample_coords[i, 1])
        if 0 <= t_idx < weight_map.shape[0] and 0 <= r_idx < weight_map.shape[1]:
            weights[i] = weight_map[t_idx, r_idx]

    return samples, weights


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute weighted median.

    Args:
        values: Array of values
        weights: Array of weights (same length as values)

    Returns:
        Weighted median value
    """
    if len(values) == 0:
        return np.nan

    # Filter out NaN values
    valid_mask = ~np.isnan(values)
    values = values[valid_mask]
    weights = weights[valid_mask]

    if len(values) == 0:
        return np.nan

    # Sort by value
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Cumulative weight
    cumsum = np.cumsum(sorted_weights)
    total = cumsum[-1]

    if total == 0:
        return float(np.median(values))

    # Find median position
    median_pos = total / 2.0
    idx = np.searchsorted(cumsum, median_pos)

    if idx >= len(sorted_values):
        idx = len(sorted_values) - 1

    return float(sorted_values[idx])


# ==============================================================================
# Integration helpers for alpha_density module
# ==============================================================================


def create_alpha_weight_map(
    polar_lab: np.ndarray,
    cluster_masks: Dict[str, np.ndarray],
    *,
    use_gradient_weights: bool = True,
    use_boundary_weights: bool = True,
    gradient_config: Optional[TransitionConfig] = None,
    boundary_width: int = 3,
    boundary_weight: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create comprehensive weight map for alpha computation.

    Combines gradient-based transition detection with cluster boundary
    detection for optimal noise reduction.

    Args:
        polar_lab: Lab image in polar coordinates (T, R, 3)
        cluster_masks: Dict of cluster masks
        use_gradient_weights: Apply gradient-based transition weights
        use_boundary_weights: Apply cluster boundary weights
        gradient_config: Config for gradient detection
        boundary_width: Width for boundary detection
        boundary_weight: Weight for boundary pixels

    Returns:
        (weight_map, meta) where weight_map is (T, R) and meta contains stats
    """
    T, R, _ = polar_lab.shape
    weight_map = np.ones((T, R), dtype=np.float32)
    meta: Dict[str, Any] = {
        "gradient_transition": None,
        "boundary_transition": None,
    }

    # Apply gradient-based transition weights
    if use_gradient_weights:
        trans_result = compute_transition_weights(polar_lab, cluster_masks, gradient_config)
        weight_map = np.minimum(weight_map, trans_result.weight_map)
        meta["gradient_transition"] = {
            "transition_ratio": trans_result.transition_ratio,
            "mean_gradient": trans_result.mean_gradient,
            "cluster_ratios": trans_result.cluster_transition_ratios,
        }

    # Apply cluster boundary weights
    if use_boundary_weights and len(cluster_masks) > 1:
        boundary_mask = compute_cluster_boundary_mask(cluster_masks, boundary_width=boundary_width)
        boundary_weight_map = np.where(boundary_mask, boundary_weight, 1.0).astype(np.float32)
        weight_map = np.minimum(weight_map, boundary_weight_map)
        meta["boundary_transition"] = {
            "boundary_ratio": float(np.mean(boundary_mask)),
            "boundary_width": boundary_width,
        }

    # Final stats
    meta["mean_weight"] = float(np.mean(weight_map))
    meta["low_weight_ratio"] = float(np.mean(weight_map < 0.5))

    return weight_map, meta
