"""
Effective Density Integration for Color Masks (P0)

Extracted from color_masks.py.
Integrates the alpha_density module with color mask output to compute
effective density = area_ratio * alpha for each cluster.

Usage:
    from .color_masks_density import compute_cluster_effective_densities
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...config_norm import get_polar_dims
from ...geometry.lens_geometry import LensGeometry, detect_lens_circle
from ...signature.radial_signature import to_polar
from ...utils import to_cie_lab


def compute_cluster_effective_densities(
    color_masks: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    polar_alpha: Optional[np.ndarray] = None,
    polar_lab: Optional[np.ndarray] = None,
    alpha_weight_map: Optional[np.ndarray] = None,
    alpha_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute effective densities for all clusters using 3-tier alpha fallback.

    This function integrates the alpha_density module with color_masks output.
    Effective density = area_ratio * alpha (where alpha uses L1/L2/L3 fallback).

    Args:
        color_masks: Dict of cluster masks from build_color_masks()
        metadata: Metadata from build_color_masks()
        polar_alpha: Optional alpha map in polar coordinates (T, R)
                     If None, effective_density = area_ratio (alpha=1.0)
        alpha_cfg: Optional configuration for alpha computation

    Returns:
        Tuple of:
        - updated_metadata: Original metadata with effective_density added to each color
        - alpha_summary: Detailed alpha analysis summary

    Example:
        masks, meta = build_color_masks(test_bgr, cfg, expected_k)
        updated_meta, alpha_summary = compute_cluster_effective_densities(
            masks, meta, polar_alpha=plate_result["alpha_polar"]
        )
        # meta["colors"][i] now has "effective_density", "alpha_used", "fallback_level"
    """
    from ..metrics.alpha_density import AlphaDensityResult, compute_effective_density, extract_alpha_summary

    # Extract area_ratios from metadata
    area_ratios = {}
    for color_info in metadata.get("colors", []):
        color_id = color_info["color_id"]
        area_ratios[color_id] = color_info.get("area_ratio", 0.0)

    # Priority 1: Map plate_lite per-ink alpha to cluster color_ids by area_ratio ordering
    # plate_lite inks are pre-sorted by area_ratio desc in single_analyzer
    effective_cfg = dict(alpha_cfg) if alpha_cfg else {}
    pl_inks = effective_cfg.pop("_plate_lite_inks", None)
    if pl_inks and area_ratios:
        # Sort clusters by area_ratio descending to align with plate_lite ink ordering
        sorted_clusters = sorted(area_ratios.items(), key=lambda x: x[1], reverse=True)
        pl_candidates = {}
        for idx, (color_id, _) in enumerate(sorted_clusters):
            if idx < len(pl_inks):
                pl_alpha = pl_inks[idx].get("alpha_mean")
                if pl_alpha is not None and pl_alpha > 0.01:
                    pl_candidates[color_id] = float(pl_alpha)
        if pl_candidates:
            effective_cfg["_plate_lite_alpha_candidates"] = pl_candidates
    else:
        effective_cfg = effective_cfg

    # Compute effective densities
    result: AlphaDensityResult = compute_effective_density(
        polar_alpha=polar_alpha,
        cluster_masks=color_masks,
        area_ratios=area_ratios,
        alpha_weight_map=alpha_weight_map,
        polar_lab=polar_lab,
        cfg=effective_cfg,
    )

    # Update metadata with effective densities
    updated_metadata = copy.deepcopy(metadata)
    for color_info in updated_metadata.get("colors", []):
        color_id = color_info["color_id"]
        if color_id in result.clusters:
            cluster_result = result.clusters[color_id]
            color_info["effective_density"] = round(cluster_result.effective_density, 4)
            color_info["alpha_used"] = round(cluster_result.alpha_used, 4)
            color_info["alpha_fallback_level"] = cluster_result.fallback_level.value
            color_info["alpha_fallback_reason"] = cluster_result.fallback_reason

    # Add summary to metadata
    quality_gate_debug = result.config_used.get("_quality_gate_debug", {})
    updated_metadata["alpha_analysis"] = {
        "global_alpha": round(result.global_alpha, 4),
        "global_alpha_std": round(result.global_alpha_std, 4),
        "quality_gate": quality_gate_debug,
        "warnings": result.warnings,
    }

    # Extend existing warnings
    updated_metadata.setdefault("warnings", []).extend(result.warnings)

    # Generate detailed summary
    alpha_summary = extract_alpha_summary(result)

    return updated_metadata, alpha_summary


def build_color_masks_with_alpha(
    test_bgr: np.ndarray,
    cfg: Dict[str, Any],
    expected_k: int,
    geom: Optional[LensGeometry] = None,
    polar_alpha: Optional[np.ndarray] = None,
    polar_lab: Optional[np.ndarray] = None,
    alpha_cfg: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Build color masks with integrated effective density calculation.

    This is a convenience wrapper that combines:
    1. build_color_masks_with_retry() - clustering and mask generation
    2. compute_cluster_effective_densities() - alpha-based density calculation

    Args:
        test_bgr: Input BGR image
        cfg: Configuration dictionary
        expected_k: Expected number of ink colors
        geom: Optional pre-computed lens geometry
        polar_alpha: Optional alpha map in polar coordinates (T, R)
        alpha_cfg: Optional configuration for alpha computation
        **kwargs: Additional arguments passed to build_color_masks_with_retry

    Returns:
        Tuple of (masks, metadata) where metadata includes effective_density
    """
    from .color_masks import build_color_masks_with_retry

    # Step 1: Build color masks
    masks, metadata = build_color_masks_with_retry(test_bgr, cfg, expected_k, geom, **kwargs)

    # Step 2: Compute effective densities
    if polar_alpha is not None or kwargs.get("compute_effective_density", True):
        need_polar_lab = (
            polar_lab is None
            and isinstance(alpha_cfg, dict)
            and bool(alpha_cfg.get("transition_weights_enabled", False))
        )
        if need_polar_lab:
            if geom is None:
                geom = detect_lens_circle(test_bgr)
            polar_R, polar_T = get_polar_dims(cfg)
            polar_lab = to_cie_lab(to_polar(test_bgr, geom, R=polar_R, T=polar_T))
        metadata, alpha_summary = compute_cluster_effective_densities(
            masks,
            metadata,
            polar_alpha,
            polar_lab=polar_lab,
            alpha_cfg=alpha_cfg,
        )
        metadata["alpha_summary"] = alpha_summary

    return masks, metadata


def build_alpha_from_plate_images(
    white_bgr: np.ndarray,
    black_bgr: np.ndarray,
    cfg: Dict[str, Any],
    geom: Optional[LensGeometry] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build alpha map in polar coordinates from white/black plate images.

    This is a helper function to generate polar_alpha for use with
    compute_cluster_effective_densities().

    Args:
        white_bgr: White backlight image
        black_bgr: Black backlight image
        cfg: Configuration dictionary
        geom: Optional pre-computed lens geometry

    Returns:
        Tuple of:
        - polar_alpha: Alpha map in polar coordinates (T, R)
        - alpha_meta: Metadata about alpha computation
    """
    from ...geometry.lens_geometry import detect_lens_circle
    from ..metrics.alpha_density import build_alpha_map_polar

    # Detect geometry if not provided
    if geom is None:
        geom = detect_lens_circle(white_bgr)

    polar_R, polar_T = get_polar_dims(cfg)

    # Build alpha map
    polar_alpha = build_alpha_map_polar(
        white_bgr,
        black_bgr,
        geom,
        polar_R,
        polar_T,
    )

    alpha_meta = {
        "polar_R": polar_R,
        "polar_T": polar_T,
        "alpha_mean": float(np.mean(polar_alpha)),
        "alpha_std": float(np.std(polar_alpha)),
        "alpha_min": float(np.min(polar_alpha)),
        "alpha_max": float(np.max(polar_alpha)),
    }

    return polar_alpha, alpha_meta
