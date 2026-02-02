"""
Hungarian Matching Integration for Color Masks (P1)

Extracted from color_masks.py.
Provides label stabilization by matching detected clusters to reference
(STD) colors using Hungarian algorithm.

Usage:
    from .color_masks_stabilize import stabilize_labels_with_reference
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...config_norm import get_polar_dims
from ...geometry.lens_geometry import LensGeometry, detect_lens_circle
from ...signature.radial_signature import to_polar
from ...utils import to_cie_lab


def stabilize_labels_with_reference(
    color_masks: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    reference_centroids: np.ndarray,
    *,
    reference_ids: Optional[List[str]] = None,
    max_distance_threshold: float = 50.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Stabilize cluster labels by matching to reference (STD) colors.

    Uses Hungarian algorithm for optimal one-to-one assignment between
    detected clusters and reference colors. This prevents "label swap"
    issues across different images of the same product.

    Args:
        color_masks: Dict of cluster masks from build_color_masks()
        metadata: Metadata from build_color_masks()
        reference_centroids: (M, 3) Lab centroids of reference colors
        reference_ids: Optional list of reference color IDs (e.g., ["ink_0", "ink_1"])
        max_distance_threshold: Maximum deltaE for valid match (default 50)

    Returns:
        Tuple of (stabilized_masks, stabilized_metadata) with labels
        reordered to match reference

    Example:
        # Load STD model centroids
        std_centroids = np.array([[30, -10, -40], [70, -30, 20]])  # [dark_blue, green]

        # Build masks (order may vary)
        masks, meta = build_color_masks(test_bgr, cfg, expected_k=2)

        # Stabilize labels to match STD order
        stable_masks, stable_meta = stabilize_labels_with_reference(
            masks, meta, std_centroids,
            reference_ids=["dark_blue", "green"]
        )
    """
    from ..matching.hungarian_matcher import (
        match_clusters_to_reference,
        reorder_clusters_by_reference,
        stabilize_cluster_labels,
    )

    # Extract detected centroids from metadata
    colors_info = metadata.get("colors", [])
    if not colors_info:
        # No colors detected - return as is
        return color_masks, metadata

    detected_centroids = []
    for c in colors_info:
        # Prefer CIE Lab centroid
        if c.get("lab_centroid_cie"):
            detected_centroids.append(c["lab_centroid_cie"])
        elif c.get("lab_centroid"):
            # Convert from CV8 to CIE if needed
            lab = c["lab_centroid"]
            # Heuristic: if L is in 0-255 range, convert
            if lab[0] > 100:
                detected_centroids.append(
                    [
                        lab[0] * 100 / 255,
                        lab[1] - 128,
                        lab[2] - 128,
                    ]
                )
            else:
                detected_centroids.append(lab)
        else:
            detected_centroids.append([50, 0, 0])  # Fallback

    detected_centroids = np.array(detected_centroids, dtype=np.float64)
    reference_centroids = np.asarray(reference_centroids, dtype=np.float64)

    # Run Hungarian matching
    stabilized_masks, stabilized_colors, stabilization_info = stabilize_cluster_labels(
        detected_centroids,
        reference_centroids,
        color_masks,
        colors_info,
        reference_ids=reference_ids,
        max_distance_threshold=max_distance_threshold,
    )

    # Update metadata
    stabilized_metadata = copy.deepcopy(metadata)
    stabilized_metadata["colors"] = stabilized_colors
    stabilized_metadata["label_stabilization"] = stabilization_info
    stabilized_metadata["label_order"] = "reference_matched"

    # Add warnings if any
    if stabilization_info.get("match_result", {}).get("n_matched", 0) < len(reference_centroids):
        stabilized_metadata.setdefault("warnings", []).append("LABEL_MATCH_INCOMPLETE")

    return stabilized_masks, stabilized_metadata


def build_color_masks_multi_source(
    test_bgr: np.ndarray,
    black_bgr: np.ndarray,
    cfg: Dict[str, Any],
    expected_k: int,
    geom: Optional[LensGeometry] = None,
    plate_kpis: Optional[Dict[str, Any]] = None,
    sample_mask_override: Optional[np.ndarray] = None,
    white_clusters_meta: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, Any]]]:
    """
    Generate alternative color segmentations from diff (white-black) and black images.

    The diff image approximates transmittance by removing backlight contribution.
    The black image captures light scattered/emitted through the lens on a dark field.

    After clustering, clusters are reordered to match the white-based cluster order
    using greedy Lab-distance matching.

    Args:
        test_bgr: White backlight image (BGR)
        black_bgr: Black backlight image (BGR)
        cfg: Configuration dict
        expected_k: Expected number of ink colors
        geom: Optional pre-computed lens geometry
        plate_kpis: Optional plate KPIs for soft gate
        sample_mask_override: Optional mask for hard gate
        white_clusters_meta: Color metadata from white clustering for order matching

    Returns:
        Dict with keys "diff" and "black", each mapping to (masks, metadata) tuple
    """
    from .color_masks import build_color_masks_with_retry

    if geom is None:
        geom = detect_lens_circle(test_bgr)

    polar_R, polar_T = get_polar_dims(cfg)

    # Convert both images to polar with the same geometry
    white_polar = to_polar(test_bgr, geom, R=polar_R, T=polar_T)
    black_polar = to_polar(black_bgr, geom, R=polar_R, T=polar_T)

    # Diff image in polar space: clip(white - black, 0, 255) -> transmittance-proportional
    diff_polar = np.clip(white_polar.astype(np.int16) - black_polar.astype(np.int16), 0, 255).astype(np.uint8)

    results: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, Any]]] = {}

    # 1. Diff-based clustering (polar override -> skip geometry detection + polar conversion)
    try:
        diff_masks, diff_meta = build_color_masks_with_retry(
            test_bgr,
            cfg,
            expected_k,
            geom=geom,
            plate_kpis=plate_kpis,
            sample_mask_override=sample_mask_override,
            _polar_override=diff_polar,
        )
        diff_meta["source"] = "diff_white_minus_black"
        results["diff"] = (diff_masks, diff_meta)
    except Exception as e:
        results["diff"] = (
            {f"color_{i}": np.zeros((polar_T, polar_R), dtype=bool) for i in range(expected_k)},
            {"colors": [], "error": str(e), "source": "diff_white_minus_black"},
        )

    # 2. Black-based clustering (polar override -> skip geometry detection + polar conversion)
    try:
        black_masks, black_meta = build_color_masks_with_retry(
            test_bgr,
            cfg,
            expected_k,
            geom=geom,
            plate_kpis=plate_kpis,
            sample_mask_override=sample_mask_override,
            _polar_override=black_polar,
        )
        black_meta["source"] = "black_backlight"
        results["black"] = (black_masks, black_meta)
    except Exception as e:
        results["black"] = (
            {f"color_{i}": np.zeros((polar_T, polar_R), dtype=bool) for i in range(expected_k)},
            {"colors": [], "error": str(e), "source": "black_backlight"},
        )

    # 3. Reorder clusters to match white clustering order (greedy Lab matching)
    if white_clusters_meta:
        white_centroids = []
        for c in white_clusters_meta:
            if c.get("lab_centroid_cie") is not None:
                white_centroids.append(np.array(c["lab_centroid_cie"], dtype=np.float64))
            elif c.get("centroid_lab_cie") is not None:
                white_centroids.append(np.array(c["centroid_lab_cie"], dtype=np.float64))
            elif c.get("centroid_lab") is not None:
                white_centroids.append(np.array(c["centroid_lab"], dtype=np.float64))
            else:
                white_centroids.append(np.array([50.0, 0.0, 0.0]))

        for source_key in ("diff", "black"):
            masks, meta = results[source_key]
            alt_colors = meta.get("colors", [])
            if not alt_colors or len(alt_colors) != len(white_centroids):
                continue

            # Extract alt centroids
            alt_centroids = []
            for c in alt_colors:
                if c.get("lab_centroid_cie") is not None:
                    alt_centroids.append(np.array(c["lab_centroid_cie"], dtype=np.float64))
                else:
                    alt_centroids.append(np.array([50.0, 0.0, 0.0]))

            # Greedy matching: for each white cluster, find closest unmatched alt cluster
            n = len(white_centroids)
            used = set()
            order = [0] * n
            for wi in range(n):
                best_dist = float("inf")
                best_ai = 0
                for ai in range(n):
                    if ai in used:
                        continue
                    d = float(np.sqrt(np.sum((white_centroids[wi] - alt_centroids[ai]) ** 2)))
                    if d < best_dist:
                        best_dist = d
                        best_ai = ai
                used.add(best_ai)
                order[wi] = best_ai

            # Reorder masks and colors
            reordered_masks = {}
            reordered_colors = []
            for new_idx, old_idx in enumerate(order):
                old_color_id = f"color_{old_idx}"
                new_color_id = f"color_{new_idx}"
                if old_color_id in masks:
                    reordered_masks[new_color_id] = masks[old_color_id]
                if old_idx < len(alt_colors):
                    c = dict(alt_colors[old_idx])
                    c["color_id"] = new_color_id
                    reordered_colors.append(c)

            meta["colors"] = reordered_colors
            results[source_key] = (reordered_masks, meta)

    return results


def build_color_masks_with_reference(
    test_bgr: np.ndarray,
    cfg: Dict[str, Any],
    reference_centroids: np.ndarray,
    *,
    geom: Optional[LensGeometry] = None,
    reference_ids: Optional[List[str]] = None,
    max_distance_threshold: float = 50.0,
    polar_alpha: Optional[np.ndarray] = None,
    polar_lab: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Build color masks with automatic label stabilization to reference.

    This is a convenience wrapper that combines:
    1. build_color_masks_with_retry() - clustering and mask generation
    2. stabilize_labels_with_reference() - Hungarian matching to reference
    3. compute_cluster_effective_densities() - alpha-based density (optional)

    Args:
        test_bgr: Input BGR image
        cfg: Configuration dictionary
        reference_centroids: (M, 3) Lab centroids of reference (STD) colors
        geom: Optional pre-computed lens geometry
        reference_ids: Optional list of reference color IDs
        max_distance_threshold: Maximum deltaE for valid match
        polar_alpha: Optional alpha map for effective density
        **kwargs: Additional arguments passed to build_color_masks_with_retry

    Returns:
        Tuple of (masks, metadata) with labels matched to reference order
    """
    from .color_masks import build_color_masks_with_retry
    from .color_masks_density import compute_cluster_effective_densities

    expected_k = len(reference_centroids)

    # Step 1: Build color masks
    masks, metadata = build_color_masks_with_retry(test_bgr, cfg, expected_k, geom, **kwargs)

    # Step 2: Stabilize labels to reference
    masks, metadata = stabilize_labels_with_reference(
        masks,
        metadata,
        reference_centroids,
        reference_ids=reference_ids,
        max_distance_threshold=max_distance_threshold,
    )

    # Step 3: Compute effective densities (optional)
    if polar_alpha is not None:
        alpha_cfg = kwargs.get("alpha_cfg")
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
