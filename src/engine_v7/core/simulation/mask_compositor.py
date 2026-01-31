"""
Mask-based Color Compositor

This module implements pixel-level color synthesis using actual segmentation masks
instead of area_ratio-based scalar mixing. This approach correctly handles:
- Overlap regions (multiple inks at same location)
- Radial contribution variations (zone-specific density)
- Actual pixel distribution patterns

Part of Direction A from Longterm Roadmap.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils import to_cie_lab


def composite_from_masks(
    lab_map: np.ndarray,
    color_masks: Dict[str, np.ndarray],
    downsample: int = 4,
    reduce: str = "trimmed_mean",
) -> Dict[str, Any]:
    """
    Compute composite color from actual pixel sampling using segmentation masks.

    This is the core implementation of mask-based pixel synthesis (Direction A).
    Instead of mixing colors based on area_ratio, we:
    1. Create a union of all color masks
    2. Sample actual pixels from the Lab map
    3. Compute statistics with proper weighting

    Args:
       lab_map: Polar Lab image (T, R, 3) in CIE scale
        color_masks: Dict of {color_id: mask} where mask is (T, R) boolean
        downsample: Downsample factor for performance (1=no downsample)
        reduce: Reduction method ("mean", "trimmed_mean", "median")

    Returns:
        {
            "composite_lab": [L, a, b],  # Final composite color
            "overlap": {
                "ratio": float,          # Fraction of pixels with multiple inks
                "regions": [...],        # List of overlapping color combinations
            },
            "zone_contributions": {      # Radial zone analysis
                "inner": float,          # 0.0-0.33 contribution
                "middle": float,         # 0.33-0.66 contribution
                "outer": float,          # 0.66-1.0 contribution
            },
            "n_pixels_sampled": int,
            "confidence": float,         # Based on coverage and variance
        }

    Example:
        >>> lab_polar = to_polar(test_bgr, geom, R=260, T=720)
        >>> lab_map = to_cie_lab(lab_polar, source="bgr")
        >>> masks = {"color_0": mask0, "color_1": mask1}
        >>> result = composite_from_masks(lab_map, masks, downsample=4)
        >>> print(result["composite_lab"])  # [L*, a*, b*]
    """
    if lab_map.size == 0 or not color_masks:
        return {
            "composite_lab": [0.0, 0.0, 0.0],
            "overlap": {"ratio": 0.0, "regions": []},
            "zone_contributions": {"inner": 0.0, "middle": 0.0, "outer": 0.0},
            "n_pixels_sampled": 0,
            "confidence": 0.0,
        }

    T, R = lab_map.shape[:2]

    # Downsample for performance if requested
    if downsample > 1:
        lab_ds = lab_map[::downsample, ::downsample]
        masks_ds = {cid: mask[::downsample, ::downsample] for cid, mask in color_masks.items()}
    else:
        lab_ds = lab_map
        masks_ds = color_masks

    # Create union mask and overlap detection
    union_mask = np.zeros(lab_ds.shape[:2], dtype=bool)
    overlap_count = np.zeros(lab_ds.shape[:2], dtype=np.int32)

    for mask in masks_ds.values():
        union_mask |= mask
        overlap_count += mask.astype(np.int32)

    # Detect overlap regions
    overlap_mask = overlap_count > 1
    overlap_ratio = float(overlap_mask.sum() / max(union_mask.sum(), 1))

    # Identify overlap combinations
    overlap_regions = []
    if overlap_ratio > 0.01:  # Only compute if significant overlap
        for i, (cid1, mask1) in enumerate(masks_ds.items()):
            for cid2, mask2 in list(masks_ds.items())[i + 1 :]:
                overlap_area = (mask1 & mask2).sum()
                if overlap_area > 0:
                    overlap_regions.append(
                        {
                            "colors": [cid1, cid2],
                            "n_pixels": int(overlap_area),
                        }
                    )

    # Sample pixels from union
    if union_mask.sum() == 0:
        return {
            "composite_lab": [0.0, 0.0, 0.0],
            "overlap": {"ratio": 0.0, "regions": []},
            "zone_contributions": {"inner": 0.0, "middle": 0.0, "outer": 0.0},
            "n_pixels_sampled": 0,
            "confidence": 0.0,
        }

    pixels = lab_ds[union_mask]  # (N, 3)

    # Compute composite using specified reduction method
    if reduce == "trimmed_mean":
        # Remove top/bottom 10% to reduce outlier influence
        if len(pixels) >= 10:
            pixels_sorted = np.sort(pixels, axis=0)
            n_trim = max(1, len(pixels) // 10)
            pixels_trimmed = pixels_sorted[n_trim:-n_trim]
            composite_lab = pixels_trimmed.mean(axis=0)
        else:
            composite_lab = pixels.mean(axis=0)
    elif reduce == "median":
        composite_lab = np.median(pixels, axis=0)
    else:  # mean
        composite_lab = pixels.mean(axis=0)

    # Radial zone contribution analysis
    zone_contributions = _compute_zone_contributions(lab_ds, masks_ds, union_mask)

    # Confidence based on coverage and variance
    coverage = union_mask.sum() / union_mask.size
    variance = np.var(pixels, axis=0).mean()  # Average variance across L, a, b
    confidence = min(1.0, coverage) * (1.0 / (1.0 + variance / 100.0))

    return {
        "composite_lab": [float(composite_lab[0]), float(composite_lab[1]), float(composite_lab[2])],
        "overlap": {
            "ratio": overlap_ratio,
            "regions": overlap_regions,
        },
        "zone_contributions": zone_contributions,
        "n_pixels_sampled": int(len(pixels)),
        "confidence": float(confidence),
    }


def _compute_zone_contributions(
    lab_map: np.ndarray,
    masks: Dict[str, np.ndarray],
    union_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Analyze radial contribution by zone (inner/middle/outer).

    Returns fraction of pixels in each radial zone.
    """
    T, R = lab_map.shape[:2]

    # Create radial coordinate map
    r_coords = np.arange(R)[None, :] / R  # (1, R) → values 0.0 to 1.0
    r_map = np.broadcast_to(r_coords, (T, R))  # (T, R)

    # Define zones
    inner_mask = (r_map < 0.33) & union_mask
    middle_mask = ((r_map >= 0.33) & (r_map < 0.66)) & union_mask
    outer_mask = (r_map >= 0.66) & union_mask

    total = union_mask.sum()
    if total == 0:
        return {"inner": 0.0, "middle": 0.0, "outer": 0.0}

    return {
        "inner": float(inner_mask.sum() / total),
        "middle": float(middle_mask.sum() / total),
        "outer": float(outer_mask.sum() / total),
    }


def compare_methods(
    lab_map: np.ndarray,
    color_masks: Dict[str, np.ndarray],
    color_centroids: Dict[str, np.ndarray],
    area_ratios: Dict[str, float],
) -> Dict[str, Any]:
    """
    Compare mask-based vs area_ratio-based methods for validation.

    Args:
        lab_map: Polar Lab image (T, R, 3)
        color_masks: Dict of {color_id: mask}
        color_centroids: Dict of {color_id: [L, a, b]}
        area_ratios: Dict of {color_id: ratio}

    Returns:
        {
            "mask_based": {...},
            "area_ratio": {...},
            "delta_e": float,  # Difference between methods
        }
    """
    # Mask-based method
    mask_result = composite_from_masks(lab_map, color_masks, downsample=4)

    # Area ratio method (legacy)
    area_composite = np.zeros(3)
    for cid, centroid in color_centroids.items():
        ratio = area_ratios.get(cid, 0.0)
        area_composite += np.array(centroid) * ratio

    # Compute difference
    delta_e = _cie2000_delta_e(mask_result["composite_lab"], area_composite)

    return {
        "mask_based": mask_result,
        "area_ratio": {
            "composite_lab": [float(area_composite[0]), float(area_composite[1]), float(area_composite[2])],
        },
        "delta_e": float(delta_e),
        "improvement": "mask_based" if delta_e > 2.0 else "similar",
    }


def _cie2000_delta_e(lab1: List[float], lab2: np.ndarray) -> float:
    """Simplified CIEDE2000 (just Euclidean for now, can upgrade)."""
    lab1_arr = np.array(lab1)
    return float(np.linalg.norm(lab1_arr - lab2))
