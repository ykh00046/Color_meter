"""
Color Mask Generation for Per-Ink Color Separation

This module generates per-color binary masks from multi-color lens images using
k-means clustering. Masks are used for training and inspection of per-color models.

Key Features:
- Automatic color identification via k-means clustering
- Stable color ID assignment (ordered by L* channel - dark to light)
- Binary mask generation in polar coordinates
- Color metadata extraction (Lab centroid, hex color, role)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..geometry.lens_geometry import LensGeometry, detect_lens_circle
from ..signature.radial_signature import to_polar
from ..utils import bgr_to_lab
from .ink_segmentation import kmeans_segment
from .preprocess import build_roi_mask, sample_ink_candidates


def lab_to_hex(lab: np.ndarray) -> str:
    """
    Convert Lab color to hex string for display.

    Args:
        lab: Lab color as [L, a, b] array

    Returns:
        Hex color string (e.g., "#2E241F")
    """
    # Lab to XYZ to RGB conversion (simplified, D65 illuminant)
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])

    # Lab to XYZ
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    def _f_inv(t):
        delta = 6.0 / 29.0
        if t > delta:
            return t**3
        return 3 * delta**2 * (t - 4.0 / 29.0)

    # D65 white point
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    X = Xn * _f_inv(fx)
    Y = Yn * _f_inv(fy)
    Z = Zn * _f_inv(fz)

    # XYZ to RGB (sRGB)
    R = X * 3.2406 + Y * -1.5372 + Z * -0.4986
    G = X * -0.9689 + Y * 1.8758 + Z * 0.0415
    B = X * 0.0557 + Y * -0.2040 + Z * 1.0570

    # Gamma correction
    def _gamma(c):
        if c <= 0.0031308:
            return 12.92 * c
        return 1.055 * (c ** (1.0 / 2.4)) - 0.055

    R = _gamma(R)
    G = _gamma(G)
    B = _gamma(B)

    # Clamp to [0, 1] and convert to [0, 255]
    R = max(0, min(1, R))
    G = max(0, min(1, G))
    B = max(0, min(1, B))

    r = int(round(R * 255))
    g = int(round(G * 255))
    b_int = int(round(B * 255))

    return f"#{r:02X}{g:02X}{b_int:02X}"


def assign_cluster_labels_to_image(
    lab_map: np.ndarray,
    cluster_centers: np.ndarray,
    l_weight: float = 0.3,
) -> np.ndarray:
    """
    Assign each pixel in lab_map to nearest cluster center.

    Args:
        lab_map: Lab image in polar coordinates (T, R, 3)
        cluster_centers: K-means centers in feature space (k, 3) = [a, b, L*weight]
        l_weight: Weight for L channel in feature space

    Returns:
        Label map (T, R) with cluster indices [0, k-1]
    """
    T, R, _ = lab_map.shape

    # Build feature map: [a, b, L*weight]
    a_map = lab_map[..., 1]
    b_map = lab_map[..., 2]
    L_map = lab_map[..., 0] * l_weight

    feat_map = np.stack([a_map, b_map, L_map], axis=-1).astype(np.float32)  # (T, R, 3)
    feat_flat = feat_map.reshape(-1, 3)  # (T*R, 3)

    # Compute distances to all centers
    # dist[i, j] = ||feat_flat[i] - center[j]||^2
    dists = np.sum((feat_flat[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :]) ** 2, axis=2)  # (T*R, k)

    # Assign to nearest center
    labels_flat = np.argmin(dists, axis=1).astype(np.int32)  # (T*R,)
    labels = labels_flat.reshape(T, R)  # (T, R)

    return labels


def build_color_masks(
    test_bgr: np.ndarray,
    cfg: Dict[str, Any],
    expected_k: int,
    geom: Optional[LensGeometry] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Generate per-color binary masks using k-means segmentation.

    This function segments the input image into k colors using k-means clustering
    on Lab color space samples, then creates binary masks for each color in polar
    coordinates. Color IDs are assigned stably by sorting on L* (dark to light).

    Args:
        test_bgr: Input BGR image
        cfg: Configuration dict with v2_ink and polar settings
        expected_k: Expected number of ink colors
        geom: Optional pre-computed lens geometry (if None, will detect)

    Returns:
        Tuple of:
        - masks: Dict mapping color_id to binary mask (T, R) in polar coords
          Example: {"color_0": mask_array, "color_1": mask_array, ...}
        - metadata: Dict with color information and diagnostics
          Example: {
              "colors": [
                  {
                      "color_id": "color_0",
                      "lab_centroid": [40.3, 131.9, 134.0],
                      "hex_ref": "#2E241F",
                      "area_ratio": 0.42,
                      "role": "ink"  # or "gap"
                  },
                  ...
              ],
              "k_expected": 3,
              "k_used": 3,
              "segmentation_method": "kmeans",
              "warnings": []
          }
    """
    v2_cfg = cfg.get("v2_ink", {})
    warnings: List[str] = []

    # 1. Detect lens geometry and convert to polar
    if geom is None:
        geom = detect_lens_circle(test_bgr)

    polar_R = int(cfg["polar"]["R"])
    polar_T = int(cfg["polar"]["T"])
    polar = to_polar(test_bgr, geom, R=polar_R, T=polar_T)
    lab_map = bgr_to_lab(polar).astype(np.float32)  # (T, R, 3)

    # 2. Build ROI mask and sample ink candidates
    r_start = float(v2_cfg.get("roi_r_start", cfg["anomaly"]["r_start"]))
    r_end = float(v2_cfg.get("roi_r_end", cfg["anomaly"]["r_end"]))
    center_excluded_frac = float(cfg["anomaly"]["center_frac"])

    roi_mask, roi_meta = build_roi_mask(
        polar_T,
        polar_R,
        r_start,
        r_end,
        center_excluded_frac=center_excluded_frac,
    )

    rng_seed = v2_cfg.get("rng_seed", None)
    samples, sample_indices, sample_meta, sample_warnings = sample_ink_candidates(
        lab_map, roi_mask, cfg, rng_seed=rng_seed
    )
    warnings.extend(sample_warnings)

    if samples.size == 0 or samples.shape[0] < expected_k:
        # Insufficient samples, return empty masks
        empty_masks = {f"color_{i}": np.zeros((polar_T, polar_R), dtype=bool) for i in range(expected_k)}
        empty_metadata = {
            "colors": [],
            "k_expected": expected_k,
            "k_used": 0,
            "segmentation_method": "kmeans",
            "warnings": warnings + ["COLOR_SEGMENTATION_FAILED"],
            "roi_meta": roi_meta,
            "sample_meta": sample_meta,
        }
        return empty_masks, empty_metadata

    # 3. Run k-means clustering on samples
    k_used = expected_k
    # Special case: if expected_k=1, cluster with k=2 to separate ink from gap
    if expected_k == 1:
        k_used = 2

    l_weight = float(v2_cfg.get("l_weight", 0.3))
    kmeans_attempts = int(v2_cfg.get("kmeans_attempts", 10))

    labels_samples, centers = kmeans_segment(
        samples,
        k_used,
        l_weight=l_weight,
        attempts=kmeans_attempts,
        rng_seed=rng_seed,
    )

    if labels_samples.size == 0 or centers.size == 0:
        # Clustering failed
        empty_masks = {f"color_{i}": np.zeros((polar_T, polar_R), dtype=bool) for i in range(expected_k)}
        empty_metadata = {
            "colors": [],
            "k_expected": expected_k,
            "k_used": 0,
            "segmentation_method": "kmeans",
            "warnings": warnings + ["COLOR_SEGMENTATION_FAILED"],
            "roi_meta": roi_meta,
            "sample_meta": sample_meta,
        }
        return empty_masks, empty_metadata

    # 4. Convert centers from feature space [a, b, L*weight] back to Lab
    # centers shape: (k, 3) where columns are [a, b, L*l_weight]
    centers_lab = np.zeros_like(centers)
    centers_lab[:, 1] = centers[:, 0]  # a
    centers_lab[:, 2] = centers[:, 1]  # b
    centers_lab[:, 0] = centers[:, 2] / l_weight  # L = (L*weight) / weight

    # 5. Assign all pixels in lab_map to nearest cluster
    label_map = assign_cluster_labels_to_image(lab_map, centers, l_weight=l_weight)  # (T, R)

    # 6. Build cluster metadata (before sorting)
    cluster_meta_unsorted = []
    for cluster_idx in range(k_used):
        cluster_mask = label_map == cluster_idx
        area_ratio = float(cluster_mask.sum()) / (polar_T * polar_R)

        # Lab centroid for this cluster (from centers)
        lab_centroid = centers_lab[cluster_idx]

        cluster_meta_unsorted.append(
            {
                "original_idx": cluster_idx,
                "lab_centroid": lab_centroid.tolist(),
                "L": float(lab_centroid[0]),
                "area_ratio": area_ratio,
                "hex_ref": lab_to_hex(lab_centroid),
            }
        )

    # 7. Sort clusters by L* (dark to light) for stable color_id assignment
    cluster_meta_sorted = sorted(cluster_meta_unsorted, key=lambda x: x["L"])

    # 8. Assign roles: "ink" vs "gap"
    # If expected_k=1 and k_used=2, assign darkest as "ink", other as "gap"
    # Otherwise, mark all as "ink"
    if expected_k == 1 and k_used == 2:
        cluster_meta_sorted[0]["role"] = "ink"  # Darkest
        cluster_meta_sorted[1]["role"] = "gap"
    else:
        for cm in cluster_meta_sorted:
            cm["role"] = "ink"

    # 9. Build color masks with stable IDs
    color_masks = {}
    colors_metadata = []

    for color_id, cluster_meta in enumerate(cluster_meta_sorted):
        original_idx = cluster_meta["original_idx"]
        mask = label_map == original_idx

        color_masks[f"color_{color_id}"] = mask

        colors_metadata.append(
            {
                "color_id": f"color_{color_id}",
                "lab_centroid": cluster_meta["lab_centroid"],
                "hex_ref": cluster_meta["hex_ref"],
                "area_ratio": cluster_meta["area_ratio"],
                "role": cluster_meta["role"],
            }
        )

    # 10. Build final metadata
    metadata = {
        "colors": colors_metadata,
        "k_expected": expected_k,
        "k_used": k_used,
        "segmentation_method": "kmeans",
        "l_weight": l_weight,
        "polar": {"order": "TR", "T": int(polar_T), "R": int(polar_R)},
        "warnings": warnings,
        "roi_meta": roi_meta,
        "sample_meta": sample_meta,
        "geom": {
            "cx": float(geom.cx),
            "cy": float(geom.cy),
            "r": float(geom.r),
        },
    }

    return color_masks, metadata


def filter_masks_by_role(
    color_masks: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    role: str = "ink",
) -> Dict[str, np.ndarray]:
    """
    Filter color masks to only include specified role (e.g., "ink" only).

    Args:
        color_masks: Dict of color masks
        metadata: Metadata from build_color_masks()
        role: Role to filter ("ink" or "gap")

    Returns:
        Filtered dict of color masks
    """
    filtered_masks = {}
    for color_info in metadata["colors"]:
        if color_info["role"] == role:
            color_id = color_info["color_id"]
            if color_id in color_masks:
                filtered_masks[color_id] = color_masks[color_id]
    return filtered_masks


def visualize_color_masks(
    polar_bgr: np.ndarray,
    color_masks: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
) -> np.ndarray:
    """
    Visualize color masks as overlay on polar image.

    Args:
        polar_bgr: Polar BGR image (T, R, 3)
        color_masks: Dict of color masks from build_color_masks()
        metadata: Metadata from build_color_masks()

    Returns:
        Visualization image (T, R, 3) with colored overlays
    """
    vis = polar_bgr.copy()

    # Generate distinct colors for each mask
    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]

    for idx, color_info in enumerate(metadata["colors"]):
        color_id = color_info["color_id"]
        if color_id not in color_masks:
            continue

        mask = color_masks[color_id]
        color = colors[idx % len(colors)]

        # Blend overlay
        overlay = np.zeros_like(vis)
        overlay[mask] = color
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

    return vis
