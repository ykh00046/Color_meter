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

from ...geometry.lens_geometry import LensGeometry, detect_lens_circle
from ...signature.radial_signature import to_polar
from ...utils import bgr_to_lab
from ..metrics.ink_metrics import (
    build_cluster_stats,
    calculate_inkness_score,
    calculate_radial_presence_curve,
    calculate_spatial_prior,
    silhouette_ab_proxy,
)
from .ink_segmentation import kmeans_segment
from .preprocess import build_roi_mask, build_sampling_mask, sample_ink_candidates


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

    # Use build_sampling_mask for better background exclusion
    sampling_mask, sample_meta, sample_warnings = build_sampling_mask(lab_map, roi_mask, cfg, rng_seed=rng_seed)
    warnings.extend(sample_warnings)

    # Extract samples from sampling mask
    samples = lab_map[sampling_mask]

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

    # 4. Build cluster stats for inkness scoring
    separation_d0 = float(v2_cfg.get("separation_d0", 3.0))
    separation_k = float(v2_cfg.get("separation_k", 1.0))
    cluster_stats = build_cluster_stats(
        samples,
        labels_samples,
        k_used,
        separation_d0=separation_d0,
        separation_k=separation_k,
    )

    # 5. Convert centers from feature space [a, b, L*weight] back to Lab
    # centers shape: (k, 3) where columns are [a, b, L*l_weight]
    centers_lab = np.zeros_like(centers)
    centers_lab[:, 1] = centers[:, 0]  # a
    centers_lab[:, 2] = centers[:, 1]  # b
    centers_lab[:, 0] = centers[:, 2] / l_weight  # L = (L*weight) / weight

    # 6. Assign all pixels in lab_map to nearest cluster
    label_map = assign_cluster_labels_to_image(lab_map, centers, l_weight=l_weight)  # (T, R)

    # Build normalized radial map (T, R) for spatial prior
    radial_bins = int(v2_cfg.get("radial_bins", 10))
    radial_bins = max(radial_bins, 1)
    r_vals = (np.arange(polar_R, dtype=np.float32) + 0.5) / max(polar_R, 1)
    polar_r_map = np.tile(r_vals[None, :], (polar_T, 1))
    labels_flat = label_map.reshape(-1)

    # 7. Build cluster metadata (before sorting)
    cluster_meta_unsorted = []
    for cluster_idx in range(k_used):
        cluster_mask = label_map == cluster_idx
        area_ratio = float(cluster_mask.sum()) / (polar_T * polar_R)

        # Lab centroid for this cluster (from centers)
        lab_centroid = centers_lab[cluster_idx]

        # Get stats for this cluster
        stats = cluster_stats["clusters"][cluster_idx]

        radial_presence_curve = calculate_radial_presence_curve(
            labels_flat,
            cluster_idx,
            polar_r_map,
            r_bins=radial_bins,
        )
        spatial_prior = calculate_spatial_prior(radial_presence_curve)

        cluster_meta_unsorted.append(
            {
                "original_idx": cluster_idx,
                "lab_centroid": lab_centroid.tolist(),
                "L": float(lab_centroid[0]),
                "area_ratio": area_ratio,
                "hex_ref": lab_to_hex(lab_centroid),
                "compactness": stats["compactness"],
                "alpha_like": stats["alpha_like"],
                "radial_presence_curve": radial_presence_curve,
                "spatial_prior": spatial_prior,
            }
        )

    auto_estimation = None
    if bool(v2_cfg.get("auto_k_enabled", True)):
        expanded = False
        candidate_set = {
            max(1, k_used - 1),
            k_used,
            max(1, k_used + 1),
        }

        min_area = float(cluster_stats["quality"].get("min_area_ratio", 0.0))
        min_area_warn = float(v2_cfg.get("min_area_ratio_warn", 0.03))
        min_delta = float(cluster_stats["quality"].get("min_deltaE_between_clusters", 0.0))

        if "INK_SEPARATION_LOW_CONFIDENCE" in warnings or min_area < min_area_warn or min_delta < separation_d0:
            expanded = True
            max_k = int(v2_cfg.get("auto_k_expand_max", 4))
            candidate_set.update(range(1, max(1, max_k) + 1))

        k_candidates = sorted(candidate_set)
        scores: Dict[int, float | None] = {}
        for k_cand in k_candidates:
            cand_labels, _ = kmeans_segment(
                samples,
                int(k_cand),
                l_weight=l_weight,
                attempts=kmeans_attempts,
                rng_seed=rng_seed,
            )
            if cand_labels.size == 0:
                scores[int(k_cand)] = None
                continue
            scores[int(k_cand)] = silhouette_ab_proxy(samples, cand_labels, int(k_cand))

        valid = [(k_i, s) for k_i, s in scores.items() if s is not None]
        valid.sort(key=lambda x: x[1], reverse=True)
        best_k = int(valid[0][0]) if valid else None
        best_score = float(valid[0][1]) if valid else 0.0
        second_score = float(valid[1][1]) if len(valid) > 1 else 0.0

        abs_min = float(v2_cfg.get("auto_k_conf_abs_min", 0.10))
        abs_span = float(v2_cfg.get("auto_k_conf_abs_span", 0.25))
        gap_span = float(v2_cfg.get("auto_k_conf_gap_span", 0.10))
        mismatch_thr = float(v2_cfg.get("auto_k_mismatch_conf_thr", 0.70))
        low_conf_thr = float(v2_cfg.get("auto_k_low_conf_thr", 0.40))

        conf_abs = (best_score - abs_min) / max(abs_span, 1e-6)
        conf_abs = float(np.clip(conf_abs, 0.0, 1.0))
        conf_gap = (best_score - second_score) / max(gap_span, 1e-6)
        conf_gap = float(np.clip(conf_gap, 0.0, 1.0))
        confidence = float(0.6 * conf_abs + 0.4 * conf_gap)

        forced_to_expected = bool(best_k is not None and best_k != k_used)

        auto_estimation = {
            "k_candidates": [int(x) for x in k_candidates],
            "metric": "silhouette_ab_proxy",
            "scores": {str(k_i): scores[k_i] for k_i in k_candidates},
            "suggested_k": int(k_used),
            "auto_k_best": best_k,
            "confidence": confidence,
            "forced_to_expected": forced_to_expected,
            "notes": [
                f"expanded_search_used:{str(expanded).lower()}",
                f"forced_to_expected:{str(forced_to_expected).lower()}",
            ],
        }

        if confidence < low_conf_thr:
            warnings.append("AUTO_K_LOW_CONFIDENCE")
        if best_k is not None and best_k != expected_k and confidence >= mismatch_thr:
            if not (expected_k == 1 and k_used == 2):
                warnings.append("INK_COUNT_MISMATCH_SUSPECTED")

    # 7. Sort clusters by L* (dark to light) for stable color_id assignment
    cluster_meta_sorted = sorted(cluster_meta_unsorted, key=lambda x: x["L"])

    # 8. Assign roles (legacy-first, inkness optional)
    role_policy = str(v2_cfg.get("role_policy", "legacy")).lower()
    inkness_threshold = float(v2_cfg.get("inkness_threshold", 0.55))
    enable_background_role = bool(v2_cfg.get("enable_background_role", False))

    for cm in cluster_meta_sorted:
        inkness = calculate_inkness_score(
            mean_lab=cm["lab_centroid"],
            compactness=cm["compactness"],
            alpha_like=cm["alpha_like"],
            spatial_prior=cm["spatial_prior"],
        )
        cm["inkness_score"] = inkness
        cm["role"] = "ink"

    if role_policy == "inkness":
        for cm in cluster_meta_sorted:
            cm["role"] = "ink" if cm["inkness_score"] >= inkness_threshold else "gap"

        if enable_background_role:
            non_ink_clusters = [cm for cm in cluster_meta_sorted if cm["role"] != "ink"]
            if non_ink_clusters:
                lightest_non_ink = max(non_ink_clusters, key=lambda x: x["L"])
                lightest_non_ink["role"] = "background"
    else:
        # legacy policy: expected_k==1 -> darkest ink, others gap; expected_k>1 -> all ink
        if expected_k == 1 and cluster_meta_sorted:
            darkest = cluster_meta_sorted[0]
            darkest["role"] = "ink"
            for cm in cluster_meta_sorted[1:]:
                cm["role"] = "gap"

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
                "inkness_score": cluster_meta.get("inkness_score"),  # Phase 2
                "radial_presence_curve": cluster_meta["radial_presence_curve"],
                "spatial_prior": cluster_meta["spatial_prior"],
            }
        )

    # 10. Build final metadata
    detected_ink_count = sum(1 for c in colors_metadata if c["role"] == "ink")

    metadata = {
        "colors": colors_metadata,
        "expected_ink_count": expected_k,  # Input: user-specified ink count
        "segmentation_k": k_used,  # Parameter: k value used for k-means
        "detected_ink_like_count": detected_ink_count,  # Result: role="ink" count
        # Legacy keys for backward compatibility
        "k_expected": expected_k,
        "k_used": k_used,
        "segmentation_method": "kmeans",
        "l_weight": l_weight,
        "polar": {"order": "TR", "T": int(polar_T), "R": int(polar_R)},
        "warnings": warnings,
        "roi_meta": roi_meta,
        "sample_meta": sample_meta,
        "radial_bins": radial_bins,
        "geom": {
            "cx": float(geom.cx),
            "cy": float(geom.cy),
            "r": float(geom.r),
        },
    }
    if warnings:
        warn_map = {
            "INK_SAMPLING_EMPTY": "sampling",
            "INK_SEPARATION_LOW_CONFIDENCE": "sampling",
            "COLOR_SEGMENTATION_FAILED": "segmentation",
            "INK_CLUSTER_TOO_SMALL": "segmentation",
            "INK_CLUSTER_OVERLAP_HIGH": "segmentation",
            "AUTO_K_LOW_CONFIDENCE": "auto_k",
            "INK_COUNT_MISMATCH_SUSPECTED": "auto_k",
        }
        warnings_by_category = {"sampling": [], "segmentation": [], "auto_k": []}
        for w in warnings:
            category = warn_map.get(w)
            if category:
                warnings_by_category[category].append(w)
        metadata["warnings_by_category"] = warnings_by_category
    if auto_estimation is not None:
        metadata["auto_estimation"] = auto_estimation

    return color_masks, metadata


def calculate_segmentation_confidence(
    metadata: Dict[str, Any],
    expected_k: int,
) -> float:
    """
    Calculate confidence score for color segmentation quality.

    Evaluates:
    1. Inter-cluster distance (separation quality)
    2. Area ratio balance (no extreme imbalance)
    3. Expected vs detected ink count match

    Args:
        metadata: Metadata from build_color_masks()
        expected_k: Expected number of ink colors

    Returns:
        Confidence score [0.0, 1.0] where:
        - 1.0 = perfect segmentation (good separation, balanced areas, count match)
        - 0.5 = uncertain segmentation
        - 0.0 = poor segmentation (clusters merged, severe imbalance)
    """
    colors = metadata.get("colors", [])
    if len(colors) < 2:
        return 0.0  # Need at least 2 clusters

    # 1. Inter-cluster distance (L* separation)
    # Good separation: darkest and lightest clusters far apart
    L_values = [c["lab_centroid"][0] for c in colors]
    L_range = max(L_values) - min(L_values)

    # Normalize to [0, 1]: L_range > 40 is good, < 15 is poor
    separation_score = min(1.0, max(0.0, (L_range - 15.0) / 25.0))

    # 2. Area ratio balance
    # Check if any cluster dominates (> 80%) or too small (< 5%)
    area_ratios = [c["area_ratio"] for c in colors]
    max_ratio = max(area_ratios)
    min_ratio = min(area_ratios)

    # Penalize extreme imbalance
    if max_ratio > 0.85 or min_ratio < 0.03:
        balance_score = 0.3
    elif max_ratio > 0.75 or min_ratio < 0.05:
        balance_score = 0.6
    else:
        balance_score = 1.0

    # 3. Expected vs detected ink count
    detected_inks = sum(1 for c in colors if c["role"] == "ink")
    count_match = detected_inks == expected_k
    count_score = 1.0 if count_match else 0.5

    # Weighted average
    confidence = separation_score * 0.4 + balance_score * 0.3 + count_score * 0.3

    return round(float(confidence), 3)


def build_color_masks_with_retry(
    test_bgr: np.ndarray,
    cfg: Dict[str, Any],
    expected_k: int,
    geom: Optional[LensGeometry] = None,
    confidence_threshold: float = 0.7,
    enable_retry: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Build color masks with 2-pass retry logic for better ink separation.

    Two-pass strategy:
    1. First pass: k = expected_k
    2. If confidence low or ink count mismatch: retry with k = expected_k + 1
    3. Choose better result based on confidence and ink count match

    Args:
        test_bgr: Input BGR image
        cfg: Configuration dict
        expected_k: Expected number of ink colors
        geom: Optional pre-computed lens geometry
        confidence_threshold: Retry if confidence < this (default: 0.7)
        enable_retry: Enable 2-pass retry logic (default: True)

    Returns:
        Tuple of (color_masks, metadata) with best result
    """
    # Pass 1: Try with expected_k
    masks1, meta1 = build_color_masks(test_bgr, cfg, expected_k, geom)

    # Calculate confidence and extract detected ink count
    confidence1 = calculate_segmentation_confidence(meta1, expected_k)
    detected_inks1 = meta1["detected_ink_like_count"]  # Use value from build_color_masks

    # Add confidence and pass info to metadata
    meta1["segmentation_confidence"] = confidence1
    meta1["segmentation_pass"] = "pass1_only"

    # Check if retry needed
    ink_count_mismatch = detected_inks1 != expected_k
    low_confidence = confidence1 < confidence_threshold

    if not enable_retry or (not ink_count_mismatch and not low_confidence):
        # Pass 1 result is good enough
        return masks1, meta1

    # Pass 2: Retry with k = expected_k + 1
    masks2, meta2 = build_color_masks(test_bgr, cfg, expected_k + 1, geom)
    confidence2 = calculate_segmentation_confidence(meta2, expected_k)
    detected_inks2 = meta2["detected_ink_like_count"]  # Use value from build_color_masks

    meta2["segmentation_confidence"] = confidence2
    meta2["segmentation_pass"] = "pass2_retry"

    # Decide which result to use
    # Priority: ink count match > confidence
    use_pass2 = False

    if detected_inks2 == expected_k and detected_inks1 != expected_k:
        # Pass 2 matches expected count, Pass 1 doesn't → use Pass 2
        use_pass2 = True
    elif detected_inks1 == expected_k and detected_inks2 != expected_k:
        # Pass 1 matches, Pass 2 doesn't → use Pass 1
        use_pass2 = False
    else:
        # Both match or both mismatch → use higher confidence
        use_pass2 = confidence2 > confidence1

    if use_pass2:
        meta2["retry_reason"] = []
        if ink_count_mismatch:
            meta2["retry_reason"].append(f"INK_COUNT_MISMATCH (expected={expected_k}, pass1_detected={detected_inks1})")
        if low_confidence:
            meta2["retry_reason"].append(f"LOW_CONFIDENCE (pass1_confidence={confidence1})")
        meta2["pass1_confidence"] = confidence1
        meta2["pass1_detected_inks"] = detected_inks1
        return masks2, meta2
    else:
        # Use Pass 1 even though we retried
        meta1["segmentation_pass"] = "pass1_chosen"
        meta1["pass2_confidence"] = confidence2
        meta1["pass2_detected_inks"] = detected_inks2
        return masks1, meta1


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
