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

import copy
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ...config_norm import get_polar_dims
from ...geometry.lens_geometry import LensGeometry, detect_lens_circle
from ...signature.radial_signature import to_polar
from ...utils import CIELabArray, lab_cie_to_cv8, lab_cv8_to_cie, to_cie_lab
from ..metrics.ink_metrics import (
    build_cluster_stats,
    calculate_inkness_score,
    calculate_radial_presence_curve,
    calculate_spatial_prior,
    silhouette_ab_proxy,
)
from .ink_segmentation import compute_adaptive_l_weight, kmeans_segment, segment_colors
from .preprocess import build_roi_mask, build_sampling_mask, sample_ink_candidates


def lab_to_hex(lab: np.ndarray) -> str:
    """
    Convert Lab color to hex string for display.

    Args:
        lab: Lab color as [L, a, b] array (CIE scale)

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


def lab_cv8_to_hex(lab_cv8: np.ndarray) -> str:
    """
    Convert OpenCV Lab (0-255) to hex via CIE Lab scale.

    Args:
        lab_cv8: Lab color as [L, a, b] array in OpenCV 8-bit scale

    Returns:
        Hex color string (e.g., "#2E241F")
    """
    lab_cie = lab_cv8_to_cie(np.array([lab_cv8], dtype=np.float32))[0]
    return lab_to_hex(lab_cie)


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
    segmentation_k: Optional[int] = None,
    plate_kpis: Optional[Dict[str, Any]] = None,
    sample_mask_override: Optional[np.ndarray] = None,
    _polar_override: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Generate per-color binary masks using k-means segmentation.
    Includes Soft Gate support via plate_kpis.

    If _polar_override is provided (T, R, 3) BGR polar image, geometry
    detection and polar conversion are skipped.  This is an internal
    parameter used by build_color_masks_multi_source().
    """
    v2_cfg = cfg.get("v2_ink", {})
    warnings: List[str] = []

    # 1. Detect lens geometry and convert to polar
    polar_R, polar_T = get_polar_dims(cfg)
    if _polar_override is not None:
        polar = _polar_override
        polar_T, polar_R = polar.shape[:2]
    else:
        if geom is None:
            geom = detect_lens_circle(test_bgr)
        polar = to_polar(test_bgr, geom, R=polar_R, T=polar_T)
    lab_map = to_cie_lab(polar)  # (T, R, 3) - CIE Lab scale

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
        plate_kpis=plate_kpis,  # Pass KPIs for dynamic ROI tightening
    )

    rng_seed = v2_cfg.get("rng_seed", None)

    # Use build_sampling_mask for better background exclusion
    # Pass sample_mask_override for Hard Gate (Plate-based sampling restriction)
    sampling_mask, sample_meta, sample_warnings = build_sampling_mask(
        lab_map, roi_mask, cfg, rng_seed=rng_seed, sample_mask_override=sample_mask_override, plate_kpis=plate_kpis
    )
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

    # 3. Run clustering on samples with configurable method (GMM or K-Means)
    # expected_k = expected number of inks (user intent)
    # segmentation_k = k used for clustering (can be over-segmented for retry)
    k_used = int(segmentation_k) if segmentation_k is not None else int(expected_k)
    # Special case: if expected_k=1 and no override, cluster with k=2 to separate ink from gap
    if expected_k == 1 and segmentation_k is None:
        k_used = 2

    # Adaptive l_weight: 이미지 특성에 따라 자동 조정
    base_l_weight = float(v2_cfg.get("l_weight", 0.3))
    use_adaptive = bool(v2_cfg.get("adaptive_l_weight", False))
    clustering_method = str(v2_cfg.get("clustering_method", "kmeans"))
    kmeans_attempts = int(v2_cfg.get("kmeans_attempts", 10))

    adaptive_meta = None
    if use_adaptive:
        l_weight, adaptive_meta = compute_adaptive_l_weight(
            samples,
            base_weight=base_l_weight,
            low_chroma_threshold=float(v2_cfg.get("adaptive_l_weight_low_chroma", 8.0)),
            high_chroma_threshold=float(v2_cfg.get("adaptive_l_weight_high_chroma", 20.0)),
        )
    else:
        l_weight = base_l_weight

    labels_samples, centers, clustering_confidence = segment_colors(
        samples,
        k_used,
        method=clustering_method,
        l_weight=l_weight,
        rng_seed=rng_seed,
        attempts=kmeans_attempts,
        covariance_type=v2_cfg.get("gmm_covariance_type", "full"),
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

        # Lab centroid for this cluster (CIE scale)
        lab_centroid_cie = centers_lab[cluster_idx]
        lab_centroid_cv8 = lab_cie_to_cv8(lab_centroid_cie).tolist()

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
                # Legacy export: lab_centroid is OpenCV Lab (0-255)
                "lab_centroid": [float(x) for x in lab_centroid_cv8],
                # Recommended: CIE Lab (L*:0-100, a/b:-128..127-ish)
                "lab_centroid_cie": lab_centroid_cie.tolist(),
                "L": float(lab_centroid_cie[0]),
                "area_ratio": area_ratio,
                "hex_ref": lab_to_hex(lab_centroid_cie),
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
            mean_lab=cm["lab_centroid_cie"],
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

        # If we intentionally over-segmented (k_used > expected_k), force exactly expected_k inks
        # by taking top-N by inkness_score. This makes 2-pass retry meaningful and prevents
        # extra gap/background clusters from being counted as inks.
        if expected_k > 0 and k_used > expected_k:
            ink_like = [cm for cm in cluster_meta_sorted if cm["role"] == "ink"]
            if len(ink_like) != expected_k:
                ranked = sorted(cluster_meta_sorted, key=lambda x: float(x.get("inkness_score", 0.0)), reverse=True)
                top_ids = {cm["original_idx"] for cm in ranked[:expected_k]}
                for cm in cluster_meta_sorted:
                    cm["role"] = "ink" if cm["original_idx"] in top_ids else "gap"
                warnings.append("INK_ROLE_FORCED_TOPK")
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
                "lab_centroid_cie": cluster_meta.get("lab_centroid_cie"),
                "hex_ref": cluster_meta["hex_ref"],
                "area_ratio": cluster_meta["area_ratio"],
                "n_pixels": int(mask.sum()),  # Store actual pixel count
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
        "segmentation_k": k_used,  # Parameter: k value used for clustering
        "detected_ink_like_count": detected_ink_count,  # Result: role="ink" count
        # Legacy keys for backward compatibility
        "k_expected": expected_k,
        "k_used": k_used,
        "segmentation_method": clustering_method,  # "kmeans" or "gmm"
        "clustering_confidence": float(clustering_confidence),  # GMM confidence or default
        "l_weight": l_weight,
        "adaptive_l_weight": adaptive_meta,  # None if not adaptive, else meta dict
        "polar": {"order": "TR", "T": int(polar_T), "R": int(polar_R)},
        "warnings": warnings,
        "roi_meta": roi_meta,
        "sample_meta": sample_meta,
        "radial_bins": radial_bins,
        "geom": {
            "cx": float(geom.cx) if geom is not None else 0.0,
            "cy": float(geom.cy) if geom is not None else 0.0,
            "r": float(geom.r) if geom is not None else 0.0,
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


def _convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_types(item) for item in obj)
    return obj


def build_color_masks_v2(
    test_bgr: np.ndarray,
    cfg: Dict[str, Any],
    expected_k: int,
    geom: Optional[LensGeometry] = None,
    same_overlap_params: Optional[Dict[str, Any]] = None,
    diff_mix_params: Optional[Dict[str, Any]] = None,
    gradient_chain_params: Optional[Dict[str, Any]] = None,
    confidence_threshold: float = 0.40,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Generate per-color binary masks using primary color extraction algorithm.

    This is an improved version that:
    1. Over-segments with high k value (8)
    2. Identifies and filters mixed colors (overlaps, mixtures, gradients)
    3. Consolidates to expected_k primary colors

    Args:
        test_bgr: Input BGR image
        cfg: Configuration dict with v2_ink and polar settings
        expected_k: Expected number of ink colors
        geom: Optional pre-computed lens geometry
        same_overlap_params: Parameters for same-color overlap detection
        diff_mix_params: Parameters for different-color mixing detection
        gradient_chain_params: Parameters for gradient chain detection
        confidence_threshold: Min confidence to classify as mixed

    Returns:
        Tuple of (masks, metadata) matching build_color_masks() format
    """
    from .primary_color_extractor import extract_primary_colors

    # Default parameters optimized for gradient lenses
    if same_overlap_params is None:
        same_overlap_params = {
            "ab_threshold": 10.0,
            "l_min_diff": 8.0,
            "l_max_diff": 35.0,
            "area_ratio_max": 0.55,
        }

    if diff_mix_params is None:
        diff_mix_params = {
            "distance_threshold": 12.0,
            "t_range": (0.15, 0.85),
            "area_ratio_max": 0.25,
        }

    if gradient_chain_params is None:
        gradient_chain_params = {
            "ab_threshold": 12.0,
            "l_step_max": 15.0,
            "min_chain_length": 3,
            "protect_area_min": 0.08,
        }

    # Use high k for over-segmentation
    max_k = 8

    # Extract primary colors
    primary_masks_raw, primary_meta = extract_primary_colors(
        test_bgr,
        cfg,
        max_k=max_k,
        expected_primary_count=expected_k,
        geom=geom,
        same_overlap_params=same_overlap_params,
        diff_mix_params=diff_mix_params,
        gradient_chain_params=gradient_chain_params,
        confidence_threshold=confidence_threshold,
    )

    if "error" in primary_meta:
        # Fall back to original method on error
        return build_color_masks(test_bgr, cfg, expected_k, geom)

    # Convert to build_color_masks() format
    v2_cfg = cfg.get("v2_ink", {})
    polar_R, polar_T = get_polar_dims(cfg)

    # Build masks dict
    color_masks = {}
    colors_metadata = []

    for idx, pc in enumerate(primary_meta["primary_colors"]):
        color_id = f"color_{idx}"
        mask_key = pc["color_id"]

        if mask_key in primary_masks_raw:
            color_masks[color_id] = primary_masks_raw[mask_key]
        else:
            color_masks[color_id] = np.zeros((polar_T, polar_R), dtype=bool)

        # Convert Lab CIE to CV8 for compatibility
        lab_cie = np.array(pc["lab_cie"], dtype=np.float32)

        # Simple CIE to CV8 conversion (inverse of lab_cv8_to_cie)
        lab_cv8 = np.array(
            [
                lab_cie[0] * 255.0 / 100.0,
                lab_cie[1] + 128.0,
                lab_cie[2] + 128.0,
            ]
        )

        colors_metadata.append(
            {
                "color_id": color_id,
                "lab_centroid": lab_cv8.tolist(),
                "lab_centroid_cie": pc.get("lab_cie"),
                "hex_ref": pc["hex_color"],
                "area_ratio": pc["area_ratio"],
                "role": "ink",
                "inkness_score": 1.0,
                "radial_presence_curve": [],
                "spatial_prior": 0.5,
            }
        )

    # Build metadata matching original format
    metadata = {
        "colors": colors_metadata,
        "expected_ink_count": expected_k,
        "segmentation_k": max_k,
        "detected_ink_like_count": len(colors_metadata),
        "k_expected": expected_k,
        "k_used": max_k,
        "segmentation_method": "primary_extraction",
        "l_weight": float(v2_cfg.get("l_weight", 0.3)),
        "polar": {"order": "TR", "T": polar_T, "R": polar_R},
        "warnings": primary_meta.get("warnings", []),
        "geom": primary_meta.get("geom", {}),
        # V2-specific metadata (convert numpy types for JSON serialization)
        "primary_extraction": _convert_numpy_types(
            {
                "total_clusters": primary_meta["total_clusters"],
                "merged_clusters": primary_meta.get("merged_clusters", 0),
                "consolidated_clusters": primary_meta.get("consolidated_clusters", 0),
                "primary_count": primary_meta.get("primary_count", 0),
                "mixed_count": primary_meta["mixed_count"],
                "primary_colors": primary_meta.get("primary_colors", []),
                "mixed_colors": primary_meta["mixed_colors"],
                "config": primary_meta["config"],
                # Clustering method info
                "clustering_method": primary_meta.get("clustering_method", "kmeans"),
                "clustering_confidence": primary_meta.get("clustering_confidence", 0.0),
                "l_weight_used": primary_meta.get("l_weight_used", 0.3),
                "adaptive_l_weight": primary_meta.get("adaptive_l_weight"),
                "auto_k": primary_meta.get("auto_k"),
                "effective_k_used": primary_meta.get("effective_k_used"),
                # Family-grouped view (all clusters grouped by color family)
                "family_groups": primary_meta.get("family_groups", {}),
            }
        ),
    }

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
    def _L_cie(c: Dict[str, Any]) -> float:
        # Prefer explicit CIE centroid
        if c.get("lab_centroid_cie") is not None:
            return float(c["lab_centroid_cie"][0])
        lab = c.get("lab_centroid") or [0, 0, 0]
        L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
        # Heuristic: OpenCV Lab tends to have a/b in [0,255] with mean > ~40
        is_cv8 = (0.0 <= a <= 255.0) and (0.0 <= b <= 255.0) and ((a + b) / 2.0 > 40.0) and (0.0 <= L <= 255.0)
        return L * 100.0 / 255.0 if is_cv8 else L

    L_values = [_L_cie(c) for c in colors]
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
    use_primary_extraction: bool = False,
    plate_kpis: Optional[Dict[str, Any]] = None,
    sample_mask_override: Optional[np.ndarray] = None,
    _polar_override: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Build color masks with 2-pass retry logic and Soft Gate scaling.

    When sample_mask_override (Plate Gate) is provided, we use the standard
    v1 path to ensure proper Hard Gate support.
    """
    # Use V2 algorithm if requested AND no Hard Gate mask
    # When sample_mask_override is provided, fall back to v1 for proper Hard Gate support
    if use_primary_extraction and sample_mask_override is None and _polar_override is None:
        # Note: build_color_masks_v2 doesn't support sample_mask_override yet
        return build_color_masks_v2(test_bgr, cfg, expected_k, geom)

    # Pass 1: Try with expected_k
    masks1, meta1 = build_color_masks(
        test_bgr,
        cfg,
        expected_k,
        geom,
        plate_kpis=plate_kpis,
        sample_mask_override=sample_mask_override,
        _polar_override=_polar_override,
    )

    # Calculate confidence
    confidence1 = calculate_segmentation_confidence(meta1, expected_k)

    # SOFT GATE: Scale confidence by plate artifacts
    if plate_kpis:
        artifact_ratio = plate_kpis.get("mask_artifact_ratio_valid", 0.0)
        # Scale factor: 1.0 at 0 artifacts, 0.25 at 0.5 artifacts
        conf_factor = max(0.2, min(1.0, 1.0 - 1.5 * artifact_ratio))
        confidence1 = round(confidence1 * conf_factor, 3)
        meta1["soft_gate_scaled"] = True
        meta1["artifact_conf_factor"] = conf_factor

    detected_inks1 = meta1["detected_ink_like_count"]
    meta1["segmentation_confidence"] = confidence1
    meta1["segmentation_pass"] = "pass1_only"

    # Check if retry needed
    ink_count_mismatch = detected_inks1 != expected_k
    low_confidence = confidence1 < confidence_threshold

    # [Safety Mode] Prevent retry if confidence is already high enough, even if count mismatch
    # This protects against over-segmentation of very clean samples.
    v2_cfg = cfg.get("v2_ink", {})
    safety_mode = bool(v2_cfg.get("auto_k_safety_mode", True))
    safety_threshold = float(v2_cfg.get("min_confidence_for_auto_k", 0.85))

    if safety_mode and confidence1 >= safety_threshold:
        # High confidence in Pass 1 -> Trust it and skip retry
        meta1["notes"] = ["retry_skipped_due_to_high_confidence"]
        return masks1, meta1

    if not enable_retry or (not ink_count_mismatch and not low_confidence):
        # Pass 1 result is good enough
        return masks1, meta1

    # Pass 2: Retry with over-segmentation (k = expected_k + 1), but keep expected_k as the ink count intent
    role_policy = str(v2_cfg.get("role_policy", "legacy")).lower()
    cfg2 = copy.deepcopy(cfg)
    if role_policy == "legacy":
        cfg2.setdefault("v2_ink", {})["role_policy"] = "inkness"
    masks2, meta2 = build_color_masks(
        test_bgr,
        cfg2,
        expected_k,
        geom,
        segmentation_k=expected_k + 1,
        plate_kpis=plate_kpis,
        sample_mask_override=sample_mask_override,
        _polar_override=_polar_override,
    )
    confidence2 = calculate_segmentation_confidence(meta2, expected_k)

    # SOFT GATE: Scale confidence 2 by plate artifacts
    if plate_kpis:
        artifact_ratio = plate_kpis.get("mask_artifact_ratio_valid", 0.0)
        conf_factor = max(0.2, min(1.0, 1.0 - 1.5 * artifact_ratio))
        confidence2 = round(confidence2 * conf_factor, 3)
        meta2["soft_gate_scaled"] = True
        meta2["artifact_conf_factor"] = conf_factor

    detected_inks2 = meta2["detected_ink_like_count"]

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


# ==============================================================================
# Backward-compatible re-exports from split modules
# ==============================================================================
from .color_masks_density import (
    build_alpha_from_plate_images,
    build_color_masks_with_alpha,
    compute_cluster_effective_densities,
)
from .color_masks_stabilize import (
    build_color_masks_multi_source,
    build_color_masks_with_reference,
    stabilize_labels_with_reference,
)
