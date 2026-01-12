from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ...geometry.lens_geometry import detect_lens_circle
from ...signature.radial_signature import to_polar
from ...utils import bgr_to_lab
from ..matching.ink_match import align_to_reference, match_clusters_ab
from ..metrics.ink_metrics import build_cluster_stats, ensure_cie_lab
from ..segmentation.color_masks import assign_cluster_labels_to_image
from ..segmentation.ink_segmentation import kmeans_segment
from ..segmentation.preprocess import build_roi_mask, sample_ink_candidates


def _segment_image(
    bgr,
    cfg: Dict[str, Any],
    expected_k: int,
) -> Optional[Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, List[List[float]]]]:
    """
    Segment image and return clusters with spatial info and pairwise ΔE.

    Returns:
        (clusters, state_id_map, polar_r_map, pairwise_deltaE) or None
        - state_id_map: (T, R) ROI pixels have [0..k-1], outside ROI is -1
        - polar_r_map: (T, R) normalized radial distance [0..1] per pixel
    """
    v2_cfg = cfg.get("v2_ink", {})
    geom = detect_lens_circle(bgr)
    polar_R = int(cfg["polar"]["R"])
    polar_T = int(cfg["polar"]["T"])
    polar = to_polar(bgr, geom, R=polar_R, T=polar_T)
    lab_map = bgr_to_lab(polar).astype(np.float32)  # (T, R, 3)

    # ROI
    r_start = float(v2_cfg.get("roi_r_start", cfg["anomaly"]["r_start"]))
    r_end = float(v2_cfg.get("roi_r_end", cfg["anomaly"]["r_end"]))
    center_excluded_frac = float(cfg["anomaly"].get("center_frac", 0.0))
    roi_mask, _ = build_roi_mask(
        polar_T,
        polar_R,
        r_start,
        r_end,
        center_excluded_frac=center_excluded_frac,
    )

    # Sample candidates
    rng_seed = v2_cfg.get("rng_seed", None)
    samples, sample_indices, sample_meta, sample_warnings, sample_mask = sample_ink_candidates(
        lab_map, roi_mask, cfg, rng_seed=rng_seed, return_mask=True
    )
    if samples.size == 0:
        return None

    kmeans_attempts = int(v2_cfg.get("kmeans_attempts", 10))
    l_weight = float(v2_cfg.get("l_weight", 0.3))
    labels_s, centers = kmeans_segment(
        samples,
        expected_k,
        l_weight=l_weight,
        attempts=kmeans_attempts,
        rng_seed=rng_seed,
    )
    if labels_s.size == 0 or centers.size == 0:
        return None

    # Assign every pixel to nearest center (same feature space) then restrict to ROI
    label_map = assign_cluster_labels_to_image(lab_map, centers, l_weight=l_weight)  # (T, R)
    state_id_map = np.full((polar_T, polar_R), -1, dtype=np.int32)
    state_id_map[roi_mask] = label_map[roi_mask].astype(np.int32)

    # Normalized radial distance map (T, R): radial axis is R (columns)
    r_vals = np.linspace(0.0, 1.0, polar_R, dtype=np.float32)
    polar_r_map = np.tile(r_vals[None, :], (polar_T, 1))

    # Compute cluster stats on ROI pixels (NOT only sampled pixels)
    roi_labs = lab_map[roi_mask]
    roi_labels = state_id_map[roi_mask]

    # IMPORTANT: Convert to CIE Lab for metrics calculation (alpha_like, inkness, etc.)
    from ...utils import lab_cv8_to_cie

    roi_labs_cie = lab_cv8_to_cie(roi_labs.reshape(-1, 1, 3)).reshape(-1, 3)

    deltaE_method = str(v2_cfg.get("deltaE_method", "76"))
    stats = build_cluster_stats(roi_labs_cie, roi_labels, int(expected_k), deltaE_method=deltaE_method)

    clusters = stats.get("clusters", [])
    pairwise_deltaE = stats.get("pairwise_deltaE", [])

    # Ensure CIE Lab
    for c in clusters:
        if "mean_lab" in c:
            c["mean_lab"] = ensure_cie_lab(np.array(c["mean_lab"], dtype=np.float32)).tolist()

    # Spatial metrics (radial curve / prior / inkness)
    from ..metrics.ink_metrics import calculate_inkness_score, calculate_radial_presence_curve, calculate_spatial_prior

    r_roi = polar_r_map[roi_mask]
    for i, cluster in enumerate(clusters):
        curve = calculate_radial_presence_curve(roi_labels, i, r_roi, r_bins=10)
        cluster["radial_presence_curve"] = curve

        spatial_prior = calculate_spatial_prior(curve)
        cluster["spatial_prior"] = spatial_prior

        inkness = calculate_inkness_score(
            cluster["mean_lab"], cluster["compactness"], cluster["alpha_like"], spatial_prior
        )
        cluster["inkness_score"] = inkness

    # Attach sampling diagnostics (useful for baseline quality debugging)
    if sample_warnings:
        for c in clusters:
            c.setdefault("warnings", [])
            c["warnings"].extend(sample_warnings)
    if sample_meta:
        for c in clusters:
            c.setdefault("sampling", {})
            c["sampling"].update(sample_meta)

    return (clusters, state_id_map, polar_r_map, pairwise_deltaE)


def build_ink_baseline(
    images_by_mode: Dict[str, List[str]],
    cfg: Dict[str, Any],
    expected_k: int,
) -> Optional[Dict[str, Any]]:
    def _effective_k(expected: int) -> int:
        return 2 if int(expected) == 1 else int(expected)

    k_expected = int(expected_k)
    k_effective = _effective_k(k_expected)
    clusters_by_mode: Dict[str, List[Dict[str, Any]]] = {}
    pairwise_by_mode: Dict[str, List[List[float]]] = {}
    for mode in ["MID", "LOW", "HIGH"]:
        paths = images_by_mode.get(mode, [])
        if not paths:
            continue
        bgr = cv2.imread(paths[0])
        if bgr is None:
            continue
        result = _segment_image(bgr, cfg, k_effective)
        if result:
            clusters, _, _, pairwise = result  # Unpack tuple
            clusters_by_mode[mode] = clusters
            pairwise_by_mode[mode] = pairwise

    if "MID" not in clusters_by_mode:
        return None
    if any(m not in clusters_by_mode for m in ["LOW", "HIGH"]):
        return None

    mid_clusters = clusters_by_mode["MID"]
    if k_expected == 1 and len(mid_clusters) == 2:
        l_vals = [float(c.get("mean_lab", [0.0, 0.0, 0.0])[0]) for c in mid_clusters]
        ink_idx = int(np.argmin(l_vals))
        for mode_key, clusters in clusters_by_mode.items():
            for idx, c in enumerate(clusters):
                c["role"] = "ink" if idx == ink_idx else "gap"

    low_aligned = align_to_reference(mid_clusters, clusters_by_mode["LOW"])
    high_aligned = align_to_reference(mid_clusters, clusters_by_mode["HIGH"])
    aligned_modes = {"LOW": low_aligned, "MID": mid_clusters, "HIGH": high_aligned}

    k = len(mid_clusters)
    weights = {"MID": 0.5, "LOW": 0.25, "HIGH": 0.25}
    baseline_clusters = []
    spreads = []
    for i in range(k):
        lab_mid = np.array(mid_clusters[i]["mean_lab"], dtype=np.float32)
        lab_low = np.array(low_aligned[i]["mean_lab"], dtype=np.float32)
        lab_high = np.array(high_aligned[i]["mean_lab"], dtype=np.float32)
        role = mid_clusters[i].get("role")

        centroid = weights["MID"] * lab_mid + weights["LOW"] * lab_low + weights["HIGH"] * lab_high
        spread = float(
            (
                np.linalg.norm(lab_mid - centroid)
                + np.linalg.norm(lab_low - centroid)
                + np.linalg.norm(lab_high - centroid)
            )
            / 3.0
        )
        spreads.append(spread)

        # Weighted average of distribution metrics (NEW!)
        def weighted_avg_3d(key):
            mid_val = np.array(mid_clusters[i].get(key, [0.0, 0.0, 0.0]))
            low_val = np.array(low_aligned[i].get(key, [0.0, 0.0, 0.0]))
            high_val = np.array(high_aligned[i].get(key, [0.0, 0.0, 0.0]))
            avg = weights["MID"] * mid_val + weights["LOW"] * low_val + weights["HIGH"] * high_val
            return [round(float(avg[0]), 2), round(float(avg[1]), 2), round(float(avg[2]), 2)]

        def weighted_avg_scalar(key):
            mid_val = mid_clusters[i].get(key, 0.0)
            low_val = low_aligned[i].get(key, 0.0)
            high_val = high_aligned[i].get(key, 0.0)
            return round(float(weights["MID"] * mid_val + weights["LOW"] * low_val + weights["HIGH"] * high_val), 3)

        def weighted_avg_curve(key, default_len=10):
            mid_val = np.array(mid_clusters[i].get(key, [0.0] * default_len))
            low_val = np.array(low_aligned[i].get(key, [0.0] * default_len))
            high_val = np.array(high_aligned[i].get(key, [0.0] * default_len))
            avg = weights["MID"] * mid_val + weights["LOW"] * low_val + weights["HIGH"] * high_val
            return [round(float(v), 3) for v in avg]

        # Calculate weighted mean_hex and mean_rgb from centroid
        from ..metrics.ink_metrics import _lab_to_rgb, _rgb_to_hex

        centroid_rgb = _lab_to_rgb(centroid)
        centroid_hex = _rgb_to_hex(centroid_rgb)

        baseline_clusters.append(
            {
                "mean_lab": centroid.tolist(),
                "mean_ab": [float(centroid[1]), float(centroid[2])],
                "mean_rgb": centroid_rgb,
                "mean_hex": centroid_hex,
                "area_ratio": float(
                    weights["MID"] * mid_clusters[i]["area_ratio"]
                    + weights["LOW"] * low_aligned[i]["area_ratio"]
                    + weights["HIGH"] * high_aligned[i]["area_ratio"]
                ),
                "spread": spread,
                "role": role,
                # NEW: Distribution width information
                "lab_std": weighted_avg_3d("lab_std"),
                "p10": weighted_avg_3d("p10"),
                "p90": weighted_avg_3d("p90"),
                "compactness": weighted_avg_scalar("compactness"),
                # NEW: Radial presence curve
                "radial_presence_curve": weighted_avg_curve("radial_presence_curve", 10),
                # NEW: Role discrimination metrics
                "alpha_like": weighted_avg_scalar("alpha_like"),
                "spatial_prior": weighted_avg_scalar("spatial_prior"),
                "inkness_score": weighted_avg_scalar("inkness_score"),
            }
        )

    match_cost_low = match_clusters_ab(mid_clusters, clusters_by_mode["LOW"])[1]
    match_cost_high = match_clusters_ab(mid_clusters, clusters_by_mode["HIGH"])[1]

    # Calculate weighted average pairwise_deltaE matrix
    baseline_pairwise = []
    if all(mode in pairwise_by_mode for mode in ["LOW", "MID", "HIGH"]):
        mid_pairwise = np.array(pairwise_by_mode["MID"], dtype=np.float32)
        low_pairwise = np.array(pairwise_by_mode["LOW"], dtype=np.float32)
        high_pairwise = np.array(pairwise_by_mode["HIGH"], dtype=np.float32)

        # Weighted average across modes
        avg_pairwise = weights["MID"] * mid_pairwise + weights["LOW"] * low_pairwise + weights["HIGH"] * high_pairwise

        # Convert back to list of lists with rounding
        for row in avg_pairwise:
            baseline_pairwise.append([round(float(v), 2) for v in row])

    return {
        "clusters": baseline_clusters,
        "spread_mean": float(np.mean(spreads)) if spreads else None,
        "spread_max": float(np.max(spreads)) if spreads else None,
        "match_cost_low_mid": match_cost_low,
        "match_cost_high_mid": match_cost_high,
        "k_expected": k_expected,
        "k_used": k_effective,
        "pairwise_deltaE": baseline_pairwise,
        "source_modes": {
            "LOW": clusters_by_mode["LOW"],
            "MID": clusters_by_mode["MID"],
            "HIGH": clusters_by_mode["HIGH"],
        },
        "aligned_modes": aligned_modes,
        "aligned_ref": "MID",
    }
