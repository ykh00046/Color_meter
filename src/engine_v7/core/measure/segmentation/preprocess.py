from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

from ...utils import CIELabArray, to_cie_lab

if TYPE_CHECKING:
    pass


# ==============================================================================
# Phase 3: Radial Distribution Analysis for Background Detection
# ==============================================================================


@dataclass
class RadialPresenceResult:
    """Result of radial presence analysis for a cluster."""

    curve: List[float]  # Presence ratio per radial bin
    uniformity: float  # How uniformly distributed (0-1, 1 = perfectly uniform)
    inner_presence: float  # Presence in inner region (r < 0.4)
    outer_presence: float  # Presence in outer region (r > 0.7)
    span_score: float  # Bonus for spanning both inner and outer regions


def _compute_radial_presence(
    labels: np.ndarray,
    cluster_idx: int,
    r_values: np.ndarray,
    n_bins: int = 10,
) -> RadialPresenceResult:
    """
    Compute radial presence curve for a specific cluster.

    Background clusters typically have high presence across all radial zones,
    while ink clusters are concentrated in specific zones (dot, ring).

    Args:
        labels: Cluster labels for each pixel (N,)
        cluster_idx: Index of cluster to analyze
        r_values: Normalized radial distance for each pixel (N,), range [0, 1]
        n_bins: Number of radial bins

    Returns:
        RadialPresenceResult with presence curve and uniformity metrics
    """
    cluster_mask = labels == cluster_idx
    n_cluster = np.sum(cluster_mask)

    if n_cluster == 0:
        return RadialPresenceResult(
            curve=[0.0] * n_bins,
            uniformity=0.0,
            inner_presence=0.0,
            outer_presence=0.0,
            span_score=0.0,
        )

    # Compute presence curve: ratio of cluster pixels in each radial bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    curve = []

    for i in range(n_bins):
        bin_mask = (r_values >= bin_edges[i]) & (r_values < bin_edges[i + 1])
        n_bin = np.sum(bin_mask)
        if n_bin > 0:
            presence = np.sum(cluster_mask & bin_mask) / n_bin
        else:
            presence = 0.0
        curve.append(float(presence))

    # Uniformity: 1 - coefficient of variation (low CV = high uniformity)
    curve_arr = np.array(curve)
    non_zero = curve_arr[curve_arr > 0]
    if len(non_zero) > 1:
        cv = float(np.std(non_zero) / (np.mean(non_zero) + 1e-6))
        uniformity = max(0.0, min(1.0, 1.0 - cv))
    else:
        uniformity = 0.0

    # Inner/outer presence (key zones for background detection)
    # Inner: first 40% of radius, Outer: last 30% of radius
    inner_bins = int(n_bins * 0.4)
    outer_start = int(n_bins * 0.7)

    inner_presence = float(np.mean(curve[:inner_bins])) if inner_bins > 0 else 0.0
    outer_presence = float(np.mean(curve[outer_start:])) if outer_start < n_bins else 0.0

    # Span score: bonus for being present in both inner AND outer regions
    # Background should span the entire radius, ink is localized
    if inner_presence > 0.05 and outer_presence > 0.05:
        span_score = min(inner_presence, outer_presence) * 2.0
    else:
        span_score = 0.0

    return RadialPresenceResult(
        curve=curve,
        uniformity=uniformity,
        inner_presence=inner_presence,
        outer_presence=outer_presence,
        span_score=span_score,
    )


def _compute_radial_background_score(
    labels: np.ndarray,
    centers: np.ndarray,
    r_values: np.ndarray,
    n_bins: int = 10,
    radial_weight: float = 0.3,
) -> Tuple[np.ndarray, List[RadialPresenceResult]]:
    """
    Compute enhanced background scores using radial distribution.

    Enhanced score = L - Chroma + radial_weight * span_score

    Args:
        labels: Cluster labels for each pixel
        centers: Cluster centers in CIE Lab space (K, 3)
        r_values: Normalized radial distance for each pixel
        n_bins: Number of radial bins
        radial_weight: Weight for radial span bonus (default 0.3)

    Returns:
        scores: Background scores for each cluster (K,)
        radial_results: RadialPresenceResult for each cluster
    """
    n_clusters = centers.shape[0]

    # Base scores: L - Chroma (existing logic)
    l_values = centers[:, 0]
    chromas = np.sqrt(centers[:, 1] ** 2 + centers[:, 2] ** 2)
    base_scores = l_values - chromas

    # Compute radial presence for each cluster
    radial_results = []
    radial_bonuses = np.zeros(n_clusters)

    for i in range(n_clusters):
        result = _compute_radial_presence(labels, i, r_values, n_bins)
        radial_results.append(result)

        # Bonus for spanning both inner and outer regions
        # This strongly favors background over ink
        radial_bonuses[i] = result.span_score * radial_weight * 100  # Scale to match L-C

    # Enhanced scores
    enhanced_scores = base_scores + radial_bonuses

    return enhanced_scores, radial_results


def build_roi_mask(
    T: int,
    R: int,
    r_start: float,
    r_end: float,
    center_excluded_frac: float = 0.0,
    plate_kpis: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build ROI mask with dynamic tightening based on Plate KPIs.
    """
    r_start_eff = max(float(r_start), float(center_excluded_frac or 0.0))
    r_end_eff = float(r_end)

    warnings = []
    # SOFT GATE 2: Proportionally tighten outer rim based on leak ratio
    # Higher leak ratio -> more conservative r_end (exclude more outer pixels)
    if plate_kpis:
        outer_leak = plate_kpis.get("outer_rim_leak_ratio", 0.0)
        # Proportional adjustment: leak 0.02 -> r_end 0.96, leak 0.10 -> r_end 0.88
        # Formula: r_end = base - (leak_ratio - threshold) * scale
        leak_threshold = 0.02
        leak_scale = 1.0  # How much to reduce r_end per leak ratio unit
        if outer_leak > leak_threshold:
            reduction = (outer_leak - leak_threshold) * leak_scale
            r_end_tightened = r_end_eff - reduction
            # Clamp to reasonable range (don't go below 0.80)
            r_end_eff = max(0.80, min(r_end_eff, r_end_tightened))
            warnings.append(f"SOFT_GATE2:r_end={r_end_eff:.3f},leak={outer_leak:.3f}")

    r0 = int(round(r_start_eff * R))
    r1 = int(round(r_end_eff * R))
    r0 = max(0, min(R - 1, r0))
    r1 = max(r0 + 1, min(R, r1))

    mask = np.zeros((T, R), dtype=bool)
    mask[:, r0:r1] = True

    meta = {
        "polar_order": "TR",
        "T": int(T),
        "R": int(R),
        "r_start_config": float(r_start),
        "r_start_effective": float(r_start_eff),
        "r_end_config": float(r_end),
        "r_end_effective": float(r_end_eff),
        "center_excluded_frac": float(center_excluded_frac or 0.0),
        "soft_gate_warnings": warnings,
    }
    return mask, meta


def _quantile_mask(values: np.ndarray, p: float, largest: bool) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(values, dtype=bool)
    q = np.quantile(values, 1 - p) if largest else np.quantile(values, p)
    if largest:
        return values >= q
    return values <= q


def build_sampling_mask(
    lab_map: np.ndarray,
    roi_mask: np.ndarray,
    cfg: Dict[str, Any],
    rng_seed: int | None = None,
    *,
    r_map: Optional[np.ndarray] = None,
    sample_mask_override: Optional[np.ndarray] = None,
    plate_kpis: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
    """
    Build a mask for ink candidate pixels with radial distribution analysis.

    Phase 3 Enhancement: Uses 2-stage clustering with radial presence analysis
    to better distinguish background (gap) from ink regions.

    Plate Gate Enhancement: When sample_mask_override is provided (e.g., from
    plate's dot_core_mask), it restricts sampling to validated ink regions,
    preventing contamination from artifacts/outer rim.

    Soft Gate Enhancement: Uses plate_kpis to dynamically tighten ROI.

    Args:
        lab_map: CIE Lab image (L:0-100, a/b:-128~127).
                 Use to_cie_lab() to convert from BGR.
        roi_mask: Boolean mask for region of interest (pre-computed base mask)
        cfg: Configuration dict
        rng_seed: Random seed for reproducibility
        r_map: Optional normalized radial distance map (T, R) with values [0, 1].
               If provided, enables radial presence analysis for better
               background detection.
        sample_mask_override: Optional pre-computed mask from plate analysis.
               If provided and sufficient pixels, overrides smart clustering
               and uses this mask directly (Hard Gate).
        plate_kpis: Optional KPIs from plate analysis for Soft Gate adjustment.

    Returns:
        mask: Boolean mask for ink candidate pixels
        meta: Metadata about the sampling process
        warnings: List of warning codes
    """
    # Ensure CIE Lab scale (auto-detect and convert if needed)
    lab_map = to_cie_lab(lab_map, validate=False)

    v2_cfg = cfg.get("v2_ink", {})
    warnings: List[str] = []

    # RE-COMPUTE ROI MASK if plate_kpis provided (Soft Gate)
    if plate_kpis:
        # We re-run build_roi_mask to get the tightened r_end
        polar_T, polar_R = lab_map.shape[:2]
        r_start = float(v2_cfg.get("roi_r_start", cfg["anomaly"]["r_start"]))
        r_end = float(v2_cfg.get("roi_r_end", cfg["anomaly"]["r_end"]))
        center_excluded_frac = float(cfg["anomaly"]["center_frac"])

        roi_mask, roi_meta_new = build_roi_mask(
            polar_T, polar_R, r_start, r_end, center_excluded_frac, plate_kpis=plate_kpis
        )
        if roi_meta_new.get("soft_gate_warnings"):
            warnings.extend(roi_meta_new["soft_gate_warnings"])

    # Initialize random generator
    rng = np.random.default_rng(rng_seed)

    # Initialize meta variables
    clear_l_min = None
    clear_c_max = None
    n_clear = 0

    # Configuration for Smart Clustering
    use_smart_clustering = bool(v2_cfg.get("use_smart_clustering", True))
    n_clusters_bg = int(v2_cfg.get("bg_n_clusters", 4))

    # Phase 3: Radial analysis configuration
    use_radial_analysis = bool(v2_cfg.get("use_radial_analysis", True))
    radial_weight = float(v2_cfg.get("radial_weight", 0.3))
    radial_n_bins = int(v2_cfg.get("radial_n_bins", 10))

    # Fallback/Legacy params
    dark_p = float(v2_cfg.get("dark_top_p", 0.25))
    chroma_p = float(v2_cfg.get("chroma_top_p", 0.35))
    min_samples = int(v2_cfg.get("min_samples", 8000))
    min_samples_warn = int(v2_cfg.get("min_samples_warn", 2000))

    # ========================================================================
    # HARD GATE: Plate-based sample mask override
    # If plate analysis provides a validated ink mask (e.g., dot_core_mask),
    # use it directly to avoid contamination from artifacts/outer rim.
    # ========================================================================
    plate_gate_min_samples = int(v2_cfg.get("plate_gate_min_samples", 1000))
    plate_gate_low_samples = 300  # Threshold for dilation fallback

    if sample_mask_override is not None:
        # Intersect with ROI mask for safety
        gated_mask = sample_mask_override & roi_mask
        n_gated = int(np.sum(gated_mask))

        # ================================================================
        # SOFT GATE 1: Erode mask when artifact ratio is high
        # This removes reflection boundaries that contaminate sampling
        # ================================================================
        artifact_erode_th = float(v2_cfg.get("soft_gate_artifact_erode_th", 0.25))
        artifact_ratio = 0.0
        eroded = False

        if plate_kpis:
            artifact_ratio = plate_kpis.get("mask_artifact_ratio_valid", 0.0)
            if artifact_ratio > artifact_erode_th and n_gated >= plate_gate_min_samples:
                # Erode by 1px to remove edge contamination
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                eroded_mask = cv2.erode(gated_mask.astype(np.uint8), kernel).astype(bool)
                eroded_mask = eroded_mask & roi_mask
                n_eroded = int(np.sum(eroded_mask))

                # Only use eroded mask if it still has sufficient samples
                if n_eroded >= plate_gate_min_samples:
                    gated_mask = eroded_mask
                    n_gated = n_eroded
                    eroded = True
                    warnings.append(f"SOFT_GATE:eroded_artifact={artifact_ratio:.2f}>{artifact_erode_th}")

        if n_gated >= plate_gate_min_samples:
            # 1. Sufficient samples -> Use Hard Gate directly
            meta = {
                "rule": "plate_gate_override_eroded" if eroded else "plate_gate_override",
                "n_pixels_used": n_gated,
                "plate_gate_applied": True,
                "artifact_ratio": artifact_ratio,
                "eroded": eroded,
                "rng_seed_used": rng_seed,
            }
            return gated_mask, meta, warnings

        elif n_gated >= plate_gate_low_samples:
            # 2. Low but usable samples -> Dilate slightly to gather more neighbors
            # Risk: might include some edge pixels, but better than heuristic fallback
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated_mask = cv2.dilate(gated_mask.astype(np.uint8), kernel).astype(bool)
            dilated_mask = dilated_mask & roi_mask  # Keep inside ROI
            n_dilated = int(np.sum(dilated_mask))

            meta = {
                "rule": "plate_gate_override_dilated",
                "n_pixels_used": n_dilated,
                "plate_gate_applied": True,
                "rng_seed_used": rng_seed,
            }
            warnings.append(f"PLATE_GATE_DILATED:n_original={n_gated}->{n_dilated}")
            return dilated_mask, meta, warnings

        else:
            # 3. Too few samples -> Fallback to Heuristic
            warnings.append(f"PLATE_GATE_FALLBACK:n_gated={n_gated}<{plate_gate_low_samples}")

    L = lab_map[..., 0]
    a = lab_map[..., 1]
    b = lab_map[..., 2]

    # ROI Extraction
    roi_idx = np.where(roi_mask)
    if roi_idx[0].size == 0:
        meta = {"rule": "none", "n_pixels_used": 0, "rng_seed_used": rng_seed}
        return np.zeros_like(roi_mask, dtype=bool), meta, ["INK_SAMPLING_EMPTY"]

    # Extract ROI pixels for analysis
    # Shape: (N, 3) where N is number of pixels in ROI
    pixels = lab_map[roi_idx].astype(np.float32)

    # Extract radial values for ROI pixels if r_map provided
    r_values_roi = None
    if r_map is not None and use_radial_analysis:
        r_values_roi = r_map[roi_idx].astype(np.float32)

    if use_smart_clustering and pixels.shape[0] >= n_clusters_bg:
        try:
            # [Smart Clustering Logic with Radial Analysis]
            # Stage 1: K-Means Clustering
            kmeans = KMeans(n_clusters=n_clusters_bg, random_state=rng_seed if rng_seed is not None else 42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            centers = kmeans.cluster_centers_

            # Stage 2: Enhanced Background Detection
            l_values = centers[:, 0]
            a_values = centers[:, 1]
            b_values = centers[:, 2]
            chromas = np.sqrt(a_values**2 + b_values**2)

            # Phase 3: Use radial distribution if available
            radial_results: Optional[List[RadialPresenceResult]] = None
            if r_values_roi is not None:
                # Enhanced scoring with radial analysis
                bg_scores, radial_results = _compute_radial_background_score(
                    labels,
                    centers,
                    r_values_roi,
                    n_bins=radial_n_bins,
                    radial_weight=radial_weight,
                )
            else:
                # Fallback to original L - Chroma scoring
                bg_scores = l_values - chromas

            bg_cluster_idx = np.argmax(bg_scores)

            # Create Mask (Include all groups EXCEPT background)
            is_ink = labels != bg_cluster_idx

            mask = np.zeros_like(roi_mask, dtype=bool)
            mask[roi_idx[0][is_ink], roi_idx[1][is_ink]] = True

            # Build metadata
            meta = {
                "rule": "smart_clustering_kmeans_v2" if r_values_roi is not None else "smart_clustering_kmeans",
                "n_pixels_used": int(mask.sum()),
                "bg_cluster_idx": int(bg_cluster_idx),
                "bg_score": float(bg_scores[bg_cluster_idx]),
                "bg_L": float(l_values[bg_cluster_idx]),
                "bg_C": float(chromas[bg_cluster_idx]),
                "ink_ratio": float(is_ink.mean()),
                "rng_seed_used": rng_seed,
            }

            # Phase 3: Add radial analysis metadata
            if radial_results is not None:
                bg_radial = radial_results[bg_cluster_idx]
                meta["radial_analysis"] = {
                    "enabled": True,
                    "radial_weight": radial_weight,
                    "n_bins": radial_n_bins,
                    "bg_uniformity": bg_radial.uniformity,
                    "bg_inner_presence": bg_radial.inner_presence,
                    "bg_outer_presence": bg_radial.outer_presence,
                    "bg_span_score": bg_radial.span_score,
                    "bg_presence_curve": bg_radial.curve,
                    "all_clusters": [
                        {
                            "idx": i,
                            "L": float(l_values[i]),
                            "C": float(chromas[i]),
                            "score": float(bg_scores[i]),
                            "uniformity": r.uniformity,
                            "span_score": r.span_score,
                        }
                        for i, r in enumerate(radial_results)
                    ],
                }
            else:
                meta["radial_analysis"] = {"enabled": False, "reason": "no_r_map"}

            return mask, meta, warnings

        except Exception as e:
            warnings.append(f"SMART_CLUSTERING_FAILED: {str(e)}")
            # Fallback to legacy logic below

    # [Legacy Logic / Fallback]
    # ... (Original quantile based logic) ...
    chroma = np.sqrt(a * a + b * b)  # Calculate chroma for full map
    L_roi = L[roi_idx]
    chroma_roi = chroma[roi_idx]

    dark_sel = _quantile_mask(L_roi, dark_p, largest=False)
    chroma_sel = _quantile_mask(chroma_roi, chroma_p, largest=True)
    union_sel = dark_sel | chroma_sel

    def _mask_from_sel(mask_sel: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(roi_mask, dtype=bool)
        rr = roi_idx[0][mask_sel]
        cc = roi_idx[1][mask_sel]
        mask[rr, cc] = True
        return mask

    mask = _mask_from_sel(union_sel)
    rule = "ink_candidate_union_v1"
    random_fallback = False
    if int(mask.sum()) < min_samples:
        mask = _mask_from_sel(dark_sel)
        rule = "dark_top_p"
    if int(mask.sum()) < min_samples:
        mask = _mask_from_sel(chroma_sel)
        rule = "high_chroma_top_p"
    if int(mask.sum()) < min_samples_warn:
        rr_all = roi_idx[0]
        cc_all = roi_idx[1]
        n_all = rr_all.size
        n_pick = min(n_all, min_samples_warn)
        if n_all > 0:
            pick = rng.choice(n_all, size=n_pick, replace=False)
            mask = np.zeros_like(roi_mask, dtype=bool)
            mask[rr_all[pick], cc_all[pick]] = True
            rule = "random_roi_fallback"
            random_fallback = True
        warnings.append("INK_SEPARATION_LOW_CONFIDENCE")

    meta = {
        "rule": rule,
        "n_pixels_used": int(mask.sum()),
        "dark_top_p": dark_p,
        "chroma_top_p": chroma_p,
        "clear_l_min": clear_l_min,
        "clear_c_max": clear_c_max,
        "n_clear_excluded": n_clear,
        "random_fallback_used": bool(random_fallback),
        "rng_seed_used": rng_seed,
    }
    return mask, meta, warnings


def sample_ink_candidates(
    lab_map: np.ndarray,
    roi_mask: np.ndarray,
    cfg: Dict[str, Any],
    rng_seed: int | None = None,
    *,
    return_mask: bool = False,
    r_map: Optional[np.ndarray] = None,
    sample_mask_override: Optional[np.ndarray] = None,
    plate_kpis: Optional[Dict[str, Any]] = None,
) -> (
    Tuple[np.ndarray, np.ndarray, Dict[str, Any], List[str]]
    | Tuple[np.ndarray, np.ndarray, Dict[str, Any], List[str], np.ndarray]
):
    """
    Sample candidate pixels (likely ink) inside ROI.

    Args:
        lab_map: Lab image (any scale, will be auto-converted to CIE Lab)
        roi_mask: Boolean mask for region of interest
        cfg: Configuration dict
        rng_seed: Random seed for reproducibility
        return_mask: If True, also return the sampling mask
        r_map: Optional normalized radial distance map (T, R) with values [0, 1].
               If provided, enables radial presence analysis (Phase 3).
        sample_mask_override: Optional pre-computed mask from plate analysis.
               If provided, acts as a hard gate (see build_sampling_mask).
        plate_kpis: Optional KPIs from plate analysis for Soft Gate adjustment.

    Returns:
        samples: (N, 3) CIE Lab samples
        sample_indices: Flat indices of samples
        meta: Metadata dict
        warnings: List of warning codes
        mask (optional): Boolean sampling mask
    """
    # Ensure CIE Lab for consistency
    lab_map = to_cie_lab(lab_map, validate=False)

    mask, meta, warnings = build_sampling_mask(
        lab_map,
        roi_mask,
        cfg,
        rng_seed=rng_seed,
        r_map=r_map,
        sample_mask_override=sample_mask_override,
        plate_kpis=plate_kpis,
    )

    # Calculate indices (flat) for mapping back to image
    sample_indices = np.flatnonzero(mask.reshape(-1)).astype(np.int64)

    if sample_indices.size == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        if return_mask:
            return empty, sample_indices, meta, warnings, mask
        return empty, sample_indices, meta, warnings

    # Extract samples using flat indices to ensure alignment
    samples = lab_map.reshape(-1, lab_map.shape[-1])[sample_indices].astype(np.float32, copy=False)

    if return_mask:
        return samples, sample_indices, meta, warnings, mask
    return samples, sample_indices, meta, warnings
