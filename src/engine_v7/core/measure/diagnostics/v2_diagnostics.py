from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...geometry.lens_geometry import detect_lens_circle
from ...signature.radial_signature import to_polar
from ...utils import lab_cv8_to_cie, to_cie_lab
from ..metrics.ink_metrics import _lab_to_rgb, calculate_pairwise_deltaE, separation_ab
from ..segmentation.color_masks import build_color_masks_with_retry, compute_cluster_effective_densities
from ..segmentation.preprocess import build_roi_mask


def build_v2_diagnostics(
    bgr,
    cfg: Dict[str, Any],
    expected_ink_count: int | None,
    expected_ink_count_registry: int | None,
    expected_ink_count_input: int | None,
    polar_alpha: Optional[np.ndarray] = None,
    alpha_cfg: Optional[Dict[str, Any]] = None,
    *,
    precomputed_geom=None,
    precomputed_masks: Optional[Tuple] = None,
    precomputed_polar: Optional[np.ndarray] = None,
) -> Dict[str, Any] | None:
    if expected_ink_count is None:
        return None

    v2_cfg = cfg.get("v2_ink", {})
    if not v2_cfg.get("enabled", True):
        return None

    warnings: List[str] = []
    if (
        expected_ink_count_input is not None
        and expected_ink_count_registry is not None
        and expected_ink_count_input != expected_ink_count_registry
    ):
        warnings.append("EXPECTED_INK_COUNT_OVERRIDE")

    geom = precomputed_geom if precomputed_geom is not None else detect_lens_circle(bgr)

    if precomputed_masks is not None:
        color_masks, metadata = precomputed_masks
    else:
        color_masks, metadata = build_color_masks_with_retry(
            bgr,
            cfg,
            expected_k=int(expected_ink_count),
            geom=geom,
            confidence_threshold=float(v2_cfg.get("confidence_threshold", 0.7)),
            enable_retry=bool(v2_cfg.get("enable_retry", True)),
        )

    metadata = metadata or {}

    # Precompute polar once for reuse within this function
    _polar = (
        precomputed_polar
        if precomputed_polar is not None
        else to_polar(bgr, geom, R=cfg["polar"]["R"], T=cfg["polar"]["T"])
    )

    # Compute effective densities if alpha_cfg provided
    alpha_summary = None
    if alpha_cfg is not None or polar_alpha is not None:
        polar_lab = to_cie_lab(_polar)
        metadata, alpha_summary = compute_cluster_effective_densities(
            color_masks,
            metadata,
            polar_alpha=polar_alpha,
            polar_lab=polar_lab,
            alpha_cfg=alpha_cfg,
        )

    warnings.extend(metadata.get("warnings") or [])
    auto_estimation = metadata.get("auto_estimation")

    clusters: List[Dict[str, Any]] = []
    mean_labs: List[np.ndarray] = []
    area_ratios: List[float] = []

    for color_info in metadata.get("colors", []):
        lab_cv8 = np.array(color_info.get("lab_centroid", [0, 128, 128]), dtype=np.float32)
        lab_cie = lab_cv8_to_cie(lab_cv8.reshape(1, 1, 3)).reshape(3)
        mean_lab = lab_cie.astype(float).tolist()
        mean_rgb = _lab_to_rgb(lab_cie)
        area_ratio = float(color_info.get("area_ratio", 0.0))

        mean_labs.append(lab_cie)
        area_ratios.append(area_ratio)
        cluster_entry = {
            "area_ratio": area_ratio,
            "mean_lab": mean_lab,
            "mean_ab": [float(lab_cie[1]), float(lab_cie[2])],
            "mean_rgb": mean_rgb,
            "mean_hex": color_info.get("hex_ref"),
            "role": color_info.get("role", "ink"),
            "inkness_score": color_info.get("inkness_score"),
            "radial_presence_curve": color_info.get("radial_presence_curve"),
            "spatial_prior": color_info.get("spatial_prior"),
        }
        # Add effective_density fields if computed
        if "effective_density" in color_info:
            cluster_entry["effective_density"] = color_info["effective_density"]
            cluster_entry["alpha_used"] = color_info.get("alpha_used")
            cluster_entry["alpha_fallback_level"] = color_info.get("alpha_fallback_level")
        clusters.append(cluster_entry)

    k_expected = int(expected_ink_count)
    k_used = int(metadata.get("segmentation_k", metadata.get("k_used", len(clusters) or 0)))
    cluster_roles = [c.get("role", "ink") for c in clusters] if clusters else None

    min_area_ratio = float(min(area_ratios)) if area_ratios else 0.0
    min_area_warn = float(v2_cfg.get("min_area_ratio_warn", 0.03))
    d0 = float(v2_cfg.get("separation_d0", 3.0))
    k_sig = float(v2_cfg.get("separation_k", 1.0))

    seg_warnings: List[str] = []
    if min_area_ratio < min_area_warn:
        seg_warnings.append("INK_CLUSTER_TOO_SMALL")

    sep = separation_ab(mean_labs, area_ratios)
    min_delta = float(sep.get("min_deltaE", 0.0))
    if min_delta < d0:
        seg_warnings.append("INK_CLUSTER_OVERLAP_HIGH")

    sep_margin = float((min_delta - d0) / max(k_sig, 1e-6))
    warnings = seg_warnings + warnings

    lab = to_cie_lab(_polar)
    r_start = float(v2_cfg.get("roi_r_start", cfg["anomaly"]["r_start"]))
    r_end = float(v2_cfg.get("roi_r_end", cfg["anomaly"]["r_end"]))
    roi_mask, roi_meta = build_roi_mask(
        lab.shape[0],
        lab.shape[1],
        r_start,
        r_end,
        center_excluded_frac=float(cfg["anomaly"]["center_frac"]),
    )
    roi_lab_mean = lab[roi_mask > 0].mean(axis=0) if np.any(roi_mask) else None
    global_lab_mean = lab.reshape(-1, 3).mean(axis=0)

    palette_colors = []
    for c in metadata.get("colors", []):
        lab_cv8 = np.array(c.get("lab_centroid", [0, 128, 128]), dtype=np.float32)
        lab_cie = lab_cv8_to_cie(lab_cv8.reshape(1, 1, 3)).reshape(3)
        palette_colors.append(
            {
                "mean_rgb": _lab_to_rgb(lab_cie),
                "mean_hex": c.get("hex_ref"),
                "area_ratio": c.get("area_ratio"),
                "mean_lab": c.get("lab_centroid"),
                "mean_lab_cie": lab_cie.astype(float).tolist(),
                "role": c.get("role"),
            }
        )

    deltae_method = str(v2_cfg.get("deltaE_method", "76"))
    pairwise_deltae = calculate_pairwise_deltaE(mean_labs, method=deltae_method)
    quality = {
        "min_area_ratio": min_area_ratio,
        "min_deltaE_between_clusters": min_delta,
        "mean_deltaE_between_clusters": sep.get("mean_deltaE", 0.0),
        "deltaE_method": deltae_method,
        "separation_margin": sep_margin,
        "pairwise_deltaE": pairwise_deltae,
    }

    diagnostics = {
        "expected_ink_count": int(expected_ink_count),
        "expected_ink_count_registry": expected_ink_count_registry,
        "expected_ink_count_input": expected_ink_count_input,
        "roi": metadata.get("roi_meta") or roi_meta,
        "sampling": {
            **(metadata.get("sample_meta") or {}),
            "warnings": [w for w in warnings if w in ("INK_SAMPLING_EMPTY", "INK_SEPARATION_LOW_CONFIDENCE")],
        },
        "segmentation": {
            "k_used": k_used,
            "k_expected": k_expected,
            "cluster_roles": cluster_roles,
            "clusters": clusters,
            "quality": quality,
            "warnings": seg_warnings,
        },
        "palette": {
            "colors": palette_colors,
            "min_deltaE_between_clusters": quality.get("min_deltaE_between_clusters"),
            "mean_deltaE_between_clusters": quality.get("mean_deltaE_between_clusters"),
        },
        "direction": {
            "roi_lab_mean": roi_lab_mean.astype(float).tolist() if roi_lab_mean is not None else None,
            "global_lab_mean": global_lab_mean.astype(float).tolist(),
        },
        "warnings": warnings,
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
            "INK_SEGMENTATION_FAILED": "segmentation",
            "EXPECTED_INK_COUNT_OVERRIDE": "auto_k",
        }
        warnings_by_category = {"sampling": [], "segmentation": [], "auto_k": []}
        for w in warnings:
            category = warn_map.get(w)
            if category:
                warnings_by_category[category].append(w)
        diagnostics["warnings_by_category"] = warnings_by_category
    if auto_estimation is not None:
        diagnostics["auto_estimation"] = auto_estimation
    # Add alpha analysis if computed
    if alpha_summary is not None:
        diagnostics["alpha_summary"] = alpha_summary
    if metadata.get("alpha_analysis"):
        diagnostics["alpha_analysis"] = metadata["alpha_analysis"]
    return diagnostics
