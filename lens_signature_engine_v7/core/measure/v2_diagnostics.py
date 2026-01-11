from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..geometry.lens_geometry import detect_lens_circle
from ..signature.radial_signature import to_polar
from ..utils import bgr_to_lab
from .ink_metrics import build_cluster_stats, calculate_inkness_score, ensure_cie_lab, silhouette_ab_proxy
from .ink_segmentation import kmeans_segment
from .preprocess import build_roi_mask, sample_ink_candidates


def build_v2_diagnostics(
    bgr,
    cfg: Dict[str, Any],
    expected_ink_count: int | None,
    expected_ink_count_registry: int | None,
    expected_ink_count_input: int | None,
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

    geom = detect_lens_circle(bgr)
    polar = to_polar(bgr, geom, R=cfg["polar"]["R"], T=cfg["polar"]["T"])
    lab = bgr_to_lab(polar).astype(np.float32)

    r_start = float(v2_cfg.get("roi_r_start", cfg["anomaly"]["r_start"]))
    r_end = float(v2_cfg.get("roi_r_end", cfg["anomaly"]["r_end"]))
    roi_mask, roi_meta = build_roi_mask(
        lab.shape[0],
        lab.shape[1],
        r_start,
        r_end,
        center_excluded_frac=float(cfg["anomaly"]["center_frac"]),
    )

    rng_seed = v2_cfg.get("rng_seed", None)
    samples, sample_indices, sampling_meta, sampling_warnings = sample_ink_candidates(
        lab,
        roi_mask,
        cfg,
        rng_seed=rng_seed,
    )
    sampling_warnings = sampling_warnings or []

    def _effective_k(expected: int) -> int:
        return 2 if int(expected) == 1 else int(expected)

    k_expected = int(expected_ink_count)
    k = _effective_k(k_expected)
    kmeans_attempts = int(v2_cfg.get("kmeans_attempts", 10))
    labels, centers = kmeans_segment(
        samples,
        k,
        l_weight=float(v2_cfg.get("l_weight", 0.3)),
        attempts=kmeans_attempts,
        rng_seed=rng_seed,
    )
    if labels.size == 0:
        warnings.extend(sampling_warnings)
        warnings.append("INK_SEGMENTATION_FAILED")
        return {
            "expected_ink_count": int(expected_ink_count),
            "expected_ink_count_registry": expected_ink_count_registry,
            "expected_ink_count_input": expected_ink_count_input,
            "roi": roi_meta,
            "sampling": {**sampling_meta, "warnings": sampling_warnings},
            "segmentation": {"k_used": k, "clusters": [], "quality": {}},
            "warnings": warnings,
        }

    stats = build_cluster_stats(samples, labels, k)
    clusters = stats["clusters"]

    # --- Enrich clusters for decision_builder compatibility ---
    # 1) role 기본값
    for c in clusters:
        c.setdefault("role", "ink")

    # 2) radial_presence_curve: sampling indices를 polar (T,R)에서 r-bin으로 집계
    #    (샘플 기반 간이 버전. 3차에서 ROI 전체 기반으로 업그레이드 권장)
    T = int((cfg.get("polar") or {}).get("T", 0) or 0)
    R = int((cfg.get("polar") or {}).get("R", 0) or 0)
    curve_bins = int((cfg.get("v2_ink") or {}).get("radial_bins", 10) or 10)

    if T > 0 and R > 0 and sample_indices is not None and len(sample_indices) > 0:
        # flat index -> (t,r)
        # sample_indices comes from flattened (T,R) mask
        # Ensure types are correct for numpy operations
        t = (sample_indices // R).astype(np.int64)
        r = (sample_indices % R).astype(np.int64)

        # Calculate radial bin for each sample
        rb = np.clip((r * curve_bins) // max(R, 1), 0, curve_bins - 1)

        # per cluster curve
        for ci in range(k):
            mask_ci = labels == ci
            if mask_ci.sum() == 0:
                clusters[ci]["radial_presence_curve"] = [0.0] * curve_bins
                continue

            rb_ci = rb[mask_ci]
            # bincount needs int array
            hist = np.bincount(rb_ci.astype(np.int64), minlength=curve_bins).astype(np.float32)

            # Normalize so max is 1.0 (relative presence)
            hist_max = hist.max()
            if hist_max > 0:
                hist = hist / hist_max

            clusters[ci]["radial_presence_curve"] = [float(x) for x in hist]
    else:
        for ci in range(k):
            clusters[ci]["radial_presence_curve"] = [0.0] * curve_bins

    # 3) spatial_prior (간이): curve peak가 있으면 0.8~1.0, 없으면 0.5
    for c in clusters:
        curve = c.get("radial_presence_curve") or []
        peak = max(curve) if curve else 0.0
        c["spatial_prior"] = float(0.5 + 0.5 * peak)

    # 4) inkness_score (정식): ink_metrics 기반 (CIE Lab)
    for c in clusters:
        cluster_lab = c.get("mean_lab") or [0.0, 0.0, 0.0]
        cluster_lab = ensure_cie_lab(np.array(cluster_lab, dtype=np.float32)).reshape(
            3,
        )
        alpha = float(c.get("alpha_like", 0.0))
        comp = float(c.get("compactness", 0.0))
        sp = float(c.get("spatial_prior", 1.0))
        c["mean_lab"] = cluster_lab.tolist()
        c["inkness_score"] = float(
            calculate_inkness_score(cluster_lab, alpha_like=alpha, compactness=comp, spatial_prior=sp)
        )

    cluster_roles: Optional[List[str]] = None
    if k_expected == 1 and k == 2:
        labs = [c.get("mean_lab", [0.0, 0.0, 0.0]) for c in stats.get("clusters", [])]
        if len(labs) == 2:
            l_vals = [float(v[0]) for v in labs]
            ink_idx = int(np.argmin(l_vals))
            roles = ["gap", "gap"]
            roles[ink_idx] = "ink"
            cluster_roles = roles
            for idx, role in enumerate(roles):
                stats["clusters"][idx]["role"] = role
    min_area = stats["quality"].get("min_area_ratio", 0.0)
    min_area_warn = float(v2_cfg.get("min_area_ratio_warn", 0.03))
    d0 = float(v2_cfg.get("separation_d0", 3.0))
    k_sig = float(v2_cfg.get("separation_k", 1.0))
    seg_warnings: List[str] = []
    samp_warnings: List[str] = list(sampling_warnings)
    if min_area < min_area_warn:
        seg_warnings.append("INK_CLUSTER_TOO_SMALL")

    min_delta = float(stats["quality"].get("min_deltaE_between_clusters", 0.0))
    if min_delta < d0:
        seg_warnings.append("INK_CLUSTER_OVERLAP_HIGH")

    # separation margin (positive = safer, negative = overlap)
    sep_margin = float((min_delta - d0) / max(k_sig, 1e-6))
    stats["quality"]["separation_margin"] = sep_margin

    warnings = seg_warnings + samp_warnings

    auto_estimation = None
    if v2_cfg.get("auto_k_enabled", True):
        expanded = False
        candidate_set = {
            max(1, k - 1),
            k,
            max(1, k + 1),
        }
        if (
            "INK_SEPARATION_LOW_CONFIDENCE" in samp_warnings
            or "INK_CLUSTER_OVERLAP_HIGH" in seg_warnings
            or min_area < min_area_warn
        ):
            expanded = True
            max_k = int(v2_cfg.get("auto_k_expand_max", 4))
            candidate_set.update(range(1, max(1, max_k) + 1))

        k_candidates = sorted(candidate_set)
        scores: Dict[int, float | None] = {}
        l_weight = float(v2_cfg.get("l_weight", 0.3))
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

        # Calculate confidence
        conf_abs = (best_score - abs_min) / max(abs_span, 1e-6)
        conf_abs = float(np.clip(conf_abs, 0.0, 1.0))
        conf_gap = (best_score - second_score) / max(gap_span, 1e-6)
        conf_gap = float(np.clip(conf_gap, 0.0, 1.0))
        confidence = float(0.6 * conf_abs + 0.4 * conf_gap)

        # Flag if auto-k was ignored in favor of expected_k
        forced_to_expected = bool(best_k is not None and best_k != k)

        auto_estimation = {
            "k_candidates": [int(x) for x in k_candidates],
            "metric": "silhouette_ab_proxy",
            "scores": {str(k_i): scores[k_i] for k_i in k_candidates},
            "suggested_k": int(k),
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
        if best_k is not None and best_k != k_expected and confidence >= mismatch_thr:
            if not (k_expected == 1 and k == 2):
                warnings.append("INK_COUNT_MISMATCH_SUSPECTED")

    # 3. Calculate ROI-specific direction (Color Shift)
    # Comparison logic moved to diagnostics summary for ROI vs Global
    roi_lab_mean = lab[roi_mask > 0].mean(axis=0) if np.any(roi_mask) else None
    global_lab_mean = lab.reshape(-1, 3).mean(axis=0)

    palette_colors = []
    from ..utils import lab_cv8_to_cie  # Import utility

    for c in stats.get("clusters", []):
        lab_cv8 = np.array(c.get("mean_lab", [0, 128, 128]), dtype=np.float32)
        lab_cie = lab_cv8_to_cie(lab_cv8.reshape(1, 1, 3)).reshape(3)

        palette_colors.append(
            {
                "mean_rgb": c.get("mean_rgb"),
                "mean_hex": c.get("mean_hex"),
                "area_ratio": c.get("area_ratio"),
                "mean_lab": c.get("mean_lab"),  # Legacy cv8
                "mean_lab_cie": lab_cie.astype(float).tolist(),  # New CIE ✅
                "role": c.get("role"),
            }
        )

    diagnostics = {
        "expected_ink_count": int(expected_ink_count),
        "expected_ink_count_registry": expected_ink_count_registry,
        "expected_ink_count_input": expected_ink_count_input,
        "roi": roi_meta,
        "sampling": {**sampling_meta, "warnings": samp_warnings},
        "segmentation": {
            "k_used": k,
            "k_expected": k_expected,
            "cluster_roles": cluster_roles,
            "clusters": stats["clusters"],
            "quality": stats["quality"],
            "warnings": seg_warnings,
        },
        "palette": {
            "colors": palette_colors,
            "min_deltaE_between_clusters": stats["quality"].get("min_deltaE_between_clusters"),
            "mean_deltaE_between_clusters": stats["quality"].get("mean_deltaE_between_clusters"),
        },
        "direction": {
            "roi_lab_mean": roi_lab_mean.astype(float).tolist() if roi_lab_mean is not None else None,
            "global_lab_mean": global_lab_mean.astype(float).tolist(),
        },
        "warnings": warnings,
    }
    if auto_estimation is not None:
        diagnostics["auto_estimation"] = auto_estimation
    return diagnostics
