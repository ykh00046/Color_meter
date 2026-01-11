from __future__ import annotations

from typing import Any, Dict, List

from ..types import AnomalyResult


def score_anomaly(
    *,
    angular_uniformity: float,
    center_blob_count: int,
    angular_unif_max: float,
    center_blob_max: int,
    blob_debug: Dict[str, Any],
) -> AnomalyResult:
    reasons: List[str] = []
    if angular_uniformity > angular_unif_max:
        reasons.append("ANGULAR_UNIFORMITY_HIGH")
    if center_blob_count > center_blob_max:
        reasons.append("CENTER_BLOBS")

    return AnomalyResult(
        passed=(len(reasons) == 0),
        reasons=reasons,
        scores={"angular_uniformity": float(angular_uniformity), "center_blob_count": float(center_blob_count)},
        debug=blob_debug,
    )


def score_anomaly_relative(
    *,
    sample_features: Dict[str, float],
    baseline: Dict[str, Any],
    margins: Dict[str, float],
) -> AnomalyResult:
    reasons: List[str] = []

    blob_base = baseline.get("center_blob") or {}
    blob_count_max = float((blob_base.get("count") or {}).get("max", 0.0))
    blob_area_max = float((blob_base.get("total_area") or {}).get("max", 0.0))
    blob_count_margin = float(margins.get("blob_count", 0.0))
    blob_area_ratio = float(margins.get("blob_area_ratio", 0.0))

    if sample_features.get("center_blob_count", 0.0) > (blob_count_max + blob_count_margin):
        reasons.append("PATTERN_CENTER_BLOB_EXCESS")
    if blob_area_max > 0 and sample_features.get("center_blob_area", 0.0) > (blob_area_max * (1.0 + blob_area_ratio)):
        reasons.append("PATTERN_CENTER_BLOB_EXCESS")

    unif_max = float((baseline.get("angular_uniformity") or {}).get("max", 0.0))
    unif_margin = float(margins.get("uniformity", 0.0))
    if sample_features.get("angular_uniformity", 0.0) > (unif_max + unif_margin):
        reasons.append("PATTERN_UNIFORMITY_EXCESS")

    dot_base = baseline.get("dot") or {}
    cov_base = dot_base.get("coverage") or {}
    cov_min = float(cov_base.get("min", 0.0))
    cov_max = float(cov_base.get("max", 0.0))
    cov_eps = float(margins.get("coverage_eps", 0.0))
    cov_val = float(sample_features.get("dot_coverage", 0.0))
    if cov_val < (cov_min - cov_eps) or cov_val > (cov_max + cov_eps):
        reasons.append("PATTERN_DOT_COVERAGE_OUT_OF_BAND")

    edge_base = dot_base.get("edge_sharpness") or {}
    edge_min = float(edge_base.get("min", 0.0))
    edge_eps = float(margins.get("sharpness_eps", 0.0))
    edge_val = float(sample_features.get("dot_edge_sharpness", 0.0))
    if edge_val < (edge_min - edge_eps):
        reasons.append("PATTERN_EDGE_SHARPNESS_LOW")

    return AnomalyResult(
        passed=(len(reasons) == 0),
        reasons=reasons,
        scores=sample_features,
        debug={"baseline": baseline, "margins": margins},
    )
