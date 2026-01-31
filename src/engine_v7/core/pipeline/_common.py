"""Common helpers shared by evaluate(), evaluate_multi(), and per_color."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Tuple

import numpy as np

from ..anomaly.angular_uniformity import angular_uniformity_score
from ..anomaly.anomaly_score import score_anomaly, score_anomaly_relative
from ..anomaly.blob_detector import detect_center_blobs
from ..anomaly.defect_classifier import classify_defect
from ..anomaly.heatmap import anomaly_heatmap
from ..anomaly.pattern_baseline import extract_pattern_features
from ..decision.decision_builder import build_decision
from ..decision.decision_engine import decide
from ..gate.gate_engine import run_gate
from ..reason_codes import reason_codes, reason_messages, split_reason
from ..types import AnomalyResult, Decision, GateResult
from ..utils import apply_white_balance
from ._diagnostics import (
    _append_ok_features,
    _attach_features,
    _attach_v2_diagnostics,
    _attach_v3_summary,
    _attach_v3_trend,
)

# ---------------------------------------------------------------------------
# reason_meta
# ---------------------------------------------------------------------------


def _reason_meta(reasons: list, overrides: Dict[str, str] | None = None) -> Tuple[list, list]:
    codes = reason_codes(reasons)
    messages = []
    for r in reasons:
        code, _detail = split_reason(r)
        if overrides and code in overrides:
            messages.append(overrides[code])
        else:
            messages.append(reason_messages([r])[0])
    return codes, messages


# ---------------------------------------------------------------------------
# White balance
# ---------------------------------------------------------------------------


def _maybe_apply_white_balance(test_bgr, geom, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    wb_cfg = cfg.get("white_balance", {}) or {}
    if not wb_cfg.get("enabled", False):
        return test_bgr, {}
    balanced, meta = apply_white_balance(test_bgr, geom, wb_cfg)
    meta = meta or {}
    meta["enabled"] = True
    return balanced, meta


# ---------------------------------------------------------------------------
# Gate check with early return
# ---------------------------------------------------------------------------


def _run_gate_check(
    geom,
    test_bgr,
    cfg: Dict[str, Any],
    mode: str,
    pattern_baseline: Dict[str, Any] | None,
    *,
    extra_decision_kwargs: Dict[str, Any] | None = None,
) -> Tuple[GateResult, Decision | None]:
    """Run gate and return early Decision if applicable, else None."""
    gate = run_gate(
        geom,
        test_bgr,
        center_off_max=cfg["gate"]["center_off_max"],
        blur_min=cfg["gate"]["blur_min"],
        illum_max=cfg["gate"]["illum_max"],
    )
    diag_on_fail = bool(cfg.get("gate", {}).get("diagnostic_on_fail", False))
    kwargs = extra_decision_kwargs or {}

    # gate-only mode
    if mode == "gate":
        codes, messages = _reason_meta(gate.reasons)
        return gate, Decision(
            label="OK" if gate.passed else "RETAKE",
            reasons=gate.reasons,
            reason_codes=codes,
            reason_messages=messages,
            gate=gate,
            signature=None,
            anomaly=None,
            debug={"test_geom": asdict(geom), "inference_valid": gate.passed},
            diagnostics={"gate": asdict(gate)},
            phase="INSPECTION",
            **kwargs,
        )

    # gate failure
    if not gate.passed and not diag_on_fail:
        codes, messages = _reason_meta(gate.reasons)
        return gate, Decision(
            label="RETAKE",
            reasons=gate.reasons,
            reason_codes=codes,
            reason_messages=messages,
            gate=gate,
            signature=None,
            anomaly=None,
            debug={"inference_valid": False},
            phase="INSPECTION",
            **kwargs,
        )

    # baseline required check
    use_relative = bool(cfg.get("pattern_baseline", {}).get("use_relative", True))
    require_baseline = bool(cfg.get("pattern_baseline", {}).get("require", False))
    if require_baseline and use_relative and pattern_baseline is None:
        codes, messages = _reason_meta(["PATTERN_BASELINE_NOT_FOUND"])
        return gate, Decision(
            label="RETAKE",
            reasons=["PATTERN_BASELINE_NOT_FOUND"],
            reason_codes=codes,
            reason_messages=messages,
            gate=gate,
            signature=None,
            anomaly=None,
            debug={"inference_valid": False, "baseline_missing": True},
            phase="INSPECTION",
            **kwargs,
        )

    return gate, None  # proceed


# ---------------------------------------------------------------------------
# Anomaly evaluation
# ---------------------------------------------------------------------------


def _evaluate_anomaly(
    polar,
    test_bgr,
    geom,
    cfg: Dict[str, Any],
    pattern_baseline: Dict[str, Any] | None,
    use_relative: bool,
) -> AnomalyResult:
    """Anomaly scoring shared by evaluate() and evaluate_multi()."""
    ang = angular_uniformity_score(polar, r_start=cfg["anomaly"]["r_start"], r_end=cfg["anomaly"]["r_end"])
    blobs = detect_center_blobs(
        test_bgr, geom, frac=cfg["anomaly"]["center_frac"], min_area=cfg["anomaly"]["center_blob_min_area"]
    )
    if use_relative and pattern_baseline is not None:
        sample_features = extract_pattern_features(test_bgr, cfg=cfg)
        anom = score_anomaly_relative(
            sample_features=sample_features,
            baseline=pattern_baseline.get("features", {}),
            margins=pattern_baseline.get("policy", {}).get(
                "margins", cfg.get("pattern_baseline", {}).get("margins", {})
            ),
        )
        anom.debug["abs_scores"] = {
            "angular_uniformity": float(ang),
            "center_blob_count": float(blobs["blob_count"]),
        }
        anom.debug["blob_debug"] = blobs
    else:
        anom = score_anomaly(
            angular_uniformity=ang,
            center_blob_count=int(blobs["blob_count"]),
            angular_unif_max=cfg["anomaly"]["angular_unif_max"],
            center_blob_max=cfg["anomaly"]["center_blob_max"],
            blob_debug=blobs,
        )
    return anom


# ---------------------------------------------------------------------------
# Heatmap + defect classification
# ---------------------------------------------------------------------------


def _attach_heatmap_defect(
    decision: Decision,
    anom: AnomalyResult | None,
    polar,
    cfg: Dict[str, Any],
) -> None:
    """Attach anomaly heatmap and defect classification to decision."""
    if anom and cfg["anomaly"].get("enable_heatmap", True) and decision.label != "OK":
        hm = anomaly_heatmap(
            polar, ds_T=int(cfg["anomaly"]["heatmap_downsample_T"]), ds_R=int(cfg["anomaly"]["heatmap_downsample_R"])
        )
        decision.debug["anomaly_heatmap"] = hm

        if decision.label == "NG_PATTERN":
            dtype, conf, det = classify_defect(anom.scores, hm)
            anom.type = dtype
            anom.type_confidence = float(conf)
            anom.type_details = det


# ---------------------------------------------------------------------------
# Finalize decision (v2/v3 + qc_decision + ok_features)
# ---------------------------------------------------------------------------


def _finalize_decision(
    dec: Decision,
    test_bgr,
    cfg: Dict[str, Any],
    ok_log_context: Dict[str, Any] | None,
    pattern_baseline: Dict[str, Any] | None,
    mode: str,
    *,
    cached_geom=None,
    cached_polar=None,
    cached_masks=None,
) -> None:
    """Attach v2/v3 diagnostics, qc_decision, and OK features in-place."""
    if mode != "signature":
        _attach_v2_diagnostics(
            test_bgr,
            dec,
            cfg,
            ok_log_context,
            cached_geom=cached_geom,
            cached_masks=cached_masks,
            cached_polar=cached_polar,
        )

    v2_diag = dec.diagnostics.get("v2_diagnostics") or {}
    _attach_v3_summary(dec, v2_diag, cfg, ok_log_context)
    _attach_v3_trend(dec, ok_log_context)

    # qc_decision
    sample_clusters = (v2_diag.get("segmentation", {}) or {}).get("clusters", []) if v2_diag else []
    match_result = v2_diag.get("ink_match") if v2_diag else None

    raw_gate = dec.gate.scores if dec.gate else {}
    gate_scores = dict(raw_gate or {})
    if "sharpness_score" not in gate_scores and "sharpness_laplacian_var" in gate_scores:
        gate_scores["sharpness_score"] = gate_scores["sharpness_laplacian_var"]

    decision_json = build_decision(
        run_id=(ok_log_context or {}).get("run_id") or "",
        phase="INSPECTION",
        cfg=cfg,
        gate_scores=gate_scores,
        expected_inks=ok_log_context.get("expected_ink_count_input") if ok_log_context else None,
        sample_clusters=sample_clusters,
        match_result=match_result,
        deltae_summary_method="max",
        inkness_summary_method="min",
    )

    dec.ops = dec.ops or {}
    dec.ops["qc_decision"] = {
        "schema_version": "qc_decision.v1",
        **decision_json,
    }

    if (cfg.get("debug") or {}).get("include_full_qc_decision", False):
        dec.debug["full_qc_decision"] = decision_json

    dec.pattern_color = decision_json.get("pattern_color", {})

    _attach_features(dec, cfg, ok_log_context)
    _append_ok_features(test_bgr, dec, cfg, pattern_baseline, ok_log_context)
