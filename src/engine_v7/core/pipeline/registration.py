"""STD registration evaluation extracted from analyzer.py."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Tuple

import numpy as np

from ..anomaly.angular_uniformity import angular_uniformity_score
from ..anomaly.anomaly_score import score_anomaly
from ..anomaly.blob_detector import detect_center_blobs
from ..gate.gate_engine import run_gate
from ..geometry.lens_geometry import detect_lens_circle
from ..signature.radial_signature import build_radial_signature, to_polar
from ..types import Decision, SignatureResult
from ._common import _maybe_apply_white_balance, _reason_meta
from ._signature import _evaluate_signature


def _registration_summary(std_models: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], list]:
    from ..utils import cie76_deltaE

    reg_cfg = cfg.get("registration", {})
    order_enabled = bool(reg_cfg.get("order_check_enabled", True))
    metric = reg_cfg.get("order_metric", "L_mean")
    direction = reg_cfg.get("order_direction", "asc")
    sep_threshold = float(reg_cfg.get("sep_threshold", 1.0))
    max_within_std = reg_cfg.get("max_within_std", None)
    max_within_val = float(max_within_std) if max_within_std is not None else None
    geom_center_max_px = reg_cfg.get("geom_center_max_px", None)
    geom_radius_max_ratio = reg_cfg.get("geom_radius_max_ratio", None)
    geom_center_max = float(geom_center_max_px) if geom_center_max_px is not None else None
    geom_radius_max = float(geom_radius_max_ratio) if geom_radius_max_ratio is not None else None
    warnings = []

    order_values: Dict[str, float] = {}
    order_ok = True
    if order_enabled:
        for mode in ["LOW", "MID", "HIGH"]:
            model = std_models.get(mode)
            if model is None:
                continue
            lab = model.radial_lab_mean
            if metric == "L_mean":
                val = float(np.mean(lab[:, 0]))
            else:
                val = float(np.mean(lab[:, 0]))
            order_values[mode] = val

        order_ok = False
        if all(m in order_values for m in ["LOW", "MID", "HIGH"]):
            if direction == "desc":
                order_ok = order_values["LOW"] > order_values["MID"] > order_values["HIGH"]
            else:
                order_ok = order_values["LOW"] < order_values["MID"] < order_values["HIGH"]

    separation = {}
    min_pairwise = None
    separation_ok = True
    if all(m in std_models for m in ["LOW", "MID", "HIGH"]):
        de_lm = cie76_deltaE(std_models["LOW"].radial_lab_mean, std_models["MID"].radial_lab_mean)
        de_mh = cie76_deltaE(std_models["MID"].radial_lab_mean, std_models["HIGH"].radial_lab_mean)
        de_lh = cie76_deltaE(std_models["LOW"].radial_lab_mean, std_models["HIGH"].radial_lab_mean)
        separation = {
            "LOW_MID": float(np.median(de_lm)),
            "MID_HIGH": float(np.median(de_mh)),
            "LOW_HIGH": float(np.median(de_lh)),
        }
        min_pairwise = min(separation.values())
        separation_ok = min_pairwise >= sep_threshold

    within_mode = {}
    unstable_modes = []
    for mode, model in std_models.items():
        std = getattr(model, "radial_lab_std", None)
        if std is None:
            within_mode[mode] = None
            continue
        val = float(np.mean(np.linalg.norm(std, axis=-1)))
        within_mode[mode] = val
        if max_within_val is not None and val > max_within_val:
            unstable_modes.append(mode)

    within_mode_ok = len(unstable_modes) == 0
    if max_within_val is None:
        within_mode_ok = None
        warnings.append("WITHIN_MODE_THRESHOLD_DISABLED")

    geom_consistency = {
        "center_drift_px": None,
        "radius_drift_ratio": None,
        "passed": None,
    }
    centers = []
    radii = []
    for model in std_models.values():
        centers.append((model.geom.cx, model.geom.cy))
        radii.append(model.geom.r)
    if centers and radii:
        cx_vals = np.array([c[0] for c in centers], dtype=np.float32)
        cy_vals = np.array([c[1] for c in centers], dtype=np.float32)
        r_vals = np.array(radii, dtype=np.float32)
        cx_mean = float(cx_vals.mean())
        cy_mean = float(cy_vals.mean())
        center_drift = np.max(np.hypot(cx_vals - cx_mean, cy_vals - cy_mean))
        r_mean = float(r_vals.mean()) if r_vals.size else 0.0
        radius_drift_ratio = float((r_vals.max() - r_vals.min()) / r_mean) if r_mean > 0 else None
        geom_consistency["center_drift_px"] = float(center_drift)
        geom_consistency["radius_drift_ratio"] = radius_drift_ratio

        if geom_center_max is not None or geom_radius_max is not None:
            passed = True
            if geom_center_max is not None and center_drift > geom_center_max:
                passed = False
            if geom_radius_max is not None and radius_drift_ratio is not None and radius_drift_ratio > geom_radius_max:
                passed = False
            geom_consistency["passed"] = passed
        else:
            warnings.append("GEOM_THRESHOLD_DISABLED")

    summary = {
        "order_enabled": order_enabled,
        "order_metric": metric,
        "order_direction": direction,
        "order_values": order_values,
        "order_ok": order_ok,
        "sep_threshold": sep_threshold,
        "separation": separation,
        "min_pairwise_separation": min_pairwise,
        "separation_ok": separation_ok,
        "within_mode_stability": within_mode,
        "within_mode_threshold": max_within_val,
        "within_mode_ok": within_mode_ok,
        "geom_consistency": geom_consistency,
        "warnings": warnings,
    }
    return summary, unstable_modes


def evaluate_registration_multi(test_bgr, std_models: Dict[str, Any], cfg: Dict[str, Any]) -> Decision:
    geom = detect_lens_circle(test_bgr)
    test_bgr, wb_meta = _maybe_apply_white_balance(test_bgr, geom, cfg)
    gate = run_gate(
        geom,
        test_bgr,
        center_off_max=cfg["gate"]["center_off_max"],
        blur_min=cfg["gate"]["blur_min"],
        illum_max=cfg["gate"]["illum_max"],
    )
    diag_on_fail = bool(cfg.get("gate", {}).get("diagnostic_on_fail", False))
    summary, unstable_modes = _registration_summary(std_models, cfg)

    label = "STD_ACCEPTABLE"
    reasons: list = []
    if not gate.passed:
        label = "STD_RETAKE"
        reasons = gate.reasons
    elif summary.get("order_enabled", True) and not summary.get("order_ok", False):
        label = "STD_UNSTABLE"
        reasons = ["MODE_ORDER_MISMATCH"]
    elif not summary.get("separation_ok", True):
        label = "STD_RETAKE"
        reasons = ["MODE_SEPARATION_LOW"]
    elif unstable_modes:
        label = "STD_RETAKE"
        reasons = [f"MODE_VARIANCE_HIGH:{m}" for m in unstable_modes]
    codes, messages = _reason_meta(reasons)

    debug = {"test_geom": asdict(geom), "std_geoms": {k: asdict(v.geom) for k, v in std_models.items()}}

    if gate.passed or diag_on_fail:
        any_model = next(iter(std_models.values()))
        polar = to_polar(test_bgr, geom, R=any_model.meta["R"], T=any_model.meta["T"])
        test_mean, _, _ = build_radial_signature(
            polar, r_start=cfg["signature"]["r_start"], r_end=cfg["signature"]["r_end"]
        )
        mode_sigs: Dict[str, SignatureResult] = {}
        for mode, m in std_models.items():
            mode_sigs[mode] = _evaluate_signature(test_mean, m, cfg)
        debug["signature_mode_scores"] = {k: asdict(v) for k, v in mode_sigs.items()}

        ang = angular_uniformity_score(polar, r_start=cfg["anomaly"]["r_start"], r_end=cfg["anomaly"]["r_end"])
        blobs = detect_center_blobs(
            test_bgr, geom, frac=cfg["anomaly"]["center_frac"], min_area=cfg["anomaly"]["center_blob_min_area"]
        )
        anom = score_anomaly(
            angular_uniformity=ang,
            center_blob_count=int(blobs["blob_count"]),
            angular_unif_max=cfg["anomaly"]["angular_unif_max"],
            center_blob_max=cfg["anomaly"]["center_blob_max"],
            blob_debug=blobs,
        )
        debug["anomaly_debug"] = asdict(anom)
    else:
        debug["inference_valid"] = False

    return Decision(
        label=label,
        reasons=reasons,
        reason_codes=codes,
        reason_messages=messages,
        gate=gate,
        signature=None,
        anomaly=None,
        debug=debug,
        phase="STD_REGISTRATION",
        registration_summary=summary,
        best_mode="",
        mode_scores={},
    )
