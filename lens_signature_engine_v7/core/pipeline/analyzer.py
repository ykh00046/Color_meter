from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from ..geometry.lens_geometry import detect_lens_circle
from ..insight.summary import build_v3_summary
from ..insight.trend import build_v3_trend
from ..measure.color_masks import build_color_masks, build_color_masks_with_retry
from ..measure.ink_match import compute_cluster_deltas
from ..measure.v2_diagnostics import build_v2_diagnostics
from ..measure.v2_flags import build_v2_flags
from ..model_registry import load_ink_baseline
from ..reason_codes import reason_codes, reason_messages, split_reason
from ..signature.radial_signature import build_radial_signature, build_radial_signature_masked, to_polar
from ..signature.signature_compare import band_violation_v2, segment_stats, signature_compare, weighted_fail_ratio
from ..types import AnomalyResult, Decision, GateResult, SignatureResult
from ..utils import apply_white_balance, bgr_to_lab

logger = logging.getLogger(__name__)


def _build_k_by_segment(R: int, segments: list, segment_k: dict, default_k: float) -> np.ndarray:
    k_arr = np.full((R, 1), float(default_k), dtype=np.float32)
    for seg in segments:
        name = seg.get("name", "")
        k = float(segment_k.get(name, default_k))
        s = int(round(float(seg["start"]) * R))
        e = int(round(float(seg["end"]) * R))
        s = max(0, min(R - 1, s))
        e = max(s + 1, min(R, e))
        k_arr[s:e, 0] = k
    return k_arr


def _evaluate_signature(test_mean: np.ndarray, std_model, cfg: Dict[str, Any]) -> SignatureResult:
    comp = signature_compare(test_mean, std_model.radial_lab_mean)
    de_curve = comp["delta_e_curve"]

    R = int(comp["test_mean_aligned"].shape[0])
    segments = cfg["signature"].get("segments", [])
    default_k = float(cfg["signature"]["band_k"])
    use_segment_k = bool(cfg["signature"].get("use_segment_k", False)) and bool(segments)
    seg_k_map = cfg["signature"].get("segment_k", {}) if use_segment_k else {}

    if use_segment_k:
        k_arr = _build_k_by_segment(R, segments, seg_k_map, default_k)
        band = band_violation_v2(
            comp["test_mean_aligned"],
            std_model.radial_lab_mean,
            getattr(std_model, "radial_lab_std", None),
            k=k_arr,
        )
    else:
        band = band_violation_v2(
            comp["test_mean_aligned"],
            std_model.radial_lab_mean,
            getattr(std_model, "radial_lab_std", None),
            k=default_k,
        )

    fail_mask = band["fail_mask"]
    base_fail_ratio = float(band["fail_ratio"])

    seg_stats = segment_stats(de_curve, fail_mask, segments) if segments else {}
    w_fail_ratio = weighted_fail_ratio(seg_stats) if seg_stats else base_fail_ratio
    use_weighted = bool(cfg["signature"].get("use_weighted_segments", False)) and bool(seg_stats)
    eff_fail_ratio = w_fail_ratio if use_weighted else base_fail_ratio

    fail_regions = np.where(fail_mask)[0].tolist()

    sig_reasons = []
    if comp["corr"] < cfg["signature"]["corr_min"]:
        sig_reasons.append("SIGNATURE_CORR_LOW")
    if comp["delta_e_p95"] > cfg["signature"]["de_p95_max"]:
        sig_reasons.append("DELTAE_P95_HIGH")
    if comp["delta_e_mean"] > cfg["signature"]["de_mean_max"]:
        sig_reasons.append("DELTAE_MEAN_HIGH")

    use_band = bool(cfg["signature"].get("use_band_model", True))
    if use_band and eff_fail_ratio > cfg["signature"]["fail_ratio_max"]:
        sig_reasons.append("BAND_VIOLATION_HIGH")

    seg_violation_reasons = []
    seg_cap = cfg["signature"].get("segment_fail_ratio_max", None)
    if seg_cap is not None and bool(seg_stats):
        for name, st in seg_stats.items():
            if float(st.get("fail_ratio", 0.0)) > float(seg_cap):
                seg_violation_reasons.append(f"SEGMENT_BAND_VIOLATION:{name}")

    return SignatureResult(
        passed=(len(sig_reasons) == 0),
        score_corr=float(comp["corr"]),
        delta_e_mean=float(comp["delta_e_mean"]),
        delta_e_p95=float(comp["delta_e_p95"]),
        fail_ratio=float(eff_fail_ratio),
        fail_regions_r=fail_regions[:200],
        reasons=sig_reasons,
        flags={"segment_violation_reasons": seg_violation_reasons},
        debug={
            "delta_e_curve": de_curve.astype(np.float32).tolist(),
            "delta_e00_mean": float(comp.get("delta_e00_mean", 0.0)),
            "delta_e00_p95": float(comp.get("delta_e00_p95", 0.0)),
            "band_fail_ratio": float(base_fail_ratio),
            "band_fail_ratio_weighted": float(w_fail_ratio),
            "segment_stats": seg_stats,
            "band_weighted_enabled": bool(use_weighted),
            "segment_k_enabled": bool(use_segment_k),
            "segment_k_map": seg_k_map,
            "test_mean_aligned": comp["test_mean_aligned"].astype(float).tolist(),
        },
    )


def _pick_best_mode(mode_sigs: Dict[str, SignatureResult]) -> str:
    # primary: fail_ratio, then delta_e_p95, then -corr
    items = []
    for mode, sig in mode_sigs.items():
        items.append((mode, sig.fail_ratio, sig.delta_e_p95, -sig.score_corr))
    items.sort(key=lambda x: (x[1], x[2], x[3]))
    return items[0][0]


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


def _mean_grad(l_map: np.ndarray) -> float:
    if l_map.size == 0:
        return 0.0
    # Handle 1D case (radial profile)
    if l_map.ndim == 1:
        grad = np.abs(np.diff(l_map))
        return float(grad.mean()) if grad.size > 0 else 0.0
    # Handle 2D case (polar map)
    gx = np.abs(np.diff(l_map, axis=1))
    gy = np.abs(np.diff(l_map, axis=0))
    return float((gx.mean() + gy.mean()) / 2.0)


def _maybe_apply_white_balance(test_bgr, geom, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    wb_cfg = cfg.get("white_balance", {}) or {}
    if not wb_cfg.get("enabled", False):
        return test_bgr, {}
    balanced, meta = apply_white_balance(test_bgr, geom, wb_cfg)
    meta = meta or {}
    meta["enabled"] = True
    return balanced, meta


def _compute_worst_case(test_lab_map: np.ndarray, std_lab_mean: np.ndarray) -> Dict[str, Any]:
    if test_lab_map.size == 0 or std_lab_mean.size == 0:
        return {}
    T = test_lab_map.shape[0]
    R = test_lab_map.shape[1]
    std_mean = std_lab_mean
    if std_mean.shape[0] != R:
        x_old = np.linspace(0, 1, std_mean.shape[0])
        x_new = np.linspace(0, 1, R)
        std_mean = np.vstack([np.interp(x_new, x_old, std_mean[:, i]) for i in range(3)]).T.astype(np.float32)
    std_lab_map = np.repeat(std_mean[None, :, :], T, axis=0)

    from ..utils import cie76_deltaE

    de_map = cie76_deltaE(test_lab_map, std_lab_map)

    if de_map.size == 0:
        return {}
    max_val = float(np.max(de_map))
    p95 = float(np.percentile(de_map, 95))
    p99 = float(np.percentile(de_map, 99))
    mean = float(np.mean(de_map))
    std = float(np.std(de_map))
    hotspot_idx = np.unravel_index(int(np.argmax(de_map)), de_map.shape)
    return {
        "schema_version": "worst_case.v1",
        "max_deltaE": max_val,
        "p95_deltaE": p95,
        "p99_deltaE": p99,
        "mean_deltaE": mean,
        "std_deltaE": std,
        "coverage_ratio": float(np.mean(de_map >= p95)),
        "hotspot": {"theta_bin": int(hotspot_idx[0]), "r_bin": int(hotspot_idx[1]), "value": max_val},
    }


def _compute_diagnostics(
    test_lab_map: np.ndarray, std_lab_mean: np.ndarray, cfg: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str], Dict[str, str]]:
    from ..utils import cie76_deltaE, cie2000_deltaE, lab_cv8_to_cie

    diag_cfg = cfg.get("diagnostics", {})
    dL_th = float(diag_cfg.get("delta_L_threshold", 1.0))
    da_th = float(diag_cfg.get("delta_a_threshold", 0.8))
    db_th = float(diag_cfg.get("delta_b_threshold", 0.8))

    test_lab_cie = lab_cv8_to_cie(test_lab_map)
    std_lab_mean_cie = lab_cv8_to_cie(std_lab_mean)

    std_mean_cie = std_lab_mean_cie.mean(axis=0)
    sample_mean_cie = test_lab_cie.reshape(-1, 3).mean(axis=0)
    delta_cie = sample_mean_cie - std_mean_cie

    T = test_lab_cie.shape[0]
    R = test_lab_cie.shape[1]
    r0 = int(R * float(cfg["signature"]["r_start"]))
    r1 = int(R * float(cfg["signature"]["r_end"]))
    r0 = max(0, min(R - 1, r0))
    r1 = max(r0 + 1, min(R, r1))

    roi_cie = test_lab_cie[:, r0:r1, :].reshape(-1, 3)
    sample_mean_roi_cie = roi_cie.mean(axis=0) if roi_cie.size else sample_mean_cie
    std_mean_roi_cie = std_lab_mean_cie.mean(axis=0) if std_lab_mean_cie.size else std_mean_cie

    de76_roi = float(cie76_deltaE(sample_mean_roi_cie, std_mean_roi_cie))
    de2000_roi = float(cie2000_deltaE(sample_mean_roi_cie, std_mean_roi_cie))

    reason_codes_extra: List[str] = []
    reason_messages_extra: Dict[str, str] = {}

    if delta_cie[2] >= db_th:
        reason_codes_extra.append("COLOR_SHIFT_YELLOW")
        reason_messages_extra["COLOR_SHIFT_YELLOW"] = f"Color shift yellow (delta_b={delta_cie[2]:+.2f} [CIE])."
    elif delta_cie[2] <= -db_th:
        reason_codes_extra.append("COLOR_SHIFT_BLUE")
        reason_messages_extra["COLOR_SHIFT_BLUE"] = f"Color shift blue (delta_b={delta_cie[2]:+.2f} [CIE])."
    if delta_cie[1] >= da_th:
        reason_codes_extra.append("COLOR_SHIFT_RED")
        reason_messages_extra["COLOR_SHIFT_RED"] = f"Color shift red (delta_a={delta_cie[1]:+.2f} [CIE])."
    elif delta_cie[1] <= -da_th:
        reason_codes_extra.append("COLOR_SHIFT_GREEN")
        reason_messages_extra["COLOR_SHIFT_GREEN"] = f"Color shift green (delta_a={delta_cie[1]:+.2f} [CIE])."
    if delta_cie[0] <= -dL_th:
        reason_codes_extra.append("COLOR_SHIFT_DARK")
        reason_messages_extra["COLOR_SHIFT_DARK"] = f"Color shift dark (delta_L={delta_cie[0]:+.2f} [CIE])."
    elif delta_cie[0] >= dL_th:
        reason_codes_extra.append("COLOR_SHIFT_LIGHT")
        reason_messages_extra["COLOR_SHIFT_LIGHT"] = f"Color shift light (delta_L={delta_cie[0]:+.2f} [CIE])."

    std_l_cie = std_lab_mean_cie[..., 0]
    sample_l_cie = test_lab_cie[..., 0]

    cov_l_delta_cie = float(diag_cfg.get("coverage_l_delta", 5.0)) * (100.0 / 255.0)

    std_l_mean_cie = float(std_l_cie.mean())
    sample_l_mean_cie = float(sample_l_cie.mean())
    std_cov = float(np.mean(std_l_cie < (std_l_mean_cie - cov_l_delta_cie)))
    sample_cov = float(np.mean(sample_l_cie < (sample_l_mean_cie - cov_l_delta_cie)))
    cov_delta_pp = (sample_cov - std_cov) * 100.0

    std_edge = _mean_grad(std_l_cie)
    sample_edge = _mean_grad(sample_l_cie)
    edge_delta = sample_edge - std_edge
    edge_th = float(diag_cfg.get("edge_sharpness_delta_threshold", 0.1))
    cov_pp_th = float(diag_cfg.get("coverage_delta_pp_threshold", 2.0))

    if cov_delta_pp >= cov_pp_th:
        reason_codes_extra.append("PATTERN_DOT_COVERAGE_HIGH")
        reason_messages_extra["PATTERN_DOT_COVERAGE_HIGH"] = f"Dot coverage high (delta={cov_delta_pp:+.2f}pp)."
    elif cov_delta_pp <= -cov_pp_th:
        reason_codes_extra.append("PATTERN_DOT_COVERAGE_LOW")
        reason_messages_extra["PATTERN_DOT_COVERAGE_LOW"] = f"Dot coverage low (delta={cov_delta_pp:+.2f}pp)."
    if edge_delta <= -edge_th:
        reason_codes_extra.append("PATTERN_EDGE_BLUR")
        reason_messages_extra["PATTERN_EDGE_BLUR"] = f"Edge sharpness down (delta={edge_delta:+.3f})."
    if cov_delta_pp >= cov_pp_th and edge_delta <= -edge_th:
        reason_codes_extra.append("PATTERN_DOT_SPREAD")
        reason_messages_extra["PATTERN_DOT_SPREAD"] = f"Dot spread (cov={cov_delta_pp:+.2f}pp, edge={edge_delta:+.3f})."

    diagnostics = {
        "color": {
            "overall": {
                "std_lab_mean_cie": std_mean_cie.astype(float).tolist(),
                "sample_lab_mean_cie": sample_mean_cie.astype(float).tolist(),
                "std_lab_mean_cv8": std_lab_mean.mean(axis=0).astype(float).tolist(),
                "deltaE76_roi": float(de76_roi),
                "deltaE2000_roi": float(de2000_roi),
                "direction": {
                    "delta_L": float(delta_cie[0]),
                    "delta_a": float(delta_cie[1]),
                    "delta_b": float(delta_cie[2]),
                },
                "_scale_note": "CIE scale: L:0-100, a/b:+-128",
            }
        },
        "pattern": {
            "dot": {
                "coverage_std": std_cov,
                "coverage_sample": sample_cov,
                "coverage_delta_pp": float(cov_delta_pp),
                "edge_sharpness_std": float(std_edge),
                "edge_sharpness_sample": float(sample_edge),
                "edge_sharpness_delta": float(edge_delta),
            }
        },
        "worst_case": _compute_worst_case(test_lab_map, std_lab_mean),
    }
    return diagnostics, reason_codes_extra, reason_messages_extra


def _append_ok_features(
    test_bgr,
    decision: Decision,
    cfg: Dict[str, Any],
    pattern_baseline: Dict[str, Any] | None,
    ok_log_context: Dict[str, Any] | None,
) -> None:
    if decision.phase != "INSPECTION" or decision.label != "OK":
        return
    if pattern_baseline is None or ok_log_context is None:
        return

    sku = ok_log_context.get("sku", "")
    ink = ok_log_context.get("ink", "")
    models_root = ok_log_context.get("models_root", "")
    if not (sku and ink and models_root):
        return

    baseline_path = pattern_baseline.get("path", "")
    baseline_schema = pattern_baseline.get("schema_version", "")
    active_versions = pattern_baseline.get("active_versions", {})
    baseline_id = Path(baseline_path).stem if baseline_path else "UNKNOWN"

    try:
        log_dir = Path(models_root) / "pattern_baselines" / "ok_logs" / sku / ink
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"OKF_{baseline_id}.jsonl"

        features = extract_pattern_features(test_bgr, cfg=cfg)
        entry = {
            "ts": datetime.now().isoformat(),
            "sku": sku,
            "ink": ink,
            "active_id": baseline_id,
            "result_path": ok_log_context.get("result_path", ""),
            "features": features,
            "baseline": {
                "pattern_baseline_path": baseline_path,
                "pattern_baseline_schema": baseline_schema,
                "active_versions": active_versions,
            },
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        decision.debug["ok_feature_log_failed"] = True
        decision.debug["ok_feature_log_error"] = str(exc)


def _attach_v2_diagnostics(
    test_bgr,
    decision: Decision,
    cfg: Dict[str, Any],
    ok_log_context: Dict[str, Any] | None,
) -> None:
    if decision.phase != "INSPECTION":
        return
    if ok_log_context is None:
        return
    expected_input = ok_log_context.get("expected_ink_count_input")
    expected_registry = ok_log_context.get("expected_ink_count_registry")
    expected = expected_input if expected_input is not None else expected_registry
    if expected is None:
        return
    diagnostics = build_v2_diagnostics(
        test_bgr,
        cfg,
        expected_ink_count=int(expected),
        expected_ink_count_registry=expected_registry,
        expected_ink_count_input=expected_input,
    )
    if diagnostics is None:
        return
    models_root = ok_log_context.get("models_root")
    sku = ok_log_context.get("sku")
    ink = ok_log_context.get("ink")
    if models_root and sku and ink:
        ink_baseline, _ = load_ink_baseline(models_root, sku, ink)
        if ink_baseline:
            sample_clusters = diagnostics.get("segmentation", {}).get("clusters", [])

            v2_cfg = cfg.get("v2_ink", {})
            deltaE_method = str(v2_cfg.get("deltaE_method", cfg.get("deltaE_method", "76")))

            match = compute_cluster_deltas(ink_baseline, sample_clusters, deltaE_method=deltaE_method)
            match["warning"] = None  # Will be set by decision builder if needed
            diagnostics["ink_match"] = match
            diagnostics.setdefault("references", {})
            diagnostics["references"]["ink_baseline_path"] = ink_baseline.get("path", "")
            diagnostics["references"]["ink_baseline_schema"] = ink_baseline.get("schema_version", "")
    decision.diagnostics.setdefault("v2_diagnostics", diagnostics)


def _attach_v3_summary(
    decision: Decision,
    v2_diag: Dict[str, Any],
    cfg: Dict[str, Any],
    ok_log_context: Dict[str, Any] | None,
) -> None:
    if decision.phase != "INSPECTION":
        return
    try:
        summary = build_v3_summary(v2_diag, decision, cfg, ok_log_context)
        if summary:
            decision.v3_summary = summary
    except Exception as e:
        decision.debug.setdefault("v3", {})["summary_error"] = str(e)


def _attach_v3_trend(
    decision: Decision,
    ok_log_context: Dict[str, Any] | None,
) -> None:
    if decision.phase != "INSPECTION":
        return

    decisions_history = []
    window = 20

    if ok_log_context:
        if "v3_trend_decisions" in ok_log_context:
            decisions_history = ok_log_context["v3_trend_decisions"]
        elif "trend_log_path" in ok_log_context:
            path = Path(ok_log_context["trend_log_path"])
            if path.exists():
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                    for line in lines[-50:]:
                        try:
                            decisions_history.append(json.loads(line))
                        except:
                            pass
                except Exception:
                    pass

        if "trend_window_requested" in ok_log_context:
            window = int(ok_log_context["trend_window_requested"])

    trend = build_v3_trend(decisions_history, window_requested=window)
    if trend:
        decision.v3_trend = trend


def _attach_pattern_color(decision: Decision, cfg: Dict[str, Any]) -> None:
    # Now handled by decision_builder, keeping for safety if needed
    pass


def _attach_features(decision: Decision, cfg: Dict[str, Any], ok_log_context: Dict[str, Any] | None) -> None:
    pass  # Reserved


def evaluate(
    test_bgr,
    std_model,
    cfg: Dict[str, Any],
    pattern_baseline: Dict[str, Any] | None = None,
    ok_log_context: Dict[str, Any] | None = None,
    mode: str = "all",
) -> Decision:
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

    if mode == "gate":
        codes, messages = _reason_meta(gate.reasons)
        return Decision(
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
        )

    if not gate.passed and not diag_on_fail:
        codes, messages = _reason_meta(gate.reasons)
        return Decision(
            label="RETAKE",
            reasons=gate.reasons,
            reason_codes=codes,
            reason_messages=messages,
            gate=gate,
            signature=None,
            anomaly=None,
            debug={"inference_valid": False},
            phase="INSPECTION",
        )

    use_relative = bool(cfg.get("pattern_baseline", {}).get("use_relative", True))
    require_baseline = bool(cfg.get("pattern_baseline", {}).get("require", False))
    if require_baseline and use_relative and pattern_baseline is None:
        codes, messages = _reason_meta(["PATTERN_BASELINE_NOT_FOUND"])
        return Decision(
            label="RETAKE",
            reasons=["PATTERN_BASELINE_NOT_FOUND"],
            reason_codes=codes,
            reason_messages=messages,
            gate=gate,
            signature=None,
            anomaly=None,
            debug={"inference_valid": False, "baseline_missing": True},
            phase="INSPECTION",
        )

    polar = to_polar(test_bgr, geom, R=std_model.meta["R"], T=std_model.meta["T"])

    sig = None
    if mode != "ink":
        test_mean, _, _ = build_radial_signature(
            polar, r_start=cfg["signature"]["r_start"], r_end=cfg["signature"]["r_end"]
        )
        sig = _evaluate_signature(test_mean, std_model, cfg)

    anom = None
    if mode not in ["signature", "ink"]:
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

    label, reasons = decide(gate, sig, anom)

    diagnostics = {}
    extra_codes = []
    extra_messages = {}

    if mode != "ink":
        test_lab_map = bgr_to_lab(polar)
        diagnostics, extra_codes, extra_messages = _compute_diagnostics(test_lab_map, std_model.radial_lab_mean, cfg)
        if sig:
            diagnostics["radial"] = {
                "summary": {
                    "score_corr": float(sig.score_corr),
                    "delta_e_mean": float(sig.delta_e_mean),
                    "delta_e_p95": float(sig.delta_e_p95),
                    "fail_ratio": float(sig.fail_ratio),
                    "best_mode": "SINGLE",
                }
            }

    if pattern_baseline is not None:
        diagnostics.setdefault("references", {})
        diagnostics["references"]["pattern_baseline_path"] = pattern_baseline.get("path", "")
        diagnostics["references"]["pattern_baseline_schema"] = pattern_baseline.get("schema_version", "")
        diagnostics["references"]["pattern_baseline_active_versions"] = pattern_baseline.get("active_versions", {})
    elif ok_log_context and ok_log_context.get("active_versions"):
        diagnostics.setdefault("references", {})
        diagnostics["references"]["active_versions"] = ok_log_context.get("active_versions", {})
    if wb_meta:
        diagnostics["white_balance"] = wb_meta
    if label != "OK":
        reasons = reasons + extra_codes
        codes, messages = _reason_meta(reasons, extra_messages)
    else:
        codes, messages = _reason_meta(reasons)

    debug = {"test_geom": asdict(geom), "std_geom": asdict(std_model.geom)}
    if not gate.passed:
        debug["inference_valid"] = False

    if anom and cfg["anomaly"].get("enable_heatmap", True) and label != "OK":
        hm = anomaly_heatmap(
            polar, ds_T=int(cfg["anomaly"]["heatmap_downsample_T"]), ds_R=int(cfg["anomaly"]["heatmap_downsample_R"])
        )
        debug["anomaly_heatmap"] = hm

    if label == "NG_PATTERN" and anom:
        dtype, conf, det = classify_defect(anom.scores, hm)
        anom.type = dtype
        anom.type_confidence = float(conf)
        anom.type_details = det

    decision = Decision(
        label=label,
        reasons=reasons,
        reason_codes=codes,
        reason_messages=messages,
        gate=gate,
        signature=sig,
        anomaly=anom,
        debug=debug,
        diagnostics=diagnostics,
        phase="INSPECTION",
    )

    if mode != "signature":
        _attach_v2_diagnostics(test_bgr, decision, cfg, ok_log_context)

    v2_diag = decision.diagnostics.get("v2_diagnostics") or {}
    _attach_v3_summary(decision, v2_diag, cfg, ok_log_context)
    _attach_v3_trend(decision, ok_log_context)

    # Final Integration: Build Decision (A. ops 덮어쓰기 방지)
    sample_clusters = (v2_diag.get("segmentation", {}) or {}).get("clusters", []) if v2_diag else []
    match_result = v2_diag.get("ink_match") if v2_diag else None

    # A. gate_scores 키 호환성 레이어 (gate_engine 개선판 대응)
    raw_gate = decision.gate.scores if decision.gate else {}
    gate_scores = dict(raw_gate or {})
    if "sharpness_score" not in gate_scores and "sharpness_laplacian_var" in gate_scores:
        gate_scores["sharpness_score"] = gate_scores["sharpness_laplacian_var"]

    thresholds_config = cfg.get("thresholds", {}) or {}
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

    # A. ✅ 절대 덮어쓰지 말고 하위 키로 넣기 (기존 ops_judgment 보존)
    decision.ops = decision.ops or {}
    decision.ops["qc_decision"] = {
        "schema_version": "qc_decision.v1",
        **decision_json,
    }

    # full dump는 debug flag 있을 때만
    if (cfg.get("debug") or {}).get("include_full_qc_decision", False):
        decision.debug["full_qc_decision"] = decision_json

    decision.pattern_color = decision_json.get("pattern_color", {})

    _attach_features(decision, cfg, ok_log_context)
    _append_ok_features(test_bgr, decision, cfg, pattern_baseline, ok_log_context)
    return decision


def evaluate_multi(
    test_bgr,
    std_models: Dict[str, Any],
    cfg: Dict[str, Any],
    pattern_baseline: Dict[str, Any] | None = None,
    ok_log_context: Dict[str, Any] | None = None,
    mode: str = "all",
) -> Tuple[Decision, Dict[str, SignatureResult]]:
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

    if mode == "gate":
        codes, messages = _reason_meta(gate.reasons)
        dec = Decision(
            label="OK" if gate.passed else "RETAKE",
            reasons=gate.reasons,
            reason_codes=codes,
            reason_messages=messages,
            gate=gate,
            signature=None,
            anomaly=None,
            debug={"test_geom": asdict(geom), "inference_valid": gate.passed},
            diagnostics={"gate": asdict(gate)},
            best_mode="",
            mode_scores={},
            phase="INSPECTION",
        )
        return dec, {}

    if not gate.passed and not diag_on_fail:
        codes, messages = _reason_meta(gate.reasons)
        dec = Decision(
            label="RETAKE",
            reasons=gate.reasons,
            reason_codes=codes,
            reason_messages=messages,
            gate=gate,
            signature=None,
            anomaly=None,
            debug={"inference_valid": False},
            best_mode="",
            mode_scores={},
            phase="INSPECTION",
        )
        return dec, {}

    use_relative = bool(cfg.get("pattern_baseline", {}).get("use_relative", True))
    require_baseline = bool(cfg.get("pattern_baseline", {}).get("require", False))
    if require_baseline and use_relative and pattern_baseline is None:
        codes, messages = _reason_meta(["PATTERN_BASELINE_NOT_FOUND"])
        dec = Decision(
            label="RETAKE",
            reasons=["PATTERN_BASELINE_NOT_FOUND"],
            reason_codes=codes,
            reason_messages=messages,
            gate=gate,
            signature=None,
            anomaly=None,
            debug={"inference_valid": False, "baseline_missing": True},
            best_mode="",
            mode_scores={},
            phase="INSPECTION",
        )
        return dec, {}

    any_model = next(iter(std_models.values()))
    polar = to_polar(test_bgr, geom, R=any_model.meta["R"], T=any_model.meta["T"])

    best_mode = ""
    best_sig = None
    mode_sigs = {}

    if mode != "ink":
        test_mean, _, _ = build_radial_signature(
            polar, r_start=cfg["signature"]["r_start"], r_end=cfg["signature"]["r_end"]
        )
        for m_mode, m in std_models.items():
            mode_sigs[m_mode] = _evaluate_signature(test_mean, m, cfg)

        best_mode = _pick_best_mode(mode_sigs)
        best_sig = mode_sigs[best_mode]

    anom = None
    if mode not in ["signature", "ink"]:
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

    label, reasons = decide(gate, best_sig, anom)

    diagnostics = {}
    extra_codes = []
    extra_messages = {}

    if mode != "ink" and best_mode:
        test_lab_map = bgr_to_lab(polar)
        diagnostics, extra_codes, extra_messages = _compute_diagnostics(
            test_lab_map, std_models[best_mode].radial_lab_mean, cfg
        )
        if best_sig:
            diagnostics["radial"] = {
                "summary": {
                    "score_corr": float(best_sig.score_corr),
                    "delta_e_mean": float(best_sig.delta_e_mean),
                    "delta_e_p95": float(best_sig.delta_e_p95),
                    "fail_ratio": float(best_sig.fail_ratio),
                    "best_mode": best_mode,
                }
            }

    if pattern_baseline is not None:
        diagnostics.setdefault("references", {})
        diagnostics["references"]["pattern_baseline_path"] = pattern_baseline.get("path", "")
        diagnostics["references"]["pattern_baseline_schema"] = pattern_baseline.get("schema_version", "")
        diagnostics["references"]["pattern_baseline_active_versions"] = pattern_baseline.get("active_versions", {})
    elif ok_log_context and ok_log_context.get("active_versions"):
        diagnostics.setdefault("references", {})
        diagnostics["references"]["active_versions"] = ok_log_context.get("active_versions", {})
    if wb_meta:
        diagnostics["white_balance"] = wb_meta
    if label != "OK":
        reasons = reasons + extra_codes
        codes, messages = _reason_meta(reasons, extra_messages)
    else:
        codes, messages = _reason_meta(reasons)

    debug = {"test_geom": asdict(geom)}
    if not gate.passed:
        debug["inference_valid"] = False
    debug["std_geoms"] = {k: asdict(v.geom) for k, v in std_models.items()}

    if anom and cfg["anomaly"].get("enable_heatmap", True) and label != "OK":
        hm = anomaly_heatmap(
            polar, ds_T=int(cfg["anomaly"]["heatmap_downsample_T"]), ds_R=int(cfg["anomaly"]["heatmap_downsample_R"])
        )
        debug["anomaly_heatmap"] = hm

    if label == "NG_PATTERN" and anom:
        dtype, conf, det = classify_defect(anom.scores, hm)
        anom.type = dtype
        anom.type_confidence = float(conf)
        anom.type_details = det

    mode_scores = {k: asdict(v) for k, v in mode_sigs.items()}

    dec = Decision(
        label=label,
        reasons=reasons,
        reason_codes=codes,
        reason_messages=messages,
        gate=gate,
        signature=best_sig,
        anomaly=anom,
        debug=debug,
        diagnostics=diagnostics,
        best_mode=best_mode,
        mode_scores=mode_scores,
        phase="INSPECTION",
    )
    if mode != "signature":
        _attach_v2_diagnostics(test_bgr, dec, cfg, ok_log_context)

    v2_diag = dec.diagnostics.get("v2_diagnostics") or {}
    _attach_v3_summary(dec, v2_diag, cfg, ok_log_context)
    _attach_v3_trend(dec, ok_log_context)

    # Final Integration: Build Decision (A. ops 덮어쓰기 방지)
    sample_clusters = (v2_diag.get("segmentation", {}) or {}).get("clusters", []) if v2_diag else []
    match_result = v2_diag.get("ink_match") if v2_diag else None

    # A. gate_scores 키 호환성 레이어 (gate_engine 개선판 대응)
    raw_gate = dec.gate.scores if dec.gate else {}
    gate_scores = dict(raw_gate or {})
    if "sharpness_score" not in gate_scores and "sharpness_laplacian_var" in gate_scores:
        gate_scores["sharpness_score"] = gate_scores["sharpness_laplacian_var"]

    thresholds_config = cfg.get("thresholds", {}) or {}
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

    # A. ✅ 절대 덮어쓰지 말고 하위 키로 넣기 (기존 ops_judgment 보존)
    dec.ops = dec.ops or {}
    dec.ops["qc_decision"] = {
        "schema_version": "qc_decision.v1",
        **decision_json,
    }

    # full dump는 debug flag 있을 때만
    if (cfg.get("debug") or {}).get("include_full_qc_decision", False):
        dec.debug["full_qc_decision"] = decision_json

    dec.pattern_color = decision_json.get("pattern_color", {})

    _attach_features(dec, cfg, ok_log_context)
    _append_ok_features(test_bgr, dec, cfg, pattern_baseline, ok_log_context)
    return dec, mode_sigs


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


def evaluate_per_color(
    test_bgr,
    per_color_models: Dict[str, Dict[str, Any]],  # {color_id: {mode: StdModel}}
    color_metadata: Dict[str, Any],  # {color_id: metadata}
    cfg: Dict[str, Any],
    expected_ink_count: int,
    pattern_baseline: Dict[str, Any] | None = None,
    ok_log_context: Dict[str, Any] | None = None,
    mode: str = "all",
) -> Tuple[Decision, Dict[str, Dict[str, SignatureResult]]]:
    """
    Per-color evaluation for multi-ink images.
    """
    # Step 1: Geometry detection
    geom = detect_lens_circle(test_bgr)
    test_bgr, wb_meta = _maybe_apply_white_balance(test_bgr, geom, cfg)

    # Step 2: Gate check (shared across all colors)
    gate = run_gate(
        geom,
        test_bgr,
        center_off_max=cfg["gate"]["center_off_max"],
        blur_min=cfg["gate"]["blur_min"],
        illum_max=cfg["gate"]["illum_max"],
    )
    diag_on_fail = bool(cfg.get("gate", {}).get("diagnostic_on_fail", False))

    if mode == "gate":
        codes, messages = _reason_meta(gate.reasons)
        dec = Decision(
            label="OK" if gate.passed else "RETAKE",
            reasons=gate.reasons,
            reason_codes=codes,
            reason_messages=messages,
            gate=gate,
            signature=None,
            anomaly=None,
            debug={"test_geom": asdict(geom), "inference_valid": gate.passed},
            diagnostics={"gate": asdict(gate), "color_mode": "per_color"},
            best_mode="",
            mode_scores={},
            phase="INSPECTION",
        )
        return dec, {}

    if not gate.passed and not diag_on_fail:
        codes, messages = _reason_meta(gate.reasons)
        dec = Decision(
            label="RETAKE",
            reasons=gate.reasons,
            reason_codes=codes,
            reason_messages=messages,
            gate=gate,
            signature=None,
            anomaly=None,
            debug={"inference_valid": False},
            diagnostics={"color_mode": "per_color"},
            best_mode="",
            mode_scores={},
            phase="INSPECTION",
        )
        return dec, {}

    # Step 3: Generate color masks with 2-pass retry logic
    try:
        color_masks, mask_metadata = build_color_masks_with_retry(
            test_bgr, cfg, expected_k=expected_ink_count, geom=geom, confidence_threshold=0.7, enable_retry=True
        )
    except Exception as e:
        codes, messages = _reason_meta(["COLOR_SEGMENTATION_FAILED"])
        dec = Decision(
            label="RETAKE",
            reasons=["COLOR_SEGMENTATION_FAILED"],
            reason_codes=codes,
            reason_messages=messages + [f"Error: {str(e)}"],
            gate=gate,
            signature=None,
            anomaly=None,
            debug={"inference_valid": False, "segmentation_error": str(e)},
            diagnostics={"color_mode": "per_color"},
            best_mode="",
            mode_scores={},
            phase="INSPECTION",
        )
        return dec, {}

    # Extract segmentation metadata for debugging/warnings
    segmentation_info = {
        "expected_ink_count": mask_metadata.get("expected_ink_count"),
        "segmentation_k": mask_metadata.get("segmentation_k"),
        "detected_ink_like_count": mask_metadata.get("detected_ink_like_count"),
        "segmentation_confidence": mask_metadata.get("segmentation_confidence"),
        "segmentation_pass": mask_metadata.get("segmentation_pass"),
        "retry_reason": mask_metadata.get("retry_reason"),
    }

    # Add warnings if ink count mismatch or retry occurred
    segmentation_warnings = []
    detected_inks = mask_metadata.get("detected_ink_like_count", 0)

    if detected_inks != expected_ink_count:
        segmentation_warnings.append(
            f"EXPECTED_INK_COUNT_MISMATCH (expected={expected_ink_count}, detected={detected_inks})"
        )

    if mask_metadata.get("segmentation_pass") == "pass2_retry":
        retry_reasons = mask_metadata.get("retry_reason", [])
        segmentation_warnings.append(f"INK_SEGMENTATION_RETRIED_K{expected_ink_count + 1}")
        if retry_reasons:
            segmentation_warnings.extend(retry_reasons)

    # Verify color count matches expected (fatal error - cannot proceed)
    detected_colors = len(mask_metadata.get("colors", []))
    if detected_colors == 0 or (
        detected_colors < expected_ink_count and mask_metadata.get("k_used", 0) == expected_ink_count
    ):
        # Hard failure: no colors detected or severe segmentation failure
        codes, messages = _reason_meta(["COLOR_SEGMENTATION_FAILED"])
        dec = Decision(
            label="RETAKE",
            reasons=["COLOR_SEGMENTATION_FAILED"] + segmentation_warnings,
            reason_codes=codes,
            reason_messages=messages
            + [f"Detected {detected_colors} colors, expected {expected_ink_count}"]
            + segmentation_warnings,
            gate=gate,
            signature=None,
            anomaly=None,
            debug={
                "inference_valid": False,
                "detected_colors": detected_colors,
                "expected_colors": expected_ink_count,
                **segmentation_info,
            },
            diagnostics={"color_mode": "per_color", "mask_metadata": mask_metadata},
            best_mode="",
            mode_scores={},
            phase="INSPECTION",
        )
        return dec, {}

    # Step 4: Evaluate each color
    any_model = next(iter(next(iter(per_color_models.values())).values()))
    polar = to_polar(test_bgr, geom, R=any_model.meta["R"], T=any_model.meta["T"])

    per_color_signatures = {}  # {color_id: {mode: SignatureResult}}
    per_color_best_modes = {}  # {color_id: best_mode}
    per_color_reasons = {}  # {color_id: [reasons]}

    if mode != "ink":
        for color_id, mode_models in per_color_models.items():
            if color_id not in color_masks:
                per_color_reasons[color_id] = ["COLOR_MASK_MISSING"]
                continue

            mask = color_masks[color_id]
            color_mode_sigs = {}

            for m_mode, m_model in mode_models.items():
                try:
                    test_mean, _, _ = build_radial_signature_masked(
                        polar, mask, r_start=cfg["signature"]["r_start"], r_end=cfg["signature"]["r_end"]
                    )
                    color_mode_sigs[m_mode] = _evaluate_signature(test_mean, m_model, cfg)
                except Exception as e:
                    per_color_reasons[color_id] = [f"SIGNATURE_EVAL_FAILED:{m_mode}:{str(e)}"]
                    break

            if color_mode_sigs:
                per_color_signatures[color_id] = color_mode_sigs
                best_mode = _pick_best_mode(color_mode_sigs)
                per_color_best_modes[color_id] = best_mode
                best_sig = color_mode_sigs[best_mode]
                if best_sig.reasons:
                    per_color_reasons[color_id] = [f"{reason}:{color_id}" for reason in best_sig.reasons]

    # Step 5: Aggregate per-color results
    all_reasons = []
    failed_colors = []

    for color_id in per_color_models.keys():
        if color_id in per_color_reasons and per_color_reasons[color_id]:
            color_meta = color_metadata.get(color_id, {})
            if color_meta.get("role") == "ink":
                all_reasons.extend(per_color_reasons[color_id])
                failed_colors.append(color_id)

    if failed_colors:
        label = "NG_COLOR"
    else:
        label = "OK"

    debug = {
        "test_geom": asdict(geom),
        "inference_valid": not failed_colors,
        "per_color_best_modes": per_color_best_modes,
        "failed_colors": failed_colors,
        "detected_colors": detected_colors,
    }

    diagnostics = {
        "gate": asdict(gate),
        "color_mode": "per_color",
        "mask_metadata": mask_metadata,
        "per_color_signatures": {
            color_id: {mode: asdict(sig) for mode, sig in mode_sigs.items()}
            for color_id, mode_sigs in per_color_signatures.items()
        },
        "per_color_best_modes": per_color_best_modes,
        "color_metadata": color_metadata,
    }

    if per_color_best_modes:
        mode_counts = {}
        for mode_name in per_color_best_modes.values():
            mode_counts[mode_name] = mode_counts.get(mode_name, 0) + 1
        overall_best_mode = max(mode_counts, key=mode_counts.get)
    else:
        overall_best_mode = ""

    # Build Decision with V2 Schema
    aggregated_clusters = []
    for color_id, sigs in per_color_signatures.items():
        best_m = per_color_best_modes.get(color_id, "MID")
        best_s = sigs.get(best_m)
        if best_s:
            aggregated_clusters.append(
                {
                    "cluster_id": color_id,
                    "inkness_score": 1.0,
                    "score_corr": float(best_s.score_corr),
                    "delta_e_mean": float(best_s.delta_e_mean),
                }
            )

    # A. gate_scores 키 호환성 레이어 (gate_engine 개선판 대응)
    raw_gate = gate.scores if gate else {}
    gate_scores = dict(raw_gate or {})
    if "sharpness_score" not in gate_scores and "sharpness_laplacian_var" in gate_scores:
        gate_scores["sharpness_score"] = gate_scores["sharpness_laplacian_var"]

    decision_json = build_decision(
        run_id=(ok_log_context or {}).get("run_id") or "",
        phase="INSPECTION",
        cfg=cfg,
        gate_scores=gate_scores,
        expected_inks=expected_ink_count,
        sample_clusters=aggregated_clusters,
        match_result=None,
        deltae_summary_method="max",
        inkness_summary_method="min",
    )

    dec = Decision(
        label=decision_json["decision"]["label"],
        reasons=all_reasons,
        reason_codes=decision_json["decision"]["reason_codes"],
        reason_messages=[],
        gate=gate,
        signature=None,
        anomaly=None,
        debug=debug,
        diagnostics=diagnostics,
        best_mode=overall_best_mode,
        mode_scores={},
        phase="INSPECTION",
    )

    # A. ✅ 절대 덮어쓰지 말고 하위 키로 넣기 (기존 ops_judgment 보존)
    dec.ops = dec.ops or {}
    dec.ops["qc_decision"] = {
        "schema_version": "qc_decision.v1",
        **decision_json,
    }

    # full dump는 debug flag 있을 때만
    if (cfg.get("debug") or {}).get("include_full_qc_decision", False):
        dec.debug["full_qc_decision"] = decision_json

    dec.pattern_color = decision_json.get("pattern_color", {})

    _attach_v3_trend(dec, ok_log_context)

    return dec, per_color_signatures
