"""Diagnostics helpers extracted from analyzer.py."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..anomaly.pattern_baseline import extract_pattern_features
from ..insight.summary import build_v3_summary
from ..insight.trend import build_v3_trend
from ..measure.diagnostics.v2_diagnostics import build_v2_diagnostics
from ..measure.matching.ink_match import compute_cluster_deltas
from ..model_registry import load_ink_baseline
from ..types import Decision

logger = logging.getLogger(__name__)


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
    polar_alpha: np.ndarray | None = None,
    *,
    cached_geom=None,
    cached_masks: tuple | None = None,
    cached_polar: np.ndarray | None = None,
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
    # Extract alpha_cfg from cfg (default enabled)
    alpha_cfg = cfg.get("alpha")
    # Allow polar_alpha from ok_log_context if not passed directly
    if polar_alpha is None and ok_log_context:
        polar_alpha = ok_log_context.get("polar_alpha")
    diagnostics = build_v2_diagnostics(
        test_bgr,
        cfg,
        expected_ink_count=int(expected),
        expected_ink_count_registry=expected_registry,
        expected_ink_count_input=expected_input,
        polar_alpha=polar_alpha,
        alpha_cfg=alpha_cfg,
        precomputed_geom=cached_geom,
        precomputed_masks=cached_masks,
        precomputed_polar=cached_polar,
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
                        except json.JSONDecodeError:
                            # Skip malformed JSON lines in trend log
                            continue
                except (IOError, OSError) as e:
                    logger.debug(f"Failed to read trend log {path}: {e}")

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
