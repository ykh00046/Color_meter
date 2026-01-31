"""
Main inspection analyzer – evaluate() and evaluate_multi() entry points.

Refactored: helpers extracted to _signature, _diagnostics, _common,
registration, and per_color sub-modules.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Tuple

import numpy as np

from ..decision.decision_engine import decide
from ..geometry.lens_geometry import detect_lens_circle
from ..signature.radial_signature import build_radial_signature, to_polar
from ..types import Decision, SignatureResult
from ..utils import to_cie_lab
from ._common import (
    _attach_heatmap_defect,
    _evaluate_anomaly,
    _finalize_decision,
    _maybe_apply_white_balance,
    _reason_meta,
    _run_gate_check,
)
from ._diagnostics import _compute_diagnostics
from ._signature import _evaluate_signature, _pick_best_mode
from .per_color import evaluate_per_color  # noqa: F401

# --- Backward compatibility re-exports ---
from .registration import _registration_summary, evaluate_registration_multi  # noqa: F401


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

    # Gate check with early return
    gate, early = _run_gate_check(geom, test_bgr, cfg, mode, pattern_baseline)
    if early is not None:
        return early

    use_relative = bool(cfg.get("pattern_baseline", {}).get("use_relative", True))
    polar = to_polar(test_bgr, geom, R=std_model.meta["R"], T=std_model.meta["T"])

    # Signature
    sig = None
    if mode != "ink":
        test_mean, _, _ = build_radial_signature(
            polar, r_start=cfg["signature"]["r_start"], r_end=cfg["signature"]["r_end"]
        )
        sig = _evaluate_signature(test_mean, std_model, cfg)

    # Anomaly
    anom = None
    if mode not in ["signature", "ink"]:
        anom = _evaluate_anomaly(polar, test_bgr, geom, cfg, pattern_baseline, use_relative)

    label, reasons = decide(gate, sig, anom)

    # Diagnostics
    diagnostics = {}
    extra_codes = []
    extra_messages = {}

    if mode != "ink":
        test_lab_map = to_cie_lab(polar)
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

    # Heatmap + defect classification
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

    _attach_heatmap_defect(decision, anom, polar, cfg)

    # v2/v3 + qc_decision + ok_features
    _finalize_decision(
        decision,
        test_bgr,
        cfg,
        ok_log_context,
        pattern_baseline,
        mode,
        cached_geom=geom,
        cached_polar=polar,
    )

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

    # Gate check with early return
    gate, early = _run_gate_check(
        geom,
        test_bgr,
        cfg,
        mode,
        pattern_baseline,
        extra_decision_kwargs={"best_mode": "", "mode_scores": {}},
    )
    if early is not None:
        return early, {}

    use_relative = bool(cfg.get("pattern_baseline", {}).get("use_relative", True))
    any_model = next(iter(std_models.values()))
    polar = to_polar(test_bgr, geom, R=any_model.meta["R"], T=any_model.meta["T"])

    # Signature (all modes)
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

    # Anomaly
    anom = None
    if mode not in ["signature", "ink"]:
        anom = _evaluate_anomaly(polar, test_bgr, geom, cfg, pattern_baseline, use_relative)

    label, reasons = decide(gate, best_sig, anom)

    # Diagnostics
    diagnostics = {}
    extra_codes = []
    extra_messages = {}

    if mode != "ink" and best_mode:
        test_lab_map = to_cie_lab(polar)
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

    _attach_heatmap_defect(dec, anom, polar, cfg)

    # v2/v3 + qc_decision + ok_features
    _finalize_decision(
        dec,
        test_bgr,
        cfg,
        ok_log_context,
        pattern_baseline,
        mode,
        cached_geom=geom,
        cached_polar=polar,
    )

    return dec, mode_sigs
