"""
Decision/InspectionResult conversion helpers.

Keeps v7 Decision -> src InspectionResult mapping in one place.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.schemas.inspection import InspectionResult


def map_v7_label(label: str, warnings: List[str]) -> str:
    if label == "OK":
        return "OK_WITH_WARNING" if warnings else "OK"
    if label.startswith("OK"):
        return "OK"
    if label in {"RETAKE", "STD_RETAKE"}:
        return "RETAKE"
    return "NG"


def build_v7_analysis_summary(decision: Any) -> Optional[Dict[str, Any]]:
    if decision is None:
        return None
    diag = getattr(decision, "diagnostics", {}) or {}
    radial = (diag.get("radial") or {}).get("summary") or {}
    v2_diag = diag.get("v2_diagnostics") if isinstance(diag, dict) else None
    seg_quality = None
    gate_diag = diag.get("gate") if isinstance(diag, dict) else None
    if isinstance(v2_diag, dict):
        seg_quality = (v2_diag.get("segmentation") or {}).get("quality")
    references = None
    white_balance = None
    anomaly = None
    pattern_color = None
    if isinstance(diag, dict):
        references = diag.get("references")
        white_balance = diag.get("white_balance")
        anomaly = diag.get("anomaly")
        pattern_color = getattr(decision, "pattern_color", None)
    diagnostics_summary = None
    if isinstance(diag, dict):
        diagnostics_summary = {
            "v2_warnings": (diag.get("v2_diagnostics") or {}).get("warnings") if diag.get("v2_diagnostics") else None,
            "references": references,
            "white_balance": white_balance,
        }
    return {
        "radial": radial,
        "gate": getattr(getattr(decision, "gate", None), "scores", None),
        "signature": getattr(getattr(decision, "signature", None), "score_corr", None),
        "segmentation_quality": seg_quality,
        "gate_diagnostics": gate_diag,
        "references": references,
        "white_balance": white_balance,
        "anomaly": anomaly,
        "pattern_color": pattern_color,
        "mode_scores": getattr(decision, "mode_scores", None),
        "mode_shift": getattr(decision, "mode_shift", None),
        "diagnostics_summary": diagnostics_summary,
    }


def build_v7_confidence_breakdown(decision: Any) -> Optional[Dict[str, Any]]:
    if decision is None:
        return None
    gate = getattr(decision, "gate", None)
    sig = getattr(decision, "signature", None)
    v2_diag = getattr(decision, "diagnostics", {}).get("v2_diagnostics") if decision else None
    seg_quality = None
    if isinstance(v2_diag, dict):
        seg_quality = (v2_diag.get("segmentation") or {}).get("quality")
        sampling_meta = v2_diag.get("sampling") or {}
    else:
        sampling_meta = None
    return {
        "gate_passed": getattr(gate, "passed", None),
        "signature_score": getattr(sig, "score_corr", None),
        "label": getattr(decision, "label", None),
        "segmentation_quality": seg_quality,
        "sampling": sampling_meta,
    }


def build_inspection_result_from_v7(
    decision: Any,
    sku: str,
    warnings: List[str],
) -> InspectionResult:
    sig = getattr(decision, "signature", None)
    overall_delta_e = 0.0
    if sig is not None:
        overall_delta_e = float(getattr(sig, "delta_e_p95", None) or getattr(sig, "delta_e_mean", 0.0) or 0.0)
    if overall_delta_e == 0.0:
        diag = getattr(decision, "diagnostics", {}) or {}
        radial = (diag.get("radial") or {}).get("summary") or {}
        overall_delta_e = float(radial.get("delta_e_p95") or radial.get("delta_e_mean") or 0.0)

    judgment = map_v7_label(decision.label, warnings)
    reasons = list(getattr(decision, "reason_codes", None) or getattr(decision, "reasons", None) or [])
    reason_messages = list(getattr(decision, "reason_messages", None) or getattr(decision, "reasons", None) or [])
    ng_reasons = reasons if judgment in {"NG", "RETAKE"} else []

    risk_factors = None
    if warnings:
        risk_factors = [{"category": "v7_warning", "severity": "warn", "message": w} for w in warnings]
    anom = getattr(decision, "anomaly", None)
    if anom is not None:
        risk_factors = risk_factors or []
        risk_factors.append(
            {
                "category": "v7_anomaly",
                "severity": "high" if not getattr(anom, "passed", True) else "info",
                "message": getattr(anom, "type", "") or "anomaly_detected",
            }
        )

    retake_reasons = None
    if judgment == "RETAKE" and reasons:
        retake_reasons = [
            {"code": r, "reason": r, "actions": ["Review v7 diagnostics."], "lever": "v7_reason"} for r in reasons
        ]

    debug = getattr(decision, "debug", {}) or {}
    debug_meta = {}
    if isinstance(debug, dict):
        for key in ("test_geom", "baseline_reasons", "mode_scores"):
            if key in debug:
                debug_meta[key] = debug[key]
    debug_meta["phase"] = getattr(decision, "phase", None)
    debug_meta["best_mode"] = getattr(decision, "best_mode", None)

    result = InspectionResult(
        sku=sku,
        timestamp=datetime.now(),
        judgment=judgment,
        overall_delta_e=float(overall_delta_e),
        ng_reasons=ng_reasons,
        confidence=1.0 if judgment in {"OK", "OK_WITH_WARNING"} else 0.5,
        decision_trace={"v7_label": decision.label, "reasons": reasons, "debug": debug_meta},
        analysis_summary=build_v7_analysis_summary(decision),
        confidence_breakdown=build_v7_confidence_breakdown(decision),
        diagnostics=reasons or None,
        warnings=warnings or None,
        suggestions=reason_messages or None,
        risk_factors=risk_factors,
        retake_reasons=retake_reasons,
    )

    v2_diag = getattr(decision, "diagnostics", {}).get("v2_diagnostics") if decision else None
    if isinstance(v2_diag, dict):
        result.ink_analysis = {
            "schema_version": "ink_analysis.v7",
            "engine": "v7",
            "v2_diagnostics": v2_diag,
        }

    return result
