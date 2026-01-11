from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..measure import threshold_policy as tp

SCHEMA_VERSION = "v3_summary.v1"
GENERATOR = "v3_summary_engine@v1.0.0"
MAX_SUMMARY_LEN = 120
MAX_SIGNAL_LEN = 80


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _fmt1(val: Optional[float]) -> Optional[str]:
    if val is None:
        return None
    return f"{val:.1f}"


def _fmt2(val: Optional[float]) -> Optional[str]:
    if val is None:
        return None
    return f"{val:.2f}"


def _descriptor_delta(value: Optional[float], pos: str, neg: str) -> str:
    if value is None:
        return "-"
    return pos if value > 0 else neg if value < 0 else "중립"


def _summary_severity(
    uncertain: bool,
    max_deltae: Optional[float],
    auto_k_mismatch: bool,
    auto_k_conf: Optional[float],
    min_deltae: Optional[float],
    has_pattern_warning: bool,
    deltae_gates: Optional[Dict[str, float]] = None,
) -> str:
    """
    Calculate severity using centralized deltae_gates (B improvement).

    Uses deltae_gates["review_max"] and ["pass_max"] instead of hardcoded 5/8.
    Falls back to legacy thresholds only if deltae_gates not provided.
    """
    if uncertain:
        return "WARN"

    # B. Use deltae_gates instead of hardcoded 5/8
    if deltae_gates:
        review_max = deltae_gates.get("review_max", 8.0)
        pass_max = deltae_gates.get("pass_max", 5.0)
    else:
        # Legacy fallback
        review_max = 8.0
        pass_max = 5.0

    if max_deltae is not None and max_deltae >= review_max:
        return "WARN"
    # Keep threshold consistent across v3 modules / ops judgment
    if auto_k_mismatch and auto_k_conf is not None and auto_k_conf >= 0.7:
        return "WARN"
    if max_deltae is not None and max_deltae >= pass_max:
        return "INFO"
    if min_deltae is not None and min_deltae < 3.0:
        return "INFO"
    if has_pattern_warning:
        return "INFO"
    return "OK"


def _metrics_basis(ok_log_context: Optional[Dict[str, Any]]) -> List[str]:
    basis = ["v2_diagnostics"]
    if not ok_log_context:
        return basis
    if ok_log_context.get("active_versions") or ok_log_context.get("expected_ink_count_registry") is not None:
        basis.append("active_index")
    return basis


def build_v3_summary(
    v2_diag: Dict[str, Any],
    decision: Any,
    cfg: Dict[str, Any],
    ok_log_context: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not v2_diag:
        return None

    # 1. Extraction
    warnings = v2_diag.get("warnings") or []
    # A. auto_estimation key fallback (multiple schemas support)
    auto = v2_diag.get("auto_estimation") or v2_diag.get("auto_k") or v2_diag.get("auto") or {}
    seg = v2_diag.get("segmentation") or {}
    quality = seg.get("quality") or {}
    ink_match = (v2_diag or {}).get("ink_match") or {}
    palette = v2_diag.get("palette") or {}
    input_quality = (v2_diag or {}).get("input_quality") or {}

    expected_k = v2_diag.get("expected_ink_count")
    suggested_k = auto.get("suggested_k")
    confidence = auto.get("confidence")
    min_deltae = palette.get("min_deltaE_between_clusters", quality.get("min_deltaE_between_clusters"))

    # 2. Match & Gate Logic (Threshold Policy)
    comparable = bool(ink_match.get("matched") is True and (ink_match.get("deltas") or ink_match.get("cluster_deltas")))

    top_item = None
    max_deltae = None
    if ink_match.get("matched"):
        for item in ink_match.get("deltas") or []:
            delta_e = item.get("deltaE")
            if delta_e is None:
                continue
            if max_deltae is None or delta_e > max_deltae:
                max_deltae = delta_e
                top_item = item

    rep_deltae = None
    if comparable and top_item and top_item.get("deltaE") is not None:
        rep_deltae = float(top_item.get("deltaE", 0.0))
    else:
        traj = ink_match.get("trajectory_summary") or {}
        if traj.get("max_off_track") is not None:
            rep_deltae = float(traj["max_off_track"])

    quality_level = str(input_quality.get("quality_level") or "unknown")
    cfg = cfg or {}
    deltae_method = str((cfg.get("v2_ink") or {}).get("deltaE_method", ink_match.get("deltaE_method", "76")))
    deltae_base_gates = cfg.get("deltaE_gates") or None

    deltae_gates = tp.get_deltae_gates(
        deltae_method=deltae_method,
        quality_level=quality_level,
        base_gates=deltae_base_gates,
    )

    match = {
        "label": "NA",
        "reason": "NO_REFERENCE_OR_MATCH_FAILED",
        "deltaE": None,
        "method": deltae_gates.get("deltaE_method"),
    }
    if comparable and rep_deltae is not None:
        cls = tp.classify_deltae(rep_deltae, deltae_gates)
        match["label"] = "PASS" if cls == "pass" else ("REVIEW" if cls == "review" else "FAIL")
        match["reason"] = "DELTAE_GATE"
        match["deltaE"] = float(rep_deltae)

    # 3. Summary Lines
    summary: List[str] = []

    # Line 1
    if not comparable:
        summary.append("기준 대비 변화: 비교 불가(참조 없음/매칭 실패).")
    elif top_item and max_deltae is not None:
        # Build direction text carefully
        directions = []
        delta_l = top_item.get("delta_L")
        delta_b = top_item.get("delta_b")
        delta_a = top_item.get("delta_a")
        if delta_l is not None and abs(delta_l) >= 1.0:
            directions.append("어두움" if delta_l < 0 else "밝아짐")
        if delta_b is not None and abs(delta_b) >= 1.0:
            directions.append("노랑 증가" if delta_b > 0 else "파랑 증가")
        if delta_a is not None and abs(delta_a) >= 1.0:
            directions.append("빨강 증가" if delta_a > 0 else "초록 증가")
        dir_text = ", ".join(directions) if directions else "shift observed"

        # Build values text
        dir_vals = []
        if delta_l is not None and abs(delta_l) >= 1.0:
            dir_vals.append(f"ΔL {_fmt1(delta_l)}")
        if delta_b is not None and abs(delta_b) >= 1.0:
            dir_vals.append(f"Δb {_fmt1(delta_b)}")
        dir_val_text = ", ".join(dir_vals) if dir_vals else f"ΔE {_fmt1(max_deltae)}"

        # Safe f-string construction
        idx_str = f"Ink{top_item.get('index', 0) + 1}"
        summary.append(f"기준 대비 변화: {idx_str} {dir_text} (ACTIVE STD, {dir_val_text}).")
    else:
        # D. Handle edge case: comparable but no deltas (schema issue)
        if comparable:
            summary.append("ACTIVE STD 매칭되었으나 deltas 비어있음(스키마 확인 필요).")
        else:
            summary.append("ACTIVE STD 대비 핵심 변화 신호 없음.")

    # Line 2
    line2_parts = []
    if suggested_k is not None and expected_k is not None:
        conf_val = confidence if confidence is not None else 0.0
        conf_txt = "low" if conf_val < 0.5 else ("med" if conf_val < 0.8 else "high")
        line2_parts.append(f"auto-k {expected_k}→{suggested_k} (conf={conf_txt}:{conf_val:.2f})")
    if min_deltae is not None:
        line2_parts.append(f"minΔE {_fmt1(min_deltae)}")

    uncertain = ink_match.get("warning") == "INK_CLUSTER_MATCH_UNCERTAIN" or "INK_CLUSTER_MATCH_UNCERTAIN" in warnings
    if ink_match.get("matched") and not uncertain:
        line2_parts.append("매칭 정상")
    if uncertain:
        line2_parts.append("참고용(indicative only)")

    if line2_parts:
        summary.append("; ".join(line2_parts) + ".")
    else:
        summary.append("보조 진단 정보가 없습니다.")

    # 4. Signals
    signals: List[Dict[str, Any]] = []

    if uncertain:
        signals.append({"priority": 0, "text": "매칭 불확실: 변화 신호는 참고용입니다."})

    # C. FAIL/REVIEW signals with actual values and thresholds
    if match["label"] == "FAIL" and rep_deltae is not None:
        review_max = deltae_gates.get("review_max", 8.0)
        signals.append({"priority": 0, "text": f"ΔE FAIL {rep_deltae:.2f} > {review_max:.2f} ({match['method']})"})
    elif match["label"] == "REVIEW" and rep_deltae is not None:
        pass_max = deltae_gates.get("pass_max", 5.0)
        signals.append({"priority": 1, "text": f"ΔE REVIEW {rep_deltae:.2f} > {pass_max:.2f} ({match['method']})"})

    # D. Comparable but no deltas signal
    if comparable and not top_item and not max_deltae:
        signals.append({"priority": 2, "text": "매칭은 되었지만 deltas가 비어있음(스키마 확인 필요)"})

    # Extra detail signal if comparable
    if comparable and top_item and max_deltae is not None:
        idx = top_item.get("index", 0)
        delta_l = top_item.get("delta_L")
        delta_b = top_item.get("delta_b")
        delta_a = top_item.get("delta_a")

        dir_parts = []
        if delta_l is not None and abs(delta_l) >= 1.0:
            dir_parts.append(f"ΔL {_fmt1(delta_l)} {_descriptor_delta(delta_l, '밝아짐', '어두움')}")
        if delta_b is not None and abs(delta_b) >= 1.0:
            dir_parts.append(f"Δb {_fmt1(delta_b)} {_descriptor_delta(delta_b, '노랑 증가', '파랑 증가')}")
        if delta_a is not None and abs(delta_a) >= 1.0:
            dir_parts.append(f"Δa {_fmt1(delta_a)} {_descriptor_delta(delta_a, '빨강 증가', '초록 증가')}")

        dir_text2 = ", ".join(dir_parts) if dir_parts else "shift observed"
        signals.append(
            {
                "priority": 2,
                "text": f"Ink{idx + 1} 최대 ΔE {_fmt1(max_deltae)} ({dir_text2})",
            }
        )

    # Auto-K
    auto_k_mismatch = (
        suggested_k is not None
        and expected_k is not None
        and suggested_k != expected_k
        and confidence is not None
        and confidence >= 0.7
    )
    if auto_k_mismatch:
        conf_val = confidence if confidence is not None else 0.0
        conf_txt = _fmt2(conf_val)
        signals.append(
            {
                "priority": 2,
                "text": f"auto-k 불일치: 기대 {expected_k}, 제안 {suggested_k} ({conf_txt})",
            }
        )
    elif min_deltae is not None and min_deltae < 3.0:
        signals.append(
            {
                "priority": 2,
                "text": f"분리 약함: minΔE {_fmt1(min_deltae)}",
            }
        )

    # Pattern
    has_pattern_warning = bool((decision.diagnostics or {}).get("pattern", {}).get("warnings"))
    if not has_pattern_warning and getattr(decision, "reason_codes", None):
        has_pattern_warning = "NG_PATTERN" in decision.reason_codes
    if has_pattern_warning and len(signals) < 3:
        signals.append({"priority": 3, "text": "패턴 경향 변화 감지"})

    signals_sorted = sorted(signals, key=lambda s: s["priority"])
    key_signals = [_truncate(s["text"], MAX_SIGNAL_LEN) for s in signals_sorted[:3]]
    summary_final = [_truncate(line, MAX_SUMMARY_LEN) for line in summary[:2]]

    # 5. Metrics
    supporting_metrics: Dict[str, Any] = {
        "max_deltaE": None,
        "mean_deltaE": None,
        "suggested_k": suggested_k,
        "expected_k": expected_k,
        "confidence": confidence,
    }
    if ink_match.get("matched"):
        deltas = ink_match.get("deltas") or []
        delta_vals = [d.get("deltaE") for d in deltas if d.get("deltaE") is not None]
        if delta_vals:
            supporting_metrics["max_deltaE"] = max(delta_vals)
            supporting_metrics["mean_deltaE"] = sum(delta_vals) / float(len(delta_vals))
    elif quality.get("mean_deltaE_between_clusters") is not None:
        supporting_metrics["mean_deltaE"] = quality.get("mean_deltaE_between_clusters")

    # 6. Severity & Meta
    confidence_source = "auto_k_silhouette" if auto else ""
    severity = _summary_severity(
        uncertain=uncertain,
        max_deltae=supporting_metrics.get("max_deltaE"),
        auto_k_mismatch=auto_k_mismatch,
        auto_k_conf=confidence,
        min_deltae=min_deltae,
        has_pattern_warning=has_pattern_warning,
        deltae_gates=deltae_gates,  # B. Pass deltae_gates for policy-based severity
    )
    if match.get("label") == "FAIL":
        severity = "WARN"
    elif match.get("label") == "REVIEW" and severity != "WARN":
        severity = "INFO"

    meta: Dict[str, Any] = {
        "window_requested": 1,
        "window_effective": 1,
        "data_sparsity": False,
        "confidence_source": confidence_source,
        "metrics_basis": _metrics_basis(ok_log_context),
        "source": "inspection",
        "generated_at": datetime.now().isoformat(),
        "generator": GENERATOR,
        "severity": severity,
    }
    if confidence is not None:
        meta["confidence_components"] = {"auto_k": confidence}

    return {
        "schema_version": SCHEMA_VERSION,
        "summary": summary_final,
        "key_signals": key_signals,
        "supporting_metrics": supporting_metrics,
        "severity": severity,
        "match": match,
        "deltaE_gates": deltae_gates,
        "meta": meta,
    }
