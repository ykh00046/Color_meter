from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..measure.metrics import threshold_policy as tp

SCHEMA_VERSION = "v3_trend.v1"
GENERATOR = "v3_trend_engine@v1.0.0"
WINDOW_DEFAULT = 20


def _get_v2(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    A. Extract v2_diagnostics from various decision schemas.

    Supports:
    - {"diagnostics": {"v2_diagnostics": ...}}  (decision.to_dict)
    - {"v2_diagnostics": ...}  (top-level)
    - {} (empty fallback)

    This makes trend less sensitive to data format variations.
    """
    if d.get("diagnostics") and (d["diagnostics"] or {}).get("v2_diagnostics"):
        return (d["diagnostics"] or {}).get("v2_diagnostics") or {}
    if d.get("v2_diagnostics"):
        return d.get("v2_diagnostics") or {}
    return {}


def _trend_direction(values: List[float]) -> str:
    if len(values) < 2:
        return "flat"
    delta = values[-1] - values[0]
    if abs(delta) < 0.2:
        return "flat"
    return "upward" if delta > 0 else "downward"


def _confidence(window_effective: int) -> str:
    if window_effective >= 16:
        return "high"
    if window_effective >= 12:
        return "medium"
    return "low"


def _collect_ink_deltas(
    decisions: List[Dict[str, Any]]
) -> Tuple[Dict[int, List[float]], Dict[int, int], List[Optional[int]]]:
    by_idx: Dict[int, List[float]] = {}
    top_counts: Dict[int, int] = {}
    top_sequence: List[Optional[int]] = []
    for dec in decisions:
        v2 = _get_v2(dec)  # A. Use unified v2 extractor
        ink_match = v2.get("ink_match") or {}
        if not ink_match.get("matched"):
            top_sequence.append(None)
            continue
        max_delta = None
        top_idx = None
        for item in ink_match.get("deltas") or []:
            idx = item.get("index")
            delta_e = item.get("deltaE")
            if idx is None or delta_e is None:
                continue
            by_idx.setdefault(int(idx), []).append(float(delta_e))
            if max_delta is None or delta_e > max_delta:
                max_delta = delta_e
                top_idx = int(idx)
        if top_idx is not None:
            top_counts[top_idx] = top_counts.get(top_idx, 0) + 1
        top_sequence.append(top_idx)
    return by_idx, top_counts, top_sequence


def build_v3_trend(decisions: List[Dict[str, Any]], window_requested: int = WINDOW_DEFAULT) -> Optional[Dict[str, Any]]:
    if not decisions:
        return {
            "meta": {"window_requested": window_requested, "window_effective": 0, "data_sparsity": True},
            "metrics": {},
            "signals": [],
        }

    # A. Use unified v2 extractor for filtering
    decisions = [d for d in decisions if _get_v2(d)]
    if not decisions:
        return {
            "meta": {"window_requested": window_requested, "window_effective": 0, "data_sparsity": True},
            "metrics": {},
            "signals": [],
        }

    # 1) sort decisions by timestamp if available (oldest -> newest)
    def _get_ts(d: dict):
        for k in ("ts", "timestamp", "created_at"):
            if k in d:
                return d.get(k)
        # also allow nested "decision.ts"
        if isinstance(d.get("decision"), dict) and "ts" in d["decision"]:
            return d["decision"].get("ts")
        return None

    # only sort if at least some ts exist and are comparable
    if any(_get_ts(d) is not None for d in decisions):
        try:
            decisions = sorted(decisions, key=lambda x: (_get_ts(x) is None, _get_ts(x)))
        except Exception:
            pass

    # 2) apply window (use most recent N)
    if window_requested and window_requested > 0 and len(decisions) > window_requested:
        decisions = decisions[-window_requested:]

    window_effective = len(decisions)

    # cfg-like globals (optional)
    base_gates = None
    for d in decisions:
        snap = d.get("cfg_snapshot") or {}
        if isinstance(snap, dict) and snap.get("deltaE_gates"):
            base_gates = snap.get("deltaE_gates")
            break

    mismatch_count = 0
    low_sep_count = 0
    uncertain_count = 0

    for dec in decisions:
        v2 = _get_v2(dec)  # A. Use unified v2 extractor
        warnings = v2.get("warnings") or []
        auto = v2.get("auto_estimation") or v2.get("auto_k") or v2.get("auto") or {}  # A. Fallback support
        expected_k = v2.get("expected_ink_count")
        suggested_k = auto.get("suggested_k")
        conf = auto.get("confidence")
        if (
            expected_k is not None
            and suggested_k is not None
            and expected_k != suggested_k
            and conf is not None
            and conf >= 0.7
        ):
            mismatch_count += 1

        palette = v2.get("palette") or {}
        seg = v2.get("segmentation") or {}
        quality = seg.get("quality") or {}
        min_deltae = palette.get("min_deltaE_between_clusters", quality.get("min_deltaE_between_clusters"))
        if min_deltae is not None and float(min_deltae) < 3.0:
            low_sep_count += 1

        ink_match = v2.get("ink_match") or {}
        uncertain = (
            ink_match.get("warning") == "INK_CLUSTER_MATCH_UNCERTAIN" or "INK_CLUSTER_MATCH_UNCERTAIN" in warnings
        )
        if uncertain:
            uncertain_count += 1

    rates = {
        "auto_k_mismatch_rate": mismatch_count / float(window_effective),
        "low_separation_rate": low_sep_count / float(window_effective),
        "uncertain_rate": uncertain_count / float(window_effective),
    }

    ink_deltas, top_counts, top_sequence = _collect_ink_deltas(decisions)

    # choose top ink by maximum observed deltaE (ties by top_count)
    top_ink_id = None
    top_ink_max = -1.0
    for idx, values in ink_deltas.items():
        vmax = max(values) if values else 0.0
        if vmax > top_ink_max or (vmax == top_ink_max and top_counts.get(idx, 0) > top_counts.get(top_ink_id, 0)):
            top_ink_max = vmax
            top_ink_id = idx

    top_ink = (top_ink_id, ink_deltas.get(top_ink_id, [])) if top_ink_id is not None else None

    signals: List[str] = []
    metrics: Dict[str, Any] = {
        "auto_k_mismatch_rate": rates["auto_k_mismatch_rate"],
        "low_separation_rate": rates["low_separation_rate"],
        "uncertain_rate": rates["uncertain_rate"],
    }

    # streak: count consecutive same top ink from MOST RECENT backwards
    streak = 0
    if top_sequence:
        current = top_sequence[-1]
        for idx in reversed(top_sequence):
            if idx == current:
                streak += 1
            else:
                break

    streak_text = ""
    if window_effective >= 10 and streak >= 3 and top_sequence and top_sequence[-1] is not None:
        streak_text = f"; 연속: Ink{top_sequence[-1] + 1} x{streak}"

    # ---- ΔE gate trend signals (threshold_policy based) ----
    fail_n = 0
    review_n = 0
    ok_n = 0
    used_method = None

    for d in decisions:
        de = None
        method = "76"
        ms = d.get("metrics_summary") or {}
        de_dict = ms.get("deltaE_value")
        if isinstance(de_dict, dict) and de_dict.get("value") is not None:
            de = float(de_dict["value"])
            method = str(de_dict.get("method", "76"))
        else:
            v2_alt = _get_v2(d)  # A. Use unified v2 extractor
            ink = v2_alt.get("ink_match") or {}
            method = str(ink.get("deltaE_method", "76"))
            traj = ink.get("trajectory_summary") or {}
            if traj.get("max_off_track") is not None:
                de = float(traj["max_off_track"])

        if de is None:
            continue

        # B. input_quality from both top-level and v2_diagnostics
        v2 = _get_v2(d)  # A. Use unified v2 extractor
        iq = d.get("input_quality") or v2.get("input_quality") or {}
        qlvl = str(iq.get("quality_level") or "unknown")
        gates = tp.get_deltae_gates(deltae_method=method, quality_level=qlvl, base_gates=base_gates)
        used_method = gates.get("deltaE_method")
        cls = tp.classify_deltae(float(de), gates)
        if cls == "fail":
            fail_n += 1
        elif cls == "review":
            review_n += 1
        else:
            ok_n += 1

    denom = max(1, (fail_n + review_n + ok_n))
    fail_rate = fail_n / denom
    review_rate = review_n / denom

    if fail_rate >= 0.20:
        signals.append(f"DELTAE_FAIL_RATE_HIGH ({fail_n}/{denom})")
    elif review_rate >= 0.35:
        signals.append(f"DELTAE_REVIEW_RATE_HIGH ({review_n}/{denom})")

    if top_ink:
        idx, values = top_ink
        direction = _trend_direction(values)
        count = top_counts.get(idx, len(values))
        ink_key = f"ink{idx + 1}_mean_deltaE"
        metrics[ink_key] = sum(values) / float(len(values))
        metrics[f"ink{idx + 1}_deltaE_trend"] = direction

        # New metrics
        metrics["top_ink_index"] = idx
        metrics["top_ink_max_deltaE"] = float(top_ink_max if top_ink_max >= 0 else 0.0)
        metrics["top_ink_trend"] = direction
        metrics["top_ink_n_obs"] = int(len(values))
        metrics["top_ink_top_count"] = int(top_counts.get(idx, 0))
        metrics["top_ink_streak_recent"] = int(streak)

        if direction == "upward":
            signals.append(f"Ink{idx + 1} ΔE 상승 경향 ({count}/{window_effective})")
        elif direction == "downward":
            signals.append(f"Ink{idx + 1} ΔE 하락 경향 ({count}/{window_effective})")
        else:
            signals.append(f"Ink{idx + 1} ΔE 변화 약함 ({count}/{window_effective})")

    metrics["window_effective"] = window_effective
    metrics["auto_k_mismatch_count"] = int(mismatch_count)
    metrics["deltae_fail_n"] = int(fail_n)
    metrics["deltae_review_n"] = int(review_n)
    metrics["deltae_ok_n"] = int(ok_n)
    metrics["deltae_fail_rate"] = float(round(fail_rate, 4))
    metrics["deltae_review_rate"] = float(round(review_rate, 4))

    if mismatch_count:
        signals.append(f"auto-k mismatch 빈도 증가 ({mismatch_count}/{window_effective})")
    if low_sep_count and len(signals) < 3:
        signals.append(f"분리 약함(minΔE<3.0) 반복 ({low_sep_count}/{window_effective})")
    if uncertain_count and len(signals) < 3:
        signals.append(f"매칭 불확실 빈도 증가 ({uncertain_count}/{window_effective})")

    data_sparsity = window_effective < 8
    if data_sparsity:
        trend_line = (
            f"Trend (참고용/희소 {window_effective}/{window_requested}): " f"{'; '.join(signals[:3])}{streak_text}"
        )
    else:
        trend_line = f"Trend: {'; '.join(signals[:3])}{streak_text}"
    meta = {
        "data_sparsity": data_sparsity,
        "confidence": _confidence(window_effective),
        "generated_at": datetime.now().isoformat(),
        "generator": GENERATOR,
        "deltae_method_used": used_method,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "window_requested": window_requested,
        "window_effective": window_effective,
        "signals": signals[:3],
        "metrics": metrics,
        "trend_line": trend_line,
        "meta": meta,
    }
