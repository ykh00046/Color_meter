# decision_builder.py (v2)
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

# ----------------------------
# Reason codes (추천 표준)
# ----------------------------
RC_QUALITY_VERY_POOR_RETAKE = "QUALITY_VERY_POOR_RETAKE"
RC_QUALITY_POOR_TIGHTENED_GATES = "QUALITY_POOR_TIGHTENED_GATES"

RC_INK_COUNT_MISMATCH = "INK_COUNT_MISMATCH"

RC_MATCHING_INCOMPLETE_TEST_HAS_UNMATCHED = "MATCHING_INCOMPLETE_TEST_HAS_UNMATCHED"
RC_MATCHING_INCOMPLETE_REF_HAS_UNMATCHED = "MATCHING_INCOMPLETE_REF_HAS_UNMATCHED"

RC_INKNESS_FAIL_BELOW_THRESHOLD = "INKNESS_FAIL_BELOW_THRESHOLD"
RC_INKNESS_REVIEW_WINDOW = "INKNESS_REVIEW_WINDOW"
RC_INKNESS_GAP_LIKELY_NON_INK = "INKNESS_GAP_LIKELY_NON_INK"

RC_DELTAE_FAIL_OVER_REVIEW_MAX = "DELTAE_FAIL_OVER_REVIEW_MAX"
RC_DELTAE_REVIEW_OVER_PASS_MAX = "DELTAE_REVIEW_OVER_PASS_MAX"
RC_DELTAE_PASS = "DELTAE_PASS"

RC_SPATIAL_LOW_ANGULAR_CONTINUITY = "SPATIAL_LOW_ANGULAR_CONTINUITY"


# ----------------------------
# Helpers
# ----------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _percentile(values: List[float], p: float) -> float:
    """Pure python percentile (0~100)."""
    if not values:
        return 0.0
    v = sorted(values)
    if len(v) == 1:
        return float(v[0])
    p = max(0.0, min(100.0, float(p)))
    k = (len(v) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(v[int(k)])
    d0 = v[int(f)] * (c - k)
    d1 = v[int(c)] * (k - f)
    return float(d0 + d1)


def _normalize_deltae_method(method: str) -> str:
    m = str(method or "76").strip().lower()
    if m in ("2000", "de2000", "ciede2000", "cie2000"):
        return "2000"
    return "76"


def _fallback_deltae_gates(deltae_method: str, gate_scores: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    threshold_policy에 ΔE gate가 아직 없을 때를 위한 fallback.
    - 품질이 나쁠수록 PASS는 엄격(pass↓), REVIEW 상한은 관대(review↑) -> review로 보내는 안전장치
    """
    method = _normalize_deltae_method(deltae_method)
    gate_scores = gate_scores or {}
    sharp = _safe_float(gate_scores.get("sharpness_score", 500.0), 500.0)
    illum = _safe_float(gate_scores.get("illumination_asymmetry", 0.03), 0.03)
    off = _safe_float(gate_scores.get("center_offset_mm", 0.5), 0.5)

    # 아주 단순한 severity (0~1)
    sev = 0.0
    if sharp < 200.0:
        sev += 0.4
    elif sharp < 500.0:
        sev += 0.2
    if illum > 0.15:
        sev += 0.3
    elif illum > 0.05:
        sev += 0.15
    if off > 3.0:
        sev += 0.3
    elif off > 1.0:
        sev += 0.15
    sev = max(0.0, min(1.0, sev))

    if method == "2000":
        base_pass, base_review = 3.5, 5.5
        pass_tighten, review_relax = 0.7, 1.0
    else:
        base_pass, base_review = 5.0, 8.0
        pass_tighten, review_relax = 1.0, 1.5

    pass_max = max(0.1, base_pass - sev * pass_tighten)
    review_max = max(pass_max + 0.3, base_review + sev * review_relax)

    # quality_level (간단 버전)
    poor_cnt = 0
    if sharp < 200.0:
        poor_cnt += 1
    if illum > 0.15:
        poor_cnt += 1
    if off > 3.0:
        poor_cnt += 1
    if poor_cnt >= 2:
        qlvl = "very_poor"
    elif poor_cnt == 1:
        qlvl = "poor"
    elif (200.0 <= sharp < 500.0) or (0.05 < illum <= 0.15) or (1.0 < off <= 3.0):
        qlvl = "medium"
    else:
        qlvl = "good"

    return {
        "deltaE_method": method,
        "pass_max": round(pass_max, 2),
        "review_max": round(review_max, 2),
        "severity": round(sev, 3),
        "quality_level": qlvl,
        "reason": "fallback_deltae_gates",
        "gate_scores": {"sharpness_score": sharp, "illumination_asymmetry": illum, "center_offset_mm": off},
    }


def _classify_deltae(deltae_value: float, gates: Dict[str, Any]) -> str:
    """pass/review/fail"""
    p = _safe_float(gates.get("pass_max"), 0.0)
    r = _safe_float(gates.get("review_max"), 0.0)
    if deltae_value <= p:
        return "pass"
    if deltae_value <= r:
        return "review"
    return "fail"


def _summarize_values(values: List[float], method: str) -> float:
    if not values:
        return 0.0
    m = str(method).lower().strip()
    if m == "max":
        return float(max(values))
    if m == "mean":
        return float(sum(values) / len(values))
    if m == "p95":
        return float(_percentile(values, 95.0))
    if m == "p90":
        return float(_percentile(values, 90.0))
    return float(max(values))  # default: conservative


def _confidence_from(label: str, quality_level: str, deltae_sev: float) -> float:
    base = {"PASS": 0.90, "REVIEW": 0.75, "FAIL": 0.85, "RETAKE": 0.65}.get(label, 0.75)
    q_pen = {"good": 0.00, "medium": 0.05, "poor": 0.10, "very_poor": 0.20}.get(quality_level, 0.08)
    c = base - q_pen - 0.10 * max(0.0, min(1.0, deltae_sev))
    return float(max(0.0, min(1.0, c)))


def _slim_cluster(c: Dict[str, Any]) -> Dict[str, Any]:
    """debug 경량화: 운영에 필요한 핵심만 남김."""
    keep = [
        "cluster_id",
        "mean_lab",
        "area_ratio",
        "inkness_score",
        "angular_score",
        "compactness",
        "alpha_like",
        "radial_presence_curve",
        # Alpha effective density fields
        "effective_density",
        "alpha_used",
        "alpha_fallback_level",
    ]
    out = {}
    for k in keep:
        if k in c:
            out[k] = c[k]
    # mean_lab 포맷 방어
    if "mean_lab" in out and isinstance(out["mean_lab"], (list, tuple)) and len(out["mean_lab"]) == 3:
        out["mean_lab"] = [float(out["mean_lab"][0]), float(out["mean_lab"][1]), float(out["mean_lab"][2])]
    return out


def _push_action(actions: List[Dict[str, str]], action: str, priority: str, note: str) -> None:
    # 중복 방지 (action+priority 동일하면 skip)
    for a in actions:
        if a.get("action") == action and a.get("priority") == priority:
            return
    actions.append({"action": action, "priority": priority, "note": note})


def _derive_next_actions(
    *,
    label: str,
    quality_level: str,
    ink_count_mismatch: bool,
    inkness_label: str,
    deltae_label: str,
    unmatched_test: List[int],
    unmatched_ref: List[int],
) -> List[Dict[str, str]]:
    """
    운영용 액션 룰:
    - RETAKE: RECAPTURE P0
    - FAIL:
        - ink_count_mismatch -> MANUAL_REVIEW P0 + STD_RETAKE P1(필요시)
        - deltae fail -> MANUAL_REVIEW P0 + RECAPTURE P1
        - inkness gap -> MANUAL_REVIEW P0 + RECALIBRATE P1(조명/마스크/세팅 점검)
    - REVIEW:
        - unmatched -> MANUAL_REVIEW P0
        - deltae review -> MANUAL_REVIEW P0
        - quality poor/medium -> RECAPTURE P1(권장)
    - PASS: NONE P2
    """
    actions: List[Dict[str, str]] = []

    if label == "RETAKE":
        _push_action(actions, "RECAPTURE", "P0", "촬영 품질 문제로 재촬영 필요")
        return actions

    if label == "FAIL":
        _push_action(actions, "MANUAL_REVIEW", "P0", "FAIL 원인 확인(자동결론 차단)")
        if ink_count_mismatch:
            _push_action(actions, "STD_RETAKE", "P1", "잉크 수 불일치: STD/레시피/공정 확인")
        if inkness_label == "gap":
            _push_action(actions, "RECALIBRATE", "P1", "inkness gap: ROI/마스크/조명/전처리 점검")
        if deltae_label == "fail":
            _push_action(actions, "RECAPTURE", "P1", "ΔE 과다: 재촬영 또는 공정 이탈 확인")
        if unmatched_test or unmatched_ref:
            _push_action(actions, "MANUAL_REVIEW", "P0", "매칭 불완전: 클러스터/정렬/마스크 점검")
        return actions

    if label == "REVIEW":
        _push_action(actions, "MANUAL_REVIEW", "P0", "REVIEW 사유 확인(임계치/매칭/품질)")
        if quality_level in ("poor", "medium"):
            _push_action(actions, "RECAPTURE", "P1", "품질 경계: 재촬영하면 PASS/FAIL 명확해질 가능성")
        if unmatched_test or unmatched_ref:
            _push_action(actions, "RECALIBRATE", "P1", "매칭 불완전: 클러스터링/정렬/ROI 설정 점검")
        return actions

    _push_action(actions, "NONE", "P2", "조치 없음")
    return actions


# ----------------------------
# Main builder
# ----------------------------
def build_decision(
    *,
    run_id: str,
    phase: str,
    engine_version: str = "v7.x",
    cfg: Optional[Dict[str, Any]] = None,
    gate_scores: Optional[Dict[str, float]] = None,
    expected_inks: Optional[int] = None,
    sample_clusters: Optional[List[Dict[str, Any]]] = None,
    # match_result: ink_match.compute_cluster_deltas() 결과를 그대로 넣는 것을 권장
    match_result: Optional[Dict[str, Any]] = None,
    # ΔE 대표값/inkness 대표값 산정 방식(운영에서 고정 추천)
    deltae_summary_method: str = "max",  # max | p95 | mean
    inkness_summary_method: str = "min",  # min | p10 | mean  (min은 보수적)
    # optional spatial gating
    spatial_fail_angular_score: Optional[float] = None,
    slim_debug: bool = True,
) -> Dict[str, Any]:
    """
    입력 데이터(클러스터/매칭/품질/설정) -> 최종 decision JSON 생성

    Required minimum:
      - sample_clusters: [{"cluster_id":int, "inkness_score":float, ...}, ...]
      - gate_scores: {"sharpness_score":..., "illumination_asymmetry":..., "center_offset_mm":...} (없어도 OK)
      - expected_inks: 1~3 (없으면 detected 기반으로만 decision)

    match_result (권장):
      ink_match.compute_cluster_deltas() 출력:
        {
          "matched": bool,
          "match_cost": float,
          "order": [sample_index aligned to baseline_index],
          "deltas": [{"index":i,"deltaE":..., ...}, ...],
          ...
        }
    """
    cfg = cfg or {}
    gate_scores = gate_scores or {}
    sample_clusters = sample_clusters or []

    # ----------------------------
    # deltaE_method / polar meta
    # ----------------------------
    v2_cfg = (cfg.get("v2_ink") or {}) if isinstance(cfg, dict) else {}
    deltae_method = _normalize_deltae_method(v2_cfg.get("deltaE_method", cfg.get("deltaE_method", "76")))

    polar = cfg.get("polar") if isinstance(cfg, dict) else None
    polar_T = int((polar or {}).get("T", 0)) if isinstance(polar, dict) else 0
    polar_R = int((polar or {}).get("R", 0)) if isinstance(polar, dict) else 0

    # ----------------------------
    # Policy snapshot (inkness + deltaE)
    # - threshold_policy가 확장되면(get_threshold_policy 등) 그걸 자동 사용
    # ----------------------------
    try:
        from ..measure.metrics import threshold_policy as tp  # Correct relative import for core.measure

        inkness_thr = tp.get_adaptive_threshold(gate_scores=gate_scores)
        classify_inkness_fn: Callable[[float, Dict[str, Any]], str] = tp.classify_inkness

        # 확장 버전이 있으면 사용
        if hasattr(tp, "get_threshold_policy"):
            policy = tp.get_threshold_policy(
                gate_scores=gate_scores,
                deltae_method=deltae_method,
                deltae_base_gates=cfg.get("deltaE_gates"),
            )
            inkness_thr = policy["inkness"]
            deltae_gates = policy["deltaE"]
            retake_recommended = bool(tp.should_retake_policy(policy)) if hasattr(tp, "should_retake_policy") else False
        else:
            deltae_gates = _fallback_deltae_gates(deltae_method, gate_scores)
            retake_recommended = bool(inkness_thr.get("quality_level") == "very_poor")
    except Exception:
        # threshold_policy import 실패 시도 대비 (최소 동작)
        inkness_thr = {
            "ink_threshold": 0.70,
            "review_lower": 0.55,
            "gap_upper": 0.50,
            "adjustment": 0.0,
            "quality_level": "good",
        }

        def classify_inkness_fn(s, t):
            if s >= t["ink_threshold"]:
                return "ink"
            elif s >= t["review_lower"]:
                return "review"
            else:
                return "gap"

        deltae_gates = _fallback_deltae_gates(deltae_method, gate_scores)
        retake_recommended = False

    # 입력 품질 레벨은 inkness 기준을 대표로 사용(필요하면 deltaE 쪽 품질도 같이 보고 결정 가능)
    quality_level = str(inkness_thr.get("quality_level", "good"))

    # deltaE gate가 확장 버전에 의해 들어왔는데 method가 meta와 다르면 meta를 gate 기준으로 맞춤
    deltae_method = _normalize_deltae_method(deltae_gates.get("deltaE_method", deltae_method))

    # ----------------------------
    # Detected inks (운영용)
    # - "gap" 아닌 클러스터 개수를 기본으로 사용
    # ----------------------------
    inkness_scores_all = [_safe_float(c.get("inkness_score"), 0.0) for c in sample_clusters]
    inkness_labels_all = [classify_inkness_fn(s, inkness_thr) for s in inkness_scores_all]
    detected_inks = int(sum(1 for lab in inkness_labels_all if lab != "gap"))

    ink_count_mismatch = False
    if expected_inks is not None:
        ink_count_mismatch = int(expected_inks) != int(detected_inks)

    # ----------------------------
    # Matching / deltaE list
    # ----------------------------
    mapping: List[Dict[str, Any]] = []
    deltas: List[float] = []
    unmatched_test: List[int] = []
    unmatched_ref: List[int] = []

    if (
        match_result
        and isinstance(match_result, dict)
        and (match_result.get("deltas") or match_result.get("cluster_deltas"))
    ):
        order = match_result.get("order") or []
        dlist = match_result.get("deltas") or match_result.get("cluster_deltas") or []
        # baseline index == ref_cluster, order[ref] == sample index == test_cluster
        for ref_idx, d in enumerate(dlist):
            de = _safe_float(d.get("deltaE"), 0.0)
            deltas.append(de)
            test_idx = order[ref_idx] if ref_idx < len(order) else None
            mapping.append(
                {
                    "test_cluster": int(test_idx) if test_idx is not None else ref_idx,
                    "ref_cluster": int(ref_idx),
                    "score": float(d.get("score", 1.0)),
                    "deltaE": float(de),
                }
            )
        # unmatched 처리(클러스터 수 불일치 등)
        k_ref = len(dlist)
        k_test = len(sample_clusters)
        if k_test > k_ref:
            unmatched_test = list(range(k_ref, k_test))
        elif k_ref > k_test:
            unmatched_ref = list(range(k_test, k_ref))
    else:
        # Fallback: per-color 모드 등에서 match_result가 없을 때
        # cluster들의 delta_e_mean이나 유사 필드를 deltaE로 사용
        if sample_clusters:
            for i, c in enumerate(sample_clusters):
                de = _safe_float(c.get("delta_e_mean"), None)
                if de is not None:
                    deltas.append(de)
            # match_result가 없고 delta_e_mean도 없으면 unmatched로 표시
            if not deltas:
                unmatched_test = [int(c.get("cluster_id", i)) for i, c in enumerate(sample_clusters)]

    # 대표 ΔE 값
    deltae_value = _summarize_values(deltas, deltae_summary_method)
    deltae_label = _classify_deltae(deltae_value, deltae_gates)

    # ----------------------------
    # 대표 inkness 값 (추천: min or p10)
    # - expected_inks가 있으면 inkness 상위 expected개를 후보로 보고 그 중 보수적으로 min
    # ----------------------------
    scored_clusters = []
    for i, c in enumerate(sample_clusters):
        scored_clusters.append((i, _safe_float(c.get("inkness_score"), 0.0), c))
    scored_clusters.sort(key=lambda x: x[1], reverse=True)

    if expected_inks is not None and expected_inks > 0:
        candidates = scored_clusters[: int(expected_inks)]
    else:
        # fallback: gap 아닌 것들을 우선 후보로
        candidates = [t for t in scored_clusters if classify_inkness_fn(t[1], inkness_thr) != "gap"] or scored_clusters

    cand_scores = [t[1] for t in candidates]
    if not cand_scores:
        inkness_value = 0.0
    else:
        m = inkness_summary_method.lower().strip()
        if m == "mean":
            inkness_value = float(sum(cand_scores) / len(cand_scores))
        elif m in ("p10", "p05"):
            inkness_value = float(_percentile(cand_scores, 10.0))
        else:
            inkness_value = float(min(cand_scores))  # default: conservative

    inkness_label = classify_inkness_fn(inkness_value, inkness_thr)

    # ----------------------------
    # Optional spatial gating
    # ----------------------------
    spatial_flag_fail = False
    spatial_min = None
    if spatial_fail_angular_score is not None:
        ang_scores = [_safe_float(c.get("angular_score"), 1.0) for c in sample_clusters]
        if ang_scores:
            spatial_min = min(ang_scores)
            spatial_flag_fail = spatial_min < float(spatial_fail_angular_score)

    # ----------------------------
    # Decide final label (우선순위 고정)
    # ----------------------------
    reason_codes: List[str] = []
    next_actions: List[Dict[str, str]] = []

    primary_reason = {"code": "UNKNOWN", "message": ""}  # Initialize

    if retake_recommended or quality_level == "very_poor":
        label = "RETAKE"
        reason_codes.append(RC_QUALITY_VERY_POOR_RETAKE)
        next_actions.append({"action": "RECAPTURE", "priority": "P0", "note": "촬영 품질 문제로 재촬영 필요"})
        primary_reason = {"code": RC_QUALITY_VERY_POOR_RETAKE, "message": "촬영 품질 문제로 재촬영 필요"}
    else:
        # FAIL 조건들
        fail_reasons = []
        if ink_count_mismatch:
            fail_reasons.append(RC_INK_COUNT_MISMATCH)
        if inkness_label == "gap":
            fail_reasons.append(RC_INKNESS_GAP_LIKELY_NON_INK)
        if deltae_label == "fail":
            fail_reasons.append(RC_DELTAE_FAIL_OVER_REVIEW_MAX)
        if spatial_flag_fail:
            fail_reasons.append(RC_SPATIAL_LOW_ANGULAR_CONTINUITY)

        if fail_reasons:
            label = "FAIL"
            reason_codes.extend(fail_reasons)
            primary_reason = {"code": fail_reasons[0], "message": "기준을 만족하지 못함"}
            next_actions.append({"action": "MANUAL_REVIEW", "priority": "P0", "note": "FAIL 원인 확인(자동결론 차단)"})
            # 상황에 따라 재촬영/STD_RETAKE 추가 권고
            if deltae_label == "fail":
                next_actions.append({"action": "RECAPTURE", "priority": "P1", "note": "ΔE 과다: 재촬영 또는 공정 확인"})
        else:
            # REVIEW 조건들
            review_reasons = []
            if inkness_label == "review":
                review_reasons.append(RC_INKNESS_REVIEW_WINDOW)
            if deltae_label == "review":
                review_reasons.append(RC_DELTAE_REVIEW_OVER_PASS_MAX)
            if unmatched_test:
                review_reasons.append(RC_MATCHING_INCOMPLETE_TEST_HAS_UNMATCHED)
            if unmatched_ref:
                review_reasons.append(RC_MATCHING_INCOMPLETE_REF_HAS_UNMATCHED)
            if quality_level in ("poor", "medium"):
                review_reasons.append(RC_QUALITY_POOR_TIGHTENED_GATES)

            if review_reasons:
                label = "REVIEW"
                reason_codes.extend(review_reasons)
                primary_reason = {"code": review_reasons[0], "message": "자동 판정 대신 검토 필요"}
                next_actions.append({"action": "MANUAL_REVIEW", "priority": "P0", "note": "REVIEW 사유 확인"})
            else:
                label = "PASS"
                reason_codes.append(RC_DELTAE_PASS)
                primary_reason = {"code": RC_DELTAE_PASS, "message": "기준 충족"}
                next_actions.append({"action": "NONE", "priority": "P2", "note": "조치 없음"})

    # confidence
    deltae_sev = _safe_float(deltae_gates.get("severity"), 0.0)
    confidence = _confidence_from(label, quality_level, deltae_sev)

    # ----------------------------
    # Build reason tree
    # ----------------------------
    reason_tree: List[Dict[str, Any]] = []

    # QUALITY gate node
    reason_tree.append(
        {
            "gate": "QUALITY",
            "result": "FAIL" if label == "RETAKE" else ("REVIEW" if quality_level in ("medium", "poor") else "PASS"),
            "code": (
                RC_QUALITY_VERY_POOR_RETAKE
                if label == "RETAKE"
                else (RC_QUALITY_POOR_TIGHTENED_GATES if quality_level in ("medium", "poor") else "QUALITY_OK")
            ),
            "evidence": {
                "value": None,
                "thresholds": None,
                "details": {"gate_scores": gate_scores, "quality_level": quality_level},
            },
        }
    )
    if expected_inks is not None:
        reason_tree.append(
            {
                "gate": "INK_COUNT",
                "result": "FAIL" if ink_count_mismatch else "PASS",
                "code": RC_INK_COUNT_MISMATCH if ink_count_mismatch else "INK_COUNT_OK",
                "evidence": {
                    "value": {"expected": int(expected_inks), "detected": int(detected_inks)},
                    "thresholds": None,
                    "details": {},
                },
            }
        )
    if match_result is not None:
        code = "MATCHING_OK"
        res = "PASS"
        if unmatched_test:
            code, res = RC_MATCHING_INCOMPLETE_TEST_HAS_UNMATCHED, "REVIEW"
        elif unmatched_ref:
            code, res = RC_MATCHING_INCOMPLETE_REF_HAS_UNMATCHED, "REVIEW"
        reason_tree.append(
            {
                "gate": "MATCHING",
                "result": res,
                "code": code,
                "evidence": {
                    "value": {"unmatched_test": unmatched_test, "unmatched_ref": unmatched_ref},
                    "thresholds": None,
                    "details": {
                        "matched": bool(match_result.get("matched")),
                        "match_cost": match_result.get("match_cost"),
                    },
                },
            }
        )
    reason_tree.append(
        {
            "gate": "INKNESS",
            "result": "FAIL" if inkness_label == "gap" else ("REVIEW" if inkness_label == "review" else "PASS"),
            "code": (
                RC_INKNESS_GAP_LIKELY_NON_INK
                if inkness_label == "gap"
                else (RC_INKNESS_REVIEW_WINDOW if inkness_label == "review" else "INKNESS_PASS")
            ),
            "evidence": {
                "value": round(float(inkness_value), 4),
                "thresholds": {
                    "ink_threshold": float(inkness_thr.get("ink_threshold", 0.0)),
                    "review_lower": float(inkness_thr.get("review_lower", 0.0)),
                    "gap_upper": float(inkness_thr.get("gap_upper", 0.0)),
                },
                "details": {"summary_method": inkness_summary_method},
            },
        }
    )
    reason_tree.append(
        {
            "gate": "DELTAE",
            "result": "FAIL" if deltae_label == "fail" else ("REVIEW" if deltae_label == "review" else "PASS"),
            "code": (
                RC_DELTAE_FAIL_OVER_REVIEW_MAX
                if deltae_label == "fail"
                else (RC_DELTAE_REVIEW_OVER_PASS_MAX if deltae_label == "review" else RC_DELTAE_PASS)
            ),
            "evidence": {
                "value": round(float(deltae_value), 4),
                "thresholds": {
                    "pass": float(deltae_gates.get("pass_max", 0.0)),
                    "review": float(deltae_gates.get("review_max", 0.0)),
                },
                "details": {
                    "deltaE_method": deltae_method,
                    "summary_method": deltae_summary_method,
                    "per_cluster": (
                        [
                            {"pair": f"test{m['test_cluster']}-ref{m['ref_cluster']}", "deltaE": m["deltaE"]}
                            for m in mapping
                        ]
                        if mapping
                        else []
                    ),
                },
            },
        }
    )
    if spatial_fail_angular_score is not None:
        reason_tree.append(
            {
                "gate": "SPATIAL",
                "result": "FAIL" if spatial_flag_fail else "PASS",
                "code": RC_SPATIAL_LOW_ANGULAR_CONTINUITY if spatial_flag_fail else "SPATIAL_OK",
                "evidence": {
                    "value": None if spatial_min is None else round(float(spatial_min), 4),
                    "thresholds": {"angular_score_min": float(spatial_fail_angular_score)},
                    "details": {},
                },
            }
        )

    # debug payload
    debug_clusters = sample_clusters
    if slim_debug:
        debug_clusters = [_slim_cluster(c) for c in sample_clusters]

    return {
        "run_id": run_id,
        "timestamp": _utc_now_iso(),
        "phase": phase,
        "engine": {"name": "LensSignatureEngine", "version": engine_version, "policy_version": "threshold_policy.v1"},
        "context": {
            "sku": cfg.get("sku"),
            "expected_inks": expected_inks,
            "detected_inks": detected_inks,
            "deltaE_method": deltae_method,
            "polar": {"order": "TR", "T": polar_T, "R": polar_R},
        },
        "input_quality": {
            "gate_scores": gate_scores,
            "quality_level": quality_level,
            "retake_recommended": bool(label == "RETAKE"),
        },
        "policy_snapshot": {
            "inkness": {
                "ink_threshold": inkness_thr.get("ink_threshold"),
                "review_lower": inkness_thr.get("review_lower"),
                "gap_upper": inkness_thr.get("gap_upper"),
                "adjustment": inkness_thr.get("adjustment"),
                "quality_level": inkness_thr.get("quality_level"),
            },
            "deltaE": {
                "pass_max": deltae_gates.get("pass_max"),
                "review_max": deltae_gates.get("review_max"),
                "severity": deltae_gates.get("severity"),
                "quality_level": deltae_gates.get("quality_level"),
                "reason": deltae_gates.get("reason"),
            },
        },
        "matching": {
            "mapping": mapping,
            "unmatched_test_clusters": unmatched_test,
            "unmatched_ref_clusters": unmatched_ref,
        },
        "metrics_summary": {
            "deltaE_value": {"method": deltae_summary_method, "value": round(float(deltae_value), 4)},
            "inkness_value": {"method": inkness_summary_method, "value": round(float(inkness_value), 4)},
            "ink_count_mismatch": bool(ink_count_mismatch),
        },
        "decision": {
            "label": label,
            "confidence": round(float(confidence), 3),
            "primary_reason": primary_reason,
            "reason_codes": reason_codes,
            "next_actions": next_actions,
        },
        "reason_tree": reason_tree,
        "debug": {"clusters": debug_clusters},
    }
