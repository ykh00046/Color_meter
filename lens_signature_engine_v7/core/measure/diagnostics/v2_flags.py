"""
V2 Flags Builder - Warning Standardization Module

Converts v2_diagnostics warnings into structured flag format for UI/reporting.

Operational Stability Improvements:
1. auto 키 fallback - Supports auto_estimation/auto_k/auto schemas
2. uncertain 체크 확장 - Checks warning/warnings/top-level warnings
3. expanded_search_used 파서 강화 - Handles dict/string/case-insensitive formats
4. flags severity 정렬 - Sorts WARN → INFO → OK for UI priority
5. index_type 명시 - Distinguishes baseline_index vs sample_index
6. None-safe formatting - Replaces None with "unknown" in detail strings

Schema: v2_flags.v1
Severity levels: WARN (critical) > INFO (notable) > OK (normal)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_v2_flags(v2_diag: Optional[Dict[str, Any]], source: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not v2_diag:
        return {
            "schema": "v2_flags.v1",
            "severity": "OK",
            "flags": [],
        }

    warnings = set(v2_diag.get("warnings") or [])
    sampling = v2_diag.get("sampling", {}) or {}
    seg = v2_diag.get("segmentation", {}) or {}
    quality = seg.get("quality", {}) or {}
    # 1) auto 키 fallback - 스키마 변화 대응
    auto = v2_diag.get("auto_estimation") or v2_diag.get("auto_k") or v2_diag.get("auto") or {}

    flags: List[Dict[str, Any]] = []

    # 6) Helper for None-safe formatting
    def _fmt(val: Any, default: str = "unknown") -> str:
        """Format value for detail string, replacing None with 'unknown'"""
        if val is None:
            return default
        if isinstance(val, float):
            return f"{val:.2f}"
        return str(val)

    def _add(code: str, severity: str, title: str, detail: str, evidence: Optional[Dict[str, Any]] = None) -> None:
        # Clean evidence: remove None values
        if evidence:
            evidence = {k: v for k, v in evidence.items() if v is not None}
        flags.append(
            {
                "code": code,
                "severity": severity,
                "title": title,
                "detail": detail,
                "evidence": evidence or {},
            }
        )

    # 7) ink_match.matched == False일 때 WARN 플래그 (매칭 실패가 조용히 묻히는 것 방지)
    ink_match = v2_diag.get("ink_match", {}) or {}
    if not ink_match.get("matched", False):
        _add(
            "V2_INK_MATCH_FAILED",
            "WARN",
            "Ink match failed",
            "baseline/sample cluster count mismatch or matching unavailable",
            {"match_cost": ink_match.get("match_cost")},
        )

    if "INK_SEPARATION_LOW_CONFIDENCE" in warnings:
        n_pixels = sampling.get("n_pixels_used")
        _add(
            "V2_LOW_CONF_SAMPLING",
            "WARN",
            "Low confidence sampling",
            f"n_pixels_used={_fmt(n_pixels)}",
            {"n_pixels_used": n_pixels},
        )

    if "INK_CLUSTER_TOO_SMALL" in warnings:
        min_area = quality.get("min_area_ratio")
        _add(
            "V2_CLUSTER_TOO_SMALL",
            "INFO",
            "Small cluster detected",
            f"min_area_ratio={_fmt(min_area)}",
            {"min_area_ratio": min_area},
        )

    if "INK_CLUSTER_OVERLAP_HIGH" in warnings:
        min_de = quality.get("min_deltaE_between_clusters")
        margin = quality.get("separation_margin")
        _add(
            "V2_OVERLAP_HIGH",
            "WARN",
            "Cluster overlap tendency",
            f"min_deltaE={_fmt(min_de)}, margin={_fmt(margin)}",
            {
                "min_deltaE_between_clusters": min_de,
                "separation_margin": margin,
            },
        )

    if "INK_COUNT_MISMATCH_SUSPECTED" in warnings:
        expected = v2_diag.get("expected_ink_count")
        suggested = auto.get("suggested_k")
        conf = auto.get("confidence")
        _add(
            "V2_AUTO_K_MISMATCH",
            "WARN",
            "Auto-k suggests different ink count",
            f"expected={_fmt(expected)}, suggested={_fmt(suggested)}, conf={_fmt(conf)}",
            {
                "expected_ink_count": expected,
                "suggested_k": suggested,
                "confidence": conf,
            },
        )

    if "AUTO_K_LOW_CONFIDENCE" in warnings:
        conf = auto.get("confidence")
        _add(
            "V2_AUTO_K_LOW_CONFIDENCE",
            "INFO",
            "Auto-k confidence is low",
            f"confidence={_fmt(conf)}",
            {"confidence": conf},
        )

    ink_match = v2_diag.get("ink_match", {}) or {}
    if ink_match.get("matched"):
        deltas = ink_match.get("deltas", []) or []
        traj_summary = ink_match.get("trajectory_summary") or {}
        max_off_track = traj_summary.get("max_off_track")
        max_idx = traj_summary.get("max_off_track_index")
        max_delta = None
        if max_off_track is None:
            for d in deltas:
                if "deltaE" in d and (max_delta is None or d["deltaE"] > max_delta):
                    max_delta = d["deltaE"]
                    max_idx = d.get("index")
        # 2) uncertain 체크 확장 - 스키마 변화 대응
        uncertain = (
            ink_match.get("warning") == "INK_CLUSTER_MATCH_UNCERTAIN"
            or "INK_CLUSTER_MATCH_UNCERTAIN" in (ink_match.get("warnings") or [])
            or "INK_CLUSTER_MATCH_UNCERTAIN" in warnings
        )
        if max_off_track is not None or max_delta is not None:
            if max_off_track is not None:
                detail = f"max_off_track={_fmt(max_off_track)}"
            else:
                detail = f"max_deltaE={_fmt(max_delta)}"
            if max_idx is not None:
                detail = f"{detail} (Ink{max_idx + 1})"
            if uncertain:
                detail = f"{detail}, uncertain=true"
            evidence = {"uncertain": uncertain}
            if max_off_track is not None:
                evidence["max_off_track"] = max_off_track
            if max_delta is not None:
                evidence["max_deltaE"] = max_delta
            if max_idx is not None:
                evidence["max_index"] = max_idx
                # 5) index_type 추가 - baseline vs sample 명시
                evidence["index_type"] = "baseline_index"
            _add(
                "V2_INK_SHIFT_SUMMARY",
                "INFO",
                "Ink shift summary",
                detail,
                evidence,
            )

    # 3) expanded_search_used 파서 강화 - 다양한 형태 대응
    notes = auto.get("notes") or []
    expanded = False
    for n in notes:
        if isinstance(n, dict) and n.get("expanded_search_used") is True:
            expanded = True
            break
        elif isinstance(n, str):
            n_lower = n.lower()
            if "expanded_search_used" in n_lower and any(x in n_lower for x in ("true", "=1", "yes")):
                expanded = True
                break

    if expanded:
        _add(
            "V2_EXPANDED_SEARCH_USED",
            "INFO",
            "Expanded auto-k search used",
            "expanded_search_used=true",
            {"expanded_search_used": True},
        )

    # 4) flags severity 정렬 - WARN → INFO → OK 순서
    severity_order = {"WARN": 0, "INFO": 1, "OK": 2}
    flags.sort(key=lambda f: severity_order.get(f["severity"], 9))

    severity = "OK"
    if any(f["severity"] == "WARN" for f in flags):
        severity = "WARN"
    elif flags:
        severity = "INFO"

    payload = {
        "schema": "v2_flags.v1",
        "severity": severity,
        "flags": flags,
    }
    if source:
        payload["source"] = source
    return payload
