from __future__ import annotations

import math
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


def compute_radial_summaries(L: List[float]) -> Dict[str, Optional[float]]:
    if not L or len(L) < 10:
        return {
            "fade_slope_outer": None,
            "knee_r": None,
            "inner_mean_L": None,
            "outer_mean_L": None,
        }

    n = len(L)

    def _idx(r: float) -> int:
        return max(0, min(n - 1, int(round(r * (n - 1)))))

    def _mean_range(r0: float, r1: float) -> float:
        i0, i1 = _idx(r0), _idx(r1)
        if i1 <= i0:
            return float(L[i0])
        xs = L[i0 : i1 + 1]
        return sum(xs) / len(xs)

    inner_mean = _mean_range(0.15, 0.35)
    outer_mean = _mean_range(0.70, 0.95)

    i0, i1 = _idx(0.70), _idx(0.95)
    xs = [(i / (n - 1)) for i in range(i0, i1 + 1)]
    ys = [L[i] for i in range(i0, i1 + 1)]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    varx = sum((x - mx) ** 2 for x in xs)
    slope = None
    if varx > 1e-12:
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        slope = cov / varx

    tail_ref = _mean_range(0.90, 0.95)
    vary = sum((y - my) ** 2 for y in ys) / max(1, (len(ys) - 1))
    stdy = math.sqrt(vary)
    thr = 0.5 * stdy

    knee = None
    for r in [i / (n - 1) for i in range(_idx(0.50), _idx(0.95) + 1)]:
        if abs(L[_idx(r)] - tail_ref) >= thr:
            knee = r
            break

    return {
        "fade_slope_outer": slope,
        "knee_r": knee,
        "inner_mean_L": inner_mean,
        "outer_mean_L": outer_mean,
    }


def compute_radial_summaries_de(de_curve: List[float]) -> Dict[str, Optional[float]]:
    if not de_curve or len(de_curve) < 10:
        return {
            "fade_slope_outer_de": None,
            "knee_r_de": None,
            "inner_mean_de": None,
            "outer_mean_de": None,
        }

    n = len(de_curve)

    def _idx(r: float) -> int:
        return max(0, min(n - 1, int(round(r * (n - 1)))))

    def _mean_range(r0: float, r1: float) -> float:
        i0, i1 = _idx(r0), _idx(r1)
        if i1 <= i0:
            return float(de_curve[i0])
        xs = de_curve[i0 : i1 + 1]
        return sum(xs) / len(xs)

    inner_mean = _mean_range(0.15, 0.35)
    outer_mean = _mean_range(0.70, 0.95)

    i0, i1 = _idx(0.70), _idx(0.95)
    xs = [(i / (n - 1)) for i in range(i0, i1 + 1)]
    ys = [de_curve[i] for i in range(i0, i1 + 1)]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    varx = sum((x - mx) ** 2 for x in xs)
    slope = None
    if varx > 1e-12:
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        slope = cov / varx

    tail_ref = _mean_range(0.90, 0.95)
    vary = sum((y - my) ** 2 for y in ys) / max(1, (len(ys) - 1))
    stdy = math.sqrt(vary)
    thr = 0.5 * stdy

    knee = None
    for r in [i / (n - 1) for i in range(_idx(0.50), _idx(0.95) + 1)]:
        if abs(de_curve[_idx(r)] - tail_ref) >= thr:
            knee = r
            break

    return {
        "fade_slope_outer_de": slope,
        "knee_r_de": knee,
        "inner_mean_de": inner_mean,
        "outer_mean_de": outer_mean,
    }


def assemble_features(
    *,
    image_path: str,
    sku: Optional[str],
    ink_set: Optional[str],
    phase: str,
    cfg: Dict[str, Any],
    cfg_hash: str,
    engine_version: str,
    generated_at_iso: Optional[str] = None,
    signature_block: Optional[Dict[str, Any]] = None,
    best_mode: Optional[str] = None,
    diagnostics_block: Optional[Dict[str, Any]] = None,
    v2_block: Optional[Dict[str, Any]] = None,
    anomaly_block: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    generated_at = generated_at_iso or datetime.now().isoformat()
    features: Dict[str, Any] = {
        "schema_version": "features.v1",
        "engine_version": engine_version,
        "generated_at": generated_at,
        "cfg_hash": cfg_hash,
        "image": {
            "path": image_path,
            "sku": sku,
            "ink_set": ink_set,
            "phase": phase,
        },
        "roi": _build_roi(cfg),
        "signature": _build_signature(signature_block, best_mode),
        "color": _build_color(diagnostics_block, signature_block),
        "ink": _build_ink(v2_block, cfg),
        "pattern": _build_pattern(diagnostics_block, anomaly_block),
        "radial": _build_radial(signature_block, cfg),
        "provenance": {
            "sources": _sources_present(signature_block, diagnostics_block, v2_block, anomaly_block),
            "shadow_only": True,
            "notes": [],
        },
    }
    return features


def _build_roi(cfg: Dict[str, Any]) -> Dict[str, Any]:
    polar = cfg.get("polar", {}) or {}
    sig = cfg.get("signature", {}) or {}
    segments = sig.get("segments", []) or []
    return {
        "polar": {
            "r_start": sig.get("r_start"),
            "r_end": sig.get("r_end"),
            "R": polar.get("R"),
            "T": polar.get("T"),
            "segments": [
                {
                    "name": s.get("name"),
                    "start": s.get("start"),
                    "end": s.get("end"),
                    "weight": s.get("weight"),
                }
                for s in segments
            ],
        },
        "semantic": {"enabled": False, "zones": []},
    }


def _build_signature(signature_block: Optional[Dict[str, Any]], best_mode: Optional[str]) -> Dict[str, Any]:
    if not signature_block:
        return {
            "best_mode": best_mode,
            "passed": None,
            "corr": None,
            "deltaE_mean": None,
            "deltaE_p95": None,
            "fail_ratio": None,
        }
    return {
        "best_mode": best_mode,
        "passed": signature_block.get("passed"),
        "corr": signature_block.get("score_corr"),
        "deltaE_mean": signature_block.get("delta_e_mean"),
        "deltaE_p95": signature_block.get("delta_e_p95"),
        "fail_ratio": signature_block.get("fail_ratio"),
    }


def _build_color(
    diagnostics_block: Optional[Dict[str, Any]],
    signature_block: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    overall = (((diagnostics_block or {}).get("color") or {}).get("overall")) or {}
    # B. analyzer 실제 출력 키 매핑 (sample_lab_mean_cie, std_lab_mean_cie)
    direction = overall.get("direction") or {}
    seg_stats = (signature_block or {}).get("debug", {}).get("segment_stats") or {}
    by_segment = {}
    for seg in ("inner", "mid", "outer"):
        seg_row = seg_stats.get(seg, {}) if isinstance(seg_stats, dict) else {}
        by_segment[seg] = {
            "lab_mean": [None, None, None],
            "deltaE_mean": seg_row.get("de_mean"),
            "deltaE_p95": seg_row.get("de_p95"),
        }
    return {
        "overall": {
            # B. 실제 analyzer 키: sample_lab_mean_cie, std_lab_mean_cie
            "sample_lab_mean_cie": overall.get("sample_lab_mean_cie") or [None, None, None],
            "std_lab_mean_cie": overall.get("std_lab_mean_cie") or [None, None, None],
            "sample_lab_mean_cv8": overall.get("std_lab_mean_cv8") or [None, None, None],
            "delta_lab_vs_active": {
                "dL": direction.get("delta_L"),
                "da": direction.get("delta_a"),
                "db": direction.get("delta_b"),
            },
            "deltaE76_roi": overall.get("deltaE76_roi"),
            "deltaE2000_roi": overall.get("deltaE2000_roi"),
            "_scale_note": overall.get("_scale_note", "CIE scale"),
        },
        "by_segment": by_segment,
    }


def _build_ink(v2_block: Optional[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not cfg.get("v2_ink", {}).get("enabled", False):
        return {"enabled": False}
    if not v2_block:
        return {"enabled": True}

    seg = v2_block.get("segmentation", {}) or {}
    auto = v2_block.get("auto_estimation", {}) or {}
    ink_match = v2_block.get("ink_match", {}) or {}
    traj_summary = ink_match.get("trajectory_summary") or {}
    deltas = ink_match.get("deltas", []) or []

    max_delta = None
    for d in deltas:
        if "deltaE" in d and (max_delta is None or d["deltaE"] > max_delta):
            max_delta = d["deltaE"]

    clusters_in = seg.get("clusters") or []
    clusters = []
    for i, c in enumerate(clusters_in):
        lab = c.get("mean_lab") or c.get("lab_center") or [None, None, None]
        sort_key = lab[0] if lab and len(lab) == 3 else None
        clusters.append(
            {
                "id": c.get("id") or f"ink{i + 1}",
                "lab_center": lab,
                "area_ratio": c.get("area_ratio"),
                "sort_key": sort_key,
            }
        )
    clusters.sort(key=lambda x: (x["sort_key"] is None, x["sort_key"]))

    return {
        "enabled": True,
        "k_expected": v2_block.get("expected_ink_count"),
        "k_expected_registry": v2_block.get("expected_ink_count_registry"),
        "k_expected_input": v2_block.get("expected_ink_count_input"),
        "k_used": seg.get("k_used"),
        "auto_k_best": auto.get("auto_k_best"),
        "auto_k_confidence": auto.get("confidence"),
        "uncertain": ink_match.get("warning") == "INK_CLUSTER_MATCH_UNCERTAIN",
        "cluster_sort_policy": "L_asc",
        "clusters": clusters,
        "shift": {
            "max_off_track": traj_summary.get("max_off_track"),
            "max_deltaE": max_delta,
            "max_deltaE_matched_inks": max_delta,
        },
    }


def _build_pattern(
    diagnostics_block: Optional[Dict[str, Any]], anomaly_block: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    dot = (((diagnostics_block or {}).get("pattern") or {}).get("dot")) or {}
    anomaly_scores = (anomaly_block or {}).get("scores") if anomaly_block else {}
    return {
        "dot": {
            "coverage": dot.get("coverage_sample"),
            "edge_sharpness": dot.get("edge_sharpness_sample"),
            "blob_count": None,
            "blob_area_p50": None,
            "blob_area_p95": None,
            "radial_uniformity": anomaly_scores.get("angular_uniformity") if anomaly_scores else None,
        },
        "units": {
            "area": "px2",
            "sharpness": "unknown",
        },
    }


def _build_radial(signature_block: Optional[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    polar = cfg.get("polar", {}) or {}
    debug = (signature_block or {}).get("debug") or {}
    L = debug.get("L_mean_profile") or []
    a = debug.get("a_mean_profile") or []
    b = debug.get("b_mean_profile") or []
    de_curve = debug.get("delta_e_curve") or []
    summaries = compute_radial_summaries(L)
    if not summaries.get("fade_slope_outer") and de_curve:
        summaries.update(compute_radial_summaries_de(de_curve))
    return {
        "r_bins": polar.get("R"),
        "t_bins": polar.get("T"),
        "L_mean_profile": L,
        "a_mean_profile": a,
        "b_mean_profile": b,
        "deltaE_profile": de_curve,
        "summaries": summaries,
    }


def _sources_present(
    signature_block: Optional[Dict[str, Any]],
    diagnostics_block: Optional[Dict[str, Any]],
    v2_block: Optional[Dict[str, Any]],
    anomaly_block: Optional[Dict[str, Any]],
) -> List[str]:
    sources = []
    if signature_block:
        sources.append("signature")
    if diagnostics_block:
        sources.append("diagnostics")
    if v2_block:
        sources.append("v2_diagnostics")
    if anomaly_block:
        sources.append("anomaly")
    return sources


def dataclass_to_dict(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    try:
        return asdict(value)
    except TypeError:
        if isinstance(value, dict):
            return value
        return None
