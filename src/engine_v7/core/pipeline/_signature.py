"""Signature evaluation helpers extracted from analyzer.py."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..signature.signature_compare import band_violation_v2, segment_stats, signature_compare, weighted_fail_ratio
from ..types import SignatureResult


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
