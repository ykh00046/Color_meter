from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from ..geometry.lens_geometry import detect_lens_circle
from .radial_signature import build_radial_signature, to_polar


def _segment_slices(Rp: int, segments: List[Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
    out: Dict[str, Tuple[int, int]] = {}
    for seg in segments:
        name = seg.get("name", "seg")
        s = int(round(float(seg["start"]) * Rp))
        e = int(round(float(seg["end"]) * Rp))
        s = max(0, min(Rp - 1, s))
        e = max(s + 1, min(Rp, e))
        out[name] = (s, e)
    return out


def suggest_segment_k_from_stds(
    std_bgr_list: List[np.ndarray],
    *,
    R: int,
    T: int,
    r_start: float,
    r_end: float,
    segments: List[Dict[str, Any]],
    percentile: float = 99.5,
    min_k: float = 2.0,
    max_k: float = 4.0,
    eps: float = 1e-3,
    std_floor: float = 1.0,
) -> Dict[str, float]:
    """
    Suggest per-segment band_k based on normalized deviation distribution in the STD set.

    For each STD image i:
      curve_i = radial signature mean (theta-mean Lab curve), shape (Rp,3)
    We compute:
      mean = average(curve_i)
      std  = std(curve_i)  (per Rp, per channel)

    Then normalized deviation:
      z = |curve_i - mean| / (max(std, std_floor) + eps)
    Collect z over (i, r, channel) within each segment and choose k as percentile(z).

    Args:
        std_floor: Minimum std value to prevent z distortion (e.g. 1.0 for Lab)

    Returns {segment_name: k} clamped to [min_k, max_k].
    """
    if len(std_bgr_list) < 2:
        # Not enough data to estimate; return neutral defaults
        return {seg.get("name", "seg"): float(min_k) for seg in segments}

    curves = []
    for bgr in std_bgr_list:
        geom = detect_lens_circle(bgr)
        polar = to_polar(bgr, geom, R=R, T=T)
        mean_curve, _, _ = build_radial_signature(polar, r_start=r_start, r_end=r_end)
        curves.append(mean_curve.astype(np.float32))

    stack = np.stack(curves, axis=0)  # (N, Rp, 3)
    mean = stack.mean(axis=0)  # (Rp, 3)
    std = stack.std(axis=0)  # (Rp, 3)

    # Apply std_floor to prevent z distortion on low variance
    std = np.maximum(std, std_floor)

    z = np.abs(stack - mean[None, ...]) / (std[None, ...] + eps)  # (N, Rp, 3)

    Rp = int(stack.shape[1])
    seg_slices = _segment_slices(Rp, segments)

    out: Dict[str, float] = {}
    for name, (s, e) in seg_slices.items():
        vals = z[:, s:e, :].reshape(-1)
        k = float(np.percentile(vals, percentile))
        k = float(np.clip(k, min_k, max_k))
        out[name] = k
    return out
