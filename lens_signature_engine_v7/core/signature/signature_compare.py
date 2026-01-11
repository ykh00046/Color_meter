from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..utils import cie76_deltaE, cie2000_deltaE, corrcoef_safe


def signature_compare(test_mean: np.ndarray, std_mean: np.ndarray) -> Dict[str, Any]:
    """
    Compare test signature with STD signature.

    Args:
        test_mean: Test radial signature (R', 3) in Lab space
        std_mean: STD radial signature (R', 3) in Lab space

    IMPORTANT:
    - Both test_mean and std_mean MUST be in the SAME Lab color space
    - Supported spaces: "opencv_lab" (0-255 scale) or "cie_lab" (L:0-100, a/b:-128~+127)
    - StdModel.meta["lab_space"] should specify which space is used
    - Mixing spaces will cause incorrect deltaE values and thresholds

    Returns:
        Dict with correlation, deltaE curves, and aligned test signature
    """
    # Lab scale verification should be done by the caller using StdModel.meta["lab_space"]
    if test_mean.shape[0] != std_mean.shape[0]:
        x_std = np.linspace(0, 1, std_mean.shape[0])
        x_test = np.linspace(0, 1, test_mean.shape[0])
        test_interp = np.vstack([np.interp(x_std, x_test, test_mean[:, i]) for i in range(3)]).T
        test_mean = test_interp.astype(np.float32)

    de_curve = cie76_deltaE(test_mean, std_mean)
    de2000_curve = cie2000_deltaE(test_mean, std_mean)
    return {
        "corr": corrcoef_safe(test_mean, std_mean),
        "delta_e_curve": de_curve,
        "delta_e_mean": float(np.mean(de_curve)),
        "delta_e_p95": float(np.percentile(de_curve, 95)),
        "delta_e00_mean": float(np.mean(de2000_curve)),
        "delta_e00_p95": float(np.percentile(de2000_curve, 95)),
        "test_mean_aligned": test_mean,
    }


def band_violation(
    test_mean_aligned: np.ndarray,
    std_mean: np.ndarray,
    std_std: Optional[np.ndarray],
    *,
    k: float,
    eps: float = 1e-3,
) -> Dict[str, Any]:
    if std_std is None:
        return {"fail_mask": np.zeros((std_mean.shape[0],), dtype=bool), "fail_ratio": 0.0}

    tol = k * std_std + eps
    diff = np.abs(test_mean_aligned - std_mean)
    fail = np.any(diff > tol, axis=1)
    return {"fail_mask": fail, "fail_ratio": float(fail.mean())}


def segment_stats(
    de_curve: np.ndarray,
    fail_mask: np.ndarray,
    segments: list,
) -> Dict[str, Any]:
    """
    Compute per-segment statistics over radius bins.
    segments: list of dicts {name,start,end,weight} where start/end in [0,1] over de_curve length
    Returns dict with per-segment mean/p95 and fail_ratio.
    """
    R = int(de_curve.shape[0])
    out = {}
    for seg in segments:
        name = seg.get("name", "seg")
        s = int(round(float(seg["start"]) * R))
        e = int(round(float(seg["end"]) * R))
        s = max(0, min(R - 1, s))
        e = max(s + 1, min(R, e))
        de_seg = de_curve[s:e]
        fm_seg = fail_mask[s:e] if fail_mask is not None else np.zeros((e - s,), dtype=bool)
        out[name] = {
            "r_idx": [s, e],
            "weight": float(seg.get("weight", 1.0)),
            "de_mean": float(np.mean(de_seg)) if de_seg.size else 0.0,
            "de_p95": float(np.percentile(de_seg, 95)) if de_seg.size else 0.0,
            "fail_ratio": float(np.mean(fm_seg)) if fm_seg.size else 0.0,
        }
    return out


def weighted_fail_ratio(seg_stats: Dict[str, Any]) -> float:
    num = 0.0
    den = 0.0
    for _, v in seg_stats.items():
        w = float(v.get("weight", 1.0))
        num += w * float(v.get("fail_ratio", 0.0))
        den += w
    return float(num / (den + 1e-9))


def band_violation_v2(
    test_mean_aligned: np.ndarray,
    std_mean: np.ndarray,
    std_std: Optional[np.ndarray],
    *,
    k: float | np.ndarray,
    eps: float = 1e-3,
) -> Dict[str, Any]:
    """
    Band violation supporting scalar k or per-r-bin k array.
      k: scalar OR shape (R',) OR shape (R',1) to broadcast to channels.
    """
    R = int(std_mean.shape[0])
    if std_std is None:
        return {"fail_mask": np.zeros((R,), dtype=bool), "fail_ratio": 0.0}

    k_arr: np.ndarray
    if isinstance(k, (float, int)):
        k_arr = np.full((R, 1), float(k), dtype=np.float32)
    else:
        k_arr = np.asarray(k, dtype=np.float32)
        if k_arr.ndim == 1:
            k_arr = k_arr.reshape(R, 1)
        elif k_arr.ndim == 2 and k_arr.shape[1] != 1:
            # allow (R,3) too
            pass

    tol = k_arr * std_std + eps
    diff = np.abs(test_mean_aligned - std_mean)
    fail = np.any(diff > tol, axis=1)
    return {"fail_mask": fail, "fail_ratio": float(fail.mean())}
