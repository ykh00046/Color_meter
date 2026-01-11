from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .ink_metrics import ensure_cie_lab


def build_roi_mask(
    T: int,
    R: int,
    r_start: float,
    r_end: float,
    center_excluded_frac: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    r_start_eff = max(float(r_start), float(center_excluded_frac or 0.0))
    r0 = int(round(r_start_eff * R))
    r1 = int(round(float(r_end) * R))
    r0 = max(0, min(R - 1, r0))
    r1 = max(r0 + 1, min(R, r1))
    mask = np.zeros((T, R), dtype=bool)
    mask[:, r0:r1] = True
    meta = {
        "polar_order": "TR",
        "T": int(T),
        "R": int(R),
        "r_start_config": float(r_start),
        "r_start_effective": float(r_start_eff),
        "r_end": float(r_end),
        "center_excluded_frac": float(center_excluded_frac or 0.0),
    }
    return mask, meta


def _quantile_mask(values: np.ndarray, p: float, largest: bool) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(values, dtype=bool)
    q = np.quantile(values, 1 - p) if largest else np.quantile(values, p)
    if largest:
        return values >= q
    return values <= q


def build_sampling_mask(
    lab_map: np.ndarray,
    roi_mask: np.ndarray,
    cfg: Dict[str, Any],
    rng_seed: int | None = None,
) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
    v2_cfg = cfg.get("v2_ink", {})
    dark_p = float(v2_cfg.get("dark_top_p", 0.25))
    chroma_p = float(v2_cfg.get("chroma_top_p", 0.35))
    min_samples = int(v2_cfg.get("min_samples", 8000))
    min_samples_warn = int(v2_cfg.get("min_samples_warn", 2000))

    warnings: List[str] = []
    L = lab_map[..., 0]
    a = lab_map[..., 1]
    b = lab_map[..., 2]
    chroma = np.sqrt(a * a + b * b)
    rng = np.random.default_rng(rng_seed)

    roi_idx = np.where(roi_mask)
    if roi_idx[0].size == 0:
        meta = {"rule": "none", "n_pixels_used": 0, "random_fallback_used": False, "rng_seed_used": rng_seed}
        return np.zeros_like(roi_mask, dtype=bool), meta, ["INK_SAMPLING_EMPTY"]

    L_roi = L[roi_idx]
    chroma_roi = chroma[roi_idx]

    dark_sel = _quantile_mask(L_roi, dark_p, largest=False)
    chroma_sel = _quantile_mask(chroma_roi, chroma_p, largest=True)
    union_sel = dark_sel | chroma_sel

    def _mask_from_sel(mask_sel: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(roi_mask, dtype=bool)
        rr = roi_idx[0][mask_sel]
        cc = roi_idx[1][mask_sel]
        mask[rr, cc] = True
        return mask

    mask = _mask_from_sel(union_sel)
    rule = "ink_candidate_union_v1"
    random_fallback = False
    if int(mask.sum()) < min_samples:
        mask = _mask_from_sel(dark_sel)
        rule = "dark_top_p"
    if int(mask.sum()) < min_samples:
        mask = _mask_from_sel(chroma_sel)
        rule = "high_chroma_top_p"
    if int(mask.sum()) < min_samples_warn:
        rr_all = roi_idx[0]
        cc_all = roi_idx[1]
        n_all = rr_all.size
        n_pick = min(n_all, min_samples_warn)
        if n_all > 0:
            pick = rng.choice(n_all, size=n_pick, replace=False)
            mask = np.zeros_like(roi_mask, dtype=bool)
            mask[rr_all[pick], cc_all[pick]] = True
            rule = "random_roi_fallback"
            random_fallback = True
        warnings.append("INK_SEPARATION_LOW_CONFIDENCE")

    meta = {
        "rule": rule,
        "n_pixels_used": int(mask.sum()),
        "dark_top_p": dark_p,
        "chroma_top_p": chroma_p,
        "random_fallback_used": bool(random_fallback),
        "rng_seed_used": rng_seed,
    }
    return mask, meta, warnings


def sample_ink_candidates(
    lab_map: np.ndarray,
    roi_mask: np.ndarray,
    cfg: Dict[str, Any],
    rng_seed: int | None = None,
    *,
    return_mask: bool = False,
) -> (
    Tuple[np.ndarray, np.ndarray, Dict[str, Any], List[str]]
    | Tuple[np.ndarray, np.ndarray, Dict[str, Any], List[str], np.ndarray]
):
    """
    Sample candidate pixels (likely ink) inside ROI.

    Returns (samples, sample_indices, meta, warnings).
    If return_mask=True, also returns the boolean sampling mask in polar space (T, R).
    """
    # Ensure CIE Lab for consistency
    lab_map = ensure_cie_lab(lab_map)

    mask, meta, warnings = build_sampling_mask(lab_map, roi_mask, cfg, rng_seed=rng_seed)

    # Calculate indices (flat) for mapping back to image
    sample_indices = np.flatnonzero(mask.reshape(-1)).astype(np.int64)

    if sample_indices.size == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        if return_mask:
            return empty, sample_indices, meta, warnings, mask
        return empty, sample_indices, meta, warnings

    # Extract samples using flat indices to ensure alignment
    samples = lab_map.reshape(-1, lab_map.shape[-1])[sample_indices].astype(np.float32, copy=False)

    if return_mask:
        return samples, sample_indices, meta, warnings, mask
    return samples, sample_indices, meta, warnings
