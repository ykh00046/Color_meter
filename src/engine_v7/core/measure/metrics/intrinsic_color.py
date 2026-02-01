from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _srgb_to_linear_gamma(srgb: np.ndarray, gamma: float) -> np.ndarray:
    """Legacy gamma-only transfer (kept for backward compatibility)."""
    arr = np.clip(srgb, 0.0, 255.0) / 255.0
    return np.power(arr, gamma)


def _linear_to_srgb_gamma(linear: np.ndarray, gamma: float) -> np.ndarray:
    """Legacy gamma-only inverse transfer (kept for backward compatibility)."""
    arr = np.clip(linear, 0.0, 1.0)
    return np.round(np.power(arr, 1.0 / gamma) * 255.0)


def _srgb_to_linear_srgb(srgb: np.ndarray) -> np.ndarray:
    """sRGB EOTF (recommended for consistency with color_simulator)."""
    arr = np.clip(srgb, 0.0, 255.0) / 255.0
    return np.where(arr <= 0.04045, arr / 12.92, np.power((arr + 0.055) / 1.055, 2.4))


def _linear_to_srgb_srgb(linear: np.ndarray) -> np.ndarray:
    """Inverse sRGB EOTF."""
    arr = np.clip(linear, 0.0, 1.0)
    srgb01 = np.where(
        arr <= 0.0031308,
        arr * 12.92,
        1.055 * np.power(np.maximum(arr, 0.0), 1.0 / 2.4) - 0.055,
    )
    return np.round(np.clip(srgb01, 0.0, 1.0) * 255.0)


def _get_transfer_fns(transfer: str, gamma: float):
    transfer = str(transfer or "gamma").lower()
    if transfer in ("srgb", "srgb_eotf", "eotf"):
        return (
            lambda x: _srgb_to_linear_srgb(x),
            lambda x: _linear_to_srgb_srgb(x),
        )
    # default: legacy gamma
    return (
        lambda x: _srgb_to_linear_gamma(x, gamma),
        lambda x: _linear_to_srgb_gamma(x, gamma),
    )


def _bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return bgr[..., ::-1]


def _masked_rgb_stat(
    polar_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    stat: str = "mean",
    trim_frac: float = 0.0,
    exclude_saturated: bool = True,
) -> np.ndarray | None:
    """Compute masked RGB statistic for robustness against highlights/speckles.

    When exclude_saturated=True, pixels with any channel at 0 or 255
    (sensor saturation) are removed before computing the statistic.
    """
    if mask is None or mask.sum() == 0:
        return None
    rgb_vals = polar_bgr[mask][:, ::-1].astype(np.float32)  # (N,3) RGB

    # Exclude fully saturated / fully black pixels (sensor clipping)
    if exclude_saturated and rgb_vals.shape[0] > 0:
        not_saturated = ~np.any((rgb_vals <= 0.5) | (rgb_vals >= 254.5), axis=1)
        if not_saturated.sum() >= max(10, rgb_vals.shape[0] // 10):
            rgb_vals = rgb_vals[not_saturated]
        # else: too few non-saturated → keep all (trim_frac will still help)

    stat = str(stat or "mean").lower()

    if stat == "median":
        return np.median(rgb_vals, axis=0)

    if stat in ("trimmed_mean", "trimmean", "trim"):
        f = float(np.clip(trim_frac, 0.0, 0.49))
        if f <= 0.0:
            return rgb_vals.mean(axis=0)
        lo = np.quantile(rgb_vals, f, axis=0)
        hi = np.quantile(rgb_vals, 1.0 - f, axis=0)
        keep = np.all((rgb_vals >= lo) & (rgb_vals <= hi), axis=1)
        if keep.sum() == 0:
            return rgb_vals.mean(axis=0)
        return rgb_vals[keep].mean(axis=0)

    # default: mean
    return rgb_vals.mean(axis=0)


def _mean_rgb(polar_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Backward-compatible alias (mean)."""
    return _masked_rgb_stat(polar_bgr, mask, stat="mean", trim_frac=0.0)


def _estimate_bg_rgb(
    polar_bgr: np.ndarray,
    outer_frac: float,
    *,
    stat: str = "median",
    pct_lo: float = 0.0,
    pct_hi: float = 100.0,
) -> np.ndarray:
    """Estimate background RGB from outer radial band.

    Args:
        outer_frac: start fraction of radius to sample (e.g. 0.95)
        stat: "median" | "mean" | "trimmed_mean"
        pct_lo/pct_hi: percentile range selection within the outer band, applied per-channel.
    """
    r_dim = polar_bgr.shape[1]
    outer_start = int(r_dim * float(outer_frac))
    outer = polar_bgr[:, outer_start:, :].reshape(-1, 3).astype(np.float32)

    lo = float(np.clip(pct_lo, 0.0, 100.0))
    hi = float(np.clip(pct_hi, 0.0, 100.0))
    if hi < lo:
        lo, hi = hi, lo
    if not (lo == 0.0 and hi == 100.0):
        qlo = np.percentile(outer, lo, axis=0)
        qhi = np.percentile(outer, hi, axis=0)
        keep = np.all((outer >= qlo) & (outer <= qhi), axis=1)
        if keep.sum() > 0:
            outer = outer[keep]

    stat = str(stat or "median").lower()
    if stat == "mean":
        bgr = outer.mean(axis=0)
    elif stat in ("trimmed_mean", "trimmean", "trim"):
        bgr = outer.mean(axis=0)
    else:
        bgr = np.median(outer, axis=0)

    return _bgr_to_rgb(bgr)


def compute_intrinsic_colors(
    polar_white_bgr: np.ndarray,
    polar_black_bgr: np.ndarray,
    color_masks: Dict[str, np.ndarray],
    cfg: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    cfg = cfg or {}
    intrinsic_cfg = cfg.get("intrinsic_color", {}) if isinstance(cfg, dict) else {}
    mode = str(intrinsic_cfg.get("mode", "DYNAMIC")).upper()
    ref_white = intrinsic_cfg.get("ref_white_srgb")
    ref_black = intrinsic_cfg.get("ref_black_srgb")
    gamma = float(intrinsic_cfg.get("gamma", 2.2))
    outer_frac = float(intrinsic_cfg.get("outer_bg_frac", 0.95))
    transfer = str(intrinsic_cfg.get("transfer", "srgb_eotf"))
    obs_stat = str(intrinsic_cfg.get("obs_stat", "trimmed_mean"))
    obs_trim_frac = float(intrinsic_cfg.get("obs_trim_frac", 0.15))
    bg_stat = str(intrinsic_cfg.get("bg_stat", "median"))
    bg_pct_lo = float(intrinsic_cfg.get("bg_pct_lo", 0.0))
    bg_pct_hi = float(intrinsic_cfg.get("bg_pct_hi", 100.0))
    verbose_debug = bool(intrinsic_cfg.get("verbose_debug", False))
    srgb_to_lin, lin_to_srgb = _get_transfer_fns(transfer, gamma)
    eps = float(intrinsic_cfg.get("eps", 1e-6))
    min_white_brightness = intrinsic_cfg.get("min_white_brightness")
    min_bright_keep_ratio = float(intrinsic_cfg.get("min_bright_keep_ratio", 0.1))
    prefer_brightest_mask = bool(intrinsic_cfg.get("prefer_brightest_mask", False))
    white_bg_margin = float(intrinsic_cfg.get("white_bg_margin", 10.0))
    debug_dir = intrinsic_cfg.get("debug_dir")
    debug_path = None
    if debug_dir:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    if mode == "FIXED" and ref_white is not None and ref_black is not None:
        bg_white_rgb = np.array(ref_white, dtype=np.float32)
        bg_black_rgb = np.array(ref_black, dtype=np.float32)
        calibrated = True
    else:
        if mode == "FIXED":
            warnings.append("intrinsic_color_calibration_missing_fallback")
        bg_white_rgb = _estimate_bg_rgb(polar_white_bgr, outer_frac, stat=bg_stat, pct_lo=bg_pct_lo, pct_hi=bg_pct_hi)
        bg_black_rgb = _estimate_bg_rgb(polar_black_bgr, outer_frac, stat=bg_stat, pct_lo=bg_pct_lo, pct_hi=bg_pct_hi)
        calibrated = False

    bg_white_lin = srgb_to_lin(bg_white_rgb)
    bg_black_lin = srgb_to_lin(bg_black_rgb)
    diff_bg = np.maximum(bg_white_lin - bg_black_lin, eps)

    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    results: Dict[str, Dict[str, Any]] = {}

    warn_k_high = float(intrinsic_cfg.get("warn_k_high", 3.0))
    warn_k_low = float(intrinsic_cfg.get("warn_k_low", 0.0))
    warn_t_high = float(intrinsic_cfg.get("warn_t_high", 0.999))

    mask_brightness: Dict[str, float] = {}
    for color_id, mask in color_masks.items():
        obs_w_rgb = _masked_rgb_stat(polar_white_bgr, mask, stat=obs_stat, trim_frac=obs_trim_frac)
        if obs_w_rgb is None:
            continue
        mask_brightness[color_id] = float(np.mean(obs_w_rgb))

    preferred_color_id = None
    if mask_brightness:
        sorted_by_brightness = sorted(mask_brightness.items(), key=lambda x: x[1], reverse=True)
        ref_brightness = None
        if ref_white is not None:
            ref_brightness = float(np.mean(np.array(ref_white, dtype=np.float32)))
        for idx, (cid, brightness) in enumerate(sorted_by_brightness):
            if min_white_brightness is not None and brightness < float(min_white_brightness):
                continue
            if ref_brightness is not None and abs(ref_brightness - brightness) < white_bg_margin:
                # Skip near-white background if a viable next candidate exists.
                if idx + 1 < len(sorted_by_brightness):
                    next_brightness = sorted_by_brightness[idx + 1][1]
                    if next_brightness >= float(min_white_brightness):
                        continue
            preferred_color_id = cid
            break
        if preferred_color_id is None:
            preferred_color_id = sorted_by_brightness[0][0]

    white_brightness_map = None
    if min_white_brightness is not None:
        white_brightness_map = polar_white_bgr.astype(np.float32).mean(axis=2)

    for color_id, mask in color_masks.items():
        mask_for_intrinsic = mask
        if (
            prefer_brightest_mask
            and preferred_color_id is not None
            and min_white_brightness is not None
            and mask_brightness.get(color_id, 0.0) < float(min_white_brightness)
        ):
            mask_for_intrinsic = color_masks[preferred_color_id]
        if min_white_brightness is not None:
            mwb = float(min_white_brightness)
            bright = (
                white_brightness_map >= mwb
                if white_brightness_map is not None
                else (polar_white_bgr.astype(np.float32).mean(axis=2) >= mwb)
            )
            filtered = mask & bright
            min_keep = max(10, int(mask.sum() * min_bright_keep_ratio))
            if filtered.sum() >= min_keep:
                mask_for_intrinsic = filtered
            else:
                # Stepwise fallback: relax threshold to 70%, then 50%
                applied_relaxed = False
                for relax_factor, relax_label in [(0.7, "RELAXED_70pct"), (0.5, "RELAXED_50pct")]:
                    bright_relaxed = (
                        white_brightness_map >= mwb * relax_factor
                        if white_brightness_map is not None
                        else (polar_white_bgr.astype(np.float32).mean(axis=2) >= mwb * relax_factor)
                    )
                    filtered_relaxed = mask & bright_relaxed
                    if filtered_relaxed.sum() >= min_keep:
                        mask_for_intrinsic = filtered_relaxed
                        warnings.append(f"INTRINSIC_BRIGHTNESS_FILTER_{relax_label}")
                        applied_relaxed = True
                        break
                if not applied_relaxed:
                    warnings.append("INTRINSIC_BRIGHTNESS_FILTER_EMPTY_FALLBACK")

        obs_w_rgb = _masked_rgb_stat(polar_white_bgr, mask_for_intrinsic, stat=obs_stat, trim_frac=obs_trim_frac)
        obs_b_rgb = _masked_rgb_stat(polar_black_bgr, mask_for_intrinsic, stat=obs_stat, trim_frac=obs_trim_frac)
        if obs_w_rgb is None or obs_b_rgb is None:
            continue
        if debug_path is not None:
            safe_id = str(color_id).replace("/", "_").replace("\\", "_")
            mask_u8 = mask_for_intrinsic.astype(np.uint8) * 255
            white_masked = cv2.bitwise_and(polar_white_bgr, polar_white_bgr, mask=mask_u8)
            black_masked = cv2.bitwise_and(polar_black_bgr, polar_black_bgr, mask=mask_u8)
            cv2.imwrite(str(debug_path / f"{safe_id}_white_masked.png"), white_masked)
            cv2.imwrite(str(debug_path / f"{safe_id}_black_masked.png"), black_masked)

        obs_w_lin = srgb_to_lin(obs_w_rgb)
        obs_b_lin = srgb_to_lin(obs_b_rgb)
        if verbose_debug:
            logger.debug(
                "intrinsic_color obs_w_rgb max=%.3f, bg_white_rgb max=%.3f, "
                "obs_w_lin max=%.6f, bg_white_lin max=%.6f",
                np.max(obs_w_rgb),
                np.max(bg_white_rgb),
                np.max(obs_w_lin),
                np.max(bg_white_lin),
            )
        diff_obs = obs_w_lin - obs_b_lin

        t_raw = diff_obs / diff_bg
        if np.any(diff_obs <= 0) or np.any(t_raw <= eps):
            # Fallback when black observation is brighter than white or equal.
            t_rgb = obs_w_lin / np.maximum(bg_white_lin, eps)
            color_warnings = ["FALLBACK_WHITE_ONLY_MODEL"]
        else:
            t_rgb = t_raw
            color_warnings = []

        t_rgb = np.clip(t_rgb, 0.0, 1.0)
        t_luma = float(np.sum(t_rgb * weights))
        alpha_y = float(np.clip(1.0 - t_luma, 0.01, 1.0))

        ink_lin = (obs_b_lin - (1.0 - alpha_y) * bg_black_lin) / max(alpha_y, eps)
        ink_lin = np.clip(ink_lin, 0.0, 1.0)
        ink_rgb = lin_to_srgb(ink_lin).astype(int).tolist()

        t_rgb_safe = np.clip(t_rgb, eps, 1.0)
        k_rgb = (-np.log(t_rgb_safe)).tolist()
        if np.any(np.array(k_rgb) > warn_k_high):
            color_warnings.append("K_VALUE_TOO_HIGH")
        if np.any(np.array(k_rgb) <= warn_k_low):
            color_warnings.append("K_VALUE_INVALID")
        if calibrated and mode == "FIXED" and np.any(t_rgb_safe >= warn_t_high):
            color_warnings.append("POSSIBLE_REFLECTION_OR_MISMATCH")
        if color_warnings:
            warnings.extend(color_warnings)
        if mask_for_intrinsic is not mask:
            color_warnings.append("INTRINSIC_BRIGHTNESS_FILTER_APPLIED")
            warnings.append("INTRINSIC_BRIGHTNESS_FILTER_APPLIED")
        if prefer_brightest_mask and mask_for_intrinsic is not mask:
            color_warnings.append("INTRINSIC_BRIGHTEST_MASK_OVERRIDE")
            warnings.append("INTRINSIC_BRIGHTEST_MASK_OVERRIDE")

        results[color_id] = {
            "alpha_y": round(alpha_y, 4),
            "ink_rgb": ink_rgb,
            "k_rgb": [round(float(k), 6) for k in k_rgb],
            "base_t": [round(float(t), 6) for t in t_rgb_safe.tolist()],
            "obs_white_rgb": np.round(obs_w_rgb, 2).tolist(),
            "obs_black_rgb": np.round(obs_b_rgb, 2).tolist(),
            "warnings": color_warnings,
        }

    meta = {
        "bg_white_rgb": np.round(bg_white_rgb, 2).tolist(),
        "bg_black_rgb": np.round(bg_black_rgb, 2).tolist(),
        "calibrated": calibrated,
        "mode": mode,
        "ref_white_srgb": ref_white,
        "ref_black_srgb": ref_black,
        "gamma": gamma,
        "outer_bg_frac": outer_frac,
        "transfer": transfer,
        "obs_stat": obs_stat,
        "obs_trim_frac": obs_trim_frac,
        "bg_stat": bg_stat,
        "bg_pct_lo": bg_pct_lo,
        "bg_pct_hi": bg_pct_hi,
        "warnings": warnings,
    }
    return results, meta


def simulate_linear(
    ink_rgb: np.ndarray,
    alpha_y: float,
    bg_rgb: np.ndarray,
    *,
    gamma: float = 2.2,
    thickness: float = 1.0,
    transfer: str = "gamma",
) -> np.ndarray:
    alpha_y = float(np.clip(alpha_y, 0.0, 1.0))
    adj_alpha = 1.0 - (1.0 - alpha_y) ** max(float(thickness), 0.0)
    srgb_to_lin, lin_to_srgb = _get_transfer_fns(transfer, gamma)
    ink_lin = srgb_to_lin(ink_rgb)
    bg_lin = srgb_to_lin(bg_rgb)
    pred_lin = adj_alpha * ink_lin + (1.0 - adj_alpha) * bg_lin
    return lin_to_srgb(pred_lin).astype(int)


def simulate_physical(
    k_rgb: np.ndarray,
    bg_rgb: np.ndarray,
    *,
    gamma: float = 2.2,
    thickness: float = 1.0,
    transfer: str = "gamma",
) -> np.ndarray:
    k_arr = np.clip(np.array(k_rgb, dtype=np.float32), 0.0, None)
    t_new = np.exp(-k_arr * max(float(thickness), 0.0))
    srgb_to_lin, lin_to_srgb = _get_transfer_fns(transfer, gamma)
    bg_lin = srgb_to_lin(bg_rgb)
    pred_lin = bg_lin * t_new
    return lin_to_srgb(pred_lin).astype(int)
