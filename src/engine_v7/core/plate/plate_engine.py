from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Savitzky-Golay filter for edge-preserving smoothing
try:
    from scipy.signal import savgol_filter

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..config_norm import get_plate_cfg, get_polar_dims
from ..geometry.lens_geometry import detect_lens_circle
from ..measure.segmentation.color_masks import lab_to_hex
from ..signature.radial_signature import to_polar
from ..types import LensGeometry
from ..utils import to_cie_lab

# Import shared helpers (aliased for backward compatibility)
from ._helpers import mask_to_polar as _mask_to_polar
from ._helpers import mean_gray_center as _mean_gray_center
from ._helpers import mean_gray_outer as _mean_gray_outer
from ._helpers import median_bgr_center as _median_bgr_center
from ._helpers import radial_mask as _radial_mask
from ._helpers import resize_to_square_centered as _resize_to_square_centered

# ==============================================================================
# Phase 2: Dynamic Radius Detection (V6 Adaptive Engine)
# ==============================================================================


@dataclass
class DynamicRadiiResult:
    """Result of dynamic radius detection."""

    r_clear: float  # Clear/Dot boundary (default 0.40)
    r_ring0: float  # Dot/Ring boundary (default 0.70)
    confidence: float  # Detection confidence (0-1)
    method: str  # Detection method used
    meta: Dict[str, Any]  # Detailed metadata


def detect_dynamic_radii(
    alpha_map: np.ndarray,
    geom: LensGeometry,
    *,
    n_bins: int = 100,
    savgol_window: int = 11,
    savgol_order: int = 3,
    r_clear_range: Tuple[float, float] = (0.20, 0.60),
    r_ring0_range: Tuple[float, float] = (0.55, 0.85),
    min_confidence: float = 0.3,
    fallback_r_clear: float = 0.40,
    fallback_r_ring0: float = 0.70,
) -> DynamicRadiiResult:
    """
    Detect optimal radii boundaries from alpha map gradient analysis.

    The clear/dot boundary (r_clear) is detected by finding where the alpha
    gradient is maximum - this indicates the transition from transparent
    center to the printed dot region.

    Args:
        alpha_map: Alpha (ink density) map from _compute_alpha_map
        geom: Lens geometry
        n_bins: Number of radial bins for profile
        savgol_window: Savitzky-Golay filter window size (must be odd)
        savgol_order: Savitzky-Golay polynomial order
        r_clear_range: Valid range for r_clear detection
        r_ring0_range: Valid range for r_ring0 detection
        min_confidence: Minimum confidence to accept detection
        fallback_r_clear: Fallback r_clear if detection fails
        fallback_r_ring0: Fallback r_ring0 if detection fails

    Returns:
        DynamicRadiiResult with detected radii and metadata
    """
    h, w = alpha_map.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((xx - geom.cx) ** 2 + (yy - geom.cy) ** 2) / max(float(geom.r), 1e-6)

    # Compute radial profile (mean alpha per radial bin)
    r_edges = np.linspace(0, 1.0, n_bins + 1)
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2
    radial_profile = np.zeros(n_bins, dtype=np.float32)
    radial_counts = np.zeros(n_bins, dtype=np.int32)

    for i in range(n_bins):
        mask = (rr >= r_edges[i]) & (rr < r_edges[i + 1])
        if np.any(mask):
            radial_profile[i] = float(np.mean(alpha_map[mask]))
            radial_counts[i] = int(np.sum(mask))

    # Handle empty bins by interpolation
    valid = radial_counts > 0
    if not np.any(valid):
        return DynamicRadiiResult(
            r_clear=fallback_r_clear,
            r_ring0=fallback_r_ring0,
            confidence=0.0,
            method="fallback_empty",
            meta={"reason": "no_valid_bins"},
        )

    # Apply Savitzky-Golay smoothing
    if HAS_SCIPY and np.sum(valid) >= savgol_window:
        try:
            profile_smooth = savgol_filter(radial_profile, savgol_window, savgol_order)
        except Exception:
            profile_smooth = radial_profile.copy()
    else:
        # Simple moving average fallback
        kernel_size = min(5, len(radial_profile) // 2)
        if kernel_size >= 3:
            kernel = np.ones(kernel_size) / kernel_size
            profile_smooth = np.convolve(radial_profile, kernel, mode="same")
        else:
            profile_smooth = radial_profile.copy()

    # Compute gradient
    gradient = np.gradient(profile_smooth)

    # Find r_clear: maximum positive gradient in valid range
    r_clear_detected, r_clear_conf = _find_gradient_peak(r_centers, gradient, r_clear_range, positive=True)

    # Find r_ring0: look for second transition or use heuristic
    # The ring region typically starts where gradient stabilizes after dot region
    r_ring0_detected, r_ring0_conf = _find_gradient_peak(r_centers, gradient, r_ring0_range, positive=False)

    # If r_ring0 detection is weak, use heuristic: r_ring0 ≈ r_clear + 0.30
    if r_ring0_conf < min_confidence:
        r_ring0_detected = min(r_clear_detected + 0.30, r_ring0_range[1])
        r_ring0_conf = r_clear_conf * 0.5  # Inherited confidence

    # Validate and apply fallback
    method = "gradient_savgol"
    if r_clear_conf < min_confidence:
        r_clear_final = fallback_r_clear
        method = "fallback_low_confidence"
    elif r_clear_detected < r_clear_range[0] or r_clear_detected > r_clear_range[1]:
        r_clear_final = fallback_r_clear
        method = "fallback_out_of_range"
    else:
        r_clear_final = r_clear_detected

    if r_ring0_detected < r_clear_final + 0.10:
        r_ring0_final = r_clear_final + 0.30  # Ensure minimum gap
    else:
        r_ring0_final = r_ring0_detected

    # Overall confidence
    confidence = (r_clear_conf + r_ring0_conf) / 2.0

    meta = {
        "r_clear_detected": float(r_clear_detected),
        "r_clear_confidence": float(r_clear_conf),
        "r_ring0_detected": float(r_ring0_detected),
        "r_ring0_confidence": float(r_ring0_conf),
        "profile_max": float(np.max(radial_profile)),
        "profile_min": float(np.min(radial_profile)),
        "gradient_max": float(np.max(gradient)),
        "gradient_min": float(np.min(gradient)),
        "n_valid_bins": int(np.sum(valid)),
        "savgol_used": HAS_SCIPY and np.sum(valid) >= savgol_window,
    }

    return DynamicRadiiResult(
        r_clear=float(r_clear_final),
        r_ring0=float(r_ring0_final),
        confidence=float(confidence),
        method=method,
        meta=meta,
    )


def _find_gradient_peak(
    r_centers: np.ndarray,
    gradient: np.ndarray,
    r_range: Tuple[float, float],
    positive: bool = True,
) -> Tuple[float, float]:
    """
    Find the radius where gradient peaks within the given range.

    Args:
        r_centers: Radial bin centers
        gradient: Gradient values
        r_range: (min_r, max_r) search range
        positive: If True, find max gradient; if False, find min gradient

    Returns:
        (detected_radius, confidence)
    """
    # Mask to valid range
    range_mask = (r_centers >= r_range[0]) & (r_centers <= r_range[1])
    if not np.any(range_mask):
        return (r_range[0] + r_range[1]) / 2, 0.0

    r_valid = r_centers[range_mask]
    g_valid = gradient[range_mask]

    if positive:
        peak_idx = np.argmax(g_valid)
        peak_val = g_valid[peak_idx]
        # Confidence based on peak prominence
        g_range = np.max(g_valid) - np.min(g_valid)
        confidence = min(1.0, abs(peak_val) / (g_range + 1e-6))
    else:
        peak_idx = np.argmin(g_valid)
        peak_val = g_valid[peak_idx]
        g_range = np.max(g_valid) - np.min(g_valid)
        confidence = min(1.0, abs(peak_val) / (g_range + 1e-6))

    return float(r_valid[peak_idx]), float(confidence)


# NOTE: _resize_to_square_centered, _mean_gray_center, _mean_gray_outer,
# _median_bgr_center are now imported from ._helpers (see imports above)


def _make_pupil_mask(gray: np.ndarray, rr: np.ndarray, r_clear: float) -> np.ndarray:
    core = rr <= float(r_clear) * 0.90
    if not np.any(core):
        return rr <= float(r_clear)
    g = gray.copy()
    g[~core] = 255
    th = float(np.percentile(g[core], 20))
    binary = (g <= th).astype(np.uint8) * 255
    binary = cv2.medianBlur(binary, 5)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return core
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _ensure_pair_order(
    w_norm: np.ndarray,
    b_norm: np.ndarray,
    geom: LensGeometry,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    r_clear = float(cfg.get("r_clear", 0.40))
    w_c = _mean_gray_center(w_norm, geom, r_clear)
    b_c = _mean_gray_center(b_norm, geom, r_clear)
    r_outer = float(cfg.get("r_outer_bg", 0.98))
    w_bg = _mean_gray_outer(w_norm, geom, r_outer)
    b_bg = _mean_gray_outer(b_norm, geom, r_outer)
    min_delta = float(cfg.get("wb_outer_min_delta", 5.0))
    action = str(cfg.get("wb_outer_action", "swap")).lower()
    swapped = False
    if w_c < b_c:
        w_norm, b_norm = b_norm, w_norm
        swapped = True
    swapped_outer = False
    if w_bg + min_delta < b_bg:
        if action == "swap":
            w_norm, b_norm = b_norm, w_norm
            swapped_outer = True
        elif action == "error":
            raise ValueError("Outer background brightness suggests white/black inputs are inverted.")
    meta = {
        "swapped": swapped,
        "swapped_outer": swapped_outer,
        "w_center_med": w_c,
        "b_center_med": b_c,
        "w_outer_med": w_bg,
        "b_outer_med": b_bg,
        "r_clear": r_clear,
        "r_outer_bg": r_outer,
        "wb_outer_min_delta": min_delta,
        "wb_outer_action": action,
    }
    return w_norm, b_norm, meta


def _reg_feature(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.2)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _phase_align_polar(
    white: np.ndarray,
    black: np.ndarray,
    geom: LensGeometry,
    T: int = 720,
    min_score: float = 0.15,
    max_dy_px: float = 4.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    w_feat = _reg_feature(white)
    b_feat = _reg_feature(black)

    R = int(geom.r)
    center = (geom.cx, geom.cy)
    w_p = cv2.warpPolar(w_feat, (T, R), center, geom.r, cv2.WARP_POLAR_LINEAR)
    b_p = cv2.warpPolar(b_feat, (T, R), center, geom.r, cv2.WARP_POLAR_LINEAR)

    r0 = int(R * 0.45)
    r1 = int(R * 0.95)
    w_band = w_p[r0:r1, :]
    b_band = b_p[r0:r1, :]

    w_prof = w_band.sum(axis=0).astype(np.float32)
    b_prof = b_band.sum(axis=0).astype(np.float32)
    win = np.hanning(len(w_prof)).astype(np.float32)
    w_prof = (w_prof - w_prof.mean()) * win
    b_prof = (b_prof - b_prof.mean()) * win

    fw = np.fft.rfft(w_prof)
    fb = np.fft.rfft(b_prof)
    corr = np.fft.irfft(fw * np.conj(fb))
    idx = int(np.argmax(corr))
    if idx > len(w_prof) // 2:
        idx -= len(w_prof)
    dtheta_px = float(idx)
    dtheta_deg = (dtheta_px / float(T)) * 360.0
    norm = float(np.linalg.norm(w_prof) * np.linalg.norm(b_prof) + 1e-6)
    response = float(corr.max() / norm)
    dy = 0.0
    low_confidence = response < float(min_score)
    note = ""
    if low_confidence:
        dtheta_deg = 0.0
        dtheta_px = 0.0
        dy = 0.0
        black_aligned = black
        note = "registration_skipped"
    else:
        M = cv2.getRotationMatrix2D(center, -dtheta_deg, 1.0)
        black_aligned = cv2.warpAffine(
            black,
            M,
            (black.shape[1], black.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    info = {
        "method": "circle_norm+polar_angular_xcorr",
        "score": float(max(0.0, min(1.0, response))),
        "dx_dy": [float(dtheta_px), float(dy)],
        "dtheta_deg": float(dtheta_deg),
        "notes": note,
        "low_confidence": low_confidence,
        "feature": "gradmag_angular",
        "band_r": [0.45, 0.95],
        "max_dy_px": float(max_dy_px),
    }
    return black_aligned, info


def _refine_alignment_ecc(
    template_bgr: np.ndarray,
    input_bgr: np.ndarray,
    cfg: Dict[str, Any],
    mask: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[float], str]:
    warp_mode_name = str(cfg.get("registration_ecc_warp", "euclidean")).lower()
    warp_mode = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
    }.get(warp_mode_name, cv2.MOTION_EUCLIDEAN)
    use_grad = bool(cfg.get("registration_ecc_use_grad", True))
    max_iter = int(cfg.get("registration_ecc_max_iter", 50))
    eps = float(cfg.get("registration_ecc_eps", 1e-5))
    min_cc = float(cfg.get("registration_ecc_min_cc", 0.1))

    if use_grad:
        temp = _reg_feature(template_bgr)
        src = _reg_feature(input_bgr)
    else:
        temp = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        src = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    temp = cv2.normalize(temp, None, 0.0, 1.0, cv2.NORM_MINMAX)
    src = cv2.normalize(src, None, 0.0, 1.0, cv2.NORM_MINMAX)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)

    try:
        cc, warp_matrix = cv2.findTransformECC(temp, src, warp_matrix, warp_mode, criteria, mask)
    except cv2.error:
        return None, None, "ecc_failed"

    if cc < min_cc:
        return None, float(cc), "ecc_low_cc"

    return warp_matrix, float(cc), ""


def _align_features_orb(
    template_bgr: np.ndarray,
    target_bgr: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], float, str]:
    max_features = int(cfg.get("registration_orb_max_features", 2000))
    edge_threshold = int(cfg.get("registration_orb_edge_threshold", 15))
    patch_size = int(cfg.get("registration_orb_patch_size", 31))
    fast_threshold = int(cfg.get("registration_orb_fast_threshold", 10))
    min_inlier = float(cfg.get("registration_orb_min_inlier_ratio", 0.2))
    orb = cv2.ORB_create(
        nfeatures=max_features,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=edge_threshold,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=patch_size,
        fastThreshold=fast_threshold,
    )
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(target_gray, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, 0.0, "orb_no_features"
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if not matches:
        return None, 0.0, "orb_no_matches"
    matches = sorted(matches, key=lambda x: x.distance)
    keep = max(4, int(len(matches) * 0.5))
    good = matches[:keep]
    if len(good) < 4:
        return None, 0.0, "orb_few_matches"
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
    if M is None or inliers is None:
        return None, 0.0, "orb_no_transform"
    score = float((np.sum(inliers) / len(good)) * min(1.0, len(good) / 50.0))
    if score < min_inlier:
        return None, score, "orb_low_inlier"
    h, w = template_bgr.shape[:2]
    aligned = cv2.warpAffine(target_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return aligned, score, ""


def _compute_alpha_map(
    white_bgr: np.ndarray,
    black_bgr: np.ndarray,
    geom: LensGeometry,
    r_bg: float = 0.35,
    r_clear: Optional[float] = None,
    diff_min: float = 1.0,
    denom_min: float = 20.0,
    spatial_bg_enabled: bool = False,
    spatial_bg_r_outer: float = 0.98,
    spatial_bg_stride: int = 4,
    alpha_clip: Tuple[float, float] = (0.02, 0.98),
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:
    w_f = white_bgr.astype(np.float32)
    b_f = black_bgr.astype(np.float32)

    h, w = w_f.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((xx - geom.cx) ** 2 + (yy - geom.cy) ** 2) / max(float(geom.r), 1e-6)

    r_clear_val = float(r_clear) if r_clear is not None else 0.40
    bg_mask = (rr < r_clear_val * 0.9) | (rr > 0.98)

    diff_raw = w_f - b_f
    raw_denom = np.linalg.norm(diff_raw, axis=2)

    if np.any(bg_mask):
        safe_denom_val = float(np.percentile(raw_denom[bg_mask], 65))
    else:
        safe_denom_val = float(np.percentile(raw_denom, 65))

    final_denom_val = max(float(safe_denom_val), float(denom_min))

    alpha_ch = 1.0 - (diff_raw / final_denom_val)
    alpha_raw = np.median(alpha_ch, axis=2)

    zero_ref_mask = rr < 0.25
    bias = 0.0
    if np.any(zero_ref_mask):
        bias = float(np.median(alpha_raw[zero_ref_mask]))
        if bias > 0:
            alpha_raw = alpha_raw - (bias * 0.95)

    alpha = np.nan_to_num(alpha_raw, nan=0.0, posinf=1.0, neginf=0.0)
    alpha = np.clip(alpha, alpha_clip[0], alpha_clip[1])
    alpha = cv2.medianBlur((alpha * 255.0).astype(np.uint8), 3).astype(np.float32) / 255.0

    ink = (white_bgr * alpha[..., None]).astype(np.uint8)

    meta = {
        "mode": "white_black_pair_v5_1_p65_autozero",
        "safe_denom_val": float(final_denom_val),
        "zero_bias_correction": float(bias),
        "alpha_clip": [float(alpha_clip[0]), float(alpha_clip[1])],
        "residual": {
            "white_recon_rmse": 0.0,
            "black_recon_rmse": 0.0,
            "mask_artifact_ratio": float(np.mean((alpha <= alpha_clip[0] + 1e-6) | (alpha >= alpha_clip[1] - 1e-6))),
            "artifact_nan_ratio": float(np.mean(~np.isfinite(alpha))),
        },
    }
    invalid_mask = raw_denom < (final_denom_val * 0.1)
    return alpha, ink, meta, invalid_mask


def _compute_alpha_map_lite(
    white_bgr: np.ndarray,
    black_bgr: np.ndarray,
    *,
    blur_ksize: int = 5,
    backlight: float = 255.0,
    alpha_clip: Tuple[float, float] = (0.02, 0.98),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    warning_list: List[str] = []

    ksize = int(blur_ksize)
    if ksize > 1:
        if ksize % 2 == 0:
            ksize += 1
        ksize = max(3, ksize)
        w_blur = cv2.GaussianBlur(white_bgr, (ksize, ksize), 0)
        b_blur = cv2.GaussianBlur(black_bgr, (ksize, ksize), 0)
    else:
        w_blur = white_bgr
        b_blur = black_bgr

    w_mean = w_blur.mean(axis=2).astype(np.float32)
    b_mean = b_blur.mean(axis=2).astype(np.float32)

    # White/Black swap detection
    w_global = float(w_mean.mean())
    b_global = float(b_mean.mean())
    if w_global < b_global:
        warning_list.append("possible_white_black_swap")

    denom = float(backlight) if float(backlight) > 1e-6 else 255.0
    alpha_map = 1.0 - ((w_mean - b_mean) / denom)
    alpha_map = np.clip(alpha_map, alpha_clip[0], alpha_clip[1]).astype(np.float32)

    alpha_mean_val = float(alpha_map.mean())

    # Alpha outlier detection
    if alpha_mean_val < 0.1:
        warning_list.append("alpha_mean_too_low")
    elif alpha_mean_val > 0.9:
        warning_list.append("alpha_mean_too_high")

    meta = {
        "method": "plate_lite_mean_diff",
        "blur_ksize": int(ksize),
        "backlight": float(denom),
        "alpha_clip": [float(alpha_clip[0]), float(alpha_clip[1])],
        "alpha_mean": alpha_mean_val,
        "alpha_std": float(alpha_map.std()),
        "warnings": warning_list,
    }
    return alpha_map, meta


def _resolve_paper_color(
    lite_cfg: Dict[str, Any],
    clear_mask: Optional[np.ndarray],
    lab: Optional[np.ndarray],
) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
    warning_list: List[str] = []
    paper_cfg = lite_cfg.get("paper_color", {}) if isinstance(lite_cfg, dict) else {}
    source = str(paper_cfg.get("source", "static")).lower()
    default_lab = np.array(paper_cfg.get("lab", [95.0, 0.0, 0.0]), dtype=np.float32).reshape(3)

    if source == "auto":
        if clear_mask is not None and lab is not None and np.any(clear_mask):
            paper_lab = lab[clear_mask].mean(axis=0).astype(np.float32)
        else:
            warning_list.append("paper_color_auto_fallback_static")
            paper_lab = default_lab
            source = "static"
    elif source == "calibration":
        cal_lab = paper_cfg.get("lab")
        if cal_lab is None:
            warning_list.append("paper_color_calibration_missing_fallback_static")
            paper_lab = default_lab
            source = "static"
        else:
            paper_lab = np.array(cal_lab, dtype=np.float32).reshape(3)
    else:
        paper_lab = default_lab

    paper_meta = {"lab": paper_lab.tolist(), "source": source}
    return paper_lab, paper_meta, warning_list


# NOTE: _radial_mask is now imported from ._helpers (see imports above)


def _get_dynamic_threshold(
    alpha_map: np.ndarray,
    mask_roi: Optional[np.ndarray] = None,
    min_th: float = 0.1,
    max_th: float = 0.6,
    fallback: float = 0.25,
) -> float:
    src = (np.clip(alpha_map, 0.0, 1.0) * 255.0).astype(np.uint8)
    pixels = src[mask_roi] if mask_roi is not None else src.reshape(-1)
    if pixels.size == 0:
        return float(fallback)
    hist, bin_edges = np.histogram(pixels, bins=256, range=(0, 256))
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    valid = (weight1 > 0) & (weight2 > 0)
    if not np.any(valid):
        return float(fallback)
    mean1 = np.cumsum(hist * bin_mids) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / np.maximum(weight2[::-1], 1))[::-1]
    inter = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = int(np.argmax(inter))
    otsu_th = float(bin_mids[:-1][idx] / 255.0)
    return float(np.clip(otsu_th, min_th, max_th))


def _estimate_spatial_bg(img: np.ndarray, mask: np.ndarray, stride: int = 4) -> np.ndarray:
    h, w = img.shape
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    y_sample = y_grid[mask][::stride]
    x_sample = x_grid[mask][::stride]
    z_sample = img[mask][::stride]
    if z_sample.size < 100:
        return np.full_like(img, np.median(img), dtype=np.float32)
    X = x_sample.flatten().astype(np.float32)
    Y = y_sample.flatten().astype(np.float32)
    A = np.c_[np.ones(X.shape), X, Y, X**2, Y**2, X * Y]
    C, _, _, _ = np.linalg.lstsq(A, z_sample, rcond=None)
    X_full = x_grid.flatten().astype(np.float32)
    Y_full = y_grid.flatten().astype(np.float32)
    A_full = np.c_[np.ones(X_full.shape), X_full, Y_full, X_full**2, Y_full**2, X_full * Y_full]
    bg_est = np.dot(A_full, C).reshape(h, w)
    return bg_est.astype(np.float32)


def _robust_mean_std(values: np.ndarray) -> Tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    sigma = 1.4826 * mad
    return med, float(sigma)


def _gaussian_intersection(mu1: float, s1: float, mu2: float, s2: float) -> Optional[float]:
    if s1 <= 1e-6 or s2 <= 1e-6:
        return None
    a = 1.0 / (2 * s1 * s1) - 1.0 / (2 * s2 * s2)
    b = mu2 / (s2 * s2) - mu1 / (s1 * s1)
    c = (mu1 * mu1) / (2 * s1 * s1) - (mu2 * mu2) / (2 * s2 * s2) + np.log(s2 / s1)
    if abs(a) < 1e-9:
        if abs(b) < 1e-9:
            return None
        return -c / b
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sqrt_disc = float(np.sqrt(disc))
    x1 = (-b + sqrt_disc) / (2 * a)
    x2 = (-b - sqrt_disc) / (2 * a)
    return x1 if mu1 < x1 < mu2 else x2 if mu1 < x2 < mu2 else None


def _make_plate_masks(
    alpha: np.ndarray,
    geom: LensGeometry,
    cfg: Dict[str, Any],
    l_map: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    h, w = alpha.shape
    alpha_mask = alpha
    blur_ksize = int(cfg.get("alpha_mask_blur_ksize", 3))
    if blur_ksize and blur_ksize > 1:
        k = max(3, blur_ksize | 1)
        alpha_mask = cv2.medianBlur((alpha_mask * 255.0).astype(np.uint8), k).astype(np.float32) / 255.0
    r_exclude_outer = float(cfg.get("r_exclude_outer", 0.95))
    alpha_clear_th = float(cfg.get("alpha_clear_th", 0.08))
    alpha_print_th_all = float(cfg.get("alpha_print_th", 0.25))
    alpha_print_th_mode = str(cfg.get("alpha_print_th_mode", "fixed")).lower()
    alpha_print_percentile = cfg.get("alpha_print_percentile")
    alpha_dot_th_cfg = cfg.get("alpha_dot_th")
    alpha_print_th_core = float(cfg.get("alpha_print_th_core", 0.40))
    alpha_dot_percentile = cfg.get("alpha_dot_percentile")
    alpha_ring_percentile = cfg.get("alpha_ring_percentile")
    clear_l_th = cfg.get("clear_l_th")
    valid_erode_px = int(cfg.get("valid_erode_px", 0))
    alpha_clip = cfg.get("alpha_clip", (0.02, 0.98))
    clip_lo = float(alpha_clip[0])
    clip_hi = float(alpha_clip[1])

    valid = _radial_mask(h, w, geom, 0.0, r_exclude_outer)
    if valid_erode_px > 0:
        valid = _erode_mask(valid, valid_erode_px)
    outer_shadow_ratio = float(cfg.get("outer_shadow_ratio", 0.92))
    outer_shadow_pct = float(cfg.get("outer_shadow_percentile", 10.0))
    if 0.0 < outer_shadow_ratio < 1.0:
        outer_band = _radial_mask(h, w, geom, outer_shadow_ratio, r_exclude_outer)
        if np.any(outer_band):
            alpha_outer = alpha_mask[outer_band]
            th = float(np.percentile(alpha_outer, outer_shadow_pct))
            valid = valid & ((~outer_band) | (alpha_mask > th))

    r_clear = float(cfg.get("r_clear", 0.40))
    r_ring0 = float(cfg.get("r_ring0", 0.70))
    dot_band = valid & _radial_mask(h, w, geom, r_clear, r_ring0)
    clip_low_mask = alpha_mask <= clip_lo + 1e-6
    if np.any(clip_low_mask & dot_band):
        neigh = cv2.medianBlur((alpha_mask * 255.0).astype(np.uint8), 3).astype(np.float32) / 255.0
        alpha_mask = np.where(clip_low_mask & dot_band, neigh, alpha_mask)
    clear_pre = valid & _radial_mask(h, w, geom, 0.0, r_clear) & (alpha_mask < alpha_clear_th)
    if l_map is not None and clear_l_th is not None:
        clear_pre = (
            valid
            & _radial_mask(h, w, geom, 0.0, r_clear)
            & ((alpha_mask < alpha_clear_th) & (l_map > float(clear_l_th)))
        )
    r_clear_guard = float(cfg.get("r_clear_guard", r_clear + 0.03))
    clear_guard = valid & _radial_mask(h, w, geom, 0.0, r_clear_guard)

    ink_candidate = valid & (~clear_guard)
    otsu_roi = ink_candidate
    if alpha_print_th_mode == "otsu":
        otsu_roi = ink_candidate & dot_band
    clear_baseline_mask = valid & _radial_mask(h, w, geom, 0.0, r_clear * 0.90)
    if alpha_print_th_mode == "otsu":
        otsu_th = _get_dynamic_threshold(
            alpha_mask,
            otsu_roi,
            min_th=float(cfg.get("alpha_otsu_min", 0.1)),
            max_th=float(cfg.get("alpha_otsu_max", 0.6)),
            fallback=float(cfg.get("alpha_print_th", 0.25)),
        )
        otsu_fallback = float(cfg.get("alpha_otsu_fallback", cfg.get("alpha_print_th", 0.25)))
        otsu_floor = float(cfg.get("alpha_otsu_floor", 0.2))
        if otsu_th <= otsu_floor:
            alpha_print_th_all = max(otsu_fallback, otsu_floor)
        else:
            alpha_print_th_all = otsu_th
    elif alpha_print_th_mode == "percentile":
        pct_raw = cfg.get("alpha_print_percentile", 30.0)
        pct = float(pct_raw) if pct_raw is not None else 30.0
        use_full = bool(cfg.get("alpha_percentile_use_full", True))
        vals = alpha_mask[ink_candidate] if use_full and np.any(ink_candidate) else alpha_mask[otsu_roi]
        if vals.size == 0:
            vals = alpha_mask.reshape(-1)
        if vals.size:
            th = float(np.percentile(vals, pct))
            alpha_print_th_all = float(
                np.clip(
                    th,
                    float(cfg.get("alpha_percentile_floor", cfg.get("alpha_otsu_min", 0.1))),
                    float(cfg.get("alpha_otsu_max", 0.6)),
                )
            )
        else:
            alpha_print_th_all = float(cfg.get("alpha_print_th", 0.25))
    elif alpha_print_th_mode == "stats":
        stats_k = float(cfg.get("alpha_stats_k", 2.5))
        stats_min = float(cfg.get("alpha_otsu_min", 0.1))
        stats_floor = float(cfg.get("alpha_stats_floor", stats_min))
        stats_max = float(cfg.get("alpha_otsu_max", 0.6))
        stats_roi = ink_candidate & dot_band
        stats_vals = alpha_mask[stats_roi] if np.any(stats_roi) else alpha_mask.reshape(-1)
        ink_mu, ink_sigma = _robust_mean_std(stats_vals)
        clear_mu, clear_sigma = (
            _robust_mean_std(alpha_mask[clear_baseline_mask]) if np.any(clear_baseline_mask) else (0.0, 0.0)
        )
        th = _gaussian_intersection(clear_mu, max(clear_sigma, 1e-6), ink_mu, max(ink_sigma, 1e-6))
        if th is None:
            th = clear_mu + stats_k * max(clear_sigma, 1e-6)
        alpha_print_th_all = float(np.clip(th, max(stats_min, stats_floor), stats_max))
    if alpha_print_th_mode == "percentile":
        floor = float(cfg.get("alpha_percentile_floor", cfg.get("alpha_otsu_min", 0.1)))
        alpha_print_th_all = max(alpha_print_th_all, floor)
    elif alpha_print_th_mode == "otsu":
        floor = float(cfg.get("alpha_otsu_floor", cfg.get("alpha_otsu_min", 0.1)))
        alpha_print_th_all = max(alpha_print_th_all, floor)
    alpha_dot_th = float(alpha_dot_th_cfg) if alpha_dot_th_cfg is not None else float(alpha_print_th_all)
    clear_alpha_median = None
    if np.any(clear_baseline_mask):
        clear_alpha_median = float(np.median(alpha[clear_baseline_mask]))

    r_ring1 = float(cfg.get("r_ring1", r_exclude_outer))
    ring_band = valid & _radial_mask(h, w, geom, r_ring0, r_ring1)
    dot_band = valid & _radial_mask(h, w, geom, r_clear, r_ring0)
    alpha_ring_th = float(cfg.get("alpha_ring_th", 0.55))
    ring_raw = ink_candidate & ring_band & (alpha_mask >= alpha_ring_th)

    ring_u8 = ring_raw.astype(np.uint8) * 255
    k = int(cfg.get("ring_morph_ksize", 7))
    k = max(3, k | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ring_closed = cv2.morphologyEx(ring_u8, cv2.MORPH_CLOSE, kernel)
    ring_closed = ring_closed > 0

    num, labels = cv2.connectedComponents(ring_closed.astype(np.uint8))
    if num > 1:
        areas = [(labels == i).sum() for i in range(1, num)]
        best = 1 + int(np.argmax(areas))
        ring = labels == best
    else:
        ring = ring_closed

    dot = ink_candidate & (~ring)

    th_ring = alpha_ring_th
    if isinstance(alpha_ring_percentile, (int, float)):
        ring_vals = alpha_mask[ring_band]
        if ring_vals.size:
            th_ring = float(np.percentile(ring_vals, float(alpha_ring_percentile)))
            th_ring = float(np.clip(th_ring, 0.40, 0.95))

    th_dot = alpha_dot_th
    if isinstance(alpha_dot_percentile, (int, float)):
        dot_vals = alpha_mask[dot_band]
        if dot_vals.size:
            th_dot = float(np.percentile(dot_vals, float(alpha_dot_percentile)))
            th_dot = float(np.clip(th_dot, 0.10, 0.35))
    if clear_alpha_median is not None:
        th_dot = max(th_dot, max(0.30, clear_alpha_median + 0.08))

    ring_mask = ring & (alpha_mask >= th_ring)
    dot_mask = dot & (alpha_mask >= th_dot)
    ink_mask = ring_mask | dot_mask
    ink_mask_core = ink_mask & (alpha_mask >= alpha_print_th_core)
    artifact_clip_ratio_valid = float(
        ((alpha_mask <= clip_lo + 1e-6) | (alpha_mask >= clip_hi - 1e-6)).sum() / max(valid.sum(), 1)
    )

    ring = ring_mask
    dot = dot_mask
    clear = clear_pre

    return {
        "ring": ring,
        "dot": dot,
        "clear": clear,
        "valid": valid,
        "ink_mask": ink_mask,
        "ink_mask_core": ink_mask_core,
        "alpha_print_th_used": float(alpha_print_th_all),
        "alpha_print_all_th_used": float(alpha_print_th_all),
        "alpha_print_core_th_used": float(alpha_print_th_core),
        "alpha_ring_th_used": float(th_ring),
        "alpha_dot_th_used": float(th_dot),
        "clear_alpha_median": clear_alpha_median,
        "artifact_clip_ratio_valid": artifact_clip_ratio_valid,
        "alpha_print_th_mode": alpha_print_th_mode,
    }


def _cluster_ink_masks(ink_est_bgr: np.ndarray, ink_mask: np.ndarray, k: int = 3) -> Dict[str, np.ndarray]:
    if ink_mask is None or ink_mask.sum() < k:
        return {}

    samples_bgr = ink_est_bgr[ink_mask]
    if samples_bgr.size == 0:
        return {}

    # Convert samples to CIE Lab
    samples_lab = to_cie_lab(samples_bgr.reshape(-1, 1, 3)).reshape(-1, 3)
    ab = samples_lab[:, 1:3].astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, _ = cv2.kmeans(ab, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)

    order = []
    for idx in range(k):
        idx_mask = labels == idx
        if np.any(idx_mask):
            order.append((idx, float(samples_lab[idx_mask, 0].mean())))
        else:
            order.append((idx, 0.0))
    order.sort(key=lambda x: x[1])

    coords = np.where(ink_mask)
    ink_masks: Dict[str, np.ndarray] = {}
    for new_id, (old_id, _) in enumerate(order, start=1):
        mask = np.zeros_like(ink_mask, dtype=bool)
        mask[coords] = labels == old_id
        ink_masks[f"ink{new_id}"] = mask
    return ink_masks


def _cluster_ink_masks_full(
    feature_map: np.ndarray,
    dist_map: np.ndarray,
    ink_mask_full: np.ndarray,
    ink_mask_core: np.ndarray,
    k: int = 3,
    block_size: int = 4,
) -> Dict[str, np.ndarray]:
    if ink_mask_core is None or ink_mask_core.sum() < k:
        return {}
    if ink_mask_full is None or ink_mask_full.sum() == 0:
        return {}

    core_samples = feature_map[ink_mask_core]
    if core_samples.size == 0:
        return {}

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels_core, centers = cv2.kmeans(core_samples.astype(np.float32), k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.float32)

    h, w = ink_mask_full.shape
    bs = max(2, int(block_size))
    block_records = []
    for y in range(0, h, bs):
        y1 = min(h, y + bs)
        for x in range(0, w, bs):
            x1 = min(w, x + bs)
            block_mask = ink_mask_full[y:y1, x:x1]
            if block_mask.sum() == 0:
                continue
            block_features = feature_map[y:y1, x:x1][block_mask]
            feat_med = np.median(block_features, axis=0).astype(np.float32)
            dist_med = float(np.median(dist_map[y:y1, x:x1][block_mask]))
            block_records.append((y, y1, x, x1, feat_med, dist_med, int(block_mask.sum())))

    if not block_records:
        return {}

    feat_blocks = np.stack([b[4] for b in block_records], axis=0)
    diff = feat_blocks[:, None, :] - centers[None, :, :]
    dist = (diff**2).sum(axis=2)
    labels_blocks = dist.argmin(axis=1)

    order = []
    for idx in range(k):
        total = 0
        dist_sum = 0.0
        for label, record in zip(labels_blocks, block_records):
            if label != idx:
                continue
            dist_med = record[5]
            count = record[6]
            dist_sum += dist_med * count
            total += count
        order.append((idx, dist_sum / total if total else 0.0))
    order.sort(key=lambda x: x[1], reverse=True)

    label_map = {old_id: new_id for new_id, (old_id, _) in enumerate(order, start=1)}
    ink_masks: Dict[str, np.ndarray] = {f"ink{i}": np.zeros_like(ink_mask_full, dtype=bool) for i in range(1, k + 1)}
    for label, record in zip(labels_blocks, block_records):
        y, y1, x, x1, _, _, _ = record
        new_id = label_map.get(label, 1)
        mask = ink_mask_full[y:y1, x:x1]
        ink_masks[f"ink{new_id}"][y:y1, x:x1] |= mask

    return ink_masks


def _erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = max(3, (2 * px + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0


def _split_core_transition(mask: np.ndarray, geom: LensGeometry, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    w_dyn = max(4, int(0.015 * float(geom.r)))
    w_cfg = cfg.get("transition_width_px")
    if isinstance(w_cfg, int) and w_cfg > 0:
        w_dyn = w_cfg

    k = max(3, (2 * w_dyn + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    core = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0
    transition = mask & (~core)
    return core, transition


def _stats_lab_alpha(lab: np.ndarray, alpha: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    if mask.sum() == 0:
        return {"empty": True}

    vals = lab[mask]
    a = alpha[mask]

    def pct(x: np.ndarray, p: int) -> List[float]:
        return np.percentile(x, p, axis=0).tolist()

    mean_lab = vals.mean(axis=0)
    hex_ref = lab_to_hex(mean_lab)

    return {
        "lab": {
            "mean": mean_lab.tolist(),
            "p95": pct(vals, 95),
            "p05": pct(vals, 5),
            "std": vals.std(axis=0).tolist(),
        },
        "hex_ref": hex_ref,
        "alpha": {
            "mean": float(a.mean()),
            "p95": float(np.percentile(a, 95)),
            "p05": float(np.percentile(a, 5)),
            "std": float(a.std()),
        },
    }


def _ink_stats_by_source(
    lab_source: np.ndarray,
    alpha: np.ndarray,
    ink_masks: Dict[str, np.ndarray],
    valid_mask: np.ndarray,
) -> Dict[str, Any]:
    total = max(int(valid_mask.sum()), 1)
    out: Dict[str, Any] = {}
    for name, mask in ink_masks.items():
        area_ratio = float(mask.sum() / total)
        stats = _stats_lab_alpha(lab_source, alpha, mask)
        stats["area_ratio"] = area_ratio
        out[name] = stats
    return out


def _prepare_pair(
    white_bgr: np.ndarray,
    black_bgr: np.ndarray,
    cfg: Dict[str, Any],
    geom_hint: Optional[LensGeometry] = None,
) -> Tuple[np.ndarray, np.ndarray, LensGeometry, Dict[str, Any]]:
    geom_w = geom_hint if geom_hint is not None else detect_lens_circle(white_bgr)
    geom_b = detect_lens_circle(black_bgr)

    out_size = int(cfg.get("norm_size", 512))
    w_norm, g = _resize_to_square_centered(white_bgr, geom_w, out_size=out_size)
    b_norm, _ = _resize_to_square_centered(black_bgr, geom_b, out_size=out_size)
    w_norm, b_norm, order_meta = _ensure_pair_order(w_norm, b_norm, g, cfg)

    _, polar_T = get_polar_dims(cfg)
    b_aligned, reg = _phase_align_polar(
        w_norm,
        b_norm,
        g,
        T=polar_T,
        min_score=float(cfg.get("registration_min_score", 0.15)),
        max_dy_px=float(cfg.get("registration_max_dy_px", 4.0)),
    )
    reg["input_order"] = order_meta
    if bool(cfg.get("registration_ecc_enabled", False)) and not reg.get("low_confidence", False):
        ecc_mask = None
        if bool(cfg.get("registration_ecc_use_mask", True)):
            r0 = float(cfg.get("registration_ecc_mask_r0", 0.35))
            r1 = float(cfg.get("registration_ecc_mask_r1", 0.95))
            ecc_mask = _radial_mask(b_aligned.shape[0], b_aligned.shape[1], g, r0, r1).astype(np.uint8) * 255
        warp, cc, reason = _refine_alignment_ecc(w_norm, b_aligned, cfg, mask=ecc_mask)
        if warp is not None:
            ecc_min_apply = float(cfg.get("registration_ecc_min_apply_cc", 0.2))
            if cc is not None and cc >= ecc_min_apply:
                b_aligned = cv2.warpAffine(
                    b_aligned,
                    warp,
                    (b_aligned.shape[1], b_aligned.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                reg["ecc"] = {
                    "applied": True,
                    "cc": cc,
                    "warp_mode": str(cfg.get("registration_ecc_warp", "euclidean")).lower(),
                    "mask_r": [
                        float(cfg.get("registration_ecc_mask_r0", 0.35)),
                        float(cfg.get("registration_ecc_mask_r1", 0.95)),
                    ],
                }
            else:
                reg["ecc"] = {
                    "applied": False,
                    "cc": cc,
                    "reason": "ecc_low_apply_cc",
                    "warp_mode": str(cfg.get("registration_ecc_warp", "euclidean")).lower(),
                    "mask_r": [
                        float(cfg.get("registration_ecc_mask_r0", 0.35)),
                        float(cfg.get("registration_ecc_mask_r1", 0.95)),
                    ],
                }
        else:
            reg["ecc"] = {
                "applied": False,
                "cc": cc,
                "reason": reason,
                "warp_mode": str(cfg.get("registration_ecc_warp", "euclidean")).lower(),
                "mask_r": [
                    float(cfg.get("registration_ecc_mask_r0", 0.35)),
                    float(cfg.get("registration_ecc_mask_r1", 0.95)),
                ],
            }
    if bool(cfg.get("registration_orb_enabled", False)):
        ecc_applied = reg.get("ecc", {}).get("applied", False)
        if not ecc_applied:
            orb_aligned, orb_score, orb_reason = _align_features_orb(w_norm, b_aligned, cfg)
            if orb_aligned is not None:
                b_aligned = orb_aligned
                reg["orb"] = {"applied": True, "score": orb_score}
            else:
                reg["orb"] = {"applied": False, "score": orb_score, "reason": orb_reason}
    return w_norm, b_aligned, g, reg


def analyze_plate_pair(
    white_bgr: np.ndarray,
    black_bgr: np.ndarray,
    cfg: Dict[str, Any],
    match_id: Optional[str] = None,
    geom_hint: Optional[LensGeometry] = None,
) -> Dict[str, Any]:
    w_norm, b_aligned, geom, reg = _prepare_pair(white_bgr, black_bgr, cfg, geom_hint=geom_hint)

    alpha, ink_est, alpha_meta, nan_mask = _compute_alpha_map(
        w_norm,
        b_aligned,
        geom,
        float(cfg.get("r_bg", cfg.get("r_clear", 0.40))),
        float(cfg.get("r_clear", 0.40)),
        float(cfg.get("wb_diff_min", 1.0)),
        float(cfg.get("wb_denom_min", 20.0)),
        bool(cfg.get("spatial_bg_enabled", False)),
        float(cfg.get("spatial_bg_r_outer", 0.98)),
        int(cfg.get("spatial_bg_stride", 4)),
        alpha_clip=tuple(cfg.get("alpha_clip", (0.02, 0.98))),
    )

    # ==== Phase 2: Dynamic Radius Detection ====
    dynamic_radii_result: Optional[DynamicRadiiResult] = None
    effective_cfg = cfg.copy()  # Use copy to avoid mutating original

    if bool(cfg.get("dynamic_radii_enabled", False)):
        dynamic_radii_result = detect_dynamic_radii(
            alpha,
            geom,
            n_bins=int(cfg.get("dynamic_radii_n_bins", 100)),
            savgol_window=int(cfg.get("dynamic_radii_savgol_window", 11)),
            savgol_order=int(cfg.get("dynamic_radii_savgol_order", 3)),
            r_clear_range=tuple(cfg.get("dynamic_radii_r_clear_range", (0.20, 0.60))),
            r_ring0_range=tuple(cfg.get("dynamic_radii_r_ring0_range", (0.55, 0.85))),
            min_confidence=float(cfg.get("dynamic_radii_min_confidence", 0.3)),
            fallback_r_clear=float(cfg.get("r_clear", 0.40)),
            fallback_r_ring0=float(cfg.get("r_ring0", 0.70)),
        )

        # Apply detected radii if confidence is sufficient
        if dynamic_radii_result.confidence >= float(cfg.get("dynamic_radii_min_confidence", 0.3)):
            effective_cfg["r_clear"] = dynamic_radii_result.r_clear
            effective_cfg["r_ring0"] = dynamic_radii_result.r_ring0
            # Also update r_clear_guard to maintain offset
            r_clear_guard_offset = float(cfg.get("r_clear_guard", 0.43)) - float(cfg.get("r_clear", 0.40))
            effective_cfg["r_clear_guard"] = dynamic_radii_result.r_clear + r_clear_guard_offset

    lab = to_cie_lab(w_norm)
    lab_black = to_cie_lab(b_aligned)
    l_map = lab[..., 0] if lab is not None else None
    masks = _make_plate_masks(alpha, geom, effective_cfg, l_map=l_map)
    plates_out: Dict[str, Any] = {}

    for name in ["ring", "dot", "clear"]:
        mask = masks[name]
        if name == "clear":
            core = mask
            transition = None
        else:
            core, transition = _split_core_transition(mask, geom, cfg)
            if "ink_mask_core" in masks:
                core = core & masks["ink_mask_core"]

        geom_info = {"area_ratio": float(mask.sum() / max(masks["valid"].sum(), 1))}
        plates_out[name] = {"geometry": geom_info, "core": _stats_lab_alpha(lab, alpha, core)}
        if transition is not None:
            plates_out[name]["transition"] = _stats_lab_alpha(lab, alpha, transition)
            plates_out[name].setdefault("notes", []).append("transition_is_design_feature")

    alpha_print_th_used = float(masks.get("alpha_print_th_used", cfg.get("alpha_print_th", 0.25)))
    valid_mask = masks["valid"]
    ink_mask = masks["ink_mask"]
    clear_mask = masks["clear"]
    dot_mask = masks["dot"]
    valid_area_ratio = float(valid_mask.sum() / max(valid_mask.size, 1))
    valid_area = max(int(valid_mask.sum()), 1)
    ink_area_ratio = float(ink_mask.sum() / valid_area)
    clear_area_ratio = float(clear_mask.sum() / valid_area)
    dot_area_ratio = float(dot_mask.sum() / valid_area)
    alpha_clip = alpha_meta.get("alpha_clip", (0.02, 0.98))
    clip_lo = float(alpha_clip[0])
    clip_hi = float(alpha_clip[1])
    clip_low = alpha <= clip_lo + 1e-6
    clip_high = alpha >= clip_hi - 1e-6
    mask_artifact_ratio_valid = float(((clip_low | nan_mask) & valid_mask).sum() / valid_area)
    artifact_nan_ratio_valid = float((nan_mask & valid_mask).sum() / valid_area)
    r_exclude_outer = float(cfg.get("r_exclude_outer", 0.95))
    leak_r0 = min(0.93, r_exclude_outer)
    if leak_r0 < 0.0:
        leak_r0 = 0.0
    rim_mask = _radial_mask(alpha.shape[0], alpha.shape[1], geom, leak_r0, r_exclude_outer)
    outer_rim_leak_ratio = float((ink_mask & rim_mask).sum() / max(int(ink_mask.sum()), 1))

    r_clear = float(effective_cfg.get("r_clear", 0.40))
    r_ring0 = float(effective_cfg.get("r_ring0", 0.70))
    r_ring1 = float(effective_cfg.get("r_ring1", r_exclude_outer))
    clear_region = _radial_mask(alpha.shape[0], alpha.shape[1], geom, 0.0, r_clear) & valid_mask
    clear_printed_ratio = float((ink_mask & clear_region).sum() / valid_area)
    outer_rim_printed_ratio = float((ink_mask & rim_mask).sum() / valid_area)

    # Generate Polar Mask for Ink Core (for single_analyzer Hard Gate)
    # This avoids coordinate transform issues (512x512 vs original)
    polar_R, polar_T = get_polar_dims(cfg)

    # Safety: ink_mask_core might be None if masks generation failed
    mask_core_src = masks.get("ink_mask_core")
    if mask_core_src is not None:
        ink_mask_core_polar = _mask_to_polar(mask_core_src, geom, polar_R, polar_T)
    else:
        ink_mask_core_polar = None

    ink_masks: Dict[str, np.ndarray] = {}
    k_used = 0
    k_expected = int(cfg.get("ink_k", 3))
    feature_map = None
    dist_map = None
    if ink_mask is not None and np.any(ink_mask):
        lab_white = to_cie_lab(w_norm)
        lab_black = to_cie_lab(b_aligned)
        h, w = alpha.shape
        yy, xx = np.mgrid[0:h, 0:w]
        dist_map = np.sqrt((xx - geom.cx) ** 2 + (yy - geom.cy) ** 2) / max(float(geom.r), 1e-6)
        dist_weight = float(cfg.get("ink_dist_weight", 1.0))
        feature_map = np.stack(
            [
                lab_white[..., 1],
                lab_white[..., 2],
                lab_black[..., 1],
                lab_black[..., 2],
                alpha,
                dist_map * dist_weight,
            ],
            axis=2,
        ).astype(np.float32)
        erode_px = int(cfg.get("ink_core_erode_px", 2))
        block_size = int(cfg.get("ink_block_size", 4))
        bg_white_l_max = cfg.get("ink_bg_white_l_max")
        bg_black_l_min = cfg.get("ink_bg_black_l_min")
        ink_mask_full = masks["dot"] | masks["ring"]
        if bg_white_l_max is not None:
            ink_mask_full = ink_mask_full & (lab_white[..., 0] <= float(bg_white_l_max))
        if bg_black_l_min is not None:
            ink_mask_full = ink_mask_full & (lab_black[..., 0] >= float(bg_black_l_min))
        ring_core, _ = _split_core_transition(masks["ring"], geom, cfg)
        dot_core, _ = _split_core_transition(masks["dot"], geom, cfg)
        if erode_px > 0:
            ring_core = _erode_mask(ring_core, erode_px)
            dot_core = _erode_mask(dot_core, erode_px)

        ink_masks = {}
        if ring_core.sum() > 0:
            ink_masks["ink1"] = ring_core
        k_dot = max(1, k_expected - 1)
        if dot_core.sum() > 0 and k_dot > 0:
            dot_masks = _cluster_ink_masks_full(
                feature_map,
                dist_map,
                ink_mask_full,
                dot_core,
                k=k_dot,
                block_size=block_size,
            )
            start_idx = 2 if "ink1" in ink_masks else 1
            for offset, name in enumerate(sorted(dot_masks.keys()), start=start_idx):
                ink_masks[f"ink{offset}"] = dot_masks[name]
        k_used = len(ink_masks)

    lab_white = to_cie_lab(w_norm)
    lab_black = to_cie_lab(b_aligned)

    center_features: Dict[str, List[float]] = {}
    pairwise_center_dist: List[Dict[str, Any]] = []
    warning_list: List[str] = []
    if ink_masks and feature_map is not None:
        for name, mask in ink_masks.items():
            if mask.sum() == 0:
                continue
            center = feature_map[mask].mean(axis=0).tolist()
            center_features[name] = [float(x) for x in center]
        names = sorted(center_features.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = np.array(center_features[names[i]], dtype=np.float32)
                b = np.array(center_features[names[j]], dtype=np.float32)
                dist = float(np.linalg.norm(a - b))
                pairwise_center_dist.append({"pair": [names[i], names[j]], "dist": dist})
        min_dist = min((item["dist"] for item in pairwise_center_dist), default=None)
        min_allowed = float(cfg.get("ink_cluster_min_dist", 4.0))
        if min_dist is not None and min_dist < min_allowed:
            warning_list.append("clusters_too_close_possible_over_split")

    inks_out = {
        "k_expected": k_expected,
        "k_used": k_used,
        "cluster_feature": "pair_lab_ab_alpha_dist",
        "core_erode_px": int(cfg.get("ink_core_erode_px", 2)),
        "cluster_centers": center_features,
        "pairwise_center_dist": pairwise_center_dist,
        "warnings": warning_list,
        "by_source": {
            "from_white": _ink_stats_by_source(lab_white, alpha, ink_masks, valid_mask),
            "from_black": _ink_stats_by_source(lab_black, alpha, ink_masks, valid_mask),
        },
    }

    return {
        "schema_version": "plate_v1.2",
        "match_id": match_id,
        "geom": {
            "cx": round(float(geom.cx), 2),
            "cy": round(float(geom.cy), 2),
            "r": round(float(geom.r), 2),
            "r_exclude_outer": r_exclude_outer,
        },
        "registration": reg,
        "alpha_model": alpha_meta,
        "masks_summary": {
            "alpha_print_th_used": alpha_print_th_used,
            "alpha_print_all_th_used": masks.get("alpha_print_all_th_used"),
            "alpha_print_core_th_used": masks.get("alpha_print_core_th_used"),
            "alpha_print_percentile": cfg.get("alpha_print_percentile", None),
            "alpha_dot_th_used": masks.get("alpha_dot_th_used"),
            "alpha_ring_th_used": masks.get("alpha_ring_th_used"),
            "valid_area_ratio": valid_area_ratio,
            "ink_area_ratio": ink_area_ratio,
            "clear_area_ratio": clear_area_ratio,
            "dot_area_ratio": dot_area_ratio,
            "outer_rim_leak_ratio": outer_rim_leak_ratio,
            "clear_printed_ratio": clear_printed_ratio,
            "outer_rim_printed_ratio": outer_rim_printed_ratio,
            "clear_alpha_median": masks.get("clear_alpha_median"),
            "mask_artifact_ratio_valid": mask_artifact_ratio_valid,
            "artifact_clip_ratio_valid": masks.get("artifact_clip_ratio_valid"),
            "artifact_nan_ratio_valid": artifact_nan_ratio_valid,
            "artifact_clip_low_ratio_valid": float((clip_low & valid_mask).sum() / valid_area),
            "artifact_clip_high_ratio_valid": float((clip_high & valid_mask).sum() / valid_area),
            "alpha_print_th_mode": masks.get("alpha_print_th_mode"),
            "zero_bias_correction": alpha_meta.get("zero_bias_correction", 0.0),
            "safe_denom_val": alpha_meta.get("safe_denom_val", 0.0),
        },
        "plate_geometry": {
            "clear": {"r_range": [0.0, r_clear]},
            "dot": {"r_range": [r_clear, r_ring0]},
            "ring": {"r_range": [r_ring0, r_ring1]},
            "dynamic_radii": (
                {
                    "enabled": bool(cfg.get("dynamic_radii_enabled", False)),
                    "detected": dynamic_radii_result is not None,
                    "r_clear": dynamic_radii_result.r_clear if dynamic_radii_result else None,
                    "r_ring0": dynamic_radii_result.r_ring0 if dynamic_radii_result else None,
                    "confidence": dynamic_radii_result.confidence if dynamic_radii_result else None,
                    "method": dynamic_radii_result.method if dynamic_radii_result else None,
                    "meta": dynamic_radii_result.meta if dynamic_radii_result else None,
                    "config_r_clear": float(cfg.get("r_clear", 0.40)),
                    "config_r_ring0": float(cfg.get("r_ring0", 0.70)),
                }
                if bool(cfg.get("dynamic_radii_enabled", False))
                else None
            ),
        },
        "plates": plates_out,
        "inks": inks_out,
        # Internal: numpy arrays for downstream use (e.g., Plate Gate)
        # Not JSON-serializable, intended for in-memory pipeline use
        "_masks": {
            "ink_mask": masks.get("ink_mask"),
            "ink_mask_core": masks.get("ink_mask_core"),
            "dot": masks.get("dot"),
            "ring": masks.get("ring"),
            "clear": masks.get("clear"),
            "valid": masks.get("valid"),
            "ink_mask_core_polar": ink_mask_core_polar,  # Direct polar mask
            "alpha": alpha,  # Raw alpha map for effective_density computation
            "alpha_polar": cv2.warpPolar(
                alpha.astype(np.float32),
                (polar_R, polar_T),
                (geom.cx, geom.cy),
                geom.r,
                cv2.WARP_POLAR_LINEAR,
            ),  # Alpha in polar coords
        },
        "_geom_internal": geom,  # LensGeometry object for coordinate transforms
    }


def analyze_plate_lite_pair(
    white_bgr: np.ndarray,
    black_bgr: np.ndarray,
    lite_cfg: Dict[str, Any],
    plate_cfg: Dict[str, Any],
    match_id: Optional[str] = None,
    geom_hint: Optional[LensGeometry] = None,
    expected_k: Optional[int] = None,
) -> Dict[str, Any]:
    w_norm, b_aligned, geom, reg = _prepare_pair(white_bgr, black_bgr, plate_cfg, geom_hint=geom_hint)

    alpha_map, alpha_meta = _compute_alpha_map_lite(
        w_norm,
        b_aligned,
        blur_ksize=int(lite_cfg.get("blur_ksize", 5)),
        backlight=float(lite_cfg.get("backlight", 255.0)),
        alpha_clip=tuple(lite_cfg.get("alpha_clip", (0.02, 0.98))),
    )

    # Collect warnings from alpha computation
    warning_list: List[str] = list(alpha_meta.get("warnings", []))

    lab = to_cie_lab(w_norm)
    lab_black = to_cie_lab(b_aligned)
    l_map = lab[..., 0] if lab is not None else None
    masks = _make_plate_masks(alpha_map, geom, plate_cfg, l_map=l_map)

    ring_core, _ = _split_core_transition(masks["ring"], geom, plate_cfg)
    dot_core, _ = _split_core_transition(masks["dot"], geom, plate_cfg)
    clear_mask = masks["clear"]

    paper_lab, paper_meta, paper_warnings = _resolve_paper_color(lite_cfg, clear_mask, lab)
    warning_list.extend(paper_warnings)
    alpha_threshold = float(lite_cfg.get("alpha_threshold", 0.1))

    zones = {
        "ring_core": ring_core,
        "dot_core": dot_core,
        "clear": clear_mask,
    }

    zones_out: Dict[str, Any] = {}
    zones_black_out: Dict[str, Any] = {}
    for name, mask in zones.items():
        if mask is None or mask.sum() == 0:
            zones_out[name] = {"empty": True}
            zones_black_out[name] = {"empty": True}
            warning_list.append(f"{name}_mask_empty")
            continue

        alpha_mean = float(alpha_map[mask].mean())
        obs_lab = lab[mask].mean(axis=0)
        obs_lab_black = lab_black[mask].mean(axis=0)
        if alpha_mean < alpha_threshold:
            ink_lab = obs_lab.copy()
            warning_list.append(f"{name}_alpha_too_low_using_observed")
        else:
            ink_lab = (obs_lab - (1.0 - alpha_mean) * paper_lab) / max(alpha_mean, 1e-6)
            ink_lab[0] = np.clip(ink_lab[0], 0.0, 100.0)
            ink_lab[1] = np.clip(ink_lab[1], -128.0, 127.0)
            ink_lab[2] = np.clip(ink_lab[2], -128.0, 127.0)

        zones_out[name] = {
            "alpha_mean": alpha_mean,
            "obs_lab": obs_lab.tolist(),
            "ink_lab": ink_lab.tolist(),
            "ink_hex": lab_to_hex(ink_lab),
        }
        zones_black_out[name] = {
            "alpha_mean": alpha_mean,
            "obs_lab": obs_lab_black.tolist(),
            "obs_hex": lab_to_hex(obs_lab_black),
        }

    # Per-ink measurement (plate_lite) using ink masks
    ink_entries: List[Dict[str, Any]] = []
    ink_mask_full = masks.get("dot") | masks.get("ring")
    if ink_mask_full is not None and np.any(ink_mask_full):
        lab_white = to_cie_lab(w_norm)
        lab_black = to_cie_lab(b_aligned)
        h, w = alpha_map.shape
        yy, xx = np.mgrid[0:h, 0:w]
        dist_map = np.sqrt((xx - geom.cx) ** 2 + (yy - geom.cy) ** 2) / max(float(geom.r), 1e-6)
        dist_weight = float(plate_cfg.get("ink_dist_weight", 1.0))
        feature_map = np.stack(
            [
                lab_white[..., 1],
                lab_white[..., 2],
                lab_black[..., 1],
                lab_black[..., 2],
                alpha_map,
                dist_map * dist_weight,
            ],
            axis=2,
        ).astype(np.float32)
        erode_px = int(plate_cfg.get("ink_core_erode_px", 2))
        block_size = int(plate_cfg.get("ink_block_size", 4))
        bg_white_l_max = plate_cfg.get("ink_bg_white_l_max")
        bg_black_l_min = plate_cfg.get("ink_bg_black_l_min")
        if bg_white_l_max is not None:
            ink_mask_full = ink_mask_full & (lab_white[..., 0] <= float(bg_white_l_max))
        if bg_black_l_min is not None:
            ink_mask_full = ink_mask_full & (lab_black[..., 0] >= float(bg_black_l_min))
        ring_core_local, _ = _split_core_transition(masks["ring"], geom, plate_cfg)
        dot_core_local, _ = _split_core_transition(masks["dot"], geom, plate_cfg)
        if erode_px > 0:
            ring_core_local = _erode_mask(ring_core_local, erode_px)
            dot_core_local = _erode_mask(dot_core_local, erode_px)

        k_expected = int(expected_k or plate_cfg.get("ink_k", 3))
        ink_masks: Dict[str, np.ndarray] = {}
        if ring_core_local.sum() > 0:
            ink_masks["ink1"] = ring_core_local
        k_dot = max(1, k_expected - 1)
        if dot_core_local.sum() > 0 and k_dot > 0:
            dot_masks = _cluster_ink_masks_full(
                feature_map,
                dist_map,
                ink_mask_full,
                dot_core_local,
                k=k_dot,
                block_size=block_size,
            )
            start_idx = 2 if "ink1" in ink_masks else 1
            for offset, name in enumerate(sorted(dot_masks.keys()), start=start_idx):
                ink_masks[f"ink{offset}"] = dot_masks[name]

        total = max(int(masks["valid"].sum()), 1)
        for name, mask in ink_masks.items():
            if mask.sum() == 0:
                continue
            obs_lab = lab_white[mask].mean(axis=0)
            obs_lab_black = lab_black[mask].mean(axis=0)
            alpha_mean = float(alpha_map[mask].mean())
            if alpha_mean < alpha_threshold:
                ink_lab = obs_lab.copy()
            else:
                ink_lab = (obs_lab - (1.0 - alpha_mean) * paper_lab) / max(alpha_mean, 1e-6)
                ink_lab[0] = np.clip(ink_lab[0], 0.0, 100.0)
                ink_lab[1] = np.clip(ink_lab[1], -128.0, 127.0)
                ink_lab[2] = np.clip(ink_lab[2], -128.0, 127.0)

            ink_entries.append(
                {
                    "ink_id": int(name.replace("ink", "")) - 1,
                    "ink_key": name,
                    "obs_lab": obs_lab.tolist(),
                    "obs_hex": lab_to_hex(obs_lab),
                    "obs_lab_black": obs_lab_black.tolist(),
                    "obs_hex_black": lab_to_hex(obs_lab_black),
                    "ink_lab": ink_lab.tolist(),
                    "ink_hex": lab_to_hex(ink_lab),
                    "alpha_mean": alpha_mean,
                    "area_ratio": float(mask.sum() / total),
                    "source": "plate_lite",
                }
            )

    return {
        "schema_version": "plate_lite_v1.0",
        "match_id": match_id,
        "registration": reg,
        "alpha_model": alpha_meta,
        "zones": zones_out,
        "zones_black": zones_black_out,
        "inks": ink_entries,
        "paper_color_used": paper_meta,
        "warnings": warning_list,
    }


# NOTE: _mask_to_polar is now imported from ._helpers (see imports above)


def _masked_radial_stats_lab(
    polar_bgr: np.ndarray, mask: np.ndarray, r_start: float = 0.0, r_end: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lab = to_cie_lab(polar_bgr)
    T, R, _ = lab.shape
    r0 = int(R * r_start)
    r1 = int(R * r_end)
    r0 = max(0, min(R - 1, r0))
    r1 = max(r0 + 1, min(R, r1))

    lab_roi = lab[:, r0:r1, :]
    mask_roi = mask[:, r0:r1]

    R_prime = r1 - r0
    mean_curve = np.full((R_prime, 3), np.nan, dtype=np.float32)
    p95_curve = np.full((R_prime, 3), np.nan, dtype=np.float32)
    p05_curve = np.full((R_prime, 3), np.nan, dtype=np.float32)
    std_curve = np.full((R_prime, 3), np.nan, dtype=np.float32)

    first_valid = None
    for r_idx in range(R_prime):
        col_mask = mask_roi[:, r_idx] > 0
        if np.any(col_mask):
            if first_valid is None:
                first_valid = r_idx
            pixels = lab_roi[col_mask, r_idx, :]
            mean_curve[r_idx] = pixels.mean(axis=0)
            p95_curve[r_idx] = np.percentile(pixels, 95, axis=0)
            p05_curve[r_idx] = np.percentile(pixels, 5, axis=0)
            std_curve[r_idx] = pixels.std(axis=0)
        elif r_idx > 0 and not np.isnan(mean_curve[r_idx - 1, 0]):
            mean_curve[r_idx] = mean_curve[r_idx - 1]
            p95_curve[r_idx] = p95_curve[r_idx - 1]
            p05_curve[r_idx] = p05_curve[r_idx - 1]
            std_curve[r_idx] = std_curve[r_idx - 1]

    if first_valid is not None and first_valid > 0:
        mean_curve[:first_valid] = mean_curve[first_valid]
        p95_curve[:first_valid] = p95_curve[first_valid]
        p05_curve[:first_valid] = p05_curve[first_valid]
        std_curve[:first_valid] = std_curve[first_valid]

    return mean_curve, p95_curve, p05_curve, std_curve


def _masked_radial_stats_scalar(
    polar_map: np.ndarray, mask: np.ndarray, r_start: float = 0.0, r_end: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T, R = polar_map.shape
    r0 = int(R * r_start)
    r1 = int(R * r_end)
    r0 = max(0, min(R - 1, r0))
    r1 = max(r0 + 1, min(R, r1))

    map_roi = polar_map[:, r0:r1]
    mask_roi = mask[:, r0:r1]

    R_prime = r1 - r0
    mean_curve = np.full((R_prime,), np.nan, dtype=np.float32)
    p95_curve = np.full((R_prime,), np.nan, dtype=np.float32)
    p05_curve = np.full((R_prime,), np.nan, dtype=np.float32)
    std_curve = np.full((R_prime,), np.nan, dtype=np.float32)

    first_valid = None
    for r_idx in range(R_prime):
        col_mask = mask_roi[:, r_idx] > 0
        if np.any(col_mask):
            if first_valid is None:
                first_valid = r_idx
            pixels = map_roi[col_mask, r_idx]
            mean_curve[r_idx] = pixels.mean(axis=0)
            p95_curve[r_idx] = np.percentile(pixels, 95, axis=0)
            p05_curve[r_idx] = np.percentile(pixels, 5, axis=0)
            std_curve[r_idx] = pixels.std(axis=0)
        elif r_idx > 0 and not np.isnan(mean_curve[r_idx - 1]):
            mean_curve[r_idx] = mean_curve[r_idx - 1]
            p95_curve[r_idx] = p95_curve[r_idx - 1]
            p05_curve[r_idx] = p05_curve[r_idx - 1]
            std_curve[r_idx] = std_curve[r_idx - 1]

    if first_valid is not None and first_valid > 0:
        mean_curve[:first_valid] = mean_curve[first_valid]
        p95_curve[:first_valid] = p95_curve[first_valid]
        p05_curve[:first_valid] = p05_curve[first_valid]
        std_curve[:first_valid] = std_curve[first_valid]

    return mean_curve, p95_curve, p05_curve, std_curve


def compute_plate_signatures(
    white_bgr: np.ndarray,
    black_bgr: np.ndarray,
    cfg: Dict[str, Any],
    geom_hint: Optional[LensGeometry] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    plate_cfg = get_plate_cfg(cfg)
    w_norm, b_aligned, geom, _ = _prepare_pair(white_bgr, black_bgr, plate_cfg, geom_hint=geom_hint)

    alpha, _, alpha_meta, _ = _compute_alpha_map(
        w_norm,
        b_aligned,
        geom,
        float(plate_cfg.get("r_bg", plate_cfg.get("r_clear", 0.40))),
        float(plate_cfg.get("r_clear", 0.40)),
        float(plate_cfg.get("wb_diff_min", 1.0)),
        float(plate_cfg.get("wb_denom_min", 20.0)),
        bool(plate_cfg.get("spatial_bg_enabled", False)),
        float(plate_cfg.get("spatial_bg_r_outer", 0.98)),
        int(plate_cfg.get("spatial_bg_stride", 4)),
        alpha_clip=tuple(plate_cfg.get("alpha_clip", (0.02, 0.98))),
    )
    l_map = to_cie_lab(w_norm)[..., 0]
    masks = _make_plate_masks(alpha, geom, plate_cfg, l_map=l_map)

    ring_core, _ = _split_core_transition(masks["ring"], geom, plate_cfg)
    dot_core, _ = _split_core_transition(masks["dot"], geom, plate_cfg)

    R, T = get_polar_dims(cfg)
    polar_white = to_polar(w_norm, geom, R=R, T=T)
    alpha_polar = cv2.warpPolar(alpha.astype(np.float32), (R, T), (geom.cx, geom.cy), geom.r, cv2.WARP_POLAR_LINEAR)

    ring_mask_polar = _mask_to_polar(ring_core, geom, R, T)
    dot_mask_polar = _mask_to_polar(dot_core, geom, R, T)

    ring_lab_mean, ring_lab_p95, ring_lab_p05, ring_lab_std = _masked_radial_stats_lab(polar_white, ring_mask_polar)
    dot_lab_mean, dot_lab_p95, dot_lab_p05, dot_lab_std = _masked_radial_stats_lab(polar_white, dot_mask_polar)
    ring_alpha_mean, ring_alpha_p95, ring_alpha_p05, ring_alpha_std = _masked_radial_stats_scalar(
        alpha_polar, ring_mask_polar
    )
    dot_alpha_mean, dot_alpha_p95, dot_alpha_p05, dot_alpha_std = _masked_radial_stats_scalar(
        alpha_polar, dot_mask_polar
    )

    signatures = {
        "plate_ring_core_radial_lab_mean": ring_lab_mean,
        "plate_ring_core_radial_lab_p95": ring_lab_p95,
        "plate_ring_core_radial_lab_p05": ring_lab_p05,
        "plate_ring_core_radial_lab_std": ring_lab_std,
        "plate_ring_core_radial_alpha_mean": ring_alpha_mean,
        "plate_ring_core_radial_alpha_p95": ring_alpha_p95,
        "plate_ring_core_radial_alpha_p05": ring_alpha_p05,
        "plate_ring_core_radial_alpha_std": ring_alpha_std,
        "plate_dot_core_radial_lab_mean": dot_lab_mean,
        "plate_dot_core_radial_lab_p95": dot_lab_p95,
        "plate_dot_core_radial_lab_p05": dot_lab_p05,
        "plate_dot_core_radial_lab_std": dot_lab_std,
        "plate_dot_core_radial_alpha_mean": dot_alpha_mean,
        "plate_dot_core_radial_alpha_p95": dot_alpha_p95,
        "plate_dot_core_radial_alpha_p05": dot_alpha_p05,
        "plate_dot_core_radial_alpha_std": dot_alpha_std,
    }

    r_clear = float(plate_cfg.get("r_clear", 0.40))
    r_ring0 = float(plate_cfg.get("r_ring0", 0.70))
    r_ring1 = float(plate_cfg.get("r_ring1", float(plate_cfg.get("r_exclude_outer", 0.95))))

    plate_meta = {
        "plate_params": {
            "transition_width": "max(4px, 0.015R)",
            "outer_rim_exclude": float(plate_cfg.get("r_exclude_outer", 0.95)),
            "alpha_clear_th": float(plate_cfg.get("alpha_clear_th", 0.08)),
        },
        "plate_geometry": {
            "ring": {"r_range": [r_ring0, r_ring1]},
            "dot": {"r_range": [r_clear, r_ring0]},
            "clear": {"r_range": [0.0, r_clear]},
        },
    }

    return signatures, plate_meta


def compute_plate_artifacts(
    white_bgr: np.ndarray,
    black_bgr: np.ndarray,
    cfg: Dict[str, Any],
    geom_hint: Optional[LensGeometry] = None,
) -> Dict[str, Any]:
    plate_cfg = get_plate_cfg(cfg)
    w_norm, b_aligned, geom, _ = _prepare_pair(white_bgr, black_bgr, plate_cfg, geom_hint=geom_hint)
    alpha, ink_est, alpha_meta, _ = _compute_alpha_map(
        w_norm,
        b_aligned,
        geom,
        float(plate_cfg.get("r_bg", plate_cfg.get("r_clear", 0.40))),
        float(plate_cfg.get("r_clear", 0.40)),
        float(plate_cfg.get("wb_diff_min", 1.0)),
        float(plate_cfg.get("wb_denom_min", 20.0)),
        bool(plate_cfg.get("spatial_bg_enabled", False)),
        float(plate_cfg.get("spatial_bg_r_outer", 0.98)),
        int(plate_cfg.get("spatial_bg_stride", 4)),
        alpha_clip=tuple(plate_cfg.get("alpha_clip", (0.02, 0.98))),
    )
    l_map = to_cie_lab(w_norm)[..., 0]
    masks = _make_plate_masks(alpha, geom, plate_cfg, l_map=l_map)
    ink_mask = masks.get("ink_mask")
    ink_masks: Dict[str, np.ndarray] = {}
    if ink_mask is not None and np.any(ink_mask):
        lab_white = to_cie_lab(w_norm)
        lab_black = to_cie_lab(b_aligned)
        h, w = alpha.shape
        yy, xx = np.mgrid[0:h, 0:w]
        dist_map = np.sqrt((xx - geom.cx) ** 2 + (yy - geom.cy) ** 2) / max(float(geom.r), 1e-6)
        dist_weight = float(plate_cfg.get("ink_dist_weight", 1.0))
        feature_map = np.stack(
            [
                lab_white[..., 1],
                lab_white[..., 2],
                lab_black[..., 1],
                lab_black[..., 2],
                alpha,
                dist_map * dist_weight,
            ],
            axis=2,
        ).astype(np.float32)
        erode_px = int(plate_cfg.get("ink_core_erode_px", 2))
        block_size = int(plate_cfg.get("ink_block_size", 4))
        bg_white_l_max = plate_cfg.get("ink_bg_white_l_max")
        bg_black_l_min = plate_cfg.get("ink_bg_black_l_min")
        ink_mask_full = masks["dot"] | masks["ring"]
        if bg_white_l_max is not None:
            ink_mask_full = ink_mask_full & (lab_white[..., 0] <= float(bg_white_l_max))
        if bg_black_l_min is not None:
            ink_mask_full = ink_mask_full & (lab_black[..., 0] >= float(bg_black_l_min))
        ring_core, _ = _split_core_transition(masks["ring"], geom, plate_cfg)
        dot_core, _ = _split_core_transition(masks["dot"], geom, plate_cfg)
        if erode_px > 0:
            ring_core = _erode_mask(ring_core, erode_px)
            dot_core = _erode_mask(dot_core, erode_px)
        k_expected = int(plate_cfg.get("ink_k", 3))
        ink_masks = {}
        if ring_core.sum() > 0:
            ink_masks["ink1"] = ring_core
        k_dot = max(1, k_expected - 1)
        if dot_core.sum() > 0 and k_dot > 0:
            dot_masks = _cluster_ink_masks_full(
                feature_map,
                dist_map,
                ink_mask_full,
                dot_core,
                k=k_dot,
                block_size=block_size,
            )
            start_idx = 2 if "ink1" in ink_masks else 1
            for offset, name in enumerate(sorted(dot_masks.keys()), start=start_idx):
                ink_masks[f"ink{offset}"] = dot_masks[name]

    return {
        "white_norm": w_norm,
        "black_aligned": b_aligned,
        "alpha": alpha,
        "masks": masks,
        "ink_masks": ink_masks,
        "geom": geom,
    }
