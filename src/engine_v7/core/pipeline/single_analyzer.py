"""
Single Sample Analysis Module

Analyzes a single sample without STD comparison.
Provides quality assessment, color distribution, pattern analysis, etc.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from ..anomaly.angular_uniformity import angular_uniformity_score
from ..anomaly.blob_detector import detect_center_blobs
from ..anomaly.pattern_baseline import extract_pattern_features
from ..config_norm import get_polar_dims
from ..gate.gate_engine import run_gate
from ..geometry.lens_geometry import detect_lens_circle
from ..measure.segmentation.preprocess import build_roi_mask
from ..signature.radial_signature import build_radial_signature, to_polar
from ..types import LensGeometry
from ..utils import apply_white_balance, to_cie_lab


def _create_lens_mask(bgr: np.ndarray, geom: LensGeometry) -> np.ndarray:
    """Create circular mask for lens region"""
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(geom.cx), int(geom.cy)), int(geom.r), 255, -1)
    return mask


def _calculate_radial_uniformity(radial_mean: np.ndarray) -> float | None:
    """
    Calculate radial uniformity score (0-1, higher is better)

    Measures how uniform the radial profile is from center to edge.
    Uses coefficient of variation of L* channel.

    Returns:
        float: 0-1 uniformity score, or None if calculation not possible
    """
    L_profile = radial_mean[:, 0]

    # Filter out invalid values (0 or NaN from empty bins)
    valid_L = L_profile[~np.isnan(L_profile) & (L_profile > 0)]

    if len(valid_L) < 2:
        return None  # Not enough valid data

    mean_L = float(np.mean(valid_L))
    std_L = float(np.std(valid_L))

    if mean_L < 1.0:  # Too dark for reliable measurement
        return None

    # Coefficient of variation (lower is more uniform)
    cv = std_L / mean_L

    # Convert to 0-1 score (invert and normalize)
    # Typical CV for good samples: 0.05-0.15
    # Bad samples: 0.3+
    uniformity = max(0.0, 1.0 - (cv / 0.3))

    return round(uniformity, 3)


def _calculate_ring_contrast(radial_mean_cie: np.ndarray) -> float:
    """
    Calculate contrast between inner and outer rings.

    Uses L* channel difference between center 20% and outer 20%.
    Higher values indicate more contrast (e.g., darker center).

    Args:
        radial_mean_cie: Radial profile in CIE L*a*b* scale (N x 3)

    Returns:
        Normalized contrast [0-1]
    """
    num_points = len(radial_mean_cie)
    inner_zone = int(num_points * 0.2)
    outer_start = int(num_points * 0.8)

    inner_L = float(np.mean(radial_mean_cie[:inner_zone, 0]))
    outer_L = float(np.mean(radial_mean_cie[outer_start:, 0]))

    # Contrast as normalized difference (0-100 L* range)
    contrast = abs(inner_L - outer_L) / 100.0

    return round(contrast, 3)


def _calculate_radial_slope(radial_mean_cie: np.ndarray) -> float:
    """
    Calculate average slope of L* radial profile.

    Uses linear regression on L* vs normalized radius.
    Positive slope = brighter at edges
    Negative slope = darker at edges

    Args:
        radial_mean_cie: Radial profile in CIE L*a*b* scale (N x 3)

    Returns:
        Slope in L* units per normalized radius
    """
    L_profile = radial_mean_cie[:, 0]
    num_points = len(L_profile)

    # Normalized radius [0-1]
    r_norm = np.linspace(0, 1, num_points)

    # Linear regression: L = slope * r + intercept
    slope, _ = np.polyfit(r_norm, L_profile, 1)

    return round(float(slope), 2)


def _calculate_radial_smoothness(radial_mean_cie: np.ndarray) -> float:
    """
    Calculate radial smoothness score (0-1, higher is better).

    Measures how smooth the L* gradient is, NOT how flat.
    A lens with consistent center-to-edge gradient scores high.
    A lens with bumpy/inconsistent gradient scores low.

    Uses polynomial fit residuals - lower residuals = smoother gradient.

    Args:
        radial_mean_cie: Radial profile in CIE L*a*b* scale (N x 3)

    Returns:
        Smoothness score [0-1]
    """
    L_profile = radial_mean_cie[:, 0]
    num_points = len(L_profile)

    if num_points < 5:
        return 0.5  # Not enough data

    # Normalized radius [0-1]
    r_norm = np.linspace(0, 1, num_points)

    # Fit 3rd-degree polynomial (allows for gradual curves)
    coeffs = np.polyfit(r_norm, L_profile, 3)
    fitted = np.polyval(coeffs, r_norm)

    # Calculate RMSE of residuals
    residuals = L_profile - fitted
    rmse = float(np.sqrt(np.mean(residuals**2)))

    # Normalize: RMSE < 2 is very smooth, RMSE > 10 is bumpy
    # Score = 1 - (rmse / 10), clamped to [0, 1]
    smoothness = max(0.0, min(1.0, 1.0 - (rmse / 10.0)))

    return round(smoothness, 3)


def _calculate_zone_uniformity(zones: List[Dict]) -> float | None:
    """
    Calculate zone uniformity score (0-1, higher is better)

    Measures how similar different zones are in color.

    Returns:
        float: 0-1 uniformity score, or None if calculation not possible
    """
    if not zones or len(zones) < 2:
        return None

    # Extract L values from all zones
    L_values = [z["mean_lab"][0] for z in zones]

    # Filter out invalid values (0 or NaN from empty bins)
    valid_L = [v for v in L_values if v > 0 and not np.isnan(v)]

    if len(valid_L) < 2:
        return None

    mean_L = float(np.mean(valid_L))
    std_L = float(np.std(valid_L))

    if mean_L < 1.0:  # Too dark for reliable measurement
        return None

    # Coefficient of variation
    cv = std_L / mean_L

    # Convert to uniformity score
    uniformity = max(0.0, 1.0 - (cv / 0.2))

    return round(uniformity, 3)


def _analyze_color_histogram(test_bgr: np.ndarray, geom: LensGeometry, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze Lab color distribution in BOTH OpenCV 8-bit and CIE scales.

    Returns:
        {
            "L_cv8": {"mean": ..., "std": ..., ...},  # OpenCV 0-255 scale
            "a_cv8": {...},
            "b_cv8": {...},
            "L_cie": {"mean": ..., "std": ..., ...},  # CIE 0-100, -128~+127 scale
            "a_cie": {...},
            "b_cie": {...},
            "histogram_L": [...],
            "histogram_a": [...],
            "histogram_b": [...],
            "_scale_note": "cv8 = OpenCV (L:0-255, a/b:0-255), cie = CIE (L:0-100, a/b:-128~+127)"
        }
    """
    from ..utils import lab_cv8_to_cie, to_cie_lab

    # Create lens mask
    mask = _create_lens_mask(test_bgr, geom)

    # Convert to Lab (OpenCV 8-bit)
    test_lab_cv8 = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    test_lab_cie = lab_cv8_to_cie(test_lab_cv8)

    # Extract pixels within mask
    masked_lab_cv8 = test_lab_cv8[mask > 0]
    masked_lab_cie = test_lab_cie[mask > 0]

    if len(masked_lab_cv8) == 0:
        empty = {"mean": 0, "std": 0, "min": 0, "max": 0, "p05": 0, "p95": 0}
        return {
            "L_cv8": empty,
            "a_cv8": empty,
            "b_cv8": empty,
            "L_cie": empty,
            "a_cie": empty,
            "b_cie": empty,
            "histogram_L": [],
            "histogram_a": [],
            "histogram_b": [],
            "_scale_note": "cv8 = OpenCV (L:0-255, a/b:0-255), cie = CIE (L:0-100, a/b:-128~+127)",
        }

    result = {}

    # Process OpenCV 8-bit values
    for i, channel in enumerate(["L", "a", "b"]):
        values_cv8 = masked_lab_cv8[:, i]
        values_cie = masked_lab_cie[:, i]

        result[f"{channel}_cv8"] = {
            "mean": round(float(np.mean(values_cv8)), 2),
            "std": round(float(np.std(values_cv8)), 2),
            "min": round(float(np.min(values_cv8)), 2),
            "max": round(float(np.max(values_cv8)), 2),
            "p05": round(float(np.percentile(values_cv8, 5)), 2),
            "p95": round(float(np.percentile(values_cv8, 95)), 2),
        }

        result[f"{channel}_cie"] = {
            "mean": round(float(np.mean(values_cie)), 2),
            "std": round(float(np.std(values_cie)), 2),
            "min": round(float(np.min(values_cie)), 2),
            "max": round(float(np.max(values_cie)), 2),
            "p05": round(float(np.percentile(values_cie, 5)), 2),
            "p95": round(float(np.percentile(values_cie, 95)), 2),
        }

        # Histogram (use CIE values for histogram)
        hist, _ = np.histogram(values_cie, bins=50)
        result[f"histogram_{channel}"] = hist.tolist()

    result["_scale_note"] = "cv8 = OpenCV (L:0-255, a/b:0-255), cie = CIE (L:0-100, a/b:-128~+127)"

    return result


def _analyze_radial_profile(test_bgr: np.ndarray, geom: LensGeometry, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze radial signature profile

    Returns:
        {
            "profile": {
                "L_mean": [...],
                "a_mean": [...],
                "b_mean": [...],
                "L_std": [...],
                "a_std": [...],
                "b_std": [...]
            },
            "meta": {...},
            "summary": {
                "inner_mean_L": ...,
                "outer_mean_L": ...,
                "uniformity": ...
            }
        }
    """
    R = cfg.get("signature", {}).get("R", 100)
    T = cfg.get("signature", {}).get("T", 360)
    r_start = cfg.get("signature", {}).get("r_start", 0.3)
    r_end = cfg.get("signature", {}).get("r_end", 0.9)

    # Build polar representation
    polar = to_polar(test_bgr, geom, R=R, T=T)

    # Build radial signature
    test_mean, test_std, meta = build_radial_signature(polar, r_start=r_start, r_end=r_end)

    # Convert to CIE scale FIRST for proper statistics
    from ..utils import lab_cv8_to_cie

    test_mean_cie = lab_cv8_to_cie(test_mean)
    test_std_cie = lab_cv8_to_cie(test_std)

    # Calculate summary statistics using CIE scale
    # Use actual radial zones: inner = first 20%, outer = last 20%
    num_points = len(test_mean_cie)
    inner_zone = int(num_points * 0.2)  # First 20% of radial profile
    outer_start = int(num_points * 0.8)  # Last 20% of radial profile

    # Calculate actual radius ranges (normalized 0-1)
    r_span = r_end - r_start
    inner_r_start = r_start
    inner_r_end = r_start + r_span * 0.2  # First 20% of profile
    outer_r_start = r_start + r_span * 0.8  # Last 20% of profile
    outer_r_end = r_end

    summary = {
        "inner_mean_L": round(float(np.mean(test_mean_cie[:inner_zone, 0])), 2),
        "outer_mean_L": round(float(np.mean(test_mean_cie[outer_start:, 0])), 2),
        "uniformity": _calculate_radial_uniformity(test_mean_cie),  # Legacy, kept for compatibility
        "smoothness": _calculate_radial_smoothness(test_mean_cie),  # NEW: measures gradient consistency
        "ring_contrast": _calculate_ring_contrast(test_mean_cie),
        "radial_slope": _calculate_radial_slope(test_mean_cie),
        "_meta": {
            "inner_range_r": [round(inner_r_start, 2), round(inner_r_end, 2)],
            "outer_range_r": [round(outer_r_start, 2), round(outer_r_end, 2)],
            "inner_zone_pct": "0-20%",
            "outer_zone_pct": "80-100%",
            "r_start": r_start,
            "r_end": r_end,
            "scale": "CIE L*a*b*",
            "num_radial_points": num_points,
        },
    }

    return {
        "profile_cv8": {
            "L_mean": test_mean[:, 0].tolist(),
            "a_mean": test_mean[:, 1].tolist(),
            "b_mean": test_mean[:, 2].tolist(),
            "L_std": test_std[:, 0].tolist(),
            "a_std": test_std[:, 1].tolist(),
            "b_std": test_std[:, 2].tolist(),
        },
        "profile_cie": {
            "L_mean": test_mean_cie[:, 0].tolist(),
            "a_mean": test_mean_cie[:, 1].tolist(),
            "b_mean": test_mean_cie[:, 2].tolist(),
            "L_std": test_std_cie[:, 0].tolist(),
            "a_std": test_std_cie[:, 1].tolist(),
            "b_std": test_std_cie[:, 2].tolist(),
        },
        "profile": {  # Backward compatibility - use CIE for new code
            "L_mean": test_mean_cie[:, 0].tolist(),
            "a_mean": test_mean_cie[:, 1].tolist(),
            "b_mean": test_mean_cie[:, 2].tolist(),
        },
        "meta": meta,
        "summary": summary,
        "_scale_note": "profile_cv8 = OpenCV scale, profile_cie = CIE scale (recommended)",
    }


def _convert_color_masks_to_legacy_format(metadata: Dict[str, Any], total_pixels: int) -> Dict[str, Any]:
    """
    Convert build_color_masks() metadata to single_analyzer legacy format.

    Notes on Lab scales:
    - metadata["colors"][i]["lab_centroid"] is **OpenCV Lab (CV8, 0-255)** for backward compatibility.
    - metadata["colors"][i]["lab_centroid_cie"] (if present) is **CIE Lab (L*:0-100, a/b centered)**.

    This converter is robust to either input (it will auto-detect and convert).

    Args:
        metadata: Metadata from build_color_masks_with_retry()
        total_pixels: Total ROI pixels for area_ratio calculation

    Returns:
        Legacy format dict compatible with existing single_analyzer consumers
    """
    from ..utils import lab_cie_to_cv8, to_cie_lab

    def _to_cie_vec(lab_any) -> np.ndarray:
        """Accept (3,) list/array in either CV8 or CIE, return (3,) CIE."""
        arr = np.array(lab_any, dtype=np.float32).reshape(1, 1, 3)
        return to_cie_lab(arr, source="auto", validate=False)[0, 0]

    cluster_list = []
    for color_info in metadata.get("colors", []):
        # Prefer explicit CIE centroid if present
        if color_info.get("lab_centroid_cie") is not None:
            lab_cie = np.array(color_info["lab_centroid_cie"], dtype=np.float32)
        else:
            lab_cie = _to_cie_vec(color_info.get("lab_centroid", [0, 0, 0]))

        # Always derive a consistent CV8 representation from CIE
        lab_cv8 = lab_cie_to_cv8(lab_cie).astype(np.float32).tolist()

        cluster_id = int(color_info["color_id"].split("_")[1])

        cluster_entry = {
            "id": cluster_id,
            "cluster_id_original": cluster_id,  # Preserve original ID for tracking
            "centroid_lab": [round(float(lab_cie[0]), 2), round(float(lab_cie[1]), 2), round(float(lab_cie[2]), 2)],
            "centroid_lab_cv8": [round(float(lab_cv8[0]), 2), round(float(lab_cv8[1]), 2), round(float(lab_cv8[2]), 2)],
            "centroid_lab_cie": [round(float(lab_cie[0]), 2), round(float(lab_cie[1]), 2), round(float(lab_cie[2]), 2)],
            "pixel_count": color_info.get(
                "n_pixels", int(float(color_info.get("area_ratio", 0.0)) * total_pixels)
            ),  # Use actual pixels if available
            "area_ratio": round(float(color_info.get("area_ratio", 0.0)), 3),
            "mean_hex": color_info.get("hex_ref"),
            "role": color_info.get("role", "ink"),
        }
        intrinsic = color_info.get("intrinsic_color")
        if intrinsic:
            cluster_entry["intrinsic_alpha_y"] = intrinsic.get("alpha_y")
            cluster_entry["intrinsic_ink_rgb"] = intrinsic.get("ink_rgb")
            cluster_entry["intrinsic_k_rgb"] = intrinsic.get("k_rgb")
            cluster_entry["intrinsic_base_t"] = intrinsic.get("base_t")
            cluster_entry["intrinsic_warnings"] = intrinsic.get("warnings")
        # Add effective_density fields if computed
        if "effective_density" in color_info:
            cluster_entry["effective_density"] = color_info["effective_density"]
            cluster_entry["alpha_used"] = color_info.get("alpha_used")
            cluster_entry["alpha_fallback_level"] = color_info.get("alpha_fallback_level")
        cluster_list.append(cluster_entry)

    # Sort by L* (darker to lighter) for consistency
    cluster_list.sort(key=lambda x: x["centroid_lab"][0])

    # Re-assign display IDs after sorting (keep original IDs intact)
    for i, cluster in enumerate(cluster_list):
        cluster["display_order"] = i  # For UI display ordering
        # DO NOT overwrite cluster["id"] to preserve original tracking

    result = {
        "k": metadata.get("segmentation_k", metadata.get("k_used", len(cluster_list))),
        "clusters": cluster_list,
        "clustering_confidence": round(float(metadata.get("segmentation_confidence", 0.7)), 3),
        "_meta": {
            "expected_ink_count": metadata.get("expected_ink_count"),
            "detected_ink_like_count": metadata.get("detected_ink_like_count"),
            "segmentation_pass": metadata.get("segmentation_pass"),
            "retry_reason": metadata.get("retry_reason"),
            "total_pixels": total_pixels,
            "warnings": metadata.get("warnings", []),
        },
    }

    if "segmentation_method" in metadata:
        result["segmentation_method"] = metadata["segmentation_method"]
    if "primary_extraction" in metadata:
        result["primary_extraction"] = metadata["primary_extraction"]

    # Include sample_meta for Plate Gate tracking
    if "sample_meta" in metadata:
        result["sample_meta"] = metadata["sample_meta"]
        # Add quick flags for easy access
        sample_meta = metadata["sample_meta"]
        if sample_meta.get("plate_gate_applied"):
            result["_meta"]["plate_gate_applied"] = True
            result["_meta"]["plate_gate_rule"] = sample_meta.get("rule")
            result["_meta"]["plate_gate_pixels"] = sample_meta.get("n_pixels_used")

    return result


def _analyze_ink_segmentation(
    test_bgr: np.ndarray,
    geom: LensGeometry,
    cfg: Dict[str, Any],
    expected_k: int = 3,
    plate_ink_mask: Optional[np.ndarray] = None,
    plate_kpis: Optional[Dict[str, Any]] = None,
    polar_alpha: Optional[np.ndarray] = None,
    alpha_cfg: Optional[Dict[str, Any]] = None,
    black_bgr: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Analyze ink color clusters using k-means

    Args:
        polar_alpha: Optional alpha map in polar coordinates (T, R) for effective_density
        alpha_cfg: Optional alpha configuration for 3-tier fallback
    """
    # Use unified color_masks engine (with 2-pass retry and role classification)
    from ..measure.segmentation.color_masks import build_color_masks_with_retry, compute_cluster_effective_densities

    # Check if V2 primary extraction is enabled in config
    v2_cfg = cfg.get("v2_ink", {})
    use_primary_extraction = bool(v2_cfg.get("use_primary_extraction", False))

    try:
        # Run color segmentation with retry logic, Hard Gate (mask) and Soft Gate (KPIs)
        color_masks, metadata = build_color_masks_with_retry(
            test_bgr,
            cfg,
            expected_k=expected_k,
            geom=geom,
            confidence_threshold=0.7,
            enable_retry=True,
            use_primary_extraction=use_primary_extraction,
            plate_kpis=plate_kpis,
            sample_mask_override=plate_ink_mask,  # Hard Gate: Plate-based sampling restriction
        )

        # Compute effective densities if alpha available
        alpha_summary = None
        if polar_alpha is not None or alpha_cfg is not None:
            # Get alpha_cfg from config if not explicitly provided
            effective_alpha_cfg = dict(alpha_cfg) if alpha_cfg is not None else dict(cfg.get("alpha") or {})

            # P2-2: Pass moire_severity from registration-less alpha quality
            # This allows quality_fail checks to include moire severity
            if polar_alpha is not None and "_moire_severity" not in effective_alpha_cfg:
                # Check if moire was detected in registration-less alpha
                from ..measure.metrics.alpha_density import PolarAlphaResult

                # Moire severity would have been passed via config from registration-less result
                pass  # Will be set below if available

            if effective_alpha_cfg is not None or polar_alpha is not None:
                from ..signature.radial_signature import to_polar
                from ..utils import to_cie_lab as util_to_cie_lab

                R, T = get_polar_dims(cfg)
                polar = to_polar(test_bgr, geom, R=R, T=T)
                polar_lab = util_to_cie_lab(polar)
                metadata, alpha_summary = compute_cluster_effective_densities(
                    color_masks,
                    metadata,
                    polar_alpha=polar_alpha,
                    polar_lab=polar_lab,
                    alpha_cfg=effective_alpha_cfg,
                )

        # Compute intrinsic ink colors from white/black pair if available
        if black_bgr is not None:
            from ..measure.metrics.intrinsic_color import compute_intrinsic_colors
            from ..signature.radial_signature import to_polar

            R, T = get_polar_dims(cfg)
            black_aligned = black_bgr
            align_info = None
            align_enabled = bool((cfg.get("intrinsic_color") or {}).get("align_black", True))
            if align_enabled:
                try:
                    from ..plate.plate_engine import _phase_align_polar

                    black_aligned, align_info = _phase_align_polar(test_bgr, black_bgr, geom, T=T)
                except Exception as exc:
                    align_info = {"method": "align_failed", "error": str(exc)}

            polar_white = to_polar(test_bgr, geom, R=R, T=T)
            polar_black = to_polar(black_aligned, geom, R=R, T=T)
            intrinsic_by_color, intrinsic_meta = compute_intrinsic_colors(
                polar_white,
                polar_black,
                color_masks,
                cfg,
            )
            for color_info in metadata.get("colors", []):
                color_id = color_info.get("color_id")
                if color_id in intrinsic_by_color:
                    color_info["intrinsic_color"] = intrinsic_by_color[color_id]
            intrinsic_meta["alignment"] = align_info
            metadata["intrinsic_color"] = intrinsic_meta

        # Calculate total ROI pixels for area_ratio conversion
        R, T = get_polar_dims(cfg)
        total_pixels = R * T  # Approximate - actual ROI may be smaller

        # Convert to legacy format
        result = _convert_color_masks_to_legacy_format(metadata, total_pixels)

        # Add alpha summary if computed
        if alpha_summary is not None:
            result["alpha_summary"] = alpha_summary
        if metadata.get("alpha_analysis"):
            result["alpha_analysis"] = metadata["alpha_analysis"]
        if metadata.get("intrinsic_color"):
            result["intrinsic_color"] = metadata["intrinsic_color"]

        return result

    except Exception as e:
        # Fallback if segmentation fails
        return {
            "k": 0,
            "clusters": [],
            "clustering_confidence": 0.0,
            "error": f"Segmentation failed: {str(e)}",
            "_meta": {"exception": str(e)},
        }


def _lab_to_hex(lab: List[float]) -> str:
    """
    Convert OpenCV Lab (0-255 scale) to hex color (DEPRECATED).
    Use _lab_cie_to_hex() instead for CIE Lab.
    """
    # Very rough approximation - just for visualization
    L, a, b = lab

    # Normalize to 0-255 range (rough)
    r = int(np.clip(L + 1.5 * a, 0, 255))
    g = int(np.clip(L - 0.5 * a + 0.5 * b, 0, 255))
    b_val = int(np.clip(L - 2.0 * b, 0, 255))

    return f"#{r:02x}{g:02x}{b_val:02x}"


def _lab_cie_to_hex(lab_cie: List[float]) -> str:
    """
    Convert CIE L*a*b* to hex color using OpenCV.

    Args:
        lab_cie: [L*, a*, b*] where L*:0-100, a*/b*:-128~+127
    """
    from ..utils import lab_cie_to_cv8

    # Convert CIE to OpenCV 8-bit
    lab_cv8 = lab_cie_to_cv8(np.array([[lab_cie]], dtype=np.float32))

    # Convert Lab to BGR using OpenCV
    bgr = cv2.cvtColor(lab_cv8.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Extract BGR values
    b, g, r = bgr[0, 0]

    return f"#{r:02x}{g:02x}{b:02x}"


def _analyze_pattern_features(test_bgr: np.ndarray, geom: LensGeometry, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze pattern quality features

    Returns:
        {
            "contrast": 45.2,
            "edge_density": 0.68,
            "texture_uniformity": 0.82,
            "angular_uniformity": 0.15,
            "center_blobs": {
                "blob_count": 0,
                "total_area": 0,
                "blobs": []
            }
        }
    """
    R = cfg.get("signature", {}).get("R", 100)
    T = cfg.get("signature", {}).get("T", 360)

    # Extract pattern features
    try:
        sample_features = extract_pattern_features(test_bgr, cfg=cfg)
    except Exception:
        sample_features = {}

    # Build polar for angular uniformity
    polar = to_polar(test_bgr, geom, R=R, T=T)
    ang = angular_uniformity_score(
        polar, r_start=cfg.get("anomaly", {}).get("r_start", 0.3), r_end=cfg.get("anomaly", {}).get("r_end", 0.9)
    )

    # Detect center blobs
    blobs = detect_center_blobs(
        test_bgr,
        geom,
        frac=cfg.get("anomaly", {}).get("center_frac", 0.2),
        min_area=cfg.get("anomaly", {}).get("center_blob_min_area", 10),
    )

    # Calculate total area from individual blobs
    blob_list = blobs.get("blobs", [])
    total_area = sum([b.get("area", 0) for b in blob_list])
    blob_count = blobs.get("blob_count", 0)

    # Select top 3 representative blobs by area and add polar coordinates
    import math

    top_blobs = sorted(blob_list, key=lambda b: b.get("area", 0), reverse=True)[:3]
    representative_blobs = []
    for blob in top_blobs:
        cx, cy = blob.get("cx", geom.cx), blob.get("cy", geom.cy)
        dx, dy = cx - geom.cx, cy - geom.cy
        r_pixels = math.hypot(dx, dy)
        r_normalized = r_pixels / (geom.r + 1e-6)  # Normalized radius [0-1]
        theta_deg = math.degrees(math.atan2(dy, dx))  # [-180, 180]

        representative_blobs.append(
            {
                "area": int(blob.get("area", 0)),
                "eccentricity": blob.get("eccentricity", 0.0),
                "r": round(r_normalized, 3),
                "theta_deg": round(theta_deg, 1),
                "cx_px": round(cx, 1),
                "cy_px": round(cy, 1),
            }
        )

    return {
        "contrast": round(float(sample_features.get("contrast", 0)), 2),
        "edge_density": round(float(sample_features.get("edge_density", 0)), 3),
        "texture_uniformity": round(float(sample_features.get("texture_uniformity", 0)), 3),
        "angular_uniformity": round(float(ang), 3),
        "center_blobs": {
            "blob_count": int(blob_count),
            "total_area": int(total_area),
            "top_3_blobs": representative_blobs,  # ✅ 대표 3개 blob with (area, eccentricity, r, θ)
            "_debug": {
                "roi_box": blobs.get("roi_box", []),
                "avg_area_per_blob": round(total_area / blob_count, 2) if blob_count > 0 else 0,
            },
        },
    }


def _analyze_zones_2d(test_bgr: np.ndarray, geom: LensGeometry, num_zones: int = 8) -> Dict[str, Any]:
    """
    Analyze zones in 2D (angular sectors)

    Returns:
        {
            "num_zones": 8,
            "zones": [
                {
                    "zone_id": 0,
                    "angle_range": [0, 45],
                    "mean_lab": [L, a, b],
                    "std_lab": [L, a, b],
                    "pixel_count": 1234
                },
                ...
            ],
            "zone_uniformity": 0.89
        }
    """
    # Convert to Lab (CIE scale)
    test_lab = to_cie_lab(test_bgr)

    # Create polar coordinates
    h, w = test_bgr.shape[:2]
    y, x = np.ogrid[:h, :w]

    cx, cy, r = geom.cx, geom.cy, geom.r

    # Calculate angle and radius for each pixel
    dx = x - cx
    dy = y - cy
    angles = np.arctan2(dy, dx) * 180 / np.pi  # -180 to 180
    angles = (angles + 360) % 360  # 0 to 360
    radii = np.sqrt(dx**2 + dy**2)

    # Create lens mask
    lens_mask = radii <= r

    # Divide into zones
    zone_angle = 360.0 / num_zones
    zones = []

    for i in range(num_zones):
        angle_start = i * zone_angle
        angle_end = (i + 1) * zone_angle

        # Create zone mask
        if angle_end <= 360:
            zone_mask = (angles >= angle_start) & (angles < angle_end) & lens_mask
        else:
            # Handle wrap-around
            zone_mask = ((angles >= angle_start) | (angles < (angle_end - 360))) & lens_mask

        # Extract pixels in this zone
        zone_pixels = test_lab[zone_mask]

        if len(zone_pixels) == 0:
            continue

        # Calculate statistics
        mean_lab = np.mean(zone_pixels, axis=0)
        std_lab = np.std(zone_pixels, axis=0)

        zones.append(
            {
                "zone_id": i,
                "angle_range": [round(angle_start, 1), round(angle_end, 1)],
                "mean_lab": [round(float(mean_lab[0]), 2), round(float(mean_lab[1]), 2), round(float(mean_lab[2]), 2)],
                "std_lab": [round(float(std_lab[0]), 2), round(float(std_lab[1]), 2), round(float(std_lab[2]), 2)],
                "pixel_count": int(np.sum(zone_mask)),
            }
        )

    # Calculate zone uniformity
    uniformity = _calculate_zone_uniformity(zones)

    return {"num_zones": num_zones, "zones": zones, "zone_uniformity": uniformity}


def _calculate_quality_score(results: Dict[str, Any]) -> float:
    """
    Calculate overall quality score (0-100)

    Weights (C. None score 재정규화):
    - Gate: 30% (재정규화: None인 score는 제외)
    - Color consistency: 20%
    - Pattern quality: 20% (angular uniformity - how uniform around 360°)
    - Zone uniformity: 15%


    Note: Soft metrics (e.g., radial_smoothness) are stored only and excluded from scoring.
    """

    # C. Gate score - None 재정규화 평균 (decision_builder 방식)
    def _weighted_gate_score(gate_scores: dict) -> float:
        """None인 score는 가중치 재정규화로 제외"""
        items = []
        if gate_scores.get("center_score") is not None:
            items.append((gate_scores["center_score"], 0.4))
        if gate_scores.get("sharpness_score") is not None:
            items.append((gate_scores["sharpness_score"], 0.3))
        if gate_scores.get("illumination_score") is not None:
            items.append((gate_scores["illumination_score"], 0.3))
        if not items:
            return 0.0
        wsum = sum(w for _, w in items)
        return sum(v * w for v, w in items) / wsum

    gate_passed = results.get("gate", {}).get("passed", False)
    gate_raw_scores = results.get("gate", {}).get("scores", {})
    # C. 재정규화된 gate 점수 (0-1 scale)
    gate_normalized = _weighted_gate_score(gate_raw_scores)
    # passed=False면 0점 처리 (품질 게이트 실패)
    gate_score = gate_normalized * 100.0 if gate_passed else 0.0

    # Color score (lower std is better)
    color_data = results.get("color", {})
    L_std = color_data.get("L_cie", {}).get("std", color_data.get("L", {}).get("std", 50))
    # Good samples typically have L_std < 10, bad samples > 20
    color_score = max(0.0, min(100.0, 100.0 - (L_std - 5) * 5))

    # Radial smoothness is a soft metric (stored only; not used in scoring).

    # Pattern score (lower angular uniformity is better - more uniform pattern)
    angular_unif = results.get("pattern", {}).get("angular_uniformity", 1.0)
    pattern_score = (1.0 - angular_unif) * 100

    # Zone score
    zone_uniformity = results.get("zones", {}).get("zone_uniformity", 0)
    zone_score = zone_uniformity * 100

    # Weighted average (Gate 30%, Color 20%, Pattern 20%, Zone 15%), normalized to 100
    weight_total = 0.30 + 0.20 + 0.20 + 0.15
    quality = (gate_score * 0.30 + color_score * 0.20 + pattern_score * 0.20 + zone_score * 0.15) / weight_total

    return round(quality, 1)


def _generate_warnings(results: Dict[str, Any]) -> List[str]:
    """Generate warnings based on analysis results"""
    warnings = []

    # Gate warnings
    gate = results.get("gate", {})
    if not gate.get("passed", True):
        for reason in gate.get("reasons", []):
            warnings.append(f"Gate: {reason}")

    # Color warnings
    color = results.get("color", {})
    L_std = color.get("L", {}).get("std", 0)
    if L_std > 15:
        warnings.append(f"High color variation (L* std: {L_std:.1f})")

    # Radial warnings - use smoothness (gradient consistency) instead of uniformity
    radial = results.get("radial", {})
    smoothness = radial.get("summary", {}).get("smoothness")
    if smoothness is None:
        warnings.append("RADIAL_SMOOTHNESS_NOT_AVAILABLE (insufficient valid data)")
    elif smoothness < 0.7:
        warnings.append(f"Low radial smoothness ({smoothness:.2f}) - inconsistent gradient")

    # Pattern warnings
    pattern = results.get("pattern", {})
    blob_count = pattern.get("center_blobs", {}).get("blob_count", 0)
    if blob_count > 0:
        warnings.append(f"Center defects detected ({blob_count} blobs)")

    # Zone warnings
    zones = results.get("zones", {})
    zone_unif = zones.get("zone_uniformity")
    if zone_unif is None:
        warnings.append("ZONE_UNIFORMITY_NOT_AVAILABLE (insufficient valid data)")
    elif zone_unif < 0.7:
        warnings.append(f"Low zone uniformity ({zone_unif:.2f})")

    return warnings


def _determine_operator_decision(results: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate operator-friendly decision summary.

    Decision rules:
    - RECAPTURE: Gate fail only
    - PASS_QC: Gate passed (quality metrics are informational)

    Returns:
        {
            "decision": "RECAPTURE" | "PASS_QC",
            "decision_reason_top2": ["REASON1", "REASON2"],
            "action": "Specific action guidance (Korean)",
            "quality_grade": "A" | "B" | "C" | "D" | "F",
            "severity": "LOW" | "HIGH",
            "quality_score": float
        }
    """
    quality_score = results.get("quality_score", 0)
    warnings = results.get("warnings", [])
    gate_passed = results.get("gate", {}).get("passed", True)
    gate_guidance = results.get("gate", {}).get("scores", {}).get("_guidance", {})

    if not gate_passed:
        decision = "RECAPTURE"
        severity = "HIGH"
    else:
        decision = "PASS_QC"
        severity = "LOW"

    if quality_score >= 90:
        grade = "A"
    elif quality_score >= 80:
        grade = "B"
    elif quality_score >= 70:
        grade = "C"
    elif quality_score >= 60:
        grade = "D"
    else:
        grade = "F"

    if gate_passed:
        top2_reasons = []
    else:
        priority_keywords = ["Gate:", "Center defects", "High color", "Low radial", "Low zone"]
        sorted_warnings = sorted(
            warnings, key=lambda w: next((i for i, p in enumerate(priority_keywords) if p in w), 999)
        )
        top2_reasons = sorted_warnings[:2] if len(sorted_warnings) >= 2 else sorted_warnings

    action = ""
    if decision == "RECAPTURE":
        if gate_guidance:
            action = " ".join(gate_guidance.values())
        else:
            if any("BLUR" in r or "SHARPNESS" in r for r in top2_reasons):
                action = "이미지가 흐릿합니다. 초점을 재조정하고 다시 촬영하세요."
            elif any("CENTER" in r or "OFFSET" in r for r in top2_reasons):
                action = "렌즈가 중심에서 벗어났습니다. 위치를 조정한 후 다시 촬영하세요."
            elif any("defect" in r or "blob" in r for r in top2_reasons):
                action = "렌즈에 이물질이나 결함이 감지되었습니다. 확인 후 다시 촬영하세요."
            elif any("ILLUMINATION" in r or "asymmetry" in r for r in top2_reasons):
                action = "조명이 불균형합니다. 조명 환경을 확인하고 다시 촬영하세요."
            else:
                action = "측정 품질이 낮습니다. 촬영 환경을 확인하고 다시 촬영하세요."
    else:
        action = "측정 품질이 양호합니다."

    return {
        "decision": decision,
        "decision_reason_top2": top2_reasons,
        "action": action,
        "quality_grade": grade,
        "severity": severity,
        "quality_score": round(quality_score, 1),
    }


def _extract_engineer_kpi(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract top 10 KPI metrics for engineer analysis.

    Returns:
        {
            "qc": {
                "sharpness": float,
                "center_offset_mm": float,
                "illumination_asymmetry": float
            },
            "ink": {
                "detected_count": int,
                "clustering_confidence": float,
                "ink_area_ratios": [float, ...]
            },
            "pattern": {
                "angular_uniformity": float,
                "zone_uniformity": float,
                "radial_uniformity": float,
                "radial_slope": float,
                "ring_contrast": float
            },
            "defect": {
                "blob_count": int,
                "blob_total_area": int
            }
        }
    """
    gate_scores = results.get("gate", {}).get("scores", {})
    ink_data = results.get("ink", {})
    pattern_data = results.get("pattern", {})
    radial_data = results.get("radial", {})
    zones_data = results.get("zones", {})

    # QC metrics
    qc_kpi = {
        "sharpness": gate_scores.get("sharpness_score", 0),
        "center_offset_mm": gate_scores.get("center_offset_mm", 0),
        "illumination_asymmetry": gate_scores.get("illumination_asymmetry", 0),
    }

    # Ink metrics
    clusters = ink_data.get("clusters", [])
    ink_kpi = {
        "detected_count": len(clusters),
        "clustering_confidence": ink_data.get("clustering_confidence", 0),
        "ink_area_ratios": [c.get("area_ratio", 0) for c in clusters],
    }

    # Pattern metrics
    radial_summary = radial_data.get("summary", {})
    pattern_kpi = {
        "angular_uniformity": pattern_data.get("angular_uniformity", 0),
        "zone_uniformity": zones_data.get("zone_uniformity", 0),
        "radial_uniformity": radial_summary.get("uniformity", 0),  # Legacy, kept for compatibility
        "radial_smoothness": radial_summary.get("smoothness", 0),  # NEW: gradient consistency
        "radial_slope": radial_summary.get("radial_slope", 0),
        "ring_contrast": radial_summary.get("ring_contrast", 0),
    }

    # Defect metrics
    blobs = pattern_data.get("center_blobs", {})
    defect_kpi = {"blob_count": blobs.get("blob_count", 0), "blob_total_area": blobs.get("total_area", 0)}

    return {"qc": qc_kpi, "ink": ink_kpi, "pattern": pattern_kpi, "defect": defect_kpi}


def analyze_single_sample(
    test_bgr: np.ndarray,
    cfg: Dict[str, Any],
    analysis_modes: Optional[List[str]] = None,
    black_bgr: Optional[np.ndarray] = None,
    match_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze single sample without STD comparison

    Args:
        test_bgr: Test image (BGR)
        cfg: Configuration dict
        analysis_modes: List of modes to run (None = all)
                       ["gate", "color", "radial", "ink", "pattern", "zones"]

    Returns:
        {
            "gate": {...},
            "color": {...},
            "radial": {...},
            "ink": {...},
            "pattern": {...},
            "zones": {...},
            "quality_score": 82.5,
            "warnings": [...]
        }
    """
    if analysis_modes is None:
        analysis_modes = ["gate", "color", "radial", "ink", "pattern", "zones"]

    if black_bgr is not None and "plate" not in analysis_modes:
        analysis_modes = list(analysis_modes) + ["plate"]

    plate_lite_cfg = cfg.get("plate_lite", {}) if isinstance(cfg, dict) else {}
    plate_lite_enabled = bool(plate_lite_cfg.get("enabled", False))
    plate_lite_override = bool(plate_lite_cfg.get("override_plate", False))

    results = {}
    extra_warnings: List[str] = []

    # Step 1: Geometry detection (always required)
    geom = detect_lens_circle(test_bgr)

    # Optional white balance
    wb_enabled = cfg.get("gate", {}).get("white_balance", {}).get("enabled", False)
    if wb_enabled and black_bgr is None:
        test_bgr, _ = apply_white_balance(test_bgr, geom, cfg)
    elif wb_enabled and black_bgr is not None:
        extra_warnings.append("white_balance_skipped_for_plate_pair")

    # Step 2: Gate analysis
    if "gate" in analysis_modes:
        gate = run_gate(
            geom,
            test_bgr,
            center_off_max=cfg.get("gate", {}).get("center_off_max", 0.5),
            blur_min=cfg.get("gate", {}).get("blur_min", 0.7),
            illum_max=cfg.get("gate", {}).get("illum_max", 0.15),
        )

        results["gate"] = {
            "passed": gate.passed,
            "geometry": {"cx": round(geom.cx, 2), "cy": round(geom.cy, 2), "r": round(geom.r, 2)},
            "scores": gate.scores,
            "reasons": gate.reasons,
        }

    # Step 3: Color histogram
    if "color" in analysis_modes:
        results["color"] = _analyze_color_histogram(test_bgr, geom, cfg)

    # Step 4: Radial profile
    if "radial" in analysis_modes:
        results["radial"] = _analyze_radial_profile(test_bgr, geom, cfg)

    # Step 5: Plate analysis (MOVED BEFORE INK for Plate Gate)
    # Plate masks can be used to restrict ink segmentation sampling
    plate_ink_mask = None
    polar_alpha = None  # Alpha map for effective_density computation
    polar_alpha_quality = None  # Quality metrics for registration-less alpha

    # P1-1: Registration-less Polar Alpha option
    # When enabled, compute alpha directly in polar coordinates without 2D registration
    alpha_cfg = cfg.get("alpha", {})
    registrationless_enabled = bool(alpha_cfg.get("registrationless_enabled", False))

    if registrationless_enabled and black_bgr is not None:
        from ..measure.metrics.alpha_density import build_polar_alpha_registrationless

        polar_R, polar_T = get_polar_dims(cfg)

        try:
            alpha_result = build_polar_alpha_registrationless(
                white_bgr=test_bgr,
                black_bgr=black_bgr,
                geom=geom,
                polar_R=polar_R,
                polar_T=polar_T,
                alpha_clip_min=float(alpha_cfg.get("clip_min", 0.02)),
                alpha_clip_max=float(alpha_cfg.get("clip_max", 0.98)),
                moire_detection_enabled=bool(alpha_cfg.get("moire_detection_enabled", True)),
                moire_threshold=float(alpha_cfg.get("moire_threshold", 0.15)),
            )

            polar_alpha = alpha_result.polar_alpha
            polar_alpha_quality = alpha_result.quality

            # Store quality info in results
            # P2-1: Include geometry confidence for debugging
            results["alpha_registrationless"] = {
                "method": alpha_result.meta["method"],
                "quality": alpha_result.quality,
                "meta": alpha_result.meta,
                "radial_profile_summary": {
                    "mean": float(np.nanmean(alpha_result.radial_profile)),
                    "std": float(np.nanstd(alpha_result.radial_profile)),
                    "min": float(np.nanmin(alpha_result.radial_profile)),
                    "max": float(np.nanmax(alpha_result.radial_profile)),
                },
                "geometry_confidence": {
                    "center_confidence": geom.center_confidence,
                    "radius_confidence": geom.radius_confidence,
                    "center_offset_ratio": geom.center_offset_ratio,
                    "source": geom.source,
                },
            }

            # P2-1: Warn if geometry confidence is low
            if geom.center_confidence < 0.7:
                extra_warnings.append(
                    f"LOW_CENTER_CONFIDENCE: {geom.center_confidence:.2f} - alpha may be spatially shifted"
                )
            if geom.radius_confidence < 0.5:
                extra_warnings.append(
                    f"LOW_RADIUS_CONFIDENCE: {geom.radius_confidence:.2f}"
                    " - alpha radial profile may be scaled incorrectly"
                )

            logger.debug(
                f"Registration-less alpha: quality={alpha_result.quality['overall']:.3f}, "
                f"nan={alpha_result.quality['nan_ratio']:.1%}, "
                f"clip={alpha_result.quality['clip_ratio']:.1%}"
            )

        except Exception as e:
            logger.warning(f"Registration-less alpha computation failed: {e}")
            results["alpha_registrationless"] = {"error": str(e)}

    if "plate" in analysis_modes and black_bgr is not None and not (plate_lite_enabled and plate_lite_override):
        from ..plate.plate_engine import analyze_plate_pair

        plate_cfg = cfg.get("plate", {})
        results["plate"] = analyze_plate_pair(
            white_bgr=test_bgr,
            black_bgr=black_bgr,
            cfg=plate_cfg,
            match_id=match_id,
            geom_hint=geom,
        )

        # Extract ink mask for Plate Gate (Hard Gate)
        # Use ink_mask_core_polar directly to avoid coordinate transform issues
        plate_masks = results["plate"].get("_masks")

        # Extract alpha_polar for effective_density computation
        # P1-1: Only use plate's alpha if registration-less was not computed
        # P1-2: When both available, plate alpha is used for verification only
        plate_alpha_polar = plate_masks.get("alpha_polar") if plate_masks is not None else None

        if polar_alpha is not None and plate_alpha_polar is not None:
            # P1-2: Verify registration-less alpha against plate alpha
            from ..measure.metrics.alpha_density import verify_alpha_agreement

            verification_cfg = alpha_cfg.get("verification", {})
            try:
                verification_result = verify_alpha_agreement(
                    registrationless_alpha=polar_alpha,
                    plate_alpha=plate_alpha_polar,
                    rmse_threshold=float(verification_cfg.get("rmse_threshold", 0.15)),
                    correlation_threshold=float(verification_cfg.get("correlation_threshold", 0.7)),
                    agreement_threshold=float(verification_cfg.get("agreement_threshold", 0.7)),
                )

                results["alpha_verification"] = {
                    "passed": verification_result.passed,
                    "agreement_score": verification_result.agreement_score,
                    "rmse": verification_result.rmse,
                    "correlation": verification_result.correlation,
                    "summary": verification_result.summary,
                    "warnings": verification_result.warnings,
                }

                if not verification_result.passed:
                    extra_warnings.append(
                        f"ALPHA_VERIFICATION_FAILED: rmse={verification_result.rmse:.3f}, "
                        f"corr={verification_result.correlation:.3f}, "
                        f"agreement={verification_result.agreement_score:.3f}"
                    )
                    logger.warning(f"Alpha verification failed: {verification_result.warnings}")
                else:
                    logger.debug(f"Alpha verification passed: agreement={verification_result.agreement_score:.3f}")
            except Exception as e:
                logger.warning(f"Alpha verification failed with error: {e}")
                results["alpha_verification"] = {"error": str(e)}

            # Keep using registration-less as primary (plate is verification only)
            logger.debug("Using registration-less alpha (plate alpha used for verification only)")
        elif polar_alpha is None and plate_alpha_polar is not None:
            # Fallback: use plate alpha if registration-less not computed
            polar_alpha = plate_alpha_polar
            logger.debug("Using plate registration-based alpha (registration-less not enabled)")

        if plate_masks is not None:
            plate_ink_mask = plate_masks.get("ink_mask_core_polar")

            # [Safety Check] Shape & Dtype validation
            if plate_ink_mask is not None:
                # Ensure boolean type
                if plate_ink_mask.dtype != bool:
                    plate_ink_mask = plate_ink_mask > 0

                # Ensure shape matches current polar config (T, R)
                # Note: We don't have polar map here yet, but we know T, R from config
                expected_R, expected_T = get_polar_dims(cfg)

                if plate_ink_mask.shape != (expected_T, expected_R):
                    # Try transpose if T/R swapped (common issue)
                    if plate_ink_mask.shape == (expected_R, expected_T):
                        plate_ink_mask = plate_ink_mask.T
                    else:
                        # Shape mismatch (e.g. different R/T configs) -> Disable Gate
                        # logger.warning(f"Plate Gate Shape Mismatch: {plate_ink_mask.shape}")  # noqa: E501
                        plate_ink_mask = None

                # Debug Logging
                if plate_ink_mask is not None:
                    logger.debug(f"Plate Ink Mask: Shape={plate_ink_mask.shape}, Sum={np.sum(plate_ink_mask)}")

    if plate_lite_enabled and black_bgr is not None:
        from ..plate.plate_engine import analyze_plate_lite_pair

        plate_cfg = cfg.get("plate", {})
        results["plate_lite"] = analyze_plate_lite_pair(
            white_bgr=test_bgr,
            black_bgr=black_bgr,
            lite_cfg=plate_lite_cfg,
            plate_cfg=plate_cfg,
            match_id=match_id,
            geom_hint=geom,
            expected_k=cfg.get("expected_ink_count"),
        )

    # Step 6: Ink segmentation (with Plate Gate if available)
    if "ink" in analysis_modes:
        expected_k = cfg.get("expected_ink_count", 3)
        # Extract Plate KPIs if available for Soft Gate
        plate_kpis = None
        if "plate" in results:
            plate_kpis = results["plate"].get("kpis") or results["plate"].get("masks_summary")

        # Extract alpha_cfg from config for effective_density computation
        # P2-2: Include moire_severity from registration-less alpha for quality_fail check
        alpha_cfg = dict(cfg.get("alpha") or {})
        if "alpha_registrationless" in results:
            alpha_quality = results["alpha_registrationless"].get("quality", {})
            if "moire_severity" in alpha_quality:
                alpha_cfg["_moire_severity"] = alpha_quality["moire_severity"]

        # P2-3: Pass verification agreement for confidence adjustment
        if "alpha_verification" in results:
            verification = results["alpha_verification"]
            if not verification.get("error"):
                alpha_cfg["_verification_enabled"] = True
                alpha_cfg["_verification_agreement"] = verification.get("agreement_score", 1.0)

        results["ink"] = _analyze_ink_segmentation(
            test_bgr,
            geom,
            cfg,
            expected_k,
            plate_ink_mask=plate_ink_mask,
            plate_kpis=plate_kpis,
            polar_alpha=polar_alpha,
            alpha_cfg=alpha_cfg,
            black_bgr=black_bgr,
        )

        # Step 6.5: Soft Gate - Adjust segmentation confidence based on plate KPIs
        # When Hard Gate isn't used or as additional validation
        if "plate" in results and results["ink"]:
            plate_kpis = results["plate"].get("masks_summary", {})

            # Key quality indicators from plate analysis
            artifact_ratio = plate_kpis.get("mask_artifact_ratio_valid", 0.0)
            leak_ratio = plate_kpis.get("outer_rim_leak_ratio", 0.0)

            # Soft Gate thresholds (can be configured in v2_ink section)
            v2_cfg = cfg.get("v2_ink", {})
            artifact_threshold = float(v2_cfg.get("soft_gate_artifact_th", 0.15))
            leak_threshold = float(v2_cfg.get("soft_gate_leak_th", 0.10))

            soft_gate_penalty = 0.0
            soft_gate_warnings = []

            # Apply penalty if plate quality indicators are poor
            if artifact_ratio > artifact_threshold:
                penalty = min(0.2, (artifact_ratio - artifact_threshold) * 1.0)
                soft_gate_penalty += penalty
                soft_gate_warnings.append(f"SOFT_GATE:high_artifact_ratio={artifact_ratio:.2f}>{artifact_threshold}")

            if leak_ratio > leak_threshold:
                penalty = min(0.2, (leak_ratio - leak_threshold) * 2.0)
                soft_gate_penalty += penalty
                soft_gate_warnings.append(f"SOFT_GATE:high_outer_rim_leak={leak_ratio:.2f}>{leak_threshold}")

            # Apply confidence penalty and add warnings
            if soft_gate_penalty > 0:
                current_conf = results["ink"].get("confidence", 1.0)
                adjusted_conf = max(0.0, current_conf - soft_gate_penalty)
                results["ink"]["confidence"] = round(adjusted_conf, 3)
                results["ink"]["soft_gate"] = {
                    "applied": True,
                    "artifact_ratio": round(artifact_ratio, 4),
                    "leak_ratio": round(leak_ratio, 4),
                    "penalty": round(soft_gate_penalty, 3),
                    "original_confidence": round(current_conf, 3),
                }
                results["ink"].setdefault("warnings", []).extend(soft_gate_warnings)
            else:
                results["ink"]["soft_gate"] = {"applied": False}

    # Step 7: Pattern features
    if "pattern" in analysis_modes:
        results["pattern"] = _analyze_pattern_features(test_bgr, geom, cfg)

    # Step 8: Zone analysis
    if "zones" in analysis_modes:
        num_zones = cfg.get("zone_analysis", {}).get("num_zones", 8)
        results["zones"] = _analyze_zones_2d(test_bgr, geom, num_zones)

    # Step 8: Calculate quality score
    results["quality_score"] = _calculate_quality_score(results)

    # Step 9: Generate warnings
    results["warnings"] = _generate_warnings(results)
    if extra_warnings:
        results["warnings"].extend(extra_warnings)

    # Step 10: Generate operator summary (NEW!)
    results["operator_summary"] = _determine_operator_decision(results, cfg)

    # Step 12: Extract engineer KPI (NEW!)
    results["engineer_kpi"] = _extract_engineer_kpi(results)

    return results
