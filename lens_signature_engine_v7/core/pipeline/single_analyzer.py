"""
Single Sample Analysis Module

Analyzes a single sample without STD comparison.
Provides quality assessment, color distribution, pattern analysis, etc.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ..anomaly.angular_uniformity import angular_uniformity_score
from ..anomaly.blob_detector import detect_center_blobs
from ..anomaly.pattern_baseline import extract_pattern_features
from ..gate.gate_engine import run_gate
from ..geometry.lens_geometry import detect_lens_circle
from ..measure.preprocess import build_roi_mask
from ..signature.radial_signature import build_radial_signature, to_polar
from ..types import LensGeometry
from ..utils import apply_white_balance, bgr_to_lab


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
    from ..utils import bgr_to_lab, lab_cv8_to_cie

    # Create lens mask
    mask = _create_lens_mask(test_bgr, geom)

    # Convert to Lab (OpenCV 8-bit)
    test_lab_cv8 = bgr_to_lab(test_bgr)
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
        "uniformity": _calculate_radial_uniformity(test_mean_cie),
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

    Args:
        metadata: Metadata from build_color_masks_with_retry()
        total_pixels: Total ROI pixels for area_ratio calculation

    Returns:
        Legacy format dict compatible with existing single_analyzer consumers
    """
    from ..utils import lab_cv8_to_cie

    # Convert colors to clusters
    cluster_list = []
    for color_info in metadata.get("colors", []):
        # color_info has: {color_id, lab_centroid, hex_ref, area_ratio, role}
        # lab_centroid is in OpenCV scale [0-255, 0-255, 0-255]

        lab_cv8 = color_info["lab_centroid"]  # Already OpenCV scale from build_color_masks

        # Convert to CIE scale for consistency
        lab_cie = lab_cv8_to_cie(np.array([lab_cv8], dtype=np.float32))[0].tolist()

        # Extract cluster ID from color_id (e.g., "color_0" -> 0)
        cluster_id = int(color_info["color_id"].split("_")[1])

        cluster_list.append(
            {
                "id": cluster_id,
                "centroid_lab": [round(lab_cie[0], 2), round(lab_cie[1], 2), round(lab_cie[2], 2)],
                "centroid_lab_cv8": [round(lab_cv8[0], 2), round(lab_cv8[1], 2), round(lab_cv8[2], 2)],
                "centroid_lab_cie": [round(lab_cie[0], 2), round(lab_cie[1], 2), round(lab_cie[2], 2)],
                "pixel_count": int(color_info["area_ratio"] * total_pixels),
                "area_ratio": round(float(color_info["area_ratio"]), 3),
                "mean_hex": color_info["hex_ref"],
                "role": color_info.get("role", "ink"),  # Include role for debugging
            }
        )

    # Sort by L* (darker to lighter) for consistency
    cluster_list.sort(key=lambda x: x["centroid_lab"][0])

    # Re-assign IDs after sorting
    for i, cluster in enumerate(cluster_list):
        cluster["id"] = i

    return {
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


def _analyze_ink_segmentation(
    test_bgr: np.ndarray, geom: LensGeometry, cfg: Dict[str, Any], expected_k: int = 3
) -> Dict[str, Any]:
    """
    Analyze ink color clusters using k-means

    Returns:
        {
            "k": 3,
            "clusters": [
                {
                    "id": 0,
                    "centroid_lab": [L, a, b],
                    "pixel_count": 1234,
                    "area_ratio": 0.35,
                    "mean_hex": "#RRGGBB"
                },
                ...
            ],
            "confidence": 0.85
        }
    """
    # Use unified color_masks engine (with 2-pass retry and role classification)
    from ..measure.color_masks import build_color_masks_with_retry

    try:
        # Run color segmentation with retry logic
        color_masks, metadata = build_color_masks_with_retry(
            test_bgr,
            cfg,
            expected_k=expected_k,
            geom=geom,
            confidence_threshold=0.7,
            enable_retry=True,
        )

        # Calculate total ROI pixels for area_ratio conversion
        R = cfg.get("polar", {}).get("R", 260)
        T = cfg.get("polar", {}).get("T", 720)
        total_pixels = R * T  # Approximate - actual ROI may be smaller

        # Convert to legacy format
        result = _convert_color_masks_to_legacy_format(metadata, total_pixels)

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
    # Convert to Lab
    test_lab = bgr_to_lab(test_bgr)

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
    - Radial uniformity: 20%
    - Pattern quality: 15%
    - Zone uniformity: 15%
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

    # Radial score
    radial_uniformity = results.get("radial", {}).get("summary", {}).get("uniformity", 0)
    radial_score = radial_uniformity * 100

    # Pattern score (lower angular uniformity is better)
    angular_unif = results.get("pattern", {}).get("angular_uniformity", 1.0)
    pattern_score = (1.0 - angular_unif) * 100

    # Zone score
    zone_uniformity = results.get("zones", {}).get("zone_uniformity", 0)
    zone_score = zone_uniformity * 100

    # Weighted average
    quality = gate_score * 0.30 + color_score * 0.20 + radial_score * 0.20 + pattern_score * 0.15 + zone_score * 0.15

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

    # Radial warnings
    radial = results.get("radial", {})
    uniformity = radial.get("summary", {}).get("uniformity")
    if uniformity is None:
        warnings.append("RADIAL_UNIFORMITY_NOT_AVAILABLE (insufficient valid data)")
    elif uniformity < 0.6:
        warnings.append(f"Low radial uniformity ({uniformity:.2f})")

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
    - RECAPTURE: Gate fail OR quality_score < 60
    - HOLD: 60 <= quality_score < 75 (engineer review needed)
    - PASS: quality_score >= 75

    Returns:
        {
            "decision": "RECAPTURE" | "HOLD" | "PASS",
            "decision_reason_top2": ["REASON1", "REASON2"],
            "action": "Specific action guidance (Korean)",
            "quality_grade": "A" | "B" | "C" | "D" | "F",
            "severity": "LOW" | "MEDIUM" | "HIGH",
            "quality_score": float
        }
    """
    quality_score = results.get("quality_score", 0)
    warnings = results.get("warnings", [])
    gate_passed = results.get("gate", {}).get("passed", True)
    gate_guidance = results.get("gate", {}).get("scores", {}).get("_guidance", {})

    # Determine decision
    if not gate_passed or quality_score < 60:
        decision = "RECAPTURE"
        severity = "HIGH"
    elif quality_score < 75:
        decision = "HOLD"
        severity = "MEDIUM"
    else:
        decision = "PASS"
        severity = "LOW"

    # Determine quality grade
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

    # Extract top 2 reasons from warnings
    # Prioritize: Gate > Defect > Color > Radial > Zone
    priority_keywords = ["Gate:", "Center defects", "High color", "Low radial", "Low zone"]
    sorted_warnings = sorted(warnings, key=lambda w: next((i for i, p in enumerate(priority_keywords) if p in w), 999))
    top2_reasons = sorted_warnings[:2] if len(sorted_warnings) >= 2 else sorted_warnings

    # Generate action from gate guidance or fallback based on warnings
    action = ""
    if decision == "RECAPTURE":
        if gate_guidance:
            # Combine all guidance messages from gate
            action = " ".join(gate_guidance.values())
        else:
            # Fallback based on warnings
            if any("BLUR" in r or "흐립" in r for r in top2_reasons):
                action = "초점을 재조정하고 재촬영하세요."
            elif any("CENTER" in r or "중심" in r for r in top2_reasons):
                action = "렌즈를 화면 중앙으로 이동하고 재촬영하세요."
            elif any("defect" in r or "결함" in r for r in top2_reasons):
                action = "렌즈 표면을 청소하고 재촬영하세요."
            elif any("ILLUMINATION" in r or "조명" in r for r in top2_reasons):
                action = "조명을 균일하게 조정하고 재촬영하세요."
            else:
                action = "측정 환경을 점검하고 재촬영하세요."
    elif decision == "HOLD":
        action = "엔지니어 검토가 필요합니다. 데이터를 보관하세요."
    else:
        action = "측정 완료. 다음 샘플로 진행하세요."

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
        "radial_uniformity": radial_summary.get("uniformity", 0),
        "radial_slope": radial_summary.get("radial_slope", 0),
        "ring_contrast": radial_summary.get("ring_contrast", 0),
    }

    # Defect metrics
    blobs = pattern_data.get("center_blobs", {})
    defect_kpi = {"blob_count": blobs.get("blob_count", 0), "blob_total_area": blobs.get("total_area", 0)}

    return {"qc": qc_kpi, "ink": ink_kpi, "pattern": pattern_kpi, "defect": defect_kpi}


def analyze_single_sample(
    test_bgr: np.ndarray, cfg: Dict[str, Any], analysis_modes: Optional[List[str]] = None
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

    results = {}

    # Step 1: Geometry detection (always required)
    geom = detect_lens_circle(test_bgr)

    # Optional white balance
    wb_enabled = cfg.get("gate", {}).get("white_balance", {}).get("enabled", False)
    if wb_enabled:
        test_bgr, _ = apply_white_balance(test_bgr, geom, cfg)

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

    # Step 5: Ink segmentation
    if "ink" in analysis_modes:
        expected_k = cfg.get("expected_ink_count", 3)
        results["ink"] = _analyze_ink_segmentation(test_bgr, geom, cfg, expected_k)

    # Step 6: Pattern features
    if "pattern" in analysis_modes:
        results["pattern"] = _analyze_pattern_features(test_bgr, geom, cfg)

    # Step 7: Zone analysis
    if "zones" in analysis_modes:
        num_zones = cfg.get("zone_analysis", {}).get("num_zones", 8)
        results["zones"] = _analyze_zones_2d(test_bgr, geom, num_zones)

    # Step 8: Calculate quality score
    results["quality_score"] = _calculate_quality_score(results)

    # Step 9: Generate warnings
    results["warnings"] = _generate_warnings(results)

    # Step 10: Generate operator summary (NEW!)
    results["operator_summary"] = _determine_operator_decision(results, cfg)

    # Step 11: Extract engineer KPI (NEW!)
    results["engineer_kpi"] = _extract_engineer_kpi(results)

    return results
