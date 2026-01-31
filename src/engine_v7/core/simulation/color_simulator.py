"""
Color Simulator Module for Digital Proofing

Calculates perceived lens color by simulating spatial mixing of ink and background.
Uses Linear RGB mixing for optical accuracy.

Phase 2 Enhancements:
- Model metadata with version and assumptions
- Black background: measured > simulated priority
- observed.on_black from plate pair analysis
- model_error.delta_e_00 for validation/learning

Phase A (Longterm Roadmap) Enhancements:
- Mask-based pixel synthesis option (Direction A)
- Overlap detection and zone contribution analysis
- Feature flag: simulation_method = "area_ratio" | "mask_based"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils import to_cie_lab
from .validation import _interpret_delta_e, calculate_delta_e_2000

# =============================================================================
# Internal Conversion Functions
# =============================================================================


def _lab_to_linear_rgb(lab_cie: List[float]) -> np.ndarray:
    """
    Convert CIE Lab [L, a, b] to Linear RGB [R, G, B] (0.0 - 1.0).
    Uses OpenCV for intermediate conversions.
    """
    lab_arr = np.array([[lab_cie]], dtype=np.float32)
    rgb_gamma = cv2.cvtColor(lab_arr, cv2.COLOR_Lab2RGB)[0, 0]  # 0..1 float

    # Inverse Gamma Correction (sRGB -> Linear RGB)
    linear_rgb = np.where(rgb_gamma <= 0.04045, rgb_gamma / 12.92, np.power((rgb_gamma + 0.055) / 1.055, 2.4))
    return linear_rgb


def _linear_rgb_to_lab(linear_rgb: np.ndarray) -> List[float]:
    """
    Convert Linear RGB [R, G, B] (0.0 - 1.0) to CIE Lab.
    """
    srgb = np.where(
        linear_rgb <= 0.0031308, linear_rgb * 12.92, 1.055 * np.power(np.maximum(linear_rgb, 0), 1.0 / 2.4) - 0.055
    )
    srgb = np.clip(srgb, 0.0, 1.0).astype(np.float32)
    srgb_arr = srgb.reshape(1, 1, 3)
    lab = cv2.cvtColor(srgb_arr, cv2.COLOR_RGB2Lab)[0, 0]
    return [float(lab[0]), float(lab[1]), float(lab[2])]


def _lab_to_hex(lab_cie: List[float]) -> str:
    """Convert CIE Lab to Hex string for UI display."""
    lab_arr = np.array([[lab_cie]], dtype=np.float32)
    rgb = cv2.cvtColor(lab_arr, cv2.COLOR_Lab2RGB)[0, 0]
    rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return f"#{rgb_u8[0]:02x}{rgb_u8[1]:02x}{rgb_u8[2]:02x}"


def _ensure_cie_lab(lab: List[float]) -> List[float]:
    """
    Phase 2-4: Lab scale defense using existing to_cie_lab utility.
    Ensures Lab values are in CIE scale (L: 0-100, a/b: -128~127).
    """
    lab_arr = np.array([[lab]], dtype=np.float32)
    # to_cie_lab auto-detects and converts OpenCV scale if needed
    lab_cie = to_cie_lab(lab_arr, validate=True)[0, 0]
    return [float(lab_cie[0]), float(lab_cie[1]), float(lab_cie[2])]


def _match_zones_to_inks_by_similarity(
    ink_clusters: List[Dict[str, Any]],
    zones: Dict[str, Dict[str, Any]],
    zone_order: List[str] = ["ring_core", "dot_core"],
) -> Dict[int, Dict[str, Any]]:
    """
    Match plate zones to ink clusters by color similarity (ΔE2000).

    Uses greedy algorithm: for each zone, find the closest unmatched ink.

    Args:
        ink_clusters: List of ink cluster data with Lab centroids
        zones: Dict of zone_name -> zone_data with obs_lab
        zone_order: Order of zones to process

    Returns:
        Dict mapping ink_id (int) -> zone_data (with source="plate_lite")
        Inks without a matching zone will not be in the dict.
    """
    if not ink_clusters or not zones:
        return {}

    # Extract ink Labs
    ink_labs = []
    for i, cluster in enumerate(ink_clusters):
        role = cluster.get("role", "ink").lower()
        if role not in ["ink", "primary"]:
            continue
        ink_lab = cluster.get("lab_centroid_cie") or cluster.get("centroid_lab_cie") or cluster.get("centroid_lab")
        if ink_lab:
            ink_labs.append((i, ink_lab))

    if not ink_labs:
        return {}

    # Extract zone Labs
    zone_labs = []
    for zone_name in zone_order:
        zone_data = zones.get(zone_name, {})
        if zone_data.get("empty", False):
            continue
        obs_lab = zone_data.get("obs_lab")
        if obs_lab:
            try:
                validated_lab = _ensure_cie_lab(obs_lab)
            except Exception:
                validated_lab = obs_lab
            zone_labs.append((zone_name, validated_lab, zone_data))

    if not zone_labs:
        return {}

    # Greedy matching: for each zone, find closest unmatched ink
    matched_inks = set()
    ink_to_zone_map: Dict[int, Dict[str, Any]] = {}

    for zone_name, zone_lab, zone_data in zone_labs:
        best_ink_idx = None
        best_delta_e = float("inf")

        for ink_idx, ink_lab in ink_labs:
            if ink_idx in matched_inks:
                continue
            delta_e = calculate_delta_e_2000(ink_lab, zone_lab)
            if delta_e < best_delta_e:
                best_delta_e = delta_e
                best_ink_idx = ink_idx

        if best_ink_idx is not None:
            matched_inks.add(best_ink_idx)
            # Build zone observation data
            obs_data = {
                "lab": zone_lab,
                "hex": _lab_to_hex(zone_lab),
                "alpha_mean": zone_data.get("alpha_mean"),
                "plate_ink_key": zone_name,
                "source": "plate_lite",
                "match_delta_e": round(best_delta_e, 2),
            }
            ink_to_zone_map[best_ink_idx] = obs_data

    return ink_to_zone_map


# =============================================================================
# Core Simulation Functions
# =============================================================================


def simulate_perceived_color(
    ink_lab: List[float], coverage: float, bg_lab: List[float] = [100.0, 0.0, 0.0]  # Default White
) -> Dict[str, Any]:
    """
    Simulate the perceived color by mixing Ink and Background in Linear RGB space.

    Args:
        ink_lab: CIE Lab of the ink [L, a, b]
        coverage: Ink coverage ratio (0.0 - 1.0)
        bg_lab: CIE Lab of the background [L, a, b]

    Returns:
        {
            "lab": [L, a, b],
            "hex": "#RRGGBB",
            "mix_ratio": float
        }
    """
    # Phase 2-4: Ensure CIE Lab scale
    ink_lab = _ensure_cie_lab(ink_lab)
    bg_lab = _ensure_cie_lab(bg_lab)

    # Safety: Clamp coverage to 0.0 - 1.0
    # Handle percentage inputs (e.g., 85.0 -> 0.85) if > 1.0
    if coverage > 1.0:
        coverage = coverage / 100.0
    coverage = max(0.0, min(1.0, float(coverage)))

    # 1. Convert to Linear RGB
    ink_lin = _lab_to_linear_rgb(ink_lab)
    bg_lin = _lab_to_linear_rgb(bg_lab)

    # 2. Spatial Mixing (Linear Interpolation)
    mixed_lin = ink_lin * coverage + bg_lin * (1.0 - coverage)

    # 3. Convert back to Lab and Hex
    mixed_lab = _linear_rgb_to_lab(mixed_lin)
    mixed_hex = _lab_to_hex(mixed_lab)

    return {"lab": [round(x, 2) for x in mixed_lab], "hex": mixed_hex, "mix_ratio": round(coverage, 3)}


def calculate_composite_color(
    simulations: List[Dict[str, Any]], bg_lab: List[float], mode: str = "white"
) -> Optional[Dict[str, Any]]:
    """
    Calculate composite color using spatial averaging (non-overlapping assumption).

    Formula: Composite = bg * (1 - total_cov) + Σ(coverage_i * ink_raw_i)

    This is equivalent to: bg + Σ(coverage_i * (ink_raw_i - bg))

    IMPORTANT: Uses raw ink Lab (not perceived), because perceived already
    has coverage baked in. Using perceived would square the coverage effect.

    Args:
        simulations: List of simulation results from build_simulation_result
        bg_lab: Background Lab color
        mode: "white" or "black" (unused in v1, kept for API compatibility)

    Returns:
        Composite color info or None if no simulations
    """
    if not simulations:
        return None

    bg_lin = _lab_to_linear_rgb(bg_lab)
    total_coverage = 0.0
    weighted_ink_sum = np.zeros(3, dtype=np.float64)

    for sim in simulations:
        # Use model_input if available, otherwise fall back to ratio
        coverage = sim["coverage"].get("model_input", sim["coverage"]["ratio"])

        # CRITICAL: Use raw_ink, NOT perceived (which already has coverage mixed)
        ink_lab = sim["raw_ink"]["lab"]
        ink_lin = _lab_to_linear_rgb(ink_lab)

        weighted_ink_sum += coverage * ink_lin
        total_coverage += coverage

    # Spatial averaging formula: bg * (1 - total) + Σ(cov_i * ink_i)
    composite_lin = bg_lin * (1.0 - total_coverage) + weighted_ink_sum

    # Clamp to valid range
    composite_lin = np.clip(composite_lin, 0.0, 1.0)

    composite_lab = _linear_rgb_to_lab(composite_lin)
    composite_hex = _lab_to_hex(composite_lab)

    return {
        "lab": [round(x, 2) for x in composite_lab],
        "hex": composite_hex,
        "total_ink_coverage": round(total_coverage, 4),
        "method": "spatial_averaging_raw_ink",
        "_note": "Non-overlapping assumption: inks occupy disjoint areas",
    }


# =============================================================================
# Main Builder Function
# =============================================================================


def build_simulation_result(
    ink_clusters: List[Dict[str, Any]],
    plate_info: Optional[Dict[str, Any]] = None,
    radial_info: Optional[Dict[str, Any]] = None,
    default_bg_lab: List[float] = [95.0, 0.0, 0.0],  # Paper white
    black_bg_lab: List[float] = [10.0, 0.0, 0.0],  # Pupil black (approx)
    # Phase 2-2: Confidence propagation
    upstream_confidence: float = 1.0,
    upstream_warnings: Optional[List[str]] = None,
    # Phase A: Mask-based simulation options
    simulation_method: str = "area_ratio",  # "area_ratio" or "mask_based"
    lab_map_polar: Optional[np.ndarray] = None,  # (T, R, 3) for mask_based
    color_masks: Optional[Dict[str, np.ndarray]] = None,  # {color_id: mask} for mask_based
    mask_downsample: int = 4,  # Downsample factor for mask_based
    # Plate-Lite support
    plate_lite_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a full simulation report from analysis results.

    Includes both White-bg (Inspection) and Black-bg (Wear) simulations.

    Args:
        ink_clusters: List of ink cluster data from segmentation
        plate_info: Legacy plate analysis result (optional)
        radial_info: Radial profile analysis result (optional)
        plate_lite_info: Plate-Lite analysis result (optional). Structure:
            {
                "zones": {
                    "ring_core": {"obs_lab": [L,a,b], "alpha_mean": float, ...},
                    "dot_core": {"obs_lab": [L,a,b], "alpha_mean": float, ...}
                }
            }
            When provided and plate_info is absent, plate_lite zones are used
            to populate color_comparison.plate_measurement.

    Returns:
        Simulation result with color_comparison, simulations, composite, etc.
        color_comparison structure per ink:
        {
            "lens_clustering": {"lab": [...], "hex": "..."},
            "plate_measurement": {"lab": [...], "hex": "...", "source": "plate_lite"},
            "proofing_simulation": {"lab": [...], "hex": "..."}
        }

    Data Priority:
        1. plate_info.inks.by_source (full plate analysis)
        2. plate_lite_info.zones (plate-lite fallback)
        3. None (no plate measurement available)

    Phase 2-3 Enhancements:
    - Black background: measured > simulated priority
    - observed.on_black from plate pair analysis when available
    - model_error.delta_e_00 for validation/learning

    Phase A Enhancements (Longterm Roadmap):
    - simulation_method: "area_ratio" (legacy) or "mask_based" (Direction A)
    """
    # Phase 2-2: Safe warnings handling
    warnings = list(upstream_warnings) if upstream_warnings else []

    if not ink_clusters:
        return {}

    # Phase 2-3: Extract black-image measurements from plate
    # plate_info.inks.by_source.from_black has per-ink Lab measured on black background
    # Mapping: plate ink keys are "ink1", "ink2", ... -> map to cluster indices 0, 1, ...
    black_observed_map: Dict[str, Dict[str, Any]] = {}
    black_observed_by_index: List[Optional[Dict[str, Any]]] = []
    # Phase 6: Also extract white-image measurements
    white_observed_map: Dict[str, Dict[str, Any]] = {}
    white_observed_by_index: List[Optional[Dict[str, Any]]] = []
    if plate_info:
        by_source = plate_info.get("inks", {}).get("by_source", {})
        from_black = by_source.get("from_black", {})
        from_white = by_source.get("from_white", {})
        # Sort by ink key to maintain consistent ordering (ink1, ink2, ink3...)
        sorted_keys = sorted(from_black.keys())
        for ink_key in sorted_keys:
            # Extract from_black
            ink_data = from_black.get(ink_key, {})
            if not ink_data.get("empty", False):
                lab_mean = ink_data.get("lab", {}).get("mean")
                if lab_mean:
                    obs_data = {
                        "lab": lab_mean,
                        "hex": ink_data.get("hex_ref") or _lab_to_hex(lab_mean),
                        "alpha_mean": ink_data.get("alpha", {}).get("mean"),
                        "area_ratio": ink_data.get("area_ratio"),
                        "plate_ink_key": ink_key,
                    }
                    black_observed_map[ink_key] = obs_data
                    black_observed_by_index.append(obs_data)
                else:
                    black_observed_by_index.append(None)
            else:
                black_observed_by_index.append(None)

            # Extract from_white
            white_data = from_white.get(ink_key, {})
            if not white_data.get("empty", False):
                white_lab_mean = white_data.get("lab", {}).get("mean")
                if white_lab_mean:
                    white_obs_data = {
                        "lab": white_lab_mean,
                        "hex": white_data.get("hex_ref") or _lab_to_hex(white_lab_mean),
                        "alpha_mean": white_data.get("alpha", {}).get("mean"),
                        "area_ratio": white_data.get("area_ratio"),
                        "plate_ink_key": ink_key,
                    }
                    white_observed_map[ink_key] = white_obs_data
                    white_observed_by_index.append(white_obs_data)
                else:
                    white_observed_by_index.append(None)
            else:
                white_observed_by_index.append(None)

    # Phase Plate-Lite: Extract measurements from plate_lite if available
    # 1) Prefer per-ink measurements (plate_lite.inks)
    # 2) Fallback to zone matching (ring/dot) when inks are not available
    plate_lite_by_ink_id: Dict[int, Dict[str, Any]] = {}
    plate_lite_black_by_ink_id: Dict[int, Dict[str, Any]] = {}
    if plate_lite_info and not white_observed_by_index:
        ink_entries = plate_lite_info.get("inks") or []
        if isinstance(ink_entries, list) and ink_entries:
            for entry in ink_entries:
                ink_id = entry.get("ink_id")
                ink_lab = entry.get("ink_lab") or entry.get("obs_lab")
                ink_lab_black = entry.get("obs_lab_black")
                if ink_id is None or ink_lab is None:
                    continue
                plate_lite_by_ink_id[int(ink_id)] = {
                    "lab": ink_lab,
                    "hex": entry.get("ink_hex") or entry.get("obs_hex") or _lab_to_hex(ink_lab),
                    "alpha_mean": entry.get("alpha_mean"),
                    "plate_ink_key": entry.get("ink_key", f"ink{int(ink_id) + 1}"),
                    "source": "plate_lite_per_ink",
                }
                if ink_lab_black is not None:
                    plate_lite_black_by_ink_id[int(ink_id)] = {
                        "lab": ink_lab_black,
                        "hex": entry.get("obs_hex_black") or _lab_to_hex(ink_lab_black),
                        "alpha_mean": entry.get("alpha_mean"),
                        "plate_ink_key": entry.get("ink_key", f"ink{int(ink_id) + 1}"),
                        "source": "plate_lite_black",
                    }
        else:
            zones = plate_lite_info.get("zones", {})
            # Match zones to inks by color similarity (?E2000)
            plate_lite_by_ink_id = _match_zones_to_inks_by_similarity(
                ink_clusters, zones, zone_order=["ring_core", "dot_core"]
            )

    # Determine White Background Color
    bg_lab = list(default_bg_lab)
    bg_source = "default_white"

    # 1. Try Plate
    if plate_info:
        clear_core = plate_info.get("plates", {}).get("clear", {}).get("core", {})
        if not clear_core.get("empty", True):
            measured_bg = clear_core.get("lab", {}).get("mean")
            if measured_bg:
                bg_lab = list(measured_bg)
                bg_source = "plate_clear_measured"

    # 2. Try Radial
    if bg_source == "default_white" and radial_info:
        summary = radial_info.get("summary", {})
        inner_L = summary.get("inner_mean_L", 0)
        outer_L = summary.get("outer_mean_L", 0)
        if outer_L > inner_L + 10:
            profile = radial_info.get("profile_cie") or radial_info.get("profile")
            if profile:
                L_mean = profile.get("L_mean", [])
                a_mean = profile.get("a_mean", [])
                b_mean = profile.get("b_mean", [])
                if L_mean and a_mean and b_mean:
                    last_L = L_mean[-1]
                    last_a = a_mean[-1]
                    last_b = b_mean[-1]
                    if last_L > 80:
                        bg_lab = [last_L, last_a, last_b]
                        bg_source = "radial_outer_measured"

    simulations = []
    # P0-1: Track alpha blending usage across all inks
    any_alpha_blending_used = False
    alpha_blending_count = 0

    for cluster in ink_clusters:
        role = cluster.get("role", "ink").lower()
        if role not in ["ink", "primary"]:
            continue

        ink_lab = cluster.get("lab_centroid_cie") or cluster.get("centroid_lab_cie") or cluster.get("centroid_lab")
        if not ink_lab:
            continue

        # Phase 2-3: Coverage separation with alpha stats
        area_ratio = float(cluster.get("area_ratio", 0.0))

        # Alpha from Plate (if available)
        # Priority: effective_density from v2_diagnostics > alpha_mean × area_ratio
        alpha_mean = cluster.get("alpha_mean", 1.0)
        alpha_p50 = cluster.get("alpha_p50", alpha_mean)
        alpha_p90 = cluster.get("alpha_p90", alpha_mean)
        alpha_source = cluster.get("alpha_source", "default_opaque")

        # P0-1: Check for pre-computed effective_density from v2_diagnostics
        # This has proper fallback logic (L1_radial > L2_zone > L3_global)
        precomputed_effective_density = cluster.get("effective_density")
        alpha_used = cluster.get("alpha_used", alpha_mean)
        alpha_fallback_level = cluster.get("alpha_fallback_level")

        # Effective density = area_ratio * alpha_used
        if precomputed_effective_density is not None:
            effective_density = float(precomputed_effective_density)
            alpha_source = alpha_fallback_level or alpha_source
        else:
            effective_density = area_ratio * alpha_mean

        # P0-1: Model input selection based on alpha availability
        # Use effective_density (alpha blending) when actual alpha is available
        # Fall back to area_ratio (opaque assumption) when only default_opaque
        use_alpha_blending = alpha_source not in ("default_opaque", None)
        if use_alpha_blending:
            model_input = effective_density
            any_alpha_blending_used = True
            alpha_blending_count += 1
        else:
            model_input = area_ratio

        # Simulate on White (Inspection)
        sim_white = simulate_perceived_color(ink_lab, model_input, bg_lab)

        # Simulate on Black (Wear)
        sim_black = simulate_perceived_color(ink_lab, model_input, black_bg_lab)

        # Get ink_id for matching with plate data
        ink_id = cluster.get("id", cluster.get("ink_id"))

        # Build simulation entry
        sim_entry = {
            "ink_id": ink_id,
            "role": role,
            "raw_ink": {
                "lab": [round(x, 2) for x in ink_lab],
                "hex": cluster.get("mean_hex") or cluster.get("rgb_hex") or _lab_to_hex(ink_lab),
            },
            # Phase 2-3 + P0-1: Enhanced coverage structure with alpha blending
            "coverage": {
                "area_ratio": round(area_ratio, 4),
                "alpha": {
                    "mean": round(alpha_mean, 3),
                    "used": round(alpha_used, 3),  # P0-1: actual alpha value used
                    "p50": round(alpha_p50, 3),
                    "p90": round(alpha_p90, 3),
                    "source": alpha_source,
                    "fallback_level": alpha_fallback_level,  # P0-1: L1_radial/L2_zone/L3_global
                },
                "effective_density": round(effective_density, 4),
                "model_input": round(model_input, 4),
                "use_alpha_blending": use_alpha_blending,  # P0-1: True if alpha is real
                # Legacy fields for backward compatibility
                "ratio": round(area_ratio, 3),
                "percent": round(area_ratio * 100, 1),
            },
            "background": {
                "white_lab": [round(x, 2) for x in bg_lab],
                "black_lab": [round(x, 2) for x in black_bg_lab],
                "source": bg_source,
            },
            # Predicted (simulated) colors
            "predicted": {
                "on_white": {"lab": sim_white["lab"], "hex": sim_white["hex"]},
                "on_black": {"lab": sim_black["lab"], "hex": sim_black["hex"]},
            },
            # Legacy alias for backward compatibility
            "perceived": {
                "on_white": {"lab": sim_white["lab"], "hex": sim_white["hex"]},
                "on_black": {"lab": sim_black["lab"], "hex": sim_black["hex"]},
            },
        }

        # Phase 2-3 + Phase 6: Add observed data from plate pair analysis
        # Try to match by:
        # 1. Direct ink_id match (e.g., "ink1" matches "ink1")
        # 2. Index-based match for plate_pair (cluster index i -> plate ink index i)
        # 3. ΔE-based match for plate_lite (cluster index i -> closest zone by color)
        observed_on_black = black_observed_map.get(str(ink_id))
        if not observed_on_black and isinstance(ink_id, int) and ink_id < len(black_observed_by_index):
            observed_on_black = black_observed_by_index[ink_id]
        if not observed_on_black and isinstance(ink_id, int) and ink_id in plate_lite_black_by_ink_id:
            observed_on_black = plate_lite_black_by_ink_id[ink_id]

        observed_on_white = white_observed_map.get(str(ink_id))
        if not observed_on_white and isinstance(ink_id, int) and ink_id < len(white_observed_by_index):
            observed_on_white = white_observed_by_index[ink_id]
        # Plate-Lite: use ΔE-based matching result
        if not observed_on_white and isinstance(ink_id, int) and ink_id in plate_lite_by_ink_id:
            observed_on_white = plate_lite_by_ink_id[ink_id]

        # Build observed structure with both on_white and on_black
        observed_entry: Dict[str, Any] = {}
        model_error_entry: Dict[str, Any] = {}

        if observed_on_white:
            white_lab = observed_on_white["lab"]
            plate_ink_key_w = observed_on_white.get("plate_ink_key", "unknown")
            # Use source from data if available (plate_lite vs plate_pair)
            white_source = observed_on_white.get("source", "plate_pair_measured")
            on_white_entry = {
                "lab": [round(x, 2) for x in white_lab],
                "hex": observed_on_white.get("hex") or _lab_to_hex(white_lab),
                "source": white_source,
                "matched_plate_ink": plate_ink_key_w,
            }
            # Add match quality for ΔE-based matching (plate_lite)
            if "match_delta_e" in observed_on_white:
                on_white_entry["match_delta_e"] = observed_on_white["match_delta_e"]
            observed_entry["on_white"] = on_white_entry
            # Calculate model error for white
            predicted_white_lab = sim_white["lab"]
            delta_e_white = calculate_delta_e_2000(predicted_white_lab, white_lab)
            model_error_entry["on_white"] = {
                "delta_e_00": round(delta_e_white, 2),
                "interpretation": _interpret_delta_e(delta_e_white),
                "comparison": f"predicted vs {plate_ink_key_w}",
            }

        if observed_on_black:
            black_lab = observed_on_black["lab"]
            plate_ink_key_b = observed_on_black.get("plate_ink_key", "unknown")
            observed_entry["on_black"] = {
                "lab": [round(x, 2) for x in black_lab],
                "hex": observed_on_black.get("hex") or _lab_to_hex(black_lab),
                "source": observed_on_black.get("source", "plate_pair_measured"),
                "matched_plate_ink": plate_ink_key_b,
            }
            # Calculate model error for black
            predicted_black_lab = sim_black["lab"]
            delta_e_black = calculate_delta_e_2000(predicted_black_lab, black_lab)
            model_error_entry["on_black"] = {
                "delta_e_00": round(delta_e_black, 2),
                "interpretation": _interpret_delta_e(delta_e_black),
                "comparison": f"predicted vs {plate_ink_key_b}",
            }

        if observed_entry:
            sim_entry["observed"] = observed_entry
        if model_error_entry:
            sim_entry["model_error"] = model_error_entry

        simulations.append(sim_entry)

    # Phase 6: Build unified color comparison structure for UI
    # Each ink has 3 color sources: lens_clustering, plate_measurement, proofing_simulation
    color_comparison = []
    for sim in simulations:
        ink_comparison = {
            "ink_id": sim.get("ink_id"),
            "role": sim.get("role"),
            # Column 1: Lens Clustering (raw extracted color)
            "lens_clustering": {
                "lab": sim.get("raw_ink", {}).get("lab"),
                "hex": sim.get("raw_ink", {}).get("hex"),
                "source": "kmeans_on_polar_lab",
            },
            # Column 2: Plate Measurement (observed on white)
            "plate_measurement": None,
            # Column 3: Proofing Simulation (predicted on white)
            "proofing_simulation": {
                "lab": sim.get("predicted", {}).get("on_white", {}).get("lab"),
                "hex": sim.get("predicted", {}).get("on_white", {}).get("hex"),
                "source": "linear_rgb_mix_simulation",
            },
        }
        # Add plate measurement if available
        observed_white = sim.get("observed", {}).get("on_white")
        if observed_white:
            plate_entry = {
                "lab": observed_white.get("lab"),
                "hex": observed_white.get("hex"),
                "source": observed_white.get("source", "plate_pair_white_image"),
                "matched_plate_ink": observed_white.get("matched_plate_ink"),
            }
            # Add match quality for ΔE-based matching (plate_lite)
            if observed_white.get("match_delta_e") is not None:
                plate_entry["match_delta_e"] = observed_white.get("match_delta_e")
            ink_comparison["plate_measurement"] = plate_entry
        color_comparison.append(ink_comparison)

    # Phase 6: Black background comparison (plate black vs proofing black)
    color_comparison_black = []
    for sim in simulations:
        ink_comparison_black = {
            "ink_id": sim.get("ink_id"),
            "role": sim.get("role"),
            "lens_clustering": {
                "lab": sim.get("raw_ink", {}).get("lab"),
                "hex": sim.get("raw_ink", {}).get("hex"),
                "source": "kmeans_on_polar_lab",
            },
            "plate_measurement": None,
            "proofing_simulation": {
                "lab": sim.get("predicted", {}).get("on_black", {}).get("lab"),
                "hex": sim.get("predicted", {}).get("on_black", {}).get("hex"),
                "source": "linear_rgb_mix_simulation",
            },
        }
        observed_black = sim.get("observed", {}).get("on_black")
        if observed_black:
            plate_entry_black = {
                "lab": observed_black.get("lab"),
                "hex": observed_black.get("hex"),
                "source": observed_black.get("source", "plate_pair_black_image"),
                "matched_plate_ink": observed_black.get("matched_plate_ink"),
            }
            ink_comparison_black["plate_measurement"] = plate_entry_black
        color_comparison_black.append(ink_comparison_black)

    # Phase 3 + Phase A: Calculate composite colors
    # Use mask_based method if requested and data available
    mask_based_result = None
    if simulation_method == "mask_based" and lab_map_polar is not None and color_masks:
        from .mask_compositor import composite_from_masks

        mask_based_result = composite_from_masks(
            lab_map_polar,
            color_masks,
            downsample=mask_downsample,
            reduce="trimmed_mean",
        )
        composite_white = {
            "lab": mask_based_result["composite_lab"],
            "hex": _lab_to_hex(mask_based_result["composite_lab"]),
            "total_ink_coverage": mask_based_result["n_pixels_sampled"] / max(lab_map_polar.size // 3, 1),
            "method": "mask_based_pixel_synthesis",
            "overlap": mask_based_result["overlap"],
            "zone_contributions": mask_based_result["zone_contributions"],
            "confidence": mask_based_result["confidence"],
        }
        # For black, we still use area_ratio (mask_based needs black lab_map)
        composite_black = calculate_composite_color(simulations, black_bg_lab, "black")
        if composite_black:
            composite_black["_note"] = "mask_based not available for black (no black lab_map)"
    else:
        # Legacy area_ratio method
        composite_white = calculate_composite_color(simulations, bg_lab, "white")
        composite_black = calculate_composite_color(simulations, black_bg_lab, "black")

    return {
        "simulations": simulations,
        # Phase 6: Unified color comparison for UI (3 columns per ink)
        "color_comparison": color_comparison,
        "color_comparison_black": color_comparison_black,
        # Phase 3: Composite colors
        "composite": {
            "on_white": composite_white,
            "on_black": composite_black,
            "method_used": simulation_method if mask_based_result else "area_ratio",
        },
        # Phase A: Mask-based analysis details (if used)
        "mask_analysis": mask_based_result if mask_based_result else None,
        # Phase 2-1: Enhanced global_background with hex
        "global_background": {
            "white_lab": [round(x, 2) for x in bg_lab],
            "white_hex": _lab_to_hex(bg_lab),
            "black_lab": [round(x, 2) for x in black_bg_lab],
            "black_hex": _lab_to_hex(black_bg_lab),
            "source": bg_source,
        },
        # Phase 2-1 + P0-1: Model metadata with alpha blending support
        "model": {
            "name": "mask_based_pixel_synthesis" if mask_based_result else "spatial_mix_linear_rgb",
            "version": "v2.1" if any_alpha_blending_used else ("v2.0" if mask_based_result else "v1.1"),
            "simulation_method": simulation_method,
            "assumptions": (
                [
                    "pixel_sampling_from_masks",
                    "overlap_aware",
                    "zone_contribution_analysis",
                ]
                if mask_based_result
                else (
                    ["alpha_weighted_spatial_averaging", "sRGB_gamma_correction", "D65_illuminant"]
                    if any_alpha_blending_used
                    else ["opaque_spatial_averaging", "sRGB_gamma_correction", "D65_illuminant"]
                )
            ),
            "coverage_definition": (
                "mask_union_pixel_count"
                if mask_based_result
                else ("effective_density_alpha_weighted" if any_alpha_blending_used else "area_ratio_of_cluster_mask")
            ),
            "model_input_field": "coverage.model_input",
            "limitations": (
                [
                    "white_background_only_for_mask_based",
                ]
                if mask_based_result
                else (
                    ["no_multi_ink_overlay"]
                    if any_alpha_blending_used
                    else ["no_alpha_transparency", "no_multi_ink_overlay"]
                )
            ),
            # P0-1: Alpha blending metadata
            "alpha_blending": {
                "enabled": any_alpha_blending_used,
                "inks_with_alpha": alpha_blending_count,
                "total_inks": len(simulations),
            },
            # Phase 2-3 + Phase 6: Observed data availability
            "observed_on_white_available": bool(white_observed_map) or bool(plate_lite_by_ink_id),
            "observed_on_black_available": bool(black_observed_map) or bool(plate_lite_black_by_ink_id),
            "observed_source": (
                "plate_pair"
                if white_observed_map or black_observed_map
                else "plate_lite" if plate_lite_by_ink_id or plate_lite_black_by_ink_id else None
            ),
            # Phase A: Mask-based options used
            "mask_based_options": (
                {
                    "downsample": mask_downsample,
                    "reduce": "trimmed_mean",
                }
                if mask_based_result
                else None
            ),
        },
        # Phase 2-2: Confidence propagation
        "confidence": {
            "upstream_factor": round(upstream_confidence, 3),
            "simulation_warnings": warnings,
        },
    }
