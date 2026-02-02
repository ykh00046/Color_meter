"""
Single Sample Analysis Module

Analyzes a single sample without STD comparison.
Provides quality assessment, color distribution, pattern analysis, etc.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

from ..config_norm import get_polar_dims
from ..gate.gate_engine import run_gate
from ..geometry.lens_geometry import detect_lens_circle
from ..types import LensGeometry
from ..utils import apply_white_balance
from .single_analysis_steps import (
    _analyze_color_histogram,
    _analyze_ink_segmentation,
    _analyze_pattern_features,
    _analyze_radial_profile,
    _analyze_zones_2d,
    _calculate_quality_score,
    _determine_operator_decision,
    _extract_engineer_kpi,
    _generate_warnings,
)


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
    registrationless_enabled = bool(alpha_cfg.get("registrationless_enabled", True))

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

        # Priority 1: Inject plate_lite per-ink alpha_mean as fallback candidates
        # Mapping is deferred to compute_cluster_effective_densities (area_ratio sort)
        if "plate_lite" in results:
            pl_inks = results["plate_lite"].get("inks", [])
            if pl_inks:
                # Sort by area_ratio descending (largest ink region first)
                pl_sorted = sorted(pl_inks, key=lambda x: x.get("area_ratio", 0), reverse=True)
                alpha_cfg["_plate_lite_inks"] = [
                    {
                        "alpha_mean": float(ink.get("alpha_mean", 0)),
                        "area_ratio": float(ink.get("area_ratio", 0)),
                        "ink_key": ink.get("ink_key", ""),
                    }
                    for ink in pl_sorted
                ]
                logger.debug(
                    f"plate_lite alpha injected: {len(pl_sorted)} inks, "
                    f"alpha_means={[round(i.get('alpha_mean', 0), 3) for i in pl_sorted]}"
                )

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
