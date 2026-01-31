"""Per-color evaluation for multi-ink images, extracted from analyzer.py."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Tuple

import numpy as np

from ..decision.decision_builder import build_decision
from ..geometry.lens_geometry import detect_lens_circle
from ..measure.segmentation.color_masks import build_color_masks_with_retry
from ..signature.radial_signature import build_radial_signature_masked, to_polar
from ..types import Decision, SignatureResult
from ._common import _finalize_decision, _maybe_apply_white_balance, _reason_meta, _run_gate_check
from ._signature import _evaluate_signature, _pick_best_mode


def evaluate_per_color(
    test_bgr,
    per_color_models: Dict[str, Dict[str, Any]],  # {color_id: {mode: StdModel}}
    color_metadata: Dict[str, Any],  # {color_id: metadata}
    cfg: Dict[str, Any],
    expected_ink_count: int,
    pattern_baseline: Dict[str, Any] | None = None,
    ok_log_context: Dict[str, Any] | None = None,
    mode: str = "all",
) -> Tuple[Decision, Dict[str, Dict[str, SignatureResult]]]:
    """Per-color evaluation for multi-ink images."""
    # Step 1: Geometry detection
    geom = detect_lens_circle(test_bgr)
    test_bgr, wb_meta = _maybe_apply_white_balance(test_bgr, geom, cfg)

    # Step 2: Gate check (shared across all colors)
    gate, early = _run_gate_check(
        geom,
        test_bgr,
        cfg,
        mode,
        pattern_baseline,
        extra_decision_kwargs={"best_mode": "", "mode_scores": {}},
    )
    if early is not None:
        if mode == "gate":
            early.diagnostics.setdefault("color_mode", "per_color")
        else:
            early.diagnostics = early.diagnostics or {}
            early.diagnostics["color_mode"] = "per_color"
        return early, {}

    # Step 3: Generate color masks with 2-pass retry logic
    try:
        color_masks, mask_metadata = build_color_masks_with_retry(
            test_bgr, cfg, expected_k=expected_ink_count, geom=geom, confidence_threshold=0.7, enable_retry=True
        )
    except Exception as e:
        codes, messages = _reason_meta(["COLOR_SEGMENTATION_FAILED"])
        dec = Decision(
            label="RETAKE",
            reasons=["COLOR_SEGMENTATION_FAILED"],
            reason_codes=codes,
            reason_messages=messages + [f"Error: {str(e)}"],
            gate=gate,
            signature=None,
            anomaly=None,
            debug={"inference_valid": False, "segmentation_error": str(e)},
            diagnostics={"color_mode": "per_color"},
            best_mode="",
            mode_scores={},
            phase="INSPECTION",
        )
        return dec, {}

    # Extract segmentation metadata for debugging/warnings
    segmentation_info = {
        "expected_ink_count": mask_metadata.get("expected_ink_count"),
        "segmentation_k": mask_metadata.get("segmentation_k"),
        "detected_ink_like_count": mask_metadata.get("detected_ink_like_count"),
        "segmentation_confidence": mask_metadata.get("segmentation_confidence"),
        "segmentation_pass": mask_metadata.get("segmentation_pass"),
        "retry_reason": mask_metadata.get("retry_reason"),
    }

    # Add warnings if ink count mismatch or retry occurred
    segmentation_warnings = []
    detected_inks = mask_metadata.get("detected_ink_like_count", 0)

    if detected_inks != expected_ink_count:
        segmentation_warnings.append(
            f"EXPECTED_INK_COUNT_MISMATCH (expected={expected_ink_count}, detected={detected_inks})"
        )

    if mask_metadata.get("segmentation_pass") == "pass2_retry":
        retry_reasons = mask_metadata.get("retry_reason", [])
        segmentation_warnings.append(f"INK_SEGMENTATION_RETRIED_K{expected_ink_count + 1}")
        if retry_reasons:
            segmentation_warnings.extend(retry_reasons)

    # Verify color count matches expected (fatal error - cannot proceed)
    detected_colors = len(mask_metadata.get("colors", []))
    if detected_colors == 0 or (
        detected_colors < expected_ink_count and mask_metadata.get("k_used", 0) == expected_ink_count
    ):
        codes, messages = _reason_meta(["COLOR_SEGMENTATION_FAILED"])
        dec = Decision(
            label="RETAKE",
            reasons=["COLOR_SEGMENTATION_FAILED"] + segmentation_warnings,
            reason_codes=codes,
            reason_messages=messages
            + [f"Detected {detected_colors} colors, expected {expected_ink_count}"]
            + segmentation_warnings,
            gate=gate,
            signature=None,
            anomaly=None,
            debug={
                "inference_valid": False,
                "detected_colors": detected_colors,
                "expected_colors": expected_ink_count,
                **segmentation_info,
            },
            diagnostics={"color_mode": "per_color", "mask_metadata": mask_metadata},
            best_mode="",
            mode_scores={},
            phase="INSPECTION",
        )
        return dec, {}

    # Step 4: Evaluate each color
    any_model = next(iter(next(iter(per_color_models.values())).values()))
    polar = to_polar(test_bgr, geom, R=any_model.meta["R"], T=any_model.meta["T"])

    per_color_signatures = {}  # {color_id: {mode: SignatureResult}}
    per_color_best_modes = {}  # {color_id: best_mode}
    per_color_reasons = {}  # {color_id: [reasons]}

    if mode != "ink":
        for color_id, mode_models in per_color_models.items():
            if color_id not in color_masks:
                per_color_reasons[color_id] = ["COLOR_MASK_MISSING"]
                continue

            mask = color_masks[color_id]
            color_mode_sigs = {}

            for m_mode, m_model in mode_models.items():
                try:
                    test_mean, _, _ = build_radial_signature_masked(
                        polar, mask, r_start=cfg["signature"]["r_start"], r_end=cfg["signature"]["r_end"]
                    )
                    color_mode_sigs[m_mode] = _evaluate_signature(test_mean, m_model, cfg)
                except Exception as e:
                    per_color_reasons[color_id] = [f"SIGNATURE_EVAL_FAILED:{m_mode}:{str(e)}"]
                    break

            if color_mode_sigs:
                per_color_signatures[color_id] = color_mode_sigs
                best_mode_for_color = _pick_best_mode(color_mode_sigs)
                per_color_best_modes[color_id] = best_mode_for_color
                best_sig = color_mode_sigs[best_mode_for_color]
                if best_sig.reasons:
                    per_color_reasons[color_id] = [f"{reason}:{color_id}" for reason in best_sig.reasons]

    # Step 5: Aggregate per-color results
    all_reasons = []
    failed_colors = []

    for color_id in per_color_models.keys():
        if color_id in per_color_reasons and per_color_reasons[color_id]:
            color_meta = color_metadata.get(color_id, {})
            if color_meta.get("role") == "ink":
                all_reasons.extend(per_color_reasons[color_id])
                failed_colors.append(color_id)

    if failed_colors:
        label = "NG_COLOR"
    else:
        label = "OK"

    debug = {
        "test_geom": asdict(geom),
        "inference_valid": not failed_colors,
        "per_color_best_modes": per_color_best_modes,
        "failed_colors": failed_colors,
        "detected_colors": detected_colors,
    }

    diagnostics = {
        "gate": asdict(gate),
        "color_mode": "per_color",
        "mask_metadata": mask_metadata,
        "per_color_signatures": {
            color_id: {m: asdict(sig) for m, sig in mode_sigs.items()}
            for color_id, mode_sigs in per_color_signatures.items()
        },
        "per_color_best_modes": per_color_best_modes,
        "color_metadata": color_metadata,
    }

    if per_color_best_modes:
        mode_counts = {}
        for mode_name in per_color_best_modes.values():
            mode_counts[mode_name] = mode_counts.get(mode_name, 0) + 1
        overall_best_mode = max(mode_counts, key=mode_counts.get)
    else:
        overall_best_mode = ""

    # Build Decision with V2 Schema
    aggregated_clusters = []
    for color_id, sigs in per_color_signatures.items():
        best_m = per_color_best_modes.get(color_id, "MID")
        best_s = sigs.get(best_m)
        if best_s:
            aggregated_clusters.append(
                {
                    "cluster_id": color_id,
                    "inkness_score": 1.0,
                    "score_corr": float(best_s.score_corr),
                    "delta_e_mean": float(best_s.delta_e_mean),
                }
            )

    # gate_scores compatibility layer
    raw_gate = gate.scores if gate else {}
    gate_scores = dict(raw_gate or {})
    if "sharpness_score" not in gate_scores and "sharpness_laplacian_var" in gate_scores:
        gate_scores["sharpness_score"] = gate_scores["sharpness_laplacian_var"]

    decision_json = build_decision(
        run_id=(ok_log_context or {}).get("run_id") or "",
        phase="INSPECTION",
        cfg=cfg,
        gate_scores=gate_scores,
        expected_inks=expected_ink_count,
        sample_clusters=aggregated_clusters,
        match_result=None,
        deltae_summary_method="max",
        inkness_summary_method="min",
    )

    dec = Decision(
        label=decision_json["decision"]["label"],
        reasons=all_reasons,
        reason_codes=decision_json["decision"]["reason_codes"],
        reason_messages=[],
        gate=gate,
        signature=None,
        anomaly=None,
        debug=debug,
        diagnostics=diagnostics,
        best_mode=overall_best_mode,
        mode_scores={},
        phase="INSPECTION",
    )

    # qc_decision
    dec.ops = dec.ops or {}
    dec.ops["qc_decision"] = {
        "schema_version": "qc_decision.v1",
        **decision_json,
    }

    if (cfg.get("debug") or {}).get("include_full_qc_decision", False):
        dec.debug["full_qc_decision"] = decision_json

    dec.pattern_color = decision_json.get("pattern_color", {})

    # v2/v3 diagnostics (uses _common._finalize_decision partially — but per_color has custom qc_decision above)
    from ._diagnostics import _attach_v2_diagnostics, _attach_v3_summary, _attach_v3_trend

    if mode != "signature":
        _attach_v2_diagnostics(
            test_bgr,
            dec,
            cfg,
            ok_log_context,
            cached_geom=geom,
            cached_masks=(color_masks, mask_metadata),
        )

    v2_diag = dec.diagnostics.get("v2_diagnostics") or {}
    _attach_v3_summary(dec, v2_diag, cfg, ok_log_context)
    _attach_v3_trend(dec, ok_log_context)

    return dec, per_color_signatures
