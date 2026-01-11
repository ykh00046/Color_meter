#!/usr/bin/env python3
"""Signature Engine Runner

- Default phase: INSPECTION
- Use --phase STD_REGISTRATION for STD validation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config_loader import load_cfg_with_sku
from core.mode.mode_tracker import load_state, save_state, update_and_check_shift
from core.model_registry import compute_cfg_hash, load_expected_ink_count, load_pattern_baseline, load_std_models
from core.pipeline.analyzer import evaluate, evaluate_multi, evaluate_registration_multi
from core.reason_codes import reason_codes, reason_messages
from core.signature.fit import fit_std
from core.signature.model_io import load_model
from core.types import Decision, GateResult


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "y"}


def load_cfg(p: str, sku: str | None = None, cfg_snapshot: bool = False):
    strict_unknown = _env_flag("LENS_CFG_STRICT")
    if cfg_snapshot:
        return load_cfg_with_sku(p, None, strict_unknown=strict_unknown)
    return load_cfg_with_sku(p, sku, strict_unknown=strict_unknown)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--cfg", default=str(Path("configs") / "default.json"))
    ap.add_argument("--cfg_snapshot", action="store_true")
    ap.add_argument("--tests", nargs="+", required=True)
    ap.add_argument("--out", default="")

    # Single-model inputs
    ap.add_argument("--std", default="", help="STD image path (if not using --std_model)")
    ap.add_argument("--std_model", default="", help="STD model prefix (prefix.npz + prefix.json)")

    # Multi-mode inputs (A-plan)
    ap.add_argument("--std_model_low", default="", help="LOW model prefix")
    ap.add_argument("--std_model_mid", default="", help="MID model prefix")
    ap.add_argument("--std_model_high", default="", help="HIGH model prefix")

    # SKU-level state for sampling QC
    ap.add_argument("--sku", default="", help="SKU id for mode shift tracking or registry lookup")
    ap.add_argument("--ink", default="INK_DEFAULT", help="Ink name for registry lookup")
    ap.add_argument("--models_root", default=str(Path("models")), help="Models root for index.json")
    ap.add_argument("--state_dir", default=None, help="Directory to store SKU states")
    ap.add_argument("--phase", default="INSPECTION", choices=["INSPECTION", "STD_REGISTRATION"])
    ap.add_argument("--expected_ink_count", type=int, default=None)
    ap.add_argument("--mode", default="all", choices=["all", "gate", "signature", "ink"], help="Inspection mode")
    ap.add_argument("--trend_jsonl", default="", help="JSONL file with decision history for trend analysis")
    ap.add_argument("--trend_window", type=int, default=20, help="Number of recent decisions to use for trend")

    args = ap.parse_args()
    if not args.state_dir:
        env_state_dir = os.getenv("LENS_STATE_DIR")
        args.state_dir = env_state_dir if env_state_dir else str(Path("state"))
    cfg, cfg_sources, cfg_warnings = load_cfg(args.cfg, args.sku or None, cfg_snapshot=args.cfg_snapshot)

    # Determine mode: multi if all three provided
    use_multi = bool(args.std_model_low and args.std_model_mid and args.std_model_high)
    use_registry = False

    std_model_single = None
    std_models = None
    pattern_baseline = None
    baseline_reasons: list = []
    expected_ink_count_registry = None

    if use_multi:
        std_models = {
            "LOW": load_model(args.std_model_low),
            "MID": load_model(args.std_model_mid),
            "HIGH": load_model(args.std_model_high),
        }
        active_versions_direct = {
            "LOW": Path(args.std_model_low).parent.name if args.std_model_low else "",
            "MID": Path(args.std_model_mid).parent.name if args.std_model_mid else "",
            "HIGH": Path(args.std_model_high).parent.name if args.std_model_high else "",
        }
        if not args.sku:
            # SKU is not strictly required, but recommended for mode shift flag.
            pass
    else:
        use_registry = bool(args.sku) and not args.std_model and not args.std
        if use_registry:
            cfg_hash = None if args.cfg_snapshot else compute_cfg_hash(cfg)
            std_models, reasons = load_std_models(
                args.models_root,
                args.sku,
                args.ink,
                cfg_hash=cfg_hash,
            )
            pattern_baseline, baseline_reasons = load_pattern_baseline(
                args.models_root,
                args.sku,
                args.ink,
            )
            expected_ink_count_registry = load_expected_ink_count(
                args.models_root,
                args.sku,
                args.ink,
            )
            if std_models is None:
                use_multi = True
            else:
                use_multi = True
        else:
            reasons = []
        if args.std_model:
            std_model_single = load_model(args.std_model)
        else:
            if not args.std:
                if not use_registry:
                    raise SystemExit("Provide --std_model or --std (or use --std_model_low/mid/high for multi-mode)")
            else:
                std_bgr = cv2.imread(args.std)
                if std_bgr is None:
                    raise SystemExit(f"Failed to read STD: {args.std}")
                std_model_single = fit_std(
                    std_bgr,
                    R=cfg["polar"]["R"],
                    T=cfg["polar"]["T"],
                    r_start=cfg["signature"]["r_start"],
                    r_end=cfg["signature"]["r_end"],
                )

    results = []

    # load sku state once (sampling QC: small volume)
    state = None
    if use_multi and args.phase == "INSPECTION" and cfg.get("mode_stability", {}).get("enabled", True) and args.sku:
        state = load_state(args.state_dir, args.sku)

    result_meta = {
        "sku": args.sku,
        "ink": args.ink,
        "phase": args.phase,
        "config_used": cfg_sources,
        "cfg_warnings": cfg_warnings,
        "cfg_hash": compute_cfg_hash(cfg),
    }

    # Pre-load trend history if provided
    trend_history_rows = []
    if args.trend_jsonl:
        trend_path = Path(args.trend_jsonl)
        if trend_path.exists() and trend_path.is_file():
            try:
                with trend_path.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            trend_history_rows.append(json.loads(line))
                        except:
                            pass
            except Exception:
                pass

    for p in args.tests:
        bgr = cv2.imread(p)
        if bgr is None:
            results.append({"path": p, "error": "read_failed", "meta": result_meta})
            continue

        # Trend setup
        trend_log_path = None
        trend_window = args.trend_window or int(cfg.get("v3", {}).get("trend_window", 20))

        if not args.trend_jsonl and args.state_dir and args.sku and args.ink:
            trend_log_path = str(Path(args.state_dir) / args.sku / args.ink / "trend_log.jsonl")

        ok_log_context = {
            "sku": args.sku,
            "ink": args.ink,
            "models_root": args.models_root,
            "result_path": p,
            "expected_ink_count_input": args.expected_ink_count,
            "expected_ink_count_registry": expected_ink_count_registry,
            "trend_log_path": trend_log_path,
            "trend_window_requested": trend_window,
        }
        if trend_history_rows:
            ok_log_context["v3_trend_decisions"] = trend_history_rows
        if use_multi and not use_registry:
            ok_log_context["active_versions"] = active_versions_direct

        if use_multi:
            if std_models is None:
                label = "RETAKE" if args.phase == "INSPECTION" else "STD_RETAKE"
                codes = reason_codes(reasons or ["MODEL_NOT_FOUND"])
                messages = reason_messages(reasons or ["MODEL_NOT_FOUND"])
                dec = Decision(
                    label=label,
                    reasons=reasons or ["MODEL_NOT_FOUND"],
                    reason_codes=codes,
                    reason_messages=messages,
                    gate=GateResult(passed=False, reasons=reasons or ["MODEL_NOT_FOUND"], scores={}),
                    signature=None,
                    anomaly=None,
                    phase=args.phase,
                )
                if reasons:
                    dec.debug = {"model_loading": {"reasons": reasons}}
            else:
                if args.phase == "STD_REGISTRATION":
                    dec = evaluate_registration_multi(bgr, std_models, cfg)
                else:
                    dec, mode_sigs = evaluate_multi(
                        bgr,
                        std_models,
                        cfg,
                        pattern_baseline=pattern_baseline,
                        ok_log_context=ok_log_context,
                        mode=args.mode,
                    )
            # mode shift flag (only when best_mode exists)
            if (
                state is not None
                and dec.best_mode
                and args.phase == "INSPECTION"
                and cfg.get("mode_stability", {}).get("emit_shift_flag", True)
            ):
                shift = update_and_check_shift(
                    state, dec.best_mode, window=int(cfg["mode_stability"].get("window", 10))
                )
                dec.mode_shift = shift
            if baseline_reasons and dec.debug is not None:
                dec.debug.setdefault("baseline_reasons", baseline_reasons)
            results.append({"path": p, "decision": dec.to_dict(), "meta": result_meta})
        else:
            if args.phase == "STD_REGISTRATION":
                codes = reason_codes(["STD_REGISTRATION_REQUIRES_MULTI"])
                messages = reason_messages(["STD_REGISTRATION_REQUIRES_MULTI"])
                dec = Decision(
                    label="STD_RETAKE",
                    reasons=["STD_REGISTRATION_REQUIRES_MULTI"],
                    reason_codes=codes,
                    reason_messages=messages,
                    gate=GateResult(passed=False, reasons=["STD_REGISTRATION_REQUIRES_MULTI"], scores={}),
                    signature=None,
                    anomaly=None,
                    phase=args.phase,
                )
            else:
                dec = evaluate(
                    bgr,
                    std_model_single,
                    cfg,
                    pattern_baseline=pattern_baseline,
                    ok_log_context=ok_log_context,
                    mode=args.mode,
                )
            results.append({"path": p, "decision": dec.to_dict(), "meta": result_meta})

    if state is not None:
        save_state(args.state_dir, state, window=int(cfg["mode_stability"].get("window", 10)))

    payload = {
        "schema_version": "v7_results.v1",
        "engine_version": "v7",
        "cfg": cfg,
        "config_used": cfg_sources,
        "cfg_warnings": cfg_warnings,
        "cfg_hash": compute_cfg_hash(cfg),
        "results": results,
        "multi_mode": use_multi,
        "sku": args.sku,
        "ink": args.ink,
        "phase": args.phase,
    }
    s = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(s)
    print(s)


if __name__ == "__main__":
    main()
