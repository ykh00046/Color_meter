#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _iter_inputs(paths: List[str]) -> Iterable[Path]:
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for item in path.glob("*.json"):
                yield item
        else:
            yield path


def _parse_results(path: Path) -> Tuple[str, str, str, List[Dict], Optional[str], Dict, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    sku = data.get("sku", "")
    ink = data.get("ink", "")
    phase = data.get("phase", "")
    generated_at = data.get("generated_at")
    active = data.get("active", {})
    active_snapshot = data.get("active_snapshot", "")
    return sku, ink, phase, data.get("results", []), generated_at, active, active_snapshot


def _base_code(reason: str) -> str:
    return reason.split(":", 1)[0]


def _get_reason_codes(decision: Dict) -> List[str]:
    codes = decision.get("reason_codes")
    if isinstance(codes, list) and codes:
        return codes
    reasons = decision.get("reasons", [])
    return [_base_code(r) for r in reasons]


def _parse_ts(ts: Optional[str], path: Path) -> float:
    if ts:
        try:
            return datetime.fromisoformat(ts).timestamp()
        except ValueError:
            pass
    return path.stat().st_mtime


def _load_index(index_path: Optional[str]) -> Dict:
    if not index_path:
        return {}
    path = Path(index_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _active_timeline(index_data: Dict) -> List[Dict]:
    timeline = []
    for item in index_data.get("items", []):
        sku = item.get("sku", "")
        ink = item.get("ink", "")
        for evt in item.get("active_history", []):
            ts = evt.get("ts") or evt.get("activated_at")
            if not ts:
                continue
            timeline.append(
                {
                    "sku": sku,
                    "ink": ink,
                    "ts": ts,
                    "action": evt.get("action", "activate"),
                    "from": evt.get("from", {}),
                    "to": evt.get("to", evt.get("active", {})),
                    "actor": evt.get("actor", evt.get("activated_by", "")),
                    "reason": evt.get("reason", ""),
                }
            )
    timeline.sort(key=lambda x: x["ts"])
    return timeline


def _window_stats(records: List[Dict]) -> Dict:
    label_counts = Counter()
    reason_counts = Counter()
    for r in records:
        label_counts[r["label"]] += 1
        for code in r["reason_codes"]:
            reason_counts[code] += 1
    total = sum(label_counts.values())
    retake = label_counts.get("RETAKE", 0)
    retake_ratio = (retake / total) if total else 0.0
    return {
        "total": total,
        "label_counts": dict(label_counts),
        "retake_ratio": retake_ratio,
        "retake_reason_top": [c for c, _ in reason_counts.most_common(3)],
    }


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def _extract_v2_row(record: Dict) -> Optional[Dict]:
    decision = record.get("decision") or {}
    diagnostics = decision.get("diagnostics") or {}
    v2 = diagnostics.get("v2_diagnostics")
    if not v2 or record.get("phase") != "INSPECTION":
        return None

    warnings = v2.get("warnings", []) or []
    sampling = v2.get("sampling", {}) or {}
    seg = v2.get("segmentation", {}) or {}
    quality = seg.get("quality", {}) or {}
    auto_est = v2.get("auto_estimation", {}) or {}

    refs = diagnostics.get("references", {}) or {}
    baseline_path = refs.get("pattern_baseline_path") or "BASELINE_UNKNOWN"
    active_versions = refs.get("pattern_baseline_active_versions") or refs.get("active_versions") or {}
    active_versions_key = "|".join([active_versions.get(m, "NA") for m in ["LOW", "MID", "HIGH"]])

    return {
        "sku": record.get("sku", ""),
        "ink": record.get("ink", ""),
        "ts": record.get("ts"),
        "baseline": baseline_path,
        "active_versions": active_versions_key,
        "warnings": warnings,
        "sampling": sampling,
        "segmentation": seg,
        "quality": quality,
        "auto_estimation": auto_est,
        "expected_ink_count": v2.get("expected_ink_count"),
    }


def _v2_group_key(row: Dict, group_by: str) -> str:
    baseline = row.get("baseline", "BASELINE_UNKNOWN")
    active_versions = row.get("active_versions", "ACTIVE_UNKNOWN")
    if group_by == "active_versions":
        return active_versions
    # fallback for manual runs without baseline path
    if baseline == "BASELINE_UNKNOWN" and active_versions != "ACTIVE_UNKNOWN":
        return active_versions
    return baseline


def _summarize_v2_group(rows: List[Dict], window: int) -> Dict:
    rows = sorted(rows, key=lambda x: x.get("ts", 0))[-window:]
    total = len(rows)
    if total == 0:
        return {"window": {"count": 0}}

    warn_counts = Counter()
    min_area_vals = []
    min_delta_vals = []
    sep_margin_vals = []
    n_pixels_vals = []
    random_fallback_vals = []
    auto_k_present = 0
    auto_k_mismatch = 0
    auto_k_conf_vals = []
    auto_k_suggested = Counter()
    auto_k_expanded = 0

    for r in rows:
        for w in r.get("warnings", []):
            warn_counts[w] += 1
        quality = r.get("quality", {})
        min_area = quality.get("min_area_ratio")
        min_delta = quality.get("min_deltaE_between_clusters")
        sep_margin = quality.get("separation_margin")
        if min_area is not None:
            min_area_vals.append(float(min_area))
        if min_delta is not None:
            min_delta_vals.append(float(min_delta))
        if sep_margin is not None:
            sep_margin_vals.append(float(sep_margin))

        sampling = r.get("sampling", {})
        n_pixels = sampling.get("n_pixels_used")
        if n_pixels is not None:
            n_pixels_vals.append(float(n_pixels))
        random_fallback_vals.append(bool(sampling.get("random_fallback_used", False)))

        auto_est = r.get("auto_estimation", {}) or {}
        suggested = auto_est.get("suggested_k")
        if suggested is not None:
            auto_k_present += 1
            auto_k_suggested[str(suggested)] += 1
            conf = auto_est.get("confidence")
            if conf is not None:
                auto_k_conf_vals.append(float(conf))
            expected = r.get("expected_ink_count")
            if expected is not None and int(suggested) != int(expected):
                auto_k_mismatch += 1
            notes = auto_est.get("notes") or []
            if any(n.endswith("true") for n in notes if "expanded_search_used" in n):
                auto_k_expanded += 1

    warning_rates = {
        "INK_CLUSTER_TOO_SMALL": warn_counts.get("INK_CLUSTER_TOO_SMALL", 0) / total,
        "INK_CLUSTER_OVERLAP_HIGH": warn_counts.get("INK_CLUSTER_OVERLAP_HIGH", 0) / total,
        "INK_SEPARATION_LOW_CONFIDENCE": warn_counts.get("INK_SEPARATION_LOW_CONFIDENCE", 0) / total,
        "AUTO_K_LOW_CONFIDENCE": warn_counts.get("AUTO_K_LOW_CONFIDENCE", 0) / total,
        "INK_COUNT_MISMATCH_SUSPECTED": warn_counts.get("INK_COUNT_MISMATCH_SUSPECTED", 0) / total,
    }

    return {
        "window": {"count": total},
        "warning_rates": warning_rates,
        "separation": {
            "min_deltaE": {"min": min(min_delta_vals) if min_delta_vals else None, "median": _median(min_delta_vals)},
            "separation_margin": {
                "min": min(sep_margin_vals) if sep_margin_vals else None,
                "median": _median(sep_margin_vals),
            },
        },
        "area_ratio": {
            "min_area_ratio": {"min": min(min_area_vals) if min_area_vals else None, "median": _median(min_area_vals)},
        },
        "sampling": {
            "random_fallback_rate": sum(1 for v in random_fallback_vals if v) / total,
            "n_pixels_used": {"min": min(n_pixels_vals) if n_pixels_vals else None, "median": _median(n_pixels_vals)},
        },
        "auto_k": {
            "present_rate": auto_k_present / total,
            "mismatch_rate": (auto_k_mismatch / auto_k_present) if auto_k_present else None,
            "expanded_search_rate": (auto_k_expanded / auto_k_present) if auto_k_present else None,
            "confidence": {
                "min": min(auto_k_conf_vals) if auto_k_conf_vals else None,
                "median": _median(auto_k_conf_vals),
            },
            "suggested_k_counts": dict(auto_k_suggested),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--model_not_found_threshold", type=int, default=5)
    ap.add_argument("--cfg_mismatch_threshold", type=int, default=3)
    ap.add_argument("--center_not_in_frame_threshold", type=int, default=5)
    ap.add_argument("--index", default="", help="index.json for active history")
    ap.add_argument("--window_size", type=int, default=20)
    ap.add_argument("--fail_on_threshold", action="store_true")
    ap.add_argument("--v2_shadow", action="store_true")
    ap.add_argument("--v2_window_size", type=int, default=20)
    ap.add_argument("--v2_group_by", default="baseline", choices=["baseline", "active_versions"])
    args = ap.parse_args()

    label_counts = Counter()
    reason_counts = Counter()
    retake_by_sku = Counter()
    phase_counts = Counter()
    sku_phase_counts = defaultdict(Counter)
    records: List[Dict] = []

    for path in _iter_inputs(args.inputs):
        if not path.exists():
            continue
        sku, ink, phase, results, generated_at, active, active_snapshot = _parse_results(path)
        ts_base = _parse_ts(generated_at, path)
        for item in results:
            decision = item.get("decision") or {}
            label = decision.get("label", "UNKNOWN")
            reason_codes = _get_reason_codes(decision)
            label_counts[label] += 1
            phase_counts[phase] += 1
            sku_phase_counts[(sku, phase)][label] += 1
            for r in reason_codes:
                reason_counts[r] += 1
            if label == "RETAKE":
                retake_by_sku[sku] += 1
            record = {
                "sku": sku,
                "ink": item.get("ink", ink),
                "phase": phase,
                "label": label,
                "reason_codes": reason_codes,
                "ts": ts_base,
                "active": active,
                "active_snapshot": active_snapshot,
                "decision": decision,
            }
            records.append(record)

    alarms = []
    if reason_counts.get("MODEL_NOT_FOUND", 0) >= args.model_not_found_threshold:
        alarms.append("MODEL_NOT_FOUND_THRESHOLD")
    if reason_counts.get("CFG_MISMATCH", 0) >= args.cfg_mismatch_threshold:
        alarms.append("CFG_MISMATCH_THRESHOLD")
    if reason_counts.get("CENTER_NOT_IN_FRAME", 0) >= args.center_not_in_frame_threshold:
        alarms.append("CENTER_NOT_IN_FRAME_THRESHOLD")

    payload = {
        "label_counts": dict(label_counts),
        "reason_counts": dict(reason_counts),
        "phase_counts": dict(phase_counts),
        "retake_by_sku": dict(retake_by_sku),
        "sku_phase_counts": {f"{k[0]}::{k[1]}": dict(v) for k, v in sku_phase_counts.items()},
        "alarms": alarms,
    }

    index_data = _load_index(args.index)
    timeline = _active_timeline(index_data) if index_data else []
    if timeline and records:
        records.sort(key=lambda x: x["ts"])
        window = max(1, int(args.window_size))
        changes = []
        for evt in timeline:
            evt_ts = _parse_ts(evt.get("ts"), Path(args.index) if args.index else Path("."))
            sku = evt.get("sku", "")
            ink = evt.get("ink", "")
            same = [r for r in records if r["sku"] == sku and r["ink"] == ink and r["phase"] == "INSPECTION"]
            before = [r for r in same if r["ts"] <= evt_ts][-window:]
            after = [r for r in same if r["ts"] > evt_ts][:window]
            changes.append(
                {
                    "sku": sku,
                    "ink": ink,
                    "ts": evt.get("ts", ""),
                    "from": evt.get("from", {}),
                    "to": evt.get("to", {}),
                    "action": evt.get("action", ""),
                    "reason": evt.get("reason", ""),
                    "before": _window_stats(before),
                    "after": _window_stats(after),
                }
            )
        payload["active_timeline"] = timeline
        payload["active_change_windows"] = changes

    if args.v2_shadow:
        rows: List[Dict] = []
        for r in records:
            row = _extract_v2_row(r)
            if row:
                rows.append(row)
        groups = defaultdict(list)
        for row in rows:
            key = _v2_group_key(row, args.v2_group_by)
            groups[key].append(row)
        v2_groups = []
        for key, items in groups.items():
            sample = items[-1]
            summary = _summarize_v2_group(items, int(args.v2_window_size))
            sample_baseline = sample.get("baseline", "BASELINE_UNKNOWN")
            sample_active_versions = sample.get("active_versions", "ACTIVE_UNKNOWN")
            active_versions_field = ""
            if args.v2_group_by == "active_versions":
                active_versions_field = sample_active_versions
            elif key != sample_baseline and sample_active_versions != "ACTIVE_UNKNOWN":
                active_versions_field = sample_active_versions
            v2_groups.append(
                {
                    "sku": sample.get("sku", ""),
                    "ink": sample.get("ink", ""),
                    "active_baseline": sample_baseline if args.v2_group_by == "baseline" else "",
                    "active_versions": active_versions_field,
                    "window_requested": int(args.v2_window_size),
                    "window_effective": summary.get("window", {}).get("count", 0),
                    **summary,
                }
            )
        payload["v2_shadow_metrics"] = {
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "v2_window_size": int(args.v2_window_size),
                "window_requested": int(args.v2_window_size),
                "group_by": args.v2_group_by,
            },
            "groups": v2_groups,
        }

    out_text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(out_text, encoding="utf-8")
    print(out_text)

    if args.fail_on_threshold and alarms:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
