#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _iter_inputs(paths: List[str]) -> Iterable[Path]:
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for item in path.glob("*.json"):
                yield item
        else:
            yield path


def _merge_counter(dst: Dict[str, int], src: Counter) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + int(v)


def summarize_file(path: Path, phase_filter: str | None, base_dir: Path | None) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = Counter()
    reasons = Counter()
    gate_reasons = Counter()
    best_modes = Counter()
    anomaly_types = Counter()
    phases = Counter()
    total = 0

    for item in data.get("results", []):
        decision = item.get("decision")
        if not decision:
            continue
        phase = decision.get("phase", "")
        if phase_filter and phase != phase_filter:
            continue
        phases[phase] += 1
        total += 1
        labels[decision.get("label", "UNKNOWN")] += 1
        for r in decision.get("reasons", []):
            reasons[r] += 1
        gate = decision.get("gate", {})
        for r in gate.get("reasons", []):
            gate_reasons[r] += 1
        if decision.get("best_mode"):
            best_modes[decision["best_mode"]] += 1
        anomaly = decision.get("anomaly")
        if isinstance(anomaly, dict):
            atype = anomaly.get("type", "")
            if atype:
                anomaly_types[atype] += 1

    file_label = path.name
    if base_dir is not None:
        try:
            file_label = str(path.resolve().relative_to(base_dir.resolve()))
        except ValueError:
            file_label = path.name

    return {
        "file": file_label,
        "total": total,
        "phases": dict(phases),
        "labels": dict(labels),
        "reasons": dict(reasons),
        "gate_reasons": dict(gate_reasons),
        "best_modes": dict(best_modes),
        "anomaly_types": dict(anomaly_types),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="JSON files or directories")
    ap.add_argument("--phase", default="", help="Filter by phase (INSPECTION or STD_REGISTRATION)")
    ap.add_argument("--out", default="", help="Optional output JSON path")
    ap.add_argument("--base_dir", default="", help="Base directory for file labels")
    args = ap.parse_args()

    phase_filter = args.phase if args.phase else None
    base_dir = Path(args.base_dir).resolve() if args.base_dir else None

    summaries = []
    totals = Counter()
    total_images = 0

    for path in _iter_inputs(args.inputs):
        if not path.exists():
            continue
        summary = summarize_file(path, phase_filter, base_dir)
        summaries.append(summary)
        total_images += summary["total"]
        _merge_counter(totals, Counter(summary["labels"]))

    payload = {
        "phase_filter": phase_filter,
        "total_images": total_images,
        "label_counts": dict(totals),
        "files": summaries,
    }

    out_text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(out_text, encoding="utf-8")
    print(out_text)


if __name__ == "__main__":
    main()
