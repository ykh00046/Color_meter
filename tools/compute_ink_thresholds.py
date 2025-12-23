#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np


def load_values(files):
    data = {}
    for path in files:
        try:
            obj = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        tests = obj.get("result", {}).get("tests", [])
        if not tests:
            continue
        test = tests[0]
        ink_deltas = test.get("ink_deltas", {})
        for ink, d in ink_deltas.items():
            entry = data.setdefault(ink, {"mean": [], "max": []})
            entry["mean"].append(float(d.get("mean_delta_e", 0)))
            entry["max"].append(float(d.get("max_delta_e", 0)))
    return data


def percentile(vals, p):
    return float(np.percentile(vals, p)) if vals else None


def summarize(data):
    summary = {}
    for ink, vals in data.items():
        mean_vals = vals["mean"]
        max_vals = vals["max"]
        summary[ink] = {
            "mean_p95": percentile(mean_vals, 95),
            "mean_p99": percentile(mean_vals, 99),
            "max_p95": percentile(max_vals, 95),
            "max_p99": percentile(max_vals, 99),
            "mean_avg": float(np.mean(mean_vals)) if mean_vals else None,
            "max_avg": float(np.mean(max_vals)) if max_vals else None,
            "count": len(mean_vals),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compute ink threshold percentiles from compare JSON files.")
    parser.add_argument("--input-dir", default="results/compare_json", help="Directory with compare JSON files")
    parser.add_argument("--output", default="results/compare_json/summary.json", help="Output summary path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = list(input_dir.glob("*.json"))
    data = load_values(files)
    summary = summarize(data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
