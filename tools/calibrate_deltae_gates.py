#!/usr/bin/env python3
"""
ΔE Gate Calibration Script

This script analyzes 'PASS' results (or other labels) from the Lens Signature Engine
to recommend optimal PASS/REVIEW thresholds for ΔE gates.

Usage:
    python tools/calibrate_deltae_gates.py --input_dir ./out --label PASS --min_n 30
    python tools/calibrate_deltae_gates.py --input_dir ./out --out deltae_gates.yaml --by_quality

Requirements:
    - numpy
    - pyyaml (optional, for yaml output)
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate DeltaE Gates")
    parser.add_argument("--input_dir", required=True, help="Directory containing result JSONs")
    parser.add_argument("--label", default="PASS", help="Filter by decision label (default: PASS)")
    parser.add_argument("--min_n", type=int, default=10, help="Minimum samples required to propose gates")
    parser.add_argument("--out", help="Output YAML file path")
    parser.add_argument("--by_quality", action="store_true", help="Split analysis by quality_level")
    return parser.parse_args()


def load_data(input_dir, target_label):
    data = []
    files = glob.glob(os.path.join(input_dir, "**/*.json"), recursive=True)
    print(f"Found {len(files)} JSON files in {input_dir}...")

    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                doc = json.load(fp)

            decision = doc.get("decision", {})
            if decision.get("label") != target_label:
                continue

            metrics = doc.get("metrics_summary", {})
            deltae_info = metrics.get("deltaE_value", {})
            val = deltae_info.get("value")

            ctx = doc.get("context", {})
            method = ctx.get("deltaE_method", "76")

            qual = doc.get("input_quality", {}).get("quality_level", "unknown")

            if val is not None:
                data.append({"val": float(val), "method": str(method), "quality": str(qual)})
        except Exception:
            continue

    print(f"Loaded {len(data)} samples with label '{target_label}'")
    return data


def propose_gates(values):
    if not values:
        return None
    arr = np.array(values)
    p95 = np.percentile(arr, 95)
    p99 = np.percentile(arr, 99)
    max_val = np.max(arr)

    # Heuristic proposal
    # Pass max: p99 (cover 99% of normal variation)
    # Review max: max_val + margin or p99 * 1.5
    pass_max = round(float(p99), 2)
    review_max = round(max(float(max_val) * 1.1, pass_max + 0.5), 2)

    return {
        "pass_max": pass_max,
        "review_max": review_max,
        "stats": {
            "n": len(values),
            "mean": round(float(np.mean(arr)), 2),
            "p95": round(float(p95), 2),
            "max": round(float(max_val), 2),
        },
    }


def main():
    args = parse_args()
    data = load_data(args.input_dir, args.label)

    if not data:
        print("No matching data found.")
        sys.exit(1)

    # Group by method
    by_method = defaultdict(list)
    for d in data:
        by_method[d["method"]].append(d)

    results = {}

    for method, items in by_method.items():
        print(f"\n--- Method: {method} (N={len(items)}) ---")

        # Global proposal for method
        vals = [d["val"] for d in items]
        if len(vals) >= args.min_n:
            proposal = propose_gates(vals)
            print(f"  [Global] Recommended: {proposal}")
            if not args.by_quality:
                results[method] = {"pass_max": proposal["pass_max"], "review_max": proposal["review_max"]}
        else:
            print(f"  [Global] Not enough samples (<{args.min_n})")

        if args.by_quality:
            by_q = defaultdict(list)
            for d in items:
                by_q[d["quality"]].append(d["val"])

            results[method] = {}
            for q in ["good", "medium", "poor", "very_poor"]:
                q_vals = by_q.get(q, [])
                if len(q_vals) >= args.min_n:
                    prop = propose_gates(q_vals)
                    print(f"  [{q}] Recommended: {prop}")
                    results[method][q] = {"pass_max": prop["pass_max"], "review_max": prop["review_max"]}
                else:
                    print(f"  [{q}] Not enough samples ({len(q_vals)})")
                    # Fallback to global if exists, else defaults
                    if len(vals) >= args.min_n:
                        # Use global stats but maybe relax slightly for poor?
                        # For now just use global stats as base
                        global_prop = propose_gates(vals)
                        results[method][q] = {
                            "pass_max": global_prop["pass_max"],
                            "review_max": global_prop["review_max"],
                        }

    if args.out:
        output = {"deltaE_gates": results}
        try:
            import yaml

            with open(args.out, "w") as f:
                yaml.dump(output, f, sort_keys=False)
            print(f"\nSaved to {args.out}")
        except ImportError:
            import json

            with open(args.out, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nSaved to {args.out} (JSON format, install pyyaml for YAML)")


if __name__ == "__main__":
    main()
