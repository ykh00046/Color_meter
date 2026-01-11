#!/usr/bin/env python3
"""
ΔE Gate Tuning Script (Grid Search Optimization)

This script finds the optimal ΔE gate thresholds (pass_max, review_max)
by minimizing a weighted cost function against Ground Truth labels.

It supports:
1. Loading GT from CSV, JSON field, or Filename tokens.
2. Grid search over potential thresholds.
3. Cost-based optimization (heavily penalizing FAIL->PASS).
4. Generating a YAML config snippet for 'deltaE_gates'.

Usage:
    # 1. Using GT CSV (Recommended)
    python tools/tune_deltae_gates.py --input_dir ./out --gt_csv ./gt.csv --by_quality

    # 2. Using JSON internal field
    python tools/tune_deltae_gates.py --input_dir ./out --gt_field decision.ground_truth.label

    # 3. Using Filename tokens
    python tools/tune_deltae_gates.py --input_dir ./out --gt_from_filename

Requirements:
    - numpy
    - pyyaml (optional, for yaml output)
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Default Costs (Penalties)
# Lower is better. We penalize dangerous errors (FAIL->PASS) the most.
DEFAULT_COSTS = {
    "fail_as_pass": 10.0,
    "fail_as_review": 4.0,
    "review_as_pass": 3.0,
    "pass_as_fail": 2.0,
    "pass_as_review": 1.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Tune DeltaE Gates using Ground Truth")
    parser.add_argument("--input_dir", required=True, help="Directory containing result JSONs")

    # GT Sources
    parser.add_argument("--gt_csv", help="Path to CSV file (path, label)")
    parser.add_argument("--gt_field", help="JSON field for GT label (e.g. decision.ground_truth.label)")
    parser.add_argument("--gt_from_filename", action="store_true", help="Infer label from filename tokens")
    parser.add_argument("--filename_labels", default="PASS,REVIEW,FAIL", help="Tokens to look for (comma-separated)")

    # Tuning Options
    parser.add_argument("--by_quality", action="store_true", help="Optimize separately per quality_level")
    parser.add_argument("--min_n", type=int, default=20, help="Minimum samples required for specific bucket")
    parser.add_argument("--deltae_step", type=float, default=0.1, help="Step size for grid search")

    # Costs
    parser.add_argument("--cost_fail_as_pass", type=float, default=DEFAULT_COSTS["fail_as_pass"])
    parser.add_argument("--cost_fail_as_review", type=float, default=DEFAULT_COSTS["fail_as_review"])
    parser.add_argument("--out", help="Output YAML file path")

    return parser.parse_args()


def load_gt_csv(csv_path):
    gt_map = {}
    if not csv_path:
        return gt_map
    print(f"Loading GT from CSV: {csv_path}")
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    # Key can be full path or filename
                    path_str = row[0].strip()
                    label = row[1].strip().upper()
                    gt_map[path_str] = label
                    gt_map[os.path.basename(path_str)] = label
    except Exception as e:
        print(f"Error loading CSV: {e}")
    return gt_map


def resolve_gt(json_path, doc, gt_map, args):
    # 1. CSV
    if gt_map:
        fname = os.path.basename(json_path)
        if json_path in gt_map:
            return gt_map[json_path]
        if fname in gt_map:
            return gt_map[fname]

    # 2. JSON Field
    if args.gt_field:
        parts = args.gt_field.split(".")
        curr = doc
        found = True
        for p in parts:
            if isinstance(curr, dict) and p in curr:
                curr = curr[p]
            else:
                found = False
                break
        if found and isinstance(curr, str):
            return curr.upper()

    # 3. Filename
    if args.gt_from_filename:
        fname = os.path.basename(json_path).upper()
        tokens = args.filename_labels.split(",")
        # Check longest tokens first to avoid substring matching issues
        tokens.sort(key=len, reverse=True)
        for t in tokens:
            if t.upper() in fname:
                return t.upper()

    return None


def extract_deltae(doc):
    # Priority 1: metrics_summary (Standard v2 output)
    try:
        val = doc["metrics_summary"]["deltaE_value"]["value"]
        if val is not None:
            return float(val)
    except (KeyError, TypeError):
        pass

    # Priority 2: trajectory max (v2 diagnostics)
    try:
        val = doc["decision"]["diagnostics"]["v2_diagnostics"]["ink_match"]["trajectory_summary"]["max_off_track"]
        if val is not None:
            return float(val)
    except (KeyError, TypeError):
        pass

    # Priority 3: Cluster deltas max
    try:
        v2 = doc["decision"]["diagnostics"]["v2_diagnostics"]
        match = v2.get("ink_match", {})
        deltas = match.get("deltas") or match.get("cluster_deltas") or []
        if deltas:
            return max([float(d.get("deltaE", 0.0)) for d in deltas])
    except (KeyError, TypeError):
        pass

    return None


def calculate_loss(gt, pred, costs):
    if gt == pred:
        return 0.0

    if gt == "FAIL":
        if pred == "PASS":
            return costs["fail_as_pass"]
        if pred == "REVIEW":
            return costs["fail_as_review"]

    if gt == "REVIEW":
        if pred == "PASS":
            return costs["review_as_pass"]
        if pred == "FAIL":
            return costs["pass_as_review"] * 1.5  # Treat REVIEW as "Soft Pass/Fail"

    if gt == "PASS":
        if pred == "FAIL":
            return costs["pass_as_fail"]
        if pred == "REVIEW":
            return costs["pass_as_review"]

    return 1.0


def optimize_gates(data_points, costs, step=0.1):
    """
    Grid search for best (pass_max, review_max)
    """
    if not data_points:
        return None

    # Extract values
    values = sorted([d["val"] for d in data_points])
    labels = [d["label"] for d in data_points]

    if not values:
        return None

    min_v = 0.0  # max(0.0, min(values) - 0.5)
    max_v = min(20.0, max(values) + 1.0)  # Cap at 20 to prevent infinite search

    # Generate candidates
    # We use percentiles + uniform grid to be efficient and precise
    candidates = set()
    for p in range(0, 101, 5):
        candidates.add(float(np.percentile(values, p)))

    # Add grid
    curr = min_v
    while curr <= max_v:
        candidates.add(round(curr, 2))
        curr += step

    candidates = sorted(list(candidates))
    candidates = [c for c in candidates if c >= 0]

    best_cost = float("inf")
    best_p = 2.0
    best_r = 4.0
    best_stats = {}

    # Pre-compute labels array for speed
    labels_arr = np.array(labels)
    values_arr = np.array([d["val"] for d in data_points])

    # Brute force pairs (p, r) where p < r
    # To optimize: iterate p, then iterate r > p + gap
    min_gap = 0.3

    for i, p in enumerate(candidates):
        if p > 10.0:
            break  # Unlikely to be a pass threshold > 10

        for r in candidates[i:]:
            if r < p + min_gap:
                continue

            # Vectorized prediction
            # PASS: val <= p
            # REVIEW: p < val <= r
            # FAIL: val > r

            # Calculate cost
            current_cost = 0.0

            # Logic:
            # Mask PASS: values_arr <= p
            # Mask REVIEW: (values_arr > p) & (values_arr <= r)
            # Mask FAIL: values_arr > r

            # We iterate simply because cost function is complex (matrix)
            # Optimization: could be vectorized but N is usually < 10000, so python loop is ok

            conf_matrix = defaultdict(int)

            for idx, val in enumerate(values_arr):
                gt = labels_arr[idx]
                if val <= p:
                    pred = "PASS"
                elif val <= r:
                    pred = "REVIEW"
                else:
                    pred = "FAIL"

                loss = calculate_loss(gt, pred, costs)
                current_cost += loss
                conf_matrix[f"{gt}->{pred}"] += 1

            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p
                best_r = r
                best_stats = {
                    "cost": round(best_cost, 2),
                    "matrix": dict(conf_matrix),
                    "avg_loss": round(best_cost / len(values_arr), 3),
                }

    return {"pass_max": best_p, "review_max": best_r, "stats": best_stats}


def print_matrix(matrix):
    # Simple confusion matrix print
    labels = ["PASS", "REVIEW", "FAIL"]
    print(f"      {'Pred PASS':<10} {'Pred REVIEW':<12} {'Pred FAIL':<10}")
    for gt in labels:
        row = []
        for pred in labels:
            row.append(str(matrix.get(f"{gt}->{pred}", 0)))
        print(f"GT {gt:<4} {row[0]:<10} {row[1]:<12} {row[2]:<10}")


def main():
    args = parse_args()

    costs = DEFAULT_COSTS.copy()
    costs["fail_as_pass"] = args.cost_fail_as_pass
    costs["fail_as_review"] = args.cost_fail_as_review

    gt_map = load_gt_csv(args.gt_csv)

    data = []
    files = glob.glob(os.path.join(args.input_dir, "**/*.json"), recursive=True)
    print(f"Scanning {len(files)} files in {args.input_dir}...")

    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                doc = json.load(fp)

            label = resolve_gt(f, doc, gt_map, args)
            if not label or label not in ("PASS", "REVIEW", "FAIL"):
                continue

            val = extract_deltae(doc)
            if val is None:
                continue

            ctx = doc.get("context", {})
            method = ctx.get("deltaE_method", "76")
            # Normalize method
            if method in ["2000", "de2000", "ciede2000"]:
                method = "2000"
            else:
                method = "76"

            qual = doc.get("input_quality", {}).get("quality_level", "unknown")

            data.append({"val": float(val), "label": label, "method": method, "quality": qual})
        except Exception:
            continue

    if not data:
        print("No valid data found (check GT source or JSON format).")
        sys.exit(1)

    print(f"Loaded {len(data)} valid samples.")

    # Grouping
    by_method = defaultdict(list)
    for d in data:
        by_method[d["method"]].append(d)

    final_config = {}

    for method, items in by_method.items():
        print(f"\n=========================================")
        print(f" Method: {method} (N={len(items)})")
        print(f"=========================================")

        # 1. Global Optimization for Method
        global_res = optimize_gates(items, costs, step=args.deltae_step)
        if not global_res:
            print("Optimization failed.")
            continue

        print(f"\n[Global] Best Gates: pass_max={global_res['pass_max']}, review_max={global_res['review_max']}")
        print(f"  Avg Loss: {global_res['stats']['avg_loss']}")
        print_matrix(global_res["stats"]["matrix"])

        if not args.by_quality:
            final_config[method] = {"pass_max": global_res["pass_max"], "review_max": global_res["review_max"]}
        else:
            final_config[method] = {}
            by_q = defaultdict(list)
            for d in items:
                by_q[d["quality"]].append(d)

            for q in ["good", "medium", "poor", "very_poor"]:
                q_items = by_q.get(q, [])
                if len(q_items) >= args.min_n:
                    print(f"\n  [Quality: {q}] (N={len(q_items)})")
                    res = optimize_gates(q_items, costs, step=args.deltae_step)
                    print(f"    Best: pass={res['pass_max']}, review={res['review_max']}")
                    print(f"    Avg Loss: {res['stats']['avg_loss']}")
                    final_config[method][q] = {"pass_max": res["pass_max"], "review_max": res["review_max"]}
                else:
                    print(
                        f"\n  [Quality: {q}] Not enough samples ({len(q_items)} < {args.min_n}). Using global fallback."
                    )
                    final_config[method][q] = {
                        "pass_max": global_res["pass_max"],
                        "review_max": global_res["review_max"],
                    }

    # Generate Output
    yaml_out = {"deltaE_gates": final_config}

    if args.out:
        try:
            import yaml

            with open(args.out, "w") as f:
                yaml.dump(yaml_out, f, sort_keys=False)
            print(f"\nSaved config to {args.out}")
        except ImportError:
            import json

            with open(args.out, "w") as f:
                json.dump(yaml_out, f, indent=2)
            print(f"\nSaved config to {args.out} (JSON format)")
    else:
        print("\n--- Recommended Configuration (deltaE_gates) ---")
        print(json.dumps(yaml_out, indent=2))


if __name__ == "__main__":
    main()
