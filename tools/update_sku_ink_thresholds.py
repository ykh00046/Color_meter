#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def update_sku(path, summary):
    sku = json.loads(Path(path).read_text(encoding="utf-8"))
    ink_thresholds = sku.setdefault("params", {}).setdefault("ink_thresholds", {})
    for ink, s in summary.items():
        mean_p95 = s.get("mean_p95")
        max_p95 = s.get("max_p95")
        mean_p99 = s.get("mean_p99")
        max_p99 = s.get("max_p99")
        ink_thresholds[ink] = {
            "mean_delta_e": round(mean_p95, 2) if mean_p95 is not None else None,
            "max_delta_e": round(max_p95, 2) if max_p95 is not None else None,
            "fail_mean_delta_e": round(mean_p99, 2) if mean_p99 is not None else None,
            "fail_max_delta_e": round(max_p99, 2) if max_p99 is not None else None,
        }

    Path(path).write_text(json.dumps(sku, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Update SKU ink thresholds from summary JSON.")
    parser.add_argument("--summary", default="results/compare_json/summary.json", help="Summary JSON path")
    parser.add_argument("--sku", required=True, help="SKU config JSON path")
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    update_sku(args.sku, summary)


if __name__ == "__main__":
    main()
