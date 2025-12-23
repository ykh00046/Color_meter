#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def render_report(summary: dict) -> str:
    lines = []
    lines.append("# Ink Threshold Report")
    lines.append("")
    lines.append("This report summarizes ink-level delta statistics and suggests thresholds.")
    lines.append("")
    if not summary:
        lines.append("No ink data found.")
        return "\n".join(lines)

    lines.append("## Summary")
    lines.append("| Ink | Count | Mean(avg) | Max(avg) | Mean P95 | Mean P99 | Max P95 | Max P99 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for ink, stats in summary.items():
        lines.append(
            f"| {ink} | {stats.get('count', 0)} | "
            f"{_fmt(stats.get('mean_avg'))} | {_fmt(stats.get('max_avg'))} | "
            f"{_fmt(stats.get('mean_p95'))} | {_fmt(stats.get('mean_p99'))} | "
            f"{_fmt(stats.get('max_p95'))} | {_fmt(stats.get('max_p99'))} |"
        )

    lines.append("")
    lines.append("## Suggested Thresholds")
    lines.append("Rule:")
    lines.append("- Warning: P95")
    lines.append("- Fail: P99")
    lines.append("")
    for ink, stats in summary.items():
        lines.append(f"### {ink}")
        lines.append(f"- mean_delta_e warning: {_fmt(stats.get('mean_p95'))}")
        lines.append(f"- mean_delta_e fail: {_fmt(stats.get('mean_p99'))}")
        lines.append(f"- max_delta_e warning: {_fmt(stats.get('max_p95'))}")
        lines.append(f"- max_delta_e fail: {_fmt(stats.get('max_p99'))}")
        lines.append("")

    return "\n".join(lines)


def _fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ink threshold report from summary JSON.")
    parser.add_argument("--summary", default="results/compare_json/summary.json", help="Summary JSON path")
    parser.add_argument("--output", default="results/compare_json/report.md", help="Report output path")
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    report = render_report(summary)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
