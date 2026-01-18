#!/usr/bin/env python
"""
Lighthouse Performance Testing Script

Runs Lighthouse audits on the web application and generates reports.
Requires: npm install -g lighthouse

Usage:
    python scripts/lighthouse_test.py [--server-url URL]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Pages to test
PAGES_TO_TEST = [
    ("/", "home"),
    ("/v7", "v7_console"),
    ("/single_analysis", "single_analysis"),
]

# Lighthouse categories
CATEGORIES = [
    "performance",
    "accessibility",
    "best-practices",
    "seo",
]


def check_lighthouse_installed():
    """Check if Lighthouse CLI is installed."""
    try:
        result = subprocess.run(
            ["lighthouse", "--version"],
            capture_output=True,
            text=True,
            shell=True,
        )
        if result.returncode == 0:
            print(f"Lighthouse version: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass

    print("Lighthouse CLI not found. Install with: npm install -g lighthouse")
    return False


def run_lighthouse(url: str, output_path: str, categories: list = None):
    """
    Run Lighthouse on a URL and save the report.

    Args:
        url: The URL to test
        output_path: Path for the output report (without extension)
        categories: List of categories to audit

    Returns:
        dict: Lighthouse scores or None if failed
    """
    if categories is None:
        categories = CATEGORIES

    # Build command
    cmd = [
        "lighthouse",
        url,
        "--output=json,html",
        f"--output-path={output_path}",
        "--chrome-flags=--headless --no-sandbox",
        "--quiet",
    ]

    # Add categories
    for cat in categories:
        cmd.append(f"--only-categories={cat}")

    print(f"  Running Lighthouse on {url}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=True,
            timeout=120,
        )

        if result.returncode != 0:
            print(f"  Error: {result.stderr}")
            return None

        # Read JSON report
        json_path = f"{output_path}.report.json"
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                report = json.load(f)

            # Extract scores
            scores = {}
            for cat in categories:
                cat_key = cat.replace("-", "")
                if cat_key == "bestpractices":
                    cat_key = "best-practices"
                if cat_key in report.get("categories", {}):
                    scores[cat] = report["categories"][cat_key]["score"] * 100
                elif cat in report.get("categories", {}):
                    scores[cat] = report["categories"][cat]["score"] * 100

            return scores

    except subprocess.TimeoutExpired:
        print("  Timeout: Lighthouse took too long")
    except Exception as e:
        print(f"  Error: {e}")

    return None


def print_scores(page_name: str, scores: dict):
    """Print formatted scores."""
    if not scores:
        print(f"  {page_name}: Failed to get scores")
        return

    print(f"\n  {page_name}:")
    for cat, score in scores.items():
        # Color coding (ANSI)
        if score >= 90:
            color = "\033[92m"  # Green
            label = "GOOD"
        elif score >= 50:
            color = "\033[93m"  # Yellow
            label = "NEEDS IMPROVEMENT"
        else:
            color = "\033[91m"  # Red
            label = "POOR"
        reset = "\033[0m"

        print(f"    {cat:20s}: {color}{score:5.1f}{reset} ({label})")


def main():
    parser = argparse.ArgumentParser(description="Run Lighthouse performance tests")
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the server (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--output-dir",
        default="lighthouse_reports",
        help="Directory for reports (default: lighthouse_reports)",
    )
    args = parser.parse_args()

    # Check Lighthouse
    if not check_lighthouse_installed():
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(exist_ok=True)

    print(f"\nLighthouse Performance Test")
    print(f"Server URL: {args.server_url}")
    print(f"Output directory: {run_dir}")
    print("=" * 50)

    # Run tests
    all_scores = {}
    for path, name in PAGES_TO_TEST:
        url = f"{args.server_url}{path}"
        output_path = str(run_dir / name)

        scores = run_lighthouse(url, output_path)
        all_scores[name] = scores
        print_scores(name, scores)

    # Generate summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    # Calculate averages
    avg_scores = {}
    for cat in CATEGORIES:
        values = [s[cat] for s in all_scores.values() if s and cat in s]
        if values:
            avg_scores[cat] = sum(values) / len(values)

    if avg_scores:
        print("\nAverage Scores:")
        for cat, score in avg_scores.items():
            print(f"  {cat:20s}: {score:5.1f}")

    # Save summary JSON
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "server_url": args.server_url,
                "pages": all_scores,
                "averages": avg_scores,
            },
            f,
            indent=2,
        )

    print(f"\nReports saved to: {run_dir}")
    print(f"Summary: {summary_path}")

    # Return exit code based on performance
    if avg_scores.get("performance", 0) < 50:
        print("\nWarning: Performance score is below 50!")
        sys.exit(1)

    print("\nLighthouse tests completed successfully!")


if __name__ == "__main__":
    main()
