"""
Coverage Summary Generator

테스트 커버리지 결과를 분석하고 요약 리포트를 생성합니다.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_coverage_data() -> Dict:
    """Load coverage.json file"""
    coverage_file = Path("coverage.json")
    if not coverage_file.exists():
        raise FileNotFoundError("coverage.json not found. Run pytest with --cov first.")

    with open(coverage_file) as f:
        return json.load(f)


def analyze_coverage(data: Dict) -> Dict:
    """Analyze coverage data and generate summary"""
    files = data.get("files", {})

    # Categorize files by module
    modules = {
        "core": [],
        "analysis": [],
        "utils": [],
        "web": [],
        "services": [],
        "data": [],
        "other": [],
    }

    for filepath, file_data in files.items():
        if "src" not in filepath:
            continue

        summary = file_data.get("summary", {})
        percent = summary.get("percent_covered", 0)
        num_statements = summary.get("num_statements", 0)
        missing_lines = summary.get("missing_lines", 0)

        file_info = {
            "path": filepath,
            "coverage": percent,
            "statements": num_statements,
            "missing": missing_lines,
        }

        # Categorize
        if "src/core" in filepath or "src\\core" in filepath:
            modules["core"].append(file_info)
        elif "src/analysis" in filepath or "src\\analysis" in filepath:
            modules["analysis"].append(file_info)
        elif "src/utils" in filepath or "src\\utils" in filepath:
            modules["utils"].append(file_info)
        elif "src/web" in filepath or "src\\web" in filepath:
            modules["web"].append(file_info)
        elif "src/services" in filepath or "src\\services" in filepath:
            modules["services"].append(file_info)
        elif "src/data" in filepath or "src\\data" in filepath:
            modules["data"].append(file_info)
        else:
            modules["other"].append(file_info)

    # Sort by coverage
    for module in modules.values():
        module.sort(key=lambda x: x["coverage"], reverse=True)

    return modules


def print_summary(modules: Dict, total_coverage: float):
    """Print coverage summary"""
    print("\n" + "=" * 80)
    print("  TEST COVERAGE SUMMARY")
    print("=" * 80)
    print(f"\n  Overall Coverage: {total_coverage:.2f}%\n")

    for module_name, files in modules.items():
        if not files:
            continue

        total_stmts = sum(f["statements"] for f in files)
        total_missing = sum(f["missing"] for f in files)
        if total_stmts > 0:
            module_coverage = ((total_stmts - total_missing) / total_stmts) * 100
        else:
            module_coverage = 0.0

        print(f"\n{module_name.upper()} MODULE:")
        print(f"  Coverage: {module_coverage:.2f}% ({len(files)} files)")
        print(f"  Statements: {total_stmts}, Missing: {total_missing}")

        # Top 3 and bottom 3 files
        if len(files) > 0:
            print("\n  Top covered:")
            for f in files[:3]:
                name = Path(f["path"]).name
                print(f"    {name:40} {f['coverage']:6.2f}%")

            if len(files) > 3:
                print("\n  Needs attention:")
                for f in files[-3:]:
                    name = Path(f["path"]).name
                    print(f"    {name:40} {f['coverage']:6.2f}%")

    print("\n" + "=" * 80)


def generate_badge(coverage: float) -> str:
    """Generate coverage badge markdown"""
    if coverage >= 80:
        color = "brightgreen"
    elif coverage >= 70:
        color = "green"
    elif coverage >= 60:
        color = "yellow"
    elif coverage >= 40:
        color = "orange"
    else:
        color = "red"

    badge = f"![Coverage](https://img.shields.io/badge/coverage-{coverage:.0f}%25-{color})"
    return badge


def main():
    try:
        data = load_coverage_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    total_coverage = data["totals"]["percent_covered"]
    modules = analyze_coverage(data)

    print_summary(modules, total_coverage)

    # Generate badge
    badge = generate_badge(total_coverage)
    print(f"\nCoverage Badge (for README.md):")
    print(f"  {badge}")

    print(f"\nHTML Report: file://{Path.cwd()}/htmlcov/index.html")
    print()

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
