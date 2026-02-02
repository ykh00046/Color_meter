#!/usr/bin/env python3
"""
Dependency and import check script (v7).
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

GREEN = "[92m"
RED = "[91m"
YELLOW = "[93m"
BLUE = "[94m"
RESET = "[0m"


def print_header(text: str) -> None:
    print(f"\n{BLUE}{'=' * 80}{RESET}")
    print(f"{BLUE}{text:^80}{RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}\n")


def print_success(text: str) -> None:
    print(f"{GREEN}[OK]{RESET} {text}")


def print_error(text: str) -> None:
    print(f"{RED}[FAIL]{RESET} {text}")


def print_warning(text: str) -> None:
    print(f"{YELLOW}[WARN]{RESET} {text}")


def check_requirements() -> Tuple[bool, List[str], List[str]]:
    print_header("1. Requirements.txt packages")
    requirements_path = Path(__file__).parent.parent / "requirements.txt"
    if not requirements_path.exists():
        print_error("requirements.txt not found")
        return False, [], []

    requirements = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    installed: List[str] = []
    missing: List[str] = []

    for req in requirements:
        pkg_name = req.split(">=")[0].split("==")[0].split("<")[0].split(">", 1)[0].split("[")[0].strip()
        result = subprocess.run(["pip", "show", pkg_name], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            version_line = [line for line in result.stdout.split("\n") if line.startswith("Version:")]
            version = version_line[0].split(":")[1].strip() if version_line else "unknown"
            print_success(f"{pkg_name:30} (v{version})")
            installed.append(pkg_name)
        else:
            print_error(f"{pkg_name:30} missing")
            missing.append(pkg_name)

    print(f"\nInstalled: {len(installed)}")
    print(f"Missing: {len(missing)}")

    return len(missing) == 0, installed, missing


def check_project_imports() -> Tuple[bool, List[str], List[Tuple[str, str]]]:
    print_header("2. Project import check")
    modules_to_check = [
        # v7 engine alias
        "src.engine_v7.core",
        "src.engine_v7.core.config_loader",
        "src.engine_v7.core.pipeline.analyzer",
        "src.engine_v7.core.measure.segmentation.color_masks",
        "src.engine_v7.core.types",
        # Project modules
        "src.services.inspection_service",
        "src.converters",
        "src.services.analysis_service",
        "src.utils.file_io",
        "src.utils.security",
        "src.web.app",
    ]

    success_modules: List[str] = []
    failed_modules: List[Tuple[str, str]] = []

    for module_name in modules_to_check:
        try:
            importlib.import_module(module_name)
            print_success(module_name)
            success_modules.append(module_name)
        except ImportError as exc:
            print_error(f"{module_name}: {exc}")
            failed_modules.append((module_name, str(exc)))
        except Exception as exc:
            print_warning(f"{module_name}: {type(exc).__name__}: {exc}")
            success_modules.append(module_name)

    print(f"\nSuccess: {len(success_modules)}")
    print(f"Failed: {len(failed_modules)}")
    return len(failed_modules) == 0, success_modules, failed_modules


def check_python_version() -> bool:
    print_header("0. Python version")
    py_version = sys.version_info
    version_str = f"{py_version.major}.{py_version.minor}.{py_version.micro}"
    if py_version.major == 3 and py_version.minor >= 8:
        print_success(f"Python {version_str} (>= 3.8)")
        return True
    print_error(f"Python {version_str} (>= 3.8)")
    return False


def generate_report(results: Dict[str, bool]) -> int:
    print_header("Summary")
    all_passed = all(results.values())
    for category, passed in results.items():
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  [{status}] {category}")
    print("\n" + "=" * 80)
    if all_passed:
        print(f"\n{GREEN}[SUCCESS] All checks passed.{RESET}\n")
        return 0
    print(f"\n{RED}[FAILED] Some checks failed.{RESET}\n")
    return 1


def main() -> int:
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    results: Dict[str, bool] = {}
    results["Python version"] = check_python_version()
    req_ok, _, _ = check_requirements()
    results["Requirements"] = req_ok
    import_ok, _, _ = check_project_imports()
    results["Project imports"] = import_ok
    return generate_report(results)


if __name__ == "__main__":
    raise SystemExit(main())
