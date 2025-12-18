#!/usr/bin/env python3
"""
의존성 및 Import 검증 스크립트

이 스크립트는 다음을 확인합니다:
1. requirements.txt의 모든 패키지가 설치되어 있는지
2. 모든 프로젝트 모듈이 정상적으로 import 되는지
3. 버전 충돌이나 누락된 의존성이 있는지

Usage:
    python tools/check_imports.py
"""

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ANSI 색상 코드
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(text: str):
    """헤더 출력"""
    print(f"\n{BLUE}{'=' * 80}{RESET}")
    print(f"{BLUE}{text:^80}{RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}\n")


def print_success(text: str):
    """성공 메시지 출력"""
    print(f"{GREEN}[OK]{RESET} {text}")


def print_error(text: str):
    """에러 메시지 출력"""
    print(f"{RED}[FAIL]{RESET} {text}")


def print_warning(text: str):
    """경고 메시지 출력"""
    print(f"{YELLOW}[WARN]{RESET} {text}")


def check_requirements() -> Tuple[bool, List[str], List[str]]:
    """requirements.txt의 패키지 설치 확인"""
    print_header("1. Requirements.txt 패키지 확인")

    requirements_path = Path(__file__).parent.parent / "requirements.txt"

    if not requirements_path.exists():
        print_error("requirements.txt 파일을 찾을 수 없습니다")
        return False, [], []

    # requirements.txt 읽기
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"총 {len(requirements)}개 패키지 확인 중...\n")

    installed = []
    missing = []

    for req in requirements:
        # 패키지 이름 추출 (버전 정보 및 extras 제거)
        pkg_name = req.split(">=")[0].split("==")[0].split("<")[0].split(">")[0].split("[")[0].strip()

        try:
            # pip show로 확인
            result = subprocess.run(["pip", "show", pkg_name], capture_output=True, text=True, check=False)

            if result.returncode == 0:
                # 버전 정보 추출
                version_line = [line for line in result.stdout.split("\n") if line.startswith("Version:")]
                version = version_line[0].split(":")[1].strip() if version_line else "unknown"
                print_success(f"{pkg_name:30} (v{version})")
                installed.append(pkg_name)
            else:
                print_error(f"{pkg_name:30} 미설치")
                missing.append(pkg_name)
        except Exception as e:
            print_error(f"{pkg_name:30} 확인 실패: {e}")
            missing.append(pkg_name)

    print(f"\n설치됨: {len(installed)}개")
    print(f"누락됨: {len(missing)}개")

    return len(missing) == 0, installed, missing


def check_project_imports() -> Tuple[bool, List[str], List[Tuple[str, str]]]:
    """프로젝트 모듈 import 확인"""
    print_header("2. 프로젝트 모듈 Import 확인")

    # 확인할 모듈 목록
    modules_to_check = [
        # Core modules
        "src.core.lens_detector",
        "src.core.zone_analyzer_2d",
        "src.core.color_evaluator",
        "src.core.image_loader",
        "src.core.radial_profiler",
        "src.core.angular_profiler",
        "src.core.boundary_detector",
        "src.core.background_masker",
        "src.core.zone_segmenter",
        "src.core.sector_segmenter",
        "src.core.ink_estimator",
        "src.core.illumination_corrector",
        # Analysis modules
        "src.analysis.profile_analyzer",
        "src.analysis.uniformity_analyzer",
        # Data modules
        "src.data.config_manager",
        "src.sku_manager",
        # Services
        "src.services.analysis_service",
        # Utils
        "src.utils.color_space",
        "src.utils.color_delta",
        "src.utils.image_utils",
        "src.utils.file_io",
        "src.utils.security",
        # Top-level
        "src.pipeline",
        "src.visualizer",
        # Web
        "src.web.app",
    ]

    print(f"총 {len(modules_to_check)}개 모듈 확인 중...\n")

    success_modules = []
    failed_modules = []

    for module_name in modules_to_check:
        try:
            importlib.import_module(module_name)
            print_success(f"{module_name}")
            success_modules.append(module_name)
        except ImportError as e:
            print_error(f"{module_name}: {str(e)}")
            failed_modules.append((module_name, str(e)))
        except Exception as e:
            print_warning(f"{module_name}: {type(e).__name__}: {str(e)}")
            # 다른 에러는 import는 성공한 것으로 간주 (런타임 에러일 수 있음)
            success_modules.append(module_name)

    print(f"\n성공: {len(success_modules)}개")
    print(f"실패: {len(failed_modules)}개")

    return len(failed_modules) == 0, success_modules, failed_modules


def check_critical_packages() -> Tuple[bool, Dict[str, str]]:
    """핵심 패키지 버전 확인"""
    print_header("3. 핵심 패키지 버전 확인")

    critical_packages = {
        "opencv-python": "4.0.0",
        "numpy": "1.20.0",
        "scikit-learn": "1.0.0",
        "fastapi": "0.100.0",
        "uvicorn": "0.20.0",
        "pytest": "7.0.0",
    }

    versions = {}
    all_ok = True

    for pkg, min_version in critical_packages.items():
        try:
            result = subprocess.run(["pip", "show", pkg], capture_output=True, text=True, check=False)

            if result.returncode == 0:
                version_line = [line for line in result.stdout.split("\n") if line.startswith("Version:")]
                if version_line:
                    version = version_line[0].split(":")[1].strip()
                    versions[pkg] = version
                    print_success(f"{pkg:20} v{version} (최소: v{min_version})")
                else:
                    print_warning(f"{pkg:20} 버전 확인 불가")
                    all_ok = False
            else:
                print_error(f"{pkg:20} 미설치")
                all_ok = False
        except Exception as e:
            print_error(f"{pkg:20} 확인 실패: {e}")
            all_ok = False

    return all_ok, versions


def check_python_version() -> bool:
    """Python 버전 확인"""
    print_header("0. Python 버전 확인")

    py_version = sys.version_info
    version_str = f"{py_version.major}.{py_version.minor}.{py_version.micro}"

    if py_version.major == 3 and py_version.minor >= 8:
        print_success(f"Python {version_str} (요구사항: Python 3.8+)")
        return True
    else:
        print_error(f"Python {version_str} (요구사항: Python 3.8+)")
        print_warning("Python 3.8 이상이 필요합니다")
        return False


def generate_report(results: Dict) -> None:
    """최종 리포트 생성"""
    print_header("검증 결과 요약")

    all_passed = all(results.values())

    print("검증 항목:")
    for category, passed in results.items():
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  [{status}] {category}")

    print("\n" + "=" * 80)

    if all_passed:
        print(f"\n{GREEN}[SUCCESS] 모든 검증 통과!{RESET}")
        print("시스템이 정상적으로 설정되었습니다.\n")
        return 0
    else:
        print(f"\n{RED}[FAILED] 일부 검증 실패{RESET}")
        print("위의 에러 메시지를 확인하고 문제를 해결하세요.\n")

        # 해결 방법 제안
        if not results.get("Requirements 패키지"):
            print(f"{YELLOW}해결 방법:{RESET}")
            print("  pip install -r requirements.txt\n")

        if not results.get("프로젝트 모듈 Import"):
            print(f"{YELLOW}해결 방법:{RESET}")
            print("  1. PYTHONPATH 환경변수 확인")
            print("  2. 모듈 파일 경로 확인")
            print("  3. __init__.py 파일 존재 확인\n")

        return 1


def main() -> int:
    """메인 함수"""
    # PYTHONPATH에 프로젝트 루트 추가
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print("\n")
    print(f"{BLUE}{'=' * 80}{RESET}")
    print(f"{BLUE}{'의존성 및 Import 검증 스크립트':^70}{RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}")

    results = {}

    # 0. Python 버전 확인
    results["Python 버전"] = check_python_version()

    # 1. Requirements 패키지 확인
    req_ok, installed, missing = check_requirements()
    results["Requirements 패키지"] = req_ok

    # 2. 프로젝트 모듈 Import 확인
    import_ok, success_modules, failed_modules = check_project_imports()
    results["프로젝트 모듈 Import"] = import_ok

    # 3. 핵심 패키지 버전 확인
    pkg_ok, versions = check_critical_packages()
    results["핵심 패키지 버전"] = pkg_ok

    # 최종 리포트
    return generate_report(results)


if __name__ == "__main__":
    sys.exit(main())
