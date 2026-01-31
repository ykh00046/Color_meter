"""
Shared path configuration for the v7 engine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def _normalize_path(value: Optional[str], default: Path) -> Path:
    if value is None or value == "":
        return default.resolve()
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    return candidate


V7_ROOT = _normalize_path(os.getenv("LSE_V7_ROOT") or os.getenv("V7_ROOT"), REPO_ROOT / "src" / "engine_v7")
V7_MODELS = _normalize_path(os.getenv("LSE_V7_MODELS"), V7_ROOT / "models")
V7_RESULTS = _normalize_path(os.getenv("LSE_V7_RESULTS"), REPO_ROOT / "results" / "v7" / "web")
V7_TEST_RESULTS = _normalize_path(os.getenv("LSE_V7_TEST_RESULTS"), REPO_ROOT / "results" / "v7" / "test")


def ensure_v7_dirs() -> None:
    V7_RESULTS.mkdir(parents=True, exist_ok=True)
    V7_TEST_RESULTS.mkdir(parents=True, exist_ok=True)


def add_repo_root_to_sys_path() -> None:
    root_str = str(REPO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
