from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _unknown_keys(base: Dict[str, Any], override: Dict[str, Any], prefix: str = "") -> List[str]:
    unknown: List[str] = []
    for key, value in override.items():
        path = f"{prefix}{key}"
        if key not in base:
            unknown.append(path)
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            unknown.extend(_unknown_keys(base[key], value, prefix=f"{path}."))
    return unknown


def load_cfg(base_path: str) -> Tuple[Dict[str, Any], List[str], List[str]]:
    base_path = str(base_path)
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    return base, [base_path], []


def load_cfg_with_sku(
    base_path: str,
    sku: str | None = None,
    sku_dir: str | None = None,
    strict_unknown: bool = False,
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    base_path = str(base_path)
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    sources = [base_path]
    warnings: List[str] = []

    if sku:
        base_dir = Path(base_path).parent
        sku_root = Path(sku_dir) if sku_dir else (base_dir / "sku")
        sku_path = sku_root / f"{sku}.json"
        if sku_path.exists():
            override = json.loads(sku_path.read_text(encoding="utf-8"))
            warnings = _unknown_keys(base, override)
            if strict_unknown and warnings:
                raise ValueError(f"Unknown cfg keys in SKU override: {', '.join(warnings)}")
            merged = deep_merge(deepcopy(base), override)
            return merged, sources + [str(sku_path)], warnings

    return base, sources, warnings
