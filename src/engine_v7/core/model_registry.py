from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .signature.model_io import load_model


def load_index(models_root: Path) -> Dict[str, Any]:
    index_path = models_root / "index.json"
    if not index_path.exists():
        return {}
    return json.loads(index_path.read_text(encoding="utf-8"))


def find_active_entry(index_data: Dict[str, Any], sku: str, ink: str) -> Dict[str, Any] | None:
    for item in index_data.get("items", []):
        if item.get("sku") == sku and item.get("ink") == ink and item.get("status") == "ACTIVE":
            return item
    return None


def _resolve_model_dir(models_root: Path, rel_path: str) -> Path | None:
    if not rel_path:
        return None
    candidate = (models_root / rel_path).resolve()
    try:
        candidate.relative_to(models_root.resolve())
    except ValueError:
        return None
    return candidate


def load_model_prefix(prefix_dir: Path):
    """
    Load model from a version directory containing model.npz + model.json.
    """
    return load_model(str(prefix_dir / "model"))


def load_std_models(
    models_root: str,
    sku: str,
    ink: str,
    required_modes: Iterable[str] = ("LOW", "MID", "HIGH"),
    cfg_hash: str | None = None,
) -> Tuple[Dict[str, Any] | None, List[str]]:
    models_root_path = Path(models_root).resolve()
    index_data = load_index(models_root_path)
    if not index_data:
        return None, ["INDEX_NOT_FOUND"]

    entry = find_active_entry(index_data, sku, ink)
    if not entry:
        return None, ["MODEL_NOT_FOUND"]

    active = entry.get("active", {})
    reasons: List[str] = []

    for mode in required_modes:
        if not active.get(mode):
            reasons.append(f"MODEL_INCOMPLETE:{mode}")

    if reasons:
        return None, reasons

    std_models: Dict[str, Any] = {}
    for mode in required_modes:
        model_dir = _resolve_model_dir(models_root_path, active.get(mode))
        if model_dir is None:
            reasons.append(f"MODEL_PATH_INVALID:{mode}")
            continue
        try:
            std_models[mode] = load_model_prefix(model_dir)
            if cfg_hash:
                meta_path = model_dir / "meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    meta_hash = (meta.get("engine") or {}).get("cfg_hash") or ""
                    if meta_hash and meta_hash != cfg_hash:
                        reasons.append(f"CFG_MISMATCH:{mode}")
        except Exception:
            reasons.append(f"MODEL_LOAD_FAILED:{mode}")

    if reasons:
        return None, reasons

    return std_models, []


def compute_cfg_hash(cfg: Dict[str, Any]) -> str:
    raw = json.dumps(cfg, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def load_pattern_baseline(
    models_root: str,
    sku: str,
    ink: str,
) -> Tuple[Dict[str, Any] | None, List[str]]:
    models_root_path = Path(models_root).resolve()
    index_data = load_index(models_root_path)
    if not index_data:
        return None, ["INDEX_NOT_FOUND"]

    entry = find_active_entry(index_data, sku, ink)
    if not entry:
        return None, ["MODEL_NOT_FOUND"]

    rel = entry.get("pattern_baseline", "")
    if not rel:
        return None, ["PATTERN_BASELINE_NOT_FOUND"]

    baseline_path = _resolve_model_dir(models_root_path, rel)
    if baseline_path is None:
        return None, ["PATTERN_BASELINE_PATH_INVALID"]
    if not baseline_path.exists():
        return None, ["PATTERN_BASELINE_NOT_FOUND"]

    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except Exception:
        return None, ["PATTERN_BASELINE_LOAD_FAILED"]

    if baseline.get("schema_version") != "pattern_baseline.v1":
        return None, ["PATTERN_BASELINE_SCHEMA_MISMATCH"]

    return baseline, []


def load_ink_baseline(
    models_root: str,
    sku: str,
    ink: str,
) -> Tuple[Dict[str, Any] | None, List[str]]:
    models_root_path = Path(models_root).resolve()
    index_data = load_index(models_root_path)
    if not index_data:
        return None, ["INDEX_NOT_FOUND"]

    entry = find_active_entry(index_data, sku, ink)
    if not entry:
        return None, ["MODEL_NOT_FOUND"]

    rel = entry.get("ink_baseline", "")
    if not rel:
        return None, ["INK_BASELINE_NOT_FOUND"]

    baseline_path = _resolve_model_dir(models_root_path, rel)
    if baseline_path is None:
        return None, ["INK_BASELINE_PATH_INVALID"]
    if not baseline_path.exists():
        return None, ["INK_BASELINE_NOT_FOUND"]

    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except Exception:
        return None, ["INK_BASELINE_LOAD_FAILED"]

    if baseline.get("schema_version") != "ink_baseline.v1":
        return None, ["INK_BASELINE_SCHEMA_MISMATCH"]

    return baseline, []


def load_expected_ink_count(
    models_root: str,
    sku: str,
    ink: str,
) -> int | None:
    models_root_path = Path(models_root).resolve()
    index_data = load_index(models_root_path)
    if not index_data:
        return None
    entry = find_active_entry(index_data, sku, ink)
    if not entry:
        return None
    val = entry.get("expected_ink_count")
    return int(val) if val is not None else None


def get_color_mode(
    models_root: str,
    sku: str,
    ink: str,
) -> str | None:
    """
    Get the color_mode for a given SKU/ink entry.

    Returns:
        "aggregate", "per_color", or None if entry not found
    """
    models_root_path = Path(models_root).resolve()
    index_data = load_index(models_root_path)
    if not index_data:
        return None
    entry = find_active_entry(index_data, sku, ink)
    if not entry:
        return None
    return entry.get("color_mode", "aggregate")  # Default to aggregate for backward compatibility


def load_per_color_models(
    models_root: str,
    sku: str,
    ink: str,
    required_modes: Iterable[str] = ("LOW", "MID", "HIGH"),
    cfg_hash: str | None = None,
) -> Tuple[Dict[str, Dict[str, Any]] | None, Dict[str, Any] | None, List[str]]:
    """
    Load per-color models for a given SKU/ink.

    Returns:
        Tuple of:
        - per_color_models: Dict[color_id, Dict[mode, StdModel]] or None
        - color_metadata: Dict[color_id, metadata] or None
        - reasons: List of error messages (empty if successful)
    """
    models_root_path = Path(models_root).resolve()
    index_data = load_index(models_root_path)
    if not index_data:
        return None, None, ["INDEX_NOT_FOUND"]

    entry = find_active_entry(index_data, sku, ink)
    if not entry:
        return None, None, ["MODEL_NOT_FOUND"]

    # Check color_mode
    color_mode = entry.get("color_mode", "aggregate")
    if color_mode != "per_color":
        return None, None, ["NOT_PER_COLOR_MODE"]

    # Get colors array
    colors = entry.get("colors", [])
    if not colors:
        return None, None, ["NO_COLORS_DEFINED"]

    reasons: List[str] = []
    per_color_models: Dict[str, Dict[str, Any]] = {}
    color_metadata: Dict[str, Any] = {}

    for color_info in colors:
        color_id = color_info.get("color_id")
        if not color_id:
            reasons.append("COLOR_ID_MISSING")
            continue

        # Extract color metadata
        color_metadata[color_id] = {
            "color_id": color_id,
            "lab_centroid": color_info.get("lab_centroid", [0, 0, 0]),
            "hex_ref": color_info.get("hex_ref", "#000000"),
            "area_ratio": color_info.get("area_ratio", 0.0),
            "role": color_info.get("role", "ink"),
        }

        # Check if all required modes are present
        active = color_info.get("active", {})
        for mode in required_modes:
            if not active.get(mode):
                reasons.append(f"COLOR_MODEL_INCOMPLETE:{color_id}:{mode}")

        if reasons:
            continue

        # Load models for each mode
        mode_models = {}
        for mode in required_modes:
            model_dir = _resolve_model_dir(models_root_path, active.get(mode))
            if model_dir is None:
                reasons.append(f"COLOR_MODEL_PATH_INVALID:{color_id}:{mode}")
                continue

            try:
                mode_models[mode] = load_model_prefix(model_dir)

                if cfg_hash:
                    meta_path = model_dir / "meta.json"
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        meta_hash = (meta.get("engine") or {}).get("cfg_hash") or ""
                        if meta_hash and meta_hash != cfg_hash:
                            reasons.append(f"CFG_MISMATCH:{color_id}:{mode}")
            except Exception as e:
                reasons.append(f"COLOR_MODEL_LOAD_FAILED:{color_id}:{mode}:{str(e)}")

        if len(mode_models) == len(required_modes):
            per_color_models[color_id] = mode_models
        else:
            reasons.append(f"COLOR_MODEL_INCOMPLETE_AFTER_LOAD:{color_id}")

    if reasons:
        return None, None, reasons

    if not per_color_models:
        return None, None, ["NO_VALID_COLOR_MODELS"]

    return per_color_models, color_metadata, []


def load_std_models_auto(
    models_root: str,
    sku: str,
    ink: str,
    required_modes: Iterable[str] = ("LOW", "MID", "HIGH"),
    cfg_hash: str | None = None,
) -> Tuple[Dict[str, Any] | None, str | None, Dict[str, Any] | None, List[str]]:
    """
    Automatically detect color_mode and load appropriate models.

    Returns:
        Tuple of:
        - models: For aggregate mode: Dict[mode, StdModel]. For per_color: Dict[color_id, Dict[mode, StdModel]]
        - color_mode: "aggregate" or "per_color"
        - color_metadata: Dict[color_id, metadata] (only for per_color mode)
        - reasons: List of error messages (empty if successful)
    """
    color_mode = get_color_mode(models_root, sku, ink)

    if color_mode == "per_color":
        per_color_models, color_metadata, reasons = load_per_color_models(
            models_root, sku, ink, required_modes, cfg_hash
        )
        return per_color_models, "per_color", color_metadata, reasons
    else:
        # Aggregate mode (default)
        std_models, reasons = load_std_models(models_root, sku, ink, required_modes, cfg_hash)
        return std_models, "aggregate", None, reasons
