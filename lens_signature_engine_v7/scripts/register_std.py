#!/usr/bin/env python3
"""STD Registration Mode

- Purpose: baseline creation & reference registration
- Decision impact: NONE (no OK/NG judgment)
- Allowed layers:
  - v1 measurement: YES
  - v2 diagnostics: YES (shadow)
  - v3 summary/trend: NO (explicitly blocked)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from pathlib import Path as _Path
from typing import Any, Dict, List

import cv2

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from core.config_loader import load_cfg_with_sku
from core.geometry.lens_geometry import detect_lens_circle
from core.measure.segmentation.color_masks import build_color_masks
from core.signature.fit import fit_std, fit_std_multi, fit_std_per_color
from core.signature.model_io import save_model
from core.utils import apply_white_balance


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "y"}


def _load_cfg(path: str, sku: str | None = None) -> Dict[str, Any]:
    strict_unknown = _env_flag("LENS_CFG_STRICT")
    data, sources, warnings = load_cfg_with_sku(path, sku, strict_unknown=strict_unknown)
    return data, sources, warnings


def _cfg_hash(cfg: Dict[str, Any]) -> str:
    raw = json.dumps(cfg, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _now_tag() -> str:
    return datetime.now().strftime("v%Y%m%d_%H%M%S")


def _ensure_index(index_path: Path) -> Dict[str, Any]:
    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        if not data.get("schema_version"):
            data["schema_version"] = "std_registry_index.v1"
        if not data.get("engine_version"):
            data["engine_version"] = "v7"
        if not data.get("created_at"):
            data["created_at"] = datetime.now().isoformat()
        return data
    return {
        "schema_version": "std_registry_index.v1",
        "engine_version": "v7",
        "created_at": datetime.now().isoformat(),
        "updated_at": "",
        "items": [],
    }


def _update_index(
    index_data: Dict[str, Any],
    *,
    sku: str,
    ink: str,
    mode: str,
    rel_path: str,
    created_by: str | None,
    notes: str | None,
    expected_ink_count: int | None,
    color_mode: str = "aggregate",
    per_color_info: Dict[str, Dict[str, str]] | None = None,  # {color_id: {mode: rel_path}}
    color_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    items = index_data.get("items", [])
    entry = None
    for item in items:
        if item.get("sku") == sku and item.get("ink") == ink:
            entry = item
            break

    if entry is None:
        entry = {
            "sku": sku,
            "ink": ink,
            "active": {},
            "status": "INCOMPLETE",
            "notes": notes or "",
            "created_by": created_by or "",
            "created_at": datetime.now().isoformat(),
        }
        items.append(entry)

    # Set color_mode
    entry["color_mode"] = color_mode

    if color_mode == "aggregate":
        # Original aggregate mode
        entry.setdefault("active", {})[mode] = rel_path
        entry["notes"] = notes or entry.get("notes", "")
        entry["created_by"] = created_by or entry.get("created_by", "")
        if expected_ink_count is not None:
            entry["expected_ink_count"] = int(expected_ink_count)

        active = entry.get("active", {})
        if all(active.get(m) for m in ["LOW", "MID", "HIGH"]):
            entry["status"] = "ACTIVE"
        else:
            entry["status"] = "INCOMPLETE"

    elif color_mode == "per_color":
        # Per-color mode
        entry.setdefault("colors", [])

        if per_color_info is not None and color_metadata is not None:
            # Update color information
            for color_id, mode_paths in per_color_info.items():
                # Find existing color entry or create new
                color_entry = None
                for c in entry["colors"]:
                    if c.get("color_id") == color_id:
                        color_entry = c
                        break

                if color_entry is None:
                    color_entry = {
                        "color_id": color_id,
                        "active": {},
                    }
                    # Add metadata from color_metadata
                    if color_id in color_metadata:
                        color_entry.update(color_metadata[color_id])
                    entry["colors"].append(color_entry)

                # Update paths for this mode
                color_entry.setdefault("active", {}).update(mode_paths)

            # Check if all colors have LOW/MID/HIGH
            all_complete = True
            for color_entry in entry["colors"]:
                color_active = color_entry.get("active", {})
                if not all(color_active.get(m) for m in ["LOW", "MID", "HIGH"]):
                    all_complete = False
                    break

            entry["status"] = "ACTIVE" if all_complete else "INCOMPLETE"

        entry["notes"] = notes or entry.get("notes", "")
        entry["created_by"] = created_by or entry.get("created_by", "")
        if expected_ink_count is not None:
            entry["expected_ink_count"] = int(expected_ink_count)

    index_data["items"] = items
    if not index_data.get("schema_version"):
        index_data["schema_version"] = "std_registry_index.v1"
    if not index_data.get("engine_version"):
        index_data["engine_version"] = "v7"
    if not index_data.get("created_at"):
        index_data["created_at"] = datetime.now().isoformat()
    index_data["updated_at"] = datetime.now().isoformat()
    return index_data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", required=True)
    ap.add_argument("--ink", default="INK_DEFAULT")
    ap.add_argument(
        "--std_level",
        "--mode",
        dest="mode",
        required=True,
        choices=["LOW", "MID", "HIGH"],
        help="STD level (LOW/MID/HIGH). --mode is deprecated alias.",
    )
    ap.add_argument("--stds", nargs="+", required=True)
    ap.add_argument("--cfg", default=str(Path("configs") / "default.json"))
    ap.add_argument("--models_root", default=str(Path("models")))
    ap.add_argument("--created_by", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--expected_ink_count", type=int, default=None)
    ap.add_argument(
        "--color_mode",
        choices=["aggregate", "per_color"],
        default="aggregate",
        help="Color mode: 'aggregate' (default) or 'per_color'",
    )
    args = ap.parse_args()

    # Validate arguments
    if args.color_mode == "per_color" and args.expected_ink_count is None:
        raise SystemExit("Error: --expected_ink_count is required when --color_mode=per_color")

    cfg, cfg_sources, cfg_warnings = _load_cfg(args.cfg, args.sku)
    wb_cfg = cfg.get("white_balance", {})
    tag = _now_tag()
    models_root = Path(args.models_root)

    # Load images
    bgrs: List[Any] = []
    for p in args.stds:
        img = cv2.imread(p)
        if img is None:
            raise SystemExit(f"Failed to read STD image: {p}")
        if wb_cfg.get("enabled", False):
            geom = detect_lens_circle(img)
            img, _ = apply_white_balance(img, geom, wb_cfg)
        bgrs.append(img)

    if args.color_mode == "aggregate":
        # Original aggregate mode
        version_dir = models_root / args.sku / args.ink / args.mode / tag
        version_dir.mkdir(parents=True, exist_ok=True)

        if len(bgrs) == 1:
            model = fit_std(
                bgrs[0],
                R=cfg["polar"]["R"],
                T=cfg["polar"]["T"],
                r_start=cfg["signature"]["r_start"],
                r_end=cfg["signature"]["r_end"],
            )
        else:
            model = fit_std_multi(
                bgrs,
                R=cfg["polar"]["R"],
                T=cfg["polar"]["T"],
                r_start=cfg["signature"]["r_start"],
                r_end=cfg["signature"]["r_end"],
            )

        save_model(model, str(version_dir / "model"))

        meta = {
            "schema_version": "std_registry_meta.v1",
            "sku": args.sku,
            "ink": args.ink,
            "mode": args.mode,
            "version": tag,
            "color_mode": "aggregate",
            "inputs": {
                "std_images": [str(p) for p in args.stds],
                "expected_ink_count": args.expected_ink_count,
            },
            "outputs": {
                "has_band": bool(model.radial_lab_std is not None and model.radial_lab_p05 is not None),
                "R": model.meta.get("R"),
                "T": model.meta.get("T"),
                "r_start": model.meta.get("r_start"),
                "r_end": model.meta.get("r_end"),
            },
            "stats": {
                "n_std": model.meta.get("n_std"),
            },
            "engine": {
                "engine_version": "v7",
                "cfg_hash": _cfg_hash(cfg),
                "cfg_sources": cfg_sources,
                "cfg_warnings": cfg_warnings,
            },
            "created_by": args.created_by,
            "created_at": datetime.now().isoformat(),
            "notes": args.notes,
        }
        (version_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        index_path = models_root / "index.json"
        index_data = _ensure_index(index_path)
        rel_path = str((version_dir.relative_to(models_root)).as_posix())
        index_data = _update_index(
            index_data,
            sku=args.sku,
            ink=args.ink,
            mode=args.mode,
            rel_path=rel_path,
            created_by=args.created_by,
            notes=args.notes,
            expected_ink_count=args.expected_ink_count,
            color_mode="aggregate",
        )
        index_path.write_text(json.dumps(index_data, ensure_ascii=False, indent=2), encoding="utf-8")

        print(
            json.dumps(
                {
                    "saved_dir": str(version_dir),
                    "index": str(index_path),
                    "color_mode": "aggregate",
                    "status": next(
                        (
                            i["status"]
                            for i in index_data.get("items", [])
                            if i.get("sku") == args.sku and i.get("ink") == args.ink
                        ),
                        "",
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    elif args.color_mode == "per_color":
        # Per-color mode
        print(f"Per-color registration: {args.sku}/{args.ink}/{args.mode} with {len(bgrs)} STD images")

        # Generate color masks
        color_masks_list = []
        color_metadata_list = []

        for idx, bgr in enumerate(bgrs):
            masks, metadata = build_color_masks(bgr, cfg, expected_k=args.expected_ink_count)
            color_masks_list.append(masks)
            color_metadata_list.append(metadata)

            print(f"  Image {idx+1}: Found {len(metadata['colors'])} colors, warnings={metadata.get('warnings', [])}")

        # Train per-color models
        try:
            per_color_models, per_color_metadata = fit_std_per_color(
                bgrs,
                color_masks_list,
                color_metadata_list,
                R=cfg["polar"]["R"],
                T=cfg["polar"]["T"],
                r_start=cfg["signature"]["r_start"],
                r_end=cfg["signature"]["r_end"],
            )
        except ValueError as e:
            raise SystemExit(f"Error training per-color models: {e}")

        # Save per-color models
        base_dir = models_root / args.sku / args.ink / "colors"
        base_dir.mkdir(parents=True, exist_ok=True)

        per_color_info = {}  # {color_id: {mode: rel_path}}

        for color_id, model in per_color_models.items():
            color_version_dir = base_dir / color_id / args.mode / tag
            color_version_dir.mkdir(parents=True, exist_ok=True)

            save_model(model, str(color_version_dir / "model"))

            # Save per-color metadata
            color_meta = {
                "schema_version": "std_registry_meta.v1",
                "sku": args.sku,
                "ink": args.ink,
                "mode": args.mode,
                "version": tag,
                "color_mode": "per_color",
                "color_id": color_id,
                "color_metadata": per_color_metadata.get(color_id, {}),
                "inputs": {
                    "std_images": [str(p) for p in args.stds],
                    "expected_ink_count": args.expected_ink_count,
                },
                "outputs": {
                    "has_band": bool(model.radial_lab_std is not None and model.radial_lab_p05 is not None),
                    "R": model.meta.get("R"),
                    "T": model.meta.get("T"),
                    "r_start": model.meta.get("r_start"),
                    "r_end": model.meta.get("r_end"),
                },
                "stats": {
                    "n_std": model.meta.get("n_std"),
                },
                "engine": {
                    "engine_version": "v7",
                    "cfg_hash": _cfg_hash(cfg),
                    "cfg_sources": cfg_sources,
                    "cfg_warnings": cfg_warnings,
                },
                "created_by": args.created_by,
                "created_at": datetime.now().isoformat(),
                "notes": args.notes,
            }
            (color_version_dir / "meta.json").write_text(
                json.dumps(color_meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            rel_path = str((color_version_dir.relative_to(models_root)).as_posix())
            per_color_info[color_id] = {args.mode: rel_path}

            print(f"  Saved {color_id}: {color_version_dir}")

        # Update index with per-color information
        index_path = models_root / "index.json"
        index_data = _ensure_index(index_path)
        index_data = _update_index(
            index_data,
            sku=args.sku,
            ink=args.ink,
            mode=args.mode,
            rel_path="",  # Not used for per_color mode
            created_by=args.created_by,
            notes=args.notes,
            expected_ink_count=args.expected_ink_count,
            color_mode="per_color",
            per_color_info=per_color_info,
            color_metadata=per_color_metadata,
        )
        index_path.write_text(json.dumps(index_data, ensure_ascii=False, indent=2), encoding="utf-8")

        print(
            json.dumps(
                {
                    "saved_base_dir": str(base_dir),
                    "per_color_dirs": {
                        color_id: str(base_dir / color_id / args.mode / tag) for color_id in per_color_models.keys()
                    },
                    "index": str(index_path),
                    "color_mode": "per_color",
                    "colors": list(per_color_models.keys()),
                    "status": next(
                        (
                            i["status"]
                            for i in index_data.get("items", [])
                            if i.get("sku") == args.sku and i.get("ink") == args.ink
                        ),
                        "",
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
