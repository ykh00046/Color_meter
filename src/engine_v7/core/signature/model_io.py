from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict

import numpy as np

from ..types import LensGeometry
from .std_model import StdModel


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_model(model: StdModel, out_prefix: str) -> Dict[str, str]:
    out_prefix = str(out_prefix)
    npz_path = out_prefix + ".npz"
    json_path = out_prefix + ".json"

    plate_signatures = getattr(model, "plate_signatures", None)
    plate_meta = getattr(model, "plate_meta", None)
    extra_npz: Dict[str, np.ndarray] = {}
    extra_shapes: Dict[str, Any] = {}
    if isinstance(plate_signatures, dict):
        for key, value in plate_signatures.items():
            if isinstance(value, np.ndarray) and value.size > 0:
                extra_npz[key] = value
                extra_shapes[key] = list(value.shape)

    schema_version = "std_model.v2_plate" if (extra_npz or plate_meta) else "std_model.v1"

    np.savez_compressed(
        npz_path,
        radial_lab_mean=model.radial_lab_mean,
        radial_lab_p95=model.radial_lab_p95,
        radial_lab_std=model.radial_lab_std if model.radial_lab_std is not None else np.array([]),
        radial_lab_p05=model.radial_lab_p05 if model.radial_lab_p05 is not None else np.array([]),
        radial_lab_median=model.radial_lab_median if model.radial_lab_median is not None else np.array([]),
        radial_lab_mad=model.radial_lab_mad if model.radial_lab_mad is not None else np.array([]),
        **extra_npz,
    )

    payload: Dict[str, Any] = {
        "schema_version": schema_version,
        "geom": asdict(model.geom),
        "meta": model.meta,
        "has_band": bool(model.radial_lab_std is not None and model.radial_lab_p05 is not None),
        "has_robust": bool(model.radial_lab_median is not None and model.radial_lab_mad is not None),
        "has_plate": bool(extra_npz),
        "shapes": {
            "radial_lab_mean": list(model.radial_lab_mean.shape),
            "radial_lab_p95": list(model.radial_lab_p95.shape),
            **extra_shapes,
        },
    }
    if plate_meta:
        payload["plate_meta"] = plate_meta
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    return {"npz": npz_path, "json": json_path}


def load_model(prefix: str) -> StdModel:
    prefix = str(prefix)
    npz_path = prefix + ".npz"
    json_path = prefix + ".json"

    # JSON load and schema version check
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load model metadata from {json_path}: {e}")

    schema_version = meta.get("schema_version", "unknown")
    if schema_version not in ["std_model.v1", "std_model.v2_plate", "unknown"]:
        raise ValueError(f"Unsupported schema version: {schema_version}")

    # NPZ load and key validation
    try:
        arr = np.load(npz_path)
    except (FileNotFoundError, IOError) as e:
        raise ValueError(f"Failed to load model arrays from {npz_path}: {e}")

    required_keys = ["radial_lab_mean", "radial_lab_p95"]
    for key in required_keys:
        if key not in arr.files:
            raise ValueError(f"Missing required key '{key}' in {npz_path}")

    radial_lab_mean = arr["radial_lab_mean"]
    radial_lab_p95 = arr["radial_lab_p95"]
    radial_lab_std = arr["radial_lab_std"] if "radial_lab_std" in arr.files else np.array([])
    radial_lab_p05 = arr["radial_lab_p05"] if "radial_lab_p05" in arr.files else np.array([])
    radial_lab_median = arr["radial_lab_median"] if "radial_lab_median" in arr.files else np.array([])
    radial_lab_mad = arr["radial_lab_mad"] if "radial_lab_mad" in arr.files else np.array([])

    # Shape validation (compare with saved shapes)
    expected_shapes = meta.get("shapes", {})
    if expected_shapes:
        mean_shape = expected_shapes.get("radial_lab_mean")
        p95_shape = expected_shapes.get("radial_lab_p95")
        if mean_shape and list(radial_lab_mean.shape) != mean_shape:
            raise ValueError(
                f"Shape mismatch for radial_lab_mean: " f"expected {mean_shape}, got {list(radial_lab_mean.shape)}"
            )
        if p95_shape and list(radial_lab_p95.shape) != p95_shape:
            raise ValueError(
                f"Shape mismatch for radial_lab_p95: " f"expected {p95_shape}, got {list(radial_lab_p95.shape)}"
            )

    # Basic shape consistency check
    if radial_lab_mean.shape != radial_lab_p95.shape:
        raise ValueError(
            f"Shape inconsistency: radial_lab_mean {radial_lab_mean.shape} " f"!= radial_lab_p95 {radial_lab_p95.shape}"
        )

    if (not meta.get("has_band", False)) or radial_lab_std.size == 0 or radial_lab_p05.size == 0:
        radial_lab_std = None
        radial_lab_p05 = None
    if (not meta.get("has_robust", False)) or radial_lab_median.size == 0 or radial_lab_mad.size == 0:
        radial_lab_median = None
        radial_lab_mad = None

    g = meta["geom"]
    geom = LensGeometry(cx=float(g["cx"]), cy=float(g["cy"]), r=float(g["r"]))

    model = StdModel(
        geom=geom,
        radial_lab_mean=radial_lab_mean,
        radial_lab_p95=radial_lab_p95,
        meta=meta["meta"],
        radial_lab_std=radial_lab_std,
        radial_lab_p05=radial_lab_p05,
        radial_lab_median=radial_lab_median,
        radial_lab_mad=radial_lab_mad,
    )
    plate_signatures: Dict[str, np.ndarray] = {}
    for key in arr.files:
        if key.startswith("plate_"):
            plate_signatures[key] = arr[key]
    if plate_signatures:
        setattr(model, "plate_signatures", plate_signatures)
    plate_meta = meta.get("plate_meta")
    if plate_meta:
        setattr(model, "plate_meta", plate_meta)

    return model
