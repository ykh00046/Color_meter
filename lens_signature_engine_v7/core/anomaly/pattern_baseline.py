from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np

from ..geometry.lens_geometry import detect_lens_circle
from ..signature.radial_signature import to_polar
from ..utils import bgr_to_lab
from .angular_uniformity import angular_uniformity_score
from .blob_detector import detect_center_blobs


def _mean_grad(l_map: np.ndarray) -> float:
    if l_map.size == 0:
        return 0.0
    gx = np.abs(np.diff(l_map, axis=1))
    gy = np.abs(np.diff(l_map, axis=0))
    return float((gx.mean() + gy.mean()) / 2.0)


def _dot_coverage(L: np.ndarray, cov_l_delta: float) -> float:
    l_mean = float(L.mean())
    return float(np.mean(L < (l_mean - cov_l_delta)))


def extract_pattern_features(bgr: np.ndarray, *, cfg: Dict[str, Any]) -> Dict[str, float]:
    geom = detect_lens_circle(bgr)
    polar = to_polar(bgr, geom, R=cfg["polar"]["R"], T=cfg["polar"]["T"])
    lab = bgr_to_lab(polar)
    L = lab[..., 0]

    ang = angular_uniformity_score(
        polar,
        r_start=cfg["anomaly"]["r_start"],
        r_end=cfg["anomaly"]["r_end"],
    )

    blobs = detect_center_blobs(
        bgr,
        geom,
        frac=cfg["anomaly"]["center_frac"],
        min_area=cfg["anomaly"]["center_blob_min_area"],
    )
    blob_count = int(blobs.get("blob_count", 0))
    blob_area = float(sum([b.get("area", 0.0) for b in blobs.get("blobs", [])]))

    cov_l_delta = float(cfg.get("diagnostics", {}).get("coverage_l_delta", 5.0))
    coverage = _dot_coverage(L, cov_l_delta)
    edge = _mean_grad(L)

    return {
        "angular_uniformity": float(ang),
        "center_blob_count": float(blob_count),
        "center_blob_area": float(blob_area),
        "dot_coverage": float(coverage),
        "dot_edge_sharpness": float(edge),
    }


def _stat(vals: Iterable[float]) -> Dict[str, float]:
    arr = np.array(list(vals), dtype=np.float32)
    if arr.size == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def build_pattern_baseline(features_list: List[Dict[str, float]]) -> Dict[str, Any]:
    blob_counts = [f["center_blob_count"] for f in features_list]
    blob_areas = [f["center_blob_area"] for f in features_list]
    unifs = [f["angular_uniformity"] for f in features_list]
    covs = [f["dot_coverage"] for f in features_list]
    edges = [f["dot_edge_sharpness"] for f in features_list]

    return {
        "center_blob": {
            "count": {"mean": _stat(blob_counts)["mean"], "max": _stat(blob_counts)["max"]},
            "total_area": {"mean": _stat(blob_areas)["mean"], "max": _stat(blob_areas)["max"]},
        },
        "angular_uniformity": {"mean": _stat(unifs)["mean"], "max": _stat(unifs)["max"]},
        "dot": {
            "coverage": _stat(covs),
            "edge_sharpness": _stat(edges),
        },
    }


def load_images(paths: Iterable[str]) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        images.append(img)
    return images
