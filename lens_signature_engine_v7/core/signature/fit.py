from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from ..geometry.lens_geometry import LensGeometry, detect_lens_circle
from ..utils import bgr_to_lab
from .radial_signature import build_radial_signature, build_radial_signature_masked, to_polar
from .std_model import StdModel


def fit_std(std_bgr, *, R: int, T: int, r_start: float, r_end: float) -> StdModel:
    geom = detect_lens_circle(std_bgr)
    polar = to_polar(std_bgr, geom, R=R, T=T)
    mean_curve, p95_curve, meta = build_radial_signature(polar, r_start=r_start, r_end=r_end)
    meta.update(
        {
            "r_start": r_start,
            "r_end": r_end,
            "R": R,
            "T": T,
            "n_std": 1,
            "lab_space": "opencv_lab",
        }
    )
    return StdModel(geom=geom, radial_lab_mean=mean_curve, radial_lab_p95=p95_curve, meta=meta)


def fit_std_multi(std_bgr_list: List[np.ndarray], *, R: int, T: int, r_start: float, r_end: float) -> StdModel:
    if len(std_bgr_list) == 0:
        raise ValueError("std_bgr_list is empty")

    curves = []
    geoms = []
    for bgr in std_bgr_list:
        geom = detect_lens_circle(bgr)
        geoms.append(geom)
        polar = to_polar(bgr, geom, R=R, T=T)
        mean_curve, _, _ = build_radial_signature(polar, r_start=r_start, r_end=r_end)
        curves.append(mean_curve)

    stack = np.stack(curves, axis=0)  # (N, R', 3)
    mean = stack.mean(axis=0).astype(np.float32)
    std = stack.std(axis=0).astype(np.float32)
    p05 = np.percentile(stack, 5, axis=0).astype(np.float32)
    p95 = np.percentile(stack, 95, axis=0).astype(np.float32)
    median = np.median(stack, axis=0).astype(np.float32)
    mad = np.median(np.abs(stack - median), axis=0).astype(np.float32)

    cx = float(np.median([g.cx for g in geoms]))
    cy = float(np.median([g.cy for g in geoms]))
    r = float(np.median([g.r for g in geoms]))
    rep_geom = type(geoms[0])(cx=cx, cy=cy, r=r)

    meta = {
        "r_start": r_start,
        "r_end": r_end,
        "R": R,
        "T": T,
        "n_std": len(std_bgr_list),
        "lab_space": "opencv_lab",
    }
    return StdModel(
        geom=rep_geom,
        radial_lab_mean=mean,
        radial_lab_p95=p95,
        meta=meta,
        radial_lab_std=std,
        radial_lab_p05=p05,
        radial_lab_median=median,
        radial_lab_mad=mad,
    )


def fit_std_per_color(
    std_bgr_list: List[np.ndarray],
    color_masks_list: List[Dict[str, np.ndarray]],
    color_metadata_list: List[Dict[str, Any]],
    *,
    R: int,
    T: int,
    r_start: float,
    r_end: float,
) -> Tuple[Dict[str, StdModel], Dict[str, Any]]:
    """
    Fit per-color STD models from multi-color images.

    Args:
        std_bgr_list: List of STD BGR images (N images)
        color_masks_list: List of color mask dicts (N dicts), each with color_id -> mask
        color_metadata_list: List of color metadata dicts (N dicts) from build_color_masks
        R: Radial bins
        T: Angular bins
        r_start: Start radius ratio
        r_end: End radius ratio

    Returns:
        Tuple of:
        - per_color_models: Dict mapping color_id to StdModel
        - per_color_metadata: Dict with aggregated color metadata

    Raises:
        ValueError: If inputs are inconsistent or empty
    """
    if len(std_bgr_list) == 0:
        raise ValueError("std_bgr_list is empty")

    if len(std_bgr_list) != len(color_masks_list) or len(std_bgr_list) != len(color_metadata_list):
        raise ValueError("Length mismatch between std_bgr_list, color_masks_list, and color_metadata_list")

    # Get color IDs from first image (should be consistent across all images due to stable sorting)
    first_metadata = color_metadata_list[0]
    color_ids = [color_info["color_id"] for color_info in first_metadata["colors"]]

    # 6A) Verify all images have same color IDs (더 친절한 에러 메시지)
    for idx, metadata in enumerate(color_metadata_list):
        img_color_ids = [color_info["color_id"] for color_info in metadata["colors"]]
        if img_color_ids != color_ids:
            # 6A) expected_k와 sorting_key 정보 추가
            expected_k = metadata.get("expected_k", "unknown")
            sorting_key = metadata.get("sorting_key", "unknown")
            raise ValueError(
                f"Color ID mismatch at image {idx}: "
                f"Image 0 has {color_ids}, but image {idx} has {img_color_ids}. "
                f"Image {idx}: expected_k={expected_k}, sorting_key={sorting_key}. "
                "Ensure all images are segmented with same expected_k and consistent color ordering."
            )

    # Train model for each color
    per_color_models = {}
    per_color_metadata = {}

    for color_id in color_ids:
        # Collect data for this color across all images
        curves = []
        geoms = []

        for img_idx, bgr in enumerate(std_bgr_list):
            geom = detect_lens_circle(bgr)
            geoms.append(geom)

            polar = to_polar(bgr, geom, R=R, T=T)
            mask = color_masks_list[img_idx][color_id]  # (T, R)

            mean_curve, _, meta = build_radial_signature_masked(polar, mask, r_start=r_start, r_end=r_end)
            curves.append(mean_curve)

        # Aggregate curves across images
        stack = np.stack(curves, axis=0)  # (N, R', 3)
        mean = stack.mean(axis=0).astype(np.float32)
        std = stack.std(axis=0).astype(np.float32)
        p05 = np.percentile(stack, 5, axis=0).astype(np.float32)
        p95 = np.percentile(stack, 95, axis=0).astype(np.float32)
        median = np.median(stack, axis=0).astype(np.float32)
        mad = np.median(np.abs(stack - median), axis=0).astype(np.float32)

        # Representative geometry (median across all images)
        cx = float(np.median([g.cx for g in geoms]))
        cy = float(np.median([g.cy for g in geoms]))
        r = float(np.median([g.r for g in geoms]))
        rep_geom = LensGeometry(cx=cx, cy=cy, r=r)

        # Model metadata
        meta = {
            "r_start": r_start,
            "r_end": r_end,
            "R": R,
            "T": T,
            "n_std": len(std_bgr_list),
            "color_id": color_id,
            "lab_space": "opencv_lab",
        }

        per_color_models[color_id] = StdModel(
            geom=rep_geom,
            radial_lab_mean=mean,
            radial_lab_p95=p95,
            meta=meta,
            radial_lab_std=std,
            radial_lab_p05=p05,
            radial_lab_median=median,
            radial_lab_mad=mad,
        )

        # Aggregate color metadata (average across images)
        # Faster lookup map for color info
        maps = [{c["color_id"]: c for c in m["colors"]} for m in color_metadata_list]
        color_info_list = []
        for i, mp in enumerate(maps):
            if color_id not in mp:
                raise ValueError(f"Color ID {color_id} not found in image {i} metadata")
            color_info_list.append(mp[color_id])

        lab_centroids = np.array([info["lab_centroid"] for info in color_info_list])
        avg_lab_centroid = lab_centroids.mean(axis=0).tolist()

        area_ratios = [info["area_ratio"] for info in color_info_list]
        avg_area_ratio = float(np.mean(area_ratios))

        roles = [info["role"] for info in color_info_list]
        # Role should be consistent, take most common
        role = max(set(roles), key=roles.count)

        per_color_metadata[color_id] = {
            "color_id": color_id,
            "lab_centroid": avg_lab_centroid,
            "hex_ref": color_info_list[0]["hex_ref"],  # Use from first image
            "area_ratio": avg_area_ratio,
            "role": role,
        }

    return per_color_models, per_color_metadata
