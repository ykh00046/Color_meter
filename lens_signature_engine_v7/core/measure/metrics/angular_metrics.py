"""
Angular Continuity Metrics

Calculates angular (theta) continuity to distinguish ink patterns from noise/reflections.
Real ink shows continuous patterns in the angular direction,
while noise or reflections show disjointed and scattered patterns.
Standardized for (T, R) polar coordinate system.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def calculate_angular_continuity(
    state_id_map: np.ndarray,
    cluster_id: int,
    r_bins: int = 10,
    *,
    polar_order: str = "TR",
    circular_theta: bool = True,
) -> float:
    """
    Calculate angular (theta) continuity score.

    Args:
        state_id_map: Cluster assignment map. (T, R) if polar_order='TR', (R, T) if 'RT'.
        cluster_id: Cluster ID to analyze.
        r_bins: Number of radial bins (default: 10).
        polar_order: 'TR' (default/recommended) or 'RT'.
        circular_theta: If True, connects theta 0 and end (circular run-length).

    Returns:
        continuity_score: 0~1 (1 = highly continuous, 0 = very disjointed)
    """
    if state_id_map.size == 0:
        return 0.0

    order = str(polar_order).upper()
    if order not in ("TR", "RT"):
        order = "TR"

    if order == "TR":
        T, R = state_id_map.shape
        theta_axis = 0
        radial_axis = 1
    else:
        R, T = state_id_map.shape
        theta_axis = 1
        radial_axis = 0

    if R == 0 or T == 0:
        return 0.0

    cluster_mask = state_id_map == cluster_id
    total_pixels = np.sum(cluster_mask)

    if total_pixels == 0:
        return 0.0

    bin_scores = []

    for r_bin in range(r_bins):
        r0 = int(R * r_bin / r_bins)
        r1 = int(R * (r_bin + 1) / r_bins)

        if r0 >= r1:
            continue

        # Slice along radial axis, check for any presence along theta axis
        if order == "TR":
            # shape (T, R_slice) -> any(axis=1) -> (T,)
            theta_slice = cluster_mask[:, r0:r1].any(axis=radial_axis)
        else:
            # shape (R_slice, T) -> any(axis=0) -> (T,)
            theta_slice = cluster_mask[r0:r1, :].any(axis=radial_axis)

        # Run-length calculation
        runs = _calculate_runs(theta_slice.astype(bool), circular=circular_theta)

        if not runs:
            continue

        avg_run = float(np.mean(runs))
        max_run = float(np.max(runs))
        num_runs = len(runs)

        avg_run_score = avg_run / T
        max_run_score = max_run / T

        # Penalty for too many breaks
        # Ideal: num_runs=1 (fully connected ring), Worst: num_runs=T (pixel dust)
        continuity_penalty = (num_runs - 1) / max(T - 1, 1)
        continuity_bonus = 1.0 - continuity_penalty

        # Weighted score
        bin_score = 0.4 * avg_run_score + 0.3 * max_run_score + 0.3 * continuity_bonus

        bin_scores.append(bin_score)

    if not bin_scores:
        return 0.0

    final_score = float(np.mean(bin_scores))
    return round(final_score, 3)


def _calculate_runs(binary_array: np.ndarray, *, circular: bool = False) -> List[int]:
    """
    Calculate run-lengths from a binary array.

    Args:
        binary_array: (N,) bool array.
        circular: If True, merges the first and last run if both ends are True.

    Returns:
        List of run lengths.
    """
    arr = np.asarray(binary_array, dtype=bool).flatten()
    n = int(arr.size)
    if n == 0:
        return []

    runs: List[int] = []
    current_run = 0

    for value in arr:
        if value:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0

    if current_run > 0:
        runs.append(current_run)

    # Handle circularity: merge first and last if applicable
    if circular and runs and arr[0] and arr[-1] and len(runs) >= 2:
        runs[0] = runs[0] + runs[-1]
        runs = runs[:-1]

    return runs


def calculate_angular_uniformity(state_id_map: np.ndarray, cluster_id: int, *, polar_order: str = "TR") -> float:
    """
    Calculate angular uniformity (CV based).
    """
    if state_id_map.size == 0:
        return 0.0

    order = str(polar_order).upper()
    if order == "TR":
        theta_axis = 0
    else:
        theta_axis = 1

    cluster_mask = state_id_map == cluster_id
    # Sum along radial axis to get counts per angle
    # If TR: sum(axis=1) -> (T,)
    # If RT: sum(axis=0) -> (T,)
    radial_axis = 1 if order == "TR" else 0
    theta_counts = np.sum(cluster_mask, axis=radial_axis)

    if np.sum(theta_counts) == 0:
        return 0.0

    mean_count = float(np.mean(theta_counts))
    std_count = float(np.std(theta_counts))

    if mean_count == 0:
        return 0.0

    cv = std_count / mean_count
    uniformity = 1.0 / (1.0 + cv)

    return round(float(uniformity), 3)


def calculate_angular_coverage(state_id_map: np.ndarray, cluster_id: int, *, polar_order: str = "TR") -> float:
    """
    Calculate angular coverage ratio (0~1).
    """
    if state_id_map.size == 0:
        return 0.0

    order = str(polar_order).upper()
    radial_axis = 1 if order == "TR" else 0

    cluster_mask = state_id_map == cluster_id
    theta_exists = np.any(cluster_mask, axis=radial_axis)  # (T,)

    T = theta_exists.size
    coverage = float(np.sum(theta_exists) / T) if T > 0 else 0.0

    return round(coverage, 3)


def build_angular_features(
    state_id_map: np.ndarray, cluster_id: int, r_bins: int = 10, *, polar_order: str = "TR"
) -> Dict[str, float]:
    """
    Build all angular features for a cluster.
    """
    continuity = calculate_angular_continuity(state_id_map, cluster_id, r_bins, polar_order=polar_order)
    uniformity = calculate_angular_uniformity(state_id_map, cluster_id, polar_order=polar_order)
    coverage = calculate_angular_coverage(state_id_map, cluster_id, polar_order=polar_order)

    angular_score = 0.50 * continuity + 0.25 * uniformity + 0.25 * coverage

    return {
        "angular_continuity": continuity,
        "angular_uniformity": uniformity,
        "angular_coverage": coverage,
        "angular_score": round(float(angular_score), 3),
    }
