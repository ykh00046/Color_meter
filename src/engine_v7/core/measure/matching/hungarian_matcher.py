"""
Hungarian Matching for Cluster Label Stabilization

This module provides optimal assignment of detected clusters to reference (STD)
colors using the Hungarian algorithm. This prevents "label swap" issues where
the same physical ink gets different color IDs across different images.

Key Features:
- Optimal one-to-one matching using Lab color distance
- Support for partial matches (fewer detected than reference)
- Support for extra clusters (more detected than reference)
- Confidence scoring for match quality

Usage:
    from .hungarian_matcher import match_clusters_to_reference

    mapping = match_clusters_to_reference(
        detected_centroids=detected_labs,
        reference_centroids=std_labs,
    )
    # mapping.assignments[detected_idx] = reference_idx
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try to import scipy for Hungarian algorithm
try:
    from scipy.optimize import linear_sum_assignment

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class MatchResult:
    """Result of Hungarian matching between detected and reference clusters."""

    # Core mapping: detected_idx -> reference_idx (or -1 if unmatched)
    assignments: Dict[int, int]

    # Reverse mapping: reference_idx -> detected_idx (or -1 if unmatched)
    reverse_assignments: Dict[int, int]

    # Cost matrix used for matching (deltaE values)
    cost_matrix: np.ndarray

    # Per-assignment distances
    distances: Dict[int, float]

    # Overall match quality
    total_cost: float
    mean_distance: float
    max_distance: float

    # Match statistics
    n_detected: int
    n_reference: int
    n_matched: int
    n_unmatched_detected: int
    n_unmatched_reference: int

    # Quality score [0, 1] - higher is better
    confidence: float

    # Warnings
    warnings: List[str] = field(default_factory=list)


def compute_lab_distance_matrix(
    detected_centroids: np.ndarray,
    reference_centroids: np.ndarray,
    *,
    use_deltaE2000: bool = False,
) -> np.ndarray:
    """
    Compute pairwise Lab distance matrix.

    Args:
        detected_centroids: (N, 3) array of detected Lab centroids
        reference_centroids: (M, 3) array of reference Lab centroids
        use_deltaE2000: Use CIEDE2000 instead of Euclidean (slower but more perceptual)

    Returns:
        (N, M) distance matrix where [i, j] = distance(detected[i], reference[j])
    """
    N = len(detected_centroids)
    M = len(reference_centroids)

    if use_deltaE2000:
        # Use CIEDE2000 for perceptually uniform distances
        from ..metrics.ink_metrics import deltaE2000

        cost_matrix = np.zeros((N, M), dtype=np.float64)
        for i in range(N):
            for j in range(M):
                cost_matrix[i, j] = deltaE2000(detected_centroids[i], reference_centroids[j])
    else:
        # Euclidean distance in Lab space (faster, good enough for matching)
        # Broadcast: (N, 1, 3) - (1, M, 3) -> (N, M, 3) -> sum -> (N, M)
        diff = detected_centroids[:, np.newaxis, :] - reference_centroids[np.newaxis, :, :]
        cost_matrix = np.sqrt(np.sum(diff**2, axis=2))

    return cost_matrix


def _hungarian_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve assignment problem using Hungarian algorithm.

    Falls back to greedy assignment if scipy is not available.

    Args:
        cost_matrix: (N, M) cost matrix

    Returns:
        (row_indices, col_indices) of optimal assignment
    """
    if SCIPY_AVAILABLE:
        return linear_sum_assignment(cost_matrix)
    else:
        # Greedy fallback (not optimal but works without scipy)
        return _greedy_assignment(cost_matrix)


def _greedy_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy assignment as fallback when scipy is unavailable.

    Iteratively assigns the minimum cost pair until no more assignments possible.
    """
    N, M = cost_matrix.shape
    row_indices = []
    col_indices = []

    # Make a copy to mark used rows/cols
    remaining_cost = cost_matrix.copy()
    used_rows = set()
    used_cols = set()

    while len(used_rows) < N and len(used_cols) < M:
        # Find minimum in remaining matrix
        min_val = np.inf
        min_i, min_j = -1, -1

        for i in range(N):
            if i in used_rows:
                continue
            for j in range(M):
                if j in used_cols:
                    continue
                if remaining_cost[i, j] < min_val:
                    min_val = remaining_cost[i, j]
                    min_i, min_j = i, j

        if min_i < 0:
            break

        row_indices.append(min_i)
        col_indices.append(min_j)
        used_rows.add(min_i)
        used_cols.add(min_j)

    return np.array(row_indices), np.array(col_indices)


def match_clusters_to_reference(
    detected_centroids: np.ndarray,
    reference_centroids: np.ndarray,
    *,
    max_distance_threshold: float = 50.0,
    use_deltaE2000: bool = False,
    detected_ids: Optional[List[str]] = None,
    reference_ids: Optional[List[str]] = None,
) -> MatchResult:
    """
    Match detected clusters to reference colors using Hungarian algorithm.

    This solves the "label swap" problem by finding the optimal one-to-one
    assignment that minimizes total color distance.

    Args:
        detected_centroids: (N, 3) Lab centroids of detected clusters
        reference_centroids: (M, 3) Lab centroids of reference (STD) colors
        max_distance_threshold: Maximum deltaE for valid match (default 50)
        use_deltaE2000: Use perceptual CIEDE2000 distance
        detected_ids: Optional string IDs for detected clusters
        reference_ids: Optional string IDs for reference colors

    Returns:
        MatchResult with optimal assignments and quality metrics

    Example:
        # Detected: [dark_blue, light_green]
        # Reference: [light_green, dark_blue]  (different order)
        detected = np.array([[30, -10, -40], [70, -30, 20]])
        reference = np.array([[70, -30, 20], [30, -10, -40]])

        result = match_clusters_to_reference(detected, reference)
        # result.assignments = {0: 1, 1: 0}  # dark_blue->ref[1], light_green->ref[0]
    """
    detected_centroids = np.asarray(detected_centroids, dtype=np.float64)
    reference_centroids = np.asarray(reference_centroids, dtype=np.float64)

    N = len(detected_centroids)
    M = len(reference_centroids)

    warnings: List[str] = []

    # Handle edge cases
    if N == 0 or M == 0:
        return MatchResult(
            assignments={},
            reverse_assignments={},
            cost_matrix=np.zeros((N, M)),
            distances={},
            total_cost=0.0,
            mean_distance=0.0,
            max_distance=0.0,
            n_detected=N,
            n_reference=M,
            n_matched=0,
            n_unmatched_detected=N,
            n_unmatched_reference=M,
            confidence=0.0,
            warnings=["EMPTY_INPUT"],
        )

    # Compute cost matrix
    cost_matrix = compute_lab_distance_matrix(
        detected_centroids,
        reference_centroids,
        use_deltaE2000=use_deltaE2000,
    )

    # Solve assignment problem
    # Handle rectangular matrices (N != M)
    if N <= M:
        # More references than detected: all detected will be assigned
        row_ind, col_ind = _hungarian_assignment(cost_matrix)
    else:
        # More detected than references: transpose, solve, then invert
        row_ind_t, col_ind_t = _hungarian_assignment(cost_matrix.T)
        row_ind, col_ind = col_ind_t, row_ind_t

    # Build assignments
    assignments: Dict[int, int] = {}
    reverse_assignments: Dict[int, int] = {j: -1 for j in range(M)}
    distances: Dict[int, float] = {}

    for i, j in zip(row_ind, col_ind):
        dist = cost_matrix[i, j]

        # Only accept matches within threshold
        if dist <= max_distance_threshold:
            assignments[i] = j
            reverse_assignments[j] = i
            distances[i] = float(dist)
        else:
            warnings.append(f"MATCH_REJECTED_dist={dist:.1f}_threshold={max_distance_threshold}")

    # Mark unassigned detected clusters
    for i in range(N):
        if i not in assignments:
            assignments[i] = -1  # Unmatched

    # Calculate statistics
    n_matched = sum(1 for v in assignments.values() if v >= 0)
    n_unmatched_detected = N - n_matched
    n_unmatched_reference = sum(1 for v in reverse_assignments.values() if v < 0)

    matched_distances = [d for d in distances.values()]
    total_cost = sum(matched_distances)
    mean_distance = np.mean(matched_distances) if matched_distances else 0.0
    max_distance = max(matched_distances) if matched_distances else 0.0

    # Confidence score
    # Based on: match rate, distance quality, count match
    match_rate = n_matched / max(N, M) if max(N, M) > 0 else 0.0
    distance_quality = 1.0 - min(1.0, mean_distance / max_distance_threshold) if max_distance_threshold > 0 else 0.0
    count_match = 1.0 if N == M else 0.7  # Penalty for count mismatch

    confidence = float(match_rate * 0.4 + distance_quality * 0.4 + count_match * 0.2)

    # Warnings
    if n_matched < min(N, M):
        warnings.append(f"PARTIAL_MATCH_{n_matched}_of_{min(N, M)}")
    if N != M:
        warnings.append(f"COUNT_MISMATCH_detected={N}_reference={M}")

    return MatchResult(
        assignments=assignments,
        reverse_assignments=reverse_assignments,
        cost_matrix=cost_matrix,
        distances=distances,
        total_cost=total_cost,
        mean_distance=mean_distance,
        max_distance=max_distance,
        n_detected=N,
        n_reference=M,
        n_matched=n_matched,
        n_unmatched_detected=n_unmatched_detected,
        n_unmatched_reference=n_unmatched_reference,
        confidence=confidence,
        warnings=warnings,
    )


def reorder_clusters_by_reference(
    detected_data: List[Dict[str, Any]],
    match_result: MatchResult,
    *,
    reference_ids: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Reorder detected cluster data to match reference order.

    Args:
        detected_data: List of cluster metadata dicts (from build_color_masks)
        match_result: MatchResult from match_clusters_to_reference
        reference_ids: Optional reference color IDs for naming

    Returns:
        (reordered_data, reorder_meta)
        - reordered_data: Clusters reordered to match reference
        - reorder_meta: Metadata about the reordering
    """
    n_ref = match_result.n_reference
    n_det = len(detected_data)

    # Build reverse lookup: reference_idx -> detected_idx
    ref_to_det = match_result.reverse_assignments

    reordered: List[Dict[str, Any]] = []
    mapping_log: List[Dict[str, Any]] = []

    for ref_idx in range(n_ref):
        det_idx = ref_to_det.get(ref_idx, -1)

        if det_idx >= 0 and det_idx < n_det:
            # Found matching detected cluster
            cluster = detected_data[det_idx].copy()

            # Update color_id to match reference
            if reference_ids and ref_idx < len(reference_ids):
                original_id = cluster.get("color_id", f"color_{det_idx}")
                new_id = reference_ids[ref_idx]
                cluster["color_id"] = new_id
                cluster["original_color_id"] = original_id
            else:
                cluster["color_id"] = f"color_{ref_idx}"
                cluster["original_color_id"] = cluster.get("color_id", f"color_{det_idx}")

            cluster["matched_to_reference"] = ref_idx
            cluster["match_distance"] = match_result.distances.get(det_idx, 0.0)

            reordered.append(cluster)
            mapping_log.append(
                {
                    "reference_idx": ref_idx,
                    "detected_idx": det_idx,
                    "distance": match_result.distances.get(det_idx, 0.0),
                    "status": "matched",
                }
            )
        else:
            # No matching detected cluster - create placeholder
            placeholder = {
                "color_id": reference_ids[ref_idx] if reference_ids else f"color_{ref_idx}",
                "matched_to_reference": ref_idx,
                "match_distance": None,
                "area_ratio": 0.0,
                "role": "missing",
                "is_placeholder": True,
            }
            reordered.append(placeholder)
            mapping_log.append(
                {
                    "reference_idx": ref_idx,
                    "detected_idx": None,
                    "distance": None,
                    "status": "missing",
                }
            )

    # Handle extra detected clusters (not matched to any reference)
    extra_clusters: List[Dict[str, Any]] = []
    for det_idx in range(n_det):
        if match_result.assignments.get(det_idx, -1) < 0:
            cluster = detected_data[det_idx].copy()
            cluster["matched_to_reference"] = -1
            cluster["match_distance"] = None
            cluster["is_extra"] = True
            extra_clusters.append(cluster)
            mapping_log.append(
                {
                    "reference_idx": None,
                    "detected_idx": det_idx,
                    "distance": None,
                    "status": "extra",
                }
            )

    reorder_meta = {
        "n_reference": n_ref,
        "n_detected": n_det,
        "n_matched": match_result.n_matched,
        "n_missing": match_result.n_unmatched_reference,
        "n_extra": len(extra_clusters),
        "mapping": mapping_log,
        "match_confidence": match_result.confidence,
        "mean_distance": match_result.mean_distance,
        "warnings": match_result.warnings,
    }

    return reordered, reorder_meta


def stabilize_cluster_labels(
    detected_centroids: np.ndarray,
    reference_centroids: np.ndarray,
    detected_masks: Dict[str, np.ndarray],
    detected_metadata: List[Dict[str, Any]],
    *,
    reference_ids: Optional[List[str]] = None,
    max_distance_threshold: float = 50.0,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Stabilize cluster labels by matching to reference colors.

    This is the main entry point for label stabilization. It takes the output
    of color_masks and reorders everything to match reference order.

    Args:
        detected_centroids: (N, 3) Lab centroids of detected clusters
        reference_centroids: (M, 3) Lab centroids of reference colors
        detected_masks: Dict mapping color_id to mask arrays
        detected_metadata: List of cluster metadata from build_color_masks
        reference_ids: Optional reference color IDs
        max_distance_threshold: Maximum deltaE for valid match

    Returns:
        (stabilized_masks, stabilized_metadata, stabilization_info)
    """
    # Step 1: Match clusters to reference
    match_result = match_clusters_to_reference(
        detected_centroids,
        reference_centroids,
        max_distance_threshold=max_distance_threshold,
    )

    # Step 2: Reorder metadata
    reordered_metadata, reorder_meta = reorder_clusters_by_reference(
        detected_metadata,
        match_result,
        reference_ids=reference_ids,
    )

    # Step 3: Build reordered mask dict
    # Create mapping from old color_id to new color_id
    old_to_new_id: Dict[str, str] = {}
    for meta in reordered_metadata:
        if "original_color_id" in meta:
            old_to_new_id[meta["original_color_id"]] = meta["color_id"]

    stabilized_masks: Dict[str, np.ndarray] = {}
    for old_id, mask in detected_masks.items():
        new_id = old_to_new_id.get(old_id, old_id)
        stabilized_masks[new_id] = mask

    stabilization_info = {
        "match_result": {
            "assignments": match_result.assignments,
            "distances": match_result.distances,
            "confidence": match_result.confidence,
            "n_matched": match_result.n_matched,
        },
        "reorder": reorder_meta,
        "id_mapping": old_to_new_id,
    }

    return stabilized_masks, reordered_metadata, stabilization_info
