"""
Primary Color Extractor - Mixed Color Filtering Module

This module extracts primary (pure) ink colors by filtering out mixed colors
that arise from ink overlapping during printing.

Key Concepts:
1. Same-color overlap: Same ink printed twice -> darker version (L* lower, a*/b* similar)
2. Different-color mixing: Two inks overlap -> intermediate color on Lab line segment

Strategy:
1. Use high k value to extract all color clusters (over-segmentation)
2. Identify mixed color candidates based on geometric and area criteria
3. Filter out mixed colors, keep only primary colors
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans

from ...config_norm import get_polar_dims
from ...geometry.lens_geometry import LensGeometry, detect_lens_circle
from ...signature.radial_signature import to_polar
from ...utils import CIELabArray, lab_cie_to_cv8, to_cie_lab
from .color_masks import build_color_masks, lab_to_hex
from .ink_segmentation import compute_adaptive_l_weight, find_optimal_k, kmeans_segment, segment_colors
from .nmf_decomposition import NMFResult, nmf_decompose_colors
from .preprocess import build_roi_mask, build_sampling_mask


@dataclass
class ColorCluster:
    """Represents a color cluster with its properties."""

    cluster_id: int
    lab_cv8: np.ndarray  # [L, a, b] in OpenCV scale (0-255)
    lab_cie: np.ndarray  # [L, a, b] in CIE scale
    hex_color: str
    area_ratio: float
    pixel_count: int
    mask: Optional[np.ndarray] = None

    # Statistical properties (for adaptive thresholding)
    lab_std: Optional[np.ndarray] = None  # [std_L, std_a, std_b] in CIE scale

    # Classification
    is_primary: bool = True
    mixed_type: str = ""  # "same_color_overlap", "different_color_mix", ""
    mixed_from: List[int] = field(default_factory=list)  # parent cluster IDs
    confidence: float = 1.0


@dataclass
class MixedColorCandidate:
    """Represents a potential mixed color."""

    cluster_id: int
    mixed_type: str  # "same_color_overlap" or "different_color_mix"
    parent_ids: List[int]
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


# Color family definitions
COLOR_FAMILIES = {
    "Blue-Cyan": {"b_max": -5, "a_max": 10},  # b* < -5, cool colors
    "Yellow-Orange": {"b_min": 10},  # b* > 10, warm colors
    "Red-Magenta": {"a_min": 15, "b_max": 15},  # a* > 15, red/pink
    "Green": {"a_max": -10, "b_min": -10, "b_max": 20},  # a* < -10, green
    "Dark-Neutral": {"L_max": 25, "chroma_max": 12},  # L* < 25, low chroma (black/dark gray)
    "Neutral": {"chroma_max": 8},  # low chroma (gray)
}


def get_chroma(a: float, b: float) -> float:
    """Calculate chroma (saturation) from a* and b*."""
    return float(np.sqrt(a * a + b * b))


def calculate_cluster_std(
    lab_map: np.ndarray,
    mask: np.ndarray,
    default_std: np.ndarray = None,
) -> np.ndarray:
    """
    Calculate standard deviation of a cluster in CIE Lab space.

    Args:
        lab_map: (H, W, 3) Lab image (CIE or OpenCV/CV8 scale)
        mask: (H, W) boolean mask for the cluster
        default_std: fallback value if calculation fails

    Returns:
        [std_L, std_a, std_b] in CIE scale
    """
    if default_std is None:
        default_std = np.array([5.0, 3.0, 3.0], dtype=np.float32)

    if mask is None or not np.any(mask):
        return default_std

    # Extract cluster pixels
    cluster_pixels = lab_map[mask]
    if cluster_pixels.shape[0] < 3:
        return default_std

    # lab_map may be either CIE Lab (L*:0-100, a/b around -128..127)
    # or OpenCV Lab (L/a/b in 0-255). Auto-detect and return std in CIE scale.
    L_max = float(np.nanmax(cluster_pixels[:, 0]))
    ab = cluster_pixels[:, 1:3]
    ab_min = float(np.nanmin(ab))
    ab_max = float(np.nanmax(ab))
    ab_mean = float(np.nanmean(ab))

    is_cv8 = (L_max > 105.0) or (ab_min >= -5.0 and ab_max <= 260.0 and ab_mean > 40.0)

    if is_cv8:
        # Calculate std in CV8 scale and convert approximately to CIE scale
        std_cv8 = np.std(cluster_pixels, axis=0)
        std_cie = np.array(
            [
                std_cv8[0] * 100.0 / 255.0,  # L std
                std_cv8[1],  # a std (same numeric scale)
                std_cv8[2],  # b std (same numeric scale)
            ],
            dtype=np.float32,
        )
        return std_cie

    # Already CIE scale
    return np.std(cluster_pixels, axis=0).astype(np.float32)


def classify_color_family(lab_cie: np.ndarray) -> str:
    """
    Classify a color into a color family based on Lab values.

    Priority: Check chromatic colors (distinct hues) first, then neutrals.
    This ensures dark blues are classified as Blue-Cyan, not Dark-Neutral.

    Args:
        lab_cie: [L*, a*, b*] in CIE scale

    Returns:
        Color family name
    """
    L, a, b = float(lab_cie[0]), float(lab_cie[1]), float(lab_cie[2])
    chroma = get_chroma(a, b)

    # 1. Check chromatic colors FIRST (colors with distinct hues)

    # Blue-Cyan: strong negative b* (even if dark)
    if b < -5 and a < 10:
        return "Blue-Cyan"

    # Red-Magenta: strong positive a*
    if a > 15 and b < 15:
        return "Red-Magenta"

    # Green: strong negative a*
    if a < -10 and -10 < b < 20:
        return "Green"

    # Yellow-Orange: strong positive b*
    if b > 10:
        return "Yellow-Orange"

    # 2. Then check neutral colors (no distinct hue)

    # Dark-Neutral: very dark AND truly achromatic (very low chroma)
    if L < 25 and chroma < 6:
        return "Dark-Neutral"

    # Default to Neutral
    return "Neutral"


def group_clusters_by_family(clusters: List[ColorCluster]) -> Dict[str, List[ColorCluster]]:
    """
    Group clusters by their color family.

    Args:
        clusters: List of color clusters

    Returns:
        Dict mapping family name to list of clusters
    """
    families: Dict[str, List[ColorCluster]] = {}

    for cluster in clusters:
        family = classify_color_family(cluster.lab_cie)
        if family not in families:
            families[family] = []
        families[family].append(cluster)

    return families


def validate_family_coverage(
    clusters: List[ColorCluster],
    expected_k: int,
) -> Dict[str, Any]:
    """
    Validate if primary clusters cover enough color families.

    Recovery triggers when:
    1. Not enough families have primaries (len < expected_k), OR
    2. A chromatic family (Blue-Cyan, Red-Magenta, Green, Yellow-Orange) exists
       in filtered but has no primary representation

    Args:
        clusters: List of all clusters (primary and filtered)
        expected_k: Expected number of ink colors

    Returns:
        Dict with validation results and recommendations
    """
    # Chromatic families are high-value (distinct hues)
    CHROMATIC_FAMILIES = {"Blue-Cyan", "Red-Magenta", "Green", "Yellow-Orange"}

    primaries = [c for c in clusters if c.is_primary]
    filtered = [c for c in clusters if not c.is_primary]

    # Group by family
    primary_families = group_clusters_by_family(primaries)
    filtered_families = group_clusters_by_family(filtered)

    # Count families with primaries
    families_with_primary = set(primary_families.keys())
    families_with_filtered = set(filtered_families.keys())

    # Find missing families (have filtered but no primary)
    missing_families = families_with_filtered - families_with_primary

    # Check basic count requirement
    count_satisfied = len(families_with_primary) >= expected_k

    # Check if any chromatic family is missing (critical!)
    missing_chromatic = missing_families & CHROMATIC_FAMILIES
    chromatic_coverage_ok = len(missing_chromatic) == 0

    # Valid only if both conditions are satisfied
    is_valid = count_satisfied and chromatic_coverage_ok

    # Find recovery candidates from filtered colors
    recovery_candidates = []
    if not is_valid:
        # Prioritize chromatic families in recovery
        for family in sorted(missing_families, key=lambda f: (f not in CHROMATIC_FAMILIES, f)):
            if family in filtered_families:
                candidates = filtered_families[family]

                # Special recovery strategy for Blue-Cyan (Sky Blue) to handle mesh/grid inks
                if family == "Blue-Cyan":
                    # 1. Try to find bright enough blue first (avoid dark navy/black)
                    bright_candidates = [c for c in candidates if c.lab_cie[0] > 30.0]

                    if bright_candidates:
                        # Sort by blueness (lowest b*)
                        candidates = sorted(bright_candidates, key=lambda c: c.lab_cie[2])
                    else:
                        # Fallback: just lowest b*
                        candidates = sorted(candidates, key=lambda c: c.lab_cie[2])
                else:
                    # Default: Get the largest filtered cluster in this family
                    candidates = sorted(candidates, key=lambda c: -c.area_ratio)

                if candidates:
                    reason = f"chromatic family missing" if family in CHROMATIC_FAMILIES else "family missing"
                    recovery_candidates.append(
                        {
                            "family": family,
                            "cluster": candidates[0],
                            "reason": reason,
                        }
                    )

    return {
        "is_valid": is_valid,
        "expected_k": expected_k,
        "families_with_primary": len(families_with_primary),
        "primary_families": list(families_with_primary),
        "missing_families": list(missing_families),
        "missing_chromatic": list(missing_chromatic),
        "recovery_candidates": recovery_candidates,
    }


def recover_filtered_by_family(
    clusters: List[ColorCluster],
    expected_k: int,
) -> Tuple[List[ColorCluster], Optional[Dict[str, Any]]]:
    """
    Recover filtered clusters if family coverage is insufficient.

    This ensures each distinct color family has at least one primary representative.

    Args:
        clusters: List of all clusters
        expected_k: Expected number of inks

    Returns:
        Tuple of (updated clusters, recovery_info dict or None if no recovery needed)
    """
    validation = validate_family_coverage(clusters, expected_k)

    if validation["is_valid"]:
        return clusters, None

    # Track recovered clusters for reporting
    recovered_list = []

    # Recover candidates
    recovered_ids = set()
    for candidate in validation["recovery_candidates"]:
        cluster = candidate["cluster"]
        cluster.is_primary = True
        cluster.mixed_type = f"recovered_{candidate['reason']}"
        recovered_ids.add(cluster.cluster_id)
        recovered_list.append(
            {
                "cluster_id": cluster.cluster_id,
                "family": candidate["family"],
                "reason": candidate["reason"],
                "lab_cie": cluster.lab_cie.tolist(),
                "hex_color": cluster.hex_color,
            }
        )

    # Check again after recovery
    primaries = [c for c in clusters if c.is_primary]
    if len(primaries) < expected_k:
        # Still not enough - recover by largest area from remaining filtered
        filtered = [c for c in clusters if not c.is_primary]
        filtered_sorted = sorted(filtered, key=lambda c: -c.area_ratio)

        for cluster in filtered_sorted:
            if len([c for c in clusters if c.is_primary]) >= expected_k:
                break
            if cluster.cluster_id not in recovered_ids:
                cluster.is_primary = True
                cluster.mixed_type = "recovered_area_fallback"
                recovered_list.append(
                    {
                        "cluster_id": cluster.cluster_id,
                        "family": classify_color_family(cluster.lab_cie),
                        "reason": "area_fallback",
                        "lab_cie": cluster.lab_cie.tolist(),
                        "hex_color": cluster.hex_color,
                    }
                )

    recovery_info = {
        "families_before": validation["families_with_primary"],
        "families_missing": validation.get("missing_families", []),
        "recovered_count": len(recovered_list),
        "recovered_clusters": recovered_list,
    }

    return clusters, recovery_info


def calculate_lab_distance(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Calculate Euclidean distance in Lab space."""
    return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))


def calculate_ab_distance(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Calculate distance in a*b* plane only (ignoring L*)."""
    return float(np.sqrt((lab1[1] - lab2[1]) ** 2 + (lab1[2] - lab2[2]) ** 2))


def is_on_line_segment(
    point: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    tolerance: float = 10.0,
    t_range: Tuple[float, float] = (0.15, 0.85),
) -> Tuple[bool, float, float]:
    """
    Check if a point lies on the line segment between p1 and p2.

    Args:
        point: The point to check [L, a, b]
        p1, p2: Line segment endpoints [L, a, b]
        tolerance: Max distance from line to be considered "on" the line
        t_range: Valid range for parameter t (0=p1, 1=p2)

    Returns:
        Tuple of (is_on_line, t_value, distance_from_line)
    """
    # Calculate t for each channel: point = p1 + t * (p2 - p1)
    diff = p2 - p1
    t_values = []

    for i in range(3):
        if abs(diff[i]) > 1.0:
            t = (point[i] - p1[i]) / diff[i]
            if t_range[0] <= t <= t_range[1]:
                t_values.append(t)

    if len(t_values) < 2:
        return False, 0.0, float("inf")

    # Check consistency of t values
    t_mean = np.mean(t_values)
    t_std = np.std(t_values)

    if t_std > 0.2:  # t values too inconsistent
        return False, t_mean, float("inf")

    # Calculate expected point and distance
    expected = p1 + t_mean * diff
    distance = calculate_lab_distance(point, expected)

    return distance < tolerance, float(t_mean), float(distance)


def detect_same_color_overlap(
    clusters: List[ColorCluster],
    ab_threshold: float = 5.0,
    l_min_diff: float = 12.0,
    l_max_diff: float = 40.0,
    area_ratio_max: float = 0.15,
    use_statistical: bool = True,
    sigma_factor: float = 2.0,
) -> List[MixedColorCandidate]:
    """
    Detect clusters that are likely same-color overlaps (darker versions).

    Criteria:
    - Similar a*, b* values (same hue)
    - Significant L* difference (darker) but not too extreme
    - Smaller area than the lighter version

    Args:
        clusters: List of color clusters
        ab_threshold: Max a*b* distance to be considered "same hue" (fallback)
        l_min_diff: Min L* difference to be considered "overlap"
        l_max_diff: Max L* difference (beyond this, it's likely a different ink)
        area_ratio_max: Max area ratio for the darker cluster
        use_statistical: If True, use cluster std for dynamic thresholds
        sigma_factor: Multiplier for sigma-based threshold (default 2.0)

    Returns:
        List of mixed color candidates
    """
    candidates = []
    n = len(clusters)

    for i in range(n):
        c_dark = clusters[i]

        for j in range(n):
            if i == j:
                continue

            c_light = clusters[j]

            # Check L* difference (dark must be darker, but not too much)
            l_diff = c_light.lab_cie[0] - c_dark.lab_cie[0]

            # Statistical L* threshold: sum of L stds
            if use_statistical and c_dark.lab_std is not None and c_light.lab_std is not None:
                l_sigma_sum = c_dark.lab_std[0] + c_light.lab_std[0]
                effective_l_min = max(l_min_diff * 0.5, sigma_factor * l_sigma_sum)
            else:
                effective_l_min = l_min_diff

            if l_diff < effective_l_min:
                continue

            # If L* difference is too large, it's likely a different ink entirely
            if l_diff > l_max_diff:
                continue

            # Check a*b* similarity with statistical threshold
            ab_dist = calculate_ab_distance(c_dark.lab_cie, c_light.lab_cie)

            # Statistical ab threshold: average of ab spreads
            if use_statistical and c_dark.lab_std is not None and c_light.lab_std is not None:
                # Combine a_std and b_std as vector norm
                ab_sigma_dark = np.sqrt(c_dark.lab_std[1] ** 2 + c_dark.lab_std[2] ** 2)
                ab_sigma_light = np.sqrt(c_light.lab_std[1] ** 2 + c_light.lab_std[2] ** 2)
                effective_ab_threshold = sigma_factor * (ab_sigma_dark + ab_sigma_light) / 2.0
                # Clamp to reasonable range [3.0, 12.0]
                effective_ab_threshold = max(3.0, min(12.0, effective_ab_threshold))
            else:
                effective_ab_threshold = ab_threshold

            if ab_dist > effective_ab_threshold:
                continue

            # Check area ratio (darker should be smaller)
            if c_dark.area_ratio > area_ratio_max:
                continue

            if c_dark.area_ratio >= c_light.area_ratio:
                continue

            # Calculate confidence based on criteria match
            conf_ab = max(0, 1.0 - ab_dist / effective_ab_threshold)
            conf_l = min(1.0, (l_diff - effective_l_min) / 20.0)
            conf_area = max(0, 1.0 - c_dark.area_ratio / area_ratio_max)
            confidence = (conf_ab + conf_l + conf_area) / 3.0

            candidates.append(
                MixedColorCandidate(
                    cluster_id=c_dark.cluster_id,
                    mixed_type="same_color_overlap",
                    parent_ids=[c_light.cluster_id],
                    confidence=confidence,
                    details={
                        "ab_distance": ab_dist,
                        "ab_threshold_used": effective_ab_threshold,
                        "l_difference": l_diff,
                        "l_threshold_used": effective_l_min,
                        "statistical_mode": use_statistical,
                        "dark_area": c_dark.area_ratio,
                        "light_area": c_light.area_ratio,
                    },
                )
            )

    return candidates


def detect_gradient_chain(
    clusters: List[ColorCluster],
    ab_threshold: float = 12.0,
    l_step_max: float = 15.0,
    min_chain_length: int = 3,
    protect_area_min: float = 0.08,
) -> List[MixedColorCandidate]:
    """
    Detect gradient chains - multiple clusters forming a continuous L* gradient.

    Gradient lenses create smooth L* transitions that result in many clusters
    with similar a*b* but progressively different L*. This function identifies
    such chains and marks intermediate clusters as mixed.

    Args:
        clusters: List of color clusters (should be sorted by L*)
        ab_threshold: Max a*b* distance to be considered same hue family
        l_step_max: Max L* step between adjacent clusters in chain
        min_chain_length: Minimum clusters to form a chain
        protect_area_min: Clusters with area >= this ratio are protected from filtering

    Returns:
        List of mixed color candidates (intermediate clusters in chains)
    """
    if len(clusters) < min_chain_length:
        return []

    candidates = []

    # Build adjacency based on a*b* similarity
    # Clusters are already sorted by L*
    n = len(clusters)

    # Find groups of clusters with similar a*b*
    visited = [False] * n
    chains = []

    for start in range(n):
        if visited[start]:
            continue

        # Try to build a chain starting from this cluster
        chain = [start]
        visited[start] = True

        current = start
        while True:
            # Look for next cluster in chain (higher L*, similar a*b*)
            found_next = False
            for next_idx in range(current + 1, n):
                if visited[next_idx]:
                    continue

                c_curr = clusters[current]
                c_next = clusters[next_idx]

                # Check a*b* similarity
                ab_dist = calculate_ab_distance(c_curr.lab_cie, c_next.lab_cie)
                if ab_dist > ab_threshold:
                    continue

                # Check L* step is reasonable
                l_diff = c_next.lab_cie[0] - c_curr.lab_cie[0]
                if l_diff > l_step_max:
                    continue

                # Add to chain
                chain.append(next_idx)
                visited[next_idx] = True
                current = next_idx
                found_next = True
                break

            if not found_next:
                break

        if len(chain) >= min_chain_length:
            chains.append(chain)

    # For each chain, mark intermediate clusters as mixed
    for chain in chains:
        if len(chain) < min_chain_length:
            continue

        # Keep first (darkest) and last (lightest) as primary
        # Mark all intermediate as gradient mixed
        chain_clusters = [clusters[i] for i in chain]
        total_area = sum(c.area_ratio for c in chain_clusters)

        # Find the cluster with largest area as the "representative"
        max_area_idx = max(range(len(chain)), key=lambda i: chain_clusters[i].area_ratio)

        for i, idx in enumerate(chain):
            c = clusters[idx]

            # Keep endpoints, max-area cluster, and large-area clusters as primary
            if i == 0 or i == len(chain) - 1 or i == max_area_idx:
                continue

            # Protect clusters with significant area
            if c.area_ratio >= protect_area_min:
                continue

            # Calculate confidence based on position in chain
            # Middle positions have higher confidence of being gradient
            position_ratio = i / (len(chain) - 1)  # 0 to 1
            conf_position = 1.0 - 2 * abs(position_ratio - 0.5)  # Peak at middle
            conf_area = 1.0 - c.area_ratio / total_area  # Smaller area = more likely gradient

            confidence = (conf_position + conf_area) / 2.0

            candidates.append(
                MixedColorCandidate(
                    cluster_id=c.cluster_id,
                    mixed_type="gradient_chain",
                    parent_ids=[
                        clusters[chain[0]].cluster_id,
                        clusters[chain[-1]].cluster_id,
                    ],  # Use cluster_id, not index
                    confidence=confidence,
                    details={
                        "chain_length": len(chain),
                        "position_in_chain": i,
                        "chain_total_area": total_area,
                        "cluster_area": c.area_ratio,
                    },
                )
            )

    return candidates


def merge_similar_clusters(
    clusters: List[ColorCluster],
    lab_threshold: float = 5.0,
) -> List[ColorCluster]:
    """
    Merge clusters that are very similar in Lab space.

    This handles cases where k-means creates multiple clusters for essentially
    the same color (e.g., due to noise or slight variations).

    Args:
        clusters: List of color clusters
        lab_threshold: Max Lab distance to merge clusters

    Returns:
        List of merged clusters
    """
    if len(clusters) <= 1:
        return clusters

    # Sort by area (largest first) to use largest as representative
    sorted_clusters = sorted(clusters, key=lambda c: -c.area_ratio)

    merged = []
    used = [False] * len(sorted_clusters)

    for i, c1 in enumerate(sorted_clusters):
        if used[i]:
            continue

        # Find all clusters similar to c1
        group = [c1]
        used[i] = True

        for j in range(i + 1, len(sorted_clusters)):
            if used[j]:
                continue

            c2 = sorted_clusters[j]
            dist = calculate_lab_distance(c1.lab_cie, c2.lab_cie)

            if dist < lab_threshold:
                group.append(c2)
                used[j] = True

        # Merge group into single cluster
        if len(group) == 1:
            merged.append(c1)
        else:
            # Combine masks and area
            total_area = sum(c.area_ratio for c in group)
            total_pixels = sum(c.pixel_count for c in group)

            # Use largest cluster's Lab as representative
            combined_mask = group[0].mask.copy() if group[0].mask is not None else None
            for c in group[1:]:
                if c.mask is not None and combined_mask is not None:
                    combined_mask = combined_mask | c.mask

            merged_cluster = ColorCluster(
                cluster_id=c1.cluster_id,
                lab_cv8=c1.lab_cv8,
                lab_cie=c1.lab_cie,
                hex_color=c1.hex_color,
                area_ratio=total_area,
                pixel_count=total_pixels,
                mask=combined_mask,
            )
            merged.append(merged_cluster)

    # Re-sort by L*
    merged.sort(key=lambda c: c.lab_cie[0])
    return merged


def consolidate_to_target_count(
    clusters: List[ColorCluster],
    target_count: int,
) -> List[ColorCluster]:
    """
    Consolidate clusters to reach target count by merging closest pairs.

    Uses agglomerative approach: repeatedly merge the two closest clusters
    until target count is reached.

    IMPORTANT: Protects the last representative of each chromatic color family
    (Blue-Cyan, Red-Magenta, Green, Yellow-Orange) to preserve color diversity.

    Args:
        clusters: List of color clusters
        target_count: Target number of clusters

    Returns:
        Consolidated list of clusters
    """
    # Chromatic families should be protected
    CHROMATIC_FAMILIES = {"Blue-Cyan", "Red-Magenta", "Green", "Yellow-Orange"}

    def count_family_members(cluster_list: List[ColorCluster], family: str) -> int:
        """Count how many clusters belong to a specific family."""
        return sum(1 for c in cluster_list if classify_color_family(c.lab_cie) == family)

    def is_sole_chromatic_rep(cluster: ColorCluster, cluster_list: List[ColorCluster]) -> bool:
        """Check if cluster is the sole representative of a chromatic family."""
        family = classify_color_family(cluster.lab_cie)
        if family not in CHROMATIC_FAMILIES:
            return False
        return count_family_members(cluster_list, family) == 1

    if len(clusters) <= target_count:
        return clusters

    working = list(clusters)

    while len(working) > target_count:
        # Find the two closest clusters (considering family protection)
        min_dist = float("inf")
        merge_i, merge_j = 0, 1

        for i in range(len(working)):
            for j in range(i + 1, len(working)):
                ci, cj = working[i], working[j]

                # Base distance: a*b* distance + L* penalty
                ab_dist = calculate_ab_distance(ci.lab_cie, cj.lab_cie)
                l_diff = abs(ci.lab_cie[0] - cj.lab_cie[0])
                dist = ab_dist + l_diff * 0.3

                # Protect sole representatives of chromatic families
                ci_sole = is_sole_chromatic_rep(ci, working)
                cj_sole = is_sole_chromatic_rep(cj, working)

                # If both are in same family, OK to merge
                ci_family = classify_color_family(ci.lab_cie)
                cj_family = classify_color_family(cj.lab_cie)
                same_family = ci_family == cj_family

                # Add large penalty if merging would eliminate a chromatic family
                if (ci_sole or cj_sole) and not same_family:
                    dist += 1000.0  # Effectively prevent this merge

                if dist < min_dist:
                    min_dist = dist
                    merge_i, merge_j = i, j

        # Merge the two closest clusters
        c1, c2 = working[merge_i], working[merge_j]

        # Weighted average based on area
        total_area = c1.area_ratio + c2.area_ratio
        total_pixels = c1.pixel_count + c2.pixel_count

        # Use area-weighted Lab values
        w1 = c1.area_ratio / total_area
        w2 = c2.area_ratio / total_area
        new_lab_cie = w1 * c1.lab_cie + w2 * c2.lab_cie
        new_lab_cv8 = w1 * c1.lab_cv8 + w2 * c2.lab_cv8

        # Combine masks
        combined_mask = None
        if c1.mask is not None and c2.mask is not None:
            combined_mask = c1.mask | c2.mask

        # Use larger cluster's ID and hex
        representative = c1 if c1.area_ratio >= c2.area_ratio else c2

        merged = ColorCluster(
            cluster_id=representative.cluster_id,
            lab_cv8=new_lab_cv8,
            lab_cie=new_lab_cie,
            hex_color=representative.hex_color,
            area_ratio=total_area,
            pixel_count=total_pixels,
            mask=combined_mask,
        )

        # Remove merged clusters and add new one
        working = [c for idx, c in enumerate(working) if idx not in (merge_i, merge_j)]
        working.append(merged)

    # Re-sort by L*
    working.sort(key=lambda c: c.lab_cie[0])
    return working


def detect_different_color_mixing(
    clusters: List[ColorCluster],
    distance_threshold: float = 8.0,
    t_range: Tuple[float, float] = (0.2, 0.8),
    area_ratio_max: float = 0.20,
    use_statistical: bool = True,
    sigma_factor: float = 1.5,
) -> List[MixedColorCandidate]:
    """
    Detect clusters that are likely mixtures of two different colors.

    Uses "Statistical Corridor" approach: the corridor width is determined by
    the interpolated standard deviations of the two parent clusters.

    Criteria:
    - Located on the line segment between two other clusters in Lab space
    - Within the statistical corridor of the parent clusters
    - Smaller area than both parent clusters

    Args:
        clusters: List of color clusters
        distance_threshold: Max distance from line to be considered "mixed" (fallback)
        t_range: Valid range for mixing ratio (0.2-0.8 = actual mixing)
        area_ratio_max: Max area ratio for the mixed cluster
        use_statistical: If True, use cluster std for dynamic corridor width
        sigma_factor: Multiplier for sigma-based corridor width (default 1.5)

    Returns:
        List of mixed color candidates
    """
    candidates = []
    n = len(clusters)

    for i in range(n):
        c_mixed = clusters[i]

        # Skip if area too large (likely a primary color)
        if c_mixed.area_ratio > area_ratio_max:
            continue

        best_candidate = None
        best_normalized_dist = float("inf")

        for j in range(n):
            for k in range(j + 1, n):
                if i == j or i == k:
                    continue

                c_p1 = clusters[j]
                c_p2 = clusters[k]

                # Check if mixed cluster is between p1 and p2
                is_on, t_val, dist = is_on_line_segment(
                    c_mixed.lab_cie,
                    c_p1.lab_cie,
                    c_p2.lab_cie,
                    tolerance=distance_threshold * 2,  # Use larger initial tolerance
                    t_range=t_range,
                )

                if not is_on:
                    continue

                # Calculate statistical corridor width
                if use_statistical and c_p1.lab_std is not None and c_p2.lab_std is not None:
                    # Interpolate std based on mixing ratio t
                    interp_std = (1 - t_val) * c_p1.lab_std + t_val * c_p2.lab_std
                    std_norm = float(np.linalg.norm(interp_std))
                    effective_threshold = max(5.0, sigma_factor * std_norm)
                else:
                    effective_threshold = distance_threshold

                # Check if within statistical corridor
                if dist > effective_threshold:
                    continue

                # Mixed cluster should have smaller area than both parents
                if c_mixed.area_ratio >= c_p1.area_ratio or c_mixed.area_ratio >= c_p2.area_ratio:
                    continue

                # Normalized distance for comparison
                normalized_dist = dist / effective_threshold

                # Keep best match (smallest normalized distance)
                if normalized_dist < best_normalized_dist:
                    best_normalized_dist = normalized_dist

                    conf_dist = max(0, 1.0 - normalized_dist)
                    conf_t = 1.0 - 2 * abs(t_val - 0.5)  # Prefer t near 0.5
                    conf_area = max(0, 1.0 - c_mixed.area_ratio / area_ratio_max)
                    confidence = (conf_dist + conf_t + conf_area) / 3.0

                    best_candidate = MixedColorCandidate(
                        cluster_id=c_mixed.cluster_id,
                        mixed_type="different_color_mix",
                        parent_ids=[c_p1.cluster_id, c_p2.cluster_id],
                        confidence=confidence,
                        details={
                            "distance_from_line": dist,
                            "threshold_used": effective_threshold,
                            "mixing_ratio": t_val,
                            "mixed_area": c_mixed.area_ratio,
                            "parent1_area": c_p1.area_ratio,
                            "parent2_area": c_p2.area_ratio,
                            "statistical_mode": use_statistical,
                        },
                    )

        if best_candidate is not None:
            candidates.append(best_candidate)

    return candidates


def identify_mixed_colors(
    clusters: List[ColorCluster],
    same_overlap_params: Optional[Dict[str, Any]] = None,
    diff_mix_params: Optional[Dict[str, Any]] = None,
    gradient_chain_params: Optional[Dict[str, Any]] = None,
    confidence_threshold: float = 0.45,  # Acting as candidate_th
) -> Tuple[List[ColorCluster], List[MixedColorCandidate]]:
    """
    Identify and classify mixed colors from a list of clusters.

    Implements 2-stage filtering:
    1. Hard Filter: High confidence cases are marked is_primary=False immediately.
    2. Soft Filter: Lower confidence cases are kept as is_primary=True but marked with mixed_type.
       These act as "first-to-drop" candidates if expected_k is exceeded.
    """
    same_params = same_overlap_params or {}
    diff_params = diff_mix_params or {}
    gradient_params = gradient_chain_params or {}

    # Detect all types of mixing
    same_overlap_candidates = detect_same_color_overlap(clusters, **same_params)
    diff_mix_candidates = detect_different_color_mixing(clusters, **diff_params)
    gradient_chain_candidates = detect_gradient_chain(clusters, **gradient_params)

    all_candidates = same_overlap_candidates + diff_mix_candidates + gradient_chain_candidates

    # Group candidates by cluster_id, keep highest confidence
    best_candidates: Dict[int, MixedColorCandidate] = {}
    for cand in all_candidates:
        cid = cand.cluster_id
        if cid not in best_candidates or cand.confidence > best_candidates[cid].confidence:
            best_candidates[cid] = cand

    # Thresholds for 2-stage strategy
    HARD_TH = 0.75
    GRADIENT_HARD_TH = 0.50
    CANDIDATE_TH = float(confidence_threshold)

    # Update cluster classifications
    for cluster in clusters:
        cid = cluster.cluster_id
        if cid in best_candidates:
            cand = best_candidates[cid]

            # Record candidate info
            cluster.mixed_type = cand.mixed_type
            cluster.mixed_from = cand.parent_ids
            cluster.confidence = cand.confidence

            # Decision: Hard vs Soft
            is_hard = False
            if cand.mixed_type == "gradient_chain":
                if cand.confidence >= GRADIENT_HARD_TH:
                    is_hard = True
            else:
                if cand.confidence >= HARD_TH:
                    is_hard = True

            if is_hard:
                cluster.is_primary = False
            elif cand.confidence >= CANDIDATE_TH:
                # Soft Filter: Keep as primary for now, mark as candidate
                cluster.is_primary = True
            else:
                # Below candidate threshold: ignore entirely
                cluster.mixed_type = ""
                cluster.is_primary = True

    return clusters, list(best_candidates.values())


def _build_family_groups(clusters: List[ColorCluster]) -> Dict[str, Any]:
    """
    Build family-grouped view of all clusters.

    Groups all clusters (primary + mixed) by their color family,
    sorted by area within each family.

    Args:
        clusters: List of all color clusters

    Returns:
        Dict with family names as keys, each containing:
        - clusters: List of cluster info sorted by area (descending)
        - total_area: Sum of all cluster areas in this family
        - primary_count: Number of primary clusters
        - mixed_count: Number of mixed clusters
    """
    families: Dict[str, Dict[str, Any]] = {}

    for c in clusters:
        family = classify_color_family(c.lab_cie)

        if family not in families:
            families[family] = {
                "clusters": [],
                "total_area": 0.0,
                "primary_count": 0,
                "mixed_count": 0,
            }

        cluster_info = {
            "cluster_id": c.cluster_id,
            "lab_cie": c.lab_cie.tolist(),
            "hex_color": c.hex_color,
            "area_ratio": c.area_ratio,
            "is_primary": c.is_primary,
            "mixed_type": c.mixed_type or "",
            "mixed_from": c.mixed_from,
            "confidence": c.confidence,
        }

        families[family]["clusters"].append(cluster_info)
        families[family]["total_area"] += c.area_ratio

        if c.is_primary:
            families[family]["primary_count"] += 1
        else:
            families[family]["mixed_count"] += 1

    # Sort clusters within each family by area (descending)
    for family_data in families.values():
        family_data["clusters"].sort(key=lambda x: -x["area_ratio"])

    # Sort families by total area (descending)
    sorted_families = dict(sorted(families.items(), key=lambda x: -x[1]["total_area"]))

    return sorted_families


def extract_primary_colors(
    test_bgr: np.ndarray,
    cfg: Dict[str, Any],
    max_k: int = 8,
    expected_primary_count: Optional[int] = None,
    geom: Optional[LensGeometry] = None,
    same_overlap_params: Optional[Dict[str, Any]] = None,
    diff_mix_params: Optional[Dict[str, Any]] = None,
    gradient_chain_params: Optional[Dict[str, Any]] = None,
    confidence_threshold: float = 0.4,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Extract primary (pure) ink colors by filtering out mixed colors.

    Strategy:
    1. Over-segment with high k value
    2. Identify mixed color clusters (same-overlap, diff-mix, gradient-chain)
    3. Return only primary color masks

    Args:
        test_bgr: Input BGR image
        cfg: Configuration dict
        max_k: Maximum k for over-segmentation
        expected_primary_count: Expected number of primary colors (optional)
        geom: Pre-computed lens geometry
        same_overlap_params: Parameters for same-color overlap detection
        diff_mix_params: Parameters for different-color mixing detection
        gradient_chain_params: Parameters for gradient chain detection
        confidence_threshold: Min confidence to classify as mixed

    Returns:
        Tuple of:
        - primary_masks: Dict mapping color_id to mask for primary colors only
        - metadata: Full analysis metadata
    """
    v2_cfg = cfg.get("v2_ink", {})

    # 1. Detect lens geometry
    if geom is None:
        geom = detect_lens_circle(test_bgr)

    polar_R, polar_T = get_polar_dims(cfg)

    # 2. Convert to polar and Lab (CIE scale)
    polar = to_polar(test_bgr, geom, R=polar_R, T=polar_T)
    lab_map = to_cie_lab(polar)

    # 3. Build ROI and sampling mask
    r_start = float(v2_cfg.get("roi_r_start", cfg.get("anomaly", {}).get("r_start", 0.3)))
    r_end = float(v2_cfg.get("roi_r_end", cfg.get("anomaly", {}).get("r_end", 0.9)))
    center_excluded_frac = float(cfg.get("anomaly", {}).get("center_frac", 0.2))

    roi_mask, roi_meta = build_roi_mask(polar_T, polar_R, r_start, r_end, center_excluded_frac)
    sampling_mask, sample_meta, warnings = build_sampling_mask(lab_map, roi_mask, cfg)
    samples = lab_map[sampling_mask]

    if samples.size == 0 or samples.shape[0] < max_k:
        return {}, {"error": "Insufficient samples", "warnings": warnings}

    # 4. Run clustering with configurable method (GMM or K-Means)
    base_l_weight = float(v2_cfg.get("l_weight", 0.3))
    use_adaptive = bool(v2_cfg.get("adaptive_l_weight", False))
    clustering_method = str(v2_cfg.get("clustering_method", "kmeans"))
    rng_seed = v2_cfg.get("rng_seed", 123)

    # Adaptive l_weight: 이미지 특성에 따라 자동 조정
    adaptive_meta = None
    if use_adaptive:
        l_weight, adaptive_meta = compute_adaptive_l_weight(
            samples,
            base_weight=base_l_weight,
            low_chroma_threshold=float(v2_cfg.get("adaptive_l_weight_low_chroma", 8.0)),
            high_chroma_threshold=float(v2_cfg.get("adaptive_l_weight_high_chroma", 20.0)),
        )
    else:
        l_weight = base_l_weight

    # Auto-K: Silhouette Score로 최적 K 탐색 (선택적)
    use_auto_k = bool(v2_cfg.get("auto_k_enabled", False))
    auto_k_expand_max = int(v2_cfg.get("auto_k_expand_max", 4))
    # expected_primary_count 기준으로 K 범위 설정 (max_k가 아닌 expected 기준)
    base_k = expected_primary_count if expected_primary_count else max_k
    auto_k_min = base_k
    auto_k_max = base_k + auto_k_expand_max
    auto_k_meta = None

    if use_auto_k:
        try:
            optimal_k, silhouette_score = find_optimal_k(
                samples,
                k_range=(auto_k_min, auto_k_max),
                l_weight=l_weight,
                rng_seed=rng_seed,
                method="silhouette",
            )
            auto_k_meta = {
                "optimal_k": optimal_k,
                "silhouette_score": round(silhouette_score, 4),
                "k_range": [auto_k_min, auto_k_max],
                "original_max_k": max_k,
            }
            # Auto-K 결과 사용:
            # - optimal_k가 expected보다 크면 over-segment (더 많은 클러스터로 분리)
            # - 단, max_k를 초과하지 않음
            effective_k = min(max(optimal_k, expected_primary_count or 3), max_k)
        except Exception as e:
            auto_k_meta = {"error": str(e)}
            effective_k = max_k
    else:
        effective_k = max_k

    labels, centers, clustering_confidence = segment_colors(
        samples,
        effective_k,
        method=clustering_method,
        l_weight=l_weight,
        rng_seed=rng_seed,
        attempts=10,  # for kmeans
        covariance_type=v2_cfg.get("gmm_covariance_type", "full"),  # for gmm
    )

    if labels.size == 0:
        return {}, {"error": "Clustering failed", "warnings": warnings, "method": clustering_method}

    # 5. Convert centers to Lab
    centers_lab = np.zeros_like(centers)
    centers_lab[:, 1] = centers[:, 0]  # a
    centers_lab[:, 2] = centers[:, 1]  # b
    centers_lab[:, 0] = centers[:, 2] / l_weight  # L

    centers_cie = centers_lab.astype(np.float32)  # already CIE scale

    # 6. Assign all pixels to clusters
    from .color_masks import assign_cluster_labels_to_image

    label_map = assign_cluster_labels_to_image(lab_map, centers, l_weight=l_weight)

    # 7. Build cluster objects
    clusters: List[ColorCluster] = []
    total_pixels = polar_T * polar_R

    for i in range(effective_k):
        mask = label_map == i
        pixel_count = int(mask.sum())

        if pixel_count == 0:
            continue

        # Calculate cluster standard deviation for statistical thresholding
        lab_std = calculate_cluster_std(lab_map, mask)

        clusters.append(
            ColorCluster(
                cluster_id=i,
                lab_cv8=lab_cie_to_cv8(centers_cie[i]),
                lab_cie=centers_cie[i],
                hex_color=lab_to_hex(centers_cie[i]),
                area_ratio=pixel_count / total_pixels,
                pixel_count=pixel_count,
                mask=mask,
                lab_std=lab_std,
            )
        )

    # Sort by L* (dark to light)
    clusters.sort(key=lambda c: c.lab_cie[0])

    # 7.5. Merge very similar clusters (handles k-means noise)
    original_count = len(clusters)
    clusters = merge_similar_clusters(clusters, lab_threshold=5.0)
    merged_count = original_count - len(clusters)

    # 8. Identify mixed colors (Hard/Soft Filtering)
    clusters, mixed_candidates = identify_mixed_colors(
        clusters,
        same_overlap_params=same_overlap_params,
        diff_mix_params=diff_mix_params,
        gradient_chain_params=gradient_chain_params,
        confidence_threshold=confidence_threshold,
    )

    # 8.5. Cleanup over-segmented primary candidates if expected_k is exceeded
    # We drop 'Soft Filtered' candidates (mixed_type exists but is_primary is still True)
    if expected_primary_count is not None:
        primary_candidates = [c for c in clusters if c.is_primary]

        if len(primary_candidates) > expected_primary_count:
            # 1. Identify soft candidates (those with a mixed_type recorded)
            soft_candidates = [c for c in primary_candidates if c.mixed_type != ""]

            if soft_candidates:
                # Sort: Confidence High (likely mixed) -> Area Ratio Low (insignificant)
                soft_candidates.sort(key=lambda c: (-c.confidence, c.area_ratio))

                # Chromatic family protection
                CHROMATIC_FAMILIES = {"Blue-Cyan", "Red-Magenta", "Green", "Yellow-Orange"}

                for cand in soft_candidates:
                    if len([c for c in clusters if c.is_primary]) <= expected_primary_count:
                        break

                    family = classify_color_family(cand.lab_cie)
                    if family in CHROMATIC_FAMILIES:
                        # Check if it's the last representative of its chromatic family
                        others_in_family = [
                            c
                            for c in clusters
                            if c.is_primary
                            and c.cluster_id != cand.cluster_id
                            and classify_color_family(c.lab_cie) == family
                        ]
                        if not others_in_family:
                            continue  # Protect the last chromatic rep

                    # Drop candidate
                    cand.is_primary = False
                    cand.mixed_type = f"dropped_soft_{cand.mixed_type}"

    # 8.6. Family-based recovery: ensure each color family has representation
    recovery_info = None
    if expected_primary_count is not None:
        clusters, recovery_info = recover_filtered_by_family(clusters, expected_primary_count)

    # 9. Separate primary and mixed clusters
    primary_clusters = [c for c in clusters if c.is_primary]
    mixed_clusters = [c for c in clusters if not c.is_primary]

    # 9.5. Consolidate primary clusters if we have more than expected
    consolidated_count = 0
    if expected_primary_count is not None and len(primary_clusters) > expected_primary_count:
        original_primary_count = len(primary_clusters)
        primary_clusters = consolidate_to_target_count(primary_clusters, expected_primary_count)
        consolidated_count = original_primary_count - len(primary_clusters)

    # 10. Build output masks (primary colors only)
    primary_masks = {}
    for idx, cluster in enumerate(primary_clusters):
        primary_masks[f"color_{idx}"] = cluster.mask

    # 11. Build metadata
    metadata = {
        "total_clusters": len(clusters),
        "merged_clusters": merged_count,
        "consolidated_clusters": consolidated_count,
        "primary_count": len(primary_clusters),
        "mixed_count": len(mixed_clusters),
        "max_k_used": max_k,
        "effective_k_used": effective_k,
        "expected_primary_count": expected_primary_count,
        "family_recovery": recovery_info,
        "clustering_method": clustering_method,
        "clustering_confidence": float(clustering_confidence),
        "l_weight_used": float(l_weight),
        "adaptive_l_weight": adaptive_meta,
        "auto_k": auto_k_meta,
        "primary_colors": [
            {
                "color_id": f"color_{idx}",
                "original_cluster_id": c.cluster_id,
                "lab_cie": c.lab_cie.tolist(),
                "lab_std": c.lab_std.tolist() if c.lab_std is not None else None,
                "hex_color": c.hex_color,
                "area_ratio": c.area_ratio,
                "color_family": classify_color_family(c.lab_cie),
            }
            for idx, c in enumerate(primary_clusters)
        ],
        "mixed_colors": [
            {
                "original_cluster_id": c.cluster_id,
                "lab_cie": c.lab_cie.tolist(),
                "hex_color": c.hex_color,
                "area_ratio": c.area_ratio,
                "mixed_type": c.mixed_type,
                "mixed_from": c.mixed_from,
                "confidence": c.confidence,
                "color_family": classify_color_family(c.lab_cie),
            }
            for c in mixed_clusters
        ],
        "all_clusters": [
            {
                "cluster_id": c.cluster_id,
                "lab_cie": c.lab_cie.tolist(),
                "hex_color": c.hex_color,
                "area_ratio": c.area_ratio,
                "is_primary": c.is_primary,
                "mixed_type": c.mixed_type,
                "mixed_from": c.mixed_from,
                "confidence": c.confidence,
                "color_family": classify_color_family(c.lab_cie),
            }
            for c in clusters
        ],
        # Family-grouped view: all clusters grouped by color family
        "family_groups": _build_family_groups(clusters),
        "mixed_candidates": [
            {
                "cluster_id": cand.cluster_id,
                "mixed_type": cand.mixed_type,
                "parent_ids": cand.parent_ids,
                "confidence": cand.confidence,
                "details": cand.details,
            }
            for cand in mixed_candidates
        ],
        "config": {
            "max_k": max_k,
            "confidence_threshold": confidence_threshold,
            "same_overlap_params": same_overlap_params or {},
            "diff_mix_params": diff_mix_params or {},
            "gradient_chain_params": gradient_chain_params or {},
        },
        "warnings": warnings,
        "sample_meta": sample_meta,
        "geom": {"cx": float(geom.cx), "cy": float(geom.cy), "r": float(geom.r)},
    }

    return primary_masks, metadata


def visualize_primary_extraction(
    polar_bgr: np.ndarray,
    clusters: List[ColorCluster],
    show_mixed: bool = True,
) -> np.ndarray:
    """
    Visualize primary vs mixed color extraction.

    Args:
        polar_bgr: Polar BGR image
        clusters: List of classified clusters
        show_mixed: Whether to show mixed colors (dimmed)

    Returns:
        Visualization image
    """
    import cv2

    vis = polar_bgr.copy()

    # Primary colors: bright overlay
    primary_colors = [
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (0, 0, 255),  # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]

    # Mixed colors: gray overlay
    mixed_color = (128, 128, 128)

    primary_idx = 0
    for cluster in clusters:
        if cluster.mask is None:
            continue

        if cluster.is_primary:
            color = primary_colors[primary_idx % len(primary_colors)]
            primary_idx += 1
            alpha = 0.4
        elif show_mixed:
            color = mixed_color
            alpha = 0.2
        else:
            continue

        overlay = np.zeros_like(vis)
        overlay[cluster.mask] = color
        vis = cv2.addWeighted(vis, 1.0, overlay, alpha, 0)

    return vis


def extract_sky_blue(img_lab: np.ndarray, roi_mask: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Extract Sky Blue ink by smart clustering and identifying the lowest b* value.
    Updated: Uses k=8 to better handle fine mesh structures.

    Args:
        img_lab: Lab image (OpenCV 8-bit scale)
        roi_mask: Boolean mask for ROI

    Returns:
        Tuple of (blue_cluster_index, center_lab_color)
    """
    # 1. 8개 그룹으로 분할 (그물망 구조 대응을 위해 더 세밀하게 분할)
    pixels = img_lab[roi_mask]
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(pixels)
    centers = kmeans.cluster_centers_

    # 2. 배경(Clear) 제외: (L - Chroma) 점수가 1등인 그룹
    l_values = centers[:, 0] * 100.0 / 255.0
    a_values = centers[:, 1] - 128.0
    b_values = centers[:, 2] - 128.0
    chromas = np.sqrt(a_values**2 + b_values**2)

    bg_idx = np.argmax(l_values - chromas)

    # 3. 하늘색(Sky Blue) 확정
    ink_candidates = [i for i in range(n_clusters) if i != bg_idx]

    if not ink_candidates:
        blue_idx = np.argmin(b_values)
    else:
        # 3-1. 밝기 조건 필터링: L* > 30 (너무 어두운 네이비/검정 제외)
        bright_candidates = [i for i in ink_candidates if l_values[i] > 30.0]

        target_candidates = bright_candidates if bright_candidates else ink_candidates

        # 3-2. b*가 가장 낮은(Blue) 클러스터 선택
        min_b_idx_in_target = np.argmin(b_values[target_candidates])
        blue_idx = target_candidates[min_b_idx_in_target]

    return int(blue_idx), centers[blue_idx]
