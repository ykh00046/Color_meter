"""
Ink State Grouping

그라데이션 대응: 여러 color state를 단일 잉크 그룹으로 병합합니다.
단일 그라데이션 잉크가 k-means에서 2-3개 클러스터로 분리되는 문제를 해결합니다.

병합 규칙:
1. radial_presence_curve 인접성 (공간적 이어짐)
2. pairwise_deltaE < 15.0 (부드러운 색 전환)
3. compactness 유사도 (밀도 일관성)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def group_ink_states(
    state_features: List[Dict[str, Any]],
    pairwise_deltaE: np.ndarray,
    merge_threshold_deltaE: float = 15.0,
    merge_threshold_compactness: float = 0.15,
    merge_threshold_radial_corr: float = 0.7,
) -> List[List[int]]:
    """
    잉크 상태 그룹화 (그라데이션 병합)

    Args:
        state_features: 각 state의 특징
            [
                {
                    "state_id": 0,
                    "inkness_score": 0.85,
                    "radial_presence_curve": [0.1, 0.3, ..., 0.8],  # len=10
                    "compactness": 0.75,
                    ...
                },
                ...
            ]
        pairwise_deltaE: (n, n) state 간 색차 행렬
        merge_threshold_deltaE: 병합 허용 최대 색차 (기본 15.0)
        merge_threshold_compactness: 병합 허용 compactness 차이 (기본 0.15)
        merge_threshold_radial_corr: 병합 허용 최소 radial curve 상관계수 (기본 0.7)

    Returns:
        groups: 병합된 그룹 리스트 (각 그룹은 state_id 리스트)
            예: [[0, 2], [1], [3, 4]] - state 0,2가 하나의 잉크, 1 단독, 3,4가 하나의 잉크

    병합 조건 (AND):
    1. pairwise_deltaE[i][j] < merge_threshold_deltaE
    2. radial_curve correlation > merge_threshold_radial_corr
    3. |compactness[i] - compactness[j]| < merge_threshold_compactness
    """
    if not state_features:
        return []

    n = len(state_features)
    if n == 1:
        return [[0]]

    # 1. 병합 가능성 그래프 구축 (adjacency matrix)
    merge_graph = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(i + 1, n):
            if _should_merge(
                state_features[i],
                state_features[j],
                pairwise_deltaE[i, j],
                merge_threshold_deltaE,
                merge_threshold_compactness,
                merge_threshold_radial_corr,
            ):
                merge_graph[i, j] = True
                merge_graph[j, i] = True

    # 2. Connected components 찾기 (Union-Find)
    groups = _find_connected_components(merge_graph)

    return groups


def _should_merge(
    state_i: Dict[str, Any],
    state_j: Dict[str, Any],
    deltaE: float,
    threshold_deltaE: float,
    threshold_compactness: float,
    threshold_radial_corr: float,
) -> bool:
    """
    두 state를 병합해야 하는지 판단

    Returns:
        True if all merge conditions met
    """
    # 조건 1: 색차가 작아야 (부드러운 전환)
    if deltaE >= threshold_deltaE:
        return False

    # 조건 2: compactness 유사해야 (밀도 일관성)
    comp_i = state_i.get("compactness", 0.0)
    comp_j = state_j.get("compactness", 0.0)
    if abs(comp_i - comp_j) >= threshold_compactness:
        return False

    # 조건 3: radial_presence_curve 상관계수 높아야 (공간적 인접)
    curve_i = state_i.get("radial_presence_curve", [])
    curve_j = state_j.get("radial_presence_curve", [])

    if not curve_i or not curve_j:
        # radial curve 없으면 병합 불가
        return False

    corr = _calculate_correlation(curve_i, curve_j)
    if corr < threshold_radial_corr:
        return False

    return True


def _calculate_correlation(curve_a: List[float], curve_b: List[float]) -> float:
    """
    두 radial_presence_curve 간 Pearson 상관계수 계산

    Args:
        curve_a, curve_b: len=10 radial bin distributions

    Returns:
        correlation: -1~1 (1 = 완전 일치)
    """
    if len(curve_a) != len(curve_b):
        return 0.0

    arr_a = np.array(curve_a, dtype=np.float64)
    arr_b = np.array(curve_b, dtype=np.float64)

    std_a = np.std(arr_a)
    std_b = np.std(arr_b)

    # 특수 케이스: 둘 다 상수 곡선
    if std_a == 0 and std_b == 0:
        # 둘 다 상수이고 값이 같으면 완전 일치
        if np.allclose(arr_a, arr_b):
            return 1.0
        else:
            return 0.0

    # 하나만 상수이면 상관 없음
    if std_a == 0 or std_b == 0:
        return 0.0

    # Pearson correlation
    corr_matrix = np.corrcoef(arr_a, arr_b)
    return float(corr_matrix[0, 1])


def _find_connected_components(adjacency: np.ndarray) -> List[List[int]]:
    """
    Adjacency matrix에서 connected components 찾기 (Union-Find)

    Args:
        adjacency: (n, n) bool array

    Returns:
        groups: [[0, 2], [1], [3, 4], ...]
    """
    n = adjacency.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x: int, y: int):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    # Union all connected pairs
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                union(i, j)

    # Group by root
    groups_dict: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in groups_dict:
            groups_dict[root] = []
        groups_dict[root].append(i)

    # Convert to list of lists
    groups = list(groups_dict.values())

    # Sort each group by state_id
    for group in groups:
        group.sort()

    # Sort groups by first element
    groups.sort(key=lambda g: g[0])

    return groups


def validate_groups(groups: List[List[int]], state_features: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    그룹 검증

    Args:
        groups: 병합 결과
        state_features: 원본 state 특징

    Returns:
        (valid, message)

    검증 항목:
    1. 모든 state가 정확히 한 그룹에 속함
    2. 그룹 ID 범위 유효
    3. 중복 없음
    """
    n = len(state_features)

    # 1. Flatten all groups
    all_ids = []
    for group in groups:
        all_ids.extend(group)

    # 2. Check range first (before count check)
    if any(sid < 0 or sid >= n for sid in all_ids):
        return False, f"State ID out of range [0, {n-1}]"

    # 3. Check uniqueness (before count check)
    if len(set(all_ids)) != len(all_ids):
        return False, "Duplicate state IDs in groups"

    # 4. Check count
    if len(all_ids) != n:
        return False, f"Expected {n} states, got {len(all_ids)} in groups"

    return True, f"OK ({len(groups)} groups from {n} states)"


def get_group_statistics(groups: List[List[int]], state_features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    그룹 통계

    Args:
        groups: 병합 결과
        state_features: 원본 state 특징

    Returns:
        {
            "num_groups": int,
            "num_states": int,
            "merge_ratio": float (0~1, 병합된 비율),
            "largest_group_size": int,
            "group_sizes": [2, 1, 3, ...]
        }
    """
    group_sizes = [len(g) for g in groups]

    num_merged = sum(1 for g in groups if len(g) > 1)
    merge_ratio = float(num_merged) / len(groups) if groups else 0.0

    return {
        "num_groups": len(groups),
        "num_states": len(state_features),
        "merge_ratio": round(merge_ratio, 3),
        "largest_group_size": max(group_sizes) if group_sizes else 0,
        "group_sizes": group_sizes,
    }
