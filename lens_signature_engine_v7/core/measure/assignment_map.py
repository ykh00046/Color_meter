"""
ROI Label Map Assignment

ROI 전체 픽셀에 대해 클러스터 ID를 할당합니다.
샘플링된 픽셀의 labels를 ROI 전체로 확장하여,
angular_continuity 같은 공간 기반 메트릭 계산을 가능하게 합니다.

NOTE:
 - 프로젝트 내 polar 표준은 (T, R) = (theta, radial) 입니다.
 - 과거 코드/데이터는 (R, T)로 들어올 수 있어 polar_order='AUTO'로 자동 추론/전환을 지원합니다.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _infer_polar_order(shape: Tuple[int, int]) -> str:
    """
    Heuristic inference for polar order.
    Returns:
        'TR' meaning (T, R) or 'RT' meaning (R, T)
    """
    a, b = int(shape[0]), int(shape[1])
    # Common theta sizes
    theta_like = {180, 360, 720}
    if a in theta_like and b not in theta_like:
        return "TR"
    if b in theta_like and a not in theta_like:
        return "RT"
    # Fallback: theta often >= radial
    return "TR" if a >= b else "RT"


def _to_tr(roi_mask: np.ndarray, lab_map: np.ndarray, order: str) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Normalize inputs to (T, R).
    Returns (roi_mask_tr, lab_map_tr, transposed_flag)
    """
    order = str(order).upper()
    if order == "RT":
        return roi_mask.T, lab_map.transpose(1, 0, 2), True
    return roi_mask, lab_map, False


def build_assignment_map(
    roi_mask: np.ndarray,
    lab_map: np.ndarray,
    centers: np.ndarray,
    *,
    l_weight: float = 0.3,
    polar_order: str = "AUTO",
) -> np.ndarray:
    """
    ROI 전체 픽셀에 state_id (클러스터 ID) 할당

    Args:
        roi_mask: polar ROI 마스크
        lab_map: polar Lab 맵 (.., .., 3) with channels [L, a, b]
        centers: (k, 3) k-means centers **in feature space** [a, b, L*l_weight]
        l_weight: L channel weight used during k-means (must match training/seg)
        polar_order: 'TR'=(T,R), 'RT'=(R,T), 'AUTO'=heuristic inference

    Returns:
        state_id_map: input과 동일한 polar_order 축을 유지한 상태의 할당 결과
            -1 = ROI 외부
            0~k-1 = 클러스터 ID

    Example:
        >>> roi_mask = np.ones((100, 360), dtype=bool)
        >>> lab_map = np.random.rand(100, 360, 3).astype(np.float32)
        >>> centers = np.array([[50, 120, 130], [80, 140, 140]], dtype=np.float32)
        >>> state_map = build_assignment_map(roi_mask, lab_map, centers)
        >>> assert state_map.shape == (100, 360)
        >>> assert np.all((state_map >= -1) & (state_map < 2))
    """
    if lab_map.ndim != 3 or lab_map.shape[2] != 3:
        raise ValueError(f"lab_map must be (*, *, 3), got {lab_map.shape}")

    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"centers must be (k, 3), got {centers.shape}")

    if roi_mask.shape != lab_map.shape[:2]:
        # allow a common mismatch case where lab_map is transposed
        if roi_mask.T.shape == lab_map.shape[:2]:
            roi_mask = roi_mask.T
        else:
            raise ValueError(f"roi_mask shape {roi_mask.shape} must match lab_map[:2] {lab_map.shape[:2]}")

    order = str(polar_order).upper()
    if order == "AUTO":
        order = _infer_polar_order(roi_mask.shape)

    roi_mask_tr, lab_tr, was_t = _to_tr(roi_mask, lab_map, order)
    T, R = roi_mask_tr.shape
    k = centers.shape[0]

    # 초기화: 전체를 -1로 (ROI 외부)
    state_id_map_tr = np.full((T, R), -1, dtype=np.int32)

    if k == 0:
        return state_id_map_tr.T if was_t else state_id_map_tr

    # ROI 내부 픽셀 좌표
    roi_coords = np.where(roi_mask_tr)
    if len(roi_coords[0]) == 0:
        return state_id_map_tr.T if was_t else state_id_map_tr

    # ROI 내부 픽셀의 feature 추출: [a, b, L*l_weight]
    a_map = lab_tr[:, :, 1].astype(np.float32, copy=False)
    b_map = lab_tr[:, :, 2].astype(np.float32, copy=False)
    L_map = (lab_tr[:, :, 0] * float(l_weight)).astype(np.float32, copy=False)
    feat_map = np.stack([a_map, b_map, L_map], axis=-1)  # (T, R, 3)
    roi_pixels = feat_map[roi_coords]  # (N_roi, 3)

    # 각 픽셀과 모든 중심점 간의 유클리드 거리 계산
    # use squared distance (faster, same argmin)
    distances = np.sum((roi_pixels[:, None, :] - centers[None, :, :]) ** 2, axis=2)

    # 가장 가까운 중심점의 ID 할당
    assigned_ids = np.argmin(distances, axis=1)  # (N_roi,)

    # state_id_map에 기록
    state_id_map_tr[roi_coords] = assigned_ids

    # restore to original order
    return state_id_map_tr.T if was_t else state_id_map_tr


def validate_assignment_map(
    state_id_map: np.ndarray,
    roi_mask: np.ndarray,
    expected_k: int,
    *,
    polar_order: str = "AUTO",
) -> Tuple[bool, str]:
    """
    Assignment map 검증

    Args:
        state_id_map: 할당 결과
        roi_mask: ROI 마스크
        expected_k: 예상 클러스터 수
        polar_order: 'TR'=(T,R), 'RT'=(R,T), 'AUTO'=heuristic inference

    Returns:
        (valid, message)

    검증 항목:
    1. ROI 외부는 모두 -1
    2. ROI 내부는 0 ~ k-1 범위
    3. 모든 클러스터 ID가 최소 1개 이상의 픽셀 보유
    """
    if roi_mask.shape != state_id_map.shape:
        if roi_mask.T.shape == state_id_map.shape:
            roi_mask = roi_mask.T
        else:
            return False, f"shape mismatch: state_id_map={state_id_map.shape}, roi_mask={roi_mask.shape}"

    order = str(polar_order).upper()
    if order == "AUTO":
        order = _infer_polar_order(state_id_map.shape)

    # normalize to TR for checks
    roi_tr, _, was_t = _to_tr(roi_mask, np.zeros((*roi_mask.shape, 3), dtype=np.float32), order)
    state_tr = state_id_map.T if was_t else state_id_map

    # 1. ROI 외부 검증
    outside_roi = state_tr[~roi_tr]
    if not np.all(outside_roi == -1):
        return False, "ROI 외부에 -1 아닌 값 존재"

    # 2. ROI 내부 범위 검증
    inside_roi = state_tr[roi_tr]
    if len(inside_roi) == 0:
        return False, "ROI 내부 픽셀 없음"

    if np.any(inside_roi < 0) or np.any(inside_roi >= expected_k):
        return False, f"ROI 내부에 0~{expected_k-1} 범위 벗어난 값 존재"

    # 3. 모든 클러스터 존재 여부
    unique_ids = np.unique(inside_roi)
    if len(unique_ids) != expected_k:
        return False, f"클러스터 {expected_k}개 중 {len(unique_ids)}개만 할당됨"

    # 4. 분포 확인 (경고용)
    counts = np.bincount(inside_roi, minlength=expected_k)
    min_count = np.min(counts)
    max_count = np.max(counts)
    ratio = min_count / max_count if max_count > 0 else 0.0

    if ratio < 0.01:  # 1% 미만이면 경고
        msg = f"클러스터 불균형 (min={min_count}, max={max_count})"
        return True, msg  # 경고지만 valid

    return True, f"OK (clusters={expected_k}, pixels={len(inside_roi)})"


def get_cluster_spatial_stats(
    state_id_map: np.ndarray,
    cluster_id: int,
    *,
    polar_order: str = "AUTO",
    circular_theta: bool = True,
) -> dict:
    """
    클러스터의 공간 분포 통계

    Args:
        state_id_map: 할당 결과
        cluster_id: 분석할 클러스터 ID
        polar_order: 'TR'=(T,R), 'RT'=(R,T), 'AUTO'=heuristic inference
        circular_theta: True면 theta 0과 끝을 연결하여 run-length 계산

    Returns:
        {
            "pixel_count": int,
            "r_min": float,  # 0~1 normalized
            "r_max": float,
            "r_mean": float,
            "theta_coverage": float,  # 0~1 (각도 방향 커버리지)
            "fragmentation": float  # 0~1 (0=연속적, 1=흩어짐)
        }
    """
    order = str(polar_order).upper()
    if order == "AUTO":
        order = _infer_polar_order(state_id_map.shape)

    if order == "RT":
        state_tr = state_id_map.T
    else:
        state_tr = state_id_map

    T, R = state_tr.shape
    cluster_mask = state_tr == cluster_id

    coords = np.where(cluster_mask)
    if len(coords[0]) == 0:
        return {
            "pixel_count": 0,
            "r_min": 0.0,
            "r_max": 0.0,
            "r_mean": 0.0,
            "theta_coverage": 0.0,
            "fragmentation": 1.0,
        }

    theta_coords = coords[0]  # theta (T)
    r_coords = coords[1]  # radial (R)

    # Radial 통계
    r_norm = r_coords / float(R) if R > 0 else 0.0
    r_min = float(np.min(r_norm))
    r_max = float(np.max(r_norm))
    r_mean = float(np.mean(r_norm))

    # Theta 커버리지
    unique_thetas = len(np.unique(theta_coords))
    theta_coverage = float(unique_thetas / T) if T > 0 else 0.0

    # Fragmentation (연속성 역수)
    # Theta 방향으로 빈 공간이 많으면 fragmentation 높음
    theta_bins = np.zeros(T, dtype=bool)
    theta_bins[theta_coords] = True

    # Run-length 계산
    runs = []
    current_run = 0
    for val in theta_bins:
        if val:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)

    # circular theta: first/last run 연결
    if circular_theta and runs and theta_bins[0] and theta_bins[-1] and len(runs) >= 2:
        runs[0] = runs[0] + runs[-1]
        runs = runs[:-1]

    if runs:
        avg_run = np.mean(runs)
        num_runs = len(runs)
        # num_runs가 많을수록 fragmentation 높음
        fragmentation = 1.0 - (avg_run / T)
    else:
        fragmentation = 1.0

    return {
        "pixel_count": int(len(r_coords)),
        "r_min": round(float(r_min), 3),
        "r_max": round(float(r_max), 3),
        "r_mean": round(float(r_mean), 3),
        "theta_coverage": round(float(theta_coverage), 3),
        "fragmentation": round(float(fragmentation), 3),
    }
