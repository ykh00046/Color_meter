from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# GMM import with fallback
try:
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute_adaptive_l_weight(
    lab_samples: np.ndarray,
    base_weight: float = 0.3,
    low_chroma_threshold: float = 8.0,
    high_chroma_threshold: float = 20.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    이미지 특성에 따라 l_weight를 자동 결정 (선형 보간 방식)

    원리:
    - 채도(Chroma) 분산이 낮으면 (그레이 렌즈) → l_weight 높임 (밝기로 구분)
    - 채도 분산이 높으면 (컬러 렌즈) → l_weight 낮춤 (색상으로 구분)
    - 중간 영역은 선형 보간으로 부드럽게 전환

    Args:
        lab_samples: (N, 3) Lab 색상 샘플 (CV8 또는 CIE 스케일)
        base_weight: 기본 l_weight 값 (중간 영역 기준)
        low_chroma_threshold: 이 이하면 저채도 (그레이) → weight 1.0
        high_chroma_threshold: 이 이상이면 고채도 (컬러) → weight 0.2

    Returns:
        l_weight: 계산된 가중치 (0.2 ~ 1.0)
        meta: 분석 메타데이터
    """
    if lab_samples.size == 0:
        return base_weight, {"error": "empty_samples"}

    # CV8 스케일 가정: a,b는 128 중심
    # CIE 스케일이면 0 중심
    a = lab_samples[:, 1]
    b = lab_samples[:, 2]

    # 스케일 자동 감지 (CV8: 0-255, CIE: -128~127)
    if a.mean() > 50:  # CV8 스케일
        a_centered = a - 128.0
        b_centered = b - 128.0
    else:  # CIE 스케일
        a_centered = a
        b_centered = b

    # Chroma 계산
    chroma = np.sqrt(a_centered**2 + b_centered**2)
    chroma_mean = float(chroma.mean())
    chroma_std = float(chroma.std())

    # L* 분산도 확인
    L = lab_samples[:, 0]
    L_std = float(L.std())

    # 선형 보간으로 Adaptive l_weight 결정
    # chroma_std: low_threshold(8) → high_threshold(20)
    # l_weight:   1.0 (밝기 중심) → 0.2 (색상 중심)
    high_w, low_w = 1.0, 0.2

    if chroma_std <= low_chroma_threshold:
        l_weight = high_w
        reason = "low_chroma_variance"
    elif chroma_std >= high_chroma_threshold:
        l_weight = low_w
        reason = "high_chroma_variance"
    else:
        # 선형 보간: chroma_std가 증가하면 l_weight 감소
        ratio = (chroma_std - low_chroma_threshold) / (high_chroma_threshold - low_chroma_threshold)
        l_weight = high_w - ratio * (high_w - low_w)
        reason = "interpolated"

        # L_std가 매우 높으면 (밝기 차이가 큼) 약간 보정
        if L_std > 50:
            l_weight = min(1.0, l_weight + 0.1)
            reason = "interpolated_L_boost"

    meta = {
        "chroma_mean": round(chroma_mean, 2),
        "chroma_std": round(chroma_std, 2),
        "L_std": round(L_std, 2),
        "base_weight": base_weight,
        "computed_weight": round(l_weight, 3),
        "reason": reason,
    }

    return l_weight, meta


def _build_features(lab_samples: np.ndarray, l_weight: float) -> np.ndarray:
    if lab_samples.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    a = lab_samples[:, 1:2]
    b = lab_samples[:, 2:3]
    L = lab_samples[:, 0:1] * float(l_weight)
    return np.hstack([a, b, L]).astype(np.float32)


def kmeans_segment(
    lab_samples: np.ndarray,
    k: int,
    l_weight: float = 0.3,
    attempts: int = 5,
    rng_seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if lab_samples.shape[0] < k or k <= 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    feats = _build_features(lab_samples, l_weight)
    if feats.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    if rng_seed is not None:
        cv2.setRNGSeed(int(rng_seed))
    compactness, labels, centers = cv2.kmeans(
        feats,
        k,
        None,
        criteria,
        attempts,
        cv2.KMEANS_PP_CENTERS,
    )
    return labels.flatten().astype(np.int32), centers.astype(np.float32)


def gmm_segment(
    lab_samples: np.ndarray,
    k: int,
    l_weight: float = 0.3,
    rng_seed: int | None = None,
    covariance_type: str = "full",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    GMM(Gaussian Mixture Model) 기반 클러스터링

    K-Means와 달리 타원형(Ellipsoidal) 분포를 처리할 수 있어
    밝기만 다른 같은 잉크 그룹을 더 정확하게 분류합니다.

    Args:
        lab_samples: (N, 3) Lab 색상 샘플
        k: 클러스터 개수
        l_weight: L* 가중치 (0.3 = 색상 중심, 1.0 = 밝기 포함)
        rng_seed: 재현성을 위한 시드값
        covariance_type: 'full' (타원형), 'diag' (축정렬 타원), 'spherical' (원형)

    Returns:
        labels: (N,) 각 샘플의 클러스터 레이블
        centers: (k, 3) 클러스터 중심 (feature space: [a, b, L*weight])
        confidence: BIC 기반 신뢰도 (0~1)
    """
    if lab_samples.shape[0] < k or k <= 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32), 0.0

    feats = _build_features(lab_samples, l_weight)
    if feats.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32), 0.0

    # sklearn 없거나 GMM 실패시 K-Means fallback
    if not HAS_SKLEARN:
        labels, centers = kmeans_segment(lab_samples, k, l_weight, rng_seed=rng_seed)
        return labels, centers, 0.7  # default confidence for kmeans

    try:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=rng_seed if rng_seed is not None else 42,
            n_init=5,
            max_iter=200,
            init_params="kmeans",  # K-Means 초기화로 안정성/속도 향상
        )

        labels = gmm.fit_predict(feats)

        # 클러스터 중심 계산 (feature 스케일: [a, b, L*weight] - kmeans와 동일)
        centers = np.zeros((k, 3), dtype=np.float32)
        for i in range(k):
            mask = labels == i
            if mask.sum() > 0:
                cluster_samples = lab_samples[mask]
                # Feature space: [a, b, L*weight]
                centers[i, 0] = cluster_samples[:, 1].mean()  # a
                centers[i, 1] = cluster_samples[:, 2].mean()  # b
                centers[i, 2] = cluster_samples[:, 0].mean() * l_weight  # L*weight

        # BIC 기반 confidence (낮을수록 좋음 → 역수로 변환)
        bic = gmm.bic(feats)
        confidence = 1.0 / (1.0 + max(0, bic) / 10000)

        return labels.astype(np.int32), centers.astype(np.float32), confidence

    except Exception as e:
        # GMM 실패시 (singular matrix 등) K-Means로 fallback
        logger.warning(f"GMM clustering failed: {e}. Using K-Means fallback.")
        labels, centers = kmeans_segment(lab_samples, k, l_weight, rng_seed=rng_seed)
        return labels, centers, 0.5  # lower confidence for fallback


def find_optimal_k(
    lab_samples: np.ndarray,
    k_range: Tuple[int, int] = (2, 8),
    l_weight: float = 0.3,
    rng_seed: int | None = None,
    method: Literal["silhouette", "bic"] = "silhouette",
    max_samples: int = 5000,
) -> Tuple[int, float]:
    """
    Silhouette Score 또는 BIC로 최적 K 탐색

    Args:
        lab_samples: (N, 3) Lab 색상 샘플
        k_range: 탐색할 K 범위 (min, max)
        l_weight: L* 가중치
        rng_seed: 재현성을 위한 시드값
        method: 'silhouette' (클러스터 분리도) 또는 'bic' (모델 복잡도)
        max_samples: 성능을 위한 최대 샘플 수 (초과시 서브샘플링)

    Returns:
        best_k: 최적 클러스터 개수
        best_score: 해당 K의 점수
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn is required. Install with: pip install scikit-learn")

    # Subsample for performance if too many samples
    rng = np.random.default_rng(rng_seed if rng_seed is not None else 42)
    if lab_samples.shape[0] > max_samples:
        idx = rng.choice(lab_samples.shape[0], size=max_samples, replace=False)
        lab_samples = lab_samples[idx]

    feats = _build_features(lab_samples, l_weight)
    if feats.shape[0] < k_range[1]:
        return k_range[0], 0.0

    best_k = k_range[0]
    best_score = -np.inf if method == "silhouette" else np.inf

    for k in range(k_range[0], k_range[1] + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=rng_seed if rng_seed is not None else 42,
                n_init=3,
            )
            labels = gmm.fit_predict(feats)

            # 모든 샘플이 하나의 클러스터에 할당되면 스킵
            if len(set(labels)) < 2:
                continue

            if method == "silhouette":
                score = silhouette_score(feats, labels)
                if score > best_score:
                    best_k, best_score = k, score
            else:  # bic (낮을수록 좋음)
                score = gmm.bic(feats)
                if score < best_score:
                    best_k, best_score = k, score

        except Exception:
            continue

    return best_k, best_score


def segment_colors(
    lab_samples: np.ndarray,
    k: int,
    method: Literal["kmeans", "gmm"] = "kmeans",
    l_weight: float = 0.3,
    rng_seed: int | None = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    통합 색상 분할 함수 (디스패처)

    Args:
        lab_samples: (N, 3) Lab 색상 샘플
        k: 클러스터 개수
        method: 'kmeans' 또는 'gmm'
        l_weight: L* 가중치
        rng_seed: 재현성을 위한 시드값
        **kwargs: 메서드별 추가 파라미터

    Returns:
        labels: (N,) 클러스터 레이블
        centers: (k, 3) 클러스터 중심
        confidence: 신뢰도 (0~1)
    """
    if method == "gmm":
        if not HAS_SKLEARN:
            # sklearn 없으면 kmeans로 fallback
            method = "kmeans"
        else:
            return gmm_segment(
                lab_samples,
                k,
                l_weight,
                rng_seed,
                covariance_type=kwargs.get("covariance_type", "full"),
            )

    # K-Means (기본)
    labels, centers = kmeans_segment(
        lab_samples,
        k,
        l_weight,
        attempts=kwargs.get("attempts", 5),
        rng_seed=rng_seed,
    )

    # K-Means는 confidence 계산 없음 → 기본값 반환
    confidence = 0.7 if labels.size > 0 else 0.0
    return labels, centers, confidence
