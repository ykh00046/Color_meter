from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


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
