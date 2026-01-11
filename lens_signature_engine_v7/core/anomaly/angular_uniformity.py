from __future__ import annotations

import numpy as np

from ..utils import bgr_to_lab


def angular_uniformity_score(polar_bgr: np.ndarray, *, r_start: float, r_end: float) -> float:
    """
    Proxy for theta non-uniformity using "inkness" derived from L channel.
    Returns std/mean of inkness(theta).
    """
    lab = bgr_to_lab(polar_bgr)
    L = lab[..., 0]  # (T,R)
    T, R = L.shape
    r0 = int(R * r_start)
    r1 = int(R * r_end)
    roi = L[:, r0:r1]
    inkness_theta = (255.0 - roi).mean(axis=1)
    return float(np.std(inkness_theta) / (np.mean(inkness_theta) + 1e-6))
