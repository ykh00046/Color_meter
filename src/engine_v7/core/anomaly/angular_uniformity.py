from __future__ import annotations

import numpy as np

from ..utils import to_cie_lab


def angular_uniformity_score(polar_bgr: np.ndarray, *, r_start: float, r_end: float) -> float:
    """
    Proxy for theta non-uniformity using "inkness" derived from L channel.
    Returns std/mean of inkness(theta).
    """
    lab = to_cie_lab(polar_bgr)
    L = lab[..., 0]  # (T,R) - CIE Lab: 0-100
    T, R = L.shape
    r0 = int(R * r_start)
    r1 = int(R * r_end)
    roi = L[:, r0:r1]
    # Inkness: darker = more ink = higher inkness (CIE Lab L*: 0-100)
    inkness_theta = (100.0 - roi).mean(axis=1)
    return float(np.std(inkness_theta) / (np.mean(inkness_theta) + 1e-6))
