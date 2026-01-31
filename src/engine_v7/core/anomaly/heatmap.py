from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from ..utils import to_cie_lab


def anomaly_heatmap(polar_bgr: np.ndarray, *, ds_T: int, ds_R: int) -> Dict[str, Any]:
    lab = to_cie_lab(polar_bgr)
    L = lab[..., 0]  # (T,R) - CIE Lab L*: 0-100
    L_norm = (L - L.min()) / (L.max() - L.min() + 1e-6)

    theta_mean = L_norm.mean(axis=0, keepdims=True)
    resid = np.abs(L_norm - theta_mean)

    resid_ds = cv2.resize(resid, (ds_R, ds_T), interpolation=cv2.INTER_AREA)
    resid_ds = np.clip(resid_ds, 0.0, 1.0)

    return {
        "shape": [int(ds_T), int(ds_R)],
        "mean": float(resid_ds.mean()),
        "p95": float(np.percentile(resid_ds, 95)),
        "map": resid_ds.astype(np.float32).tolist(),
    }
