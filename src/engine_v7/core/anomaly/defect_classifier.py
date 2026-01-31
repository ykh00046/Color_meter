from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def classify_defect(
    anomaly_scores: Dict[str, float], heatmap: Dict[str, Any] | None
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Heuristic classifier for NG_PATTERN explanation.
    Returns (defect_type, confidence, details)
      defect_type: BLOB_DEFECT / SECTOR_DEFECT / RING_DEFECT / UNIFORMITY_DEFECT / UNKNOWN
    """
    details: Dict[str, Any] = {}
    blob = float(anomaly_scores.get("center_blob_count", 0.0))
    ang = float(anomaly_scores.get("angular_uniformity", 0.0))

    # blob dominates
    if blob >= 1.0:
        return "BLOB_DEFECT", 0.85, {"center_blob_count": blob}

    # heatmap-based ring/sector cues
    if heatmap and "map" in heatmap:
        m = np.array(heatmap["map"], dtype=np.float32)  # (T,R)
        # radial profile (mean over theta) -> ring cue
        radial_mean = m.mean(axis=0)
        ring_peak = float(radial_mean.max() / (radial_mean.mean() + 1e-6))
        # angular profile (mean over radius) -> sector cue
        ang_mean = m.mean(axis=1)
        sector_peak = float(ang_mean.max() / (ang_mean.mean() + 1e-6))

        details.update({"ring_peak_ratio": ring_peak, "sector_peak_ratio": sector_peak})

        if ring_peak > 2.2 and ring_peak >= sector_peak:
            conf = min(0.9, 0.55 + 0.15 * (ring_peak - 2.2))
            return "RING_DEFECT", float(conf), details
        if sector_peak > 2.2:
            conf = min(0.9, 0.55 + 0.15 * (sector_peak - 2.2))
            return "SECTOR_DEFECT", float(conf), details

    # fallback using angular uniformity score (already in engine)
    if ang > 0.35:
        # higher ang => more likely sector-type nonuniformity
        conf = min(0.85, 0.55 + 0.8 * (ang - 0.35))
        return "SECTOR_DEFECT", float(conf), {"angular_uniformity": ang}

    return "UNIFORMITY_DEFECT", 0.55, details
