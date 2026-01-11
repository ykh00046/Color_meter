from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..types import LensGeometry


@dataclass
class StdModel:
    """
    STD signature model.

    - Single STD: radial_lab_mean + radial_lab_p95
    - Multi STD:  radial_lab_mean + radial_lab_std + radial_lab_p05/p95
    """

    geom: LensGeometry
    radial_lab_mean: np.ndarray  # (R', 3)
    radial_lab_p95: np.ndarray  # (R', 3)
    meta: Dict[str, Any]

    radial_lab_std: Optional[np.ndarray] = None  # (R', 3)
    radial_lab_p05: Optional[np.ndarray] = None  # (R', 3)
    radial_lab_median: Optional[np.ndarray] = None  # (R', 3)
    radial_lab_mad: Optional[np.ndarray] = None  # (R', 3)
