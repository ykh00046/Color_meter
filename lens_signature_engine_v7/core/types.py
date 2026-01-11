from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LensGeometry:
    cx: float
    cy: float
    r: float
    confidence: float = 1.0
    source: str = "hough"


@dataclass
class GateResult:
    passed: bool
    reasons: List[str]
    scores: Dict[str, float]


@dataclass
class SignatureResult:
    passed: bool
    score_corr: float
    delta_e_mean: float
    delta_e_p95: float
    fail_ratio: float
    fail_regions_r: List[int]
    reasons: List[str]
    flags: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyResult:
    passed: bool
    reasons: List[str]
    scores: Dict[str, float]
    debug: Dict[str, Any] = field(default_factory=dict)
    type: str = ""
    type_confidence: float = 0.0
    type_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    label: str
    reasons: List[str]
    gate: GateResult
    signature: Optional[SignatureResult]
    anomaly: Optional[AnomalyResult]
    debug: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # v7 extensions (sampling QC friendly)
    phase: str = "INSPECTION"
    best_mode: str = ""
    mode_scores: Dict[str, Any] = field(default_factory=dict)  # mode -> signature dict
    mode_shift: Dict[str, Any] = field(default_factory=dict)  # shift meta (SKU-level)
    registration_summary: Dict[str, Any] = field(default_factory=dict)
    reason_codes: List[str] = field(default_factory=list)
    reason_messages: List[str] = field(default_factory=list)
    v3_summary: Dict[str, Any] = field(default_factory=dict)
    v3_trend: Dict[str, Any] = field(default_factory=dict)
    ops: Dict[str, Any] = field(default_factory=dict)
    pattern_color: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
