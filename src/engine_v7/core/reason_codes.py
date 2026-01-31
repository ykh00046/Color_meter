from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

REASON_MESSAGES: Dict[str, str] = {
    "MODEL_NOT_FOUND": "Active model not found for SKU/INK.",
    "MODEL_INCOMPLETE": "Model set is incomplete.",
    "MODEL_PATH_INVALID": "Model path is invalid.",
    "MODEL_LOAD_FAILED": "Model load failed.",
    "CFG_MISMATCH": "Config hash mismatch.",
    "INDEX_NOT_FOUND": "Model index.json not found.",
    "INDEX_INVALID": "Model index.json is invalid.",
    "PATTERN_BASELINE_NOT_FOUND": "Pattern baseline not found.",
    "PATTERN_BASELINE_PATH_INVALID": "Pattern baseline path is invalid.",
    "PATTERN_BASELINE_LOAD_FAILED": "Pattern baseline load failed.",
    "PATTERN_BASELINE_SCHEMA_MISMATCH": "Pattern baseline schema mismatch.",
    "CENTER_NOT_IN_FRAME": "Lens center is outside the frame.",
    "BLUR_LOW": "Image blur is below minimum.",
    "ILLUMINATION_UNEVEN": "Illumination is uneven.",
    "SIGNATURE_CORR_LOW": "Signature correlation is below threshold.",
    "DELTAE_P95_HIGH": "DeltaE p95 is above threshold.",
    "DELTAE_MEAN_HIGH": "DeltaE mean is above threshold.",
    "BAND_VIOLATION_HIGH": "Signature band violation is high.",
    "SEGMENT_BAND_VIOLATION": "Segment band violation detected.",
    "ANGULAR_UNIFORMITY_HIGH": "Angular uniformity is high.",
    "CENTER_BLOBS": "Center blob defects detected.",
    "MODE_ORDER_MISMATCH": "Mode order mismatch.",
    "MODE_SEPARATION_LOW": "Mode separation is low.",
    "MODE_VARIANCE_HIGH": "Mode variance is high.",
    "STD_REGISTRATION_REQUIRES_MULTI": "STD registration requires LOW/MID/HIGH models.",
    "PACK_NOT_FOUND": "Approval pack not found.",
    "PACK_MISMATCH": "Approval pack context mismatch.",
    "PACK_BLOCKED": "Approval pack status BLOCKED cannot be used.",
    "PACK_UPDATE_FAILED": "Approval pack final update failed.",
    "ACTIVATE_SUCCESS": "Activated new ACTIVE model.",
    "ROLLBACK_SUCCESS": "Rolled back to previous ACTIVE model.",
    "COLOR_SHIFT_YELLOW": "Color shift: yellow (b*+).",
    "COLOR_SHIFT_BLUE": "Color shift: blue (b*-).",
    "COLOR_SHIFT_RED": "Color shift: red (a*+).",
    "COLOR_SHIFT_GREEN": "Color shift: green (a*-).",
    "COLOR_SHIFT_DARK": "Color shift: dark (L*-).",
    "COLOR_SHIFT_LIGHT": "Color shift: light (L*+).",
    "PATTERN_DOT_COVERAGE_HIGH": "Pattern: dot coverage high.",
    "PATTERN_DOT_COVERAGE_LOW": "Pattern: dot coverage low.",
    "PATTERN_EDGE_BLUR": "Pattern: edge sharpness decreased.",
    "PATTERN_DOT_SPREAD": "Pattern: dot spread detected.",
    "PATTERN_CENTER_BLOB_EXCESS": "Pattern: center blob count exceeds baseline.",
    "PATTERN_UNIFORMITY_EXCESS": "Pattern: angular non-uniformity exceeds baseline.",
    "PATTERN_DOT_COVERAGE_OUT_OF_BAND": "Pattern: dot coverage out of baseline band.",
    "PATTERN_EDGE_SHARPNESS_LOW": "Pattern: edge sharpness below baseline band.",
    "SIGNATURE_ZERO_DE_GLOBAL_TONE_SHIFT": "Signature deltaE is zero but global tone shift detected.",
}


def split_reason(reason: str) -> Tuple[str, str]:
    if ":" in reason:
        code, detail = reason.split(":", 1)
        return code, detail
    return reason, ""


def reason_codes(reasons: Iterable[str]) -> List[str]:
    return [split_reason(r)[0] for r in reasons]


def reason_messages(reasons: Iterable[str]) -> List[str]:
    out: List[str] = []
    for r in reasons:
        code, detail = split_reason(r)
        base = REASON_MESSAGES.get(code, code)
        if detail:
            out.append(f"{base} ({detail})")
        else:
            out.append(base)
    return out
