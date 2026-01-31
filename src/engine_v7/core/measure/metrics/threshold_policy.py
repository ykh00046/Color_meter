"""
Inkness Threshold Policy

Gate 품질 지표 기반 적응형 threshold 조정 정책
3구간 분류: INK / REVIEW / GAP
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

# 기본 threshold 설정
DEFAULT_BASE_INK_THRESHOLD = 0.70
DEFAULT_REVIEW_WINDOW = 0.15  # INK threshold - 0.15 = REVIEW lower
DEFAULT_GAP_MARGIN = 0.05  # REVIEW lower - 0.05 = GAP upper

# 보정 한계
MAX_ADJUSTMENT = 0.10

# Gate 품질 임계값
BLUR_THRESHOLD_GOOD = 500.0
BLUR_THRESHOLD_POOR = 200.0
BLUR_PENALTY_MEDIUM = 0.03
BLUR_PENALTY_HIGH = 0.08

ILLUMINATION_THRESHOLD_GOOD = 0.05
ILLUMINATION_THRESHOLD_POOR = 0.15
ILLUMINATION_PENALTY_MEDIUM = 0.03
ILLUMINATION_PENALTY_HIGH = 0.05

OFFSET_THRESHOLD_GOOD = 1.0
OFFSET_THRESHOLD_POOR = 3.0
OFFSET_PENALTY_MEDIUM = 0.02
OFFSET_PENALTY_HIGH = 0.05


# ΔE gate (PASS/REVIEW/FAIL) 기본값 (method별)
# - PASS: deltaE <= pass_max
# - REVIEW: pass_max < deltaE <= review_max
# - FAIL: deltaE > review_max
#
# 주의: 아래 값은 '초기 기본값'이며 제품/조명/카메라/STD 품질에 맞춰 튜닝이 필요합니다.
DEFAULT_DELTAE_METHOD = "76"
DEFAULT_DELTAE_GATES_BY_METHOD = {
    # ΔE76 (Lab Euclidean)
    "76": {"pass_max": 2.0, "review_max": 4.0},
    # CIEDE2000 (ΔE00) — 보통 수치가 더 작게 나오는 편이라 threshold도 더 작게 시작
    "2000": {"pass_max": 1.8, "review_max": 3.0},
}

# 품질(blur/illum/offset)이 나쁠수록:
# - PASS 구간은 '조금 더 엄격'하게(=pass_max 감소) → false pass 감소
# - REVIEW 구간 상한은 '조금 더 관대'하게(=review_max 증가) → false fail 감소(대신 review 증가)
DELTAE_PASS_TIGHTEN_MAX_BY_METHOD = {"76": 1.0, "2000": 0.7}
DELTAE_REVIEW_RELAX_MAX_BY_METHOD = {"76": 1.5, "2000": 1.0}

# pass_max와 review_max 사이 최소 간격
MIN_DELTAE_REVIEW_GAP = 0.3


def normalize_deltae_method(method: str) -> str:
    """Normalize deltaE method string to '76' or '2000'."""
    m = str(method or DEFAULT_DELTAE_METHOD).strip().lower()
    if m in ("2000", "de2000", "ciede2000", "cie2000"):
        return "2000"
    return "76"


def get_adaptive_threshold(
    base_threshold: float = DEFAULT_BASE_INK_THRESHOLD,
    gate_scores: Optional[Dict[str, float]] = None,
    review_window: float = DEFAULT_REVIEW_WINDOW,
    gap_margin: float = DEFAULT_GAP_MARGIN,
    enable_adjustment: bool = True,
) -> Dict[str, Any]:
    """
    Gate 품질 지표 기반 적응형 threshold 계산

    Args:
        base_threshold: 기본 INK threshold (기본 0.70)
        gate_scores: Gate 품질 지표
            {
                "sharpness_score": float,      # 높을수록 선명 (300~800)
                "illumination_asymmetry": float, # 낮을수록 좋음 (0.0~0.3)
                "center_offset_mm": float       # 낮을수록 좋음 (0.0~10.0)
            }
        review_window: REVIEW 구간 폭 (기본 0.15)
        gap_margin: GAP 구간 마진 (기본 0.05)
        enable_adjustment: 보정 활성화 여부 (기본 True)

    Returns:
        {
            "ink_threshold": float,    # 보정된 INK 판정 기준
            "review_lower": float,     # REVIEW 구간 하한
            "gap_upper": float,        # GAP 구간 상한
            "adjustment": float,       # 보정량 (0.00 ~ 0.10)
            "reason": str,             # 보정 이유
            "quality_level": str       # "good" | "medium" | "poor" | "very_poor"
        }

    보정 규칙:
    - blur < 200: +0.08
    - blur 200~500: +0.03
    - illumination > 0.15: +0.05
    - illumination 0.05~0.15: +0.03
    - offset > 3.0: +0.05
    - offset 1.0~3.0: +0.02
    - 최대 adjustment: 0.10
    """
    if gate_scores is None:
        gate_scores = {}

    # Gate 지표 추출
    sharpness = gate_scores.get("sharpness_score", 500.0)  # 기본값: 좋음
    illumination = gate_scores.get("illumination_asymmetry", 0.03)  # 기본값: 좋음
    offset = gate_scores.get("center_offset_mm", 0.5)  # 기본값: 좋음

    # 보정량 계산
    if enable_adjustment:
        blur_penalty = _calculate_blur_penalty(sharpness)
        illum_penalty = _calculate_illumination_penalty(illumination)
        offset_penalty = _calculate_offset_penalty(offset)

        adjustment = min(blur_penalty + illum_penalty + offset_penalty, MAX_ADJUSTMENT)
    else:
        adjustment = 0.0
        blur_penalty = 0.0
        illum_penalty = 0.0
        offset_penalty = 0.0

    # 최종 threshold 계산
    ink_threshold = base_threshold + adjustment
    review_lower = ink_threshold - review_window
    gap_upper = review_lower - gap_margin

    # 보정 이유 생성
    reason_parts = []
    if blur_penalty > 0:
        reason_parts.append(f"low_sharpness(+{blur_penalty:.2f})")
    if illum_penalty > 0:
        reason_parts.append(f"illumination_issue(+{illum_penalty:.2f})")
    if offset_penalty > 0:
        reason_parts.append(f"center_offset(+{offset_penalty:.2f})")

    if reason_parts:
        reason = ", ".join(reason_parts)
    else:
        reason = "no_adjustment"

    # 품질 레벨 판단
    quality_level = _determine_quality_level(sharpness, illumination, offset)

    return {
        "ink_threshold": round(ink_threshold, 3),
        "review_lower": round(review_lower, 3),
        "gap_upper": round(gap_upper, 3),
        "adjustment": round(adjustment, 3),
        "reason": reason,
        "quality_level": quality_level,
        "gate_scores": {
            "sharpness_score": sharpness,
            "illumination_asymmetry": illumination,
            "center_offset_mm": offset,
        },
    }


def _calculate_blur_penalty(sharpness: float) -> float:
    """
    Blur penalty 계산

    Args:
        sharpness: sharpness_score (높을수록 선명)

    Returns:
        penalty: 0.00 ~ 0.08
    """
    if sharpness >= BLUR_THRESHOLD_GOOD:
        return 0.0
    elif sharpness >= BLUR_THRESHOLD_POOR:
        return BLUR_PENALTY_MEDIUM
    else:
        return BLUR_PENALTY_HIGH


def _calculate_illumination_penalty(illumination: float) -> float:
    """
    Illumination penalty 계산

    Args:
        illumination: illumination_asymmetry (낮을수록 좋음)

    Returns:
        penalty: 0.00 ~ 0.05
    """
    if illumination <= ILLUMINATION_THRESHOLD_GOOD:
        return 0.0
    elif illumination <= ILLUMINATION_THRESHOLD_POOR:
        return ILLUMINATION_PENALTY_MEDIUM
    else:
        return ILLUMINATION_PENALTY_HIGH


def _calculate_offset_penalty(offset: float) -> float:
    """
    Center offset penalty 계산

    Args:
        offset: center_offset_mm (낮을수록 좋음)

    Returns:
        penalty: 0.00 ~ 0.05
    """
    if offset <= OFFSET_THRESHOLD_GOOD:
        return 0.0
    elif offset <= OFFSET_THRESHOLD_POOR:
        return OFFSET_PENALTY_MEDIUM
    else:
        return OFFSET_PENALTY_HIGH


def _determine_quality_level(sharpness: float, illumination: float, offset: float) -> str:
    """
    전체 품질 레벨 판단

    Returns:
        "good" | "medium" | "poor" | "very_poor"
    """
    # Very poor: 2개 이상 지표가 poor
    poor_count = 0
    if sharpness < BLUR_THRESHOLD_POOR:
        poor_count += 1
    if illumination > ILLUMINATION_THRESHOLD_POOR:
        poor_count += 1
    if offset > OFFSET_THRESHOLD_POOR:
        poor_count += 1

    if poor_count >= 2:
        return "very_poor"

    # Poor: 1개 지표가 poor
    if poor_count == 1:
        return "poor"

    # Medium: 1개 이상 지표가 medium
    medium_count = 0
    if BLUR_THRESHOLD_POOR <= sharpness < BLUR_THRESHOLD_GOOD:
        medium_count += 1
    if ILLUMINATION_THRESHOLD_GOOD < illumination <= ILLUMINATION_THRESHOLD_POOR:
        medium_count += 1
    if OFFSET_THRESHOLD_GOOD < offset <= OFFSET_THRESHOLD_POOR:
        medium_count += 1

    if medium_count > 0:
        return "medium"

    return "good"


def classify_inkness(inkness_score: float, thresholds: Dict[str, float]) -> str:
    """
    Inkness score를 3구간으로 분류

    Args:
        inkness_score: 0~1 사이 inkness 점수
        thresholds: get_adaptive_threshold() 결과

    Returns:
        "ink" | "review" | "gap"
    """
    ink_threshold = thresholds["ink_threshold"]
    review_lower = thresholds["review_lower"]

    if inkness_score >= ink_threshold:
        return "ink"
    elif inkness_score >= review_lower:
        return "review"
    else:
        return "gap"


def validate_threshold_policy(thresholds: Dict[str, float]) -> bool:
    """
    Threshold policy 검증

    Args:
        thresholds: get_adaptive_threshold() 결과

    Returns:
        valid: True if valid

    Raises:
        ValueError: 검증 실패 시
    """
    ink_threshold = thresholds["ink_threshold"]
    review_lower = thresholds["review_lower"]
    gap_upper = thresholds["gap_upper"]

    # 1. 순서 검증
    if not (ink_threshold > review_lower > gap_upper):
        raise ValueError(f"Invalid threshold order: ink={ink_threshold}, " f"review={review_lower}, gap={gap_upper}")

    # 2. 범위 검증
    if ink_threshold > 1.0 or gap_upper < 0.0:
        raise ValueError(f"Threshold out of range [0, 1]: ink={ink_threshold}, gap={gap_upper}")

    # 3. REVIEW 구간 최소 폭 검증 (0.05 이상)
    review_width = ink_threshold - review_lower
    if review_width < 0.05:
        warnings.warn(f"REVIEW window too narrow: {review_width:.3f} (recommend ≥0.10)", UserWarning)

    return True


def get_threshold_summary(thresholds: Dict[str, float]) -> str:
    """
    Threshold policy 요약 문자열

    Args:
        thresholds: get_adaptive_threshold() 결과

    Returns:
        summary: Human-readable summary
    """
    lines = [
        "=== Inkness Threshold Policy ===",
        f"Quality: {thresholds['quality_level'].upper()}",
        f"Adjustment: +{thresholds['adjustment']:.3f} ({thresholds['reason']})",
        "",
        "Zones:",
        f"  INK:    inkness ≥ {thresholds['ink_threshold']:.3f}",
        f"  REVIEW: {thresholds['review_lower']:.3f} ~ {thresholds['ink_threshold']:.3f}",
        f"  GAP:    inkness < {thresholds['gap_upper']:.3f}",
        "",
        "Gate Scores:",
        f"  Sharpness:     {thresholds['gate_scores']['sharpness_score']:.1f}",
        f"  Illumination:  {thresholds['gate_scores']['illumination_asymmetry']:.3f}",
        f"  Center Offset: {thresholds['gate_scores']['center_offset_mm']:.2f} mm",
    ]

    return "\n".join(lines)


def should_retake(thresholds: Dict[str, float]) -> bool:
    """
    재촬영 권고 여부 판단

    Args:
        thresholds: get_adaptive_threshold() 결과

    Returns:
        True if retake recommended
    """
    quality_level = thresholds["quality_level"]

    # Very poor 품질이면 재촬영 권고
    if quality_level == "very_poor":
        return True

    # Adjustment가 최대치면 재촬영 권고
    if thresholds["adjustment"] >= MAX_ADJUSTMENT:
        return True

    return False


def _resolve_base_gates(deltae_base_gates: dict | None, method: str, quality_level: str) -> dict | None:
    """
    Accept flexible shapes:
      1) {"pass_max": 2.0, "review_max": 4.0}
      2) {"76": {...}, "2000": {...}}
      3) {"76": {"good": {...}, "poor": {...}}, "2000": {...}}
    """
    if not isinstance(deltae_base_gates, dict):
        return None

    m = normalize_deltae_method(method)
    base = deltae_base_gates

    # method layer
    if m in base and isinstance(base[m], dict):
        base = base[m]

    # quality layer
    if isinstance(base, dict) and quality_level in base and isinstance(base[quality_level], dict):
        base = base[quality_level]

    if not isinstance(base, dict):
        return None

    # accept pass_max/review_max keys only
    if "pass_max" in base and "review_max" in base:
        return {"pass_max": float(base["pass_max"]), "review_max": float(base["review_max"])}

    return None


def get_deltae_gates(*, deltae_method: str, quality_level: str, base_gates: dict | None = None) -> dict:
    method = normalize_deltae_method(deltae_method)

    defaults = {
        "76": {"pass_max": 2.0, "review_max": 4.0},
        "2000": {"pass_max": 1.8, "review_max": 3.0},
    }[method]

    # cfg override (supports nested dict)
    resolved = _resolve_base_gates(base_gates, method, quality_level)
    if resolved:
        defaults = resolved

    # 품질에 따른 보정(표본이 부족할 때의 안전장치)
    bump = 0.0
    tighten = 0.0
    if quality_level == "medium":
        bump, tighten = 0.2, 0.0
    elif quality_level == "poor":
        bump, tighten = 0.5, 0.1
    elif quality_level == "very_poor":
        bump, tighten = 1.0, 0.2

    pass_max = max(0.1, float(defaults["pass_max"]) - tighten)
    review_max = max(pass_max + 0.3, float(defaults["review_max"]) + bump)

    severity = {"good": 0.0, "medium": 0.2, "poor": 0.5, "very_poor": 1.0}.get(quality_level, 0.3)

    return {
        "deltaE_method": method,
        "pass_max": round(pass_max, 3),
        "review_max": round(review_max, 3),
        "quality_level": quality_level,
        "severity": severity,
        "reason": "cfg_or_quality_adjusted",
    }


def classify_deltae(deltae_value: float, deltae_gates: dict) -> str:
    p = float(deltae_gates.get("pass_max", 0.0))
    r = float(deltae_gates.get("review_max", 0.0))
    if deltae_value <= p:
        return "pass"
    if deltae_value <= r:
        return "review"
    return "fail"


def get_threshold_policy(
    *, gate_scores: dict, deltae_method: str = "76", deltae_base_gates: dict | None = None
) -> dict:
    ink = get_adaptive_threshold(gate_scores=gate_scores)
    qlvl = ink.get("quality_level", "good")
    de = get_deltae_gates(deltae_method=deltae_method, quality_level=qlvl, base_gates=deltae_base_gates)
    return {"inkness": ink, "deltaE": de}


def should_retake_policy(policy: dict) -> bool:
    ink = (policy or {}).get("inkness", {})
    return should_retake(ink)
