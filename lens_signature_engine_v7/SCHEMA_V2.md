# Uncertainty & Evidence Schema V2

## 개요

기존 단순 warnings 리스트 대신, 구조화된 uncertainty/evidence 체계로 개선하여
의사결정 분기 로직(RETAKE/HOLD/NG_COLOR)을 지원합니다.

---

## 1. 현재 문제점 (V1)

### 기존 구조
```python
Decision = {
    "result": "NG_COLOR",
    "reasons": ["INK_COUNT_MISMATCH", "BLUR_TOO_LOW"],
    "warnings": [
        "LOW_SHARPNESS",
        "ILLUMINATION_ASYMMETRY_HIGH",
        "INK_COUNT_MISMATCH_SUSPECTED"
    ]
}
```

### 문제
1. **구조화 부족**: 문자열 나열로 심각도/원인 파악 어려움
2. **중복 표현**: `reasons`와 `warnings` 혼재
3. **의사결정 불가**: RETAKE vs HOLD vs NG 구분 불가능
4. **근거 부족**: 왜 불확실한지, 어느 정도 확신하는지 알 수 없음

---

## 2. 새 스키마 구조 (V2)

### 2.1 Decision 최상위 구조

```python
Decision = {
    # 기존 필드
    "result": str,  # "OK" | "RETAKE" | "HOLD" | "NG_COLOR"
    "confidence": float,  # 0~1 (1=매우 확신, 0=불확실)

    # 새 필드
    "uncertainty": {
        "level": str,  # "low" | "medium" | "high"
        "factors": List[UncertaintyFactor],
        "total_score": float  # 0~1 (1=매우 불확실)
    },

    "evidence": {
        "positive": List[Evidence],  # NG를 지지하는 증거
        "negative": List[Evidence],  # OK를 지지하는 증거
        "neutral": List[Evidence]    # 중립적 관찰
    },

    "reason_codes": List[str],  # 간소화된 reason codes

    # 기존 필드 (호환성)
    "reasons": List[str],  # deprecated, use reason_codes
    "warnings": List[str]  # deprecated, use uncertainty.factors
}
```

### 2.2 UncertaintyFactor 구조

불확실성 요인 (의사결정을 어렵게 만드는 요소)

```python
UncertaintyFactor = {
    "category": str,  # "gate" | "sampling" | "segmentation" | "signature"
    "code": str,      # "BLUR_TOO_LOW", "SAMPLING_SPARSE", etc.
    "severity": str,  # "low" | "medium" | "high" | "critical"
    "impact": float,  # 0~1 (불확실성 기여도)
    "message": str,   # Human-readable 설명
    "details": dict   # 세부 정보
}
```

**예시**:
```python
{
    "category": "gate",
    "code": "BLUR_TOO_LOW",
    "severity": "high",
    "impact": 0.35,
    "message": "Image sharpness too low (150.3 < 200 threshold)",
    "details": {
        "sharpness_score": 150.3,
        "threshold": 200.0,
        "recommendation": "Retake with better focus"
    }
}
```

### 2.3 Evidence 구조

의사결정 근거 (확신을 높이거나 낮추는 증거)

```python
Evidence = {
    "type": str,      # "gate" | "signature" | "ink" | "spatial"
    "code": str,      # "HIGH_CHROMA", "ANGULAR_CONTINUITY_GOOD", etc.
    "strength": str,  # "weak" | "moderate" | "strong"
    "weight": float,  # 0~1 (의사결정 기여도)
    "value": Any,     # 측정값
    "message": str    # Human-readable 설명
}
```

**예시 (positive - NG 지지)**:
```python
{
    "type": "ink",
    "code": "EXCESSIVE_INK_COUNT",
    "strength": "strong",
    "weight": 0.80,
    "value": 5,
    "message": "Detected 5 inks (expected 2), strong evidence of defect"
}
```

**예시 (negative - OK 지지)**:
```python
{
    "type": "signature",
    "code": "SIGNATURE_MATCH_EXCELLENT",
    "strength": "strong",
    "weight": 0.75,
    "value": 0.92,
    "message": "Signature score 0.92 (threshold 0.85), excellent match"
}
```

---

## 3. 의사결정 분기 로직

### 3.1 분기 조건

```python
if uncertainty.level == "critical" or should_retake(gate_scores):
    result = "RETAKE"
    # Gate 품질 문제 → 재촬영 권고

elif uncertainty.level == "high":
    result = "HOLD"
    # 불확실성 높음 → 사람 검토 필요

elif count(positive_evidence) > count(negative_evidence):
    result = "NG_COLOR"
    # 확실한 불량 증거

else:
    result = "OK"
    # 정상
```

### 3.2 Uncertainty Level 계산

```python
def calculate_uncertainty_level(factors: List[UncertaintyFactor]) -> str:
    """
    불확실성 레벨 계산

    Returns:
        "low": total_impact < 0.20
        "medium": 0.20 ≤ total_impact < 0.50
        "high": 0.50 ≤ total_impact < 0.80
        "critical": total_impact ≥ 0.80
    """
    total_impact = sum(f["impact"] for f in factors)

    if total_impact >= 0.80:
        return "critical"
    elif total_impact >= 0.50:
        return "high"
    elif total_impact >= 0.20:
        return "medium"
    else:
        return "low"
```

### 3.3 Confidence 계산

```python
def calculate_confidence(uncertainty_score: float, evidence_balance: float) -> float:
    """
    최종 확신도 계산

    Args:
        uncertainty_score: 0~1 (높을수록 불확실)
        evidence_balance: -1~1 (positive - negative 증거 비율)

    Returns:
        confidence: 0~1 (1=매우 확신)
    """
    # 불확실성이 높으면 확신도 낮음
    base_confidence = 1.0 - uncertainty_score

    # 증거가 한쪽으로 치우칠수록 확신도 높음
    evidence_boost = abs(evidence_balance) * 0.3

    confidence = min(base_confidence + evidence_boost, 1.0)
    return confidence
```

---

## 4. UncertaintyFactor 카테고리별 정의

### 4.1 Gate (이미지 품질)

| Code | Severity | Impact | Condition |
|------|----------|--------|-----------|
| `BLUR_CRITICAL` | critical | 0.80 | sharpness < 150 |
| `BLUR_TOO_LOW` | high | 0.35 | sharpness < 200 |
| `BLUR_SLIGHTLY_LOW` | medium | 0.15 | sharpness < 300 |
| `ILLUMINATION_SEVERE` | high | 0.30 | asymmetry > 0.20 |
| `ILLUMINATION_MODERATE` | medium | 0.15 | asymmetry > 0.10 |
| `CENTER_OFFSET_HIGH` | medium | 0.20 | offset > 3.0 mm |
| `CENTER_OFFSET_MODERATE` | low | 0.10 | offset > 2.0 mm |

### 4.2 Sampling (샘플링 품질)

| Code | Severity | Impact | Condition |
|------|----------|--------|-----------|
| `SAMPLING_TOO_SPARSE` | high | 0.40 | sample_count < 500 |
| `SAMPLING_IMBALANCED` | medium | 0.25 | cluster size ratio > 10:1 |
| `SAMPLING_LOW_SILHOUETTE` | medium | 0.20 | silhouette_score < 0.3 |

**중요**: auto_k 결과는 더 이상 "INK_COUNT_MISMATCH"가 아님
→ `SAMPLING_SUGGESTS_DIFFERENT_K` (severity: low, impact: 0.10)

### 4.3 Segmentation (색상 분리)

| Code | Severity | Impact | Condition |
|------|----------|--------|-----------|
| `SEGMENTATION_POOR_SEPARATION` | high | 0.35 | min ΔE between clusters < 10 |
| `SEGMENTATION_OVERLAP` | medium | 0.25 | cluster overlap ratio > 0.15 |

### 4.4 Signature (프로파일 매칭)

| Code | Severity | Impact | Condition |
|------|----------|--------|-----------|
| `SIGNATURE_AMBIGUOUS` | medium | 0.30 | score near threshold (±0.05) |
| `SIGNATURE_MODE_CONFLICT` | low | 0.15 | LOW/MID/HIGH 선택 불일치 |

---

## 5. Evidence 타입별 정의

### 5.1 Positive Evidence (NG 지지)

| Code | Strength | Weight | Condition |
|------|----------|--------|-----------|
| `EXCESSIVE_INK_COUNT` | strong | 0.80 | ink_count > expected + 2 |
| `HIGH_DELTAEE` | strong | 0.75 | signature ΔE > threshold + 0.10 |
| `ANGULAR_FRAGMENTATION` | moderate | 0.50 | angular_continuity < 0.40 |
| `LOW_INKNESS_SCORE` | moderate | 0.45 | inkness < gap_threshold |

### 5.2 Negative Evidence (OK 지지)

| Code | Strength | Weight | Condition |
|------|----------|--------|-----------|
| `SIGNATURE_MATCH_EXCELLENT` | strong | 0.75 | signature score > 0.90 |
| `INK_COUNT_EXACT` | strong | 0.70 | ink_count == expected |
| `ANGULAR_CONTINUITY_GOOD` | moderate | 0.50 | angular_continuity > 0.85 |
| `HIGH_CHROMA` | weak | 0.30 | chroma > 40 |

### 5.3 Neutral Evidence (관찰)

| Code | Value |
|------|-------|
| `GRADIENT_DETECTED` | gradient_groups count |
| `REFLECTION_POSSIBLE` | reflection areas count |
| `GATE_ADJUSTMENT_APPLIED` | adjustment amount |

---

## 6. 실제 사용 예시

### 6.1 정상 케이스 (OK)

```python
{
    "result": "OK",
    "confidence": 0.92,

    "uncertainty": {
        "level": "low",
        "total_score": 0.10,
        "factors": [
            {
                "category": "gate",
                "code": "BLUR_SLIGHTLY_LOW",
                "severity": "medium",
                "impact": 0.10,
                "message": "Sharpness 280 (slightly below ideal 300+)"
            }
        ]
    },

    "evidence": {
        "positive": [],
        "negative": [
            {
                "type": "ink",
                "code": "INK_COUNT_EXACT",
                "strength": "strong",
                "weight": 0.70,
                "value": 2,
                "message": "Detected 2 inks (expected 2)"
            },
            {
                "type": "signature",
                "code": "SIGNATURE_MATCH_EXCELLENT",
                "strength": "strong",
                "weight": 0.75,
                "value": 0.91,
                "message": "Signature score 0.91"
            }
        ],
        "neutral": []
    },

    "reason_codes": []
}
```

### 6.2 재촬영 케이스 (RETAKE)

```python
{
    "result": "RETAKE",
    "confidence": 0.30,

    "uncertainty": {
        "level": "critical",
        "total_score": 0.85,
        "factors": [
            {
                "category": "gate",
                "code": "BLUR_CRITICAL",
                "severity": "critical",
                "impact": 0.80,
                "message": "Sharpness 120 (critical, threshold 150)",
                "details": {
                    "sharpness_score": 120.0,
                    "threshold": 150.0,
                    "recommendation": "Check camera focus and lens cleanliness"
                }
            },
            {
                "category": "gate",
                "code": "ILLUMINATION_SEVERE",
                "severity": "high",
                "impact": 0.30,
                "message": "Illumination asymmetry 0.25 (severe)"
            }
        ]
    },

    "evidence": {
        "positive": [],
        "negative": [],
        "neutral": [
            {
                "type": "gate",
                "code": "GATE_QUALITY_POOR",
                "value": "very_poor",
                "message": "Image quality too poor for reliable inspection"
            }
        ]
    },

    "reason_codes": ["RETAKE_REQUIRED"]
}
```

### 6.3 보류 케이스 (HOLD)

```python
{
    "result": "HOLD",
    "confidence": 0.45,

    "uncertainty": {
        "level": "high",
        "total_score": 0.60,
        "factors": [
            {
                "category": "sampling",
                "code": "SAMPLING_LOW_SILHOUETTE",
                "severity": "medium",
                "impact": 0.20,
                "message": "Silhouette score 0.28 (ambiguous clustering)"
            },
            {
                "category": "signature",
                "code": "SIGNATURE_AMBIGUOUS",
                "severity": "medium",
                "impact": 0.30,
                "message": "Signature score 0.87 (near threshold 0.85)"
            },
            {
                "category": "segmentation",
                "code": "SEGMENTATION_POOR_SEPARATION",
                "severity": "high",
                "impact": 0.35,
                "message": "Min ΔE between clusters: 8.5 (threshold 10)"
            }
        ]
    },

    "evidence": {
        "positive": [
            {
                "type": "spatial",
                "code": "ANGULAR_FRAGMENTATION",
                "strength": "moderate",
                "weight": 0.50,
                "value": 0.38,
                "message": "Angular continuity 0.38 (suspicious pattern)"
            }
        ],
        "negative": [
            {
                "type": "ink",
                "code": "INK_COUNT_EXACT",
                "strength": "strong",
                "weight": 0.70,
                "value": 2,
                "message": "Detected 2 inks (expected 2)"
            }
        ],
        "neutral": [
            {
                "type": "sampling",
                "code": "SAMPLING_SUGGESTS_DIFFERENT_K",
                "value": 3,
                "message": "Auto-k suggests 3 clusters (expected 2), may indicate gradient"
            }
        ]
    },

    "reason_codes": ["UNCERTAINTY_HIGH", "MANUAL_REVIEW_REQUIRED"]
}
```

### 6.4 불량 케이스 (NG_COLOR)

```python
{
    "result": "NG_COLOR",
    "confidence": 0.88,

    "uncertainty": {
        "level": "low",
        "total_score": 0.15,
        "factors": [
            {
                "category": "gate",
                "code": "BLUR_SLIGHTLY_LOW",
                "severity": "medium",
                "impact": 0.15,
                "message": "Sharpness 260 (acceptable but not ideal)"
            }
        ]
    },

    "evidence": {
        "positive": [
            {
                "type": "ink",
                "code": "EXCESSIVE_INK_COUNT",
                "strength": "strong",
                "weight": 0.80,
                "value": 5,
                "message": "Detected 5 ink groups (expected 2)"
            },
            {
                "type": "signature",
                "code": "HIGH_DELTAEE",
                "strength": "strong",
                "weight": 0.75,
                "value": 0.25,
                "message": "Signature ΔE 0.25 (threshold 0.10)"
            }
        ],
        "negative": [],
        "neutral": []
    },

    "reason_codes": ["INK_COUNT_MISMATCH", "SIGNATURE_DEVIATION"]
}
```

---

## 7. 마이그레이션 가이드

### 7.1 기존 코드 (V1)
```python
decision = {
    "result": "NG_COLOR",
    "reasons": ["INK_COUNT_MISMATCH"],
    "warnings": ["BLUR_TOO_LOW", "INK_COUNT_MISMATCH_SUSPECTED"]
}
```

### 7.2 새 코드 (V2)
```python
from core.decision.uncertainty import (
    UncertaintyBuilder,
    EvidenceBuilder,
    calculate_decision_v2
)

# 1. Uncertainty 수집
uncertainty_builder = UncertaintyBuilder()
if sharpness < 200:
    uncertainty_builder.add_gate_factor(
        code="BLUR_TOO_LOW",
        severity="high",
        sharpness=sharpness
    )

# 2. Evidence 수집
evidence_builder = EvidenceBuilder()
if ink_count > expected + 2:
    evidence_builder.add_positive(
        code="EXCESSIVE_INK_COUNT",
        strength="strong",
        value=ink_count
    )

# 3. Decision 계산
decision = calculate_decision_v2(
    uncertainty=uncertainty_builder.build(),
    evidence=evidence_builder.build(),
    gate_scores=gate_scores
)
```

---

## 8. 구현 파일

| 파일 | 내용 |
|------|------|
| `core/decision/uncertainty.py` | UncertaintyFactor, Evidence 클래스 |
| `core/decision/decision_engine.py` | calculate_decision_v2() 함수 |
| `core/reason_codes.py` | 정리된 reason codes 정의 |
| `tests/test_uncertainty_schema.py` | 단위 테스트 |

---

**작성일**: 2026-01-10
**버전**: 2.0
**작성자**: Claude Code (Track C)
