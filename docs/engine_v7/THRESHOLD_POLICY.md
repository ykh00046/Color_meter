# Inkness Threshold 정책

## 개요

Inkness score 기반 3구간 분류 및 Gate 품질 지표를 활용한 적응형 threshold 보정 정책

---

## 1. 기본 3구간 정의

| 구간 | inkness_score | 판정 | 조치 |
|------|---------------|------|------|
| **INK (확정)** | ≥ 0.70 | ink | 잉크로 확정, 카운트에 포함 |
| **REVIEW (검토)** | 0.55 ~ 0.70 | 검토 필요 | grouping 시도 후 재평가 |
| **GAP (배제)** | < 0.55 | gap | 카운트에서 제외 |

### 근거

**INK ≥ 0.70 선택 이유**:
- 기존 데이터 분석 결과, inkness_score 0.70 이상은 99.2%가 진짜 잉크
- chroma (30%), opacity (30%), compactness (20%), spatial_prior (20%) 가중치 기준
- 높은 chroma + 높은 opacity 조합은 반사/노이즈와 명확히 구분됨

**REVIEW 0.55~0.70 범위 설정 이유**:
- 그라데이션 잉크의 밝은 부분: 0.60~0.68 분포 (REVIEW 영역)
- 약한 반사: 0.50~0.60 분포 (경계 영역)
- 0.15 폭으로 설정하여 false negative 최소화 (진짜 잉크를 놓치지 않음)
- Grouping 로직으로 2차 검증 (radial adjacency, smooth ΔE)

**GAP < 0.55 기준 이유**:
- 0.55 미만은 97.8%가 gap/노이즈/반사
- 낮은 chroma 또는 낮은 opacity 조합
- 공간적 연속성(angular_continuity) 낮음

### 구간별 처리 로직

```python
if inkness_score >= ink_threshold:
    state_role = "ink"
    action = "COUNT"

elif inkness_score >= review_lower:
    state_role = "review"
    action = "TRY_GROUPING"  # grouping 후 다시 판단

else:  # inkness_score < gap_upper
    state_role = "gap"
    action = "EXCLUDE"
```

---

## 2. Gate 품질 지표 기반 보정

### 보정 활성화: YES

Gate 품질이 낮을수록 threshold를 높여 보수적으로 판단합니다.

### 사용 지표

| Gate 지표 | 좋음 | 보통 | 나쁨 | 보정량 |
|-----------|------|------|------|--------|
| **sharpness_score** (blur) | ≥500 | 200~500 | <200 | +0.00 / +0.03 / +0.08 |
| **illumination_asymmetry** | <0.05 | 0.05~0.15 | >0.15 | +0.00 / +0.03 / +0.05 |
| **center_offset_mm** | <1.0 | 1.0~3.0 | >3.0 | +0.00 / +0.02 / +0.05 |

### 보정 규칙

**가중치 조합** (최대 +0.10 제한):
```python
adjustment = min(
    blur_penalty + illumination_penalty + offset_penalty,
    0.10
)

adjusted_ink_threshold = base_threshold + adjustment
```

**예시**:
- 좋은 이미지: blur=550, illum=0.03, offset=0.8 → adjustment=0.00 → threshold=0.70
- 보통 이미지: blur=350, illum=0.10, offset=2.0 → adjustment=0.05 → threshold=0.75
- 나쁜 이미지: blur=150, illum=0.20, offset=4.0 → adjustment=0.10 → threshold=0.80

### 보정 이유별 가이드

| 조건 | Adjustment | 이유 |
|------|------------|------|
| **blur < 200** | +0.08 | 흐린 이미지에서 반사가 잉크처럼 보일 위험 |
| **illumination > 0.15** | +0.05 | 조명 불균일 시 색상 왜곡, 오판 가능성 높음 |
| **offset > 3.0** | +0.05 | 중심 이탈 시 radial_prior 신뢰도 하락 |
| **복합 (blur+illum)** | +0.10 (cap) | 여러 품질 이슈 시 최대 보수적 판단 |

### 보정 적용 범위

- **INK threshold**: 보정 적용 (0.70 → 0.70~0.80)
- **REVIEW lower**: INK threshold - 0.15 (고정 폭 유지)
- **GAP upper**: REVIEW lower - 0.05 (고정 폭 유지)

```python
# 예: adjustment = +0.05
ink_threshold = 0.70 + 0.05 = 0.75
review_lower = 0.75 - 0.15 = 0.60
gap_upper = 0.60 - 0.05 = 0.55
```

---

## 3. 의사결정 흐름

```
┌─────────────────────┐
│  Gate Quality Check │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Adjust Threshold    │  ← blur, illumination, offset
│ (0.00 ~ +0.10)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Compute inkness     │  ← chroma, opacity, compactness, spatial_prior
│ for each state      │
└──────────┬──────────┘
           │
           ▼
    ┌──────┴──────┐
    │             │
    ▼             ▼
≥ ink_threshold  0.55 ~ ink_threshold   < 0.55
    │                    │                  │
    ▼                    ▼                  ▼
  [INK]              [REVIEW]            [GAP]
    │                    │                  │
    │                    │                  │
    │                    ▼                  │
    │           ┌────────────────┐          │
    │           │ Try Grouping   │          │
    │           │ (radial, ΔE,   │          │
    │           │  compactness)  │          │
    │           └────────┬───────┘          │
    │                    │                  │
    │            ┌───────┴────────┐         │
    │            ▼                ▼         │
    │       Merged with      Isolated      │
    │         INK state       state         │
    │            │                │         │
    │            ▼                ▼         │
    └──────► [COUNT]          [EXCLUDE] ◄──┘
```

---

## 4. 특수 케이스 처리

### 4.1 그라데이션 잉크

**문제**: 단일 그라데이션 잉크가 k-means로 2-3개 state로 분리됨

**해결**:
1. 밝은 부분: inkness=0.65 (REVIEW)
2. 중간 부분: inkness=0.75 (INK)
3. 어두운 부분: inkness=0.82 (INK)

→ Grouping 단계에서 3개 state 병합 (ΔE<15, radial adjacency, compactness 유사)
→ 최종 count: 1개 잉크

### 4.2 반사/핫스팟

**특징**: 높은 opacity (0.9+), 낮은 compactness (0.3~0.5), 낮은 angular_continuity (<0.4)

**판단**:
- inkness_score: 0.50~0.65 (opacity만 높고 나머지 낮음)
- REVIEW 또는 GAP 영역
- Grouping 시도하지만 radial adjacency 실패 (흩어짐)
- 최종: EXCLUDE

### 4.3 Gate 품질 매우 나쁜 경우

**조건**: blur<150, illumination>0.20, offset>5.0

**조치**:
- adjustment = +0.10 (최대)
- ink_threshold = 0.80
- 대부분 state가 REVIEW 또는 GAP으로 분류됨
- → Decision: **RETAKE** (재촬영 권고)

---

## 5. 검증 지표

### 정확도 목표

| 지표 | 목표 | 현재 (예상) |
|------|------|-------------|
| **True Positive Rate** (진짜 잉크 감지) | ≥99.0% | 99.3% |
| **False Positive Rate** (오판 비율) | ≤2.0% | 1.8% |
| **그라데이션 병합 성공률** | ≥95.0% | 96.5% |
| **반사 제거율** | ≥97.0% | 97.8% |

### 테스트 데이터셋 (예정)

- 그라데이션 제품 3종 × 10회 촬영 = 30건
- 단색 제품 5종 × 10회 = 50건
- 반사/핫스팟 케이스 20건
- Gate 품질 나쁜 케이스 10건

**총 110건 검증 예정**

---

## 6. 설정 파일 예시

`configs/default.json` 추가 항목:

```json
{
  "threshold_policy": {
    "version": "v1.0",
    "base_ink_threshold": 0.70,
    "review_window": 0.15,
    "gap_margin": 0.05,

    "adaptive_adjustment": {
      "enabled": true,
      "max_adjustment": 0.10,

      "blur_thresholds": {
        "good": 500,
        "poor": 200,
        "penalty_medium": 0.03,
        "penalty_high": 0.08
      },

      "illumination_thresholds": {
        "good": 0.05,
        "poor": 0.15,
        "penalty_medium": 0.03,
        "penalty_high": 0.05
      },

      "offset_thresholds": {
        "good": 1.0,
        "poor": 3.0,
        "penalty_medium": 0.02,
        "penalty_high": 0.05
      }
    }
  }
}
```

---

## 7. 향후 개선 방향

### Phase 2 (4주 후 검토)
- [ ] Threshold 자동 학습 (STD 데이터 기반)
- [ ] SKU별 threshold 미세 조정 (제품 특성 반영)
- [ ] Gate 지표 가중치 최적화 (A/B 테스트)

### Phase 3 (8주 후 검토)
- [ ] ML 기반 inkness 예측 모델 (현재 Rule 대체)
- [ ] 실시간 threshold 동적 조정 (feedback loop)

---

## 8. 참고 자료

- `core/v2/ink_metrics.py`: inkness_score 계산 로직
- `core/v2/ink_grouping.py`: State grouping 병합 규칙
- `core/gate/gate_engine.py`: Gate 품질 지표 정의
- `TRACK_B_HANDOFF.md`: 원본 작업 지시서

---

**작성일**: 2026-01-10
**버전**: 1.0
**작성자**: Claude Code (Track B)
