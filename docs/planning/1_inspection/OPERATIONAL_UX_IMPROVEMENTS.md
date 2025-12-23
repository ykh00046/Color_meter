# 운영 UX 개선 (Operational UX Improvements)

**작성일:** 2025-12-13
**상태:** ✅ 구현 완료 (Phase 1 Quick Wins)

---

## 📋 개요

Contact Lens Color Inspection System의 **현장 운영성**을 높이기 위한 UX 개선 작업입니다.

### 핵심 목표
1. **오탐 감소**: "신뢰할 수 있는 시스템"으로 만들기
2. **조치 가능성**: 판정 결과에서 "다음 행동"까지 제시
3. **투명성**: OK/NG 이유를 명확하게 설명

---

## ✅ 구현 완료 (2025-12-13)

### 1. Diff Summary (색상 변화 방향 설명)

**위치:** `src/core/color_evaluator.py`, `src/core/zone_analyzer_2d.py`

**기능:**
- Zone별로 측정값과 기준값의 차이를 **사람이 이해하기 쉬운 말**로 번역
- 6개 카테고리: 어두워짐/밝아짐, 붉어짐/녹색화, 황색화/청색화
- 가장 큰 변화 2개만 강조 (정보 과부하 방지)

**출력 예시:**
```json
{
  "zone_name": "B",
  "delta_e": 7.65,
  "threshold": 8.0,
  "is_ok": true,
  "diff": {
    "dL": -6.14,
    "da": +0.07,
    "db": +4.55,
    "direction": "어두워짐(ΔL=-6.1), 황색화(Δb=+4.6)"
  }
}
```

**효과:**
- 사용자가 **공정 조정 방향**을 즉시 파악
- "ΔE가 크다"에서 → "어떤 방향으로 다른지" 명확화

---

### 2. RETAKE 상태 도입 (OK/NG/RETAKE 3단계)

**위치:** `src/core/zone_analyzer_2d.py:1125-1166`

**판정 우선순위:**
```
RETAKE (촬영 문제) > NG (품질 불량) > OK (합격)
```

**RETAKE Reason Codes (4개):**

| Code | 조건 | 설명 | 조치 레버 |
|------|------|------|-----------|
| **R1_DetectionLow** | lens_conf < 0.7 | 렌즈 검출 신뢰도 낮음 | 촬영 |
| **R2_CoverageLow** | pixels < 50% | 픽셀 커버리지 부족 | 촬영 |
| **R3_BoundaryUncertain** | fallback + conf < 0.8 | 경계 탐지 불확실 | 촬영 |
| **R4_UniformityLow** | std_L > 10.0 | 각도 불균일 높음 | 촬영 |

**각 Reason 구조:**
```json
{
  "code": "R1_DetectionLow",
  "reason": "렌즈 검출 신뢰도 낮음",
  "actions": [
    "렌즈 중앙 정렬 확인",
    "반사 감소 (조명 각도 조정)",
    "초점 재조정"
  ],
  "lever": "촬영"
}
```

**효과:**
- 오탐 방지: 촬영 문제를 NG로 보고하지 않음
- 조치 가능성: 구체적인 액션 가이드 제공
- 책임 명확화: lever 필드로 누가 조치할지 표시

---

### 3. 각도 불균일 검출 및 대응

**위치:** `src/core/zone_analyzer_2d.py:1157-1166`

**로직:**

| std_L 범위 | 동작 | 이유 |
|-----------|------|------|
| **> 10.0** | → **RETAKE (R4)** | 촬영/반사 의심, 오탐 방지 |
| **5.0 ~ 10.0** | → **Confidence penalty** | 신뢰도 감소 (이미 구현됨) |
| **< 5.0** | → 정상 | 균일함 |

**효과:**
- **False OK 방지**: 평균은 OK인데 일부 sector만 NG인 경우 감지
- 국부 결함(도트 불량, 얼룩) 검출
- NG보다 RETAKE 우선 → 사용자 수용성 향상

---

### 4. OK 여유도/신뢰도/Soft Warning

**위치:** `src/core/zone_analyzer_2d.py:1173-1195`

**OK 상태일 때 제공되는 컨텍스트:**

```json
{
  "judgment": "OK",
  "confidence": 0.87,
  "debug": {
    "ok_context": [
      "Zone 여유도: A:ΔE=2.97(여유 63%), B:ΔE=7.65(여유 4%), C:ΔE=2.71(여유 73%)",
      "신뢰도: 0.87 (재촬영 불필요)",
      "주의: Zone B는 fallback 범위 사용 (전이 탐지 불확실)"
    ]
  }
}
```

**제공 정보:**
1. **Zone별 여유도**: 각 Zone이 기준 대비 얼마나 여유가 있는지 (%)
2. **신뢰도**: 재촬영 필요 여부 판단 기준
3. **Soft Warning**:
   - 각도 균일성 경계값 근접 (std_L 5~10)
   - B zone fallback 사용

**효과:**
- 시스템이 "판정기"가 아니라 **"도구"**가 됨
- OK여도 "얼마나 안전한지" 파악 가능
- 예방적 조치 가능 (경계값 근접 경고)

---

## 📊 JSON 응답 구조 (전체)

### 기본 구조
```json
{
  "run_id": "3e1a6338",
  "judgment": "OK",  // "OK", "NG", "RETAKE"
  "overall_delta_e": 4.44,
  "confidence": 0.87,
  "zone_results": [
    {
      "zone_name": "C",
      "measured_lab": [97.69, 0.15, 1.89],
      "target_lab": [95.0, 0.5, 2.0],
      "delta_e": 2.71,
      "threshold": 10.0,
      "is_ok": true,
      "diff": {
        "dL": +2.69,
        "da": -0.35,
        "db": -0.11,
        "direction": "밝아짐(ΔL=+2.7)"
      }
    },
    // ... Zone B, A
  ],
  "analysis": {
    "profile": { /* ... */ },
    "derivatives": { /* ... */ },
    "boundary_candidates": [ /* ... */ ],
    "lens_info": { /* ... */ },
    "debug": {
      "B_candidate_range": [0.33, 0.66],
      "B_selected_range": [0.38, 0.61],
      "used_fallback_B": true,
      "retake_reasons": null,  // RETAKE일 때만 채워짐
      "ok_context": [          // OK일 때만 채워짐
        "Zone 여유도: A:ΔE=2.97(여유 63%), ...",
        "신뢰도: 0.87 (재촬영 불필요)"
      ]
    }
  }
}
```

### RETAKE 예시
```json
{
  "judgment": "RETAKE",
  "ng_reasons": [
    "R4_UniformityLow: 각도 불균일 높음 (std_L=10.2)"
  ],
  "debug": {
    "retake_reasons": [
      {
        "code": "R4_UniformityLow",
        "reason": "각도 불균일 높음 (std_L=10.2)",
        "actions": [
          "반사 감소 (조명 각도)",
          "렌즈 표면 이물질 제거",
          "재촬영"
        ],
        "lever": "촬영"
      }
    ]
  }
}
```

---

## 🔬 테스트 방법

### 1. 정상 OK 케이스
```bash
python src/web/app.py
# 브라우저: http://localhost:8000
# 업로드: 정상 품질 이미지
```

**확인 사항:**
- ✅ Zone별 `diff.direction` 표시
- ✅ `ok_context`에 여유도/신뢰도 메시지
- ✅ Soft warning (있으면)

### 2. RETAKE 케이스
**테스트 조건:**
- 반사가 심한 이미지 (R4_UniformityLow)
- 렌즈 중심이 벗어난 이미지 (R1_DetectionLow)
- 흐린 이미지 (R2_CoverageLow)

**확인 사항:**
- ✅ `judgment: "RETAKE"`
- ✅ `retake_reasons`에 code/reason/actions/lever
- ✅ Raw Data 탭에서 JSON 확인

### 3. NG 케이스
**테스트 조건:**
- 색차가 큰 불량 이미지

**확인 사항:**
- ✅ `judgment: "NG"`
- ✅ `diff.direction`으로 어떤 방향으로 불량인지 확인

---

## ✅ P1 개선 완료 (2025-12-13 오후)

**핵심 이슈 해결:** "Zone은 OK인데 결과는 RETAKE" → 사용자 혼란 해소

### 1. Judgment 구조 개선 (decision_trace, next_actions 승격)

**변경 내용:**
- RETAKE/NG 이유와 조치를 `debug`가 아닌 **최상위 필드**로 승격
- `decision_trace`: 판정 과정 추적 (final, because, overrides)
- `next_actions`: 권장 조치 목록 (즉시 실행 가능)
- `retake_reasons`: RETAKE 상세 사유 (1급 시민)

**JSON 응답 구조:**
```json
{
  "judgment": {
    "result": "RETAKE",
    "decision_trace": {
      "final": "RETAKE",
      "because": ["R4_UniformityLow"],
      "overrides": "zones_all_ok"  // Zone OK → RETAKE override 명시
    },
    "next_actions": [
      "반사 감소 (조명 각도)",
      "렌즈 표면 이물질 제거",
      "재촬영"
    ],
    "retake_reasons": [
      {
        "code": "R4_UniformityLow",
        "reason": "각도 불균일 높음 (std_L=12.5)",
        "actions": [...],
        "lever": "촬영"
      }
    ]
  }
}
```

**효과:**
- 운영자가 debug 없이 **판정 이유와 조치를 즉시 확인** 가능
- "왜 RETAKE인지" 명확히 설명 (overrides 필드)

---

### 2. OK_WITH_WARNING 상태 추가 (히스테리시스)

**변경 내용:**
- 판정 순서: **RETAKE > NG > OK_WITH_WARNING > OK**
- R4_UniformityLow 히스테리시스:
  - std_L > **12.0** → RETAKE (기존 10.0에서 완화)
  - **10.0 < std_L ≤ 12.0** → OK_WITH_WARNING
- 추가 경고 조건:
  - B zone fallback + confidence < 0.85
  - Confidence 0.6~0.7 경계값

**효과:**
- 오탐 감소: 경계값 근처는 경고로 처리
- 예방적 조치 가능: OK_WITH_WARNING 시 모니터링 권장

---

### 3. Diff Summary 실제 적용

**변경 내용:**
- Zone별 결과에 `diff` 필드 추가 (dL, da, db, direction)
- "어두워짐", "황색화" 등 **사람이 이해하기 쉬운 방향성** 제공
- `ZoneResult` dataclass에 diff 필드 추가
- API 응답의 `judgment.zones[].diff`에 포함

**예시:**
```json
{
  "zones": [
    {
      "name": "C",
      "delta_e": 2.7,
      "threshold": 10.0,
      "is_ok": true,
      "diff": {
        "dL": 2.69,
        "da": -0.35,
        "db": -0.11,
        "direction": "밝아짐(ΔL=+2.7)"
      }
    }
  ]
}
```

**효과:**
- 공정 담당자가 **어느 방향으로 조정**해야 하는지 즉시 파악
- NG/OK_WITH_WARNING 시 구체적인 색상 변화 확인 가능

---

### 4. 코드 위치

| 파일 | 변경 내용 | Line |
|------|----------|------|
| `color_evaluator.py` | ZoneResult에 diff 필드 추가 | 78 |
| `color_evaluator.py` | InspectionResult에 decision_trace, next_actions, retake_reasons 추가 | 108-110 |
| `zone_analyzer_2d.py` | R4 히스테리시스 (12.0), OK_WITH_WARNING 로직 | 1161-1216 |
| `zone_analyzer_2d.py` | decision_trace 생성 | 1218-1253 |
| `zone_analyzer_2d.py` | ZoneResult에 diff 포함 | 1099 |
| `app.py` | judgment_result에 decision_trace, next_actions, retake_reasons 포함 | 678-681 |
| `app.py` | zones에 diff 포함 | 675 |

---

## ✅ Step 1 완료 (2025-12-13 오후) - 데이터→판정 번역 레이어

**핵심 이슈 해결:** "프로파일 데이터가 풍부하지만 결론에 기여한 정도를 알 수 없음"

### 1. confidence_breakdown (Confidence 분해)

**목적:** "Confidence=0.65인데 왜 0.65인지" 설명

**구조:**
```json
{
  "confidence_breakdown": {
    "overall": 0.85,
    "factors": [
      {
        "name": "pixel_count",
        "weight": 0.30,
        "score": 0.95,
        "contribution": 0.285,
        "status": "good",
        "description": "Zone별 픽셀 수 충분도"
      },
      {
        "name": "uniformity",
        "weight": 0.25,
        "score": 0.70,
        "contribution": 0.175,
        "status": "warning",
        "description": "각도 균일성 (std_L 낮을수록 좋음)"
      }
      // ... 5개 factors 총합 = overall
    ]
  }
}
```

**5개 요소 (가중치):**
- pixel_count (30%): Zone별 픽셀 수
- transition (25%): 전이 구간 제거 정도
- uniformity (25%): 각도 균일성 (std_L)
- sector_uniformity (10%): 섹터 간 균일성
- lens_detection (10%): 렌즈 검출 신뢰도

**효과:**
- "어떤 요소가 Confidence를 깎았는지" 한눈에 파악
- status: good/warning/poor로 즉시 판단 가능

---

### 2. analysis_summary (프로파일 데이터 핵심 요약)

**목적:** "어디가 여유 있고, 어디가 위험했는지" 요약

**구조:**
```json
{
  "analysis_summary": {
    "uniformity": {
      "max_std_L": 11.2,
      "threshold_retake": 12.0,
      "threshold_warning": 10.0,
      "status": "warning",
      "impact": "OK_WITH_WARNING 트리거"
    },
    "boundary_quality": {
      "B_zone_method": "fallback",
      "confidence_contribution": 0.75,
      "status": "warning",
      "impact": "Confidence 페널티"
    },
    "coverage": {
      "total_pixels": 15234,
      "expected_min": 12000,
      "status": "good",
      "impact": "정상"
    }
  }
}
```

**3개 카테고리:**
- uniformity: 각도 균일성 (std_L)
- boundary_quality: Zone B 경계 탐지 품질
- coverage: 픽셀 커버리지

**효과:**
- 프로파일 데이터 핵심만 추출
- "이 특징이 어떤 판정에 기여했는지" 명시 (impact 필드)

---

### 3. risk_factors (위험 요소 목록)

**목적:** "무엇이 위험한지" 우선순위 리스트

**구조:**
```json
{
  "risk_factors": [
    {
      "category": "uniformity",
      "severity": "medium",
      "message": "각도 불균일 경계값 근접 (std_L=11.2, 경고=10.0)",
      "source": "OK_WITH_WARNING 조건"
    },
    {
      "category": "boundary",
      "severity": "medium",
      "message": "Zone B 경계 자동 탐지 실패 (fallback 사용)",
      "source": "경고"
    }
  ]
}
```

**카테고리:**
- uniformity: 각도 균일성 위험
- boundary: 경계 탐지 위험
- lens_detection: 렌즈 검출 위험
- coverage: 픽셀 커버리지 위험
- zone_quality: Zone ΔE 초과

**severity 레벨:**
- high: 즉시 조치 필요 (RETAKE/NG 트리거)
- medium: 주의 필요 (OK_WITH_WARNING)
- low: 모니터링 (정보성)

**효과:**
- 위험 요소를 severity별로 정렬하여 우선순위 파악
- source 필드로 어떤 판정 로직과 연결되는지 추적 가능

---

### 4. 전체 JSON 응답 구조 (완성)

```json
{
  "run_id": "abc123",
  "judgment": {
    "result": "OK_WITH_WARNING",
    "overall_delta_e": 4.5,
    "confidence": 0.85,

    // P1: 판정 추적
    "decision_trace": {
      "final": "OK_WITH_WARNING",
      "because": ["각도 균일성 경계값 근접 (std_L=11.2)"],
      "overrides": null
    },

    // P1: 권장 조치
    "next_actions": [
      "경고 사항 모니터링",
      "조명 균일성 확인"
    ],

    // P1: RETAKE 상세 (RETAKE일 때만)
    "retake_reasons": null,

    // ✨ Step 1: Confidence 분해
    "confidence_breakdown": {
      "overall": 0.85,
      "factors": [...]
    },

    // ✨ Step 1: 프로파일 요약
    "analysis_summary": {
      "uniformity": {...},
      "boundary_quality": {...},
      "coverage": {...}
    },

    // ✨ Step 1: 위험 요소
    "risk_factors": [
      {
        "category": "uniformity",
        "severity": "medium",
        "message": "각도 불균일 경계값 근접 (std_L=11.2)",
        "source": "OK_WITH_WARNING 조건"
      }
    ],

    // Zone 결과 (P1: Diff 포함)
    "zones": [...]
  },

  // 분석 데이터 (감사용, 별도 탭)
  "analysis": {...}
}
```

---

### 5. 코드 위치

| 파일 | 변경 내용 | Line |
|------|----------|------|
| `color_evaluator.py` | InspectionResult에 analysis_summary, confidence_breakdown, risk_factors 추가 | 116-118 |
| `zone_analyzer_2d.py` | confidence_breakdown 생성 (5개 요소) | 1256-1301 |
| `zone_analyzer_2d.py` | analysis_summary 생성 (3개 카테고리) | 1303-1324 |
| `zone_analyzer_2d.py` | risk_factors 생성 (5개 카테고리) | 1326-1384 |
| `app.py` | judgment_result에 3개 필드 포함 | 683-685 |

---

## ✅ 잉크 색 도출 기능 추가 (2025-12-13 오후)

**사용자 목표:** 사용된 잉크의 수와 각 잉크의 색을 도출

### 현황

**이미 측정하고 있었습니다!** ✅
- Zone 개수 검출 (transition 기반, expected_zones 힌트)
- 각 Zone의 Lab 값 측정
- 기준값과 비교 (ΔE)

**Zone = 잉크 영역**이 맞습니다.

---

### ink_analysis 필드 추가

**목적:** "잉크" 관점의 명시적 표현

**구조:**
```json
{
  "ink_analysis": {
    "detected_ink_count": 3,
    "detection_method": "transition_based",  // or "fallback"
    "expected_ink_count": 3,  // SKU 설정값
    "inks": [
      {
        "ink_number": 1,
        "zone_name": "C",
        "position": "inner",  // inner, middle, outer
        "radial_range": [0.0, 0.33],  // normalized radius
        "measured_color": {
          "L": 95.0,
          "a": 0.5,
          "b": 2.0
        },
        "reference_color": {
          "L": 95.0,
          "a": 0.5,
          "b": 2.0
        },
        "delta_e": 2.7,
        "is_within_spec": true,
        "pixel_count": 5234
      },
      {
        "ink_number": 2,
        "zone_name": "B",
        "position": "middle",
        "radial_range": [0.38, 0.61],
        "measured_color": {
          "L": 68.0,
          "a": 5.0,
          "b": 22.0
        },
        "reference_color": {...},
        "delta_e": 7.65,
        "is_within_spec": true,
        "pixel_count": 6123
      },
      {
        "ink_number": 3,
        "zone_name": "A",
        "position": "outer",
        "radial_range": [0.66, 1.0],
        "measured_color": {
          "L": 45.0,
          "a": 8.0,
          "b": 28.0
        },
        "reference_color": {...},
        "delta_e": 2.97,
        "is_within_spec": true,
        "pixel_count": 8412
      }
    ]
  }
}
```

---

### 잉크 번호 매기기 방식

**중심→바깥 순서:**
- **Ink_1**: Zone C (inner, 중심)
- **Ink_2**: Zone B (middle, 중간)
- **Ink_3**: Zone A (outer, 바깥)

**이유:** 콘택트렌즈 프린팅은 일반적으로 중심에서 바깥으로 진행

---

### 주요 정보

| 필드 | 설명 |
|------|------|
| `detected_ink_count` | 자동 검출된 잉크 수 |
| `detection_method` | transition_based (자동) or fallback (힌트 사용) |
| `expected_ink_count` | SKU params.expected_zones 설정값 |
| `ink_number` | 중심→바깥 순 번호 (1, 2, 3...) |
| `position` | inner/middle/outer |
| `radial_range` | 정규화된 반경 범위 [r_start, r_end] |
| `measured_color` | 측정된 잉크 색 (Lab) |
| `reference_color` | 기준 잉크 색 (Lab) |
| `delta_e` | 색차 (CIEDE2000) |
| `is_within_spec` | 허용 범위 내 여부 |
| `pixel_count` | 해당 잉크 영역 픽셀 수 |

---

### 코드 위치

| 파일 | 변경 내용 | Line |
|------|----------|------|
| `color_evaluator.py` | InspectionResult에 ink_analysis 추가 | 119 |
| `zone_analyzer_2d.py` | ink_analysis 생성 (잉크 정보 추출) | 1386-1423 |
| `app.py` | judgment_result에 ink_analysis 포함 | 687 |

---

## 📈 향후 개선 계획 (Priority 2)

### 2.1 B Zone 안정성 점수 (2일)
**목표:** B zone 경계 선택의 신뢰도를 정량화

**방법:**
- Peak prominence (전이 명확도)
- Smoothing invariance (파라미터 강건성)
- Sector consistency (각도 일관성)

**효과:** R3_BoundaryUncertain의 정확도 향상

---

### 2.2 Sector별 ΔE 분포 판정 (3일)
**목표:** 부분 불량 검출 강화

**방법:**
- Ring × Sector 2D 분석 활성화
- Percentile 기반 판정 (p90, outlier_ratio)

**효과:**
- "평균은 OK인데 특정 섹터만 NG" 검출
- 도트 인쇄 국부 결함 감지

---

## 📝 변경 이력

| 날짜 | 내용 | 담당 |
|------|------|------|
| 2025-12-13 (오후) | **잉크 색 도출**: ink_analysis 필드 추가 (사용자 목표 달성) | Claude |
| 2025-12-13 (오후) | **Step 1 완료**: confidence_breakdown, analysis_summary, risk_factors 추가 (데이터→판정 번역 레이어) | Claude |
| 2025-12-13 (오후) | **P1 개선 완료**: decision_trace 승격, OK_WITH_WARNING 추가, Diff Summary 적용 | Claude |
| 2025-12-13 (오전) | Phase 1 Quick Wins 구현 완료 (4개 항목) | Claude |
| 2025-12-13 | 버그 수정: used_fallback_B 변수 순서 문제 | Claude |

---

## 🔗 관련 문서

- **[PHASE7_CORE_IMPROVEMENTS.md](PHASE7_CORE_IMPROVEMENTS.md)** - 핵심 품질 개선 계획
- **[ANALYSIS_UI_DEVELOPMENT_PLAN.md](ANALYSIS_UI_DEVELOPMENT_PLAN.md)** - 분석 UI 개발 계획
- **코드 위치:**
  - `src/core/color_evaluator.py:25-53` - describe_color_shift()
  - `src/core/zone_analyzer_2d.py:922-935` - Diff 계산
  - `src/core/zone_analyzer_2d.py:1125-1195` - RETAKE + OK context

---

**➡ 참고:** 이 문서는 실제 구현 결과를 기반으로 작성되었으며, 코드와 함께 유지보수됩니다.
