# STD vs Sample 비교 UI 가이드

## 목적
STD(Reference)와 Sample 비교 결과를 UI에서 읽는 방법과 출력 항목의 의미를 정리한다.

## 사용 흐름
1. `/compare` 화면 접속
2. STD 이미지 1장 + Sample 이미지 1장 업로드
3. `분석 실행` 클릭
4. 결과 확인 + 필요 시 `JSON 다운로드`

## 결과 항목 요약
- `mean/max Delta E`: 전체 평균/최대 색상 편차
- `Delta L/Delta a/Delta b`: 명도/적-녹/황-청 방향의 편차
- `overall shift`: 설명용 요약(판정에 사용하지 않음)
- `zone_deltas`: Zone별 Delta E와 Delta L/a/b
- `ink_deltas`: 잉크 단위 통계(ink_mapping 필요)
- `ink_flags`: 잉크 단위 경고/불합격 플래그(ink_thresholds 필요)
- `metrics`: 측정 지표(blur/histogram/dot_stats)
- `comparison`: STD 대비 지표(blur_delta/hist_diff/dot_delta)
- `quality_thresholds`: 품질 지표 경고 기준(quality_thresholds)

## 편차 해석 규칙(부호 기준)
- `Delta L`:
  - 음수: 어두워짐(Darker)
  - 양수: 밝아짐(Lighter)
- `Delta a`:
  - 음수: 녹색 방향(Green shift)
  - 양수: 적색 방향(Red shift)
- `Delta b`:
  - 음수: 청색 방향(Blue shift)
  - 양수: 황색 방향(Yellow shift)
- `overall shift`는 설명용 지표이며 판정/점수 계산에는 사용하지 않는다.
- `stable` 기준: |Delta L| < 1.0 AND |Delta a| < 1.0 AND |Delta b| < 1.0

## Ink 기반 결과
### ink_mapping
SKU 설정의 `params.ink_mapping`으로 Zone → Ink 매핑을 정의한다.

예시:
```json
"params": {
  "ink_mapping": {
    "A": "ink1",
    "B": "ink1",
    "C": "ink2"
  }
}
```

### ink_thresholds
SKU 설정의 `params.ink_thresholds`로 잉크별 기준을 정의한다.

예시:
```json
"params": {
  "ink_thresholds": {
    "default": { "max_delta_e": 8.0 },
    "ink1": { "max_delta_e": 8.0 },
    "ink2": { "max_delta_e": 10.0 }
  }
}
```

### ink_flags
`ink_thresholds`와 비교해 잉크 단위 경고/불합격을 표시한다.

## 품질 지표(blur / histogram / dot_stats)
### metrics
- `blur.score`: Laplacian variance 기반 선명도 지표
- `histogram`: Lab/HSV 채널별 히스토그램(정규화)
- `dot_stats`: dot_count, dot_coverage, dot_area_mean/std 등

### comparison
- `blur_delta`: STD 대비 blur 변화량
- `hist_diff`: 히스토그램 차이(l1 기준)
- `dot_delta`: dot_count/coverage 등 변화량

### quality_thresholds
SKU 설정의 `params.quality_thresholds`로 품질 경고 기준을 정의한다.

예시:
```json
"params": {
  "quality_thresholds": {
    "blur": { "delta_warn": -50.0 },
    "histogram": { "lab_mean": 0.2, "hsv_mean": 0.2 },
    "dot_stats": { "dot_count_delta": 50, "dot_coverage_delta": 0.05 }
  }
}
```

## 정형 텍스트 템플릿(시스템 출력용)
아래 템플릿을 그대로 채워서 로그/리포트로 사용한다.

```
📊 상세 편차 및 디자인 양상 분석

1) Overall Summary
- 최대 Delta E: {max_delta_e} (Zone {max_zone})
- Overall Shift: {overall_shift}
- 판정 요약: {summary}

2) Zone별 상세 편차 분석
Zone {zone_name}
- Delta E: {delta_e}
- Delta L: {delta_l}  -> {l_shift}
- Delta a: {delta_a}  -> {a_shift}
- Delta b: {delta_b}  -> {b_shift}
- 해석: {zone_interpretation}

3) Ink 기반 편차 분석
Ink {ink_name}
- Mean Delta E: {mean_delta_e}
- Max Delta E: {max_delta_e}
- Mean Delta L/a/b: {mean_delta_l}/{mean_delta_a}/{mean_delta_b}
- 해석: {ink_interpretation}

4) Design Pattern Diagnosis
- Primary: {primary_defect}
- Secondary: {secondary_defect}
- Impact: {design_impact}

5) Decision Trace
- {decision_trace_1}
- {decision_trace_2}
- {decision_trace_3}

6) Next Actions
- Immediate: {action_immediate}
- Root Cause: {action_root_cause}
- Corrective: {action_corrective}
- System: {action_system}
```

### 템플릿 내 부호 변환 규칙
- `Delta L` 음수: Darker / 양수: Lighter
- `Delta a` 음수: Green shift / 양수: Red shift
- `Delta b` 음수: Blue shift / 양수: Yellow shift
