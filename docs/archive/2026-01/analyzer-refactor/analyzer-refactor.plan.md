# analyzer.py 리팩토링 계획서

> **Summary**: 1,437줄 모놀리식 analyzer.py를 책임별 모듈로 분할
>
> **Project**: Color Meter
> **Author**: PDCA Auto
> **Date**: 2026-01-31
> **Status**: Draft

---

## 1. Overview

### 1.1 Purpose

`src/engine_v7/core/pipeline/analyzer.py` (1,437줄)는 검사(INSPECTION)·등록(STD_REGISTRATION)·색상별 평가(PER_COLOR)의 3가지 평가 로직을 단일 파일에 담고 있다. 함수 간 코드 중복이 심하고, 변경 시 의도치 않은 영향 범위가 넓다. 책임별 모듈로 분할하여 유지보수성을 개선한다.

### 1.2 Problem Analysis

#### 코드 중복

| 중복 패턴 | 위치 (evaluate / evaluate_multi) | 줄 수 |
|-----------|--------------------------------|-------|
| Gate 체크 + 조기 반환 | 502-554 / 720-782 | ~65줄 × 2 |
| Anomaly 평가 | 566-592 / 801-827 | ~27줄 × 2 |
| Diagnostics 생성 + references | 596-628 / 831-865 | ~33줄 × 2 |
| Heatmap + defect classify | 634-644 / 872-882 | ~11줄 × 2 |
| v2/v3 attach + qc_decision | 659-706 / 900-948 | ~48줄 × 2 |

**중복 합계**: 약 370줄 (전체의 ~26%)

#### 파일 복잡도

| 메트릭 | 값 |
|--------|-----|
| 총 줄 수 | 1,437 |
| 공개 함수 | 4 (evaluate, evaluate_multi, evaluate_registration_multi, evaluate_per_color) |
| 비공개 함수 | 14 |
| Import 수 | 33 |

### 1.3 Related Documents

- `docs/02-design/features/phase5-diagnostics.design.md` - v2_diagnostics 캐싱 설계
- `docs/engine_v7/INSPECTION_FLOW.md` - 검사 흐름 다이어그램

---

## 2. Scope

### 2.1 In Scope

- [ ] 공통 로직을 헬퍼 모듈로 추출
- [ ] evaluate()와 evaluate_multi()의 중복 제거
- [ ] 등록 검증 로직을 별도 모듈로 분리
- [ ] 색상별 평가 로직을 별도 모듈로 분리
- [ ] 외부 import 경로 호환성 유지
- [ ] 기능 동작 무변경 (pure refactoring)

### 2.2 Out of Scope

- 새 기능 추가
- evaluate() / evaluate_multi() API 시그니처 변경
- v7.py 라우터 분할 (별도 PDCA)
- inspection_service.py 분할 (별도 PDCA)

---

## 3. Design Concept

### 3.1 목표 구조

```
core/pipeline/
├── __init__.py                    # 공개 API re-export
├── analyzer.py                    # 축소: evaluate(), evaluate_multi() (진입점)
├── _common.py                     # 공통 헬퍼 (NEW)
│   ├── _reason_meta()
│   ├── _maybe_apply_white_balance()
│   ├── _gate_early_return()       # Gate 체크 + 조기 반환 패턴
│   ├── _evaluate_anomaly()        # Anomaly 평가 패턴
│   ├── _build_diagnostics_block() # diagnostics + references 패턴
│   ├── _attach_heatmap_defect()   # Heatmap + defect classify 패턴
│   └── _finalize_decision()       # v2/v3 attach + qc_decision 패턴
├── _signature.py                  # 시그니처 관련 (NEW)
│   ├── _build_k_by_segment()
│   ├── _evaluate_signature()
│   └── _pick_best_mode()
├── _diagnostics.py                # 진단 관련 (NEW)
│   ├── _mean_grad()
│   ├── _compute_worst_case()
│   ├── _compute_diagnostics()
│   ├── _attach_v2_diagnostics()
│   ├── _attach_v3_summary()
│   ├── _attach_v3_trend()
│   └── _append_ok_features()
├── registration.py                # 등록 검증 (NEW)
│   ├── _registration_summary()
│   └── evaluate_registration_multi()
├── per_color.py                   # 색상별 평가 (NEW)
│   └── evaluate_per_color()
├── single_analyzer.py             # 기존 유지
└── feature_export.py              # 기존 유지
```

### 3.2 예상 줄 수

| 모듈 | 예상 줄 수 | 원본 줄 수 |
|------|-----------|-----------|
| analyzer.py (축소) | ~350 | 1,437 |
| _common.py | ~200 | (추출) |
| _signature.py | ~110 | (추출) |
| _diagnostics.py | ~250 | (추출) |
| registration.py | ~200 | (추출) |
| per_color.py | ~300 | (추출) |
| **합계** | **~1,410** | 1,437 |

### 3.3 외부 호환성

현재 외부 import:
```python
# v7.py
from src.engine_v7.core.pipeline import analyzer as analyzer_mod
from src.engine_v7.core.pipeline.analyzer import _registration_summary, evaluate_multi, evaluate_registration_multi

# api.py
from src.engine_v7.core.pipeline.analyzer import evaluate_multi, evaluate_per_color

# tests
from core.pipeline.analyzer import evaluate
from core.pipeline.analyzer import evaluate_per_color
```

**호환성 전략**: `analyzer.py`에서 분리된 함수들을 re-export하여 기존 import 경로 유지.
```python
# analyzer.py (축소 후)
from ._common import ...
from ._signature import ...
from ._diagnostics import _attach_v2_diagnostics, _append_ok_features
from .registration import evaluate_registration_multi, _registration_summary
from .per_color import evaluate_per_color
```

---

## 4. Requirements

### 4.1 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | 공통 Gate 패턴을 _common.py로 추출 | High |
| FR-02 | Signature 헬퍼를 _signature.py로 추출 | High |
| FR-03 | Diagnostics 로직을 _diagnostics.py로 추출 | High |
| FR-04 | Registration 로직을 registration.py로 분리 | High |
| FR-05 | Per-color 로직을 per_color.py로 분리 | High |
| FR-06 | evaluate()와 evaluate_multi() 중복 제거 | Medium |
| FR-07 | 외부 import 호환성 re-export | High |

### 4.2 Non-Functional Requirements

| Category | Criteria | Measurement |
|----------|----------|-------------|
| 호환성 | 기존 import 경로 100% 유지 | grep 검증 |
| 동작 무변경 | 동일 입력 → 동일 출력 | smoke test |
| 줄 수 감소 | analyzer.py 400줄 이하 | wc -l |

---

## 5. Success Criteria

### 5.1 Definition of Done

- [ ] analyzer.py 400줄 이하로 축소
- [ ] 새 모듈 5개 생성 (_common, _signature, _diagnostics, registration, per_color)
- [ ] 기존 import 경로 모두 동작 (grep 검증)
- [ ] 코드 중복 370줄 → 0줄 (공통 함수 호출)
- [ ] python -c "import ast; ast.parse(...)" 통과
- [ ] 기존 테스트 통과 (test_backward_compat.py, test_per_color_e2e.py)

### 5.2 Quality Criteria

- [ ] 각 모듈이 단일 책임 원칙 준수
- [ ] 순환 import 없음

---

## 6. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| 순환 import | High | Medium | _common.py는 core 모듈만 import, 역방향 금지 |
| 외부 import 깨짐 | High | Low | analyzer.py에서 re-export + grep 검증 |
| 동작 변경 (regression) | High | Low | 순수 추출 (코드 변경 없이 이동만) |
| evaluate/evaluate_multi 공통화 시 미묘한 차이 놓침 | Medium | Medium | diff 비교 후 공통화 |

---

## 7. Next Steps

1. [ ] Design 문서 작성 (`/pdca design analyzer-refactor`)
2. [ ] 구현 (모듈 분할)
3. [ ] Gap analysis
4. [ ] 완료 보고서

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-01-31 | Initial draft | PDCA Auto |
