# Phase 5 Diagnostics Analysis Report

> **Analysis Type**: Gap Analysis (Design vs Implementation)
>
> **Project**: Color Meter
> **Analyst**: PDCA Auto
> **Date**: 2026-01-30
> **Design Doc**: [phase5-diagnostics.design.md](../02-design/features/phase5-diagnostics.design.md)

---

## 1. Analysis Overview

### 1.1 Analysis Purpose

v2_diagnostics Engine B 통합 설계서(Phase 5)에 명시된 12개 점검 항목에 대해 실제 구현 코드와의 일치율을 검증한다.

### 1.2 Analysis Scope

- **Design Document**: `docs/02-design/features/phase5-diagnostics.design.md`
- **Implementation Files**:
  - `src/engine_v7/core/measure/diagnostics/v2_diagnostics.py`
  - `src/engine_v7/core/pipeline/analyzer.py`
- **Analysis Date**: 2026-01-30

---

## 2. Gap Analysis (Design vs Implementation)

### 2.1 Per-Item Comparison

| # | Check Item | Design Section | Match | Status |
|:-:|:-----------|:--------------:|:-----:|:------:|
| 1 | `build_v2_diagnostics()` keyword-only params 3개 추가 | 3.1 | YES | MATCH |
| 2 | geom = precomputed_geom or detect_lens_circle(bgr) | 3.1 | YES | MATCH |
| 3 | precomputed_masks 사용 시 build_color_masks_with_retry 스킵 | 3.1 | YES | MATCH |
| 4 | _polar 변수로 to_polar() 1회 통합 | 3.1 | YES | MATCH |
| 5 | `_attach_v2_diagnostics()` 시그니처에 cached_geom, cached_masks, cached_polar 추가 | 3.2 | PARTIAL | GAP |
| 6 | build_v2_diagnostics() 호출에 precomputed_geom, precomputed_masks, precomputed_polar 전달 | 3.2 | PARTIAL | GAP |
| 7 | evaluate() 호출부에서 cached_geom=geom, cached_polar=polar 전달 | 3.3.1 | YES | MATCH |
| 8 | evaluate_multi() 호출부에서 cached_geom=geom, cached_polar=polar 전달 | 3.3.2 | YES | MATCH |
| 9 | evaluate_per_color() 에서 cached_masks=(color_masks, mask_metadata) 전달 | 3.3.3 | NO | GAP |
| 10 | v2_diagnostics 내부 to_polar() 2회 -> 1회 통합 | 3.4 | YES | MATCH |
| 11 | 변경 파일 목록 일치 | 4 | YES | MATCH |
| 12 | 하위 호환성 유지 (precomputed=None 시 기존 동작) | 6 | YES | MATCH |

### 2.2 Match Rate Summary

```
+---------------------------------------------+
|  Overall Match Rate: 83% (10/12)            |
+---------------------------------------------+
|  MATCH:          9 items  (75%)             |
|  PARTIAL:        2 items  (17%)             |
|  GAP:            1 item   ( 8%)             |
+---------------------------------------------+
```

---

## 3. Gap Detail

### 3.1 Item 5 -- `_attach_v2_diagnostics()` 시그니처 불완전

**설계** (`phase5-diagnostics.design.md` Section 3.2):
```python
cached_geom=None,
cached_masks: Optional[Tuple] = None,
cached_polar: Optional[np.ndarray] = None,
```

**구현** (`analyzer.py`):
```python
cached_geom=None,
cached_polar: np.ndarray | None = None,
```

**차이**: `cached_masks: Optional[Tuple] = None` 파라미터가 누락됨. 3개 중 2개만 구현.

**영향도**: Medium

---

### 3.2 Item 6 -- build_v2_diagnostics() 호출 시 precomputed_masks 미전달

**설계** (`phase5-diagnostics.design.md` Section 3.2):
```python
precomputed_geom=cached_geom,
precomputed_masks=cached_masks,
precomputed_polar=cached_polar,
```

**구현** (`analyzer.py`):
```python
precomputed_geom=cached_geom,
precomputed_polar=cached_polar,
```

**차이**: `precomputed_masks=cached_masks` 전달 누락. Item 5의 직접적 결과.

**영향도**: Medium

---

### 3.3 Item 9 -- evaluate_per_color()에서 _attach_v2_diagnostics 미호출

**설계** (`phase5-diagnostics.design.md` Section 3.3.3):
```python
if mode != "signature":
    _attach_v2_diagnostics(
        test_bgr, dec, cfg, ok_log_context,
        cached_geom=geom,
        cached_masks=(color_masks, mask_metadata),
    )
```

**구현**: `evaluate_per_color()` 함수 내에서 `_attach_v2_diagnostics()`를 호출하지 않음.

**차이**: 설계서에서 "High" 심각도로 식별한 `build_color_masks_with_retry()` 2회 호출 중복이 해소되지 않은 상태.

**영향도**: High

---

## 4. Root Cause Analysis

Items 5, 6, 9는 연쇄 관계:

```
Item 5: cached_masks 파라미터 미추가
   -> Item 6: masks 전달 불가
      -> Item 9: evaluate_per_color()에서 v2_diagnostics 연결 불가
```

---

## 5. Overall Score

| Category | Score | Status |
|:---------|:-----:|:------:|
| Design Match | 83% | Partial Gap |
| Architecture Compliance | 100% | PASS |
| Backward Compatibility | 100% | PASS |
| **Overall** | **83%** | **Needs Action** |

---

## 6. Recommended Actions

| Priority | Item | File | Action |
|:--------:|:-----|:-----|:-------|
| 1 | `cached_masks` 파라미터 추가 | `analyzer.py` | `_attach_v2_diagnostics()` 시그니처에 `cached_masks: tuple \| None = None` 추가 |
| 2 | `precomputed_masks` 전달 | `analyzer.py` | `build_v2_diagnostics()` 호출에 `precomputed_masks=cached_masks` 추가 |
| 3 | `evaluate_per_color()` v2 연결 | `analyzer.py` | `_attach_v2_diagnostics()` 호출 추가 with cached_geom, cached_masks |

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-01-30 | Initial gap analysis | PDCA Auto |
