# Phase 5: v2_diagnostics Engine B 통합 설계서

> **Summary**: v2_diagnostics와 analyzer.py 간 색상 마스크 중복 계산 제거 및 Engine B SSoT 확립
>
> **Project**: Color Meter
> **Author**: PDCA Auto
> **Date**: 2026-01-30
> **Status**: Draft
> **Plan Reference**: `docs/01-plan/features/phase5-diagnostics.plan.md`

---

## 1. 현재 상태 분석

### 1.1 호출 흐름 (AS-IS)

```
evaluate() / evaluate_multi()
  ├── detect_lens_circle(test_bgr)                    [1회]
  ├── to_polar(test_bgr, geom)                        [1회]
  ├── build_radial_signature(polar)                    [1회]
  ├── _evaluate_signature()                            [1회]
  ├── score_anomaly / score_anomaly_relative           [1회]
  ├── decide(gate, sig, anom)                          [1회]
  └── _attach_v2_diagnostics(test_bgr, decision, cfg)
        └── build_v2_diagnostics(bgr, cfg, ...)
              ├── detect_lens_circle(bgr)              [중복! 2회차]
              ├── build_color_masks_with_retry(bgr)    [v2 전용 호출]
              ├── to_polar(bgr, geom)                  [중복! 2회차]
              ├── to_cie_lab(polar)                     [v2 전용]
              ├── build_roi_mask()                      [v2 전용]
              └── separation_ab(), pairwise_deltaE()   [v2 전용]

evaluate_per_color()
  ├── detect_lens_circle(test_bgr)                    [1회]
  ├── build_color_masks_with_retry(bgr)               [1회 - per_color 전용]
  ├── per-color signature evaluation                   [N회]
  └── _attach_v2_diagnostics(test_bgr, decision, cfg)
        └── build_v2_diagnostics(bgr, cfg, ...)
              ├── detect_lens_circle(bgr)              [중복! 2회차]
              ├── build_color_masks_with_retry(bgr)    [중복! 2회차]
              └── ...                                  [중복 계산]
```

### 1.2 중복 분석 결과

| 계산 항목 | evaluate() | evaluate_per_color() | 심각도 |
|----------|-----------|---------------------|--------|
| `detect_lens_circle()` | 2회 (evaluate + v2_diag) | 2회 | Medium |
| `build_color_masks_with_retry()` | 1회 (v2_diag만) | 2회 (per_color + v2_diag) | **High** |
| `to_polar()` | 2회 (evaluate + v2_diag) | 1+회 | Medium |
| `to_cie_lab()` | 2회 (diagnostics + v2_diag) | N+1회 | Low |

**핵심 발견**:
- `evaluate()`에서는 `build_color_masks_with_retry()`가 v2_diagnostics 내부에서만 호출됨 (중복 없음)
- `evaluate_per_color()`에서는 **라인 1196과 v2_diagnostics 내부에서 2회 호출** (진짜 중복)
- `detect_lens_circle()`과 `to_polar()`는 두 경로 모두에서 중복

### 1.3 기존 코드 특성

- `build_v2_diagnostics()`는 독립 실행 가능한 함수 (bgr 입력만으로 전체 진단 생성)
- `_attach_v2_diagnostics()`는 `build_v2_diagnostics()` + `ink_match` + `v2_flags`를 조합
- `evaluate_per_color()`는 자체적으로 `color_masks`를 사용하여 per-color 시그니처를 평가

---

## 2. 설계 목표

### 2.1 TO-BE 아키텍처

```
evaluate() / evaluate_multi()
  ├── detect_lens_circle(test_bgr)                    [1회]
  ├── to_polar(test_bgr, geom)                        [1회, 캐시]
  ├── (signature, anomaly, decision - 기존 유지)
  └── _attach_v2_diagnostics(test_bgr, decision, cfg,
        cached_geom=geom, cached_polar=polar)          [캐시 전달]
        └── build_v2_diagnostics(...,
              precomputed_geom=geom,
              precomputed_polar=polar)                  [중복 제거]

evaluate_per_color()
  ├── detect_lens_circle(test_bgr)                    [1회]
  ├── build_color_masks_with_retry(bgr)               [1회, 결과 보존]
  ├── per-color signature evaluation                   [기존 유지]
  └── _attach_v2_diagnostics(test_bgr, decision, cfg,
        cached_geom=geom,
        cached_masks=(color_masks, mask_metadata))     [캐시 전달]
        └── build_v2_diagnostics(...,
              precomputed_geom=geom,
              precomputed_masks=(color_masks, metadata)) [중복 제거]
```

### 2.2 설계 원칙

1. **하위 호환성 유지**: 기존 파라미터 인터페이스 유지, 새 파라미터는 Optional
2. **최소 변경**: 함수 시그니처에 `precomputed_*` 파라미터 추가만, 내부 로직 변경 최소화
3. **Fail-safe**: precomputed 값이 None이면 기존처럼 내부 계산 수행
4. **API 출력 불변**: JSON 응답 스키마 변경 없음

---

## 3. 상세 설계

### 3.1 `build_v2_diagnostics()` 인터페이스 변경

**파일**: `src/engine_v7/core/measure/diagnostics/v2_diagnostics.py`

```python
# AS-IS
def build_v2_diagnostics(
    bgr,
    cfg: Dict[str, Any],
    expected_ink_count: int | None,
    expected_ink_count_registry: int | None,
    expected_ink_count_input: int | None,
    polar_alpha: Optional[np.ndarray] = None,
    alpha_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any] | None:

# TO-BE (새 파라미터 추가)
def build_v2_diagnostics(
    bgr,
    cfg: Dict[str, Any],
    expected_ink_count: int | None,
    expected_ink_count_registry: int | None,
    expected_ink_count_input: int | None,
    polar_alpha: Optional[np.ndarray] = None,
    alpha_cfg: Optional[Dict[str, Any]] = None,
    *,  # keyword-only 구분자
    precomputed_geom=None,                              # NEW
    precomputed_masks: Optional[Tuple] = None,          # NEW: (color_masks, metadata)
    precomputed_polar: Optional[np.ndarray] = None,     # NEW
) -> Dict[str, Any] | None:
```

**내부 로직 변경**:

```python
# AS-IS (라인 39)
geom = detect_lens_circle(bgr)

# TO-BE
geom = precomputed_geom if precomputed_geom is not None else detect_lens_circle(bgr)
```

```python
# AS-IS (라인 40-46)
color_masks, metadata = build_color_masks_with_retry(
    bgr, cfg, expected_k=int(expected_ink_count), geom=geom, ...)

# TO-BE
if precomputed_masks is not None:
    color_masks, metadata = precomputed_masks
else:
    color_masks, metadata = build_color_masks_with_retry(
        bgr, cfg, expected_k=int(expected_ink_count), geom=geom, ...)
```

```python
# AS-IS (라인 54, 119)
polar = to_polar(bgr, geom, R=cfg["polar"]["R"], T=cfg["polar"]["T"])

# TO-BE
if precomputed_polar is not None:
    polar = precomputed_polar
else:
    polar = to_polar(bgr, geom, R=cfg["polar"]["R"], T=cfg["polar"]["T"])
```

> **주의**: `to_polar()` 호출이 함수 내 2곳 (라인 54, 119)에 존재. 두 곳 모두 동일한 `bgr, geom, R, T` 파라미터를 사용하므로, 함수 상단에서 1회 계산하여 내부 변수로 공유.

### 3.2 `_attach_v2_diagnostics()` 인터페이스 변경

**파일**: `src/engine_v7/core/pipeline/analyzer.py`

```python
# AS-IS (라인 373)
def _attach_v2_diagnostics(
    test_bgr,
    decision: Decision,
    cfg: Dict[str, Any],
    ok_log_context: Dict[str, Any] | None,
    polar_alpha: np.ndarray | None = None,
) -> None:

# TO-BE
def _attach_v2_diagnostics(
    test_bgr,
    decision: Decision,
    cfg: Dict[str, Any],
    ok_log_context: Dict[str, Any] | None,
    polar_alpha: np.ndarray | None = None,
    *,
    cached_geom=None,                                   # NEW
    cached_masks: Optional[Tuple] = None,               # NEW
    cached_polar: Optional[np.ndarray] = None,          # NEW
) -> None:
```

**내부 전달**:
```python
# AS-IS (라인 394)
diagnostics = build_v2_diagnostics(
    test_bgr, cfg,
    expected_ink_count=int(expected), ...)

# TO-BE
diagnostics = build_v2_diagnostics(
    test_bgr, cfg,
    expected_ink_count=int(expected), ...,
    precomputed_geom=cached_geom,
    precomputed_masks=cached_masks,
    precomputed_polar=cached_polar,
)
```

### 3.3 호출부 변경

#### 3.3.1 `evaluate()` (라인 652-653)

```python
# AS-IS
if mode != "signature":
    _attach_v2_diagnostics(test_bgr, decision, cfg, ok_log_context)

# TO-BE
if mode != "signature":
    _attach_v2_diagnostics(
        test_bgr, decision, cfg, ok_log_context,
        cached_geom=geom,
        cached_polar=polar,  # 라인 549에서 이미 계산됨
    )
```

#### 3.3.2 `evaluate_multi()` (라인 890-891)

```python
# AS-IS
if mode != "signature":
    _attach_v2_diagnostics(test_bgr, dec, cfg, ok_log_context)

# TO-BE
if mode != "signature":
    _attach_v2_diagnostics(
        test_bgr, dec, cfg, ok_log_context,
        cached_geom=geom,
        cached_polar=polar,  # 라인 774에서 이미 계산됨
    )
```

#### 3.3.3 `evaluate_per_color()` (v2 호출부 확인 필요)

```python
# evaluate_per_color 내부에서 이미 계산한 masks 전달
# (color_masks, mask_metadata)는 라인 1196에서 계산됨
if mode != "signature":
    _attach_v2_diagnostics(
        test_bgr, dec, cfg, ok_log_context,
        cached_geom=geom,
        cached_masks=(color_masks, mask_metadata),
    )
```

### 3.4 `build_v2_diagnostics()` 내부 `to_polar()` 중복 제거

현재 함수 내 `to_polar()`가 2곳에서 호출됨:
- **라인 54**: alpha_summary 계산용 (`if alpha_cfg is not None or polar_alpha is not None`)
- **라인 119**: ROI LAB mean 계산용

**설계**: 함수 상단에서 polar를 1회만 계산하고 재사용

```python
# TO-BE: 함수 초반에 polar 계산 통합
_polar = precomputed_polar
if _polar is None:
    _polar = to_polar(bgr, geom, R=cfg["polar"]["R"], T=cfg["polar"]["T"])

# 이후 라인 54, 119에서 _polar 재사용
```

---

## 4. 변경 파일 목록

| 파일 | 변경 유형 | 변경 내용 | 영향도 |
|------|----------|----------|--------|
| `src/engine_v7/core/measure/diagnostics/v2_diagnostics.py` | 수정 | precomputed_* 파라미터 추가, 내부 중복 제거 | Medium |
| `src/engine_v7/core/pipeline/analyzer.py` | 수정 | cached_* 파라미터 전달, 호출부 3곳 변경 | Medium |
| `docs/INTEGRATION_STATUS.md` | 수정 | Phase 5 상태 100% 업데이트 | Low |

### 변경하지 않는 파일

| 파일 | 이유 |
|------|------|
| `v2_flags.py` | Engine B 출력 형식과 이미 호환됨 (검증 완료) |
| `color_masks.py` | 변경 불필요 (호출측에서 캐시 전달) |
| `single_analyzer.py` | v2_diagnostics를 직접 호출하지 않음 |
| Web API 라우터 | 내부 함수 변경이므로 API 인터페이스 불변 |

---

## 5. 데이터 흐름도

```
┌─────────────────────────────────────────────────────────────┐
│                    evaluate() / evaluate_multi()             │
│                                                             │
│  ┌──────────────┐     ┌──────────┐     ┌──────────────┐    │
│  │detect_lens_  │     │to_polar()│     │build_radial_ │    │
│  │circle()      │────▶│          │────▶│signature()   │    │
│  │  → geom      │     │  → polar │     │              │    │
│  └──────┬───────┘     └────┬─────┘     └──────────────┘    │
│         │                  │                                │
│         │   cached_geom    │  cached_polar                  │
│         ▼                  ▼                                │
│  ┌─────────────────────────────────────┐                    │
│  │   _attach_v2_diagnostics()          │                    │
│  │     → build_v2_diagnostics(         │                    │
│  │         precomputed_geom=geom,      │                    │
│  │         precomputed_polar=polar)    │                    │
│  │       [중복 계산 SKIP]              │                    │
│  └─────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 evaluate_per_color()                         │
│                                                             │
│  ┌──────────────┐     ┌──────────────────────┐             │
│  │detect_lens_  │     │build_color_masks_    │             │
│  │circle()      │────▶│with_retry()          │             │
│  │  → geom      │     │  → (masks, metadata) │             │
│  └──────┬───────┘     └──────────┬───────────┘             │
│         │                        │                          │
│         │   cached_geom          │  cached_masks            │
│         ▼                        ▼                          │
│  ┌─────────────────────────────────────┐                    │
│  │   _attach_v2_diagnostics()          │                    │
│  │     → build_v2_diagnostics(         │                    │
│  │         precomputed_geom=geom,      │                    │
│  │         precomputed_masks=(...))    │                    │
│  │       [중복 masks 계산 SKIP]        │                    │
│  └─────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 테스트 전략

### 6.1 회귀 테스트

| 테스트 | 명령 | 기준 |
|--------|------|------|
| smoke_tests | `python scripts/engine_v7/smoke_tests.py` | 24/24 PASS |
| 전체 pytest | `pytest tests/` | 기존 결과 유지 |
| API 스냅샷 | `/api/v7/inspect` 응답 diff | JSON 구조 동일 |

### 6.2 신규 테스트 (선택)

| 테스트 | 내용 |
|--------|------|
| `test_v2_diag_precomputed` | precomputed 파라미터 전달 시 결과 일치 확인 |
| `test_v2_diag_no_precomputed` | precomputed=None 시 기존 동작 유지 확인 |

### 6.3 성능 검증

```python
# 검증 포인트: evaluate_per_color()에서 masks 중복 제거 효과
# 기대: build_color_masks_with_retry() 호출 횟수 2→1 감소
# 측정: 처리 시간 비교 (동일 이미지, 전후)
```

---

## 7. 구현 순서

### Step 1: v2_diagnostics.py 수정
1. `build_v2_diagnostics()` 시그니처에 keyword-only 파라미터 3개 추가
2. 함수 상단에서 precomputed 값 우선 사용 로직 추가
3. 내부 `to_polar()` 중복 호출 → 단일 변수로 통합

### Step 2: analyzer.py 수정
1. `_attach_v2_diagnostics()` 시그니처에 cached_* 파라미터 추가
2. `build_v2_diagnostics()` 호출부에 precomputed_* 전달
3. `evaluate()` 호출부 수정 (cached_geom, cached_polar 전달)
4. `evaluate_multi()` 호출부 수정 (cached_geom, cached_polar 전달)
5. `evaluate_per_color()` 호출부 수정 (cached_geom, cached_masks 전달)

### Step 3: 테스트 & 검증
1. smoke_tests 실행
2. pytest 전체 실행
3. API 응답 스냅샷 비교

### Step 4: 문서 업데이트
1. INTEGRATION_STATUS.md Phase 5 상태 → 100%
2. 전체 진행률 업데이트

---

## 8. 리스크 및 완화

| 리스크 | 영향 | 완화 방안 |
|--------|------|----------|
| `evaluate()`에서 polar 변수가 gate 실패 시 미정의 | High | gate 실패 early return 이후에만 cached_polar 전달 |
| precomputed_masks의 cfg 불일치 | Medium | 동일 cfg에서 계산된 경우만 캐시 전달 (호출부에서 보장) |
| `v2_diagnostics` 내 `to_polar()` R/T가 다를 가능성 | Low | 코드 확인 결과 동일한 `cfg["polar"]` 사용 확인 |

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-01-30 | Initial design | PDCA Auto |
