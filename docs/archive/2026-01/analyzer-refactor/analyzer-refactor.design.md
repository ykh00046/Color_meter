# analyzer.py 리팩토링 설계서

> **Summary**: 1,437줄 모놀리식 analyzer.py를 6개 모듈로 분할하는 상세 설계
>
> **Project**: Color Meter
> **Author**: PDCA Auto
> **Date**: 2026-01-31
> **Status**: Draft
> **Plan Doc**: [analyzer-refactor.plan.md](../../01-plan/features/analyzer-refactor.plan.md)

---

## 1. Architecture Overview

### 1.1 AS-IS

```
core/pipeline/
├── __init__.py          (empty)
├── analyzer.py          (1,437줄 - 모든 평가 로직)
├── single_analyzer.py   (단독 분석)
└── feature_export.py    (특성 내보내기)
```

### 1.2 TO-BE

```
core/pipeline/
├── __init__.py          (re-export: evaluate, evaluate_multi, etc.)
├── analyzer.py          (~350줄 - evaluate, evaluate_multi 진입점)
├── _common.py           (~180줄 - 공통 패턴 헬퍼)
├── _signature.py        (~110줄 - 시그니처 평가)
├── _diagnostics.py      (~250줄 - v1/v2/v3 진단 + OK features)
├── registration.py      (~200줄 - 등록 검증)
├── per_color.py         (~300줄 - 색상별 평가)
├── single_analyzer.py   (기존 유지)
└── feature_export.py    (기존 유지)
```

---

## 2. Module Design

### 2.1 `_common.py` — 공통 헬퍼 (~180줄)

evaluate()와 evaluate_multi()에서 반복되는 패턴을 추출.

#### 2.1.1 함수 목록

| 함수 | 원본 위치 | 설명 |
|------|----------|------|
| `reason_meta(reasons, overrides)` | L139-148 | reason_codes + messages 빌드 |
| `maybe_apply_white_balance(bgr, geom, cfg)` | L164-171 | WB 적용 |
| `run_gate_check(geom, bgr, cfg)` | NEW (추출) | gate 실행 + 조기 반환 Decision 생성 |
| `evaluate_anomaly(polar, bgr, geom, cfg, ...)` | NEW (추출) | anomaly 평가 공통 로직 |
| `build_diagnostics_block(polar, std_lab_mean, cfg, sig, ...)` | NEW (추출) | diagnostics dict + extra reasons |
| `attach_heatmap_defect(dec, anom, polar, cfg)` | NEW (추출) | heatmap + defect classify |
| `finalize_decision(dec, cfg, ok_log_context, ...)` | NEW (추출) | v2/v3 attach + qc_decision + ok_features |

#### 2.1.2 `run_gate_check` 상세 설계

evaluate()와 evaluate_multi()의 Gate 체크 패턴은 거의 동일:

**AS-IS** (`evaluate` L502-554, `evaluate_multi` L720-782):
```python
# 패턴 1: gate-only 모드
if mode == "gate":
    codes, messages = _reason_meta(gate.reasons)
    return Decision(label=..., gate=gate, ...)

# 패턴 2: gate 실패 + diag_on_fail=False
if not gate.passed and not diag_on_fail:
    codes, messages = _reason_meta(gate.reasons)
    return Decision(label="RETAKE", gate=gate, ...)

# 패턴 3: baseline 필수인데 없음
if require_baseline and use_relative and pattern_baseline is None:
    codes, messages = _reason_meta(["PATTERN_BASELINE_NOT_FOUND"])
    return Decision(label="RETAKE", ...)
```

**TO-BE**: `run_gate_check()` → `Decision | None` 반환. None이면 계속 진행.

```python
def run_gate_check(
    geom,
    test_bgr,
    cfg: Dict[str, Any],
    mode: str,
    pattern_baseline: Dict[str, Any] | None,
    *,
    extra_decision_kwargs: Dict[str, Any] | None = None,
) -> tuple[GateResult, Decision | None]:
    """Gate 실행 + 조기 반환 판단.

    Returns:
        (gate, early_decision): early_decision이 None이 아니면 즉시 반환.
    """
    gate = run_gate(
        geom, test_bgr,
        center_off_max=cfg["gate"]["center_off_max"],
        blur_min=cfg["gate"]["blur_min"],
        illum_max=cfg["gate"]["illum_max"],
    )
    diag_on_fail = bool(cfg.get("gate", {}).get("diagnostic_on_fail", False))
    kwargs = extra_decision_kwargs or {}

    # gate-only 모드
    if mode == "gate":
        codes, messages = reason_meta(gate.reasons)
        return gate, Decision(
            label="OK" if gate.passed else "RETAKE",
            reasons=gate.reasons,
            reason_codes=codes, reason_messages=messages,
            gate=gate, signature=None, anomaly=None,
            debug={"test_geom": asdict(geom), "inference_valid": gate.passed},
            diagnostics={"gate": asdict(gate)},
            phase="INSPECTION",
            **kwargs,
        )

    # gate 실패
    if not gate.passed and not diag_on_fail:
        codes, messages = reason_meta(gate.reasons)
        return gate, Decision(
            label="RETAKE",
            reasons=gate.reasons,
            reason_codes=codes, reason_messages=messages,
            gate=gate, signature=None, anomaly=None,
            debug={"inference_valid": False},
            phase="INSPECTION",
            **kwargs,
        )

    # baseline 필수 체크
    use_relative = bool(cfg.get("pattern_baseline", {}).get("use_relative", True))
    require_baseline = bool(cfg.get("pattern_baseline", {}).get("require", False))
    if require_baseline and use_relative and pattern_baseline is None:
        codes, messages = reason_meta(["PATTERN_BASELINE_NOT_FOUND"])
        return gate, Decision(
            label="RETAKE",
            reasons=["PATTERN_BASELINE_NOT_FOUND"],
            reason_codes=codes, reason_messages=messages,
            gate=gate, signature=None, anomaly=None,
            debug={"inference_valid": False, "baseline_missing": True},
            phase="INSPECTION",
            **kwargs,
        )

    return gate, None  # 계속 진행
```

#### 2.1.3 `evaluate_anomaly` 상세 설계

**AS-IS** (`evaluate` L566-592, `evaluate_multi` L801-827 — 동일):
```python
ang = angular_uniformity_score(polar, r_start=..., r_end=...)
blobs = detect_center_blobs(test_bgr, geom, frac=..., min_area=...)
if use_relative and pattern_baseline is not None:
    sample_features = extract_pattern_features(test_bgr, cfg=cfg)
    anom = score_anomaly_relative(...)
    anom.debug["abs_scores"] = {...}
    anom.debug["blob_debug"] = blobs
else:
    anom = score_anomaly(...)
```

**TO-BE**:
```python
def evaluate_anomaly(
    polar, test_bgr, geom, cfg: Dict[str, Any],
    pattern_baseline: Dict[str, Any] | None,
    use_relative: bool,
) -> AnomalyResult:
    """anomaly 평가 공통 로직."""
    ang = angular_uniformity_score(polar, ...)
    blobs = detect_center_blobs(test_bgr, geom, ...)
    if use_relative and pattern_baseline is not None:
        sample_features = extract_pattern_features(test_bgr, cfg=cfg)
        anom = score_anomaly_relative(...)
        anom.debug["abs_scores"] = {
            "angular_uniformity": float(ang),
            "center_blob_count": float(blobs["blob_count"]),
        }
        anom.debug["blob_debug"] = blobs
    else:
        anom = score_anomaly(...)
    return anom
```

#### 2.1.4 `finalize_decision` 상세 설계

**AS-IS** (`evaluate` L659-706, `evaluate_multi` L900-948 — 거의 동일):
```python
# v2/v3 attach
_attach_v2_diagnostics(test_bgr, dec, cfg, ok_log_context, ...)
v2_diag = dec.diagnostics.get("v2_diagnostics") or {}
_attach_v3_summary(dec, v2_diag, cfg, ok_log_context)
_attach_v3_trend(dec, ok_log_context)

# qc_decision 빌드
sample_clusters = ...
match_result = ...
gate_scores = ...  (gate_engine 호환 레이어)
decision_json = build_decision(...)
dec.ops["qc_decision"] = {"schema_version": "qc_decision.v1", **decision_json}
dec.pattern_color = decision_json.get("pattern_color", {})
_attach_features(dec, cfg, ok_log_context)
_append_ok_features(test_bgr, dec, cfg, pattern_baseline, ok_log_context)
```

**TO-BE**:
```python
def finalize_decision(
    dec: Decision,
    test_bgr,
    cfg: Dict[str, Any],
    ok_log_context: Dict[str, Any] | None,
    pattern_baseline: Dict[str, Any] | None,
    mode: str,
    *,
    cached_geom=None,
    cached_polar=None,
    cached_masks=None,
) -> None:
    """v2/v3 diagnostics + qc_decision + OK features 첨부 (in-place)."""
    if mode != "signature":
        attach_v2_diagnostics(
            test_bgr, dec, cfg, ok_log_context,
            cached_geom=cached_geom,
            cached_masks=cached_masks,
            cached_polar=cached_polar,
        )

    v2_diag = dec.diagnostics.get("v2_diagnostics") or {}
    attach_v3_summary(dec, v2_diag, cfg, ok_log_context)
    attach_v3_trend(dec, ok_log_context)

    # qc_decision
    sample_clusters = (v2_diag.get("segmentation", {}) or {}).get("clusters", []) if v2_diag else []
    match_result = v2_diag.get("ink_match") if v2_diag else None

    raw_gate = dec.gate.scores if dec.gate else {}
    gate_scores = dict(raw_gate or {})
    if "sharpness_score" not in gate_scores and "sharpness_laplacian_var" in gate_scores:
        gate_scores["sharpness_score"] = gate_scores["sharpness_laplacian_var"]

    decision_json = build_decision(
        run_id=(ok_log_context or {}).get("run_id") or "",
        phase="INSPECTION", cfg=cfg, gate_scores=gate_scores,
        expected_inks=ok_log_context.get("expected_ink_count_input") if ok_log_context else None,
        sample_clusters=sample_clusters, match_result=match_result,
        deltae_summary_method="max", inkness_summary_method="min",
    )

    dec.ops = dec.ops or {}
    dec.ops["qc_decision"] = {"schema_version": "qc_decision.v1", **decision_json}
    if (cfg.get("debug") or {}).get("include_full_qc_decision", False):
        dec.debug["full_qc_decision"] = decision_json
    dec.pattern_color = decision_json.get("pattern_color", {})

    _attach_features(dec, cfg, ok_log_context)
    append_ok_features(test_bgr, dec, cfg, pattern_baseline, ok_log_context)
```

---

### 2.2 `_signature.py` — 시그니처 평가 (~110줄)

| 함수 | 원본 줄 | 설명 |
|------|---------|------|
| `build_k_by_segment(R, segments, segment_k, default_k)` | L38-48 | segment별 k 배열 |
| `evaluate_signature(test_mean, std_model, cfg)` | L51-127 | 시그니처 비교+판정 |
| `pick_best_mode(mode_sigs)` | L130-136 | 최적 모드 선택 |

이동만 수행, 코드 변경 없음. 함수명에서 `_` prefix 유지 (private).

---

### 2.3 `_diagnostics.py` — 진단 로직 (~250줄)

| 함수 | 원본 줄 | 설명 |
|------|---------|------|
| `mean_grad(l_map)` | L151-161 | 평균 그래디언트 |
| `compute_worst_case(test_lab_map, std_lab_mean)` | L174-207 | worst case 메트릭 |
| `compute_diagnostics(test_lab_map, std_lab_mean, cfg)` | L210-321 | 색상/패턴 진단 |
| `append_ok_features(test_bgr, dec, cfg, pb, ctx)` | L324-371 | OK 로그 기록 |
| `attach_v2_diagnostics(test_bgr, dec, cfg, ctx, ...)` | L373-429 | v2 diagnostics |
| `attach_v3_summary(dec, v2_diag, cfg, ctx)` | L432-445 | v3 요약 |
| `attach_v3_trend(dec, ctx)` | L448-480 | v3 트렌드 |
| `attach_pattern_color(dec, cfg)` | L483-485 | stub |
| `attach_features(dec, cfg, ctx)` | L488-489 | stub |

---

### 2.4 `registration.py` — 등록 검증 (~200줄)

| 함수 | 원본 줄 | 설명 |
|------|---------|------|
| `_registration_summary(std_models, cfg)` | L951-1068 | 등록 요약 통계 |
| `evaluate_registration_multi(test_bgr, std_models, cfg)` | L1071-1141 | 등록 검증 평가 |

이동만 수행. `_registration_summary`는 `v7.py`에서 직접 import하므로 re-export 필수.

---

### 2.5 `per_color.py` — 색상별 평가 (~300줄)

| 함수 | 원본 줄 | 설명 |
|------|---------|------|
| `evaluate_per_color(...)` | L1144-1437 | 색상별 멀티잉크 평가 |

`_common.py`의 `run_gate_check()`, `finalize_decision()`을 사용하도록 리팩토링.

---

### 2.6 `analyzer.py` (축소) — 진입점 (~350줄)

남는 내용:
- imports + re-exports
- `evaluate()` (~150줄, 공통 함수 호출)
- `evaluate_multi()` (~150줄, 공통 함수 호출)

---

## 3. Implementation Order

| Step | 작업 | 파일 | 위험도 |
|------|------|------|--------|
| 1 | `_signature.py` 생성 (순수 이동) | NEW | Low |
| 2 | `_diagnostics.py` 생성 (순수 이동) | NEW | Low |
| 3 | `_common.py` 생성 (추출+공통화) | NEW | Medium |
| 4 | `registration.py` 생성 (순수 이동) | NEW | Low |
| 5 | `per_color.py` 생성 (이동+공통화) | NEW | Medium |
| 6 | `analyzer.py` 축소 (공통 함수 호출로 교체) | MODIFY | High |
| 7 | `__init__.py` re-export 추가 | MODIFY | Low |
| 8 | Import 경로 검증 | - | Low |

**원칙**: Step 1-2는 순수 코드 이동. Step 3에서 공통 함수 생성. Step 6에서 analyzer.py를 공통 함수 호출로 교체. 각 단계에서 syntax check.

---

## 4. External Import Compatibility

### 4.1 현재 외부 참조

```python
# src/web/routers/v7.py
from src.engine_v7.core.pipeline import analyzer as analyzer_mod          # 모듈 참조
from src.engine_v7.core.pipeline.analyzer import _registration_summary    # private 함수
from src.engine_v7.core.pipeline.analyzer import evaluate_multi           # public
from src.engine_v7.core.pipeline.analyzer import evaluate_registration_multi  # public

# src/engine_v7/api.py
from src.engine_v7.core.pipeline.analyzer import evaluate_multi           # public
from src.engine_v7.core.pipeline.analyzer import evaluate_per_color       # public

# tests
from core.pipeline.analyzer import evaluate                               # public
from core.pipeline.analyzer import evaluate_per_color                     # public
```

### 4.2 Re-export 전략

**analyzer.py (축소)**에 다음 re-export 추가:
```python
# Re-exports for backward compatibility
from .registration import evaluate_registration_multi, _registration_summary
from .per_color import evaluate_per_color
```

이로써 기존 `from ...analyzer import X` 패턴이 모두 유지됨.

---

## 5. Detailed Diff: evaluate() vs evaluate_multi()

중복 제거 대상 구간의 정밀 비교:

### 5.1 차이점 (공통화 불가)

| 구간 | evaluate() | evaluate_multi() |
|------|-----------|-----------------|
| 모델 접근 | `std_model` (단일) | `std_models` (Dict) |
| Polar 생성 | `std_model.meta["R"]` | `any_model.meta["R"]` |
| Signature | 단일 모델 비교 | 모든 모드 순회 + `_pick_best_mode` |
| Debug | `"std_geom": asdict(std_model.geom)` | `"std_geoms": {k: asdict(v.geom) ...}` |
| Return | `Decision` | `(Decision, Dict[str, SignatureResult])` |

### 5.2 공통 구간 (추출 대상)

| 구간 | 줄 수 | 추출 대상 함수 |
|------|-------|---------------|
| Gate 체크 3패턴 | ~65 | `run_gate_check()` |
| Anomaly 평가 | ~27 | `evaluate_anomaly()` |
| Diagnostics + references | ~33 | `build_diagnostics_block()` |
| Heatmap + defect | ~11 | `attach_heatmap_defect()` |
| v2/v3 + qc_decision + ok_features | ~48 | `finalize_decision()` |
| **합계** | **~184** | 5개 공통 함수 |

---

## 6. Changed Files Summary

| File | Action | Lines |
|------|--------|-------|
| `core/pipeline/_signature.py` | CREATE | ~110 |
| `core/pipeline/_diagnostics.py` | CREATE | ~250 |
| `core/pipeline/_common.py` | CREATE | ~180 |
| `core/pipeline/registration.py` | CREATE | ~200 |
| `core/pipeline/per_color.py` | CREATE | ~300 |
| `core/pipeline/analyzer.py` | MODIFY (1437→~350) | -1087 |
| `core/pipeline/__init__.py` | MODIFY (re-exports) | +10 |
| **Net** | | **~-37줄** (중복 제거) |

---

## 7. Success Criteria

- [ ] `analyzer.py` 400줄 이하
- [ ] 5개 신규 모듈 생성
- [ ] `python -c "import ast; ast.parse(open(f, encoding='utf-8').read())"` 모든 파일 통과
- [ ] 기존 import 경로 grep 검증 (5개 참조 모두 동작)
- [ ] 순환 import 없음 검증
- [ ] evaluate() / evaluate_multi() 간 중복 코드 0줄

---

## 8. Backward Compatibility

모든 외부 참조가 `analyzer.py`의 re-export를 통해 유지됨:

```python
# analyzer.py 끝부분
# --- Backward compatibility re-exports ---
from .registration import evaluate_registration_multi, _registration_summary  # noqa: F401
from .per_color import evaluate_per_color  # noqa: F401
```

**검증**: 리팩토링 후 아래 grep이 모두 유효한 import를 반환해야 함:
```bash
grep -r "from.*pipeline.analyzer import" src/ --include="*.py"
```

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-01-31 | Initial design | PDCA Auto |
