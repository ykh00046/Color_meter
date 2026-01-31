# v7.py 라우터 분할 계획서

> **Summary**: 2,670줄 모놀리식 v7.py를 도메인별 서브 라우터로 분할
>
> **Project**: Color Meter
> **Author**: PDCA Auto
> **Date**: 2026-01-31
> **Status**: Draft

---

## 1. Overview

### 1.1 Purpose

`src/web/routers/v7.py` (2,670줄)는 등록·활성화·검사·플레이트·메트릭 등 5개 도메인의 15개 라우트와 47개 헬퍼 함수를 단일 파일에 담고 있다. 파일 크기가 과도하여 탐색·수정·리뷰가 어렵다. 도메인별 서브 라우터로 분할하여 유지보수성을 개선한다.

### 1.2 Problem Analysis

#### 파일 복잡도

| 메트릭 | 값 |
|--------|-----|
| 총 줄 수 | 2,670 |
| 라우트 함수 | 15 |
| 헬퍼 함수 | 47 |
| Pydantic 모델 | 5 |
| Import 수 | 40+ |

#### 도메인 혼재

| 도메인 | 라우트 수 | 줄 수 (추정) |
|--------|----------|-------------|
| Registration & Model Management | 5 | ~325 |
| Activation & Governance | 2 | ~126 |
| Metrics & Diagnostics | 3 | ~56 |
| Inspection & Quality Control | 3 | ~512 |
| Plate & Calibration | 3 | ~251 |
| Core Utilities (비-라우트) | 0 | ~1,244 |

#### 헬퍼 함수 의존 그래프

Core Utilities (1,244줄)가 전체의 47%를 차지. 여러 도메인에서 공유하는 함수가 많아 분할 시 공통 모듈 추출이 핵심.

### 1.3 Related Documents

- `docs/01-plan/features/analyzer-refactor.plan.md` — 동일 패턴의 리팩토링 선행 사례
- `docs/02-design/features/analyzer-refactor.design.md` — 모듈 분할 설계 패턴 참고

---

## 2. Scope

### 2.1 In Scope

- [ ] 공통 유틸리티를 `_helpers.py`로 추출
- [ ] 등록(Registration) 라우트를 `v7_registration.py`로 분리
- [ ] 활성화/거버넌스 라우트를 `v7_activation.py`로 분리
- [ ] 검사(Inspection) 라우트를 `v7_inspection.py`로 분리
- [ ] 플레이트/교정 라우트를 `v7_plate.py`로 분리
- [ ] 메트릭/진단 라우트를 `v7_metrics.py`로 분리
- [ ] v7.py를 서브 라우터 조립 진입점으로 축소
- [ ] 외부 import 경로 호환성 유지 (`app.py`에서 `v7.router` 참조)

### 2.2 Out of Scope

- 라우트 API 시그니처 변경
- 비즈니스 로직 변경
- inspection_service.py 리팩토링 (별도 PDCA)
- 테스트 추가 (별도 PDCA)

---

## 3. Design Concept

### 3.1 목표 구조

```
src/web/routers/
├── v7.py                      # 축소: 서브 라우터 조립 진입점 (~80줄)
├── v7_helpers.py              # 공통 유틸리티 (~450줄)
│   ├── NumpyEncoder
│   ├── _env_flag(), _load_cfg(), _resolve_cfg_path()
│   ├── _atomic_write_json(), _safe_float(), _safe_delete_path()
│   ├── _compute_center_crop_mean_rgb()
│   ├── _load_snapshot_config(), _load_active_snapshot()
│   ├── _normalize_expected_ink_count(), _parse_match_ids()
│   ├── _require_role(), _load_bgr()
│   ├── _save_uploads(), _save_single_upload()
│   ├── _validate_subprocess_arg(), _run_script(), _run_script_async()
│   ├── _read_index(), _read_index_at(), _find_entry()
│   └── Pydantic models (공통)
├── v7_registration.py         # 등록 + 모델 관리 (~650줄)
│   ├── 5 routes: register_validate, get_status, list_entries,
│   │            register_cleanup, get_candidates
│   ├── _resolve_sku_for_ink(), _auto_tune_cfg_from_std()
│   ├── _build_std_model(), _run_gate_check()
│   ├── _active_versions(), _warning_diff(), _approval_status()
│   ├── _resolve_pack_path(), _load_approval_pack()
│   ├── _validate_pack_for_activate(), _build_approval_pack()
│   └── Pydantic: RegisterCleanupRequest
├── v7_activation.py           # 활성화 + 거버넌스 (~400줄)
│   ├── 2 routes: activate_model, rollback_model
│   ├── _finalize_activation(), _finalize_pack()
│   ├── _write_active_snapshot()
│   ├── _write_pattern_baseline(), _write_ink_baseline()
│   └── Pydantic: ActivateRequest, RollbackRequest
├── v7_inspection.py           # 검사 + QC (~550줄)
│   ├── 3 routes: test_run, inspect, analyze_single
│   ├── _set_inspection_metadata()
│   ├── _write_inspection_artifacts()
│   ├── _generate_single_analysis_artifacts()
│   ├── _polar_mask_to_base64()
│   ├── _load_latest_v2_review(), _load_latest_v3_summary()
│   └── _load_recent_decisions_for_trend()
├── v7_metrics.py              # 메트릭 + 진단 + 삭제 (~200줄)
│   ├── 3 routes: get_v2_metrics, get_trend_line, delete_entry
│   ├── _delete_approval_packs(), _delete_results_dirs()
│   └── Pydantic: DeleteEntryRequest
└── v7_plate.py                # 플레이트 + 교정 (~350줄)
    ├── 3 routes: extract_plate_gate_api, intrinsic_calibrate_api,
    │            intrinsic_simulate_api
    ├── _generate_plate_pair_artifacts()
    └── Pydantic: PlateGateRequest
```

### 3.2 예상 줄 수

| 모듈 | 예상 줄 수 | 원본 비율 |
|------|-----------|----------|
| v7.py (축소, 조립) | ~80 | 3% |
| v7_helpers.py | ~450 | 17% |
| v7_registration.py | ~650 | 24% |
| v7_activation.py | ~400 | 15% |
| v7_inspection.py | ~550 | 21% |
| v7_metrics.py | ~200 | 7% |
| v7_plate.py | ~350 | 13% |
| **합계** | **~2,680** | 100% |

### 3.3 서브 라우터 패턴

각 도메인 모듈은 자체 `APIRouter`를 생성하고, `v7.py`에서 조립:

```python
# v7.py (축소 후)
from fastapi import APIRouter
from .v7_registration import router as registration_router
from .v7_activation import router as activation_router
from .v7_inspection import router as inspection_router
from .v7_metrics import router as metrics_router
from .v7_plate import router as plate_router

router = APIRouter(prefix="/api/v7", tags=["V7 MVP"])

router.include_router(registration_router)
router.include_router(activation_router)
router.include_router(inspection_router)
router.include_router(metrics_router)
router.include_router(plate_router)
```

### 3.4 외부 호환성

현재 외부 참조:
```python
# app.py
from src.web.routers import inspection, v7
app.include_router(v7.router)
```

`v7.router`가 서브 라우터를 포함하므로 **API 경로 변경 없음**. `app.py` 수정 불필요.

---

## 4. Requirements

### 4.1 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | 공통 유틸리티를 v7_helpers.py로 추출 | High |
| FR-02 | 등록 라우트를 v7_registration.py로 분리 | High |
| FR-03 | 활성화 라우트를 v7_activation.py로 분리 | High |
| FR-04 | 검사 라우트를 v7_inspection.py로 분리 | High |
| FR-05 | 메트릭 라우트를 v7_metrics.py로 분리 | High |
| FR-06 | 플레이트 라우트를 v7_plate.py로 분리 | High |
| FR-07 | v7.py를 서브 라우터 조립 진입점으로 축소 | High |
| FR-08 | API 경로 100% 유지 | Critical |

### 4.2 Non-Functional Requirements

| Category | Criteria | Measurement |
|----------|----------|-------------|
| 호환성 | 모든 API 경로 동일 | curl 검증 |
| 호환성 | `app.py`의 `v7.router` 참조 무변경 | grep 검증 |
| 동작 무변경 | 동일 요청 → 동일 응답 | smoke test |
| 줄 수 감소 | v7.py 100줄 이하 | wc -l |

---

## 5. Success Criteria

### 5.1 Definition of Done

- [ ] v7.py 100줄 이하로 축소
- [ ] 6개 서브 모듈 생성 (helpers, registration, activation, inspection, metrics, plate)
- [ ] 모든 15개 API 경로 유지
- [ ] `app.py` 수정 불필요 (`v7.router` 인터페이스 유지)
- [ ] python -c "import ast; ast.parse(...)" 전 파일 통과
- [ ] 순환 import 없음

### 5.2 Quality Criteria

- [ ] 각 모듈이 단일 도메인 책임
- [ ] 공통 함수 중복 0줄

---

## 6. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| 순환 import (helpers ↔ domain) | High | Medium | helpers는 engine/config만 import, 역방향 금지 |
| API 경로 변경 | Critical | Low | 서브 라우터에 prefix 없이 포함 + curl 검증 |
| 공유 상태 (모듈 변수) | Medium | Medium | `_SUBPROC_SEM`, `V7_*` 상수를 helpers에 배치 |
| 함수 분류 애매 (cross-domain) | Low | High | 2개 이상 도메인에서 사용 시 helpers 배치 |

---

## 7. Next Steps

1. [ ] Design 문서 작성 (`/pdca design v7-router-split`)
2. [ ] 구현 (모듈 분할)
3. [ ] Gap analysis
4. [ ] 완료 보고서

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-01-31 | Initial draft | PDCA Auto |
