# Phase 6-8: 레거시 마이그레이션 현황 재평가 및 문서 정비 계획서

> **Summary**: Phase 6-8 레거시 마이그레이션이 이미 완료되었음을 확인하고, 문서 정비 및 잔존 개선 사항 도출
>
> **Project**: Color Meter
> **Author**: PDCA Auto
> **Date**: 2026-01-30
> **Status**: Draft

---

## 1. Overview

### 1.1 Purpose

INTEGRATION_STATUS.md에 Phase 6(pipeline.py 마이그레이션), Phase 7(레거시 API 라우터), Phase 8(src/core/ 제거)이 미완료로 기재되어 있으나, 코드 조사 결과 **이 3단계 모두 실질적으로 완료**된 것으로 확인되었습니다. 문서와 실제 코드 상태의 불일치를 해소합니다.

### 1.2 Background - 코드 조사 결과

#### Phase 6 (src/pipeline.py 마이그레이션): **이미 완료**
- 레거시 `src/pipeline.py` 파일이 존재하지 않음
- `src/services/inspection_service.py`가 v7 엔진 기반으로 완전 재작성 (1,233줄)
- `src/engine_v7/api.py` Facade API가 모든 호출을 처리
- 레거시 `from src.core.` import가 전체 코드베이스에서 **0건**

#### Phase 7 (레거시 API 라우터 마이그레이션): **이미 완료**
- `src/web/routers/v7.py` (2,670줄)가 완전히 v7 import만 사용
- `src/web/routers/inspection.py` (797줄)는 DB 쿼리 기반 (엔진 미참조)
- 레거시 `std.py`, `sku.py` 라우터가 존재하지 않음 (삭제 완료)

#### Phase 8 (src/core/ 제거): **이미 완료**
- `src/core/` 폴더가 git status에서 `D` (deleted) 상태
- 모든 하위 파일(color_evaluator.py, ink_estimator.py, lens_detector.py 등) 삭제됨
- 어떤 파일도 `src.core`를 import하지 않음

### 1.3 Related Documents

- `docs/INTEGRATION_STATUS.md` - 현재 Phase 6-8 "미완료" 기재 (문서 오류)
- `docs/HANDOFF.md` - Phase 5~8 미완료 언급 (문서 오류)
- `docs/Legacy_Cleanup_Summary.md` - 레거시 정리 현황

---

## 2. Scope

### 2.1 In Scope

- [ ] INTEGRATION_STATUS.md Phase 6-8 상태를 100% 완료로 업데이트
- [ ] INTEGRATION_STATUS.md 전체 진행률을 100%로 업데이트
- [ ] HANDOFF.md Phase 5~8 관련 내용 업데이트
- [ ] 잔존 개선 기회 식별 및 문서화:
  - inspection_service.py (1,233줄) 리팩토링 검토
  - v7.py (2,670줄) 라우터 분할 검토

### 2.2 Out of Scope

- inspection_service.py 리팩토링 구현 (별도 PDCA)
- v7.py 라우터 분할 구현 (별도 PDCA)
- 새 기능 추가

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | INTEGRATION_STATUS.md Phase 6 상태 → 100% 완료 | High | Pending |
| FR-02 | INTEGRATION_STATUS.md Phase 7 상태 → 100% 완료 | High | Pending |
| FR-03 | INTEGRATION_STATUS.md Phase 8 상태 → 100% 완료 | High | Pending |
| FR-04 | 전체 진행률 → 100% 업데이트 | High | Pending |
| FR-05 | HANDOFF.md Phase 5~8 관련 내용 수정 | Medium | Pending |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement Method |
|----------|----------|-------------------|
| 정확성 | 문서와 코드 상태 100% 일치 | grep 검증 |
| 완전성 | Phase 1~8 모든 상태 반영 | 문서 리뷰 |

---

## 4. Success Criteria

### 4.1 Definition of Done

- [ ] INTEGRATION_STATUS.md에 Phase 6-8이 완료로 표시
- [ ] 전체 진행률이 100%로 표시
- [ ] HANDOFF.md 관련 섹션 수정
- [ ] `from src.core.` import 0건 확인 (이미 확인됨)
- [ ] 잔존 개선 기회 목록 작성

### 4.2 Quality Criteria

- [ ] 문서 정확성 검증 (코드 기반)
- [ ] 후속 PDCA 사이클 식별

---

## 5. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| git에서 삭제되었지만 커밋 안 된 파일이 복구될 가능성 | Low | Low | git status 확인으로 검증 |
| HANDOFF.md 수정 시 다른 섹션 영향 | Low | Low | 해당 섹션만 수정 |

---

## 6. 잔존 개선 기회 (후속 PDCA 후보)

| 항목 | 현재 상태 | 개선 방향 | 우선순위 |
|------|----------|----------|----------|
| `inspection_service.py` | 1,233줄 모놀리식 | 클래스 분할 (로딩/검출/분석/결과) | Medium |
| `v7.py` 라우터 | 2,670줄 단일 파일 | 기능별 분할 (inspect/register/config) | Medium |
| `analyzer.py` | 1,436줄 | sub-analyzer 분할 | Medium |

---

## 7. Next Steps

1. [ ] Design 문서 작성 (`/pdca design phase6-pipeline`)
2. [ ] 문서 업데이트 구현
3. [ ] 후속 PDCA 사이클 계획

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-01-30 | Initial draft - Phase 6-8 재평가 | PDCA Auto |
