# Phase 5: v2_diagnostics Engine B 통합 계획서

> **Summary**: v2_diagnostics 모듈의 Engine B 완전 통합으로 잉크 분석 파이프라인 일원화
>
> **Project**: Color Meter
> **Author**: PDCA Auto
> **Date**: 2026-01-30
> **Status**: Draft

---

## 1. Overview

### 1.1 Purpose

v2_diagnostics 모듈이 현재 Engine B(color_masks_with_retry 기반)를 호출하지만, 반환값 변환 및 analyzer.py와의 통합이 불완전합니다. 이 작업은 Engine B를 Single Source of Truth(SSoT)로 완전 통합하여 잉크 분석 파이프라인의 일관성을 확보합니다.

### 1.2 Background

- Phase 1~4에서 single_analyzer, color_masks, measure 폴더 구조, src/analysis 마이그레이션 완료 (100%)
- Phase 5는 현재 30% 진행 (파일 이동만 완료, 로직 교체 대기)
- `build_v2_diagnostics()`는 이미 Engine B 함수(`build_color_masks_with_retry`)를 호출하나, analyzer.py의 `evaluate()` / `evaluate_multi()` 내부에서 중복 호출이 발생할 수 있음
- ENGINE_UNIFICATION_STATUS.md, INTEGRATION_STATUS.md에 남은 작업이 명시됨

### 1.3 Related Documents

- `docs/INTEGRATION_STATUS.md` - 통합 현황 (Phase 5 섹션)
- `docs/engine_v7/ENGINE_UNIFICATION_STATUS.md` - Engine A/B 비교 및 통합 계획
- `docs/HANDOFF.md` - 인계 문서 (Phase 5~8 언급)
- `docs/engine_v7/INSPECTION_FLOW.md` - 검사 흐름도

---

## 2. Scope

### 2.1 In Scope

- [x] `v2_diagnostics.py` 파일이 Engine B 경로에 위치 (완료)
- [ ] `build_v2_diagnostics()` 내 `build_color_masks_with_retry()` 호출 검증 및 중복 제거
- [ ] analyzer.py `_attach_v2_diagnostics()` 호출 경로에서 중복 세분화 방지
- [ ] `v2_flags.py` 경고 매핑 Engine B 출력 형식과 일치 확인
- [ ] 회귀 테스트 (smoke_tests 24/24 유지)
- [ ] STD 등록 API 경로 검증

### 2.2 Out of Scope

- Phase 6: pipeline.py 마이그레이션 (별도 PDCA)
- Phase 7: 레거시 API 라우터 마이그레이션 (별도 PDCA)
- analyzer.py 전체 리팩토링 (별도 PDCA)
- 새로운 진단 항목 추가

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | `build_v2_diagnostics()`가 Engine B `build_color_masks_with_retry()` 결과만 사용 | High | 진행중 |
| FR-02 | analyzer.py `evaluate()`에서 v2_diagnostics 호출 시 중복 색상 마스크 계산 제거 | High | Pending |
| FR-03 | `_attach_v2_diagnostics()`가 ok_log_context의 polar_alpha를 올바르게 전달 | Medium | 검증 필요 |
| FR-04 | v2_flags 출력 형식이 Engine B 클러스터 메타데이터와 일치 | Medium | 검증 필요 |
| FR-05 | 기존 API 응답(`/api/v7/inspect`, `/api/v7/analyze_single`) 형식 유지 | High | 검증 필요 |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement Method |
|----------|----------|-------------------|
| 호환성 | 기존 API 응답 JSON 스키마 변경 없음 | smoke_tests + API diff |
| 성능 | 중복 색상 마스크 계산 제거로 처리 시간 감소 | 프로파일링 전후 비교 |
| 안정성 | smoke_tests 24/24 유지 | `scripts/engine_v7/smoke_tests.py` |

---

## 4. Success Criteria

### 4.1 Definition of Done

- [ ] `build_v2_diagnostics()` Engine B 전용 경로만 사용 확인
- [ ] analyzer.py에서 중복 `build_color_masks_with_retry()` 호출 제거 또는 캐시 활용
- [ ] smoke_tests 24/24 통과
- [ ] `/api/v7/inspect` API 응답 형식 변경 없음
- [ ] `/api/v7/analyze_single` API 응답 형식 변경 없음
- [ ] v2_flags 경고 코드 매핑 정상 동작

### 4.2 Quality Criteria

- [ ] 기존 테스트 전체 통과 (`pytest tests/`)
- [ ] 처리 시간 회귀 없음 (동일 이미지 기준)
- [ ] 코드 lint 통과 (flake8, mypy)

---

## 5. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| 중복 호출 제거 시 v2_diagnostics 출력 누락 | High | Medium | 기존 출력 스냅샷과 diff 비교 |
| analyzer.py 복잡도로 인한 부작용 | Medium | Medium | 변경 최소화, 단위 테스트 추가 |
| ENGINE_UNIFICATION_STATUS.md 인코딩 깨짐 | Low | High | 실제 코드 기반으로 작업 (문서 참고용) |
| ok_log_context → polar_alpha 전달 오류 | Medium | Low | 디버그 로깅 추가 후 검증 |

---

## 6. Architecture Considerations

### 6.1 현재 호출 흐름

```
/api/v7/inspect
  → analyzer.py::evaluate()
    → _attach_v2_diagnostics()
      → build_v2_diagnostics()
        → build_color_masks_with_retry()  [Engine B]
        → to_polar(), to_cie_lab()
        → compute_cluster_effective_densities()
        → separation_ab(), calculate_pairwise_deltaE()
      → build_v2_flags()
    → (evaluate 내부에서도 color_masks 관련 호출 가능 - 중복 위험)
```

### 6.2 목표 호출 흐름

```
/api/v7/inspect
  → analyzer.py::evaluate()
    → Engine B 색상 마스크 1회 계산 (캐시 또는 전달)
    → _attach_v2_diagnostics(cached_masks=...)
      → build_v2_diagnostics(cached_masks=...)  [중복 제거]
      → build_v2_flags()
```

### 6.3 핵심 변경 포인트

| 파일 | 변경 내용 | 복잡도 |
|------|----------|--------|
| `core/pipeline/analyzer.py` | 색상 마스크 캐시/전달 로직 추가 | Medium |
| `core/measure/diagnostics/v2_diagnostics.py` | cached_masks 파라미터 수용 | Low |
| `core/measure/diagnostics/v2_flags.py` | Engine B 출력 형식 매핑 검증 | Low |

---

## 7. Implementation Plan

### Step 1: 현재 상태 분석 (분석)
1. analyzer.py에서 `build_color_masks_with_retry()` 호출 지점 전수 조사
2. `build_v2_diagnostics()` 내부와의 중복 여부 확인
3. 기존 API 응답 스냅샷 저장 (회귀 테스트용)

### Step 2: 중복 제거 구현 (구현)
1. `build_v2_diagnostics()`에 선택적 `cached_masks` 파라미터 추가
2. analyzer.py에서 이미 계산된 마스크를 전달하도록 수정
3. 캐시 미스 시 기존 로직 유지 (backward compatible)

### Step 3: 검증 (테스트)
1. smoke_tests 실행 (24/24 확인)
2. API 응답 스냅샷 diff 비교
3. pytest 전체 실행
4. v2_flags 경고 코드 매핑 검증

### Step 4: 문서 업데이트
1. INTEGRATION_STATUS.md Phase 5 상태 업데이트 (100%)
2. ENGINE_UNIFICATION_STATUS.md 업데이트

---

## 8. Next Steps

1. [ ] Design 문서 작성 (`/pdca design phase5-diagnostics`)
2. [ ] 팀 리뷰 및 승인
3. [ ] 구현 시작

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-01-30 | Initial draft | PDCA Auto |
