# Phase 6-8: 레거시 마이그레이션 문서 정비 설계서

> **Summary**: Phase 6-8이 이미 완료된 상태를 문서에 정확히 반영하는 변경 설계
>
> **Project**: Color Meter
> **Author**: PDCA Auto
> **Date**: 2026-01-31
> **Status**: Approved

---

## 1. Overview

### 1.1 Purpose

코드 조사 결과 Phase 6(pipeline.py 마이그레이션), Phase 7(레거시 API 라우터), Phase 8(src/core/ 제거)이 모두 **실질적으로 완료**되었으나, `INTEGRATION_STATUS.md`와 `HANDOFF.md`에는 미완료로 기재되어 있음. 이 문서-코드 불일치를 해소한다.

### 1.2 Evidence (코드 기반 확인)

| Phase | 확인 사항 | 결과 |
|-------|----------|------|
| Phase 6 | `src/pipeline.py` 존재 여부 | 파일 없음 (삭제됨) |
| Phase 6 | `src/services/inspection_service.py` v7 기반 | 1,233줄, v7 import만 사용 |
| Phase 6 | `src/engine_v7/api.py` Facade API | 모든 호출 처리 |
| Phase 7 | `src/web/routers/v7.py` v7 import | 2,670줄, 레거시 import 0건 |
| Phase 7 | `std.py`, `sku.py` 라우터 존재 | 삭제됨 |
| Phase 8 | `src/core/` 폴더 | git status `D` (전체 삭제) |
| Phase 8 | `from src.core.` import | grep 결과 **0건** |

---

## 2. Changes

### 2.1 INTEGRATION_STATUS.md 변경

#### Change 1: Phase 6 상태 업데이트
**AS-IS** (line 162-169):
```markdown
### Phase 6: src/pipeline.py 마이그레이션
**상태**: 미착수
(작업 필요 목록...)
```

**TO-BE**:
```markdown
### Phase 6: src/pipeline.py 마이그레이션
**상태**: 완료 (2026-01-30 확인)
- 레거시 `src/pipeline.py` 삭제 완료
- `src/services/inspection_service.py`가 v7 엔진 기반으로 완전 재작성
- `src/engine_v7/api.py` Facade API가 모든 호출 처리
- 레거시 `from src.core.` import 전체 코드베이스에서 0건
```

#### Change 2: Phase 7 상태 업데이트
**AS-IS** (line 171-179):
```markdown
### Phase 7: 레거시 API 라우터 마이그레이션
**상태**: 미착수
(대상 파일/작업 목록...)
```

**TO-BE**:
```markdown
### Phase 7: 레거시 API 라우터 마이그레이션
**상태**: 완료 (2026-01-30 확인)
- `src/web/routers/v7.py` (2,670줄) v7 import만 사용
- `src/web/routers/inspection.py` (797줄) DB 쿼리 기반 (엔진 미참조)
- 레거시 `std.py`, `sku.py` 라우터 삭제 완료
```

#### Change 3: Phase 8 상태 업데이트
**AS-IS** (line 183-188):
```markdown
### Phase 8: src/core/ 제거
**상태**: 보류 (Phase 6-7 완료 후)
(제거 대상 목록...)
```

**TO-BE**:
```markdown
### Phase 8: src/core/ 제거
**상태**: 완료 (2026-01-30 확인)
- `src/core/` 전체 폴더 삭제 완료 (git status `D`)
- 모든 하위 파일 삭제 (color_evaluator.py, ink_estimator.py, lens_detector.py 등)
- 어떤 파일도 `src.core`를 import하지 않음
```

#### Change 4: 진행률 업데이트
**AS-IS** (line 194-204):
```
⬜ Phase 6: ... (0%)
⬜ Phase 7: ... (0%)
⬜ Phase 8: ... (0%)
전체 진행률: 62.5%
```

**TO-BE**:
```
✅ Phase 6: ... (100%)
✅ Phase 7: ... (100%)
✅ Phase 8: ... (100%)
전체 진행률: 100%
```

### 2.2 HANDOFF.md 변경

#### Change 5: Section 8.1 현황 업데이트
**AS-IS** (line 199-200):
```markdown
- Phase 5~8 미완료 (v2_diagnostics 통합, pipeline/routers 마이그레이션, legacy core 제거)
- 통합 진행률 약 50%
```

**TO-BE**:
```markdown
- Phase 1~8 **전체 완료** (2026-01-30 최종 확인)
- 통합 진행률 **100%**
```

#### Change 6: Section 9.1 단기 로드맵 업데이트
**AS-IS** (line 232):
```markdown
- Phase 5~7 엔진 통합 완료 (v2_diagnostics, pipeline, legacy routers)
```

**TO-BE**:
```markdown
- ~~Phase 5~7 엔진 통합~~ → **완료** (Phase 1~8 전체 완료)
- 대형 파일 리팩토링 (analyzer.py 1,436줄, v7.py 2,670줄, inspection_service.py 1,233줄)
```

#### Change 7: Section 14 다음 작업 업데이트
**AS-IS** (line 311):
```markdown
1. **엔진 통합 Phase 5~7 완료**
```

**TO-BE**:
```markdown
1. ~~엔진 통합 Phase 5~7~~ → ✅ 완료
```

---

## 3. Success Criteria

- [ ] INTEGRATION_STATUS.md Phase 6-8 모두 "완료"로 표시
- [ ] 전체 진행률 100% 표시
- [ ] HANDOFF.md Phase 5~8 관련 내용 수정
- [ ] 문서 내용과 코드 상태 100% 일치

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-01-31 | Initial design | PDCA Auto |
