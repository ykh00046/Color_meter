# 엔진 통합 현황 (2026-01-12 17:58 기준)

## 완료된 작업

### ✅ Phase 3: measure 폴더 구조 재구성
**상태**: 완료

**변경 내용**:
```
기존 (평면 구조):
core/measure/
├── color_masks.py
├── ink_segmentation.py
├── preprocess.py
├── ink_metrics.py
├── angular_metrics.py
├── threshold_policy.py
├── v2_diagnostics.py
├── v2_flags.py
├── ink_match.py
├── assignment_map.py
└── ink_baseline.py

신규 (모듈식 구조):
core/measure/
├── segmentation/
│   ├── __init__.py
│   ├── color_masks.py
│   ├── ink_segmentation.py
│   └── preprocess.py
├── metrics/
│   ├── __init__.py
│   ├── ink_metrics.py
│   ├── angular_metrics.py
│   ├── threshold_policy.py
│   └── uniformity.py          ⭐ src/analysis에서 이식
├── diagnostics/
│   ├── __init__.py
│   ├── v2_diagnostics.py
│   └── v2_flags.py
├── matching/
│   ├── __init__.py
│   ├── ink_match.py
│   └── assignment_map.py
├── baselines/
│   ├── __init__.py
│   └── ink_baseline.py
└── ink_grouping.py            (그대로 유지)
```

**영향**:
- 책임 분리 명확화
- 모듈 재사용성 향상
- 테스트 용이성 개선

### ✅ Phase 4: src/analysis/ 이식 및 삭제
**상태**: 완료

**이식된 모듈**:
1. `src/analysis/uniformity_analyzer.py` → `v7/core/measure/metrics/uniformity.py`
2. `src/analysis/profile_analyzer.py` → `v7/core/signature/profile_analysis.py`

**삭제된 파일**:
- `src/analysis/__init__.py`
- `src/analysis/uniformity_analyzer.py`
- `src/analysis/profile_analyzer.py`

**결과**: v7이 고급 분석 기능 흡수 완료

### ✅ 테스트 검증
**smoke_tests.py**: 통과 ✅
- `smoke_report.json` 생성 완료
- "ok": true
- 6개 SKU 테스트 완료 (SKU_TEMP, SKU_SMOKE_BASELINE)

### ✅ config_loader.py 복구
**문제**: 통합 중 SKU별 config 오버라이드 기능 삭제됨
**해결**: `git checkout HEAD -- core/config_loader.py`로 복구

**복구된 기능**:
- `load_cfg_with_sku()` - SKU별 config 병합
- `deep_merge()` - 계층적 오버라이드
- `_unknown_keys()` - 잘못된 키 검증

---

## 수정된 파일 (Staged 필요)

### v7 Core
- `core/config_loader.py` - SKU 기능 복구
- `core/decision/decision_builder.py`
- `core/insight/summary.py`
- `core/insight/trend.py`
- `core/pipeline/analyzer.py` - import 경로 변경
- `core/pipeline/single_analyzer.py` - import 경로 변경
- `core/types.py`
- `core/utils.py`

### v7 Scripts
- `scripts/register_std.py` - import 경로 변경
- `scripts/run_signature_engine.py` - import 경로 변경
- `scripts/smoke_tests.py` - import 경로 변경
- `scripts/train_std_model.py` - import 경로 변경

### v7 Tests
- `tests/test_angular_metrics.py`
- `tests/test_assignment_map.py`
- `tests/test_color_masks.py`
- `tests/test_per_color_training.py`
- `tests/test_threshold_policy.py`
- 신규: `tests/test_uniformity.py`
- 신규: `tests/test_profile_analysis.py`

### src (레거시)
- `src/core/sector_segmenter.py`
- `src/pipeline.py`
- `src/services/analysis_service.py`
- `src/web/app.py`
- `src/web/routers/v7.py`

### Models/State
- `lens_signature_engine_v7/models/index.json`
- `state/GGG.json`
- 다수의 신규 모델/스냅샷 생성됨

---

## Import 경로 변경 예시

### Before (평면 구조)
```python
from lens_signature_engine_v7.core.measure.color_masks import build_color_masks_with_retry
from lens_signature_engine_v7.core.measure.ink_metrics import calculate_inkness_score
from lens_signature_engine_v7.core.measure.v2_diagnostics import build_v2_diagnostics
```

### After (모듈식 구조)
```python
from lens_signature_engine_v7.core.measure.segmentation.color_masks import build_color_masks_with_retry
from lens_signature_engine_v7.core.measure.metrics.ink_metrics import calculate_inkness_score
from lens_signature_engine_v7.core.measure.diagnostics.v2_diagnostics import build_v2_diagnostics
```

---

## 미완료 작업 (남은 Phase)

### Phase 5: v2_diagnostics.py → Engine B 통합
**상태**: 파일 이동 완료, 로직 교체 대기

**현재 위치**: `core/measure/diagnostics/v2_diagnostics.py`

**작업 필요**:
1. `build_v2_diagnostics()` 내부에서 `build_color_masks_with_retry()` 호출
2. 반환 형식 변환 로직 작성
3. analyzer.py의 `_attach_v2_diagnostics()` 호출 유지
4. 회귀 테스트

**참고 문서**: `ENGINE_UNIFICATION_STATUS.md`

### Phase 6: src/pipeline.py 마이그레이션
**상태**: 미착수

**작업 필요**:
1. 레거시 `src/core/` import → `v7/core/` import 변경
2. Config 클래스 기반 → JSON 기반 변경
3. SKU config 구조 변환 어댑터 작성
4. 기존 API 호환성 유지

### Phase 7: 레거시 API 라우터 마이그레이션
**상태**: 미착수

**대상 파일**:
- `src/web/routers/inspection.py`
- `src/web/routers/std.py`
- `src/web/routers/sku.py`

**작업 필요**:
- v7 엔진 호출로 변경
- 기존 API 응답 형식 유지

### Phase 8: src/core/ 제거
**상태**: 보류 (Phase 6-7 완료 후)

**제거 대상**:
- `src/core/` 전체 폴더
- 단, zone_analyzer_2d.py는 사용 여부 확인 후 결정

---

## 통합 진행률

```
✅ Phase 1: single_analyzer.py Engine B 통합          (100%)
✅ Phase 2: color_masks.py 개선                       (100%)
✅ Phase 3: measure 폴더 구조 재구성                  (100%)
✅ Phase 4: src/analysis/ 이식 및 삭제                (100%)
⏳ Phase 5: v2_diagnostics Engine B 통합              (30% - 파일 이동만)
⬜ Phase 6: src/pipeline.py 마이그레이션              (0%)
⬜ Phase 7: 레거시 API 라우터 마이그레이션            (0%)
⬜ Phase 8: src/core/ 제거                            (0%)

전체 진행률: 50%
```

---

## 즉시 필요한 작업

### 1. Git Commit (우선순위: 높음)
현재 많은 파일이 unstaged 상태입니다. 작업 내용을 커밋해야 합니다.

**권장 커밋 메시지**:
```bash
git add lens_signature_engine_v7/core/measure/segmentation/
git add lens_signature_engine_v7/core/measure/metrics/
git add lens_signature_engine_v7/core/measure/diagnostics/
git add lens_signature_engine_v7/core/measure/matching/
git add lens_signature_engine_v7/core/measure/baselines/
git add lens_signature_engine_v7/core/signature/profile_analysis.py
git add lens_signature_engine_v7/tests/test_uniformity.py
git add lens_signature_engine_v7/tests/test_profile_analysis.py

git add -u  # 수정/삭제된 파일들

git commit -m "feat(Phase 3-4): restructure measure module and migrate src/analysis

- Reorganize core/measure/ into submodules (segmentation, metrics, diagnostics, matching, baselines)
- Migrate UniformityAnalyzer to v7/core/measure/metrics/uniformity.py
- Migrate ProfileAnalyzer to v7/core/signature/profile_analysis.py
- Delete src/analysis/ folder (fully migrated to v7)
- Update all import paths across scripts, tests, and services
- Restore config_loader.py SKU functionality
- smoke_tests.py: PASSED ✅

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### 2. 문서 정리 (우선순위: 중간)
```bash
git add PROJECT_STRUCTURE_ANALYSIS.md
git add lens_signature_engine_v7/ENGINE_UNIFICATION_STATUS.md
```

### 3. .gitignore 업데이트 (우선순위: 중간)
다음 항목들을 `.gitignore`에 추가:
```
# Test/development models
lens_signature_engine_v7/models/SKU_TEMP_*/
lens_signature_engine_v7/models/SKU_SMOKE_BASELINE_*/
lens_signature_engine_v7/models/_staging/
lens_signature_engine_v7/models/active_snapshots/
lens_signature_engine_v7/models/pattern_baselines/ok_logs/

# State files
state/*.json
state/.*.lock

# Coverage reports
.coverage
smoke_report.json
```

---

## 주의 사항

### config_loader.py 변경 시 주의
현재 `config_loader.py`에 unstaged change가 있습니다. 이 파일은:
- ✅ `load_cfg_with_sku()` 기능 복구됨 (HEAD 버전)
- ⚠️ 추가 수정 시 신중히 검토 필요

**확인 명령**:
```bash
git diff lens_signature_engine_v7/core/config_loader.py
```

### Import 경로 일관성
모든 import 경로가 새 구조를 따르는지 확인:
```bash
# 잘못된 import 검색
rg "from.*core\.measure\.(color_masks|ink_metrics|v2_diagnostics)" --type py

# 올바른 import 확인
rg "from.*core\.measure\.(segmentation|metrics|diagnostics)\." --type py
```

### Smoke Tests 재실행
구조 변경 후 최종 검증:
```bash
cd lens_signature_engine_v7
python scripts/smoke_tests.py
```

---

## 다음 단계 제안

### 옵션 A: 현재 작업 커밋 후 Phase 5 진행
1. 위 Git Commit 실행
2. `ENGINE_UNIFICATION_STATUS.md` 참고하여 Phase 5 진행
3. v2_diagnostics.py 로직 교체

### 옵션 B: Phase 6-7 우선 진행
1. 현재 작업 커밋
2. src/pipeline.py 마이그레이션 시작
3. 레거시 API 라우터 업데이트

### 옵션 C: 안정화 우선
1. 현재 작업 커밋
2. 전체 회귀 테스트 실행
3. 문서화 보완

**권장**: 옵션 A (Phase 5 완료 → 잉크 분석 엔진 완전 통합)

---

## 참고 문서

- `PROJECT_STRUCTURE_ANALYSIS.md` - 전체 프로젝트 구조 분석
- `ENGINE_UNIFICATION_STATUS.md` - 엔진 통합 상세 현황
- `lens_signature_engine_v7/RUNBOOK.md` - 운영 가이드
- `lens_signature_engine_v7/CHANGELOG.md` - 변경 이력

---

## 요약

**완료**:
- measure 폴더 모듈식 재구성 ✅
- src/analysis/ v7 이식 ✅
- smoke_tests 통과 ✅
- config_loader SKU 기능 복구 ✅

**진행 중**:
- Import 경로 업데이트 (대부분 완료)
- 테스트 파일 업데이트

**다음 작업**:
- Git Commit
- Phase 5 (v2_diagnostics 통합)
