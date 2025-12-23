# 테스트 커버리지 리포트

**생성일**: 2025-12-15
**전체 커버리지**: 60%
**테스트 통과율**: 271/275 (98.5%)

---

## 📊 요약

| 지표 | 값 | 목표 | 상태 |
|------|-----|------|------|
| **전체 커버리지** | 60% | 70% | ⚠️ -10% |
| **총 코드 라인** | 4,395 lines | - | - |
| **테스트된 라인** | 2,627 lines | - | - |
| **미테스트 라인** | 1,768 lines | - | ⚠️ |
| **총 테스트** | 319 tests | - | ✅ |
| **통과 테스트** | 271 tests | - | ✅ |
| **실패 테스트** | 4 tests | 0 | ⚠️ |
| **스킵 테스트** | 44 tests | - | ℹ️ |

---

## 🟢 높은 커버리지 모듈 (90%+)

| 모듈 | 커버리지 | Stmts | Miss | 상태 |
|------|----------|-------|------|------|
| **radial_profiler.py** | 100% | 79 | 0 | ✅ Perfect |
| **visualizer.py** | 98% | 202 | 4 | ✅ Excellent |
| **angular_profiler.py** | 96% | 101 | 4 | ✅ Excellent |
| **uniformity_analyzer.py** | 95% | 131 | 6 | ✅ Excellent |
| **boundary_detector.py** | 95% | 99 | 5 | ✅ Excellent |
| **profile_analyzer.py** | 91% | 105 | 9 | ✅ Excellent |

### 분석
핵심 분석 모듈들이 90% 이상의 높은 커버리지를 유지하고 있습니다. 특히 radial_profiler.py는 100% 커버리지를 달성했습니다.

---

## 🟡 보통 커버리지 모듈 (70-89%)

| 모듈 | 커버리지 | Stmts | Miss | 누락 라인 |
|------|----------|-------|------|-----------|
| **security.py** | 91% | 32 | 3 | 54, 87-88 |
| **image_utils.py** | 90% | 31 | 3 | 21, 23, 50 |
| **image_loader.py** | 89% | 114 | 13 | 45-47, 55-57, 63, 91, 98, 102, 105, 109, 138 |
| **color_space.py** | 89% | 28 | 3 | 126-129 |
| **lens_detector.py** | 87% | 174 | 22 | 59, 124, 139, 166, 193, 225-251, 315-316, 328-329 |
| **config_manager.py** | 87% | 30 | 4 | 13, 33-35 |
| **color_evaluator.py** | 86% | 175 | 24 | 33-49, 286-288, 373, 427-430, 475-481 |
| **file_io.py** | 83% | 36 | 6 | 17, 20-21, 24-25, 41 |
| **sku_manager.py** | 81% | 179 | 34 | 168, 171-175, 203, 227-229, 301-305, ... |
| **services/analysis_service.py** | 80% | 15 | 3 | 56-60 |
| **zone_segmenter.py** | 76% | 200 | 49 | 75, 83-85, 116-117, 120-129, 159-160, ... |
| **color_delta.py** | 74% | 108 | 28 | 46, 51, 169, 174, 213, 218, 233, 253-281 |

### 분석
유틸리티 및 보조 모듈들이 양호한 커버리지를 보이고 있습니다. 에러 처리 및 엣지 케이스 추가 테스트가 필요합니다.

---

## 🔴 낮은 커버리지 모듈 (70% 미만) - 개선 필요

| 모듈 | 커버리지 | Stmts | Miss | 우선순위 | 비고 |
|------|----------|-------|------|----------|------|
| **zone_analyzer_2d.py** | 35% | 728 | 476 | 🔴 Critical | 주요 분석 모듈 |
| **web/app.py** | 39% | 512 | 313 | ⚠️ High | Web API |
| **illumination_corrector.py** | 55% | 204 | 92 | ⚠️ High | PHASE7 신규 |
| **ink_estimator.py** | 57% | 230 | 100 | ⚠️ High | 잉크 분석 |
| **sector_segmenter.py** | 60% | 128 | 51 | 💡 Medium | 섹터 분할 |
| **background_masker.py** | 67% | 126 | 42 | 💡 Medium | 배경 마스킹 |
| **pipeline.py** | 68% | 212 | 68 | 💡 Medium | 파이프라인 |
| **camera.py** | 67% | 3 | 1 | ℹ️ Low | 간단한 모듈 |

### 상세 분석

#### 🔴 Critical: zone_analyzer_2d.py (35%)

**현황**:
- 728 statements 중 476 미테스트 (65% 미커버)
- 시스템의 핵심 2D 분석 엔진

**주요 누락 라인**:
- 32-40: `lab_to_rgb_hex()` 함수
- 133-142: 초기화 및 검증 로직
- 184-270: Polar transform 관련
- 406-409: 에러 처리
- 1191-1775: **analyze_lens_zones_2d()** 메인 함수 (585 라인!)

**문제점**:
- analyze_lens_zones_2d() 함수가 1400+ 라인으로 너무 큼
- 통합 테스트는 많으나 단위 테스트 부족
- 복잡한 로직 분기가 많아 테스트 케이스 작성 어려움

**개선 방안**:
1. **Priority 1**: analyze_lens_zones_2d() 함수 리팩토링
   ```python
   analyze_lens_zones_2d()
   ├── _prepare_polar_transform()
   ├── _detect_zone_boundaries()
   ├── _calculate_zone_statistics()
   ├── _perform_ink_analysis()
   ├── _evaluate_quality_metrics()
   └── _generate_judgment()
   ```
2. **Priority 2**: 각 하위 함수별 단위 테스트 작성
3. **Priority 3**: 엣지 케이스 및 에러 처리 테스트

**예상 효과**: 35% → 70% (목표 +35%)

---

#### ⚠️ High: web/app.py (39%)

**현황**:
- 512 statements 중 313 미테스트 (61% 미커버)
- FastAPI Web API endpoints

**주요 누락 라인**:
- 39-48: 초기화 코드
- 59-96: 에러 핸들러
- 165-173: 파일 업로드 처리
- 561-618: `/recompute` endpoint
- 638-677: `/compare` endpoint
- 712-797: 배치 처리 로직
- 891-1047: 헬퍼 함수들

**문제점**:
- API endpoint 테스트 부족 (통합 테스트 위주)
- 에러 처리 경로 미테스트
- 파일 업로드/다운로드 로직 미검증

**개선 방안**:
1. **Priority 1**: 각 endpoint별 단위 테스트
   - `test_web_api.py`에 9개 endpoints × 3 scenarios = 27 tests
2. **Priority 2**: 에러 처리 테스트
   - 400, 404, 500 에러 케이스
3. **Priority 3**: 파일 처리 테스트
   - 다양한 파일 포맷 (JPG, PNG, ZIP)

**예상 효과**: 39% → 75% (목표 +36%)

---

#### ⚠️ High: ink_estimator.py (57%)

**현황**:
- 230 statements 중 100 미테스트 (43% 미커버)
- GMM 기반 잉크 분석 엔진

**주요 누락 라인**:
- 39-58: `lab_to_rgb_hex()` (색상 변환)
- 177-181: 샘플링 에러 처리
- 185-196: 하이라이트 제거 로직
- 205-212: 크로마 필터링
- 232-267: **mixing correction 핵심 로직**
- 272-287: GMM 클러스터링 분기
- 434-453: 결과 포매팅

**문제점**:
- 핵심 알고리즘 (mixing correction) 미테스트
- 엣지 케이스 (빈 이미지, 단색 이미지 등) 미검증
- 현재 3개 테스트 실패 중

**개선 방안**:
1. **Priority 1**: 실패한 3개 테스트 수정
   - `test_sample_ink_pixels_basic`
   - `test_chroma_threshold_filtering`
   - `test_black_ink_preservation`
2. **Priority 2**: Mixing correction 테스트 추가
   - Collinear 케이스 (3→2 보정)
   - Non-collinear 케이스 (3 유지)
3. **Priority 3**: GMM 클러스터링 테스트
   - 단일 잉크 (k=1)
   - 2-3개 잉크 (k=2,3)
   - BIC 점수 검증

**예상 효과**: 57% → 85% (목표 +28%)

---

#### ⚠️ High: illumination_corrector.py (55%)

**현황**:
- 204 statements 중 92 미테스트 (45% 미커버)
- PHASE7에서 새로 추가된 조명 보정 모듈

**주요 누락 라인**:
- 149, 242, 252-253: 에러 처리
- 281, 339-340: 파라미터 검증
- 362-396: **polynomial correction 메서드** (35 라인)
- 409-421: **retinex correction 메서드** (13 라인)
- 436-468: **CLAHE correction 메서드** (33 라인)
- 483-514: **auto selection 로직** (32 라인)
- 530-549: 헬퍼 함수들

**문제점**:
- 3가지 보정 방법 모두 미테스트
- Auto selection 로직 미검증
- 통합 테스트만 존재 (12개), 단위 테스트 부족

**개선 방안**:
1. **Priority 1**: 각 보정 방법별 단위 테스트
   - Polynomial (degree 1-4)
   - Retinex (scale 파라미터 변화)
   - CLAHE (clip limit, grid size)
2. **Priority 2**: Auto selection 로직 테스트
   - 균일한 조명 → none
   - 그라디언트 → polynomial
   - 극심한 그라디언트 → retinex
3. **Priority 3**: 엣지 케이스 테스트
   - 이미 균일한 이미지
   - 극단적으로 어두운/밝은 이미지

**예상 효과**: 55% → 80% (목표 +25%)

---

## ⚪ 특수 모듈 (테스트 불필요/불가능)

| 모듈 | 커버리지 | 비고 |
|------|----------|------|
| **main.py** | 0% | CLI 진입점 (실행용) |
| **telemetry.py** | 0% | 미사용 모듈 (제거 검토) |

---

## 🧪 테스트 실패 분석

### 실패한 4개 테스트

#### 1. test_ink_estimator.py::test_sample_ink_pixels_basic

**에러**: AssertionError 또는 샘플링 로직 변경으로 인한 실패

**원인**:
- `sample_ink_pixels()` 함수가 최근 수정됨 (sampling_info 반환 추가)
- 테스트가 구 버전 API 기대

**해결 방안**:
```python
# 수정 전
samples = estimator.sample_ink_pixels(image)

# 수정 후
samples, sampling_info = estimator.sample_ink_pixels(image)
```

---

#### 2. test_ink_estimator.py::test_chroma_threshold_filtering

**에러**: 크로마 필터링 결과 불일치

**원인**:
- 크로마 임계값 로직 변경
- 테스트 데이터가 변경된 로직에 맞지 않음

**해결 방안**:
- 테스트 이미지 재생성 또는 임계값 조정
- Assertion 기대값 업데이트

---

#### 3. test_ink_estimator.py::test_black_ink_preservation

**에러**: 검정 잉크 샘플링 실패

**원인**:
- L_dark_thresh 파라미터 변경
- 검정 잉크 보존 로직 수정

**해결 방안**:
- L_dark_thresh 값 검증
- 검정 잉크 필터링 로직 재확인

---

#### 4. test_uniformity_analyzer.py::test_analyze_uniform_cells

**에러**: Assertion failure

**원인**:
- 최근 uniformity_analyzer.py 수정 (quartile 추가 등)
- 테스트가 구 버전 출력 형식 기대

**해결 방안**:
- 테스트 업데이트하여 새 데이터 스키마 반영

---

## 📈 개선 로드맵

### Phase 1: Critical (즉시 - 1주)

**목표**: 실패 테스트 수정 + 주요 모듈 60% → 70%

| 작업 | 예상 시간 | 예상 효과 |
|------|----------|----------|
| 1. 실패 테스트 4개 수정 | 2시간 | +1% (테스트 통과율 100%) |
| 2. ink_estimator.py 테스트 추가 | 6시간 | 57% → 85% (+28%) |
| 3. zone_analyzer_2d.py 리팩토링 시작 | 8시간 | 35% → 50% (+15%) |

**예상 전체 커버리지**: 60% → 65% (+5%)

---

### Phase 2: High Priority (2주)

**목표**: 주요 모듈 70% 이상 달성

| 작업 | 예상 시간 | 예상 효과 |
|------|----------|----------|
| 4. zone_analyzer_2d.py 완전 분할 | 12시간 | 50% → 70% (+20%) |
| 5. web/app.py API 테스트 | 10시간 | 39% → 75% (+36%) |
| 6. illumination_corrector.py 테스트 | 6시간 | 55% → 80% (+25%) |

**예상 전체 커버리지**: 65% → 72% (+7%)

---

### Phase 3: 최적화 (1개월)

**목표**: 전체 커버리지 80% 달성

| 작업 | 예상 시간 | 예상 효과 |
|------|----------|----------|
| 7. pipeline.py 통합 테스트 강화 | 6시간 | 68% → 85% (+17%) |
| 8. background_masker.py 테스트 | 4시간 | 67% → 85% (+18%) |
| 9. sector_segmenter.py 테스트 | 4시간 | 60% → 80% (+20%) |
| 10. color_delta.py 엣지 케이스 | 3시간 | 74% → 85% (+11%) |

**예상 전체 커버리지**: 72% → 80% (+8%)

---

## 🎯 액션 아이템

### 즉시 실행 (이번 주)

- [ ] **Task 1**: 실패 테스트 4개 수정 (2시간)
  - test_ink_estimator.py: 3개
  - test_uniformity_analyzer.py: 1개

- [ ] **Task 2**: ink_estimator.py 테스트 추가 (6시간)
  - Mixing correction 테스트 (collinear/non-collinear)
  - GMM 클러스터링 테스트 (k=1,2,3)
  - 엣지 케이스 (빈 이미지, 단색 등)

- [ ] **Task 3**: zone_analyzer_2d.py 리팩토링 계획 수립 (2시간)
  - 함수 분할 전략 문서화
  - 테스트 케이스 설계

### 다음 주 실행

- [ ] **Task 4**: zone_analyzer_2d.py 리팩토링 실행 (20시간)
  - 함수 분할 (6개 하위 함수)
  - 각 함수별 단위 테스트 작성
  - 통합 테스트 유지 확인

- [ ] **Task 5**: web/app.py API 테스트 추가 (10시간)
  - 9개 endpoints × 3 scenarios = 27 tests
  - 에러 처리 테스트
  - 파일 업로드 테스트

---

## 📊 커버리지 트렌드 (계획)

```
Week 0 (현재):  60% ████████████░░░░░░░░
Week 1 (Phase1): 65% █████████████░░░░░░░
Week 2 (Phase2): 72% ██████████████░░░░░░
Week 4 (Phase3): 80% ████████████████░░░░
```

---

## 🔗 관련 문서

- [IMPROVEMENT_PLAN.md](../planning/IMPROVEMENT_PLAN.md): 전체 개선 계획
- [PHASE7_COMPLETION_REPORT.md](planning/PHASE7_COMPLETION_REPORT.md): PHASE7 완료 리포트
- [pytest.ini](../pytest.ini): 테스트 설정
- [htmlcov/index.html](../reports/coverage/htmlcov/index.html): 상세 HTML 리포트

---

## 📞 문의 및 제안

커버리지 개선 관련 제안이나 질문이 있다면:
1. GitHub Issues에 등록
2. docs/planning/IMPROVEMENT_PLAN.md에 코멘트 추가
3. 팀 회의에서 논의

---

**생성**: 2025-12-15
**다음 업데이트**: Phase 1 완료 후 (1주 후)
