# 프로젝트 구조 분석 및 통합 방안

## 현재 문제점

**2개의 core 폴더가 병행 존재**
- `src/core/` (레거시 v1-v6)
- `lens_signature_engine_v7/core/` (신규 v7)

**결과**:
- 코드 중복
- 유지보수 복잡도 증가
- 기능 불일치 가능성

---

## 1. Core 폴더 비교

### src/core/ (레거시)

**구조**: 평면 구조 (단일 파일들)

```
src/core/
├── __init__.py
├── angular_profiler.py       # 각도별 프로파일 분석
├── background_masker.py       # 배경 마스킹
├── boundary_detector.py       # 경계선 검출
├── color_evaluator.py         # 색상 평가 (메인)
├── illumination_corrector.py  # 조명 보정
├── image_loader.py            # 이미지 로딩
├── ink_estimator.py           # 잉크 추정
├── lens_detector.py           # 렌즈 검출 (원형)
├── quality_metrics.py         # 품질 지표
├── radial_profiler.py         # 반경 프로파일
├── sector_segmenter.py        # 섹터 분할
├── zone_analyzer_2d.py        # 2D 존 분석 (85KB - 대형)
└── zone_segmenter.py          # 존 분할
```

**특징**:
- 모놀리식 구조
- 파일 크기 큼 (zone_analyzer_2d.py 85KB)
- 의존성 높음

**사용처**:
- `src/pipeline.py` (메인 파이프라인)
- `src/web/routers/sku.py`
- 레거시 API 엔드포인트

### lens_signature_engine_v7/core/ (신규)

**구조**: 모듈식 구조 (하위 폴더들)

```
lens_signature_engine_v7/core/
├── __init__.py
├── config_loader.py
├── model_registry.py
├── reason_codes.py
├── types.py
├── utils.py
├── anomaly/               # 이상 패턴 검출
│   ├── angular_uniformity.py
│   ├── anomaly_score.py
│   ├── blob_detector.py
│   ├── defect_classifier.py
│   ├── heatmap.py
│   └── pattern_baseline.py
├── calibration/           # 보정 (미래 확장)
├── decision/              # 판정 로직
│   ├── decision_builder.py
│   ├── decision_engine.py
│   └── uncertainty.py
├── gate/                  # 게이트 검사 (blur, illum 등)
│   └── gate_engine.py
├── geometry/              # 기하 검출
│   └── lens_geometry.py
├── insight/               # v3 분석 (요약, 트렌드)
│   ├── summary.py
│   └── trend.py
├── measure/               # 측정 (잉크 분석 핵심)
│   ├── color_masks.py         ⭐ Engine B
│   ├── ink_grouping.py
│   ├── ink_match.py
│   ├── ink_metrics.py
│   ├── ink_segmentation.py
│   ├── preprocess.py
│   ├── threshold_policy.py
│   ├── v2_diagnostics.py      ⚠️ Engine A (교체 대상)
│   └── v2_flags.py
├── mode/                  # 모드 추적 (LOW/MID/HIGH)
│   └── mode_tracker.py
├── pipeline/              # 파이프라인
│   ├── analyzer.py            ⚠️ 부분 미통합
│   └── single_analyzer.py     ✅ Phase 1 완료
└── signature/             # 서명 (radial profile)
    ├── fit.py
    ├── model_io.py
    ├── radial_signature.py
    ├── segment_k_suggest.py
    ├── signature_compare.py
    └── std_model.py
```

**특징**:
- 관심사 분리 (SoC)
- 테스트 용이
- 확장 가능

**사용처**:
- `lens_signature_engine_v7/scripts/` (모든 스크립트)
- `src/web/routers/v7.py` (subprocess로 실행)

---

## 2. 사용 현황 맵

```
┌─────────────────────────────────────────────────┐
│                  Web API                        │
└─────────────────────────────────────────────────┘
           │                      │
           ▼                      ▼
    ┌─────────────┐        ┌──────────────┐
    │ 레거시 API   │        │  v7 API      │
    │ (inspection,│        │  (v7 router) │
    │  sku, std)  │        │              │
    └─────────────┘        └──────────────┘
           │                      │
           ▼                      ▼ (subprocess)
    ┌─────────────┐        ┌──────────────────────┐
    │src/pipeline │        │v7/scripts/           │
    │   .py       │        │run_signature_engine  │
    └─────────────┘        └──────────────────────┘
           │                      │
           ▼                      ▼
    ┌─────────────┐        ┌──────────────────────┐
    │ src/core/   │        │lens_signature_engine │
    │  (레거시)   │        │_v7/core/             │
    │             │        │  (신규)              │
    └─────────────┘        └──────────────────────┘
```

**병렬 동작**: 두 엔진이 독립적으로 존재

---

## 3. 중복 기능 분석

### 3.1 렌즈 검출

| 기능 | src/core/ | v7/core/ |
|------|-----------|----------|
| 원형 검출 | `lens_detector.py::LensDetector` | `geometry/lens_geometry.py::detect_lens_circle` |
| 구현 방식 | Class 기반 | Function 기반 |
| 사용 | pipeline.py | analyzer.py, single_analyzer.py |

**중복도**: 🔴 높음 (핵심 기능 동일)

### 3.2 품질 게이트

| 기능 | src/core/ | v7/core/ |
|------|-----------|----------|
| Blur 검사 | `quality_metrics.py` | `gate/gate_engine.py` |
| 조명 검사 | `illumination_corrector.py` | `gate/gate_engine.py` |
| 사용 | pipeline.py | analyzer.py |

**중복도**: 🟡 중간 (로직 유사)

### 3.3 잉크 분석

| 기능 | src/core/ | v7/core/ |
|------|-----------|----------|
| 색상 클러스터링 | `ink_estimator.py` | `measure/color_masks.py` (Engine B) |
| k-means | `color_evaluator.py` | `measure/ink_segmentation.py` |
| 사용 | pipeline.py | single_analyzer.py, analyzer.py |

**중복도**: 🔴 높음 (핵심 로직)

### 3.4 반경 프로파일

| 기능 | src/core/ | v7/core/ |
|------|-----------|----------|
| Radial 추출 | `radial_profiler.py` | `signature/radial_signature.py` |
| Polar 변환 | `radial_profiler.py` | `signature/radial_signature.py::to_polar` |

**중복도**: 🔴 높음

### 3.5 존 분석

| 기능 | src/core/ | v7/core/ |
|------|-----------|----------|
| 2D 존 | `zone_analyzer_2d.py` (85KB) | ❌ 없음 |
| 섹터 분할 | `sector_segmenter.py` | `signature/` (segments) |

**중복도**: 🟢 낮음 (v7에 2D 존 없음)

---

## 4. 미사용 코드 후보

### src/core/ (레거시)

**미사용 가능성**:
- `background_masker.py` (v7은 preprocess.py에서 ROI 마스크 사용)
- `boundary_detector.py` (v7은 geometry/lens_geometry 사용)
- `angular_profiler.py` (v7은 anomaly/angular_uniformity 사용)

**확인 필요**: import/호출 여부를 확정해야 삭제 가능

**검증 절차(권장 순서)**:
1. 정적 import 확인
   - `rg "from src\.core\.(background_masker|boundary_detector|angular_profiler)" -g"*.py"`
2. 동적 import/문자열 호출 확인
   - `rg "background_masker|boundary_detector|angular_profiler" -g"*.py"`
3. 파이프라인/라우터 호출 경로 확인
   - `src/pipeline.py`, `src/web/routers/`에서 직접 호출이 없는지 확인
4. 사용 여부 기록
   - 사용 없음 확인 시 "제거 가능" 표기, 있으면 "마이그레이션 필요" 표기

**검증 결과(현재 코드 기준)**:
- `src/core/sector_segmenter.py`에서 3개 모듈 모두 직접 사용
- `src/web/app.py`에서 `SectorSegmenter`가 사용되어 런타임 경로 존재
- `src/analysis/uniformity_analyzer.py`, `src/utils/telemetry.py`에서 `angular_profiler` 타입 사용
- 관련 테스트 및 도구(`tests/`, `tools/check_imports.py`)에서도 참조됨

**판정**: 삭제 후보가 아니라 "사용 중(마이그레이션 필요)"로 분류

### lens_signature_engine_v7/core/

**미사용 확정**:
- `calibration/` (폴더만 존재, 파일 없음)

**Phase 3 후 제거 대상**:
- `measure/v2_diagnostics.py` (Engine A → Engine B 통합 후)

---

## 5. 통합 전략

### 전략 A: v7을 메인으로 (권장)

**방향**: `src/core/` 레거시 제거, v7을 유일한 엔진으로

**단계**:

#### 5.1 Phase 3 완료 (우선)
**목표**: Engine A(legacy 진단) 기능을 Engine B 파이프라인에 흡수

**구체 작업(초안)**:
1. `measure/v2_diagnostics.py` 기능 목록화
   - 입력/출력 타입, 내부 단계, 사용되는 임계값/정책 정리
2. Engine B 진단 지점 정의
   - `pipeline/analyzer.py` 또는 `measure/` 계층 어디에서 진단을 수행할지 결정
3. 결과 포맷 통합
   - 기존 출력 스키마(v2) vs 신규 출력 스키마(B) 비교, 필드 매핑 정의
4. 임계값/정책 이관
   - `threshold_policy.py`와 충돌 여부 확인 및 단일 정책으로 정리
5. 회귀 비교
   - 동일 입력에 대해 v2 vs B 진단 결과 비교 스냅샷 확보
6. 문서 갱신
   - ENGINE_UNIFICATION_STATUS.md 체크리스트 완료

**v2_diagnostics 기능 목록(초안)**:
- 렌즈 검출 + polar 변환: `detect_lens_circle`, `to_polar`
- ROI 마스크/샘플링: `build_roi_mask`, `sample_ink_candidates`
- k-means 분할: `kmeans_segment` (L-weight, attempts, seed)
- 클러스터 통계/품질: `build_cluster_stats`, `min_deltaE`, `min_area_ratio`, `separation_margin`
- 클러스터 보강: `radial_presence_curve`, `spatial_prior`, `inkness_score`
- auto-k 추정: silhouette proxy 기반, confidence 계산, mismatch 경고
- 팔레트/색상: mean_lab(CV8) + CIE 변환 팔레트 제공
- 경고 체계: sampling/segmentation/auto-k 경고 문자열

**업그레이드/추가 가치 후보(Engine B에 흡수 가능)**:
- auto-k 추정 및 confidence 스코어링 로직
- `radial_presence_curve` 산출을 ROI 샘플 기반에서 전체 ROI 기반으로 고도화
- `inkness_score`/`spatial_prior`를 일관된 품질지표로 표준화
- `separation_margin`/`min_area_ratio`를 품질 게이트에 포함

#### 5.2 src/pipeline.py 마이그레이션
```python
# Before (레거시)
from src.core.lens_detector import LensDetector
from src.core.color_evaluator import ColorEvaluator
from src.core.radial_profiler import RadialProfiler

# After (v7)
from lens_signature_engine_v7.core.geometry.lens_geometry import detect_lens_circle
from lens_signature_engine_v7.core.measure.color_masks import build_color_masks_with_retry
from lens_signature_engine_v7.core.signature.radial_signature import to_polar, build_radial_signature
from lens_signature_engine_v7.core.pipeline.analyzer import evaluate
```

**작업량**: 중간 (pipeline.py 리팩토링 필요)
**핵심 확인 포인트**:
- 입력 포맷 차이(이미지, ROI, 메타데이터)
- 출력 포맷 차이(결과 구조, reason codes)
- 예외/에러 처리 방식 차이

#### 5.3 레거시 API 라우터 업데이트
- `src/web/routers/inspection.py` → v7 엔진 호출
- `src/web/routers/std.py` → v7 엔진 호출
- `src/web/routers/sku.py` → v7 엔진 호출

**작업량**: 소 (import 변경 위주)

#### 5.4 src/core/ 제거
- 모든 마이그레이션 완료 후
- 레거시 파일들 삭제

**이득**:
- 단일 코드베이스
- 중복 제거
- 유지보수 간소화

---

### 전략 B: 기능별 선택적 통합

**방향**: src/core/의 일부 기능만 v7로 이동

**대상**:
- `zone_analyzer_2d.py` (v7에 없는 기능, 85KB 대형 파일)
- `zone_segmenter.py`

**작업**:
- v7/core/zones/ 폴더 생성
- 위 파일들 이동 및 v7 구조에 맞게 리팩토링

**이득**: zone 기능을 v7에도 제공

**단점**: 여전히 일부 중복 유지

---

### 전략 C: 점진적 병합

**1차**: Phase 3 완료 (잉크 분석 완전 통합)
**2차**: 공통 모듈 통합 (lens_detector, quality_metrics)
**3차**: pipeline.py 리팩토링
**4차**: 레거시 API 라우터 마이그레이션
**5차**: src/core/ 제거

**이득**: 리스크 분산
**단점**: 시간 소요

---

## 6. 권장 로드맵

### 즉시 (Phase 3)
1. ✅ `v2_diagnostics.py` → Engine B 통합
2. ✅ smoke_tests.py 검증
3. ✅ 커밋

**토큰**: 약 20-30K (문서 참고로 절약)

### 단기 (1-2주)
4. src/pipeline.py 분석
   - 레거시 core 사용처 목록화
   - v7 대응 함수 매핑 테이블 작성

5. 마이그레이션 우선순위 결정
   - 핵심 기능 먼저 (lens_detector, color_evaluator)
   - 주변 기능 나중 (angular_profiler, boundary_detector)

**토큰**: 약 50-80K (구조 분석)

### 중기 (2-4주)
6. src/pipeline.py 리팩토링
   - v7 엔진으로 전환
   - 기존 API 호환성 유지

7. 레거시 API 라우터 마이그레이션
   - inspection.py, std.py, sku.py
   - v7 엔진 사용

**토큰**: 약 100-150K (구현 + 테스트)

### 장기 (1-2개월)
8. zone_analyzer_2d.py 마이그레이션 또는 폐기
   - 사용 빈도 확인
   - 필요 시 v7/core/zones/ 생성

9. src/core/ 제거
   - 모든 마이그레이션 완료 후
   - 레거시 폴더 삭제

10. 문서화
    - 새 구조 README 작성
    - API 변경사항 기록

---

## 7. 체크리스트 (Phase 3 이후)

### src/core/ 파일별 상태

- [ ] angular_profiler.py → v7 대응: `anomaly/angular_uniformity.py`
- [ ] background_masker.py → v7 대응: `measure/preprocess.py::build_roi_mask`
- [ ] boundary_detector.py → v7 대응: `geometry/lens_geometry.py`
- [ ] color_evaluator.py → v7 대응: `measure/color_masks.py` + `pipeline/analyzer.py`
- [ ] illumination_corrector.py → v7 대응: `gate/gate_engine.py` (일부)
- [ ] image_loader.py → v7 대응: cv2 직접 사용
- [ ] ink_estimator.py → v7 대응: `measure/color_masks.py`
- [ ] lens_detector.py → v7 대응: `geometry/lens_geometry.py`
- [ ] quality_metrics.py → v7 대응: `gate/gate_engine.py` + `measure/ink_metrics.py`
- [ ] radial_profiler.py → v7 대응: `signature/radial_signature.py`
- [ ] sector_segmenter.py → v7 대응: `signature/` (segments)
- [ ] zone_analyzer_2d.py → v7 대응: ❌ 없음 (마이그레이션 필요?)
- [ ] zone_segmenter.py → v7 대응: ❌ 없음 (pipeline.py, sku_manager.py에서 사용 중)

### 마이그레이션 진행률
- ✅ Engine B (잉크 분석 핵심): Phase 1-2 완료
- ⏳ Engine A → B 통합: Phase 3 진행 중 (구조 리팩토링 완료)
- ✅ 고급 분석 이식 (Phase 4): `src/analysis` → `v7/core` 이식 완료 (Uniformity, Profile)
- ⬜ src/pipeline.py: 미착수
- ⬜ 레거시 API 라우터: 미착수
- ⬜ src/core/ 제거: 미착수

**예상 완료율**: 현재 40% (고급 분석 이식 완료)

---

## 7.1 현황 정리 (2026-01-12 업데이트)
- **Phase 3 (잉크 분석)**: `core/measure` 폴더 구조 리팩토링 완료. `v2_diagnostics.py` 로직 교체 대기 중.
- **Phase 4 (고급 분석)**: `src/analysis`의 핵심 로직을 `v7`으로 이식 및 테스트 완료.
    - `UniformityAnalyzer` → `v7/core/measure/metrics/uniformity.py`
    - `ProfileAnalyzer` → `v7/core/signature/profile_analysis.py`
- **테스트**: `v7/tests/test_uniformity.py`, `v7/tests/test_profile_analysis.py` 통과.
- **잔여 과제**: `src/web` 및 `src/pipeline.py`가 구형 `src/analysis` 대신 신규 `v7` 모듈을 사용하도록 수정 완료.
- **파일 정리**: `src/analysis` 폴더 삭제 완료.

---

## 8. 즉시 실행 가능한 정리 작업

### 8.1 미사용 파일 삭제 (안전)

```bash
# lens_signature_engine_v7/
rm -rf core/calibration/  # 빈 폴더
```

### 8.2 명명 정리

**현재 혼란**:
- `v2_diagnostics.py` (v2가 뭔지 불명확)
- `v2_flags.py`
- `v3_summary.py` (v3?)

**제안**:
- `v2_diagnostics.py` → `ink_diagnostics.py` (Phase 3 후)
- `v2_flags.py` → `ink_flags.py`
- `v3_summary.py` → `inspection_summary.py`

### 8.3 폴더 구조 평탄화 (선택)

**현재**: `lens_signature_engine_v7/core/measure/`에 10개 파일

**제안**: 하위 폴더로 분리
```
measure/
├── segmentation/
│   ├── color_masks.py
│   ├── ink_segmentation.py
│   └── preprocess.py
├── metrics/
│   ├── ink_metrics.py
│   └── threshold_policy.py
└── diagnostics/
    ├── ink_diagnostics.py  # 구 v2_diagnostics
    └── ink_flags.py        # 구 v2_flags
```

---

## 9. 토큰 절약 팁

### 다음 작업 시:
1. ✅ ENGINE_UNIFICATION_STATUS.md 먼저 읽기
2. ✅ 이 문서 (PROJECT_STRUCTURE_ANALYSIS.md) 참고
3. 파일 읽기 최소화
   - Grep으로 import 확인
   - 필요한 함수만 Read
4. 병렬 작업
   - 여러 파일 동시 수정 가능하면 한 번에
5. 테스트 계획
   - smoke_tests.py 먼저
   - 실패 시에만 디버깅

---

## 10. 최종 목표 아키텍처

```
Color_meter/
├── lens_signature_engine_v7/    ⭐ 유일한 엔진
│   ├── core/
│   │   ├── anomaly/
│   │   ├── decision/
│   │   ├── gate/
│   │   ├── geometry/
│   │   ├── insight/
│   │   ├── measure/
│   │   │   ├── segmentation/
│   │   │   ├── metrics/
│   │   │   └── diagnostics/
│   │   ├── mode/
│   │   ├── pipeline/
│   │   ├── signature/
│   │   └── zones/         (zone_analyzer_2d 이동)
│   └── scripts/
├── src/
│   ├── pipeline.py        (v7 엔진 사용)
│   ├── schemas/
│   ├── services/
│   ├── utils/
│   └── web/
│       └── routers/       (모두 v7 엔진 사용)
└── [src/core/ 삭제됨]     ✅ 레거시 제거
```

**이득**:
- 단일 코드베이스
- 명확한 책임 분리
- 테스트/유지보수 용이
- 토큰 사용량 감소 (중복 코드 제거)

---

## 11. 결정 기준 (추가)

### 11.1 zone_analyzer_2d.py 마이그레이션 vs 폐기
**마이그레이션 조건**:
- 실제 사용 빈도 높음 (실행 로그/요청 비율 기준)
- 고객/검사 스펙에서 2D 존 결과가 필수
- v7 결과로 대체 불가하거나 성능 저하가 명확

**폐기 조건**:
- 최근 N개월 동안 호출/사용 없음
- v7 결과로 충분히 대체 가능
- 유지보수 비용 대비 가치 낮음

**검증 방법(예시)**:
- `rg "zone_analyzer_2d" -g"*.py"`로 호출 경로 확인
- 최근 배치/로그에서 기능 사용 여부 확인(가능 시)

**현재 사용처(코드 기준)**:
- `src/web/app.py` (inspect/recompute/inspect_v2 경로)
- `src/core/quality_metrics.py` (InkMaskConfig, build_ink_mask 사용)
- `tools/`, `tests/`에서도 참조됨

### 11.2 pipeline.py 마이그레이션 인터페이스 체크리스트
- 입력: 이미지 로딩/전처리 경로가 동일한가
- 파라미터: 레거시 옵션/플래그가 v7에 존재하는가
- 출력: 결과 키/필드가 API 응답 스키마와 호환되는가
- 에러: 실패 시 반환 포맷이 동일한가

### 11.3 Engine B 매핑 (v2_diagnostics ↔ v7/measure)
**이미 v7에 존재(재사용 가능)**:
- 렌즈 검출/Polar 변환: `geometry.lens_geometry.detect_lens_circle`, `signature.radial_signature.to_polar`
- ROI 마스크/샘플링: `measure.preprocess.build_roi_mask`, `sample_ink_candidates`, `build_sampling_mask`
- k-means 분할: `measure.ink_segmentation.kmeans_segment`
- 클러스터 통계/품질: `measure.ink_metrics.build_cluster_stats`
- 분리도/델타E: `build_cluster_stats.quality` + `pairwise_deltaE`

**v7에 있으나 연결되지 않음(통합 후보)**:
- radial presence curve: `measure.ink_metrics.calculate_radial_presence_curve`
- spatial prior: `measure.ink_metrics.calculate_spatial_prior`
- inkness score: `measure.ink_metrics.calculate_inkness_score` (현재 `color_masks`에서 중립값 사용)

**v7에 없음/부족(업그레이드 후보)**:
- auto-k 추정 + confidence 스코어링 (v2_diagnostics의 `auto_estimation`)
- separation_margin 계산(현재 min_deltaE만 제공)
- 경고 체계 통합(샘플링/세그먼트/auto-k 경고를 단일 스키마로)

**정리 방향(제안)**:
- `color_masks.py`에 `radial_presence_curve`/`spatial_prior` 실제 연결
- `build_cluster_stats` 출력에 `separation_margin` 추가
- `build_color_masks_with_retry`와 v2 `auto_k` 로직을 비교 후 하나로 정리

### 11.4 통합 설계 초안 (저위험 우선)
**A. `color_masks.py`에 spatial_prior 연결**
- `label_map`(T,R)과 `polar_r`(T,R, 0~1)로 `calculate_radial_presence_curve` 계산
- `calculate_spatial_prior`로 prior 산출 후 `calculate_inkness_score`에 주입
- 초기엔 `v2_ink.radial_bins`(없으면 10)만 사용, 나머지 기본값 유지

**B. separation_margin 추가**
- `build_cluster_stats`의 `quality`에 `separation_margin` 추가
  예: `(min_deltaE - separation_d0) / max(separation_k, 1e-6)`
- 파라미터는 `v2_ink.separation_d0`, `v2_ink.separation_k` 사용 (없으면 기본값)

**C. auto-k 로직 통합(보수적 적용)**
- 기존 `build_color_masks_with_retry` 유지하되,
  `auto_k_enabled`가 켜진 경우에만 후보 k 탐색(silhouette proxy)
- 후보 k는 `[k-1, k, k+1]` 기본 + 경고 조건 시 확장
- confidence 낮으면 경고 추가만 하고, k 강제 변경은 Phase 3 후반에 결정

### 11.5 경고 스키마 통합 초안
**목표**: 샘플링/세그먼트/auto-k 경고를 단일 schema로 통합해 파이프라인/보고서에서 일관 사용

**권장 카테고리**:
- `sampling`: 샘플링 품질/수량 경고
- `segmentation`: k-means 품질/클러스터 분리 경고
- `auto_k`: 자동 k 추정 신뢰도 경고

**현재 경고 → 카테고리 매핑(초안)**:
- `INK_SAMPLING_EMPTY` → `sampling`
- `INK_SEPARATION_LOW_CONFIDENCE` → `sampling`
- `COLOR_SEGMENTATION_FAILED` → `segmentation`
- `INK_CLUSTER_TOO_SMALL` → `segmentation`
- `INK_CLUSTER_OVERLAP_HIGH` → `segmentation`
- `AUTO_K_LOW_CONFIDENCE` → `auto_k`
- `INK_COUNT_MISMATCH_SUSPECTED` → `auto_k`

**적용 방식(저위험)**:
- 기존 `warnings: List[str]` 유지
- `warnings_by_category: Dict[str, List[str]]`를 추가로 제공

---

## 12. measure 모듈 분리 설계안(초안)

### 12.1 제안 폴더 구조
```
core/measure/
├── segmentation/
│   ├── ink_segmentation.py
│   ├── preprocess.py
│   └── color_masks.py
├── metrics/
│   ├── ink_metrics.py
│   ├── angular_metrics.py
│   └── threshold_policy.py
├── diagnostics/
│   ├── v2_diagnostics.py
│   └── v2_flags.py
├── matching/
│   ├── ink_match.py
│   └── assignment_map.py
├── baselines/
│   └── ink_baseline.py
└── ink_grouping.py
```

### 12.2 임포트 변경 표(초안)
| 기존 경로 | 변경 경로 |
|---|---|
| `core.measure.color_masks` | `core.measure.segmentation.color_masks` |
| `core.measure.ink_segmentation` | `core.measure.segmentation.ink_segmentation` |
| `core.measure.preprocess` | `core.measure.segmentation.preprocess` |
| `core.measure.ink_metrics` | `core.measure.metrics.ink_metrics` |
| `core.measure.angular_metrics` | `core.measure.metrics.angular_metrics` |
| `core.measure.threshold_policy` | `core.measure.metrics.threshold_policy` |
| `core.measure.v2_diagnostics` | `core.measure.diagnostics.v2_diagnostics` |
| `core.measure.v2_flags` | `core.measure.diagnostics.v2_flags` |
| `core.measure.ink_match` | `core.measure.matching.ink_match` |
| `core.measure.assignment_map` | `core.measure.matching.assignment_map` |
| `core.measure.ink_baseline` | `core.measure.baselines.ink_baseline` |
| `core.measure.ink_grouping` | `core.measure.ink_grouping` (유지) |

### 12.3 영향 범위(우선 스캔 기준)
- `lens_signature_engine_v7/scripts/` (train/register 스크립트)
- `lens_signature_engine_v7/tests/` (measure 관련 테스트 전반)
- `src/web/routers/v7.py`
- `lens_signature_engine_v7/core/decision/decision_builder.py`

### 12.4 적용 순서(권장)
1. 폴더 생성 및 파일 이동
2. 내부 상대 import 수정(먼저 core 내부)
3. 외부 import 수정(scripts/tests/web)
4. 정적 검사(`rg "core\\.measure"`로 잔여 경로 확인)
