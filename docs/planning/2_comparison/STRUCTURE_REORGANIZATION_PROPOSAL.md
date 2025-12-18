# 프로젝트 구조 재구성 제안서

**작성일**: 2025-12-18
**목적**: 단일 분석(Inspection) vs 비교 시스템(Comparison) 명확한 분리를 위한 구조 개선

---

## 📊 현재 구조 분석

### 1. 코드 구조 (src/)

#### 현재 상태: **혼재**
```
src/
├── core/                    # 🔵 기존: 단일 분석 시스템
│   ├── lens_detector.py
│   ├── zone_segmenter.py
│   ├── radial_profiler.py
│   ├── color_evaluator.py
│   └── ... (13개 모듈)
│
├── models/                  # 🟢 신규: 비교 시스템 (DB)
│   ├── database.py
│   ├── std_models.py
│   ├── test_models.py
│   └── user_models.py
│
├── schemas/                 # 🟢 신규: 비교 시스템 (API)
│   ├── std_schemas.py
│   ├── comparison_schemas.py
│   └── judgment_schemas.py
│
├── services/                # 🔵 기존: analysis_service.py
│   └── analysis_service.py
│
├── web/                     # 🔵 기존: 웹 UI
├── utils/                   # 🔵 공통: 유틸리티
├── pipeline.py              # 🔵 기존: InspectionPipeline
├── sku_manager.py           # 🔵 기존: SKU 관리
└── visualizer.py            # 🔵 기존: 시각화
```

**문제점**:
- ✅ models/, schemas/ 신규 생성으로 이미 일부 분리됨
- ⚠️ 하지만 "어디에 새 코드를 넣어야 하는가"가 불명확
- ⚠️ services/ 폴더가 현재 기존 시스템 파일만 포함

---

### 2. 문서 구조 (docs/)

#### 현재 상태: **혼재**
```
docs/
├── planning/
│   ├── 🔵 PHASE7_CORE_IMPROVEMENTS.md              # 기존 시스템
│   ├── 🔵 ANALYSIS_UI_DEVELOPMENT_PLAN.md          # 기존 시스템
│   ├── 🟢 ROADMAP_REVIEW_AND_ARCHITECTURE.md       # 비교 시스템 ⭐
│   ├── 🟢 STD_BASED_QC_SYSTEM_PLAN.md              # 비교 시스템
│   ├── 🟢 TECHNICAL_ENHANCEMENTS_ADVANCED.md       # 비교 시스템
│   ├── 🟢 WEEK1_M0_READINESS_CHECKLIST.md          # 비교 시스템
│   ├── 🟢 JUDGMENT_CRITERIA_WORKSHOP.md            # 비교 시스템
│   └── ... (30+ 파일 혼재)
│
├── design/
│   ├── 🔵 PIPELINE_DESIGN.md                       # 기존 시스템
│   ├── 🔵 SKU_MANAGEMENT_DESIGN.md                 # 기존 시스템
│   ├── 🔵 VISUALIZER_DESIGN.md                     # 기존 시스템
│   └── 🔵 COLOR_EXTRACTION_DUAL_SYSTEM.md          # 기존 시스템
│
└── guides/
    ├── 🔵 USER_GUIDE.md                            # 기존 시스템
    ├── 🔵 DEPLOYMENT_GUIDE.md                      # 기존 시스템
    └── 🔵 WEB_UI_GUIDE.md                          # 기존 시스템
```

**문제점**:
- ❌ planning/ 폴더에 30+ 파일이 혼재 (기존 + 신규)
- ❌ 파일명만으로는 어느 시스템에 속하는지 불명확
- ❌ 신규 개발자가 "어떤 문서부터 읽어야 하는가" 혼란

---

## 💡 재구성 옵션 (3가지)

### ✅ **Option A: 문서만 분리 (권장)**

**변경 범위**: `docs/` 구조만 개선
**코드**: 현재 상태 유지 (models/, schemas/ 이미 분리됨)

#### 새 구조
```
docs/
├── planning/
│   ├── 1_inspection/                    # 🔵 기존 단일 분석 시스템
│   │   ├── PHASE7_CORE_IMPROVEMENTS.md
│   │   ├── ANALYSIS_UI_DEVELOPMENT_PLAN.md
│   │   ├── OPERATIONAL_UX_IMPROVEMENTS.md
│   │   └── archive/
│   │
│   ├── 2_comparison/                    # 🟢 신규 비교 시스템 ⭐
│   │   ├── ROADMAP_REVIEW_AND_ARCHITECTURE.md
│   │   ├── STD_BASED_QC_SYSTEM_PLAN.md
│   │   ├── TECHNICAL_ENHANCEMENTS_ADVANCED.md
│   │   ├── WEEK1_M0_READINESS_CHECKLIST.md
│   │   ├── JUDGMENT_CRITERIA_WORKSHOP.md
│   │   └── ROADMAP_COMPARISON_REVIEW.md
│   │
│   └── ACTIVE_PLANS.md                  # 공통 (루트에 유지)
│
├── design/
│   ├── inspection/                      # 🔵 기존
│   │   ├── PIPELINE_DESIGN.md
│   │   ├── SKU_MANAGEMENT_DESIGN.md
│   │   └── COLOR_EXTRACTION_*.md
│   │
│   └── comparison/                      # 🟢 신규 (향후)
│       └── (STD 비교 설계 문서 추가 예정)
│
└── guides/
    ├── inspection/                      # 🔵 기존
    │   ├── USER_GUIDE.md
    │   ├── DEPLOYMENT_GUIDE.md
    │   └── WEB_UI_GUIDE.md
    │
    └── comparison/                      # 🟢 신규 (Week 6+)
        └── STD_COMPARISON_GUIDE.md
```

#### 장점 ✅
1. **명확한 분리**: 문서만 봐도 어느 시스템인지 즉시 이해
2. **낮은 리스크**: 코드 변경 없음 (import 경로 유지)
3. **점진적 작업**: Week 1-2에 문서만 정리 가능
4. **신규 팀원 친화**: "2_comparison/ 폴더만 보면 됨"

#### 단점 ⚠️
1. 코드는 여전히 일부 혼재 (하지만 models/schemas 이미 분리됨)
2. docs/ INDEX.md 업데이트 필요

---

### Option B: 코드만 분리

**변경 범위**: `src/` 구조만 개선
**문서**: 현재 상태 유지

#### 새 구조
```
src/
├── inspection/              # 🔵 기존 단일 분석 시스템
│   ├── core/
│   │   ├── lens_detector.py
│   │   ├── zone_segmenter.py
│   │   └── ... (13개 모듈)
│   ├── pipeline.py
│   ├── sku_manager.py
│   ├── visualizer.py
│   └── services/
│       └── analysis_service.py
│
├── comparison/              # 🟢 신규 비교 시스템
│   ├── models/
│   │   ├── std_models.py
│   │   ├── test_models.py
│   │   └── user_models.py
│   ├── schemas/
│   │   ├── std_schemas.py
│   │   ├── comparison_schemas.py
│   │   └── judgment_schemas.py
│   └── services/            # Week 2-3 추가 예정
│       └── std_service.py
│
├── shared/                  # 공통
│   ├── utils/
│   └── data/
│
└── web/                     # 웹 UI (통합)
```

#### 장점 ✅
1. 코드 레벨 명확한 분리
2. import 경로로 시스템 구분 명확
   ```python
   from src.inspection.core import LensDetector
   from src.comparison.models import STDModel
   ```

#### 단점 ❌
1. **높은 리스크**: 모든 import 경로 변경 필요 (100+ 파일 수정)
2. **테스트 깨짐**: pytest 경로 수정 필수
3. **시간 소요**: 1-2일 작업 (Week 1 일정 압박)
4. **코드 충돌**: 진행 중인 작업과 conflict 가능성

---

### Option C: 코드 + 문서 모두 분리

**변경 범위**: Option A + Option B

#### 장점 ✅
1. 가장 명확한 분리
2. 장기적으로 유지보수 최적

#### 단점 ❌
1. **매우 높은 리스크**: 코드 + 문서 모두 변경
2. **시간 소모**: 2-3일 작업
3. **Week 1 일정 지연 우려**

---

## 🎯 권장 사항

### ✅ **Option A 채택 (문서만 분리)**

#### 이유
1. **Week 1 일정 보호**: 코드 변경 없어 리스크 최소
2. **즉시 효과**: 문서만 정리해도 명확도 크게 향상
3. **코드는 이미 부분 분리됨**: models/, schemas/ 신규 폴더 존재
4. **향후 확장 여지**: P1-P2에서 코드 분리 고려 가능

#### 실행 계획

**Week 1 Day 1 (오늘)**:
1. 문서 분류 (30분)
   - planning/ 파일 분류 (기존 vs 신규)
   - design/ 파일 분류
   - guides/ 파일 분류

2. 폴더 생성 및 이동 (30분)
   ```bash
   # 폴더 생성
   mkdir -p docs/planning/1_inspection
   mkdir -p docs/planning/2_comparison
   mkdir -p docs/design/inspection
   mkdir -p docs/design/comparison
   mkdir -p docs/guides/inspection
   mkdir -p docs/guides/comparison

   # 파일 이동 (git mv 사용)
   git mv docs/planning/ROADMAP_REVIEW_AND_ARCHITECTURE.md docs/planning/2_comparison/
   git mv docs/planning/STD_BASED_QC_SYSTEM_PLAN.md docs/planning/2_comparison/
   # ... (계속)
   ```

3. INDEX.md 업데이트 (20분)
   - 새 구조 반영
   - 1_inspection/, 2_comparison/ 섹션 분리

4. README.md 업데이트 (10분)
   - 문서 구조 설명 추가

**총 소요 시간**: 1.5시간

---

## 📋 파일 분류 목록

### Planning 폴더

#### 🟢 2_comparison/ (신규 비교 시스템)
- `ROADMAP_REVIEW_AND_ARCHITECTURE.md` ⭐ (실행 계획)
- `STD_BASED_QC_SYSTEM_PLAN.md` (장기 비전)
- `TECHNICAL_ENHANCEMENTS_ADVANCED.md` (P1-P2 기술)
- `WEEK1_M0_READINESS_CHECKLIST.md` (Week 1 체크리스트)
- `JUDGMENT_CRITERIA_WORKSHOP.md` (판정 기준 워크샵)
- `ROADMAP_COMPARISON_REVIEW.md` (로드맵 비교)
- `AI_REVIEW_CONTEXT_BUNDLE.md` (AI 검토용)
- `REVIEW_FEEDBACK_AND_IMPROVEMENTS.md` (검토 피드백)

#### 🔵 1_inspection/ (기존 단일 분석)
- `PHASE7_CORE_IMPROVEMENTS.md`
- `PHASE7_*.md` (모든 Phase 7 문서)
- `ANALYSIS_UI_DEVELOPMENT_PLAN.md`
- `OPERATIONAL_UX_IMPROVEMENTS.md`
- `INK_ANALYSIS_ENHANCEMENT_PLAN.md`
- `PHASE4_UI_OVERHAUL_PLAN.md`
- `ANALYSIS_IMPROVEMENTS.md`
- `CODE_QUALITY_REPORT.md`
- `REFACTORING_COMPLETION_REPORT.md`
- `QUICK_WINS_COMPLETE.md`
- 모든 `TEST_*.md`, `OPTION*.md`, `PRIORITY*.md`

#### 📌 루트 유지 (공통)
- `ACTIVE_PLANS.md` (SSOT)

---

## 🔄 코드 구조 (현재 상태 유지)

### 현재 상태: 이미 부분 분리됨 ✅

```
src/
├── core/                    # 🔵 기존 (공유 가능)
├── models/                  # 🟢 신규 비교 전용
├── schemas/                 # 🟢 신규 비교 전용
├── services/                # 🔵 기존 (Week 2에 std_service.py 추가 예정)
├── web/                     # 공통 (Week 4-6에 비교 UI 추가)
├── utils/                   # 공통
└── pipeline.py              # 🔵 기존 (비교 시스템도 재사용) ⭐
```

**Week 2-6 추가 예정**:
- `src/services/std_service.py` (Week 2-3)
- `src/services/comparison_service.py` (Week 4-6)
- `src/web/routers/std.py` (Week 2-3)
- `src/web/routers/comparison.py` (Week 4-6)

**코드 분리는 P1-P2에서 검토** (선택적):
- MVP 완료 후 필요성 재평가
- 현재 구조로도 충분히 명확

---

## ✅ 실행 여부 (의사결정 필요)

### 질문
1. **Option A (문서만 분리) 진행할까요?** ✅ 권장
   - 소요 시간: 1.5시간
   - 리스크: 낮음
   - Week 1 일정 영향: 없음

2. **아니면 현재 상태 유지?** (파일명 프리픽스만 추가)
   - 예: `[CMP] ROADMAP_*.md`, `[INS] PHASE7_*.md`
   - 소요 시간: 30분
   - 리스크: 매우 낮음

3. **Option B/C (코드 분리)?** ❌ 비권장 (Week 1에는)
   - P1-P2에서 재검토 권장

---

## 📊 결정 매트릭스

| 옵션 | 명확도 | 리스크 | 시간 | Week 1 영향 | 권장 |
|------|--------|--------|------|-------------|------|
| **Option A (문서 분리)** | ⭐⭐⭐⭐ | 낮음 | 1.5h | 없음 | ✅ **권장** |
| 현재 유지 + 프리픽스 | ⭐⭐ | 매우 낮음 | 0.5h | 없음 | 🟡 차선 |
| Option B (코드 분리) | ⭐⭐⭐⭐⭐ | 높음 | 1-2일 | 지연 | ❌ Week 1 비권장 |
| Option C (코드+문서) | ⭐⭐⭐⭐⭐ | 매우 높음 | 2-3일 | 지연 | ❌ Week 1 비권장 |

---

## 🎬 Next Action

**사용자 결정 필요**:
1. ✅ Option A (문서 분리) 진행? → 1.5시간 작업
2. 🟡 현재 유지 (프리픽스만)? → 30분 작업
3. ❓ 다른 의견?

**결정 후**:
- Week 1 M0 작업 계속 (Alembic, 판정 기준 워크샵)

---

**작성자**: Claude Sonnet 4.5
**검토 필요**: 사용자 의사결정
**Status**: ⏳ **결정 대기 중**
