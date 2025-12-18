# STD 기반 QC 시스템: 로드맵 검토 및 아키텍처 분리 설계

**작성일**: 2025-12-17
**목적**: 사용자 제안 로드맵 검토 및 단일 분석 vs 비교 시스템 아키텍처 분리

---

## 📋 Executive Summary

사용자가 제안한 **MVP 우선 로드맵**은 기존 계획 대비 **훨씬 현실적이고 실행 가능**합니다.

### 핵심 변경사항
1. **MVP 최우선** (Week 1-6): "현장에 써볼 수 있는" 최소 기능 제품
2. **ML 도입 연기**: P2로 미루고, P1에서는 Anomaly Detection만 보조 역할
3. **복잡도 관리**: DTW, 정량 조치 권장 등 고급 기능은 P1-P2로 분산
4. **데이터 수집 UX 강조**: 이것이 ML 성공의 핵심

### 검토 결과: ✅ **승인 권장** (단, 아키텍처 분리 필요)

---

## 📋 Quick Summary

### 🎯 핵심 질문 (각 단계의 목표)

- 🔴 **MVP (Week 1-6)**: **"STD랑 다른가?"**
  - 상관계수, ΔE tail, Explainability
  - 완료 기준: Week 6에 현장 시범 운영 가능

- 🟡 **P1 (Week 7-10)**: **"이거 진짜 봐야 하나?"**
  - DTW 옵션, Anomaly ML, 데이터 수집 UX
  - 완료 기준: 100 샘플 데이터 수집 + 시스템 vs 사람 판정 일치도 측정

- 🟢 **P2 (Week 11+)**: **"왜 이런 불량이 나왔나?"**
  - ML 불량 분류, 통계 STD, 조치 권장
  - 완료 기준: 시스템 판정 정확도 > 95%

### 📊 단계별 요약 표

| 단계 | 핵심 질문 | 기술 포인트 | 완료 기준 | ML 역할 |
|------|-----------|-------------|-----------|---------|
| 🔴 **MVP** | "STD랑 다른가?" | 상관계수, ΔE tail, Explainability | Week 6: 현장 시범 | ❌ 없음 |
| 🟡 **P1** | "이거 진짜 봐야 하나?" | DTW 옵션, Anomaly ML | Week 10: 100 샘플 | 🟡 보조 (Review 플래그) |
| 🟢 **P2** | "왜 이런 불량이 나왔나?" | ML 분류, 통계 STD | 정확도 > 95% | ✅ 본격 (판정 보정) |

### 🎯 5가지 핵심 원칙 (프로젝트 성공 조건)

1. ✅ **룰 기반이 먼저, ML은 나중**
   - MVP/P1: 규칙 기반 판정
   - P2: ML 본격 도입 (데이터 충분히 쌓인 후)

2. ✅ **판정과 신뢰도는 분리**
   - `FAIL + 신뢰도 95%` → "확실한 불량"
   - `FAIL + 신뢰도 60%` → "재검사 필요"

3. ✅ **숫자보다 '현상 설명'을 우선**
   - ❌ "점수 65.3점"
   - ✅ "Zone B 경계 +4.2% 외측 이동 (허용 3% 초과)"

4. ✅ **STD는 결국 '분포'가 되어야 함**
   - MVP/P1: 단일 기준 프로파일
   - P2: 5-10 샘플 통계 모델 (mean ± σ)

5. ✅ **데이터 수집 UX가 ML 성능을 결정**
   - P1에서 사람 판정 입력 UI 필수
   - 시스템 vs 사람 불일치 케이스가 학습 데이터

---

## 1. 로드맵 비교 분석

### 1.1 기존 계획 vs 사용자 제안

| 항목 | 기존 계획 (TECHNICAL_ENHANCEMENTS) | 사용자 제안 (MVP 로드맵) | 비고 |
|------|-----------------------------------|--------------------------|------|
| **완료 시점** | Week 10 (전체 완성) | **Week 6 (MVP)** | ✅ 현실적 |
| **STD 모델** | 통계 모델 (5-10 샘플) | **단일 기준 프로파일** | ✅ 단순화 |
| **정렬 알고리즘** | Elastic Alignment (Anchor + DTW) | **C-zone 기준 정렬만** | ✅ 충분함 |
| **비교 지표** | 7가지 (percentile, hotspot, ...) | **상관계수 + ΔE_95p** | ✅ 핵심만 |
| **ML** | P2 (정량 조치 권장) | **P1: ML-lite (보조), P2: 본격** | ✅ 단계적 |
| **Explainability** | Top 3 reasons | **Top 3 reasons** | ✅ 동일 |
| **현장 투입** | Week 10 | **Week 6** | ✅ 빠름 |

**결론**: 사용자 제안이 **더 빠르고, 더 안전하고, 더 현실적**입니다.

---

## 2. 단일 분석 시스템 vs 비교 시스템 아키텍처 분리

### 2.1 현재 시스템 (단일 분석)

```
[이미지]
  ↓
[InspectionPipeline]
  ├─ ImageLoader
  ├─ LensDetector
  ├─ ZoneSegmenter
  ├─ RadialProfiler
  ├─ ColorEvaluator
  └─ InkEstimator
  ↓
[AnalysisResult]
  - zones: {A, B, C}
  - radial_profile: [L, a, b]
  - ink_colors: [...]
  - expected_zones: SKU 설정
  ↓
[Visualizer] → 결과 이미지/차트
```

**특징**:
- **절대적 기준**: SKU 설정 파일 (expected_zones, zone_specs)
- **독립적 분석**: 각 이미지를 단독으로 분석
- **출력**: 분석 결과 (Zone 색상, 경계, 잉크 개수 등)

### 2.2 비교 시스템 (STD 기반)

```
[STD 이미지] → [InspectionPipeline] → [STD Profile] → [DB 저장]
                                             ↓
                                        [std_profiles]

[TEST 이미지] → [InspectionPipeline] → [TEST Profile]
                                             ↓
                                    [ComparisonEngine]
                                        ↙        ↘
                            [STD Profile]   [Alignment]
                                             ↓
                                    [SimilarityScoring]
                                        - Structure
                                        - Color
                                             ↓
                                    [JudgmentEngine]
                                        - PASS/FAIL
                                        - Confidence
                                             ↓
                                    [ExplainabilityEngine]
                                        - Top 3 Reasons
                                             ↓
                                    [ComparisonResult] → [DB 저장]
```

**특징**:
- **상대적 기준**: STD 프로파일과의 비교
- **의존적 분석**: STD가 있어야 비교 가능
- **출력**: 비교 결과 (유사도, 판정, 이유)

### 2.3 아키텍처 분리 원칙

#### ✅ DO: 공유해야 할 것
1. **InspectionPipeline** (동일한 분석 로직)
   - LensDetector, ZoneSegmenter, RadialProfiler 등
   - 이유: STD와 TEST를 동일한 방법으로 분석해야 공정한 비교

2. **AnalysisResult 데이터 구조**
   - zones, radial_profile, ink_analysis
   - 이유: STD Profile과 TEST Profile이 동일한 형식

3. **Visualizer** (일부)
   - Radial profile 차트, Zone overlay
   - 이유: 시각화 로직 재사용

#### ❌ DON'T: 분리해야 할 것
1. **판정 로직**
   - 단일 분석: "Zone B L=65±5인가?" (절대적)
   - 비교 시스템: "STD 대비 유사도 85%인가?" (상대적)

2. **기준 데이터**
   - 단일 분석: SKU 설정 JSON (고정 값)
   - 비교 시스템: STD Profile DB (동적, 버전 관리)

3. **출력 형식**
   - 단일 분석: AnalysisResult (분석 결과)
   - 비교 시스템: ComparisonResult (비교 결과)

---

## 3. MVP 범위 현실성 검증

### 3.1 M0. 기반 정비 (Week 1) ✅ **현실적**

**작업량**: 1주
- DB 스키마 구축 (std_profiles, test_samples, comparison_results)
- ORM 모델 (이미 완료됨!)
- Pydantic 모델 (API 입출력)

**현재 진행도**: 70% 완료
- ✅ SQLAlchemy 모델 완성 (src/models/)
- ✅ DB 테스트 통과 (12/12)
- ⏳ Alembic 마이그레이션 필요
- ⏳ Pydantic 모델 정의 필요

**리스크**: 낮음

### 3.2 M1. STD 프로파일 생성 & 저장 (Week 2-3) ✅ **현실적**

**작업량**: 2주
- STD 등록 API (FastAPI)
- STD 분석 파이프라인 (기존 InspectionPipeline 재사용)
- STD 상세 조회 UI (간단한 템플릿)

**재사용 가능 코드**:
- ✅ InspectionPipeline (전체 재사용)
- ✅ Visualizer (일부 재사용)

**새로 작성할 코드**:
```python
# src/services/std_service.py
class STDService:
    def register_std(sku_code, image_path, version="v1.0"):
        # 1. InspectionPipeline으로 분석
        result = pipeline.run(image_path)

        # 2. STD Profile 생성
        std_profile = STDProfile(
            sku_code=sku_code,
            version=version,
            zone_boundaries=result.transitions,
            radial_profile=result.radial_profile,
            zone_colors={z: result.zones[z].color_lab for z in ['A','B','C']}
        )

        # 3. DB 저장
        db.add(std_profile)
        db.commit()

        return std_profile
```

**라인 수 추정**: ~800줄 (Service 300 + API 200 + UI 300)

**리스크**: 낮음 (기존 코드 재사용)

### 3.3 M2. STD vs TEST 비교 + 리포트 (Week 4-6) ⚠️ **타이트**

**작업량**: 3주 (기능이 많음)

**필수 기능 (P0)**:
1. 구조 유사도
   - Radial profile 상관계수 (scipy.corrcoef)
   - Zone 경계 위치 차이 (간단한 뺄셈)
   - C-zone 기준 정렬 (np.roll)

2. 색상 유사도
   - Zone별 평균 ΔE (기존 코드 재사용)
   - Zone별 ΔE_95percentile (np.percentile)

3. 판정 로직
   - PASS/WARNING/FAIL (if-else)
   - Confidence 계산 (간단한 가중 평균)

4. Explainability
   - Top 3 reasons (정렬 + 슬라이싱)

5. 비교 리포트 UI
   - STD vs TEST Overlay (Plotly)
   - Zone별 비교 테이블 (Bootstrap)
   - NG 사유 카드 (Jinja2 템플릿)

**코드 추정**:
```python
# src/services/comparison_service.py (~500줄)
class ComparisonService:
    def compare(test_result, std_profile):
        # 1. Alignment
        aligned = self.align_c_zone(test_result.radial_profile, std_profile.radial_profile)

        # 2. Structure similarity
        corr = np.corrcoef(aligned, std_profile.radial_profile)[0, 1]
        boundary_diff = {
            z: (test_result.transitions[z] - std_profile.boundaries[z]) / std_profile.boundaries[z] * 100
            for z in ['AB', 'BC', 'C']
        }

        # 3. Color similarity
        zone_delta_e = {}
        for zone in ['A', 'B', 'C']:
            test_lab = test_result.zones[zone].pixels_lab
            std_lab = std_profile.zone_colors[zone]
            delta_e_array = [delta_e_cie2000(t, std_lab) for t in test_lab]
            zone_delta_e[zone] = {
                'mean': np.mean(delta_e_array),
                'p95': np.percentile(delta_e_array, 95)
            }

        # 4. Judgment
        structure_score = corr * 100
        color_score = 100 - np.mean([z['mean'] for z in zone_delta_e.values()]) * 10
        total_score = structure_score * 0.4 + color_score * 0.6

        if total_score > 80:
            judgment = 'PASS'
        elif total_score > 60:
            judgment = 'WARNING'
        else:
            judgment = 'FAIL'

        # 5. Explainability
        reasons = []
        for zone, de in zone_delta_e.items():
            if de['p95'] > 5.0:
                reasons.append({
                    'severity': de['p95'] * 10,
                    'message': f"Zone {zone} ΔE_95p={de['p95']:.1f} (허용 5.0 초과)"
                })

        for zone, diff in boundary_diff.items():
            if abs(diff) > 3.0:
                reasons.append({
                    'severity': abs(diff) * 20,
                    'message': f"Zone {zone} 경계 {diff:+.1f}% {'외측' if diff > 0 else '내측'} 이동"
                })

        top_reasons = sorted(reasons, key=lambda x: x['severity'], reverse=True)[:3]

        return ComparisonResult(
            total_score=total_score,
            judgment=judgment,
            top_reasons=top_reasons,
            ...
        )
```

**라인 수 추정**: ~2000줄 (Service 500 + API 300 + UI 800 + Tests 400)

**리스크**: 중간 (기능이 많지만 각각은 단순)

**완화 방안**:
- UI를 Week 6에만 집중 (Week 4-5는 API만)
- Plotly 차트 대신 간단한 테이블부터 시작
- 점진적 개선 (Week 6에 완성, Week 7에 개선)

### 3.4 MVP 총 작업량 추정

| 주차 | 작업 | 라인 수 | 리스크 |
|------|------|---------|--------|
| Week 1 | M0 기반 정비 | 500줄 | 낮음 (70% 완료) |
| Week 2-3 | M1 STD 등록 | 800줄 | 낮음 (기존 코드 재사용) |
| Week 4-6 | M2 비교 + UI | 2000줄 | 중간 (타이트) |
| **합계** | **6주** | **3300줄** | **중간** |

**결론**: ✅ **6주 MVP 달성 가능** (단, Week 4-6가 타이트함)

---

## 4. 기존 계획과의 통합

### 4.1 기존 문서 재배치

| 기존 문서 | 해당 단계 | 활용 방법 |
|----------|-----------|-----------|
| **STD_BASED_QC_SYSTEM_PLAN.md** | 전체 비전 | 참고 문서 (장기 계획) |
| **TECHNICAL_ENHANCEMENTS_ADVANCED.md** | P1-P2 | **P1-P2 구현 시 참고** |
| **REVIEW_FEEDBACK_AND_IMPROVEMENTS.md** | P1 | 알고리즘 최적화 시 참고 |

### 4.2 7대 고급 기술의 단계별 배치

| 기술 | MVP | P1 | P2 | 비고 |
|------|-----|----|----|------|
| 1. **STD 통계 모델** | ❌ | ❌ | ✅ | P2에서 도입 |
| 2. **Elastic Alignment** | 🟡 | ✅ | - | MVP: C-zone만, P1: DTW |
| 3. **Worst-Case Metrics** | 🟡 | ✅ | - | MVP: p95만, P1: hotspot |
| 4. **Ink-Aware Comparison** | ❌ | ✅ | - | P1에서 추가 |
| 5. **Explainability** | ✅ | ✅ | ✅ | MVP부터 필수 |
| 6. **Performance & Stability** | 🟡 | ✅ | ✅ | MVP: 기본만, P1: 캐싱 |
| 7. **Phenomenological Classification** | ❌ | ❌ | ✅ | P2 ML 도입 시 |

**범례**:
- ✅ 전체 도입
- 🟡 부분 도입
- ❌ 제외

---

## 5. 아키텍처 설계 (단일 vs 비교 시스템 분리)

### 5.1 디렉토리 구조

```
src/
├── core/                    # 공유 분석 엔진
│   ├── image_loader.py
│   ├── lens_detector.py
│   ├── zone_segmenter.py
│   ├── radial_profiler.py
│   ├── color_evaluator.py
│   └── ink_estimator.py
│
├── pipeline.py              # 단일 분석 파이프라인
│
├── services/                # 비즈니스 로직 (신규)
│   ├── std_service.py       # STD 등록/조회/관리
│   ├── comparison_service.py # STD vs TEST 비교
│   ├── judgment_service.py  # 판정 로직
│   └── explainability_service.py # 설명 생성
│
├── models/                  # DB 모델 (완성)
│   ├── std_models.py
│   ├── test_models.py
│   └── user_models.py
│
├── schemas/                 # Pydantic 모델 (신규)
│   ├── std_schemas.py       # STD 입출력
│   ├── comparison_schemas.py # 비교 결과
│   └── judgment_schemas.py  # 판정 결과
│
├── web/                     # Web UI
│   ├── app.py               # FastAPI 앱
│   ├── routers/             # API 라우터 (신규)
│   │   ├── analysis.py      # 단일 분석 API (기존)
│   │   ├── std.py           # STD 관리 API
│   │   ├── comparison.py    # 비교 API
│   │   └── admin.py         # 관리자 API
│   └── templates/
│       ├── index.html       # 단일 분석 UI (기존)
│       ├── std_list.html    # STD 목록
│       ├── std_detail.html  # STD 상세
│       └── comparison.html  # 비교 결과
│
└── utils/
    ├── color_delta.py
    └── alignment.py         # 정렬 알고리즘 (신규)
```

### 5.2 모듈 간 의존성

```
[Web UI / API]
    ↓ 호출
[Services Layer] ← 비즈니스 로직 (신규)
    ↓ 사용
[Pipeline / Core] ← 분석 엔진 (기존, 공유)
    ↓ 저장/조회
[Models / DB] ← 데이터 계층 (완성)
```

**핵심 원칙**:
1. **Pipeline은 "어떻게 분석하는가"만 담당** (STD/TEST 구분 없음)
2. **Services가 "어떻게 비교하는가" 담당** (비교 시스템 전용)
3. **API/UI는 Services를 호출** (두 시스템 모두)

### 5.3 데이터 흐름 (MVP)

#### Case 1: STD 등록
```
[사용자] → POST /api/std/register
              ↓
    [STDService.register_std()]
              ↓
    [InspectionPipeline.run()] ← 기존 코드 재사용
              ↓
         [AnalysisResult]
              ↓
    [STDProfile 생성 및 DB 저장]
              ↓
         [STDModel] → DB
```

#### Case 2: TEST 비교
```
[사용자] → POST /api/comparison/compare
              ↓
    [ComparisonService.compare()]
              ↓
    ┌─────────────────────┬─────────────────────┐
    ↓                     ↓                     ↓
[InspectionPipeline]  [STD 조회]     [Alignment]
    ↓                     ↓                     ↓
[TEST Profile]      [STD Profile]        [정렬된 Profile]
    └─────────────────────┴─────────────────────┘
                          ↓
              [SimilarityScoring]
                          ↓
              [JudgmentEngine]
                          ↓
              [ExplainabilityEngine]
                          ↓
              [ComparisonResult] → DB
```

---

## 6. 핵심 원칙 검증

사용자가 제시한 5가지 원칙을 MVP에서 어떻게 구현하는가?

### 원칙 1: "룰 기반이 먼저, ML은 나중"
✅ **MVP 준수**
- MVP: 상관계수, ΔE threshold 등 룰 기반 판정
- P1: ML-lite (Anomaly Detection) 보조만
- P2: ML 본격 도입

### 원칙 2: "판정과 신뢰도는 분리"
✅ **MVP 준수**
```python
ComparisonResult(
    judgment='FAIL',          # 판정 (PASS/FAIL/WARNING)
    confidence_score=72.5,    # 신뢰도 (0-100)
    total_score=65.3          # 유사도 점수 (0-100)
)
```

**사용 시나리오**:
- `judgment='FAIL' + confidence=95` → 확실한 불량
- `judgment='FAIL' + confidence=65` → 재검사 필요
- `judgment='PASS' + confidence=95` → 확실한 합격

### 원칙 3: "숫자보다 '현상 설명'을 우선"
✅ **MVP 준수**
```python
top_reasons = [
    {
        'rank': 1,
        'message': "Zone B 경계 +4.2% 외측 이동 (허용 3.0% 초과)",
        'severity': 84,
        'action': "내측 방향 조정 필요"
    },
    {
        'rank': 2,
        'message': "Zone A ΔE_95p=6.1 (허용 5.0 초과)",
        'severity': 61,
        'action': "색상 농도 조정"
    }
]
```

**UI 표시**:
```
❌ FAIL (총점 65.3)

주요 불량 원인:
1. Zone B 경계가 4.2% 외측으로 이동했습니다 (기준: ±3% 이내)
   → 조치: 내측 방향 조정 필요

2. Zone A 색상이 기준 대비 95% 픽셀에서 ΔE 6.1로 차이가 큽니다 (기준: 5.0 이하)
   → 조치: 색상 농도 조정
```

### 원칙 4: "STD는 결국 '분포'가 되어야 함"
🟡 **MVP: 단일 기준, P2: 분포**
- MVP: STD = 하나의 "좋은 샘플"
- P1: 여전히 단일 기준
- **P2**: STD = 5-10 샘플의 통계 분포 (mean ± σ)

**이유**: MVP에서는 복잡도 최소화, P2에서 데이터 쌓인 후 도입

### 원칙 5: "데이터 수집 UX가 ML 성능을 결정"
✅ **P1부터 준비**

**P1 필수 기능**:
```python
# 사람의 최종 판정 입력
class ManualReview:
    system_judgment = 'FAIL'
    human_judgment = 'PASS'  # 사람이 오버라이드
    reason_code = 'ACCEPTABLE_DEVIATION'
    action_taken = 'APPROVED_WITH_COMMENT'
    comment = "경계 이동 있으나 시각적으로 양호"
```

**데이터 수집 포인트**:
1. 시스템 판정 vs 사람 판정 (불일치 케이스 학습)
2. NG 사유 코드 (분류 학습)
3. 조치 후 결과 (조치 권장 학습)

**ML 학습 데이터셋** (P2에서 활용):
```
training_data = [
    {
        'features': {
            'zone_A_delta_e': 4.2,
            'zone_B_boundary_diff': 4.8,
            'structure_corr': 0.89,
            ...
        },
        'system_judgment': 'FAIL',
        'human_judgment': 'PASS',  # Ground truth
        'reason_code': 'ACCEPTABLE_DEVIATION'
    },
    ...
]
```

---

## 7. 리스크 분석 및 완화 방안

### 7.1 MVP 리스크

| 리스크 | 확률 | 영향 | 완화 방안 |
|--------|------|------|-----------|
| Week 4-6 일정 지연 | 중간 | 높음 | UI 간소화, Week 7 버퍼 사용 |
| STD 기준 부족 (단일 샘플) | 높음 | 중간 | P2 통계 모델로 해결 예정 |
| 현장 저항 (기존 방식 변경) | 중간 | 높음 | **병렬 운영** (기존 방식 유지 + 신규 시스템 시범) |
| 판정 기준 불명확 | 높음 | 높음 | Week 1-2에 **임계값 협의** 필수 |

### 7.2 완화 방안 상세

#### 리스크 1: Week 4-6 일정 지연
**완화**:
- UI 우선순위: 비교 결과 테이블 > Overlay 차트 > 고급 시각화
- Week 6 목표: "동작하는 최소 UI" (예쁘지 않아도 됨)
- Week 7-8: UI 개선 (P1 기능 추가 전)

#### 리스크 3: 현장 저항
**완화**: 병렬 운영 전략
```
Week 6-8: 시범 운영
├─ 기존 방식: 계속 사용 (공식 판정)
└─ 신규 시스템: 참고용으로만 사용 (판정 효력 없음)

Week 9-10: 신뢰도 검증
├─ 100 샘플 비교
├─ 사람 vs 시스템 일치도 측정
└─ 임계값 튜닝

Week 11+: 단계적 전환
├─ PASS 케이스만 시스템 판정 신뢰
├─ FAIL 케이스는 사람 재확인
└─ 3개월 후 완전 전환
```

#### 리스크 4: 판정 기준 불명확
**완화**: Week 1-2 기준 설정 워크샵
```
필요한 답:
1. "구조 유사도 몇 % 이상이면 OK?"
   → 제안: 상관계수 > 0.85 (85%)

2. "색상 차이 ΔE 몇까지 허용?"
   → 제안: 평균 ΔE < 3.0, 95% ΔE < 5.0

3. "경계 위치 몇 % 이동까지 OK?"
   → 제안: ±3% 이내

4. "WARNING 구간 필요한가?"
   → 제안: 80% > score > 60% = WARNING (재검사)
```

---

## 8. 통합 로드맵 (최종안)

### 🔴 Phase M: MVP (Week 1-6)

**핵심 질문**: **"STD랑 다른가?"**

**목표**: "STD 기준으로 양산 샘플을 비교하고, 왜 OK/NG인지 사람에게 설명할 수 있는 시스템"

**기술 포인트**: 상관계수, ΔE tail, Explainability

**완료 기준**: ✅ Week 6에 "현장에서 1개 샘플을 STD와 비교하고, PASS/FAIL + 이유를 볼 수 있다"

---

#### M0. 기반 정비 (Week 1)

**핵심 목표**: STD/TEST를 저장할 수 있는 구조 완성

**포함**:
- [x] DB 스키마 (std_profiles, test_samples, comparison_results)
- [x] SQLAlchemy 모델 (7 tables)
- [x] Pydantic 스키마 (19 schemas) ✅ 완료
- [ ] Alembic 마이그레이션
- [ ] **판정 기준 협의 워크샵** ⭐

**제외**:
- ❌ 범위 관리 (Upper/Lower) → P2로 연기
- ❌ 버전 관리 UI → P2로 연기

---

#### M1. STD 등록 (Week 2-3)

**핵심 목표**: "STD란 무엇인가"를 시스템이 명확히 이해하게 만들기

**포함**:
- [ ] STDService 구현
- [ ] STD 등록 API
- [ ] STD 분석 파이프라인 (InspectionPipeline 재사용)
  - Zone 경계
  - Radial profile (L, a, b)
  - Zone별 LAB 평균 / std
- [ ] STD 목록/상세 조회 UI
- [ ] STD = **단일 기준 프로파일**

**제외**:
- ❌ STD 다중 샘플 통계 → P2로 연기
- ❌ 자동 범위 계산 → P2로 연기

---

#### M2. 비교 & 판정 (Week 4-6) ← **MVP 종료선**

**핵심 목표**: "이 렌즈는 STD와 얼마나 다른가?"를 명확히 보여주기

**포함 (P0 기능)**:

1️⃣ **구조 유사도**
- [ ] Radial profile 상관계수
- [ ] Zone 경계 위치 차이 (%)
- [ ] 간단한 Alignment (C-zone 기준 정렬)

2️⃣ **색상 유사도**
- [ ] Zone별 평균 ΔE
- [ ] Zone별 ΔE_95percentile (tail)

3️⃣ **판정 로직**
- [ ] PASS / WARNING / FAIL
- [ ] 판정 + 신뢰도(confidence) 분리

4️⃣ **Explainability** (중요)
- [ ] Top 3 FAIL 원인 생성
- [ ] 예시: "Zone B 경계 +4.2% 외측 이동", "Zone A ΔE_95p = 6.1 (허용 5.0 초과)"

5️⃣ **비교 리포트 UI**
- [ ] STD vs TEST Overlay
- [ ] Zone별 수치 비교 테이블
- [ ] NG 사유 요약 카드

**제외 (MVP에서 뺄 것)**:
- ❌ DTW 전체 적용 → P1로 연기 (조건부 실행)
- ❌ KS/Wasserstein 분포 비교 → P2로 연기
- ❌ 정량 조치 권장 (L +5 같은 수치) → P2로 연기
- ❌ 상한/하한 UI → P2로 연기

**M2 완료 기준**: ✅ "현장에서 1개 샘플을 STD와 비교하고, PASS/FAIL + 이유를 볼 수 있다"

---

### 🟡 Phase P1: 운영 안정화 & 보조 지능 (Week 7-10)

**핵심 질문**: **"이거 진짜 봐야 하나?"**

**목표**: "불확실한 케이스를 줄이고, 사람이 판단하기 쉬운 시스템으로 만들기"

**기술 포인트**: DTW 옵션, Anomaly ML, **데이터 수집 UX** ⭐

**완료 기준**: ✅ "100개 샘플 데이터 수집 완료 + ML 학습 준비"

#### P1-1. 비교 고도화 (Week 7-8)

**핵심 목표**: "상관계수 0.70~0.80 애매한 케이스"를 DTW로 재평가

**포함**:
- [ ] DTW 정렬 (상관계수 < 0.80 케이스만, 선택적 적용)
- [ ] Hotspot 검출 (p95 외에 공간 정보 추가)
- [ ] Ink-wise 비교 (Zone vs Ink 분리 분석)

**제외**:
- ❌ 전체 샘플에 DTW 적용 (성능 이슈)
- ❌ KS/Wasserstein 분포 비교 → P2로 연기

---

#### P1-2. ML-lite: 보조 지능 (Week 8-9)

**핵심 목표**: "판정 변경 없이" 검토 필요 케이스만 flagging

**포함**:
- [ ] Isolation Forest / One-Class SVM (이상치 탐지)
- [ ] "Review 추천" 플래그 생성 (판정 변경 ❌)
- [ ] 신뢰도 점수에만 반영 (Confidence 감소)

**제외**:
- ❌ ML로 판정 변경 → P2로 연기
- ❌ 불량 유형 분류 → P2로 연기

---

#### P1-3. 데이터 수집 UX (Week 9-10) ⭐ **가장 중요**

**핵심 목표**: "ML 학습을 위한 Ground Truth 데이터 수집"

**포함**:
- [ ] 사람 최종 판정 입력 UI (PASS/FAIL 오버라이드)
- [ ] NG 사유 코드 선택 (4가지 유형 준비)
- [ ] 조치 후 결과 기록 (재검사 결과 추적)
- [ ] 시스템 vs 사람 판정 일치도 대시보드

**제외**:
- ❌ 자동 조치 권장 → P2로 연기

**완료 기준**: ✅ "100개 샘플 데이터 수집 완료 + ML 학습 준비"

---

### 🟢 Phase P2: 고도화 & 예측 지능 (Week 11+)

**핵심 질문**: **"왜 이런 불량이 나왔나?"**
**목표**: "불량 원인을 자동 분류하고, 정량적 조치를 권장하는 시스템"
**기술 포인트**: ML 불량 분류, 통계 STD, 조치 권장, 분포 비교
**완료 기준**: ✅ "시스템 판정 정확도 > 95%"

---

#### P2-1. STD 통계 모델: "분포로 진화" (Week 11-12)

**핵심 목표**: STD = "단일 기준"에서 "범위(분포)"로 진화

**포함**:
- [ ] 다중 STD 샘플 수집 (5-10개)
- [ ] 통계 계산 (mean, std, covariance)
- [ ] 자동 Upper/Lower 범위 생성
- [ ] KS/Wasserstein 분포 비교

**제외**:
- ❌ STD 자동 업데이트 → 추후 검토

---

#### P2-2. ML 본격 도입: "불량 분류 & 조치 권장" (Week 13-14)

**핵심 목표**: "왜 NG인가"를 자동으로 설명하고 조치 제안

**포함**:
- [ ] 판정 보정 모델 (XGBoost/LightGBM)
  - 입력: 상관계수, ΔE, boundary 차이, confidence
  - 출력: 판정 보정 (PASS → FAIL, FAIL → WARNING)
- [ ] 불량 유형 분류 (4가지 유형)
  1. "색상 불균일"
  2. "경계 불명확"
  3. "전체 색상 shift"
  4. "구조 왜곡"
- [ ] 조치 권장 정량화
  - "인쇄압 +5% 필요" (예시)
  - "잉크 농도 조정 권장" (예시)

**제외**:
- ❌ 공정 파라미터 자동 제어 → 추후 검토
- ❌ 실시간 공정 최적화 → 추후 검토

**완료 기준**: ✅ "시스템 판정 정확도 > 95% + 불량 원인 설명 자동화"

---

## 9. 권장 사항

### 9.1 즉시 실행
1. ✅ **사용자 제안 로드맵 승인**
   - 기존 계획 대비 훨씬 현실적
   - 6주 MVP 달성 가능

2. ⚠️ **판정 기준 협의 (Week 1 필수)**
   - 임계값 미정은 치명적 리스크
   - 제안값: 상관계수 > 0.85, 평균 ΔE < 3.0, 95% ΔE < 5.0

3. ✅ **병렬 운영 전략 수립**
   - 기존 방식 유지 + 신규 시스템 시범
   - 3개월 검증 기간

### 9.2 Week 1 작업
1. Alembic 설치 및 마이그레이션
2. Pydantic 스키마 정의
3. **판정 기준 워크샵** (가장 중요)
4. STDService 프로토타입

### 9.3 문서 재구성
- `STD_BASED_QC_SYSTEM_PLAN.md` → "장기 비전" 참고 문서
- `TECHNICAL_ENHANCEMENTS_ADVANCED.md` → P1-P2 구현 가이드
- `ROADMAP_REVIEW_AND_ARCHITECTURE.md` (본 문서) → **실행 계획** ⭐

---

## 10. 결론

### 승인 권장 이유
1. ✅ **현실적**: 6주 MVP 달성 가능 (3300줄, 중간 리스크)
2. ✅ **점진적**: MVP → P1 → P2 단계적 복잡도 증가
3. ✅ **데이터 중심**: P1에서 데이터 수집 UX 강조
4. ✅ **안전함**: 병렬 운영으로 현장 저항 최소화
5. ✅ **원칙 준수**: 5가지 핵심 원칙 모두 반영

### 주요 변경사항 요약
| 항목 | 기존 계획 | 사용자 제안 | 변경 이유 |
|------|-----------|-------------|-----------|
| 완료 시점 | Week 10 | **Week 6** | MVP 우선 |
| STD 모델 | 통계 (5-10개) | **단일** | 복잡도 감소 |
| 정렬 | Elastic | **C-zone만** | 충분함 |
| ML | Week 10 | **P2 (Week 11+)** | 데이터 먼저 |

### Next Action
**Week 1 시작 전 필수**:
1. 판정 기준 협의 (상관계수, ΔE threshold, 경계 허용 오차)
2. Alembic 설치 및 DB 마이그레이션
3. Pydantic 스키마 정의

**Status**: ✅ **승인 권장 - MVP 로드맵 실행 준비 완료**

---

**작성자**: Claude Sonnet 4.5
**검토 필요**: 판정 임계값 협의, 병렬 운영 전략 승인
**승인 후**: Week 1 M0 작업 착수
