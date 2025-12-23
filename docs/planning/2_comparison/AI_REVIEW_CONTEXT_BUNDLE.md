# AI 검토용 컨텍스트 번들: STD 기준 품질 관리 시스템 계획

> **작성일**: 2025-12-17
> **목적**: AI 전문가 검토를 위한 종합 컨텍스트 문서
> **프로젝트**: Contact Lens Color Inspection System

---

## 📌 검토 요청 사항 (AI에게)

다음 사항들을 검토하고 피드백 부탁드립니다:

### 1. 기술적 타당성
- [ ] 제안된 아키텍처가 현재 시스템과 잘 통합될 수 있는가?
- [ ] 비교 분석 알고리즘 (상관계수, DTW, KS test)이 적절한가?
- [ ] 데이터베이스 스키마가 요구사항을 충족하는가?
- [ ] 성능 목표 (비교 < 3초)가 현실적인가?

### 2. 누락된 요구사항
- [ ] 간과된 기능이나 시나리오가 있는가?
- [ ] 추가로 고려해야 할 엣지 케이스는?
- [ ] 보안, 백업, 감사 추적 등 운영 요구사항은?

### 3. 우선순위 및 일정
- [ ] Phase 우선순위(P0/P1/P2)가 적절한가?
- [ ] 10주 일정이 현실적인가?
- [ ] MVP 범위(Week 6)가 적절한가?

### 4. 리스크 및 대안
- [ ] 주요 기술적 리스크는?
- [ ] 대안적 접근 방식 제안
- [ ] 단계별 rollback 전략 필요성

### 5. 개선 제안
- [ ] 더 나은 알고리즘이나 기술 스택
- [ ] 사용자 경험(UX) 개선
- [ ] 확장성/유지보수성 향상

---

## 📖 목차

1. [프로젝트 배경](#1-프로젝트-배경)
2. [현재 시스템 (As-Is)](#2-현재-시스템-as-is)
3. [목표 시스템 (To-Be)](#3-목표-시스템-to-be)
4. [기술 스택](#4-기술-스택)
5. [데이터 구조 예시](#5-데이터-구조-예시)
6. [검토 체크리스트](#6-검토-체크리스트)

---

## 1. 프로젝트 배경

### 1.1 프로젝트 개요

**Contact Lens Color Inspection System**은 컴퓨터 비전 기술을 활용하여 콘택트렌즈 제조 공정 중 색상 불량을 자동으로 검출하는 시스템입니다.

**핵심 기술**:
- 극좌표 변환 (Polar Transform) 기반 회전 불변 분석
- CIEDE2000 색차 공식으로 미세한 색상 차이 감지
- Zone 기반 구역 분할 (방사형 프로파일 분석)
- GMM (Gaussian Mixture Model) 기반 잉크 검출

**현재 단계**:
- ✅ 단일 이미지 분석 시스템 완성 (94.7% 테스트 커버리지)
- ✅ Zone-based 및 Image-based 이중 색상 추출
- ✅ 4단계 판정 (OK/WARNING/NG/RETAKE)
- ✅ Web UI (FastAPI + Bootstrap)

**목표**:
- 🎯 **STD(표준) 기반 양산 품질 관리 시스템 구축**
- 🎯 STD 이미지와 양산 샘플 비교 분석
- 🎯 구조/색상 유사도 자동 계산 및 조치 권장
- 🎯 합격/불합격 자동 판정

---

## 2. 현재 시스템 (As-Is)

### 2.1 구현된 기능

| 모듈 | 상태 | 주요 기능 |
|------|------|----------|
| **렌즈 검출** | ✅ | Hough Circle + Contour 기반 자동 검출 |
| **Zone 분할** | ✅ | Radial Profiling + Gradient 기반 경계 검출 |
| **색상 측정** | ✅ | Zone별 LAB 평균/std/percentiles |
| **ΔE 계산** | ✅ | CIEDE2000 (Zone별, Overall) |
| **판정** | ✅ | OK/OK_WITH_WARNING/NG/RETAKE (4단계) |
| **SKU 관리** | ✅ | JSON 기반 기준값 관리 |
| **시각화** | ✅ | Overlay, Profile Chart, Heatmap |
| **Web UI** | ✅ | FastAPI + 단일 이미지 분석 대시보드 |
| **Ink 분석** | ✅ | Zone-based (3단계 필터링, 정확도 높음)<br>Image-based (GMM, 보조용) |

### 2.2 현재 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    Web UI (FastAPI)                     │
│  - 단일 이미지 업로드                                     │
│  - 분석 결과 표시 (요약, 잉크, 상세, 그래프)                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  Analysis Pipeline                       │
├─────────────────────────────────────────────────────────┤
│  1. ImageLoader → 이미지 로드                            │
│  2. LensDetector → 렌즈 위치/반경 검출                    │
│  3. RadialProfiler → 극좌표 프로파일 생성                 │
│  4. ZoneSegmenter → Zone 경계 검출                       │
│  5. ZoneAnalyzer2D → Zone별 색상 측정                    │
│  6. ColorEvaluator → ΔE 계산 및 판정                     │
│  7. InkEstimator → GMM 기반 잉크 검출                    │
│  8. Visualizer → 결과 시각화                             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│               SKU Config (JSON Files)                   │
│  - config/sku_db/SKU001.json                            │
│  - Zone별 기준값 (L, a, b, threshold)                    │
│  - expected_zones, optical_clear_ratio                  │
└─────────────────────────────────────────────────────────┘
```

### 2.3 모듈 구조

```
src/
├── core/                   # 핵심 알고리즘
│   ├── lens_detector.py    # 렌즈 검출
│   ├── radial_profiler.py  # 극좌표 프로파일
│   ├── zone_segmenter.py   # Zone 분할
│   ├── zone_analyzer_2d.py # 2D Zone 분석 (메인 엔진)
│   ├── color_evaluator.py  # 색상 평가 및 판정
│   └── ink_estimator.py    # GMM 잉크 분석
├── web/                    # Web UI
│   ├── app.py              # FastAPI 앱
│   ├── templates/          # HTML 템플릿
│   └── static/             # JS, CSS
├── data/                   # 데이터 관리
│   └── config_manager.py   # SKU 설정 로드
├── utils/                  # 유틸리티
│   ├── color_delta.py      # CIEDE2000
│   └── color_space.py      # LAB 변환
├── main.py                 # CLI 진입점
└── pipeline.py             # 검사 파이프라인
```

### 2.4 현재 작업 흐름

```
[사용자]
   ↓ 이미지 업로드 (sample.jpg)
   ↓ SKU 선택 (SKU001)
[시스템]
   ↓ SKU JSON 로드 (config/sku_db/SKU001.json)
   ↓ 분석 실행 (analyze_lens_zones_2d)
   ↓ Zone별 측정값 vs 기준값 비교
   ↓ ΔE 계산 → OK/NG 판정
[결과 표시]
   ✅ Judgment: OK (ΔE=2.5)
   ✅ Zone A: ΔE=2.1 (OK)
   ✅ Zone B: ΔE=2.9 (OK)
```

### 2.5 현재 시스템의 한계 ⚠️

| 한계 | 설명 | 영향 |
|------|------|------|
| **단일 분석** | 이미지 하나만 분석, 비교 대상 없음 | STD와 비교 불가 |
| **휘발성 결과** | 분석 결과를 DB에 저장하지 않음 | 히스토리 추적 불가 |
| **수동 기준값** | SKU JSON을 사람이 직접 작성 | 오류 가능성, 비효율 |
| **색상만 비교** | 구조(프로파일 형태) 무시 | 도포 패턴 이상 감지 못함 |
| **범위 없음** | 상한/하한 개념 없음 | 양산 범위 관리 불가 |
| **조치 부재** | 불량 시 무엇을 할지 모름 | 현장 조치 어려움 |

---

## 3. 목표 시스템 (To-Be)

> **전체 계획 문서를 아래에 포함합니다.**

---

# STD 기준 품질 관리 시스템 구축 계획

> **작성일**: 2025-12-17
> **목적**: 표준 이미지(STD) 기반 양산 품질 관리 시스템 구축 로드맵
> **상태**: 🔴 계획 수립 단계

---

## 📋 목차

1. [현재 상황 분석](#1-현재-상황-분석-1)
2. [목표 시스템 정의](#2-목표-시스템-정의-1)
3. [갭 분석](#3-갭-분석-1)
4. [시스템 아키텍처](#4-시스템-아키텍처-1)
5. [단계별 구현 계획](#5-단계별-구현-계획-1)
6. [데이터베이스 설계](#6-데이터베이스-설계-1)
7. [비교 분석 알고리즘](#7-비교-분석-알고리즘-1)
8. [우선순위 및 마일스톤](#8-우선순위-및-마일스톤-1)

---

## 1. 현재 상황 분석

### 1.1 구현된 기능 ✅

| 기능 | 상태 | 비고 |
|------|------|------|
| **렌즈 검출** | ✅ 완료 | Hough Circle + Contour |
| **Zone 분할** | ✅ 완료 | Radial Profiling + Transition Detection |
| **색상 측정** | ✅ 완료 | Zone별 LAB 평균, std, percentiles |
| **ΔE 계산** | ✅ 완료 | CIEDE2000 (Zone별, Overall) |
| **OK/NG 판정** | ✅ 완료 | 4단계 (OK/WARNING/NG/RETAKE) |
| **SKU 관리** | ✅ 완료 | JSON 기반 기준값 관리 |
| **시각화** | ✅ 완료 | Overlay, Profile Chart, Heatmap |
| **Web UI** | ✅ 완료 | FastAPI + 단일 이미지 분석 |
| **Ink 분석** | ✅ 완료 | Zone-based + Image-based (GMM) |

### 1.2 현재 작업 흐름

```
사용자 → 이미지 업로드 → 분석 실행 → 결과 표시
   ↓
SKU 기준값 (수동 설정)
   ↓
Zone별 측정값 vs 기준값
   ↓
ΔE < threshold → OK/NG
```

**한계**:
- ❌ **단일 분석**: 이미지 하나만 분석 (비교 대상 없음)
- ❌ **수동 기준값**: SKU JSON을 사람이 직접 작성
- ❌ **상세 저장 없음**: 분석 결과를 DB에 저장하지 않음
- ❌ **구조 비교 없음**: 색상만 비교 (프로파일 형태 무시)
- ❌ **범위 관리 없음**: 상한/중/하한 개념 없음

---

## 2. 목표 시스템 정의

### 2.1 핵심 개념

```
┌─────────────────────────────────────────────────────────────┐
│                    STD (Standard) 이미지                     │
│  - 양산 전 승인된 표준 렌즈의 기준 이미지                      │
│  - 구조 + 색상 정보를 DB에 상세 저장                          │
│  - 추후 상한/중/하한으로 확장                                 │
└─────────────────────────────────────────────────────────────┘
                          ↓ 비교
┌─────────────────────────────────────────────────────────────┐
│                  양산 샘플 (Test Sample)                     │
│  - 양산 중 생산된 렌즈                                        │
│  - STD와 비교하여 합격/불합격 판정                            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 요구사항

#### Phase 1: STD 프로파일 생성 및 저장
- [ ] STD 이미지 지정 (SKU당 1개 이상)
- [ ] 상세 분석 실행 (구조 + 색상)
- [ ] DB에 STD 프로파일 저장
  - 구조: Zone 경계, Radial Profile, Transition 패턴
  - 색상: Zone별 LAB 평균/std/percentiles
  - 메타: 촬영 조건, 잉크 정보, 승인자, 일시

#### Phase 2: 양산 샘플 비교 분석
- [ ] STD와 Test Sample 자동 매칭 (SKU 기반)
- [ ] 구조 유사도 계산
  - Radial Profile 형태 비교 (상관계수, DTW, MSE)
  - Zone 경계 위치 비교
  - Transition 패턴 비교
- [ ] 색상 유사도 계산
  - Zone별 ΔE (평균, 최대, std)
  - 색상 분포 비교 (KS test, Wasserstein)
- [ ] 종합 점수 산출 (0~100점)
- [ ] 같은 렌즈인지 판단 (의견 + 신뢰도)

#### Phase 3: 조치 권장 시스템
- [ ] 색상 편차 → 잉크 조정 방향 및 정도
  - 예: "Zone A를 +3.2 ΔE만큼 더 밝게 (L +5.0)"
- [ ] 구조 편차 → 공정 조정 권장
  - 예: "Zone B 경계가 3% 안쪽으로 이동, 도포 범위 확인"

#### Phase 4: 범위 기반 합격/불합격
- [ ] STD 상한/중/하한 설정
  - 상한: 허용 가능한 최대 편차
  - 중: STD 자체
  - 하한: 허용 가능한 최소 편차
- [ ] 범위 내 판정 로직
  - 구조 점수 + 색상 점수 모두 범위 내 → 합격
  - 하나라도 범위 밖 → 불합격

### 2.3 최종 사용자 시나리오

```
[1] STD 등록 (1회)
    엔지니어 → STD 이미지 업로드 → 상세 분석 → DB 저장 → 승인

[2] 양산 검사 (반복)
    검사자 → 양산 샘플 업로드 → STD 자동 매칭 → 비교 분석 → 리포트

    리포트 내용:
    - 종합 점수: 87/100 (합격)
    - 구조 유사도: 92% (우수)
    - 색상 유사도: 82% (양호, Zone A 편차 있음)
    - 조치 권장: "Zone A 잉크를 +2.5 ΔE 더 밝게 조정 권장"

[3] 범위 설정 (추후)
    관리자 → STD 상한/하한 설정 → 자동 판정 기준 적용
```

---

## 3. 갭 분석

### 3.1 기능 갭

| 필요 기능 | 현재 상태 | 우선순위 |
|----------|----------|---------|
| **STD 프로파일 저장** | ❌ 없음 | 🔴 P0 (필수) |
| **비교 분석 엔진** | ❌ 없음 | 🔴 P0 (필수) |
| **구조 유사도 계산** | ❌ 없음 | 🔴 P0 (필수) |
| **조치 권장 로직** | ❌ 없음 | 🟡 P1 (중요) |
| **범위 관리 시스템** | ❌ 없음 | 🟢 P2 (추후) |
| **비교 리포트 UI** | ❌ 없음 | 🔴 P0 (필수) |
| **STD 관리 UI** | ❌ 없음 | 🟡 P1 (중요) |

### 3.2 데이터 갭

| 데이터 | 현재 상태 | 필요 상태 |
|--------|----------|----------|
| **SKU 기준값** | JSON (L, a, b, threshold만) | DB에 전체 STD 프로파일 |
| **분석 결과** | 메모리에만 (휘발) | DB에 영구 저장 |
| **프로파일 데이터** | 시각화만 (저장 안 함) | Raw 프로파일 저장 |
| **메타데이터** | 없음 | 촬영 조건, 승인 이력 등 |

---

## 4. 시스템 아키텍처

### 4.1 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (Web UI)                     │
├─────────────────────────────────────────────────────────────┤
│  - STD 등록 페이지                                            │
│  - 양산 검사 페이지 (비교 리포트)                              │
│  - STD 관리 페이지 (상한/하한 설정)                           │
└─────────────────────────────────────────────────────────────┘
                          ↓ API
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                         │
├─────────────────────────────────────────────────────────────┤
│  [1] STD Profile Service                                     │
│      - register_std()                                        │
│      - get_std_profile()                                     │
│      - update_std_range()                                    │
│                                                              │
│  [2] Comparison Service                                      │
│      - compare_to_std()                                      │
│      - calculate_structure_similarity()                      │
│      - calculate_color_similarity()                          │
│      - generate_recommendation()                             │
│                                                              │
│  [3] Judgment Service                                        │
│      - judge_pass_fail()                                     │
│      - check_in_range()                                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                      Database (SQLite/PostgreSQL)            │
├─────────────────────────────────────────────────────────────┤
│  [1] std_profiles                                            │
│      - id, sku_code, version, image_path                     │
│      - profile_data (JSON: radial profile, zone boundaries)  │
│      - color_data (JSON: zone LAB stats)                     │
│      - meta_data (JSON: 촬영 조건, 승인자, 일시)              │
│      - upper_limit, lower_limit (JSON, nullable)             │
│                                                              │
│  [2] test_samples                                            │
│      - id, sku_code, image_path, tested_at                   │
│      - analysis_result (JSON: 분석 결과)                      │
│      - std_profile_id (FK)                                   │
│                                                              │
│  [3] comparison_results                                      │
│      - id, test_sample_id, std_profile_id                    │
│      - structure_score, color_score, total_score             │
│      - pass_fail, recommendations (JSON)                     │
│      - created_at                                            │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 핵심 컴포넌트

#### 4.2.1 STD Profile Manager
```python
class STDProfileManager:
    def create_std_profile(self, image_path: str, sku_code: str) -> STDProfile:
        """STD 이미지 분석 및 프로파일 생성"""
        # 1. 기존 분석 파이프라인 실행
        result = analyze_lens_zones_2d(...)

        # 2. 구조 정보 추출
        structure = {
            "radial_profile": {
                "r": profile.r_normalized.tolist(),
                "L": profile.L.tolist(),
                "a": profile.a.tolist(),
                "b": profile.b.tolist(),
            },
            "zone_boundaries": [
                {"zone": "A", "r_start": 0.65, "r_end": 1.0},
                {"zone": "B", "r_start": 0.38, "r_end": 0.65},
            ],
            "transitions": transition_ranges,
        }

        # 3. 색상 정보 추출
        color = {
            "zones": [
                {
                    "name": "A",
                    "mean_lab": [72.2, 137.3, 122.8],
                    "std_lab": [2.5, 1.8, 2.1],
                    "percentiles": {...},
                }
            ]
        }

        # 4. DB 저장
        return self.db.save_std_profile(sku_code, structure, color, meta)
```

#### 4.2.2 Comparison Engine
```python
class ComparisonEngine:
    def compare(self, test_result: dict, std_profile: STDProfile) -> ComparisonResult:
        """Test Sample과 STD 비교"""

        # 1. 구조 유사도
        structure_score = self.calculate_structure_similarity(
            test_profile=test_result["radial_profile"],
            std_profile=std_profile.structure["radial_profile"]
        )
        # - Pearson 상관계수 (0~1)
        # - DTW (Dynamic Time Warping) 거리
        # - Zone 경계 위치 차이 (%)

        # 2. 색상 유사도
        color_score = self.calculate_color_similarity(
            test_zones=test_result["zones"],
            std_zones=std_profile.color["zones"]
        )
        # - Zone별 ΔE 평균
        # - 분포 유사도 (KS test)

        # 3. 종합 점수
        total_score = (structure_score * 0.4 + color_score * 0.6) * 100

        # 4. 조치 권장
        recommendations = self.generate_recommendations(test_result, std_profile)

        return ComparisonResult(
            structure_score=structure_score,
            color_score=color_score,
            total_score=total_score,
            recommendations=recommendations
        )
```

#### 4.2.3 Recommendation Engine
```python
class RecommendationEngine:
    def generate(self, test_zones: list, std_zones: list) -> list:
        """색상 조정 권장사항 생성"""

        recommendations = []
        for tz, sz in zip(test_zones, std_zones):
            delta_e = tz["delta_e"]

            if delta_e > threshold:
                # LAB 차이 분석
                dL = tz["mean_lab"][0] - sz["mean_lab"][0]
                da = tz["mean_lab"][1] - sz["mean_lab"][1]
                db = tz["mean_lab"][2] - sz["mean_lab"][2]

                # 조정 방향
                direction = self.describe_shift(dL, da, db)
                # 예: "너무 어두움 (L -5.2), 약간 황색 편향 (b +3.1)"

                # 조치 사항
                action = f"Zone {tz['name']} 잉크 조정 필요: {direction}"
                adjustment = {
                    "dL": -dL,  # 보정값 (반대 방향)
                    "da": -da,
                    "db": -db,
                    "target_lab": sz["mean_lab"],
                    "current_lab": tz["mean_lab"],
                }

                recommendations.append({
                    "zone": tz["name"],
                    "action": action,
                    "adjustment": adjustment,
                    "priority": "high" if delta_e > 5.0 else "medium"
                })

        return recommendations
```

---

## 5. 단계별 구현 계획

### Phase 0: 기반 작업 (1주)
- [ ] 데이터베이스 스키마 설계 및 마이그레이션
- [ ] ORM 모델 정의 (SQLAlchemy)
- [ ] STD Profile 데이터 구조 정의 (Pydantic)

### Phase 1: STD 프로파일 생성 (2주)
- [ ] STDProfileManager 구현
  - [ ] create_std_profile() - 분석 + 저장
  - [ ] get_std_profile() - STD 조회
  - [ ] list_std_profiles() - SKU별 목록
- [ ] API 엔드포인트
  - [ ] POST /std/register - STD 등록
  - [ ] GET /std/{sku} - STD 조회
  - [ ] GET /std/list - 전체 목록
- [ ] Web UI
  - [ ] STD 등록 페이지
  - [ ] STD 상세 보기 페이지

### Phase 2: 비교 분석 엔진 (3주)
- [ ] ComparisonEngine 구현
  - [ ] calculate_structure_similarity()
    - [ ] Radial Profile 상관계수
    - [ ] DTW 거리 계산
    - [ ] Zone 경계 위치 비교
  - [ ] calculate_color_similarity()
    - [ ] Zone별 ΔE
    - [ ] 분포 유사도 (KS test)
  - [ ] 종합 점수 산출
- [ ] API 엔드포인트
  - [ ] POST /compare - Test Sample vs STD 비교
  - [ ] GET /compare/{id} - 비교 결과 조회
- [ ] Web UI
  - [ ] 비교 리포트 페이지
    - 구조 유사도 차트
    - 색상 유사도 표
    - 종합 점수 게이지

### Phase 3: 조치 권장 시스템 (2주)
- [ ] RecommendationEngine 구현
  - [ ] 색상 편차 → 잉크 조정 권장
  - [ ] 구조 편차 → 공정 조정 권장
- [ ] Web UI
  - [ ] 조치 권장 섹션 (카드 형태)
  - [ ] 조정 시뮬레이션 (슬라이더)

### Phase 4: 범위 관리 시스템 (2주)
- [ ] 상한/하한 설정 UI
- [ ] 범위 기반 자동 판정 로직
- [ ] 대시보드 (합격률, 트렌드)

---

## 6. 데이터베이스 설계

### 6.1 ERD

```
┌──────────────────────┐
│   std_profiles       │
├──────────────────────┤
│ id (PK)              │
│ sku_code             │
│ version              │
│ image_path           │
│ profile_data (JSON)  │◄───┐
│ color_data (JSON)    │    │
│ meta_data (JSON)     │    │
│ upper_limit (JSON)   │    │
│ lower_limit (JSON)   │    │
│ created_at           │    │
│ approved_by          │    │
└──────────────────────┘    │
                            │ FK
┌──────────────────────┐    │
│   test_samples       │    │
├──────────────────────┤    │
│ id (PK)              │    │
│ sku_code             │    │
│ image_path           │    │
│ analysis_result (JSON)│   │
│ std_profile_id (FK)  ├────┘
│ tested_at            │
│ tester_id            │
└──────────────────────┘
        │
        │ FK
        ▼
┌──────────────────────┐
│ comparison_results   │
├──────────────────────┤
│ id (PK)              │
│ test_sample_id (FK)  │
│ std_profile_id (FK)  │
│ structure_score      │
│ color_score          │
│ total_score          │
│ pass_fail            │
│ recommendations (JSON)│
│ created_at           │
└──────────────────────┘
```

### 6.2 JSON 스키마

#### profile_data (구조 정보)
```json
{
  "radial_profile": {
    "r": [0.0, 0.01, 0.02, ..., 1.0],
    "L": [72.5, 72.3, ..., 35.2],
    "a": [137.2, 137.5, ..., 45.8],
    "b": [122.8, 122.5, ..., 39.1]
  },
  "zone_boundaries": [
    {
      "zone": "C",
      "r_start": 0.0,
      "r_end": 0.38,
      "method": "optical_clear"
    },
    {
      "zone": "B",
      "r_start": 0.38,
      "r_end": 0.65,
      "method": "transition_based"
    },
    {
      "zone": "A",
      "r_start": 0.65,
      "r_end": 1.0,
      "method": "outer_band"
    }
  ],
  "transitions": [
    {
      "position": 0.38,
      "strength": 8.5,
      "width": 0.03
    }
  ]
}
```

#### color_data (색상 정보)
```json
{
  "zones": [
    {
      "name": "A",
      "mean_lab": [72.2, 137.3, 122.8],
      "std_lab": [2.5, 1.8, 2.1],
      "percentiles": {
        "L": {"p5": 68.2, "p25": 70.5, "p50": 72.2, "p75": 73.8, "p95": 76.1},
        "a": {...},
        "b": {...}
      },
      "pixel_count": 125000,
      "ink_pixel_count": 123500
    }
  ],
  "overall": {
    "mean_chroma": 185.3,
    "mean_hue": 41.8
  }
}
```

#### meta_data (메타 정보)
```json
{
  "capture": {
    "camera": "Canon EOS R5",
    "lens": "RF 100mm Macro",
    "iso": 100,
    "exposure": "1/200s",
    "aperture": "f/11",
    "white_balance": "5500K",
    "date": "2025-12-17T10:30:00Z"
  },
  "approval": {
    "approved_by": "홍길동",
    "approved_at": "2025-12-17T14:00:00Z",
    "notes": "최종 승인됨, 양산 기준으로 사용"
  },
  "ink": {
    "count": 2,
    "colors": ["#7C3830", "#A58073"],
    "batch": "INK-2025-001"
  }
}
```

#### upper_limit / lower_limit (범위 설정)
```json
{
  "structure": {
    "profile_correlation": 0.92,  // 최소 상관계수
    "zone_boundary_tolerance": 0.05  // ±5% 허용
  },
  "color": {
    "zone_delta_e_max": 5.0,  // Zone별 최대 ΔE
    "overall_delta_e_max": 4.0  // 전체 평균 최대 ΔE
  }
}
```

---

## 7. 비교 분석 알고리즘

### 7.1 구조 유사도

#### 7.1.1 Radial Profile 상관계수
```python
def profile_correlation(test_profile: np.ndarray, std_profile: np.ndarray) -> float:
    """
    두 프로파일 간 Pearson 상관계수
    1.0 = 완전 일치, 0.0 = 무관, -1.0 = 역상관
    """
    # L, a, b 채널별 상관계수 계산
    corr_L = np.corrcoef(test_profile["L"], std_profile["L"])[0, 1]
    corr_a = np.corrcoef(test_profile["a"], std_profile["a"])[0, 1]
    corr_b = np.corrcoef(test_profile["b"], std_profile["b"])[0, 1]

    # 가중 평균 (L 50%, a 25%, b 25%)
    return corr_L * 0.5 + corr_a * 0.25 + corr_b * 0.25
```

#### 7.1.2 DTW (Dynamic Time Warping)
```python
from dtaidistance import dtw

def profile_dtw_distance(test_profile: np.ndarray, std_profile: np.ndarray) -> float:
    """
    두 프로파일 간 DTW 거리
    작을수록 유사 (0 = 완전 일치)
    """
    # L 채널 DTW (가장 중요)
    distance_L = dtw.distance(test_profile["L"], std_profile["L"])

    # 정규화 (0~1)
    max_distance = np.max(std_profile["L"]) - np.min(std_profile["L"])
    normalized = distance_L / (max_distance * len(std_profile["L"]))

    return 1.0 - np.clip(normalized, 0, 1)  # 1에 가까울수록 유사
```

#### 7.1.3 Zone 경계 위치 비교
```python
def zone_boundary_similarity(test_zones: list, std_zones: list) -> float:
    """
    Zone 경계 위치가 얼마나 일치하는지
    """
    diffs = []
    for tz, sz in zip(test_zones, std_zones):
        # r_start, r_end 차이
        diff_start = abs(tz["r_start"] - sz["r_start"])
        diff_end = abs(tz["r_end"] - sz["r_end"])
        diffs.append((diff_start + diff_end) / 2)

    avg_diff = np.mean(diffs)
    # 5% 이내 차이 → 1.0, 10% 차이 → 0.5, 20% 차이 → 0.0
    return max(0, 1.0 - avg_diff / 0.2)
```

#### 7.1.4 종합 구조 점수
```python
def calculate_structure_score(test, std) -> float:
    """
    구조 유사도 종합 점수 (0~1)
    """
    corr = profile_correlation(test["radial_profile"], std["radial_profile"])
    dtw_score = profile_dtw_distance(test["radial_profile"], std["radial_profile"])
    boundary = zone_boundary_similarity(test["zone_boundaries"], std["zone_boundaries"])

    # 가중 평균
    return corr * 0.4 + dtw_score * 0.3 + boundary * 0.3
```

### 7.2 색상 유사도

#### 7.2.1 Zone별 ΔE
```python
def zone_color_similarity(test_zones: list, std_zones: list) -> dict:
    """
    Zone별 색상 유사도
    """
    zone_scores = {}
    for tz, sz in zip(test_zones, std_zones):
        delta_e = delta_e_cie76(tz["mean_lab"], sz["mean_lab"])

        # ΔE → 점수 변환 (0 = 1.0, 5 = 0.5, 10 = 0.0)
        score = max(0, 1.0 - delta_e / 10.0)

        zone_scores[tz["name"]] = {
            "delta_e": delta_e,
            "score": score,
            "current_lab": tz["mean_lab"],
            "target_lab": sz["mean_lab"]
        }

    return zone_scores
```

#### 7.2.2 분포 유사도
```python
from scipy.stats import ks_2samp

def distribution_similarity(test_pixels: np.ndarray, std_pixels: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov 테스트로 분포 유사도 측정
    """
    # L, a, b 채널별 KS 테스트
    ks_L = ks_2samp(test_pixels[:, 0], std_pixels[:, 0])
    ks_a = ks_2samp(test_pixels[:, 1], std_pixels[:, 1])
    ks_b = ks_2samp(test_pixels[:, 2], std_pixels[:, 2])

    # p-value가 높을수록 유사 (같은 분포)
    avg_p_value = (ks_L.pvalue + ks_a.pvalue + ks_b.pvalue) / 3

    return avg_p_value  # 0~1
```

#### 7.2.3 종합 색상 점수
```python
def calculate_color_score(test, std) -> float:
    """
    색상 유사도 종합 점수 (0~1)
    """
    zone_scores = zone_color_similarity(test["zones"], std["zones"])
    avg_zone_score = np.mean([z["score"] for z in zone_scores.values()])

    # 분포 유사도는 선택적 (픽셀 데이터 있을 때만)
    # dist_score = distribution_similarity(test_pixels, std_pixels)

    return avg_zone_score  # 또는 가중 평균
```

### 7.3 종합 점수 및 판정

```python
def calculate_total_score(structure_score: float, color_score: float) -> dict:
    """
    최종 점수 계산 및 판정
    """
    # 가중치: 색상 60%, 구조 40%
    total = structure_score * 0.4 + color_score * 0.6

    # 100점 환산
    total_100 = total * 100

    # 판정
    if total_100 >= 90:
        grade = "S (우수)"
        pass_fail = "PASS"
    elif total_100 >= 80:
        grade = "A (양호)"
        pass_fail = "PASS"
    elif total_100 >= 70:
        grade = "B (보통)"
        pass_fail = "WARNING"
    else:
        grade = "C (불량)"
        pass_fail = "FAIL"

    return {
        "total_score": total_100,
        "structure_score": structure_score * 100,
        "color_score": color_score * 100,
        "grade": grade,
        "pass_fail": pass_fail
    }
```

---

## 8. 우선순위 및 마일스톤

### 8.1 Phase 우선순위

| Phase | 우선순위 | 기간 | 의존성 |
|-------|---------|------|--------|
| **Phase 0: 기반 작업** | 🔴 P0 | 1주 | - |
| **Phase 1: STD 프로파일** | 🔴 P0 | 2주 | Phase 0 |
| **Phase 2: 비교 분석** | 🔴 P0 | 3주 | Phase 1 |
| **Phase 3: 조치 권장** | 🟡 P1 | 2주 | Phase 2 |
| **Phase 4: 범위 관리** | 🟢 P2 | 2주 | Phase 2 |

### 8.2 마일스톤

```
Week 1: ✅ DB 스키마, ORM 모델, 기본 CRUD
Week 2-3: ✅ STD 등록 기능 (API + UI)
Week 4-6: ✅ 비교 분석 엔진 (알고리즘 + API)
Week 7-8: ✅ 조치 권장 시스템
Week 9-10: ✅ 범위 관리 및 대시보드

MVP: Week 6 (STD 등록 + 비교 분석 기본)
v1.0: Week 10 (전체 기능 완성)
```

### 8.3 핵심 성공 지표 (KPI)

1. **STD 등록 완료율**: SKU당 최소 1개 STD 등록
2. **비교 정확도**: 수동 검사 vs 자동 비교 일치율 > 95%
3. **조치 권장 정확도**: 엔지니어 피드백 기반 개선
4. **처리 속도**: 비교 분석 < 3초 (이미지 당)

---

## 9. 다음 단계

### 즉시 착수 가능한 작업
1. ✅ **본 계획 문서 검토 및 승인** (현재)
2. ⏭️ DB 스키마 설계 (SQLAlchemy models)
3. ⏭️ STDProfile 데이터 클래스 정의 (Pydantic)
4. ⏭️ 구조 유사도 알고리즘 프로토타입 (Jupyter Notebook)

### 검토 필요 사항
- [ ] DB 선택: SQLite (개발) vs PostgreSQL (운영)?
- [ ] 이미지 저장 위치: DB blob vs 파일 시스템?
- [ ] STD 버전 관리 전략: 단일 active vs 다중 버전?
- [ ] 범위 설정 단위: SKU별 vs Zone별?

---

**이상으로 STD 기준 품질 관리 시스템 구축 계획을 종료합니다.**

---
---

## 4. 기술 스택

### 4.1 현재 기술 스택

```yaml
언어: Python 3.10+

이미지 처리:
  - opencv-python: 4.8.0+ (이미지 로드, 극좌표 변환, Hough Circle)
  - Pillow: 10.0.0+ (이미지 포맷 처리)

수치 계산:
  - numpy: 1.24.0+ (배열 연산)
  - scipy: 1.11.0+ (신호 처리, 통계)

색상 과학:
  - Custom CIEDE2000 구현 (src/utils/color_delta.py)

시각화:
  - matplotlib: 3.8.0+
  - seaborn: 0.13.0+
  - plotly: 5.18.0+ (Web UI 차트)

웹 프레임워크:
  - FastAPI: 0.104.0+ (REST API)
  - uvicorn: 0.24.0+ (ASGI 서버)
  - Jinja2: 3.1.0+ (템플릿)

데이터 관리:
  - pandas: 2.1.0+ (CSV 처리)
  - SQLAlchemy: 2.0.0+ (ORM, 현재 미사용)

머신러닝:
  - scikit-learn: 1.3.0+ (GMM 클러스터링)

테스트:
  - pytest: 7.4.0+ (319개 테스트, 94.7% 커버리지)
  - pytest-cov: 4.1.0+

코드 품질:
  - black: 23.0.0+ (포맷팅)
  - flake8: 7.0.0+ (린팅)
  - mypy: 1.8.0+ (타입 체킹)
  - pre-commit: 4.0.0+ (자동화)
```

### 4.2 추가 필요 기술

```yaml
DB:
  - SQLite (개발/소규모) 또는 PostgreSQL (운영/대규모)
  - Alembic (마이그레이션)

비교 알고리즘:
  - dtaidistance: DTW 계산 (추가 필요)
  - scipy.stats: KS test (이미 있음)

프론트엔드:
  - Chart.js 또는 Plotly.js (비교 차트)
  - Bootstrap 5 (이미 있음)
```

---

## 5. 데이터 구조 예시

### 5.1 현재 SKU 설정 (JSON)

**파일**: `config/sku_db/SKU001.json`

```json
{
  "sku_code": "SKU001",
  "description": "3-zone colored contact lens (brown printing)",
  "default_threshold": 8.0,
  "zones": {
    "A": {
      "L": 45.0,
      "a": 8.0,
      "b": 28.0,
      "threshold": 8.0,
      "description": "Outer zone (darkest brown)"
    },
    "B": {
      "L": 68.0,
      "a": 5.0,
      "b": 22.0,
      "threshold": 8.0,
      "description": "Middle zone (medium brown)"
    },
    "C": {
      "L": 95.0,
      "a": 0.5,
      "b": 2.0,
      "threshold": 10.0,
      "description": "Inner zone (near transparent)"
    }
  },
  "params": {
    "expected_zones": 3,
    "optical_clear_ratio": 0.15
  },
  "metadata": {
    "created_at": "2025-12-11",
    "baseline_samples": 5,
    "last_updated": "2025-12-11T21:56:00",
    "calibration_method": "Measured from dummy data"
  }
}
```

**한계**: L, a, b 평균값만 있음 (구조, 분포, 프로파일 없음)

### 5.2 현재 분석 결과 (메모리)

```json
{
  "judgment": "OK",
  "overall_delta_e": 2.5,
  "confidence": 0.87,
  "zones": [
    {
      "zone_name": "A",
      "measured_lab": [45.2, 8.3, 28.5],
      "target_lab": [45.0, 8.0, 28.0],
      "delta_e": 2.1,
      "threshold": 8.0,
      "is_ok": true,
      "pixel_count": 125000
    }
  ],
  "ink_analysis": {
    "zone_based": {
      "detected_ink_count": 3,
      "inks": [...]
    },
    "image_based": {
      "detected_ink_count": 2,
      "inks": [...]
    }
  }
}
```

**한계**: DB에 저장 안 됨, 구조 정보 없음

---

## 6. 검토 체크리스트

### 6.1 기술적 검토

- [ ] **아키텍처**: 제안된 3-tier 구조 (UI/Backend/DB)가 적절한가?
- [ ] **DB 스키마**: JSON 필드 사용이 적절한가? 정규화가 필요한가?
- [ ] **알고리즘 선택**:
  - [ ] Pearson 상관계수 vs Spearman/Kendall?
  - [ ] DTW vs Euclidean distance?
  - [ ] KS test vs Wasserstein distance?
- [ ] **성능**: 프로파일 비교 < 3초가 현실적인가?
  - Radial profile 길이: ~500 포인트
  - DTW 복잡도: O(n²)
  - 최적화 필요 여부?
- [ ] **확장성**:
  - SKU 100개, 일일 검사 1000건 처리 가능한가?
  - 이미지 저장 전략 (파일 시스템 vs DB blob)?

### 6.2 기능 검토

- [ ] **STD 버전 관리**:
  - SKU당 1개 active STD vs 다중 버전?
  - 버전 업데이트 시 기존 결과 재평가 필요 여부?
- [ ] **비교 매칭**:
  - SKU 기반 자동 매칭만으로 충분한가?
  - Lot 번호, 생산 일자 추가 필요 여부?
- [ ] **조치 권장**:
  - LAB 차이 → 잉크 조정량 변환 공식 현실적인가?
  - 엔지니어 피드백 루프 필요 여부?
- [ ] **범위 설정**:
  - 상한/하한을 자동 계산하는 방법은? (평균 ± 3σ?)
  - Zone별로 다른 허용 범위가 필요한가?

### 6.3 사용성 검토

- [ ] **사용자 시나리오**:
  - 3가지 시나리오 (STD 등록/양산 검사/범위 설정)로 충분한가?
  - 추가 시나리오: 배치 검사, 리포트 생성, 히스토리 조회?
- [ ] **UI/UX**:
  - 비교 리포트 화면 구성은?
  - 조치 권장을 어떻게 표시할 것인가?
  - 모바일 지원 필요 여부?

### 6.4 운영 검토

- [ ] **보안**:
  - 사용자 인증/권한 필요 여부? (STD 승인, 결과 수정 방지)
  - 감사 로그 (Audit Trail) 필요 여부?
- [ ] **백업**:
  - STD 프로파일 백업 전략?
  - 이미지 파일 백업 전략?
- [ ] **모니터링**:
  - 시스템 헬스 체크?
  - 알림 (합격률 급감 시)?

### 6.5 리스크 검토

- [ ] **기술 리스크**:
  - DTW 라이브러리 의존성 문제?
  - 프로파일 길이 불일치 처리?
- [ ] **데이터 리스크**:
  - STD와 Test Sample의 Zone 개수 불일치?
  - 렌즈 검출 실패 시 처리?
- [ ] **일정 리스크**:
  - 10주 일정의 버퍼는?
  - 우선순위 조정 전략은?

---

## 7. 추가 고려사항 제안

### 7.1 단기 (Phase 0-2)

- **DTW 대안 검토**: scipy.spatial.distance로 충분할 수도
- **DB 마이그레이션**: Alembic 스크립트 자동화
- **프로파일 정규화**: 길이 불일치 시 interpolation
- **캐싱**: 자주 조회되는 STD 프로파일 Redis 캐시

### 7.2 중기 (Phase 3-4)

- **조치 권장 학습**: 엔지니어 피드백 기반 ML 모델
- **범위 자동 설정**: 통계적 공정 관리 (SPC) 적용
- **트렌드 분석**: 시계열 분석으로 공정 변화 감지

### 7.3 장기 (v2.0)

- **실시간 모니터링**: 양산 라인 직결
- **AI 이상 감지**: Anomaly Detection (Isolation Forest, Autoencoder)
- **다중 촬영 조건**: 조명, 각도 변화 보정

---

## 📝 검토 결과 제출 양식

AI 검토자는 다음 형식으로 피드백을 주시기 바랍니다:

```markdown
# STD QC 시스템 계획 검토 결과

## ✅ 긍정적 측면
1. ...
2. ...

## ⚠️ 우려 사항
1. ...
2. ...

## 🔧 개선 제안
1. ...
2. ...

## 🚨 치명적 리스크 (있다면)
1. ...
2. ...

## 📊 종합 의견
- 기술적 타당성: ⭐⭐⭐⭐☆ (4/5)
- 일정 현실성: ⭐⭐⭐☆☆ (3/5)
- 우선순위 적절성: ⭐⭐⭐⭐⭐ (5/5)
- 전반적 평가: [요약 한 문장]
```

---

**문서 끝**

**작성자**: Human + Claude Sonnet 4.5
**검토 요청일**: 2025-12-17
**문서 위치**: `docs/planning/AI_REVIEW_CONTEXT_BUNDLE.md`
