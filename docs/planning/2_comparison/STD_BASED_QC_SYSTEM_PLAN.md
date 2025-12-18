# STD 기준 품질 관리 시스템 구축 계획

> **작성일**: 2025-12-17
> **목적**: 표준 이미지(STD) 기반 양산 품질 관리 시스템 구축 로드맵
> **상태**: 🔴 계획 수립 단계

---

## 📋 목차

1. [현재 상황 분석](#1-현재-상황-분석)
2. [목표 시스템 정의](#2-목표-시스템-정의)
3. [갭 분석](#3-갭-분석)
4. [시스템 아키텍처](#4-시스템-아키텍처)
5. [단계별 구현 계획](#5-단계별-구현-계획)
6. [데이터베이스 설계](#6-데이터베이스-설계)
7. [비교 분석 알고리즘](#7-비교-분석-알고리즘)
8. [우선순위 및 마일스톤](#8-우선순위-및-마일스톤)

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

**다음 문서**:
- `DB_SCHEMA_DESIGN.md` - 데이터베이스 상세 설계
- `COMPARISON_ALGORITHM_SPEC.md` - 비교 알고리즘 상세 명세
- `STD_MANAGEMENT_UI_DESIGN.md` - STD 관리 UI 설계

---

**작성자**: Claude Sonnet 4.5
**프로젝트**: Contact Lens Color Inspection System
**문서 위치**: `docs/planning/STD_BASED_QC_SYSTEM_PLAN.md`
