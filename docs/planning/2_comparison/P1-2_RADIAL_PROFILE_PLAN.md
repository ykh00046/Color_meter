# P1-2: Radial Profile Comparison Implementation Plan

> **작성일**: 2025-12-18
> **목적**: P1-2 - Radial Profile 비교 기능 추가
> **상태**: 🟡 구현 진행 중

---

## 📋 목차

1. [현재 상태](#1-현재-상태)
2. [P1-2 목표](#2-p1-2-목표)
3. [설계](#3-설계)
4. [구현 계획](#4-구현-계획)
5. [테스트 계획](#5-테스트-계획)

---

## 1. 현재 상태

### 1.1 M3에서 구현된 것

- ✅ Ink Comparison (ink_score, ink_details)
- ✅ Zone-based Comparison (zone_score, zone_details)
- ✅ Total score 통합 (zone 50%, ink 30%, confidence 20%)

### 1.2 RadialProfiler 현황

**위치**: `src/core/radial_profiler.py`

**RadialProfile 구조**:
```python
@dataclass
class RadialProfile:
    r_normalized: np.ndarray  # 정규화된 반경 (0-1)
    L: np.ndarray             # L* 프로파일
    a: np.ndarray             # a* 프로파일
    b: np.ndarray             # b* 프로파일
    std_L: np.ndarray         # L* 표준편차
    std_a: np.ndarray         # a* 표준편차
    std_b: np.ndarray         # b* 표준편차
    pixel_count: np.ndarray   # 픽셀 개수
```

**사용처**:
- InspectionPipeline: radial_profile 생성 및 저장
- ZoneSegmenter: radial_profile로부터 zone 분할

### 1.3 문제점: radial_profile이 저장되지 않음 ⚠️

**현황**:
- ✅ InspectionPipeline은 radial_profile 생성 (inspection_result.radial_profile)
- ❌ zone_analyzer_2d는 radial_profile 저장하지 않음
- ❌ TestSample/STDSample의 analysis_result에 radial_profile 없음

**원인**:
- TestService/STDService는 zone_analyzer_2d를 사용
- zone_analyzer_2d는 InspectionResult에 radial_profile 포함하지 않음

**해결 방안**:
1. **Option A**: zone_analyzer_2d 수정하여 radial_profile 추가
2. **Option B**: TestService/STDService에서 RadialProfiler 별도 호출
3. **Option C**: ComparisonService에서 이미지 재로드하여 radial_profile 생성

---

## 2. P1-2 목표

### 2.1 핵심 질문

**"STD와 TEST의 Radial Profile이 구조적으로 유사한가?"**

### 2.2 구현 범위

#### ✅ 포함

1. **Pearson Correlation Coefficient**
   - L, a, b 각 채널별 상관계수
   - 전체 평균 상관계수

2. **Structural Similarity**
   - Profile shape similarity
   - Gradient similarity (구조 변화 패턴)

3. **profile_score 계산**
   - Correlation과 Structural Similarity 종합 점수 (0-100)

4. **profile_details 저장**
   - 비교 결과 상세 정보

#### ❌ 제외 (P1-3 이후)

- DTW (Dynamic Time Warping) alignment
- Profile segmentation (zone 재분할)
- Hotspot detection (국소 이상 감지)

---

## 3. 설계

### 3.1 Radial Profile 저장 방안

**선택: Option A - zone_analyzer_2d 수정**

**이유**:
- ✅ 한 곳에서만 수정 (관심사 집중)
- ✅ 기존 InspectionPipeline과 일관성 유지
- ✅ 추후 모든 분석에서 radial_profile 사용 가능

**구현**:
```python
# zone_analyzer_2d.py
def analyze_2d(...) -> Tuple[InspectionResult, Dict]:
    ...
    # 1. RadialProfiler 생성
    from src.core.radial_profiler import RadialProfiler, ProfilerConfig
    profiler = RadialProfiler(ProfilerConfig())

    # 2. Radial profile 추출
    radial_profile = profiler.extract_profile(img_bgr, lens_detection)

    # 3. radial_profile을 dict로 변환하여 InspectionResult에 포함
    radial_profile_dict = {
        "r_normalized": radial_profile.r_normalized.tolist(),
        "L": radial_profile.L.tolist(),
        "a": radial_profile.a.tolist(),
        "b": radial_profile.b.tolist(),
        "std_L": radial_profile.std_L.tolist(),
        "std_a": radial_profile.std_a.tolist(),
        "std_b": radial_profile.std_b.tolist(),
        "pixel_count": radial_profile.pixel_count.tolist(),
    }

    # 4. InspectionResult에 radial_profile 추가
    result = InspectionResult(
        ...
        radial_profile=radial_profile_dict,  # 추가
    )
```

**InspectionResult 수정**:
```python
# color_evaluator.py
@dataclass
class InspectionResult:
    ...
    radial_profile: Optional[Dict[str, List[float]]] = None  # 추가
```

### 3.2 Profile 비교 알고리즘

#### Step 1: 데이터 추출 및 정규화

```python
def _compare_radial_profiles(
    test_radial_profile: Dict[str, List[float]],
    std_radial_profile: Dict[str, List[float]]
) -> Dict[str, Any]:
    """
    Radial profile 비교

    Returns:
        {
            "correlation": {
                "L": 0.95,
                "a": 0.92,
                "b": 0.90,
                "avg": 0.92
            },
            "structural_similarity": 0.88,
            "gradient_similarity": 0.85,
            "profile_score": 89.5,
            "message": "High correlation (r=0.92)"
        }
    """
    # 1. numpy 배열로 변환
    test_L = np.array(test_radial_profile["L"])
    test_a = np.array(test_radial_profile["a"])
    test_b = np.array(test_radial_profile["b"])

    std_L = np.array(std_radial_profile["L"])
    std_a = np.array(std_radial_profile["a"])
    std_b = np.array(std_radial_profile["b"])

    # 2. 길이 확인 (다를 경우 interpolation)
    if len(test_L) != len(std_L):
        # Interpolate to match lengths
        ...
```

#### Step 2: Pearson Correlation Coefficient

```python
from scipy.stats import pearsonr

def _calculate_correlation(profile1: np.ndarray, profile2: np.ndarray) -> float:
    """Calculate Pearson correlation coefficient"""
    if len(profile1) < 2 or len(profile2) < 2:
        return 0.0

    corr, p_value = pearsonr(profile1, profile2)
    return float(corr)

# L, a, b 각 채널별 계산
corr_L = _calculate_correlation(test_L, std_L)
corr_a = _calculate_correlation(test_a, std_a)
corr_b = _calculate_correlation(test_b, std_b)
corr_avg = (corr_L + corr_a + corr_b) / 3.0
```

#### Step 3: Structural Similarity

**방법 1: SSIM (Structural Similarity Index Measure)**
```python
from skimage.metrics import structural_similarity as ssim

# Normalize profiles to 0-1 range
test_L_norm = (test_L - test_L.min()) / (test_L.max() - test_L.min() + 1e-8)
std_L_norm = (std_L - std_L.min()) / (std_L.max() - std_L.min() + 1e-8)

# Calculate SSIM (1D version)
ssim_L = ssim(test_L_norm, std_L_norm, data_range=1.0)
```

**방법 2: Gradient Similarity**
```python
# Calculate gradients
test_grad = np.gradient(test_L)
std_grad = np.gradient(std_L)

# Correlation of gradients (구조 변화 패턴 유사도)
grad_similarity = _calculate_correlation(test_grad, std_grad)
```

#### Step 4: profile_score 계산

```python
def _calculate_profile_score(
    correlation_avg: float,
    structural_similarity: float,
    gradient_similarity: float
) -> float:
    """
    Calculate overall profile score (0-100)

    Weights:
    - Correlation: 50%
    - Structural Similarity: 30%
    - Gradient Similarity: 20%
    """
    # Convert correlation (-1 to 1) to score (0-100)
    corr_score = (correlation_avg + 1) * 50.0  # -1→0, 0→50, 1→100

    # Convert SSIM (-1 to 1) to score (0-100)
    ssim_score = (structural_similarity + 1) * 50.0

    # Convert gradient corr (-1 to 1) to score (0-100)
    grad_score = (gradient_similarity + 1) * 50.0

    # Weighted average
    profile_score = (
        corr_score * 0.5 +
        ssim_score * 0.3 +
        grad_score * 0.2
    )

    return max(0.0, min(100.0, profile_score))
```

### 3.3 데이터 구조

#### profile_details (ComparisonResult)

```python
profile_details = {
    "correlation": {
        "L": 0.95,
        "a": 0.92,
        "b": 0.90,
        "avg": 0.92
    },
    "structural_similarity": {
        "L": 0.88,
        "a": 0.85,
        "b": 0.87,
        "avg": 0.87
    },
    "gradient_similarity": {
        "L": 0.82,
        "a": 0.80,
        "b": 0.83,
        "avg": 0.82
    },
    "profile_score": 89.5,
    "length_match": True,
    "test_length": 348,
    "std_length": 348,
    "message": "High correlation (r=0.92), strong structural similarity"
}
```

### 3.4 Schema 업데이트

**comparison_schemas.py**:

```python
class CorrelationData(BaseModel):
    """Correlation coefficients"""
    L: float = Field(..., description="L* correlation", ge=-1, le=1)
    a: float = Field(..., description="a* correlation", ge=-1, le=1)
    b: float = Field(..., description="b* correlation", ge=-1, le=1)
    avg: float = Field(..., description="Average correlation", ge=-1, le=1)

class StructuralSimilarityData(BaseModel):
    """Structural similarity metrics"""
    L: float = Field(..., description="L* SSIM", ge=-1, le=1)
    a: float = Field(..., description="a* SSIM", ge=-1, le=1)
    b: float = Field(..., description="b* SSIM", ge=-1, le=1)
    avg: float = Field(..., description="Average SSIM", ge=-1, le=1)

class ProfileDetailsData(BaseModel):
    """Radial profile comparison details (P1-2)"""
    correlation: CorrelationData = Field(..., description="Pearson correlation coefficients")
    structural_similarity: StructuralSimilarityData = Field(..., description="SSIM metrics")
    gradient_similarity: CorrelationData = Field(..., description="Gradient correlations")
    profile_score: float = Field(..., description="Overall profile score (0-100)", ge=0, le=100)
    length_match: bool = Field(..., description="Whether profile lengths match")
    test_length: int = Field(..., description="Test profile length", ge=0)
    std_length: int = Field(..., description="STD profile length", ge=0)
    message: Optional[str] = Field(None, description="Summary message")

class ComparisonDetailResponse(BaseModel):
    # ... existing fields ...
    profile_details: Optional[ProfileDetailsData] = Field(None, description="Radial profile comparison details (P1-2)")
```

### 3.5 total_score 업데이트

**M3 → P1-2 변경**:

```python
# M3 (Before P1-2)
total_score = zone_score * 0.5 + ink_score * 0.3 + confidence_score * 0.2

# P1-2 (After profile added)
total_score = (
    zone_score * 0.35 +      # 50% → 35%
    ink_score * 0.25 +       # 30% → 25%
    profile_score * 0.25 +   # NEW: 25%
    confidence_score * 0.15  # 20% → 15%
)
```

**Fallback Logic**:
```python
if profile_details:
    # P1-2: Include profile_score
    total_score = zone * 0.35 + ink * 0.25 + profile * 0.25 + confidence * 0.15
elif ink_details:
    # M3 fallback: No profile
    total_score = zone * 0.5 + ink * 0.3 + confidence * 0.2
else:
    # M2 fallback: No ink, no profile
    total_score = zone_score
```

---

## 4. 구현 계획

### 4.1 작업 순서

#### Task 1: InspectionResult에 radial_profile 필드 추가
- [ ] src/core/color_evaluator.py 수정
- [ ] InspectionResult에 radial_profile: Optional[Dict] 추가
- [ ] 타입 힌트 및 docstring 업데이트

#### Task 2: zone_analyzer_2d에 RadialProfiler 통합
- [ ] src/core/zone_analyzer_2d.py 수정
- [ ] RadialProfiler import 및 생성
- [ ] radial_profile 추출 및 dict 변환
- [ ] InspectionResult에 radial_profile 포함

#### Task 3: 기존 STD/Test 샘플 재분석
- [ ] STD 샘플 재등록 (radial_profile 포함)
- [ ] Test 샘플 재등록 (radial_profile 포함)

#### Task 4: ComparisonService._compare_radial_profiles() 구현
- [ ] _calculate_correlation() helper 메서드
- [ ] _calculate_structural_similarity() helper 메서드
- [ ] _calculate_profile_score() 메서드
- [ ] Profile length mismatch 처리 (interpolation)

#### Task 5: ComparisonService._calculate_scores() 수정
- [ ] profile_score를 total_score에 포함
- [ ] 가중치 조정: zone 35%, ink 25%, profile 25%, confidence 15%
- [ ] Fallback logic 추가

#### Task 6: Schema 업데이트
- [ ] CorrelationData, StructuralSimilarityData, ProfileDetailsData 추가
- [ ] ComparisonDetailResponse에 profile_details 추가
- [ ] __init__.py export 업데이트

#### Task 7: 테스트
- [ ] End-to-end 테스트 (STD 등록 → Test 등록 → 비교)
- [ ] profile_score 계산 검증
- [ ] profile_details 데이터 검증
- [ ] API 응답 검증

### 4.2 코드 위치

**수정 파일**:
- `src/core/color_evaluator.py` - InspectionResult
- `src/core/zone_analyzer_2d.py` - RadialProfiler 통합
- `src/services/comparison_service.py` - _compare_radial_profiles(), _calculate_scores()
- `src/schemas/comparison_schemas.py` - ProfileDetailsData

---

## 5. 테스트 계획

### 5.1 단위 테스트

#### Test Case 1: 완벽한 일치 (Perfect Match)
```python
test_profile = {"L": [50.0, 60.0, 70.0], "a": [10.0, 12.0, 14.0], "b": [20.0, 22.0, 24.0]}
std_profile = {"L": [50.0, 60.0, 70.0], "a": [10.0, 12.0, 14.0], "b": [20.0, 22.0, 24.0]}

result = _compare_radial_profiles(test_profile, std_profile)
assert result["correlation"]["avg"] == 1.0
assert result["profile_score"] == 100.0
```

#### Test Case 2: 높은 상관계수 (High Correlation)
```python
test_profile = {"L": [50.0, 60.0, 70.0], "a": [10.0, 12.0, 14.0], "b": [20.0, 22.0, 24.0]}
std_profile = {"L": [51.0, 61.0, 71.0], "a": [10.5, 12.5, 14.5], "b": [20.2, 22.2, 24.2]}

result = _compare_radial_profiles(test_profile, std_profile)
assert result["correlation"]["avg"] > 0.95
assert result["profile_score"] > 90.0
```

#### Test Case 3: 낮은 상관계수 (Low Correlation)
```python
test_profile = {"L": [50.0, 60.0, 70.0], "a": [10.0, 12.0, 14.0], "b": [20.0, 22.0, 24.0]}
std_profile = {"L": [70.0, 60.0, 50.0], "a": [14.0, 12.0, 10.0], "b": [24.0, 22.0, 20.0]}  # Reversed

result = _compare_radial_profiles(test_profile, std_profile)
assert result["correlation"]["avg"] < -0.5  # Negative correlation
assert result["profile_score"] < 30.0
```

#### Test Case 4: 길이 불일치 (Length Mismatch)
```python
test_profile = {"L": [50.0, 60.0, 70.0], "a": [10.0, 12.0, 14.0], "b": [20.0, 22.0, 24.0]}
std_profile = {"L": [50.0, 60.0], "a": [10.0, 12.0], "b": [20.0, 22.0]}  # Shorter

result = _compare_radial_profiles(test_profile, std_profile)
assert result["length_match"] == False
assert result["profile_score"] >= 0.0  # Should handle gracefully
```

### 5.2 통합 테스트

**Workflow**:
1. STD 등록 (SKU001 이미지)
   - RadialProfiler로 radial_profile 생성
   - radial_profile 저장 확인

2. Test Sample 등록 (SKU001 이미지)
   - RadialProfiler로 radial_profile 생성
   - radial_profile 저장 확인

3. 비교 실행
   - profile_details 생성 확인
   - profile_score 계산 확인
   - total_score에 profile_score 반영 확인

4. API 응답 검증
   - GET /api/compare/{id}에서 profile_details 반환 확인

---

## 6. 성공 기준

### 6.1 기능 요구사항
- [ ] radial_profile이 TestSample/STDSample에 저장됨
- [ ] _compare_radial_profiles() 메서드 구현
- [ ] profile_score 계산 (0-100)
- [ ] profile_details 생성
- [ ] Schema 업데이트
- [ ] API 응답에 profile 데이터 포함

### 6.2 품질 요구사항
- [ ] 단위 테스트 4개 이상
- [ ] 통합 테스트 통과
- [ ] 코드 커버리지 > 80%
- [ ] Flake8, Black 통과

### 6.3 성능 요구사항
- [ ] Profile 비교 처리 시간 < 50ms

---

## 7. 다음 단계 (P1-3)

P1-2 완료 후:
- **P1-3**: DTW (Dynamic Time Warping) 조건부 실행 (상관계수 < 0.80인 경우만)

---

**작성자**: Claude Sonnet 4.5
**프로젝트**: Contact Lens Color Inspection System
**문서 위치**: `docs/planning/2_comparison/P1-2_RADIAL_PROFILE_PLAN.md`
