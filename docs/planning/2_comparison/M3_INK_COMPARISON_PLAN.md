# M3: Ink Comparison Implementation Plan

> **작성일**: 2025-12-18
> **목적**: P1-1 - GMM 기반 잉크 비교 기능 추가
> **상태**: 🟡 구현 진행 중

---

## 📋 목차

1. [현재 상태](#1-현재-상태)
2. [M3 목표](#2-m3-목표)
3. [설계](#3-설계)
4. [구현 계획](#4-구현-계획)
5. [테스트 계획](#5-테스트-계획)

---

## 1. 현재 상태

### 1.1 M2에서 구현된 것

- ✅ TestService (test sample 등록)
- ✅ ComparisonService (zone-based comparison)
- ✅ Zone 색상 비교 (color_score)
- ✅ 보수적 판정 로직 (PASS/FAIL/RETAKE/MANUAL_REVIEW)
- ✅ Top-3 failure reasons

### 1.2 M2에서 Placeholder로 남긴 것

```python
# ComparisonResult (line 135-141)
ink_score=0.0,  # MVP: not implemented yet
ink_details=None,  # MVP: not implemented
```

### 1.3 기존 InkEstimator (이미 구현됨)

**위치**: `src/core/ink_estimator.py`

**주요 기능**:
- GMM + BIC로 최적 잉크 개수 추정 (1-3개)
- 잉크별 LAB, RGB, HEX 색상
- 픽셀 비율 (weight)
- 혼합색 보정 (3개 → 2개)

**반환 구조**:
```python
{
  "ink_count": 2,
  "inks": [
    {
      "weight": 0.65,
      "lab": [72.2, 137.3, 122.8],
      "rgb": [255, 180, 165],
      "hex": "#FFB4A5",
      "is_mix": False
    }
  ],
  "meta": {
    "bic": 12345.6,
    "sample_count": 15000,
    "correction_applied": True,
    "sampling_config": {...}
  }
}
```

---

## 2. M3 목표

### 2.1 핵심 질문

**"STD와 TEST의 잉크가 같은가?"**

### 2.2 구현 범위

#### ✅ 포함
1. **잉크 개수 비교**
   - STD와 TEST의 잉크 개수가 일치하는가?

2. **잉크 색상 비교**
   - 각 잉크 쌍의 LAB ΔE 계산
   - 잉크 매칭 (weight-based matching)

3. **ink_score 계산**
   - 잉크 개수 + 색상 유사도 종합 점수 (0-100)

4. **ink_details 저장**
   - 비교 결과 상세 정보

#### ❌ 제외 (P1-3 이후)
- Defect Classification (UNDERDOSE/OVERDOSE)
- 조치 권장 (잉크 조정량)
- Hotspot 감지

---

## 3. 설계

### 3.1 Ink 비교 알고리즘

#### Step 1: 잉크 개수 체크
```python
def _compare_inks(test_ink_analysis, std_ink_analysis):
    test_count = test_ink_analysis["ink_count"]
    std_count = std_ink_analysis["ink_count"]

    if test_count != std_count:
        return {
            "ink_count_match": False,
            "ink_pairs": [],
            "ink_score": 0.0,
            "message": f"Ink count mismatch: TEST={test_count}, STD={std_count}"
        }
```

#### Step 2: 잉크 매칭 (Weight-based)
```python
# 무게 기준으로 정렬 (가장 많이 사용된 잉크부터)
test_inks_sorted = sorted(test_inks, key=lambda x: x["weight"], reverse=True)
std_inks_sorted = sorted(std_inks, key=lambda x: x["weight"], reverse=True)

# 순서대로 페어링
ink_pairs = []
for test_ink, std_ink in zip(test_inks_sorted, std_inks_sorted):
    delta_e = calculate_delta_e(test_ink["lab"], std_ink["lab"])

    ink_pairs.append({
        "test_ink": test_ink,
        "std_ink": std_ink,
        "delta_e": delta_e,
        "weight_diff": abs(test_ink["weight"] - std_ink["weight"])
    })
```

**매칭 전략**:
- Weight-based matching (주 잉크부터 매칭)
- Alternative: Hungarian algorithm (최적 매칭, P2)

#### Step 3: ink_score 계산
```python
def _calculate_ink_score(ink_pairs):
    """
    ink_score 계산 (0-100)

    Components:
    - Color similarity: ΔE → score (70%)
    - Weight similarity: weight diff → score (30%)
    """

    if not ink_pairs:
        return 0.0

    color_scores = []
    weight_scores = []

    for pair in ink_pairs:
        # Color score: ΔE=0 → 100, ΔE=10 → 0
        color_score = max(0, 100 - pair["delta_e"] * 10)
        color_scores.append(color_score)

        # Weight score: diff=0 → 100, diff=0.3 → 0
        weight_score = max(0, 100 - pair["weight_diff"] * 333)
        weight_scores.append(weight_score)

    # 가중 평균
    avg_color = sum(color_scores) / len(color_scores)
    avg_weight = sum(weight_scores) / len(weight_scores)

    ink_score = avg_color * 0.7 + avg_weight * 0.3

    return ink_score
```

### 3.2 데이터 구조

#### ink_details (ComparisonResult)
```python
ink_details = {
    "ink_count_match": True,
    "test_ink_count": 2,
    "std_ink_count": 2,
    "ink_pairs": [
        {
            "rank": 1,  # Primary ink
            "test_ink": {
                "weight": 0.65,
                "lab": [72.2, 137.3, 122.8],
                "hex": "#FFB4A5"
            },
            "std_ink": {
                "weight": 0.63,
                "lab": [72.5, 136.8, 123.1],
                "hex": "#FFB3A4"
            },
            "delta_e": 0.8,
            "weight_diff": 0.02,
            "color_score": 92.0,
            "weight_score": 93.3,
            "pair_score": 92.4
        }
    ],
    "avg_delta_e": 1.2,
    "max_delta_e": 1.5,
    "ink_score": 88.5
}
```

### 3.3 Schema 업데이트

**comparison_schemas.py**:
```python
class InkPairData(BaseModel):
    rank: int = Field(..., description="Ink rank (1=primary)")
    test_ink: Dict[str, Any] = Field(..., description="Test ink data")
    std_ink: Dict[str, Any] = Field(..., description="STD ink data")
    delta_e: float = Field(..., description="Color difference")
    weight_diff: float = Field(..., description="Weight difference")
    color_score: float = Field(..., description="Color similarity score (0-100)")
    weight_score: float = Field(..., description="Weight similarity score (0-100)")
    pair_score: float = Field(..., description="Overall pair score")

class InkDetailsData(BaseModel):
    ink_count_match: bool = Field(..., description="Ink count match")
    test_ink_count: int = Field(..., description="Test ink count")
    std_ink_count: int = Field(..., description="STD ink count")
    ink_pairs: List[InkPairData] = Field(..., description="Ink pair comparisons")
    avg_delta_e: float = Field(..., description="Average ink ΔE")
    max_delta_e: float = Field(..., description="Maximum ink ΔE")
    ink_score: float = Field(..., description="Overall ink score (0-100)")

class ComparisonDetailResponse(BaseModel):
    # ... existing fields ...
    ink_details: Optional[InkDetailsData] = Field(None, description="Ink comparison details")
```

---

## 4. 구현 계획

### 4.1 작업 순서

#### Task 1: ink_analysis 존재 여부 확인
- [ ] TestSample.analysis_result에 ink_analysis 필드가 있는지 확인
- [ ] STDSample.analysis_result에 ink_analysis 필드가 있는지 확인
- [ ] 없으면 TestService/STDService에서 InkEstimator 호출 추가

#### Task 2: ComparisonService._compare_inks() 구현
- [ ] 잉크 개수 체크
- [ ] 잉크 매칭 (weight-based)
- [ ] ink_pairs 생성
- [ ] ink_score 계산

#### Task 3: ComparisonService._calculate_scores() 수정
- [ ] ink_score를 total_score에 포함
- [ ] 가중치: color 50%, ink 30%, confidence 20% (변경)

#### Task 4: Schema 업데이트
- [ ] InkPairData, InkDetailsData 추가
- [ ] ComparisonDetailResponse에 ink_details 추가
- [ ] ScoresData에 ink 필드 추가

#### Task 5: 테스트
- [ ] End-to-end 테스트 (STD 등록 → Test 등록 → 비교)
- [ ] ink_score 계산 검증
- [ ] ink_details 데이터 검증

### 4.2 코드 위치

**수정 파일**:
- `src/services/comparison_service.py` - _compare_inks(), _calculate_scores()
- `src/schemas/comparison_schemas.py` - InkPairData, InkDetailsData
- `src/web/routers/comparison.py` - Response mapping

**확인 필요**:
- `src/services/test_service.py` - ink_analysis 저장 여부
- `src/services/std_service.py` - ink_analysis 저장 여부

---

## 5. 테스트 계획

### 5.1 단위 테스트

#### Test Case 1: 잉크 개수 일치
```python
test_inks = [
    {"weight": 0.65, "lab": [72.2, 137.3, 122.8], ...},
    {"weight": 0.35, "lab": [35.5, 45.2, 39.1], ...}
]
std_inks = [
    {"weight": 0.63, "lab": [72.5, 136.8, 123.1], ...},
    {"weight": 0.37, "lab": [35.8, 44.9, 39.5], ...}
]

result = _compare_inks(test_inks, std_inks)
assert result["ink_count_match"] == True
assert len(result["ink_pairs"]) == 2
assert result["ink_score"] > 80.0
```

#### Test Case 2: 잉크 개수 불일치
```python
test_inks = [{"weight": 1.0, "lab": [72.2, 137.3, 122.8]}]
std_inks = [
    {"weight": 0.65, "lab": [72.5, 136.8, 123.1]},
    {"weight": 0.35, "lab": [35.5, 45.2, 39.1]}
]

result = _compare_inks(test_inks, std_inks)
assert result["ink_count_match"] == False
assert result["ink_score"] == 0.0
```

#### Test Case 3: 색상 차이 큼
```python
test_inks = [
    {"weight": 0.65, "lab": [80.0, 150.0, 130.0], ...}  # 매우 다른 색상
]
std_inks = [
    {"weight": 0.63, "lab": [72.5, 136.8, 123.1], ...}
]

result = _compare_inks(test_inks, std_inks)
assert result["ink_pairs"][0]["delta_e"] > 10.0
assert result["ink_score"] < 50.0
```

### 5.2 통합 테스트

**Workflow**:
1. STD 등록 (SKU001 이미지)
   - InkEstimator로 잉크 분석
   - ink_analysis 저장 확인

2. Test Sample 등록 (SKU001 이미지)
   - InkEstimator로 잉크 분석
   - ink_analysis 저장 확인

3. 비교 실행
   - ink_details 생성 확인
   - ink_score 계산 확인
   - total_score에 ink_score 반영 확인

4. API 응답 검증
   - GET /api/compare/{id}에서 ink_details 반환 확인

---

## 6. 성공 기준

### 6.1 기능 요구사항
- [x] ink_analysis가 TestSample/STDSample에 저장됨
- [ ] _compare_inks() 메서드 구현
- [ ] ink_score 계산 (0-100)
- [ ] ink_details 생성
- [ ] Schema 업데이트
- [ ] API 응답에 ink 데이터 포함

### 6.2 품질 요구사항
- [ ] 단위 테스트 3개 이상
- [ ] 통합 테스트 통과
- [ ] 코드 커버리지 > 80%
- [ ] Flake8, Black 통과

### 6.3 성능 요구사항
- [ ] 잉크 비교 처리 시간 < 100ms (zone 비교 제외)

---

## 7. 다음 단계 (P1-2)

M3 완료 후:
- **P1-2**: Radial Profile 비교 (Pearson 상관계수, 구조 유사도)

---

**작성자**: Claude Sonnet 4.5
**프로젝트**: Contact Lens Color Inspection System
**문서 위치**: `docs/planning/2_comparison/M3_INK_COMPARISON_PLAN.md`
