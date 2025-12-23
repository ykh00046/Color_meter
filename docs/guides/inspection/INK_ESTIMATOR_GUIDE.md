# 📊 InkEstimator 기술 가이드 (v2.0)

**작성일**: 2025-12-14
**대상**: 개발자, 품질 엔지니어, 시스템 통합 담당자
**목적**: GMM 기반 잉크 분석 엔진의 원리와 활용법 상세 설명

---

## 1. 개요

### 1.1 InkEstimator란?

InkEstimator는 콘택트렌즈 이미지에서 **잉크의 개수와 색상을 자동으로 추정**하는 비지도 학습(Unsupervised Learning) 모듈입니다.

**핵심 특징**:
- ✅ **SKU 독립적**: 사전 정의된 Zone 정보 없이도 실제 잉크 개수 파악
- ✅ **GMM + BIC**: 최적 군집 수를 자동 선택 (k=1, 2, 3)
- ✅ **Mixing Correction**: 도트 패턴의 "가짜 중간 톤" 감지 및 제거
- ✅ **고속 처리**: 50K 픽셀 샘플링 + GMM 학습 ~0.5초

**역할**:
- **Main Pipeline (판정)**: Rule-based (SKU 기준값 비교) → 빠르고 일관됨
- **InkEstimator (진단)**: Data-driven (픽셀 분포 분석) → 이상 징후 감지 및 설명

---

## 2. 알고리즘 원리

### 2.1 4단계 추론 파이프라인

```
입력 이미지 (BGR)
    ↓
[Step 1] 지능형 후보 픽셀 선별
    ↓
[Step 2] 반사/하이라이트 제거
    ↓
[Step 3] 적응형 군집화 (GMM + BIC)
    ↓
[Step 4] "중간 톤 = 혼합" 추론
    ↓
출력: {ink_count, inks[], meta{}}
```

### 2.2 Step 1: 지능형 후보 픽셀 선별 (Intelligent Sampling)

**목적**: 배경, 투명부, 노이즈 제거 → 유의미한 잉크 픽셀만 추출

**알고리즘**:
```python
# BGR → CIELAB 변환 (표준 스케일: L 0-100, a,b -128-127)
L = lab_cv[:, :, 0] * (100.0 / 255.0)
a = lab_cv[:, :, 1] - 128.0
b = lab_cv[:, :, 2] - 128.0

# Chroma 계산
chroma = sqrt(a² + b²)

# 필터링 조건
is_colored = (chroma >= chroma_thresh)          # 유채색 잉크
is_dark = (L <= L_dark_thresh)                  # 무채색(Black) 잉크
is_not_highlight = (L <= L_max)                 # 하이라이트 제거

# 최종 마스크
mask = (is_colored OR is_dark) AND is_not_highlight
```

**파라미터**:
| 파라미터 | 기본값 | 의미 | 조정 시나리오 |
|---------|-------|------|-------------|
| `chroma_thresh` | 6.0 | 유채색 판단 기준 | 낮춤: 연한 색 포함, 높임: 선명한 색만 |
| `L_dark_thresh` | 45.0 | Black 잉크 보존 | 높임: 더 밝은 회색도 포함 |
| `L_max` | 98.0 | 하이라이트 제거 | 낮춤: 엄격한 반사 제거 |

**샘플링 전략**:
```python
# 고해상도에서도 대표성 확보
target_samples = min(n_pixels, max(5000, min(50000, n_pixels * 0.05)))
```
- 전체의 5% 또는 50K 중 작은 값 (최소 5K 확보 시도)
- 랜덤 샘플링 (Seed 고정으로 재현성 확보)

### 2.3 Step 2: 반사/하이라이트 제거 (Specular Rejection)

**목적**: 잉크 색상과 유사하지만 빛 반사로 밝게 뜬 픽셀 제거

**조건**:
```python
if (L >= 95.0) AND (chroma <= 5.0):
    exclude  # 반사광으로 판단
```

**효과**: 렌즈 표면 반사가 독립 군집으로 검출되는 것 방지

### 2.4 Step 3: 적응형 군집화 (Adaptive Clustering)

**3.1 GMM (Gaussian Mixture Model) 학습**

```python
for k in [1, 2, 3]:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",      # 타원형 군집 지원
        random_state=42,
        reg_covar=1e-4,               # 안정성 향상
        n_init=3                      # Local minima 방지
    )
    gmm.fit(samples)  # shape: (N, 3) - L, a, b
    bic = gmm.bic(samples)

    if bic < best_bic:
        best_bic = bic
        best_gmm = gmm
```

**3.2 BIC (Bayesian Information Criterion) 최소화**

$$
\text{BIC} = -2 \ln(\mathcal{L}) + k \ln(n)
$$

- $\mathcal{L}$: 모델 가능도 (Likelihood)
- $k$: 파라미터 개수 (군집 수 증가 → 패널티)
- $n$: 샘플 수

**선택 로직**:
- k=1: 단일 잉크 (단색 렌즈)
- k=2: 2가지 잉크 (2도 렌즈)
- k=3: 3가지 잉크 또는 2도 + 혼합 톤

**3.3 KMeans Fallback**

GMM 실패 시 KMeans로 대체:
```python
if best_gmm is None:
    kmeans = KMeans(n_clusters=k_max, random_state=42, n_init=10)
    # FakeGMM wrapper로 인터페이스 통일
```

### 2.5 Step 4: "중간 톤 = 혼합" 추론 (Linearity Check) ⭐핵심

**목적**: 물리적으로 섞여서 생긴 색을 독립 잉크로 오판하는 것 방지

**조건**: k=3일 때만 수행

**알고리즘**:
```python
# 1. L값 기준 정렬
order = argsort(centers[:, 0])  # Dark, Mid, Bright

# 2. 벡터 정의
vec_DB = C_bright - C_dark
u_DB = vec_DB / ||vec_DB||

# 3. 투영 거리 계산
vec_DM = C_mid - C_dark
projection_len = dot(vec_DM, u_DB)
closest_point = C_dark + u_DB * projection_len
distance = ||C_mid - closest_point||

# 4. 선형성 판단
if distance < linearity_thresh:
    # Mid를 혼합으로 간주 → 2개로 병합
    ratio = clip(projection_len / ||vec_DB||, 0.0, 1.0)

    new_weights[0] = w_dark + w_mid * (1 - ratio)
    new_weights[1] = w_bright + w_mid * ratio

    return [C_dark, C_bright], new_weights, True
```

**기하학적 의미**:
```
Dark ●────────────● Mid ────────────● Bright
      ↑           ↑                  ↑
    Ink 1      Mixing             Ink 2

Mid가 직선 위에 있으면 (distance < 3.0):
  → Ink 1과 Ink 2의 도트 혼합으로 판단
  → 독립 잉크가 아님!
```

**가중치 재분배**:
- ratio=0.3 → Mid가 Dark에 가까움 → w_dark += w_mid * 0.7
- ratio=0.7 → Mid가 Bright에 가까움 → w_bright += w_mid * 0.7

---

## 3. 파라미터 가이드

### 3.1 전체 파라미터 테이블

| 파라미터 | 기본값 | 범위 | 설명 | 조정 시나리오 |
|---------|-------|------|------|-------------|
| `chroma_thresh` | 6.0 | 3.0~15.0 | 유채색 잉크 판단 | 연한 파스텔: 낮춤, 선명한 원색: 높임 |
| `L_dark_thresh` | 45.0 | 30.0~60.0 | Black 잉크 보존 | 진한 검정만: 낮춤, 회색 포함: 높임 |
| `L_max` | 98.0 | 90.0~100.0 | 하이라이트 제거 | 반사 심함: 낮춤 (95.0) |
| `merge_de_thresh` | 5.0 | 3.0~10.0 | 유사 색상 병합 | 엄격한 분리: 낮춤 |
| `linearity_thresh` | 3.0 | 1.0~5.0 | 혼합 판단 거리 | 엄격한 선형성: 낮춤 |
| `max_samples` | 50000 | 10K~100K | 최대 샘플 수 | 고해상도: 높임 |
| `random_seed` | 42 | - | 재현성 확보 | 변경 금지 권장 |

### 3.2 시나리오별 튜닝 예시

**시나리오 A: 도트 패턴 2도 렌즈 (갈색 계열)**
```python
estimator.estimate_from_array(
    bgr=img,
    chroma_thresh=6.0,          # 기본값
    linearity_thresh=3.0,       # 기본값 (혼합 감지 활성화)
    merge_de_thresh=5.0         # 기본값
)
# 예상 결과: k=3 → Mixing Correction → 2개
```

**시나리오 B: 검은색 써클라인 + 투명**
```python
estimator.estimate_from_array(
    bgr=img,
    chroma_thresh=6.0,
    L_dark_thresh=45.0,         # Black 보존
    L_max=98.0
)
# 예상 결과: k=1 (Black만 검출)
```

**시나리오 C: 3도 실제 독립 잉크**
```python
estimator.estimate_from_array(
    bgr=img,
    linearity_thresh=3.0        # 기본값
)
# 예상 결과: k=3, Mixing Correction=False (3개 유지)
```

---

## 4. 출력 구조

### 4.1 JSON Schema

```json
{
  "ink_count": 2,
  "detection_method": "gmm_bic",
  "inks": [
    {
      "weight": 0.45,
      "lab": [35.2, 15.8, -8.3],
      "rgb": [120, 80, 95],
      "hex": "#78505F",
      "is_mix": false
    },
    {
      "weight": 0.55,
      "lab": [68.5, 12.3, -25.6],
      "rgb": [180, 165, 210],
      "hex": "#B4A5D2",
      "is_mix": false
    }
  ],
  "meta": {
    "correction_applied": true,
    "original_cluster_count": 3,
    "sample_count": 15234,
    "bic": -45231.2,
    "mean_L": 52.3,
    "algorithm": "GMM+BIC"
  }
}
```

### 4.2 필드 설명

**inks[] 배열**:
| 필드 | 타입 | 설명 |
|-----|------|------|
| `weight` | float | 픽셀 비율 (0.0~1.0, 합=1.0) |
| `lab` | [L, a, b] | CIELAB 색공간 값 |
| `rgb` | [R, G, B] | sRGB 변환 값 (0~255) |
| `hex` | string | HEX 색상 코드 (#RRGGBB) |
| `is_mix` | bool | 혼합색 여부 (현재 미사용) |

**meta 객체**:
| 필드 | 설명 |
|-----|------|
| `correction_applied` | Mixing Correction 적용 여부 |
| `original_cluster_count` | 보정 전 군집 수 (보정 시에만) |
| `sample_count` | 분석에 사용된 픽셀 수 |
| `bic` | BIC 점수 (낮을수록 좋음) |
| `mean_L` | 전체 이미지 평균 명도 |

---

## 5. 사용법

### 5.1 Python API

#### 기본 사용
```python
from src.core.ink_estimator import InkEstimator

estimator = InkEstimator(random_seed=42)

# 이미지 로드
import cv2
img_bgr = cv2.imread("lens_image.jpg")

# 분석 실행
result = estimator.estimate_from_array(img_bgr)

# 결과 확인
print(f"Ink Count: {result['ink_count']}")
for i, ink in enumerate(result['inks']):
    print(f"Ink {i+1}: Lab={ink['lab']}, Hex={ink['hex']}, Weight={ink['weight']:.2f}")

if result['meta']['correction_applied']:
    print("⚠️ Mixing Correction Applied (3→2)")
```

#### 파라미터 튜닝
```python
result = estimator.estimate_from_array(
    bgr=img_bgr,
    k_max=3,                    # 최대 군집 수
    chroma_thresh=8.0,          # 엄격한 유채색 필터
    L_max=95.0,                 # 엄격한 하이라이트 제거
    merge_de_thresh=4.0,        # 유사 색상 병합
    linearity_thresh=2.5        # 엄격한 혼합 판단
)
```

### 5.2 Pipeline 통합

InkEstimator는 `zone_analyzer_2d.py`에 통합되어 자동 실행됩니다:

```python
# src/core/zone_analyzer_2d.py:1422-1460

def analyze_lens_zones_2d(...) -> ZoneAnalysisResult:
    # ... 기존 Zone 분석 ...

    # InkEstimator 실행 (자동)
    ink_estimator = InkEstimator(random_seed=42)
    image_based_result = ink_estimator.estimate_from_array(img_bgr)

    # 결과 구조화
    ink_analysis = {
        "zone_based": {...},      # Zone 기반 결과
        "image_based": image_based_result  # InkEstimator 결과
    }

    return result
```

**접근 방법**:
```python
result = analyze_lens_zones_2d(...)
zone_count = result.ink_analysis["zone_based"]["detected_ink_count"]
gmm_count = result.ink_analysis["image_based"]["ink_count"]

if zone_count != gmm_count:
    print("⚠️ 불일치 감지! SKU 설정 검토 필요")
```

---

## 6. 품질 보증 (QA)

### 6.1 테스트 커버리지

`tests/test_ink_estimator.py`에 9개 통과 테스트:
- ✅ Sampling (3개): 픽셀 선별, Chroma 필터링, Black 보존
- ✅ Clustering (2개): 단일/다중 군집 GMM + BIC
- ✅ Mixing Correction (2개): 적용/미적용 시나리오
- ✅ Edge Cases (2개): 빈 이미지, Trimmed Mean

**실행**:
```bash
pytest tests/test_ink_estimator.py -v
# 9 passed, 3 skipped in 11.03s
```

### 6.2 검증 데이터셋

**Case A (2도 Dot)**: 갈색 도트 렌즈
- 기존 알고리즘: 3개 오판 (Dark, Mid, Light)
- InkEstimator: 3→2 보정 ✅

**Case B (Black Circle)**: 검은색 써클 렌즈
- 기존 알고리즘: 0개 (Chroma 낮아서 누락)
- InkEstimator: 1개 (Black) 검출 ✅

**Case C (3도 Real)**: 실제 3가지 잉크
- InkEstimator: 3개 유지 (Mixing Correction 미적용) ✅

---

## 7. 문제 해결 (Troubleshooting)

### Q1. "sample_count가 너무 적습니다" (< 1000)

**원인**: 유효 픽셀 부족 (배경이 대부분, 또는 과도한 필터링)

**해결**:
```python
# chroma_thresh 낮춤 (더 많은 픽셀 포함)
result = estimator.estimate_from_array(img, chroma_thresh=4.0)

# L_dark_thresh 높임 (더 밝은 회색 포함)
result = estimator.estimate_from_array(img, L_dark_thresh=55.0)
```

### Q2. "GMM이 항상 k=3을 선택합니다"

**원인**: 노이즈나 도트 패턴으로 인한 과잉 군집화

**해결**:
1. Mixing Correction 확인 (보정 후 2개로 줄어드는지)
2. k_max 조정:
   ```python
   result = estimator.estimate_from_array(img, k_max=2)  # 최대 2개로 제한
   ```

### Q3. "Mixing Correction이 작동하지 않습니다"

**원인**: linearity_thresh가 너무 엄격

**해결**:
```python
# 임계값 완화 (3.0 → 5.0)
result = estimator.estimate_from_array(img, linearity_thresh=5.0)
```

**디버깅**:
```python
# 중간 톤 거리 확인
if result['meta']['correction_applied'] == False:
    # 수동으로 거리 계산하여 임계값 결정
    pass
```

### Q4. "mean_L이 0.0입니다 (어두운 이미지)"

**경고**: 입력 이미지가 노출 부족 (Underexposed)

**해결**:
- 이미지 전처리: 히스토그램 평활화
- 촬영 조건 개선: 조명 강화

---

## 8. 성능 최적화

### 8.1 실행 시간 분석

| 단계 | 시간 | 비율 |
|-----|------|------|
| 픽셀 샘플링 | ~0.1s | 20% |
| GMM 학습 (k=1,2,3) | ~0.3s | 60% |
| Mixing Correction | <0.01s | 2% |
| Lab→RGB 변환 | ~0.05s | 10% |
| **Total** | **~0.5s** | 100% |

### 8.2 최적화 팁

**1. 이미지 다운스케일링**:
```python
# InkEstimator 내부에서 자동 수행
# max(h, w) > 1200이면 자동 다운샘플링
```

**2. 샘플 수 조정**:
```python
# 고해상도: 샘플 수 증가
result = estimator.estimate_from_array(img, max_samples=100000)

# 저해상도/빠른 처리: 샘플 수 감소
result = estimator.estimate_from_array(img, max_samples=10000)
```

**3. GMM n_init 조정**:
```python
# 빠른 처리 (정확도 약간 낮음)
gmm = GaussianMixture(n_init=1)  # 기본값: 3

# 정확도 우선 (느림)
gmm = GaussianMixture(n_init=10)
```

---

## 9. 향후 개발 방향

### 9.1 Phase 3 계획 (예정)

**SKU 관리 기능 연동**:
- Web UI에 "Auto-Detect Ink Config" 버튼 추가
- InkEstimator 결과로 SKU 기준값 자동 생성

```python
# tools/generate_sku_baseline.py 개선
def auto_detect_sku_config(golden_sample_images):
    estimator = InkEstimator()

    ink_counts = []
    for img in golden_sample_images:
        result = estimator.estimate_from_array(img)
        ink_counts.append(result['ink_count'])

    # 다수결로 expected_zones 결정
    expected_zones = mode(ink_counts)

    # 각 잉크의 Lab 평균값을 Zone 기준값으로 사용
    ...
```

### 9.2 알고리즘 개선 아이디어

**1. 4+ 잉크 지원**:
```python
# 현재: k_max=3
# 향후: k_max=5 (고급 멀티컬러 렌즈)
```

**2. Adaptive linearity_thresh**:
```python
# 데이터 분포에 따라 임계값 자동 조정
linearity_thresh = auto_tune_threshold(samples, centers)
```

**3. Sector-wise 분석**:
```python
# 각 섹터(상/하/좌/우)별로 독립 분석 후 통합
# 비균일 렌즈(그라데이션)에 효과적
```

---

## 10. 참고 문서

- **알고리즘 설계**: `docs/planning/INK_ANALYSIS_ENHANCEMENT_PLAN.md`
- **통합 완료 보고**: `docs/planning/INK_ANALYSIS_ENHANCEMENT_PLAN.md` (Phase 2)
- **테스트 보고서**: `docs/planning/TEST_INK_ESTIMATOR_COMPLETION.md`
- **사용자 가이드**: `docs/guides/USER_GUIDE.md` (Section 6)
- **API 레퍼런스**: `src/core/ink_estimator.py` (Docstrings)

---

## 11. 라이선스 및 기여

이 모듈은 프로젝트 전체 라이선스를 따릅니다.
버그 리포트 및 개선 제안은 GitHub Issues에 등록해 주세요.

**개발 이력**:
- v1.0 (2025-12-13): 초기 구현 (GMM + BIC)
- v2.0 (2025-12-14): Pipeline 통합 + Mixing Correction
