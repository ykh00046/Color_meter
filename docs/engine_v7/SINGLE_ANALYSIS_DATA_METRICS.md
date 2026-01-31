# 단독 샘플 분석 - 데이터 수치화 방식 상세 분석

**작성일**: 2026-01-09
**버전**: v1.0

---

## 📊 개요

단독 샘플 분석은 STD 모델 없이 샘플 자체의 품질을 평가하는 방식입니다. 6개 분석 모드를 통해 다양한 측면을 수치화하고, 이를 종합하여 0-100점의 품질 점수를 산출합니다.

---

## 🔍 분석 모드별 수치화 방식

### 1. Gate Check (기하학적 검사)

**목적**: 렌즈 기하학적 품질 평가 (중심 편차, 선명도, 조명 균일성)

**수치화 항목**:
```python
{
    "passed": True/False,  # 최종 합격/불합격
    "geometry": {
        "cx": 512.34,        # 중심 X 좌표 (px)
        "cy": 512.89,        # 중심 Y 좌표 (px)
        "r": 480.12          # 반지름 (px)
    },
    "scores": {
        "center_offset_mm": 0.08,     # 중심 편차 (mm)
        "sharpness_score": 45.2,      # 선명도 점수
        "illumination_asymmetry": 0.05 # 조명 비대칭도
    }
}
```

**평가 기준** (configs/default.json):
- `center_off_max`: 0.12 mm (초과 시 실패)
- `blur_min`: 40.0 (미만 시 실패)
- `illum_max`: 0.1 (초과 시 실패)

**Quality Score 기여도**: **30%**
- 합격: 100점
- 불합격: 0점

**장점**:
- ✅ 명확한 Pass/Fail 기준
- ✅ 물리적 의미가 분명함 (mm 단위)
- ✅ 조명 문제 조기 감지

**단점**:
- ❌ 이진 판정 (중간 상태 없음)
- ❌ 임계값에 민감 (0.11mm vs 0.13mm의 차이가 100점 vs 0점)

---

### 2. Color Distribution (색상 분포 분석)

**목적**: Lab 색공간에서 색상 분포의 일관성 평가

**수치화 항목**:
```python
{
    "L": {
        "mean": 45.3,      # L* 평균
        "std": 8.2,        # L* 표준편차
        "min": 30.1,       # L* 최소값
        "max": 62.5,       # L* 최대값
        "p05": 35.2,       # 5th percentile
        "p95": 58.7        # 95th percentile
    },
    "a": {...},            # a* 동일 구조
    "b": {...},            # b* 동일 구조
    "histogram_L": [array of 50 bins],  # L* 히스토그램
    "histogram_a": [...],
    "histogram_b": [...]
}
```

**평가 기준**:
```python
# Color score 계산 (L* 표준편차 기반)
L_std = color_data["L"]["std"]
color_score = max(0.0, min(100.0, 100.0 - (L_std - 5) * 5))

# 예시:
# L_std = 5  → color_score = 100
# L_std = 10 → color_score = 75
# L_std = 15 → color_score = 50
# L_std = 25 → color_score = 0
```

**Quality Score 기여도**: **20%**

**장점**:
- ✅ 연속적인 점수 (fine-grained)
- ✅ 색상 일관성을 직관적으로 표현
- ✅ 히스토그램으로 분포 시각화

**단점**:
- ❌ L* 채널에만 의존 (a*, b* 무시)
- ❌ 다색상 샘플의 경우 높은 std가 정상일 수 있음
- ❌ "좋은" std 값 (5-10)이 하드코딩됨

**보완 방안**:
- a*, b* 채널도 가중 평균으로 포함
- Multi-modal 분포 감지 (클러스터별 std 계산)
- SKU별 기준값 설정 가능하도록 개선

---

### 3. Radial Profile (방사형 균일성)

**목적**: 중심에서 외곽으로의 색상 변화 균일성 평가

**수치화 항목**:
```python
{
    "profile": {
        "L_mean": [array of R values],  # 각 반지름에서 L* 평균
        "a_mean": [...],
        "b_mean": [...],
        "L_std": [...],                  # 각 반지름에서 L* 표준편차
        "a_std": [...],
        "b_std": [...]
    },
    "summary": {
        "inner_mean_L": 48.2,   # 내부 영역 평균 L*
        "outer_mean_L": 42.5,   # 외부 영역 평균 L*
        "uniformity": 0.87      # 균일성 점수 (0-1)
    }
}
```

**Uniformity 계산**:
```python
# Coefficient of Variation (CV) 기반
L_profile = radial_mean[:, 0]  # L* 프로파일
mean_L = np.mean(L_profile)
std_L = np.std(L_profile)
cv = std_L / mean_L

# 0-1 점수로 변환 (CV가 낮을수록 균일)
uniformity = max(0.0, 1.0 - (cv / 0.3))

# 예시:
# cv = 0.05 → uniformity = 0.83
# cv = 0.15 → uniformity = 0.50
# cv = 0.30+ → uniformity = 0.00
```

**Quality Score 기여도**: **0% (soft metric, stored only)**

**장점**:
- ✅ 렌즈 특성상 중요한 중심-외곽 균일성 평가
- ✅ Coefficient of Variation은 통계적으로 타당
- ✅ 프로파일 시각화로 패턴 파악 가능

**단점**:
- ❌ CV 기준값 0.3이 하드코딩됨
- ❌ 각도별 차이 무시 (방사형만 봄)
- ❌ 그라데이션이 정상인 렌즈도 저점수 가능

**보완 방안**:
- 선형 그라데이션 패턴 감지 (정상으로 처리)
- SKU별 expected_profile 학습 가능
- 각도별 프로파일 분산도 고려

---

### 4. Ink Segmentation (잉크 색상 클러스터링) ⚠️ 문제 있음

**목적**: K-means 클러스터링으로 잉크 색상 분리 및 분석

**수치화 항목**:
```python
{
    "k": 3,                # 클러스터 개수
    "clusters": [
        {
            "id": 0,
            "centroid_lab": [40.3, 131.9, 134.0],  # Lab 중심값
            "pixel_count": 12450,
            "area_ratio": 0.35,                     # 전체 대비 비율
            "mean_hex": "#2E241F"                   # 근사 RGB 색상
        },
        {...},
        {...}
    ],
    "confidence": 0.7      # 클러스터링 신뢰도 (placeholder)
}
```

**클러스터링 방식**:
```python
# 1. Polar 좌표계 변환
polar_lab = to_polar(test_bgr, geom, R=260, T=720)

# 2. ROI 마스크 적용 (r_start=0.15 ~ r_end=0.95)
lab_samples = polar_lab[roi_mask > 0]  # Shape: (N, 3)

# 3. Feature 변환 (a, b, L*0.3)
features = [a, b, L*0.3]

# 4. K-means 클러스터링
k = cfg.get("expected_ink_count", 3)  # ⚠️ 하드코딩 기본값 3
labels, centers = kmeans_segment(lab_samples, k=k, l_weight=0.3)

# 5. 클러스터별 픽셀 수 집계
for i in range(k):
    cluster_mask = (labels == i)
    count = np.sum(cluster_mask)
    ...

# 6. L* 값 기준 정렬 (어두운 색 → 밝은 색)
clusters.sort(key=lambda x: x["centroid_lab"][0])
```

**Quality Score 기여도**: **0% (현재 미사용)**

---

### ⚠️ **K 값 결정 방식의 문제점**

#### 비교/등록 모드 (정상)
```bash
# 사용자가 명시적으로 지정
python scripts/register_std.py --sku GGG --ink INK_RGB \
  --expected_ink_count 3 \
  --stds low1.png low2.png

# 각 SKU마다 다른 k 사용 가능
# - 단색 렌즈: k=1
# - RGB 렌즈: k=3
# - CMYK 렌즈: k=4
```

#### 단독 분석 모드 (문제) ⚠️
```python
# single_analyzer.py:644
expected_k = cfg.get("expected_ink_count", 3)  # 기본값 3

# configs/default.json에 "expected_ink_count" 필드 없음!
# → 모든 샘플에 대해 k=3으로 고정

# 문제 시나리오:
# - 단색 렌즈 (실제 k=1) → 강제로 3개 클러스터 생성 → 과분할
# - CMYK 렌즈 (실제 k=4) → 3개로 강제 병합 → 정보 손실
# - 결과: 의미 없는 클러스터링
```

**장점**:
- ✅ 잉크별 색상 분리 가능 (k가 올바른 경우)
- ✅ L* 정렬로 안정적인 ID 부여
- ✅ Area ratio로 커버리지 파악

**단점**:
- ❌ **k 값이 하드코딩됨 (가장 심각한 문제)**
- ❌ Confidence score가 placeholder (0.7 고정)
- ❌ Quality score에 반영 안됨 (ink 분석이 무의미)
- ❌ Lab → RGB 변환이 근사치 (시각화만 가능)

**보완 방안 (필수)**:
1. **Auto k detection 추가**
   ```python
   # BIC (Bayesian Information Criterion) 사용
   def auto_detect_k(lab_samples, k_max=5):
       bic_scores = []
       for k in range(1, k_max+1):
           labels, centers = kmeans_segment(lab_samples, k)
           bic = calculate_bic(lab_samples, labels, centers, k)
           bic_scores.append(bic)
       return np.argmin(bic_scores) + 1
   ```

2. **Silhouette score로 신뢰도 계산**
   ```python
   from sklearn.metrics import silhouette_score
   confidence = silhouette_score(lab_samples, labels)
   ```

3. **UI에서 k 값 입력 받기**
   ```html
   <label>Expected Ink Count:</label>
   <input type="number" id="inkCount" value="3" min="1" max="5">
   ```

4. **configs/default.json에 추가**
   ```json
   {
     "expected_ink_count": 3,
     "ink_auto_detect": true,
     "ink_k_range": [1, 5]
   }
   ```

---

### 5. Pattern Quality (패턴 품질)

**목적**: 표면 패턴 특징 추출 (각도 균일성, 중심 결함, 대비, 엣지)

**수치화 항목**:
```python
{
    "angular_uniformity": 0.15,  # 각도별 균일성 (0-1, 낮을수록 좋음)
    "center_blobs": {
        "blob_count": 0,         # 중심부 blob 개수
        "max_area": 0            # 최대 blob 면적
    },
    "contrast": 25.3,            # 대비
    "edge_density": 0.12         # 엣지 밀도
}
```

**Angular Uniformity 계산**:
```python
# 각 각도(theta)에서 표준편차 계산
std_per_theta = np.std(polar_lab, axis=0)  # axis=0: radius 방향

# 표준편차의 평균 (각도 간 변동)
angular_unif = np.mean(std_per_theta)

# 정규화 (0-1 범위)
# 좋은 샘플: 0.05-0.15
# 나쁜 샘플: 0.3+
```

**Quality Score 기여도**: **15%**
```python
pattern_score = (1.0 - angular_uniformity) * 100

# 예시:
# angular_unif = 0.10 → pattern_score = 90
# angular_unif = 0.20 → pattern_score = 80
# angular_unif = 0.50 → pattern_score = 50
```

**장점**:
- ✅ 각도별 일관성 평가 (radial과 보완)
- ✅ Blob detection으로 결함 감지
- ✅ 다양한 feature 조합

**단점**:
- ❌ Angular uniformity 해석이 직관적이지 않음
- ❌ Contrast, edge_density가 점수에 미반영
- ❌ "좋은" 기준값이 경험적

---

### 6. Zone Analysis (영역별 분석)

**목적**: 8개 각도 섹터별 색상 균일성 평가

**수치화 항목**:
```python
{
    "zones": [
        {
            "zone_id": 0,
            "angle_range": [0, 45],       # 각도 범위
            "mean_lab": [45.2, 12.3, -5.1],  # 평균 Lab
            "std_lab": [3.2, 1.1, 0.8]       # 표준편차 Lab
        },
        # ... 총 8개 zone
    ],
    "zone_uniformity": 0.92  # 전체 균일성 (0-1)
}
```

**Zone Uniformity 계산**:
```python
# 각 zone의 L* 값 추출
L_values = [zone["mean_lab"][0] for zone in zones]

# Coefficient of Variation
mean_L = np.mean(L_values)
std_L = np.std(L_values)
cv = std_L / mean_L

# 균일성 점수
zone_uniformity = max(0.0, 1.0 - (cv / 0.2))

# 예시:
# cv = 0.05 → uniformity = 0.75
# cv = 0.10 → uniformity = 0.50
# cv = 0.20+ → uniformity = 0.00
```

**Quality Score 기여도**: **15%**

**장점**:
- ✅ 각도별 색상 변화 정량화
- ✅ Canvas 시각화로 직관적
- ✅ 국소 불량 감지 가능

**단점**:
- ❌ 8개 고정 (설정 가능하지만 UI 미지원)
- ❌ Radial profile과 중복되는 정보
- ❌ CV 기준값 0.2가 하드코딩

---

## 🎯 Quality Score 종합 계산

### 공식
```python
quality_score = (
    gate_score    * 0.30 +  # 30%
    color_score   * 0.20 +  # 20%
    pattern_score * 0.20 +  # 20%
    zone_score    * 0.15    # 15%
) / 0.85  # normalized to 100; soft metrics excluded
```

Note: radial metrics are soft metrics (stored only; excluded from scoring).

### 가중치 설계 철학
1. **Gate (30%)** - 가장 중요 (불합격 시 분석 무의미)
2. **Color (20%)** - 색상 일관성의 두 축
3. **Pattern + Zone (35%)** - 보조 지표

### 점수 분포
- **80-100**: 우수 (녹색)
- **60-79**: 보통 (노란색)
- **0-59**: 불량 (빨간색)

---

## ⚠️ 현재 시스템의 주요 문제점

### 1. K 값 하드코딩 (심각)
```python
# ❌ 현재: 모든 샘플에 k=3 강제
expected_k = cfg.get("expected_ink_count", 3)

# ✅ 필요: UI 입력 or Auto detection
expected_k = user_input or auto_detect_k(lab_samples)
```

### 2. 하드코딩된 임계값
```python
# 좋은 L_std 기준
color_score = 100.0 - (L_std - 5) * 5  # 5가 하드코딩

# Radial CV 기준
uniformity = 1.0 - (cv / 0.3)  # 0.3이 하드코딩

# Zone CV 기준
zone_uniformity = 1.0 - (cv / 0.2)  # 0.2가 하드코딩
```

**해결**: SKU별 baseline 학습 기능 추가

### 3. 단일 채널 의존
```python
# ❌ 현재: L* 채널만 사용
color_score = f(L_std)

# ✅ 개선: a*, b* 포함
color_score = f(L_std, a_std, b_std)
```

### 4. Ink 분석 미활용
- 클러스터링 결과가 Quality score에 반영 안됨
- Confidence score가 placeholder

---

## 💡 보완 방안 요약

### 단기 (High Priority)
1. **K 값 입력 UI 추가** ⭐⭐⭐
   ```html
   <div class="form-group">
       <label>Expected Ink Count (k):</label>
       <input type="number" id="expectedInkCount" value="3" min="1" max="5">
       <span class="help-text">Enter number of ink colors (1-5)</span>
   </div>
   ```

2. **configs/default.json에 추가**
   ```json
   {
     "expected_ink_count": 3,
     "ink_analysis": {
       "auto_detect_k": false,
       "k_range": [1, 5]
     }
   }
   ```

3. **Lab 전체 채널 활용**
   ```python
   color_score = (
       f(L_std) * 0.5 +
       f(a_std) * 0.25 +
       f(b_std) * 0.25
   ) / 0.85  # normalized to 100; soft metrics excluded
   ```

### 중기 (Medium Priority)
4. **Auto k detection**
   - BIC, Silhouette score 기반
   - k=1~5 범위 자동 탐색

5. **Ink 분석 품질 점수 반영**
   ```python
   quality_score = (
       gate_score   * 0.25 +
       color_score  * 0.15 +
       ink_score    * 0.15 +  # 신규
       pattern_score * 0.15 +
       zone_score   * 0.15
   )
   ```

6. **SKU별 기준값 학습**
   - N개 양품 샘플로 baseline 생성
   - expected_L_std, expected_cv 등 저장

### 장기 (Future Enhancement)
7. **ML 기반 이상 탐지**
   - Autoencoder로 reconstruction error
   - 정상 패턴 학습 후 anomaly detection

8. **다중 모드 분포 감지**
   - 다색상 렌즈의 경우 클러스터별 분석

9. **시간별 트렌드 분석**
   - 동일 SKU의 과거 데이터와 비교

---

## 📊 장단점 종합

### 장점 ✅
1. **다면적 평가**: 6개 독립 지표로 종합 품질 평가
2. **STD 불필요**: 연구 초기 단계에서 즉시 활용 가능
3. **시각화**: 히스토그램, 프로파일, Zone map 제공
4. **자동화**: 입력만 하면 즉시 결과 생성

### 단점 ❌
1. **K 값 문제**: 가장 심각, 즉시 해결 필요
2. **하드코딩**: 임계값이 경험 기반, SKU 독립적이지 않음
3. **단일 채널**: L*에 과도하게 의존
4. **미활용 데이터**: Ink 분석 결과가 점수에 미반영
5. **절대 평가**: 상대 비교 없이 절대 기준만 사용

### 비교 분석 vs 단독 분석

| 항목 | 비교 분석 (STD vs Test) | 단독 분석 |
|------|------------------------|----------|
| **판정 기준** | 상대적 (STD 대비) | 절대적 (임계값) |
| **K 값** | 사용자 명시 (SKU별) | 고정 (3) ⚠️ |
| **정확도** | 높음 (STD 기준 명확) | 중간 (baseline 없음) |
| **활용 시기** | 양산 단계 | 연구/개발 단계 |
| **False Positive** | 낮음 | 높음 (임계값 의존) |

---

## 🎯 우선순위 액션 아이템

### 즉시 적용 (이번 주)
- [ ] UI에 "Expected Ink Count" 입력 필드 추가
- [ ] configs/default.json에 expected_ink_count: 3 추가
- [ ] Lab a*, b* 채널도 color_score에 반영

### 다음 단계 (이번 달)
- [ ] Auto k detection (BIC 기반) 구현
- [ ] Silhouette score로 confidence 계산
- [ ] Ink 분석을 Quality score에 반영 (가중치 15%)

### 장기 계획 (분기별)
- [ ] SKU별 baseline 학습 기능
- [ ] ML 기반 anomaly detection
- [ ] 시간별 트렌드 분석 대시보드

---

**작성자**: Claude
**검토 필요**: K 값 결정 방식, 임계값 설정
**다음 업데이트**: Auto k detection 구현 후
