# AI Telemetry & 데이터 분석 가이드

**목적**: JSON 텔레메트리를 활용한 AI 기반 품질 분석 고도화

---

## 📊 현재 JSON 정보 커버리지

### ✅ 현재 포함된 정보 (약 70%)

```json
{
  // 1. 기본 판정 정보
  "sku": "SKU001",
  "timestamp": "2025-12-12T21:11:32",
  "judgment": "NG",
  "overall_delta_e": 17.90,  // ✅ 수정 후 정확한 값
  "confidence": 0.0,
  "ng_reasons": ["Zone A: ΔE=17.90 > 4.00"],

  // 2. Zone별 상세 평가 (3개 Zone)
  "zone_results": [
    {
      "zone_name": "A",
      "measured_lab": [71.6, -0.5, 9.5],  // ✅ 표준 Lab
      "target_lab": [72.2, 9.3, -5.2],
      "delta_e": 17.90,
      "threshold": 4.0,
      "is_ok": false
    }
  ],

  // 3. Zone 원본 통계
  "zones": [
    {
      "name": "A",
      "r_start": 1.0,
      "r_end": 0.667,
      "mean_L": 71.6,
      "std_L": 2.99,  // 표준편차도 포함
      "zone_type": "pure"
    }
  ],

  // 4. 렌즈 검출 정보
  "lens_detection": {
    "center_x": 454.2,
    "center_y": 415.8,
    "radius": 348.1,
    "confidence": 0.9,
    "method": "hybrid",
    "roi": [71, 32, 765, 765]
  },

  // 5. Ring×Sector 2D 분석 (36개 셀)
  "ring_sector_cells": [
    {
      "ring_index": 0,  // 0=중심, 1=중간, 2=외곽
      "sector_index": 0,  // 0~11 (30도씩)
      "r_start": 0.15,
      "r_end": 0.33,
      "angle_start": 0.0,
      "angle_end": 30.0,
      "mean_L": 99.6,  // ✅ 표준 Lab
      "std_L": 0.15,
      "pixel_count": 2721
    }
    // ... 35개 셀 더
  ],

  // 6. 균일성 분석
  "uniformity_analysis": {
    "is_uniform": false,
    "global_mean_lab": [59.0, 4.1, 23.5],
    "max_delta_e": 31.9,
    "mean_delta_e": 18.9,
    "outlier_cells": [],
    "ring_uniformity": {
      "0": {"mean_de": 31.8, "is_uniform": false},
      "1": {"mean_de": 8.9, "is_uniform": false},
      "2": {"mean_de": 15.9, "is_uniform": false}
    },
    "sector_uniformity": { ... }
  }
}
```

---

### ❌ 현재 누락된 정보 (약 30%)

| 정보 | AI 활용도 | 추가 방법 | 예상 크기 |
|------|----------|----------|----------|
| **Radial Profile 원본** | ⭐⭐⭐⭐⭐ | `radial_profile` 필드 추가 | ~5KB |
| **미분/피크 정보** | ⭐⭐⭐⭐ | `derivative`, `peaks` 필드 | ~3KB |
| **Boundary Detection 상세** | ⭐⭐⭐⭐ | `boundary_detection` 필드 | ~1KB |
| **Background Mask 통계** | ⭐⭐⭐ | `background_mask` 필드 | ~500B |
| **단계별 처리 시간** | ⭐⭐⭐ | `processing_times` 필드 | ~500B |
| **설정값 스냅샷** | ⭐⭐⭐ | `config_snapshot` 필드 | ~2KB |
| **이미지 Base64** | ⭐⭐ | `images.original.data` 필드 | ~200KB |

**총 추가 크기:** ~200-210KB (이미지 포함 시), ~10KB (이미지 제외 시)

---

## 🎯 완전한 텔레메트리 JSON 구조

### Enhanced JSON 스키마 (모든 정보 포함)

```json
{
  "version": "1.0.0",
  "timestamp": "2025-12-12T21:11:32.799083",

  "metadata": {
    "image_filename": "sample_001.jpg",
    "image_width": 800,
    "image_height": 800,
    "image_format": "JPEG",
    "sku_code": "SKU001",
    "inspection_id": "8fe34d36"
  },

  "inspection": {
    "sku": "SKU001",
    "judgment": "NG",
    "overall_delta_e": 17.90,
    "confidence": 0.42,
    "ng_reasons": ["Zone A: ΔE=17.90 > 4.00"],
    "zone_count": 3
  },

  "zone_results": [ ... ],  // 기존과 동일

  "lens_detection": { ... },  // 기존과 동일

  // ========== 추가 정보 ==========

  "radial_profile": {
    "r_normalized": [0.0, 0.003, 0.006, ..., 1.0],  // 348개 샘플
    "L": [99.8, 99.7, 99.5, ..., 25.3],
    "a": [0.1, 0.2, 0.3, ..., 8.5],
    "b": [0.5, 0.6, 0.7, ..., -3.2],
    "std_L": [0.5, 0.6, 0.8, ..., 2.1],
    "std_a": [0.1, 0.1, 0.2, ..., 0.5],
    "std_b": [0.2, 0.2, 0.3, ..., 0.8],
    "pixel_count": [360, 360, 360, ..., 360],
    "length": 348,

    "statistics": {
      "L_mean": 72.3,
      "L_std": 18.5,
      "L_min": 25.3,
      "L_max": 99.8,
      "a_mean": 3.2,
      "a_std": 2.1,
      "b_mean": 5.8,
      "b_std": 4.3
    }
  },

  "boundary_detection": {
    "r_inner": 0.150,
    "r_outer": 0.948,
    "confidence": 0.85,
    "method": "auto",
    "peaks": [
      {"r": 0.15, "height": 12.5, "prominence": 8.3},
      {"r": 0.45, "height": 8.2, "prominence": 5.1}
    ],
    "valleys": [
      {"r": 0.30, "depth": -5.2},
      {"r": 0.70, "depth": -3.8}
    ]
  },

  "background_mask": {
    "valid_pixel_ratio": 0.451,
    "total_pixels": 704536,
    "valid_pixels": 317821,
    "filtered_by_luminance": 317268,
    "filtered_by_saturation": 316031,
    "filtered_by_circular": 69447,
    "mask_method": "luminance_saturation"
  },

  "zones": [ ... ],  // 기존과 동일

  "ring_sector_cells": [ ... ],  // 기존과 동일 (36개)

  "uniformity_analysis": { ... },  // 기존과 동일

  "processing_times": {
    "total_ms": 245.3,
    "image_load_ms": 15.2,
    "lens_detection_ms": 42.5,
    "radial_profile_ms": 28.1,
    "zone_segmentation_ms": 12.3,
    "color_evaluation_ms": 8.5,
    "boundary_detection_ms": 18.7,
    "background_mask_ms": 35.2,
    "angular_profile_ms": 62.8,
    "uniformity_analysis_ms": 22.0
  },

  "config_snapshot": {
    "image_config": {
      "target_size": 800,
      "interpolation": "lanczos"
    },
    "detector_config": {
      "method": "hybrid",
      "min_radius": 50,
      "max_radius": 500
    },
    "profiler_config": {
      "r_start_ratio": 0.15,
      "r_step_pixels": 1,
      "smoothing_enabled": true
    },
    "segmenter_config": {
      "expected_zones": 3,
      "derivative_threshold": 2.0
    }
  },

  "images": {
    "original": {
      "format": "jpeg",
      "encoding": "base64",
      "data": "/9j/4AAQSkZJRg...",  // Base64 인코딩
      "width": 800,
      "height": 800,
      "channels": 3
    }
  }
}
```

**완전한 JSON 크기:**
- 이미지 포함: ~250KB
- 이미지 제외: ~50KB

---

## 🤖 AI 활용 시나리오

### 1. Supervised Learning: 불량 패턴 학습

**목적**: 과거 검사 데이터로 AI 모델 학습 → 자동 OK/NG 판정

**필요 데이터:**
- ✅ `radial_profile.L`, `radial_profile.a`, `radial_profile.b` (특징 벡터)
- ✅ `ring_sector_cells` (36개 셀 Lab 값)
- ✅ `uniformity_analysis.max_delta_e`, `mean_delta_e`
- ✅ `judgment` (라벨)

**모델 예시:**
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import json

# 1. 데이터 로드
training_data = []
labels = []

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)

    # 특징 추출
    features = []

    # Radial profile 통계
    features.append(data['radial_profile']['statistics']['L_mean'])
    features.append(data['radial_profile']['statistics']['L_std'])
    features.append(data['radial_profile']['statistics']['a_std'])
    features.append(data['radial_profile']['statistics']['b_std'])

    # Ring×Sector 평균
    for cell in data['ring_sector_cells']:
        features.append(cell['mean_L'])
        features.append(cell['std_L'])

    # 균일성 지표
    features.append(data['uniformity_analysis']['max_delta_e'])
    features.append(data['uniformity_analysis']['mean_delta_e'])

    training_data.append(features)
    labels.append(1 if data['inspection']['judgment'] == 'OK' else 0)

# 2. 모델 학습
X = np.array(training_data)
y = np.array(labels)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 3. 예측
accuracy = model.score(X, y)
print(f"Model Accuracy: {accuracy:.2%}")

# 4. 중요 특징 분석
importances = model.feature_importances_
print("Top 5 Important Features:")
for idx in np.argsort(importances)[-5:]:
    print(f"  Feature {idx}: {importances[idx]:.3f}")
```

**기대 효과:**
- 자동 판정 정확도: 95%+
- 검사 시간 단축: 30% (수동 검증 감소)
- 일관성 향상: 검사자 편차 제거

---

### 2. Anomaly Detection: 이상치 탐지

**목적**: 정상 패턴 학습 → 새로운 불량 유형 자동 탐지

**필요 데이터:**
- ✅ `radial_profile` (전체 곡선)
- ✅ `ring_sector_cells` (36개 셀)
- ✅ `uniformity_analysis.outlier_cells`

**모델 예시:**
```python
from sklearn.ensemble import IsolationForest

# 1. OK 샘플만 사용
ok_profiles = []
for json_file in ok_json_files:
    with open(json_file) as f:
        data = json.load(f)

    # Radial profile을 벡터로 변환
    profile = data['radial_profile']['L']
    ok_profiles.append(profile)

X_ok = np.array(ok_profiles)

# 2. Isolation Forest 학습 (비지도 학습)
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X_ok)

# 3. 새 샘플 이상치 판정
def is_anomaly(new_json_file):
    with open(new_json_file) as f:
        data = json.load(f)

    profile = np.array(data['radial_profile']['L']).reshape(1, -1)
    prediction = iso_forest.predict(profile)

    return prediction[0] == -1  # -1 = 이상치

# 4. 결과
print(f"Anomaly: {is_anomaly('new_sample.json')}")
```

**기대 효과:**
- 신규 불량 패턴 자동 감지
- False Negative 감소 (놓치는 불량 감소)
- 품질 관리 고도화

---

### 3. Adaptive Thresholding: 자동 임계값 최적화

**목적**: SKU별 최적 threshold 자동 학습

**필요 데이터:**
- ✅ `zone_results[].delta_e`
- ✅ `zone_results[].threshold`
- ✅ `judgment` (실제 판정)
- ✅ 검사자 피드백 (Optional)

**분석 예시:**
```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# 1. 데이터 수집
delta_e_values = []
true_labels = []  # 0=OK, 1=NG

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)

    for zone in data['zone_results']:
        delta_e_values.append(zone['delta_e'])
        true_labels.append(0 if zone['is_ok'] else 1)

df = pd.DataFrame({
    'delta_e': delta_e_values,
    'label': true_labels
})

# 2. ROC 곡선으로 최적 threshold 찾기
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(df['label'], df['delta_e'])

# Youden's J statistic으로 최적점 찾기
J = tpr - fpr
optimal_idx = np.argmax(J)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal Threshold: {optimal_threshold:.2f}")
print(f"TPR: {tpr[optimal_idx]:.2%}, FPR: {fpr[optimal_idx]:.2%}")

# 3. Zone별 최적 threshold 계산
zone_thresholds = {}
for zone_name in df['zone_name'].unique():
    zone_df = df[df['zone_name'] == zone_name]
    # ... ROC 계산 반복
    zone_thresholds[zone_name] = optimal_threshold
```

**기대 효과:**
- SKU별 맞춤 threshold → 정확도 +10%
- Over-rejection 감소 → 수율 향상
- 데이터 기반 의사결정

---

### 4. Predictive Quality: 품질 예측

**목적**: 실시간 Lab 값으로 최종 ΔE 예측

**필요 데이터:**
- ✅ `radial_profile` (초반 30% 구간)
- ✅ `ring_sector_cells` (Ring 0만)
- ✅ `overall_delta_e` (타겟)

**모델 예시:**
```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# 1. 초반 30% 구간만 사용
X_train = []
y_train = []

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)

    profile = data['radial_profile']
    length = len(profile['L'])
    early_portion = int(length * 0.3)

    # 초반 구간 통계
    features = [
        np.mean(profile['L'][:early_portion]),
        np.std(profile['L'][:early_portion]),
        np.mean(profile['a'][:early_portion]),
        np.std(profile['b'][:early_portion])
    ]

    X_train.append(features)
    y_train.append(data['inspection']['overall_delta_e'])

# 2. 모델 학습
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

model = Ridge(alpha=1.0)
model.fit(X_scaled, y_train)

# 3. 조기 예측
def predict_quality_early(new_json_file):
    with open(new_json_file) as f:
        data = json.load(f)

    profile = data['radial_profile']
    length = len(profile['L'])
    early_portion = int(length * 0.3)

    features = [
        np.mean(profile['L'][:early_portion]),
        np.std(profile['L'][:early_portion]),
        np.mean(profile['a'][:early_portion]),
        np.std(profile['b'][:early_portion])
    ]

    X_new = scaler.transform([features])
    predicted_de = model.predict(X_new)[0]

    return predicted_de

# 결과
predicted = predict_quality_early('sample.json')
print(f"Predicted ΔE: {predicted:.2f}")
```

**기대 효과:**
- 조기 품질 예측 (30% 진행 시점)
- 불량 렌즈 조기 제외 → 비용 절감
- 실시간 공정 제어

---

### 5. Root Cause Analysis: 불량 원인 분석

**목적**: NG 판정 시 원인 자동 분석

**필요 데이터:**
- ✅ `uniformity_analysis.ring_uniformity`
- ✅ `uniformity_analysis.sector_uniformity`
- ✅ `ring_sector_cells` (outlier 위치)
- ✅ `zones` (zone_type: pure/mix)

**분석 예시:**
```python
def diagnose_ng_cause(json_file):
    with open(json_file) as f:
        data = json.load(f)

    if data['inspection']['judgment'] == 'OK':
        return "No issue"

    causes = []

    # 1. Ring별 불균일성 체크
    for ring_idx, stats in data['uniformity_analysis']['ring_uniformity'].items():
        if not stats['is_uniform']:
            causes.append(f"Ring {ring_idx} 불균일 (ΔE={stats['mean_de']:.1f})")

    # 2. Sector별 불균일성 체크
    non_uniform_sectors = []
    for sector_idx, stats in data['uniformity_analysis']['sector_uniformity'].items():
        if not stats['is_uniform']:
            non_uniform_sectors.append(int(sector_idx))

    if len(non_uniform_sectors) >= 3:
        angles = [s * 30 for s in non_uniform_sectors]
        causes.append(f"특정 각도 불량: {angles}°")

    # 3. Zone 경계 문제
    for zone in data['zones']:
        if zone['zone_type'] == 'mix':
            causes.append(f"Zone {zone['name']}: 경계 혼합 영역 발견")

    # 4. 전체 평균 ΔE
    if data['inspection']['overall_delta_e'] > 30:
        causes.append("전체적 색상 편차 심각")

    return causes

# 결과
causes = diagnose_ng_cause('ng_sample.json')
print("NG 원인:")
for cause in causes:
    print(f"  - {cause}")
```

**출력 예시:**
```
NG 원인:
  - Ring 0 불균일 (ΔE=31.8)
  - 특정 각도 불량: [180, 210, 240]°
  - 전체적 색상 편차 심각
```

**기대 효과:**
- 불량 원인 자동 분류
- 공정 개선 포인트 식별
- 생산 라인 피드백 자동화

---

## 🚀 텔레메트리 사용 방법

### 1. 현재 JSON 활용 (바로 가능)

```python
import json
import glob

# 1. 모든 검사 결과 로드
json_files = glob.glob("results/web/*/result.json")

ok_samples = []
ng_samples = []

for json_file in json_files:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if data['judgment'] == 'OK':
        ok_samples.append(data)
    else:
        ng_samples.append(data)

print(f"OK: {len(ok_samples)}, NG: {len(ng_samples)}")

# 2. 통계 분석
import pandas as pd

# Zone A의 ΔE 분포
zone_a_de = [
    zr['delta_e']
    for data in ok_samples + ng_samples
    for zr in data['zone_results']
    if zr['zone_name'] == 'A'
]

df = pd.DataFrame({'delta_e': zone_a_de})
print(df.describe())

# 3. 시각화
import matplotlib.pyplot as plt

plt.hist(zone_a_de, bins=50)
plt.xlabel('ΔE')
plt.ylabel('Count')
plt.title('Zone A ΔE Distribution')
plt.axvline(4.0, color='red', linestyle='--', label='Threshold')
plt.legend()
plt.show()
```

---

### 2. 완전한 텔레메트리 활용 (추가 구현 필요)

```python
from src.utils.telemetry import TelemetryExporter

# 검사 실행 후
exporter = TelemetryExporter(
    include_images=False,  # 용량 절약
    include_radial_profile=True,  # AI 학습 필수
    include_processing_times=True,
    include_config_snapshot=True
)

full_telemetry = exporter.export_full_telemetry(
    inspection_result=result,
    radial_profile=radial_profile,
    lens_detection=lens_detection,
    zones=zones,
    ring_sector_cells=cells,
    uniformity_analysis=uniformity_data,
    boundary_detection=boundary_info,  # 추가 필요
    background_mask_stats=mask_stats,   # 추가 필요
    processing_times=times,             # 추가 필요
    config_snapshot=configs             # 추가 필요
)

# JSON 저장
exporter.save_json(full_telemetry, "results/full_telemetry.json")
```

---

## 📈 데이터 분석 워크플로우

### Step 1: 데이터 수집 (최소 100개 샘플)
- OK 샘플: 50개 이상
- NG 샘플: 50개 이상
- 다양한 SKU 포함

### Step 2: 탐색적 데이터 분석 (EDA)
```python
import seaborn as sns

# Zone별 ΔE 분포
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='zone_name', y='delta_e', hue='judgment')
plt.title('ΔE Distribution by Zone')
plt.show()

# Ring별 균일성
ring_uniformity = [
    data['uniformity_analysis']['ring_uniformity'][str(i)]['mean_de']
    for data in all_samples
    for i in range(3)
]

plt.hist(ring_uniformity, bins=30)
plt.xlabel('Ring Mean ΔE')
plt.ylabel('Count')
plt.show()
```

### Step 3: 특징 엔지니어링
- Radial profile 통계량
- Ring×Sector 공간 분포
- 균일성 지표
- 미분 피크 위치/높이

### Step 4: 모델 학습 및 검증
- Train/Test 분할 (80/20)
- Cross-validation
- Hyperparameter tuning

### Step 5: 모델 배포
- FastAPI 엔드포인트 추가
- 실시간 예측
- 모델 성능 모니터링

---

## ✅ 다음 단계

### 즉시 가능 (현재 JSON 활용)
1. ✅ 과거 검사 결과 수집 (`results/web/*/result.json`)
2. ✅ Zone별 ΔE 통계 분석
3. ✅ Ring×Sector 불균일성 패턴 분석
4. ✅ 간단한 머신러닝 모델 학습

### 추가 구현 필요 (완전한 텔레메트리)
1. ⏳ `radial_profile` JSON 추가
2. ⏳ `boundary_detection` 상세 추가
3. ⏳ `processing_times` 추적
4. ⏳ `config_snapshot` 저장

### AI 모델 개발
1. ⏳ Supervised Learning (OK/NG 분류)
2. ⏳ Anomaly Detection (이상치 탐지)
3. ⏳ Adaptive Thresholding (최적 임계값)
4. ⏳ Root Cause Analysis (원인 분석)

---

**결론:** 현재 JSON만으로도 상당한 AI 분석이 가능하며, 추가 정보를 포함하면 더욱 고도화된 분석이 가능합니다!
