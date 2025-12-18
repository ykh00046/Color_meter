# Contact Lens Color Extraction: Dual Analysis System

> **작성일**: 2025-12-17
> **버전**: 2.0
> **대상**: 개발자 및 품질 관리 팀

## 📋 개요

Contact Lens Color Inspection System은 **두 가지 독립적인 색상 추출 방법**을 병렬로 실행하여 렌즈의 잉크 개수와 색상을 분석합니다:

1. **Zone-Based Method** (구역 기반)
2. **Image-Based Method** (이미지 기반)

두 방법은 서로 다른 알고리즘과 목적을 가지며, 상호 보완적으로 작동합니다.

---

## 🎯 두 가지 방법의 비교

| 특징 | Zone-Based | Image-Based |
|------|------------|-------------|
| **알고리즘** | Radial Profiling + Transition Detection | GMM (Gaussian Mixture Model) + BIC |
| **입력** | SKU 기준값 필요 (expected_zones, baseline colors) | SKU 독립적 (기준값 불필요) |
| **분석 단위** | 방사형 구역 (Zone A, B, C) | 전체 픽셀 (공간 무관) |
| **검출 방식** | 미분 + 경계 검출 → 구역 매핑 | 비지도 학습 클러스터링 |
| **장점** | 구조화된 공간 정보 제공 (inner/middle/outer)<br>SKU 기준과 비교 가능 (ΔE, OK/NG) | SKU 없이도 실제 잉크 개수 추정<br>도트 패턴 혼합 자동 보정<br>신규 제품 탐색에 유용 |
| **단점** | SKU 설정 필수<br>경계 검출 실패 시 fallback 필요 | 공간 정보 부족 (어느 위치에 있는지 모름)<br>노이즈에 민감할 수 있음 |
| **출력** | Zone별 측정값, ΔE, OK/NG, 위치 정보 | Ink별 LAB/RGB/HEX, 픽셀 비중, 혼합 여부 |
| **활용** | 품질 검사 판정 (OK/NG/WARNING/RETAKE) | 잉크 개수 검증, 새 제품 분석, 배합 이상 탐지 |

---

## 🔬 방법 1: Zone-Based Color Extraction

### 📍 위치 및 구현

- **파일**: `src/core/zone_analyzer_2d.py`
- **함수**: `_perform_ink_analysis()` (Line 1737+)
- **호출**: `analyze_lens_zones_2d()` → `_perform_ink_analysis()`

### 📐 작동 원리

```
[1] Radial Profiling
    ↓ (극좌표 변환 후 방사형 프로파일 생성)
[2] Gradient Analysis
    ↓ (1차/2차 미분으로 경계 후보 검출)
[3] Transition Detection
    ↓ (스무딩 + 임계값으로 Zone 경계 확정)
[4] Zone Segmentation
    ↓ (구역 분할: C(inner), B(middle), A(outer))
[5] Color Measurement per Zone
    ↓ (각 Zone의 LAB 평균값 계산)
[6] Pixel Ratio Filtering
    ↓ (전체 잉크 픽셀의 5% 이상인 Zone만 잉크로 인정)
[7] Zone → Ink Mapping
    ↓
[Result] Zone-Based Inks (position, measured_color, delta_e, is_within_spec)
```

### 🔑 핵심 코드 (zone_analyzer_2d.py:1752-1824)

```python
# 🔧 FIX: Zone ≠ Ink. 충분한 잉크 픽셀이 있는 Zone만 잉크로 인정
MIN_INK_PIXEL_RATIO = 0.05  # 전체 잉크 픽셀의 5% 이상이어야 잉크로 간주

# 전체 잉크 픽셀 수 계산
total_ink_pixels = sum(zr["pixel_count_ink"] for zr in zone_results_raw)

# Zone 순서: C (inner) → B (middle) → A (outer)
inks_zone = []
ink_num = 1

for zr, zspec in zip(zone_results_raw, zone_specs):
    # 잉크 픽셀 비율 계산
    ink_pixel_ratio = zr["pixel_count_ink"] / total_ink_pixels

    # 충분한 잉크 픽셀이 있는 Zone만 잉크로 카운트
    if ink_pixel_ratio >= MIN_INK_PIXEL_RATIO:
        ink_info = {
            "ink_number": ink_num,
            "zone_name": zr["zone_name"],
            "position": "inner/middle/outer",
            "measured_color": {"L": ..., "a": ..., "b": ..., "rgb": ..., "hex": ...},
            "reference_color": {"L": ..., "a": ..., "b": ...},
            "delta_e": ...,
            "is_within_spec": ...,
            "pixel_count_ink": ...,
            "ink_pixel_ratio": ...,
        }
        inks_zone.append(ink_info)
        ink_num += 1
```

### 📊 출력 구조 (zone_based)

```json
{
  "zone_based": {
    "detected_ink_count": 2,
    "detection_method": "transition_based",  // or "fallback"
    "expected_ink_count": 2,  // from SKU config
    "inks": [
      {
        "ink_number": 1,
        "zone_name": "B",
        "position": "middle",
        "radial_range": [0.45, 0.75],
        "measured_color": {
          "L": 35.2, "a": 45.1, "b": 38.7,
          "rgb": [124, 56, 48], "hex": "#7C3830"
        },
        "reference_color": {"L": 36.0, "a": 44.0, "b": 40.0},
        "delta_e": 2.1,
        "is_within_spec": true,
        "pixel_count": 125000,
        "pixel_count_ink": 98000,
        "ink_pixel_ratio": 0.65
      },
      {
        "ink_number": 2,
        "zone_name": "A",
        "position": "outer",
        // ... (similar structure)
      }
    ],
    "all_zones": [ /* 모든 Zone 정보 (잉크 아닌 것도 포함) */ ],
    "filter_threshold": 0.05
  }
}
```

---

## 🎨 방법 2: Image-Based Color Extraction (InkEstimator)

### 📍 위치 및 구현

- **파일**: `src/core/ink_estimator.py`
- **클래스**: `InkEstimator`
- **메서드**: `estimate_from_array(bgr)`
- **호출**: `zone_analyzer_2d.py:_perform_ink_analysis()` → `InkEstimator.estimate_from_array()`

### 🤖 작동 원리 (GMM + BIC)

```
[1] Pixel Sampling
    ↓ (Chroma ≥ 6.0, L ≤ 98.0, 다운스케일 최적화)
[2] Pre-Check: Exposure Warnings
    ↓ (mean_L < 25 or > 90이면 경고)
[3] GMM Clustering (k=1~3)
    ↓ (Gaussian Mixture Model, Full Covariance)
[4] BIC Selection
    ↓ (Bayesian Information Criterion으로 최적 k 선택)
[5] Robustify Centers
    ↓ (Trimmed Mean으로 아웃라이어 제거)
[6] Merge Close Clusters
    ↓ (ΔE < 5.0인 군집 병합)
[7] Mixing Correction (k=3인 경우)
    ↓ (중간 톤이 두 극단의 혼합인지 판단 → 3→2 보정)
[8] Format Results
    ↓ (L값 순 정렬, LAB→RGB/HEX 변환)
[Result] Image-Based Inks (weight, lab, rgb, hex, is_mix)
```

### 🔑 핵심 알고리즘

#### 1️⃣ Pixel Sampling (ink_estimator.py:76-152)

```python
def sample_ink_pixels(self, bgr, max_samples=50000, chroma_thresh=6.0, L_max=98.0):
    """
    이미지에서 잉크로 추정되는 픽셀만 샘플링
    - 유채색 잉크: Chroma ≥ thresh
    - 무채색(Black) 잉크: L ≤ dark_thresh (Chroma 낮아도 허용)
    - 하이라이트 제거: L ≤ L_max
    """
    # LAB 변환
    lab_cv = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    L = lab_cv[..., 0] * (100.0 / 255.0)
    a = lab_cv[..., 1] - 128.0
    b = lab_cv[..., 2] - 128.0
    chroma = np.sqrt(a*a + b*b)

    # 필터링
    is_colored = chroma >= chroma_thresh
    is_dark = L <= L_dark_thresh
    is_not_highlight = L <= L_max
    mask = (is_colored | is_dark) & is_not_highlight

    # 샘플링 (최대 50,000개 or 5%)
    return samples, sampling_info
```

#### 2️⃣ GMM + BIC Selection (ink_estimator.py:154-201)

```python
def select_k_clusters(self, samples, k_min=1, k_max=3):
    """GMM + BIC로 최적 k 선택"""
    best_gmm = None
    best_bic = np.inf

    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",  # 타원형 클러스터 지원
            random_state=self.random_seed,
            reg_covar=1e-4,  # 안정성 향상
            n_init=3  # Local minima 방지
        )
        gmm.fit(samples)
        bic = gmm.bic(samples)  # Bayesian Information Criterion

        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm

    return best_gmm, best_bic
```

#### 3️⃣ Mixing Correction (ink_estimator.py:319-418)

**핵심 아이디어**: 도트 패턴 렌즈는 두 가지 잉크를 섞어 중간 톤을 만들지만, GMM은 이를 3개 군집으로 감지합니다. 중간 톤이 두 극단의 "가짜 혼합"인지 판단하여 3→2로 보정합니다.

```python
def correct_ink_count_by_mixing(self, centers, weights, linearity_thresh=3.0):
    """
    3개 군집일 때, 중간 톤이 두 극단(Dark, Bright)의 혼합인지 판단
    """
    if len(centers) != 3:
        return centers, weights, False

    # L값 기준 정렬 (Dark → Mid → Bright)
    order = np.argsort(centers[:, 0])
    c_dark, c_mid, c_bright = centers[order]
    w_dark, w_mid, w_bright = weights[order]

    # Vector: Dark → Bright
    vec_db = c_bright - c_dark
    len_db = np.linalg.norm(vec_db)
    u_db = vec_db / len_db

    # Mid의 투영 위치
    vec_dm = c_mid - c_dark
    projection_len = np.dot(vec_dm, u_db)
    projection_ratio = projection_len / len_db  # 0~1 사이면 "중간 위치"

    # 수직 거리 (선형성 오차)
    closest_point = c_dark + u_db * projection_len
    distance = np.linalg.norm(c_mid - closest_point)
    relative_distance = distance / len_db  # 스케일 독립적

    # 🔧 다중 조건 체크
    cond1_between = -0.1 <= projection_ratio <= 1.1  # Mid가 Dark-Bright 사이
    cond2_close_to_line = relative_distance < 0.15  # 직선 거리 < 15%
    cond3_mid_weight_ok = 0.05 < w_mid < 0.7  # Mid 비중 적절

    # 모든 조건 만족 시 혼합으로 판정
    if cond1_between and cond2_close_to_line and cond3_mid_weight_ok:
        print("[MIXING_CHECK] Mid-tone IS mixed. Merging to 2 inks.")

        # Mid 비중을 Dark/Bright에 분배
        ratio = np.clip(projection_ratio, 0.0, 1.0)
        new_weights = [
            w_dark + w_mid * (1.0 - ratio),  # Dark
            w_bright + w_mid * ratio          # Bright
        ]
        new_centers = [c_dark, c_bright]
        return new_centers, new_weights, True

    return centers, weights, False
```

**혼합 판정 기준**:
- ✅ **조건 1**: Mid가 Dark-Bright 사이에 위치 (projection_ratio: -10% ~ 110%)
- ✅ **조건 2**: Mid가 직선에서 멀지 않음 (relative_distance < 15%)
- ✅ **조건 3**: Mid 비중이 적절 (5% < weight < 70%)

**보정 효과**:
```
Before: [Dark(30%), Mid(40%), Bright(30%)] → 3 inks
After:  [Dark(50%), Bright(50%)] → 2 inks (Mid는 혼합으로 판단)
```

### 📊 출력 구조 (image_based)

```json
{
  "image_based": {
    "detected_ink_count": 2,
    "detection_method": "gmm_bic",
    "inks": [
      {
        "weight": 0.52,  // 픽셀 비율 (전체의 52%)
        "lab": [34.8, 46.2, 39.1],
        "rgb": [126, 54, 46],
        "hex": "#7E362E",
        "is_mix": false
      },
      {
        "weight": 0.48,
        "lab": [58.3, 28.5, 25.2],
        "rgb": [165, 128, 115],
        "hex": "#A58073",
        "is_mix": false
      }
    ],
    "meta": {
      "bic": 1234567.8,
      "sample_count": 45000,
      "correction_applied": true,  // 3→2 보정 적용 여부
      "sampling_config": {
        "chroma_threshold": 6.0,
        "L_max": 98.0,
        "downscale_factor": 0.8,
        "candidate_pixels": 150000,
        "sampled_pixels": 45000
      }
    }
  }
}
```

---

## 🔄 통합 구조 (Dual Analysis)

### 📦 최종 출력 (ink_analysis)

두 가지 방법의 결과가 하나의 `ink_analysis` 딕셔너리에 통합됩니다:

```json
{
  "ink_analysis": {
    "zone_based": {
      "detected_ink_count": 2,
      "detection_method": "transition_based",
      "expected_ink_count": 2,
      "inks": [ /* Zone별 잉크 정보 */ ],
      "all_zones": [ /* 모든 Zone 정보 */ ],
      "filter_threshold": 0.05
    },
    "image_based": {
      "detected_ink_count": 2,
      "detection_method": "gmm_bic",
      "inks": [ /* GMM 클러스터링 결과 */ ],
      "meta": {
        "bic": 1234567.8,
        "sample_count": 45000,
        "correction_applied": true
      }
    }
  }
}
```

### 🎯 사용 시나리오

| 시나리오 | Zone-Based | Image-Based |
|---------|-----------|-------------|
| **정상 검사** | ✅ 주 판정 기준 (ΔE, OK/NG) | ✅ 참고용 (잉크 개수 검증) |
| **SKU 없음** | ❌ 실행 불가 | ✅ 유일한 방법 |
| **경계 검출 실패** | ⚠️ Fallback 모드 | ✅ 정상 작동 (보조 역할) |
| **도트 패턴** | ⚠️ 3개로 과검출 가능 | ✅ Mixing Correction으로 2개 보정 |
| **신규 제품 탐색** | ❌ 기준값 필요 | ✅ 실제 잉크 개수 추정 가능 |
| **공간 정보 필요** | ✅ inner/middle/outer 제공 | ❌ 위치 정보 없음 |

---

## 📈 Web UI 통합

Web UI의 **"잉크 정보"** 탭에서 두 방법을 비교 확인할 수 있습니다:

### 화면 구성

```
┌─────────────────────────────────────────────────┐
│ 잉크 정보 (Ink Analysis)                         │
├─────────────────────────────────────────────────┤
│ [Zone-Based Analysis]                           │
│ - Detected: 2 inks                              │
│ - Method: transition_based                      │
│ - Expected: 2 (from SKU config)                 │
│                                                 │
│   Ink #1 (Zone B - Middle)                      │
│   ├─ Color: L=35.2, a=45.1, b=38.7 (#7C3830)   │
│   ├─ ΔE: 2.1 (✅ OK)                            │
│   └─ Pixels: 98,000 (65%)                       │
│                                                 │
│   Ink #2 (Zone A - Outer)                       │
│   ├─ Color: L=58.3, a=28.5, b=25.2 (#A58073)   │
│   ├─ ΔE: 3.5 (✅ OK)                            │
│   └─ Pixels: 52,000 (35%)                       │
│                                                 │
├─────────────────────────────────────────────────┤
│ [Image-Based Analysis (GMM)]                    │
│ - Detected: 2 inks                              │
│ - Method: gmm_bic                               │
│ - Mixing Correction: ✅ Applied (3→2)           │
│                                                 │
│   Ink #1 (Dark)                                 │
│   ├─ Color: L=34.8, a=46.2, b=39.1 (#7E362E)   │
│   └─ Weight: 52%                                │
│                                                 │
│   Ink #2 (Bright)                               │
│   ├─ Color: L=58.3, a=28.5, b=25.2 (#A58073)   │
│   └─ Weight: 48%                                │
│                                                 │
│ BIC Score: 1,234,567.8                          │
│ Sampled Pixels: 45,000 / 150,000               │
└─────────────────────────────────────────────────┘
```

---

## 🛠️ 개발자 가이드

### Zone-Based 방법 수정

**파일**: `src/core/zone_analyzer_2d.py`

```python
# 픽셀 필터링 임계값 조정
MIN_INK_PIXEL_RATIO = 0.05  # Line 1753
# → 5% 미만 Zone은 잉크로 간주하지 않음

# expected_zones 활용
sku_config.get("params", {}).get("expected_zones")  # Line 1850
# → SKU JSON에서 예상 잉크 개수 읽기
```

### Image-Based 방법 수정

**파일**: `src/core/ink_estimator.py`

```python
# 샘플링 파라미터 조정
chroma_thresh=6.0,     # 유채색 임계값 (높일수록 더 진한 잉크만)
L_max=98.0,            # 하이라이트 제거 (낮출수록 밝은 영역 제외)
max_samples=50000,     # 최대 샘플 수 (속도 vs 정확도)

# 클러스터링 파라미터
k_max=3,               # 최대 잉크 개수
merge_de_thresh=5.0,   # 유사 색상 병합 기준 (ΔE < 5.0)
linearity_thresh=3.0,  # Mixing correction 기준 (사용 안 함)

# Mixing correction 조건
RELATIVE_DIST_THRESH = 0.15  # Line 368
MIN_MID_WEIGHT = 0.05         # Line 369
MAX_MID_WEIGHT = 0.7          # Line 370
```

---

## 🚨 주의사항

### 1. 입력 이미지 품질

**Zone-Based와 Image-Based 모두**:
- ✅ White Balance 보정 완료 필수
- ✅ 적절한 노출 (mean_L: 25~90)
- ✅ 충분한 해상도 (최소 800×800)

**Image-Based 추가 요구사항**:
- ⚠️ 과다 노출 시 하이라이트 제거로 샘플 부족 가능
- ⚠️ 저조도 이미지에서 Chroma 임계값 부적절할 수 있음

### 2. SKU 설정 의존성

| 방법 | SKU 필수 여부 | 필수 필드 |
|------|-------------|----------|
| Zone-Based | ✅ 필수 | `zones`, `params.expected_zones` |
| Image-Based | ❌ 선택 | 없음 (독립적 실행) |

### 3. 결과 불일치 처리

두 방법의 잉크 개수가 다를 수 있습니다:

```python
zone_count = ink_analysis["zone_based"]["detected_ink_count"]
image_count = ink_analysis["image_based"]["detected_ink_count"]

if zone_count != image_count:
    # 원인 1: Zone 경계 검출 실패 (fallback 모드 확인)
    if ink_analysis["zone_based"]["detection_method"] == "fallback":
        print("Zone detection used fallback - may be inaccurate")

    # 원인 2: 도트 패턴 (Mixing correction 확인)
    if ink_analysis["image_based"]["meta"]["correction_applied"]:
        print("Dot pattern detected - Image-based corrected 3→2")

    # 원인 3: 미세한 Zone의 픽셀 비율 부족 (< 5%)
    # → all_zones 확인
```

---

## 📚 관련 문서

- [InkEstimator Guide](../guides/INK_ESTIMATOR_GUIDE.md): GMM 기반 잉크 분석 상세 설명
- [User Guide](../guides/USER_GUIDE.md): SKU 설정 및 검사 실행 방법
- [Web UI Guide](../guides/WEB_UI_GUIDE.md): 잉크 정보 탭 사용법
- [API Reference](../guides/API_REFERENCE.md): `/inspect`, `/recompute` API 스키마

---

## 🔍 디버깅 팁

### Zone-Based 디버깅

```python
# zone_analyzer_2d.py에서 출력되는 로그 확인
[INK_ZONE] Zone B counted as ink (ink_ratio=65.00%, ink_pixels=98000)
[INK_ZONE] Zone C excluded (ink_ratio=2.31% < 5%, ink_pixels=3500)
```

### Image-Based 디버깅

```python
# ink_estimator.py에서 출력되는 로그 확인
[MIXING_CHECK] 3 clusters detected - checking if middle is mixed
[MIXING_CHECK] Condition checks:
  - Mid between Dark-Bright: True (ratio=0.523)
  - Close to line: True (rel_dist=0.08 < 0.15)
  - Mid weight OK: True (0.05 < 0.42 < 0.7)
[MIXING_CHECK] OK Mid-tone IS mixed. Merging to 2 inks.
```

### 공통 디버깅

```bash
# 로그 레벨 조정
export LOG_LEVEL=DEBUG
python src/main.py --image sample.jpg --sku SKU001 --debug

# 시각화 활성화
python src/main.py --image sample.jpg --sku SKU001 --visualize
# → results/에 프로파일 그래프, 오버레이 이미지 생성
```

---

## ✅ 완료 상태

- ✅ Zone-Based 구현 완료 (zone_analyzer_2d.py)
- ✅ Image-Based 구현 완료 (ink_estimator.py)
- ✅ Dual Analysis 통합 완료 (_perform_ink_analysis)
- ✅ Web UI 통합 완료 (잉크 정보 탭)
- ✅ 테스트 커버리지 확보 (test_zone_analyzer_2d: 40개, test_ink_estimator: 12개)
- ✅ 문서화 완료 (본 문서)

---

## 📝 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2025-12-14 | 1.0 | 초기 작성 (GMM + Mixing Correction 통합) |
| 2025-12-16 | 1.1 | 테스트 추가 (52개 신규 테스트) |
| 2025-12-17 | 2.0 | **본 문서 작성** (Dual System 전체 설명) |

---

**작성자**: Claude Sonnet 4.5
**프로젝트**: Contact Lens Color Inspection System
**문서 위치**: `docs/design/COLOR_EXTRACTION_DUAL_SYSTEM.md`
