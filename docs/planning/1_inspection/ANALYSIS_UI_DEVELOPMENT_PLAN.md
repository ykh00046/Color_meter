# Analysis-Focused Web UI Development Plan

## 문서 정보
- **작성일**: 2025-12-12
- **목적**: 콘택트렌즈 색상 검사 시스템의 분석 및 검증 중심 웹 UI 개발 계획
- **핵심 원칙**: 분석 모드 우선 (Analysis Mode First), 판정은 옵션

---

## 1. 프로젝트 목표 및 배경

### 1.1 핵심 목표

이 프로젝트의 웹 UI는 **단순한 OK/NG 판정 결과 표시 도구가 아닌**, **렌즈 색상 프로파일의 분석 및 경계 검출 알고리즘 검증 도구**입니다.

### 1.2 이론적 배경

**극좌표 변환 기반 방사형 프로파일 분석:**
- 렌즈 중심으로부터의 거리(r)에 따른 L*, a*, b* 값 추출
- 도넛(Donut) 형태의 동심원 구조 → 회전 불변성(Rotation Invariance)
- 각 반경별 평균 색상 계산 → 1차원 프로파일 생성

**경계 검출 방법론:**
1. **1차 미분 (Gradient)**: 색상 변화율이 높은 지점 = 경계 후보
2. **2차 미분 (Inflection Point)**: 변화율의 변화가 큰 지점 = 변곡점
3. **ΔE 기반 검출**: CIEDE2000 색차가 급격히 변하는 지점
4. **Change Point Detection**: PELT, Binary Segmentation 등 통계적 알고리즘

**검증 필요성:**
- 자동 검출된 경계가 실제 렌즈 Zone 경계와 일치하는지 시각적 확인
- 파라미터(스무딩 윈도우, 임계값 등) 조정 후 실시간 재계산
- `expected_zones`는 자동 검출 실패 시 보정용 힌트로만 사용

---

## 2. 현재 구현 상태 분석

### 2.1 기존 구현 (Worker B)

**Backend (src/web/app.py):**
- ✅ FastAPI 기본 구조
- ✅ `/inspect` endpoint: 단일 이미지 검사
- ✅ `/batch` endpoint: 배치 처리
- ✅ Radial profile 데이터 추출 및 반환 (`radius`, `L`, `a`, `b`)
- ✅ Overlay 이미지 생성 및 반환
- ❌ 1차 미분 데이터 계산/반환 없음
- ❌ 2차 미분 데이터 계산/반환 없음
- ❌ ΔE 프로파일 계산/반환 없음
- ❌ 경계 후보(boundary candidates) 구조화된 데이터 없음
- ❌ 파라미터 조정 API 없음

**Frontend (src/web/templates/index.html):**
- ✅ 단순한 Pico CSS 기반 UI
- ✅ Chart.js로 L*, a*, b* vs radius 그래프 표시
- ✅ Overlay 이미지 표시
- ❌ 1차/2차 미분 그래프 없음
- ❌ ΔE vs radius 그래프 없음
- ❌ 경계 후보 테이블 없음
- ❌ 테이블 클릭 시 이미지에 원 표시 기능 없음
- ❌ 파라미터 조정 UI 없음
- ❌ 실시간 재계산 기능 없음

### 2.2 문서 업데이트 (Worker B)

Worker B가 다음 문서들을 "분석 모드 우선" 원칙에 맞춰 업데이트:

**README.md (lines 161-166):**
```markdown
## 현재 진행 원칙 (중요)
- 기본 흐름은 **분석 모드**(프로파일/스무딩/미분/피크)이며, OK/NG 판정은 옵션으로 뒤에서 실행합니다.
- `expected_zones`는 자동 경계 검출이 실패했을 때 보정용 힌트로만 사용합니다.
- 광학부(중심부) 배제를 위해 SKU에 `params.optical_clear_ratio`(또는 r_min) 필드를 설정해 앞 구간을 제외할 수 있습니다.
- 웹 UI 단건 탭에서 프로파일·미분 그래프와 경계 후보를 먼저 확인한 뒤, 필요 시 판정/비교를 수행하세요.
```

**docs/USER_GUIDE.md Section 8 (lines 249-254):**
- 동일한 원칙 반복 강조
- Analysis mode as default, judgment as optional
- expected_zones는 correction hint only
- optical_clear_ratio로 중심 광학부 제외 가능

**docs/WEB_UI.md:**
- Usage 설명: "기본은 분석 모드로 프로파일(원본/스무딩), 미분, 피크/경계 후보와 overlay를 먼저 보여주고, 필요 시 판정 실행"
- Notes: expected_zones는 자동 경계 검출 실패 시 보정용으로만 사용

---

## 3. 개발 요구사항 상세

### 3.1 Backend API 개선

#### 3.1.1 새로운 데이터 계산 모듈

**위치:** `src/analysis/` (신규 디렉토리)

**파일: `src/analysis/profile_analyzer.py`**
```python
class ProfileAnalyzer:
    """
    RadialProfile 데이터를 받아 분석용 데이터를 생성
    """

    def compute_gradient(self, profile, smoothing_window=5):
        """1차 미분 계산 (Savitzky-Golay 필터 적용)"""
        pass

    def compute_second_derivative(self, profile, smoothing_window=5):
        """2차 미분 계산"""
        pass

    def compute_delta_e_profile(self, profile, baseline_lab):
        """반경별 ΔE 계산 (CIEDE2000)"""
        pass

    def detect_inflection_points(self, second_derivative, threshold=0.1):
        """변곡점 검출"""
        pass

    def detect_change_points(self, profile, method='gradient', **params):
        """
        경계 후보 검출
        method: 'gradient', 'delta_e', 'pelt', 'binary_seg'
        """
        pass
```

#### 3.1.2 API Endpoint 확장

**파일: `src/web/app.py`**

**기존 `/inspect` endpoint 수정:**
```python
@app.post("/inspect")
async def inspect_image(
    file: UploadFile = File(...),
    sku: str = Form(...),
    expected_zones: Optional[int] = Form(None),
    run_judgment: bool = Form(False),  # 판정 실행 여부 (기본값: False)
    smoothing_window: int = Form(5),
    gradient_threshold: float = Form(0.5),
    inflection_threshold: float = Form(0.1)
):
    """
    분석 모드:
    1. Radial profile 추출 (raw + smoothed)
    2. 1차 미분 계산
    3. 2차 미분 및 변곡점 검출
    4. ΔE 프로파일 계산
    5. 경계 후보 검출 (여러 방법)
    6. Overlay 생성 (경계 후보 표시)
    7. (옵션) run_judgment=True 시 OK/NG 판정
    """
    # ... existing code ...

    # 분석 수행
    from src.analysis.profile_analyzer import ProfileAnalyzer
    analyzer = ProfileAnalyzer()

    analysis_results = {
        "profile": {
            "radius": profile.r_normalized.tolist(),
            "L_raw": profile.L.tolist(),
            "a_raw": profile.a.tolist(),
            "b_raw": profile.b.tolist(),
            "L_smoothed": analyzer.smooth(profile.L, smoothing_window).tolist(),
            "a_smoothed": analyzer.smooth(profile.a, smoothing_window).tolist(),
            "b_smoothed": analyzer.smooth(profile.b, smoothing_window).tolist()
        },
        "derivatives": {
            "gradient_L": analyzer.compute_gradient(profile.L, smoothing_window).tolist(),
            "gradient_a": analyzer.compute_gradient(profile.a, smoothing_window).tolist(),
            "gradient_b": analyzer.compute_gradient(profile.b, smoothing_window).tolist(),
            "second_derivative_L": analyzer.compute_second_derivative(profile.L, smoothing_window).tolist()
        },
        "delta_e": {
            "profile": analyzer.compute_delta_e_profile(profile, baseline_lab).tolist()
        },
        "boundary_candidates": [
            {
                "method": "gradient_L",
                "radius_px": 120.5,
                "radius_normalized": 0.35,
                "value": 2.3,
                "confidence": 0.85
            },
            {
                "method": "inflection_point",
                "radius_px": 122.0,
                "radius_normalized": 0.36,
                "value": 0.15,
                "confidence": 0.72
            }
            # ... more candidates
        ],
        "judgment": None  # run_judgment=False일 때
    }

    # run_judgment=True일 때만 판정 실행
    if run_judgment:
        analysis_results["judgment"] = {
            "result": result.judgment,
            "overall_delta_e": result.overall_delta_e,
            "zone_results": [...]
        }

    return analysis_results
```

**신규 endpoint: `/recompute`**
```python
@app.post("/recompute")
async def recompute_analysis(
    run_id: str = Form(...),
    smoothing_window: int = Form(5),
    gradient_threshold: float = Form(0.5),
    # ... 기타 파라미터
):
    """
    이미 업로드된 이미지에 대해 파라미터만 변경하여 재계산
    (이미지 재업로드 불필요)
    """
    pass
```

### 3.2 Frontend UI 개선

#### 3.2.1 레이아웃 구조

**파일: `src/web/templates/index_analysis.html` (신규)**

```
┌─────────────────────────────────────────────────────────────┐
│  Header: Color Meter - Analysis & Verification Tool         │
├─────────────────────────────────────────────────────────────┤
│  Tabs: [ Single Analysis ] [ Batch Processing ]             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌─────────────────────────────────────┐ │
│  │  Upload      │  │  Parameter Controls                  │ │
│  │  - File      │  │  - Smoothing Window: [5]    [Apply] │ │
│  │  - SKU       │  │  - Gradient Threshold: [0.5]        │ │
│  │  - (옵션)    │  │  - Inflection Threshold: [0.1]      │ │
│  │   expected_  │  │  - Detection Method: [Gradient ▼]   │ │
│  │   zones      │  │                                      │ │
│  │              │  │  [☐ Run OK/NG Judgment]              │ │
│  │  [Analyze]   │  │                                      │ │
│  └──────────────┘  └─────────────────────────────────────┘ │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Image with Overlay (Interactive)                     │  │
│  │  - Click boundary in table → Show circle on image    │  │
│  │  - Display detected zones, boundaries                │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Graphs (Chart.js / Plotly.js)                        │  │
│  │                                                        │  │
│  │  Graph 1: Radial Profile (L*, a*, b* vs Radius)      │  │
│  │  - Raw (thin line) + Smoothed (thick line)           │  │
│  │  - Vertical lines for detected boundaries            │  │
│  │                                                        │  │
│  │  Graph 2: ΔE vs Radius                                │  │
│  │  - Show tolerance threshold line                      │  │
│  │                                                        │  │
│  │  Graph 3: 1st Derivative (Gradient)                   │  │
│  │  - dL/dr, da/dr, db/dr                                │  │
│  │  - Mark peaks above threshold                         │  │
│  │                                                        │  │
│  │  Graph 4: 2nd Derivative (Inflection Points)          │  │
│  │  - d²L/dr²                                             │  │
│  │  - Mark inflection points                             │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Boundary Candidates Table (Interactive)              │  │
│  │  ┌────────┬──────────┬────────┬────────┬──────────┐  │  │
│  │  │ Method │ Radius   │ Value  │ Conf.  │ Action   │  │  │
│  │  ├────────┼──────────┼────────┼────────┼──────────┤  │  │
│  │  │ Grad_L │ 120.5 px │ 2.30   │ 85%    │ [Show]   │  │  │
│  │  │ Inflec │ 122.0 px │ 0.15   │ 72%    │ [Show]   │  │  │
│  │  │ ΔE     │ 119.8 px │ 3.45   │ 90%    │ [Show]   │  │  │
│  │  └────────┴──────────┴────────┴────────┴──────────┘  │  │
│  │  - Click [Show] → Highlight circle on image above    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Judgment Results (Optional, shown when requested)    │  │
│  │  - Overall: OK/NG                                     │  │
│  │  - Overall ΔE: 2.45                                   │  │
│  │  - Zone-wise results table                            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### 3.2.2 주요 기능 구현

**1. Interactive Image Overlay**
- Canvas or SVG overlay on image
- Click boundary candidate in table → Draw circle at that radius
- Multiple boundaries can be shown simultaneously with different colors
- Legend: Auto-detected (green), Expected zones (blue), Selected (red)

**2. Multi-Graph Visualization**
- Use Chart.js or Plotly.js
- All graphs synchronized: Zoom/pan on one graph affects all
- Vertical line markers for detected boundaries
- Tooltip showing exact values on hover

**3. Boundary Candidates Table**
- Sortable by method, radius, confidence
- Filter by method (checkbox: Gradient, Inflection, ΔE, etc.)
- Click row → Highlight on image + vertical line on graphs
- Export table as CSV

**4. Parameter Adjustment Interface**
- Sliders for smoothing_window, thresholds
- [Apply] button → Call `/recompute` API → Update all visualizations
- Show loading spinner during recomputation
- No need to re-upload image (use run_id)

**5. Optional Judgment Section**
- Collapsed by default
- Checkbox: "Run OK/NG Judgment"
- When checked and [Analyze] clicked → Show judgment results below

#### 3.2.3 기술 스택

**Frontend Libraries:**
- **Chart.js** or **Plotly.js**: 그래프 시각화
  - Chart.js: 가볍고 빠름, 기본 라인 차트에 적합
  - Plotly.js: 더 많은 기능 (동기화, 3D 등), 파일 크기 큼
  - **권장: Chart.js** (현재 이미 사용 중, 단순함)

- **Canvas API** or **SVG**: 이미지 위 원 그리기
  - Canvas: 픽셀 기반, 빠름
  - SVG: 벡터 기반, 확대/축소 시 품질 유지
  - **권장: Canvas** (overlay 이미지 위에 그리기만 하면 되므로)

- **Alpine.js** (옵션): 가벼운 반응형 상태 관리
  - 또는 Vanilla JS로 충분

**CSS Framework:**
- 현재 Pico CSS 사용 중 → 유지 (깔끔하고 가벼움)
- 또는 Tailwind CSS (더 많은 커스터마이징 가능하지만 무거움)
- **권장: Pico CSS 유지**

---

## 4. 구현 단계별 계획

### Phase 1: Backend Analysis Module (우선순위: 최고)

**목표:** 분석 데이터 계산 및 API 반환

**작업:**
1. `src/analysis/profile_analyzer.py` 생성
   - `compute_gradient()`: Savitzky-Golay 필터 + 1차 미분
   - `compute_second_derivative()`: 2차 미분
   - `compute_delta_e_profile()`: 반경별 ΔE 계산
   - `detect_peaks()`: Scipy signal.find_peaks 활용
   - `detect_inflection_points()`: 2차 미분의 zero-crossing 검출
   - `detect_change_points()`: 여러 방법 구현 (gradient, delta_e 기반)

2. `src/web/app.py` 수정
   - `/inspect` endpoint에 ProfileAnalyzer 통합
   - `run_judgment` 파라미터 추가 (기본값: False)
   - 분석 결과 JSON 구조 설계 (위 3.1.2 참고)
   - 경계 후보 데이터 구조화

3. 테스트
   - 샘플 이미지로 API 호출 테스트
   - JSON 응답 검증

**예상 소요:** 1일

### Phase 2: Frontend Graph Visualization (우선순위: 높음)

**목표:** 4개 그래프 표시 (Profile, ΔE, Gradient, 2nd Derivative)

**작업:**
1. `index_analysis.html` 생성 (또는 기존 index.html 전면 수정)
2. Chart.js로 4개 그래프 구현
   - 각 그래프에 raw + smoothed 데이터 표시
   - 경계 후보 위치에 수직선 마커 추가
   - 범례 및 툴팁 설정
3. 그래프 동기화 (옵션)
   - 줌/팬 동기화 (Chart.js plugin 필요)

**예상 소요:** 1일

### Phase 3: Interactive Boundary Table + Image Overlay (우선순위: 높음)

**목표:** 테이블 클릭 시 이미지에 원 표시

**작업:**
1. 경계 후보 테이블 렌더링
   - JSON 데이터 파싱하여 HTML table 생성
   - Sortable.js 또는 네이티브 JS로 정렬 기능
2. Canvas overlay 구현
   - 이미지 위에 Canvas 레이어 추가
   - 테이블 행 클릭 → `drawCircle(centerX, centerY, radius)` 호출
   - 여러 경계 동시 표시 (색상 구분)
3. 그래프와 연동
   - 테이블 클릭 → 그래프에도 수직선 강조 표시

**예상 소요:** 1일

### Phase 4: Parameter Adjustment & Recompute (우선순위: 중간)

**목표:** 파라미터 변경 후 실시간 재계산

**작업:**
1. `/recompute` API endpoint 구현
   - run_id로 기존 이미지 로드
   - 새 파라미터로 ProfileAnalyzer 재실행
   - 동일한 JSON 구조 반환
2. Frontend에 파라미터 조정 UI 추가
   - 슬라이더: smoothing_window (1-15)
   - 숫자 입력: gradient_threshold, inflection_threshold
   - 드롭다운: detection_method
   - [Apply] 버튼 → `/recompute` 호출 → 모든 그래프/테이블 업데이트
3. Loading state 표시
   - 재계산 중 스피너 표시

**예상 소요:** 0.5일

### Phase 5: Optional Judgment Section (우선순위: 낮음)

**목표:** 필요 시 OK/NG 판정 실행 및 표시

**작업:**
1. Frontend에 체크박스 추가: "Run OK/NG Judgment"
2. 체크 시 `/inspect` 호출 시 `run_judgment=True` 전달
3. 판정 결과 섹션 렌더링
   - Collapsible section (기본 접힌 상태)
   - Overall judgment, ΔE, zone-wise 결과 테이블

**예상 소요:** 0.5일

### Phase 6: Documentation & Testing (우선순위: 중간)

**작업:**
1. `docs/WEB_UI.md` 업데이트
   - 새로운 분석 UI 사용법 상세 설명
   - 스크린샷 추가
2. 단위 테스트 작성
   - `test_profile_analyzer.py`
   - Mock 데이터로 각 분석 함수 테스트
3. 통합 테스트
   - 실제 샘플 이미지로 엔드투엔드 테스트

**예상 소요:** 1일

---

## 5. 데이터 구조 설계

### 5.1 Backend Response JSON

```json
{
  "run_id": "a3b4c5d6",
  "image": "sample.jpg",
  "sku": "SKU001",
  "analysis": {
    "profile": {
      "radius": [0, 0.01, 0.02, ..., 1.0],
      "L_raw": [72.3, 72.5, ..., 85.2],
      "a_raw": [137.1, 137.3, ..., -1.5],
      "b_raw": [122.6, 122.8, ..., 2.3],
      "L_smoothed": [72.3, 72.4, ..., 85.2],
      "a_smoothed": [137.1, 137.2, ..., -1.5],
      "b_smoothed": [122.6, 122.7, ..., 2.3]
    },
    "derivatives": {
      "gradient_L": [0.0, 0.05, ..., -0.02],
      "gradient_a": [0.0, 0.03, ..., -0.8],
      "gradient_b": [0.0, 0.02, ..., 0.15],
      "second_derivative_L": [0.0, 0.001, ..., -0.003]
    },
    "delta_e": {
      "profile": [0.0, 0.2, 0.5, ..., 12.3],
      "baseline_lab": {"L": 72.2, "a": 137.3, "b": 122.8}
    },
    "boundary_candidates": [
      {
        "method": "gradient_L",
        "radius_px": 120.5,
        "radius_normalized": 0.35,
        "value": 2.30,
        "confidence": 0.85,
        "is_peak": true
      },
      {
        "method": "inflection_point",
        "radius_px": 122.0,
        "radius_normalized": 0.36,
        "value": 0.15,
        "confidence": 0.72,
        "is_peak": true
      },
      {
        "method": "delta_e",
        "radius_px": 119.8,
        "radius_normalized": 0.35,
        "value": 3.45,
        "confidence": 0.90,
        "is_peak": true
      }
    ],
    "detected_zones": [
      {"name": "Zone_A", "r_start": 0.0, "r_end": 0.35},
      {"name": "Zone_B", "r_start": 0.35, "r_end": 1.0}
    ]
  },
  "overlay": "/results/a3b4c5d6/overlay.png",
  "judgment": null  // or {...} if run_judgment=true
}
```

### 5.2 Boundary Candidate 구조

```python
@dataclass
class BoundaryCandidate:
    method: str  # 'gradient_L', 'gradient_a', 'gradient_b', 'inflection_point', 'delta_e', 'pelt', etc.
    radius_px: float  # 픽셀 단위 반경
    radius_normalized: float  # 0-1 정규화 반경
    value: float  # 해당 메서드의 값 (예: gradient 크기, 2차 미분 값)
    confidence: float  # 0-1, 신뢰도 (옵션)
    is_peak: bool  # 피크 여부
```

---

## 6. 핵심 알고리즘 구현 방침

### 6.1 Smoothing (스무딩)

**방법:** Savitzky-Golay Filter
- `scipy.signal.savgol_filter(data, window_length, polyorder)`
- 장점: 데이터의 전반적 형태 유지하면서 노이즈 제거, 피크 보존
- 파라미터: `window_length` (홀수, 예: 5, 7, 11)

### 6.2 Gradient (1차 미분)

**방법:** NumPy gradient
- `np.gradient(smoothed_data, radius)`
- 또는 Savitzky-Golay의 deriv=1 옵션 사용

### 6.3 Second Derivative (2차 미분)

**방법:**
- `np.gradient(gradient, radius)` 다시 적용
- 또는 `savgol_filter(..., deriv=2)`

### 6.4 Inflection Point Detection

**방법:** 2차 미분의 Zero-Crossing
- 2차 미분이 양수→음수 또는 음수→양수로 바뀌는 지점
- `np.where(np.diff(np.sign(second_derivative)))[0]`
- Threshold 적용하여 미세한 변화 무시

### 6.5 ΔE Profile Calculation

**방법:** CIEDE2000
- 각 반경 r에 대해 Lab(r)와 baseline Lab 간 ΔE 계산
- 이미 구현된 `color_utils.py`의 `ciede2000()` 함수 활용

### 6.6 Change Point Detection (고급)

**방법 1: Gradient-based**
- Gradient의 피크 검출 (`scipy.signal.find_peaks`)
- Threshold: `prominence`, `height`, `distance` 설정

**방법 2: ΔE-based**
- ΔE profile의 급격한 변화 지점 검출
- 동일하게 `find_peaks` 활용

**방법 3: Statistical (옵션)**
- PELT (Pruned Exact Linear Time): `ruptures` 라이브러리
- Binary Segmentation: `ruptures.Binseg()`
- 장점: 사전 threshold 불필요, 자동 검출
- 단점: 계산량 증가, 추가 라이브러리 필요

**권장:** 우선 Gradient-based + ΔE-based 구현, 필요 시 PELT 추가

---

## 7. 예상 이슈 및 해결 방안

### 7.1 이슈: 노이즈로 인한 과도한 경계 후보 검출

**해결:**
- Smoothing window 크게 설정
- Peak detection의 `prominence`, `distance` 파라미터 조정
- Confidence threshold 적용 (예: confidence < 0.5인 후보 필터링)

### 7.2 이슈: 경계 후보가 너무 적음

**해결:**
- Smoothing window 작게 설정
- Threshold 낮춤
- 여러 detection method 병행 사용

### 7.3 이슈: 실시간 재계산 성능 저하

**해결:**
- 이미지 전처리 결과 캐싱 (run_id 기반)
- ProfileAnalyzer 함수 최적화 (NumPy 벡터화)
- 필요 시 백그라운드 작업으로 전환 (WebSocket으로 진행률 전송)

### 7.4 이슈: 그래프 동기화 복잡도

**해결:**
- Chart.js의 경우 `zoom` plugin 사용
- 또는 단순하게 동일한 x축 범위 설정으로 시작
- 고급 동기화는 Phase 2 이후 옵션으로 추가

### 7.5 이슈: expected_zones와 자동 검출 결과 비교 UI

**해결:**
- 경계 후보 테이블에 "Expected" 열 추가
- Expected zones의 경계를 파란색으로, Auto-detected를 초록색으로 구분
- Overlay 이미지에도 동일한 색상 코드 적용

---

## 8. 성공 기준 (Definition of Done)

1. **분석 우선 원칙 준수**
   - UI 로드 시 기본적으로 분석 모드로 진입
   - OK/NG 판정은 명시적으로 요청했을 때만 실행

2. **필수 그래프 표시**
   - ✅ Radial Profile (L*, a*, b* raw + smoothed)
   - ✅ ΔE vs Radius
   - ✅ 1st Derivative (Gradient)
   - ✅ 2nd Derivative (Inflection Points)

3. **경계 후보 검출 및 표시**
   - ✅ 최소 3가지 방법으로 경계 후보 검출 (Gradient, Inflection, ΔE)
   - ✅ 테이블 형태로 정리하여 표시
   - ✅ Confidence score 함께 표시

4. **Interactive Verification**
   - ✅ 테이블 클릭 → 이미지에 원 표시
   - ✅ 여러 경계 동시 표시 가능
   - ✅ 색상 코드로 구분 (method별 또는 confidence별)

5. **Parameter Adjustment**
   - ✅ 최소 3개 파라미터 조정 가능 (smoothing, gradient threshold, inflection threshold)
   - ✅ 파라미터 변경 후 재계산 기능
   - ✅ 재계산 시 이미지 재업로드 불필요

6. **Documentation**
   - ✅ `docs/WEB_UI.md` 업데이트
   - ✅ 사용법 스크린샷 포함
   - ✅ 각 그래프 및 기능 설명

7. **Testing**
   - ✅ 최소 3개 샘플 이미지로 테스트
   - ✅ 1-zone, 2-zone, 3-zone 렌즈 각각 테스트
   - ✅ 노이즈가 많은 이미지(도트 인쇄)로 테스트

---

## 9. 다음 단계 (Next Steps)

### 즉시 시작 가능한 작업:

1. **Phase 1 시작: Backend Analysis Module 구현**
   - `src/analysis/` 디렉토리 생성
   - `profile_analyzer.py` 뼈대 작성
   - `compute_gradient()` 함수 구현 및 테스트

2. **샘플 데이터 준비**
   - 1-zone, 2-zone, 3-zone 샘플 이미지 각 1개씩 준비
   - 노이즈가 많은 도트 인쇄 이미지 1개 추가

3. **API 테스트 환경 구축**
   - Postman 또는 curl로 `/inspect` endpoint 테스트 스크립트 작성
   - 예상 JSON 응답 구조 mock 데이터 생성

---

## 10. 참고 자료

### 관련 문서:
- `docs/USER_GUIDE.md` Section 8: 현재 진행 원칙
- `docs/WEB_UI.md`: 웹 UI 가이드 (기존)
- `README.md`: 프로젝트 개요 및 진행 원칙

### 기술 참고:
- Chart.js 공식 문서: https://www.chartjs.org/
- Scipy Signal Processing: https://docs.scipy.org/doc/scipy/reference/signal.html
- Savitzky-Golay Filter: https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
- CIEDE2000: https://en.wikipedia.org/wiki/Color_difference#CIEDE2000

### 알고리즘 논문 (옵션):
- Change Point Detection: Killick et al., "Optimal Detection of Changepoints With a Linear Computational Cost"
- Inflection Point Detection in Noisy Data: Various signal processing literature

---

**작성자:** Claude (Assistant)
**검토 필요:** User
**버전:** 1.0
**최종 수정:** 2025-12-12
