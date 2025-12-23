# Phase 7: Core Algorithm Improvements (Backend Focus)

**작성일**: 2025-12-12
**목적**: 사용자 수동 분석 방식 및 전문가 피드백 기반 핵심 알고리즘 개선
**범위**: Backend 알고리즘 및 API (UI 작업은 별도 문서)

---

## 📋 문서 개요

이 문서는 다음 세 가지 분석을 통합한 개선 계획입니다:

1. **사용자 수동 분석 방식 비교**
   - 배경색 기반 중심 검출
   - r_inner, r_outer 자동 검출
   - 균등 분할 + 자기 참조(Self-Reference) 균일성 분석
   - CIE76 기반 빠른 ΔE 계산

2. **전문가 피드백**
   - 조명 편차 보정 (Gray World / White Patch)
   - 2단계 배경 마스킹 (ROI + Otsu)
   - 가변 폭 링 분할
   - 표준편차/사분위수 지표
   - Lot 간 비교 워크플로우
   - 파라미터 노출 및 재계산

3. **AI 템플릿 분석** (2025-12-12 추가)
   - **Ring × Sector 2D 분할** (각도별 분석)
   - counts 기반 r_inner/outer 검출 (fallback용)
   - 균등 분할의 단순성 및 예측 가능성
   - 전체 평균 대비 ΔE (균일성 분석)

### ⚠️ UI 작업 분리

- **이 문서**: Backend 알고리즘, API endpoint, 데이터 구조
- **별도 작업 (다른 작업자)**: Frontend (HTML/CSS/JS, Chart.js, Canvas)
- **연동 지점**: API Response 형식, 파라미터 전달 방식

---

## 🎯 개선 목표

### 핵심 문제점

현재 시스템과 사용자 수동 분석 / AI 템플릿의 가장 큰 차이:

| 항목 | 현재 시스템 | 사용자 분석 / AI 템플릿 | 문제점 |
|------|-------------|------------------------|--------|
| **분석 범위** | 렌즈 전체 (0~1) | 실제 인쇄 영역만 (r_inner~r_outer) | 투명 외곽이 포함되어 색상 평균 희석 |
| **분석 차원** | ❌ 1D (Radial only) | ✅ 2D (**Ring × Sector**) | **각도별 불균일 검출 불가** ⭐⭐⭐ |
| **ΔE 기준** | SKU 절대 기준값 대비 | 전체 평균 대비 (자기 참조) | 균일성 분석 불가 |
| **조명 보정** | 없음 | 수동 조정 | 조명 불균일 시 왜곡 |
| **경계 검출** | Gradient/ΔE 피크 | 색도(chroma) 임계값 | 노이즈 민감도 차이 |

### 개선 효과 예상

- ✅ **각도별 불균일 검출 가능** (Ring × Sector 2D 분석) ⭐⭐⭐
- ✅ 색상 평균 정확도 **20-30% 향상** (r_inner/outer 자동 검출)
- ✅ 균일성 이상 패턴 검출 (자기 참조 모드)
- ✅ 조명 불균일 환경에서 안정성 확보
- ✅ 사용자가 파라미터 튜닝 가능 → 알고리즘 검증
- ✅ Heatmap 시각화로 품질 문제 직관적 파악

---

## 📦 개선 항목 전체 목록 (우선순위 재조정)

| # | 개선 항목 | 분류 | 우선순위 | 예상 시간 | Backend | Frontend | 출처 |
|---|-----------|------|----------|-----------|---------|----------|------|
| **0** | **Ring × Sector 2D 분할** | 알고리즘 | **🔴🔴🔴 Critical** | **1.5일** | ✅ | ⚠️ Heatmap | **AI 템플릿** ⭐ |
| 1 | r_inner, r_outer 자동 검출 | 알고리즘 | 🔴🔴 Highest | 1일 | ✅ | - | PHASE7 + 템플릿 |
| 2 | 2단계 배경 마스킹 | 전처리 | 🔴 High | 1일 | ✅ | - | PHASE7 |
| 3 | 자기 참조 균일성 분석 | 분석 | 🔴 High | 1일 | ✅ | ⚠️ 테이블 | 사용자 + 템플릿 |
| 4 | 조명 편차 보정 | 전처리 | 🟠 High | 1일 | ✅ | - | 전문가 |
| 5 | 에러 처리 및 제안 메시지 | 품질 | 🟠 High | 0.5일 | ✅ | ⚠️ 표시만 | PHASE7 |
| 6 | 표준편차/사분위수 지표 | 분석 | 🟢 Medium-High | 0.5일 | ✅ | ⚠️ 표시만 | 전문가 |
| 7 | 가변 폭 링 분할 개선 | 알고리즘 | 🟡 Medium | 1일 | ✅ | - | 전문가 |
| 8 | 파라미터 API (/recompute) | API | 🟡 Medium | 1.5일 | ✅ | ⚠️ UI 컨트롤 | 전문가 |
| 9 | Lot 간 비교 API (/compare) | API | 🟡 Medium | 2일 | ✅ | ⚠️ 비교 화면 | 전문가 |
| 10 | 배경색 기반 중심 검출 | 알고리즘 | 🟢 Low | 1일 | ✅ | - | 사용자 |
| 11 | 균등 분할 우선 옵션 | 알고리즘 | 🟢 Low | 0.5일 | ✅ | - | 사용자 |

**범례**:
- ✅ Backend: 이 Phase에서 구현
- ⚠️ Frontend: API만 제공, UI는 다른 작업자
- - : Frontend 작업 없음

**총 예상 시간 (Backend만)**: 약 **12.5일** (Sector 분할 추가)

**우선순위 설명**:
- 🔴🔴🔴 **Critical**: Sector 분할 - ANALYSIS_IMPROVEMENTS 핵심, 각도별 불균일 검출 필수
- 🔴🔴 **Highest**: r_inner/outer - 색상 정확도 향상의 기초
- 🔴 **High**: 균일성, 배경 마스킹 - 핵심 품질 개선
- 🟠 **Medium-High**: 조명 보정, 에러 처리 - 실제 환경 대응
- 🟡 **Medium**: API, 파라미터 - 사용성 개선
- 🟢 **Low**: Fallback 옵션 - 선택적 기능

---

## 🔧 Phase A: 핵심 품질 개선 (Backend Only, 5일)

### 1. r_inner, r_outer 자동 검출 ⭐⭐⭐

**목적**: 실제 인쇄 영역만 분석하여 색상 평균 정확도 향상

#### 구현 위치
```
src/analysis/profile_analyzer.py
```

#### 구현 내용

```python
def detect_print_boundaries(
    profile: RadialProfile,
    method: str = "chroma",  # "chroma", "gradient", "hybrid"
    chroma_threshold: float = 2.0
) -> Tuple[float, float]:
    """
    radial profile에서 실제 인쇄 영역의 r_inner, r_outer 자동 검출

    Args:
        profile: RadialProfile 객체
        method: 검출 방법
            - "chroma": 색도(sqrt(a^2 + b^2)) 기반
            - "gradient": 색도 그래디언트 기반
            - "hybrid": 둘 다 사용
        chroma_threshold: 배경 노이즈 임계값

    Returns:
        (r_inner, r_outer): 정규화된 반경 (0~1)

    Example:
        사용자 분석: r_inner=119px, r_outer=387px, lens_radius=400px
        → r_inner=0.2975, r_outer=0.9675
    """
    # 1. 색도(Chroma) 계산
    chroma = np.sqrt(profile.a**2 + profile.b**2)

    # 2. 배경 노이즈 레벨 추정 (최소값 10% 평균)
    noise_level = np.percentile(chroma, 10)

    # 3. 색이 있는 구간 검출
    threshold = noise_level + chroma_threshold
    colored_mask = chroma > threshold

    if not np.any(colored_mask):
        logger.warning("No colored area detected, using full range")
        return (0.0, 1.0)

    # 4. 첫/마지막 색 영역 찾기
    colored_indices = np.where(colored_mask)[0]
    inner_idx = colored_indices[0]
    outer_idx = colored_indices[-1]

    r_inner = float(profile.r_normalized[inner_idx])
    r_outer = float(profile.r_normalized[outer_idx])

    # 5. 안전성 체크
    if r_outer - r_inner < 0.2:
        logger.warning(f"Print area too narrow ({r_outer - r_inner:.3f}), may be detection error")

    logger.info(f"Detected print area: r_inner={r_inner:.3f}, r_outer={r_outer:.3f}")

    return (r_inner, r_outer)
```

#### Config 추가

```python
# src/core/radial_profiler.py
@dataclass
class ProfilerConfig:
    # ... 기존 필드
    auto_crop_print_area: bool = False  # 신규
    print_area_detection_method: str = "chroma"  # 신규
    chroma_threshold: float = 2.0  # 신규
```

#### Pipeline 통합

```python
# src/pipeline.py
def inspect_image(self, image_path: str, sku: str) -> InspectionResult:
    # ... (렌즈 검출, 프로파일 추출)

    # 신규: 인쇄 영역 자동 검출
    if self.config.auto_crop_print_area:
        r_inner, r_outer = detect_print_boundaries(profile,
                                                   method=self.config.print_area_detection_method,
                                                   chroma_threshold=self.config.chroma_threshold)

        # 프로파일 crop
        mask = (profile.r_normalized >= r_inner) & (profile.r_normalized <= r_outer)
        profile = RadialProfile(
            r_normalized=profile.r_normalized[mask],
            L=profile.L[mask],
            a=profile.a[mask],
            b=profile.b[mask],
            # ... (나머지)
        )

        # 결과에 기록
        result.print_boundaries = {"r_inner": r_inner, "r_outer": r_outer}
```

#### 테스트

```python
# tests/test_print_area_detection.py
def test_detect_print_boundaries_chroma():
    # 시뮬레이션: 중심 투명, 중간 색상, 외곽 투명
    r = np.linspace(0, 1, 100)
    chroma = np.zeros(100)
    chroma[20:80] = 15.0  # 인쇄 영역

    profile = create_mock_profile(r, chroma)
    r_inner, r_outer = detect_print_boundaries(profile)

    assert 0.15 < r_inner < 0.25
    assert 0.75 < r_outer < 0.85
```

#### UI 연동 (Frontend 작업자용)

**API Response에 추가**:
```json
{
  "print_boundaries": {
    "r_inner": 0.2975,
    "r_outer": 0.9675,
    "method": "chroma",
    "confidence": 0.92
  }
}
```

**Frontend 표시 예시** (참고용, 구현은 다른 작업자):
- Canvas에 r_inner, r_outer 원으로 표시
- "Print Area: 29.8% ~ 96.8%" 텍스트

---

### 2. 조명 편차 보정 ⭐⭐⭐

**목적**: 조명 불균일 환경에서 색상 안정성 확보

#### 구현 위치
```
src/utils/illumination.py (신규 파일)
```

#### 구현 내용

```python
import cv2
import numpy as np
from typing import Optional, Literal

class IlluminationCorrector:
    """조명 편차 보정 클래스"""

    def __init__(self, method: Literal['none', 'gray_world', 'white_patch', 'auto'] = 'auto'):
        self.method = method

    def correct(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        조명 편차 보정

        Args:
            image: BGR 이미지
            mask: 렌즈 영역 마스크 (1=렌즈, 0=배경). None이면 전체 이미지 사용

        Returns:
            보정된 BGR 이미지
        """
        if self.method == 'none':
            return image
        elif self.method == 'gray_world':
            return self._gray_world(image, mask)
        elif self.method == 'white_patch':
            return self._white_patch(image, mask)
        else:  # auto
            return self._auto_select(image, mask)

    def _gray_world(self, image: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Gray World Assumption: 전체 평균이 회색(128, 128, 128)이 되도록 조정

        원리: 자연 이미지의 RGB 채널 평균은 대체로 동일하다는 가정
        """
        if mask is not None:
            masked_pixels = image[mask > 0]
            if len(masked_pixels) == 0:
                return image
            channel_means = masked_pixels.mean(axis=0)
        else:
            channel_means = image.mean(axis=(0, 1))

        # 각 채널을 128 기준으로 스케일링
        scale = 128.0 / (channel_means + 1e-6)
        corrected = image.astype(np.float32) * scale

        return np.clip(corrected, 0, 255).astype(np.uint8)

    def _white_patch(self, image: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """
        White Patch: 가장 밝은 영역이 흰색(255, 255, 255)이 되도록 조정

        원리: 이미지에서 가장 밝은 픽셀이 흰색이라고 가정
        """
        if mask is not None:
            masked_pixels = image[mask > 0]
            if len(masked_pixels) == 0:
                return image
            channel_max = masked_pixels.max(axis=0)
        else:
            channel_max = image.max(axis=(0, 1))

        scale = 255.0 / (channel_max + 1e-6)
        corrected = image.astype(np.float32) * scale

        return np.clip(corrected, 0, 255).astype(np.uint8)

    def _auto_select(self, image: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """
        두 방법을 시도하고 더 안정적인 것 선택

        기준: 원본 평균 밝기와의 차이가 작은 것
        """
        gw = self._gray_world(image, mask)
        wp = self._white_patch(image, mask)

        orig_mean = np.mean(image)
        gw_diff = abs(np.mean(gw) - orig_mean)
        wp_diff = abs(np.mean(wp) - orig_mean)

        selected = 'gray_world' if gw_diff < wp_diff else 'white_patch'
        logger.debug(f"Auto-selected illumination correction: {selected}")

        return gw if gw_diff < wp_diff else wp
```

#### Config 추가

```python
# src/core/image_loader.py
@dataclass
class ImageLoaderConfig:
    # ... 기존 필드
    illumination_correction: str = 'none'  # 'none', 'gray_world', 'white_patch', 'auto'
```

#### Pipeline 통합

```python
# src/core/image_loader.py
class ImageLoader:
    def load(self, image_path: str) -> np.ndarray:
        # ... (이미지 로드)

        # 신규: 조명 보정
        if self.config.illumination_correction != 'none':
            from src.utils.illumination import IlluminationCorrector
            corrector = IlluminationCorrector(self.config.illumination_correction)
            image = corrector.correct(image)
            logger.info(f"Applied illumination correction: {self.config.illumination_correction}")

        return image
```

#### UI 연동 (Frontend 작업자용)

**API 파라미터**:
```json
POST /inspect
{
  "illumination_correction": "auto"  // "none", "gray_world", "white_patch", "auto"
}
```

**Frontend 컨트롤** (참고용):
```html
<select id="illumination">
  <option value="none">No Correction</option>
  <option value="auto" selected>Auto</option>
  <option value="gray_world">Gray World</option>
  <option value="white_patch">White Patch</option>
</select>
```

---

### 3. 2단계 배경 마스킹 ⭐⭐⭐

**목적**: 강건한 배경/렌즈 분리 (케이스, 그림자, 오염 대응)

#### 구현 위치
```
src/core/lens_detector.py (메서드 추가)
```

#### 구현 내용

```python
def create_background_mask(
    self,
    image: np.ndarray,
    lens: Optional[LensDetection] = None
) -> np.ndarray:
    """
    2단계 배경 마스크 생성

    Stage 1: ROI 밖에서 배경색 샘플링 (lens 정보 활용)
    Stage 2: Otsu + 색상 거리 이중 마스킹

    Args:
        image: BGR 이미지
        lens: 렌즈 검출 결과 (있으면 ROI 활용)

    Returns:
        mask: 0=배경, 255=렌즈
    """
    h, w = image.shape[:2]

    # Stage 1: ROI 기반 배경 샘플링
    if lens:
        # 렌즈 영역 마스크
        lens_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(lens_mask,
                   (int(lens.center_x), int(lens.center_y)),
                   int(lens.radius * 1.2),  # 여유 20%
                   255, -1)

        # ROI 밖에서만 배경 샘플링
        bg_samples = image[lens_mask == 0]
    else:
        # ROI 없으면 네 모서리 샘플링
        corners = [
            image[0:10, 0:10],
            image[0:10, w-10:w],
            image[h-10:h, 0:10],
            image[h-10:h, w-10:w]
        ]
        bg_samples = np.vstack([c.reshape(-1, 3) for c in corners])

    # 배경색: 중앙값 사용 (outlier에 강함)
    bg_color = np.median(bg_samples, axis=0)
    logger.debug(f"Detected background color: {bg_color}")

    # Stage 2a: Otsu 이진화
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Stage 2b: 색상 거리 마스크
    color_dist = np.linalg.norm(image - bg_color, axis=2)
    _, color_mask = cv2.threshold(color_dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # AND 결합
    combined_mask = cv2.bitwise_and(otsu_mask, color_mask.astype(np.uint8))

    # Morphology 정제
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    return combined_mask
```

#### 사용 예시

```python
# src/pipeline.py
def inspect_image(self, image_path: str, sku: str) -> InspectionResult:
    image = self.image_loader.load(image_path)
    lens = self.lens_detector.detect(image)

    # 배경 마스크 생성 (옵션)
    if self.config.use_background_mask:
        bg_mask = self.lens_detector.create_background_mask(image, lens)
        result.background_mask = bg_mask  # 저장
```

---

### 4. 에러 처리 및 제안 메시지 ⭐⭐

**목적**: 실패 시 명확한 원인 및 해결 방법 제시

#### 구현 내용

```python
# src/pipeline.py
@dataclass
class InspectionResult:
    # ... 기존 필드
    diagnostics: List[str] = field(default_factory=list)  # 진단 메시지
    suggestions: List[str] = field(default_factory=list)  # 개선 제안
    warnings: List[str] = field(default_factory=list)     # 경고

class InspectionPipeline:
    def inspect_image(self, image_path: str, sku: str) -> InspectionResult:
        diagnostics = []
        suggestions = []
        warnings = []

        try:
            # 1. Lens Detection
            try:
                lens = self.lens_detector.detect(image)
                diagnostics.append(f"✓ Lens detected: center=({lens.center_x:.1f}, {lens.center_y:.1f}), radius={lens.radius:.1f}, confidence={lens.confidence:.2f}")
            except LensDetectionError as e:
                diagnostics.append(f"✗ Lens detection failed: {e}")
                suggestions.append("→ Adjust hough_param1/param2 or use manual center/radius input")
                suggestions.append("→ Check if image background is clean")
                raise

            # 2. Print Area Detection
            if self.config.auto_crop_print_area:
                r_inner, r_outer = detect_print_boundaries(profile)
                diagnostics.append(f"✓ Print area detected: r_inner={r_inner:.3f}, r_outer={r_outer:.3f}")

                # 경고: 너무 좁음
                if r_outer - r_inner < 0.2:
                    warnings.append(f"⚠ Print area is very narrow ({r_outer - r_inner:.1%})")
                    suggestions.append("→ Check if lens detection center/radius is accurate")
                    suggestions.append("→ Adjust chroma_threshold parameter")

            # 3. Zone Segmentation
            expected = sku_config.get('params', {}).get('expected_zones')
            try:
                zones = self.zone_segmenter.segment(profile, expected_zones=expected)
                diagnostics.append(f"✓ Segmented into {len(zones)} zones: {[z.name for z in zones]}")
            except ZoneSegmentationError as e:
                diagnostics.append(f"✗ Zone segmentation failed: {e}")
                suggestions.append("→ Set 'expected_zones' in SKU config to force uniform split")
                suggestions.append("→ Increase smoothing_window or decrease min_gradient")

                # Fallback
                zones = self._fallback_3zone_split(profile)
                warnings.append("⚠ Using fallback 3-zone split")

            # 경고: 개수 불일치
            if expected and len(zones) != expected:
                warnings.append(f"⚠ Expected {expected} zones but got {len(zones)}")
                suggestions.append(f"→ Adjust min_gradient (current: {self.zone_segmenter.config.min_gradient}) or min_delta_e")
                suggestions.append(f"→ Or update expected_zones to {len(zones)} if this is correct")

            # 4. Color Evaluation
            result = self.color_evaluator.evaluate(zones, sku, sku_config)

            # 진단 정보 추가
            result.diagnostics = diagnostics
            result.suggestions = suggestions
            result.warnings = warnings

            return result

        except Exception as e:
            # 최종 에러 메시지 구성
            error_msg = f"Inspection failed: {str(e)}\n\n"
            error_msg += "=== Diagnostics ===\n" + "\n".join(diagnostics) + "\n\n"
            error_msg += "=== Suggestions ===\n" + "\n".join(suggestions)

            logger.error(error_msg)
            raise InspectionError(error_msg)
```

#### API Response

```json
{
  "judgment": "OK",
  "diagnostics": [
    "✓ Lens detected: center=(512.3, 498.7), radius=385.2, confidence=0.95",
    "✓ Print area detected: r_inner=0.298, r_outer=0.968",
    "✓ Segmented into 3 zones: ['A', 'B', 'C']"
  ],
  "warnings": [
    "⚠ Expected 3 zones but got 2"
  ],
  "suggestions": [
    "→ Adjust min_gradient (current: 0.25) or min_delta_e",
    "→ Or update expected_zones to 2 if this is correct"
  ]
}
```

#### UI 연동 (Frontend 작업자용)

**Frontend 표시** (참고용):
```html
<section class="diagnostics">
  <h3>Diagnostics</h3>
  <ul>
    <li class="success">✓ Lens detected: center=(512.3, 498.7), radius=385.2</li>
    <li class="warning">⚠ Expected 3 zones but got 2</li>
  </ul>

  <h3>Suggestions</h3>
  <ul>
    <li>→ Adjust min_gradient parameter</li>
    <li>→ Or update expected_zones to 2</li>
  </ul>
</section>
```

---

### 5. 표준편차/사분위수 지표 추가 ⭐⭐

**목적**: Zone 내부 균일도 분석 (색상 분산 검출)

#### 구현 내용

```python
# src/core/zone_segmenter.py의 Zone 클래스는 이미 std_L, std_a, std_b를 가지고 있음
# 추가로 사분위수 계산

# src/core/color_evaluator.py에 추가
def calculate_zone_statistics(
    self,
    zone: Zone,
    profile: RadialProfile
) -> Dict[str, Any]:
    """
    Zone의 상세 통계 계산

    Returns:
        {
            'mean_lab': (L, a, b),
            'std_lab': (std_L, std_a, std_b),
            'internal_uniformity': float,  # 0~1, 1=완벽히 균일
            'chroma_stats': {
                'q25': float,
                'median': float,
                'q75': float,
                'iqr': float
            },
            'pixel_count': int,
            'uniformity_grade': str  # 'Good', 'Medium', 'Poor'
        }
    """
    # Zone 구간 추출
    mask = (profile.r_normalized >= zone.r_end) & (profile.r_normalized < zone.r_start)
    zone_a = profile.a[mask]
    zone_b = profile.b[mask]
    zone_L = profile.L[mask]

    # 색도(Chroma) 계산
    chroma = np.sqrt(zone_a**2 + zone_b**2)

    # 사분위수
    q25, q50, q75 = np.percentile(chroma, [25, 50, 75])
    iqr = q75 - q25

    # 내부 균일도 점수
    internal_std = np.mean([zone.std_L, zone.std_a, zone.std_b])
    uniformity_score = 1.0 - min(internal_std / 20.0, 1.0)  # 0~1, std가 0이면 1

    # 등급
    if internal_std < 5:
        grade = 'Good'
    elif internal_std < 10:
        grade = 'Medium'
    else:
        grade = 'Poor'

    # 픽셀 수 추정 (반지름 비율로)
    pixel_count = int(np.pi * (zone.r_start**2 - zone.r_end**2) * (profile.r_normalized.shape[0])**2)

    return {
        'mean_lab': (zone.mean_L, zone.mean_a, zone.mean_b),
        'std_lab': (zone.std_L, zone.std_a, zone.std_b),
        'internal_uniformity': uniformity_score,
        'chroma_stats': {
            'q25': float(q25),
            'median': float(q50),
            'q75': float(q75),
            'iqr': float(iqr)
        },
        'pixel_count': pixel_count,
        'uniformity_grade': grade
    }
```

#### API Response

```json
{
  "zones": [
    {
      "name": "A",
      "mean_lab": [75.03, 3.02, 17.25],
      "std_lab": [4.2, 1.1, 2.3],
      "internal_uniformity": 0.78,
      "chroma_stats": {
        "q25": 15.1,
        "median": 17.5,
        "q75": 19.8,
        "iqr": 4.7
      },
      "pixel_count": 12345,
      "uniformity_grade": "Good"
    }
  ]
}
```

---

### 6. 가변 폭 링 분할 개선 ⭐

**목적**: 검출된 경계를 신뢰하되, expected_zones로 보정

#### 구현 내용

```python
# src/core/zone_segmenter.py 수정
@dataclass
class SegmenterConfig:
    detection_method: str = "hybrid"  # "uniform", "gradient", "delta_e", "hybrid", "variable_width"
    uniform_split_priority: bool = False
    # ... 기존 필드

def segment(self, profile: RadialProfile, expected_zones: Optional[int] = None) -> List[Zone]:
    hint_zones = expected_zones or self.config.expected_zones
    smooth_profile = self._smooth_profile(profile)

    # 1) 실제 경계 검출
    grad_pts = self._detect_by_gradient(smooth_profile)
    de_pts = self._detect_by_delta_e(smooth_profile)
    detected_boundaries = sorted(list(set(grad_pts + de_pts)), reverse=True)
    detected_boundaries = self._merge_close_boundaries(detected_boundaries, self.config.min_zone_width)

    # 2) 전략 선택
    if self.config.detection_method == "variable_width":
        # 가변 폭: 검출된 경계 우선, hint로 개수 조정
        boundaries = detected_boundaries
        if hint_zones and len(boundaries) != hint_zones - 1:
            boundaries = self._adjust_to_hint(boundaries, hint_zones, smooth_profile)

    elif self.config.detection_method == "uniform" or self.config.uniform_split_priority:
        # 균등 분할 우선
        boundaries = self._uniform_boundaries(hint_zones or 3)

    else:  # hybrid (기존)
        if detected_boundaries and (not hint_zones or len(boundaries) == hint_zones - 1):
            boundaries = detected_boundaries
        else:
            logger.info(f"Fallback to uniform split (detected {len(detected_boundaries)}, expected {hint_zones})")
            boundaries = self._uniform_boundaries(hint_zones or 3)

    # ... (Zone 생성)

def _adjust_to_hint(self, boundaries: List[float], hint_zones: int, profile: RadialProfile) -> List[float]:
    """
    검출된 경계를 hint_zones에 맞게 조정

    Strategy:
    - 경계가 많으면: 피크 강도가 약한 것부터 제거
    - 경계가 부족하면: 가장 넓은 구간을 분할
    """
    target_count = hint_zones - 1

    if len(boundaries) > target_count:
        # 피크 강도 재계산
        grad = np.gradient(profile.a)
        grad_strengths = []
        for b in boundaries:
            idx = np.argmin(np.abs(profile.r_normalized - b))
            strength = abs(grad[idx])
            grad_strengths.append(strength)

        # 강도 순으로 정렬, 상위 N개만 유지
        sorted_indices = np.argsort(grad_strengths)[::-1]
        keep_indices = sorted(sorted_indices[:target_count])
        boundaries = [boundaries[i] for i in keep_indices]

        logger.info(f"Reduced boundaries from {len(boundaries) + len(keep_indices)} to {target_count}")

    elif len(boundaries) < target_count:
        # 가장 넓은 구간 찾아서 분할
        boundaries_with_edges = [1.0] + boundaries + [0.0]

        while len(boundaries) < target_count:
            widths = [boundaries_with_edges[i] - boundaries_with_edges[i+1]
                     for i in range(len(boundaries_with_edges)-1)]
            widest_idx = np.argmax(widths)

            # 중간에 새 경계 추가
            new_boundary = (boundaries_with_edges[widest_idx] + boundaries_with_edges[widest_idx+1]) / 2
            boundaries.insert(widest_idx, new_boundary)
            boundaries = sorted(boundaries, reverse=True)
            boundaries_with_edges = [1.0] + boundaries + [0.0]

        logger.info(f"Expanded boundaries to {target_count}")

    return boundaries
```

---

## 🔌 Phase B: 분석 도구 고도화 (Backend + API, 4일)

### 7. 자기 참조 균일성 분석 ⭐⭐

**목적**: "이 렌즈가 균일한가?" 분석 (사용자 수동 분석 방식)

#### 구현 내용

```python
# src/core/color_evaluator.py에 추가
def evaluate_uniformity(
    self,
    zones: List[Zone],
    reference_mode: Literal['overall_mean', 'outer_zone', 'custom'] = 'overall_mean',
    custom_reference: Optional[Tuple[float, float, float]] = None
) -> Dict[str, Any]:
    """
    균일성 분석: 자기 참조(Self-Reference) 방식

    사용자 수동 분석 방식:
    - 전체 인쇄 영역 평균을 기준으로 각 Zone의 ΔE 계산
    - "Ring1이 전체보다 16 ΔE 밝다" 같은 인사이트 제공

    Args:
        zones: Zone 리스트
        reference_mode:
            - 'overall_mean': 전체 인쇄 영역 평균 (사용자 방식)
            - 'outer_zone': 가장 바깥 zone을 기준
            - 'custom': 특정 Lab 값 제공
        custom_reference: reference_mode='custom'일 때 사용

    Returns:
        {
            'reference_lab': (L, a, b),
            'reference_mode': str,
            'zone_uniformity': [
                {
                    'zone': 'A',
                    'delta_e_vs_ref': 16.92,
                    'delta_L': +16.13,
                    'delta_a': -0.77,
                    'delta_b': -5.07,
                    'deviation': 'high',
                    'description': 'Much brighter than average'
                },
                ...
            ],
            'overall_uniformity_score': 0.65,  # 0~1, 1=완벽히 균일
            'max_deviation': 16.92
        }
    """
    # 1. 기준 Lab 값 계산
    if reference_mode == 'custom':
        ref_lab = custom_reference
    elif reference_mode == 'outer_zone':
        ref_lab = (zones[-1].mean_L, zones[-1].mean_a, zones[-1].mean_b)
    else:  # overall_mean
        # 픽셀 수 가중 평균
        total_L, total_a, total_b = 0.0, 0.0, 0.0
        total_pixels = 0

        for zone in zones:
            # 면적 추정 (π * (r_start^2 - r_end^2))
            area = np.pi * (zone.r_start**2 - zone.r_end**2)
            total_L += zone.mean_L * area
            total_a += zone.mean_a * area
            total_b += zone.mean_b * area
            total_pixels += area

        ref_lab = (total_L / total_pixels, total_a / total_pixels, total_b / total_pixels)

    logger.info(f"Uniformity reference Lab (mode={reference_mode}): L*={ref_lab[0]:.2f}, a*={ref_lab[1]:.2f}, b*={ref_lab[2]:.2f}")

    # 2. 각 Zone의 ΔE 계산
    uniformity_results = []
    max_de = 0.0

    for zone in zones:
        measured = (zone.mean_L, zone.mean_a, zone.mean_b)
        de = self.calculate_delta_e(measured, ref_lab, method='cie1976')  # CIE76: 빠르고 직관적

        delta_L = zone.mean_L - ref_lab[0]
        delta_a = zone.mean_a - ref_lab[1]
        delta_b = zone.mean_b - ref_lab[2]

        max_de = max(max_de, de)

        # 편차 등급
        if de > 10:
            deviation = 'high'
            desc = self._describe_deviation(delta_L, delta_a, delta_b, 'high')
        elif de > 5:
            deviation = 'medium'
            desc = self._describe_deviation(delta_L, delta_a, delta_b, 'medium')
        else:
            deviation = 'low'
            desc = 'Similar to average'

        uniformity_results.append({
            'zone': zone.name,
            'delta_e_vs_ref': round(de, 2),
            'delta_L': round(delta_L, 2),
            'delta_a': round(delta_a, 2),
            'delta_b': round(delta_b, 2),
            'deviation': deviation,
            'description': desc
        })

    # 3. 전체 균일성 점수
    uniformity_score = 1.0 - min(max_de / 20.0, 1.0)

    return {
        'reference_lab': tuple(round(x, 2) for x in ref_lab),
        'reference_mode': reference_mode,
        'zone_uniformity': uniformity_results,
        'overall_uniformity_score': round(uniformity_score, 3),
        'max_deviation': round(max_de, 2)
    }

def _describe_deviation(self, dL: float, da: float, db: float, level: str) -> str:
    """편차 설명 자동 생성"""
    parts = []

    # 밝기
    if abs(dL) > 5:
        if dL > 0:
            parts.append("much brighter" if level == 'high' else "brighter")
        else:
            parts.append("much darker" if level == 'high' else "darker")

    # 색상
    if abs(da) > 3:
        if da > 0:
            parts.append("more red")
        else:
            parts.append("more green")

    if abs(db) > 3:
        if db > 0:
            parts.append("more yellow")
        else:
            parts.append("more blue")

    if not parts:
        return "Similar to average"

    return " and ".join(parts).capitalize() + " than average"
```

#### API Endpoint

```python
# src/web/app.py
@app.post("/inspect")
async def inspect_image(
    file: UploadFile,
    sku: str,
    run_judgment: bool = False,
    analyze_uniformity: bool = True,  # 신규
    uniformity_reference: str = "overall_mean"  # 신규
):
    # ... (기존 검사 로직)

    # 균일성 분석
    if analyze_uniformity:
        uniformity = evaluator.evaluate_uniformity(
            zones,
            reference_mode=uniformity_reference
        )
        response['uniformity_analysis'] = uniformity

    return response
```

#### API Response 예시

```json
{
  "uniformity_analysis": {
    "reference_lab": [58.90, 3.79, 22.32],
    "reference_mode": "overall_mean",
    "zone_uniformity": [
      {
        "zone": "Ring1",
        "delta_e_vs_ref": 16.92,
        "delta_L": 16.13,
        "delta_a": -0.77,
        "delta_b": -5.07,
        "deviation": "high",
        "description": "Much brighter and more blue than average"
      },
      {
        "zone": "Ring2",
        "delta_e_vs_ref": 11.01,
        "delta_L": -7.75,
        "delta_a": 1.41,
        "delta_b": 7.69,
        "deviation": "high",
        "description": "Much darker and more yellow than average"
      },
      {
        "zone": "Ring3",
        "delta_e_vs_ref": 3.79,
        "delta_L": -1.52,
        "delta_a": -0.69,
        "delta_b": -3.40,
        "deviation": "low",
        "description": "Similar to average"
      }
    ],
    "overall_uniformity_score": 0.154,
    "max_deviation": 16.92
  }
}
```

#### UI 연동 (Frontend 작업자용)

**테이블 예시** (참고용):
```html
<section id="uniformity-section">
  <h3>Uniformity Analysis (Self-Reference)</h3>
  <p>Reference Lab (Overall Mean): L*=58.90, a*=3.79, b*=22.32</p>
  <p>Uniformity Score: <strong>0.154</strong> (Low uniformity detected)</p>

  <table>
    <thead>
      <tr>
        <th>Zone</th>
        <th>L*, a*, b*</th>
        <th>ΔE (vs Ref)</th>
        <th>ΔL / Δa / Δb</th>
        <th>Deviation</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr class="deviation-high">
        <td>Ring 1</td>
        <td>75.03, 3.02, 17.25</td>
        <td class="high">16.92</td>
        <td>+16.13 / -0.77 / -5.07</td>
        <td>🔴 High</td>
        <td>Much brighter and more blue than average</td>
      </tr>
      <tr class="deviation-high">
        <td>Ring 2</td>
        <td>51.15, 5.20, 30.01</td>
        <td class="high">11.01</td>
        <td>-7.75 / +1.41 / +7.69</td>
        <td>🟠 High</td>
        <td>Much darker and more yellow than average</td>
      </tr>
      <tr class="deviation-low">
        <td>Ring 3</td>
        <td>57.38, 3.10, 18.92</td>
        <td class="low">3.79</td>
        <td>-1.52 / -0.69 / -3.40</td>
        <td>🟢 Low</td>
        <td>Similar to average</td>
      </tr>
    </tbody>
  </table>
</section>
```

---

### 8. 파라미터 API (/recompute) ⭐⭐⭐

**목적**: 이미지 재업로드 없이 파라미터만 변경하여 재분석

#### Backend 구현

```python
# src/web/app.py

# 이미지 캐시 (메모리)
image_cache = {}  # {image_id: np.ndarray}
cache_lock = asyncio.Lock()

@app.post("/inspect")
async def inspect_image(...):
    # ... (기존 검사 로직)

    # 캐시에 저장
    image_id = str(uuid.uuid4())
    async with cache_lock:
        image_cache[image_id] = image

    response['image_id'] = image_id  # 클라이언트에 반환
    return response

@app.post("/recompute")
async def recompute_analysis(
    image_id: str,
    sku: str,
    params: Dict[str, Any]  # 조정할 파라미터
):
    """
    동일 이미지를 파라미터만 변경하여 재분석

    Args:
        image_id: /inspect에서 받은 이미지 ID
        sku: SKU 코드
        params: {
            # Zone Segmentation
            "smoothing_window": 11,
            "min_gradient": 0.25,
            "min_delta_e": 2.0,
            "expected_zones": 3,

            # Radial Profiling
            "auto_crop_print_area": true,
            "chroma_threshold": 2.0,

            # Uniformity
            "uniformity_reference": "overall_mean"
        }

    Returns:
        동일한 형식의 검사 결과
    """
    # 캐시에서 이미지 로드
    async with cache_lock:
        image = image_cache.get(image_id)

    if image is None:
        raise HTTPException(status_code=404, detail="Image not found in cache. Please re-upload.")

    # Config 오버라이드
    segmenter_config = SegmenterConfig(
        smoothing_window=params.get('smoothing_window', 11),
        min_gradient=params.get('min_gradient', 0.25),
        min_delta_e=params.get('min_delta_e', 2.0),
        expected_zones=params.get('expected_zones')
    )

    profiler_config = ProfilerConfig(
        auto_crop_print_area=params.get('auto_crop_print_area', False),
        chroma_threshold=params.get('chroma_threshold', 2.0)
    )

    # 파이프라인 재실행
    pipeline = create_pipeline_with_configs(segmenter_config, profiler_config)
    result = pipeline.inspect(image, sku)

    # 균일성 분석
    if params.get('analyze_uniformity', True):
        uniformity = evaluator.evaluate_uniformity(
            result.zones,
            reference_mode=params.get('uniformity_reference', 'overall_mean')
        )
        result.uniformity_analysis = uniformity

    return result
```

#### UI 연동 (Frontend 작업자용)

**Frontend 컨트롤 예시** (참고용):
```html
<section id="advanced-params">
  <h3>Advanced Parameters</h3>

  <div class="param-group">
    <label>
      Smoothing Window:
      <input type="range" id="smoothing_window" min="5" max="51" step="2" value="11">
      <span id="smoothing_window_value">11</span>
    </label>

    <label>
      Min Gradient:
      <input type="number" id="min_gradient" min="0" max="5" step="0.05" value="0.25">
    </label>

    <label>
      Min ΔE:
      <input type="number" id="min_delta_e" min="0" max="10" step="0.1" value="2.0">
    </label>

    <label>
      Expected Zones:
      <input type="number" id="expected_zones" min="1" max="5" value="3">
    </label>
  </div>

  <button id="recompute-btn" onclick="recompute()">🔄 Recompute with New Parameters</button>
</section>

<script>
let currentImageId = null;

async function analyze() {
  const formData = new FormData();
  formData.append('file', document.getElementById('image-input').files[0]);
  formData.append('sku', document.getElementById('sku-select').value);

  const response = await fetch('/inspect', {
    method: 'POST',
    body: formData
  });

  const result = await response.json();
  currentImageId = result.image_id;  // 저장

  updateUI(result);
}

async function recompute() {
  if (!currentImageId) {
    alert('Please analyze an image first');
    return;
  }

  const params = {
    smoothing_window: parseInt(document.getElementById('smoothing_window').value),
    min_gradient: parseFloat(document.getElementById('min_gradient').value),
    min_delta_e: parseFloat(document.getElementById('min_delta_e').value),
    expected_zones: parseInt(document.getElementById('expected_zones').value)
  };

  const response = await fetch('/recompute', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      image_id: currentImageId,
      sku: document.getElementById('sku-select').value,
      params: params
    })
  });

  const result = await response.json();
  updateUI(result);  // 그래프 및 테이블 업데이트
}
</script>
```

---

### 9. Lot 간 비교 API (/compare) ⭐

**목적**: 레퍼런스 대비 테스트 이미지들의 차이 분석

#### Backend 구현

```python
# src/web/app.py
@app.post("/compare")
async def compare_lots(
    reference_file: UploadFile,
    test_files: List[UploadFile],
    sku: str
):
    """
    레퍼런스 이미지 대비 테스트 이미지들의 차이 분석

    Returns:
        {
            'reference': {
                'filename': 'ref.jpg',
                'zones': [...],
                'profile': [...]
            },
            'tests': [
                {
                    'filename': 'lot_002_001.jpg',
                    'zone_deltas': [
                        {
                            'zone': 'A',
                            'delta_L': -2.3,
                            'delta_a': 0.5,
                            'delta_b': 1.2,
                            'delta_e': 2.7
                        },
                        ...
                    ],
                    'overall_shift': 'Darker and more yellow',
                    'max_delta_e': 3.5
                }
            ],
            'batch_summary': {
                'mean_delta_e_per_zone': {'A': 2.3, 'B': 1.8, 'C': 3.1},
                'max_delta_e_per_zone': {'A': 4.5, 'B': 3.2, 'C': 5.8},
                'stability_score': 0.82,  # 0~1, 1=모두 동일
                'outliers': ['lot_002_005.jpg']
            }
        }
    """
    # 레퍼런스 검사
    ref_image = await load_image(reference_file)
    ref_result = pipeline.inspect(ref_image, sku)
    ref_zones = ref_result.zones

    # 테스트 이미지들 검사
    test_results = []
    all_deltas_per_zone = {zone.name: [] for zone in ref_zones}

    for test_file in test_files:
        test_image = await load_image(test_file)
        test_result = pipeline.inspect(test_image, sku)

        # Zone별 차이 계산
        zone_deltas = []
        max_de = 0.0

        for ref_zone, test_zone in zip(ref_zones, test_result.zones):
            delta_L = test_zone.mean_L - ref_zone.mean_L
            delta_a = test_zone.mean_a - ref_zone.mean_a
            delta_b = test_zone.mean_b - ref_zone.mean_b
            delta_e = np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)

            max_de = max(max_de, delta_e)

            zone_deltas.append({
                'zone': ref_zone.name,
                'delta_L': round(delta_L, 2),
                'delta_a': round(delta_a, 2),
                'delta_b': round(delta_b, 2),
                'delta_e': round(delta_e, 2)
            })

            all_deltas_per_zone[ref_zone.name].append(delta_e)

        # 전체 shift 설명
        avg_dL = np.mean([d['delta_L'] for d in zone_deltas])
        avg_da = np.mean([d['delta_a'] for d in zone_deltas])
        avg_db = np.mean([d['delta_b'] for d in zone_deltas])
        overall_shift = _describe_shift(avg_dL, avg_da, avg_db)

        test_results.append({
            'filename': test_file.filename,
            'zone_deltas': zone_deltas,
            'overall_shift': overall_shift,
            'max_delta_e': round(max_de, 2)
        })

    # 배치 요약
    batch_summary = {
        'mean_delta_e_per_zone': {
            zone: round(np.mean(deltas), 2)
            for zone, deltas in all_deltas_per_zone.items()
        },
        'max_delta_e_per_zone': {
            zone: round(np.max(deltas), 2)
            for zone, deltas in all_deltas_per_zone.items()
        },
        'stability_score': round(_calculate_stability_score(test_results), 3),
        'outliers': _detect_outliers(test_results)
    }

    return {
        'reference': {
            'filename': reference_file.filename,
            'zones': [serialize_zone(z) for z in ref_zones]
        },
        'tests': test_results,
        'batch_summary': batch_summary
    }

def _describe_shift(dL: float, da: float, db: float) -> str:
    """색상 shift 설명"""
    parts = []
    if abs(dL) > 3:
        parts.append("Darker" if dL < 0 else "Brighter")
    if abs(da) > 2:
        parts.append("more green" if da < 0 else "more red")
    if abs(db) > 2:
        parts.append("more blue" if db < 0 else "more yellow")

    return " and ".join(parts) if parts else "No significant shift"

def _calculate_stability_score(test_results: List[Dict]) -> float:
    """안정성 점수 (0~1, 1=모두 동일)"""
    all_max_des = [t['max_delta_e'] for t in test_results]
    mean_de = np.mean(all_max_des)
    return 1.0 - min(mean_de / 10.0, 1.0)

def _detect_outliers(test_results: List[Dict], threshold: float = 2.0) -> List[str]:
    """이상치 검출 (평균보다 threshold * std 이상)"""
    all_max_des = [t['max_delta_e'] for t in test_results]
    mean = np.mean(all_max_des)
    std = np.std(all_max_des)

    outliers = []
    for t in test_results:
        if t['max_delta_e'] > mean + threshold * std:
            outliers.append(t['filename'])

    return outliers
```

#### UI 연동 (Frontend 작업자용)

**화면 구성** (참고용):
```html
<section id="lot-comparison">
  <h2>Lot Comparison</h2>

  <div class="file-inputs">
    <label>Reference Image: <input type="file" id="ref-image"></label>
    <label>Test Images (multiple): <input type="file" id="test-images" multiple></label>
    <button onclick="compareLots()">Compare</button>
  </div>

  <div id="comparison-results">
    <h3>Batch Summary</h3>
    <p>Stability Score: <strong>0.82</strong> (Good consistency)</p>
    <p>Outliers: <span class="outlier">lot_002_005.jpg</span></p>

    <h4>Average ΔE per Zone</h4>
    <canvas id="zone-delta-chart"></canvas>

    <h3>Individual Results</h3>
    <table>
      <thead>
        <tr>
          <th>Filename</th>
          <th>Zone A ΔE</th>
          <th>Zone B ΔE</th>
          <th>Zone C ΔE</th>
          <th>Max ΔE</th>
          <th>Overall Shift</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>lot_002_001.jpg</td>
          <td>2.3</td>
          <td>1.8</td>
          <td>3.1</td>
          <td>3.1</td>
          <td>Darker and more yellow</td>
        </tr>
      </tbody>
    </table>
  </div>
</section>
```

---

## 📊 Phase C: 제조 라인 통합 (3일)

*(Lot 비교 워크플로우는 Phase B에 포함되었으므로, 추가 항목)*

### 배치 검사 요약 통계

```python
# src/web/app.py
@app.post("/batch")
async def batch_inspect(
    files: List[UploadFile],
    sku: str
):
    """
    배치 검사 + 통계 요약

    Returns:
        {
            'results': [...],  # 개별 검사 결과
            'summary': {
                'total': 10,
                'ok_count': 8,
                'ng_count': 2,
                'ok_rate': 0.80,
                'mean_delta_e': 2.3,
                'max_delta_e': 5.8,
                'zone_statistics': {
                    'A': {'mean_de': 2.1, 'std_de': 0.5, 'max_de': 3.2},
                    'B': {...},
                    'C': {...}
                }
            }
        }
    """
    results = []
    all_deltas_per_zone = {}

    for file in files:
        result = await inspect_single(file, sku)
        results.append(result)

        # Zone별 ΔE 수집
        for zone_result in result.zone_results:
            if zone_result.zone_name not in all_deltas_per_zone:
                all_deltas_per_zone[zone_result.zone_name] = []
            all_deltas_per_zone[zone_result.zone_name].append(zone_result.delta_e)

    # 요약 통계
    ok_count = sum(1 for r in results if r.judgment == 'OK')
    all_des = [r.overall_delta_e for r in results]

    summary = {
        'total': len(results),
        'ok_count': ok_count,
        'ng_count': len(results) - ok_count,
        'ok_rate': round(ok_count / len(results), 3),
        'mean_delta_e': round(np.mean(all_des), 2),
        'max_delta_e': round(np.max(all_des), 2),
        'zone_statistics': {
            zone: {
                'mean_de': round(np.mean(deltas), 2),
                'std_de': round(np.std(deltas), 2),
                'max_de': round(np.max(deltas), 2)
            }
            for zone, deltas in all_deltas_per_zone.items()
        }
    }

    return {
        'results': results,
        'summary': summary
    }
```

---

## 🧪 테스트 계획

### 단위 테스트

```python
# tests/test_print_area_detection.py
def test_detect_print_boundaries_chroma()
def test_detect_print_boundaries_gradient()
def test_detect_print_boundaries_hybrid()
def test_print_area_too_narrow_warning()

# tests/test_illumination.py
def test_gray_world_correction()
def test_white_patch_correction()
def test_auto_selection()
def test_correction_with_mask()

# tests/test_background_masking.py
def test_create_background_mask_with_roi()
def test_create_background_mask_without_roi()
def test_masking_with_case()

# tests/test_uniformity_analysis.py
def test_evaluate_uniformity_overall_mean()
def test_evaluate_uniformity_outer_zone()
def test_evaluate_uniformity_custom_reference()
def test_uniformity_score_calculation()

# tests/test_variable_width_segmentation.py
def test_adjust_to_hint_reduce_boundaries()
def test_adjust_to_hint_expand_boundaries()
def test_variable_width_mode()
```

### 통합 테스트

```python
# tests/test_pipeline_integration.py
def test_pipeline_with_print_area_detection()
def test_pipeline_with_illumination_correction()
def test_pipeline_with_uniformity_analysis()
def test_pipeline_error_handling_and_suggestions()
```

### API 테스트

```python
# tests/test_api.py
def test_inspect_with_uniformity()
def test_recompute_endpoint()
def test_compare_lots_endpoint()
def test_batch_summary()
```

---

## 📅 구현 일정

### Week 1: Phase A (5일)

| Day | 작업 | 담당 | 비고 |
|-----|------|------|------|
| 1 | r_inner/outer 자동 검출 + 테스트 | Backend | - |
| 2 | 조명 편차 보정 + 2단계 마스킹 | Backend | - |
| 3 | 에러 처리 + 표준편차/사분위수 | Backend | - |
| 4 | 가변 폭 링 분할 개선 | Backend | - |
| 5 | 통합 테스트 및 버그 수정 | Backend | - |

### Week 2: Phase B (4일)

| Day | 작업 | 담당 | 비고 |
|-----|------|------|------|
| 6 | 자기 참조 균일성 분석 + API | Backend | Frontend: 테이블 UI |
| 7 | /recompute API + 캐싱 | Backend | Frontend: 파라미터 컨트롤 |
| 8 | /compare API | Backend | Frontend: 비교 화면 |
| 9 | 배경색 기반 중심 검출 Fallback | Backend | - |

### Week 3: Phase C + 검증 (3일)

| Day | 작업 | 담당 | 비고 |
|-----|------|------|------|
| 10 | /batch 요약 통계 | Backend | Frontend: 요약 화면 |
| 11 | 사용자 이미지로 비교 검증 | All | - |
| 12 | 문서화 및 배포 준비 | All | - |

---

## 🔗 Frontend 작업 항목 (다른 작업자용)

### UI 컴포넌트 목록

| 컴포넌트 | 설명 | Backend API | 우선순위 |
|----------|------|-------------|----------|
| **균일성 분석 테이블** | Zone별 ΔE vs Ref, 편차 설명 표시 | `GET /inspect` → `uniformity_analysis` | High |
| **파라미터 조정 패널** | 슬라이더/입력 필드 + Recompute 버튼 | `POST /recompute` | High |
| **Lot 비교 화면** | Reference + 다중 Test 업로드, Diff 그래프/테이블 | `POST /compare` | Medium |
| **진단 메시지 패널** | Diagnostics, Warnings, Suggestions 표시 | `GET /inspect` → `diagnostics`, `suggestions` | High |
| **배치 요약 대시보드** | OK/NG 비율, Zone별 통계, 차트 | `POST /batch` → `summary` | Medium |
| **Print Area 오버레이** | r_inner, r_outer 원 표시 | `GET /inspect` → `print_boundaries` | Low |

### Frontend API 연동 가이드

**1. /inspect 응답 구조**
```typescript
interface InspectResponse {
  judgment?: string;  // "OK" | "NG" (run_judgment=true일 때만)
  zones: Zone[];
  profile: RadialProfile;
  print_boundaries?: {
    r_inner: number;
    r_outer: number;
    method: string;
    confidence: number;
  };
  uniformity_analysis?: UniformityAnalysis;
  diagnostics: string[];
  warnings: string[];
  suggestions: string[];
  image_id: string;  // recompute용
}
```

**2. /recompute 파라미터**
```typescript
interface RecomputeRequest {
  image_id: string;
  sku: string;
  params: {
    smoothing_window?: number;      // 5-51 (odd)
    min_gradient?: number;           // 0-5
    min_delta_e?: number;            // 0-10
    expected_zones?: number;         // 1-5
    auto_crop_print_area?: boolean;
    chroma_threshold?: number;       // 0-10
    uniformity_reference?: "overall_mean" | "outer_zone" | "custom";
  };
}
```

**3. /compare 응답 구조**
```typescript
interface CompareResponse {
  reference: {
    filename: string;
    zones: Zone[];
  };
  tests: TestResult[];
  batch_summary: {
    mean_delta_e_per_zone: Record<string, number>;
    max_delta_e_per_zone: Record<string, number>;
    stability_score: number;
    outliers: string[];
  };
}
```

---

## 🏗️ 하이브리드 아키텍처 (통합 전략)

### 모듈 구조

```
src/core/
├── radial_profiler.py        ✅ 현재 유지 (cv2.warpPolar, 빠름)
├── zone_segmenter.py          ✅ 현재 유지 (Ring 분할, Gradient/ΔE)
└── sector_segmenter.py        🆕 신규 (AI 템플릿 기반, 각도별 분할)

src/analysis/
├── profile_analyzer.py        ✅ 현재 유지 (경계 후보 검출)
└── print_area_detector.py     🆕 신규 (r_inner/outer 자동 검출)

src/utils/
├── background_mask.py         🆕 신규 (2단계 배경 마스킹)
├── illumination.py            🆕 신규 (조명 편차 보정)
└── color_delta.py             ✅ 현재 유지 (CIEDE2000)
```

### 통합 전략

**1. Radial Analysis (현재 방식 유지)**
- `cv2.warpPolar` 사용 → 회전 불변성 + 고속 처리
- Savgol 스무딩 → 노이즈 제거
- ProfileAnalyzer → 경계 후보 검출

**2. Sector Analysis (AI 템플릿 채택)**
- 원본 이미지에서 직접 각도 계산
- Ring × Sector 2D 그리드 생성
- Heatmap 데이터 출력

**3. r_inner/outer 검출 (하이브리드)**
- **Primary**: Chroma 기반 (PHASE7) - 정밀도 우선
- **Fallback**: Counts 기반 (AI 템플릿) - 단순성, 속도

**4. 배경 마스킹 (PHASE7 방식)**
- Stage 1: ROI 기반 배경 샘플링
- Stage 2: Otsu + 색상 거리 이중 마스킹

**5. 균일성 분석 (템플릿 방식)**
- 전체 평균 Lab 계산
- 각 Ring/Sector별 ΔE (vs 전체 평균)

---

## 🎯 Priority 0: Ring × Sector 2D 분할 구현

### 목적
- 각도별 색상 불균일 검출 (방사형 분석의 한계 극복)
- Heatmap 시각화로 품질 문제 직관적 파악
- ANALYSIS_IMPROVEMENTS의 핵심 기능

### 구현 위치
```
src/core/sector_segmenter.py (신규 파일)
```

### AI 템플릿 기반 구현

```python
"""
Sector Segmenter Module

Ring × Sector 2D 그리드로 렌즈를 분할하고, 각 셀의 Lab 평균 및 ΔE를 계산합니다.
AI 템플릿 분석 방식을 기반으로 구현되었습니다.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SectorCell:
    """단일 (Ring, Sector) 셀 데이터"""
    ring: int              # 1-based
    sector: int            # 1-based
    r_start: float         # 정규화 반경
    r_end: float
    theta_start: float     # 각도 (도)
    theta_end: float
    mean_L: float
    mean_a: float
    mean_b: float
    std_L: float
    std_a: float
    std_b: float
    pixel_count: int
    coverage: float        # 픽셀 비율 (0~1)


@dataclass
class SectorSegmentationResult:
    """Sector 분할 결과"""
    n_rings: int
    n_sectors: int
    r_inner: float
    r_outer: float
    cells: List[SectorCell]
    overall_lab: Tuple[float, float, float]
    heatmap_delta_e: np.ndarray  # shape: (n_rings, n_sectors)


class SectorSegmenter:
    """Ring × Sector 2D 분할 (AI 템플릿 기반)"""

    def __init__(self, n_rings: int = 3, n_sectors: int = 12):
        """
        Args:
            n_rings: Ring 개수 (기본 3)
            n_sectors: Sector 개수 (기본 12 = 30도씩)
        """
        self.n_rings = n_rings
        self.n_sectors = n_sectors

    def segment(
        self,
        image_lab: np.ndarray,
        mask: np.ndarray,
        cx: float,
        cy: float,
        r_inner: float,
        r_outer: float,
        lens_radius: float
    ) -> SectorSegmentationResult:
        """
        Ring × Sector 2D 분할

        Args:
            image_lab: Lab 색공간 이미지
            mask: 배경 마스크 (1=렌즈, 0=배경)
            cx, cy: 렌즈 중심 (픽셀)
            r_inner: 인쇄 시작 반경 (픽셀)
            r_outer: 인쇄 끝 반경 (픽셀)
            lens_radius: 렌즈 전체 반경 (정규화용)

        Returns:
            SectorSegmentationResult
        """
        h, w = image_lab.shape[:2]
        yy, xx = np.indices((h, w))

        # 반경 및 각도 계산
        rr = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        theta = (np.degrees(np.arctan2(yy - cy, xx - cx)) + 360) % 360

        # Ring 경계 (균등 분할)
        ring_bounds = np.linspace(r_inner, r_outer, self.n_rings + 1)

        cells: List[SectorCell] = []

        # 각 (Ring, Sector) 셀 처리
        for r in range(self.n_rings):
            for s in range(self.n_sectors):
                r0, r1 = ring_bounds[r], ring_bounds[r + 1]
                theta0 = 360 / self.n_sectors * s
                theta1 = 360 / self.n_sectors * (s + 1)

                # 셀 마스크
                cell_mask = (
                    (mask == 1) &
                    (rr >= r0) & (rr < r1) &
                    (theta >= theta0) & (theta < theta1)
                )

                pixels = image_lab[cell_mask]
                pixel_count = len(pixels)

                if pixel_count > 0:
                    mean_lab = np.mean(pixels, axis=0)
                    std_lab = np.std(pixels, axis=0)
                else:
                    mean_lab = np.array([0, 0, 0])
                    std_lab = np.array([0, 0, 0])

                # Coverage 계산 (이론적 픽셀 수 대비 실제 픽셀 수)
                theoretical_area = np.pi * (r1**2 - r0**2) / self.n_sectors
                coverage = pixel_count / max(theoretical_area, 1)

                cells.append(SectorCell(
                    ring=r + 1,
                    sector=s + 1,
                    r_start=r1 / lens_radius,  # 정규화
                    r_end=r0 / lens_radius,
                    theta_start=theta0,
                    theta_end=theta1,
                    mean_L=float(mean_lab[0]),
                    mean_a=float(mean_lab[1]),
                    mean_b=float(mean_lab[2]),
                    std_L=float(std_lab[0]),
                    std_a=float(std_lab[1]),
                    std_b=float(std_lab[2]),
                    pixel_count=pixel_count,
                    coverage=min(coverage, 1.0)
                ))

        # 전체 평균 Lab 계산
        all_vals = np.array([[c.mean_L, c.mean_a, c.mean_b] for c in cells if c.pixel_count > 0])
        overall_lab = tuple(np.mean(all_vals, axis=0))

        # Heatmap ΔE 계산
        heatmap = self._compute_heatmap(cells, overall_lab)

        return SectorSegmentationResult(
            n_rings=self.n_rings,
            n_sectors=self.n_sectors,
            r_inner=r_inner / lens_radius,
            r_outer=r_outer / lens_radius,
            cells=cells,
            overall_lab=overall_lab,
            heatmap_delta_e=heatmap
        )

    def _compute_heatmap(self, cells: List[SectorCell], overall_lab: Tuple[float, float, float]) -> np.ndarray:
        """Heatmap 생성 (Ring × Sector ΔE)"""
        from src.utils.color_delta import delta_e_cie1976  # 빠른 계산용

        heatmap = np.zeros((self.n_rings, self.n_sectors))

        for cell in cells:
            if cell.pixel_count > 0:
                cell_lab = (cell.mean_L, cell.mean_a, cell.mean_b)
                de = delta_e_cie1976(overall_lab, cell_lab)
                heatmap[cell.ring - 1, cell.sector - 1] = de

        return heatmap
```

### API Response 형식

```json
{
  "sector_analysis": {
    "n_rings": 3,
    "n_sectors": 12,
    "r_inner": 0.298,
    "r_outer": 0.968,
    "overall_lab": [58.90, 3.79, 22.32],
    "cells": [
      {
        "ring": 1,
        "sector": 1,
        "mean_L": 75.03,
        "mean_a": 3.02,
        "mean_b": 17.25,
        "delta_e_vs_mean": 16.92,
        "coverage": 0.95
      },
      ...
    ],
    "heatmap_delta_e": [
      [16.9, 17.2, 16.5, ...],  // Ring 1
      [11.0, 10.8, 11.3, ...],  // Ring 2
      [3.8, 3.9, 3.7, ...]      // Ring 3
    ]
  }
}
```

### Frontend 연동 (참고용)

**Heatmap 렌더링**:
```javascript
// Chart.js Heatmap
const ctx = document.getElementById('heatmap').getContext('2d');
new Chart(ctx, {
  type: 'matrix',
  data: {
    datasets: [{
      data: sector_analysis.heatmap_delta_e.flat().map((value, i) => ({
        x: i % sector_analysis.n_sectors,
        y: Math.floor(i / sector_analysis.n_sectors),
        v: value
      })),
      backgroundColor(context) {
        const value = context.dataset.data[context.dataIndex].v;
        const alpha = value / 20;  // 0~20 ΔE 범위
        return `rgba(255, ${255 - value * 12}, 0, ${alpha})`;
      }
    }]
  }
});
```

### 테스트

```python
# tests/test_sector_segmenter.py
def test_sector_segmentation():
    segmenter = SectorSegmenter(n_rings=3, n_sectors=12)

    # Mock 데이터
    image_lab = np.random.rand(500, 500, 3) * 100
    mask = np.ones((500, 500), dtype=np.uint8)

    result = segmenter.segment(
        image_lab, mask,
        cx=250, cy=250,
        r_inner=119, r_outer=387,
        lens_radius=400
    )

    assert len(result.cells) == 3 * 12  # 36 cells
    assert result.heatmap_delta_e.shape == (3, 12)
```

---

## 📋 AI 템플릿에서 가져올 코드

### ✅ 즉시 사용 가능

#### 1. Sector 분할 로직
```python
# 템플릿: segment_pixels 함수
theta = (np.degrees(np.arctan2(yy - cy, xx - cx)) + 360) % 360

cell_mask = (
    (mask == 1) &
    (rr >= r0) & (rr < r1) &
    (theta >= theta0) & (theta < theta1)
)
```

#### 2. counts 기반 r_inner/outer (Fallback)
```python
# 템플릿: detect_radii 함수
def detect_radii_by_counts(mask, cx, cy):
    """픽셀 카운트 기반 반경 검출 (빠르고 단순)"""
    h, w = mask.shape
    yy, xx = np.indices((h, w))
    rr = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    max_r = int(rr.max())
    counts = np.zeros(max_r)

    for r in range(max_r):
        ring_mask = (rr >= r) & (rr < r + 1)
        counts[r] = np.sum(mask[ring_mask])

    # 인쇄 시작/끝
    edges = np.where(counts > 10)[0]
    if len(edges) == 0:
        return None, None

    inner = edges[0]
    outer = edges[-1]

    return inner, outer
```

#### 3. 균등 Ring 분할
```python
# 템플릿: np.linspace 방식
ring_bounds = np.linspace(r_inner, r_outer, n_rings + 1)
```

### ⚠️ 개선 후 사용

#### 1. 배경 마스킹
템플릿의 단순 RGB 거리 방식 → PHASE7 2단계 방식으로 강화

#### 2. Lab 변환
템플릿의 `skimage.rgb2lab` → 현재 `cv2.cvtColor` 유지 (더 빠름)

#### 3. ΔE 계산
템플릿의 CIE76 → 현재 CIEDE2000 유지 (Heatmap은 CIE76 사용 가능)

---

## 📚 참고 자료

### 관련 문서
- `docs/planning/ANALYSIS_IMPROVEMENTS.md`: Ring×Sector 분할, Ink Mask (기존 계획)
- `docs/planning/ANALYSIS_UI_DEVELOPMENT_PLAN.md`: UI 개발 계획 (710줄)
- `docs/USER_GUIDE.md`: 사용자 가이드
- `docs/WEB_UI.md`: 웹 UI 사용법

### 알고리즘 참고
- **Gray World Assumption**: [Wikipedia](https://en.wikipedia.org/wiki/Color_constancy)
- **CIEDE2000**: `src/utils/color_delta.py`
- **Savitzky-Golay Filter**: `scipy.signal.savgol_filter`
- **Peak Detection**: `scipy.signal.find_peaks`

---

## ✅ 체크리스트

### Backend 구현 완료 체크

- [ ] **Ring × Sector 2D 분할** (Priority 0) ⭐⭐⭐
- [ ] r_inner, r_outer 자동 검출 (chroma + counts fallback)
- [ ] 2단계 배경 마스킹
- [ ] 자기 참조 균일성 분석
- [ ] 조명 편차 보정 (Gray World / White Patch)
- [ ] 에러 처리 및 제안 메시지
- [ ] 표준편차/사분위수 지표
- [ ] 가변 폭 링 분할 개선
- [ ] /recompute API
- [ ] /compare API
- [ ] /batch 요약 통계
- [ ] 배경색 기반 중심 검출 Fallback
- [ ] 균등 분할 우선 옵션

### 테스트 완료 체크

- [ ] 단위 테스트 (각 모듈별)
- [ ] 통합 테스트 (파이프라인)
- [ ] API 테스트 (모든 endpoint)
- [ ] 사용자 이미지로 검증 (수동 분석 결과와 비교)

### 문서화 완료 체크

- [ ] 각 모듈 docstring 작성
- [ ] API 문서 업데이트 (README.md)
- [ ] Frontend 연동 가이드 작성
- [ ] CHANGELOG.md 업데이트

---

**작성자**: Claude Code
**최종 업데이트**: 2025-12-12
**관련 Issue**: Phase 7 Core Algorithm Improvements
