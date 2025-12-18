# Ink Mask 통합 계획

## 🎯 목적

도트 인쇄의 "잉크 픽셀만" 평균내서 희석 문제 해결

## 📋 AI 코드 핵심 아이디어

### 1. ink_mask 생성
```python
# HSV 기반: 채도 높고 명도 낮은 영역 = 잉크
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)
ink_mask = ((S > 40) & (V < 200)).astype(np.uint8) * 255
```

### 2. mean_all vs mean_ink
```python
mean_all = zone 전체 평균  # 현재 방식
mean_ink = zone ∩ ink_mask 평균  # 도트만

# ΔE 계산 시 mean_ink 우선 사용
if mean_ink is not None:
    delta_e = calculate(mean_ink, target)
else:
    delta_e = calculate(mean_all, target)
```

### 3. ink_pixel_ratio
```python
ink_pixel_ratio = n_ink / n_all

# 너무 낮으면 (예: <0.03) ink_mask 임계값 조정 필요
```

---

## 🔧 통합 방법

### Option 1: BackgroundMasker 확장 (권장)

**File**: `src/core/background_masker.py`

```python
class BackgroundMasker:
    def create_ink_mask(
        self,
        image_lab: np.ndarray,
        center_x: float,
        center_y: float,
        radius: float,
        method: str = "sat_val"
    ) -> InkMaskResult:
        """
        도트 인쇄의 잉크 픽셀만 분리.

        Args:
            method: "sat_val" (HSV 기반) or "gray_otsu" (그레이 기반)

        Returns:
            InkMaskResult with mask and statistics
        """
        # AI 코드 적용
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        ink = ((S > self.config.ink_saturation_min) &
               (V < self.config.ink_value_max)).astype(np.uint8) * 255

        return InkMaskResult(mask=ink, ink_ratio=...)
```

### Option 2: Zone 계산에 ink_mask 통합

**File**: `src/core/zone_segmenter.py`

```python
@dataclass
class Zone:
    # ... 기존 필드 ...
    mean_L_ink: Optional[float] = None  # 잉크만 평균
    mean_a_ink: Optional[float] = None
    mean_b_ink: Optional[float] = None
    ink_pixel_ratio: float = 0.0
```

### Option 3: ColorEvaluator에서 선택적 사용

**File**: `src/core/color_evaluator.py`

```python
def evaluate(
    self,
    zones: List[Zone],
    sku: str,
    sku_config: dict,
    use_ink_mask: bool = False  # 옵션으로 제공
):
    if use_ink_mask:
        # mean_ink 우선 사용
        measured_lab = (zone.mean_L_ink, zone.mean_a_ink, zone.mean_b_ink)
    else:
        measured_lab = (zone.mean_L, zone.mean_a, zone.mean_b)
```

---

## 📊 검증 방법

### 1. mean_all vs mean_ink 비교 로그
```python
logger.info(
    f"Zone {zone.name}:\n"
    f"  mean_all: Lab=({mean_L:.1f}, {mean_a:.1f}, {mean_b:.1f})\n"
    f"  mean_ink: Lab=({mean_L_ink:.1f}, {mean_a_ink:.1f}, {mean_b_ink:.1f})\n"
    f"  ink_ratio: {ink_ratio:.2%}\n"
    f"  pixels: {n_all} (ink: {n_ink})"
)
```

**기대 결과:**
```
Zone A:
  mean_all: Lab=(71.0, -0.4, 9.7)   ← 희석됨
  mean_ink: Lab=(45.0, 8.0, 28.0)   ← 진짜 잉크색
  ink_ratio: 35%
  pixels: 5234 (ink: 1832)
```

### 2. ink_mask 시각화
```python
# AI 코드 참고
cv2.imwrite("debug_ink_mask.png", ink_mask)
overlay = draw_mask_overlay(image, ink_mask, (0, 255, 0), alpha=0.3)
cv2.imwrite("debug_ink_overlay.png", overlay)
```

---

## ⚠️ 주의사항

### 1. 임계값 조정 필요
```python
# HSV 기준값 (제품별 조정 필요)
S_min = 40   # 채도 하한 (낮으면 배경 포함, 높으면 잉크 누락)
V_max = 200  # 명도 상한 (낮으면 하이라이트 제거, 높으면 하이라이트 포함)
```

### 2. 조명 보정 필수
- ink_mask는 조명에 민감
- 조명 보정 후 적용 권장

### 3. 도트 패턴
- 너무 작은 도트는 morphology로 날아갈 수 있음
- opening kernel size 조정 필요

---

## 🎯 통합 우선순위

**현재 수정(r_inner/r_outer) 검증 후:**

1. ✅ **Zone Lab이 Ring과 유사** → ink_mask 불필요
2. ❌ **여전히 L=71 근처** → ink_mask 즉시 도입

**ink_mask 도입 시 순서:**
1. BackgroundMasker에 create_ink_mask() 추가
2. Zone에 mean_ink 필드 추가
3. ColorEvaluator에서 mean_ink 우선 사용
4. 디버깅 로그 + 시각화 추가

---

## 📝 참고

- AI 제공 코드: 전체 파이프라인 템플릿
- 핵심: `build_ink_mask()`, `compute_zone_results()`
- 우리 구조에 맞게 분해해서 통합 필요
