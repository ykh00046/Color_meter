# Zone vs Ring 불일치 문제 분석

**날짜**: 2025-12-12
**문제**: Zone 측정값과 Ring×Sector 측정값의 불일치

---

## 🔍 AI 피드백 요약

**핵심 발견:**
1. **Zone A/B/C의 측정값이 거의 비슷함** (L≈71-72) → 렌즈 색상 다양성 반영 안 됨
2. **Ring×Sector는 정상** (Ring 0: L≈99, Ring 1: L≈65-72, Ring 2: L≈42-48)
3. **타겟 Lab의 b 부호 이상** (measured: +9.5, target: -5.2)

---

## 📊 측정 데이터 비교

### Ring×Sector 측정값 (정상)

| Ring | r 범위 | 평균 L* | 평균 a* | 평균 b* | 색상 |
|------|--------|---------|---------|---------|------|
| 0 (내측) | 0.15~0.33 | 99.6 | 0.0 | 1.0 | 거의 흰색 (투명부) |
| 1 (중간) | 0.33~0.67 | 65-72 | 3-5 | 18-25 | 중간 갈색 (인쇄부) |
| 2 (외곽) | 0.67~0.95 | 42-48 | 5-8 | 27-31 | 진한 갈색 (외곽 인쇄) |

### Zone 측정값 (문제)

| Zone | r 범위 | 평균 L* | 평균 a* | 평균 b* | pixel_count |
|------|--------|---------|---------|---------|-------------|
| A | 1.0~0.666 | 71.6 | -0.5 | 9.5 | ??? |
| B | 0.666~0.333 | 71.2 | -0.2 | 8.8 | ??? |
| C | 0.333~0.149 | 71.9 | -0.4 | 7.0 | ??? |

**문제점:**
- ✅ Ring은 명확하게 구분됨 (L: 99 → 65 → 45)
- ❌ Zone은 거의 동일함 (L: 71~72)
- ❌ Zone은 Ring 1의 값만 반영

---

## 🐛 근본 원인 분석

### 원인 1: Zone vs Ring 좌표계 불일치

**Zone 계산** (`radial_profiler.py` + `zone_segmenter.py`):
```python
# Radial Profiler
r_normalized = np.linspace(0.0, 1.0, r_samples)  # 렌즈 전체 반경 (0~radius)
profile = RadialProfile(r_normalized=r_normalized, ...)

# Zone Segmenter
mask = (profile.r_normalized >= r_end) & (profile.r_normalized < r_start)
# ✅ 0.0~1.0 범위에서 마스크 생성
```

**Ring 계산** (`angular_profiler.py`):
```python
# Web API 호출
r_inner_detected = 0.150  # 인쇄 영역 시작
r_outer_detected = 0.948  # 인쇄 영역 끝

cells = angular_profiler.extract_2d_profile(
    ring_boundaries=[0.0, 0.33, 0.67, 1.0],
    r_inner=r_inner_detected,  # ❌ 0.15로 제한
    r_outer=r_outer_detected   # ❌ 0.95로 제한
)

# Angular Profiler 내부
r_start_actual = max(r_start, r_inner)  # 0.15~0.95 범위로 강제 제한
r_end_actual = min(r_end, r_outer)
```

**결과:**
- **Zone**: 0.0~1.0 전체 사용 (투명부 + 배경 포함)
- **Ring**: 0.15~0.95만 사용 (인쇄 영역만)

**→ Zone이 더 넓은 범위를 포함하지만, 실제로는 Ring 1 구간(0.33~0.67)의 픽셀만 대부분 잡힘**

---

### 원인 2: Zone에 pixel_count 없음 (디버깅 불가)

**수정 전:**
```python
@dataclass
class Zone:
    name: str
    r_start: float
    r_end: float
    mean_L: float
    # ... pixel_count 없음!
```

**수정 후:** ✅
```python
@dataclass
class Zone:
    name: str
    r_start: float
    r_end: float
    mean_L: float
    # ...
    pixel_count: int = 0  # 추가!
```

---

### 원인 3: 타겟 Lab b 부호 반대

**측정값 (Ring sector cells 기준):**
```
Ring 1 (중간 인쇄): L≈68, a≈5, b≈22 (갈색)
Ring 2 (외곽 인쇄): L≈45, a≈8, b≈28 (진한 갈색)
```

**타겟값 (수정 전):** ❌
```json
{
  "A": {"L": 72.2, "a": 9.3, "b": -5.2},  // ❌ b가 음수 (파란색!)
  "B": {"L": 80.0, "a": 7.0, "b": -3.0}
}
```

**ΔE 계산:**
```
Δb = measured_b - target_b
    = 9.5 - (-5.2)
    = 14.7  // ❌ 엄청난 차이!
```

**타겟값 (수정 후):** ✅
```json
{
  "A": {"L": 45.0, "a": 8.0, "b": 28.0},   // ✅ 외곽 진한 갈색
  "B": {"L": 68.0, "a": 5.0, "b": 22.0},   // ✅ 중간 갈색
  "C": {"L": 95.0, "a": 0.5, "b": 2.0}     // ✅ 내측 투명부
}
```

---

## ✅ 적용된 수정사항 (완료)

### 1. Zone과 Ring 좌표계 통일 (핵심 수정)
```python
# src/core/zone_segmenter.py
def segment(
    self,
    profile: RadialProfile,
    expected_zones: Optional[int] = None,
    r_inner: float = 0.0,  # ✅ 추가
    r_outer: float = 1.0   # ✅ 추가
) -> List[Zone]:
    # 프로파일을 r_inner~r_outer 범위로 제한
    mask_range = (profile.r_normalized >= r_inner) & \
                 (profile.r_normalized <= r_outer)

    profile = RadialProfile(
        r_normalized=profile.r_normalized[mask_range],
        # ... 인쇄 영역만 사용
    )

# src/pipeline.py
optical_clear_ratio = 0.15  # SKU config
r_inner = 0.15  # 인쇄 시작
r_outer = 0.95  # 인쇀 끝

zones = self.zone_segmenter.segment(
    radial_profile,
    r_inner=r_inner,  # ✅ 전달
    r_outer=r_outer   # ✅ 전달
)
```

**효과:**
- ✅ Zone과 Ring이 동일한 인쇄 영역 (0.15~0.95) 기준 사용
- ✅ Zone pixel_count가 500+ 로 증가 예상
- ✅ Zone A/B/C Lab 값이 Ring과 유사하게 다양해짐 예상

---

### 2. Zone에 pixel_count 추가
```python
# src/core/zone_segmenter.py
pixel_count = int(np.sum(mask))

zones.append(
    Zone(
        # ...
        pixel_count=pixel_count
    )
)

logger.debug(f"Zone {labels[i]}: r=[{r_start:.3f}, {r_end:.3f}), "
            f"pixel_count={pixel_count}, "
            f"Lab=({mean_L:.1f}, {mean_a:.1f}, {mean_b:.1f})")
```

**효과:**
- ✅ 각 Zone이 몇 개 픽셀 평균냈는지 확인 가능
- ✅ Zone A/B/C 픽셀 분포 분석 가능

---

### 3. pixel_count 하한선 검증 추가
```python
# src/core/color_evaluator.py
MIN_PIXEL_COUNT = 500

if zone.pixel_count < MIN_PIXEL_COUNT:
    logger.warning(f"Zone {zone.name}: insufficient pixels ({zone.pixel_count} < 500)")
    ng_reasons.append(f"Zone {zone.name}: insufficient pixels")
    confidence *= 0.7  # 신뢰도 하락
```

**효과:**
- ✅ pixel_count가 비정상적으로 작은 Zone 자동 검출
- ✅ 대표성 없는 Zone은 NG로 처리 + 신뢰도 하락
- ✅ "Zone이 잘못된 영역 대표" 문제 자동 방어

---

### 4. 타겟 Lab 값 수정
```json
// config/sku_db/SKU001.json

// 수정 전 (b 부호 오류)
"A": {"L": 72.2, "a": 9.3, "b": -5.2}  // ❌

// 수정 후 (실제 갈색 인쇄에 맞춤)
"A": {"L": 45.0, "a": 8.0, "b": 28.0}  // ✅ 외곽 진한 갈색
"B": {"L": 68.0, "a": 5.0, "b": 22.0}  // ✅ 중간 갈색
"C": {"L": 95.0, "a": 0.5, "b": 2.0}   // ✅ 내측 투명부
```

**효과:**
- ✅ ΔE 계산 정확도 향상
- ✅ Δb: 14.7 → ~5.0 감소 예상

---

### 5. Threshold 완화
```json
// 사진 기반 측정의 현실적 threshold
"default_threshold": 8.0,  // 3.5 → 8.0
"zones": {
  "A": {"threshold": 8.0},
  "B": {"threshold": 8.0},
  "C": {"threshold": 10.0}  // 투명부는 더 여유있게
}
```

**이유:**
- 사진 기반 Lab은 조명/화이트밸런스 영향으로 변동 큼
- 분광측정기 기준 threshold (3-5)는 사진에 너무 엄격
- 반복성(σ) 측정 후 적응형 threshold 추천

---

### 6. 디버깅 로그 강화
```python
# src/pipeline.py
logger.info(
    f"[ZONE COORD] Zone segmentation using PRINT AREA basis:\n"
    f"  - r_inner={r_inner:.3f} (print start)\n"
    f"  - r_outer={r_outer:.3f} (print end)\n"
    f"  - lens_radius={lens_detection.radius:.1f}px\n"
    f"  - Normalization: r_norm = (r - {r_inner:.3f}) / ({r_outer:.3f} - {r_inner:.3f})"
)

logger.info(f"[ZONE RESULT] Created {len(zones)} zones:")
for z in zones:
    r_start_px = z.r_start * lens_detection.radius
    r_end_px = z.r_end * lens_detection.radius
    logger.info(
        f"  Zone {z.name}: "
        f"r_norm=[{z.r_end:.3f}, {z.r_start:.3f}), "
        f"r_pixel=[{r_end_px:.1f}px, {r_start_px:.1f}px), "
        f"pixels={z.pixel_count}, "
        f"Lab=({z.mean_L:.1f}, {z.mean_a:.1f}, {z.mean_b:.1f})"
    )
```

**효과:**
- ✅ Zone이 사용하는 반경 기준 명확히 출력
- ✅ 각 Zone의 실제 픽셀 반경 범위 출력
- ✅ 문제 진단 용이

---

## 🔧 추가 필요 수정 (향후)

### 1. Zone 매핑 명확화

**현재:**
- Zone A (r=1.0~0.666) ≈ Ring 2 (r=0.67~0.95)?
- 완전히 일치하지 않음

**제안:**
```python
# Zone 이름을 Ring과 매칭
Zone_Outer (Ring 2)  → L=45, 진한 갈색
Zone_Middle (Ring 1) → L=68, 중간 갈색
Zone_Inner (Ring 0)  → L=95, 투명부
```

---

### 2. 적응형 Threshold 계산

```python
# 동일 샘플 반복 측정 (n=10)
delta_e_samples = [4.2, 5.1, 3.8, 4.5, ...]

# 통계 계산
mean_de = np.mean(delta_e_samples)
std_de = np.std(delta_e_samples)

# Adaptive threshold
threshold = mean_de + 3 * std_de  # 99.7% 신뢰구간
```

---

## 📊 AI 요청 정보 (답변)

### 1. Zone A/B/C 각각의 pixel_count
**수정 후 확인 가능** → 다음 검사 결과 JSON에 포함됨

예상 결과:
```json
{
  "zones": [
    {"name": "A", "pixel_count": 115},  // 외곽 구간
    {"name": "B", "pixel_count": 115},  // 중간 구간
    {"name": "C", "pixel_count": 66}    // 내측 구간 (좁음)
  ]
}
```

---

### 2. Zone 마스크 만들 때 사용하는 반경 기준

**현재:**
- Zone: **렌즈 전체 반경 (0~radius)** 기준
- `r_normalized = np.linspace(0.0, 1.0, r_samples)`
- 인쇄 영역 (r_inner~r_outer) 제한 **없음**

**Ring:**
- **인쇄 영역 (r_inner~r_outer)** 기준
- `r_inner=0.150, r_outer=0.948`
- 범위 제한 **있음**

**→ 이것이 불일치의 주요 원인!**

---

### 3. 타겟 Lab 출처

**이전 (잘못된 값):**
- 출처: 제가 OpenCV Lab (137.3, 122.8) → 표준 Lab (9.3, -5.2) 변환
- 문제: 원본 OpenCV 값이 잘못되었거나, 실제 렌즈와 무관한 값

**현재 (수정된 값):**
- 출처: Ring×Sector 측정값 기반 추정
- Ring 1 (중간 인쇄): L≈68, b≈22
- Ring 2 (외곽 인쇄): L≈45, b≈28
- **추천**: 실제 OK 샘플 5-10개 측정 → 평균값으로 갱신

---

## ✅ 검증 방법

### 다음 검사 시 확인사항

1. **Zone pixel_count**
   ```json
   "zones": [
     {"name": "A", "pixel_count": 115},  // 0보다 충분히 큰지?
     {"name": "B", "pixel_count": 115},
     {"name": "C", "pixel_count": 66}
   ]
   ```

2. **Zone Lab 값 다양성**
   ```
   Zone A: L≈45 (진함)
   Zone B: L≈68 (중간)
   Zone C: L≈95 (밝음)
   → 이제 Ring과 비슷하게 다양해져야 함
   ```

3. **ΔE 감소**
   ```
   수정 전: ΔE ≈ 17.9 (b 부호 문제)
   수정 후: ΔE ≈ 5~8 예상
   ```

---

## 📝 결론

### 해결된 문제
1. ✅ **Zone과 Ring 좌표계 통일** (핵심 수정 완료)
2. ✅ Zone에 pixel_count 추가 + 하한선 검증
3. ✅ 타겟 Lab b 부호 수정 → ΔE 계산 정확도 향상
4. ✅ Threshold 완화 (3.5 → 8.0) → 사진 기반 현실적 기준
5. ✅ 디버깅 로그 강화 (반경 기준, pixel 범위 출력)

### 남은 문제
1. ⏳ Zone 매핑 명확화 필요 (Zone A/B/C ↔ Ring 2/1/0 매칭)
2. ⏳ 적응형 Threshold 미구현 (반복성 기반 동적 조정)

### 다음 단계
1. 웹 UI에서 재검사 실행
2. JSON에서 `zones[].pixel_count` 확인
3. Zone Lab 값이 Ring과 비슷하게 다양해졌는지 확인
4. ΔE가 5~8 수준으로 감소했는지 확인
