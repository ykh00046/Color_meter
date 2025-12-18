# 2D Zone Analysis 테스트 가이드

## ✅ 통합 완료

AI 템플릿 코드가 독립 모듈로 통합되었습니다!

### 생성된 파일
- `src/core/zone_analyzer_2d.py` - AI 템플릿 코드
- `src/web/app.py` - 웹 UI 옵션 추가

---

## 🚀 테스트 방법

### 1. 서버 재시작
```bash
# Ctrl+C로 기존 서버 종료
cd C:\X\Color_total\Color_meter
python -m src.web.app
```

### 2. 웹 UI 접속
```
http://localhost:8000 또는 http://localhost:8001
```

### 3. 이미지 검사 실행
- 기본적으로 **2D 분석이 활성화**되어 있습니다
- `use_2d_analysis=True` (기본값)

---

## 📊 예상 결과

### Before (1D RadialProfile 방식)
```json
{
  "zone_results": [
    {"zone_name": "A", "measured_lab": [71.57, -0.43, 9.68], "pixel_count": 115},
    {"zone_name": "B", "measured_lab": [71.20, -0.23, 8.84], "pixel_count": 116},
    {"zone_name": "C", "measured_lab": [71.97, -0.43, 6.95], "pixel_count": 64}
  ]
}
```

### After (2D 이미지 직접 방식 - AI 템플릿)
```json
{
  "zone_results": [
    {
      "zone_name": "A",
      "measured_lab": [45.0, 8.0, 28.0],  // ← Ring 2와 유사
      "pixel_count": 5234,                // ← 실제 픽셀 수
      "pixel_count_ink": 1832,            // ← 잉크만
      "ink_pixel_ratio": 0.35,            // ← 35%
      "measured_lab_ink": [42.5, 8.5, 29.1],  // ← 잉크 픽셀만 평균
      "delta_e_basis": "mean_ink"         // ← 어느 평균 사용했는지
    },
    {
      "zone_name": "B",
      "measured_lab": [68.0, 5.0, 22.0],
      "pixel_count": 6128
    },
    {
      "zone_name": "C",
      "measured_lab": [95.0, 0.5, 2.0],
      "pixel_count": 3421
    }
  ]
}
```

**핵심 변화:**
- ✅ Zone Lab 값이 Ring과 유사 (A: 45, B: 68, C: 95)
- ✅ pixel_count가 정상 범위 (3000~6000)
- ✅ ink_pixel_ratio로 도트 비율 확인
- ✅ mean_ink로 실제 잉크색 측정

---

## 🔍 로그 확인 사항

검사 실행 시 다음 로그가 출력되어야 합니다:

```
[INSPECT] Using 2D zone analysis (AI template)
[2D ZONE ANALYSIS] Starting...
[2D ZONE ANALYSIS] Building ink mask...
[2D ZONE ANALYSIS] Estimating print boundaries...
[PRINT BOUNDARIES] inner=78.0px (0.150), outer=494.0px (0.950), confidence=0.85
[2D ZONE ANALYSIS] Building zone masks...
[2D ZONE ANALYSIS] Computing zone results...
  Zone C: pixels_all=3421, pixels_ink=1205, ink_ratio=35.22%, Lab_all=[94.8, 0.6, 2.3], Lab_ink=[95.2, 0.4, 1.8], ΔE=0.31 (basis=mean_ink)
  Zone B: pixels_all=6128, pixels_ink=2145, ink_ratio=35.01%, Lab_all=[67.5, 5.2, 21.8], Lab_ink=[68.2, 5.0, 22.1], ΔE=0.23 (basis=mean_ink)
  Zone A: pixels_all=5234, pixels_ink=1832, ink_ratio=35.00%, Lab_all=[44.8, 8.2, 27.5], Lab_ink=[45.1, 8.0, 28.2], ΔE=0.15 (basis=mean_ink)
[2D ZONE ANALYSIS] Complete: OK, ΔE=0.23, confidence=0.92
[INSPECT] 2D analysis complete: OK, ΔE=0.23
```

**확인 포인트:**
1. ✅ `[PRINT BOUNDARIES]` - print_inner/outer 자동 추정
2. ✅ `pixels_all` - 실제 픽셀 수 (수천 단위)
3. ✅ `pixels_ink` - 잉크 픽셀 수
4. ✅ `ink_ratio` - 30-40% 정도 정상
5. ✅ `Lab_all` vs `Lab_ink` - 차이 확인
6. ✅ `ΔE` - Ring과 유사하므로 작아야 함

---

## 🐛 문제 발생 시

### 1. 2D 분석 실패 시
```
[INSPECT] 2D analysis failed: ..., falling back to 1D result
```
**→ 기존 1D 결과 사용, 에러 로그 확인**

### 2. ink_ratio가 너무 낮으면 (<0.1)
```python
# InkMaskConfig 조정
from src.core.zone_analyzer_2d import InkMaskConfig

config = InkMaskConfig(
    saturation_min=30,  # 40 → 30 (낮춤)
    value_max=220       # 200 → 220 (높임)
)
```

### 3. Zone Lab이 여전히 비슷하면
- `debug_2d_zones.png` 확인
- `debug_2d_ink.png` 확인
- Zone 마스크가 인쇄부를 덮는지 확인

---

## 📁 디버그 파일

검사 후 다음 파일 생성됨:
```
results/web/{run_id}/
├── debug_2d_zones.png  // Zone A/B/C 오버레이
├── debug_2d_ink.png    // 잉크 마스크 오버레이
├── result.json         // 검사 결과
└── ...
```

**확인 방법:**
1. `debug_2d_zones.png` 열기
2. Zone A(빨강), B(노랑), C(파랑)이 인쇄부를 덮는지 확인
3. `debug_2d_ink.png` 열기
4. 초록색이 잉크 도트만 잡는지 확인

---

## 🔄 1D 방식으로 되돌리기

필요하면 1D 방식 사용 가능:
```python
# src/web/app.py:121
use_2d_analysis: bool = Form(False)  # True → False
```

또는 API 호출 시:
```bash
curl -X POST http://localhost:8000/inspect \
  -F "file=@image.jpg" \
  -F "sku=SKU001" \
  -F "use_2d_analysis=false"
```

---

## ✅ 성공 기준

다음 조건이 만족되면 성공:

1. ✅ Zone A Lab ≈ Ring 2 (L≈45, b≈28)
2. ✅ Zone B Lab ≈ Ring 1 (L≈68, b≈22)
3. ✅ Zone C Lab ≈ Ring 0 (L≈95, b≈2)
4. ✅ pixel_count > 2000 (각 Zone)
5. ✅ ink_ratio ≈ 0.3-0.4 (30-40%)
6. ✅ ΔE < 5.0 (Ring과 유사하므로 작아야 함)

---

## 🎯 다음 단계

### 성공 시:
1. ✅ AI 템플릿 방식 채택
2. 🔄 기존 1D 방식 deprecate
3. 📝 문서화 업데이트

### 실패 시:
1. 로그 분석
2. InkMaskConfig 조정
3. print_boundaries 확인
4. 필요 시 1D로 되돌리기

---

## 💡 참고

**AI 템플릿의 핵심 개선:**
- ✅ 2D 이미지에서 직접 Zone 마스크 생성
- ✅ print_inner/outer 자동 추정
- ✅ ink_mask로 도트만 평균
- ✅ 정확한 pixel_count
- ✅ mean_all vs mean_ink 비교
- ✅ Zone 마스크 시각화

**기존 1D 방식의 한계:**
- ❌ RadialProfile에서 정확한 pixel_count 불가능
- ❌ 도트 희석 문제 (바탕+잉크 혼합)
- ❌ Zone 좌표계 불일치

---

**테스트를 시작하세요!** 🚀

서버를 재시작하고 이미지를 업로드한 후 로그와 JSON 결과를 확인하세요.
