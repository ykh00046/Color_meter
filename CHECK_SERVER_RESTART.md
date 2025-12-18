# 서버 재시작 확인 체크리스트

## ✅ 확인해야 할 로그

재검사 실행 시 다음 로그가 출력되어야 합니다:

```
[ZONE COORD] Zone segmentation using PRINT AREA basis:
  - r_inner=0.150 (print start, from optical_clear_ratio=0.150)
  - r_outer=0.950 (print end)
  - lens_radius=520.0px
  - Normalization: r_norm = (r - 0.150) / (0.950 - 0.150)

[ZONE RESULT] Created 3 zones:
  Zone A: r_norm=[0.633, 0.950), r_pixel=[329.2px, 494.0px), pixels=5234, Lab=(45.0, 8.0, 28.0), mainly Ring 2 (outer print)
  Zone B: r_norm=[0.317, 0.633), r_pixel=[164.8px, 329.2px), pixels=6128, Lab=(68.0, 5.0, 22.0), mainly Ring 1 (middle print)
  Zone C: r_norm=[0.150, 0.317), r_pixel=[78.0px, 164.8px), pixels=3421, Lab=(95.0, 0.5, 2.0), mainly Ring 0 (inner clear)
```

## ❌ 이 로그가 없으면

서버가 재시작되지 않았습니다!

```bash
# 1. Ctrl+C로 현재 서버 종료
# 2. 서버 재시작
cd C:\X\Color_total\Color_meter
python -m src.web.app

# 3. http://localhost:8000 또는 8001 접속
# 4. 재검사 실행
```

## 📊 예상 결과

### Before (수정 전)
```json
{
  "zone_results": [
    {"zone_name": "A", "measured_lab": [71.57, -0.43, 9.68], "pixel_count": 115},
    {"zone_name": "B", "measured_lab": [71.20, -0.23, 8.84], "pixel_count": 116},
    {"zone_name": "C", "measured_lab": [71.97, -0.43, 6.95], "pixel_count": 64}
  ]
}
```

### After (수정 후 기대)
```json
{
  "zone_results": [
    {"zone_name": "A", "measured_lab": [45.0, 8.0, 28.0], "pixel_count": 5234},
    {"zone_name": "B", "measured_lab": [68.0, 5.0, 22.0], "pixel_count": 6128},
    {"zone_name": "C", "measured_lab": [95.0, 0.5, 2.0], "pixel_count": 3421}
  ]
}
```

## 🔧 여전히 L=71 근처이면

AI가 제공한 ink_mask 코드를 도입해야 합니다.
