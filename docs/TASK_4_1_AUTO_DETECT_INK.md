# Task 4.1: Auto-Detect Ink Configuration

**Status**: ✅ Completed (2025-12-19)
**Estimated Time**: 12 hours
**Actual Time**: ~3 hours

---

## Overview

Auto-Detect Ink Configuration은 대표 이미지를 분석하여 SKU의 잉크 색상 설정을 자동으로 제안하는 기능입니다.

## Features

### 1. Auto-Detect Ink Colors
- InkEstimator를 사용하여 이미지에서 잉크 색상 자동 검출
- GMM 클러스터링 기반 색상 분석
- LAB 색공간 값 추출

### 2. Suggest Zone Configuration
- 검출된 잉크를 Zone A, B, C에 자동 매핑
- L 값 기준으로 darkest → brightest 정렬
- 잉크 weight 기반 ΔE threshold 자동 계산

### 3. User Review & Approval
- API로 제안된 설정 반환
- 사용자가 검토 후 승인 필요
- PUT /api/sku/{sku_code}/zones로 최종 저장

---

## API Endpoints

### POST /api/sku/auto-detect-ink

대표 이미지에서 잉크 색상을 자동 검출합니다.

**Request Body**:
```json
{
  "sku_code": "SKU002",
  "image_path": "C:/X/Color_total/Color_meter/data/raw_images/SKU002_OK_001.jpg",
  "chroma_thresh": 6.0,
  "L_max": 98.0,
  "merge_de_thresh": 5.0,
  "linearity_thresh": 3.0
}
```

**Response (200 OK)**:
```json
{
  "sku_code": "SKU002",
  "ink_count": 3,
  "detected_inks": [
    {
      "L": 0.0,
      "a": 0.0,
      "b": 0.0,
      "weight": 0.792,
      "hex": "#000000",
      "suggested_threshold": 6.0
    },
    {
      "L": 7.6,
      "a": 16.2,
      "b": -29.4,
      "weight": 0.154,
      "hex": "#0A113F",
      "suggested_threshold": 10.0
    },
    {
      "L": 62.7,
      "a": 12.4,
      "b": -54.1,
      "weight": 0.054,
      "hex": "#5F95F9",
      "suggested_threshold": 10.0
    }
  ],
  "suggested_zones": {
    "A": {
      "L": 0.0,
      "a": 0.0,
      "b": 0.0,
      "threshold": 6.0,
      "description": "Outer zone (darkest) - Auto-detected"
    },
    "B": {
      "L": 7.6,
      "a": 16.2,
      "b": -29.4,
      "threshold": 10.0,
      "description": "Middle zone (darkest) - Auto-detected"
    },
    "C": {
      "L": 62.7,
      "a": 12.4,
      "b": -54.1,
      "threshold": 10.0,
      "description": "Inner zone (medium) - Auto-detected"
    }
  },
  "meta": {
    "bic": -119142.38,
    "sample_count": 7538,
    "correction_applied": false
  },
  "warnings": [],
  "message": "Successfully detected 3 ink(s). Review and approve the suggested zone configuration."
}
```

### GET /api/sku/

모든 SKU 코드를 조회합니다.

**Response**:
```json
{
  "skus": ["SKU001", "SKU002", "SKU003", "VIS_TEST"],
  "count": 4
}
```

### GET /api/sku/{sku_code}

특정 SKU 설정을 조회합니다.

### PUT /api/sku/{sku_code}/zones

Zone 설정을 업데이트합니다 (auto-detect 결과 적용 시 사용).

**Request Body**:
```json
{
  "zones": {
    "A": {
      "L": 0.0,
      "a": 0.0,
      "b": 0.0,
      "threshold": 6.0,
      "description": "Outer zone (darkest) - Auto-detected"
    },
    "B": {...},
    "C": {...}
  },
  "notes": "Auto-detected from SKU002_OK_001.jpg"
}
```

---

## Usage Example

### 1. Auto-Detect Ink Colors

```python
import requests

url = 'http://127.0.0.1:8000/api/sku/auto-detect-ink'
payload = {
    'sku_code': 'SKU002',
    'image_path': 'C:/X/Color_total/Color_meter/data/raw_images/SKU002_OK_001.jpg',
    'chroma_thresh': 6.0,
    'L_max': 98.0,
    'merge_de_thresh': 5.0,
    'linearity_thresh': 3.0
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Detected {result['ink_count']} inks")
for zone, config in result['suggested_zones'].items():
    print(f"Zone {zone}: L={config['L']}, a={config['a']}, b={config['b']}")
```

### 2. Apply Suggested Configuration

```python
# Review and approve suggested zones
update_url = f'http://127.0.0.1:8000/api/sku/{sku_code}/zones'
update_payload = {
    'zones': result['suggested_zones'],
    'notes': 'Auto-detected and approved'
}

response = requests.put(update_url, json=update_payload)
print(f"Updated: {response.json()['message']}")
```

---

## Threshold Calculation Logic

잉크 weight에 따라 자동으로 threshold를 계산합니다:

```python
def _calculate_threshold_from_weight(weight: float) -> float:
    if weight > 0.6:
        return 6.0  # Strict threshold for dominant ink
    elif weight > 0.3:
        return 8.0  # Medium threshold
    else:
        return 10.0  # Loose threshold for minor inks
```

**Rationale**:
- 높은 weight (dominant ink) → 엄격한 threshold (6.0)
- 낮은 weight (minor ink) → 느슨한 threshold (10.0)

---

## Zone Mapping Strategy

검출된 잉크를 Zone에 매핑하는 전략:

1. **Sort by L value**: darkest → brightest
2. **Assign to zones**:
   - Zone A (Outer): 가장 어두운 잉크
   - Zone B (Middle): 중간 톤 잉크
   - Zone C (Inner): 가장 밝은 잉크
3. **Generate descriptions**: L 값 기준으로 "darkest", "medium", "lightest" 자동 생성

---

## Parameter Tuning Guide

### chroma_thresh (default: 6.0)
- **높이면**: 유채색 잉크만 검출 (무채색 제외)
- **낮추면**: 무채색 포함하여 검출
- **권장**: 6.0 (대부분의 렌즈에 적합)

### L_max (default: 98.0)
- **높이면**: 밝은 영역도 포함
- **낮추면**: 하이라이트 제거 강화
- **권장**: 98.0

### merge_de_thresh (default: 5.0)
- **높이면**: 유사 색상 적극 병합
- **낮추면**: 유사 색상도 분리
- **권장**: 5.0

### linearity_thresh (default: 3.0)
- **낮추면**: Mixing correction 적극 적용
- **높이면**: Mixing correction 억제
- **권장**: 3.0

---

## Test Results

### Test Case: SKU002_OK_001.jpg

**Input**:
- SKU: SKU002
- Image: SKU002_OK_001.jpg
- Parameters: default values

**Output**:
- Ink Count: 3
- Detected Inks:
  1. L=0.0, a=0.0, b=0.0 (Black, weight=0.792) → Zone A, threshold=6.0
  2. L=7.6, a=16.2, b=-29.4 (Dark Blue, weight=0.154) → Zone B, threshold=10.0
  3. L=62.7, a=12.4, b=-54.1 (Bright Blue, weight=0.054) → Zone C, threshold=10.0

**Result**: ✅ Successfully detected and mapped

---

## Warnings and Error Handling

### Warning: Only 1 ink detected
**Cause**: 색상 변화가 적거나 파라미터 부적절
**Action**: chroma_thresh 낮추거나 다른 이미지 사용

### Warning: Mixing correction applied
**Cause**: 중간 톤이 양 극단의 혼합으로 판정
**Action**: 정보성 경고, 조치 불필요

### Error: Image not found
**Cause**: image_path가 잘못되었거나 파일이 없음
**Action**: 경로 확인

### Error: Insufficient ink pixels
**Cause**: 검출된 잉크 픽셀이 500개 미만
**Action**: 이미지 품질 확인 또는 파라미터 조정

---

## Implementation Details

### Files Modified/Created
- `src/web/routers/sku.py`: 새로 생성 (400 lines)
- `src/web/app.py`: SKU router 등록
- `src/sku_manager.py`: `list_skus()` 메서드 추가

### Key Functions
- `auto_detect_ink_config()`: 메인 엔드포인트
- `_map_inks_to_zones()`: 잉크 → 존 매핑
- `_calculate_threshold_from_weight()`: Threshold 계산

---

## Future Enhancements (Optional)

### 1. UI Integration
SKU 관리 페이지에 "Auto-Detect" 버튼 추가 (현재는 API만 제공)

### 2. Batch Auto-Detect
여러 이미지를 분석하여 평균값 계산

### 3. Advanced Mapping
4-zone 또는 2-zone 렌즈 지원

### 4. Confidence Score
검출 신뢰도 점수 제공

---

## Related Documentation

- **InkEstimator Guide**: `docs/guides/INK_ESTIMATOR_GUIDE.md`
- **API Reference**: `API_REFERENCE.md`
- **SKU Manager**: `src/sku_manager.py`

---

**Completed**: 2025-12-19
**Tested**: ✅ Passed
**Deployed**: ✅ Ready for production
