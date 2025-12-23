# Quick Start JSON ↔ M2 ComparisonResult 매핑

## 목적
Quick Start(STD vs Sample 단발 비교) 결과를 M2 ComparisonService의 정식 결과 구조로 흡수하기 위한 표준 매핑 정의입니다.

## 1. 개념 정리 (전제)
- Quick Start: Reference Image 기반의 즉시 비교/데모
- M2: STD Model(DB, versioned) 기반의 판정/이력/운영
- Quick Start 결과는 M2 결과의 부분집합입니다.

## 2. Quick Start JSON (현재)
```json
{
  "mean_delta_e": 2.8,
  "max_delta_e": 6.1,
  "delta_lab": { "dL": -3.2, "da": 1.1, "db": 0.5 },
  "overall_shift": "slightly darker, warmer",
  "zone_deltas": {
    "A": { "delta_e": 3.1, "delta_lab": { "dL": -2.5, "da": 1.0, "db": 0.3 } },
    "B": { "delta_e": 5.9, "delta_lab": { "dL": -4.1, "da": 1.5, "db": 0.7 } }
  }
}
```

## 3. M2 ComparisonResult (정식 구조)
```json
{
  "id": 12,
  "test_sample_id": 34,
  "std_model_id": null,
  "scores": {
    "total_score": 78.4,
    "zone_score": 78.4,
    "confidence_score": 86.0
  },
  "judgment": "RETAKE",
  "zone_details": {
    "A": { "delta_e": 3.1, "color_score": 69.0 },
    "B": { "delta_e": 5.9, "color_score": 41.0 }
  },
  "top_failure_reasons": [
    {
      "rank": 1,
      "category": "ZONE_COLOR",
      "zone": "B",
      "message": "Zone B: Delta E=5.9 exceeds threshold",
      "severity": 59
    }
  ]
}
```

## 4. 매핑 규칙 (핵심)
### 4.1 Zone 매핑
- `zone_deltas[A].delta_e` → `zone_details[A].delta_e`
- `zone_deltas[A].delta_lab` → `zone_details[A].delta_lab` (optional)

### 4.2 Score 매핑
- `mean_delta_e`는 `total_score` 계산에 참고하는 보조 지표
- `max_delta_e`는 failure reason severity 산정에 참고
- `delta_lab` 방향은 failure reason message 작성에 참고

### 4.3 overall_shift 처리 규칙 (중요)
- `overall_shift`는 설명용 지표입니다.
- 판정/점수 계산에는 사용하지 않습니다.
- M2에는 meta로 보관합니다.

```json
"meta": {
  "quick_start_summary": {
    "overall_shift": "slightly darker, warmer"
  }
}
```

## 5. 통합 JSON (권장 포맷)
```json
{
  "comparison_version": "M2",
  "derived_from": "quick_start",
  "scores": { ... },
  "judgment": "...",
  "zone_details": { ... },
  "top_failure_reasons": [ ... ],
  "quick_metrics": {
    "mean_delta_e": 2.8,
    "max_delta_e": 6.1,
    "overall_shift": "slightly darker, warmer"
  }
}
```
