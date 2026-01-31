# Plate Engine 설계 문서 (v1.0)

> 기존 색 추출 엔진과 병행 사용. 대체가 아닌 추가 엔진.

---

## 1. 개요

### 1.1 목적
- 렌즈 인쇄 영역을 **판(plate) 단위로 먼저 분리**한 뒤 색상 분석
- 흰판/검은판 2장을 활용하여 **발색(커버리지)과 색상을 분리** 추정
- 배경 픽셀이 색 클러스터를 지배하는 문제 해결
- "하얗게 뜸/연함/덜찍힘" 현상의 정량적 원인 분석

### 1.2 기존 방식 vs 신규 방식

| 항목 | 기존 방식 | 신규 방식 (Plate Engine) |
|------|----------|-------------------------|
| 입력 | 이미지 1장 | 흰판 + 검은판 2장 |
| 분석 단위 | ROI 전체 | 판(ring/dot/clear) 단위 |
| 색 추출 | k-means 클러스터링 | 판 core 영역에서만 추출 |
| 발색 분석 | 불가 (색과 커버리지 혼합) | α(알파) 맵으로 분리 |
| 혼합 구간 | 클러스터에 섞임 | transition으로 별도 관리 |

### 1.3 병행 사용 전략
- 기존 엔진: 유지 (단일 이미지 빠른 스크리닝)
- 신규 엔진: 추가 (정밀 분석, 원인 진단)
- API 응답에 `plate` 블록 추가 (기존 필드 유지)

---

## 2. 핵심 개념

### 2.1 판(Plate) 정의

렌즈 인쇄 영역을 3개 판으로 분리:

| 판 | 설명 | 특성 |
|----|------|------|
| **Ring** | 외곽 솔리드 링 | α 높음, 연속성 큼 |
| **Dot** | 도트(halftone) 영역 | α 점 형태 반복, 고주파 |
| **Clear** | 중심부 비인쇄 | α ≈ 0, 투명 |

### 2.2 알파(α) 블렌딩 모델

흰판/검은판 2장으로 커버리지와 잉크색 분리:

```
검은판: I_black = α × I_ink
흰판:   I_white = α × I_ink + (1-α) × 255

α = 1 - (I_white - I_black) / 255
I_ink ≈ I_black / α
```

- **α (알파)**: 픽셀의 잉크 커버리지/농도 (0~1)
- **I_ink**: 배경 영향 제거한 잉크 자체 색

### 2.3 Core / Transition 분리

혼합(그라데이션) 구간 처리:

```
ring_core = erode(ring_mask, w)
dot_core = erode(dot_mask, w)
transition = (ring ∪ dot) - (ring_core ∪ dot_core)
```

- **Core**: 순수 구간 → 대표색 추출 영역
- **Transition**: 혼합 구간 → 관측 지표로만 사용 (판정에 미사용)
- **w (폭)**: `max(4px, 0.015 × R)`

> ⚠️ Transition은 디자인 특성. NG 트리거로 사용하지 않음.

---

## 3. 처리 파이프라인

### Step 0: 정렬 (Registration)
1. 각 이미지에서 렌즈 외곽 원(cx, cy, r) 검출
2. 동일 스케일로 정규화
3. (필요시) polar 변환 후 위상상관으로 각도 정렬
4. 정렬 품질 점수(score) 산출

### Step 1: α(커버리지) 맵 계산
1. 픽셀별 α 계산 (RGB 평균/중앙값 기반)
2. 클리핑 (0.02 ~ 0.98)
3. 노이즈 안정화 (0 근처 분모 보호)
4. Residual 계산 (아티팩트 탐지용)

### Step 2: 판(Plate) 마스크 생성
1. 외곽 반사 링 제외 (r > 0.95R)
2. Clear 마스크: α < 0.08 + 중심 반경 조건
3. Ring 마스크: α 높음 + 연속 띠 + 반경 방향
4. Dot 마스크: 블롭/고주파/LoG 기반

### Step 3: Core/Transition 분리
1. ring_core = erode(ring, w)
2. dot_core = erode(dot, w)
3. transition = 나머지 영역

### Step 4: 지표 계산
- 각 판 Core에서:
  - Lab 통계 (mean, p95, p05, std)
  - α 통계 (mean, p95, p05, std)
- Dot 전용: count, density, size, missing_score, smear_score
- Ring 전용: width, uniformity_theta, edge_softness
- Transition: Lab/α 통계 (관측용)

### Step 5: JSON 생성
- plate_v1.0 스키마로 결과 생성

---

## 4. 출력 JSON 스키마 (plate_v1.0)

```json
{
  "schema_version": "plate_v1.0",
  "match": {
    "match_id": "UI에서 입력한 키",
    "product_id": "옵션(있으면)",
    "captured_at": "2026-01-14T09:00:00+09:00"
  },
  "inputs": {
    "white": {"filename": "white.png", "size": [1024, 1024]},
    "black": {"filename": "black.png", "size": [1024, 1024]}
  },
  "geom": {
    "cx": 84.6,
    "cy": 84.6,
    "r": 75.32,
    "r_exclude_outer": 0.95,
    "r_exclude_inner": 0.00
  },
  "registration": {
    "method": "circle_norm+polar_phase",
    "score": 0.92,
    "dx_dy": [0.3, -0.2],
    "dtheta_deg": 0.8,
    "notes": ""
  },
  "alpha_model": {
    "mode": "white_black_pair",
    "alpha_clip": [0.02, 0.98],
    "residual": {
      "white_recon_rmse": 4.1,
      "mask_artifact_ratio": 0.013
    }
  },

  "plates": {
    "ring": {
      "geometry": {
        "r_range": [0.72, 0.92],
        "area_ratio": 0.18
      },
      "core": {
        "mask_ratio": 0.12,
        "lab": {"mean": [45.2, 6.1, 18.9], "p95": [], "p05": [], "std": []},
        "alpha": {"mean": 0.78, "p95": 0.91, "p05": 0.55, "std": 0.08},
        "uniformity": {
          "theta_std_lab": 1.9,
          "theta_std_alpha": 0.06
        }
      },
      "transition": {
        "mask_ratio": 0.06,
        "lab": {"mean": [], "std": []},
        "alpha": {"mean": 0.52, "std": 0.12}
      },
      "notes": ["transition_is_design_feature"]
    },

    "dot": {
      "geometry": {
        "r_range": [0.40, 0.78],
        "area_ratio": 0.42
      },
      "core": {
        "lab": {"mean": [], "std": []},
        "alpha": {"mean": 0.34, "p05": 0.12, "p95": 0.58, "std": 0.10},
        "dot_metrics": {
          "count": 12840,
          "density_per_mm2": 0.0,
          "size_px_mean": 3.2,
          "size_px_std": 0.9,
          "missing_score": 0.03,
          "smear_score": 0.05
        }
      },
      "transition": {
        "lab": {"mean": [], "std": []},
        "alpha": {"mean": 0.28, "std": 0.11}
      }
    },

    "clear": {
      "geometry": {"r_range": [0.00, 0.40], "area_ratio": 0.40},
      "core": {
        "alpha": {"mean": 0.03, "p95": 0.07, "std": 0.02}
      }
    }
  },

  "summary": {
    "expected_plates": ["ring", "dot", "clear"],
    "artifact_tags": ["OUTER_RIM_EXCLUDED"],
    "diagnosis_tags": ["OK"],
    "notes": "혼합(transition)은 디자인 특성으로 관측만 수행"
  },

  "debug": {
    "params": {
      "transition_width": {"mode": "dynamic", "value": "max(4px, 0.015R)"},
      "outer_rim_exclude": 0.95,
      "alpha_threshold_clear": 0.08
    }
  }
}
```

---

## 5. 기본 파라미터 (추천값)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| transition_width | `max(4px, 0.015R)` | Core/Transition 분리 폭 |
| r_exclude_outer | 0.95 | 외곽 반사 링 제외 |
| alpha_threshold_clear | 0.08 | Clear 판 판정 기준 |
| alpha_clip | [0.02, 0.98] | α 클리핑 범위 |

---

## 6. API 통합

### 6.1 엔드포인트
- 기존 `POST /api/v7/analyze_single` 유지
- 응답에 `plate` 블록 추가

### 6.2 입력
```json
{
  "match_id": "매칭 키",
  "white_image": "흰판 이미지",
  "black_image": "검은판 이미지",
  "product_id": "옵션"
}
```

### 6.3 응답 확장
```json
{
  "...기존필드": "...",
  "plate": { /* plate_v1.0 결과 전체 */ }
}
```

---

## 7. STD 저장 확장

### 7.1 model.json 확장
```json
{
  "schema_version": "std_v2_plate",
  "geom": {},
  "meta": {},
  "plate_params": {
    "transition_width": "max(4px,0.015R)",
    "outer_rim_exclude": 0.95,
    "alpha_clear_th": 0.08
  },
  "plate_geometry": {
    "ring": {"r_range": [0.72, 0.92]},
    "dot": {"r_range": [0.40, 0.78]},
    "clear": {"r_range": [0.00, 0.40]}
  }
}
```

### 7.2 model.npz 확장
- `ring_core_radial_lab_mean`, `ring_core_radial_lab_std`, ...
- `dot_core_radial_lab_mean`, `dot_core_radial_lab_std`, ...
- `ring_core_radial_alpha_mean`, `ring_core_radial_alpha_std`, ...
- `dot_core_radial_alpha_mean`, `dot_core_radial_alpha_std`, ...

### 7.3 호환성
- 기존 모델(std_v1): plate 키 없음 → 로더가 skip
- 비교 엔진: plate signature 있으면 plate 비교, 없으면 기존 방식 폴백

---

## 8. 구현 로드맵

### Phase 0: 스펙 고정 (0.5~1일)
- [ ] 판 정의 확정 (ring/dot/clear)
- [ ] 분석 반경 범위 확정
- [ ] JSON 스키마 확정

### Phase 1: 정렬 + 기본 마스크
- [ ] 외곽 원 검출 (geom)
- [ ] 흰/검 정규화 (센터/스케일)
- [ ] 각도 정렬 (필요시)
- [ ] registration.score 산출

### Phase 2: α 맵 + 잉크색 근사
- [ ] α 계산 (채널 평균)
- [ ] 클리핑 + 노이즈 안정화
- [ ] Residual 계산 (아티팩트 탐지)

### Phase 3: 판 마스크 생성
- [ ] 외곽 림 제외
- [ ] Clear 마스크
- [ ] Ring 마스크
- [ ] Dot 마스크

### Phase 4: Core/Transition 분리
- [ ] erode로 core 생성
- [ ] transition 계산
- [ ] 지표 계산 (Lab/α/dot_metrics/ring_uniformity)

### Phase 5: JSON 생성 + API 통합
- [ ] plate_v1.0 JSON 생성
- [ ] /api/v7/analyze_single 응답 확장

### Phase 6: STD 저장 확장
- [ ] model.json 확장
- [ ] model.npz 확장
- [ ] 호환성 처리

### Phase 7: 검증
- [ ] 정상 5 + 이슈 5 세트 테스트
- [ ] 반복 스캔 재현성 확인
- [ ] 정렬 실패 시 fail-fast 확인

---

## 9. 개발 체크리스트

### A. Plate Engine 코어 모듈
- [ ] `plate_engine.analyze_pair(white_bgr, black_bgr, cfg, geom_hint=None)`
  - 원 검출/정규화/정렬
  - alpha_map 계산 + residual 마스크
  - plate mask 생성 + outer rim 제외
  - core/transition 분리
  - 지표 계산
  - JSON 생성

### B. API 통합
- [ ] `src/web/routers/v7.py`: analyze_single_sample() 확장
- [ ] 입력: white, black, match_id
- [ ] 응답에 `"plate": plate_result` 추가

### C. STD 학습/등록
- [ ] `scripts/train_std_model.py` 확장
- [ ] `model_io.py` 확장

### D. STD 비교
- [ ] plate signature 있으면 plate 비교
- [ ] 없으면 기존 radial 비교 폴백

### E. 테스트
- [ ] 10세트 golden test
- [ ] registration score 안정성
- [ ] core 대표색 재현성

---

## 10. 진단 태그

| 태그 | 의미 |
|------|------|
| `DOT_ALPHA_LOW` | 도트 커버리지 부족 |
| `RING_DARKER_THAN_STD` | 링이 기준보다 어두움 |
| `DOT_MISSING` | 도트 결손 |
| `EDGE_REFLECTION_ARTIFACT` | 외곽 반사 아티팩트 |
| `OUTER_RIM_EXCLUDED` | 외곽 림 제외됨 |
| `REGISTRATION_FAILED` | 정렬 실패 |

---

## 11. 참고: 혼합색 발생 원인

1. **공정 혼합**: 오버프린트/번짐/도트게인
2. **공간 혼합**: 픽셀이 여러 잉크+투명 포함 → 평균색
3. **광학 혼합**: 렌즈 재질/산란/배경 의존

→ Transition으로 분리하여 Core 대표색 오염 방지

---

## 12. 파일 구조 (예정)

```
src/engine_v7/core/
├── plate/                      # [NEW] Plate Engine
│   ├── __init__.py
│   ├── plate_engine.py         # 메인 분석 함수
│   ├── registration.py         # 정렬 로직
│   ├── alpha_model.py          # α 맵 계산
│   ├── plate_mask.py           # 판 마스크 생성
│   └── plate_metrics.py        # 판별 지표 계산
```

---

## 13. 코드 패치 상세

### 13.1 STD 저장 전략 (호환성 유지)

| 버전 | schema_version | 설명 |
|------|---------------|------|
| 기존 | `std_model.v1` | 기존 모델 그대로 로드 가능 |
| 신규 | `std_model.v2_plate` | plate 키 추가 저장 |

- 로더는 v1/v2 모두 허용
- v2에서만 plate 키를 읽음
- **기존 STD 자산 깨지지 않음**

### 13.2 single_analyzer.py 패치

**목표**: 기존 호출 깨지 않고 plate 모드 추가

```python
def analyze_single_sample(
    test_bgr: np.ndarray,
    cfg: Dict[str, Any],
    analysis_modes: Optional[List[str]] = None,
    black_bgr: Optional[np.ndarray] = None,  # NEW
    match_id: Optional[str] = None,           # NEW
) -> Dict[str, Any]:
    if analysis_modes is None:
        analysis_modes = ["gate", "color", "radial", "ink", "pattern", "zones"]
        if black_bgr is not None:
            analysis_modes.append("plate")  # 자동 추가

    results = {}

    # Geometry (white 기준)
    geom = detect_lens_circle(test_bgr)

    # ⚠️ plate(pair) 모드에서는 white_balance 스킵 (동일 보정 보장 어려움)
    wb_enabled = cfg.get("gate", {}).get("white_balance", {}).get("enabled", False)
    if wb_enabled and black_bgr is None:
        test_bgr, _ = apply_white_balance(test_bgr, geom, cfg)
    elif wb_enabled and black_bgr is not None:
        results.setdefault("warnings", [])
        results["warnings"].append("white_balance_skipped_for_plate_pair")

    # gate/color/radial/ink/pattern/zones 기존 로직 그대로...

    # plate (NEW)
    if "plate" in analysis_modes and black_bgr is not None:
        from .plate.plate_engine import analyze_plate_pair

        plate_cfg = cfg.get("plate", {})
        results["plate"] = analyze_plate_pair(
            white_bgr=test_bgr,
            black_bgr=black_bgr,
            cfg=plate_cfg,
            match_id=match_id,
            geom_hint=geom,
        )

    return results
```

**핵심**:
- `black_bgr` 안 넘기면 → 기존 동작
- `black_bgr` 넘기면 → plate 자동 추가

### 13.3 model_io.py 패치

#### 13.3.1 save_model 확장

```python
def save_model(model: StdModel, out_prefix: str) -> Dict[str, str]:
    out_prefix = str(out_prefix)
    npz_path = out_prefix + ".npz"
    json_path = out_prefix + ".json"

    # --- (NEW) optional plate data ---
    plate_signatures = getattr(model, "plate_signatures", None)
    plate_meta = getattr(model, "plate_meta", None)

    extra_npz: Dict[str, np.ndarray] = {}
    extra_shapes: Dict[str, Any] = {}

    if isinstance(plate_signatures, dict):
        for k, v in plate_signatures.items():
            if isinstance(v, np.ndarray) and v.size > 0:
                extra_npz[k] = v
                extra_shapes[k] = list(v.shape)

    schema_version = "std_model.v2_plate" if (extra_npz or plate_meta) else "std_model.v1"

    np.savez_compressed(
        npz_path,
        radial_lab_mean=model.radial_lab_mean,
        radial_lab_p95=model.radial_lab_p95,
        radial_lab_std=model.radial_lab_std if model.radial_lab_std is not None else np.array([]),
        radial_lab_p05=model.radial_lab_p05 if model.radial_lab_p05 is not None else np.array([]),
        radial_lab_median=model.radial_lab_median if model.radial_lab_median is not None else np.array([]),
        radial_lab_mad=model.radial_lab_mad if model.radial_lab_mad is not None else np.array([]),
        **extra_npz,  # <-- NEW
    )

    payload: Dict[str, Any] = {
        "schema_version": schema_version,
        "geom": asdict(model.geom),
        "meta": model.meta,
        "has_band": bool(model.radial_lab_std is not None and model.radial_lab_p05 is not None),
        "has_robust": bool(model.radial_lab_median is not None and model.radial_lab_mad is not None),
        "has_plate": bool(extra_npz),
        "shapes": {
            "radial_lab_mean": list(model.radial_lab_mean.shape),
            "radial_lab_p95": list(model.radial_lab_p95.shape),
            **extra_shapes,  # <-- NEW
        },
    }
    if plate_meta:
        payload["plate_meta"] = plate_meta  # <-- NEW

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {"npz": npz_path, "json": json_path}
```

#### 13.3.2 load_model 확장

```python
schema_version = meta.get("schema_version", "unknown")
if schema_version not in ["std_model.v1", "std_model.v2_plate", "unknown"]:
    raise ValueError(f"Unsupported schema version: {schema_version}")

# ... 기존 로직으로 radial 로드 ...

# NEW: plate signatures 로드
plate_signatures: Dict[str, np.ndarray] = {}
for k in arr.files:
    if k.startswith("plate_"):
        plate_signatures[k] = arr[k]

plate_meta = meta.get("plate_meta", None)

m = StdModel(
    geom=geom,
    radial_lab_mean=radial_lab_mean,
    radial_lab_p95=radial_lab_p95,
    meta=meta["meta"],
    radial_lab_std=radial_lab_std,
    radial_lab_p05=radial_lab_p05,
    radial_lab_median=radial_lab_median,
    radial_lab_mad=radial_lab_mad,
)

# 동적으로 plate 데이터 추가 (호환성)
if plate_signatures:
    setattr(m, "plate_signatures", plate_signatures)
if plate_meta:
    setattr(m, "plate_meta", plate_meta)

return m
```

### 13.4 NPZ 저장 키 네이밍

```
plate_ring_core_radial_lab_mean
plate_ring_core_radial_lab_p95
plate_ring_core_radial_lab_p05
plate_ring_core_radial_lab_std
plate_ring_core_radial_alpha_mean
plate_ring_core_radial_alpha_std

plate_dot_core_radial_lab_mean
plate_dot_core_radial_lab_p95
plate_dot_core_radial_lab_p05
plate_dot_core_radial_lab_std
plate_dot_core_radial_alpha_mean
plate_dot_core_radial_alpha_std
```

---

## 14. plate_engine.py 최소 구현

### 14.1 위치
```
src/engine_v7/core/plate/plate_engine.py
```

### 14.2 핵심 함수

```python
def analyze_plate_pair(
    white_bgr: np.ndarray,
    black_bgr: np.ndarray,
    cfg: Dict[str, Any],
    match_id: Optional[str] = None,
    geom_hint: Optional[LensGeometry] = None,
) -> Dict[str, Any]:
    """
    흰판/검은판 2장을 분석하여 plate_v1.0 JSON 반환.

    Args:
        white_bgr: 흰판 이미지 (BGR)
        black_bgr: 검은판 이미지 (BGR)
        cfg: plate 설정 (transition_width, outer_rim_exclude 등)
        match_id: 매칭 키 (UI 입력)
        geom_hint: 이미 검출된 geometry (재사용 가능)

    Returns:
        plate_v1.0 스키마 딕셔너리
    """
    # Step 0: Geometry 검출 (또는 재사용)
    # Step 1: Registration (정렬)
    # Step 2: Alpha map 계산
    # Step 3: Plate mask 생성 (ring/dot/clear)
    # Step 4: Core/Transition 분리
    # Step 5: 지표 계산
    # Step 6: JSON 생성

    return plate_result
```

---

## 15. STD 학습에서 Plate Signature 생성

### 15.1 학습 흐름

```python
# train_std_model.py 확장

# 1. 흰/검 이미지 쌍 로드
white_images = [...]
black_images = [...]

# 2. plate_engine으로 분석
plate_results = []
for white, black in zip(white_images, black_images):
    result = analyze_plate_pair(white, black, cfg)
    plate_results.append(result)

# 3. plate별 radial signature 계산 (core 마스크 기반)
plate_signatures = {
    "plate_ring_core_radial_lab_mean": ...,
    "plate_ring_core_radial_alpha_mean": ...,
    "plate_dot_core_radial_lab_mean": ...,
    "plate_dot_core_radial_alpha_mean": ...,
}

plate_meta = {
    "plate_params": {
        "transition_width": "max(4px,0.015R)",
        "outer_rim_exclude": 0.95,
        "alpha_clear_th": 0.08
    },
    "plate_geometry": {
        "ring": {"r_range": [0.72, 0.92]},
        "dot": {"r_range": [0.40, 0.78]},
        "clear": {"r_range": [0.00, 0.40]}
    }
}

# 4. StdModel에 추가
model.plate_signatures = plate_signatures
model.plate_meta = plate_meta

# 5. 저장
save_model(model, out_prefix)
```

---

## 16. End-to-End 흐름 요약

```
[입력]
  흰판 이미지 + 검은판 이미지
          ↓
[단일 분석] analyze_single_sample(..., black_bgr=black)
          ↓
[Plate Engine] analyze_plate_pair()
  - Registration
  - Alpha map
  - Plate masks
  - Core/Transition
  - Metrics
          ↓
[출력] results["plate"] = plate_v1.0 JSON
          ↓
[STD 학습] train_std_model.py
  - plate_signatures 계산
  - plate_meta 생성
          ↓
[STD 저장] save_model()
  - NPZ: plate_* 키 추가
  - JSON: schema_version = std_model.v2_plate
          ↓
[STD 비교]
  - plate signature 있으면 → plate별 비교
  - 없으면 → 기존 radial 비교 폴백
```

---

## 17. 추가 구현 필요 파일

| 파일 | 역할 | 우선순위 |
|------|------|---------|
| `core/plate/plate_engine.py` | 메인 분석 함수 | 1 |
| `core/plate/registration.py` | 정렬 로직 | 1 |
| `core/plate/alpha_model.py` | α 맵 계산 | 1 |
| `core/plate/plate_mask.py` | 판 마스크 생성 | 2 |
| `core/plate/plate_metrics.py` | 지표 계산 | 2 |

---

## 18. 설정 파라미터 (default.json 추가)

```json
{
  "plate": {
    "enabled": true,
    "transition_width": {
      "mode": "dynamic",
      "value": "max(4px, 0.015R)"
    },
    "outer_rim_exclude": 0.95,
    "alpha_clear_threshold": 0.08,
    "alpha_clip": [0.02, 0.98],
    "registration": {
      "method": "circle_norm+polar_phase",
      "score_threshold": 0.7
    }
  }
}
```

---

## 19. 라우터 확장 상세

### 19.1 기존 연계 유지 원칙

- `/api/v7/analyze_single` 경로 그대로 유지
- 기존 파라미터 `files` 그대로 유지 (기존 UI 안 깨짐)
- **추가 선택 파라미터만 추가**

### 19.2 라우터 시그니처 (호환 유지)

```python
@router.post("/analyze_single")
async def analyze_single(
    x_user_role: Optional[str] = Header(default=""),
    analysis_modes: Optional[str] = Form("all"),
    expected_ink_count: Optional[int] = Form(None),
    files: List[UploadFile] = File(...),                  # 기존: white로 사용
    black_files: Optional[List[UploadFile]] = File(None), # NEW: black
    match_ids: Optional[List[str]] = Form(None),          # NEW: 페어 식별
):
```

### 19.3 동작 규칙

| 조건 | 동작 |
|------|------|
| `black_files is None` | 기존과 동일: files 각각 단독 분석 |
| `black_files is not None` | 페어 분석 모드 |

**페어 분석 모드**:
1. 길이 동일 검증 (아니면 에러)
2. `files[i]` ↔ `black_files[i]` 순서대로 페어 매칭
3. `match_id` 있으면 사용, 없으면 stem 기반 생성
4. 결과는 페어 단위로 반환

```python
# 페어 분석 호출
analysis = analyze_single_sample(
    white_bgr,
    cfg,
    modes,
    black_bgr=black_bgr,
    match_id=match_id
)
```

### 19.4 응답 구조 (plate 포함)

```json
{
  "gate": {...},
  "color": {...},
  "radial": {...},
  "ink": {...},
  "pattern": {...},
  "zones": {...},
  "quality_score": 82.5,
  "warnings": [...],
  "operator_summary": {...},
  "engineer_kpi": {...},

  "plate": {
    "schema_version": "plate_v1.0",
    "match_id": "...",
    "registration": {...},
    "geom": {...},
    "alpha_model": {...},
    "plates": {
      "ring": {"core": {...}, "transition": {...}, "geometry": {...}},
      "dot":  {"core": {...}, "transition": {...}, "geometry": {...}},
      "clear":{"core": {...}, "geometry": {...}}
    },
    "summary": {...},
    "debug": {...}
  }
}
```

---

## 20. StdModel 정식 필드 추가

### 20.1 변경 위치
`src/engine_v7/core/signature/std_model.py`

### 20.2 변경 내용

```python
from typing import Any, Dict, Optional

@dataclass
class StdModel:
    geom: LensGeometry
    radial_lab_mean: np.ndarray
    radial_lab_p95: np.ndarray
    meta: Dict[str, Any]
    radial_lab_std: Optional[np.ndarray] = None
    radial_lab_p05: Optional[np.ndarray] = None
    radial_lab_median: Optional[np.ndarray] = None
    radial_lab_mad: Optional[np.ndarray] = None

    # NEW (plate 확장)
    plate_signatures: Optional[Dict[str, np.ndarray]] = None  # npz에 저장
    plate_meta: Optional[Dict[str, Any]] = None               # json에 저장
```

### 20.3 장점
- `@dataclass`라 확장 저장에 유리
- `slots=True`가 아니라 동적 확장도 가능
- 정식 필드 추가로 장기 운영 안정

---

## 21. White Balance 운영 규칙

### 21.1 페어 분석에서 WB 처리

> ⚠️ **중요**: WB가 이미지마다 다르게 걸리면 α 계산이 흔들림

| 모드 | 동작 |
|------|------|
| 기본 (페어) | WB OFF |
| `from_white` (옵션) | 흰판에서 계산한 WB gain을 검은판에 동일 적용 |

### 21.2 권장 설정

```python
# plate(pair) 모드에서는 white_balance 기본 스킵
wb_enabled = cfg.get("gate", {}).get("white_balance", {}).get("enabled", False)

if wb_enabled and black_bgr is None:
    # 단독 분석: WB 적용
    test_bgr, _ = apply_white_balance(test_bgr, geom, cfg)
elif wb_enabled and black_bgr is not None:
    # 페어 분석: WB 스킵 + 경고
    results.setdefault("warnings", [])
    results["warnings"].append("white_balance_skipped_for_plate_pair")
```

---

## 22. train_std_model.py 확장

### 22.1 추가 인자

```bash
# 옵션 1: 분리 리스트
--white_stds w1.png w2.png w3.png
--black_stds b1.png b2.png b3.png

# 옵션 2: 페어 문자열 (운영 편의)
--std_pairs "w1:b1" "w2:b2" "w3:b3"
```

### 22.2 동작 흐름

```python
if 페어 인자 없음:
    # 기존 그대로 (흰판 1장으로 radial signature 저장)
    pass
else:
    # 1. 기존 모델(radial_lab_mean/p95 등)은 흰판 기준으로 저장 (기존 체계 유지)

    # 2. plate_engine으로 페어 분석
    for white, black in pairs:
        result = analyze_plate_pair(white, black, cfg)

    # 3. plate signatures 계산
    #    - ring_core/dot_core의 radial Lab signature
    #    - ring_core/dot_core의 radial alpha signature

    # 4. model에 추가
    model.plate_signatures = {
        "plate_ring_core_radial_lab_mean": ...,
        "plate_ring_core_radial_alpha_mean": ...,
        "plate_dot_core_radial_lab_mean": ...,
        "plate_dot_core_radial_alpha_mean": ...,
    }
    model.plate_meta = {
        "plate_params": {...},
        "plate_geometry": {...}
    }

    # 5. 저장
    save_model(model, out_prefix)
```

---

## 23. 개발 체크리스트 (완료 조건 포함)

### A. 데이터 계약 고정
- [ ] plate_v1.0 JSON 스키마 고정
- [ ] transition은 관측 지표(디자인 특성), w=max(4px,0.015R)

### B. 모델 저장 확장
- [ ] StdModel에 `plate_signatures`/`plate_meta` 필드 추가
- [ ] model_io.save/load v2_plate 지원

✅ **완료 조건**: v1 모델도 로드 가능 + v2_plate 모델 저장/로드 시 plate 키 유지

### C. Plate Engine 구현 (신규 모듈)
- [ ] `plate_engine.analyze_plate_pair()` 구현
  - [ ] 정렬 (registration)
  - [ ] alpha_map + residual
  - [ ] plate mask (ring/dot/clear)
  - [ ] core/transition 분리
  - [ ] 지표 산출 + JSON 생성

✅ **완료 조건**: 흰/검 페어 입력 시 plate JSON이 항상 생성되고, core 통계가 반복 스캔에서 안정적

### D. 단일 분석 연결
- [ ] `analyze_single_sample`에 `black_bgr`/`match_id` optional 추가
- [ ] `results["plate"]` 삽입 (black_bgr 있을 때만)

✅ **완료 조건**: 기존 1장 분석 결과 변경 없음 + 2장 페어에서만 plate 추가

### E. 라우터 확장
- [ ] `/api/v7/analyze_single`에 `black_files`, `match_ids` 추가
- [ ] pair 분석 시 페어 단위 결과 반환

✅ **완료 조건**: 기존 클라이언트는 그대로 동작 + 신규 UI에서 페어 업로드 시 plate 포함 결과 확인

### F. STD 학습 확장
- [ ] `train_std_model.py`에 페어 입력 옵션 추가
- [ ] plate signatures를 model에 넣고 save_model로 저장

✅ **완료 조건**: .npz에 `plate_*` 키가 생성되고 .json에 `plate_meta`가 함께 저장됨

---

## 24. 구현 우선순위 정리

| 순서 | 작업 | 파일 |
|------|------|------|
| 1 | StdModel 필드 추가 | `std_model.py` |
| 2 | model_io v2_plate 지원 | `model_io.py` |
| 3 | plate_engine 핵심 함수 | `plate/plate_engine.py` (신규) |
| 4 | single_analyzer 확장 | `single_analyzer.py` |
| 5 | 라우터 확장 | `routers/v7.py` |
| 6 | STD 학습 확장 | `train_std_model.py` |
| 7 | 테스트 | 10세트 golden test |

---

## 25. 현재 코드 기준 핵심 확인 사항

### A. StdModel
- `@dataclass`라 확장 저장에 유리
- `slots=True`가 아니라 동적 확장 가능
- **결정**: 정식 필드 추가 (장기 운영 최적)

### B. analyze_single_sample
- 현재 `test_bgr` 1장만 받음
- **결정**: optional 인자로 `black_bgr`, `match_id` 확장

### C. 라우터
- 현재 `files: List[UploadFile]`만 받음
- **결정**: `black_files`, `match_ids` 추가 (기존 호출 유지)

---

## 26. plate_engine.py 전체 구현 코드 (v1)

### 26.1 파일 경로
```
src/engine_v7/core/pipeline/plate_engine.py
```

### 26.2 전체 코드

```python
# src/engine_v7/core/pipeline/plate_engine.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Tuple, Optional

import cv2
import numpy as np

from ..geometry.lens_geometry import detect_lens_circle


def _to_float01(bgr: np.ndarray) -> np.ndarray:
    x = bgr.astype(np.float32)
    return np.clip(x / 255.0, 0.0, 1.0)


def _resize_to_square_centered(bgr: np.ndarray, geom, out_size: int = 512) -> Tuple[np.ndarray, Any]:
    """Center+scale normalize by circle. Returns normalized image and normalized geom."""
    target_r = out_size * 0.45
    scale = target_r / max(geom.r, 1e-6)

    M = np.array([[scale, 0, out_size / 2 - geom.cx * scale],
                  [0, scale, out_size / 2 - geom.cy * scale]], dtype=np.float32)

    warped = cv2.warpAffine(bgr, M, (out_size, out_size),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    class G:
        pass
    g2 = G()
    g2.cx = out_size / 2
    g2.cy = out_size / 2
    g2.r = target_r
    return warped, g2


def _phase_align_polar(white: np.ndarray, black: np.ndarray, geom) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Polar transform + phase correlation to align rotation.
    Returns aligned black and registration info.
    """
    w_gray = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)

    R = int(geom.r)
    T = 720
    center = (geom.cx, geom.cy)
    w_p = cv2.warpPolar(w_gray, (T, R), center, geom.r, cv2.WARP_POLAR_LINEAR)
    b_p = cv2.warpPolar(b_gray, (T, R), center, geom.r, cv2.WARP_POLAR_LINEAR)

    shift, response = cv2.phaseCorrelate(w_p.astype(np.float32), b_p.astype(np.float32))
    dtheta_px = shift[0]
    dtheta_deg = (dtheta_px / T) * 360.0

    M = cv2.getRotationMatrix2D(center, -dtheta_deg, 1.0)
    black_aligned = cv2.warpAffine(black, M, (black.shape[1], black.shape[0]),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    info = {
        "method": "circle_norm+polar_phase",
        "score": float(max(0.0, min(1.0, response))),
        "dx_dy": [0.0, 0.0],
        "dtheta_deg": float(dtheta_deg),
        "notes": ""
    }
    return black_aligned, info


def _compute_alpha_map(white_bgr: np.ndarray, black_bgr: np.ndarray,
                       alpha_clip=(0.02, 0.98)) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    alpha = 1 - (Iwhite - Iblack)/255 (채널 평균 기반)
    """
    w = white_bgr.astype(np.float32)
    b = black_bgr.astype(np.float32)
    w_m = w.mean(axis=2)
    b_m = b.mean(axis=2)

    alpha = 1.0 - (w_m - b_m) / 255.0
    alpha = np.clip(alpha, alpha_clip[0], alpha_clip[1])

    # residual: reconstruct white from alpha & ink
    ink = np.where(alpha[..., None] > 0.01, b / alpha[..., None], 0)
    ink = np.clip(ink, 0.0, 255.0)
    white_pred = alpha[..., None] * ink + (1.0 - alpha[..., None]) * 255.0
    rmse = float(np.sqrt(np.mean((white_pred - w) ** 2)))

    return alpha.astype(np.float32), {
        "mode": "white_black_pair",
        "alpha_clip": [float(alpha_clip[0]), float(alpha_clip[1])],
        "residual": {"white_recon_rmse": rmse}
    }


def _radial_mask(h: int, w: int, geom, r0: float, r1: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((xx - geom.cx) ** 2 + (yy - geom.cy) ** 2) / max(geom.r, 1e-6)
    return (rr >= r0) & (rr <= r1)


def _make_plate_masks(alpha: np.ndarray, geom, cfg: dict) -> Dict[str, np.ndarray]:
    """
    v1: alpha 기반 3판(ring/dot/clear) 분리 (템플릿 없이)
    """
    h, w = alpha.shape
    r_exclude_outer = float(cfg.get("r_exclude_outer", 0.95))
    alpha_clear_th = float(cfg.get("alpha_clear_th", 0.08))

    valid = _radial_mask(h, w, geom, 0.0, r_exclude_outer)

    # clear: 중심부 + alpha 낮음
    r_clear = float(cfg.get("r_clear", 0.40))
    clear = valid & _radial_mask(h, w, geom, 0.0, r_clear) & (alpha < alpha_clear_th)

    # 인쇄 후보
    ink_candidate = valid & (~clear)

    # ring: 바깥쪽 + alpha 높음
    r_ring0 = float(cfg.get("r_ring0", 0.70))
    r_ring1 = float(cfg.get("r_ring1", r_exclude_outer))
    alpha_ring_th = float(cfg.get("alpha_ring_th", 0.55))
    ring_raw = ink_candidate & _radial_mask(h, w, geom, r_ring0, r_ring1) & (alpha >= alpha_ring_th)

    # morphology close + largest CC
    ring_u8 = (ring_raw.astype(np.uint8) * 255)
    k = int(cfg.get("ring_morph_ksize", 7))
    k = max(3, k | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ring_closed = cv2.morphologyEx(ring_u8, cv2.MORPH_CLOSE, kernel)
    ring_closed = (ring_closed > 0)

    num, labels = cv2.connectedComponents(ring_closed.astype(np.uint8))
    if num > 1:
        areas = [(labels == i).sum() for i in range(1, num)]
        best = 1 + int(np.argmax(areas))
        ring = (labels == best)
    else:
        ring = ring_closed

    # dot = 나머지
    dot = ink_candidate & (~ring)

    return {"ring": ring, "dot": dot, "clear": clear, "valid": valid}


def _split_core_transition(mask: np.ndarray, geom, cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    core = erode(mask, w), transition = mask - core
    w = max(4px, 0.015R)
    """
    w_dyn = max(4, int(0.015 * float(geom.r)))
    w_cfg = cfg.get("transition_width_px")
    if isinstance(w_cfg, int) and w_cfg > 0:
        w_dyn = w_cfg

    k = max(3, (2 * w_dyn + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    core = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0
    transition = mask & (~core)
    return core, transition


def _bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    return np.stack([L, a, b], axis=2)


def _stats_lab_alpha(lab: np.ndarray, alpha: np.ndarray, m: np.ndarray) -> Dict[str, Any]:
    if m.sum() == 0:
        return {"empty": True}
    vals = lab[m]
    a = alpha[m]
    def pct(x, p): return np.percentile(x, p, axis=0).tolist()
    return {
        "lab": {
            "mean": vals.mean(axis=0).tolist(),
            "p95": pct(vals, 95),
            "p05": pct(vals, 5),
            "std": vals.std(axis=0).tolist(),
        },
        "alpha": {
            "mean": float(a.mean()),
            "p95": float(np.percentile(a, 95)),
            "p05": float(np.percentile(a, 5)),
            "std": float(a.std()),
        }
    }


def analyze_plate_pair(
    white_bgr: np.ndarray,
    black_bgr: np.ndarray,
    cfg: dict,
    match_id: Optional[str] = None,
    geom_hint: Any = None,
) -> dict:
    """
    흰판/검은판 2장을 분석하여 plate_v1.0 JSON 반환.
    """
    # 1) geom detect
    geom_w = geom_hint if geom_hint is not None else detect_lens_circle(white_bgr)
    geom_b = detect_lens_circle(black_bgr)

    # 2) normalize
    out_size = int(cfg.get("norm_size", 512))
    w_norm, g = _resize_to_square_centered(white_bgr, geom_w, out_size=out_size)
    b_norm, _ = _resize_to_square_centered(black_bgr, geom_b, out_size=out_size)

    # 3) polar phase align
    b_aligned, reg = _phase_align_polar(w_norm, b_norm, g)

    # 4) alpha map
    alpha, alpha_meta = _compute_alpha_map(
        w_norm, b_aligned,
        alpha_clip=tuple(cfg.get("alpha_clip", (0.02, 0.98)))
    )

    # 5) plate masks
    masks = _make_plate_masks(alpha, g, cfg)

    # 6) core/transition + stats
    lab = _bgr_to_lab(w_norm)
    plates_out: Dict[str, Any] = {}

    for name in ["ring", "dot", "clear"]:
        m = masks[name]
        if name == "clear":
            core = m
            transition = None
        else:
            core, transition = _split_core_transition(m, g, cfg)

        geom_info = {
            "area_ratio": float(m.sum() / max(masks["valid"].sum(), 1)),
        }
        plates_out[name] = {"geometry": geom_info}
        plates_out[name]["core"] = _stats_lab_alpha(lab, alpha, core)
        if transition is not None:
            plates_out[name]["transition"] = _stats_lab_alpha(lab, alpha, transition)
            plates_out[name].setdefault("notes", []).append("transition_is_design_feature")

    # 7) output JSON
    return {
        "schema_version": "plate_v1.0",
        "match_id": match_id,
        "geom": {
            "cx": round(float(g.cx), 2),
            "cy": round(float(g.cy), 2),
            "r": round(float(g.r), 2),
            "r_exclude_outer": float(cfg.get("r_exclude_outer", 0.95))
        },
        "registration": reg,
        "alpha_model": alpha_meta,
        "plates": plates_out,
        "summary": {
            "expected_plates": ["ring", "dot", "clear"],
            "diagnosis_tags": ["OK"],
            "notes": "transition은 디자인 특성으로 관측만 수행"
        },
        "debug": {
            "params": {
                "transition_width": {"mode": "dynamic", "value": "max(4px, 0.015R)"},
                "outer_rim_exclude": float(cfg.get("r_exclude_outer", 0.95)),
                "alpha_clear_th": float(cfg.get("alpha_clear_th", 0.08)),
            }
        }
    }
```

### 26.3 v1 범위
- 판 분리/통계까지 확실히 구현
- dot_metrics (count/밀도) 는 v1.1에서 추가 (마스크 안정화 후)

---

## 27. configs/default.json plate 섹션

```json
{
  "plate": {
    "norm_size": 512,
    "r_exclude_outer": 0.95,
    "alpha_clear_th": 0.08,
    "alpha_clip": [0.02, 0.98],
    "r_clear": 0.40,
    "r_ring0": 0.70,
    "r_ring1": 0.95,
    "alpha_ring_th": 0.55,
    "ring_morph_ksize": 7
  }
}
```

---

## 28. 라우터 로직 상세

### 28.1 동작 규칙

```python
# black_files 없으면 기존대로 1장씩
if black_files is None:
    for f in files:
        bgr = decode_image(f)
        result = analyze_single_sample(bgr, cfg, modes)
        results.append(result)

# black_files 있으면 페어 분석
else:
    if len(files) != len(black_files):
        raise HTTPException(400, "files와 black_files 개수가 다릅니다")

    match_id_list = parse_match_ids(match_ids) if match_ids else None

    for i, (wf, bf) in enumerate(zip(files, black_files)):
        white_bgr = decode_image(wf)
        black_bgr = decode_image(bf)
        mid = match_id_list[i] if match_id_list else f"{wf.filename}_{bf.filename}"

        result = analyze_single_sample(
            white_bgr, cfg, modes,
            black_bgr=black_bgr,
            match_id=mid
        )
        results.append(result)
```

### 28.2 match_ids 처리
- 없으면: 파일 stem 조합으로 자동 생성
- 있으면: JSON 문자열 또는 CSV 파싱해서 사용

---

## 29. STD 학습 확장 상세

### 29.1 v1.0 저장 범위

| 항목 | 저장 위치 | 내용 |
|------|----------|------|
| plate_meta | model.json | params + plate summary (스칼라) |
| plate_signatures | model.npz | v1에서는 비움 또는 최소 키만 |

### 29.2 v1.0 plate_meta 예시

```json
{
  "plate_params": {
    "transition_width": "max(4px, 0.015R)",
    "outer_rim_exclude": 0.95,
    "alpha_clear_th": 0.08
  },
  "plate_summary": {
    "ring": {
      "core": {
        "alpha": {"mean": 0.78, "p05": 0.55, "p95": 0.91, "std": 0.08},
        "lab": {"mean": [45.2, 6.1, 18.9], "std": [3.1, 2.2, 4.5]}
      }
    },
    "dot": {
      "core": {
        "alpha": {"mean": 0.34, "p05": 0.12, "p95": 0.58, "std": 0.10},
        "lab": {"mean": [52.1, 3.2, 12.4], "std": [4.2, 1.8, 3.1]}
      }
    }
  }
}
```

### 29.3 v1.1+ 확장 예정
- plate_signatures에 radial 벡터 저장
- 비교에서 plate별 radial ΔE 계산

---

## 30. 최종 개발 체크리스트 (완료 조건)

### (A) Plate Engine v1 동작 확인
- [ ] 흰/검 페어에서 plate JSON 생성

✅ **완료 조건**:
- `plates.ring.core`와 `plates.dot.core`가 empty가 아님
- alpha 평균이 ring > dot > clear 경향

### (B) API 연계
- [ ] 라우터에 `black_files` 추가 (선택 필드)

✅ **완료 조건**:
- 기존 1장 업로드 동작 그대로
- black_files 포함 업로드 시 결과에 plate 추가

### (C) STD 저장 확장
- [ ] StdModel에 plate 필드 추가
- [ ] model_io 저장/로드 확장 (v2_plate)

✅ **완료 조건**:
- .json에 `has_plate`/`plate_meta`가 저장됨
- v1 모델도 그대로 로드 가능

---

## 31. 버전 로드맵

| 버전 | 범위 | 상태 |
|------|------|------|
| v1.0 | 판 분리 + 통계 + API 연계 | 구현 예정 |
| v1.1 | dot_metrics (count/밀도/결손) | 계획 |
| v1.2 | plate별 radial signature 저장 | 계획 |
| v2.0 | STD 비교에서 plate별 ΔE | 계획 |
