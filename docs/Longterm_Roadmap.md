# Color Meter v7 장기 로드맵 (Long-term Roadmap)

**작성일**: 2026-01-17
**현재 상태**: P0/P1/P2 완료 (안정화 단계)
**목적**: 차세대 아키텍처 개선 방향 정의

---

## 개요

현재 시스템은 안정화되었지만, 두 가지 근본적인 한계가 있습니다:

1. **Simulation의 단순화**: `area_ratio` 기반 스칼라 혼합 → 실제 겹침/농도 변화 반영 불가
2. **관측색 의존성**: 배경/조명에 영향받는 관측색으로 클러스터링 → 고유색 추출 불가

다음 두 가지 방향이 장기적으로 시스템을 한 단계 도약시킬 수 있습니다.

---

## 방향 A: 마스크 기반 시뮬레이션 (Mask-based Pixel Synthesis)

### 현재 방식의 한계

```python
# 현재: area_ratio 기반 스칼라 혼합
composite_lab = sum(ink_i.mean_lab * ink_i.area_ratio for ink_i in inks)
```

**문제점:**

- ❌ Overlap 영역 미반영 (겹친 부분은 더 진해져야 함)
- ❌ Radial zone별 기여도 무시 (중심/외곽 농도 차이)
- ❌ 실제 dot 분포 패턴 손실

### 제안: 마스크 기반 합성

```python
# 제안: 마스크 유니온에서 픽셀 샘플링
union_mask = mask_1 | mask_2 | mask_3
overlap_mask = mask_1 & mask_2  # 자동으로 겹침 감지

# 각 마스크 영역에서 실제 픽셀 추출
pixels_union = lab_map[union_mask]
composite_lab = pixels_union.mean(axis=0)  # 실제 관측값 평균
```

**장점:**

- ✅ Overlap이 자연스럽게 "추가 혼합"으로 반영
- ✅ Zone별 기여도 자동 반영 (중심이 많으면 자동으로 가중)
- ✅ 체감색이 설명 가능해짐 (실제 픽셀 합성)

---

### 구현 계획 (Phase A)

#### Phase A1: 프로토타입 (2-3일)

**파일**: `src/engine_v7/core/simulation/mask_based_compositor.py` (NEW)

```python
def composite_from_masks(
    lab_map: np.ndarray,  # (T, R, 3) polar Lab map
    color_masks: Dict[str, np.ndarray],  # {color_id: mask}
    downsample: int = 4,  # 성능 최적화
) -> Dict[str, Any]:
    """
    Mask union에서 실제 픽셀 샘플링하여 composite color 계산.

    Returns:
        {
            "composite_lab": [L, a, b],
            "overlap_regions": [...],  # 겹침 영역 분석
            "zone_contributions": {...},  # zone별 기여도
        }
    """
```

**작업:**

1. ✅ 마스크 유니온 생성
2. ✅ Overlap 영역 감지 및 분석
3. ✅ Downsampling으로 성능 최적화
4. ✅ 기존 `area_ratio` 방식과 비교

#### Phase A2: 통합 (1주)

**업데이트 파일:**

- `color_simulator.py`: 새 compositor 옵션 추가
- `single_analyzer.py`: Feature flag로 전환 가능하게

```python
# Config 옵션
{
    "simulation": {
        "method": "mask_based",  # or "area_ratio" (기존)
        "downsample": 4,
    }
}
```

#### Phase A3: 검증 (1-2주)

**검증 항목:**

1. 실제 샘플과 시뮬레이션 비교 (ΔE 측정)
2. Overlap 많은 샘플에서 정확도 향상 확인
3. 성능 영향 측정 (downsampling 최적화)

**성공 기준:**

- Overlap 있는 샘플: ΔE 개선 20% 이상
- 단일 잉크: ΔE 동등 이상
- 성능: 10% 이내 slowdown

---

## 방향 B: Intrinsic Color Clustering (고유색 클러스터링)

### 현재 방식의 한계

```python
# 현재: 관측색(observed color)에서 직접 클러스터링
samples = lab_map[roi_mask]  # 배경/조명 영향 포함
labels = kmeans(samples, k=3)
```

**문제점:**

- ❌ 배경색/조명 변화에 민감
- ❌ 반사/클립/그림자 영향
- ❌ Hard Gate/Soft Gate로 간접 보정 필요

### 제안: W/B Pair에서 Intrinsic 역산

**핵심 아이디어:**

Plate W/B 쌍에서:

```
observed_white = alpha * intrinsic + (1-alpha) * bg_white
observed_black = alpha * intrinsic + (1-alpha) * bg_black
```

두 식을 연립하면:

```
intrinsic = (observed_white - observed_black) / (alpha_white - alpha_black)
```

**장점:**

- ✅ 배경 독립적 고유색 추출
- ✅ Hard Gate/Soft Gate 의존도 감소
- ✅ Simulation 정확도 자동 향상

---

### 구현 계획 (Phase B)

#### Phase B1: 이론 검증 (1주)

**파일**: `notebooks/intrinsic_color_extraction.ipynb` (NEW)

**작업:**

1. W/B 쌍에서 intrinsic + alpha 역산 구현
2. 실제 샘플로 검증 (물리적으로 타당한가?)
3. 수치 안정성 확인 (alpha 값이 너무 작을 때)

#### Phase B2: 클러스터링 적용 (2-3주)

**파일**: `src/engine_v7/core/measure/segmentation/intrinsic_clustering.py` (NEW)

```python
def cluster_intrinsic_colors(
    white_bgr: np.ndarray,
    black_bgr: np.ndarray,
    alpha_map: np.ndarray,
    cfg: Dict[str, Any],
    k: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Intrinsic color space에서 k-means 클러스터링.

    Returns:
        - color_masks: 기존과 동일한 포맷
        - metadata: intrinsic_colors, alpha_maps 추가
    """
```

**핵심 과제:**

1. ✅ 수치 안정성 (alpha ≈ 0인 영역 처리)
2. ✅ 배경 추정 정확도 (bg_white, bg_black)
3. ✅ Outlier 필터링 (비물리적 값 제거)

#### Phase B3: 통합 및 비교 (2주)

**업데이트 파일:**

- `color_masks.py`: intrinsic clustering 옵션 추가
- Feature flag로 기존 방식과 전환 가능

```python
# Config 옵션
{
    "v2_ink": {
        "clustering_method": "intrinsic",  # or "observed" (기존)
        "intrinsic_alpha_min": 0.1,  # 수치 안정성
    }
}
```

**검증:**

1. 다양한 조명 조건에서 일관성 확인
2. Hard Gate 없이도 정확도 유지되는지
3. Simulation과 결합 시너지 확인

---

## 우선순위 및 타임라인

### Short-term (1-2개월)

**우선순위: A1 → A2**

- 이유: 구현 난이도 낮음, 즉각적 효과 가능
- 목표: Mask-based simulation 프로토타입 완성

### Mid-term (3-6개월)

**우선순위: A3 + B1**

- A3: Mask simulation 검증 및 안정화
- B1: Intrinsic extraction 이론 검증

### Long-term (6-12개월)

**우선순위: B2 → B3**

- Intrinsic clustering 본격 구현
- A + B 결합 시너지 확인

---

## 시너지 효과 (A + B 결합)

```
Intrinsic Clustering → 배경 독립적 고유색
         ↓
Mask-based Simulation → 실제 픽셀 합성
         ↓
최종 결과: 물리 기반 정확도 + 설명 가능성
```

**예상 효과:**

- Simulation ΔE: 30-50% 개선
- Gate 의존도: 50% 감소
- 복잡한 overlap 패턴 처리 가능

---

## 위험 관리

| 위험                | 완화 방안                              |
| ------------------- | -------------------------------------- |
| Phase A 성능 저하   | Downsampling 적극 활용                 |
| Phase B 수치 불안정 | Alpha min threshold, outlier filtering |
| 기존 시스템 호환성  | Feature flag로 점진적 전환             |
| 검증 데이터 부족    | Ground truth 샘플 수집 계획 필요       |

---

## 의사결정 지점 (Decision Points)

### Checkpoint 1 (A1 완료 후)

- **질문**: Mask-based simulation이 실제로 더 정확한가?
- **기준**: Overlap 샘플에서 ΔE \u003c 현재 방식

### Checkpoint 2 (B1 완료 후)

- **질문**: Intrinsic 역산이 물리적으로 타당한가?
- **기준**: 다양한 조명에서 intrinsic 일관성

### Checkpoint 3 (A3 + B2 완료 후)

- **질문**: 두 방향을 결합할 가치가 있는가?
- **기준**: 시너지 효과 \u003e 개별 효과 합

---

## 다음 단계

### 즉시 실행 가능 (Quick Win)

1. **Phase A1 프로토타입** (2-3일)
   - 기존 코드 활용도 높음 (마스크 이미 존재)
   - 검증 쉬움 (기존 결과와 비교)

2. **Phase B1 이론 검증** (1주, 병렬 가능)
   - Notebook으로 빠른 실험
   - 물리적 타당성만 확인

### 승인 필요

- **리소스**: A1+B1 합쳐서 약 2주 분량
- **위험도**: 낮음 (기존 시스템 변경 없음)
- **ROI**: 높음 (두 방향 모두 가능성 확인)

---

## 참고 자료

**관련 모듈:**

- `color_masks.py`: 마스크 생성
- `color_simulator.py`: 시뮬레이션
- `plate_engine.py`: W/B alpha 맵

**관련 논문/개념:**

- Intrinsic image decomposition
- Alpha matting theory
- Color mixing models

**기존 안정화 완료:**

- [P0/P1/P2 Walkthrough](file:///C:/Users/interojo/.gemini/antigravity/brain/1df7e6e1-2c07-4927-bca8-a4667fa933ab/walkthrough.md)

---

**문서 버전**: 2026-01-17
**상태**: 계획 단계 (미승인)
**다음 리뷰**: Phase A1 프로토타입 완성 후

# Color Meter v7 — Long-term Roadmap Review & Recommendations
**Date**: 2026-01-17
**Scope**: Review of `Longterm_Roadmap.md` with concrete, code-level recommendations for the next long-term leap (A: mask-based synthesis, B: intrinsic clustering, plus alternative directions).

> Assumption: The previously discussed short-term fixes (Hard/Soft Gate stability, confidence scaling, simulation metadata/UI, composite, config normalization, etc.) have already been implemented.

---

## 1) What’s strong in the roadmap
### Direction A — Mask-based Pixel Synthesis
- The roadmap correctly targets the biggest limitation of scalar `area_ratio` mixing: it cannot represent **overlap**, **radial contribution**, or **pattern distribution**.
- Phase structure (A1 prototype → A2 integration → A3 validation) is practical and low-risk.

### Direction B — Intrinsic Color Clustering
- The goal is right: reduce sensitivity to background/illumination by moving clustering into a “background-independent” space.
- The roadmap’s emphasis on numeric stability and feature flags is correct.

### Decision points
- Checkpoints (A1, B1, A3+B2) create a healthy “stop/ship” cadence.

---

## 2) Critical correction: B’s intrinsic formula is not implementable as written
In the current roadmap, the intrinsic derivation is written as:

```text
observed_white = alpha * intrinsic + (1-alpha) * bg_white
observed_black = alpha * intrinsic + (1-alpha) * bg_black
intrinsic = (observed_white - observed_black) / (alpha_white - alpha_black)
```

This is risky because `alpha_white` and `alpha_black` are typically **not separately defined** in a physically consistent way. In most pipelines, alpha is a *property of the ink/print + optics*, not “per-background”. If `alpha_white == alpha_black`, the denominator collapses.

### Recommended intrinsic approach (robust and implementable)
Work in **Linear RGB** (not gamma sRGB), and treat White/Black as:
- **(1) intrinsic estimation from a single background** given alpha
- **(2) consistency check using the other background** to measure reliability

#### Intrinsic (single-background inversion)
```python
# in linear RGB space
intrinsic_lin = (observed_lin - (1 - alpha) * bg_lin) / max(alpha, eps)
intrinsic_lin = clip(intrinsic_lin, 0, 1)
```

#### Consistency check (validation, not solving)
Compute `intrinsic_from_white` and `intrinsic_from_black`, then:
- `consistency = ΔE(intrinsic_white, intrinsic_black)` (or RGB distance)
- Use consistency to produce a **reliability map**:
  - reliability ↓ if alpha small
  - reliability ↓ if intrinsic inversion goes out-of-gamut (clipping)
  - reliability ↓ if mask_artifact/leak high

This turns W/B into a powerful QA tool instead of a brittle algebraic solve.

---

## 3) Missing “hard deliverable” that A depends on: mask/label data contract
Direction A (pixel synthesis) cannot be reliable unless simulation receives one of:

1) `label_map_polar (T×R)`: int map (cluster id per pixel), **or**
2) `cluster_masks_polar`: dict of boolean masks (per cluster), preferably compressed (RLE)

### Concrete contract recommendation
Add a single field under the single analysis result:

```json
"ink": {
  "clusters": [...],
  "masks": {
    "space": "polar",
    "shape": [T, R],
    "encoding": "rle",
    "label_map_rle": "...",          // preferred
    "cluster_masks_rle": { ... },  // optional
    "downsample": 2                  // optional for UI/debug payloads
  }
}
```

Why this matters long-term:
- Enables A without re-running segmentation.
- Enables “observed-on-black measurement” using the exact same mask basis.
- Enables future overlay models, GT harnessing, and offline evaluation.

**Where to implement**
- Producer: `src/engine_v7/core/measure/segmentation/color_masks.py` (where masks are created)
- Carrier: `src/engine_v7/core/pipeline/single_analyzer.py` (include in output)
- Consumer: `src/engine_v7/core/simulation/color_simulator.py` (mask-based compositor)

---

## 4) Direction A: how to implement mask-based synthesis without turning it into a slow monster
### A.1 Minimal v1 compositor (fast, useful, consistent)
For the initial mask-based composite, assume *non-overlap* between clusters (or choose “winner takes all” in overlap):
- Construct a “painted pixel” image in linear RGB:
  - for each pixel, pick cluster id (or none)
  - if none, it’s background
  - if some, it’s cluster color (possibly alpha-aware later)
- Downsample early (e.g., 2–8×) and compute mean/trimmed mean.

#### Suggested function placement
- File: `src/engine_v7/core/simulation/color_simulator.py`
- New function:
```python
def composite_from_label_map(
    label_map,          # (T,R) int32
    cluster_lin_rgb,     # dict: id -> (3,)
    bg_lin_rgb,          # (3,)
    downsample=4,
    reduce="trimmed_mean"
):
    ...
```

### A.2 Overlap strategy (make it explicit)
Roadmap correctly highlights overlap, but you must define *what overlap means* in data:
- Is overlap “multiple masks true at same pixel”?
- Or is it “two inks printed at same location” (true overlay)?
- Or “segmentation uncertainty” (artifact of clustering)?

Pick one and encode in outputs:
- `overlap_map` (bool) or `overlap_ratio`
- If overlap is real overlay, you need a layer order or a commutative blend model.

### A.3 Performance guardrails
Set budgets early so the feature doesn’t get disabled in production:
- For 720×260 polar:
  - Downsample 4 → ~180×65 ≈ 11.7k pixels (cheap)
- Target: compositor < 10–15 ms per sample (numpy-only, vectorized)

---

## 5) Direction B: intrinsic clustering that won’t explode numerically
### B.1 Build an intrinsic feature space with reliability weights
Instead of directly clustering intrinsic RGB/Lab equally:
- Compute `(intrinsic, reliability)` per pixel
- Only sample pixels above a reliability threshold for clustering
- Keep the old observed-space clustering as fallback

Recommended features:
- `intrinsic_lab` (or intrinsic linear RGB)
- optional: `alpha`, `radial r`, `distance_to_center`, `plate_gate_quality`

### B.2 Feature-flag rollout (must have)
Config idea in roadmap is good; keep it:
```json
"v2_ink": {
  "clustering_method": "intrinsic",  // or "observed"
  "intrinsic_alpha_min": 0.15,
  "intrinsic_consistency_max_de": 8.0
}
```

### B.3 Success criteria (tighten)
For B, don’t only check ΔE improvement; also check:
- “candidate count” stability vs expected_k
- “forced_topk frequency” does not spike
- “rare light inks survive” rate improves (recall)

---

## 6) A + B synergy: the clean architecture
The best long-term architecture is:

1) **Plate module** produces:
   - `alpha_map_polar`
   - `bg_white_lab`, `bg_black_lab`
   - `gate masks` (dot/ring/valid/core)
2) **Intrinsic module** (new) produces:
   - `intrinsic_map_polar` + `reliability_map_polar`
3) **Segmentation module** samples from:
   - hard gate masks ∩ high reliability
4) **Simulation module** supports:
   - `area_ratio` (legacy)
   - `mask_based` (A)
   - `alpha_aware` (v2)
   - optional: overlay blending (future)

This avoids “plate ↔ segmentation ↔ simulation” circular coupling.

---

## 7) Alternative “different direction” that often wins in practice: Hybrid residual learning
If you can collect ground truth (measured Lab) for a modest set of samples:
- Keep A/B as physics-informed baseline
- Train a small model to predict residual `ΔLab`:
  - Inputs: predicted Lab, alpha stats, coverage stats, radial features, warnings
  - Outputs: correction vector

Benefits:
- Huge ΔE gains with minimal complexity
- Robust to camera/lighting drift
- Keeps explainability (physics + small correction)

---

## 8) Two-week execution package (high leverage)
### Week 1 — Make A possible end-to-end
1) Add mask/label contract into output (label_map + RLE)
2) Implement `mask_based` compositor with downsampling
3) Add composite output (on_white/on_black) + confidence metadata

### Week 2 — Make B measurable, not theoretical
4) Notebook: intrinsic inversion + consistency metrics
5) Implement reliability-weighted sampling prototype behind feature flag
6) Checkpoint:
   - overlap-heavy samples: ΔE00 improves ≥ 20%
   - non-overlap samples: ΔE00 same or better
   - runtime within budget

---

## 9) Checklist by file (suggested edits)
### `src/engine_v7/core/measure/segmentation/color_masks.py`
- Export `label_map_polar` or per-cluster masks (compressed)
- Include mask meta: shape, space, encoding, downsample

### `src/engine_v7/core/pipeline/single_analyzer.py`
- Carry mask payload into the final `results["ink"]["masks"]`
- Include simulation config in `_meta` for reproducibility

### `src/engine_v7/core/simulation/color_simulator.py`
- Add `mask_based` compositor (fast v1)
- Add `reduce` strategies: mean, trimmed mean
- Output: `composite` plus `model` and `confidence` blocks

### (New) `src/engine_v7/core/intrinsic/intrinsic_color.py`
- Functions: `invert_intrinsic()`, `compute_consistency()`, `build_reliability_map()`

---

## 10) Appendix: reference pseudocode snippets
### RLE encoding (label map)
```python
def rle_encode_int_map(arr):
    # encode flattened int map into (value, run) pairs
    ...
```

### Mask-based composite (vectorized idea)
```python
# label_map_ds: (t,r)
# lin_rgb: (K,3), bg_lin: (3,)
pix = bg_lin[None,None,:].repeat(t,0).repeat(r,1)
for k in ids:
    pix[label_map_ds == k] = lin_rgb_for_id[k]
composite_lin = trimmed_mean(pix.reshape(-1,3))
```

---

### Final note
Your roadmap is directionally excellent. The big win is **turning B into a stability-first intrinsic inversion + consistency QA**, and unlocking A by enforcing a **mask/label data contract** so simulation can actually operate on pixels. Once those two are in place, everything else becomes incremental rather than fragile.

---

## [추가] 11. Phase 6 이후 현황 반영 (2026-01-17)

### 현재 구현된 인프라 자산

Phase 6 리팩토링 및 P2 작업 완료로 다음 자산들이 이미 준비되어 있습니다:

#### A. Mask 데이터 계약 (이미 부분 구현됨)

**`plate_engine.py`** (L1319-1331):
```python
"_masks": {
    "ink_mask": masks.get("ink_mask"),
    "ink_mask_core": masks.get("ink_mask_core"),
    "dot": masks.get("dot"),
    "ring": masks.get("ring"),
    "clear": masks.get("clear"),
    "valid": masks.get("valid"),
    "ink_mask_core_polar": ink_mask_core_polar,  # ← 이미 polar 변환됨
},
```

**활용 가능**:
- `ink_mask_core_polar`이 이미 `(T, R)` boolean mask로 생성됨
- Direction A의 mask-based compositor가 바로 사용 가능

#### B. Lightweight Gate Path (P2 완료)

**`plate_gate.py`** 신규 모듈:
```python
def extract_plate_gate(...) -> Dict[str, Any]:
    """Fast extraction without full plate analysis."""
    return {
        "ink_mask_core_polar": ...,  # (T, R) boolean
        "valid_polar": ...,
        "geom": LensGeometry,
        "gate_quality": {...},
    }
```

**활용 가능**:
- Direction A 프로토타입 시 full analysis 없이 mask만 빠르게 추출 가능
- 성능 최적화에 유리

#### C. Alpha Map 계산 (이미 구현됨)

**`plate_engine.py:_compute_alpha_map()`** (L493-555):
- W/B pair에서 alpha 역산 이미 구현
- Direction B의 intrinsic inversion 기반 코드로 활용 가능

```python
alpha_ch = 1.0 - (diff_raw / final_denom_val)
alpha_raw = np.median(alpha_ch, axis=2)
```

---

### Direction A 구현 시 활용할 기존 코드

| 필요 기능 | 기존 코드 위치 | 상태 |
|----------|--------------|------|
| Polar mask 변환 | `plate_engine._mask_to_polar()` | ✅ 구현됨 |
| Radial mask 생성 | `plate_engine._radial_mask()` | ✅ 구현됨 |
| Color masks | `color_masks.build_color_masks_v2()` | ✅ 구현됨 |
| Polar transform | `radial_signature.to_polar()` | ✅ 구현됨 |
| Lab 변환 | `utils.to_cie_lab()` | ✅ 구현됨 |

**즉시 구현 가능한 compositor 위치**:
```
src/engine_v7/core/simulation/color_simulator.py
  └── composite_from_masks()  # 신규 함수 추가
```

---

### Direction B 구현 시 활용할 기존 코드

| 필요 기능 | 기존 코드 위치 | 상태 |
|----------|--------------|------|
| Alpha map 계산 | `plate_engine._compute_alpha_map()` | ✅ 구현됨 |
| Background 추정 | `plate_engine._estimate_spatial_bg()` | ✅ 구현됨 |
| Dynamic radii | `plate_engine.detect_dynamic_radii()` | ✅ 구현됨 |
| Clustering | `color_masks._cluster_ink_masks_full()` | ✅ 구현됨 |

**신규 모듈 권장 위치**:
```
src/engine_v7/core/intrinsic/
  ├── __init__.py
  ├── intrinsic_color.py      # invert_intrinsic(), compute_consistency()
  └── reliability_map.py      # build_reliability_map()
```

---

### 수정된 타임라인 (기존 인프라 활용 시)

| Phase | 원래 예상 | 수정 예상 | 이유 |
|-------|----------|----------|------|
| A1 프로토타입 | 2-3일 | **1-2일** | `ink_mask_core_polar` 이미 존재 |
| A2 통합 | 1주 | **3-4일** | `single_analyzer` 구조 이미 정리됨 |
| B1 이론 검증 | 1주 | **3-4일** | `_compute_alpha_map` 활용 가능 |

---

### 추가 고려사항

#### 1. plate_gate.py vs plate_engine.py 통합 결정

현재 중복 코드 존재 (5개 헬퍼 함수). 장기 로드맵 구현 전 결정 필요:

| 옵션 | 장점 | 단점 |
|-----|------|------|
| 분리 유지 | 독립적 테스트, lazy import 효과 | 중복 유지보수 |
| 공통 헬퍼 분리 | DRY 원칙, 일관성 | 추가 모듈 생성 |
| plate_gate를 plate_engine 내부로 | 단일 모듈 | 모듈 크기 증가 |

**권장**: 공통 헬퍼를 `plate/_helpers.py`로 분리 후 Direction A/B 구현

#### 2. Config 구조 선정리 필요

Direction A/B 구현 전 config 정리 권장:

```json
{
  "simulation": {
    "method": "area_ratio",  // → "mask_based" (A)
    "downsample": 4
  },
  "clustering": {
    "method": "observed",    // → "intrinsic" (B)
    "intrinsic_alpha_min": 0.15
  }
}
```

현재 config에 legacy 파라미터가 혼재되어 있으므로 `Legacy_Cleanup_Plan.md`의 Config 파편화 정리와 병행 권장.

#### 3. 테스트 데이터 준비

Direction A/B 검증을 위한 Ground Truth 샘플 수집 계획:

| 샘플 유형 | 필요 수량 | 용도 |
|----------|----------|------|
| Overlap 있는 샘플 | 20+ | A 검증 (mask-based vs area_ratio) |
| 다양한 조명 조건 | 10+ | B 검증 (intrinsic 일관성) |
| 측정된 Lab 값 | 50+ | 전체 ΔE 검증 |

---

## 요약: 단기 실행 항목 (Phase 6 이후)

### 즉시 가능 (0-1주)

1. **A1 프로토타입**: `composite_from_masks()` 구현
   - 입력: `ink_mask_core_polar` (이미 존재)
   - 출력: `composite_lab`, `overlap_ratio`

2. **헬퍼 분리**: `plate/_helpers.py` 생성
   - 중복 코드 제거
   - A/B 구현 기반 마련

### 단기 (1-2주)

3. **B1 이론 검증**: `intrinsic_color.py` 프로토타입
   - `_compute_alpha_map()` 활용
   - consistency metric 계산

4. **Config 정리**: v2 설정 구조화
   - `Legacy_Cleanup_Plan.md` Task 1.3과 병행

---

**문서 업데이트**: 2026-01-17
**상태**: Phase 6 이후 현황 반영 완료
