# Production Lens Color System v1.0.7 - 작업자 인수인계

> Contact Lens 색 추출 파이프라인 (Production v1.0.7) 기술 문서

**정본 파일**: `production_lens_color_system_v107_final.py`
**버전**: v1.0.7 (최종 안전화)
**작성일**: 2026-02

---

## 0. 한 줄 요약

정상 렌즈(시중 제품) 이미지에서 **"눈으로 보이는 디자인 색"**을 최대한 정확히 추출하기 위한 파이프라인이며, **white-only 스캔(black 없음)에서도 하이라이트/배경 누출을 안정적으로 제거**하도록 v1.0.7까지 완결됨.

---

## 1. 배경: v7 트랙과의 관계 (혼선 방지)

### v7(intrinsic_color/alpha) 트랙
- **목적**: white/black plate 페어 기반 alpha/투명도 포함 색 추정, 업로드/검증/동시성/plate_gate 안정화 등 **"서버 운영" 중심**
- **현재 상태**: 참고 지식으로만 유지

### Production(색 추출) 트랙
- **목적**: 실제 제품 이미지에서 **"디자인 대표색/비율"**을 정확히 뽑는 것이 최우선
- **특징**:
  - 입력이 white-only일 수 있음
  - 크롭/렌즈 위치가 매번 달라도 견딜 수 있어야 함
- **결과**: v1.0.7에서 "디자인 색 보호 + 배경 누출 제거 + 경고 의미 분리"까지 완료

> **결론**: v7 서버/alpha 트랙과 결합하지 말고, 색 추출은 **Production v1.0.7을 기준**으로 운영/확장

---

## 2. 현재 기준 파일 (정본)

```
production_lens_color_system_v107_final.py
```

- 이 파일이 **Production v1.0.7 완결본(정본)**
- 로컬/서버에 복사 시 **"실행 중 파일 1개 고정"** 필요

### 운영 핵심 룰
> 돌아가는 파일 1개를 고정하고, **로그로 파일 경로+해시를 찍어서 버전 혼선 차단**

---

## 3. 핵심 로직 요약 (v1.0.7 스펙)

### 3.1 색 추출 아키텍처 (기본 흐름)

```
1. 전경(fg_mask) 추출 (동적 임계값)
   ↓
2. ROI 강제: largest connected component만 사용 (먼지/물띠 제거)
   ↓
3. Lab 변환 + L down-weight (조명 변화에 강건)
   ↓
4. Chroma 기반 샘플링 + KMeans
   ↓
5. ROI 전체 픽셀 기준으로 라벨 할당/비율 계산 (샘플링 비율 오판 문제 해결)
   ↓
6. ΔE 기반 클러스터 병합 + small cluster 처리
   ↓
7. 대표색(center)은 ΔE 거리 기반 trimmed mean (치명 버그 해결)
   ↓
8. 운영 가드레일(coverage/warnings) 생성
```

### 3.2 v1.0.7의 "핵심 업그레이드"

#### A) 2D 공간 기반 하이라이트 제거 (핵심)

| 구분 | 설명 |
|------|------|
| **코어 + 헤일로** | 중심부 반사/번짐을 **"공간 팽창(dilate)"**로 제거 |
| **이전 문제** | 기존 1D(flat) 필터로는 번짐(halo) 제거가 불가 |
| **해결** | v1.0.6에서 2D spatial 필터로 해결 |

#### B) 배경 누출 제거를 "밝고 저채도"가 아니라 배경 유사도(ΔE) 기반으로 변경 (v1.0.7 핵심)

| 항목 | 값/방식 |
|------|---------|
| **bg_lab 추정** | ROI 주변 링(1.02~1.08R)에서 추정 (조명 그라데이션 대응) |
| **bg_leak 조건** | `(ΔE_to_bg < 8) & (chroma < 15)` ← **AND 조건으로 디자인 색 보호** |

**효과**:
- Natural 톤 밝은 크림/베이지 디자인은 배경과 ΔE가 충분히 크면 **보존**
- 진짜 배경 누출만 정확히 **제거**

#### C) 경고 의미를 "반사 vs 배경정리"로 분리 (운영 혼선 제거)

| 지표 | 의미 |
|------|------|
| `specular_ratio` | `(core + halo) / ROI` → **반사(조명 문제) 경고** |
| `bg_leak_ratio` | `bg_leak / ROI` → **배경 정리 정보(INFO)** |

**결과**:
- **WARNING** = 반사 과다 (실제 조명/노출 문제)
- **INFO** = 배경 누출 정리 (정상 동작)

---

## 4. 운영 로그/경고 해석 가이드 (작업자용)

### 4.1 핵심 지표

| 지표 | 의미 |
|------|------|
| `ROI%` | 전경 추출/크롭 품질 |
| `filtered% vs ROI%` | 필터링이 과도한지 판단 |
| `specular_ratio` | 반사 영향 (조명/노출 이슈) |
| `bg_leak_ratio` | 배경 누출 제거량 (정상 처리 정보) |
| `components` | 노이즈/먼지 수준 |
| `clusters k→k(merged n)` | 세그먼트 안정성 |

### 4.2 경고 의미

| 코드 | 레벨 | 의미 |
|------|------|------|
| `HIGHLIGHT_VERY_HIGH` | WARNING | 반사(specular) 과다 → 스캔 조명/노출/반사 억제 필요 |
| `HIGHLIGHT_HIGH` | INFO | 반사 다소 높음 (관찰) |
| `BG_LEAK_REMOVED` | INFO | 배경 누출 정리 완료 (문제 아님) |

> **운영자가 실수하기 쉬운 포인트**:
> `BG_LEAK_REMOVED`는 **"정상적으로 잘 정리했다"**는 의미이지 불량 경고가 **아님**

---

## 5. 작업자에게 요구되는 유지보수 포인트

### 5.1 파라미터 튜닝은 어디를 건드려야 하나

| 상황 | 튜닝 대상 |
|------|----------|
| 반사 과다 WARNING가 너무 자주 뜬다 | `core_L`, `halo_L`, `halo_dilate_ratio` 튜닝 (조명 변화에 따라) |
| 배경 누출이 남는다 | `bg_delta_e_threshold` (기본 8) 또는 링 추정 방식 (링 폭/반경) 튜닝 |
| 디자인 색이 날아간다 | bg_leak AND 조건에서 `chroma<15`를 더 낮추거나, ΔE threshold를 낮춤 |

### 5.2 절대 바꾸면 안 되는 것 (회귀 위험)

| 금지 사항 | 이유 |
|----------|------|
| **비율 계산을 샘플링 기준으로 하지 말 것** | KMeans 학습은 샘플링, **비율은 ROI 전체 픽셀로 계산**이 핵심 |
| **대표색 계산에서 "채널별 정렬 trimmed mean" 금지** | 반드시 **ΔE 거리 기반 trim** 유지 |
| **merge 이후 size 재계산 없이 small 판단 금지** | merge 후 **root_percentages 재계산 필수** |

---

## 6. 핵심 함수 참조

### `filter_highlights_spatial()` (lines 183-285)
- 2D 공간 기반 하이라이트 제거
- 코어 + 헤일로 + 배경 유사도(ΔE & chroma) AND 조건

### `trimmed_mean_lab()` (lines 68-97)
- ΔE 거리 기반 trimmed mean (치명 버그 수정)
- median Lab 기준 거리 계산 후 양 끝 k개 제거

### `merge_similar_clusters()` (lines 385-475)
- ΔE 기반 클러스터 병합
- **병합 후 크기 재계산** (root_percentages) 필수

### `extract_colors_production_v107()` (lines 706-1017)
- 메인 색 추출 함수
- 자동 복구 (dynamic_low_ratio 증가) 포함

---

## 7. v7 트랙과의 통합 금지 사유

| 항목 | Production v1.0.7 | v7 트랙 |
|------|------------------|---------|
| **입력** | white-only 가능 | white+black 페어 필수 |
| **목표** | 디자인 색/비율 | alpha/투명도 |
| **배경 처리** | ΔE 기반 bg_leak | plate 기반 |
| **하이라이트** | 2D spatial | 다름 |

> 통합 시 **양쪽 모두 품질 저하** 위험

---

## 8. 관련 파일

- **정본**: `production_lens_color_system_v107_final.py` (루트)
- **v7 트랙**: `src/engine_v7/` (참고용)
- **이 문서**: `docs/01-plan/knowledge/production-lens-color-system-v107.md`

---

**Last Updated**: 2026-02-04
