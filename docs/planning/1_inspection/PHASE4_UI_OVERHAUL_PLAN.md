# Phase 4: 차세대 UI/UX 및 기능 확장 구현 계획

**작성일:** 2025-12-12
**수정일:** 2025-12-12 (기술적 정밀도 및 알고리즘 표준화 반영)
**목표:** 단순 웹 뷰어를 전문 분석 장비 수준의 대시보드(Lab Software)로 격상
**범위:** UI 레이아웃 재설계, 정밀 분석(CIEDE2000, r_norm), 도트 커버리지, DB 구축

---

## 1. 시스템 아키텍처

### A. Frontend
- **Framework:** **Bootstrap 5.3** (Grid System, Card, Nav Tabs, Form Controls)
- **Visualization:**
  - **Chart.js v3.9.1**: Radial Profile, Delta E Trend
  - **Canvas API + Panzoom.js**: 메인 이미지 뷰어 (Zoom/Pan), 실시간 오버레이(Ring/Sector Grid)
- **Interaction:** Vanilla JS (ES6+) - 설정 변경 시 Canvas 즉시 리드로잉(Preview)

### B. Backend (FastAPI)
- **Schema 확장:** 메타데이터, 잉크 설정, 수동 반지름 보정값 수신
- **Logic:**
  - `PreProcessor`: 배경 마스킹(Threshold/Otsu), 초기 중심점/반경 자동 검출
  - `AngularProfiler`: 0~360도 섹터 분할 통계 (좌표계 표준화 적용)
  - `InkColorExtractor`: K-means 기반 잉크 분리 및 Dot Coverage 계산
  - `ColorEvaluator`: **CIEDE2000** 알고리즘 적용
- **DB:** SQLite (분석 이력, 레퍼런스 데이터, 파라미터 스냅샷)
- **Export:** Pandas (Excel), WeasyPrint (PDF)

---

## 2. 화면 레이아웃 및 기능 명세

### 전체 구조 (Grid Layout)
```
[ Header: Meta Info & Mode Selection ]
--------------------------------------
[ Left: Viewer ] | [ Right: Controls ]
                 | [ Right: Results  ]
```

### A. 상단 헤더 (Meta Info)
*   **좌측 (제품 정보):** `Sample ID`, `Product Name`, `Lot No.`, `Capture Date`
*   **환경 설정:** `Light Source` (e.g., D65), `Magnification` (e.g., 10x)
*   **분석 모드:** `단일 샘플` / `샘플 vs 레퍼런스` / `LOT 트렌드`

### B. 좌측: 이미지 뷰어 (Interactive Viewer)
*   **기능:**
    *   자동 검출된 **렌즈 외곽선** 및 **중심점(●)** 표시.
    *   설정 패널 조작 시 **Ring/Sector 그리드 실시간 프리뷰**.
*   **Layering:**
    1.  Background Image (Original)
    2.  Mask Layer (Background Masking Toggle - 유효 영역 확인용)
    3.  Grid Overlay (Center, Inner/Outer Radius, Rings, Sectors)
    4.  Heatmap Overlay (분석 후 표시)

### C. 우측 상단: 분석 설정 패널 (Settings)
1.  **영역 미세 조정 (Radius Fine-tuning)**
    *   **슬라이더:** `Inner Radius` (홀 경계), `Outer Radius` (인쇄 끝)
    *   *Action:* 슬라이더 조작 시 뷰어의 원이 실시간으로 크기 변경됨.
2.  **세그먼트 설정**
    *   `Ring 개수`: 슬라이더 (2~6)
    *   `Ring 분할 방식`: Radio (`균등`, `중앙 강조`)
    *   `각도 섹터`: Radio (`사용 안 함`, `12분할(30°)`, `24분할(15°)`)
3.  **잉크/색상 설정**
    *   `잉크 개수`: 1~3
    *   Checkbox: `[ ] 잉크 자동 추출(K-means)`
4.  **기준 설정**
    *   `ΔE 기준`: `전체 평균` / `레퍼런스` / `Manual Input`
    *   `공식 선택`: `CIEDE2000 (권장)` / `CIE76`
    *   `허용 상한`: Number Input
5.  **Action**
    *   Button: `[분석 실행]` (파라미터 변경 후 재계산 Trigger)

### D. 우측 하단: 결과 탭 (Results)
1.  **Tab 1: Ring Summary (Table)**
    *   Cols: `Ring`, `Radius(r_norm)`, `L*`, `a*`, `b*`, `ΔE`, **`Coverage(%)`**, `Judgment`
    *   *Note:* `r_norm` (0.0~1.0)을 표시하여 해상도 무관 비교 가능하게 함.
2.  **Tab 2: Sector Heatmap (Grid)**
    *   Ring x Sector 매트릭스. ΔE 값에 따라 Green-Yellow-Red 그라데이션.
    *   *Standard:* **0도 = 3시 방향, 시계 방향(CW)** 기준.
3.  **Tab 3: Graph**
    *   Radial Profile, Delta E Trend.
4.  **Tab 4: Report**
    *   `[PDF 생성]`, `[Excel 내보내기]`

---

## 3. 백엔드 처리 로직 (Pipeline Flow)

1.  **Preprocessing & Masking**
    *   `PreProcessor`:
        *   기본: RGB distance from White > 20 (User Request)
        *   확장: Otsu Thresholding 지원 (Option)
    *   `Dot Coverage` 분모(전체 영역 픽셀 수) 계산.
2.  **Geometric Segmentation**
    *   User가 보정한 `Inner`, `Outer` 반경 적용.
    *   **Normalization:** $r_{norm} = (r - r_{inner}) / (r_{outer} - r_{inner})$
    *   **Angular Mapping:** `atan2(y, x)`로 각도 계산 (3시=0도, CW 보정).
3.  **Color Analysis**
    *   **Color Space:** sRGB -> XYZ -> CIELAB.
    *   **Ink Extraction:** K-means (if enabled) or Mean shifting.
4.  **Statistics & Delta E**
    *   각 세그먼트(Ring/Sector)별 L, a, b 평균 계산.
    *   **Distance Metric:** **CIEDE2000** (Default) for perceptual accuracy.
    *   `Dot Coverage` = (세그먼트 내 유효 픽셀 수 / 세그먼트 전체 면적 픽셀 수) * 100.
5.  **Persistence**
    *   SQLite DB에 결과 저장.
    *   **Snapshot:** 분석 당시 사용된 `settings` (threshold, ring count 등)를 함께 저장하여 재현성 확보.

---

## 4. 엑셀 내보내기 사양 (Excel Spec)

### Sheet 1: RingSummary
| Row | A | B | C | D | E | F | G | H |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1~6 | Meta | (Include Config Snapshot) | | | | | | |
| 8+ | **Ring** | **r_norm(%)** | **L*** | **a*** | **b*** | **dE(2000)** | **Pixels** | **Cov(%)** |
| Data | 1 | 0.33 | 71.4 | 3.5 | 19.8 | 13.9 | 88071 | 95.2 |

### Sheet 2: RingSectorDeltaE
| Row | A | B | ... | M |
|:---:|:---:|:---:|:---:|:---:|
| 1 | **Ring \ Sector** | **S1 (0-30°)** | ... | **S12** |
| 2 | Ring 1 | 2.6 | ... | 4.5 |

---

## 5. 단계별 구현 로드맵

### Step 1: UI 스켈레톤 및 인터랙티브 뷰어 (Day 7)
- `dashboard.html` 생성 (Bootstrap 5).
- Canvas View Controller: 이미지 로드, 줌/팬, 실시간 오버레이.

### Step 2: 백엔드 로직 확장 (Day 8)
- `AngularProfiler` (좌표계 표준화) 및 `ColorEvaluator` (CIEDE2000) 구현.
- API 스키마 업데이트 (파라미터 스냅샷 포함).

### Step 3: 데이터 연동 및 시각화 (Day 9)
- 분석 실행 및 결과 바인딩.
- 히트맵 및 그래프 연동.

### Step 4: 리포트 및 DB (Day 10)
- 엑셀/PDF 내보내기.
- SQLite DB 연동 및 이력 관리.
