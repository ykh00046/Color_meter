﻿# 🖥️ Color Meter Web UI – 사용 가이드 (Ver 2.0)

이 문서는 **전면 재설계된 Color Meter Web UI**의 사용 방법을 안내합니다. 기존의 경량 UI를 넘어, 콘택트렌즈 색상 분석을 위한 **전문적인 대시보드** 형태로 재탄생했습니다. Interactive Viewer, 상세 설정, 다채로운 시각화 기능을 통해 직관적이고 효율적인 분석 경험을 제공합니다.

---

## 1. 개요
새로운 Web UI는 FastAPI 백엔드와 Bootstrap 5, Chart.js, Panzoom.js를 결합하여 구현된 단일 페이지 애플리케이션(SPA) 형태입니다.
**주요 목표:** 분석 워크플로우를 최적화하고, 렌즈 이미지 및 분석 결과에 대한 깊이 있는 시각적 탐색을 지원합니다.

**주요 기능:**
*   **통합 대시보드:** 모든 주요 기능(이미지 뷰어, 설정, 결과)이 한 화면에 통합.
*   **대화형 이미지 뷰어:**
    *   Panzoom을 이용한 확대/축소 및 이동.
    *   렌즈 검출 결과(중심점, 반경) 및 조절 가능한 Grid Overlay (Ring, Sector) 실시간 표시.
*   **세밀한 분석 설정:**
    *   Inner/Outer Radius, Ring/Sector 개수, 분할 모드 등 정밀한 Grid 설정.
    *   SKU 선택 및 분석 실행.
*   **다채로운 결과 시각화:**
    *   **Ring Summary Table:** 각 Ring(영역)별 측정 Lab 값, ΔE, OK/NG 판정 요약.
    *   **Graphs:** Radial Color Profile(L*, a*, b* 스무딩 곡선) 및 Zone별 ΔE 바 차트.
    *   **Sector Heatmap (준비 중):** 렌즈 내 섹터별 ΔE 분포를 시각화 (예정).
*   **백엔드 API 연동:** `/inspect` API를 통해 분석 파이프라인과 완벽하게 통합.

---

## 2. 실행 방법
### 2.1 서버 실행
Web UI를 사용하려면 FastAPI 서버가 가동되어야 합니다. 프로젝트 루트 폴더에서 다음 스크립트를 실행합니다.
```bash
scripts/run_web_ui.bat
```
*   **개발 중:** `uvicorn src.web.app:app --host 0.0.0.0 --port 8000 --reload` 명령어를 직접 사용하면 코드 변경 시 자동 재시작됩니다.
*   **Docker 환경:** Docker 환경에서 이미지를 실행할 경우, 기본적으로 uvicorn 서버가 8000번 포트로 실행됩니다. (자세한 내용은 [배포 가이드](../guides/DEPLOYMENT_GUIDE.md) 참고)

서버가 정상적으로 시작되면 콘솔에 관련 로그가 출력되고, 브라우저에서 UI에 접속할 수 있습니다.

### 2.2 브라우저 접속
로컬 서버를 실행했다면 웹 브라우저에서 `http://localhost:8000`으로 접속합니다. (포트를 변경했다면 URL도 해당 포트에 맞춰 수정)

---

## 3. UI 레이아웃 및 주요 기능 사용법
새로운 Web UI는 크게 세 부분으로 나뉩니다: **상단 헤더**, **좌측 이미지 뷰어**, **우측 설정 & 결과 패널**.

### 3.1 상단 헤더
*   **로고 및 제목:** "Color Meter" 프로젝트 이름.
*   **JUDGMENT Badge:** 현재 분석 결과의 최종 판정(`OK`, `NG`, `READY`)을 실시간으로 표시합니다.

### 3.2 좌측: 이미지 뷰어 (Interactive Viewer)
렌즈 이미지를 표시하고, Panzoom 기능과 Grid Overlay를 통해 시각적으로 탐색하는 핵심 영역입니다.
*   **이미지 업로드:**
    1.  뷰어 상단의 **"Load Image"** 버튼을 클릭하거나, 뷰어 영역으로 이미지를 직접 **드래그 앤 드롭**하여 파일을 불러옵니다.
    2.  업로드된 이미지는 Canvas에 렌더링됩니다.
*   **확대/축소 및 이동 (Panzoom):**
    1.  **확대/축소:** 마우스 휠을 스크롤하여 이미지를 확대/축소할 수 있습니다.
    2.  **이동:** 마우스 왼쪽 버튼을 누른 채 드래그하여 이미지를 이동할 수 있습니다.
*   **Grid Overlay (실시간 프리뷰):**
    1.  이미지가 로드되면 렌즈 검출 결과(중심점, 반경)를 기반으로 Grid Overlay가 기본적으로 표시됩니다.
    2.  우측 "SETTINGS" 패널에서 **Inner/Outer Radius, Rings, Sectors** 값을 변경하면 뷰어에 Grid Overlay가 **실시간으로 업데이트**되어 분석 영역을 미리 확인할 수 있습니다.

### 3.3 우측: 설정 & 결과 패널

이 패널은 **"SETTINGS"** 카드와 **"RESULTS"** 카드로 구성됩니다.

#### 3.3.1 SETTINGS 카드 (분석 설정)
다양한 분석 파라미터를 조정하는 영역입니다.
*   **Analysis Configuration:**
    *   **Target SKU:** 드롭다운에서 분석에 사용할 SKU 코드를 선택합니다. (예: `SKU001`, `SKU_EXAMPLE`)
    *   **Advanced (API) 파라미터:** `/recompute` 요청 시 `sample_percentile`(0~100)을 지정하면 프로파일 평균 대신 분위수 집계가 적용됩니다.
*   **Region of Interest (Grid):** 렌즈 내 분석할 영역(ROI) 및 Grid 분할 방식을 설정합니다.
    *   **Inner Radius / Outer Radius:** 렌즈의 중심부와 외곽부를 제외하거나 포함할 정규화된 반지름(0.0~1.0)을 슬라이더로 조절합니다. 뷰어에 실시간으로 반영됩니다.
    *   **Rings:** 렌즈를 몇 개의 동심원 영역(Ring)으로 나눌지 설정합니다. 슬라이더 조작 시 뷰어에 반영됩니다.
    *   **Sectors:** 렌즈를 몇 개의 각도 섹터로 나눌지 설정합니다. (예: 12 (30°), 24 (15°)). 드롭다운 선택 시 뷰어에 반영됩니다.
    *   **Ring 분할 모드:** Ring의 분할 방식을 `Uniform (균등)` 또는 `Center Focus (중앙 강조)` 중 선택합니다. (현재는 `Uniform`만 구현)
*   **"RUN ANALYSIS" 버튼:** 현재 업로드된 이미지와 설정된 파라미터로 분석을 실행합니다. 분석이 완료되면 "RESULTS" 패널이 갱신됩니다.

#### 3.3.2 RESULTS 카드 (결과 시각화)

분석 결과는 6개의 Tab으로 제공되어 다양한 관점에서 데이터를 확인할 수 있습니다.

##### Tab 1: Summary (기본 탭)
- 전체 판정 결과 (OK / OK_WITH_WARNING / NG / RETAKE)
- 각 Zone별 측정된 **Lab 값**, **기준 임계값(Threshold)**, 계산된 **ΔE 값**, 최종 **판정**을 테이블 형태로 요약
- ΔE 값과 판정 결과에 따라 색상 구분 (녹색: OK, 주황색: Warning, 빨간색: NG)
- **Decision Trace**: 판정 근거 및 이유 표시
- **Next Actions**: 권장 조치 사항 표시
- **Confidence Score**: 분석 신뢰도 (0.0~1.0) 및 구성 요소 breakdown

##### Tab 2: Ink Info (잉크 정보) ⭐신규 기능

렌즈의 잉크 개수와 색상 정보를 **2가지 방법**으로 분석하여 비교 제공합니다:

**Zone-Based Analysis (파란색 헤더)**
- **방법**: SKU 설정의 Zone 구조를 기반으로 추출
- **표시 내용**:
  - 검출된 잉크 개수 (예: 3개)
  - 각 잉크의 Lab 값, RGB 값, HEX 색상 배지
  - Zone 매핑 정보 (Zone C → Ink 1, Zone B → Ink 2, Zone A → Ink 3)
- **활용**: SKU 기준값과의 직접 비교, OK/NG 판정

**Image-Based Analysis (녹색 헤더)**
- **방법**: GMM(Gaussian Mixture Model)으로 이미지 전체를 분석
- **표시 내용**:
  - GMM으로 검출된 잉크 개수 (예: 2개)
  - 각 잉크의 Lab 값, RGB/HEX 색상 배지, 픽셀 비율(Weight)
  - **색상 팔레트**: 검출된 잉크 색상을 큰 칩으로 표시 (동적 개수)
  - **Meta 정보 Alert**:
    - `Mixing Correction Applied`: 혼합색 보정 적용 여부 (true/false)
    - `Sample Count`: 분석에 사용된 픽셀 수
    - `BIC Score`: 모델 선택 지표 (낮을수록 좋음)
    - **Sampling Config** (상세 정보):
      - `chroma_threshold`: 크로마 필터 임계값 (기본 6.0)
      - `L_max`: 하이라이트 제거 기준 (기본 98.0)
      - `candidate_pixels`: 필터링 후 후보 픽셀 수
      - `sampled_pixels`: 실제 GMM 분석에 사용된 픽셀 수
      - `sampling_ratio`: 샘플링 비율 (0.0~1.0)
- **활용**: Zone-Based 결과와 불일치 시 원인 진단, 신규 SKU 기준값 설정

**Sampling Config 해석 팁**:
- `sampling_ratio`가 낮으면 (<0.5): 유효 데이터 부족, 이미지 품질 문제 가능성
- `highlight_removed: true`: 하이라이트 픽셀 제거됨 (L > L_max)
- `chroma_filter_applied: true`: 낮은 채도 픽셀(배경) 제거됨

**불일치 케이스 해석**:
- Zone-Based 3개, Image-Based 2개 (Mixing Correction) → 도트 패턴 렌즈
- Zone-Based 2개, Image-Based 3개 → 실제 3도 잉크, SKU 설정 오래됨
- 개수 같지만 색상 다름 → Zone 경계 검출 오류 가능성

##### Tab 3: Detailed Analysis (상세 분석)

**Uniformity Metrics (균일도 지표)**
- `max_std_L`: Zone별 최대 L* 표준편차 (낮을수록 균일함)
  - < 10.0: 양호 (OK)
  - 10.0 ~ 12.0: 경고 (OK_WITH_WARNING)
  - > 12.0: 불량 (RETAKE)
- **해석**: Zone 내 밝기 편차가 크면 얼룩, 오염, 조명 문제 가능성

**Sector Uniformity (섹터별 균일성) ⭐신규 기능**
- 렌즈를 **8개 섹터**(45°씩)로 나누어 국부 결함 감지
- 각 섹터의 L* 표준편차를 계산하여 한쪽만 색이 다른 경우 감지
- **표시 내용**:
  - `Max Sector Std L`: 가장 편차가 큰 섹터의 std_L 값
  - `Worst Zone`: 문제가 있는 Zone 이름
  - `Worst Sector`: 문제가 있는 섹터 번호 (0~7)
- **임계값**:
  - > 8.0: High severity (심각한 국부 결함)
  - 5.0 ~ 8.0: Medium severity (경미한 편차)
  - < 5.0: 정상

**Confidence Breakdown (신뢰도 구성 요소)**
- **Pixel Count Score**: 유효 픽셀 충분도
  - 분석에 사용된 픽셀 수가 충분한지 평가
- **Transition Score**: Zone 경계 명확도
  - Zone 경계가 얼마나 명확하게 검출되었는지
  - 낮으면 → 초점 불량, 해상도 부족
- **Std Score**: 균일도 점수
  - Zone 내 색상 일관성
  - 낮으면 → 얼룩, 조명 편향
- **Sector Uniformity**: 섹터별 일관성 ⭐신규
  - 8개 섹터 간 색상 편차
  - 낮으면 → 한쪽 방향 결함, 조명 편향
- **Lens Detection**: 렌즈 검출 신뢰도
  - 렌즈 영역이 얼마나 정확하게 검출되었는지
- **Overall**: 종합 신뢰도 점수 (0.0~1.0)
  - 0.9+: HIGH, 0.7~0.9: GOOD, 0.6~0.7: REVIEW, <0.6: LOW

**Risk Factors (위험 요소)**
위험 요소가 감지된 경우에만 표시됩니다:

- **Category 종류**:
  - `delta_e_exceeded`: Zone ΔE 초과
  - `sector_uniformity`: 섹터별 편차 높음 ⭐신규
  - `uniformity_low`: 전체 균일도 낮음
  - `boundary_unclear`: 경계 검출 불명확
  - `coverage_low`: 유효 픽셀 부족

- **Severity 레벨**:
  - `high`: 심각 (빨간색) - 즉시 조치 필요
  - `medium`: 보통 (주황색) - 검토 권장
  - `low`: 경미 (노란색) - 참고용

- **예시**:
  ```json
  {
    "category": "sector_uniformity",
    "severity": "high",
    "message": "Zone B 섹터 간 편차 높음",
    "details": {
      "zone": "B",
      "max_sector_std_L": 9.2,
      "worst_sector": 3
    }
  }
  ```

**Risk Factors 활용 방법**:
1. Severity가 high인 항목부터 우선 확인
2. Category별로 대응 방법 다름:
   - `delta_e_exceeded` → 색상 불량, 공정 점검
   - `sector_uniformity` → 한쪽 얼룩/오염, 조명 확인
   - `uniformity_low` → 전체적 불균일, 재촬영
3. Details 정보를 참고하여 원인 파악

##### Tab 4: Graphs (그래프)
- **Radial Color Profile**: 렌즈 중심에서 외곽으로 L\*, a\*, b\* 값 변화를 보여주는 라인 차트
- **Zone Delta E Analysis**: 각 Zone별 ΔE 값 막대 그래프 (임계값 기준선 포함)
- **Confidence Factors**: 신뢰도 구성 요소별 막대 그래프

##### Tab 5: Candidates (후보) - 개발자용
- **Transition Ranges**: 자동 검출된 Zone 경계 후보 목록
  - 각 후보의 반경(r), 신뢰도(confidence), 상태(SELECTED/CANDIDATE/REJECTED)
  - 표를 클릭하면 이미지 뷰어에 해당 반경 표시
- **Fallback Usage**: 경계 검출 실패 시 Fallback 사용 여부
- **활용**: Zone 분할 결과 검증, 파라미터 튜닝

##### Tab 6: Raw JSON
- 전체 분석 결과의 원본 JSON 데이터
- 개발자가 데이터 구조를 확인하거나 디버깅 시 활용
- Pretty-print 형식으로 가독성 확보

---

## 4. 트러블슈팅 및 팁
*   **데이터 미표시:** "RUN ANALYSIS" 버튼 클릭 후 결과 패널에 데이터가 나타나지 않는 경우:
    *   서버 로그(`scripts/run_web_ui.bat` 콘솔)에 오류 메시지(`PipelineError`)가 있는지 확인합니다.
    *   SKU 선택이 올바른지, `expected_zones` 설정이 SKU 구성과 일치하는지 확인합니다. (예: `SKU001`은 1-Zone이므로 UI의 Rings 슬라이더를 1로 설정)
*   **성능:** 대용량 이미지를 처리하거나 많은 Rings/Sectors를 설정할 경우 분석 시간이 길어질 수 있습니다.
*   **백엔드 API:** 이 UI는 내부적으로 `/inspect` API 엔드포인트를 사용합니다. 개발자는 `src/web/app.py`를 참조하여 API의 입출력 구조를 확인할 수 있습니다.
*   **확장성:** `src/web/static/js/` 디렉토리의 `viewer.js`, `controls.js`, `charts.js` 파일을 수정하여 UI 기능을 확장할 수 있습니다.

새로운 Web UI가 콘택트렌즈 색상 분석 작업에 큰 도움이 되기를 바랍니다. 궁금한 점이나 개선 요청이 있다면 언제든지 문의해주세요.
