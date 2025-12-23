# 지능형 잉크 분석 엔진 향상 계획 (v2.0)

**작성일:** 2025-12-13
**상태:** 계획 단계 (Planned)
**목표:** 기계적 군집화의 한계를 넘어, "사람의 눈과 직관"을 재현하는 잉크 분석 알고리즘 구현

---

## 1. 개요 및 스코프 (Scope)

### 1.1 역할 정의
`InkEstimator`는 기존 판정 파이프라인(`InspectionPipeline`)을 대체하는 것이 아니라, **진단 및 설명 보조 도구(Diagnostic & Explanatory Aid)**로 동작합니다.

*   **Main Pipeline (판정)**: Rule-based (SKU 기준값 비교) → 빠르고 일관됨.
*   **InkEstimator (진단)**: Unsupervised (데이터 분포 분석) → 이상 징후 감지 및 설명 제공.

### 1.2 데이터 표준 (Data Standard)
모든 색상 분석은 **표준 CIELAB 색공간**을 기준으로 합니다.
*   **Lightness ($L^*$)**: 0.0 (Black) ~ 100.0 (White)
*   **Channels ($a^*, b^*$)**: -128.0 ~ +127.0
*   *Note*: OpenCV의 Lab 스케일($L$:0~255, $a,b$:0~255)은 입력 즉시 표준형으로 정규화하여 사용합니다.

---

## 2. 핵심 알고리즘: 4단계 추론 파이프라인

### Step 1: 지능형 후보 픽셀 선별 (Intelligent Sampling)
배경, 투명부, 노이즈를 제거하고 "유의미한 잉크 후보"만 남깁니다.

*   **입력**: BGR 이미지
*   **변환**: BGR $\rightarrow$ CIELAB
*   **필터링 조건**:
    1.  **유채색 잉크**: $C = \sqrt{a^2 + b^2} \ge T_{chroma}$ (기본값 6.0)
    2.  **무채색(Black) 잉크 보존**: $C < T_{chroma}$ 이더라도, $L \le T_{dark}$ (기본값 45.0) 이면 포함.
    3.  **최종 마스크**: `(Condition 1 OR Condition 2)`

### Step 2: 반사/하이라이트 제거 (Specular Rejection)
잉크 색상과 유사하지만 "빛 반사"로 인해 밝게 뜬 픽셀을 제거합니다.

*   **조건**: $L \ge T_{high}$ (기본값 95.0) AND $C \le T_{low}$ (기본값 5.0)
*   **처리**: 해당 픽셀은 잉크 군집화에서 **완전 배제**합니다.

### Step 3: 적응형 군집화 (Adaptive Clustering)
도트 인쇄의 타원형 분포 특성을 반영하여 색 덩어리를 찾습니다.

*   **샘플링**: 전체 유효 픽셀 중 $N=20,000$개를 랜덤 샘플링 (Seed 고정).
*   **모델**: GMM (Gaussian Mixture Model)
*   **최적화**: $k \in \{1, 2, 3\}$에 대해 각각 GMM을 학습하고, **BIC (Bayesian Information Criterion)**가 최소인 $k$를 선택합니다.
*   **초기화**: K-Means 결과를 초기값으로 사용하여 수렴 속도와 안정성을 확보합니다.

### Step 4: "중간 톤 = 혼합" 추론 (Human Intuition Logic) ⭐핵심
물리적으로 섞여서 생긴 색을 독립 잉크로 오판하는 것을 방지합니다.

*   **전제**: $k=3$일 때만 수행합니다.
*   **알고리즘 (Linearity Check)**:
    1.  3개의 군집 중심을 $L$값 기준으로 정렬: $C_{dark}, C_{mid}, C_{bright}$.
    2.  양 끝점 벡터 정의: $\vec{V} = C_{bright} - C_{dark}$.
    3.  중간점의 투영 오차(Projection Error) 계산:
        *   $C_{mid}$에서 직선 $\vec{V}$까지의 수직 거리 $d$.
    4.  **판단**: $d < T_{linear}$ (기본값 3.0) 이면, $C_{mid}$는 독립 잉크가 아니라 **혼합(Mixing)**으로 간주합니다.
*   **조치**: 잉크 개수를 2개로 수정하고, $C_{mid}$의 가중치를 거리에 비례하여 $C_{dark}, C_{bright}$에 분배합니다.

---

## 3. 구현 명세 (Implementation Specs)

### 3.1 파일 구조
*   `src/core/ink_estimator.py`: 독립 모듈 구현체.

### 3.2 주요 파라미터 테이블

| 파라미터 | 기본값 | 설명 |
| :--- | :--- | :--- |
| `chroma_thresh` | 6.0 | 유채색 잉크 판단 기준 (낮으면 배경 노이즈 증가) |
| `L_dark_thresh` | 45.0 | 무채색(Black) 잉크 판단 기준 (높으면 그림자 오인) |
| `L_highlight` | 95.0 | 반사광 제거 기준 |
| `merge_de_thresh` | 5.0 | 유사 색상 병합 기준 ($\Delta E_{76}$) |
| `linearity_thresh` | 3.0 | 중간 톤 혼합 판단 기준 (거리 오차) |
| `random_seed` | 42 | 결과 재현성 확보를 위한 시드값 |

---

## 4. 기대 효과 및 검증 계획

### 4.1 기대 효과
*   **과잉 검출 방지**: 도트 밀도 차이로 인한 "가짜 3도" 판정을 획기적으로 줄임.
*   **Black 잉크 인식**: 기존 로직에서 누락되던 검은색 써클라인을 정확히 잡아냄.
*   **설명력 강화**: "왜 2개인가요?" $\rightarrow$ "중간색은 혼합으로 판단되어 제외했습니다."

### 4.2 검증 데이터셋
*   **Case A (2도 Dot)**: 갈색 도트 렌즈 (기존 알고리즘이 3개로 오판하던 케이스) $\rightarrow$ 2개로 나와야 성공.
*   **Case B (Black Circle)**: 검은색 써클 렌즈 $\rightarrow$ 1개(Black)가 잡혀야 성공.
*   **Case C (3도 Real)**: 실제로 3가지 다른 색 잉크를 쓴 렌즈 $\rightarrow$ 3개로 유지되어야 성공.

---

## 5. 향후 일정 (Roadmap)

*   **Phase 1 (✅ 완료 - 2025-12-13)**: 알고리즘 설계 및 `InkEstimator` 프로토타입 구현.
*   **Phase 2 (✅ 완료 - 2025-12-14)**: `InspectionPipeline` 통합 및 대시보드 [잉크 정보] 탭 연동.
    *   ✅ `estimate_from_array()` 메서드 추가 (numpy array 직접 입력)
    *   ✅ `zone_analyzer_2d.py`에서 InkEstimator 호출 통합
    *   ✅ `ink_analysis` 구조 변경 (zone_based + image_based)
    *   ✅ Web UI 병렬 표시 (Zone-Based + Image-Based 섹션)
    *   ✅ Meta 정보 표시 (Mixing Correction, BIC 점수, 샘플 수)
*   **Phase 3 (예정)**: SKU 관리 기능에 "Auto-Detect Ink Config" 버튼 추가하여 기준값 자동 설정 지원.

## 6. Phase 2 구현 상세 (2025-12-14)

### 통합 방식
기존 Zone 기반 판정은 유지하고, InkEstimator 결과를 보조 정보로 병렬 제공하는 "변형 옵션 1" 방식 채택.

**이유**:
- 안정성 우선: 기존 판정 로직 유지로 리스크 최소화
- 비교 가능성: Zone 기반 vs Image 기반 결과 비교로 불일치 감지
- 단계적 접근: 충분한 검증 후 Phase 3 진행

### 구현 내용
1. **Backend** (`src/core/zone_analyzer_2d.py:1422-1460`):
   ```python
   ink_analysis = {
       "zone_based": {
           "detected_ink_count": 3,
           "detection_method": "transition_based",
           "inks": [...]  # Zone C/B/A → Ink 1/2/3 매핑
       },
       "image_based": {
           "detected_ink_count": 2,  # GMM 검출 결과
           "detection_method": "gmm_bic",
           "inks": [...],  # 실제 잉크 Lab/RGB/HEX 값
           "meta": {
               "correction_applied": true,  # Mixing Correction 적용 여부
               "sample_count": 15000,
               "bic": -45231.2
           }
       }
   }
   ```

2. **Frontend** (`src/web/templates/index.html:183-214, 534-657`):
   - Zone-Based Analysis 섹션 (파란색 헤더)
   - Image-Based Analysis 섹션 (녹색 헤더)
   - Meta 정보 Alert (Mixing Correction, 샘플 수, BIC)
   - 색상 배지 표시 (HEX 코드 + 실제 색상)

### 검증 결과
- ✅ 문법 오류 없음 (py_compile 통과)
- ✅ Web UI 정상 동작 확인 (사용자 보고)
- ⚠️ 테스트 코드 미작성 (다음 단계 필요)
