# Image Normalization & QC Guide

**목적:** 분석 대상 이미지의 **기하학적 일관성(Geometry Consistency)**을 확보하여 분석 정밀도를 높이고, 색상 값(Color Integrity)은 원본 그대로 유지합니다.

---

## 1. 정규화 파이프라인 (Pipeline)

분석 전처리 단계에서 다음 순서로 수행합니다.

1.  **원 검출 (Circle Detection):** 렌즈의 중심 $(c_x, c_y)$과 반지름 $r$을 정밀하게 검출.
2.  **품질 검사 (Quality Control - QC):** 검출된 렌즈가 분석 가능한 상태인지 1차 판정.
3.  **좌표 변환 행렬 계산 (Transformation):**
    *   **Centering:** 렌즈 중심을 출력 이미지의 중심 $(W/2, H/2)$으로 이동.
    *   **Scaling:** 검출된 반지름 $r$을 목표 반지름 $r_{target}$으로 변환 (Scale Factor $s = r_{target} / r$).
4.  **이미지 워핑 (Warping):**
    *   계산된 행렬을 적용하여 이미지 변환.
    *   **중요:** 보간법(Interpolation)은 기하학적 매끄러움을 위해 `Bicubic` 또는 `Lanczos`를 권장하나, 색상 값의 미세한 변화를 방지해야 하는 엄격한 기준이 있다면 `Nearest Neighbor` 또는 고해상도 처리를 고려해야 함. (기본: `Linear` or `Cubic`)
5.  **마스킹 (Masking):** 렌즈 영역 바깥(배경)을 0(검정)으로 마스킹하여 노이즈 제거.

---

## 2. 입출력 정의 (I/O)

### 입력 (Input)
*   **Raw Image:** 원본 촬영 이미지 (예: 4000x3000)
*   **Options:**
    *   `do_normalize` (bool): 정규화 수행 여부
    *   `out_size` (int): 출력 정사각형 크기 (예: 800 -> 800x800)
    *   `r_target` (int): 정규화 후 렌즈 반지름 (예: 350)

### 출력 (Output)
*   **Normalized Image:** 정규화된 이미지 (예: 800x800, 렌즈 중심이 (400,400), 반지름 350)
*   **QC Result:** PASS / WARN / FAIL 및 세부 수치
*   **Transform Info:** 적용된 이동($tx, ty$) 및 스케일($s$) 값

---

## 3. QC 임계값 예시 (Quality Control)

정규화 과정에서 다음 항목을 검사하여 불량 이미지를 사전에 걸러냅니다.

| 항목 | 설명 | FAIL 기준 (예시) | 비고 |
| :--- | :--- | :--- | :--- |
| **Center Offset** | 원본 중심에서 렌즈가 벗어난 정도 | 이미지 중심에서 > 15% 벗어남 | 촬영 위치 불량 |
| **Radius Range** | 렌즈 크기의 적절성 | $r < 300px$ 또는 $r > 1800px$ | 거리/줌 불량 |
| **Circularity** | 원형 비율 (Short axis / Long axis) | < 0.90 | 렌즈 찌그러짐 또는 오검출 |
| **Sharpness** | Laplacian Variance 등 선명도 | < 100.0 | 초점 불량 |

---

## 4. UI/API 통합 시나리오

### 추천 운영 정책 (Default Policy)
*   **기본값 (Default):** `do_normalize=True`, `qc_check=True`
*   **목표:** 모든 분석 이미지를 동일한 규격(800x800, r=350)으로 맞춰서, Ring/Sector 설정이 고정된 템플릿처럼 동작하게 함.

### API 파라미터 제안 (POST /inspect)

```json
{
  "sku": "SKU001",
  "options": {
    "do_normalize": true,       // [기본: true] 정규화 수행 (좌표 중심 정렬 + 스케일링)
    "scale_normalize": true,    // [기본: true] 크기까지 통일할지 여부 (false면 중심만 맞춤)
    "qc_only": false,           // [기본: false] 분석 없이 QC만 하고 종료 (빠른 촬영 점검용)
    "return_preview": false,    // [기본: false] 정규화된 이미지를 base64로 반환 (트래픽 절약)
    "out_size": 800,            // [고정값 권장] 출력 이미지 크기
    "r_target": 350             // [고정값 권장] 정규화된 렌즈 반지름
  }
}
```

### UI 노출 가이드 (Frontend)

사용자 혼란을 막기 위해 **"고급 설정(Advanced Settings)"** 패널에 숨기거나 토글 형태로 제공합니다.

1.  **Normalization (스위치 ON/OFF):**
    *   Default: **ON**
    *   설명: "이미지 위치와 크기를 자동으로 보정합니다."
2.  **QC Only Mode (체크박스):**
    *   Default: **OFF**
    *   설명: "분석을 건너뛰고 촬영 상태(초점, 위치)만 점검합니다."
3.  **Preview Normalized (버튼):**
    *   클릭 시 `return_preview=true`로 요청하여 팝업으로 정규화된 이미지를 확인.

---

## 5. 데이터 보존 원칙
*   **색상 보존:** 히스토그램 평활화(Equalization), 화이트 밸런스 조정 등 **픽셀 값을 변조하는 전처리는 금지**합니다.
*   **좌표 변환:** 오직 Affine Transform(이동, 확대/축소)만 허용합니다.
