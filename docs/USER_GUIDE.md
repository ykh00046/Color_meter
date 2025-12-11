# Contact Lens Color Inspection System - User Guide

이 문서는 콘택트렌즈 색상 검사 시스템의 사용자를 위한 상세 가이드입니다. SKU 등록부터 검사 실행, 결과 분석 및 시각화 활용까지의 전체 워크플로우를 다룹니다.

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [사전 준비](#2-사전-준비)
3. [SKU 관리 및 등록](#3-sku-관리-및-등록)
4. [검사 실행](#4-검사-실행)
5. [시각화 및 결과 분석](#5-시각화-및-결과-분석)
6. [문제 해결 (Troubleshooting)](#6-문제-해결-troubleshooting)
7. [배포 및 운영](#7-배포-및-운영)

---

## 1. 시스템 개요

이 시스템은 컴퓨터 비전 기술을 활용하여 콘택트렌즈 이미지의 색상을 분석하고 품질을 자동으로 판정합니다.

*   **자동 검사**: 이미지에서 렌즈를 감지하고, 정의된 구역별로 색상 차이($\Delta E$)를 계산하여 OK/NG를 판정합니다.
*   **다중 SKU 지원**: 다양한 제품 모델(SKU)에 대한 기준 색상 정보를 관리하고 적용할 수 있습니다.
*   **정밀한 색상 분석**: Lab 색 공간 및 CIEDE2000 ΔE 알고리즘을 적용합니다.
*   **시각화**: 검사 결과를 히트맵, 차트, 오버레이 이미지로 시각화하여 직관적인 분석을 제공합니다.
*   **배치 처리**: 대량의 이미지를 한 번에 처리하고 결과를 CSV로 저장할 수 있습니다.
*   **성능 최적화**: 멀티스레딩 기반 이미지 로딩 병렬화, NumPy 벡터화 최적화, 메모리 관리 등.
*   **Zone Segmentation 개선**: 렌즈의 Zone 분할 알고리즘이 개선되어, 노이즈가 많은 도트 인쇄 환경에서도 안정적인 경계 검출이 가능합니다. `expected_zones` 힌트, 스무딩 필터, 혼합 구간(Transition Buffer) 처리 등이 적용되었습니다.

---

## 2. 사전 준비

시스템을 사용하기 전에 다음 사항을 확인하세요.

*   **Python 환경**: Python 3.8 이상이 설치되어 있어야 합니다.
*   **패키지 설치**: `requirements.txt`에 명시된 필수 패키지가 설치되어 있어야 합니다.
    ```bash
    pip install -r requirements.txt
    ```
*   **디렉토리 구조**: 입력 이미지와 설정 파일이 올바른 위치에 있어야 합니다.
    *   이미지: `data/raw_images/` (기본 경로)
    *   SKU 설정: `config/sku_db/`

---

## 3. SKU 관리 및 등록

새로운 제품(SKU)을 검사하려면 먼저 해당 SKU의 기준 정보(Baseline)를 등록해야 합니다.

### 3.1. 기준 데이터(Baseline) 생성

가장 쉬운 방법은 양품(Golden Sample) 이미지를 사용하여 자동으로 기준 데이터를 생성하는 것입니다.

1.  **양품 이미지 준비**: 해당 SKU의 결함 없는 깨끗한 이미지를 준비합니다.
2.  **생성 도구 실행**: 다음 명령어를 사용하여 기준 JSON 파일을 생성합니다.

    ```bash
    # 사용법: python tools/generate_sku_baseline.py --image <양품_이미지_경로> --sku <SKU_ID> --expected-zones <예상_존_개수>
    
    python tools/generate_sku_baseline.py --image data/raw_images/sample_sku002.jpg --sku SKU002 --expected-zones 2
    ```

    *   `--expected-zones`: `ZoneSegmenter`에 제공할 Zone 개수에 대한 힌트입니다. 도트 인쇄 렌즈와 같이 노이즈가 많은 이미지의 Zone 분할 정확도를 높이는 데 사용됩니다.

3.  **결과 확인**: `config/sku_db/SKU002.json` 파일이 생성되었는지 확인합니다.

### 3.2. SKU 수동 등록 및 수정 (고급)

JSON 파일을 직접 생성하거나 수정하여 세부 설정을 조정할 수 있습니다.

**파일 경로**: `config/sku_db/<SKU_ID>.json`

**JSON 구조 예시**:
```json
{
    "sku_id": "SKU002",
    "zones": {
        "Zone_A": { "L": 85.2, "a": -1.5, "b": 2.3, "tolerance": 2.5 },
        "Zone_B": { "L": 70.1, "a": 5.2, "b": -10.1, "tolerance": 3.0 },
        "Zone_C": { "L": 92.0, "a": 0.5, "b": 0.2, "tolerance": 2.0 }
    },
    "params": {
        "lens_radius": 350,
        "center_x": 1024,
        "center_y": 1024,
        "expected_zones": 2,          // ZoneSegmenter에 제공할 기대 Zone 개수 힌트
        "transition_buffer_px": 5     // Zone 경계 주변 무시할 픽셀 수 (혼합 구간 처리)
    }
}
```

*   **zones**: 각 구역(Zone A, B, C 등)별 L\*a\*b\* 기준값과 허용 오차(`tolerance`)를 설정합니다.
*   **params**: 렌즈 반경이나 중심점 보정 등 추가 파라미터를 설정할 수 있습니다.
    *   `expected_zones`: `ZoneSegmenter`가 렌즈를 몇 개의 Zone으로 분할할지 기대하는 값입니다. 이 힌트는 노이즈가 많은 환경에서 Zone 분할 정확도를 높이는 데 활용됩니다.
    *   `transition_buffer_px`: Zone 경계 주변의 혼합 구간(Transition Zone)을 처리하기 위한 설정입니다. 지정된 픽셀 수만큼 경계 양쪽의 데이터를 색상 평가에서 제외하거나 가중치를 낮출 수 있습니다.

### 3.3. SKU 목록 확인

현재 등록된 SKU 목록을 확인하려면 CLI 명령을 사용합니다.

```bash
python src/main.py sku list
```

---

## 4. 검사 실행

### 4.1. 단일 이미지 검사

특정 이미지 한 장을 검사하고 결과를 확인합니다.

```bash
python src/main.py --image data/raw_images/test_image.jpg --sku SKU001
```

*   **--image**: 검사할 이미지 파일 경로
*   **--sku**: 적용할 SKU ID

### 4.2. 배치 검사 (폴더 단위)

폴더 내의 모든 이미지를 검사합니다.

```bash
python src/main.py --batch data/raw_images/ --sku SKU001
```

*   **--batch**: 이미지가 저장된 디렉토리 경로
*   결과는 `results/` 디렉토리에 CSV 파일로 저장됩니다 (예: `inspection_results_20241212_103000.csv`).

---

## 5. 시각화 및 결과 분석

검사 결과를 시각적으로 확인하여 불량 원인을 분석할 수 있습니다.

### 5.1. 시각화 옵션 사용

검사 실행 시 `--visualize` 옵션을 추가하면 시각화 결과물이 생성됩니다.

```bash
python src/main.py --image data/raw_images/ng_sample.jpg --sku SKU001 --visualize
```

### 5.2. 시각화 결과물 확인

결과물은 `results/visualization/<timestamp>/` 디렉토리에 저장됩니다.

1.  **Overlay Image**: 원본 이미지 위에 구역 경계와 판정 결과(OK/NG)가 표시됩니다.
2.  **$\Delta E$ Heatmap**: 색상 차이($\Delta E$) 분포를 히트맵으로 보여줍니다. 붉은색일수록 기준 색상과의 차이가 큼을 의미합니다. 극좌표 또는 직교좌표 형식으로 표시될 수 있습니다.
3.  **Profile Chart**: 중심으로부터의 거리에 따른 색상 변화(L\*a\*b\* 값) 그래프를 보여주며, Zone 경계 및 기준값과의 비교를 표시합니다.
4.  **Dashboard**: 배치 처리 결과에 대한 요약 통계 및 분포를 보여줍니다.

---

## 6. 문제 해결 (Troubleshooting)

### Q1. "SKU not found" 에러가 발생합니다.
*   해당 SKU ID에 대한 JSON 설정 파일이 `config/sku_db/` 디렉토리에 존재하는지 확인하세요.
*   SKU ID의 대소문자가 정확한지 확인하세요.

### Q2. 렌즈가 검출되지 않습니다 ("Lens detection failed").
*   이미지가 너무 어둡거나 밝지 않은지 확인하세요.
*   이미지 배경이 복잡하지 않고 균일한지 확인하세요.
*   `config/system_config.json`의 렌즈 검출 파라미터(`min_radius`, `max_radius`)를 조정해 보세요.

### Q3. 색상 판정이 너무 민감하거나 둔감합니다.
*   해당 SKU의 JSON 설정 파일에서 각 Zone의 `tolerance` 값을 조정하세요.
    *   값이 작을수록 엄격하게 판정합니다 (민감).
    *   값이 클수록 관대하게 판정합니다 (둔감).
*   `transition_buffer_px` 값을 조정하여 혼합 구간의 영향을 조절해 보세요.

### Q4. Zone 분할이 정확하지 않습니다 (Zone 개수가 틀리거나 경계가 부정확).
*   SKU JSON 설정 파일의 `params` 내 `expected_zones` 값을 실제 렌즈의 Zone 개수에 맞춰 정확히 설정했는지 확인하세요. 이 힌트는 `ZoneSegmenter`의 성능을 크게 향상시킵니다.
*   `ZoneSegmenter`는 `expected_zones`를 힌트로 사용하여 `gradient` 기반 검출, `ΔE` 기반 보조 검출, 그리고 `expected_zones` 기반 균등 분할 및 최소 폭 병합 등 다양한 Fallback 전략을 순차적으로 시도합니다.
*   입력 이미지의 노이즈가 심한 경우, `ZoneSegmenter`의 내부 스무딩 파라미터 튜닝이 필요할 수 있습니다.

### Q5. 프로그램 실행 속도가 느립니다.
*   이미지 해상도가 너무 높은 경우 처리 속도가 느려질 수 있습니다.
*   `--visualize` 옵션은 추가적인 이미지 처리가 필요하므로, 대량 배치 처리 시에는 필요한 경우에만 사용하세요.
*   시스템 성능 프로파일링 결과(docs/PERFORMANCE_ANALYSIS.md 참고)를 통해 병목 지점을 확인하고 최적화 방안을 고려하세요.

---

## 7. 배포 및 운영

### 7.1. Docker를 이용한 배포

시스템은 Docker 컨테이너를 통해 쉽게 빌드하고 배포할 수 있습니다. 이는 환경 설정의 복잡성을 줄이고 일관된 실행 환경을 제공합니다.

1.  **Docker 이미지 빌드**:
    ```bash
    scripts/build_docker.sh
    ```
2.  **Docker 컨테이너 실행**:
    ```bash
    scripts/run_docker.sh
    ```

    *   더 자세한 내용은 [배포 가이드](docs/DEPLOYMENT.md)를 참조하세요.

### 7.2. 환경 변수 설정

Docker 환경 또는 로컬에서 실행 시 다음과 같은 환경 변수를 설정할 수 있습니다.

*   `LOG_LEVEL`: `INFO`, `DEBUG`, `WARNING`, `ERROR` 중 하나 (기본값: `INFO`)
*   `DATA_PATH`: 이미지 및 결과 파일 경로 (기본값: `data/`)

---
**추가 문의사항이나 기술 지원이 필요한 경우 개발팀에 문의해 주세요.**