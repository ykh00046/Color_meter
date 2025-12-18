# 📘 콘택트렌즈 색상 검사 시스템 – 사용자 가이드

이 가이드는 콘택트렌즈 인쇄/색상 품질 검사 시스템을 사용하는 방법을 단계별로 설명합니다. **SKU 등록**에서 **검사 실행**, **결과 시각화**, **문제 해결**에 이르는 전체 워크플로우를 다룹니다. 내부 개발자뿐 아니라 품질 담당자 등 **시스템 사용자** 관점에서 작성되었습니다.

## 1. 시스템 개요
이 시스템은 **컴퓨터 비전** 알고리즘으로 렌즈 이미지를 분석하여 구역별 색상 오차(ΔE)를 계산하고 품질을 **자동 판정(OK/NG)**합니다. 주요 특징:
- **자동 색상 검사** – 이미지를 입력하면 렌즈 영역을 검출하고 사전에 정의된 각 **Zone**별로 ΔE를 계산하여 기준과 비교합니다. 결과는 OK/NG로 판정되고 세부 ΔE 값도 제공합니다.
- **다중 SKU 지원** – 다양한 제품 모델(SKU)에 대한 **기준 색상 데이터**(LAB 값 및 허용 오차)를 JSON 파일로 관리하며, 검사 시 해당 SKU의 기준을 적용합니다.
- **정밀한 분석 모드** – 판정 이전에 **분석용 데이터**(반지름 프로파일, 1차/2차 미분 곡선 등)를 확인할 수 있어, 자동 경계 검출 및 판정의 타당성을 **검증**할 수 있습니다.
- **결과 시각화** – 검사 결과를 **히트맵, 차트, 오버레이 이미지**로 제공하여 불량 원인을 직관적으로 파악할 수 있습니다.
- **배치 처리 & 성능 최적화** – 다수의 이미지를 한 번에 처리하는 **배치 기능**과, NumPy 병렬화 등을 활용한 **고속 처리**로 대량 검사에 견딜 수 있게 설계되었습니다 (단일 이미지 처리 ~**90ms**, 100장 배치 ~**7ms/장** 평균).

## 2. 사전 준비
시스템을 사용하기 전에 다음을 준비하세요:
- **Python 환경**: Python 3.10+ (또는 Docker 컨테이너 이용). 로컬에서 실행한다면 `requirements.txt`의 패키지를 모두 설치해야 합니다.
    ```bash
    pip install -r requirements.txt
    ```
- **데이터 디렉토리 구조**: 프로젝트 루트에 `data/` 및 `config/` 폴더가 존재해야 합니다.
  - 입력 이미지 위치: `data/raw_images/` (예시 이미지를 여기에 넣으면 편리합니다)
  - SKU 기준값 파일: `config/sku_db/` 내에 SKU별 JSON 파일 (예: `SKU001.json`)이 있어야 합니다.
- **(옵션) Docker 사용**: 개발 환경 구성 없이 바로 사용하려면 Docker 컨테이너를 이용할 수 있습니다. Docker 사용법은 [배포 가이드](DEPLOYMENT_GUIDE.md)를 참고하세요.

## 3. SKU 등록 및 관리
새로운 제품 **SKU**를 검사하려면 해당 SKU의 기준 색상 정보를 시스템에 등록해야 합니다. 두 가지 방식이 있습니다:
### 3.1 베이스라인 자동 생성 (권장)
양품 이미지를 사용하여 자동으로 SKU 기준 JSON 파일을 생성합니다:
1. **양품(Golden Sample) 이미지 확보:** 해당 SKU의 결함 없는 대표 이미지를 준비합니다 (가능하면 여러 장 평균).
2. **생성 도구 실행:** 프로젝트의 도구 스크립트를 사용하여 기준 정보를 생성합니다.
    ```bash
    python tools/generate_sku_baseline.py --image <경로/양품이미지.jpg> --sku <SKU_ID> --expected-zones <Zone개수>
    ```
    - `--sku`: 신규 SKU의 ID (예: `SKU002`).
    - `--expected-zones`: 해당 렌즈의 예상 Zone 개수 (예: 3). 이 값은 ZoneSegmenter에 **힌트**로 주어져 복잡한 패턴에서도 안정적으로 구역을 나누는 데 도움을 줍니다.
3. **결과 확인:** 실행 후 `config/sku_db/<SKU_ID>.json` 파일이 생성됩니다. 이 파일에 SKU의 기준 LAB 값과 허용 오차(`tolerance` 또는 `threshold`)가 채워져 있습니다. 필요하면 텍스트 편집기로 열어 값들을 검토하세요.

### 3.2 수동 등록 또는 수정 (고급)
SKU JSON 파일을 수동으로 작성하거나 자동 생성된 파일을 편집하여 세부 설정을 조정할 수 있습니다.
- **파일 위치:** `config/sku_db/<SKU_ID>.json`
- **파일 구조:** `"sku_code"`, `"default_threshold"`, `"zones"` 등의 필드를 포함합니다. 예시는 다음과 같습니다:
    ```json
    {
      "sku_code": "SKU002",
      "description": "Blue lens - 3 zones",
      "default_threshold": 3.5,
      "zones": {
        "A": { "L": 70.5, "a": -10.2, "b": -30.8, "threshold": 4.0 },
        "B": { "L": 68.3, "a": -8.5, "b": -28.2, "threshold": 3.5 },
        "C": { "L": 65.1, "a": -6.8, "b": -25.5, "threshold": 3.0 }
      },
      "metadata": { "created_at": "...", "baseline_samples": 5, ... }
    }
    ```
    *상세한 스키마 설명은 [SKU 관리 설계 문서](../design/SKU_MANAGEMENT_DESIGN.md)의 **2. SKU JSON 스키마** 섹션을 참고하세요.* 주요 항목:
    - `default_threshold`: 모든 Zone에 공통으로 적용될 ΔE 기준값 (Zone별 threshold가 없을 때 사용).
    - `zones`: 각 Zone의 기준 L\*, a\*, b\* 값과 허용 임계값 (`threshold`).
    - `metadata`: 기준값 생성에 대한 메타데이터 (샘플 수, 생성 방법 등).

### 3.3 SKU 목록 조회
현재 등록된 SKU들의 목록을 확인하거나 세부 정보를 보려면 **CLI 명령어**를 사용할 수 있습니다 (개발자 전용 기능):
```bash
# 모든 SKU 목록 출력
python -m src.main sku list

# 특정 SKU 정보 출력 (예: SKU002)
python -m src.main sku show SKU002
```
위 명령은 `config/sku_db/` 폴더를 스캔하여 SKU 코드를 나열하거나 해당 JSON 내용을 표시합니다.
(개발자 참고: 더 자세한 SKU 관리 CLI 사용법은 [SKU 관리 설계 문서](../design/SKU_MANAGEMENT_DESIGN.md)의 5. CLI 인터페이스를 참고하세요.)

## 4. 검사항목 실행
이 시스템은 **명령줄 인터페이스(CLI)**와 웹 인터페이스(UI) 두 가지 방법으로 검사를 실행할 수 있습니다. 여기서는 CLI 사용법을 먼저 설명하고, 5장에서 Web UI 사용법을 다룹니다.

### 4.1 명령줄을 통한 단일 이미지 검사
한 장의 이미지를 즉시 검사하려면 다음 명령을 실행합니다:
```bash
python src/main.py --image data/raw_images/test_image.jpg --sku SKU001
```
파라미터:
- `--image`: 검사할 이미지 파일 경로
- `--sku`: 적용할 SKU ID (해당 SKU의 기준값으로 판정)

명령 실행 후 콘솔에 검사 결과(OK/NG 여부, 전체 ΔE 등)가 출력되며, `results/` 폴더에 상세 결과 파일들이 저장됩니다:
- `results/latest/result.json`: 검사 결과 상세 (JSON 형식, ΔE 값 등 포함)
- `results/latest/overlay.png`: 원본 이미지에 Zone 경계와 판정 상태를 중첩 표시한 이미지

### 4.2 명령줄을 통한 배치 검사 (여러 이미지)
여러 이미지를 한 번에 처리하려면 `--batch` 옵션에 폴더 경로를 지정합니다:
```bash
python src/main.py --batch data/raw_images/ --sku SKU001
```
지정된 폴더 내의 모든 이미지를 순차 처리하며, 결과는 `results/<timestamp>/` 하위 폴더에 저장됩니다. 주요 산출물:
- `<timestamp>/batch_summary.csv`: 이미지별 결과 요약 (파일명, 판정, ΔE 값 등)
- `<timestamp>/logs/`: 검사 중 로그 파일 (optional, 로그 수준에 따라 생성)

**Tip:** 배치 검사 시 이미지가 많으면 시간이 걸릴 수 있습니다. CLI 실행 대신 비동기 처리를 원하면, 추후 지원 예정인 REST API 또는 멀티스레딩 기능을 활용할 수 있습니다. (개발자 참고: CLI 명령어의 상세 동작, 예외 코드 정의 등은 [파이프라인 설계 문서](../design/PIPELINE_DESIGN.md) 5. CLI 인터페이스 설계 부분에 있습니다.)

## 5. 검사 결과 확인 및 시각화
검사 완료 후 결과 데이터를 다양한 방식으로 확인할 수 있습니다.

### 5.1 Web UI를 통한 시각화 (단건/배치)
보다 인터랙티브한 방식으로 결과를 보고 싶다면 내장된 경량 웹 UI를 사용할 수 있습니다.
1. **Web UI 실행:** 터미널에서 `uvicorn src.web.app:app --port 8000` 명령으로 FastAPI 서버를 시작합니다. (자세한 내용은 [Web UI 가이드](WEB_UI_GUIDE.md)를 참고하세요.)
2. **브라우저 접속:** `http://localhost:8000` 에 접속하면 단일 및 배치 검사용 간단한 웹 인터페이스가 나타납니다. 이미지를 업로드하고 SKU를 입력한 뒤 **Analyze** 버튼을 누르면, 분석 모드로 프로파일 그래프 및 경계 후보가 먼저 표시됩니다. 필요 시 **Run Judgment** 옵션을 체크하여 OK/NG 판정을 수행할 수 있습니다.
3. **시각화 내용:** 웹 UI 단건 결과 화면에서는 원본+오버레이 이미지, 반지름-색상 프로파일 그래프(L*, a*, b*), ΔE 분포 그래프, 1차 미분 그래프, 2차 미분 (변곡점) 그래프, 그리고 자동 검출된 경계 후보 목록이 표시됩니다. 테이블의 경계 항목을 클릭하면 이미지 상에 해당 반경 위치에 표시가 나타나 검증 작업에 활용할 수 있습니다.
4. 배치 모드로 실행하면 업로드된 ZIP 내 모든 이미지를 처리하고, 각 이미지별 결과는 CSV 파일로 제공하며, ΔE 통계와 OK/NG 요약 표를 UI에 표시합니다.

(Web UI의 내부 동작과 API 엔드포인트에 대한 개발 정보는 [Web UI 가이드 문서](WEB_UI_GUIDE.md)를 확인하세요.)

### 5.2 오프라인 시각화 결과물 활용
CLI로 검사한 경우 생성된 결과 이미지/파일을 직접 열어볼 수 있습니다.
- **Overlay 이미지:** `results/<run_id>/overlay.png` 파일을 열면 각 Zone 경계와 판정(OK=녹색, NG=빨간색)이 그려진 검사항목 이미지를 확인할 수 있습니다.
- **JSON 결과:** `results/<run_id>/result.json`에는 전체 ΔE 값(`overall_delta_e`), 경과 시간, NG 원인(`ng_reasons`) 등 상세 정보가 담겨 있습니다. 이 파일을 통해 필요하면 별도의 리포트를 생성하거나 통계를 낼 수 있습니다.
- **추가 시각화 도구:** 개발팀은 Jupyter 노트북 (`notebooks/analysis_visualization.ipynb` 등)을 활용하여 결과 프로파일을 커스텀 분석하기도 합니다. 관심 있는 경우 개발 저장소의 해당 노트북을 참고하십시오.

## 6. 잉크 분석 기능

시스템은 렌즈의 잉크 개수와 색상을 자동으로 분석하는 **2가지 방법**을 제공합니다:

### 6.1 Zone-Based vs Image-Based 분석

#### Zone-Based Analysis (구역 기반)
- **방법**: SKU 설정에 정의된 Zone 구조를 기반으로 각 Zone의 대표 색상을 추출합니다.
- **장점**: SKU 기준값과 직접 비교가 가능하며, 판정이 명확합니다.
- **단점**: SKU 설정이 부정확하면 잉크 개수를 잘못 판단할 수 있습니다.
- **사용 시점**: 일반적인 검사 및 OK/NG 판정

#### Image-Based Analysis (이미지 기반 - InkEstimator)
- **방법**: 이미지 전체 픽셀을 분석하여 GMM(Gaussian Mixture Model)으로 잉크 군집을 찾습니다.
- **장점**: SKU 설정과 무관하게 실제 잉크 개수를 추정할 수 있습니다.
- **단점**: 도트 패턴 등에서 과잉 검출(3→2 보정 필요) 가능성이 있습니다.
- **사용 시점**: Zone-Based 결과와 불일치 시 원인 진단, 신규 SKU 기준값 설정 시

### 6.2 Web UI에서 잉크 정보 확인하기

Web UI의 **잉크 정보(Ink Info)** 탭에서 두 분석 결과를 함께 확인할 수 있습니다:

1. **Zone-Based Analysis 섹션 (파란색 헤더)**
   - 검출된 잉크 개수 (예: 3개)
   - 각 잉크의 Lab 값 및 RGB/HEX 색상 배지
   - Zone 매핑 정보 (Zone C → Ink 1, Zone B → Ink 2 등)

2. **Image-Based Analysis 섹션 (녹색 헤더)**
   - GMM으로 검출된 잉크 개수 (예: 2개)
   - 각 잉크의 Lab 값, RGB/HEX 색상 배지, 픽셀 비율(Weight)
   - **Meta 정보**:
     - `Mixing Correction Applied`: 혼합색 보정 여부 (true/false)
     - `Sample Count`: 분석에 사용된 픽셀 수
     - `BIC Score`: 모델 선택 지표 (낮을수록 좋음)

### 6.3 Mixing Correction이란?

**Mixing Correction**은 도트 인쇄 패턴에서 발생하는 "가짜 중간 톤"을 감지하여 제거하는 기능입니다.

**발생 원인**:
- 2가지 잉크로 도트 인쇄된 렌즈의 경우, 도트 밀도 차이로 인해 시각적으로 "중간 톤"이 나타남
- GMM은 이를 3번째 잉크로 오판할 수 있음

**보정 로직**:
1. 3개 군집이 검출되었을 때, Dark-Mid-Light 순서로 정렬
2. Mid 점이 Dark와 Light를 잇는 직선 위에 있는지 계산 (투영 거리 < 3.0)
3. 선형 배치가 확인되면 Mid를 혼합으로 판단하고 2개 잉크로 병합

**결과 해석**:
- `Mixing Correction Applied: true` → 실제 잉크는 2개, 3번째는 혼합색
- `Mixing Correction Applied: false` → 3개 모두 독립적인 잉크

### 6.4 결과 불일치 시 대처법

Zone-Based와 Image-Based 결과가 다를 경우:

**Case 1: Zone-Based 3개, Image-Based 2개 (Mixing Correction)**
- **원인**: 도트 패턴 렌즈, SKU 설정이 혼합 톤을 독립 Zone으로 정의
- **조치**: Image-Based 결과를 참고하여 SKU 설정의 `expected_zones`를 2로 수정

**Case 2: Zone-Based 2개, Image-Based 3개**
- **원인**: 실제로 3가지 잉크 사용, SKU 설정이 오래됨
- **조치**: Image-Based 결과를 참고하여 SKU 기준값 재생성 (3 zones)

**Case 3: 둘 다 같은 개수지만 색상 값이 다름**
- **원인**: Zone 경계 검출 오류, 또는 샘플링 차이
- **조치**: 경계 검출 정확도 확인, 필요 시 `transition_buffer_px` 조정

### 6.5 신규 SKU 기준값 설정 시 활용

새로운 SKU를 등록할 때 Image-Based 분석을 활용하면 정확한 잉크 개수를 파악할 수 있습니다:

```bash
# 1. 양품 이미지로 분석 실행
python src/main.py --image golden_sample.jpg --sku TEMP_SKU

# 2. Web UI에서 Image-Based 결과 확인
#    - 잉크 개수 확인 (예: 2개)
#    - 각 잉크의 Lab 값 확인

# 3. 확인된 정보로 SKU 기준값 생성
python tools/generate_sku_baseline.py \
  --image golden_sample.jpg \
  --sku NEW_SKU \
  --expected-zones 2  # Image-Based 결과 참고
```

---

## 7. 판정 시스템 이해하기

시스템은 렌즈 품질을 **4단계**로 판정하며, 각 판정 결과에 따른 조치 사항을 제공합니다.

### 7.1 4단계 판정 (OK / OK_WITH_WARNING / NG / RETAKE)

#### ✅ OK (합격)
- **조건**: 모든 Zone의 ΔE가 임계값 이내, 균일도 양호 (std_L < 10.0)
- **의미**: 품질 기준을 완전히 만족하는 양품
- **조치**: 출하 가능

#### ⚠️ OK_WITH_WARNING (경고 포함 합격)
- **조건**: 모든 Zone 합격이지만, 균일도가 경계값 (10.0 ≤ std_L < 12.0)
- **의미**: 기준은 통과했으나 품질 변동성이 다소 큼
- **조치**: 추가 검토 권장, 반복 발생 시 공정 점검
- **히스테리시스**: OK와 NG 사이의 완충 구간으로, 미세한 변동에 대한 과민 반응 방지

#### ❌ NG (불합격)
- **조건**: 1개 이상의 Zone에서 ΔE 초과
- **의미**: 색상 품질 불량
- **조치**: 불량품으로 분류, `ng_reasons`에서 불량 Zone 확인
- **상세 정보**:
  - `ng_reasons`: 불량 Zone 목록 (예: `["Zone_A", "Zone_B"]`)
  - 각 Zone의 `delta_e` 값과 `threshold` 비교

#### 🔄 RETAKE (재촬영 필요)
- **조건**: 이미지 품질 문제로 정상 검사 불가
- **의미**: 검사 자체가 신뢰할 수 없음
- **조치**: 재촬영 후 다시 검사
- **발생 원인**:
  - `R1_LensNotDetected`: 렌즈 영역 검출 실패
  - `R2_CoverageLow`: 유효 픽셀 부족 (렌즈가 이미지의 < 30%)
  - `R3_TransitionAmbiguous`: Zone 경계가 불분명 (Fallback 사용)
  - `R4_UniformityLow`: 균일도 심각하게 낮음 (std_L > 12.0)
  - `R5_OverallConfidenceLow`: 전체 신뢰도 < 0.6

### 7.2 Decision Trace (판정 근거 추적)

모든 검사 결과에는 `decision_trace` 필드가 포함되어 판정 과정을 투명하게 확인할 수 있습니다:

```json
{
  "decision_trace": {
    "final": "OK_WITH_WARNING",
    "because": "All zones passed but uniformity in warning range (std_L=10.8)",
    "overrides": null,
    "zone_checks": {
      "Zone_C": {"delta_e": 2.1, "threshold": 4.0, "passed": true},
      "Zone_B": {"delta_e": 3.5, "threshold": 4.0, "passed": true},
      "Zone_A": {"delta_e": 2.8, "threshold": 4.0, "passed": true}
    },
    "uniformity_check": {
      "max_std_l": 10.8,
      "warning_threshold": 10.0,
      "retake_threshold": 12.0,
      "status": "warning"
    }
  }
}
```

**활용 방법**:
- `because`: 판정 이유를 한 문장으로 요약
- `overrides`: Zone은 통과했지만 다른 이유로 RETAKE된 경우 표시
- `zone_checks`: 각 Zone의 ΔE 값과 통과 여부
- `uniformity_check`: 균일도 점검 상세 내역

### 7.3 Next Actions (권장 조치)

판정 결과와 함께 제공되는 `next_actions` 필드는 다음 단계를 안내합니다:

**OK 판정**:
```json
"next_actions": ["출하 승인"]
```

**OK_WITH_WARNING 판정**:
```json
"next_actions": [
  "품질 모니터링 강화",
  "반복 발생 시 공정 점검"
]
```

**NG 판정**:
```json
"next_actions": [
  "불량품 분류",
  "Zone_A 색상 편차 원인 분석 (ΔE=5.2 > 4.0)"
]
```

**RETAKE 판정**:
```json
"next_actions": [
  "조명 개선 후 재촬영",
  "렌즈 위치 조정",
  "카메라 초점 확인"
]
```

### 7.4 Confidence Score (신뢰도 점수)

모든 검사 결과에는 `confidence` 점수 (0.0~1.0)가 포함됩니다:

- **0.9 이상 (HIGH)**: 매우 신뢰할 수 있는 결과
- **0.7~0.9 (GOOD)**: 신뢰할 수 있는 결과
- **0.6~0.7 (REVIEW)**: 재검토 권장
- **0.6 미만 (LOW)**: RETAKE 권장

**Confidence 구성 요소** (`confidence_breakdown`):
```json
{
  "pixel_count_score": 0.95,      // 유효 픽셀 충분도
  "transition_score": 0.90,       // Zone 경계 명확도
  "std_score": 0.85,              // 균일도
  "sector_uniformity": 0.92,      // 섹터별 일관성
  "lens_detection": 0.95,         // 렌즈 검출 신뢰도
  "overall": 0.91                 // 종합 점수
}
```

**활용 방법**:
- Confidence가 낮은 경우 `confidence_breakdown`에서 원인 파악
- `transition_score`가 낮으면 → Zone 경계 검출 문제
- `std_score`가 낮으면 → 균일도 문제 (재촬영 권장)

---

## 8. FAQ (자주 묻는 질문)

### 8.1 잉크 분석 관련

**Q1. Zone-Based와 Image-Based 결과가 다른 이유는?**

Zone-Based와 Image-Based는 서로 다른 방식으로 색상을 분석합니다:

- **Zone-Based**: SKU에 정의된 Zone 구조를 따라 각 Zone의 평균 색상을 계산
  - 장점: SKU 기준과 직접 비교 가능, 빠름
  - 단점: Zone 경계에 영향받음, SKU 설정에 의존적

- **Image-Based**: 이미지 전체 픽셀을 GMM으로 클러스터링
  - 장점: Zone 구조와 무관하게 실제 잉크 검출
  - 단점: 샘플링과 알고리즘에 영향받음

**색상 값이 다른 이유**:
- Zone-Based는 Zone 내 **모든 픽셀 평균** (경계, 반사 포함)
- Image-Based는 **하이라이트/배경 제외 후 순수 잉크만 샘플링**

**어느 것이 정확한가?**
- 잉크 **개수** 확인: Image-Based가 더 정확
- Zone별 **색상 편차** 확인: Zone-Based가 유용
- 신규 SKU 설정: 두 결과를 함께 참고

---

**Q2. Mixing Correction은 언제 발생하나요?**

다음 조건을 **모두** 만족할 때 발생합니다:

1. **GMM이 3개 군집 검출**
2. **Dark-Mid-Light 순서로 정렬 가능**
3. **Mid 점이 Dark↔Light 직선 위에 위치** (상대 거리 < 0.15)
4. **Mid 가중치가 중간 범위** (5% < weight < 70%)

**예시**:
```json
// Mixing Correction 적용됨
{
  "raw_clusters": 3,
  "final_inks": 2,
  "meta": {
    "correction_applied": true,
    "distance": 5.94,
    "relative_distance": 0.143  // < 0.15 ✓
  }
}
```

**주의**: 실제로 3가지 잉크를 사용했는데 보정되면 안 됩니다. 이 경우 `linearity_thresh`를 높이거나 (기본 3.0 → 4.0) Image-Based 결과를 무시하고 Zone-Based를 신뢰하세요.

---

**Q3. RETAKE가 자주 나오는 경우 대처법은?**

RETAKE는 이미지 품질 문제로 정상 검사가 불가능할 때 발생합니다. Reason code별 해결책:

**R1_LensNotDetected** (렌즈 검출 실패)
- ✅ 조명: 균일한 조명 사용, 과다 반사 제거
- ✅ 배경: 단순하고 균일한 배경 (흰색/검정색 권장)
- ✅ 위치: 렌즈가 중앙에 위치, 이미지의 30% 이상 차지
- ⚙️ 설정: `config/system_config.json`의 `min_radius`, `max_radius` 조정

**R2_CoverageLow** (유효 픽셀 부족)
- ✅ 렌즈 크기: 이미지에서 렌즈가 더 크게 보이도록 줌인
- ✅ 해상도: 최소 800×800 픽셀 이상 권장

**R3_TransitionAmbiguous** (Zone 경계 불분명)
- ✅ 초점: 카메라 초점이 맞는지 확인
- ✅ 해상도: 더 높은 해상도로 촬영
- ⚙️ 설정: SKU의 `expected_zones` 값 확인

**R4_UniformityLow** (균일도 낮음, std_L > 12.0)
- ✅ 조명: 편향 조명 제거, 다중 조명 사용
- ✅ 렌즈: 실제로 불량일 가능성 (얼룩, 오염)
- ⚙️ 설정: `illumination_correction` 활성화

**R5_OverallConfidenceLow** (신뢰도 < 0.6)
- ✅ 종합 점검: R1~R4 모두 확인
- 📊 분석: `confidence_breakdown`에서 낮은 항목 확인

**빈도 높은 RETAKE 대응**:
1. 최근 10장 결과 분석 → 공통 Reason code 파악
2. 해당 원인에 집중 대응 (조명 / 초점 / 배경)
3. 문제 지속 시 카메라 캘리브레이션 점검

---

**Q4. Confidence Score가 낮은 이유는?**

`confidence_breakdown`을 확인하여 원인을 파악하세요:

```json
{
  "confidence_breakdown": {
    "pixel_count_score": 0.95,    // ← 유효 픽셀 수
    "transition_score": 0.55,     // ← 낮음! Zone 경계 불명확
    "std_score": 0.85,
    "sector_uniformity": 0.78,
    "lens_detection": 0.95,
    "overall": 0.73               // ← 0.7대 (REVIEW 구간)
  }
}
```

**항목별 대응**:
- `transition_score` 낮음 → Zone 경계 검출 실패 (초점, 해상도 확인)
- `std_score` 낮음 → 균일도 문제 (조명, 렌즈 품질)
- `sector_uniformity` 낮음 → 섹터별 편차 큼 (국부 결함, 조명 편향)
- `lens_detection` 낮음 → 렌즈 검출 신뢰도 낮음 (배경, 대비)

**종합 신뢰도 기준**:
- **0.9 이상**: 매우 신뢰 (HIGH)
- **0.7~0.9**: 신뢰 가능 (GOOD)
- **0.6~0.7**: 재검토 권장 (REVIEW)
- **0.6 미만**: RETAKE 권장 (LOW)

---

**Q5. 섹터별 분석(Sector Uniformity)이란?**

시스템은 렌즈를 **8개 섹터**(45°씩)로 나누어 국부 결함을 감지합니다:

```
      0°
   ↑ (Top)
   |  S1
S8 |     | S2
   |     |
←──┼─────┼──→
   |     |
S7 |     | S3
   | S4  ↓
    Bottom
```

**검출 로직**:
- 각 섹터의 L* 채널 표준편차 계산
- 한 섹터의 std_L이 다른 섹터보다 현저히 높으면 국부 결함 의심

**Risk Factor 생성**:
```json
{
  "category": "sector_uniformity",
  "severity": "high",           // std_L > 8.0
  "message": "Zone B 섹터 간 편차 높음",
  "details": {
    "zone": "B",
    "max_sector_std_L": 9.2,
    "worst_sector": 3
  }
}
```

**활용**:
- **high severity** (std_L > 8.0): 한쪽에만 얼룩/오염 가능성
- **medium severity** (5.0 < std_L ≤ 8.0): 경미한 국부 편차
- 조명 편향인지 실제 결함인지 판단 필요

---

**Q6. Image-Based 샘플링 ROI 정보는 어디서 확인하나요?**

Web UI의 **잉크 정보 → Image-Based Analysis → Meta** 섹션에서 확인 가능:

```json
"sampling_config": {
  "chroma_threshold": 6.0,        // 크로마 필터 임계값
  "L_max": 98.0,                  // 하이라이트 제거 기준
  "highlight_removed": true,      // 하이라이트 제거 여부
  "candidate_pixels": 50000,      // 필터링 후 후보 픽셀 수
  "sampled_pixels": 45000,        // 실제 사용된 픽셀 수
  "sampling_ratio": 0.90          // 샘플링 비율
}
```

**의미**:
- GMM 분석에 사용된 정확한 픽셀 수와 필터 조건 확인
- `sampling_ratio`가 너무 낮으면 (<0.5) 유효 데이터 부족 의심

---

## 9. 문제 해결 (Troubleshooting)
사용 중 만날 수 있는 자주 발생하는 문제와 해결 방법입니다:

**Q1. "SKU not found" 에러가 발생합니다.**
⇒ 지정한 SKU 코드에 대응하는 JSON 파일이 `config/sku_db/`에 존재하지 않을 때 발생합니다. SKU 파일 이름과 경로를 확인하세요 (`SKUXYZ.json` 형식). 또한 JSON 내부의 `"sku_code"` 필드값이 파일명과 일치해야 합니다. 참고: SKU 파일 생성은 3.1절을 보세요.

**Q2. "Lens detection failed" – 렌즈가 이미지에서 검출되지 않습니다.**
⇒ 이미지 품질이나 설정 문제일 수 있습니다. 먼저 이미지 밝기/초점을 확인하세요. 너무 어둡거나 밝으면 검출이 어려울 수 있습니다. 배경 단순성도 중요합니다 – 복잡한 배경보다는 균일한 배경에서 촬영된 이미지를 사용하세요.
또한 `config/system_config.json`의 렌즈 검출 파라미터(예: `min_radius`, `max_radius`)를 조정해보세요. 이미지 해상도에 따라 이 값들을 튜닝해야 할 수 있습니다.

**Q3. 색상 판정 결과가 지나치게 엄격하거나 느슨합니다.**
⇒ SKU 설정의 허용 오차(`threshold` 또는 `tolerance`) 값을 조정해야 합니다. 각 Zone별 threshold 값을 늘리면 관대해져 NG 판정이 줄고, 줄이면 엄격해져 NG 판정이 늘어납니다. 조정 시에는 ΔE 계산 특성을 감안하여 0.5 단위 정도로 미세하게 변경하는 것을 권장합니다.

**Q4. Zone 분할 결과가 부정확합니다 (예: Zone 개수가 틀리게 검출됨).**
✅ **해결 방법:**
1. **Expected Zones 힌트 활용:** SKU JSON의 `expected_zones` 값을 설정하거나 조정하세요. 이 값이 올바르게 설정되면 ZoneSegmenter가 그 개수에 맞춰 분할하려고 시도합니다. (예: 실제 Zone이 3개인 렌즈라면 `expected_zones: 3`으로 설정)
2. **스무딩 및 혼합구간 설정:** `params.transition_buffer_px` (혼합구간 버퍼)나 프로파일 스무딩 필터 크기 등을 조정하면 경계 검출 안정성이 높아집니다. 해당 파라미터들은 이미지 노이즈나 도트무늬 렌즈 등에 대응하기 위한 것으로, 약간씩 값을 바꿔보며 최적치를 찾습니다.
3. **영상 품질 확인:** Zone 경계가 애매하게 보일 정도로 흐릿한 이미지라면 정확한 분할이 어렵습니다. 선명한 이미지를 사용하고, 필요시 대비 향상 등의 전처리를 고려하세요.

**Q5. 처리 속도가 예상보다 느립니다.**
⇒ 이미지 해상도가 매우 높으면 처리에 시간이 많이 걸릴 수 있습니다. 가능하면 검사용 해상도를 적절히 낮춰주세요 (예: 4K 이미지를 1080p 정도로 다운샘플링).
또한 `--visualize` 옵션을 사용하는 경우 그래프 및 이미지를 추가 생성하므로 부하가 커집니다. 대량 배치 시에는 꼭 필요한 경우에만 시각화를 활성화하세요.
참고: 보다 자세한 성능 최적화 내용과 주요 병목 분석은 [PERFORMANCE_ANALYSIS.md](../design/PERFORMANCE_ANALYSIS.md) 문서를 참고할 수 있습니다. Radial Profiling 최적화 등으로 처리 속도를 크게 향상시킬 수 있습니다.

---

## 10. 배포 및 운영 안내
마지막으로, 시스템을 실제 운영 환경에서 사용하기 위한 배포 방법을 간략히 설명합니다 (자세한 내용은 [배포 가이드](DEPLOYMENT_GUIDE.md) 참조).

### 10.1 Docker로 실행하기
이 프로젝트는 Docker 컨테이너로 손쉽게 배포할 수 있습니다. Docker를 사용하면 환경 세팅을 일관되게 유지할 수 있습니다.
1. **이미지 빌드:** 프로젝트 루트에서 제공된 스크립트를 실행하거나 수동으로 Docker 빌드합니다.
    ```bash
    ./scripts/build_docker.sh   # 빌드 스크립트 사용 (IMAGE_NAME 등 내부 정의)
    ```
    (위 스크립트는 기본적으로 `colormeter:dev` 태그로 이미지를 빌드합니다. 수동으로 하려면 `docker build -t colormeter:dev .` 명령을 사용할 수 있습니다.)
2. **컨테이너 실행:** Docker 이미지를 실행하여 서비스를 시작합니다.
    ```bash
    ./scripts/run_docker.sh --image data/raw_images/OK_001.jpg --sku SKU001
    ```
    (위 스크립트는 1회성 검사를 수행합니다. 지속적인 서버 모드로 실행하려면 `docker compose up -d --build` 명령을 사용할 수 있습니다.)
3. **결과 및 볼륨:** 호스트의 `./data`, `./config`, `./results` 디렉토리가 컨테이너 내부로 마운트됩니다. 컨테이너 내부에서 생성된 결과나 로그는 호스트의 `results/` 폴더에서 확인 가능합니다.
   *배포 시 추가 고려사항과 문제 해결 팁은 [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)를 참고하십시오 (예: Windows 파일공유 설정, Linux 퍼미션 오류 해결 등).*

### 10.2 환경 변수 설정
컨테이너 환경 또는 로컬 환경에서 다음과 같은 환경 변수를 통해 동작을 제어할 수 있습니다:
- `LOG_LEVEL` – 로그 출력 수준을 설정합니다. (`DEBUG`, `INFO`(기본), `WARNING`, `ERROR`)
- `DATA_PATH` – 기본 데이터 디렉토리 경로를 변경할 때 사용합니다. (`data/`가 기본값이며, 하위에 `raw_images`, `results` 등을 두는 구조를 가정)
- `(기타)` 그 밖에 필요에 따라 `SKU` (기본 SKU 코드 지정) 등의 변수가 사용됩니다. *주의: 임시로 쓰는 용도로만 사용하며, 정식 운영 시에는 SKU 코드는 각 검사 요청별로 지정하는 것을 권장합니다.*

문의나 추가 도움이 필요하면, 프로젝트 개발팀에 Slack으로 연락하거나 저장소의 Issues에 남겨주세요. 내부 위키에도 유용한 팁이 업데이트되오니 참고하시기 바랍니다.
