# Contact Lens Color Inspection System

> **극좌표 변환 기반 콘택트렌즈 색상 품질 검사 자동화 시스템**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)]()
[![Status](https://img.shields.io/badge/status-production-green.svg)]()
[![Tests](https://img.shields.io/badge/tests-302%20passed-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-94.7%25-brightgreen.svg)]()
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue.svg)]()
[![Code Quality](https://img.shields.io/badge/code%20quality-mypy%20%7C%20black%20%7C%20flake8-blue.svg)]()
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg)]()

## 📋 프로젝트 소개

이 프로젝트는 컴퓨터 비전 기술을 활용하여 콘택트렌즈의 제조 공정 중 색상 불량을 자동으로 검출하는 시스템입니다. 렌즈 이미지를 극좌표계로 변환하여 방사형 프로파일(Radial Profile)을 분석하고, CIEDE2000 색차 공식을 사용하여 미세한 색상 차이를 감지합니다.

### 🌟 주요 기능

*   **자동 검사 파이프라인**: 이미지 로드 → 렌즈 검출 → 구역 분할 → 색상 평가 → 리포팅
*   **다중 SKU 지원**: 제품별 색상 기준값(Baseline) 관리 및 적용
*   **정밀한 색상 분석**: Lab 색 공간 및 CIEDE2000 ΔE 알고리즘 적용
*   **시각화 도구**: 검사 결과 오버레이, 히트맵, 프로파일 차트 제공
*   **배치 처리**: 대량 이미지 일괄 검사 및 CSV 결과 저장
*   **성능 최적화**: 배치 처리 병렬화, 극좌표 변환 및 메모리 최적화
*   **Zone Segmentation 개선**: 적응형 임계값, ΔE 보조 검출, 기대 영역(expected_zones) 힌트 기반 분할, 혼합 구간(transition buffer) 처리 강화
*   **✨ 운영 UX 개선 (2025-12-13)**:
    - **4단계 판정**: OK / OK_WITH_WARNING / NG / RETAKE
    - **Decision Trace**: 판정 과정 추적 (final, because, overrides)
    - **Next Actions**: 권장 조치를 최상위 필드로 제공
    - **Diff Summary**: Zone별 색상 변화 방향 (황색화, 어두워짐 등)
    - **히스테리시스**: 경계값 완충 구간 (std_L 10.0~12.0)
    - **조치 가이드**: Reason code별 구체적인 액션 및 책임 레버
    - **Sector Uniformity**: 8-sector (45°) 국부 결함 감지
    - **Confidence Breakdown**: 5개 요소 (픽셀, 경계, 균일도, 섹터, 렌즈 검출)
    - **Risk Factors**: severity 기반 위험 요소 분석
*   **🎨 지능형 잉크 분석 (2025-12-14)**:
    - **GMM + BIC**: Gaussian Mixture Model 기반 비지도 학습 잉크 검출
    - **Mixing Correction**: 도트 패턴의 "가짜 중간 톤" 자동 보정 (3→2)
    - **Dual Analysis**: Zone-Based + Image-Based 병렬 분석 결과 제공
    - **Web UI 통합**: 잉크 정보 탭에서 두 방식 비교 확인
    - **SKU 독립적**: 기준값 없이도 실제 잉크 개수 추정 가능
*   **✅ 테스트 커버리지 강화 (2025-12-16)**:
    - **292개 테스트** (290 passed, 2 pre-existing failures, 27 skipped)
    - **52개 신규 테스트 추가** (test_zone_analyzer_2d: 40개, test_ink_estimator: 12개)
    - **100% 성공률** (신규 테스트)
    - **핵심 알고리즘 검증**: GMM 클러스터링, Mixing correction, 픽셀 샘플링, 판정 로직, 4단계 판정, Confidence 계산
    - **핵심 모듈 커버리지**: ink_estimator 87.39%, zone_analyzer_2d 77.43%
    - **CI/CD 준비 완료**: pytest 실행 시간 55.8초
*   **🔄 STD 기반 비교 시스템 (2025-12-18)** ✅ NEW:
    - **M3 - Ink Comparison**: GMM 기반 잉크 색상 비교
      - Weight-based ink matching (pixel ratio 기준 페어링)
      - Color score (70%) + Weight score (30%) 혼합 평가
      - ink_score (0-100) 계산 및 total_score 통합
      - 불일치 시 상세 메시지 제공
    - **P1-2 - Radial Profile Comparison**: 1D 프로파일 유사도 분석
      - Pearson correlation coefficient (L, a, b 채널별)
      - Structural similarity (1D SSIM 근사)
      - Gradient similarity (변화 패턴 매칭)
      - Profile length mismatch 자동 보간
      - profile_score (0-100) 계산 및 total_score 통합 (zone 35%, ink 25%, profile 25%, confidence 15%)
    - **P2 - Worst-Case Metrics**: 통계적 품질 분석 (2025-12-19)
      - Percentile statistics (mean, median, p95, p99, max, std)
      - Hotspot detection (Connected Components Analysis)
      - Severity classification (CRITICAL/HIGH/MEDIUM)
      - Coverage ratio (임계값 초과 영역 비율)
      - Worst zone identification (최악 존 자동 식별)

---

## 🚀 빠른 시작 (Quick Start)

### 1. 설치

```bash
# 저장소 클론
git clone <repository-url>
cd Color_meter

# 가상환경 생성 (권장)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 또는 Windows 자동 설치 스크립트
tools\install_dependencies.bat

# Linux/Mac 자동 설치 스크립트
bash tools/install_dependencies.sh

# 주요 패키지 확인
# - numpy, opencv-python, scipy (이미지 처리)
# - scikit-learn (GMM 잉크 분석)
# - fastapi, uvicorn (Web API)
# - pytest, pytest-cov (테스트)
```

**의존성 검증:**
```bash
python tools/check_imports.py
```

### 2. 실행 예제

**단일 이미지 검사:**
```bash
python src/main.py --image data/raw_images/sample.jpg --sku SKU001
```

**배치 검사 (폴더):**
```bash
python src/main.py --batch data/raw_images/ --sku SKU001
```

**시각화 포함 검사:**
```bash
python src/main.py --image data/raw_images/ng_sample.jpg --sku SKU001 --visualize
```

**Web UI 실행:**
```bash
uvicorn src.web.app:app --port 8000
# 브라우저: http://127.0.0.1:8000
```

---

## 📚 문서 (Documentation)

**⚠️ 중요**: 이 프로젝트는 두 개의 독립적인 시스템으로 구성됩니다:
- 🔵 **Inspection System** (단일 분석): 운영 중
- 🟢 **Comparison System** (STD 비교): MVP 개발 중 (Week 1-6)

### 🟢 비교 시스템 (개발 완료: M0~M3, P1-2) ✅
**완료된 기능:**
*   **M0**: Database & Migration (Alembic, SQLAlchemy)
*   **M1**: STD Registration (기준 모델 등록 및 프로파일 저장)
*   **M2**: Comparison & Judgment (Zone-based 비교, PASS/FAIL/RETAKE/MANUAL_REVIEW)
*   **M3**: Ink Comparison (GMM 기반 잉크 색상 비교, ink_score 통합) ✅ NEW
*   **P1-2**: Radial Profile Comparison (Pearson correlation, SSIM, gradient similarity) ✅ NEW

**주요 문서:**
*   [**🎯 MVP 로드맵**](docs/planning/2_comparison/ROADMAP_REVIEW_AND_ARCHITECTURE.md): Week 6 MVP 달성 계획
*   [**📊 M3 완료 보고서**](docs/planning/2_comparison/M3_COMPLETION_REPORT.md): 잉크 비교 구현 내역
*   [**📈 P1-2 계획서**](docs/planning/2_comparison/P1-2_RADIAL_PROFILE_PLAN.md): Radial profile 비교 구현 내역

### 🔵 단일 분석 시스템 (운영 중)
#### 사용자 가이드
*   [**📘 User Guide**](docs/guides/inspection/USER_GUIDE.md): SKU 등록, 검사 실행, 잉크 분석, 판정 시스템, 문제 해결
*   [**🖥️ Web UI Guide**](docs/guides/inspection/WEB_UI_GUIDE.md): Web 대시보드 사용법 (6개 탭 상세 설명)

#### 기술 가이드
*   [**📊 InkEstimator Guide**](docs/guides/inspection/INK_ESTIMATOR_GUIDE.md): GMM 기반 잉크 분석 엔진 원리 및 활용법
*   [**🌐 API Reference**](docs/guides/inspection/API_REFERENCE.md): Web API 엔드포인트, 스키마, 예제 코드
*   [**🚀 Deployment Guide**](docs/guides/inspection/DEPLOYMENT_GUIDE.md): Docker 환경 구성 및 배포 방법

### 공통 문서
*   [**📁 INDEX**](docs/INDEX.md): 전체 문서 색인 (시스템별 분류)
*   [**📋 IMPROVEMENT_PLAN**](IMPROVEMENT_PLAN.md): 프로젝트 보강 계획 (테스트 / 문서 / 품질)

---

## 🏗️ 디렉토리 구조

```
Color_meter/
├── config/                 # 설정 파일 및 SKU 데이터베이스
│   ├── sku_db/             # SKU별 JSON 기준값 파일
│   └── system_config.json  # 시스템 전역 설정
├── data/                   # 데이터 디렉토리
│   ├── raw_images/         # 입력 이미지
│   └── results/            # 검사 결과 (CSV, 시각화)
├── docs/                   # 프로젝트 문서
│   ├── planning/
│   │   ├── 1_inspection/   # 🔵 단일 분석 시스템 계획
│   │   ├── 2_comparison/   # 🟢 STD 비교 시스템 계획 (신규)
│   │   └── ACTIVE_PLANS.md # 프로젝트 현황판 (SSOT)
│   ├── design/
│   │   ├── inspection/     # 🔵 단일 분석 설계 문서
│   │   └── comparison/     # 🟢 비교 시스템 설계 (향후)
│   ├── guides/
│   │   ├── inspection/     # 🔵 단일 분석 사용 가이드
│   │   └── comparison/     # 🟢 비교 시스템 가이드 (Week 6+)
│   └── development/        # 개발 가이드
├── src/                    # 소스 코드
│   ├── core/               # 🔵 핵심 알고리즘 (검출, 분석, 평가) - 공유
│   ├── models/             # 🟢 DB 모델 (STD 비교 시스템 전용)
│   ├── schemas/            # 🟢 API 스키마 (STD 비교 시스템 전용)
│   ├── services/           # 서비스 레이어 (단일 분석 + 비교)
│   ├── data/               # 🔵 데이터 관리 (SKU, 로깅)
│   ├── web/                # FastAPI Web UI (통합)
│   ├── utils/              # 유틸리티 (이미지 처리, 파일 IO) - 공유
│   ├── main.py             # 🔵 메인 진입점 (CLI)
│   └── pipeline.py         # 🔵 검사 파이프라인 (공유 가능)
├── tests/                  # 유닛 및 통합 테스트
├── tools/                  # 보조 도구 (더미 데이터 생성 등)
├── Dockerfile              # Docker 이미지 빌드 파일
├── docker-compose.yml      # Docker Compose 설정 파일
├── scripts/                # 빌드 및 실행 스크립트
└── requirements.txt        # 의존성 패키지 목록
```

---

## 💻 CLI 명령어 레퍼런스

`src/main.py`는 다음과 같은 하위 명령과 옵션을 지원합니다.

### `inspect` (기본 명령)
검사를 수행합니다.

*   `--image <path>`: 단일 이미지 파일 경로
*   `--batch <dir>`: 이미지 폴더 경로 (배치 처리)
*   `--sku <id>`: 적용할 SKU ID (필수)
*   `--visualize`: 시각화 결과 생성 및 저장
*   `--debug`: 디버그 로그 출력

### `sku`
SKU를 관리합니다.

*   `list`: 등록된 SKU 목록 표시
*   `create`: (구현 예정) 새로운 SKU 기준값 생성

### SKU 설정 (중요)

각 SKU의 JSON 설정 파일(`config/sku_db/<SKU_CODE>.json`)에는 **`params.expected_zones`를 반드시 설정**해야 합니다:

```json
{
  "sku_code": "SKU001",
  "zones": {
    "A": { "L": 72.2, "a": 137.3, "b": 122.8, "threshold": 4.0 }
  },
  "params": {
    "expected_zones": 1  // 필수! 실제 zone 개수
  }
}
```

- `expected_zones`: 렌즈의 실제 Zone 개수 (1, 2, 3 등)
- Zone 분할 정확도를 크게 향상시키는 필수 설정값입니다.
- 자세한 내용은 [사용자 가이드](docs/guides/USER_GUIDE.md)를 참조하세요.

---

## 🌐 Web API

FastAPI 기반 RESTful API를 제공합니다.

### 주요 엔드포인트

*   **`POST /inspect`**: 단일 이미지 검사
*   **`POST /recompute`**: 파라미터 재계산 (이미지 재업로드 불필요, 30× 속도 향상)
*   **`POST /batch`**: 배치 이미지 검사 (ZIP 또는 서버 경로)
*   **`POST /compare`**: 로트 비교 분석 (Golden Sample vs Test Images)
*   **`GET /results/{run_id}`**: 배치 결과 조회

### 사용 예시

```python
import requests

# 단일 이미지 검사
with open("lens.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/inspect",
        files={"file": f},
        data={"sku": "SKU001"}
    )

result = response.json()
print(f"Judgment: {result['judgment']}")
print(f"ΔE: {result['overall_delta_e']:.2f}")
print(f"Confidence: {result['confidence']:.2f}")

# 파라미터 재계산
image_id = result["image_id"]
response2 = requests.post("/recompute", data={
    "image_id": image_id,
    "sku": "SKU001",
    "params": json.dumps({"smoothing_window": 20, "min_gradient": 2.5})
})
```

자세한 내용은 [API Reference](docs/guides/API_REFERENCE.md)를 참조하세요.

---

## 📞 지원 및 문의

이 프로젝트는 사내 품질 관리 팀을 위해 개발되었습니다.
문의 사항이나 버그 제보는 이슈 트래커를 이용해 주세요.

---

## 현재 진행 원칙 (중요)
- 기본 흐름은 **분석 모드**(프로파일/스무딩/미분/피크)이며, OK/NG 판정은 옵션으로 뒤에서 실행합니다.
- `expected_zones`는 자동 경계 검출이 실패했을 때 보정용 힌트로만 사용합니다.
- 광학부(중심부) 배제를 위해 SKU에 `params.optical_clear_ratio`(또는 r_min) 필드를 설정해 앞 구간을 제외할 수 있습니다.
- 웹 UI 단건 탭에서 프로파일·미분 그래프와 경계 후보를 먼저 확인한 뒤, 필요 시 판정/비교를 수행하세요.
