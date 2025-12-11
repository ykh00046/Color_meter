# Contact Lens Color Inspection System

> **극좌표 변환 기반 콘택트렌즈 색상 품질 검사 자동화 시스템**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)]()
[![Status](https://img.shields.io/badge/status-active-green.svg)]()

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

---

## 📚 문서 (Documentation)

*   [**사용자 가이드 (User Guide)**](docs/USER_GUIDE.md): SKU 등록, 검사 실행, 문제 해결 방법
*   [**상세 구현 계획 (Implementation Plan)**](docs/DETAILED_IMPLEMENTATION_PLAN.md): 시스템 아키텍처 및 모듈 상세 설계
*   [**개발 가이드 (Development Guide)**](docs/DEVELOPMENT_GUIDE.md): 개발 환경 설정 및 기여 방법
*   [**배포 가이드 (Deployment Guide)**](docs/DEPLOYMENT.md): Docker 환경 구성 및 배포 방법

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
├── src/                    # 소스 코드
│   ├── core/               # 핵심 알고리즘 (검출, 분석, 평가)
│   ├── data/               # 데이터 관리 (SKU, 로깅)
│   ├── ui/                 # 사용자 인터페이스 (예정)
│   ├── utils/              # 유틸리티 (이미지 처리, 파일 IO)
│   ├── main.py             # 메인 진입점 (CLI)
│   └── pipeline.py         # 검사 파이프라인
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
- 자세한 내용은 [사용자 가이드](docs/USER_GUIDE.md)를 참조하세요.

---

## 📞 지원 및 문의

이 프로젝트는 사내 품질 관리 팀을 위해 개발되었습니다.
문의 사항이나 버그 제보는 이슈 트래커를 이용해 주세요.
