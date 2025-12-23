# 콘택트렌즈 색상 검사 시스템 작업 분담 계획

## 1. 개요

본 문서는 콘택트렌즈 색상 검사 시스템 개발 프로젝트의 효율적인 진행을 위해 개발자 A, 개발자 B의 작업 분담 계획을 명시한다.
Claude (가상의 선행 작업자)가 핵심 모듈 3개(ImageLoader, LensDetector, RadialProfiler)를 구현하고 단위 테스트를 완료하였다는 전제 하에, 남은 작업을 분담하여 진행한다.

## 2. 작업 분담 계획

### 🔴 개발자 A: 인프라 & 데이터 (병렬 작업 가능 ✅)

**목표**: 개발 환경을 구축하고, 핵심 모듈의 동작을 검증하며, 초기 데이터 및 설정을 준비한다.

1.  **Task A1: 환경 세팅 (1-2시간)**
    *   **내용**: Python 가상환경 생성 및 활성화, `requirements.txt` 기반 필수 패키지 설치.
    *   **검증**: `pytest tests/test_image_loader.py -v` 실행하여 55개 테스트 케이스 통과 확인.
    *   **결과**: 개발 환경 준비 완료.

2.  **Task A2: Git 초기화 (30분)**
    *   **내용**: Git 저장소 초기화, `.gitignore` 파일 확인, 모든 파일 Stage 후 초기 커밋 수행. 원격 저장소 연결 및 푸시.
    *   **결과**: 프로젝트 초기 버전 관리 시스템 구축.

3.  **Task A3: 샘플 데이터 수집 (반나절)**
    *   **내용**: `data/raw_images/` 경로에 양품 5장, 불량 5장 (총 10장)의 렌즈 이미지 파일 준비. (현재는 Claude가 만든 가상의 렌즈 이미지를 활용)
    *   **내용**: 각 이미지에 대한 메타데이터 (`file_name`, `sku`, `judgment`, `description`)를 포함하는 `data/raw_images/metadata.csv` 파일 작성.
    *   **결과**: 핵심 알고리즘 테스트 및 검증을 위한 초기 데이터셋 확보.

4.  **Task A4: 설정 파일 초기화 (30분)**
    *   **내용**: `config/system_config.example.json` 파일을 `config/system_config.json`으로 복사하고, 로컬 개발 환경에 맞게 (예: 카메라 미연동 모드, 디버그 로그 레벨 등) 일부 파라미터 수정.
    *   **결과**: 시스템 설정 파일 준비 완료.

---

### 🔵 개발자 B: 유틸리티 & 프로토타입 (병렬 작업 가능 ✅)

**목표**: 핵심 모듈의 활용도를 높이기 위한 공통 유틸리티를 구현하고, Jupyter Notebook을 통해 빠른 검증 환경을 구축한다.

1.  **Task B1: 유틸리티 함수 구현 (2-3시간)**
    *   **내용**: `src/utils/file_io.py`에 JSON 파일 로드/저장, 이미지 파일 로드/저장, 디렉토리 생성/관리 함수 구현. `src/utils/image_utils.py`에 이미지 리사이즈, BGR↔RGB 변환, OpenCV 기반 이미지 위에 원/텍스트 그리기 등 공통 영상 처리 유틸리티 함수 구현.
    *   **결과**: 재사용 가능한 유틸리티 라이브러리 확보.

2.  **Task B2: ConfigManager 구현 (2시간)**
    *   **내용**: `src/data/config_manager.py`에 설정 파일(`system_config.json`)을 로드하고, 중첩된 키(nested key)에 쉽게 접근할 수 있는 `ConfigManager` 클래스 구현. (예: `config.get('camera.resolution.width')`)
    *   **결과**: 유연한 시스템 설정 관리 기능.

3.  **Task B3: Jupyter 프로토타입 환경 구성 (1-2시간)**
    *   **내용**: `notebooks/01_prototype.ipynb` 파일 생성 및 환경 설정.
    *   **내용**: Claude가 구현한 3가지 핵심 모듈(`ImageLoader`, `LensDetector`, `RadialProfiler`)을 Jupyter Notebook에서 임포트하고, 샘플 이미지에 대해 실행하여 중간 결과(이미지, 그래프)를 시각화하는 코드 작성.
    *   **결과**: 핵심 모듈의 동작을 시각적으로 빠르게 검증할 수 있는 환경.

4.  **Task B4: 테스트 템플릿 준비 (1시간)**
    *   **내용**: `tests/conftest.py` 파일에 `pytest fixtures` (예: 임시 파일 경로, 더미 이미지 생성 픽스처) 추가.
    *   **내용**: `tests/test_template.py` 파일을 생성하여 새로운 모듈 개발 시 활용할 수 있는 단위 테스트 템플릿 제공.
    *   **결과**: 효율적인 테스트 코드 작성을 위한 기반 마련.

---

## 3. 타임라인 및 마일스톤

**Day 1 (오늘)** - **핵심 모듈 구현 및 환경 준비**
*   **Claude**: `ImageLoader`, `LensDetector`, `RadialProfiler` 구현 및 테스트 완료.
*   **개발자 A**: 환경 세팅, Git 초기화, 샘플 데이터 수집 (가상 데이터), 설정 파일 초기화.
*   **개발자 B**: 유틸리티, `ConfigManager`, Jupyter 프로토타입 환경 구성, 테스트 템플릿 준비.

**Day 2 (내일)** - **알고리즘 고도화 및 파이프라인 통합**
*   **Claude**: `ZoneSegmenter`, `ColorEvaluator` 구현 및 테스트 완료.
*   **개발자 A, B**: `ImageLoader`부터 `ColorEvaluator`까지의 파이프라인을 `src/main.py`에 통합하고, `notebooks/01_prototype.ipynb`를 완성하여 최종 검증.

**Day 3** - **SKU 관리 및 전체 테스트**
*   **Claude**: `SkuConfigManager`, `Logger` 구현 및 테스트 완료.
*   **전체**: 모든 모듈을 통합하여 10장 샘플 이미지에 대해 전체 파이프라인 테스트 및 최종 검증.

---

## 4. 기타 문서 및 소통 채널

*   **문서**:
    *   `WORK_ASSIGNMENT.md` (이 문서): 작업 분담 상세 계획
    *   `CLAUDE_WORK_SUMMARY.md`: Claude 작업 완료 보고서
    *   `docs/DEVELOPMENT_GUIDE.md`: 실무 가이드 (매일 업데이트)
    *   `src/core/image_loader.py` 등: 핵심 모듈 소스 코드 및 Docstring
*   **소통**: Slack 채널을 통한 실시간 소통 및 진행 상황 공유. GitHub Pull Request를 통한 코드 리뷰.
*   **타임라인**: 각 개발자는 위에 명시된 타임라인에 따라 작업을 진행하며, 블로킹 이슈 발생 시 즉시 공유하여 해결 방안을 모색한다.

---
**END OF DOCUMENT**
