# 🚀 개발 실무 가이드 (Development Guide)

> **⚠️ 이 문서는 지속적으로 업데이트되는 Living Document입니다.**
> **최종 업데이트:** 2025-12-15

프로젝트 개발팀원 모두가 참고하는 **중앙 가이드** 문서입니다. 환경 세팅에서 코드 컨벤션, 진행 현황까지 개발 과정에 필요한 정보를 담고 있습니다. **새로 합류한 개발자**나 **현재 진행 상황을 파악하려는 팀원**에게 특히 유용합니다.

## 1. 개발 환경 구축 (Setup)
### 1.1 Python 환경 & 의존성 설치
```bash
# 1. Python 3.10+ 설치 확인
python --version   # Python 3.10.x 이상이어야 함

# 2. 가상환경 생성 (Windows 예시)
cd C:\Projects\ColorMeter
python -m venv venv

# 3. 가상환경 활성화
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. pip 최신화
python -m pip install --upgrade pip

# 5. 패키지 설치
pip install -r requirements.txt

# 6. 설치 검증
python -c "import cv2; print(cv2.__version__)"    # OpenCV 버전 출력 (예: 4.8.1)
python -c "import numpy; print(numpy.__version__)"  # NumPy 버전 출력 (예: 1.26.x)
```
**Note:** 권장 의존성 버전은 `requirements.txt`에 명시되어 있습니다. (OpenCV, NumPy 등 주요 패키지 버전은 상세 구현 계획서에 근거하여 선정됨)

### 1.2 Git 저장소 설정
```bash
# 1. Git 저장소 초기화 (최초 1회)
git init

# 2. .gitignore 설정 (필요한 내용 예시는 아래 참고)

# 3. 첫 커밋 & 원격 저장소 연결
git add .
git commit -m "Initial project structure and docs"
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```
**.gitignore 예시:** (프로젝트 루트에 생성)
```text
# Python artifacts
__pycache__/
*.py[cod]
*.so

# Virtual env
venv/
env/
ENV/

# IDE settings
.vscode/
.idea/

# OS files
.DS_Store
Thumbs.db

# Data (large or sensitive files)
data/raw_images/
data/ng_images/
results/
logs/

# Config (sensitive information)
config/system_config.json

# Models (if any)
models/

# Tests output
test_results/
```

### 1.3 IDE 및 도구
VS Code를 사용하는 경우, 팀에서 다음 확장팩을 권장합니다:
- **Python** (ms-python.python) – Python 지원
- **Pylance** (ms-python.vscode-pylance) – 강화된 IntelliSense
- **Jupyter** (ms-toolsai.jupyter) – 노트북 지원
- **Ruff** (charliermarsh.ruff) – Python 린터 (PEP8 검사 대체용)
- **Markdown All in One** (yzhang.markdown-all-in-one) – 마크다운 편집 보조

또한 `.vscode/settings.json`에 아래와 같은 설정을 해두면 편리합니다:
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe",
  "editor.formatOnSave": true,
  "editor.rulers": [88, 120],
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true
}
```

## 2. 프로젝트 디렉토리 구조
프로젝트의 폴더 구조는 다음과 같이 구성되어 있습니다 (확정):
```text
ColorMeter/
├── config/                  # 설정 파일 및 기준 데이터
│   ├── system_config.json      # 시스템 전역 설정
│   └── sku_db/                 # SKU별 기준 JSON 폴더
│       ├── SKU001.json
│       └── SKU002.json
├── data/                    # 데이터 (대용량 파일은 Git 관리 제외)
│   ├── raw_images/             # 입력 원본 이미지
│   ├── ng_images/              # NG 사례 이미지 (분석용)
│   └── results/                # 검사 결과 (JSON, overlay 이미지 등)
├── docs/                    # 📚 문서 (설계, 가이드 등)
│   ├── guides/
│   ├── development/
│   ├── design/
│   ├── daily_reports/
│   └── planning/
├── src/                     # 애플리케이션 소스 코드
│   ├── main.py                 # CLI 진입점 및 인자 파싱
│   ├── core/                   # 핵심 알고리즘 모듈
│   │   ├── image_loader.py         # 이미지 로드 & 전처리
│   │   ├── lens_detector.py        # 렌즈 검출 (원 탐지)
│   │   ├── radial_profiler.py      # 극좌표 변환 및 프로파일 추출
│   │   ├── zone_segmenter.py      # Zone 자동 분할
│   │   └── color_evaluator.py     # ΔE 계산 및 판정
│   ├── data/                   # 데이터 관리 모듈
│   │   ├── sku_manager.py         # SKUConfigManager 클래스 구현
│   │   └── logger.py              # 로깅 모듈
│   ├── web/                    # 웹 UI 모듈 (FastAPI)
│   │   ├── app.py                 # FastAPI 앱 정의 (엔드포인트 설정)
│   │   └── templates/            # 웹 UI HTML 템플릿
│   ├── visualizer/             # 시각화 모듈
│   │   ├── __init__.py
│   │   └── visualizer.py          # InspectionVisualizer 클래스 구현
│   └── utils/                  # 유틸리티 모듈 (카메라, 파일 I/O 등)
│       ├── camera.py
│       └── file_io.py
├── tests/                   # 테스트 코드
│   ├── test_image_loader.py
│   ├── test_lens_detector.py
│   ├── test_radial_profiler.py
│   └── ... (etc)
├── scripts/                 # 편의 실행 스크립트
│   ├── build_docker.sh
│   └── run_docker.sh
├── notebooks/               # Jupyter Notebook 실험 및 프로토타입
│   └── analysis_prototype.ipynb
├── requirements.txt         # 요구 패키지 목록 (버전 명시)
├── CHANGELOG.md
└── README.md
```
구조는 가능하면 변경하지 않으며, 새 파일은 위 분류에 맞게 추가합니다.

## 3. 코드 컨벤션 및 규칙
팀의 일관된 코드 스타일을 위해 다음 컨벤션을 따릅니다:

### 3.1 Git 브랜치 및 커밋 규칙
- **브랜치 전략:** 기능 단위로 분기 생성. 브랜치 이름 예시:
  - `feature/<기능명>` – 새로운 기능 개발
  - `fix/<버그명>` – 버그 수정
  - `docs/<문서명>` – 문서 작업
  - `refactor/<모듈명>` – 코드 리팩토링
  ```bash
  git checkout -b feature/visualizer-module
  git checkout -b fix/lens-detect-crash
  ```
  작업 완료 후 원격에 푸시하고 PR을 생성해 코드 리뷰를 받습니다.

- **커밋 메시지 템플릿:**
  ```
  <type>: <subject>  (한 줄 요약)

  <body> (선택사항으로 상세 설명)
  ```
  타입 예시: `feat`(기능), `fix`(버그수정), `docs`(문서), `refactor`(리팩토링), `test`(테스트), `chore`(기타 잡일)
  예:
  ```
  feat: Add ImageLoader module

  - Implemented ImageLoader.load_from_file()
  - Added basic preprocessing (denoise)
  - Includes unit tests for the new functionality
  ```

### 3.2 Python 코드 스타일
- **Formatter & Linter:** `Black`(format 88cols)으로 자동 포매팅, `Flake8`/`Ruff`로 린팅. 가능한 IDE 저장 시 자동 적용되도록 합니다.
- **타입 힌트 필수:** 모든 함수/메서드 정의에 Python type hint를 작성합니다.
- **Docstring:** Google style 또는 일관된 형태로 작성. 주요 함수/클래스에는 한 줄 설명과 Args/Returns 등을 포함합니다.
- **Import 순서:** 표준 라이브러리 -> 서드파티 -> 로컬모듈, 섹션별로 개행.
- **명명법:** `snake_case` (파일, 함수, 변수), `PascalCase` (클래스). 상수는 `UPPER_SNAKE_CASE`.

```python
# 예시
class ImageLoader:
    def load_from_file(self, path: str) -> np.ndarray:
        """Load an image from file into a NumPy array."""
        ...
```

### 3.3 테스트와 품질
- **단위 테스트 필수:** 새 모듈 작성 시 `tests/`에 대응 테스트를 작성합니다. 최소 happy path와 edge case를 검증해주세요.
- **테스트 실행:** `pytest`로 전체 테스트를 수시로 돌려봐야 합니다 (Docker 컨테이너 내에서 `docker run --rm -it ... pytest`로 실행 가능).
- **코드 리뷰:** 모든 PR은 최소 한 명 이상의 리뷰어 승인을 받아야 합니다. 특히 핵심 모듈(`src/core/` 아래)의 변경은 담당자(김개발 등)에게 리뷰 요청하세요.

### 3.4 Pre-commit Hooks (코드 품질 자동 검사)
**⭐ 신규 추가 (2025-12-15)**: 코드 품질을 자동으로 검사하는 pre-commit hook이 설정되었습니다.

#### 설치 방법
```bash
# 1. pre-commit 패키지 설치 (이미 requirements.txt에 포함됨)
pip install pre-commit

# 2. Git hook 설치
pre-commit install

# 3. 설치 확인 - 모든 파일에 대해 수동 실행
pre-commit run --all-files
```

#### 자동 검사 항목
commit 시 다음 항목들이 자동으로 검사됩니다:

1. **Black** - 코드 포매팅 (자동 수정)
   - Line length: 120 characters
   - PEP 8 스타일 자동 적용

2. **Flake8** - 코드 린팅
   - 코드 스타일 검사
   - 일부 규칙은 `.flake8` 파일에서 설정
   - 현재 일시적으로 무시되는 규칙들:
     - E226, E402, F401, F841, E722, E712, F541
     - (점진적으로 코드 개선하면서 제거 예정)

3. **isort** - Import 정렬 (자동 수정)
   - Black 프로필 사용
   - Import 순서 자동 정렬

4. **Pre-commit-hooks** - 기본 검사
   - 파일 끝 개행 추가
   - Trailing whitespace 제거
   - YAML/JSON 문법 검사
   - 대용량 파일 체크 (5MB 이상)
   - Merge conflict 마커 검사
   - Debug statements 검사

#### Hook 동작 방식
```bash
# Commit 시도
git add .
git commit -m "feat: Add new feature"

# Pre-commit hook이 자동 실행됨:
# - 문제가 없으면: Commit 진행
# - 자동 수정 가능한 문제: 파일 수정 후 다시 add 필요
# - 수동 수정 필요한 문제: 에러 메시지 확인 후 수정

# 자동 수정된 경우 다시 커밋
git add .
git commit -m "feat: Add new feature"
```

#### Hook 우회 (긴급 상황)
```bash
# 긴급한 경우에만 hook을 우회할 수 있습니다
git commit --no-verify -m "hotfix: Critical bug fix"
```
**⚠️ 주의**: Hook 우회는 긴급 상황에만 사용하고, 이후 반드시 코드 품질 문제를 수정해야 합니다.

#### 설정 파일
- `.pre-commit-config.yaml` - Pre-commit hook 설정
- `.flake8` - Flake8 린팅 규칙
- 두 파일 모두 프로젝트 루트에 위치

## 4. 모듈별 담당자 및 진행 상태
아래 표는 주요 모듈 구현의 담당자와 현재 상태를 나타냅니다. (주 단위 스프린트 계획에 따라 업데이트됨)

| 모듈 (파일) | 담당자 | 상태 | 관련 문서 | 비고 |
|---|---|---|---|---|
| ImageLoader (`core/image_loader.py`) | 김개발 | 🟢 완료 | 설계서 §3.2.1 | 이미지 파일 I/O 및 전처리 |
| LensDetector (`core/lens_detector.py`) | 김개발 | 🟢 완료 | 설계서 §3.2.2 | 원 검출 (허프 변환 기반) |
| RadialProfiler (`core/radial_profiler.py`) | 김개발 | 🟢 완료 | 설계서 §3.2.3 | 극좌표 변환 및 프로파일 |
| ZoneSegmenter (`core/zone_segmenter.py`) | 김개발 | 🟡 개발중 | 설계서 §3.2.4 | Zone 자동 분할 알고리즘 |
| ColorEvaluator (`core/color_evaluator.py`) | 김개발 | ⚪ 대기 | 설계서 §3.2.5 | ΔE 계산 및 OK/NG 판정 |
| Visualizer (`visualizer/visualizer.py`) | 이비전 | ⚪ 대기 | 시각화 설계 | 분석용 시각화 (Phase 2) |
| Logger (`data/logger.py`) | 이비전 | ⚪ 대기 | 설계서 §3.2.7 | 통합 로깅 모듈 |
| SkuConfigManager (`data/sku_manager.py`) | 김개발, 이비전 | ⚪ 대기 | SKU 설계 | SKU CRUD 및 베이스라인 |

*상태 표시: ⚪ 대기 (Not Started), 🟡 개발중, 🟢 완료, 🔴 블로커(진행 불가)*
*(업데이트: 2025-12-12 현재 Week 0 마무리 단계 – ImageLoader/RadialProfiler 등 핵심 모듈 프로토타입 완료)*

## 5. 현재 이슈 현황 (Issues & Blockers)
**🚨 긴급/블로킹:**
- 현재 블로킹 이슈 없음 (2025-12-12 기준)

**⚠️ 주요 이슈:**
1. **OpenCV warpPolar 축 설정 재검증** – 상태: 해결됨 ✅ (RadialProfiler 구현 시 axis=1로 수정 완료)
   - 설명: 극좌표 변환 후 평균 계산 축이 잘못되어 있었던 문제. 평균을 axis=1 (360도 방향)으로 해야 함을 확인.
2. **샘플 데이터 확보 일정 지연** – 상태: 진행 중 🟡 (담당: 박매니저)
   - 설명: SKU별 양품 이미지 30장씩 필요. 현재 수집 지연으로 0장/30장 (SKU-A 기준). 12/13 품질팀 미팅에서 일정 조율 예정.

*(그 외 일반 이슈 및 버그는 프로젝트 보드 또는 GitHub Issues 참조)*

## 6. 자주 찾는 참고 정보
- **프로젝트 개요** – 상세 개념 및 전체 흐름: `README.md` 또는 구 버전 통합 문서 `planning/total.md` 참고.
- **기능별 상세 설계** – `docs/design` 폴더 내 각 모듈 설계서를 참조 (예: 알고리즘 구현 세부 내용 등).
- **성능 튜닝 가이드** – 성능 개선 관련 아이디어는 `PERFORMANCE_ANALYSIS.md`에 정리됨.
- **API 명세** – FastAPI 엔드포인트와 응답 형식은 `src/web/app.py` 및 Web UI 가이드 문서에 상세.

## 7. 문의처 (Who to Ask)
프로젝트 진행 중 궁금한 사항이 있으면 아래 담당자를 찾아주세요:

| 질문 주제 | 담당자 (역할) | 연락 채널 |
|---|---|---|
| 기술 설계 및 아키텍처 | 김개발 (Tech Lead) | Slack @kim-dev |
| 코드 리뷰 & 품질 | 김개발 | GitHub PR 리뷰 |
| UI/UX 및 프론트 | 이비전 (UI 엔지니어) | Slack @lee-vision |
| 품질 기준(ΔE 등) | 박매니저 (PM) + 품질팀 | Slack @quality-team |
| 일정/리소스 조율 | 박매니저 (PM) | Slack @pm-park |
| 하드웨어(카메라 등) | 하드웨어엔지니어 | Slack @hw-engineer |
| 문서 수정/오류 보고 | (전체) | GitHub Issues 활용 |

## 8. 최근 회의 및 결정사항
- **2025-12-10 Kick-off 미팅:** 문서 구조 개편 (INDEX.md 도입 등), Python 3.10 / OpenCV 4.8.1 등 기술스택 확정, Week 0 목표(환경 구축 & 샘플 10장 확보) 설정.
- **다가오는 회의:** 12/13 요구사항 재확인 회의 (품질팀 참석), 12/17 1단계 킥오프.
*(회의 상세 내용과 과거 로그는 `docs/daily_reports/`의 회의록 섹션에 기록되어 있습니다.)*

## 9. 신규 입사자용 – 학습 자료
프로젝트 도메인 및 기술에 대한 배경 지식을 쌓는 데 도움이 되는 자료:
- **OpenCV 극좌표 변환** – 공식 docs(opencv.org) 중 warpPolar 함수 설명
- **CIELAB 색 공간** – 위키백과: CIELAB (색공간 개념 이해)
- **ΔE (색상 차이) 이해** – 기술 블로그 – ΔE 계산 방식과 시각적 의미

**권장 독서:**
1. 프로젝트 개요 문서 (요약본) – `README.md` 또는 `planning/total.md`의 시스템 구성 소개 (30분 소요)
2. 기술 설계 명세 – 상세 구현 계획서 3.2 모듈별 설계 부분 (약 1시간 소요)

*(이 개발 가이드는 프로젝트 진행 상황에 따라 수시로 업데이트됩니다. 변경 사항이 있을 때마다 팀에 공지하고 본 문서를 최신 상태로 유지해주세요.)*
