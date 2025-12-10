# 🚀 개발 실무 가이드 (Living Document)

> **⚠️ 이 문서는 매일 업데이트됩니다**
>
> **최종 업데이트**: 2025-12-10
> **현재 단계**: 0단계 - 준비 단계 (Week 0)
> **다음 마일스톤**: 1단계 시작 (2025-12-17 예정)

---

## ✅ 오늘 해야 할 일 (Today - 2025-12-10)

### 🎯 전체 팀 목표
- [ ] 개발 환경 세팅 완료
- [ ] Git 저장소 초기화 및 권한 설정
- [ ] 샘플 데이터 수집 계획 확정

### 👤 개인별 할 일

**김개발 (ML/Vision 엔지니어)**
- [ ] Python 3.10 가상환경 생성
- [ ] requirements.txt 패키지 설치 확인
- [ ] OpenCV warpPolar 테스트 코드 작성
- [ ] 샘플 렌즈 이미지 5장 확보

**이비전 (SW/UI 엔지니어)**
- [ ] Git 저장소 클론 및 브랜치 전략 확인
- [ ] 프로젝트 디렉토리 구조 생성
- [ ] PyQt6 설치 및 Hello World 테스트
- [ ] SQLite 테스트 DB 생성

**박매니저 (PM/도메인 전문가)**
- [ ] 품질팀과 ΔE 허용 기준 논의
- [ ] SKU별 양품 샘플 수집 일정 조율 (최소 3개 SKU × 30장)
- [ ] 하드웨어 (카메라, 조명) 사양 최종 확정

---

## 🗓️ 현재 스프린트 (Week 0: 준비 단계)

**기간**: 2025-12-10 ~ 2025-12-17 (1주)

**목표**: 개발 환경 구축 및 데이터 수집 계획 수립

### 📊 진행 상황 (3/10 완료)

- [x] 프로젝트 문서 구조 정리 (INDEX.md, DEVELOPMENT_GUIDE.md)
- [x] 기술 스택 결정 (Python 3.10, OpenCV 4.8.1)
- [x] 역할 분담 완료
- [x] Claude 핵심 모듈 3개 구현 및 테스트 완료
- [ ] Python 가상환경 세팅 (개발자 A)
- [ ] Git 저장소 초기화 (개발자 A)
- [ ] 코드 스타일 가이드 공유
- [ ] 샘플 데이터 수집 (목표: 10장) (개발자 A + PM)
- [ ] 하드웨어 사양 확정 (PM)
- [ ] 요구사항 재확인 (품질팀 미팅) (PM)
- [ ] 캘리브레이션 컬러차트 구매 (PM)

### 📅 이번 주 일정

| 날짜 | 주요 작업 | 담당 |
|------|----------|------|
| **12/10 (화)** | 환경 세팅, 문서 정리 | 전체 |
| **12/11 (수)** | Git 세팅, requirements.txt | 개발팀 |
| **12/12 (목)** | 샘플 데이터 수집 시작 | PM + 품질팀 |
| **12/13 (금)** | 요구사항 재확인 회의 | 전체 |
| **12/16 (월)** | 하드웨어 사양 최종 확정 | PM |
| **12/17 (화)** | **1단계 킥오프** 🎉 | 전체 |

---

## 🏗️ 개발 환경 구축 (Setup Guide)

### 1️⃣ Python 환경 세팅

```bash
# 1. Python 3.10 설치 확인
python --version  # Python 3.10.x 이상

# 2. 가상환경 생성
cd C:\X\Color_meter
python -m venv venv

# 3. 가상환경 활성화
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. pip 업그레이드
python -m pip install --upgrade pip

# 5. 패키지 설치
pip install -r requirements.txt

# 6. 설치 확인
python -c "import cv2; print(cv2.__version__)"  # 4.8.1 확인
python -c "import numpy; print(numpy.__version__)"  # 1.26.x 확인
```

### 2️⃣ Git 저장소 설정

```bash
# 1. 저장소 초기화 (이미 완료되었다면 스킵)
cd C:\X\Color_meter
git init

# 2. .gitignore 설정
# (아래 내용 참고)

# 3. 첫 커밋
git add .
git commit -m "Initial commit: 프로젝트 구조 및 문서"

# 4. 원격 저장소 연결 (GitHub/GitLab 주소)
git remote add origin <저장소_URL>
git push -u origin main
```

### 3️⃣ .gitignore 설정

프로젝트 루트에 `.gitignore` 파일 생성:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
virtualenv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# 데이터 (용량 큰 파일)
data/raw_images/*.jpg
data/raw_images/*.png
data/ng_images/

# 로그
logs/*.log
*.log

# 설정 (민감 정보 포함 가능)
config/system_config.json

# 모델 (용량 큼)
models/*.pth
models/*.h5

# 테스트 결과
test_results/
```

### 4️⃣ IDE 설정 (VS Code 권장)

**추천 확장 프로그램:**
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-toolsai.jupyter",
    "charliermarsh.ruff",
    "tamasfe.even-better-toml",
    "yzhang.markdown-all-in-one"
  ]
}
```

**settings.json 설정:**
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "editor.formatOnSave": true,
  "editor.rulers": [88, 120]
}
```

---

## 📦 requirements.txt

```txt
# 영상 처리
opencv-python==4.8.1
opencv-contrib-python==4.8.1
scikit-image==0.22.0
pillow==10.1.0

# 수치 계산
numpy==1.26.2
scipy==1.11.4

# 색상 과학
colormath==3.0.0

# 시각화
matplotlib==3.8.2
seABorn==0.13.0
plotly==5.18.0

# GUI
PyQt6==6.6.1
pyqtgraph==0.13.3

# 데이터 관리
pandas==2.1.4
h5py==3.10.0
sqlalchemy==2.0.25

# 설정 관리
pydantic==2.5.3
python-dotenv==1.0.0

# 로깅
loguru==0.7.2

# 테스트
pytest==7.4.3
pytest-cov==4.1.0

# 코드 품질
black==23.12.1
flake8==7.0.0
mypy==1.8.0

# 카메라 (선택 - 하드웨어에 따라)
# pypylon==3.0.1  # Basler 카메라
```

---

## 📋 모듈별 담당자 및 상태

| 모듈 | 담당자 | 상태 | 문서 참고 | 예상 시작일 |
|------|--------|------|----------|------------|
| **ImageLoader** | 김개발 | 🟢 완료 | [DETAILED § 3.2.1](./DETAILED_IMPLEMENTATION_PLAN.md#321-imageloader) | Week 1 |
| **LensDetector** | 김개발 | 🟢 완료 | [DETAILED § 3.2.2](./DETAILED_IMPLEMENTATION_PLAN.md#322-lensdetector) | Week 2 |
| **RadialProfiler** | 김개발 | 🟢 완료 | [DETAILED § 3.2.3](./DETAILED_IMPLEMENTATION_PLAN.md#323-radialprofiler) | Week 2 |
| **ZoneSegmenter** | 김개발 | ⚪ 대기 | [DETAILED § 3.2.4](./DETAILED_IMPLEMENTATION_PLAN.md#324-zonesegmenter) | Week 3 |
| **ColorEvaluator** | 김개발 | ⚪ 대기 | [DETAILED § 3.2.5](./DETAILED_IMPLEMENTATION_PLAN.md#325-colorevaluator) | Week 3 |
| **Visualizer** | 이비전 | ⚪ 대기 | [DETAILED § 3.2.6](./DETAILED_IMPLEMENTATION_PLAN.md#326-visualizer) | Week 4 (2단계) |
| **Logger** | 이비전 | ⚪ 대기 | [DETAILED § 3.2.7](./DETAILED_IMPLEMENTATION_PLAN.md#327-logger) | Week 4 (2단계) |
| **SkuConfigManager** | 김개발 + 이비전 | ⚪ 대기 | [DETAILED § 3.2.8](./DETAILED_IMPLEMENTATION_PLAN.md#328-skuconfigmanager) | Week 5 (2단계) |

**범례:**
- ⚪ 대기 (Not Started)
- 🟡 개발 중 (In Progress)
- 🟢 완료 (Completed)
- 🔴 블로킹 (Blocked)

---

## 🐛 현재 이슈 & 블로커

### 🚨 블로커 (긴급)
> 현재 블로킹 이슈 없음

### ⚠️ 중요 이슈

**#1: OpenCV warpPolar axis 확인 필요**
- **상태**: ✅ 해결됨 (RadialProfiler 구현 시 `axis=1` 적용 완료)
- **설명**: total.md의 코드 예시에서 `polar_img.mean(axis=0)`이 맞는지 검증 필요
- **담당**: 김개발
- **기한**: 2025-12-11 (내일)
- **참고**: [DETAILED § 3.2.3](./DETAILED_IMPLEMENTATION_PLAN.md#323-radialprofiler)
- **해결책**:
  ```python
  # 검증 후 확정:
  # polar_img shape: (maxRadius, 360, 3)
  # 올바른 코드: polar_img.mean(axis=1)  # 360도 방향 평균
  ```

**#2: 샘플 데이터 확보 일정**
- **상태**: 🟡 진행 중
- **설명**: SKU별 양품 이미지 최소 30장씩 필요
- **담당**: 박매니저 + 품질팀
- **기한**: 2025-12-12 (목요일 회의)
- **현황**:
  - SKU-A: 0/30장
  - SKU-B: 0/30장
  - SKU-C: 0/30장

### ✅ 해결된 이슈
- 없음 (프로젝트 시작 단계)

---

## 💡 개발 컨벤션 (Quick Reference)

### Git Commit 메시지

```bash
# 형식
<type>: <subject>

<body (선택)>

# Type 종류
feat:     새 기능 추가
fix:      버그 수정
docs:     문서 변경
style:    코드 스타일 (formatting, 세미콜론 등)
refactor: 리팩토링
test:     테스트 추가/수정
chore:    빌드, 설정 변경

# 예시
feat: ImageLoader 클래스 구현

- 이미지 로드 기능 추가
- 노이즈 제거 전처리 구현
- 단위 테스트 작성

Refs: #123
```

### 브랜치 전략

```bash
# 브랜치 명명 규칙
feature/<기능명>     # 새 기능
fix/<버그명>         # 버그 수정
docs/<문서명>        # 문서 작업
refactor/<모듈명>    # 리팩토링

# 예시
git checkout -b feature/image-loader
git checkout -b fix/lens-detection-bug
git checkout -b docs/update-readme

# 작업 완료 후
git push origin feature/image-loader
# GitHub에서 PR 생성 → 코드 리뷰 → Merge
```

### 코드 스타일

**Python (Black + Flake8)**
```python
# 1. Black 포매터 사용 (88자 제한)
# 저장 시 자동 포맷팅 설정 권장

# 2. Type Hints 필수
def detect_lens(image: np.ndarray) -> tuple[float, float, float]:
    """렌즈 검출 함수.

    Args:
        image: 입력 이미지 (H, W, 3)

    Returns:
        (center_x, center_y, radius) 튜플
    """
    pass

# 3. Docstring: Google 스타일
# 4. Import 순서: 표준 라이브러리 → 서드파티 → 로컬
import os
from typing import Optional

import numpy as np
import cv2

from src.core.utils import preprocess
```

### 파일/클래스 명명

```python
# 파일명: snake_case
image_loader.py
lens_detector.py

# 클래스명: PascalCase
class ImageLoader:
    pass

class LensDetector:
    pass

# 함수/변수명: snake_case
def load_image(file_path: str) -> np.ndarray:
    pass

lens_radius = 100
```

### 디렉토리 구조 (확정)

```
C:\X\Color_meter\
├── config/                   # 설정 파일
│   ├── system_config.json
│   └── sku_db/
│       ├── SKU001.json
│       └── SKU002.json
│
├── data/                     # 데이터 (Git 미포함)
│   ├── raw_images/          # 원본 테스트 이미지
│   ├── ng_images/           # NG 샘플 아카이브
│   └── logs/                # 검사 로그
│
├── docs/                     # 📚 문서
│   ├── INDEX.md                   # 문서 내비게이션 허브
│   ├── DEVELOPMENT_GUIDE.md       # 실무 가이드 (매일 업데이트)
│   ├── DETAILED_IMPLEMENTATION_PLAN.md  # 기술 상세 명세
│   └── references/                # 원본 제안서
│
├── src/                      # 소스 코드
│   ├── __init__.py
│   ├── main.py              # 엔트리포인트
│   │
│   ├── core/                # 핵심 알고리즘
│   │   ├── __init__.py
│   │   ├── image_loader.py
│   │   ├── lens_detector.py
│   │   ├── radial_profiler.py
│   │   ├── zone_segmenter.py
│   │   ├── color_evaluator.py
│   │
│   ├── ui/                  # GUI
│   │   ├── __init__.py
│   │   ├── main_window.py
│   │   ├── plotting.py
│   │   └── settings_dialog.py
│   │
│   ├── data/                # 데이터 관리
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   └── sku_manager.py
│   │
│   └── utils/               # 유틸리티
│       ├── __init__.py
│       ├── camera.py
│       └── file_io.py
│
├── tests/                   # 테스트
│   ├── test_image_loader.py
│   ├── test_lens_detector.py
│   ├── test_radial_profiler.py
│   └── ...
│
├── notebooks/               # Jupyter 실험
│   └── prototype.ipynb
│
├── .gitignore
├── requirements.txt
├── README.md
└── total.md
```

---

## 📞 Who to Ask (연락처)

| 질문 유형 | 담당자 | 연락처 |
|----------|--------|--------|
| **기술 설계 질문** | 김개발 (Tech Lead) | Slack: @kim-dev |
| **코드 리뷰** | 김개발 | GitHub PR |
| **UI/UX 관련** | 이비전 | Slack: @lee-vision |
| **품질 기준 (ΔE 등)** | 박매니저 + 품질팀 | Slack: @quality-team |
| **일정/리소스** | 박매니저 (PM) | Slack: @pm-park |
| **하드웨어 (카메라 등)** | 하드웨어 엔지니어 | Slack: @hardware |
| **문서 오류** | GitHub Issues | - |

---

## 🔗 자주 쓰는 링크

### 개발 도구
- 🐙 [GitHub Repository](링크_추가_필요)
- 📊 [Jira/Trello Board](링크_추가_필요)
- 💬 [Slack #colormeter](링크_추가_필요)

### 문서
- 📄 [INDEX.md - 문서 내비게이션](./INDEX.md)
- 📄 [total.md - 프로젝트 개요](../total.md)
- 📄 [DETAILED - 기술 상세](./DETAILED_IMPLEMENTATION_PLAN.md)

### 참고 자료
- 📚 [OpenCV Documentation](https://docs.opencv.org/4.8.0/)
- 📚 [scikit-image](https://scikit-image.org/)
- 📚 [PyQt6 Docs](https://www.riverbankcomputing.com/static/Docs/PyQt6/)
- 📚 [ΔE 계산 (CIEDE2000)](https://en.wikipedia.org/wiki/Color_difference#CIEDE2000)

### 데이터
- 📁 [샘플 이미지 공유 폴더](링크_추가_필요)
- 📁 [컬러 캘리브레이션 차트](링크_추가_필요)

---

## 📝 회의록 (최근 3건)

### 2025-12-10 (화) - 킥오프 미팅
**참석**: 김개발, 이비전, 박매니저

**결정 사항**:
1. ✅ 문서 구조를 INDEX + DEVELOPMENT_GUIDE 방식으로 확정
2. ✅ Python 3.10, OpenCV 4.8.1 사용
3. ✅ Week 0 목표: 환경 구축 + 샘플 데이터 10장 확보

**액션 아이템**:
- [ ] 김개발: requirements.txt 작성 및 테스트 (12/11)
- [ ] 이비전: Git 저장소 초기화 (12/11)
- [ ] 박매니저: 품질팀과 ΔE 기준 논의 (12/13)

---

### 향후 회의 일정
- **12/13 (금) 10:00**: 요구사항 재확인 회의 (품질팀 참석)
- **12/17 (화) 14:00**: 1단계 킥오프 미팅

---

## 🎓 학습 자료 (신규 입사자용)

### 필수 개념
1. **극좌표 변환**: [OpenCV warpPolar Tutorial](https://docs.opencv.org/4.8.0/da/d54/group__imgproc__transform.html)
2. **LAB 색공간**: [Understanding LAB Color Space](https://en.wikipedia.org/wiki/CIELAB_color_space)
3. **ΔE (색차)**: [Color Difference CIEDE2000](https://zschuessler.github.io/DeltaE/learn/)

### 추천 읽기
- 📖 [total.md § 2](../total.md) - 시스템 구성 (30분)
- 📖 [DETAILED § 3.2](./DETAILED_IMPLEMENTATION_PLAN.md) - 모듈별 설계 (1시간)

---

## 🔄 다음 스프린트 미리보기

### Week 1 (1단계 시작)
**목표**: ImageLoader + LensDetector 프로토타입

**예상 작업**:
- ImageLoader 클래스 구현
- 렌즈 검출 알고리즘 (Hough + Contour)
- 단위 테스트 작성
- Jupyter Notebook 프로토타입

**참고 문서**:
- [total.md § 5 - 1단계](../total.md)
- [DETAILED § 4.1](./DETAILED_IMPLEMENTATION_PLAN.md)

---

## 📌 중요 알림

### ⚠️ 주의사항
1. **total.md, DETAILED.md는 함부로 수정 금지** → PR 필수
2. **이 문서(DEVELOPMENT_GUIDE.md)는 매일 업데이트** → 직접 수정 OK
3. **코드 푸시 전 반드시 Black 포맷터 실행**
4. **민감 정보 (카메라 IP, 비밀번호 등) Git에 올리지 말 것**

### ✅ 좋은 습관
- 🕘 매일 아침 이 문서의 "오늘 해야 할 일" 확인
- 🕒 작업 완료 시 즉시 체크박스 체크
- 💬 막히는 부분은 Slack에 즉시 질문
- 📝 새로운 이슈 발견 시 GitHub Issues 등록

---

**최종 업데이트**: 2025-12-10 by 김개발
**다음 업데이트 예정**: 2025-12-11 (매일)

---

**Happy Coding! 🚀**

*질문이나 제안 사항은 Slack #colormeter 채널로 언제든지 공유해주세요!*