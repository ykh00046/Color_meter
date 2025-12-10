# 콘택트렌즈 색상 검사 시스템

> **Contact Lens Color Inspection System**
> 극좌표 변환(r-profile) 기반 콘택트렌즈 색상 품질 검사 자동화 시스템

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)]()

---

## 📋 프로젝트 개요

본 프로젝트는 콘택트렌즈 제조 공정에서 색상 품질을 자동으로 검사하는 시스템입니다.

**핵심 기술**:
- 극좌표 변환(r-profile)을 통한 동심원 색상 분석
- LAB 색공간 기반 CIEDE2000 ΔE 계산
- SKU별 베이스라인 자동 학습 및 관리

**주요 특징**:
- ⚡ 실시간 처리 (목표: <200ms/장)
- 🎯 높은 정확도 (목표: 검출률 95%+, 오탐률 5% 미만)
- 🔧 Multi-SKU 지원 (SKU 등록만으로 즉시 적용 가능)
- 📊 높은 설명가능성 (ΔE 값 기반 객관적 판정)

---

## 🚀 빠른 시작

### 1️⃣ 환경 설정

```bash
# Python 3.10+ 필요
python --version

# 저장소 클론
git clone <저장소_URL>
cd Color_meter

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2️⃣ 프로그램 실행

```bash
# GUI 모드 (개발 중)
python src/main.py --gui

# CLI 모드 (테스트용)
python src/main.py --image data/raw_images/sample.jpg --sku SKU001
```

---

## 📚 문서 구조

프로젝트 문서는 역할과 목적에 따라 계층적으로 구성되어 있습니다.

### 🗺️ 시작점: [docs/INDEX.md](docs/INDEX.md)

**문서 내비게이션의 허브입니다.** 역할별 읽기 가이드, 주제별 빠른 링크를 제공합니다.

### 📄 핵심 문서

| 문서 | 대상 독자 | 설명 |
|------|----------|------|
| **[total.md](total.md)** | 전체 (특히 PM/경영진) | 프로젝트 전체 개요 및 실행 계획 (235줄) |
| **[docs/DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md)** | 개발자 (매일 참고) | 실무 가이드 - 오늘 할 일, 모듈 상태, 이슈 트래킹 |
| **[docs/DETAILED_IMPLEMENTATION_PLAN.md](docs/DETAILED_IMPLEMENTATION_PLAN.md)** | 개발자 (구현 시) | 기술 상세 명세 - 클래스 설계, API, 코드 예시 (2513줄) |

### 📂 참고 자료

- **[docs/references/](docs/references/)** - 원본 제안서 및 요구사항 문서

---

## 🏗️ 시스템 아키텍처

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Image Input │────▶│ Preprocessing│────▶│ Lens Detection  │
└─────────────┘     └──────────────┘     └─────────────────┘
                                                    │
                                                    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Judgment   │◀────│Zone Segment. │◀────│Radial Profiling │
│   & Report  │     │              │     │  (r-profile)    │
└─────────────┘     └──────────────┘     └─────────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │Color Quality │
                    │  Evaluation  │
                    └──────────────┘
```

**8개 핵심 모듈**:
1. `ImageLoader` - 이미지 로드 및 전처리
2. `LensDetector` - 렌즈 중심/반경 검출
3. `RadialProfiler` - 극좌표 변환 및 r-profile 추출
4. `ZoneSegmenter` - 색상 존 자동 분할
5. `ColorEvaluator` - ΔE 계산 및 품질 판정
6. `SkuConfigManager` - SKU 베이스라인 관리
7. `Visualizer` - 결과 시각화
8. `Logger` - 검사 이력 저장 및 통계

---

## 📁 프로젝트 구조

```
Color_meter/
├── docs/                          # 📚 문서
│   ├── INDEX.md                   # 문서 내비게이션 허브
│   ├── DEVELOPMENT_GUIDE.md       # 실무 가이드 (매일 업데이트)
│   ├── DETAILED_IMPLEMENTATION_PLAN.md  # 기술 상세 명세
│   └── references/                # 원본 제안서
│
├── src/                           # 소스 코드
│   ├── core/                      # 핵심 알고리즘
│   │   ├── image_loader.py
│   │   ├── lens_detector.py
│   │   ├── radial_profiler.py
│   │   ├── zone_segmenter.py
│   │   └── color_evaluator.py
│   ├── ui/                        # GUI (PyQt6)
│   ├── data/                      # 데이터 관리
│   │   ├── logger.py
│   │   └── sku_manager.py
│   └── utils/                     # 유틸리티
│
├── tests/                         # 단위/통합 테스트
├── notebooks/                     # Jupyter 실험 노트북
├── config/                        # 설정 파일
│   └── sku_db/                    # SKU 베이스라인 DB
├── data/                          # 데이터 (Git 미포함)
│   ├── raw_images/                # 테스트 이미지
│   └── logs/                      # 검사 로그
│
├── total.md                       # 프로젝트 개요 (전체용)
├── requirements.txt               # Python 패키지 목록
└── README.md                      # 이 파일
```

---

## 🛠️ 기술 스택

### 핵심 라이브러리

| 분야 | 라이브러리 | 버전 | 용도 |
|------|-----------|------|------|
| 영상 처리 | OpenCV | 4.8.1 | 극좌표 변환, 렌즈 검출 |
| 색상 과학 | colormath | 3.0.0 | CIEDE2000 ΔE 계산 |
| 수치 계산 | NumPy | 1.26+ | 배열 연산 |
| 신호 처리 | SciPy | 1.11+ | 변곡점 검출 |
| GUI | PyQt6 | 6.6+ | 사용자 인터페이스 |
| 시각화 | Matplotlib | 3.8+ | 그래프 및 차트 |
| 데이터 관리 | Pandas | 2.1+ | 로그 분석 |
| DB | SQLAlchemy | 2.0+ | 검사 이력 저장 |

### 개발 도구

- **테스트**: pytest, pytest-cov
- **코드 품질**: Black (formatter), Flake8 (linter), mypy (type checker)
- **버전 관리**: Git
- **문서화**: Markdown, Sphinx (계획 중)

---

## 📊 개발 로드맵

### Stage 0: 준비 단계 (1주) - 진행 중 ✅

- [x] 문서 구조 정리
- [x] 기술 스택 확정
- [ ] Python 환경 세팅
- [ ] 샘플 데이터 수집 (10장)
- [ ] 요구사항 재확인

### Stage 1: 프로토타입 (3주) - 예정 ⏳

- 렌즈 검출 알고리즘 구현
- 극좌표 변환 및 r-profile 생성
- 기본 ΔE 계산
- Jupyter Notebook 검증

**목표**: 10장 샘플 이미지로 알고리즘 검증 완료

### Stage 2: SKU 관리 시스템 (4주)

- SKU 베이스라인 자동 생성
- CLI/GUI 관리 도구
- 존 분할 알고리즘 고도화

**목표**: 5개 SKU 등록 및 검증 완료

### Stage 3: 품질 최적화 (4주)

- 조명 견고성 향상
- 성능 최적화 (목표: <200ms)
- 불량 데이터 테스트 (검출률 95%+)

**목표**: 성능 지표 달성

### Stage 4: GUI & 배포 (4주)

- PyQt6 GUI 통합
- 카메라 연동 (Basler 등)
- MES 연동
- 설치 패키지 생성

**목표**: 현장 배포 가능한 완성품

### Stage 5: PoC (4주)

- 현장 시범 운영 (200+ 샘플/일)
- 캘리브레이션 프로토콜 수립
- 오탐 분석 및 개선

**목표**: 현장 검증 완료

---

## 🧪 테스트

```bash
# 전체 테스트 실행
pytest

# 커버리지 포함
pytest --cov=src --cov-report=html

# 특정 모듈만 테스트
pytest tests/test_lens_detector.py -v
```

**테스트 전략**:
- 단위 테스트: 모듈별 독립 테스트 (목표: 80%+ 커버리지)
- 통합 테스트: 전체 파이프라인 안정성
- 성능 테스트: 95-백분위수 지연시간
- 정확도 테스트: 30장 양품 반복성(ΔE 분산 <0.5), 20장 불량 검출률 95%+

---

## 📈 성능 목표

| 지표 | 목표 | 현재 상태 |
|------|------|----------|
| **처리 속도** | 평균 <150ms, 95%ile <200ms | - |
| **검출률** | ≥95% (필수) / ≥98% (목표) | - |
| **오탐률** | ≤5% (필수) / ≤2% (목표) | - |
| **처리량** | 300장/분 이상 | - |
| **반복성** | ΔE 분산 <0.5 (동일 샘플) | - |

---

## 🤝 기여 가이드

### Git Workflow

```bash
# 1. 브랜치 생성
git checkout -b feature/image-loader

# 2. 작업 및 커밋
git add .
git commit -m "feat: ImageLoader 클래스 구현"

# 3. 푸시 및 PR
git push origin feature/image-loader
# GitHub에서 Pull Request 생성
```

### 커밋 메시지 규칙

```
<type>: <subject>

<body (선택)>
```

**Type 종류**:
- `feat`: 새 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 코드 스타일 (formatting)
- `refactor`: 리팩토링
- `test`: 테스트 추가/수정
- `chore`: 빌드/설정 변경

---

## 👥 팀 & 역할

| 역할 | 담당자 | 책임 |
|------|--------|------|
| **Tech Lead** | 김개발 | 알고리즘 설계, 코드 리뷰 |
| **SW/UI Engineer** | 이비전 | GUI 개발, 시스템 통합 |
| **PM** | 박매니저 | 일정 관리, 요구사항 조율 |

---

## 📞 연락처 & 지원

- **Slack**: #colormeter 채널
- **이슈 트래킹**: GitHub Issues
- **문서 질문**: [docs/INDEX.md](docs/INDEX.md) 참고

---

## 📝 라이선스

본 프로젝트는 회사 내부 사용을 위한 독점 소프트웨어입니다.

**Copyright © 2025 [회사명]. All Rights Reserved.**

---

## 🎓 참고 자료

### 핵심 개념
- [극좌표 변환 (cv2.warpPolar)](https://docs.opencv.org/4.8.0/da/d54/group__imgproc__transform.html)
- [LAB 색공간](https://en.wikipedia.org/wiki/CIELAB_color_space)
- [CIEDE2000 색차(ΔE)](https://en.wikipedia.org/wiki/Color_difference#CIEDE2000)

### 추천 읽기 순서 (신규 입사자)
1. **[total.md](total.md)** - 전체 흐름 이해 (30분)
2. **[docs/INDEX.md](docs/INDEX.md)** - 문서 구조 파악 (5분)
3. **[docs/DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md)** - 실무 가이드 (15분)
4. **[docs/DETAILED_IMPLEMENTATION_PLAN.md](docs/DETAILED_IMPLEMENTATION_PLAN.md)** - 기술 상세 (필요 시 참고)

---

## ⭐ Quick Links

- 📁 [프로젝트 개요 (total.md)](total.md)
- 🗺️ [문서 내비게이션 (INDEX.md)](docs/INDEX.md)
- 🚀 [개발 실무 가이드 (DEVELOPMENT_GUIDE.md)](docs/DEVELOPMENT_GUIDE.md)
- 🔧 [기술 상세 (DETAILED_IMPLEMENTATION_PLAN.md)](docs/DETAILED_IMPLEMENTATION_PLAN.md)

---

**프로젝트 상태**: 🟡 개발 중 (Stage 0 - 준비 단계)
**최종 업데이트**: 2025-12-10
**다음 마일스톤**: Stage 1 킥오프 (2025-12-17 예정)
