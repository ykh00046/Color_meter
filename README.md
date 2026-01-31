# Color Meter - 렌즈 색상 분석 시스템

타사 렌즈의 색상을 측정/분석하여 자사에서 동일한 렌즈를 제작할 수 있도록 지원하는 시스템입니다.

## 핵심 워크플로우

```
[타사 렌즈] → 측정/수치화 → AI 분석 → 자사 BOM 추천
                                ↓
                         자사 렌즈 제작
                                ↓
[자사 렌즈] → 측정/수치화 → 비교 분석 → 차이량 도출
                                ↓
                         AI 조정 권고 → 반복
```

## 주요 기능

| 기능 | 상태 | 설명 |
|------|------|------|
| 렌즈 이미지 측정 | ✅ | 색상 추출, 잉크 세그먼테이션 |
| 서명 분석 | ✅ | 극좌표 변환, 프로파일 추출 |
| STD 등록/비교 | ✅ | 표준 모델 학습 및 비교 |
| Target vs Sample 비교 | 🔧 | 타사 vs 자사 직접 비교 (개발 예정) |
| AI 분석/추천 | 📋 | BOM 추천, 조정 권고 (계획) |

## 빠른 시작

```bash
# 환경 설정
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 서버 실행
python -m src.web.app

# 웹 UI 접속
http://localhost:8000
```

## 프로젝트 구조

```
Color_meter/
├── src/
│   ├── engine_v7/     # v7 분석 엔진 (메인)
│   │   ├── core/           # 핵심 모듈
│   │   │   ├── measure/    # 측정 (색상, 잉크)
│   │   │   ├── signature/  # 서명 분석
│   │   │   ├── decision/   # 판정 로직
│   │   │   └── pipeline/   # 분석 파이프라인
│   │   └── configs/        # 설정 파일
│   ├── web/                # FastAPI 웹 서버
│   └── pipeline.py         # 통합 파이프라인
├── docs/                   # 문서
└── data/                   # 데이터 (이미지, 모델)
```

## API 엔드포인트

| 엔드포인트 | 설명 |
|-----------|------|
| `POST /api/v7/inspect` | 렌즈 검사 |
| `POST /api/v7/register` | STD 등록 |
| `POST /api/v7/compare` | 비교 분석 |
| `GET /compare` | 비교 UI |

## 문서

- [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) - 프로젝트 로드맵 및 계획
- [INTEGRATION_STATUS.md](INTEGRATION_STATUS.md) - 엔진 통합 현황
- [docs/INDEX.md](docs/INDEX.md) - 전체 문서 색인

## 기술 스택

- **Backend**: Python, FastAPI, NumPy, OpenCV, scikit-learn
- **Frontend**: HTML/CSS/JavaScript (정적)
- **분석**: k-means 클러스터링, LAB 색공간, 극좌표 변환
