# 🗂️ 콘택트렌즈 색상 검사 시스템 - 개발 문서 인덱스

> **프로젝트 코드명**: ColorMeter
> **최종 업데이트**: 2025-12-10
> **문서 버전**: v1.0

---

## 📚 문서 체계 안내

이 프로젝트는 **역할별 최적화된 문서 구조**를 사용합니다.
처음 보시는 분은 아래 **"역할별 읽기 가이드"**를 먼저 확인하세요.

---

## 👥 역할별 읽기 가이드

### 👔 경영진 / 프로젝트 관리자 (PM)

**추천 순서:**
1. 📄 [total.md](../total.md) - **프로젝트 전체 개요** (필수, 30분)
   - 시스템 목적 및 기대 효과
   - 6단계 개발 로드맵 (0-6단계)
   - 리스크 관리 및 대응 방안
   - PoC 계획

2. 📄 [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md) - **현재 진행 상황** (선택, 10분)
   - 현재 스프린트 상태
   - 주간 목표 및 마일스톤

**필요시 참고:**
- 예산 및 리소스 계획 → [total.md § 2, 5장](../total.md)
- 품질 검증 지표 → [total.md § 6장](../total.md)

---

### 👨‍💻 개발자 - 신규 입사 / 온보딩

**Day 1 - 프로젝트 이해 (2시간):**
1. 📄 [total.md § 1-2](../total.md) - 프로젝트 개요 및 시스템 구성 (30분)
2. 📄 [DETAILED_IMPLEMENTATION_PLAN.md § 1-3](./DETAILED_IMPLEMENTATION_PLAN.md) - 기술 스택 및 아키텍처 (1시간)
3. 📄 [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md) - 개발 환경 세팅 (30분)

**Day 2 - 코드 시작:**
- 환경 세팅: [DEVELOPMENT_GUIDE.md § 환경 구축](./DEVELOPMENT_GUIDE.md)
- 담당 모듈 파악: [DEVELOPMENT_GUIDE.md § 모듈별 담당](./DEVELOPMENT_GUIDE.md)
- API 참고: [DETAILED_IMPLEMENTATION_PLAN.md § 3](./DETAILED_IMPLEMENTATION_PLAN.md)

---

### 👨‍💻 개발자 - 실무 (매일 사용)

**아침 루틴:**
1. 📄 [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md)
   - ✅ 오늘 할 일 확인
   - 🐛 현재 블로커 이슈 확인
   - 📋 모듈별 상태 확인

**코딩 중:**
- **모듈 설계 참고**: [DETAILED_IMPLEMENTATION_PLAN.md § 3.2](./DETAILED_IMPLEMENTATION_PLAN.md)
- **코드 예시**: [DETAILED_IMPLEMENTATION_PLAN.md § 3.2.x](./DETAILED_IMPLEMENTATION_PLAN.md)
- **기술 스택**: [total.md § 4](../total.md)

**코드 리뷰 / PR:**
- 컨벤션: [DEVELOPMENT_GUIDE.md § 개발 컨벤션](./DEVELOPMENT_GUIDE.md)
- 테스트: [total.md § 6](../total.md)

---

### 🧪 QA / 품질 관리

**테스트 계획:**
1. 📄 [total.md § 6장](../total.md) - 품질 검증 지표 및 테스트 계획
   - 단위/통합/성능 테스트
   - 정확도 평가 방법
   - PoC 검증 절차

**테스트 시나리오:**
- 성능 목표: 처리 시간 ≤200ms, 검출률 ≥95%
- 정확도 평가: [total.md § 6 - 정확도 평가](../total.md)

---

### 🔬 연구 개발 (R&D)

**고급 기술 검토:**
1. 📄 [total.md § 6단계](../total.md) - PatchCore, PaDiM 등 ML 기법
2. 📄 [references/콘택트렌즈 색상 검사 시스템 개선 제안서.pdf](./references/) - 경쟁 기술 비교

---

## 🔍 주제별 빠른 찾기

### 알고리즘 / 기술

| 찾는 내용 | 문서 | 섹션 |
|----------|------|------|
| **극좌표 변환 (warpPolar)** | [total.md](../total.md) | § 2. 시스템 구성 - 색상 프로파일 추출 |
| **ΔE 계산 (CIEDE2000)** | [total.md](../total.md), [DETAILED](./DETAILED_IMPLEMENTATION_PLAN.md) | § 2, § 3.2.5 |
| **렌즈 검출 알고리즘** | [total.md](../total.md), [DETAILED](./DETAILED_IMPLEMENTATION_PLAN.md) | § 2, § 3.2.2 |
| **Zone 분할 로직** | [total.md](../total.md), [DETAILED](./DETAILED_IMPLEMENTATION_PLAN.md) | § 2, § 3.2.4 |
| **SKU 관리 시스템** | [total.md](../total.md), [DETAILED](./DETAILED_IMPLEMENTATION_PLAN.md) | § 2, § 3.2.8 |

### 모듈별 상세 설계

| 모듈 | 상세 문서 |
|------|----------|
| ImageLoader | [DETAILED § 3.2.1](./DETAILED_IMPLEMENTATION_PLAN.md#321-imageloader-영상-로더) |
| LensDetector | [DETAILED § 3.2.2](./DETAILED_IMPLEMENTATION_PLAN.md#322-lensdetector-렌즈-검출) |
| RadialProfiler | [DETAILED § 3.2.3](./DETAILED_IMPLEMENTATION_PLAN.md#323-radialprofiler-r-프로파일-추출) |
| ZoneSegmenter | [DETAILED § 3.2.4](./DETAILED_IMPLEMENTATION_PLAN.md#324-zonesegmenter-zone-분할) |
| ColorEvaluator | [DETAILED § 3.2.5](./DETAILED_IMPLEMENTATION_PLAN.md#325-colorevaluator-색상-평가-및-판정) |
| Visualizer | [DETAILED § 3.2.6](./DETAILED_IMPLEMENTATION_PLAN.md#326-visualizer-시각화) |
| Logger | [DETAILED § 3.2.7](./DETAILED_IMPLEMENTATION_PLAN.md#327-logger-데이터-저장-및-로깅) |
| SkuConfigManager | [DETAILED § 3.2.8](./DETAILED_IMPLEMENTATION_PLAN.md#328-skuconfigmanager-sku-설정-관리) |

### 개발 일정 / 프로세스

| 내용 | 문서 |
|------|------|
| **전체 로드맵 (0-6단계)** | [total.md § 5](../total.md) |
| **주차별 상세 계획** | [DETAILED § 4](./DETAILED_IMPLEMENTATION_PLAN.md) |
| **현재 스프린트** | [DEVELOPMENT_GUIDE](./DEVELOPMENT_GUIDE.md) |
| **리스크 관리** | [total.md § 7](../total.md) |
| **PoC 계획** | [total.md § 5단계](../total.md) |

### 품질 / 테스트

| 내용 | 문서 |
|------|------|
| **성능 목표** | [total.md § 6](../total.md) |
| **테스트 전략** | [total.md § 6](../total.md) |
| **정확도 평가** | [total.md § 6 - 정확도 평가](../total.md) |

---

## 📁 문서 구조 전체 맵

```
C:\X\Color_meter\
├── 📄 README.md                          # 프로젝트 소개 (GitHub 메인)
├── 📄 total.md                           # ⭐ 프로젝트 통합 계획서
│
├── 📁 docs/                              # 📚 모든 문서 모음
│   ├── 📄 INDEX.md                       # ⭐ 이 파일 (내비게이션)
│   ├── 📄 DEVELOPMENT_GUIDE.md           # ⭐ 실무 개발 가이드 (Living Doc)
│   ├── 📄 DETAILED_IMPLEMENTATION_PLAN.md # 기술 상세 설계 (2,500줄)
│   │
│   ├── 📁 references/                    # 참고 자료
│   │   ├── 콘택트렌즈_색상_검사_시스템_개발_플랜_제안서.md
│   │   └── 콘택트렌즈 색상 검사 시스템 개선 제안서.pdf
│   │
│   └── 📁 archive/                       # 과거 버전 아카이브
│       └── (이전 버전 문서들)
│
├── 📁 src/                               # 소스 코드
├── 📁 config/                            # 설정 파일
├── 📁 data/                              # 데이터
└── 📁 tests/                             # 테스트
```

---

## 📌 현재 작업 중 (Quick Access)

### 🚀 현재 단계: **1단계 - 알고리즘 프로토타입** (Week 1)

**이번 주 목표**: 개발 환경 구축 + ImageLoader 구현

**바로 가기:**
- 📋 [오늘 할 일](./DEVELOPMENT_GUIDE.md#오늘-해야-할-일)
- 🐛 [현재 이슈](./DEVELOPMENT_GUIDE.md#현재-이슈--블로커)
- 👥 [담당자 현황](./DEVELOPMENT_GUIDE.md#모듈별-담당자-및-상태)

**참고 문서:**
- 환경 세팅: [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md)
- ImageLoader 설계: [DETAILED § 3.2.1](./DETAILED_IMPLEMENTATION_PLAN.md#321-imageloader-영상-로더)
- 1단계 상세 계획: [total.md § 5 - 1단계](../total.md)

---

## 🔄 문서 업데이트 규칙

### 언제 어떤 문서를 수정하는가?

| 변경 사항 | 수정할 문서 | 담당자 | 프로세스 |
|----------|------------|--------|----------|
| **전략/일정 변경** | [total.md](../total.md) | PM | PR 필요 |
| **기술 설계 변경** | [DETAILED](./DETAILED_IMPLEMENTATION_PLAN.md) | Tech Lead | PR 필요 |
| **일일 작업 상태** | [DEVELOPMENT_GUIDE](./DEVELOPMENT_GUIDE.md) | 각 개발자 | 직접 수정 OK |
| **회의 결정사항** | [DEVELOPMENT_GUIDE](./DEVELOPMENT_GUIDE.md) | PM | 회의 후 즉시 |
| **버그/이슈** | GitHub Issues | 발견자 | Issue 등록 |
| **코드 변경** | 코드 주석 + PR | 개발자 | PR + 코드 리뷰 |

### 주의사항
- ⚠️ **total.md, DETAILED.md**는 안정적인 문서 → 함부로 수정 금지, PR 필수
- ✅ **DEVELOPMENT_GUIDE.md**는 살아있는 문서 → 매일 업데이트 OK
- 📝 문서 수정 시 **"최종 업데이트"** 날짜 갱신 필수

---

## 📞 문의 / 지원

### 문서 관련 질문
- 📧 문서 구조: Tech Lead
- 📧 내용 오류 제보: GitHub Issues에 등록

### 기술 질문
- 💬 알고리즘 질문: [DEVELOPMENT_GUIDE § Who to Ask](./DEVELOPMENT_GUIDE.md)
- 💬 환경 세팅: [DEVELOPMENT_GUIDE](./DEVELOPMENT_GUIDE.md)

---

## 🎯 처음 이 문서를 보는 분께

**30초 안내:**

1. **당신의 역할**을 위 "[역할별 읽기 가이드](#-역할별-읽기-가이드)"에서 찾으세요
2. 추천된 **문서 순서대로** 읽으세요
3. 궁금한 내용은 "[주제별 빠른 찾기](#-주제별-빠른-찾기)"에서 검색하세요

**1분 안내 (개발자):**
1. [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md) 열기
2. "환경 구축" 섹션 따라하기
3. "오늘 할 일" 확인하고 시작!

---

**Happy Coding! 🚀**

*이 문서가 도움이 되었다면, 신규 팀원에게도 공유해주세요!*
