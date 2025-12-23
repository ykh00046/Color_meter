# Color Lens Color QA System - Implementation Plan

## 0. Assumptions & Targets
- **Scope:** Radial-profile rule-based 검사 우선, 추후 딥러닝 이상탐지(PatchCore/PaDiM) 옵션 검토.
- **Timeline:** 4~5주 내 핵심 기능 완료, PoC 진행 및 현장 최적화.
- **Throughput:** 실시간 검사 200ms/장, 렌즈 직경 1000px 이상, SKU 혼입 방지.

## 1. Workstreams & Owners
- **Vision/ML:** 렌즈 검출, r-profile 추출, ΔE 계산, 동적 Zone 설정.
- **SW/UI:** 백엔드/API/DB, SKU 관리, UI/카메라 연동.
- **PM/QA:** 요구사항 정의, 정상/불량 데이터 확보, 현장 테스트 지원.

## 2. Data & HW Readiness
- **데이터:** SKU 약 30종(정상/불량), 불량 유형 약 20종(찍힘/기포/이물/미세파손 등).
- **카메라:** 5MP 이상, 렌즈 영역 1000px, 매크로/조명 고정.
- **조명:** 돔/링, 5000K, 균일도 보장; 1일 1회 캘리브레이션 SOP.

## 3. Modules & Interfaces
- **ImageLoader:** 파일/카메라 입력, 전처리(크롭, 노이즈 제거).
- **LensDetector:** 동공/외곽 원 중심/반경 검출.
- **RadialProfiler:** 0.3R~1.0R 반경 데이터 추출 (RGB -> LAB).
- **ZoneSegmenter:** ΔE 변화율 기반 Zone 자동 분할 (동공/홍채/외곽 등).
- **ColorEvaluator:** SKU 기준(LAB, 허용 오차)과 비교하여 OK/NG 판정.
- **SkuConfigManager:** SKU별 기준값 JSON 관리.
- **Visualizer:** 분석 결과 시각화 (그래프, 오버레이).
- **Logger/Reporter:** DB 저장, 리포트 생성, MES 연동.

## 4. Milestones & Deliverables

### 0단계: 준비 (완료)
- 개발환경/CI 구축, ΔE2000 유틸, 세부 계획 수립.

### 1단계: 프로토타입 (완료)
- 핵심 로직: 렌즈 검출, r-profile, LAB 변환.
- CLI/노트북: 기본 분석 기능 검증.

### 2단계: SKU 관리 자동화 (완료)
- 동적 Zone 설정, SKU JSON 구조 설계.
- 웹 UI: 기본 검사 기능 구현.

### 3단계: 로직 최적화 (완료)
- 평활화(Smoothing), 미분(Derivative) 적용으로 경계 검출 정밀도 향상.
- 4종 그래프 시각화 및 Web UI 연동.

### 4단계: UI/UX 전면 개편 및 통합 (진행 중)
- **참조:** `docs/planning/PHASE4_UI_OVERHAUL_PLAN.md` (상세 계획 별도 문서화)
- **주요 목표:** 단순 웹 뷰어를 전문 분석 장비 수준의 대시보드로 격상
- **세부 작업:**
  - [ ] **UI:** Bootstrap 5 기반의 반응형 Grid 레이아웃 (Dashboard).
  - [ ] **Viewer:** Canvas 기반 인터랙티브 이미지 뷰어 (Zoom/Pan/Overlay).
  - [ ] **DB:** SQLite 기반 분석 이력 및 메타데이터 관리 시스템.
  - [ ] **Logic:** 분석 로직 확장 (각도 분할 분석, K-means 색상 추출).
  - [ ] **Report:** 엑셀/PDF 리포트 생성 및 내보내기.

### 5단계: 현장 PoC (예정)
- 현장 설치 및 캘리브레이션 SOP 검증.
- 200ea 연속 검사 및 신뢰성 테스트.

### 6단계: 고도화 (R&D)
- 딥러닝(PatchCore) 기반 비정형 불량 검출 모듈 추가 검토.
