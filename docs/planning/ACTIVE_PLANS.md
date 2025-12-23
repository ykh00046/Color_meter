﻿# Active Project Plans & Documentation Status



**작성일:** 2025-12-12

**최종 업데이트:** 2025-12-18

**목적:** 프로젝트 문서의 위계 구조, 신뢰도 상태, 참조 우선순위 정의

**SSOT (Single Source of Truth):** 본 문서는 현재 유효한 계획과 레거시 문서를 구분하는 기준점입니다.



---



## 🟢 Active (Follow These First)

현재 개발의 기준이 되는 문서들입니다. 구현 시 반드시 이 문서들의 내용을 최우선으로 준수해야 합니다.



### 1. Planning & Direction

*   **[README.md](../../README.md)**: 프로젝트 개요, 설치 및 실행 방법 (✅ M3, P1-2, P2 업데이트 완료)

*   **[ORGANIZATION_AND_DIRECTION_SUMMARY.md](../planning/ORGANIZATION_AND_DIRECTION_SUMMARY.md)**: 프로젝트의 핵심 방향성 (Analysis-first, Judgment-optional).



### 2. Comparison System (STD-based QC) ✅ 최신 업데이트

작업 완료 상태:

*   **M0**: Database & Migration ✅ 완료

*   **M1**: STD Registration ✅ 완료

*   **M2**: Comparison & Judgment ✅ 완료

*   **M3**: Ink Comparison ✅ 완료 (2025-12-18)

*   **P1-2**: Radial Profile Comparison ✅ 완료 (2025-12-18)

*   **P2**: Worst-Case Metrics ✅ 완료 (2025-12-19)



관련 문서:

*   **[M3_COMPLETION_REPORT.md](archive/2_comparison/M3_COMPLETION_REPORT.md)**: 잉크 비교 구현 완료 보고서

*   **[P1-2_RADIAL_PROFILE_PLAN.md](2_comparison/P1-2_RADIAL_PROFILE_PLAN.md)**: Radial profile 비교 계획 및 구현

*   **[P2_COMPLETION_REPORT.md](archive/2_comparison/P2_COMPLETION_REPORT.md)**: 통계적 품질 분석 완료 보고서



### 3. Implementation Roadmap (Priority Order)

작업자는 자신의 역할(Backend/UI)에 따라 아래 문서들을 참조하십시오.



1.  **[PHASE7_CORE_IMPROVEMENTS.md](PHASE7_CORE_IMPROVEMENTS.md)** (⭐ **Highest Priority**)

    *   **주제:** 기초 품질 개선 (Core Quality)

    *   **내용:** r_inner/outer 자동 검출, 조명 편차 보정, 2단계 배경 마스킹, 균일성 분석.

    *   **대상:** Backend / Algorithm Engineer.

    *   *Note:* UI 개발 전 데이터 신뢰성을 확보하기 위한 선결 과제.



2.  **[PHASE4_UI_OVERHAUL_PLAN.md](PHASE4_UI_OVERHAUL_PLAN.md)** (🎨 **UI Main** ✅ Completed)

    *   **주제:** 차세대 UI/UX 및 기능 확장

    *   **내용:** Grid Layout, 인터랙티브 뷰어, Ring/Sector 히트맵, 리포트, DB 스키마.

    *   **대상:** Frontend / UI Engineer.

    *   *Note:* UI 개발의 메인 실행 계획서.



3.  **[ANALYSIS_IMPROVEMENTS.md](ANALYSIS_IMPROVEMENTS.md)** (⚙️ **Backend Spec**)

    *   **주제:** 분석 로직 상세 및 Backend 고도화

    *   **내용:** API 상세 명세, Fallback 로직, 에러 처리, 단위 테스트 계획.

    *   **대상:** Backend Engineer.

    *   *Note:* PHASE4 UI를 뒷받침하는 Backend의 상세 기술 명세.



### 3. User Guides

*   **[USER_GUIDE.md](../guides/USER_GUIDE.md)**: 사용자 매뉴얼.

*   **[WEB_UI_GUIDE.md](../guides/WEB_UI_GUIDE.md)**: 웹 인터페이스 사용 가이드.



---



## 🟡 Reference (Technical Background)

구현 시 참고할 수 있는 배경 지식, 초기 설계, 알고리즘 이론 문서입니다. 최신 계획(Active)과 충돌 시 Active 문서가 우선합니다.



### 알고리즘 및 초기 기획

*   **[ANALYSIS_UI_DEVELOPMENT_PLAN.md](ANALYSIS_UI_DEVELOPMENT_PLAN.md)**: 알고리즘 상세 이론 (미분, 스무딩, 피크 검출, Change Point Detection).

*   **[DETAILED_IMPLEMENTATION_PLAN.md](DETAILED_IMPLEMENTATION_PLAN.md)**: 초기 기획 및 기술 선택 배경 (제안서 비교, 경쟁 기술 분석, 초기 로드맵).



### 시스템 설계

*   **[PIPELINE_DESIGN.md](../design/PIPELINE_DESIGN.md)**: 파이프라인 아키텍처 설계 (5단계 파이프라인).

*   **[VISUALIZER_DESIGN.md](../design/VISUALIZER_DESIGN.md)**: 시각화 모듈 설계.

*   **[SKU_MANAGEMENT_DESIGN.md](../design/SKU_MANAGEMENT_DESIGN.md)**: SKU 관리 시스템 설계.



### 개발 및 운영

*   **[DEVELOPMENT_GUIDE.md](../development/DEVELOPMENT_GUIDE.md)**: 코딩 컨벤션 및 개발 환경 세팅.

*   **[DEPLOYMENT_GUIDE.md](../guides/DEPLOYMENT_GUIDE.md)**: Docker 배포 및 운영 가이드.

*   **[PERFORMANCE_ANALYSIS.md](../design/PERFORMANCE_ANALYSIS.md)**: 성능 프로파일링 결과 (88-142ms/이미지).



---



## 🔴 Deprecated & Archived

더 이상 유효하지 않거나 완료된 마일스톤의 기록입니다. 역사적 맥락 확인용으로만 보십시오.



*   **위치:** `docs/daily_reports/` 및 `docs/planning/` (구버전)

*   **포함 문서:**

    *   `INDEX.md` (구버전 문서 인덱스)

    *   `total.md` (초기 기획서)

    *   `PHASE2_IMPROVEMENTS_SUMMARY.md`

    *   `PHASE3_COMPLETION_SUMMARY.md`

    *   `PHASE3_UI_FIX_REPORT.md`

    *   `DAY*_WORK_PLAN.md` (일일 계획)

    *   기타 히스토리 문서



---



## 🔄 문서 통합 가이드 (For Contributors)



1.  **UI 수정 시:** `PHASE4_UI_OVERHAUL_PLAN.md`를 따르십시오.

2.  **알고리즘 수정 시:** `PHASE7`을 최우선으로 하고, 상세 스펙은 `ANALYSIS_IMPROVEMENTS.md`를 참조하십시오.

3.  **새로운 기능 제안:** `ACTIVE_PLANS.md`에 새 로드맵을 추가하고 상태를 정의하십시오.
