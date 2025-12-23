﻿# 📖 Documentation Index (Color Meter Project)







프로젝트 관련 모든 문서들의 색인입니다. 각 문서의 목적과 주요 내용을 함께 정리하여, 필요한 정보를 빠르게 찾아볼 수 있습니다.







---







## 🎯 시스템 구분







이 프로젝트는 두 개의 독립적인 시스템으로 구성됩니다:







### 🔵 **Inspection System** (단일 분석 시스템)



- **목적**: 개별 렌즈의 절대적 품질 검사



- **기준**: SKU 설정 파일 (고정된 임계값)



- **상태**: ✅ 운영 중



- **문서 위치**: `1_inspection/`, `design/inspection/`, `guides/inspection/`







### 🟢 **Comparison System** (STD 비교 시스템) ⭐ **신규**



- **목적**: STD 샘플 대비 상대적 품질 비교



- **기준**: STD 프로파일 (동적 기준)



- **상태**: 🚧 개발 중 (MVP Week 1-6)



- **문서 위치**: `2_comparison/`, `design/comparison/`, `guides/comparison/`
- 가이드: `guides/comparison/COMPARE_UI_GUIDE.md` (STD vs Sample UI)
- 가이드: `guides/comparison/QUICK_START_M2_MAPPING.md` (Quick Start ↔ M2 매핑)
- 가이드: `guides/comparison/COMPARISON_SYSTEM_OVERVIEW.md` (시스템 개념 정의)
- 가이드: `guides/comparison/MEASUREMENT_ROADMAP.md` (측정/비교 지표 확장 순서)













---







## 🟢 Comparison System (STD 기반 비교 시스템) - 신규 개발







### 📌 실행 계획 (Execution Plan)



**Week 1-6 MVP 개발 중 - 여기부터 읽으세요!**







#### 필수 문서 (읽는 순서대로)



1. **[ROADMAP_REVIEW_AND_ARCHITECTURE.md](planning/2_comparison/ROADMAP_REVIEW_AND_ARCHITECTURE.md)** – 🎯 **MVP 로드맵 및 아키텍처 (최우선 읽기):** Week 6 MVP 달성을 위한 현실적 실행 계획. Quick Summary, 3단계 로드맵 (M0-M2 → P1 → P2), 단일 분석 vs 비교 시스템 아키텍처 분리.







2. **[WEEK1_M0_READINESS_CHECKLIST.md](planning/2_comparison/WEEK1_M0_READINESS_CHECKLIST.md)** – ✅ **Week 1 준비 완료 체크리스트 (95% 완료):** DB 스키마, ORM 모델, Pydantic 스키마 완료. Alembic 마이그레이션 및 판정 기준 워크샵 실행 대기 중.







3. **[JUDGMENT_CRITERIA_WORKSHOP.md](planning/2_comparison/JUDGMENT_CRITERIA_WORKSHOP.md)** – ⭐ **판정 기준 협의 워크샵 (Week 1 Day 2 필수):** 상관계수, ΔE 임계값, 경계 허용 오차 등 판정 기준 합의를 위한 워크샵 템플릿 (2-3시간).







#### 참고 문서 (Reference Documents)



- **[STD_BASED_QC_SYSTEM_PLAN.md](planning/2_comparison/STD_BASED_QC_SYSTEM_PLAN.md)** – 🔴 **장기 비전 문서:** 표준 이미지(STD) 기반 양산 비교 검사 시스템 전체 비전. 구조/색상 유사도 분석, 조치 권장, 합격/불합격 자동 판정 (10주 계획).







- **[TECHNICAL_ENHANCEMENTS_ADVANCED.md](planning/2_comparison/TECHNICAL_ENHANCEMENTS_ADVANCED.md)** – ⭐ **P1-P2 구현 가이드 (고급 기술):** STD 시스템 고도화를 위한 7대 핵심 기술. STD 통계 모델, Elastic Alignment, Worst-Case 지표, Ink-Aware 비교, Explainability Layer, 안정성 설계, 현상학적 분류.







- **[REVIEW_FEEDBACK_AND_IMPROVEMENTS.md](planning/2_comparison/REVIEW_FEEDBACK_AND_IMPROVEMENTS.md)** – ✅ **검토 의견 및 개선 계획:** 전문가 검토 결과 및 반영 사항. 알고리즘 최적화, DB 개선, 운영 요구사항, 리스크 완화 방안.







#### 내부 문서 (Internal)



- **[STRUCTURE_REORGANIZATION_PROPOSAL.md](planning/2_comparison/STRUCTURE_REORGANIZATION_PROPOSAL.md)** – 📁 **문서 구조 재구성 제안서:** 단일 분석 vs 비교 시스템 문서 분리 방안 (본 재구성의 기반 문서).







- **[ROADMAP_COMPARISON_REVIEW.md](planning/2_comparison/ROADMAP_COMPARISON_REVIEW.md)** – 📊 **로드맵 비교 분석:** 사용자 제안 로드맵과 기존 계획 비교 분석 (95% 일치 확인).







- **[AI_REVIEW_CONTEXT_BUNDLE.md](planning/2_comparison/AI_REVIEW_CONTEXT_BUNDLE.md)** – 📦 **AI 검토용 컨텍스트 번들:** STD QC 시스템 계획 검토를 위한 통합 문서 (현재 시스템, 목표, 기술 스택, 검토 체크리스트).







### Design (설계 문서) - 향후 추가 예정



현재 비어 있음. Week 2-6 개발 과정에서 추가될 예정:



- STD 비교 알고리즘 설계



- 판정 로직 설계



- API 설계 등







### Guides (사용 가이드) - Week 6+ 추가 예정



현재 비어 있음. MVP 완료 후 작성 예정:



- STD 등록 가이드



- 비교 검사 실행 가이드



- 비교 리포트 해석 가이드







---







## 🔵 Inspection System (단일 분석 시스템) - 운영 중







### Planning (계획 및 개선)



#### 핵심 개선 계획 (Active)



- **[PHASE7_CORE_IMPROVEMENTS.md](planning/1_inspection/PHASE7_CORE_IMPROVEMENTS.md)** – **핵심 품질 개선 계획:** 백엔드 알고리즘(조명 보정, 배경 마스킹 등)의 신뢰성 향상을 위한 기술 과제 (10개 우선순위).







- **[OPERATIONAL_UX_IMPROVEMENTS.md](planning/1_inspection/OPERATIONAL_UX_IMPROVEMENTS.md)** – ✅ **운영 UX 개선 (구현 완료):** RETAKE 상태, Diff Summary, OK 컨텍스트 등 현장 운영성 향상 기능.







- **[ANALYSIS_UI_DEVELOPMENT_PLAN.md](planning/1_inspection/ANALYSIS_UI_DEVELOPMENT_PLAN.md)** – **분석 중심 UI 개발 계획:** "분석 우선(Analysis-First)" 원칙에 따른 UI/UX 및 알고리즘 검증 도구 개발 로드맵.







- **[PHASE4_UI_OVERHAUL_PLAN.md](planning/1_inspection/PHASE4_UI_OVERHAUL_PLAN.md)** – **차세대 UI/UX 계획:** Grid Layout, 인터랙티브 뷰어 등 UI 고도화 마일스톤.







#### 완료된 개선 작업 (Completed)



- **[PHASE7_COMPLETION_REPORT.md](planning/1_inspection/PHASE7_COMPLETION_REPORT.md)** – Phase 7 전체 완료 보고서



- **[PHASE7_PRIORITY0_COMPLETE.md](planning/1_inspection/PHASE7_PRIORITY0_COMPLETE.md)** – 최우선 작업 완료



- **[PHASE7_PRIORITY3-4_COMPLETE.md](planning/1_inspection/PHASE7_PRIORITY3-4_COMPLETE.md)** – 중간 우선순위 완료



- **[PHASE7_PRIORITY5-6_COMPLETE.md](planning/1_inspection/PHASE7_PRIORITY5-6_COMPLETE.md)** – 추가 개선 완료



- **[PHASE7_PRIORITY8_COMPLETE.md](planning/1_inspection/PHASE7_PRIORITY8_COMPLETE.md)** – 테스트 개선 완료



- **[PHASE7_PRIORITY9_COMPLETE.md](planning/1_inspection/PHASE7_PRIORITY9_COMPLETE.md)** – 추가 테스트 완료



- **[PHASE7_PRIORITY10_COMPLETE.md](planning/1_inspection/PHASE7_PRIORITY10_COMPLETE.md)** – 최종 완료



- **[PHASE7_MEDIUM_PRIORITY_COMPLETE.md](planning/1_inspection/PHASE7_MEDIUM_PRIORITY_COMPLETE.md)** – 중간 우선순위 전체 완료



- **[QUICK_WINS_COMPLETE.md](planning/1_inspection/QUICK_WINS_COMPLETE.md)** – 빠른 개선 사항 완료



- **[REFACTORING_COMPLETION_REPORT.md](planning/1_inspection/REFACTORING_COMPLETION_REPORT.md)** – 리팩토링 완료



- **[DOCUMENTATION_UPDATE_COMPLETION.md](planning/1_inspection/DOCUMENTATION_UPDATE_COMPLETION.md)** – 문서 업데이트 완료



- **[CODE_QUALITY_REPORT.md](planning/1_inspection/CODE_QUALITY_REPORT.md)** – 코드 품질 개선 보고서







#### 특정 기능 개선



- **[INK_ANALYSIS_ENHANCEMENT_PLAN.md](planning/1_inspection/INK_ANALYSIS_ENHANCEMENT_PLAN.md)** – 잉크 분석 기능 개선 계획



- **[ANALYSIS_IMPROVEMENTS.md](planning/1_inspection/ANALYSIS_IMPROVEMENTS.md)** – 분석 알고리즘 개선







#### 리팩토링 및 옵션 구현



- **[OPTION2_REFACTORING_COMPLETE.md](planning/1_inspection/OPTION2_REFACTORING_COMPLETE.md)** – 옵션 2 리팩토링 완료



- **[OPTION3_PHASE7_PROGRESS.md](planning/1_inspection/OPTION3_PHASE7_PROGRESS.md)** – 옵션 3 진행 상황







#### 테스트 완료 보고서



- **[TEST_INK_ESTIMATOR_COMPLETION.md](planning/1_inspection/TEST_INK_ESTIMATOR_COMPLETION.md)** – 잉크 추정기 테스트 완료



- **[TEST_ZONE_ANALYZER_2D_COMPLETION.md](planning/1_inspection/TEST_ZONE_ANALYZER_2D_COMPLETION.md)** – 2D Zone 분석기 테스트 완료







### Design (설계 및 기술 명세)



- **[PIPELINE_DESIGN.md](design/inspection/PIPELINE_DESIGN.md)** – **검사 파이프라인 설계:** `InspectionPipeline` 모듈의 엔드투엔드 처리 흐름 설계 상세.



- **[SKU_MANAGEMENT_DESIGN.md](design/inspection/SKU_MANAGEMENT_DESIGN.md)** – **SKU 관리 모듈 설계:** SKU 설정 JSON 스키마, CRUD, 베이스라인 자동 생성 설계.



- **[VISUALIZER_DESIGN.md](design/inspection/VISUALIZER_DESIGN.md)** – **검사 결과 시각화 모듈 설계:** Overlay, Heatmap 등 시각화 기능 설계.



- **[COLOR_EXTRACTION_DUAL_SYSTEM.md](design/inspection/COLOR_EXTRACTION_DUAL_SYSTEM.md)** – **색상 추출 이중 시스템 설계:** Zone-Based(구역 기반)와 Image-Based(GMM 비지도 학습) 두 가지 잉크 검출 방법의 알고리즘, 차이점, 통합 방식 상세 설명.



- **[COLOR_EXTRACTION_COMPARISON.md](design/inspection/COLOR_EXTRACTION_COMPARISON.md)** – **색상 추출 품질 비교:** Zone-Based가 Image-Based보다 색상을 더 정확하게 추출하는 이유 (3단계 필터링, 공간적 정밀성, 노이즈 제거) 및 실험적 증거.



- **[PERFORMANCE_ANALYSIS.md](design/inspection/PERFORMANCE_ANALYSIS.md)** – **성능 분석 보고서:** 파이프라인 프로파일링 결과 및 최적화 제안.







### Guides (사용/운영 가이드)



- **[USER_GUIDE.md](guides/inspection/USER_GUIDE.md)** – **시스템 사용자 가이드:** CLI 및 웹 UI를 통해 콘택트렌즈 검사 시스템을 사용하는 방법을 단계별로 안내합니다.



- **[WEB_UI_GUIDE.md](guides/inspection/WEB_UI_GUIDE.md)** – **경량 Web UI 사용 가이드:** FastAPI 기반의 간이 웹 인터페이스 실행 방법과 주요 기능을 설명합니다.



- **[DEPLOYMENT_GUIDE.md](guides/inspection/DEPLOYMENT_GUIDE.md)** – **배포 및 환경 구성 가이드:** Docker를 이용한 컨테이너 빌드/실행 방법, 볼륨 구성, 환경 변수 설정 등을 다룹니다.



- **[API_REFERENCE.md](guides/inspection/API_REFERENCE.md)** – **API 레퍼런스:** Web API 엔드포인트 상세 설명.



- **[INK_ESTIMATOR_GUIDE.md](guides/inspection/INK_ESTIMATOR_GUIDE.md)** – **잉크 추정기 가이드:** GMM 기반 잉크 색상 추정 모듈 사용법.



- **[image_normalization.md](guides/inspection/image_normalization.md)** – **이미지 정규화 가이드:** 조명 보정 및 이미지 전처리 설명.







---







## 📚 공통 문서







### Planning (프로젝트 전체 관리)



- **[ACTIVE_PLANS.md](planning/ACTIVE_PLANS.md)** – **(SSOT) 프로젝트 계획 현황판:** 현재 유효한 계획과 레거시를 구분하고, 문서 간 우선순위를 정의하는 기준점입니다.







### Development (개발 환경 및 프로세스)



- **[DEVELOPMENT_GUIDE.md](development/DEVELOPMENT_GUIDE.md)** – **개발 실무 가이드 (Living Doc):** 개발 환경 세팅, 코드 컨벤션, Git 브랜치 전략, 디렉토리 구조 등을 다루는 실시간 업데이트 문서입니다.



- **CHANGELOG.md** – **변경 이력:** 프로젝트의 버전 별 주요 변경 사항 기록.







### Technical Guides (기술 가이드)



- **[AI_TELEMETRY_GUIDE.md](AI_TELEMETRY_GUIDE.md)** – **AI 텔레메트리 가이드:** AI 기반 코드 분석 및 성능 모니터링 시스템 사용법.



- **[ZONE_RING_ANALYSIS.md](ZONE_RING_ANALYSIS.md)** – **Zone/Ring 분석 가이드:** 렌즈의 구역별 분석 방법론.



- **[LAB_SCALE_FIX.md](LAB_SCALE_FIX.md)** – **LAB 색공간 스케일 수정:** LAB 색공간 처리 관련 버그 수정 및 표준화.



- **[SECURITY_FIXES.md](SECURITY_FIXES.md)** – **보안 수정 사항:** 보안 관련 버그 수정 및 개선 내역.



- **[TEST_COVERAGE_REPORT.md](TEST_COVERAGE_REPORT.md)** – **테스트 커버리지 보고서:** 현재 테스트 커버리지 현황 및 개선 계획.







### Special Topics



- **[INK_MASK_INTEGRATION_PLAN.md](INK_MASK_INTEGRATION_PLAN.md)** – **잉크 마스크 통합 계획:** 잉크 영역 검출 알고리즘 통합 방안.



- **[TEST_2D_ZONE_ANALYSIS.md](reports/TEST_2D_ZONE_ANALYSIS.md)** – **2D Zone 분석 테스트:** 2D 구역 분석 알고리즘 테스트 결과.







### Project Management



- **[IMPROVEMENT_PLAN.md](planning/IMPROVEMENT_PLAN.md)** – **프로젝트 전반 개선 계획**



- **[PROJECT_COMPREHENSIVE_REVIEW.md](PROJECT_COMPREHENSIVE_REVIEW.md)** – **프로젝트 종합 검토**







---







## 📂 Archives (참고용 기록)







과거 일일 보고서 및 레거시 문서는 다음 위치에 보관됩니다:



- **planning/archive/** - 과거 계획 문서



- **daily_reports/archive/** - 일일 작업 보고서







---







## 🧭 빠른 시작 가이드







### 🟢 비교 시스템 개발에 참여하는 경우 (신규 개발자)



1. **[ROADMAP_REVIEW_AND_ARCHITECTURE.md](planning/2_comparison/ROADMAP_REVIEW_AND_ARCHITECTURE.md)** 읽기 (Quick Summary 섹션부터)



2. **[WEEK1_M0_READINESS_CHECKLIST.md](planning/2_comparison/WEEK1_M0_READINESS_CHECKLIST.md)** 현재 진행 상황 확인



3. **[STD_BASED_QC_SYSTEM_PLAN.md](planning/2_comparison/STD_BASED_QC_SYSTEM_PLAN.md)** 장기 비전 이해







### 🔵 기존 시스템 유지보수/개선하는 경우



1. **[USER_GUIDE.md](guides/inspection/USER_GUIDE.md)** 시스템 사용법 학습



2. **[PIPELINE_DESIGN.md](design/inspection/PIPELINE_DESIGN.md)** 파이프라인 구조 이해



3. **[PHASE7_CORE_IMPROVEMENTS.md](planning/1_inspection/PHASE7_CORE_IMPROVEMENTS.md)** 현재 개선 계획 확인







### 📚 문서 작성자



1. **[ACTIVE_PLANS.md](planning/ACTIVE_PLANS.md)** 문서 우선순위 및 SSOT 확인



2. 새 문서는 적절한 시스템 폴더에 배치:



   - 비교 시스템 → `planning/2_comparison/`, `design/comparison/`



   - 단일 분석 → `planning/1_inspection/`, `design/inspection/`



   - 공통 사항 → 루트 폴더







---







**➡ 참고:**



- 문서 내용은 지속적으로 업데이트되며, **굵은 텍스트**나 *강조*로 표시된 부분은 중요 포인트입니다.



- 최신 정보 확인을 위해 각 문서의 업데이트 날짜를 참고하세요.



- 🟢 **비교 시스템**과 🔵 **단일 분석 시스템**은 독립적이지만, 핵심 알고리즘(`InspectionPipeline`)을 공유합니다.




## 추가 문서
- `MEASUREMENT_UI_REQUIREMENTS.md`: Inspection/Comparison 공통 지표 UI 요구사항
