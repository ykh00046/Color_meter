# 문서 색인

> **최종 업데이트**: 2026-01-17

## 루트 문서

| 문서                                                         | 설명                                |
| ------------------------------------------------------------ | ----------------------------------- |
| [README.md](../README.md)                                    | 프로젝트 개요 및 빠른 시작          |
| [INTEGRATION_STATUS.md](../INTEGRATION_STATUS.md)            | v7 엔진 통합 진행 현황              |
| [WEB_UI_MODERNIZATION_PLAN.md](WEB_UI_MODERNIZATION_PLAN.md) | 웹 UI 병합 및 현대화 계획           |
| [Longterm_Roadmap.md](Longterm_Roadmap.md)                   | 장기 로드맵 (Digital Proofing 등)   |
| [Legacy_Cleanup_Summary.md](Legacy_Cleanup_Summary.md)       | 레거시 코드 정리 현황               |
| [Direction_A_Implementation.md](Direction_A_Implementation.md) | Direction A 구현 상세             |

---

## 계획 문서 (planning/)

| 문서                                                                             | 설명                                      |
| -------------------------------------------------------------------------------- | ----------------------------------------- |
| [PLATE_LITE_COLOR_EXTRACTION_PLAN.md](planning/PLATE_LITE_COLOR_EXTRACTION_PLAN.md) | Plate-Lite 색상 추출 개선 계획 (물리 기반 복원) |

---

## 설계 문서 (design/)

| 문서                                                        | 설명                                            |
| ----------------------------------------------------------- | ----------------------------------------------- |
| [PLATE_ENGINE_DESIGN.md](design/PLATE_ENGINE_DESIGN.md)     | 판 분리 + 흰/검 2장 기반 색 추출 엔진           |
| [PIPELINE_DESIGN.md](design/PIPELINE_DESIGN.md)             | 검사 파이프라인 설계                            |
| [SKU_MANAGEMENT_DESIGN.md](design/SKU_MANAGEMENT_DESIGN.md) | SKU 설정 관리 설계                              |

---

## Engine v7 문서 (engine_v7/)

### 핵심 문서

| 문서                                                                   | 설명                        |
| ---------------------------------------------------------------------- | --------------------------- |
| [README.md](engine_v7/README.md)                                       | Engine v7 개요              |
| [CHANGELOG.md](engine_v7/CHANGELOG.md)                                 | 변경 이력                   |
| [ENGINE_UNIFICATION_STATUS.md](engine_v7/ENGINE_UNIFICATION_STATUS.md) | 엔진 통합 현황 (Phase 6)    |

### 플로우 및 아키텍처

| 문서                                                                   | 설명                        |
| ---------------------------------------------------------------------- | --------------------------- |
| [INSPECTION_FLOW.md](engine_v7/INSPECTION_FLOW.md)                     | 검사 플로우 다이어그램      |
| [STD_REGISTRATION_FLOW.md](engine_v7/STD_REGISTRATION_FLOW.md)         | 표준 등록 플로우            |
| [UI_FLOW_DIAGRAM.md](engine_v7/UI_FLOW_DIAGRAM.md)                     | UI 플로우 다이어그램        |
| [UI_REGISTRATION_DESIGN.md](engine_v7/UI_REGISTRATION_DESIGN.md)       | UI 등록 설계                |

### 기술 참조

| 문서                                                                         | 설명                        |
| ---------------------------------------------------------------------------- | --------------------------- |
| [MODEL_REGISTRY_API.md](engine_v7/MODEL_REGISTRY_API.md)                     | 모델 레지스트리 API         |
| [SCHEMA_V2.md](engine_v7/SCHEMA_V2.md)                                       | 데이터 스키마 v2            |
| [THRESHOLD_POLICY.md](engine_v7/THRESHOLD_POLICY.md)                         | 임계값 정책                 |
| [SINGLE_ANALYSIS_DATA_METRICS.md](engine_v7/SINGLE_ANALYSIS_DATA_METRICS.md) | 단일 분석 데이터 메트릭     |
| [RUNBOOK.md](engine_v7/RUNBOOK.md)                                           | 운영 런북                   |

---

## 가이드 (guides/)

| 문서                                      | 설명                |
| ----------------------------------------- | ------------------- |
| [USER_GUIDE.md](guides/USER_GUIDE.md)     | 시스템 사용 가이드  |

---

## 폴더 구조

```
docs/
├── INDEX.md                      # 이 파일
├── WEB_UI_MODERNIZATION_PLAN.md  # 웹 UI 현대화 계획
├── Longterm_Roadmap.md           # 장기 로드맵
├── Legacy_Cleanup_Summary.md     # 레거시 정리 현황
├── Direction_A_Implementation.md # Direction A 구현
├── Session_Summary_2026-01-17.md # 세션 요약 (작업 기록)
│
├── planning/                     # 계획 문서
│   └── PLATE_LITE_COLOR_EXTRACTION_PLAN.md
│
├── design/                       # 설계 문서
│   ├── PLATE_ENGINE_DESIGN.md
│   ├── PIPELINE_DESIGN.md
│   └── SKU_MANAGEMENT_DESIGN.md
│
├── engine_v7/                    # Engine v7 문서
│   ├── README.md
│   ├── CHANGELOG.md
│   ├── ENGINE_UNIFICATION_STATUS.md
│   ├── INSPECTION_FLOW.md
│   ├── STD_REGISTRATION_FLOW.md
│   ├── UI_FLOW_DIAGRAM.md
│   ├── UI_REGISTRATION_DESIGN.md
│   ├── MODEL_REGISTRY_API.md
│   ├── SCHEMA_V2.md
│   ├── THRESHOLD_POLICY.md
│   ├── SINGLE_ANALYSIS_DATA_METRICS.md
│   └── RUNBOOK.md
│
└── guides/                       # 사용 가이드
    └── USER_GUIDE.md
```

---

## 문서 상태

| 상태 | 의미 |
|------|------|
| ✅ Active | 현재 유효한 문서 |
| 🔄 In Progress | 작성/업데이트 중 |
| 📦 Archive | 참고용 아카이브 |

---

## 변경 이력

| 날짜       | 변경 내용                                      |
| ---------- | ---------------------------------------------- |
| 2026-01-17 | engine_v7/ 폴더 추가, 누락 문서 반영, 구조 정리 |

---

## Planning Additions

| Document | Description |
| --- | --- |
| [PLATE_LITE_AB_RESULT.md](planning/PLATE_LITE_AB_RESULT.md) | Plate-Lite A/B 비교 결과 및 결정 (초기) |
