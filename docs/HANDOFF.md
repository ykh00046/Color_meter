# Color Meter 인계 문서 (기술 스펙 + 현황)

**작성일**: 2026-01-29
**대상**: 동료 개발자, 외주 개발자
**목적**: 프로젝트 구조, 핵심 기능, 현황, 방향성, 운영/디버깅 방법을 상세히 전달

---

## 1) 프로젝트 요약

**Color Meter**는 콘택트렌즈(컬러 렌즈)의 인쇄 색상/패턴 품질을 이미지 기반으로 분석하는 시스템입니다. 타사 렌즈를 분석하고, 자사 렌즈를 유사하게 만들기 위한 **측정 → 비교 → 조정**의 반복 워크플로우를 지원합니다.

핵심 흐름(README/ROADMAP 기반):
```
[타사 렌즈] → 측정/수치화 → AI 분석(계획) → 자사 BOM/레시피 추천
[자사 렌즈] → 측정/수치화 → 타사 대비 비교 → 차이량 도출 → AI 조정 권고(계획)
```

현재 구현된 범위는 **v7 엔진 기반의 검사/등록/비교**, **웹 UI**, **Plate-Lite(White/Black 페어 기반 경량 plate 분석)** 입니다.

---

## 2) 레포지토리 구조 (핵심 디렉터리)

```
Color_meter/
├── src/
│   ├── engine_v7/               # v7 분석 엔진 (메인)
│   │   ├── core/                # 분석 코어 (measure/signature/decision/anomaly/gate/plate 등)
│   │   ├── configs/             # 엔진 기본 설정 (default.json 등)
│   │   ├── api.py               # 엔진 파사드 API
│   │   └── tests/               # 엔진 단위 테스트
│   ├── web/                     # FastAPI 웹 앱 + Jinja 템플릿 + 정적 JS/CSS
│   │   ├── app.py               # 메인 서버 엔트리
│   │   ├── routers/             # REST API 라우터 (/api/v7, /api/inspection)
│   │   ├── templates/           # UI 템플릿
│   │   └── static/              # 프론트 JS/CSS/이미지
│   ├── services/                # 서비스 레이어 (analysis_service 등)
│   ├── models/                  # DB 모델 (InspectionHistory 등)
│   ├── schemas/                 # Pydantic 스키마
│   ├── utils/                   # 유틸리티
│   └── config/                  # v7 경로/환경 설정
├── config/                      # SKU 설정 JSON (config/sku_db/*.json)
├── data/                        # 입력/샘플 이미지
├── results/                     # 검사 결과 (웹/배치)
├── v7_results/                  # v7 전용 결과/디버그
├── scripts/                     # 운영 스크립트 (register_std, run_signature_engine 등)
├── tools/                       # 진단/보정/유틸리티
├── tests/                       # 앱/라우터/E2E 테스트
├── docs/                        # 문서 (로드맵, 설계, 런북 등)
└── color_meter.db               # SQLite DB (InspectionHistory)
```

---

## 3) 실행/운영 진입점

### 3.1 Web UI (권장 운영 경로)
- 실행 스크립트: `scripts/run_web_ui_noreload.bat`
- 실제 실행 커맨드: `uvicorn src.web.app:app --host 0.0.0.0 --port 8000`
- 브라우저: `http://localhost:8000`

> 참고: 문서/가이드 일부에 8888 포트가 등장합니다. 현재 스크립트와 서버 래퍼는 **8000 포트**를 사용합니다. (정확한 포트는 `scripts/server_wrapper.py` 참고)

### 3.2 엔진 CLI (검증/배치)
- STD 등록: `scripts/register_std.py`
- STD 학습: `scripts/train_std_model.py`
- 검사 실행: `scripts/run_signature_engine.py`
- 요약 리포트: `scripts/summarize_results.py`
- 메트릭 리포트: `scripts/metrics_report.py`
- 무결성 체크: `scripts/integrity_check.py`
- 하우스키핑: `scripts/housekeeping.py`

---

## 4) 핵심 워크플로우 (v7 기준)

### 4.1 STD 등록/검증/활성화
- **등록**: STD 이미지(LOW/MID/HIGH)를 모델로 저장 + 레지스트리 업데이트
- **검증(등록 검증)**: `phase=STD_REGISTRATION`으로 안정성/분리도 확인
- **활성화**: STD_REGISTRATION 통과 후 ACTIVE 포인터 갱신

관련 문서:
- `docs/engine_v7/STD_REGISTRATION_FLOW.md`
- `docs/engine_v7/RUNBOOK.md`
- `docs/engine_v7/MODEL_REGISTRY_API.md`

**출력 예시 라벨**
- 등록 검증: `STD_ACCEPTABLE`, `STD_RETAKE`, `STD_UNSTABLE`
- 인스펙션: `OK`, `RETAKE`, `NG_COLOR`, `NG_PATTERN`

### 4.2 검사(INSPECTION)
- Gate(품질) → Signature(색) → Anomaly(패턴) → Decision
- **패턴 기준**은 ACTIVE baseline 기반 비교
- 패턴 baseline이 없으면 `RETAKE`

관련 문서:
- `docs/engine_v7/INSPECTION_FLOW.md`
- `docs/engine_v7/RUNBOOK.md`

### 4.3 Plate-Lite (White/Black 페어)
- White/Black 이미지 쌍에서 **알파(커버리지)와 잉크색 복원**을 목표로 한 경량 plate 분석
- **현재 상태**: `plate_lite.enabled=true`, `override_plate=true` (초기 결론, 샘플 2개)
- **추가 검증 필요**: 페어 샘플 13개 이상

관련 문서:
- `docs/planning/PLATE_LITE_COLOR_EXTRACTION_PLAN.md`
- `docs/planning/PLATE_LITE_AB_RESULT.md`

---

## 5) API 구조

### 5.1 Web 앱 라우트 (HTML 페이지)
`src/web/app.py`에서 제공:
- `GET /` (메인)
- `GET /v7` (v7 UI)
- `GET /single_analysis`
- `GET /calibration`
- `GET /history`
- `GET /stats`
- `GET /design_system_demo`

### 5.2 Legacy/기본 검사 API
`src/web/app.py`에서 제공:
- `POST /inspect`
- `POST /inspect_v2`
- `POST /batch`
- `POST /recompute`
- `GET /results/{run_id}`
- `GET /results/{run_id}/{filename}`

### 5.3 V7 API (`/api/v7`)
`src/web/routers/v7.py` 참고. 주요 엔드포인트:
- `POST /api/v7/inspect`
- `POST /api/v7/analyze_single`
- `POST /api/v7/register_validate`
- `POST /api/v7/activate`, `POST /api/v7/rollback`
- `GET /api/v7/status`, `GET /api/v7/entries`, `GET /api/v7/candidates`
- `POST /api/v7/plate_gate`
- `POST /api/v7/intrinsic_calibrate`, `POST /api/v7/intrinsic_simulate`

### 5.4 Inspection History API (`/api/inspection`)
`src/web/routers/inspection.py`:
- `GET /api/inspection/history`
- `GET /api/inspection/history/{history_id}`
- `GET /api/inspection/history/session/{session_id}`
- `DELETE /api/inspection/history/{history_id}`
- `GET /api/inspection/history/stats/*`
- `GET /api/inspection/history/export`

---

## 6) 데이터/스토리지

### 6.1 DB
- SQLite: `color_meter.db`
- 테이블: `InspectionHistory` (alembic migrations 적용)
- Alembic 스크립트: `alembic/versions/*.py`

### 6.2 모델 레지스트리
- 위치: `src/engine_v7/models/index.json` 및 SKU/INK/MODE별 폴더
- 구조 예시:
```
models/
  index.json
  SKU001/
    INK_A/
      LOW/vYYYYMMDD_HHMMSS/{model.npz, model.json, meta.json}
```

### 6.3 결과/아티팩트
- `results/`, `v7_results/` 아래에 JSON/이미지 출력
- UI에서 overlay/heatmap 이미지 경로 사용

---

## 7) 설정 체계

### 7.1 엔진 설정
- 기본: `src/engine_v7/configs/default.json`
- SKU 오버라이드: `config/sku_db/*.json`
- 경로 규칙: `src/config/v7_paths.py`

### 7.2 시스템 설정
- `config/system_config.json` (렌즈 검출 관련 파라미터 등)
- 예시: `config/system_config.example.json`

### 7.3 참고: 설정 파편화 이슈
문서에 따르면 설정은 **default.json + SKU config + 런타임 override + 하드코딩**이 섞여 있어 정리 대상입니다.
관련 문서:
- `docs/TECHNICAL_REVIEW_AND_IMPROVEMENTS.md`

---

## 8) 현황 요약 (최근 상태 기준)

### 8.1 엔진 통합 현황 (2026-01-31 최종 확인)
- Phase 1~8 **전체 완료**
  - Phase 1-4: measure 구조 재편, analysis 이식
  - Phase 5: v2_diagnostics Engine B 통합 (precomputed caching 패턴)
  - Phase 6: pipeline.py 마이그레이션 (inspection_service.py v7 기반 재작성)
  - Phase 7: 레거시 API 라우터 (v7 import 전환, 레거시 라우터 삭제)
  - Phase 8: src/core/ 제거 완료
- 통합 진행률 **100%**

관련 문서:
- `INTEGRATION_STATUS.md` (정상 인코딩)
- `docs/engine_v7/ENGINE_UNIFICATION_STATUS.md` (현재 파일 인코딩 깨짐)

### 8.2 안정화 작업 (2026-01-17 요약)
- Config normalization
- Plate gate 분리
- Mask-based simulation prototype (Direction A)
- Legacy bgr_to_lab_cie 제거

관련 문서:
- `docs/Session_Summary_2026-01-17.md`
- `docs/Direction_A_Implementation.md`
- `docs/Legacy_Cleanup_Summary.md`

### 8.3 기술 부채/리스크 (2026-01-20 리뷰)
- `analyzer.py` 과도한 복잡도
- async endpoint에서 CPU-bound 실행 (블로킹 위험)
- 파일 업로드 검증 불충분
- 테스트 커버리지 낮음

관련 문서:
- `docs/TECHNICAL_REVIEW_AND_IMPROVEMENTS.md`

---

## 9) 방향성 (로드맵)

### 9.1 단기
- ~~Phase 5~7 엔진 통합~~ → **완료** (Phase 1~8 전체 완료)
- 대형 파일 리팩토링 (analyzer.py 1,436줄, v7.py 2,670줄, inspection_service.py 1,233줄)
- Plate-Lite A/B 재검증 (샘플 13+)
- Web UI 현대화 (design_system, ES6 모듈화, 템플릿 통합)

### 9.2 중장기
- Direction A: **Mask-based simulation** 고도화
- Direction B: **Intrinsic color clustering**

관련 문서:
- `docs/Longterm_Roadmap.md`
- `docs/WEB_UI_MODERNIZATION_PLAN.md`

---

## 10) 디버깅/트러블슈팅 포인트

### 10.1 자주 보는 로그/결과
- `results/` 및 `v7_results/` JSON/이미지
- `debug_output/`, `reports/`

### 10.2 Gate 실패 원인
- BLUR, ILLUMINATION, CENTER_OFFSET 등 (Runbook에 표준 reason_codes 정의)

### 10.3 Plate-Lite 관련
- White/Black 페어 순서 오류 가능 (Plan 문서에서 swap 검출/경고 로직 권장)
- `plate_lite.enabled`/`override_plate` 설정 확인

---

## 11) 인계 체크리스트 (실무용)

1. **환경 준비**
   - `requirements.txt` 설치
   - `config/sku_db`에 SKU JSON 존재 확인

2. **서버 기동**
   - `scripts/run_web_ui_noreload.bat` 실행
   - `http://localhost:8000` 접속 확인

3. **STD 등록/검증**
   - `scripts/register_std.py`로 LOW/MID/HIGH 등록
   - `run_signature_engine.py --phase STD_REGISTRATION` 확인
   - `/api/v7/activate`로 ACTIVE 설정

4. **검사 파이프라인 검증**
   - `/api/v7/inspect` 정상 응답 확인
   - `results/` 결과 생성 확인

5. **Plate-Lite 확인**
   - 페어 이미지 2장 이상으로 분석
   - `plate_lite` 결과 포함 여부 확인

6. **테스트**
   - `pytest tests/` (가능한 범위)
   - `scripts/smoke_tests.py` (엔진 smoke)

---

## 12) 주의/정정 사항

- `WORK_INSTRUCTIONS.md`, `docs/engine_v7/ENGINE_UNIFICATION_STATUS.md`는 파일 내용이 깨진 상태(UTF-8 replacement). 원본 복구/재작성 필요.
- 일부 문서에서 포트(8000/8888)나 CLI 존재 여부가 상충함. 현재 운영 스크립트 기준은 8000 포트 + 웹 UI 중심.

---

## 13) 핵심 파일 빠른 참조

- 엔진 흐름: `docs/engine_v7/INSPECTION_FLOW.md`, `docs/engine_v7/STD_REGISTRATION_FLOW.md`
- 운영 런북: `docs/engine_v7/RUNBOOK.md`
- Plate-Lite: `docs/planning/PLATE_LITE_COLOR_EXTRACTION_PLAN.md`
- 통합 현황: `INTEGRATION_STATUS.md`
- 기술 리뷰: `docs/TECHNICAL_REVIEW_AND_IMPROVEMENTS.md`
- 장기 로드맵: `docs/Longterm_Roadmap.md`
- 웹 UI 계획: `docs/WEB_UI_MODERNIZATION_PLAN.md`

---

## 14) 다음 작업 제안 (우선순위)

1. ~~엔진 통합 Phase 5~7~~ → ✅ 완료 (Phase 1~8 전체 완료)
2. **Plate-Lite 재검증(샘플 확장)**
3. **대형 파일 리팩토링** (analyzer.py, v7.py, inspection_service.py)
4. **Web UI 통합/모듈화 적용**
5. **테스트 커버리지 개선**

---

## 15) 문의/추가 정보

이 문서는 현재 레포 내 문서와 코드 기준으로 작성했습니다. 실제 운영 흐름/데이터 파이프라인이 문서와 다르다면 알려주면 즉시 반영하겠습니다.
