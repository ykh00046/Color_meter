# v7.py 라우터 분할 설계서

> **Summary**: 2,670줄 모놀리식 v7.py를 7개 파일로 분할하는 상세 설계
>
> **Project**: Color Meter
> **Author**: PDCA Auto
> **Date**: 2026-01-31
> **Status**: Draft
> **Plan Doc**: [v7-router-split.plan.md](../../01-plan/features/v7-router-split.plan.md)

---

## 1. Architecture Overview

### 1.1 AS-IS

```
src/web/routers/
├── v7.py               (2,670줄 - 15 routes + 47 helpers + 5 Pydantic models)
└── inspection.py        (기존 유지)
```

### 1.2 TO-BE

```
src/web/routers/
├── v7.py                (~50줄 - 서브 라우터 조립 진입점)
├── v7_helpers.py        (~500줄 - 공통 유틸리티, 인코더, 상수)
├── v7_registration.py   (~660줄 - 등록 + 모델 빌드 + 승인 팩)
├── v7_activation.py     (~360줄 - 활성화 + 롤백 + 스냅샷/베이스라인)
├── v7_inspection.py     (~560줄 - 검사 + 테스트 + 단일 분석)
├── v7_metrics.py        (~220줄 - 메트릭 + 트렌드 + 삭제)
├── v7_plate.py          (~330줄 - 플레이트 게이트 + 교정 + 시뮬레이션)
└── inspection.py        (기존 유지)
```

---

## 2. Module Design

### 2.1 `v7_helpers.py` — 공통 유틸리티 (~500줄)

여러 도메인 모듈에서 공유하는 함수와 상수를 배치.

#### 2.1.1 상수 및 모듈 변수

| 항목 | 원본 줄 | 사용처 |
|------|---------|--------|
| `logger` | L34 | 전체 |
| `V7_SUBPROCESS_MAX_CONCURRENCY` | L41 | inspection, registration |
| `V7_SUBPROCESS_TIMEOUT_SEC` | L42 | inspection, registration |
| `_SUBPROC_SEM` | L43 | inspection, registration |

#### 2.1.2 클래스

| 클래스 | 원본 줄 | 사용처 |
|--------|---------|--------|
| `NumpyEncoder` | L65-83 | inspection (analyze_single) |

#### 2.1.3 함수 목록

| # | 함수 | 원본 줄 | 사용처 | 설명 |
|:-:|------|---------|--------|------|
| 1 | `_env_flag(name)` | L86-88 | registration | 환경변수 bool 파싱 |
| 2 | `_load_cfg(sku)` | L91-102 | registration, inspection, activation, plate, metrics | SKU별 설정 로드 |
| 3 | `_resolve_cfg_path(sku)` | L105-110 | plate (calibrate) | 설정 파일 경로 해석 |
| 4 | `_atomic_write_json(path, data)` | L113-116 | plate (calibrate) | 원자적 JSON 쓰기 |
| 5 | `_compute_center_crop_mean_rgb(bgr, crop)` | L119-133 | plate (calibrate) | 중심 크롭 평균 RGB |
| 6 | `_load_snapshot_config(entry)` | L136-150 | inspection (inspect) | 스냅샷 설정 로드 |
| 7 | `_safe_float(value)` | L153-159 | registration (auto_tune) | 안전한 float 변환 |
| 8 | `_normalize_expected_ink_count(value)` | L162-171 | registration, inspection | 잉크 카운트 정규화 |
| 9 | `_parse_match_ids(raw, count)` | L174-191 | inspection (analyze_single) | match ID 파싱 |
| 10 | `_require_role(expected, user_role)` | L714-726 | 전체 라우트 | 역할 기반 접근 제어 |
| 11 | `_save_uploads(files, run_dir, max_mb)` | L729-744 | registration, inspection | 파일 업로드 저장 (async) |
| 12 | `_save_single_upload(file, dest, max_mb)` | L747-749 | inspection (test_run) | 단일 파일 저장 (async) |
| 13 | `_load_bgr(path)` | L752-756 | inspection (test_run) | BGR 이미지 로드 |
| 14 | `_validate_subprocess_arg(value, name, max_len)` | L902-928 | (unused - 보존) | subprocess 인자 검증 |
| 15 | `_run_script(script, args, timeout_sec)` | L931-990 | registration, inspection | subprocess 실행 |
| 16 | `_run_script_async(script, args)` | L993-1000 | registration, inspection | async subprocess 실행 |
| 17 | `_read_index()` | L1003-1007 | registration, activation, inspection, metrics | index.json 읽기 |
| 18 | `_read_index_at(root)` | L1010-1014 | registration | 지정 root에서 index 읽기 |
| 19 | `_find_entry(index_data, sku, ink)` | L1017-1021 | registration, activation, inspection, metrics | SKU/INK 엔트리 찾기 |
| 20 | `_safe_delete_path(rel_path)` | L1032-1047 | activation (rollback), metrics (delete) | 안전한 경로 삭제 |
| 21 | `_active_versions(active)` | L381-385 | registration, activation | 버전 문자열 추출 |
| 22 | `_load_std_images_from_versions(active)` | L619-633 | activation | STD 이미지 로드 |

**의존성**: v7_paths, security, engine_v7 core imports (config_loader, model_registry)

#### 2.1.4 Imports 블록

```python
import asyncio
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from src.config.v7_paths import (
    REPO_ROOT, V7_MODELS, V7_RESULTS, V7_ROOT, V7_TEST_RESULTS,
    add_repo_root_to_sys_path, ensure_v7_dirs,
)
from src.utils.security import sanitize_filename, validate_file_extension, validate_file_size
from src.engine_v7.core.config_loader import load_cfg_with_sku
from src.engine_v7.core.model_registry import compute_cfg_hash
```

---

### 2.2 `v7_registration.py` — 등록 + 모델 관리 (~660줄)

#### 2.2.1 라우트

| # | 메서드 | 경로 | 함수 | 원본 줄 |
|:-:|--------|------|------|---------|
| 1 | POST | `/register_validate` | `register_and_validate()` | L1278-1518 |
| 2 | GET | `/status` | `get_status()` | L1521-1527 |
| 3 | GET | `/entries` | `list_entries()` | L1530-1549 |
| 4 | POST | `/register_cleanup` | `register_cleanup()` | L1552-1556 |
| 5 | GET | `/candidates` | `get_candidates()` | L1559-1569 |

#### 2.2.2 도메인 헬퍼 (이 모듈 전용)

| # | 함수 | 원본 줄 | 설명 |
|:-:|------|---------|------|
| 23 | `_resolve_sku_for_ink(sku, ink)` | L194-199 | SKU 해석 (inspection에서도 사용 → helpers로 이동) |
| 24 | `_auto_tune_cfg_from_std(cfg, std_models, entry, active)` | L202-378 | 자동 튜닝 (177줄) |
| 25 | `_build_std_model(bgr, cfg)` | L864-881 | STD 모델 빌드 |
| 26 | `_run_gate_check(bgr, cfg)` | L884-899 | Gate 체크 |
| 27 | `_warning_diff(active_warnings, new_warnings)` | L388-394 | 경고 diff |
| 28 | `_approval_status(candidate_label, delta_sep, warn_diff)` | L397-404 | 승인 상태 결정 |
| 29 | `_resolve_pack_path(pack_path)` | L407-420 | 승인 팩 경로 해석 |
| 30 | `_load_approval_pack(pack_path)` | L423-428 | 승인 팩 로드 |
| 31 | `_validate_pack_for_activate(pack, sku, ink)` | L431-437 | 승인 팩 검증 |
| 32 | `_build_approval_pack(sku, ink, run_id, ...)` | L457-558 | 승인 팩 생성 (102줄) |

**NOTE**: `_resolve_sku_for_ink`은 inspection에서도 사용 → **helpers에 배치**.
**NOTE**: `_load_approval_pack`, `_validate_pack_for_activate`는 activation에서도 사용 → **이 모듈에 두고 activation에서 import**.

#### 2.2.3 Pydantic 모델

| 클래스 | 원본 줄 |
|--------|---------|
| `RegisterCleanupRequest` | L1274-1275 |

#### 2.2.4 Sub-router 설정

```python
from fastapi import APIRouter, File, Form, Header, UploadFile
router = APIRouter()  # prefix 없음 — v7.py에서 include할 때 부여
```

---

### 2.3 `v7_activation.py` — 활성화 + 거버넌스 (~360줄)

#### 2.3.1 라우트

| # | 메서드 | 경로 | 함수 | 원본 줄 |
|:-:|--------|------|------|---------|
| 6 | POST | `/activate` | `activate_model()` | L1657-1693 |
| 7 | POST | `/rollback` | `rollback_model()` | L1696-1781 |

#### 2.3.2 도메인 헬퍼

| # | 함수 | 원본 줄 | 설명 |
|:-:|------|---------|------|
| 33 | `_finalize_activation(sku, ink, versions, ...)` | L1572-1654 | 활성화 핵심 로직 (83줄) |
| 34 | `_finalize_pack(pack, pack_path, action, ...)` | L440-454 | 승인 팩 마무리 |
| 35 | `_write_active_snapshot(entry, active)` | L561-616 | 활성 스냅샷 기록 (56줄) |
| 36 | `_write_pattern_baseline(entry, active)` | L636-681 | 패턴 베이스라인 기록 (46줄) |
| 37 | `_write_ink_baseline(entry, active)` | L684-711 | 잉크 베이스라인 기록 (28줄) |

#### 2.3.3 Pydantic 모델

| 클래스 | 원본 줄 |
|--------|---------|
| `ActivateRequest` | L1245-1254 |
| `RollbackRequest` | L1257-1263 |

#### 2.3.4 Cross-module 의존성

- `v7_registration._load_approval_pack` — activation에서 import
- `v7_registration._validate_pack_for_activate` — activation에서 import
- `v7_helpers._load_std_images_from_versions` — snapshot/baseline 생성 시 사용
- `src.engine_v7.core.pipeline.analyzer._registration_summary` — snapshot 생성 시 사용

---

### 2.4 `v7_inspection.py` — 검사 + QC (~560줄)

#### 2.4.1 라우트

| # | 메서드 | 경로 | 함수 | 원본 줄 |
|:-:|--------|------|------|---------|
| 8 | POST | `/test_run` | `test_run()` | L1871-2016 |
| 9 | POST | `/inspect` | `inspect()` | L2019-2116 |
| 10 | POST | `/analyze_single` | `analyze_single()` | L2119-2280 |

#### 2.4.2 도메인 헬퍼

| # | 함수 | 원본 줄 | 설명 |
|:-:|------|---------|------|
| 38 | `_set_inspection_metadata(payload, entry)` | L1024-1029 | 검사 메타데이터 설정 |
| 39 | `_write_inspection_artifacts(run_dir, data)` | L1106-1161 | 검사 아티팩트 기록 (56줄) |
| 40 | `_generate_single_analysis_artifacts(bgr, analysis, run_dir, basename)` | L2283-2375 | 단일분석 아티팩트 (93줄) |
| 41 | `_load_active_snapshot(entry)` | L1164-1173 | 활성 스냅샷 로드 |
| 42 | `_load_latest_v2_review(sku, ink)` | L1176-1200 | 최신 v2 리뷰 로드 |
| 43 | `_load_latest_v3_summary(sku, ink)` | L1203-1222 | 최신 v3 요약 로드 |
| 44 | `_load_recent_decisions_for_trend(sku, ink, window)` | L1225-1242 | 최근 결정 트렌드 로드 |

**NOTE**: `_load_active_snapshot`, `_load_latest_v2_review`, `_load_latest_v3_summary`는 registration에서도 사용 (approval pack 빌드 시).
→ **이 모듈에 두고 registration에서 import** (주 사용처가 inspection이므로).

---

### 2.5 `v7_metrics.py` — 메트릭 + 진단 + 삭제 (~220줄)

#### 2.5.1 라우트

| # | 메서드 | 경로 | 함수 | 원본 줄 |
|:-:|--------|------|------|---------|
| 11 | GET | `/v2_metrics` | `get_v2_metrics()` | L1784-1801 |
| 12 | GET | `/trend_line` | `get_trend_line()` | L1804-1808 |
| 13 | POST | `/delete_entry` | `delete_entry()` | L1811-1868 |

#### 2.5.2 도메인 헬퍼

| # | 함수 | 원본 줄 | 설명 |
|:-:|------|---------|------|
| 45 | `_delete_approval_packs(sku, ink)` | L1050-1071 | 승인 팩 삭제 |
| 46 | `_delete_results_dirs(sku, ink)` | L1074-1103 | 결과 디렉토리 삭제 |

#### 2.5.3 Pydantic 모델

| 클래스 | 원본 줄 |
|--------|---------|
| `DeleteEntryRequest` | L1266-1271 |

---

### 2.6 `v7_plate.py` — 플레이트 + 교정 (~330줄)

#### 2.6.1 라우트

| # | 메서드 | 경로 | 함수 | 원본 줄 |
|:-:|--------|------|------|---------|
| 14 | POST | `/plate_gate` | `extract_plate_gate_api()` | L2418-2564 |
| 15 | POST | `/intrinsic_calibrate` | `intrinsic_calibrate_api()` | L2567-2629 |
| 16 | POST | `/intrinsic_simulate` | `intrinsic_simulate_api()` | L2632-2668 |

#### 2.6.2 도메인 헬퍼

| # | 함수 | 원본 줄 | 설명 |
|:-:|------|---------|------|
| 47 | `_generate_plate_pair_artifacts(white_bgr, ...)` | L759-861 | 플레이트 아티팩트 생성 (103줄) |
| 48 | `_polar_mask_to_base64(polar_mask)` | L2383-2407 | 폴라 마스크 Base64 변환 |

#### 2.6.3 Pydantic 모델

| 클래스 | 원본 줄 |
|--------|---------|
| `PlateGateRequest` | L2410-2413 |

**NOTE**: `_generate_plate_pair_artifacts`는 inspection (analyze_single)에서도 호출됨.
→ **plate 모듈에 두고 inspection에서 import** (주 도메인이 plate이므로).

---

### 2.7 `v7.py` (축소) — 서브 라우터 조립 진입점 (~50줄)

```python
"""
Lens Signature Engine v7 UI/API bridge (MVP)

Sub-routers assembled here. Individual route handlers live in:
  - v7_registration.py  (register_validate, status, entries, cleanup, candidates)
  - v7_activation.py    (activate, rollback)
  - v7_inspection.py    (test_run, inspect, analyze_single)
  - v7_metrics.py       (v2_metrics, trend_line, delete_entry)
  - v7_plate.py         (plate_gate, intrinsic_calibrate, intrinsic_simulate)
"""
from fastapi import APIRouter

from src.config.v7_paths import add_repo_root_to_sys_path, ensure_v7_dirs

ensure_v7_dirs()
add_repo_root_to_sys_path()

from .v7_registration import router as registration_router
from .v7_activation import router as activation_router
from .v7_inspection import router as inspection_router
from .v7_metrics import router as metrics_router
from .v7_plate import router as plate_router

router = APIRouter(prefix="/api/v7", tags=["V7 MVP"])

router.include_router(registration_router, tags=["V7 Registration"])
router.include_router(activation_router, tags=["V7 Activation"])
router.include_router(inspection_router, tags=["V7 Inspection"])
router.include_router(metrics_router, tags=["V7 Metrics"])
router.include_router(plate_router, tags=["V7 Plate"])
```

---

## 3. Cross-Module Dependency Map

```
v7_helpers.py  ←  (모든 도메인 모듈이 import)
     │
     ├── v7_registration.py
     │        │
     │        ├── v7_activation.py  (imports: _load_approval_pack, _validate_pack_for_activate)
     │        │
     │        └── v7_inspection.py  (imports: _load_active_snapshot, _load_latest_v2_review, etc. -- OR from inspection)
     │
     ├── v7_inspection.py
     │        │
     │        └── v7_plate.py  (imports: _generate_plate_pair_artifacts via inspection)
     │
     ├── v7_metrics.py  (독립)
     │
     └── v7_plate.py  (독립, inspection에서 artifact 함수만 import당함)
```

### 3.1 순환 import 방지 전략

**규칙**: 단방향 import만 허용. helpers → (없음). domain → helpers. domain → domain (단방향만).

| Import 방향 | 허용 |
|-------------|------|
| `v7_activation → v7_registration` | YES (승인 팩 관련) |
| `v7_inspection → v7_plate` | YES (plate_pair_artifacts) |
| `v7_registration → v7_inspection` | YES (snapshot/v2/v3 로드 함수) |
| `v7_inspection → v7_registration` | NO (순환 위험) |

**해결**: `_load_active_snapshot`, `_load_latest_v2_review`, `_load_latest_v3_summary`를 **helpers로 이동** (registration과 inspection 모두에서 사용하므로).

### 3.2 수정된 helpers 함수 목록 (순환 방지 후)

helpers에 추가 이동할 함수:

| # | 함수 | 원래 배치 | 이유 |
|:-:|------|----------|------|
| 23 | `_resolve_sku_for_ink` | registration | inspection에서도 사용 |
| 41 | `_load_active_snapshot` | inspection | registration에서도 사용 |
| 42 | `_load_latest_v2_review` | inspection | registration에서도 사용 |
| 43 | `_load_latest_v3_summary` | inspection | registration에서도 사용 |
| 44 | `_load_recent_decisions_for_trend` | inspection | metrics에서도 사용 |

---

## 4. Implementation Order

| Step | 작업 | 파일 | 위험도 |
|------|------|------|--------|
| 1 | `v7_helpers.py` 생성 (공통 함수 추출) | NEW | Medium |
| 2 | `v7_plate.py` 생성 (독립적, 가장 간단) | NEW | Low |
| 3 | `v7_metrics.py` 생성 (독립적) | NEW | Low |
| 4 | `v7_inspection.py` 생성 | NEW | Medium |
| 5 | `v7_registration.py` 생성 | NEW | High |
| 6 | `v7_activation.py` 생성 | NEW | Medium |
| 7 | `v7.py` 축소 (서브 라우터 조립) | MODIFY | High |
| 8 | Import 경로 검증 + syntax check | - | Low |

**원칙**: Step 1에서 helpers 모듈 완성 후, Step 2-6은 독립적 도메인부터. Step 7에서 v7.py를 최종 축소. 각 단계에서 syntax check.

---

## 5. Function Placement Summary

### 5.1 v7_helpers.py (27개 함수/클래스)

| # | 항목 | 원본 줄 |
|:-:|------|---------|
| 1 | `NumpyEncoder` | L65-83 |
| 2 | `_env_flag` | L86-88 |
| 3 | `_load_cfg` | L91-102 |
| 4 | `_resolve_cfg_path` | L105-110 |
| 5 | `_atomic_write_json` | L113-116 |
| 6 | `_compute_center_crop_mean_rgb` | L119-133 |
| 7 | `_load_snapshot_config` | L136-150 |
| 8 | `_safe_float` | L153-159 |
| 9 | `_normalize_expected_ink_count` | L162-171 |
| 10 | `_parse_match_ids` | L174-191 |
| 11 | `_resolve_sku_for_ink` | L194-199 |
| 12 | `_active_versions` | L381-385 |
| 13 | `_require_role` | L714-726 |
| 14 | `_save_uploads` | L729-744 |
| 15 | `_save_single_upload` | L747-749 |
| 16 | `_load_bgr` | L752-756 |
| 17 | `_validate_subprocess_arg` | L902-928 |
| 18 | `_run_script` | L931-990 |
| 19 | `_run_script_async` | L993-1000 |
| 20 | `_read_index` | L1003-1007 |
| 21 | `_read_index_at` | L1010-1014 |
| 22 | `_find_entry` | L1017-1021 |
| 23 | `_safe_delete_path` | L1032-1047 |
| 24 | `_load_std_images_from_versions` | L619-633 |
| 25 | `_load_active_snapshot` | L1164-1173 |
| 26 | `_load_latest_v2_review` | L1176-1200 |
| 27 | `_load_latest_v3_summary` | L1203-1222 |
| 28 | `_load_recent_decisions_for_trend` | L1225-1242 |

### 5.2 v7_registration.py (5 routes + 10 helpers + 1 model)

| # | 항목 | 원본 줄 |
|:-:|------|---------|
| R1 | `register_and_validate` | L1278-1518 |
| R2 | `get_status` | L1521-1527 |
| R3 | `list_entries` | L1530-1549 |
| R4 | `register_cleanup` | L1552-1556 |
| R5 | `get_candidates` | L1559-1569 |
| 24 | `_auto_tune_cfg_from_std` | L202-378 |
| 25 | `_build_std_model` | L864-881 |
| 26 | `_run_gate_check` | L884-899 |
| 27 | `_warning_diff` | L388-394 |
| 28 | `_approval_status` | L397-404 |
| 29 | `_resolve_pack_path` | L407-420 |
| 30 | `_load_approval_pack` | L423-428 |
| 31 | `_validate_pack_for_activate` | L431-437 |
| 32 | `_build_approval_pack` | L457-558 |
| 33 | `_write_inspection_artifacts` | L1106-1161 |
| M1 | `RegisterCleanupRequest` | L1274-1275 |

**NOTE**: `_write_inspection_artifacts`는 registration에서 사용 (register_and_validate에서 호출). inspection에서도 사용 → **registration에 두고 inspection에서 import**.

### 5.3 v7_activation.py (2 routes + 5 helpers + 2 models)

| # | 항목 | 원본 줄 |
|:-:|------|---------|
| R6 | `activate_model` | L1657-1693 |
| R7 | `rollback_model` | L1696-1781 |
| 34 | `_finalize_activation` | L1572-1654 |
| 35 | `_finalize_pack` | L440-454 |
| 36 | `_write_active_snapshot` | L561-616 |
| 37 | `_write_pattern_baseline` | L636-681 |
| 38 | `_write_ink_baseline` | L684-711 |
| M2 | `ActivateRequest` | L1245-1254 |
| M3 | `RollbackRequest` | L1257-1263 |

### 5.4 v7_inspection.py (3 routes + 2 helpers)

| # | 항목 | 원본 줄 |
|:-:|------|---------|
| R8 | `test_run` | L1871-2016 |
| R9 | `inspect` | L2019-2116 |
| R10 | `analyze_single` | L2119-2280 |
| 39 | `_set_inspection_metadata` | L1024-1029 |
| 40 | `_generate_single_analysis_artifacts` | L2283-2375 |

### 5.5 v7_metrics.py (3 routes + 2 helpers + 1 model)

| # | 항목 | 원본 줄 |
|:-:|------|---------|
| R11 | `get_v2_metrics` | L1784-1801 |
| R12 | `get_trend_line` | L1804-1808 |
| R13 | `delete_entry` | L1811-1868 |
| 41 | `_delete_approval_packs` | L1050-1071 |
| 42 | `_delete_results_dirs` | L1074-1103 |
| M4 | `DeleteEntryRequest` | L1266-1271 |

### 5.6 v7_plate.py (3 routes + 2 helpers + 1 model)

| # | 항목 | 원본 줄 |
|:-:|------|---------|
| R14 | `extract_plate_gate_api` | L2418-2564 |
| R15 | `intrinsic_calibrate_api` | L2567-2629 |
| R16 | `intrinsic_simulate_api` | L2632-2668 |
| 43 | `_generate_plate_pair_artifacts` | L759-861 |
| 44 | `_polar_mask_to_base64` | L2383-2407 |
| M5 | `PlateGateRequest` | L2410-2413 |

---

## 6. External Compatibility

### 6.1 현재 외부 참조

```python
# src/web/app.py
from src.web.routers import inspection, v7
app.include_router(v7.router)
```

### 6.2 호환성 전략

`v7.py`의 `router` 변수가 모든 서브 라우터를 include하므로:
- `app.py`에서 `v7.router` 참조 → **변경 없음**
- 모든 API 경로 `/api/v7/*` → **변경 없음** (prefix는 v7.py에서 설정)
- 서브 라우터는 prefix 없이 경로만 정의

### 6.3 검증 항목

```bash
# 외부 참조 검증
grep -r "from.*routers.*v7" src/ --include="*.py"
grep -r "v7\.router" src/ --include="*.py"

# API 경로 완전성 검증 (15개 라우트)
grep -rn "@router\." src/web/routers/v7_*.py --include="*.py" | wc -l
# 기대값: 15
```

---

## 7. Success Criteria

| # | Criterion | Measurement |
|:-:|-----------|-------------|
| 1 | v7.py 50줄 이하 | wc -l |
| 2 | 6개 서브 모듈 생성 | ls v7_*.py |
| 3 | 15개 API 경로 유지 | grep @router 카운트 |
| 4 | `app.py` 수정 불필요 | git diff app.py |
| 5 | python syntax check 통과 | ast.parse 7/7 OK |
| 6 | 순환 import 없음 | python -c "import" 테스트 |
| 7 | 공통 함수 중복 0줄 | 코드 리뷰 |

---

## 8. Changed Files Summary

| File | Action | Est. Lines |
|------|--------|-----------|
| `v7.py` | MODIFY (2670→~50) | -2620 |
| `v7_helpers.py` | CREATE | ~500 |
| `v7_registration.py` | CREATE | ~660 |
| `v7_activation.py` | CREATE | ~360 |
| `v7_inspection.py` | CREATE | ~560 |
| `v7_metrics.py` | CREATE | ~220 |
| `v7_plate.py` | CREATE | ~330 |
| **Net** | | **~0줄** (순수 분할) |

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-01-31 | Initial design | PDCA Auto |
