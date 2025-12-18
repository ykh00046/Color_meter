# Changelog

All notable changes to the Contact Lens Color Inspection System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Major] - 2025-12-17 - STD-Based QC System: Technical Enhancements & Database Models

### Added
- **Technical Enhancement Specification** (`docs/planning/TECHNICAL_ENHANCEMENTS_ADVANCED.md`):
  - 7 advanced technical enhancements for STD-based QC system (15,000+ words)
  - STD Statistical Model (multiple samples → mean ± σ)
  - Elastic Alignment (anchor zone + DTW)
  - Worst-Case Color Metrics (percentile ΔE, hotspot detection)
  - Ink-Aware Comparison (dual scoring: zone + ink)
  - Explainability Layer (Top 3 failure reasons)
  - Performance & Stability (caching, fail-safe)
  - Phenomenological Classification (defect taxonomy)

- **Algorithm Benchmark Script** (`tools/benchmark_alignment.py`):
  - Performance comparison: Cross-Correlation vs Circular Shift vs DTW
  - 100 synthetic profiles, 500 points each
  - Results: Cross-Correlation 0.09ms avg (12,000x faster than 1s target)
  - Auto-generated charts and JSON reports

- **Database Models** (`src/models/`):
  - SQLAlchemy ORM models for STD-based QC system
  - 7 tables: std_models, std_samples, std_statistics, test_samples, comparison_results, users, audit_logs
  - Full RBAC support (Admin, Engineer, Inspector, Viewer)
  - Comprehensive audit logging for compliance
  - JSON minimized, searchable fields separated
  - 15+ indexes for query performance

- **Database Configuration** (`src/models/database.py`):
  - Shared SQLAlchemy Base
  - Session management (get_session, get_db)
  - init_database(), create_tables(), drop_tables()

- **Database Validation** (`tools/test_db_models.py`):
  - 12 test cases validating all models
  - Relationship testing (cascade, lazy loading)
  - Query testing (filters, joins)
  - to_dict() serialization testing

### Changed
- **Web Dashboard** (`src/web/templates/index.html`):
  - Swapped Zone-Based and Image-Based sections
  - Zone-Based now prominent (top, green border, "권장" badge)
  - Image-Based collapsed below (click to expand)

- **Documentation Index** (`docs/INDEX.md`):
  - Added TECHNICAL_ENHANCEMENTS_ADVANCED.md to Key Planning Documents
  - Updated with high-priority star icon (⭐)

- **Requirements** (`requirements.txt`):
  - Added dtaidistance>=2.3.0 as optional dependency (for DTW benchmarking)

### Performance
- ✅ Cross-Correlation: 0.09 ms avg, 0.12 ms p99, 98.9% correlation
- ✅ Circular Shift: 3.25 ms avg, 4.44 ms p99, 99.1% correlation
- ✅ Both algorithms far exceed targets (avg < 1s, p99 < 3s)

### Documentation
- Created `docs/planning/TECHNICAL_ENHANCEMENTS_ADVANCED.md` (15,000 words)
- Created `docs/daily_reports/2025-12-17_COMPLETION_REPORT.md` (comprehensive summary)
- Updated `docs/INDEX.md` with new planning document
- Benchmark results: `results/alignment_benchmark.json`, `results/alignment_benchmark.png`

### Technical Debt
- TODO: Install Alembic for database migrations
- TODO: Implement STD statistical model aggregation service
- TODO: Create batch STD upload API
- TODO: Implement elastic alignment algorithm

---

## [Patch] - 2025-12-12 - UI Overhaul & Web Integration

### Added
- New Web UI dashboard based on Bootstrap 5.3 for a professional analysis experience.
- Interactive Image Viewer with Panzoom for zoom/pan functionality.
- Real-time Grid Overlay (Rings, Sectors) updated by UI sliders.
- Dynamic Result Display with Summary Table and Chart.js integration (Radial Profile, Delta E).
- Modular JavaScript for UI components (viewer.js, controls.js, charts.js, main.js).
- Dedicated CSS for modern styling (main.css).
- Analysis Service Layer (`src/services/analysis_service.py`) for centralized profile analysis.

### Fixed
- `/inspect` API response updated to include `zone_results` for UI rendering.
- `charts.js` data mapping corrected to match `ProfileAnalysisResult` structure.

### Changed
- `src/web/app.py` refactored to use `AnalysisService`.
- `docs/guides/WEB_UI_GUIDE.md` completely rewritten to reflect the new UI.

---

## [Patch] - 2025-12-11 - 시스템 개선 및 품질 향상

### Fixed
- DEPLOYMENT.md 한글 인코딩 깨짐 수정 (Notes 섹션)
- Windows 콘솔 UTF-8 로그 출력 문제 해결 (ΔE 기호 정상 표시)
- SKU001.json에 `expected_zones: 1` 추가로 over-segmentation 방지

### Changed
- 에러 핸들링 대폭 개선:
  - 이미지 로드 실패 시 3회 retry 로직 추가
  - 파일 없음 에러 명확한 메시지 ("Image file not found")
  - 렌즈 검출 실패 시 이미지 정보 포함 (크기, 경로)
  - Zone 분할 실패 시 3-zone fallback 자동 복구 시도
  - 모든 에러에 "Suggestion" 제공으로 문제 해결 용이성 향상
- 로깅 설정 개선: UTF-8 인코딩 명시, Windows 호환성 강화

### Added
- DEPLOYMENT.md에 Troubleshooting 섹션 추가

### Tested
- SKU001 OK/NG 샘플 E2E 검증 완료
- 시각화 기능 정상 동작 확인
- 배치 처리 (112 images) 정상 동작 확인
- 에러 케이스 (파일 없음) 핸들링 검증 완료

---

## [Day 5] - 2025-12-11 - 성능 최적화 및 프로덕션 준비

### Added
- 성능 프로파일링 도구 (`tools/profiler.py`, `tools/detailed_profiler.py`)
- 배치 처리 병렬화 지원 (`pipeline.process_batch(parallel=True)`)
- Zone Segmentation에 `expected_zones` 파라미터 추가
- 성능 회귀 테스트 5개 (`tests/test_performance.py`)
- Docker 배포 환경 (Dockerfile, docker-compose.yml)
- 배포 자동화 스크립트 3개 (`scripts/*.sh`)
- 문서 3개 신규 작성:
  - `docs/PERFORMANCE_ANALYSIS.md` - 성능 분석 리포트
  - `docs/USER_GUIDE.md` - 사용자 가이드
  - `docs/DEPLOYMENT.md` - 배포 가이드

### Changed
- `README.md` 전면 개선 (프로젝트 소개, Quick Start, CLI 레퍼런스)
- Zone Segmentation: 변곡점 검출 실패 시 uniform split fallback
- VIS_TEST SKU: `expected_zones: 3` 추가

### Fixed
- VIS_TEST 이미지에서 1개 zone만 검출되던 문제 → 3개 zone 검출 성공

### Performance
- 단일 이미지 처리: 88-142ms (안정적)
- 배치 처리: 병렬화로 대용량 처리 효율 향상
- 메모리: 배치 크기와 무관하게 일정 수준 유지

---

## [Day 4] - 2025-12-11 - Visualization System

### Added
- `InspectionVisualizer` 클래스 (3가지 시각화 타입)
  - Zone Overlay: 이미지 위 zone 경계 표시
  - Comparison Chart: 측정 vs 기준값 비교
  - Dashboard: 배치 처리 4-panel 요약
- CLI 시각화 옵션 (`--visualize`, `--viz-output`)
- 통합 테스트 18개 (`tests/test_visualizer.py`)
- Jupyter notebook (`notebooks/03_visualization_demo.ipynb`)
- 설계 문서 (`docs/VISUALIZER_DESIGN.md`)

### Changed
- `InspectionResult`에 시각화 필드 추가 (lens_detection, zones, image)
- `Pipeline.process()`: 시각화 데이터 자동 주입

---

## [Day 3] - 2025-12-10 - SKU Management System

### Added
- `SkuConfigManager` 클래스 (SKU CRUD 관리)
- `sku` CLI 명령어 (list, show, create, update, delete, generate-baseline)
- 베이스라인 자동 생성 기능 (OK 샘플 기반)
- SKU 관리 통합 테스트 15개
- Multi-SKU 분석 Jupyter notebook

### Changed
- SKU 설정을 JSON 파일 기반으로 관리 (`config/sku_db/`)
- 임계값 자동 계산 (mean + 2σ, mean + 3σ, fixed)

---

## [Day 2] - 2025-12-10 - E2E Pipeline & Prototype

### Added
- `InspectionPipeline` 클래스 (E2E 파이프라인)
- 배치 처리 지원 (`process_batch()`)
- 중간 결과 저장 옵션 (`save_intermediates`)
- Jupyter prototype notebook (`notebooks/01_prototype.ipynb`)
- CLI 진입점 (`src/main.py`)

---

## [Day 1] - 2025-12-09 - Core Modules

### Added
- 5개 핵심 모듈 구현:
  - `ImageLoader`: 이미지 로드 및 전처리
  - `LensDetector`: Hough Circle 기반 렌즈 검출
  - `RadialProfiler`: 극좌표 기반 radial profile 추출
  - `ZoneSegmenter`: 그래디언트 기반 zone 분할
  - `ColorEvaluator`: CIEDE2000 색차 계산 및 판정
- 유틸리티 모듈 (color_delta, file_io, image_utils)
- 단위 테스트 50+ 개

---

## Project Statistics

**Total Code:** ~8,000 lines
**Total Tests:** 133 tests (all passing)
**Test Coverage:** Core modules 100%
**Documentation:** 10+ documents
**Performance:** 88-142ms per image
**Ready for:** Production deployment

---

**Generated:** 2025-12-11
