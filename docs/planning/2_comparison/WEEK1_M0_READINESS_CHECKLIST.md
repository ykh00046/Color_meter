# Week 1 M0 작업 준비 완료 체크리스트

**작성일**: 2025-12-17
**목적**: MVP Week 1 시작 전 준비 상태 확인
**상태**: ✅ **READY TO START**

---

## 📋 M0 목표 (Week 1)

**핵심 목표**: STD/TEST를 저장할 수 있는 구조 완성

**포함 작업**:
1. ✅ DB 스키마 구축
2. ✅ ORM 모델 정의
3. ✅ Pydantic 스키마 정의
4. 🟡 Alembic 마이그레이션 (가이드 완성)
5. ⭐ 판정 기준 협의 워크샵 (템플릿 완성)

---

## ✅ 완료 항목

### 1. Database Schema (100% 완료)

#### 1.1 SQLAlchemy ORM 모델 ✅
**위치**: `src/models/`
- ✅ `database.py` - Base, init_database, session management
- ✅ `std_models.py` - STDModel, STDSample, STDStatistics (3 tables)
- ✅ `test_models.py` - TestSample, ComparisonResult (2 tables)
- ✅ `user_models.py` - User, AuditLog (2 tables)
- ✅ `__init__.py` - Package exports

**총 7개 테이블**:
1. `std_models` - STD 메타데이터
2. `std_samples` - STD 이미지 (향후 P2에서 다중 샘플 지원)
3. `std_statistics` - Zone 통계 (향후 P2에서 통계 모델)
4. `test_samples` - TEST 샘플
5. `comparison_results` - 비교 결과
6. `users` - 사용자 (RBAC)
7. `audit_logs` - 감사 로그

**검증 상태**: ✅ 12/12 테스트 통과 (`tools/test_db_models.py`)

---

### 2. API Schemas (100% 완료)

#### 2.1 Pydantic 모델 ✅
**위치**: `src/schemas/`
- ✅ `std_schemas.py` - STD 등록/조회 API (8 schemas)
  - `STDProfileData` - 완전한 프로파일 데이터
  - `STDRegisterRequest` - 등록 요청
  - `STDRegisterResponse` - 등록 응답
  - `STDListResponse` - 목록 조회
  - `STDDetailResponse` - 상세 조회
  - `ZoneColorData`, `ZoneBoundaryData` - 세부 구조

- ✅ `comparison_schemas.py` - 비교 API (8 schemas)
  - `ComparisonRequest` - 비교 요청
  - `ComparisonResponse` - 비교 응답
  - `StructureSimilarity` - 구조 유사도
  - `ColorSimilarity`, `ZoneColorSimilarity` - 색상 유사도
  - `FailureReason` - 실패 원인
  - `ComparisonSummary` - 요약

- ✅ `judgment_schemas.py` - 판정 기준 (3 schemas)
  - `JudgmentCriteria` - 판정 임계값 설정
  - `ConfidenceScore` - 신뢰도 점수
  - `JudgmentResult` - 판정 결과

- ✅ `__init__.py` - Package exports

**총 19개 Pydantic 모델** 정의 완료

**기능**:
- ✅ FastAPI 자동 API 문서 생성 지원
- ✅ 입력 검증 (validator, Field constraints)
- ✅ 예제 데이터 (json_schema_extra)
- ✅ 타입 힌팅 (IDE 자동완성)

---

### 3. Judgment Criteria Workshop (100% 완료)

#### 3.1 워크샵 템플릿 ✅
**위치**: `docs/planning/JUDGMENT_CRITERIA_WORKSHOP.md`

**포함 내용**:
- ✅ 워크샵 진행 순서 (4 Phase)
- ✅ 결정해야 할 기준값:
  - 구조 유사도 (상관계수 >= 0.85, 경계 차이 <= ±3%)
  - 색상 유사도 (평균 ΔE <= 3.0, 95% ΔE <= 5.0)
  - 종합 점수 (PASS >= 80, WARNING >= 60)
  - 신뢰도 (자동 판정 >= 80%)
- ✅ 검증 프로세스 (OK 5개, NG 5개 샘플 테스트)
- ✅ 예외 케이스 (RETAKE, MANUAL_REVIEW)
- ✅ 가중치 조정 (구조 40% + 색상 60%)
- ✅ 최종 합의 JSON 템플릿
- ✅ 사후 관리 계획 (Week 7-10 튜닝)

**상태**: ⭐ **Week 1 시작 전 필수 실행 필요**

---

### 4. Alembic Migration Setup (80% 완료)

#### 4.1 가이드 스크립트 ✅
**위치**: `tools/init_alembic.py`

**포함 기능**:
- ✅ Alembic 설치 확인
- ✅ 초기화 가이드
- ✅ alembic.ini 설정 안내
- ✅ env.py 설정 안내
- ✅ 마이그레이션 생성/적용 가이드

**실행 순서**:
```bash
# 1. Alembic 설치
pip install alembic>=1.13.0

# 2. 가이드 실행
python tools/init_alembic.py

# 3. 수동 설정 (스크립트 안내에 따라)
# - alembic.ini 편집 (DB URL)
# - alembic/env.py 편집 (target_metadata)

# 4. 마이그레이션 생성
alembic revision --autogenerate -m "Initial schema"

# 5. 마이그레이션 적용
alembic upgrade head
```

**상태**: 🟡 **실행 필요** (Week 1 첫날)

---

### 5. Documentation Updates (100% 완료)

#### 5.1 로드맵 검토 문서 ✅
**위치**: `docs/planning/ROADMAP_REVIEW_AND_ARCHITECTURE.md`

**핵심 내용**:
- ✅ MVP 로드맵 승인 (Week 6 목표)
- ✅ 단일 분석 vs 비교 시스템 아키텍처 분리
- ✅ 3단계 로드맵 (M0-M2 → P1 → P2)
- ✅ 리스크 분석 및 완화 방안
- ✅ 작업량 추정 (3300줄, 6주)

#### 5.2 문서 색인 업데이트 ✅
**위치**: `docs/INDEX.md`
- ✅ "실행 계획" 섹션 추가
- ✅ ROADMAP_REVIEW_AND_ARCHITECTURE.md 최우선 표시

#### 5.3 Requirements 업데이트 ✅
**위치**: `requirements.txt`
- ✅ `alembic>=1.13.0` 추가

---

## 🟡 남은 작업 (Week 1 실행 필요)

### 1. Alembic 마이그레이션 초기화
**예상 시간**: 30분
**난이도**: 낮음

**실행**:
```bash
# 1. Alembic 설치 (requirements.txt에 추가됨)
pip install -r requirements.txt

# 2. 가이드 실행
python tools/init_alembic.py

# 3. 수동 편집 (스크립트가 안내)
#    - alembic.ini
#    - alembic/env.py

# 4. 마이그레이션 생성 및 적용
alembic revision --autogenerate -m "Initial schema"
alembic upgrade head

# 5. 확인
sqlite3 color_meter.db ".tables"
```

---

### 2. 판정 기준 협의 워크샵 실행 ⭐
**예상 시간**: 2-3시간
**난이도**: 중간 (협의 필요)

**준비물**:
- [ ] OK 샘플 5개 (이미지 파일)
- [ ] NG 샘플 5개 (이미지 파일)
- [ ] 현재 시스템으로 10개 샘플 분석 (상관계수, ΔE 측정)

**참석자**:
- [ ] 품질 관리자
- [ ] 생산 엔지니어
- [ ] 검사원 대표
- [ ] 시스템 개발자

**결과물**:
- [ ] 합의된 판정 기준 JSON 파일
- [ ] 서명된 승인 문서

**실행 후**:
```bash
# 합의 결과를 설정 파일로 저장
mkdir -p config
cat > config/judgment_criteria.json << 'EOF'
{
  "min_profile_correlation": 0.85,
  "max_boundary_difference_percent": 3.0,
  "max_mean_delta_e": 3.0,
  "max_p95_delta_e": 5.0,
  "pass_score_threshold": 80.0,
  "warning_score_threshold": 60.0,
  "min_confidence_for_auto_judgment": 80.0
}
EOF

# src/schemas/judgment_schemas.py의 기본값 업데이트
```

---

## 📊 진행 상황 요약

| 카테고리 | 완료 | 진행 중 | 남음 | 진행률 |
|---------|------|---------|------|--------|
| **DB 스키마** | 7 tables | 0 | 0 | 100% |
| **ORM 모델** | 6 models | 0 | 0 | 100% |
| **API 스키마** | 19 schemas | 0 | 0 | 100% |
| **마이그레이션** | 가이드 | 0 | 실행 | 80% |
| **판정 기준** | 템플릿 | 0 | 워크샵 | 90% |
| **문서화** | 완료 | 0 | 0 | 100% |
| **전체** | - | - | - | **95%** |

---

## 🎯 Week 1 시작 체크리스트

### Day 1 (월요일)
- [ ] Alembic 설치 및 초기화 (30분)
- [ ] DB 마이그레이션 실행 (10분)
- [ ] DB 테이블 생성 확인 (10분)
- [ ] 판정 기준 워크샵 준비 (샘플 10개 수집) (1시간)

### Day 2 (화요일)
- [ ] 판정 기준 협의 워크샵 실행 (2-3시간) ⭐
- [ ] 합의 결과 JSON 파일 생성 (30분)
- [ ] 기본값 업데이트 (30분)

### Day 3-5 (수-금요일)
- [ ] (옵션) 검증 스크립트 작성
- [ ] M1 준비 (STDService 설계)

**M0 완료 기준**: ✅ DB 테이블 생성 + 판정 기준 합의 완료

---

## 📦 산출물 목록

### 코드 파일 (신규 생성)
```
src/
├── models/                     # DB 모델 (완료)
│   ├── __init__.py
│   ├── database.py
│   ├── std_models.py
│   ├── test_models.py
│   └── user_models.py
│
└── schemas/                    # API 스키마 (완료)
    ├── __init__.py
    ├── std_schemas.py
    ├── comparison_schemas.py
    └── judgment_schemas.py
```

### 도구 스크립트 (신규 생성)
```
tools/
├── test_db_models.py           # DB 모델 테스트 (완료)
└── init_alembic.py             # Alembic 초기화 가이드 (완료)
```

### 문서 (신규/업데이트)
```
docs/
├── INDEX.md                    # 업데이트 (완료)
└── planning/
    ├── ROADMAP_REVIEW_AND_ARCHITECTURE.md  # 신규 (완료)
    ├── JUDGMENT_CRITERIA_WORKSHOP.md       # 신규 (완료)
    └── WEEK1_M0_READINESS_CHECKLIST.md     # 신규 (본 문서)
```

---

## 🔄 다음 단계 (M1, Week 2-3)

M0 완료 후:
1. STDService 구현 (`src/services/std_service.py`)
2. STD 등록 API (`src/web/routers/std.py`)
3. STD 목록/상세 UI (`src/web/templates/std_*.html`)
4. InspectionPipeline 재사용 (기존 코드)

**목표**: Week 3 종료 시 "STD 등록 및 조회 가능"

---

## ✅ 승인 및 시작

### 준비 상태
- ✅ DB 스키마: **READY**
- ✅ API 스키마: **READY**
- 🟡 Alembic: **PENDING** (30분 작업)
- ⭐ 판정 기준: **PENDING** (2-3시간 워크샵)

### 시작 가능 여부
**✅ YES** - Week 1 시작 가능

### 필수 선행 작업 (Week 1 Day 1-2)
1. Alembic 초기화 (Day 1)
2. 판정 기준 워크샵 (Day 2) ⭐

---

**작성자**: Claude Sonnet 4.5
**상태**: ✅ **READY TO START WEEK 1**
**최종 업데이트**: 2025-12-17
