# M2: Comparison & Judgment Implementation Plan

> **작성일**: 2025-12-18
> **상태**: 🟡 진행 중
> **우선순위**: 🔴 P0 (필수)

---

## 목차

1. [개요](#1-개요)
2. [선행 작업 확인](#2-선행-작업-확인)
3. [M2 범위 정의](#3-m2-범위-정의)
4. [아키텍처 설계](#4-아키텍처-설계)
5. [구현 계획](#5-구현-계획)
6. [검증 계획](#6-검증-계획)

---

## 1. 개요

### 1.1 목표

양산 샘플(Test Sample)을 STD와 비교하여 합격/불합격을 판정하는 시스템 구축

### 1.2 설계 원칙 (Review 반영)

**🔧 핵심 수정사항**:
1. **Zone 매칭**: `zip()` → `name` 기반 매칭 (순서/개수 불일치 대응)
2. **점수 계산**: boundary/area score 제외, MVP는 color_score만
3. **Confidence 정의**: InspectionPipeline 신뢰도로 명확화
4. **판정 기준**: FAIL을 보수적으로 조정 (`< 55` OR `min_zone < 45`)

### 1.3 핵심 기능

```
[M2 Flow]
1. Test Sample 등록 → InspectionPipeline 실행 → DB 저장
2. STD 자동 매칭 (SKU 기반)
3. 비교 분석 실행
   - Zone 비교 (색상 + 경계 + 면적)
   - Ink 비교 (잉크 색상 매칭)
   - Radial Profile 비교 (구조 유사도)
4. 종합 점수 산출 (0-100)
5. 판정 (PASS/FAIL/RETAKE/MANUAL_REVIEW)
6. 실패 이유 상위 3개 추출
7. 결과 저장 및 반환
```

### 1.3 MVP vs P2 범위

**MVP (M2)**:
- ✅ Test Sample 등록 API
- ✅ 기본 비교 분석 (Zone 색상 + 경계)
- ✅ 판정 로직 (PASS/FAIL)
- ✅ 비교 결과 조회 API

**P2 (향후)**:
- ⏭️ Ink 비교 (GMM 기반)
- ⏭️ Radial Profile 상관계수
- ⏭️ Worst-case metrics
- ⏭️ Defect classification
- ⏭️ 조치 권장 시스템

---

## 2. 선행 작업 확인

### 2.1 완료된 작업 ✅

| 작업 | 상태 | 비고 |
|------|------|------|
| **M0: Database** | ✅ 완료 | 7개 테이블 생성 (Alembic) |
| **M1: STD Registration** | ✅ 완료 | STDService, STD API |
| **DB Models** | ✅ 완료 | TestSample, ComparisonResult |
| **InspectionPipeline** | ✅ 완료 | 기존 분석 파이프라인 |

### 2.2 사용 가능한 리소스

```python
# Models
from src.models.std_models import STDModel, STDSample
from src.models.test_models import TestSample, ComparisonResult, JudgmentStatus

# Services
from src.services.std_service import STDService

# Pipeline
from src.pipeline import InspectionPipeline
from src.core.color_evaluator import InspectionResult, ZoneResult
```

---

## 3. M2 범위 정의

### 3.1 서비스 레이어

#### 3.1.1 TestService
```python
class TestService:
    """Test Sample 등록 및 관리"""

    def register_test_sample(
        self,
        sku_code: str,
        image_path: str,
        batch_number: Optional[str] = None,
        sample_id: Optional[str] = None,
        operator: Optional[str] = None,
        notes: Optional[str] = None
    ) -> TestSample:
        """
        Test Sample 등록

        Workflow:
        1. Validate image path
        2. Load SKU config
        3. Run InspectionPipeline
        4. Create TestSample with analysis_result
        5. Audit log
        6. Return TestSample
        """
        pass

    def get_test_sample(self, test_id: int) -> TestSample:
        """Test Sample 조회"""
        pass

    def list_test_samples(
        self,
        sku_code: Optional[str] = None,
        batch_number: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[TestSample]:
        """Test Sample 목록 조회"""
        pass
```

#### 3.1.2 ComparisonService
```python
class ComparisonService:
    """STD vs Test 비교 분석"""

    def compare(
        self,
        test_sample_id: int,
        std_model_id: Optional[int] = None  # None이면 자동 매칭
    ) -> ComparisonResult:
        """
        비교 분석 실행

        Workflow:
        1. Load TestSample
        2. Find active STD (if not provided)
        3. Compare zones
           - Color similarity (ΔE)
           - Boundary position
           - Area difference
        4. Calculate scores
           - Zone score (0-100)
           - Total score (0-100)
        5. Determine judgment (PASS/FAIL)
        6. Extract top 3 failure reasons
        7. Save ComparisonResult
        8. Return result
        """
        pass

    def get_comparison_result(self, comparison_id: int) -> ComparisonResult:
        """비교 결과 조회"""
        pass

    def list_comparison_results(
        self,
        sku_code: Optional[str] = None,
        judgment: Optional[JudgmentStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ComparisonResult]:
        """비교 결과 목록 조회"""
        pass

    # Private methods
    def _compare_zones(
        self,
        test_zones: List[ZoneResult],
        std_zones: List[Dict]
    ) -> Dict[str, Any]:
        """Zone별 비교"""
        pass

    def _calculate_zone_score(self, zone_details: Dict) -> float:
        """Zone 점수 계산 (0-100)"""
        pass

    def _determine_judgment(
        self,
        total_score: float,
        zone_details: Dict
    ) -> JudgmentStatus:
        """판정 결정"""
        pass

    def _extract_failure_reasons(
        self,
        zone_details: Dict
    ) -> List[Dict[str, Any]]:
        """실패 이유 추출 (상위 3개)"""
        pass
```

### 3.2 API 엔드포인트

#### 3.2.1 Test Sample API (`/api/test`)

```python
# POST /api/test/register
# - Request: { sku_code, image_path, batch_number?, sample_id?, operator?, notes? }
# - Response: { id, sku_code, sample_id, lens_detected, created_at, ... }

# GET /api/test/list
# - Query: sku_code?, batch_number?, limit?, offset?
# - Response: { total, items: [TestSample] }

# GET /api/test/{test_id}
# - Response: TestSample detail
```

#### 3.2.2 Comparison API (`/api/compare`)

```python
# POST /api/compare
# - Request: { test_sample_id, std_model_id? }
# - Response: ComparisonResult

# GET /api/compare/{comparison_id}
# - Response: ComparisonResult detail

# GET /api/compare/list
# - Query: sku_code?, judgment?, limit?, offset?
# - Response: { total, items: [ComparisonResult] }

# GET /api/compare/test/{test_sample_id}
# - Response: 해당 Test Sample의 모든 비교 결과
```

### 3.3 Pydantic Schemas

```python
# Test Sample Schemas
class TestRegisterRequest(BaseModel):
    sku_code: str
    image_path: str
    batch_number: Optional[str] = None
    sample_id: Optional[str] = None
    operator: Optional[str] = None
    notes: Optional[str] = None

class TestListItem(BaseModel):
    id: int
    sku_code: str
    batch_number: Optional[str]
    sample_id: Optional[str]
    lens_detected: bool
    created_at: datetime
    operator: Optional[str]

class TestDetailResponse(BaseModel):
    id: int
    sku_code: str
    batch_number: Optional[str]
    sample_id: Optional[str]
    image_path: str
    lens_detected: bool
    lens_detection_score: Optional[float]
    created_at: datetime
    operator: Optional[str]
    notes: Optional[str]
    analysis_result: Dict[str, Any]

# Comparison Schemas
class CompareRequest(BaseModel):
    test_sample_id: int
    std_model_id: Optional[int] = None  # Auto-match if None

class CompareResponse(BaseModel):
    id: int
    test_sample_id: int
    std_model_id: int
    scores: Dict[str, float]  # {total, zone, ink?, confidence?}
    judgment: str  # PASS/FAIL/RETAKE/MANUAL_REVIEW
    top_failure_reasons: Optional[List[Dict[str, Any]]]
    created_at: datetime
    processing_time_ms: int

class ComparisonListItem(BaseModel):
    id: int
    test_sample_id: int
    std_model_id: int
    total_score: float
    judgment: str
    created_at: datetime
```

---

## 4. 아키텍처 설계

### 4.1 비교 알고리즘 (MVP)

```python
def compare_zones(test_zones, std_zones):
    """
    Zone별 비교 분석

    IMPORTANT: Zone은 name 기반으로 매칭 (순서/개수 불일치 대응)

    Returns:
        zone_details: {
            "A": {
                "color_score": 85.3,  # 0-100 (ΔE 기반)
                "total_score": 85.3,  # MVP: color_score와 동일
                "delta_e": 2.5,
                "measured_lab": [72.2, 137.3, 122.8],
                "target_lab": [70.5, 135.1, 120.2]
            },
            "B": {...},
            "C": {...}
        }
    """
    zone_details = {}

    # Zone을 name 기반으로 매핑 (zip 대신)
    std_map = {z["zone_name"]: z for z in std_zones}

    for tz in test_zones:
        zone_name = tz.zone_name
        sz = std_map.get(zone_name)

        if not sz:
            # STD에 해당 Zone이 없음 → FAIL 사유
            zone_details[zone_name] = {
                "error": "STD_ZONE_MISSING",
                "color_score": 0,
                "total_score": 0,
                "delta_e": 999.9,
                "measured_lab": tz.measured_lab,
                "target_lab": None
            }
            continue

        # 1. Color similarity (ΔE → 0-100 score)
        delta_e = tz.delta_e
        color_score = max(0, 100 - delta_e * 10)  # ΔE=10 → 0점

        # 2. Total zone score (MVP: color_score만 사용)
        # boundary/area는 P1 이후로 연기
        total = color_score

        zone_details[zone_name] = {
            "color_score": color_score,
            "total_score": total,
            "delta_e": delta_e,
            "measured_lab": tz.measured_lab,
            "target_lab": sz["target_lab"]
        }

    return zone_details
```

### 4.2 점수 계산

```python
def calculate_scores(zone_details, inspection_result):
    """
    종합 점수 계산

    Returns:
        {
            "zone_score": 82.5,  # Zone별 평균 (color_score)
            "total_score": 82.5,  # MVP: zone_score와 동일
            "confidence_score": 88.0  # InspectionPipeline 신뢰도 (0-100)
        }
    """
    # Zone score (평균)
    zone_scores = [z["total_score"] for z in zone_details.values()]
    zone_score = sum(zone_scores) / len(zone_scores) if zone_scores else 0

    # Total score (MVP: zone_score와 동일)
    # P1 이후: ink_score 추가 시 가중 평균으로 변경
    total_score = zone_score

    # Confidence score (InspectionPipeline 신뢰도)
    # - 렌즈 검출 성공 여부
    # - Zone 분할 성공 여부
    # - InspectionResult.confidence (0.0-1.0)
    confidence_score = inspection_result.confidence * 100 if inspection_result.confidence else 0

    return {
        "zone_score": zone_score,
        "total_score": total_score,
        "confidence_score": confidence_score
    }
```

### 4.3 판정 로직

```python
def determine_judgment(total_score, zone_details, lens_detected):
    """
    판정 결정 (보수적 기준)

    Logic:
    - MANUAL_REVIEW: 렌즈 미검출 or STD Zone 누락
    - PASS: total_score >= 80 AND all zones >= 70
    - FAIL: total_score < 55 OR any zone < 45  (보수적 조정)
    - RETAKE: 나머지 (55 <= total_score < 80)
    """
    # Check for manual review cases
    if not lens_detected:
        return JudgmentStatus.MANUAL_REVIEW

    # Check for zone errors
    for zone_name, details in zone_details.items():
        if details.get("error"):
            return JudgmentStatus.MANUAL_REVIEW

    # Check zone thresholds
    min_zone_score = min(z["total_score"] for z in zone_details.values())

    # PASS: 높은 기준 (안정적)
    if total_score >= 80 and min_zone_score >= 70:
        return JudgmentStatus.PASS

    # FAIL: 보수적 기준 (초기 현장 안정화)
    elif total_score < 55 or min_zone_score < 45:
        return JudgmentStatus.FAIL

    # RETAKE: 중간 구간 (넓게 설정)
    else:
        return JudgmentStatus.RETAKE
```

### 4.4 실패 이유 추출

```python
def extract_failure_reasons(zone_details, judgment):
    """
    실패 이유 추출 (상위 3개)

    Returns:
        [
            {
                "rank": 1,
                "category": "ZONE_COLOR",
                "message": "Zone B: ΔE=8.5 exceeds threshold",
                "severity": 85  # 0-100 (100=심각)
            },
            ...
        ]
    """
    if judgment == JudgmentStatus.PASS:
        return []

    issues = []

    for zone_name, details in zone_details.items():
        # Color issues
        if details["color_score"] < 70:
            severity = 100 - details["color_score"]
            issues.append({
                "category": "ZONE_COLOR",
                "zone": zone_name,
                "message": f"Zone {zone_name}: ΔE={details['delta_e']:.1f}",
                "severity": severity,
                "score": details["color_score"]
            })

        # Boundary issues
        if details["boundary_score"] < 80:
            severity = 100 - details["boundary_score"]
            issues.append({
                "category": "ZONE_BOUNDARY",
                "zone": zone_name,
                "message": f"Zone {zone_name}: Boundary shift detected",
                "severity": severity,
                "score": details["boundary_score"]
            })

    # Sort by severity (desc) and take top 3
    issues.sort(key=lambda x: x["severity"], reverse=True)

    # Add rank
    for i, issue in enumerate(issues[:3]):
        issue["rank"] = i + 1

    return issues[:3]
```

---

## 5. 구현 계획

### 5.1 Phase 1: TestService (1일)

**파일**: `src/services/test_service.py`

```python
"""
Test Service Layer

Handles test sample registration and management.
"""

class TestServiceError(Exception):
    pass

class TestService:
    def __init__(self, db_session: Session, sku_manager: SkuConfigManager):
        self.db = db_session
        self.sku_manager = sku_manager

    def register_test_sample(...):
        # Implementation
        pass

    def get_test_sample(...):
        # Implementation
        pass

    def list_test_samples(...):
        # Implementation
        pass
```

**구현 체크리스트**:
- [ ] TestService 클래스 구현
- [ ] register_test_sample() - InspectionPipeline 실행 + DB 저장
- [ ] get_test_sample() - 단일 조회
- [ ] list_test_samples() - 목록 조회 (필터링 지원)
- [ ] AuditLog 기록

### 5.2 Phase 2: ComparisonService (1-2일)

**파일**: `src/services/comparison_service.py`

```python
"""
Comparison Service Layer

Handles comparison between test samples and STD models.
"""

class ComparisonServiceError(Exception):
    pass

class ComparisonService:
    def __init__(self, db_session: Session):
        self.db = db_session

    def compare(...):
        # Implementation
        pass

    def _compare_zones(...):
        # Zone별 비교 로직
        pass

    def _calculate_scores(...):
        # 점수 계산
        pass

    def _determine_judgment(...):
        # 판정 로직
        pass

    def _extract_failure_reasons(...):
        # 실패 이유 추출
        pass
```

**구현 체크리스트**:
- [ ] ComparisonService 클래스 구현
- [ ] compare() - 전체 비교 워크플로
- [ ] _compare_zones() - Zone별 비교 (MVP: 색상만)
- [ ] _calculate_scores() - 점수 계산
- [ ] _determine_judgment() - 판정 로직
- [ ] _extract_failure_reasons() - Top 3 추출
- [ ] get_comparison_result() - 조회
- [ ] list_comparison_results() - 목록 조회

### 5.3 Phase 3: Pydantic Schemas (0.5일)

**파일**: `src/schemas/test_schemas.py`, `src/schemas/comparison_schemas.py`

**구현 체크리스트**:
- [ ] test_schemas.py - TestRegisterRequest, TestListItem, TestDetailResponse
- [ ] comparison_schemas.py - CompareRequest, CompareResponse, ComparisonListItem

### 5.4 Phase 4: API Routers (1일)

**파일**:
- `src/web/routers/test.py` - Test Sample API
- `src/web/routers/comparison.py` - Comparison API

**구현 체크리스트**:
- [ ] test.py - POST /api/test/register, GET /api/test/list, GET /api/test/{id}
- [ ] comparison.py - POST /api/compare, GET /api/compare/{id}, GET /api/compare/list
- [ ] app.py에 router 등록
- [ ] Dependency injection 설정

### 5.5 Phase 5: 통합 테스트 (0.5일)

**구현 체크리스트**:
- [ ] Test Sample 등록 테스트
- [ ] 비교 분석 테스트 (기존 STD 사용)
- [ ] 판정 로직 검증
- [ ] API 엔드포인트 전체 테스트

---

## 6. 검증 계획

### 6.1 단위 테스트

```python
# tests/test_comparison_service.py

def test_compare_zones():
    """Zone 비교 로직 테스트"""
    # Mock test_zones, std_zones
    # Compare
    # Assert zone_details structure and scores
    pass

def test_calculate_scores():
    """점수 계산 테스트"""
    # Mock zone_details
    # Calculate
    # Assert score ranges (0-100)
    pass

def test_determine_judgment():
    """판정 로직 테스트"""
    # Test PASS case (score >= 80)
    # Test FAIL case (score < 60)
    # Test RETAKE case (60 <= score < 80)
    pass

def test_extract_failure_reasons():
    """실패 이유 추출 테스트"""
    # Mock zone_details with failures
    # Extract
    # Assert top 3, ranked, sorted by severity
    pass
```

### 6.2 통합 테스트 시나리오

```bash
# 1. STD 등록 (이미 완료)
curl -X POST http://localhost:8000/api/std/register \
  -H "Content-Type: application/json" \
  -d '{"sku_code": "SKU001", "image_path": "data/raw_images/SKU001_OK_001.jpg", "version": "v1.0"}'
# → STD ID=1

# 2. Test Sample 등록
curl -X POST http://localhost:8000/api/test/register \
  -H "Content-Type: application/json" \
  -d '{"sku_code": "SKU001", "image_path": "data/raw_images/SKU001_OK_002.jpg", "batch_number": "B001"}'
# → Test ID=1

# 3. 비교 분석 실행
curl -X POST http://localhost:8000/api/compare \
  -H "Content-Type: application/json" \
  -d '{"test_sample_id": 1}'
# → Comparison ID=1, judgment=PASS, total_score=85.3

# 4. 비교 결과 조회
curl http://localhost:8000/api/compare/1
# → Full comparison details

# 5. 불량 샘플 테스트
curl -X POST http://localhost:8000/api/test/register \
  -d '{"sku_code": "SKU001", "image_path": "data/raw_images/SKU001_NG_001.jpg"}'
# → Test ID=2

curl -X POST http://localhost:8000/api/compare \
  -d '{"test_sample_id": 2}'
# → Comparison ID=2, judgment=FAIL, top_failure_reasons=[...]
```

### 6.3 검증 기준

| 항목 | 기준 |
|------|------|
| **Test 등록** | InspectionPipeline 정상 실행, analysis_result 저장 |
| **STD 자동 매칭** | sku_code 기반으로 active STD 매칭 성공 |
| **Zone 비교** | 3개 Zone (A, B, C) 모두 비교 성공 |
| **점수 범위** | 모든 점수 0-100 사이 |
| **판정 일관성** | score >= 80 → PASS, < 60 → FAIL |
| **실패 이유** | FAIL 시 최대 3개 이유 추출, rank/severity 포함 |
| **처리 시간** | 비교 분석 < 3초 |

---

## 7. 타임라인

| Day | 작업 | 산출물 |
|-----|------|--------|
| **Day 1** | TestService 구현 | test_service.py |
| **Day 2** | ComparisonService 구현 (1/2) | comparison_service.py (핵심 로직) |
| **Day 3** | ComparisonService 구현 (2/2) + Schemas | comparison_service.py (완성), schemas |
| **Day 4** | API Routers | test.py, comparison.py |
| **Day 5** | 통합 테스트 + 문서화 | M2_COMPLETE.md |

**예상 완료**: 5일 (2025-12-23)

---

## 8. 참고 자료

- `STD_BASED_QC_SYSTEM_PLAN.md` - 전체 로드맵
- `M1_STD_REGISTRATION_COMPLETE.md` - M1 완료 보고서
- `src/models/test_models.py` - TestSample, ComparisonResult 모델
- `src/pipeline.py` - InspectionPipeline 기존 구현

---

**작성자**: Claude Sonnet 4.5
**프로젝트**: Contact Lens Color Inspection System
**문서 위치**: `docs/planning/2_comparison/M2_COMPARISON_IMPLEMENTATION_PLAN.md`
