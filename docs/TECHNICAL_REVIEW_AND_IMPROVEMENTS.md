# Color Meter 기술 리뷰 및 코드 개선 로드맵

**작성일**: 2026-01-20
**버전**: V7 Engine
**심각도 분류**: 🔴 Critical | 🟡 Medium | 🟢 Low

---

## 1. 코드 품질 이슈

### 1.1 과도하게 큰 함수 (높은 순환 복잡도)

#### 🔴 Critical: `analyzer.py` - 1,406줄, 복잡한 함수들

**위치**: `src/engine_v7/core/pipeline/analyzer.py`

| 함수명 | 라인 수 | 문제점 |
|--------|---------|--------|
| `evaluate()` | 213줄 (477-689) | Gate, Signature, Anomaly, Diagnostics 혼합 |
| `evaluate_multi()` | 236줄 (692-927) | evaluate()와 80% 코드 중복 |
| `evaluate_per_color()` | 283줄 (1123-1406) | 가장 복잡한 함수 |
| `_compute_diagnostics()` | 111줄 (210-321) | 7단계 중첩 if-else |

**문제 코드 예시** (라인 244-291):
```python
# 7단계 중첩된 진단 계산
if delta_cie[2] >= db_th:
    reason_codes_extra.append(...)
elif delta_cie[2] <= -db_th:
    ...
if delta_cie[1] >= da_th:
    ...
elif delta_cie[1] <= -da_th:
    ...
# ... 8회 이상 반복
```

**개선 방안**:
```python
# BEFORE: 단일 거대 함수
def evaluate(self, test_bgr, config, ...):
    # 213줄의 복잡한 로직
    ...

# AFTER: 책임 분리
def evaluate(self, test_bgr, config, ...):
    gate_result = self._evaluate_gate_phase(test_bgr, config)
    if not gate_result.passed:
        return self._build_gate_failure_decision(gate_result)

    signature_result = self._evaluate_signature_phase(test_bgr, config)
    anomaly_result = self._evaluate_anomaly_phase(test_bgr, config)

    return self._build_final_decision(gate_result, signature_result, anomaly_result)

def _evaluate_gate_phase(self, test_bgr, config) -> GateResult:
    """게이트 평가만 담당 (50줄)"""
    ...

def _evaluate_signature_phase(self, test_bgr, config) -> SignatureResult:
    """시그니처 평가만 담당 (60줄)"""
    ...

def _evaluate_anomaly_phase(self, test_bgr, config) -> AnomalyResult:
    """이상 탐지만 담당 (40줄)"""
    ...
```

---

### 1.2 코드 중복 패턴

#### 🔴 Critical: Gate 처리 로직 3회 중복

**위치**: `analyzer.py` 라인 511-523, 729-744, 1168-1183

**중복 코드**:
```python
# 3곳에서 동일하게 반복됨
if not gate.passed and not diag_on_fail:
    codes, messages = _reason_meta(gate.reasons)
    return Decision(
        label="RETAKE",
        reasons=gate.reasons,
        reason_codes=codes,
        reason_messages=messages,
        ...
    )
```

**개선 방안**:
```python
# 공통 함수 추출
def _handle_gate_failure(self, gate: GateResult, diag_on_fail: bool) -> Optional[Decision]:
    """Gate 실패 시 공통 처리 로직"""
    if not gate.passed and not diag_on_fail:
        codes, messages = _reason_meta(gate.reasons)
        return Decision(
            label="RETAKE",
            reasons=gate.reasons,
            reason_codes=codes,
            reason_messages=messages,
            gate=gate,
        )
    return None

# 사용
def evaluate(self, ...):
    gate_failure = self._handle_gate_failure(gate, diag_on_fail)
    if gate_failure:
        return gate_failure
    # 계속 진행...
```

---

### 1.3 일관성 없는 예외 처리

#### 🔴 Critical: 베어 except 절

**위치**: `src/engine_v7/api.py:69`

```python
# 현재 코드 - 위험!
except:  # 모든 예외 무시 (KeyboardInterrupt 포함)
    return {}  # 조용한 실패 - 디버깅 불가
```

**전체 예외 처리 패턴 불일치**:

| 파일 | 라인 | 문제 |
|------|------|------|
| `api.py` | 69 | 베어 except |
| `model_registry.py` | 84 | 로깅 없는 except Exception |
| `analyzer.py` | 368 | 올바른 패턴 (로깅 + 계속) |

**개선 방안**:
```python
# 표준 예외 처리 패턴
import logging

logger = logging.getLogger(__name__)

def load_config(sku_id: str) -> dict:
    try:
        config_path = CONFIG_DIR / f"{sku_id}.json"
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config not found for SKU: {sku_id}, using defaults")
        return DEFAULT_CONFIG.copy()
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config for SKU {sku_id}: {e}")
        raise ConfigurationError(f"Malformed config for {sku_id}") from e
    except Exception as e:
        logger.exception(f"Unexpected error loading config for {sku_id}")
        raise
```

---

### 1.4 매직 넘버 및 하드코딩된 값

#### 🟡 Medium: 47개 매직 넘버 발견

**주요 위치**:

| 파일 | 라인 | 값 | 용도 |
|------|------|-----|------|
| `analyzer.py` | 266 | `5.0` | coverage_l_delta |
| `analyzer.py` | 267 | `0.1` | edge_sharpness_delta_threshold |
| `analyzer.py` | 268 | `2.0` | coverage_delta_pp_threshold |
| `analyzer.py` | 441 | `20` | window size (미설명) |
| `color_masks.py` | 336 | `0.03` | min_area_ratio_warn |
| `threshold_policy.py` | 14-35 | 다수 | 임계값 중복 정의 |

**개선 방안**:
```python
# BEFORE
cov_l_delta_cie = float(diag_cfg.get("coverage_l_delta", 5.0)) * (100.0 / 255.0)
edge_th = float(diag_cfg.get("edge_sharpness_delta_threshold", 0.1))

# AFTER: 상수 클래스 + 문서화
class DiagnosticThresholds:
    """진단 임계값 상수

    Note: 이 값들은 실험적으로 결정됨 (2025-12 캘리브레이션 기준)
    """
    # 커버리지 L* 델타 (CIE Lab 단위)
    # 5.0 이상이면 명도 차이가 눈에 띄는 수준
    COVERAGE_L_DELTA: float = 5.0

    # 에지 선명도 델타 임계값
    # 0.1 이상이면 에지가 흐릿함
    EDGE_SHARPNESS_DELTA: float = 0.1

    # 커버리지 pp 임계값
    COVERAGE_DELTA_PP: float = 2.0

# 사용
cov_l_delta_cie = float(diag_cfg.get(
    "coverage_l_delta",
    DiagnosticThresholds.COVERAGE_L_DELTA
)) * (100.0 / 255.0)
```

---

## 2. 성능 병목 지점

### 2.1 동기 블로킹 연산

#### 🔴 Critical: async 핸들러에서 동기 블로킹

**위치**: `src/web/app.py:310+`

```python
# 현재 코드 - 이벤트 루프 블로킹!
@app.post("/v7/inspect")
async def inspect_image(file: UploadFile, ...):
    file_content, input_path, original_name = await validate_and_save_file(...)

    # 🔴 이 부분이 5-30초 동기 블로킹!
    pipeline = InspectionPipeline(...)
    result = pipeline.run(test_bgr)  # CPU-bound 작업
```

**영향**:
- 하나의 느린 이미지가 모든 요청 블로킹
- 실제 동시성 없음
- 타임아웃 위험

**개선 방안**:
```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

# 프로세스 풀 생성 (애플리케이션 시작 시)
executor = ProcessPoolExecutor(max_workers=4)

@app.post("/v7/inspect")
async def inspect_image(file: UploadFile, ...):
    file_content, input_path, original_name = await validate_and_save_file(...)

    # CPU-bound 작업을 별도 프로세스에서 실행
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        _run_inspection_sync,  # 동기 함수
        test_bgr, config
    )
    return result

def _run_inspection_sync(test_bgr, config):
    """별도 프로세스에서 실행되는 동기 함수"""
    pipeline = InspectionPipeline(config)
    return pipeline.run(test_bgr)
```

---

### 2.2 불필요한 데이터 복사

#### 🟡 Medium: LAB 변환 중복 실행

**위치**: `analyzer.py:545-586`

```python
# 1차 변환 (라인 545-548)
test_mean, _, _ = build_radial_signature(polar, ...)
# build_radial_signature 내부에서 LAB 변환 수행

# 2차 변환 (라인 586) - 중복!
test_lab_map = to_cie_lab(polar)

# 3차 변환 (라인 1084-1085) - 또 중복!
```

**개선 방안**:
```python
class AnalysisContext:
    """분석 중 공유되는 중간 결과 캐싱"""
    def __init__(self, image_bgr: np.ndarray):
        self._image_bgr = image_bgr
        self._image_lab: Optional[np.ndarray] = None
        self._polar: Optional[np.ndarray] = None
        self._polar_lab: Optional[np.ndarray] = None

    @property
    def image_lab(self) -> np.ndarray:
        if self._image_lab is None:
            self._image_lab = cv2.cvtColor(self._image_bgr, cv2.COLOR_BGR2LAB)
        return self._image_lab

    @property
    def polar_lab(self) -> np.ndarray:
        if self._polar_lab is None:
            self._polar_lab = to_cie_lab(self.polar)
        return self._polar_lab
```

---

### 2.3 메모리 집약적 히트맵 생성

#### 🟡 Medium: 전체 히트맵이 JSON 응답에 포함

**위치**: `analyzer.py:619-623`

```python
if anom and cfg["anomaly"].get("enable_heatmap", True) and label != "OK":
    hm = anomaly_heatmap(polar, ...)  # 360x512x3 = 552KB
    debug["anomaly_heatmap"] = hm  # JSON으로 직렬화!
```

**영향**:
- 검사 실패 시 응답 페이로드 수 MB
- 배치 작업 시 메모리 폭발

**개선 방안**:
```python
@dataclass
class InspectionOptions:
    include_debug_heatmap: bool = False
    heatmap_downsample_factor: int = 4
    max_heatmap_size: int = 128 * 128

def evaluate(self, test_bgr, config, options: InspectionOptions = None):
    options = options or InspectionOptions()

    if anom and options.include_debug_heatmap:
        hm = anomaly_heatmap(
            polar,
            ds_T=options.heatmap_downsample_factor,
            ds_R=options.heatmap_downsample_factor
        )
        # 크기 제한
        if hm.size <= options.max_heatmap_size:
            debug["anomaly_heatmap"] = hm
        else:
            debug["anomaly_heatmap_path"] = self._save_heatmap_to_file(hm)
```

---

### 2.4 비효율적인 루프 패턴

#### 🟡 Medium: O(n²) 거리 계산

**위치**: `color_masks.py:136-141`

```python
# 현재 코드 - 184K × 8 = 1.5M float 연산
dists = np.sum(
    (feat_flat[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :]) ** 2,
    axis=2
)
labels_flat = np.argmin(dists, axis=1)
```

**개선 방안**:
```python
from scipy.spatial.distance import cdist

# 최적화된 거리 계산 (BLAS 활용)
dists = cdist(feat_flat, cluster_centers, metric='sqeuclidean')
labels_flat = np.argmin(dists, axis=1)

# 또는 sklearn 활용
from sklearn.metrics import pairwise_distances_argmin
labels_flat = pairwise_distances_argmin(feat_flat, cluster_centers)
```

---

## 3. 아키텍처 이슈

### 3.1 모듈 간 강한 결합

#### 🟡 Medium: 순환 의존성

**현재 구조**:
```
analyzer.py
    ├── decision_builder.py
    │       └── decision_engine.py
    ├── anomaly_score.py
    │       └── heatmap.py
    ├── pattern_baseline.py
    └── ... 10+ imports
```

**개선 방안 - 인터페이스 레이어 추출**:
```python
# interfaces.py - 추상 인터페이스 정의
from abc import ABC, abstractmethod
from typing import Protocol

class IGateEvaluator(Protocol):
    def evaluate(self, image: np.ndarray, config: dict) -> GateResult: ...

class ISignatureAnalyzer(Protocol):
    def analyze(self, polar: np.ndarray, std_model: StdModel) -> SignatureResult: ...

class IAnomalyDetector(Protocol):
    def detect(self, polar: np.ndarray, config: dict) -> AnomalyResult: ...

# analyzer.py - 인터페이스에만 의존
class Analyzer:
    def __init__(
        self,
        gate_evaluator: IGateEvaluator,
        signature_analyzer: ISignatureAnalyzer,
        anomaly_detector: IAnomalyDetector,
    ):
        self._gate = gate_evaluator
        self._signature = signature_analyzer
        self._anomaly = anomaly_detector
```

---

### 3.2 설정 관리 혼란

#### 🟡 Medium: 3-4단계 설정 오버라이드

**현재 구조**:
```
1. default.json (기본값)
2. sku_db/{sku}.json (SKU별 오버라이드)
3. threshold_policy.py (하드코딩된 중복 값)
4. Runtime cfg_override (API 파라미터)
```

**개선 방안 - 단일 설정 소스**:
```python
# config_schema.py
from pydantic import BaseModel, Field
from typing import Optional

class GateConfig(BaseModel):
    blur_threshold: float = Field(100.0, ge=0, description="블러 임계값")
    illumination_threshold: float = Field(0.3, ge=0, le=1)
    center_offset_threshold: float = Field(10.0, ge=0)

class SignatureConfig(BaseModel):
    min_correlation: float = Field(0.85, ge=0, le=1)
    max_delta_e: float = Field(8.0, ge=0)

class EngineConfig(BaseModel):
    gate: GateConfig = Field(default_factory=GateConfig)
    signature: SignatureConfig = Field(default_factory=SignatureConfig)

    @classmethod
    def load(cls, sku_id: str, overrides: dict = None) -> "EngineConfig":
        """설정 로드 with 검증"""
        base = cls.parse_file(DEFAULT_CONFIG_PATH)

        sku_path = SKU_CONFIG_DIR / f"{sku_id}.json"
        if sku_path.exists():
            sku_config = cls.parse_file(sku_path)
            base = base.copy(update=sku_config.dict(exclude_unset=True))

        if overrides:
            base = base.copy(update=overrides)

        return base
```

---

### 3.3 의존성 주입 부재

#### 🟡 Medium: 전역 상태 및 하드코딩된 경로

**위치**: `app.py:71-92`

```python
# 현재 코드 - 전역 인스턴스
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR.parent / "results" / "web"
analysis_service = AnalysisService()  # 전역!
```

**개선 방안 - FastAPI 의존성 주입**:
```python
# dependencies.py
from functools import lru_cache

class Settings(BaseSettings):
    base_dir: Path = Path(__file__).resolve().parent.parent
    results_dir: Path = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.results_dir is None:
            self.results_dir = self.base_dir.parent / "results" / "web"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

def get_analysis_service(settings: Settings = Depends(get_settings)) -> AnalysisService:
    return AnalysisService(settings.results_dir)

# app.py
@app.post("/v7/inspect")
async def inspect_image(
    file: UploadFile,
    service: AnalysisService = Depends(get_analysis_service)
):
    return await service.inspect(file)
```

---

## 4. 보안 이슈

### 4.1 입력 검증 불완전

#### 🔴 Critical: 파일 업로드 검증 부족

**위치**: `app.py:273-299`

```python
# 현재 코드 - 확장자만 검사!
if not validate_file_extension(file.filename, [".jpg", ".jpeg", ".png", ".bmp"]):
    raise HTTPException(...)

# 검증되지 않는 항목:
# - JPEG 헤더 (조작된 익스플로잇 가능)
# - 이미지 크기 (100k x 100k로 DoS 가능)
# - 실제 파일 내용 vs 확장자 (스푸핑)
```

**개선 방안**:
```python
from PIL import Image
import io

MAX_IMAGE_DIMENSION = 8192  # 8K 최대
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

async def validate_image_upload(file: UploadFile) -> bytes:
    """이미지 업로드 종합 검증"""
    # 1. 파일 크기 검증
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large: {len(content)} bytes")

    # 2. 확장자 검증
    ext = Path(file.filename).suffix.lower()
    if ext not in {'.jpg', '.jpeg', '.png', '.bmp'}:
        raise HTTPException(400, f"Invalid file type: {ext}")

    # 3. 실제 이미지 형식 검증 (헤더 확인)
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()  # 손상된 이미지 검출
    except Exception as e:
        raise HTTPException(400, f"Invalid image file: {e}")

    # 4. 이미지 크기 검증 (DoS 방지)
    img = Image.open(io.BytesIO(content))
    if img.width > MAX_IMAGE_DIMENSION or img.height > MAX_IMAGE_DIMENSION:
        raise HTTPException(400, f"Image too large: {img.width}x{img.height}")

    # 5. 확장자와 실제 형식 일치 검증
    format_map = {'.jpg': 'JPEG', '.jpeg': 'JPEG', '.png': 'PNG', '.bmp': 'BMP'}
    if img.format != format_map.get(ext):
        raise HTTPException(400, f"Extension mismatch: {ext} vs {img.format}")

    return content
```

---

### 4.2 Rate Limiting 부재

#### 🟡 Medium: API 무제한 호출 가능

**개선 방안**:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/v7/inspect")
@limiter.limit("10/minute")  # 분당 10회 제한
async def inspect_image(request: Request, file: UploadFile, ...):
    ...

@app.post("/v7/register")
@limiter.limit("5/minute")  # 등록은 더 엄격하게
async def register_std(request: Request, ...):
    ...
```

---

### 4.3 경로 순회 보호 개선

**위치**: `app.py:137-151`

```python
# 현재 코드 - run_id 검증이 경로 사용 후
def _safe_result_path(run_id: str, filename: Optional[str] = None) -> Path:
    run_dir = (RESULTS_DIR / run_id).resolve()  # 이미 사용됨
    try:
        run_dir.relative_to(RESULTS_DIR.resolve())  # 그 다음 검증
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run_id")
```

**개선 방안**:
```python
import re

def _validate_run_id(run_id: str) -> str:
    """run_id를 경로에 사용하기 전에 검증"""
    if not re.match(r'^[a-zA-Z0-9_-]+$', run_id):
        raise HTTPException(400, "Invalid run_id format")
    if '..' in run_id or '/' in run_id or '\\' in run_id:
        raise HTTPException(400, "Path traversal detected")
    return run_id

def _safe_result_path(run_id: str, filename: Optional[str] = None) -> Path:
    run_id = _validate_run_id(run_id)  # 먼저 검증
    run_dir = (RESULTS_DIR / run_id).resolve()
    # 이중 검증
    if not run_dir.is_relative_to(RESULTS_DIR.resolve()):
        raise HTTPException(400, "Invalid path")
    return run_dir
```

---

## 5. 테스트 갭

### 5.1 테스트 커버리지 현황

#### 🔴 Critical: 11% 파일 커버리지 (13/116)

**테스트되지 않은 핵심 모듈**:

| 모듈 | 라인 수 | 테스트 상태 |
|------|---------|-------------|
| `analyzer.py` | 1,406 | ❌ 없음 |
| `color_masks.py` | 950 | ⚠️ E2E만 |
| `decision_builder.py` | 500+ | ❌ 없음 |
| `api.py` | 200+ | ❌ 없음 |
| `app.py` | 600+ | ⚠️ 최소 |

### 5.2 누락된 엣지 케이스

```python
# 테스트되지 않은 시나리오
- 빈 이미지 (전체 검정 또는 흰색)
- 단색 렌즈 (세그멘테이션 불가)
- 극단적 조명 (과다/부족 노출)
- 유효하지 않은 geometry 검출
- 손상된 SKU 설정
- 동시 요청 처리
- 대규모 배치 처리
```

### 5.3 테스트 추가 계획

```python
# tests/unit/test_analyzer.py (신규)
import pytest
from src.engine_v7.core.pipeline.analyzer import Analyzer

class TestAnalyzerEvaluate:
    """evaluate() 함수 단위 테스트"""

    def test_evaluate_gate_failure_returns_retake(self, mock_image, mock_config):
        """게이트 실패 시 RETAKE 반환"""
        analyzer = Analyzer()
        mock_config["gate"]["blur_threshold"] = 0  # 무조건 실패

        result = analyzer.evaluate(mock_image, mock_config)

        assert result.label == "RETAKE"
        assert "blur" in result.reasons[0].lower()

    def test_evaluate_signature_mismatch_returns_ng(self, mock_image, mock_config, mock_std):
        """시그니처 불일치 시 NG 반환"""
        analyzer = Analyzer()
        mock_std.correlation = 0.5  # 낮은 상관관계

        result = analyzer.evaluate(mock_image, mock_config, std_model=mock_std)

        assert result.label == "NG"
        assert "signature" in str(result.reasons).lower()

    def test_evaluate_empty_image_handles_gracefully(self, mock_config):
        """빈 이미지 처리"""
        analyzer = Analyzer()
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = analyzer.evaluate(empty_image, mock_config)

        assert result.label in ["RETAKE", "NG"]
        assert len(result.reasons) > 0

# tests/integration/test_inspection_pipeline.py (신규)
class TestInspectionPipeline:
    """검사 파이프라인 통합 테스트"""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, test_client, sample_images):
        """동시 요청 처리"""
        import asyncio

        tasks = [
            test_client.post("/v7/inspect", files={"file": img})
            for img in sample_images[:10]
        ]

        results = await asyncio.gather(*tasks)

        assert all(r.status_code == 200 for r in results)

    @pytest.mark.asyncio
    async def test_large_batch_memory_usage(self, test_client, sample_images):
        """대규모 배치 메모리 사용량"""
        import tracemalloc

        tracemalloc.start()

        for img in sample_images[:100]:
            await test_client.post("/v7/inspect", files={"file": img})

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < 2 * 1024 * 1024 * 1024  # 2GB 미만
```

---

## 6. 개선 우선순위 로드맵

### Phase 1: Critical (1주차)

| 작업 | 파일 | 예상 시간 |
|------|------|----------|
| 베어 except 수정 | `api.py:69` | 1시간 |
| 이미지 검증 강화 | `app.py:273-299` | 4시간 |
| async 태스크 오프로딩 | `app.py:310+` | 8시간 |

### Phase 2: High (2-3주차)

| 작업 | 파일 | 예상 시간 |
|------|------|----------|
| evaluate() 함수 분리 | `analyzer.py` | 16시간 |
| 중복 코드 제거 | `analyzer.py` | 8시간 |
| 매직 넘버 설정화 | 다수 | 8시간 |
| analyzer.py 테스트 추가 | `tests/` | 16시간 |

### Phase 3: Medium (4-5주차)

| 작업 | 파일 | 예상 시간 |
|------|------|----------|
| analyzer.py 모듈 분리 | `analyzer.py` | 24시간 |
| Rate limiting 구현 | `app.py` | 4시간 |
| 80% 테스트 커버리지 | `tests/` | 40시간 |
| 설정 스키마 검증 | `config_schema.py` | 8시간 |

### Phase 4: Nice-to-have (6주차+)

| 작업 | 파일 | 예상 시간 |
|------|------|----------|
| 의존성 주입 구현 | 전체 | 24시간 |
| 종합 에러 로깅 | 전체 | 16시간 |
| 성능 최적화 (LAB 캐싱) | `analyzer.py` | 8시간 |
| 히트맵 최적화 | `analyzer.py` | 4시간 |

---

## 7. 이슈 요약

### 심각도별 분류

| 심각도 | 개수 | 주요 이슈 |
|--------|------|----------|
| 🔴 Critical | 8 | 베어 except, 블로킹 I/O, 입력 검증, 테스트 부족 |
| 🟡 Medium | 10 | 매직 넘버, 메모리, 결합도, Rate limiting |
| 🟢 Low | 6 | 코멘트 언어 혼합, 설정 중복 |

### 영역별 분류

| 영역 | 이슈 수 |
|------|---------|
| 코드 품질 | 8 |
| 성능 | 5 |
| 아키텍처 | 5 |
| 보안 | 4 |
| 테스트 | 3 |

---

## 8. 즉시 적용 가능한 Quick Wins

### 8.1 베어 except 수정 (5분)

```python
# api.py:69 수정
# BEFORE
except:
    return {}

# AFTER
except Exception as e:
    logger.warning(f"Config load failed: {e}")
    return {}
```

### 8.2 이미지 크기 제한 추가 (10분)

```python
# app.py에 추가
MAX_IMAGE_DIMENSION = 8192

async def validate_and_save_file(...):
    # 기존 코드 후에 추가
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
        raise HTTPException(400, f"Image too large: {w}x{h}")
```

### 8.3 Rate Limiting 추가 (15분)

```bash
pip install slowapi
```

```python
# app.py 상단에 추가
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/v7/inspect")
@limiter.limit("30/minute")
async def inspect_image(...):
    ...
```

---

*기술 리뷰 작성: Claude Code*
*작성일: 2026-01-20*
*분석된 파일: 116개*
*발견된 이슈: 24개 패턴, 50+ 위치*
