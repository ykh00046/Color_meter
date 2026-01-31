# Color Meter 전체 구조 및 코드 분석 보고서

**작성일**: 2026-01-20
**버전**: V7 Engine

---

## 1. 프로젝트 개요

Color Meter는 컬러 콘택트렌즈의 품질 검사를 위한 머신비전 기반 분석 시스템입니다.
렌즈 이미지를 분석하여 색상 균일성, 패턴 결함, 잉크 분포 등을 자동으로 검사하고 OK/NG 판정을 수행합니다.

### 핵심 기능
- 이미지 품질 게이팅 (블러, 조명, 중심 오프셋 검증)
- 방사형 시그니처 분석 (720개 각도 샘플의 극좌표 변환)
- STD(Standard) 모델 등록 및 비교
- 다중 잉크 색상 분리 및 분석
- 이상 탐지 (블롭 검출, 각도 균일성, 히트맵 생성)
- 판정 로직 (Gate → Signature → Anomaly 의사결정 트리)

---

## 2. 디렉토리 구조

```
Color_meter/
├── src/                           # 메인 소스코드
│   ├── engine_v7/                 # V7 분석 엔진 (핵심)
│   │   ├── core/                  # 핵심 분석 모듈
│   │   ├── configs/               # 엔진 설정 파일
│   │   ├── tests/                 # 엔진 유닛 테스트
│   │   └── api.py                 # 엔진 파사드 API
│   ├── web/                       # FastAPI 웹 애플리케이션
│   │   ├── app.py                 # 메인 FastAPI 앱
│   │   ├── routers/               # API 라우트 핸들러
│   │   ├── templates/             # Jinja2 HTML 템플릿
│   │   ├── static/                # 클라이언트 에셋
│   │   └── schemas.py             # Pydantic 스키마
│   ├── models/                    # 데이터베이스 모델
│   ├── services/                  # 비즈니스 로직 서비스
│   ├── schemas/                   # Pydantic 스키마
│   ├── utils/                     # 유틸리티 모듈
│   └── config/                    # 설정 관리
├── tests/                         # 테스트 스위트
├── alembic/                       # DB 마이그레이션
├── config/                        # 사용자 설정
│   └── sku_db/                    # SKU별 설정
├── data/                          # 데이터 파일
├── scripts/                       # 유틸리티 스크립트
├── docs/                          # 문서
└── requirements.txt               # Python 의존성
```

---

## 3. V7 엔진 코어 구조 (`src/engine_v7/core/`)

### 3.1 모듈 개요

| 모듈 | 용도 | 주요 파일 |
|------|------|----------|
| **measure/** | 색상/잉크 추출, 세그멘테이션, 메트릭 계산 | `ink_segmentation.py`, `ink_metrics.py` |
| **signature/** | 방사형 시그니처 분석, STD 모델 피팅 | `radial_signature.py`, `std_model.py` |
| **decision/** | 판정 로직, 불확실성 정량화 | `decision_engine.py`, `uncertainty.py` |
| **pipeline/** | 분석 파이프라인 오케스트레이션 | `analyzer.py`, `single_analyzer.py` |
| **anomaly/** | 패턴 결함 탐지, 히트맵 분석 | `blob_detector.py`, `heatmap.py` |
| **calibration/** | 색상 정확도 검증, 바이어스 분석 | `bias_analyzer.py` |
| **gate/** | 이미지 품질 게이팅 | `gate_engine.py` |
| **geometry/** | 렌즈 검출 및 기하학 분석 | `lens_geometry.py` |
| **plate/** | Plate 분석 (White/Black 쌍) | `plate_engine.py` |
| **simulation/** | 색상 시뮬레이션 | `color_simulator.py` |

### 3.2 세그멘테이션 & 측정 (`measure/`)

```
measure/
├── segmentation/
│   ├── color_masks.py              # LAB 색공간 마스킹
│   ├── ink_segmentation.py         # 잉크 클러스터 추출
│   ├── preprocess.py               # 이미지 전처리
│   └── primary_color_extractor.py  # 주요 색상 분석
├── metrics/
│   ├── ink_metrics.py              # 잉크별 메트릭
│   ├── angular_metrics.py          # 각도 균일성
│   ├── uniformity.py               # 색상 균일성 분석
│   └── threshold_policy.py         # 임계값 정책 관리
├── matching/
│   ├── ink_match.py                # 잉크 매칭 알고리즘
│   └── assignment_map.py           # 잉크-존 할당
└── diagnostics/
    ├── v2_diagnostics.py           # 진단 메트릭
    └── v2_flags.py                 # 진단 플래그
```

### 3.3 시그니처 분석 (`signature/`)

```
signature/
├── radial_signature.py           # 극좌표 변환
├── profile_analysis.py           # 방사형 프로파일 분석
├── fit.py                        # STD 모델 피팅
├── std_model.py                  # 표준 모델 구조
├── model_io.py                   # 모델 직렬화
├── signature_compare.py          # 시그니처 비교
└── segment_k_suggest.py          # 세그먼트 수 제안
```

### 3.4 판정 & 파이프라인 (`decision/`, `pipeline/`)

```
decision/
├── decision_engine.py            # 메인 판정 로직
├── decision_builder.py           # 판정 구성
└── uncertainty.py                # 불확실성 정량화

pipeline/
├── analyzer.py                   # 다중 샘플 분석
├── single_analyzer.py            # 단일 샘플 특징 추출
└── feature_export.py             # 특징 내보내기
```

### 3.5 핵심 데이터 타입 (`types.py`)

```python
@dataclass
class LensGeometry:
    cx: float          # 중심 X
    cy: float          # 중심 Y
    r: float           # 반경
    confidence: float  # 검출 신뢰도

@dataclass
class GateResult:
    passed: bool
    reasons: List[str]
    scores: Dict[str, float]

@dataclass
class Decision:
    judgment: str      # OK/NG/RETAKE
    confidence: float
    reasons: List[str]
    gate_result: GateResult
    signature_result: SignatureResult
    anomaly_result: AnomalyResult
```

---

## 4. 웹 애플리케이션 구조 (`src/web/`)

### 4.1 메인 애플리케이션 (`app.py`)

**주요 엔드포인트:**

| 엔드포인트 | 메소드 | 용도 |
|-----------|--------|------|
| `/inspect` | POST | 메인 검사 |
| `/recompute` | POST | 캐시된 이미지 재분석 |
| `/batch` | POST | 배치 처리 |
| `/v7` | GET | V7 MVP 인터페이스 |
| `/single_analysis` | GET | 단일 샘플 분석 |

### 4.2 라우터 (`routers/`)

**inspection.py:**
```python
POST /api/inspection/inspect      # 이미지 검사
GET  /api/inspection/history      # 검사 이력
GET  /api/inspection/stats        # 통계
```

**v7.py (핵심 V7 API):**
```python
POST /api/v7/inspect              # V7 엔진 검사
POST /api/v7/register             # STD 등록
POST /api/v7/analyze_single       # 단일 분석
POST /api/v7/compare              # 비교
POST /api/v7/simulation           # 색상 시뮬레이션
GET  /api/v7/status               # STD 상태
```

### 4.3 템플릿 (`templates/`)

```
_base_layout.html              # 기본 레이아웃
index.html                     # 메인 검사 UI
v7_mvp.html                    # V7 MVP 인터페이스
single_analysis.html           # 단일 샘플 분석
calibration.html               # 색상 캘리브레이션
history.html                   # 검사 이력 뷰어
```

### 4.4 프론트엔드 구조 (`static/`)

```
js/
├── core/
│   ├── api.js                # API 클라이언트
│   └── state.js              # 상태 관리
├── components/
│   ├── base.js               # 기본 컴포넌트
│   ├── tabs.js               # 탭 컴포넌트
│   ├── viewer.js             # 이미지 뷰어
│   └── visuals.js            # 시각화
├── features/
│   ├── inspection/           # 검사 UI
│   ├── analysis/             # 분석 UI
│   │   ├── single.js         # 단일 분석
│   │   └── ink_visuals.js    # 잉크 시각화
│   ├── registration/         # 등록 UI
│   └── history/              # 이력 UI
└── utils/
    ├── helpers.js            # 헬퍼 함수
    ├── i18n.js               # 다국어 지원
    └── notifications.js      # 알림
```

---

## 5. 데이터베이스 스키마

### 5.1 InspectionHistory 모델

```python
class InspectionHistory(Base):
    __tablename__ = "inspection_history"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, index=True)        # run_id
    sku_code = Column(String, index=True)          # 제품 코드
    image_filename = Column(String)
    image_path = Column(String)

    # 판정 결과
    judgment = Column(String, index=True)          # OK/NG/RETAKE
    overall_delta_e = Column(Float)
    confidence = Column(Float)

    # 분석 결과 (JSON)
    analysis_result = Column(JSON)
    ng_reasons = Column(JSON)
    retake_reasons = Column(JSON)
    decision_trace = Column(JSON)
    next_actions = Column(JSON)

    # 렌즈 정보
    lens_detected = Column(Boolean, index=True)
    lens_confidence = Column(Float)

    # 메타데이터
    created_at = Column(DateTime, index=True)
    operator = Column(String)
    batch_number = Column(String)
    processing_time_ms = Column(Integer)
```

### 5.2 마이그레이션 이력

```
e377e0730c8c  initial_schema
5cd42af34616  add_inspection_history_table
0f3c5bb4c5f2  add_batch_number
9d2b1c7a4f13  drop_zones_count
a77c82cbb191  add_profile_score
```

---

## 6. 설정 파일 구조

### 6.1 엔진 설정 (`src/engine_v7/configs/default.json`)

```json
{
  "polar": {
    "num_angles": 720,
    "num_radii": 100,
    "r_inner_ratio": 0.2,
    "r_outer_ratio": 0.95
  },
  "gate": {
    "blur_threshold": 100,
    "illumination_threshold": 0.3,
    "center_offset_threshold": 10
  },
  "signature": {
    "min_correlation": 0.85,
    "max_delta_e": 8.0
  },
  "anomaly": {
    "blob_min_area": 50,
    "angular_uniformity_threshold": 0.1
  },
  "plate_lite": {
    "enabled": true,
    "override_plate": true,
    "blur_ksize": 5
  }
}
```

### 6.2 SKU 설정 (`config/sku_db/SKU001.json`)

```json
{
  "sku_code": "SKU001",
  "description": "3-zone colored contact lens",
  "zones": {
    "A": {"L": 45.0, "a": 8.0, "b": 28.0, "threshold": 8.0},
    "B": {"L": 68.0, "a": 5.0, "b": 22.0, "threshold": 8.0},
    "C": {"L": 95.0, "a": 0.5, "b": 2.0, "threshold": 10.0}
  },
  "params": {
    "expected_zones": 3,
    "optical_clear_ratio": 0.15
  }
}
```

---

## 7. 데이터 흐름

### 7.1 검사 요청 플로우

```
┌─────────────────────────────────────────────────────────────┐
│                    웹 UI / API 레이어                        │
│  (FastAPI app.py + routers + templates)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         v             v             v
    InspectionAPI  V7 API        DatabaseAPI
    (inspection.py) (v7.py)  (inspection_models.py)
         │             │             │
         └─────────────┼─────────────┘
                       │
         ┌─────────────v──────────────┐
         │   서비스 레이어             │
         │ inspection_service.py      │
         │ analysis_service.py        │
         └─────────────┬──────────────┘
                       │
         ┌─────────────v──────────────────────────────┐
         │      엔진 V7 코어 파이프라인                │
         │  (src/engine_v7/core/pipeline/)            │
         │  • analyzer.py (다중 샘플)                  │
         │  • single_analyzer.py (단일 샘플)          │
         └──────────┬─────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┬────────────────┬──────────────┐
    │               │               │                │              │
    v               v               v                v              v
  Measure       Signature       Decision          Anomaly         Gate
  (세그멘테이션,  (방사형         (판정            (결함            (이미지
   잉크 매칭,     프로파일,       엔진,            탐지,            품질)
   메트릭)        STD 모델)       판정)            히트맵)
```

### 7.2 완전한 검사 데이터 흐름

```
사용자 업로드 (index.html)
    ↓
POST /inspect (app.py)
    ↓
[파일 검증 & 저장]
    ↓
V7 엔진 API (api.py의 inspect_single)
    ├─ 설정 로드 (config_loader.py)
    └─ analyze_single_sample() ← 핵심 분석
        ├─ Gate: detect_lens_circle() → LensGeometry
        ├─ Color: 세그멘테이션 → 색상 마스크 → LAB 추출
        ├─ Radial: to_polar() → 프로파일 분석
        ├─ Ink: 주요 색상 추출 → 클러스터링
        ├─ Plate: plate_lite 분석 (White/Black 쌍)
        └─ Pattern: 이상 탐지
    ↓
Decision Builder (decision_engine.py)
    ├─ Gate 판정 (품질 검사)
    ├─ Signature 비교 (STD 모델 매칭)
    ├─ Anomaly 탐지 (결함)
    └─ 최종 판정 (OK/NG/RETAKE)
    ↓
이미지 캐시 (UUID)
    ↓
데이터베이스 저장 (InspectionHistory)
    ↓
응답 JSON 생성
    ├─ 판정 결과
    ├─ 메트릭 & 분석
    ├─ 렌즈 정보
    └─ 시각화 경로
    ↓
클라이언트 반환
    ↓
결과 표시 (v7_mvp.html + viewer.js)
```

---

## 8. 핵심 알고리즘

### 8.1 렌즈 검출 (Hough Circle Transform)

```python
# lens_geometry.py
def detect_lens_circle(image, config):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=config['min_radius'],
        maxRadius=config['max_radius']
    )
    return LensGeometry(cx, cy, r, confidence)
```

### 8.2 방사형 시그니처 추출

```python
# radial_signature.py
def to_polar(image, geometry, num_angles=720, num_radii=100):
    """
    직교 좌표를 극좌표로 변환
    - 720개 각도 (0.5° 간격)
    - 100개 반경 샘플
    """
    polar = np.zeros((num_radii, num_angles, 3))
    for r_idx in range(num_radii):
        for a_idx in range(num_angles):
            r = r_inner + (r_outer - r_inner) * r_idx / num_radii
            theta = 2 * np.pi * a_idx / num_angles
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            polar[r_idx, a_idx] = bilinear_interpolate(image, x, y)
    return polar
```

### 8.3 잉크 세그멘테이션 (K-Means 클러스터링)

```python
# ink_segmentation.py
def segment_inks(image, mask, expected_k=3):
    """
    LAB 색공간에서 K-Means 클러스터링으로 잉크 분리
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    pixels = lab[mask > 0].reshape(-1, 3)

    kmeans = KMeans(n_clusters=expected_k, random_state=42)
    labels = kmeans.fit_predict(pixels)

    clusters = []
    for k in range(expected_k):
        cluster_pixels = pixels[labels == k]
        clusters.append({
            'mean_lab': cluster_pixels.mean(axis=0),
            'std_lab': cluster_pixels.std(axis=0),
            'area_ratio': len(cluster_pixels) / len(pixels)
        })
    return clusters
```

### 8.4 Plate-Lite 알파 맵 계산

```python
# plate_engine.py
def _compute_alpha_map_lite(white_bgr, black_bgr, blur_ksize=5, backlight=255.0):
    """
    White/Black 이미지 쌍으로 알파 맵 계산
    alpha = 1 - (White - Black) / Backlight
    """
    white_gray = cv2.cvtColor(white_bgr, cv2.COLOR_BGR2GRAY).astype(float)
    black_gray = cv2.cvtColor(black_bgr, cv2.COLOR_BGR2GRAY).astype(float)

    if blur_ksize > 1:
        white_gray = cv2.GaussianBlur(white_gray, (blur_ksize, blur_ksize), 0)
        black_gray = cv2.GaussianBlur(black_gray, (blur_ksize, blur_ksize), 0)

    alpha = 1.0 - (white_gray - black_gray) / backlight
    alpha = np.clip(alpha, 0.0, 1.0)

    return alpha
```

### 8.5 색상 시뮬레이션 (color_simulator.py)

```python
def build_simulation_result(
    ink_clusters,
    plate_info=None,
    radial_info=None,
    plate_lite_info=None,
):
    """
    잉크 클러스터와 plate 데이터를 결합하여
    3열 색상 비교 생성:
    - lens_clustering: 렌즈에서 추출한 색상
    - plate_measurement: plate/plate_lite에서 측정한 색상
    - proofing_simulation: 시뮬레이션된 색상
    """
    color_comparison = []
    for idx, cluster in enumerate(ink_clusters):
        comparison = {
            'role': cluster.get('role'),
            'lens_clustering': {
                'lab': cluster['mean_lab'],
                'hex': lab_to_hex(cluster['mean_lab'])
            },
            'plate_measurement': None,  # plate/plate_lite에서 채움
            'proofing_simulation': None
        }
        # plate_lite 데이터가 있으면 plate_measurement 채움
        if plate_lite_info:
            observed = white_observed_by_index[idx]
            if observed:
                comparison['plate_measurement'] = {
                    'lab': observed['lab'],
                    'hex': observed['hex'],
                    'source': 'plate_lite'  # P-Lite 표시용
                }
        color_comparison.append(comparison)
    return {'color_comparison': color_comparison}
```

---

## 9. 보안 기능

### 9.1 경로 순회 방지

```python
# app.py
def safe_sku_path(sku_code: str) -> Path:
    """SKU 코드 검증 및 안전한 경로 생성"""
    if not re.match(r'^[A-Za-z0-9_-]+$', sku_code):
        raise ValueError("Invalid SKU code")
    path = SKU_DB_PATH / f"{sku_code}.json"
    if not path.resolve().is_relative_to(SKU_DB_PATH.resolve()):
        raise ValueError("Path traversal detected")
    return path
```

### 9.2 파일 검증

```python
def validate_upload(file: UploadFile):
    """업로드 파일 검증"""
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(400, "Invalid file type")

    max_size = 50 * 1024 * 1024  # 50MB
    if file.size > max_size:
        raise HTTPException(400, "File too large")
```

### 9.3 이미지 캐시 관리

```python
# 캐시 TTL 및 메모리 제한
IMAGE_CACHE_TTL = 600  # 10분
IMAGE_CACHE_MAX_SIZE = 100  # 최대 100개 이미지
```

---

## 10. 기술 스택

### 백엔드
| 기술 | 버전 | 용도 |
|------|------|------|
| Python | 3.10+ | 메인 언어 |
| FastAPI | 0.104+ | 웹 프레임워크 |
| SQLAlchemy | 2.0+ | ORM |
| NumPy | - | 수치 연산 |
| OpenCV | 4.8+ | 이미지 처리 |
| scikit-learn | - | 클러스터링 |
| Matplotlib/Plotly | - | 시각화 |

### 프론트엔드
| 기술 | 용도 |
|------|------|
| HTML5/CSS3 | 마크업/스타일 |
| JavaScript (ES6) | 클라이언트 로직 |
| Jinja2 | 서버사이드 템플릿 |
| TailwindCSS | 스타일링 |

### 인프라
| 기술 | 용도 |
|------|------|
| Uvicorn | ASGI 서버 |
| SQLite | 기본 데이터베이스 |
| Alembic | DB 마이그레이션 |

---

## 11. 테스트 구조

```
tests/
├── conftest.py                 # Pytest 설정 및 픽스처
├── test_plate_lite.py          # Plate-Lite 테스트 (26개)
├── test_single_analyzer.py     # 단일 분석 테스트
├── test_color_simulator.py     # 색상 시뮬레이션 테스트
├── test_api_endpoints.py       # API 엔드포인트 테스트
└── e2e/
    └── test_inspection_flow.py # E2E 테스트
```

### 테스트 실행

```bash
# 전체 테스트
pytest tests/

# 특정 모듈 테스트
pytest tests/test_plate_lite.py -v

# 커버리지 포함
pytest tests/ --cov=src --cov-report=html
```

---

## 12. 최근 구현된 기능: Plate-Lite

### 12.1 개요
Plate-Lite는 White/Black 이미지 쌍을 사용하여 간소화된 plate 분석을 수행하는 기능입니다.

### 12.2 데이터 흐름

```
White 이미지 + Black 이미지
    ↓
plate_engine.analyze_plate_lite_pair()
    ├─ _compute_alpha_map_lite() → 알파 맵
    ├─ _make_plate_masks() → 존 마스크
    └─ zones: ring_core, dot_core, clear
    ↓
single_analyzer.py
    └─ results["plate_lite"] = {...}
    ↓
color_simulator.build_simulation_result()
    └─ plate_lite_info → color_comparison[i].plate_measurement
        └─ source: "plate_lite"
    ↓
ink_visuals.js / single.js
    └─ source === "plate_lite" → "P-Lite" 라벨 표시
```

### 12.3 UI 표시

```javascript
// ink_visuals.js:55-58
const plateSource = comparison.plate_measurement?.source;
const isPlate_Lite = plateSource === "plate_lite";
const plateLabel = isPlate_Lite ? "P-Lite" : "Plate";
const plateLabelColor = isPlate_Lite ? "text-orange-400" : "text-amber-400";
```

---

## 13. 결론

Color Meter는 모듈화된 구조로 설계되어 있으며, 핵심 분석 엔진(V7)과 웹 인터페이스가 명확하게 분리되어 있습니다.

**장점:**
- 명확한 계층 분리 (Engine → Service → API → UI)
- 확장 가능한 모듈 구조
- 포괄적인 테스트 커버리지
- 보안 기능 내장

**개선 가능 영역:**
- 비동기 처리 확대
- 캐싱 전략 고도화
- API 문서화 강화 (OpenAPI/Swagger)

---

*보고서 작성: Claude Code*
*작성일: 2026-01-20*
