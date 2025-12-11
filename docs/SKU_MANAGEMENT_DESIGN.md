# SKU Management System - Design Document

> **작성일**: 2025-12-11
> **버전**: 1.0
> **목적**: SKU 구성 관리 및 베이스라인 자동 생성 시스템 설계

---

## 1. 개요

### 1.1 배경
현재 시스템은 단일 SKU(SKU001)만 지원하며, 베이스라인 값을 수동으로 측정하여 JSON에 입력해야 합니다. 실제 생산 환경에서는 수십~수백 개의 SKU를 관리해야 하므로, 자동화된 SKU 관리 시스템이 필요합니다.

### 1.2 목표
1. **CRUD 기능**: SKU 생성, 조회, 수정, 삭제
2. **베이스라인 자동 생성**: OK 샘플 이미지 → SKU JSON 자동화
3. **다중 SKU 지원**: 여러 SKU 동시 관리 및 검사
4. **CLI 통합**: 사용자 친화적 명령어 인터페이스

### 1.3 범위
- SkuConfigManager 클래스 구현
- 베이스라인 생성 도구 (`tools/generate_sku_baseline.py`)
- CLI 확장 (`src/main.py` - `sku` 서브커맨드)
- SKU JSON 스키마 표준화

---

## 2. SKU JSON 스키마

### 2.1 기본 구조

```json
{
  "sku_code": "SKU002",
  "description": "Blue colored contact lens - 3 zones",
  "default_threshold": 3.5,
  "zones": {
    "A": {
      "L": 70.5,
      "a": -10.2,
      "b": -30.8,
      "threshold": 4.0,
      "description": "Outer zone (blue)"
    },
    "B": {
      "L": 68.3,
      "a": -8.5,
      "b": -28.2,
      "threshold": 3.5,
      "description": "Middle zone (blue)"
    },
    "C": {
      "L": 65.1,
      "a": -6.8,
      "b": -25.5,
      "threshold": 3.0,
      "description": "Inner zone (blue)"
    }
  },
  "metadata": {
    "created_at": "2025-12-11T14:30:00",
    "last_updated": "2025-12-11T14:30:00",
    "baseline_samples": 5,
    "calibration_method": "auto_generated",
    "author": "system",
    "statistics": {
      "zone_A": {
        "L_std": 0.5,
        "a_std": 0.3,
        "b_std": 0.4,
        "samples": 5
      },
      "zone_B": {
        "L_std": 0.4,
        "a_std": 0.2,
        "b_std": 0.3,
        "samples": 5
      },
      "zone_C": {
        "L_std": 0.3,
        "a_std": 0.2,
        "b_std": 0.2,
        "samples": 5
      }
    },
    "notes": "Generated from SKU002_OK_001-005.jpg"
  }
}
```

### 2.2 필드 설명

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `sku_code` | string | ✓ | SKU 고유 식별자 (예: "SKU002") |
| `description` | string | ✓ | SKU 설명 |
| `default_threshold` | float | ✓ | 기본 ΔE 임계값 (Zone별 미지정 시 사용) |
| `zones` | object | ✓ | Zone별 LAB 기준값 및 임계값 |
| `zones.<name>.L` | float | ✓ | L* 값 (0-100) |
| `zones.<name>.a` | float | ✓ | a* 값 (-128~127) |
| `zones.<name>.b` | float | ✓ | b* 값 (-128~127) |
| `zones.<name>.threshold` | float | ✓ | Zone별 ΔE 임계값 |
| `zones.<name>.description` | string | - | Zone 설명 |
| `metadata.created_at` | ISO8601 | ✓ | 생성 일시 |
| `metadata.last_updated` | ISO8601 | ✓ | 최종 수정 일시 |
| `metadata.baseline_samples` | int | ✓ | 베이스라인 생성에 사용된 샘플 수 |
| `metadata.calibration_method` | string | ✓ | `manual` 또는 `auto_generated` |
| `metadata.statistics` | object | - | Zone별 통계 (표준편차, 샘플 수) |

### 2.3 스키마 검증

```python
# JSON Schema (JSONSchema Draft 7)
SKU_SCHEMA = {
    "type": "object",
    "required": ["sku_code", "description", "default_threshold", "zones", "metadata"],
    "properties": {
        "sku_code": {"type": "string", "pattern": "^SKU[0-9]+$"},
        "description": {"type": "string", "minLength": 1},
        "default_threshold": {"type": "number", "minimum": 0},
        "zones": {
            "type": "object",
            "minProperties": 1,
            "patternProperties": {
                "^[A-Z]$": {
                    "type": "object",
                    "required": ["L", "a", "b", "threshold"],
                    "properties": {
                        "L": {"type": "number", "minimum": 0, "maximum": 100},
                        "a": {"type": "number", "minimum": -128, "maximum": 127},
                        "b": {"type": "number", "minimum": -128, "maximum": 127},
                        "threshold": {"type": "number", "minimum": 0}
                    }
                }
            }
        },
        "metadata": {
            "type": "object",
            "required": ["created_at", "last_updated", "baseline_samples", "calibration_method"],
            "properties": {
                "calibration_method": {"enum": ["manual", "auto_generated"]}
            }
        }
    }
}
```

---

## 3. SkuConfigManager 클래스

### 3.1 클래스 다이어그램

```
SkuConfigManager
├─ __init__(db_path: Path)
├─ create_sku(sku_code, description, zones, ...) -> Dict
├─ get_sku(sku_code: str) -> Dict
├─ update_sku(sku_code: str, updates: Dict) -> Dict
├─ delete_sku(sku_code: str) -> bool
├─ list_all_skus() -> List[Dict]
├─ generate_baseline(sku_code, ok_images, ...) -> Dict
├─ validate_sku(sku_data: Dict) -> bool
└─ _save_sku(sku_code: str, data: Dict) -> Path
```

### 3.2 주요 메서드

#### 3.2.1 `create_sku()`

```python
def create_sku(
    self,
    sku_code: str,
    description: str,
    default_threshold: float = 3.5,
    zones: Optional[Dict[str, Dict[str, float]]] = None,
    author: str = "user"
) -> Dict[str, Any]:
    """
    새로운 SKU 생성

    Args:
        sku_code: SKU 코드 (예: "SKU002")
        description: SKU 설명
        default_threshold: 기본 ΔE 임계값
        zones: Zone별 LAB 값 및 임계값
               예: {"A": {"L": 70, "a": -10, "b": -30, "threshold": 4.0}}
        author: 작성자

    Returns:
        생성된 SKU 데이터 (Dict)

    Raises:
        ValueError: SKU 코드가 이미 존재하거나 유효하지 않음
    """
```

#### 3.2.2 `generate_baseline()`

```python
def generate_baseline(
    self,
    sku_code: str,
    ok_images: List[Path],
    description: str = "",
    default_threshold: float = 3.5,
    threshold_method: str = "mean_plus_2std"
) -> Dict[str, Any]:
    """
    OK 샘플 이미지로부터 베이스라인 자동 생성

    Args:
        sku_code: SKU 코드
        ok_images: OK 샘플 이미지 경로 리스트 (최소 3장, 권장 5-10장)
        description: SKU 설명
        default_threshold: 기본 임계값
        threshold_method: 임계값 계산 방법
            - "mean_plus_2std": mean + 2 * std (기본, 95% 신뢰구간)
            - "mean_plus_3std": mean + 3 * std (99.7% 신뢰구간)
            - "fixed": default_threshold 사용

    Returns:
        생성된 SKU 데이터 (Dict)

    Raises:
        ValueError: 샘플 수가 부족하거나 처리 실패

    Algorithm:
        1. 각 OK 이미지를 파이프라인으로 처리
        2. Zone별 LAB 값 추출
        3. Zone별 평균 계산: mean_L, mean_a, mean_b
        4. Zone별 표준편차 계산: std_L, std_a, std_b
        5. Threshold 계산:
           - method="mean_plus_2std": default_threshold + 2 * max(std_L, std_a, std_b)
           - method="fixed": default_threshold
        6. SKU JSON 생성 및 저장
    """
```

#### 3.2.3 `get_sku()`

```python
def get_sku(self, sku_code: str) -> Dict[str, Any]:
    """
    SKU 데이터 조회

    Args:
        sku_code: SKU 코드

    Returns:
        SKU 데이터 (Dict)

    Raises:
        FileNotFoundError: SKU가 존재하지 않음
    """
```

#### 3.2.4 `update_sku()`

```python
def update_sku(
    self,
    sku_code: str,
    updates: Dict[str, Any]
) -> Dict[str, Any]:
    """
    SKU 데이터 수정

    Args:
        sku_code: SKU 코드
        updates: 수정할 필드 (Dict)
                 예: {"zones.A.threshold": 4.5, "description": "Updated"}

    Returns:
        수정된 SKU 데이터 (Dict)

    Raises:
        FileNotFoundError: SKU가 존재하지 않음
        ValueError: 유효하지 않은 업데이트
    """
```

#### 3.2.5 `delete_sku()`

```python
def delete_sku(self, sku_code: str) -> bool:
    """
    SKU 삭제

    Args:
        sku_code: SKU 코드

    Returns:
        성공 여부 (bool)

    Raises:
        FileNotFoundError: SKU가 존재하지 않음
    """
```

#### 3.2.6 `list_all_skus()`

```python
def list_all_skus(self) -> List[Dict[str, Any]]:
    """
    모든 SKU 목록 조회

    Returns:
        SKU 요약 정보 리스트
        예: [
            {
                "sku_code": "SKU001",
                "description": "...",
                "zones_count": 1,
                "created_at": "2025-12-11T10:10:00"
            },
            ...
        ]
    """
```

### 3.3 내부 메서드

```python
def _validate_sku(self, sku_data: Dict) -> bool:
    """JSON Schema 검증"""

def _save_sku(self, sku_code: str, data: Dict) -> Path:
    """SKU JSON 파일 저장 (config/sku_db/{sku_code}.json)"""

def _load_sku(self, sku_code: str) -> Dict:
    """SKU JSON 파일 로드"""

def _get_sku_path(self, sku_code: str) -> Path:
    """SKU JSON 파일 경로 반환"""
```

---

## 4. 베이스라인 생성 알고리즘

### 4.1 입력

- **OK 샘플 이미지**: 최소 3장, 권장 5-10장
- **SKU 코드**: 생성할 SKU 식별자
- **Threshold 계산 방법**: `mean_plus_2std` (기본)

### 4.2 처리 과정

```
Step 1: 이미지 로드 및 검증
  - OK 샘플 이미지 경로 리스트 입력
  - 각 이미지 존재 확인
  - 최소 샘플 수 검증 (≥3)

Step 2: 파이프라인 처리
  FOR each image IN ok_images:
    - ImageLoader.load_from_file()
    - ImageLoader.preprocess()
    - LensDetector.detect()
    - RadialProfiler.extract_profile()
    - ZoneSegmenter.segment()
    - Zone별 LAB 값 추출
    - zone_data[zone_name].append((L, a, b))

Step 3: Zone별 통계 계산
  FOR each zone IN zone_data:
    - mean_L = np.mean([L for L, a, b in zone_data[zone]])
    - mean_a = np.mean([a for L, a, b in zone_data[zone]])
    - mean_b = np.mean([b for L, a, b in zone_data[zone]])
    - std_L = np.std([L for L, a, b in zone_data[zone]])
    - std_a = np.std([a for L, a, b in zone_data[zone]])
    - std_b = np.std([b for L, a, b in zone_data[zone]])

Step 4: Threshold 계산
  IF threshold_method == "mean_plus_2std":
    FOR each zone:
      max_std = max(std_L, std_a, std_b)
      zone_threshold = default_threshold + 2.0 * max_std
  ELIF threshold_method == "mean_plus_3std":
    zone_threshold = default_threshold + 3.0 * max_std
  ELSE:
    zone_threshold = default_threshold

Step 5: SKU JSON 생성
  - sku_code, description, default_threshold
  - zones: {zone_name: {L, a, b, threshold, description}}
  - metadata: {created_at, baseline_samples, statistics, ...}

Step 6: 검증 및 저장
  - JSON Schema 검증
  - config/sku_db/{sku_code}.json 저장
  - 생성 완료 로그 출력
```

### 4.3 Threshold 계산 방법 비교

| 방법 | 수식 | 신뢰구간 | 용도 |
|------|------|----------|------|
| `mean_plus_2std` | μ + 2σ | 95% | 일반적 (기본값) |
| `mean_plus_3std` | μ + 3σ | 99.7% | 엄격한 품질 관리 |
| `fixed` | default | - | 수동 설정 |

### 4.4 예시

```python
# 입력
ok_images = [
    "data/raw_images/SKU002_OK_001.jpg",
    "data/raw_images/SKU002_OK_002.jpg",
    "data/raw_images/SKU002_OK_003.jpg",
    "data/raw_images/SKU002_OK_004.jpg",
    "data/raw_images/SKU002_OK_005.jpg"
]

# 처리 후 Zone A 데이터
zone_A_samples = [
    (70.5, -10.2, -30.8),
    (70.3, -10.0, -30.5),
    (70.7, -10.4, -31.0),
    (70.4, -10.1, -30.7),
    (70.6, -10.3, -30.9)
]

# 통계 계산
mean_L = 70.5, std_L = 0.15
mean_a = -10.2, std_a = 0.15
mean_b = -30.78, std_b = 0.18

# Threshold 계산 (mean_plus_2std)
max_std = 0.18
threshold = 3.5 + 2.0 * 0.18 = 3.86 ≈ 3.9

# 결과
{
  "A": {
    "L": 70.5,
    "a": -10.2,
    "b": -30.8,
    "threshold": 3.9
  }
}
```

---

## 5. CLI 인터페이스

### 5.1 명령어 구조

```bash
python -m src.main sku <subcommand> [options]
```

### 5.2 서브커맨드

#### 5.2.1 `list` - 모든 SKU 목록 조회

```bash
python -m src.main sku list

# 출력 예시
SKU Code    Description                      Zones  Created At
----------  ------------------------------  ------  -------------------
SKU001      3-zone colored lens (dummy)          1  2025-12-11 10:10:00
SKU002      Blue colored contact lens            3  2025-12-11 14:30:00
SKU003      Brown colored contact lens           3  2025-12-11 14:35:00

Total: 3 SKUs
```

#### 5.2.2 `show` - 특정 SKU 상세 조회

```bash
python -m src.main sku show SKU002

# 출력 예시
SKU Code: SKU002
Description: Blue colored contact lens - 3 zones
Default Threshold: 3.5

Zones:
  Zone A: L=70.5, a=-10.2, b=-30.8, threshold=3.9
  Zone B: L=68.3, a=-8.5, b=-28.2, threshold=3.6
  Zone C: L=65.1, a=-6.8, b=-25.5, threshold=3.3

Metadata:
  Created: 2025-12-11 14:30:00
  Samples: 5
  Method: auto_generated
```

#### 5.2.3 `create` - SKU 생성

```bash
# 빈 SKU 생성 (Zone 없음)
python -m src.main sku create \
  --code SKU004 \
  --description "Green colored lens"

# Zone과 함께 생성
python -m src.main sku create \
  --code SKU004 \
  --description "Green colored lens" \
  --zone A:70.0:-5.0:-20.0:4.0

# 여러 Zone
python -m src.main sku create \
  --code SKU004 \
  --description "Green colored lens" \
  --zone A:70.0:-5.0:-20.0:4.0 \
  --zone B:68.0:-4.0:-18.0:3.5
```

#### 5.2.4 `generate-baseline` - 베이스라인 자동 생성

```bash
python -m src.main sku generate-baseline \
  --sku SKU002 \
  --images data/raw_images/SKU002_OK_*.jpg \
  --description "Blue colored lens" \
  --method mean_plus_2std

# 출력 예시
Processing 5 OK samples...
  [1/5] SKU002_OK_001.jpg - OK (1 zone detected)
  [2/5] SKU002_OK_002.jpg - OK (1 zone detected)
  [3/5] SKU002_OK_003.jpg - OK (1 zone detected)
  [4/5] SKU002_OK_004.jpg - OK (1 zone detected)
  [5/5] SKU002_OK_005.jpg - OK (1 zone detected)

Calculated baselines:
  Zone A: L=70.5±0.15, a=-10.2±0.15, b=-30.8±0.18, threshold=3.9

SKU002 baseline generated successfully!
Saved to: config/sku_db/SKU002.json
```

#### 5.2.5 `update` - SKU 수정

```bash
# Description 수정
python -m src.main sku update SKU002 --description "Updated description"

# Zone threshold 수정
python -m src.main sku update SKU002 --zone-threshold A:4.5

# 여러 필드 동시 수정
python -m src.main sku update SKU002 \
  --description "Updated" \
  --default-threshold 4.0 \
  --zone-threshold A:4.5
```

#### 5.2.6 `delete` - SKU 삭제

```bash
python -m src.main sku delete SKU002

# 확인 프롬프트
Are you sure you want to delete SKU002? (y/N): y
SKU002 deleted successfully.
```

### 5.3 기존 명령어 (변경 없음)

```bash
# 단일 이미지 검사
python -m src.main inspect --image data/raw_images/SKU002_OK_001.jpg --sku SKU002

# 배치 처리
python -m src.main batch --dir data/raw_images --pattern "SKU002_*.jpg" --sku SKU002 --output results/sku002_results.csv
```

---

## 6. 파일 구조

```
config/sku_db/
├── SKU001.json        # 기존
├── SKU002.json        # 신규 (자동 생성)
├── SKU003.json        # 신규 (자동 생성)
└── ...

src/
├── sku_manager.py     # 신규 (SkuConfigManager)
├── pipeline.py        # 기존 (변경 없음)
└── main.py            # 수정 (sku 서브커맨드 추가)

tools/
└── generate_sku_baseline.py  # 신규 (CLI 래퍼)

tests/
└── test_sku_manager.py        # 신규
```

---

## 7. 에러 처리

### 7.1 에러 타입

| 에러 | 발생 조건 | 처리 방법 |
|------|----------|-----------|
| `SkuNotFoundError` | SKU 코드가 존재하지 않음 | 404 메시지, 사용 가능한 SKU 목록 출력 |
| `SkuAlreadyExistsError` | SKU 코드가 이미 존재 | 409 메시지, update 사용 권장 |
| `InvalidSkuDataError` | JSON 스키마 검증 실패 | 유효하지 않은 필드 목록 출력 |
| `InsufficientSamplesError` | 베이스라인 샘플 수 부족 (<3) | 최소 3장 필요 메시지 |
| `PipelineProcessingError` | 이미지 처리 중 실패 | 실패한 이미지 건너뛰기, 계속 진행 |

### 7.2 예시

```python
# SkuNotFoundError
try:
    sku_data = manager.get_sku("SKU999")
except SkuNotFoundError:
    print(f"Error: SKU999 not found.")
    print("Available SKUs:", manager.list_all_skus())

# InvalidSkuDataError
try:
    manager.create_sku("SKU002", zones={"A": {"L": 150}})  # L > 100
except InvalidSkuDataError as e:
    print(f"Invalid SKU data: {e.validation_errors}")
```

---

## 8. 성능 고려사항

### 8.1 베이스라인 생성 속도

- **목표**: 샘플 10장 처리 < 1초
- **현재 파이프라인 속도**: ~7ms/장
- **예상 소요 시간**: 70ms (10장) + 오버헤드 ≈ 150ms

### 8.2 SKU 조회 속도

- **방법**: 파일 시스템 (JSON 파일)
- **예상**: <10ms (파일 읽기)
- **대안**: 향후 SQLite 또는 인메모리 캐시 고려

### 8.3 확장성

- **현재 방식**: 파일 기반 (수백 개 SKU 지원 가능)
- **한계**: 수천 개 SKU 시 파일 시스템 한계
- **향후**: SQLite 또는 PostgreSQL 마이그레이션

---

## 9. 보안 고려사항

### 9.1 SKU 코드 검증

- **패턴**: `^SKU[0-9]+$` (예: SKU001, SKU002)
- **목적**: 디렉터리 순회 공격 방지

### 9.2 파일 시스템 접근

- **제한**: `config/sku_db/` 디렉터리만 접근
- **검증**: 생성된 경로가 베이스 디렉터리 내부인지 확인

```python
def _get_sku_path(self, sku_code: str) -> Path:
    if not re.match(r'^SKU[0-9]+$', sku_code):
        raise ValueError(f"Invalid SKU code: {sku_code}")

    path = self.db_path / f"{sku_code}.json"

    # Path traversal 방지
    if not path.resolve().is_relative_to(self.db_path.resolve()):
        raise ValueError(f"Invalid SKU path: {path}")

    return path
```

---

## 10. 테스트 전략

### 10.1 단위 테스트 (15개)

1. `test_create_sku_success()` - 정상 생성
2. `test_create_sku_already_exists()` - 중복 생성 실패
3. `test_get_sku_success()` - 정상 조회
4. `test_get_sku_not_found()` - 존재하지 않는 SKU
5. `test_update_sku_success()` - 정상 수정
6. `test_delete_sku_success()` - 정상 삭제
7. `test_list_all_skus()` - 목록 조회
8. `test_generate_baseline_single_zone()` - 단일 Zone 베이스라인
9. `test_generate_baseline_multi_zone()` - 다중 Zone 베이스라인
10. `test_generate_baseline_insufficient_samples()` - 샘플 부족
11. `test_threshold_calculation_mean_plus_2std()` - Threshold 계산
12. `test_validate_sku_schema()` - JSON 스키마 검증
13. `test_invalid_sku_code()` - 유효하지 않은 SKU 코드
14. `test_multi_sku_batch_processing()` - 다중 SKU 배치 처리
15. `test_cli_sku_commands()` - CLI 명령어 테스트

### 10.2 통합 테스트

- SKU002, SKU003 베이스라인 자동 생성
- 3개 SKU 동시 배치 처리 (30장)
- CLI 전체 워크플로우 테스트

---

## 11. 향후 확장 가능성

### 11.1 Phase 4: Database 마이그레이션
- SQLite → PostgreSQL
- SKU 버전 관리 (히스토리)
- 베이스라인 재생성 이력 추적

### 11.2 Phase 5: 웹 UI
- SKU 관리 대시보드
- 베이스라인 생성 시각화
- 검사 이력 통계

### 11.3 Phase 6: AI 기반 최적화
- 자동 Threshold 최적화 (ML)
- Zone 분할 자동 학습
- 이상 패턴 자동 검출

---

## 12. 참고 자료

- [JSON Schema](https://json-schema.org/)
- [Click CLI Framework](https://click.palletsprojects.com/)
- [ISO 8601 DateTime](https://en.wikipedia.org/wiki/ISO_8601)
- `DAY2_COMPLETION_REPORT.md` - 기존 파이프라인 구조
- `PIPELINE_DESIGN.md` - 파이프라인 아키텍처

---

**작성자**: Claude (AI Assistant)
**버전**: 1.0
**최종 수정**: 2025-12-11
