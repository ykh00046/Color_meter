# 📦 SKU 관리 모듈 설계 (SkuConfigManager & Baseline Generator)

> **작성:** 2025-12-11
> **목적:** 다수 SKU의 기준값을 효율적으로 관리하고, 양품 이미지로부터 **자동으로 기준값을 생성하는 기능**을 설계합니다. 또한 CLI를 통한 SKU CRUD 인터페이스를 정의합니다.

## 1. 개요
### 1.1 배경
현재 시스템은 초기 버전에서 `SKU001` 하나만 지원하며, 해당 SKU의 기준 LAB 값과 허용 오차를 수동으로 JSON에 입력해두었습니다. 실제 생산 환경에서는 **수십~수백 개 SKU**를 운영해야 하므로:
- 신규 SKU 추가 시 일일이 JSON을 작성하는 것은 번거롭습니다.
- 기준값 측정에 사람이 관여하면 오류 가능성이 있습니다.

### 1.2 목표
1. **중앙 관리**: SKU **생성(Create)**, 조회(Read), 수정(Update), 삭제(Delete)의 CRUD 기능을 제공하여 운영 중 SKU 구성을 동적으로 관리.
2. **베이스라인 자동 생성**: 양품 샘플 이미지를 입력하면 해당 SKU의 JSON 기준 파일을 자동 생성 (ΔE 통계 등 메타정보 포함).
3. **다중 SKU 지원**: Pipeline에서 SKU를 바꿔가며 검사하거나, 한 번에 여러 SKU의 결과를 관리.
4. **CLI 통합**: 개발/운영자가 쉽게 접근하도록 `src/main.py`에 SKU 관리용 **서브커맨드**를 추가합니다 (예: `python src/main.py sku create ...`).

### 1.3 범위
- `SkuConfigManager` 클래스 설계 및 구현.
- 베이스라인 생성 도구 (`tools/generate_sku_baseline.py`) 설계.
- CLI 인터페이스 (`src/main.py`의 `sku` subcommand) 설계.
- SKU JSON 스키마 표준화 및 검증 방법.

## 2. SKU JSON 스키마
SKU의 기준값은 JSON 파일로 저장되며, **표준 스키마**를 갖습니다.
### 2.1 기본 구조
SKU 설정 JSON의 기본 예는 다음과 같습니다:
```json
{
  "sku_code": "SKU002",
  "description": "Blue colored contact lens - 3 zones",
  "default_threshold": 3.5,
  "zones": {
    "A": {
      "L": 70.5, "a": -10.2, "b": -30.8,
      "threshold": 4.0,
      "description": "Outer zone (blue)"
    },
    "B": {
      "L": 68.3, "a": -8.5, "b": -28.2,
      "threshold": 3.5,
      "description": "Middle zone (blue)"
    },
    "C": {
      "L": 65.1, "a": -6.8, "b": -25.5,
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
      "zone_A": { "L_std": 0.5, "a_std": 0.3, "b_std": 0.4, "samples": 5 },
      "zone_B": { "L_std": 0.4, "a_std": 0.2, "b_std": 0.3, "samples": 5 },
      "zone_C": { "L_std": 0.3, "a_std": 0.2, "b_std": 0.2, "samples": 5 }
    },
    "notes": "Generated from SKU002_OK_001-005.jpg"
  }
}
```
필수 필드는 `sku_code`, `default_threshold`, `zones`, `metadata` 등입니다. zones 객체의 key는 보통 알파벳 대문자로 Zone명을 표시합니다 (A, B, C...).

### 2.2 주요 필드 설명
- `sku_code` (string, required): SKU 고유 식별자 (예: "SKU002"). 파일명 및 내부 코드에 모두 사용됩니다.
- `description` (string, optional): 사람이 읽기 쉬운 SKU 설명.
- `default_threshold` (number, required): ΔE 기본 허용 오차. 특정 Zone에 threshold가 별도 지정되지 않으면 이 값을 기준으로 판정합니다 (일반적으로 3.0 ~ 5.0).
- `zones` (object, required): Zone별 기준값과 허용치:
  - 각 Zone 키 아래 객체는 해당 구역의 기준 L*, a*, b* 값과 허용 threshold를 가집니다.
  - `threshold` 필드는 그 Zone만의 허용 ΔE (없으면 default_threshold를 사용).
  - `description` (optional): Zone에 대한 부가 설명 (예: "center clear zone").
- `metadata` (object, required): 기준값이 생성된 시점과 방법 등의 정보:
  - `created_at`, `last_updated`: ISO8601 타임스탬프. 생성 및 최종 수정 시각.
  - `baseline_samples`: 이 기준을 만드는 데 사용한 샘플 이미지 개수 (auto gen의 경우).
  - `calibration_method`: "manual" 또는 "auto_generated" 등으로 값이 채워짐.
  - `author`: 기준값 작성자 (수동 작성 시 사용자 이름이나, 자동 생성 시 "system").
  - `statistics`: Zone별 통계 (표준편차 등) – 자동 생성 시 계산됨.
  - `notes`: 비고 또는 출처 등.

### 2.3 스키마 검증
JSON 구조의 유효성을 보장하기 위해 JSON Schema를 정의합니다 (Draft7 예시):
```python
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
        "^[A-Z]$": {  # Zone 키는 한 글자 대문자 (단일 알파벳)
          "type": "object",
          "required": ["L", "a", "b", "threshold"],
          "properties": {
            "L": {"type": "number", "minimum": 0, "maximum": 100},
            "a": {"type": "number", "minimum": -128, "maximum": 127},
            "b": {"type": "number", "minimum": -128, "maximum": 127},
            "threshold": {"type": "number", "minimum": 0},
            "description": {"type": "string"}
          }
        }
      }
    },
    "metadata": {
      "type": "object",
      "required": ["created_at", "last_updated", "baseline_samples", "calibration_method"],
      "properties": {
        "created_at": {"type": "string", "format": "date-time"},
        "last_updated": {"type": "string", "format": "date-time"},
        "baseline_samples": {"type": "integer", "minimum": 1},
        "calibration_method": {"type": "string", "enum": ["manual", "auto_generated"]},
        "author": {"type": "string"},
        "statistics": {"type": "object"},  # 상세 내용까지 검증하지는 않음 (있을 경우 object)
        "notes": {"type": "string"}
      }
    }
  }
}
```
위 스키마를 `jsonschema` 등을 통해 SkuConfigManager 내에서 활용하여 로드 시 검증할 계획입니다.

## 3. SkuConfigManager 클래스
SKU 설정을 메모리 상에서 관리하고 JSON 파일 입출력을 담당하는 클래스로 설계합니다.

### 3.1 클래스 역할 및 다이어그램
```
SkuConfigManager
├─ __init__(db_path: Path)
├─ create_sku(sku_code, description, zones, ...) -> Dict
├─ get_sku(sku_code: str) -> Dict
├─ update_sku(sku_code: str, updates: Dict) -> Dict
├─ delete_sku(sku_code: str) -> bool
├─ list_all_skus() -> List[Dict]
├─ generate_baseline(sku_code, image_paths, ...) -> Dict
├─ validate_sku(sku_data: Dict) -> bool
└─ (internal) _load_file(sku_code), _save_file(sku_code, data) ...
```

### 3.2 주요 메서드

#### 3.2.1 create_sku()
```python
def create_sku(self, sku_code: str, description: str, zones: Dict[str, Any], default_threshold: float = 3.0) -> Dict:
    # 1. 이미 존재하는 SKU 코드인지 확인
    if (self.db_path / f"{sku_code}.json").exists():
        raise SkuExistsError(f"{sku_code} already exists")
    # 2. 기본 구조 생성
    sku_data = {
        "sku_code": sku_code,
        "description": description,
        "default_threshold": default_threshold,
        "zones": zones,
        "metadata": {
            "created_at": now_iso(), "last_updated": now_iso(),
            "baseline_samples": 0,
            "calibration_method": "manual",
            "author": current_user or "unknown"
        }
    }
    # 3. 검증 및 저장
    self.validate_sku(sku_data, raise_exc=True)
    self._save_file(sku_code, sku_data)
    return sku_data
```
- 수동으로 zone 딕셔너리를 만들어 넣으면 새 SKU JSON을 생성합니다. (`zones` 딕셔너리는 `{"A": {"L": ..., "a":..., "b":..., "threshold":...}, ...}` 형태)
- 생성한 데이터는 JSON 파일로 저장되고 내장 딕셔너리에도 로드합니다.

#### 3.2.2 generate_baseline()
```python
def generate_baseline(self, sku_code: str, ok_images: List[Path], expected_zones: int = None) -> Dict:
    # 1. 이미지들을 처리하여 평균 LAB 값 계산
    profiles = [analyze_image(img_path) for img_path in ok_images]  # analyze_image -> RadialProfiler + 평균
    avg_lab = compute_average_lab(profiles)
    zones = {}

    # 2. expected_zones 수에 따라 프로파일을 등분 or 자동 검출
    if expected_zones:
        boundaries = split_evenly(expected_zones, profiles)
    else:
        boundaries = detect_boundaries_from_profile(avg_lab)

    # 3. 각 zone 평균 계산 및 threshold 추정
    for idx, zone_range in enumerate(boundaries):
        L_mean, a_mean, b_mean = calc_zone_lab(avg_lab, zone_range)
        zones[chr(65+idx)] = {
            "L": round(L_mean, 1), "a": round(a_mean, 1), "b": round(b_mean, 1),
            "threshold": round(default_threshold * 1.0, 1)  # 초기값으로 default_threshold 사용
        }

    # 4. SKU JSON 생성
    description = f"Auto-generated baseline for {expected_zones or len(zones)} zones"
    sku_data = self.create_sku(sku_code, description, zones, default_threshold=self.default_threshold)

    # 5. metadata 보완
    sku_data["metadata"].update({
        "baseline_samples": len(ok_images),
        "calibration_method": "auto_generated",
        "author": "system",
        "statistics": compute_statistics(profiles, boundaries)
    })
    sku_data["metadata"]["last_updated"] = now_iso()
    self._save_file(sku_code, sku_data)
    return sku_data
```
- OK 이미지 리스트를 받아 각 이미지를 RadialProfiler 등을 사용해 LAB 프로파일을 얻고, 평균 프로파일 `avg_lab`을 구합니다.
- `expected_zones`가 주어지면 해당 개수로 균등 분할하거나, 아니면 자동 경계 검출 알고리즘(`detect_boundaries_from_profile`)으로 Zone 경계를 결정합니다.
- 각 Zone별 평균 L, a, b 값을 계산하여 JSON 구조를 채웁니다. `threshold`는 일단 `default_threshold` 값으로 넣거나, Zone별 표준편차 등을 고려해 살짝 높일 수도 있습니다.
- `create_sku()`를 호출해 JSON 생성 및 저장을 수행합니다. (create_sku 내부에서 validate 후 저장)
- 추가로 metadata의 `baseline_samples`, `calibration_method` 등을 업데이트하고 다시 저장합니다. (또는 create_sku에 파라미터로 넘겨 한 번에 저장할 수도 있음)

#### 3.2.3 get_sku()
```python
def get_sku(self, sku_code: str) -> Dict:
    path = self.db_path / f"{sku_code}.json"
    if not path.exists():
        raise SkuNotFoundError(f"{sku_code} not found")
    return json.loads(path.read_text(encoding='utf-8'))
```
- 단순히 JSON 파일을 읽어 파싱하여 반환. (실제 구현에서는 캐싱하거나, 클래스 초기화 시 모든 SKU를 메모리에 로드해둘 수도 있음)

#### 3.2.4 update_sku()
```python
def update_sku(self, sku_code: str, updates: Dict) -> Dict:
    data = self.get_sku(sku_code)
    # 특정 필드들만 업데이트 허용 또는 전체 덮어쓰기. 여기서는 예시로 description만.
    if "description" in updates:
        data["description"] = updates["description"]
    # zones나 threshold 업데이트 로직도 필요할 수 있음 (생략)
    data["metadata"]["last_updated"] = now_iso()
    self.validate_sku(data, raise_exc=True)
    self._save_file(sku_code, data)
    return data
```
- 부분 업데이트를 수행 (구현 시 업데이트 가능한 필드를 제한하거나, zones 수정 기능 등을 추가할 수 있습니다).

#### 3.2.5 delete_sku()
```python
def delete_sku(self, sku_code: str) -> bool:
    path = self.db_path / f"{sku_code}.json"
    if not path.exists():
        return False
    path.unlink()  # 파일 삭제
    return True
```
- 단순히 파일을 지웁니다. (삭제 전 백업을 뜬다든지, running pipeline이 있으면 사용 중인지 체크 등의 고려가 필요할 수 있음)

#### 3.2.6 list_all_skus()
```python
def list_all_skus(self) -> List[Dict]:
    files = sorted(self.db_path.glob("SKU*.json"))
    return [self.get_sku(file.stem) for file in files]
```
- 현재 등록된 모든 SKU의 설정을 리스트로 반환. (file.stem으로 SKU 코드 획득)

### 3.3 내부 메서드
- `_load_file(sku_code)` / `_save_file(sku_code, data)`: JSON 파일 입출력 담당. `_save_file`에서는 `indent=2` 등의 형식으로 저장.
- `validate_sku(data, raise_exc=False)`: JSON Schema를 이용해 data 구조 검증. `raise_exc=True`이면 유효하지 않을 때 예외 발생.

## 4. 베이스라인 생성 알고리즘
이미 `generate_baseline` 메서드에서 간략히 다뤘지만, 추가로 어떤 접근 방법들이 있을지 정리합니다.

### 4.1 입력
- SKU 코드 및 해당 SKU의 양품 이미지 집합 (권장 최소 5장 이상).
- (옵션) `expected_zones`: 렌즈의 기대 Zone 개수 (정보가 있으면 더 정확한 분할 가능).

### 4.2 처리 과정
1. **이미지 프로파일 추출:** 각 이미지를 RadialProfiler로 분석 → 반경별 LAB 3채널 곡선 얻기.
2. **대표 프로파일 계산:** 모든 이미지의 프로파일을 평균내거나, 또는 중앙값 등을 사용하여 하나의 대표 프로파일을 생성.
3. **Zone 경계 검출:** 대표 프로파일 상에서 ΔE 변화가 큰 지점을 찾아 경계로 설정. 방법:
   - 1차 미분(Gradient) 커브 사용: 피크를 경계 후보로 추출.
   - 또는 `expected_zones` 값이 있다면 프로파일 길이를 해당 개수로 등분하는 초기 경계를 설정하고 미세 조정.
4. **Zone별 LAB 계산:** 결정된 Zone 구간마다, 해당 구간의 모든 프로파일 데이터에서 LAB 평균값을 계산.
5. **허용 오차 추정:** 각 Zone 내 샘플들 간 변동성을 바탕으로 threshold를 산정. 예:
   - 각 Zone의 L, a, b 값 표준편차를 구하고, 이를 ΔE 관점에서 통합한 값의 xσ를 threshold로 삼는다.
   - 또는 모든 ΔE(각 픽셀 vs 평균)를 구해 95% 신뢰구간 상한을 threshold로 설정.
   - 간략히는 `default_threshold`를 그대로 쓰거나, 0.5~1.0 정도 여유를 더하는 방식도 가능.
   *(다양한 방법이 있을 수 있으며, 현재 구현은 기본 threshold를 사용하고, 통계만 metadata에 기록하는 수준입니다.)*

### 4.3 Threshold 계산 방법 비교
| 방법 | 계산식 | 신뢰수준 | 장단점 |
|---|---|---|---|
| 표준편차 기반 | μ + k·σ (예: k=2) | ~95% | 통계적 근거 있으나 outlier에 민감 |
| 최대편차 기반 | max(ΔE_i) | 100% | 안전하지만 너무 보수적 (높은 값) |
| 기본값 사용 | 기본값 그대로 | - | 구현 용이, 경험에 의존 |

현 단계에선 **기본값 사용 + 통계 참고** 정도로 하며, 필요시 표준편차 기반으로 개선할 계획입니다.

### 4.4 예시
SKU002의 5장 양품 이미지로부터:
- 대표 프로파일 그래프 상에서 3개의 피크를 발견 → 3 zones 결정.
- 각 Zone 평균 LAB: A(70.5, -10.2, -30.8), B(68.3, -8.5, -28.2), C(65.1, -6.8, -25.5).
- 표준편차로 볼 때 Zone A는 ±0.5 내외 등으로 안정 → threshold 약 4.0 설정 (기본 3.5보다 약간 상향).
- JSON 생성 결과는 2.1절 예시와 같습니다.
*(실제 계산 세부치는 `DETAILED_IMPLEMENTATION_PLAN.md` 문서의 6장 데이터 요구사항에 일부 논의되어 있습니다.)*

## 5. CLI 인터페이스 설계 (sku subcommands)

> **⚠️ 중요 - CLI 도구 제거됨 (2025-01-12)**
> 본 섹션은 역사적 설계 문서로 보존됩니다.
> `src/main.py` CLI 도구는 제거되었으며, 모든 SKU 관리 및 검사 기능은 **웹 UI** (`http://localhost:8888`)를 통해 제공됩니다.
> - SKU 목록 조회: 웹 UI "History" 또는 "Stats" 페이지
> - SKU 상세 정보: `config/sku_db/` 폴더의 JSON 파일 직접 확인
> - 검사 실행: 웹 UI 홈페이지에서 이미지 업로드

---

**아래 내용은 설계 참고용입니다 (구현 제거됨):**

이제 SKU 관리를 위한 CLI 명령 설계를 정의합니다. `src/main.py`에서 `sku`라는 하위 명령군을 추가하여 여러 기능을 제공합니다.

### 5.1 명령어 구조
```bash
usage: main.py sku <subcommand> [options]

subcommands:
    list                현재 모든 SKU 목록 출력
    show <SKU>          특정 SKU의 상세정보 출력
    create <SKU> <JSON> 신규 SKU 생성 (JSON 문자열 또는 파일 경로 입력)
    generate-baseline <SKU> <img_dir_or_list> [--expected-zones N]
                        양품 이미지들로 기준 생성
    update <SKU> <JSON> SKU 정보 업데이트
    delete <SKU>        SKU 삭제
```
각 서브커맨드별 상세 동작은 아래와 같습니다:

### 5.2 서브커맨드 상세

#### 5.2.1 list – 모든 SKU 목록 조회
```bash
python -m src.main sku list
```
- 동작: `SkuConfigManager.list_all_skus()`를 호출, 각 SKU의 sku_code와 description을 콘솔에 출력.
- 출력 예:
  ```
  SKU001 - "Default sample SKU"
  SKU002 - "Blue colored contact lens - 3 zones"
  SKU003 - "Green lens - 2 zones"
  ... (etc)
  ```
  (JSON으로 출력할 필요까진 없고 가독성 위해 정리된 텍스트로)

#### 5.2.2 show – 특정 SKU 상세 조회
```bash
python -m src.main sku show SKU002
```
- 동작: 해당 SKU JSON을 불러와 포맷팅하여 출력.
- 출력 예: (요약해서 보여줄 수도 있고, raw JSON 출력 옵션을 둘 수도 있음)
  ```
  SKU002: Blue colored contact lens - 3 zones
  Zones:
    A: L=70.5, a=-10.2, b=-30.8 (threshold 4.0) - Outer zone (blue)
    B: L=68.3, a=-8.5, b=-28.2 (threshold 3.5) - Middle zone (blue)
    C: L=65.1, a=-6.8, b=-25.5 (threshold 3.0) - Inner zone (blue)
  Default ΔE threshold: 3.5
  Created at: 2025-12-11 14:30, Last updated: 2025-12-11 14:30
  Generated from 5 samples (auto_generated by system)
  ```

#### 5.2.3 create – SKU 생성
```bash
# 빈 SKU 생성 (Zone 정보는 추후 추가)
python -m src.main sku create SKU004 --desc "New Lens - manual entry"
```
혹은 JSON 파일을 이용:
```bash
python -m src.main sku create SKU005 --file config/sku_db/template_sku.json
```
- 동작: 입력으로 SKU 코드와 description, 그리고 zones 정보를 받아 `SkuConfigManager.create_sku` 호출.
- `--desc` 옵션과 `--file` 또는 `--json` 중 하나를 선택하여 Zone 정의를 입력받는 방식을 설계.
- 파일 입력 시 해당 JSON 파일 내용을 그대로 사용 (validate 후 저장).
- 출력: 성공 시 "SKU004 created." 등의 메시지.

#### 5.2.4 generate-baseline – 베이스라인 자동 생성
```bash
python -m src.main sku generate-baseline SKU006 --images data/raw_images/SKU6_ok/ --expected-zones 2
```
- 동작: `--images` 옵션으로 폴더 경로나 파일 리스트를 받아, 해당 이미지를 사용하여 `SkuConfigManager.generate_baseline` 실행. (`tools/generate_sku_baseline.py` 스크립트는 내부적으로 이 로직을 호출하거나, vice versa)
- `--expected-zones` 옵션으로 힌트 개수를 받을 수 있음.
- 출력: 생성된 SKU JSON의 요약 또는 경로. 예: "SKU006.json generated with 2 zones (baseline from 10 images)."

#### 5.2.5 update – SKU 수정
```bash
# Description 수정
python -m src.main sku update SKU002 --json '{"description": "Blue lens - updated description"}'
```
- 동작: 업데이트 내용(JSON)을 인자로 받아 `update_sku` 호출. 특정 필드만 변경.
- 출력: "SKU002 updated." 또는 변경 후 show와 동일한 정보를 출력.

#### 5.2.6 delete – SKU 삭제
```bash
python -m src.main sku delete SKU002
```
- 동작: 파일을 삭제하고 메모리 목록에서 제거.
- 출력: "SKU002 deleted." (없는 경우 "SKU002 not found.")

### 5.3 기존 검사 명령과의 관계
SKU 서브커맨드는 기존 `--image`/`--batch` 검사 명령과는 별개로 동작합니다. 즉 `main.py sku ...` 형태로 완전히 분리되어 있으므로, 혼용하지 않도록 argparse 설정합니다.
검사 명령 실행 시 SKU가 존재하지 않으면 에러가 나는데, 그 경우 사용자에게 `sku create` 또는 `generate-baseline`을 먼저 하라는 안내를 출력할 수도 있습니다.
(CLI 설계에서는 사용자 편의와 명령의 일관성을 고려했습니다.)

## 6. 파일 구조 연계
SKU 설정 파일들은 `config/sku_db/` 아래에 저장되므로, Deployment 시 이 폴더가 볼륨 마운트되어야 합니다 (이미 docker-compose에 반영됨).
또한 SKUManager 관련 코드는 `src/data/sku_manager.py`로 작성하고, Pipeline이나 main에서 임포트하여 사용합니다. 프로젝트 구조 상의 변경점:
- `src/data/sku_manager.py`     # 신규 추가 (SkuConfigManager 구현)
- `src/pipeline.py`             # 기존 pipeline (변경 없음, 다만 SKUManager를 import 가능)
- `src/main.py`                 # 수정: sku subcommands 추가
(DEVELOPMENT_GUIDE 등 문서에도 이 변경을 반영해야 함).

## 7. 에러 처리
SKU 관리 기능에서 발생 가능한 오류와 대처:

### 7.1 에러 타입
- `SkuNotFoundError`: 요청한 SKU 파일이 없음 (show/update/delete 시).
- `SkuExistsError`: 생성하려는 SKU 코드가 이미 존재.
- `ValidationError`: 입력한 JSON 데이터가 스키마에 안 맞음 (필드 누락 등).
- `BaselineGenerationError`: 이미지 처리 실패 등으로 자동 생성 못함.

### 7.2 예외 처리 전략
CLI에서 이러한 예외가 발생하면, 표준 오류로 메시지를 출력하고 종료 코드를 1로 반환합니다.
예를 들어:
```
ERROR: SKU "SKU002" not found.
```
또는
```
ERROR: Validation failed - default_threshold must be >=0
```
`generate-baseline`의 경우 이미지 읽기 실패 시 해당 파일명을 표시하고 건너뛰거나 전체 중단 여부를 `continue_on_error` 정책으로 결정합니다. 5장 중 1장 실패해도 나머지 4장으로 진행할 수 있도록 구현 가능합니다.
(종합적으로 에러 메시지를 사용자 친화적으로 작성합니다.)

## 8. 성능 고려사항
SKU 관리 자체는 파일 CRUD 위주이므로 성능 이슈는 크지 않으나, 수백~수천 개 SKU 시 고려사항:

### 8.1 베이스라인 생성 속도
이미지 개수가 많을 경우 (예: 100장), RadialProfiler 반복 실행이 병목일 수 있습니다. 추후 멀티스레드로 여러 이미지를 병렬 처리하거나, 프로파일 추출 자체를 Cython/Numba로 최적화하는 방안을 고려합니다.
현재 5~10장 정도는 수 초 내 처리 가능하므로 큰 문제는 없습니다.

### 8.2 SKU 조회 속도
- **현행 방식:** 파일 시스템 접근 (JSON 파일 읽기). 수백 개일 때도 필요한 시점에 개별 파일 읽는 것은 수 ms 이내라 충분히 빠릅니다.
- **잠재 한계:** SKU 수가 수천 개를 넘어서면 파일 접근 오버헤드가 누적될 수 있습니다. 그때는 SQLite 등 DB로 저장하거나, SKUManager가 초기 로드시 모두 읽어 캐싱하는 방법도 고려할 수 있습니다.

### 8.3 확장성
현재 JSON 파일 기반 구현은 수백 개 수준까지 실용적입니다. 그러나 SKU가 매우 많아질 경우 파일 관리에 부담이 생길 수 있으므로, DBMS로 마이그레이션을 생각합니다 (예: SQLite -> PostgreSQL).
또한 멀티프로세스 환경에서 동시 SKU 생성/수정 시 충돌 관리가 필요할 수 있습니다. 간단한 락킹이나 atomic write 방법을 적용합니다.

## 9. 보안 고려사항
SKU 정보는 민감할 수 있으므로:
### 9.1 SKU 코드 검증
`sku_code`에 특수문자나 경로문자("/", "\") 등이 들어오지 않도록 패턴 검증 (정규식 `^SKU[0-9]+$` 적용). 이는 디렉토리 트래버설 등을 막기 위함입니다.

### 9.2 파일 시스템 접근
SkuConfigManager는 기본적으로 `config/sku_db/` 이외 경로를 다루지 않습니다. 혹시라도 `../` 등이 들어간 경로가 생성되지 않도록 철저히 검사합니다.
JSON 파일에 임의의 스크립트를 넣는 등의 행위는 실행 시 아무 영향이 없지만, 그래도 입력 값을 그대로 eval하는 일은 피합니다 (모두 문자열로 처리).

## 10. 테스트 전략
### 10.1 단위 테스트 (15개 정도 예상)
- `create_sku` 성공 케이스 (새 JSON 파일 생성 확인)
- `create_sku` 중복 SKU 코드 에러
- `get_sku` 존재/비존재 케이스
- `update_sku` 특정 필드 수정 후 반영 확인
- `delete_sku` 파일 삭제 확인
- `generate_baseline` 간이 테스트: 임의의 LAB 프로파일로 구성된 데이터를 넣고 결과 JSON 유효성 검사
- `validate_sku` 스키마 테스트: 잘못된 데이터로 False/예외 확인 등.

### 10.2 통합 테스트
실제 main.py CLI 호출을 통한 시나리오 테스트:
1. SKU 생성 -> 목록 -> 삭제까지 흐름 테스트 (예: SKU007 만들고 지우기)
2. `generate-baseline` 함수로 생성한 JSON을 곧바로 pipeline에 적용, 이미지 검사 수행 (pipeline 결과가 OK 나오도록 샘플 사용).
3. **Visual inspection:** generate-baseline 결과로 나온 overlay를 한두 개 확인하여, Zone 구분이 의도대로 되었는지 품질팀이 한번 검증.

*(이상으로 SKU 관리 설계에 대한 설명을 마칩니다. 구현 이후 실제 적용하면서 필요에 따라 수정/보완될 수 있습니다.)*
