# 🔧 검사 파이프라인 설계 (InspectionPipeline Module)

> **작성:** 2025-12-11
> **목적:** 5개의 핵심 모듈(ImageLoader, LensDetector, RadialProfiler, ZoneSegmenter, ColorEvaluator)을 **엔드투엔드**로 연결하는 검사 파이프라인의 구조와 동작을 설계합니다. 또한 CLI와 연동, 에러 처리, 확장 가능성에 대해서도 다룹니다.

## 1. 개요 및 목표
**InspectionPipeline** 클래스는 콘택트렌즈 이미지와 SKU 코드를 입력받아 일련의 분석 단계를 거쳐 **최종 품질 판정 결과**를 반환하는 엔드투엔드 파이프라인입니다. 목표는 다음과 같습니다:
- 다양한 모듈의 **순차적 실행 흐름** 관리 (로드 → 검출 → 프로파일 → 분할 → 판정).
- 각 단계의 **입출력 데이터 구조 정의**와 모듈 간 **인터페이스** 표준화.
- 단일 이미지뿐 아니라 **다중 이미지 일괄 처리(batch)** 기능 제공.
- 예외 발생 시 처리 중단 없이 대응 (오류를 결과에 반영하거나 batch의 경우 다음 항목으로 진행).
- 향후 Visualizer 등의 **플러그인 모듈 연동** 지원 (후술하는 확장성 고려사항 참조).

아래는 파이프라인 전체 흐름의 다이어그램입니다:
```mermaid
Input: [이미지 경로] + [SKU 코드]
      ↓
[1] ImageLoader – 이미지 로드 및 전처리
      ↓
[2] LensDetector – 렌즈 위치/반경 검출
      ↓
[3] RadialProfiler – 렌즈를 극좌표 변환하여 색상 프로파일 추출
      ↓
[4] ZoneSegmenter – 색상 변화 지점으로 Zone 자동 분할
      ↓
[5] ColorEvaluator – 각 Zone의 ΔE 계산, OK/NG 판정
      ↓
Output: InspectionResult (판정 OK/NG, 전체 ΔE, Zone별 세부 결과 등)
```
*(참고: 각 단계의 알고리즘 상세 내용은 [상세 구현 계획](../planning/DETAILED_IMPLEMENTATION_PLAN.md) 3.2장에 설명되어 있습니다.)*

## 2. InspectionPipeline 클래스 설계
```python
class InspectionPipeline:
    """
    엔드투엔드 콘택트렌즈 색상 검사 파이프라인.
    5개 핵심 모듈을 순차적으로 실행하여 최종 판정 결과를 생성합니다.
    """
    def __init__(
        self,
        sku_config: Dict[str, Any],
        image_config: Optional[ImageConfig] = None,
        detector_config: Optional[DetectorConfig] = None,
        profiler_config: Optional[ProfilerConfig] = None,
        segmenter_config: Optional[SegmenterConfig] = None,
        save_intermediates: bool = False
    ):
        """파이프라인 초기화"""
        self.sku_config = sku_config
        self.image_loader = ImageLoader(image_config or {})
        self.lens_detector = LensDetector(detector_config or {})
        self.radial_profiler = RadialProfiler(profiler_config or {})
        self.zone_segmenter = ZoneSegmenter(segmenter_config or {})
        self.color_evaluator = ColorEvaluator()  # 평가 모듈은 특별한 설정 없음
        self.save_intermediates = save_intermediates

    def process(self, image_path: str, sku: str, save_dir: Optional[Path] = None) -> InspectionResult:
        """단일 이미지 검사 처리"""
        ...

    def process_batch(self, image_paths: List[str], sku: str, output_csv: Optional[Path] = None,
                      continue_on_error: bool = True) -> List[InspectionResult]:
        """다중 이미지 배치 처리"""
        ...
```

### 2.1 핵심 속성 및 초기화
- `sku_config`: SKU 기준값 딕셔너리. 보통 `config/sku_db/<SKU>.json`을 읽어온 내용으로 생성됩니다. (SKUConfigManager에서 JSON 로드 후 딕셔너리를 넘겨주는 방식)
- **모듈 인스턴스들:** ImageLoader, LensDetector, RadialProfiler, ZoneSegmenter, ColorEvaluator를 초기화합니다. 각 모듈은 자체 설정(config)을 가질 수 있으며, 필요 시 파이프라인 생성 시 인자로 조절 가능합니다.
- `save_intermediates`: True일 경우 중간 단계 결과(예: polar 변환 이미지, 프로파일 데이터 등)를 파일로 저장합니다 (디버깅용).

### 2.2 process() 메서드 (단일 이미지)
단일 이미지를 검사하는 상세 흐름은 아래와 같습니다 (의사코드):
```python
def process(self, image_path: str, sku: str, save_dir=None) -> InspectionResult:
    # 1. 이미지 로드 및 전처리
    img = self.image_loader.load_from_file(image_path)
    processed_img = self.image_loader.preprocess(img)

    # 2. 렌즈 검출
    lens = self.lens_detector.detect(processed_img)
    if lens is None:
        raise PipelineError("Lens detection failed")

    # 3. 극좌표 프로파일 추출
    radial_profile = self.radial_profiler.extract_profile(processed_img, lens)

    # 4. Zone 자동 분할
    zones = self.zone_segmenter.segment(radial_profile)

    # 5. 색상 평가 및 품질 판정
    result = self.color_evaluator.evaluate(zones, sku, self.sku_config)

    # (옵션) 중간 결과 저장
    if self.save_intermediates and save_dir:
        self._save_intermediates(save_dir, {
            "profile": radial_profile, "zones": zones, "lens": lens, "result": result
        })

    return result
```
**각 단계에서 필요한 데이터 형태:**
- `LensDetector.detect`: 전처리된 이미지를 받아 렌즈의 중심 좌표와 반지름을 반환합니다 (`lens` 객체 혹은 (x, y, r) 튜플).
- `RadialProfiler.extract_profile`: 입력 이미지와 검출된 렌즈 정보 -> 극좌표로 변환된 `RadialProfile` 객체 반환 (반지름 방향으로 L*, a*, b* 값 리스트).
- `ZoneSegmenter.segment`: `RadialProfile` -> `Zone` 리스트 반환. Zone에는 각 구간의 범위(반경)와 대표 색상값, ΔE 기준 등이 포함될 수 있습니다.
- `ColorEvaluator.evaluate`: `Zone` 리스트 + SKU 기준값 -> `InspectionResult` 반환. InspectionResult에는 overall 판정 (OK/NG), overall ΔE, 각 Zone별 ΔE 및 판정 등이 담깁니다.

### 2.3 process_batch() 메서드 (배치 처리)
여러 이미지를 처리할 때의 동작:
1. 입력으로 파일 경로 리스트와 SKU를 받습니다. 필요에 따라 결과를 CSV로 저장하는 경로(`output_csv`)를 받을 수 있습니다.
2. 각 이미지에 대해 `process()`를 호출하고 결과 리스트를 모읍니다.
3. `continue_on_error=True`인 경우, 개별 이미지 처리 중 오류가 발생해도 해당 오류를 로그하고 다음 이미지로 넘어갑니다. (오류 발생 이미지는 결과 리스트에 None 또는 에러 표시를 넣거나, PipelineError를 던지지 않고 내부 처리)
4. 마지막에 `output_csv` 경로가 주어졌다면 결과 리스트를 CSV로 직렬화하여 저장합니다.

```python
def process_batch(self, image_paths: List[str], sku: str, output_csv=None, continue_on_error=True) -> List[InspectionResult]:
    results = []
    for path in image_paths:
        try:
            res = self.process(path, sku)
            results.append(res)
        except Exception as e:
            logging.error(f"Error processing {path}: {e}")
            if not continue_on_error:
                raise
            # 오류 발생한 경우도 리스트 길이를 맞추기 위해 None 추가 또는 특별 객체 추가
            results.append(PipelineErrorResult(path, str(e)))
    if output_csv:
        self._save_results_csv(results, output_csv)
    return results
```
(참고: 실제 구현에서는 `PipelineErrorResult`와 같은 구조를 정의해 오류 케이스를 표시했습니다.)

## 3. SKU 기준값 관리 및 적용
파이프라인은 `sku_config` 정보를 이용하여 판정을 수행합니다. `SKUConfigManager`에서 JSON 파일을 읽어 파싱한 딕셔너리를 파이프라인에 넘겨준다고 가정합니다.
- `ColorEvaluator.evaluate(zones, sku, sku_config)`: 여기서 `sku_config[sku]`의 정보를 참조하여 각 Zone의 ΔE를 기준값과 비교합니다. Zone 이름 또는 인덱스를 키로 하여 `sku_config[sku]["zones"]`에 접근하고 threshold를 가져오는 구조입니다.

SKU JSON 구조의 상세는 **SKU 관리 설계 문서**를 참고하고, 본 파이프라인 코드에서는 해당 구조를 알고 있다고 간주하여 구현합니다.
(만약 SKU 코드에 해당하는 설정이 없으면 예외 또는 오류 처리를 해야 하는데, 이는 8장 예외 상황 처리에서 다룹니다.)

## 4. 예외 처리 전략
파이프라인 실행 중 발생할 수 있는 주요 예외 상황과 처리 방식을 정의합니다.

### 4.1 렌즈 검출 실패
만약 렌즈 검출 단계에서 아무 것도 찾지 못한 경우:
- `process()`에서는 `PipelineError("Lens detection failed")` 예외를 발생시킵니다. 이 예외는 상위 호출자(CLI 또는 API)가 잡아서 사용자에게 “렌즈를 이미지에서 찾지 못했습니다” 등의 메시지를 주게 됩니다.
- `process_batch()`에서는 기본적으로 `continue_on_error=True` 이므로, 해당 이미지를 건너뛰고 다음으로 진행합니다. 이때 실패한 파일 경로와 에러 원인을 별도로 기록하거나 결과 리스트에 표시합니다.

### 4.2 Zone 분할 실패
일부 이미지에서 색상 변화가 미미하여 ZoneSegmenter가 변화를 못 찾을 수 있습니다. 이 경우:
- 최소한 1개 Zone (전체 렌즈 영역)을 반환하도록 설계합니다. 즉, 분할 실패 시엔 fallback으로 “Zone A = 전체”와 같이 처리하여 이후 단계가 진행되도록 합니다.
- 이 상황을 `InspectionResult`에 플래그로 남겨 품질 관리자가 인지할 수 있게 할 수도 있습니다 (예: `result.zones_auto_completed = True`).

### 4.3 SKU 기준값 없음
주어진 SKU 코드에 대한 기준 JSON을 찾지 못한 경우:
- 파이프라인 초기화 시 또는 `evaluate` 단계에서 체크하여, 없으면 `PipelineError("SKU config not found")`를 발생시킵니다. 이 역시 상위 레벨에서 처리하여 사용자에게 “해당 SKU의 기준값이 미등록”이라는 메시지를 표시하게 합니다.
- 개선 여지: 추후 SKUConfigManager와 연동하여, SKU가 없으면 자동 생성 프로세스를 트리거하거나, 사용자에게 JSON 파일 생성 방법을 안내할 수도 있습니다.

(이 밖에도 파일 I/O 오류, 이미지 파싱 오류 등 일반 예외는 Python 예외로 발생할 수 있으며, 공통 PipelineError로 래핑하거나 로그로만 남기는 등 일관되게 처리합니다.)

## 5. CLI 인터페이스 설계 (src/main.py)
파이프라인은 CLI를 통해 사용될 수 있도록 `main.py`에서 `argparse`로 인터페이스를 제공합니다. 주요 설계 포인트:

### 인자 구조
- `--image <경로>`: 단일 이미지 검사 모드 (mutually exclusive with --batch)
- `--batch <폴더경로>`: 배치 검사 모드 (지정 폴더 내 모든 이미지 처리)
- `--sku <SKU_ID>`: 필수 인자, 적용할 SKU 코드
- `--visualize`: (옵션) 이 플래그를 주면 검사 후 결과 시각화까지 수행 (Visualizer 모듈 활용)
- 그 외 `--output <경로>` 등 추가 옵션 가능 (예: CSV 저장 경로 지정)

### 예시 사용법
```bash
# 단일 이미지 검사 예시
python src/main.py --image data/raw_images/OK_001.jpg --sku SKU001

# 배치 검사 예시
python src/main.py --batch data/raw_images/ --sku SKU001 --output results/batch_output.csv
```

### 출력 형식
콘솔에 한 줄 요약 출력 + 결과 저장.
- 단일 이미지의 경우 JSON 결과 경로를 출력하거나 JSON 내용을 바로 출력할 수도 있습니다. (현재 구현은 JSON 파일 경로만 보여줌)
- 배치의 경우 CSV 경로 출력.

### 종료 코드
프로세스 종료 시 반환값을 정의하여 상위 스크립트가 성공/실패를 인지하도록 합니다:
- `0`: 정상 실행 (및 모든 이미지 OK인 경우)
- `1`: 일반 에러 (입력 파일 없음 등 사용자 잘못)
- `2`: 검사 수행 중 오류 발생 (예: 렌즈 검출 실패, 모듈 예외 등). 배치의 경우 일부 실패가 있어도 `continue_on_error`이면 2 대신 0을 쓸지 정해야 하나, 현재 설계에서는 한 장이라도 실패하면 2로 처리.

### 5.1 단일 검사 CLI 흐름
`main.py` 내부 논리는 다음과 같습니다:
```python
if args.image:
    pipeline = InspectionPipeline(sku_config_manager.get_config(args.sku))
    try:
        result = pipeline.process(args.image, args.sku)
        print(f"[OK] {args.image} -> ΔE={result.overall_delta_e:.2f}, Result={result.judgment}")
        save_path = save_result_json(result)
        print(f"Result saved: {save_path}")
        sys.exit(0 if result.judgment == 'OK' else 0)  # 모두 OK여도 0
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(2)
```
(실제 구현과 다를 수 있지만 개념적으로 이렇게 동작하도록 설계)

출력 예 (콘솔):
```
[OK] data/raw_images/OK_001.jpg -> ΔE=2.35, Result=OK
Result saved: results/20251212_103000/result.json
```
JSON 결과 예 (`result.json`):
```json
{
  "image_path": "data/raw_images/OK_001.jpg",
  "sku": "SKU001",
  "judgment": "OK",
  "overall_delta_e": 2.35,
  "confidence": 0.87,
  "processing_time_ms": 234,
  "zones": [
    {"name": "A", "delta_e": 1.2, "judgment": "OK"},
    {"name": "B", "delta_e": 3.5, "judgment": "NG", "ng_reason": "DeltaE > threshold"},
    ...
  ]
}
```
(예시는 구조를 보여주기 위한 것으로, 실제 필드명과 구성은 구현 시 결정됩니다.)

### 5.2 배치 검사 CLI 흐름
배치 모드의 경우:
```python
elif args.batch:
    images = load_images_from_folder(args.batch)
    pipeline = InspectionPipeline(...); results = pipeline.process_batch(images, args.sku, output_csv=args.output)
    num_ok = sum(1 for r in results if r.judgment == 'OK')
    num_ng = len(results) - num_ok
    print(f"Processed {len(results)} images -> OK: {num_ok}, NG: {num_ng}")
    if args.output:
        print(f"CSV saved: {args.output}")
    sys.exit(0 if num_ng == 0 else 0)  # NG가 있어도 프로세스는 성공적으로 완료했으므로 0 (또는 정책에 따라 0/2 결정 가능)
```
CSV 결과 예시 (배치 요약):
```csv
image_path,sku,judgment,overall_delta_e,confidence,processing_time_ms,ng_reasons
data/raw_images/OK_001.jpg,SKU001,OK,2.35,0.87,234,
data/raw_images/NG_002.jpg,SKU001,NG,5.12,0.90,250,"Zone B high ΔE"
...
```
CSV에는 이미지별 요약 행이 포함됩니다.

### 5.3 기타 옵션
- `--visualize` 플래그가 있으면, InspectionPipeline 수행 후 **Visualizer** 모듈을 호출하여 결과 그래프/이미지를 생성합니다. 이때 CLI에서 추가 출력(예: “visualization saved at ...”)을 하도록 설계합니다. (Phase 2에서 구현 예정)
- 추후 `--config` 등으로 config 파일 경로를 지정하거나 `--no-error-stop` 등의 옵션을 추가할 수 있습니다.

## 6. 중간 결과 저장 (디버깅용)
`InspectionPipeline.save_intermediates=True` 설정 시 `_save_intermediates()` 메서드가 중간 산출물을 파일로 남깁니다. 개발 및 디버깅에 유용합니다.

### 6.1 저장 구조
중간 결과 저장이 활성화되면, `process()` 호출 시 지정된 `save_dir` 경로 아래에 다음 파일들을 저장하도록 구현했습니다:
- `polar_image.png`: 극좌표로 변환된 이미지 (RadialProfiler 결과)
- `profile_data.json`: 반경별 LAB 값 목록 (RadialProfile 객체 직렬화)
- `zones.json`: ZoneSegmenter 결과 (경계 리스트 및 구간 정보)
- `lens_detection.json`: 검출된 렌즈 중심/반경 정보
- `metadata.json`: 기타 메타 정보 (처리 시간, SKU, timestamp 등)

이런 파일들은 개발자가 결과를 분석하거나, Visualizer 개발 시 활용하기 위함입니다.
`metadata.json` 예시:
```json
{
  "lens_detection": { "cx": 512, "cy": 512, "radius": 500 },
  "zones_detected": 3,
  "profile_length": 360,
  "timestamp": "2025-12-12T10:30:00"
}
```

### 6.2 활용
중간 파일들은 실제 운영 모드에서는 생성되지 않으며(성능 영향을 주므로), 개발 단계에서 알고리즘 동작을 검증하는 용도로 사용합니다. 필요 없어진다면 `save_intermediates=False`로 두고 사용하면 됩니다.

## 7. 성능 요구사항
시스템은 실시간 또는 준실시간으로 동작해야 하므로, 파이프라인에 대한 성능 목표를 설정합니다:

### 7.1 처리 속도
- **단일 이미지:** 평균 < **200ms** 내 처리 (목표). 현재 프로토타입은 ~90ms 수준으로 목표를 이미 달성했으며, 추가 여유를 확보할 계획입니다.
- **배치 모드:** 1장당 100ms 이내, 1초에 최소 10장 이상 처리 (현재 6.5~7ms/장 수준으로 매우 양호). 향후 I/O 병렬화로 더 개선 가능.

### 7.2 메모리 사용
- **단일 이미지:** 프로세스 전체 메모리 사용량 100MB 이내.
- **배치 처리:** 10장 처리 기준 500MB 이내. (현재는 이미지 처리 완료 시 메모리를 반환하고 있어 문제 없음)

### 7.3 정확도 지표
- **렌즈 검출 성공률:** ≥ 95% (실패시 수동 개입 필요하므로 높게 설정)
- **오탐률 (False NG):** ≤ 5% (정상 렌즈를 NG로 판정하는 경우).
- ΔE 정확도 자체는 CIEDE2000 공식에 따르므로 별도 지표 없음. 대신 시각적 검증을 통해 품질팀이 만족할 수준인지 주기적으로 점검.
*(실제 성능 측정 결과는 [성능 분석 보고서](../design/PERFORMANCE_ANALYSIS.md)에서 확인합니다.)*

## 8. 확장성 고려사항
향후 요구사항 변경이나 기능 추가에 대비한 설계 요소들입니다:

### 8.1 향후 추가 기능
- **다중 SKU 동시 비교:** 한 번에 여러 SKU의 기준과 비교하여 가장 가까운 SKU 찾기 등 (현재는 단일 SKU 기준으로만 평가).
- **다양한 조명 조건 보정:** 이미지 입력 시 조명 색온도나 밝기 차이를 보정하는 전처리 단계 추가.
- **UI 연계:** 현재 CLI/웹UI 외에 GUI 데스크탑 앱(PyQt) 연계 가능성. 이를 위해 Pipeline을 싱글톤이 아닌 인스턴스로 설계하고 결과를 객체로 유지.

### 8.2 플러그인 아키텍처
파이프라인에 플러그인 모듈을 쉽게 붙일 수 있도록 유연성을 줍니다. 예를 들어:
- `VisualizerPlugin`: 파이프라인 완료 후 결과를 받아 시각화 수행.
- `LoggerPlugin`: 각 단계 결과를 실시간 로깅/모니터링.

파이프라인 클래스에 `add_plugin(plugin)` 메서드를 제공하여, plugin은 사전에 정의된 훅(hook)을 구현하도록 할 수 있습니다:
```python
pipeline = InspectionPipeline(...)
pipeline.add_plugin(VisualizerPlugin())
pipeline.add_plugin(LoggerPlugin())
```
이렇게 하면 특정 이벤트(예: `after_process`)에 등록된 플러그인들의 메서드가 호출되도록 구현할 수 있습니다. 세부 구현은 추후 설계. (현재 Visualizer의 CLI 통합은 `--visualize` 옵션으로 간단히 처리하지만, 구조적으로는 위와 같이 개선할 여지가 있습니다.)

## 9. 테스트 전략
### 9.1 단위 테스트
각 모듈(ImageLoader 등)의 단위 테스트가 이미 개별적으로 작성되어 있으며, 현재 100+개 테스트 케이스가 모두 통과한 상태입니다. 파이프라인 모듈 자체에 대한 단위 테스트도 중요합니다:
- 정상 입력에 대해 `process()`가 `InspectionResult`를 올바르게 반환하는지 확인.
- 극단 상황 (렌즈 없는 이미지 등)에 대해 예외를 제대로 발생시키는지 확인.
- Batch 모드에서 continue_on_error 동작 확인 등.

### 9.2 통합 테스트 (tests/test_pipeline.py)
시스템 통합 테스트로, 실제 여러 모듈이 연결된 상태에서의 동작을 검증합니다:
```python
def test_pipeline_ok_case():
    pipeline = InspectionPipeline(sku_config=load_sku("SKU001"))
    res = pipeline.process("tests/data/OK_example.jpg", "SKU001")
    assert res.judgment == "OK"
    assert res.overall_delta_e < res.sku_threshold
```
또한 배치 처리에 대한 테스트, Visualizer 옵션 켜고 실행해보기 등도 포함됩니다.

### 9.3 수동 검증
자동화된 테스트 외에, 시각적 검증이 필요한 부분(예: Overlay 그림이 제대로 표시되는지 등)은 QA를 통해 수동 테스트합니다. 특히 “분석 우선” 모드가 UI에서 잘 작동하는지, 경계 후보 표시가 정확한지 등을 수시로 확인하고 있습니다.

## 10. 구현 단계별 우선순위
개발 편의를 위해 파이프라인 및 관련 기능 구현을 몇 단계로 나누었습니다:
- **Phase 1 (현재 단계):** InspectionPipeline 기본 동작 구현 및 단일 이미지 처리 완료. (완료: 핵심 클래스 구현 및 단위 테스트 통과)
- **Phase 2 (다음):** `process_batch()` 구현, CSV 출력 기능, 중간 결과 저장 기능 추가. (진행 중: 배치 처리 모듈 개발)
- **Phase 3 (이후):** 성능 최적화(벡터화 적용, 멀티스레딩 등), 병렬 처리 지원, GUI 연동 등.

현재 Phase 1을 마쳤으며, Phase 2 작업이 진행 중입니다. 상세 일정은 별도 프로젝트 관리 문서 참고.

## 11. 참고 자료
- **핵심 알고리즘 상세:** 상세 구현 계획서 §3.2 – 각 모듈별 알고리즘 설명.
- **SKU 데이터 구조:** [SKU 관리 설계 문서](SKU_MANAGEMENT_DESIGN.md) §2 – SKU JSON 스키마 정의 및 예시.
- **개발 가이드:** [DEVELOPMENT_GUIDE.md](../development/DEVELOPMENT_GUIDE.md) – 개발 환경 설정 및 디렉토리 구조 등 (파이프라인 소스코드 위치 등 참고 가능).
- **성능 분석 보고서:** [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md) – 실제 성능 수치와 최적화 방안. 특히 RadialProfiler 최적화 방향 관련.

*(이상 파이프라인 설계 문서 끝)*
