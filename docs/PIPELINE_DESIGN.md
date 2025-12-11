# 통합 파이프라인 설계 문서

> **작성일**: 2025-12-11
> **목적**: 5개 핵심 모듈을 연결하는 엔드투엔드 검사 파이프라인 설계
> **버전**: 1.0

---

## 1. 개요

### 1.1 목표
콘택트렌즈 이미지를 입력받아 자동으로 색상 품질을 검사하고 OK/NG 판정을 반환하는 통합 파이프라인 구현.

### 1.2 파이프라인 구조
```
Input: 이미지 파일 경로 + SKU 코드
    ↓
[1] ImageLoader - 이미지 로드 및 전처리
    ↓
[2] LensDetector - 렌즈 중심/반경 검출
    ↓
[3] RadialProfiler - 극좌표 변환 및 색상 프로파일 추출
    ↓
[4] ZoneSegmenter - 색상 Zone 자동 분할
    ↓
[5] ColorEvaluator - ΔE 계산 및 OK/NG 판정
    ↓
Output: InspectionResult (판정, ΔE 값, Zone별 결과)
```

---

## 2. InspectionPipeline 클래스 설계

### 2.1 클래스 인터페이스

```python
class InspectionPipeline:
    """
    엔드투엔드 렌즈 색상 검사 파이프라인.

    5개 핵심 모듈을 순차적으로 실행하여 최종 판정 결과를 생성.
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
        """
        파이프라인 초기화.

        Args:
            sku_config: SKU별 기준값 설정 (dict)
            image_config: ImageLoader 설정 (옵션)
            detector_config: LensDetector 설정 (옵션)
            profiler_config: RadialProfiler 설정 (옵션)
            segmenter_config: ZoneSegmenter 설정 (옵션)
            save_intermediates: 중간 결과 저장 여부 (디버깅용)
        """
        pass

    def process(
        self,
        image_path: str,
        sku: str,
        save_dir: Optional[Path] = None
    ) -> InspectionResult:
        """
        단일 이미지 처리.

        Args:
            image_path: 입력 이미지 경로
            sku: SKU 코드 (예: 'SKU001')
            save_dir: 중간 결과 저장 디렉토리 (옵션)

        Returns:
            InspectionResult: 검사 결과

        Raises:
            PipelineError: 파이프라인 실행 중 오류 발생 시
        """
        pass

    def process_batch(
        self,
        image_paths: List[str],
        sku: str,
        output_csv: Optional[Path] = None,
        continue_on_error: bool = True
    ) -> List[InspectionResult]:
        """
        배치 처리.

        Args:
            image_paths: 입력 이미지 경로 리스트
            sku: SKU 코드
            output_csv: 결과 CSV 저장 경로 (옵션)
            continue_on_error: 오류 발생 시 계속 진행 여부

        Returns:
            List[InspectionResult]: 검사 결과 리스트
        """
        pass
```

---

## 3. SKU 기준값 JSON 포맷

### 3.1 파일 구조
```
config/sku_db/
├── SKU001.json
├── SKU002.json
└── ...
```

### 3.2 JSON 스키마
```json
{
  "sku_code": "SKU001",
  "description": "Blue colored contact lens - 3 zones",
  "default_threshold": 3.0,
  "zones": {
    "A": {
      "L": 45.0,
      "a": 10.0,
      "b": -40.0,
      "threshold": 4.0,
      "description": "Outer blue zone"
    },
    "B": {
      "L": 70.0,
      "a": -5.0,
      "b": 60.0,
      "threshold": 3.5,
      "description": "Middle transition zone"
    },
    "C": {
      "L": 85.0,
      "a": 0.0,
      "b": 5.0,
      "threshold": 3.0,
      "description": "Center clear zone"
    }
  },
  "metadata": {
    "created_at": "2025-12-11",
    "baseline_samples": 30,
    "last_updated": "2025-12-11"
  }
}
```

### 3.3 필드 설명
- `sku_code`: SKU 식별자 (필수)
- `description`: SKU 설명 (옵션)
- `default_threshold`: 기본 ΔE 허용치 (필수, 일반적으로 3.0)
- `zones`: Zone별 기준값 (필수)
  - `L`, `a`, `b`: 기준 LAB 값
  - `threshold`: Zone별 허용치 (없으면 default_threshold 사용)
  - `description`: Zone 설명 (옵션)
- `metadata`: 메타데이터 (옵션)

---

## 4. 파이프라인 실행 흐름

### 4.1 process() 메서드 상세 흐름

```python
def process(self, image_path: str, sku: str, save_dir=None) -> InspectionResult:
    # 1. 이미지 로드 및 전처리
    image = self.image_loader.load_from_file(image_path)
    processed_image = self.image_loader.preprocess(image)

    # 2. 렌즈 검출
    lens_detection = self.lens_detector.detect(processed_image)
    if lens_detection is None:
        raise PipelineError("Lens detection failed")

    # 3. 극좌표 변환 및 프로파일 추출
    radial_profile = self.radial_profiler.extract_profile(
        processed_image,
        lens_detection
    )

    # 4. Zone 분할
    zones = self.zone_segmenter.segment(radial_profile)

    # 5. 색상 평가 및 판정
    inspection_result = self.color_evaluator.evaluate(
        zones,
        sku,
        self.sku_config
    )

    # 6. 중간 결과 저장 (옵션)
    if self.save_intermediates and save_dir:
        self._save_intermediates(save_dir, {
            'processed_image': processed_image,
            'lens_detection': lens_detection,
            'radial_profile': radial_profile,
            'zones': zones
        })

    return inspection_result
```

### 4.2 에러 처리

각 단계별 예외 처리:
```python
try:
    # 각 단계 실행
except LensDetectionError as e:
    logger.error(f"Lens detection failed: {e}")
    raise PipelineError(f"Pipeline failed at lens detection: {e}")
except ZoneSegmentationError as e:
    logger.error(f"Zone segmentation failed: {e}")
    raise PipelineError(f"Pipeline failed at zone segmentation: {e}")
except ColorEvaluationError as e:
    logger.error(f"Color evaluation failed: {e}")
    raise PipelineError(f"Pipeline failed at color evaluation: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise PipelineError(f"Pipeline failed: {e}")
```

---

## 5. CLI 인터페이스 설계 (main.py)

### 5.1 명령어 구조

#### 단일 이미지 처리
```bash
python src/main.py \
    --image data/raw_images/OK_001.jpg \
    --sku SKU001 \
    [--output results/OK_001_result.json] \
    [--save-intermediates] \
    [--debug]
```

#### 배치 처리
```bash
python src/main.py \
    --batch data/raw_images/ \
    --sku SKU001 \
    --output results/batch_results.csv \
    [--continue-on-error] \
    [--debug]
```

### 5.2 출력 형식

#### JSON 출력 (단일 이미지)
```json
{
  "image_path": "data/raw_images/OK_001.jpg",
  "sku": "SKU001",
  "timestamp": "2025-12-11T10:30:45",
  "judgment": "OK",
  "overall_delta_e": 2.35,
  "confidence": 0.87,
  "processing_time_ms": 234,
  "zone_results": [
    {
      "zone_name": "A",
      "measured_lab": [45.2, 10.3, -39.8],
      "target_lab": [45.0, 10.0, -40.0],
      "delta_e": 0.42,
      "threshold": 4.0,
      "is_ok": true
    }
  ],
  "ng_reasons": []
}
```

#### CSV 출력 (배치)
```csv
image_path,sku,judgment,overall_delta_e,confidence,processing_time_ms,ng_reasons
data/raw_images/OK_001.jpg,SKU001,OK,2.35,0.87,234,
data/raw_images/NG_001.jpg,SKU001,NG,8.42,0.45,251,"Zone A: ΔE=8.2 > 4.0"
```

### 5.3 종료 코드
- `0`: 성공 (모든 이미지 OK 또는 정상 처리 완료)
- `1`: 일반 오류 (파일 없음, 설정 오류 등)
- `2`: 파이프라인 실행 오류 (렌즈 검출 실패 등)

---

## 6. 중간 결과 저장 (디버깅용)

### 6.1 저장 구조
`save_intermediates=True` 시:
```
results/intermediates/{image_name}/
├── 01_original.jpg
├── 02_preprocessed.jpg
├── 03_lens_detection.jpg (중심/반경 오버레이)
├── 04_polar_image.jpg
├── 05_radial_profile.png (LAB 그래프)
├── 06_zones.jpg (Zone 경계선 오버레이)
└── metadata.json
```

### 6.2 metadata.json 예시
```json
{
  "lens_detection": {
    "center_x": 200.5,
    "center_y": 198.3,
    "radius": 95.2,
    "confidence": 0.95,
    "method": "hybrid"
  },
  "zones": [
    {
      "name": "A",
      "r_start": 1.0,
      "r_end": 0.7,
      "mean_lab": [45.2, 10.3, -39.8]
    }
  ]
}
```

---

## 7. 성능 요구사항

### 7.1 처리 속도
- **목표**: 평균 <200ms/장
- **허용**: 95-백분위수 <500ms/장

### 7.2 메모리
- **단일 이미지**: <100MB
- **배치 처리**: <500MB (순차 처리로 메모리 재사용)

### 7.3 정확도
- **검출률**: ≥95% (렌즈 검출 성공률)
- **오탐률**: ≤5% (OK를 NG로 잘못 판정)

---

## 8. 예외 상황 처리

### 8.1 렌즈 검출 실패
```python
if lens_detection is None or lens_detection.confidence < 0.5:
    return InspectionResult(
        sku=sku,
        timestamp=datetime.now(),
        judgment='ERROR',
        overall_delta_e=float('inf'),
        zone_results=[],
        ng_reasons=['Lens detection failed or low confidence'],
        confidence=0.0
    )
```

### 8.2 Zone 분할 실패
- 최소 1개 Zone으로 폴백 (전체 렌즈를 단일 Zone으로 처리)

### 8.3 SKU 기준값 없음
```python
if sku not in sku_database:
    raise PipelineError(f"SKU {sku} not found in database")
```

---

## 9. 확장성 고려사항

### 9.1 향후 추가 기능
- [ ] 다중 SKU 동시 비교
- [ ] 실시간 카메라 스트림 처리
- [ ] GPU 가속 (OpenCV CUDA)
- [ ] 병렬 배치 처리 (multiprocessing)

### 9.2 플러그인 아키텍처
추후 Visualizer, Logger 등을 플러그인으로 추가 가능:
```python
pipeline.add_plugin(VisualizerPlugin())
pipeline.add_plugin(LoggerPlugin(db_path='logs/inspections.db'))
```

---

## 10. 테스트 전략

### 10.1 단위 테스트
각 모듈은 이미 개별 테스트 완료 (103개 테스트 통과)

### 10.2 통합 테스트 (`tests/test_pipeline.py`)
```python
def test_pipeline_ok_case():
    """OK 이미지 처리 테스트"""
    pipeline = InspectionPipeline(sku_config)
    result = pipeline.process('data/raw_images/OK_001.jpg', 'SKU001')
    assert result.judgment == 'OK'
    assert result.overall_delta_e < 4.0

def test_pipeline_ng_case():
    """NG 이미지 처리 테스트"""
    pipeline = InspectionPipeline(sku_config)
    result = pipeline.process('data/raw_images/NG_001.jpg', 'SKU001')
    assert result.judgment == 'NG' or result.overall_delta_e > 4.0

def test_pipeline_batch():
    """배치 처리 테스트"""
    pipeline = InspectionPipeline(sku_config)
    results = pipeline.process_batch(
        ['OK_001.jpg', 'OK_002.jpg'],
        'SKU001'
    )
    assert len(results) == 2
```

---

## 11. 구현 우선순위

### Phase 1 (현재)
1. ✅ InspectionPipeline 클래스 기본 구조
2. ✅ process() 메서드 구현
3. ✅ SKU JSON 로드 기능
4. ✅ main.py CLI 기본 구현

### Phase 2 (다음)
1. ⏳ process_batch() 구현
2. ⏳ CSV 출력 기능
3. ⏳ 중간 결과 저장 기능
4. ⏳ 에러 처리 강화

### Phase 3 (향후)
1. ⏳ 성능 최적화
2. ⏳ 병렬 처리
3. ⏳ GUI 연동
4. ⏳ 실시간 스트림 처리

---

## 12. 참고 자료

- **핵심 모듈 문서**: `docs/DETAILED_IMPLEMENTATION_PLAN.md`
- **개발 가이드**: `docs/DEVELOPMENT_GUIDE.md`
- **작업 분담**: `WORK_ASSIGNMENT.md`, `DAY2_WORK_PLAN.md`

---

**문서 버전**: 1.0
**작성자**: Claude (AI Assistant)
**마지막 업데이트**: 2025-12-11
