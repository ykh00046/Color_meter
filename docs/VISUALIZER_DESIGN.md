# Inspection Visualizer - Design Document

> **작성일**: 2025-12-11
> **버전**: 1.0
> **목적**: 검사 결과 시각화 시스템 설계

---

## 1. 개요

### 1.1 배경
현재 시스템은 텍스트 기반 검사 결과만 제공합니다. 품질 관리자와 검사자가 다음을 필요로 합니다:
- Zone 검출 결과의 시각적 확인
- 색상 차이(ΔE) 분포의 직관적 파악
- 배치 처리 결과의 요약 통계
- 불량 원인의 시각적 분석

### 1.2 목표
1. **Zone Overlay**: 검출된 Zone과 판정 결과를 원본 이미지 위에 표시
2. **ΔE Heatmap**: 색상 차이 분포를 히트맵으로 시각화
3. **Comparison Chart**: 측정값과 기준값의 정량적 비교
4. **Dashboard**: 배치 처리 결과 요약 대시보드

### 1.3 범위
- InspectionVisualizer 클래스 구현
- CLI 통합 (--visualize 옵션)
- 파일 출력 지원 (PNG, PDF)
- Jupyter Notebook 통합

---

## 2. 아키텍처

### 2.1 클래스 다이어그램

```
InspectionVisualizer
├─ __init__(config: VisualizerConfig)
├─ visualize_zone_overlay(image, detection, zones, result) -> np.ndarray
├─ visualize_delta_e_heatmap(profile, zones, sku_config) -> Tuple[np.ndarray, np.ndarray]
├─ visualize_comparison(zones, result) -> plt.Figure
├─ visualize_dashboard(results) -> plt.Figure
├─ save_visualization(image, output_path, format)
└─ _draw_zone_boundary(image, zone, color)
   _draw_zone_label(image, zone, text, color)
   _create_polar_heatmap(profile, zones)
   _create_cartesian_heatmap(profile, zones)
   _get_zone_color(zone_result)
```

### 2.2 데이터 흐름

```
Pipeline 처리
    ↓
InspectionResult
    ↓
Visualizer
    ├─ Zone Overlay → PNG/PDF
    ├─ ΔE Heatmap → PNG/PDF
    ├─ Comparison → PNG/PDF
    └─ Dashboard → PNG/PDF
```

---

## 3. 시각화 타입별 명세

### 3.1 Zone Overlay

**목적**: Zone 검출 결과 및 판정 상태 시각적 확인

**입력**:
- `image`: 원본 이미지 (BGR, np.ndarray)
- `lens_detection`: 렌즈 검출 결과 (LensDetection)
- `zones`: Zone 리스트 (List[Zone])
- `inspection_result`: 검사 결과 (InspectionResult)

**출력**: 오버레이된 이미지 (BGR, np.ndarray)

**시각화 요소**:
1. **렌즈 원**: 검출된 렌즈 외곽선 (회색, 점선)
2. **렌즈 중심**: 중심점 표시 (십자가, 빨간색)
3. **Zone 경계선**: Zone별 경계선
   - OK Zone: 녹색 (0, 255, 0), 두께 2px
   - NG Zone: 빨간색 (0, 0, 255), 두께 3px
4. **Zone 레이블**: Zone 이름 + ΔE 값
   - 위치: Zone 중간 반지름 위치
   - 배경: 반투명 사각형 (가독성)
   - 텍스트: Zone명, ΔE, OK/NG
5. **전체 판정**: 좌상단에 크게 표시
   - OK: 녹색 배경
   - NG: 빨간색 배경

**예시**:
```
┌─────────────────────────┐
│ [OK] ΔE=0.5            │ ← 전체 판정
│                         │
│         ⊕               │ ← 렌즈 중심
│      ╱     ╲            │
│    ╱  Zone A ╲          │ ← Zone 경계선 + 레이블
│   │  ΔE=0.3   │         │
│    ╲  [OK]   ╱          │
│      ╲     ╱            │
│         ○               │ ← 렌즈 외곽선
└─────────────────────────┘
```

### 3.2 ΔE Heatmap

**목적**: 색상 차이 분포의 공간적 시각화

**입력**:
- `radial_profile`: 극좌표 프로파일 (RadialProfile)
- `zones`: Zone 리스트 (List[Zone])
- `sku_config`: SKU 기준값 (Dict)

**출력**: (polar_heatmap, cartesian_heatmap) (각각 np.ndarray)

**3.2.1 극좌표 히트맵 (Polar Heatmap)**

**계산 방법**:
1. 각 (r, θ) 위치에서 ΔE 계산
2. Zone별 기준 LAB와 비교
3. 컬러맵 적용: RdYlGn_r (Red=높음, Green=낮음)

**시각화 요소**:
- **원형 표시**: 렌즈 형태 유지
- **컬러바**: ΔE 범위 (0~10)
- **Zone 경계**: 검은색 점선
- **기준값 초과 영역**: 빨간색 하이라이트

**3.2.2 직교좌표 히트맵 (Cartesian Heatmap)**

**계산 방법**:
1. 극좌표 → 직교좌표 변환 (unwrap)
2. 각도별 ΔE 프로파일 전개
3. 2D 히트맵 생성

**시각화 요소**:
- **X축**: 각도 (0°~360°)
- **Y축**: 반지름 (0~1, normalized)
- **컬러맵**: RdYlGn_r
- **Zone 경계**: 수평선

### 3.3 Comparison Chart

**목적**: 측정값과 기준값의 정량적 비교

**입력**:
- `zones`: Zone 리스트 (List[Zone])
- `inspection_result`: 검사 결과 (InspectionResult)

**출력**: matplotlib Figure

**차트 구성 (2개 서브플롯)**:

**3.3.1 LAB 비교 (막대 그래프)**
```
    L*  a*  b*
    ║   ║   ║
100 ║▓▓▓║   ║
 75 ║▒▒▒║▓▓▓║
 50 ║░░░║▒▒▒║▓▓▓
 25 ║   ║░░░║▒▒▒
  0 ╚═══╩═══╩═══
    Zone A

▓ = 측정값 (Measured)
▒ = 기준값 (Target)
```

- **X축**: Zone 이름
- **Y축**: LAB 값
- **그룹**: L, a, b 각각 막대
- **색상**: 측정값=파란색, 기준값=회색

**3.3.2 ΔE vs Threshold (라인 차트)**
```
ΔE
10 ┤           ●  (NG)
 8 ┤       ●
 6 ┤   ●
 4 ├───────────────── Threshold
 2 ┤ ●     (OK)
 0 └─────────────────
   A   B   C   Zone
```

- **X축**: Zone 이름
- **Y축**: ΔE 값
- **포인트**: 측정 ΔE (원)
- **선**: Threshold (점선, 빨간색)
- **영역**: Pass(녹색), Fail(빨간색) 배경

### 3.4 Dashboard (배치 처리 요약)

**목적**: 배치 처리 결과의 종합적 시각화

**입력**:
- `results`: 검사 결과 리스트 (List[InspectionResult])

**출력**: matplotlib Figure (4개 서브플롯)

**레이아웃 (2x2 grid)**:
```
┌─────────────┬─────────────┐
│ 1. 판정 비율│ 2. ΔE 분포  │
│   (파이)    │  (박스플롯) │
├─────────────┼─────────────┤
│ 3. Zone NG  │ 4. 처리속도 │
│   빈도(히트맵)│ (타임라인)  │
└─────────────┴─────────────┘
```

**4.1 판정 비율 (파이 차트)**
- OK: 녹색
- NG: 빨간색
- 백분율 표시

**4.2 ΔE 분포 (박스 플롯)**
- **X축**: SKU 코드
- **Y축**: ΔE 값
- **박스**: Q1, Median, Q3
- **Whiskers**: Min, Max
- **Threshold 라인**: 빨간색 점선

**4.3 Zone별 NG 빈도 (히트맵)**
- **행**: SKU 코드
- **열**: Zone 이름
- **값**: NG 발생 횟수
- **컬러맵**: Reds (진할수록 많음)

**4.4 처리 속도 타임라인**
- **X축**: 이미지 번호
- **Y축**: 처리 시간 (ms)
- **평균선**: 점선
- **목표선**: 빨간색 (<200ms)

---

## 4. VisualizerConfig

```python
@dataclass
class VisualizerConfig:
    """Visualizer 설정"""

    # Zone overlay
    zone_line_thickness: int = 2
    zone_color_ok: Tuple[int, int, int] = (0, 255, 0)  # BGR: Green
    zone_color_ng: Tuple[int, int, int] = (0, 0, 255)  # BGR: Red
    zone_label_font_scale: float = 0.6
    zone_label_thickness: int = 2
    show_zone_labels: bool = True
    show_lens_circle: bool = True
    show_center_mark: bool = True

    # Heatmap
    heatmap_colormap: str = "RdYlGn_r"  # Red=high, Green=low
    heatmap_resolution: int = 360  # Angular resolution
    show_colorbar: bool = True
    delta_e_range: Tuple[float, float] = (0.0, 10.0)

    # Comparison chart
    comparison_figure_size: Tuple[int, int] = (12, 6)
    comparison_dpi: int = 100
    show_threshold_line: bool = True
    show_pass_fail_zones: bool = True

    # Dashboard
    dashboard_figure_size: Tuple[int, int] = (14, 10)
    dashboard_dpi: int = 100

    # Output
    output_format: str = "png"  # "png", "pdf", "both"
    output_quality: int = 95  # JPEG quality (1-100)
```

---

## 5. CLI 인터페이스

### 5.1 단일 이미지 시각화

```bash
# 기본 (모든 타입)
python -m src.main inspect --image data/raw_images/OK_001.jpg --sku SKU001 \
  --visualize --output results/OK_001_viz.png

# 특정 타입만
python -m src.main inspect --image data/raw_images/OK_001.jpg --sku SKU001 \
  --visualize overlay,heatmap --output results/OK_001_overlay.png

# PDF 출력
python -m src.main inspect --image data/raw_images/OK_001.jpg --sku SKU001 \
  --visualize --output results/OK_001_report.pdf --format pdf
```

### 5.2 배치 시각화

```bash
# 배치 처리 + 각 이미지 시각화
python -m src.main batch --batch data/raw_images/ --sku SKU001 \
  --visualize --output-dir results/visualizations/

# 출력 구조:
# results/visualizations/
#   ├── OK_001_overlay.png
#   ├── OK_002_overlay.png
#   ├── ...
#   └── dashboard.png (요약)

# Dashboard만
python -m src.main batch --batch data/raw_images/ --sku SKU001 \
  --visualize dashboard --output results/dashboard.png
```

### 5.3 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--visualize [types]` | 시각화 타입 (overlay,heatmap,comparison,dashboard) | 모두 |
| `--output PATH` | 출력 파일 경로 | None (화면 표시) |
| `--output-dir PATH` | 배치 출력 디렉토리 | None |
| `--format FORMAT` | 출력 포맷 (png, pdf) | png |
| `--no-colorbar` | 컬러바 숨기기 | False |
| `--dpi DPI` | 해상도 | 100 |

---

## 6. API 사용 예제

### 6.1 Python 스크립트

```python
from src.pipeline import InspectionPipeline
from src.visualizer import InspectionVisualizer, VisualizerConfig
from src.utils.file_io import read_json
from pathlib import Path

# Setup
sku_config = read_json("config/sku_db/SKU001.json")
pipeline = InspectionPipeline(sku_config)
visualizer = InspectionVisualizer()

# Process image
result = pipeline.process("data/raw_images/OK_001.jpg", "SKU001")

# Visualize
overlay = visualizer.visualize_zone_overlay(
    image=pipeline.image_loader.load_from_file("data/raw_images/OK_001.jpg"),
    lens_detection=result.lens_detection,  # Added to result
    zones=result.zones,
    inspection_result=result
)

# Save
visualizer.save_visualization(overlay, "results/overlay.png")
```

### 6.2 Jupyter Notebook

```python
from src.visualizer import InspectionVisualizer
import matplotlib.pyplot as plt

visualizer = InspectionVisualizer()

# Zone overlay
overlay = visualizer.visualize_zone_overlay(...)
plt.imshow(overlay)
plt.axis('off')
plt.show()

# Comparison chart
fig = visualizer.visualize_comparison(zones, result)
plt.show()
```

---

## 7. 성능 요구사항

### 7.1 처리 시간

| 시각화 타입 | 목표 | 최대 허용 |
|------------|------|----------|
| Zone Overlay | <50ms | <100ms |
| ΔE Heatmap | <100ms | <200ms |
| Comparison | <50ms | <100ms |
| Dashboard | <200ms | <500ms |

### 7.2 메모리

- **최대 메모리 사용**: <500MB (단일 이미지)
- **배치 처리**: <2GB (100장)

### 7.3 출력 파일 크기

- **PNG**: <2MB (1920x1080)
- **PDF**: <5MB (다중 페이지)

---

## 8. 에러 처리

### 8.1 에러 타입

| 에러 | 발생 조건 | 처리 방법 |
|------|----------|-----------|
| `VisualizationError` | 시각화 실패 | 원본 이미지 반환 + 경고 |
| `InvalidConfigError` | 잘못된 설정 | 기본값 사용 |
| `OutputError` | 파일 저장 실패 | 재시도 + 대체 경로 |

### 8.2 Fallback 전략

1. **컬러맵 로드 실패**: 기본 컬러맵 사용 (viridis)
2. **폰트 로드 실패**: OpenCV 기본 폰트 사용
3. **메모리 부족**: 이미지 다운샘플링
4. **출력 경로 없음**: 임시 디렉토리 사용

---

## 9. 확장성

### 9.1 향후 추가 기능

1. **3D 시각화**: Zone 높이를 ΔE로 표현
2. **애니메이션**: 배치 처리 과정 동영상
3. **인터랙티브**: Plotly 기반 줌/팬
4. **템플릿**: 커스텀 리포트 템플릿
5. **AR 오버레이**: 실시간 카메라 피드

### 9.2 플러그인 시스템

```python
class CustomVisualizer(InspectionVisualizer):
    def visualize_custom(self, ...):
        # 사용자 정의 시각화
        pass
```

---

## 10. 테스트 전략

### 10.1 단위 테스트 (12개)

1. `test_visualizer_initialization()` - 초기화
2. `test_zone_overlay_ok_image()` - OK 이미지 오버레이
3. `test_zone_overlay_ng_image()` - NG 이미지 오버레이
4. `test_delta_e_heatmap_polar()` - 극좌표 히트맵
5. `test_delta_e_heatmap_cartesian()` - 직교좌표 히트맵
6. `test_comparison_chart()` - 비교 차트
7. `test_dashboard_single_sku()` - 단일 SKU 대시보드
8. `test_dashboard_multi_sku()` - 다중 SKU 대시보드
9. `test_save_png()` - PNG 저장
10. `test_save_pdf()` - PDF 저장
11. `test_custom_config()` - 커스텀 설정
12. `test_error_handling()` - 에러 처리

### 10.2 통합 테스트

- CLI 명령어 테스트
- Jupyter Notebook 실행 테스트
- 성능 벤치마크 테스트

### 10.3 시각적 회귀 테스트

- 기준 이미지와 픽셀 단위 비교
- 허용 오차: 1% (압축 손실 고려)

---

## 11. 보안 고려사항

### 11.1 파일 출력

- **경로 검증**: Path traversal 방지
- **파일 크기 제한**: 최대 10MB
- **포맷 검증**: 허용된 포맷만 (png, pdf)

### 11.2 메모리 관리

- **이미지 크기 제한**: 최대 4K (3840x2160)
- **배치 크기 제한**: 최대 1000장
- **메모리 누수 방지**: 명시적 자원 해제

---

## 12. 접근성

### 12.1 색맹 지원

- **대체 컬러맵**: viridis, plasma (색맹 친화적)
- **패턴 추가**: 점선, 해칭
- **텍스트 레이블**: 색상만 의존하지 않음

### 12.2 고대비 모드

- **배경**: 흰색 또는 검은색
- **텍스트**: 자동 대비 조절
- **선 두께**: 2배 증가 옵션

---

## 13. 참고 자료

- [Matplotlib Documentation](https://matplotlib.org/stable/index.html)
- [OpenCV Drawing Functions](https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html)
- [Colormap Reference](https://matplotlib.org/stable/gallery/color/colormap_reference.html)
- CIEDE2000 ΔE 논문: Sharma et al. (2005)

---

**작성자**: Claude (AI Assistant)
**버전**: 1.0
**최종 수정**: 2025-12-11
