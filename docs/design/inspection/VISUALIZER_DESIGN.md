# 🎨 검사 결과 시각화 모듈 설계 (InspectionVisualizer)

> **작성:** 2025-12-11
> **목적:** 검사 결과 데이터를 다양한 형태로 시각화하여 품질 관리자 및 개발자가 직관적으로 이해하고 **검증 작업**에 활용할 수 있는 도구를 설계합니다.

## 1. 개요
### 1.1 배경
현재 시스템은 텍스트 기반으로 OK/NG 결과와 일부 수치만 제공합니다. 그러나 현장 사용자(품질 관리자 등)는 다음 사항을 원하고 있습니다:
- **Zone 결과의 시각적 확인:** 각 Zone이 어디인지, 해당 구역이 양호한지 불량인지 한눈에 보고 싶습니다.
- **ΔE 분포 파악:** 렌즈 전반에 걸쳐 색차(ΔE)가 어떻게 변화하는지 히트맵 등으로 보고 싶습니다.
- **배치 결과 요약:** 다수 이미지 검사 시, 전체적인 불량율이나 경향을 대시보드 형태로 확인하고 싶습니다.
- **원인 분석 지원:** NG 발생 시 어떤 Zone 때문에, 어떤 패턴의 변화 때문에 발생했는지 시각적으로 분석하고 싶습니다.

### 1.2 목표
이에 따라 **InspectionVisualizer** 모듈의 목표를 설정합니다:
1. **Zone Overlay:** 원본 검사 이미지 위에 검출된 렌즈 영역과 Zone 경계를 표시, 그리고 각 Zone의 ΔE 및 판정 결과(OK/NG)를 시각적으로 표시.
2. **ΔE Heatmap:** 렌즈 전체의 색상 오차 분포를 극좌표/직교좌표 히트맵으로 시각화하여, 국소적으로 색차가 큰 부분을 쉽게 찾음.
3. **Comparison Chart:** 각 Zone의 측정값(L\*, a\*, b\*)과 기준값을 바 차트 등으로 비교 제시.
4. **Dashboard:** 여러 이미지의 결과를 요약한 대시보드 (예: 배치 검사 시 OK/NG 개수, 평균 ΔE 등).

### 1.3 범위
- **InspectionVisualizer 클래스** 구현 설계.
- 기존 Pipeline/CLI와의 통합: `--visualize` 옵션 처리 및 `VisualizerPlugin` 구상.
- **시각화 결과 저장** (PNG 이미지, PDF 보고서 등) 지원.
- **Jupyter Notebook**에서 시각화 모듈 활용 예시.

UI(Web UI나 Desktop 앱)에서의 사용은 본 모듈의 결과물(이미지, 그래프)을 임베드하는 형태로 고려합니다.

## 2. 아키텍처
### 2.1 클래스 다이어그램
```
InspectionVisualizer
├─ init(config: VisualizerConfig)
├─ visualize_zone_overlay(image, lens_detection, zones, result) -> np.ndarray
├─ visualize_delta_e_heatmap(profile, zones, sku_config) -> Tuple[np.ndarray, np.ndarray]
├─ visualize_comparison(zones, result) -> plt.Figure
├─ visualize_dashboard(results: List[InspectionResult]) -> plt.Figure
├─ save_visualization(image_or_figure, output_path: Path, format: str)
└─ _draw_zone_boundary(image, zone, color),
   _draw_zone_label(image, zone, text, color),
   _create_polar_heatmap(profile, zones),
   _create_cartesian_heatmap(profile, zones),
   _get_zone_color(zone_result)
```
- **VisualizerConfig**: 시각화 옵션 (컬러맵, 폰트 크기 등)을 담을 수 있는 설정 객체(dataclass)로 가정합니다.

### 2.2 데이터 흐름
Visualizer는 **파이프라인 처리 완료 후**의 결과물을 입력으로 동작합니다:
```
(검사 Pipeline 처리 완료)
      ↓ InspectionResult (data structure)
Visualizer 모듈 호출
  ├─ Zone Overlay 생성 → 출력 이미지 (PNG 등)
  ├─ ΔE Heatmap 생성 → 출력 이미지 2종 (polar, cartesian)
  ├─ Comparison Chart 생성 → Matplotlib Figure (또는 이미지)
  └─ Dashboard 생성 → Matplotlib Figure
(필요 시 파일로 저장)
```
즉, 파이프라인 자체에는 영향을 주지 않고, **사후 처리(Post-processing)**로 결과를 시각화합니다. CLI에서는 `--visualize`가 켜지면 Pipeline 결과를 받아 Visualizer의 각 기능을 호출하고 저장하도록 할 계획입니다. Web UI에서는 Pipeline 결과 JSON을 받아 JavaScript 차트로 그리므로, Python Visualizer는 주로 **오프라인 보고서**나 **데스크탑 앱**에서 활용될 것입니다.

## 3. 시각화 타입별 명세
### 3.1 Zone Overlay
**목적:** 검사한 렌즈 이미지 위에 **검출된 영역과 Zone별 판정**을 한눈에 보여줍니다.

**입력:**
- `image`: 원본 또는 전처리된 컬러 이미지 (`np.ndarray`, BGR)
- `lens_detection`: 렌즈 검출 결과 (예: `LensDetection` 객체 또는 (center_x, center_y, radius))
- `zones`: Zone 리스트 (각 Zone에 반경 범위, 이름 등이 있음)
- `inspection_result`: 검사 결과 (`InspectionResult`, Zone별 판정 포함)

**출력:** Overlay가 그려진 이미지 (`np.ndarray`, BGR).

**시각화 요소:**
1. **렌즈 외곽선**: 검출된 렌즈 경계를 원 형태로 그림. 스타일은 회색 점선 등으로 표시.
2. **렌즈 중심**: 검출 중심에 십자(+) 마크 표시 (색상: 빨간색 등).
3. **Zone 경계선**: 각 Zone의 경계 반경 위치에 원을 그림.
   - 해당 Zone이 OK이면 녹색 원, NG이면 빨간색 원 (선 두께도 NG일 때 더 두껍게).
4. **Zone 레이블**: 각 Zone에 라벨을 표시. 내용: Zone 이름 + ΔE 값 + 판정(OK/NG).
   - 레이블 위치: Zone 영역의 중간 반지름 지점 (예: Zone 경계 두 개 사이의 중간 거리 지점).
   - 레이블 스타일: 반투명 박스에 흰색/검은색 글씨 (가독성 위해).
5. **전체 판정 표시**: 이미지 구석 (좌상단 등)에 이번 검사 전체 결과 표시. OK면 녹색 배경 박스에 "OK", NG면 빨간 배경에 "NG (ΔE=..)" 등 주요 정보를 요약.

**간단한 레이아웃 예시:**
```
┌─────────────────────────┐
│ [OK] ΔE=0.5 │           │ ← 전체 판정
│             │           │
│      ⊕      │           │ ← 렌즈 중심 표시
│    ╱   ╲    │           │
│   ╱ Zone A ╲ │          │ ← Zone A 경계선 + 레이블 (예: [OK] ΔE=0.3)
│   │         │           │
│   ╲ Zone B ╱ │          │ ← Zone B 경계선 + 레이블 (예: [NG] ΔE=5.2)
│    ╲   ╱    │           │
│      ○      │           │ ← 렌즈 외곽선
└─────────────────────────┘
```
*(위 그림은 개념을 나타낸 것이며 실제 디자인은 코드로 구현)*

### 3.2 ΔE Heatmap
**목적:** 렌즈 표면 전체의 ΔE 분포를 **공간적으로 시각화**하여, 특정 부위의 색상 편차가 큰지 등을 직관적으로 확인합니다.

**입력:**
- `radial_profile`: 극좌표 프로파일 데이터 (`RadialProfile`, 각 (반지름 r, 각도 θ)에 대해 LAB 및 ΔE 계산 가능)
- `zones`: Zone 리스트 (경계 정보를 활용)
- `sku_config`: 해당 SKU의 기준값 딕셔너리 (Zone별 기준 LAB 및 threshold 사용)

**출력:**
- `polar_heatmap`: 극좌표 형태의 ΔE 히트맵 이미지 (`np.ndarray`)
- `cartesian_heatmap`: 직교좌표로 펼친 ΔE 히트맵 (`np.ndarray`)

**3.2.1 극좌표 히트맵 (Polar)**
- **계산:** RadialProfile의 각 (r, θ) 지점에서 ΔE 값 계산. (기준 값은 해당 Zone의 기준 LAB; 우선 각 (r,θ)이 어떤 Zone에 속하는지 판별 필요)
- 각 점의 ΔE를 색으로 표현하여 원 형태 이미지 생성. OpenCV `circle` 혹은 polar→cartesian 변환을 응용할 수 있습니다.
- **컬러맵:** Red-Yellow-Green 등의 연속 컬러맵 (예: `RdYlGn_r` – 빨간색: ΔE 높음, 녹색: 낮음).
- **표시 요소:**
  - 렌즈 실제 모양대로 **원형으로 출력** (즉 이미지의 극좌표 변환 버전).
  - ΔE 값에 따라 픽셀 색 채움.
  - **Zone 경계**: 검출된 Zone 경계 반경 위치에 원형 경계선을 점선 등으로 표시 (검은색 또는 흰색 점선).
  - **허용 기준 초과 영역**: ΔE가 해당 Zone threshold를 넘는 지점들을 강조 (예: 특정 색으로 덧칠 또는 깜빡임 표시, 정적 이미지에서는 진한 빨강 등).

**3.2.2 직교좌표 히트맵 (Cartesian)**
- **계산:** 극좌표 데이터를 각도(θ) vs 반지름(r) 2D 평면으로 펼칩니다 (θ = x-axis 0~360°, r = y-axis 0~1 normalized).
- **표시:**
  - X축: 각도 0°~360° (혹은 0~180° 대칭이므로 180°까지만 표현 가능하지만 이해를 돕기 위해 360° 풀 사용).
  - Y축: 반지름 0 (중심) ~ 1 (가장자리) 정규화.
  - 컬러맵: 위와 동일 (ΔE 크기에 따른 색).
  - **Zone 경계선**: 각 Zone 반경 경계를 y축 상에 선으로 표시 (예: y=0.33, 0.66 등의 수평선).
- 이 차트는 렌즈 주변부 vs 중심부, 그리고 각도 방향으로 ΔE가 어디서 튀는지 볼 수 있습니다.

*(두 히트맵 모두, ΔE의 절대값 범위는 0~10 범위로 컬러맵을 정규화할 계획입니다. 0 이 녹색, 10 이상은 빨간색 saturate.)*

### 3.3 Comparison Chart
**목적:** **각 Zone의 측정된 색상값과 기준값**을 한눈에 비교합니다. 예를 들어 Zone A의 측정 LAB vs 기준 LAB.

**입력:**
- `zones`: Zone 리스트 (각 Zone에 측정 평균 LAB 등이 포함됐다고 가정, 없으면 InspectionResult에서 구해와야 함)
- `inspection_result`: 검사 결과 (Zone별 ΔE 및 판정 등 포함)

**출력:** matplotlib Figure (또는 np.ndarray 이미지로 렌더링).

**차트 구성:** 두 가지 서브플롯으로 구상:
- **LAB 막대 그래프:** 각 Zone마다 3개의 막대(또는 점) 그룹: *기준 L\**, *측정 L\** (그리고 a\*, b\* 마찬가지). 한 눈에 어느 채널에서 오차가 큰지 볼 수 있도록.
  - X축: Zone (A, B, C,...)
  - Y축: 값 (0~100 정도 스케일 for L, -128~127 for a/b – 이를 같은 차트에 나타내기 어렵다면 별도 normalization이나 분리 고려)
  - 막대: 예를 들어 Zone A 기준 L=80 vs 실측 L=75, 기준 a=5 vs 실측 a=3, 기준 b=-10 vs 실측 b=-12.
  - ΔE 값이 크면 차이가 확연히 보일 것.
- **ΔE 막대 그래프:** 각 Zone의 ΔE 값을 한 개 막대로 표시하고, threshold 값을 기준선으로 함께 표시.
  - X축: Zone
  - Y축: ΔE
  - 막대 색: OK이면 녹색, NG이면 빨간색.
  - 그래프 위에 threshold를 가로선으로 그어서 넘었는지 바로 보이게.

*(Alternatively, a combined chart with 4 bars per Zone – L*, a*, b* difference, and ΔE – but that might be too cluttered. 나누는 것이 가독성에 좋습니다.)*

### 3.4 Dashboard (배치 처리 요약)
**목적:** 여러 이미지 검사 결과를 요약하여 **전체 품질 현황**을 보여줍니다. (Phase 2 or future)

**입력:**
- `results`: InspectionResult 리스트 (배치 검사 실행 결과들)

**출력:** matplotlib Figure (대시보드 그래프들).

**구성 요소:**
- **OK/NG 비율 차트:** 파이 차트 또는 막대 그래프로 OK vs NG 개수.
- **ΔE 분포 히스토그램:** 모든 이미지의 overall ΔE 값 분포. NG 이미지들은 ΔE가 높게 몰릴 테니 threshold 주변으로 쏠림 확인 가능.
- **평균/최대 ΔE:** 텍스트로 표시 (예: "Max ΔE = 7.5 at image XYZ.jpg").
- **Zone별 NG 빈도:** 만약 Zone별로 NG 발생 카운트를 집계할 수 있으면, 어떤 Zone에서 주로 불량나는지 막대그래프로 표시 (예: Zone C가 5회 NG로 가장 많음).
- **기타:** SKU별 (만약 혼합 SKU 검사했다면) 통계 등.

*(Dashboard는 구체적 설계는 유연하지만, 위 요소들을 조합해 한 화면에 보이도록 합니다.)*

## 4. VisualizerConfig
Visualizer의 설정 객체로, 시각화 출력의 세부사항을 조정 가능합니다. 예:
```python
 @dataclass
class VisualizerConfig:
    overlay_font_scale: float = 0.5
    overlay_font_color: Tuple[int,int,int] = (255,255,255)
    overlay_bg_color: Tuple[int,int,int] = (0,0,0,150)  # RGBA last value for opacity if using image blending
    colormap: str = "RdYlGn_r"
    save_format: str = "png"
    ...
```
사용자가 필요시 config를 InspectionVisualizer에 넘겨 특수한 표시(예: 색맹 지원 팔레트 등) 가능하게.

## 5. CLI 통합 및 사용 예
Visualizer 모듈은 CLI와 연계되어 사용할 수 있습니다. Phase 1에서는 CLI에서 Visualizer를 호출해 파일 출력만 구현하고, Phase 2에서는 Pipeline과의 플러그인 구조로 개선할 예정입니다.

### 5.1 단일 이미지 시각화 (CLI)
`--visualize` 옵션 처리 예:
```bash
python src/main.py --image data/raw_images/NG_001.jpg --sku SKU001 --visualize
```
Pipeline 검사를 수행한 후:
1. `InspectionVisualizer.visualize_zone_overlay` → `results/<id>/overlay.png`
2. `visualize_delta_e_heatmap` → `results/<id>/heatmap_polar.png` & `heatmap_unwrap.png`
3. `visualize_comparison` → `results/<id>/comparison.png`
콘솔 출력으로 저장 경로를 안내:
`Visualization saved: results/20251212_104500/overlay.png (etc.)`
(CLI에서는 PDF 한 파일로 합치는 기능은 현재 고려 안 함, 필요시 추후)

### 5.2 배치 시각화 (CLI)
`--visualize`와 `--batch` 동시 사용:
```bash
python src/main.py --batch data/raw_images/ --sku SKU001 --visualize
```
- 각 이미지별 overlay 생성 (원본 파일명 기반으로 결과 폴더 구조 가능: e.g., `results/batch_<id>/image1_overlay.png` ...)
- 전체 배치 대시보드 그래프 생성 → `batch_dashboard.png`
- CSV와 함께 이러한 파일들을 제공.
(Batch의 각 이미지를 다 시각화하면 양이 많아질 수 있어, CLI에선 배치 시각화 옵션 사용 시 특정 대표이미지만 하거나, overlay만 생성하고 대시보드만 표시하는 등 정책 필요. 우선은 가능하도록 구현하고 사용자에게 양을 알리는 것으로.)

### 5.3 Visualizer 모듈 코드 사용 (예: Notebook)
개발자가 Notebook에서 수동으로 시각화를 원할 때:
```python
from src.visualizer import InspectionVisualizer, VisualizerConfig

vis = InspectionVisualizer()
result = pipeline.process("test.jpg", "SKU001")

overlay_img = vis.visualize_zone_overlay(image, lens, zones, result)
plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
```
혹은:
```python
fig = vis.visualize_comparison(result.zones, result)
fig.show()
```
이런 식으로 Notebook에서 바로 그래프 확인이 가능하게, matplotlib figure를 반환하도록 설계했습니다.

## 6. 성능 요구사항
시각화 모듈은 주로 오프라인 분석 용도라 실시간성은 덜 중요하지만, 그래도 원활한 UX를 위해 아래 목표로 합니다:

### 7.1 처리 시간
- **단일 이미지 Overlay:** < 50ms (이미지 드로잉 위주, OpenCV 빠름).
- **Heatmap 계산:** < 100ms (ΔE 계산 및 이미지 생성, 360xradius 해상도 기준).
- **Comparison 차트:** < 50ms (matplotlib drawing).
- 전체 Web UI 모드: front-end Chart.js 사용 시 이미 충분히 빠름.
(주로 Python 시각화는 Notebook/CLI용이므로 성능 여유 충분).

### 7.2 메모리
- 단일 시각화는 이미지 한두 장, figure 하나이므로 메모리 << 500MB (유의미한 이슈 없음).
- **배치 대시보드:** 100장 기준 데이터 집계 정도이므로 문제 없음.
- 단, **대용량 이미지 (예: 4K)**를 다룰 시 OpenCV 메모리 사용량 증가 가능. Overlay 생성 전 이미지 리사이즈(디스플레이 목적) 등을 고려 (최대 1920x1080 정도로 축소 저장).
- PDF 등 생성 시 Pillow, matplotlib 내 메모리 사용이 있으나 수십 MB 수준으로 예상.

### 7.3 출력 파일 크기
- PNG Overlay: 1920x1080 기준 <2MB.
- PDF Report: 수십 페이지일 경우 5MB 내 (배치 100장 * 1page + summary).
(파일 크기가 너무 크면 공유 어려우니, 압축/해상도 조정 등의 대책 있을 수 있음)

## 7. 에러 처리
시각화 과정에서의 에러는 비교적 드물지만:

### 8.1 에러 타입
- **Invalid Data:** 전달된 profile이나 zones 데이터에 문제가 있을 경우 (ex: zone 경계가 profile 길이 초과 등).
- **Matplotlib Backend Error:** 서버 환경 등에서 DISPLAY 없어 에러나는 경우 (CLI에서 PDF 만들 때 headless backend 설정해야).
- **File I/O Error:** save_visualization 시 경로 문제.

### 8.2 Fallback 전략
- **컬러맵 로드 실패:** 예를 들어 custom 컬러맵 파일을 쓰려다 실패하면, 기본 컬러맵(예: viridis)로 대체합니다.
- **폰트 로드 실패:** 특수 폰트 사용시 문제 생기면 OpenCV 기본 폰트로 텍스트 표시하도록.
- **Matplotlib rendering fail:** 간혹 Agg backend 문제시 PDF 생략하고 PNG만 저장 등 대응.
전반적으로 시각화 실패가 본 시스템 핵심 기능에 영향 주지 않도록, 오류 시 경고 로그만 남기고 나머지 흐름은 진행하는 식으로 처리합니다.

## 9. 확장성
### 9.1 향후 추가 기능
- **Interactive UI:** Desktop UI에서 시각화 결과를 interactive하게 탐색 (예: zone 경계 슬라이더로 조정 즉시 반영).
- **동영상 지원:** 연속 이미지(동영상 프레임)에 대한 시각화 (moving lens 검사 등 확장).
- **다른 차트:** ΔE 시계열, 3D surface plot 등 요구에 따라 추가 가능.

### 9.2 플러그인 시스템
앞서 파이프라인 설계에서 언급한 VisualizerPlugin 형태로 통합:
```python
class VisualizerPlugin(PipelinePlugin):
    def after_process(self, image, result, **kwargs):
        vis = InspectionVisualizer()
        vis.visualize_zone_overlay(image, result.lens, result.zones, result)
        vis.save_visualization(..., format='png')
```
이런 식으로 Pipeline `process()` 내부에서 hook 호출하면, CLI가 `--visualize` 옵션을 신경쓰지 않아도 항상 Visualizer 실행 가능. (옵션으로 plugin 추가를 제어하면 됨). 현재는 우선 Visualizer 모듈을 standalone으로 만들고, 나중에 Plugin 체계에 녹이는 것으로 계획.

## 10. 테스트 전략
### 10.1 단위 테스트 (예: 12개)
- **Overlay correctness:** 임의의 간단한 이미지와 가짜 lens/zones 데이터로 overlay 호출, 반환 이미지의 픽셀 수량/색상이 예상대로 변했는지 (예: 중심점 좌표에 빨간색 픽셀이 있는지 등).
- **Heatmap array size:** profile → heatmap 생성 후, polar 이미지가 원형 형태 (예: 일정 반지름 밖은 0 또는 특정색) 확인, cartesian 이미지 크기 확인.
- **Comparison chart data:** Matplotlib 객체에서 예상 막대 개수, 값 검증 (MPL testing tools 활용).
- **Dashboard logic:** few dummy results feed in, ensure figure contains expected text (like "OK: X, NG: Y").
- **Save_visualization:** test that given a numpy image or figure, the file is saved (check file existence and format).

### 10.2 통합 테스트
- **CLI integration test:** `main.py --visualize` 실행 후 결과 폴더에 overlay, heatmap, etc. 모두 생성되었는지.
- **Visual inspection by QA:** 실제 다양한 결과에 대해 생성된 시각화들을 품질팀이 확인하여 유용성 피드백. 특히 색맹 모드 필요성, 색상 선택 적절성 등에 대한 의견 수렴 (접근성 12장 참고).

### 10.3 시각적 회귀 테스트
시각화 모듈은 그림 자체가 품질이므로, baseline 이미지를 통한 시각적 회귀 테스트를 도입할 수도 있습니다:
- 예를 들어 동일한 input `InspectionResult`에 대해 overlay 이미지를 생성하고, 이전 버전 결과 이미지와 픽셀 단위 비교하여 차이가 없음을 검증.
- 허용 오차: 1% 이내 픽셀 차이 (이미지 압축/랜덤성 요소 고려).
이 기법으로 코드 변경이 시각적 결과에 나쁜 영향을 주지 않았는지 자동 확인 가능.

## 11. 보안 및 접근성
### 11.1 파일 출력 보안
시각화 결과는 주로 내부에서 보며, 보안 이슈 적음. 그래도 경로 traversal 등 일반적 주의만. (Visualizer는 파일을 읽지는 않고 InspectionResult 객체 사용)

### 11.2 메모리 관리
대용량 이미지를 반복 생성 시 메모리 누수 유의:
- OpenCV 객체는 numpy 배열로 관리되므로 Python GC 대상.
- Matplotlib Figure는 사용 후 `plt.close(fig)`로 명시 해제 추천.
- **이미지 크기 제한:** 가급적 4K 이상 이미지는 다루지 않음 (UI에서 업로드시 제한하거나, Visualizer에서 강제 축소).
- **배치 크기 제한:** 한 번에 1000장 이상 시각화는 비현실적이므로, 필요 시 경고 및 중지.

## 12. 접근성 (Accessibility)
일부 사용자가 색약/색맹일 경우 시각화 색상에 대한 고려가 필요합니다:
### 12.1 색맹 지원
ΔE 히트맵의 Red-Green 컬러맵은 적록색약자에게 구분이 어려울 수 있습니다. 대안으로 색맹 친화 팔레트(예: Viridis 등)를 옵션으로 제공하거나, 패턴/명도 차이를 함께 활용.
UI단에서는 차트.js 등에서 패턴 채우기 등을 고려 가능.
VisualizerConfig에 `colormap_colorblind: True` 옵션을 두어 팔레트 변경 가능하도록.

### 12.2 고대비 모드
보고서를 인쇄하거나, 시각 장애가 있는 분을 위해 고대비 테마 지원:
- 배경을 흰색 또는 검은색으로 통일하고 텍스트를 대비되게 (예: 검정 배경+흰 글씨).
- 이 모드는 config로 설정하면 Overlay 등에서 채색 대신 라벨 텍스트 위주 표시 등의 변화도 가능.
(현재는 기본 기능 개발 우선, 접근성 모드는 추후 품질팀 요구에 따라 추가 계획.)

## 13. 참고 자료
- **Matplotlib Documentation** – Colormap (적절한 컬러맵 선택 참고)
- **OpenCV Drawing Functions** (이미지 위 그리기 참고)
- **Color Brewer Guidelines for Colorblind-safe Palettes** (컬러 선택 가이드)
- 기존 프로젝트의 Web UI Chart.js 구현 (Worker B 작성, 이미 L*a*b* 그래프 구현됨) – 해당 코드 참고하여 일관성 유지.
