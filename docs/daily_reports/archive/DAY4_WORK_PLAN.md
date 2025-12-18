# Day 4 작업 계획서

> **목표**: Visualizer 구현 - Zone 오버레이, ΔE 히트맵, 판정 결과 시각화
> **예상 소요 시간**: 3.5시간
> **작업 방식**: 병렬 작업 + 명확한 핸드오프 포인트

---

## 📋 선택된 옵션

**Option 1: Visualizer 구현** ✅

**선택 이유:**
- 디버깅 및 품질 검증 효율 향상
- 사용자(검사자, 품질 관리자)에게 직관적 인터페이스 제공
- Jupyter Notebook 프로토타입과 자연스럽게 통합
- 생산 환경에서도 즉시 활용 가능

---

## 🎯 목표

1. **Visualizer 클래스** 구현 (Zone 오버레이, ΔE 히트맵)
2. **CLI 시각화 명령어** 추가 (`inspect --visualize`, `batch --visualize`)
3. **판정 결과 대시보드** (요약 통계 시각화)
4. **Jupyter Notebook 확장** (시각화 섹션 추가)

---

## 👥 작업 분담

### Phase 1: 준비 및 설계 (병렬) - 20분

#### 👤 Claude Task C1: 설계 문서 작성 (15분)
**산출물:**
- `docs/VISUALIZER_DESIGN.md`
  - Visualizer 클래스 설계
  - 시각화 타입별 명세 (overlay, heatmap, dashboard)
  - CLI 인터페이스 확장
  - 파일 출력 포맷 (PNG, PDF)

**시작:** 즉시
**완료 조건:** 설계 문서 커밋

---

#### 👤 개발자 A Task A1: 시각화 테스트 케이스 준비 (20분)
**작업 내용:**
- **대표 이미지 선정** (시각화 품질 검증용):
  - OK 샘플 3장 (SKU001/002/003 각 1장)
  - NG 샘플 3장 (다양한 ΔE 수준: 약한 NG, 중간 NG, 강한 NG)

- **테스트 케이스 목록 작성** (`data/visualizer_test_cases.csv`):
  ```csv
  image_path,sku,expected_judgment,test_purpose
  data/raw_images/SKU001_OK_001.jpg,SKU001,OK,Baseline overlay test
  data/raw_images/SKU002_NG_001.jpg,SKU002,NG,Weak defect visualization
  data/raw_images/SKU003_NG_005.jpg,SKU003,NG,Strong defect heatmap
  ...
  ```

**도구:** 기존 이미지 선별 + CSV 작성
**시작:** 즉시
**완료 조건:** 6장 선정 + CSV 파일 생성

---

### Phase 2: 핵심 구현 (순차) - 120분

#### 👤 Claude Task C2: Visualizer 구현 + CLI 통합 (120분)

**⏸️ 대기 조건:**
- Phase 1 완료 후 시작 (Task A1 완료 필요)
- 테스트 케이스 이미지로 검증하며 구현

**작업 내용:**

**2-1. Visualizer 클래스** (`src/visualizer.py`, 400+ lines)
```python
class InspectionVisualizer:
    def __init__(self, config: Optional[VisualizerConfig] = None):
        pass

    def visualize_zone_overlay(
        self,
        image: np.ndarray,
        lens_detection: LensDetection,
        zones: List[Zone],
        inspection_result: InspectionResult
    ) -> np.ndarray:
        """Zone 경계선 + 판정 결과 오버레이"""

    def visualize_delta_e_heatmap(
        self,
        radial_profile: RadialProfile,
        zones: List[Zone],
        sku_config: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ΔE 히트맵 (극좌표 + 직교좌표)"""

    def visualize_comparison(
        self,
        zones: List[Zone],
        inspection_result: InspectionResult
    ) -> plt.Figure:
        """측정값 vs 기준값 비교 차트"""

    def visualize_dashboard(
        self,
        results: List[InspectionResult]
    ) -> plt.Figure:
        """배치 처리 요약 대시보드"""

    def save_visualization(
        self,
        image: Union[np.ndarray, plt.Figure],
        output_path: Path,
        format: str = "png"
    ):
        """시각화 결과 저장"""
```

**2-2. CLI 확장** (`src/main.py` 수정, +100 lines)
```bash
# 단일 이미지 시각화
python -m src.main inspect --image data/raw_images/OK_001.jpg --sku SKU001 \
  --visualize --output results/OK_001_viz.png

# 배치 시각화 (각 이미지별 + 요약 대시보드)
python -m src.main batch --batch data/raw_images/ --sku SKU001 \
  --visualize --output-dir results/visualizations/

# 시각화 타입 선택
python -m src.main inspect --image data/raw_images/OK_001.jpg --sku SKU001 \
  --visualize overlay,heatmap,comparison \
  --output results/OK_001_viz.png
```

**2-3. VisualizerConfig** (`src/visualizer.py`)
```python
@dataclass
class VisualizerConfig:
    # Zone overlay
    zone_line_thickness: int = 2
    zone_color_ok: Tuple[int, int, int] = (0, 255, 0)
    zone_color_ng: Tuple[int, int, int] = (0, 0, 255)
    show_zone_labels: bool = True

    # Heatmap
    colormap: str = "RdYlGn_r"  # Red=high ΔE, Green=low ΔE
    show_colorbar: bool = True

    # Dashboard
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
```

**2-4. 시각화 타입별 구현**

**Zone Overlay:**
- 원본 이미지 위에 Zone 경계선 표시
- Zone별 색상 코드: OK=녹색, NG=빨간색
- Zone 이름 + ΔE 값 텍스트 표시
- 전체 판정 결과 (OK/NG) 상단에 표시

**ΔE Heatmap:**
- Radial profile 기반 ΔE 분포 계산
- 극좌표 히트맵 (원형)
- 직교좌표 히트맵 (전개도)
- 기준값 초과 영역 하이라이트

**Comparison Chart:**
- Zone별 측정 LAB vs 기준 LAB (막대 그래프)
- Zone별 ΔE vs Threshold (라인 차트)
- Pass/Fail 영역 색상 구분

**Dashboard (배치 처리):**
- 전체 판정 비율 (파이 차트)
- SKU별 ΔE 분포 (박스 플롯)
- Zone별 NG 빈도 (히트맵)
- 처리 속도 타임라인

**완료 조건:**
- Visualizer 클래스 완성
- CLI 명령어 동작 확인
- 6개 테스트 케이스 시각화 성공

---

### Phase 3: 검증 및 확장 (병렬) - 60분

#### 👤 Claude Task C3: 통합 테스트 (30분)

**⏸️ 대기 조건:** Task C2 완료 후

**작업 내용:**
- `tests/test_visualizer.py` (250+ lines, 12개 테스트)
  - test_visualizer_initialization()
  - test_zone_overlay_ok_image()
  - test_zone_overlay_ng_image()
  - test_delta_e_heatmap()
  - test_comparison_chart()
  - test_dashboard_single_sku()
  - test_dashboard_multi_sku()
  - test_save_png()
  - test_save_pdf()
  - test_cli_visualize_command()
  - test_batch_visualize()
  - test_custom_config()

**검증:**
- 전체 테스트 통과 (123개 → 135개, +12)
- 시각화 출력 품질 확인
- 성능 테스트 (시각화 시간 <100ms/장)

**완료 조건:** 전체 테스트 통과

---

#### 👤 개발자 B Task B1: Jupyter Notebook 확장 (60분)

**⏸️ 대기 조건:** Task C2 완료 후 (Visualizer 클래스 사용 필요)

**작업 내용:**
- `notebooks/03_visualization_demo.ipynb` (신규 생성) 또는
- `notebooks/01_prototype.ipynb` 업데이트 (시각화 섹션 추가)

**섹션 구성 (신규 노트북 생성 시, 6개 섹션):**
1. **환경 설정** - Visualizer 임포트
2. **Zone Overlay 데모** - OK/NG 이미지 비교
3. **ΔE Heatmap 데모** - 극좌표/직교좌표 비교
4. **Comparison Chart** - 측정값 vs 기준값
5. **다중 이미지 비교** - SKU별 시각화
6. **Dashboard 데모** - 배치 처리 요약

**또는 기존 노트북 업데이트 시:**
- `01_prototype.ipynb`의 각 섹션에 시각화 추가
  - Section 3 (렌즈 검출) → Zone overlay 추가
  - Section 6 (색상 평가) → ΔE heatmap 추가
  - Section 7 (배치 처리) → Dashboard 추가

**완료 조건:** Notebook 실행 가능 + 모든 시각화 표시

---

### Phase 4: 최종 검증 및 문서화 (순차) - 30분

#### 👥 전체 작업자 (Claude + 개발자 A + 개발자 B)

**⏸️ 대기 조건:** Task C3, B1 모두 완료

**검증 항목:**
1. ✅ Zone overlay 시각화 품질 확인
2. ✅ ΔE heatmap 정확도 확인
3. ✅ CLI 명령어 동작 확인
4. ✅ 전체 테스트 통과 (135개)
5. ✅ Jupyter Notebook 실행 가능
6. ✅ 시각화 성능 기준 충족 (<100ms/장)
7. ✅ 문서화 완료

**문서 작성 (Claude):**
- `DAY4_COMPLETION_REPORT.md`
- `docs/VISUALIZER_DESIGN.md` (이미 작성됨)
- `README.md` 업데이트 (시각화 사용 예제)

**Git 커밋:**
```bash
git add -A
git commit -m "feat: Day 4 - Implement visualization system

- Add InspectionVisualizer class (zone overlay, ΔE heatmap, dashboard)
- Extend CLI with --visualize option
- Add 12 visualization tests (123 → 135 total)
- Add visualization demo notebook

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## 📊 성공 기준 (7개)

| # | 기준 | 목표 | 검증 방법 |
|---|------|------|-----------|
| 1 | Zone overlay 시각화 | OK/NG 구분 표시 | 육안 검증 + 테스트 |
| 2 | ΔE heatmap 생성 | 극좌표/직교좌표 | 파일 출력 확인 |
| 3 | Comparison chart | 측정 vs 기준 비교 | 그래프 정확도 |
| 4 | Dashboard 생성 | 배치 요약 통계 | 다중 이미지 처리 |
| 5 | CLI 통합 | --visualize 옵션 동작 | 명령어 실행 |
| 6 | 테스트 통과 | 135개 (123→+12) | pytest 실행 |
| 7 | 성능 기준 | <100ms/장 (시각화) | 성능 테스트 |

---

## ⏱️ 타임라인

```
00:00 ━━━━━━━━━━━━━━━━━ Phase 1 시작 (병렬)
       ├─ Claude C1 (설계 문서)
       └─ 개발자 A (테스트 케이스 준비)

00:20 ━━━━━━━━━━━━━━━━━ Phase 2 시작 (순차)
       └─ Claude C2 (Visualizer + CLI)
       ⏸️ 개발자 A, B 대기

02:20 ━━━━━━━━━━━━━━━━━ Phase 3 시작 (병렬)
       ├─ Claude C3 (테스트)
       └─ 개발자 B (Notebook)

03:20 ━━━━━━━━━━━━━━━━━ Phase 4 시작 (전체)
       └─ 최종 검증 + 문서화

03:50 ━━━━━━━━━━━━━━━━━ 완료 🎉
```

---

## 📦 예상 산출물

### 코드
- `src/visualizer.py` (400+ lines) - InspectionVisualizer 클래스
- `src/main.py` (+100 lines) - CLI --visualize 옵션
- `tests/test_visualizer.py` (250+ lines, 12개 테스트)

### 데이터
- `data/visualizer_test_cases.csv` (by 개발자 A)
- `results/visualizations/` (시각화 샘플 출력)

### 문서
- `docs/VISUALIZER_DESIGN.md` (설계)
- `DAY4_COMPLETION_REPORT.md` (완료 보고서)
- `notebooks/03_visualization_demo.ipynb` (by 개발자 B) 또는
- `notebooks/01_prototype.ipynb` (업데이트)

**총 신규 코드:** ~750 lines
**총 테스트:** 135개 (123 → +12)

---

## 🔄 핸드오프 포인트 요약

### 🚦 누가 누구를 기다리는가?

**Phase 1 → Phase 2:**
- ✋ **Claude Task C2**는 **개발자 A Task A1** 완료 대기
  - 이유: 테스트 케이스 이미지로 시각화 품질 검증 필요

**Phase 2 → Phase 3:**
- ✋ **Claude Task C3**는 **Claude Task C2** 완료 대기
  - 이유: Visualizer 클래스 구현 완료 후 테스트 가능
- ✋ **개발자 B Task B1**은 **Claude Task C2** 완료 대기
  - 이유: Visualizer API 사용 필요

**Phase 3 → Phase 4:**
- ✋ **전체 작업자**는 **Task C3, B1 모두** 완료 대기
  - 이유: 통합 검증 및 최종 문서화

---

## 💡 주요 설계 결정

### 1. 시각화 타입 분리
- **Zone Overlay**: 검출 결과 확인용 (디버깅)
- **ΔE Heatmap**: 색상 분포 분석용 (품질 분석)
- **Comparison**: 정량적 비교용 (리포트)
- **Dashboard**: 전체 요약용 (관리자)

### 2. 출력 포맷
- **PNG**: 기본 (빠름, 웹 호환)
- **PDF**: 리포트용 (고품질, 벡터)
- **Interactive**: Jupyter에서만 (matplotlib)

### 3. 컬러맵 선택
- **ΔE Heatmap**: RdYlGn_r (Red=높음, Green=낮음)
- **Zone별 판정**: OK=녹색, NG=빨간색
- **색맹 고려**: 추가 패턴/기호 옵션

### 4. 성능 최적화
- **이미지 크기**: 원본 크기 유지 (품질 우선)
- **캐싱**: matplotlib figure 재사용
- **병렬화**: 배치 시각화 시 고려

---

## ❓ 리스크 및 대응

| 리스크 | 확률 | 영향 | 대응 방안 |
|--------|------|------|-----------|
| matplotlib 버전 호환성 | 낮 | 중 | 최신 버전 사용, 대체 라이브러리 준비 |
| 시각화 성능 저하 | 중 | 중 | 이미지 다운샘플링, 캐싱 |
| 컬러맵 가독성 | 낮 | 낮 | 사용자 피드백 반영, 커스터마이징 |
| 파일 출력 크기 | 낮 | 낮 | 압축 옵션, 해상도 조절 |

---

## ✅ 시작 전 체크리스트

- [ ] Day 3 완료 확인 (Git 커밋 완료)
- [ ] 123개 테스트 통과 확인
- [ ] matplotlib 설치 확인 (`pip install matplotlib`)
- [ ] 개발자 A, B 준비 상태 확인
- [ ] Option 1 (Visualizer) 최종 승인

---

**작성자**: Claude (AI Assistant)
**검토 필요**: 개발자 A, 개발자 B
**작성일**: 2025-12-11
