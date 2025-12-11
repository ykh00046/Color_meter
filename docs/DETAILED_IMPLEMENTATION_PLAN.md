# 콘택트렌즈 색상 검사 시스템 - 초상세 구현 계획서

## 문서 개요
- **작성일**: 2025-12-10
- **프로젝트명**: 콘택트렌즈 인쇄/색상 품질 자동 검사 시스템
- **목적**: 두 제안서를 종합하여 실제 구현 가능한 초상세 개발 계획 수립
- **대상 독자**: 개발팀, 프로젝트 관리자, 경영진

---

## 목차
1. [제안서 종합 분석](#1-제안서-종합-분석)
2. [기술 선택 및 타당성](#2-기술-선택-및-타당성)
3. [시스템 아키텍처 상세 설계](#3-시스템-아키텍처-상세-설계)
4. [상세 구현 계획 (단계별)](#4-상세-구현-계획-단계별)
5. [개발 환경 및 기술 스택](#5-개발-환경-및-기술-스택)
6. [데이터 요구사항 및 관리](#6-데이터-요구사항-및-관리)
7. [하드웨어 사양 및 설치](#7-하드웨어-사양-및-설치)
8. [품질 보증 및 테스트 계획](#8-품질-보증-및-테스트-계획)
9. [리스크 관리](#9-리스크-관리)
10. [일정 및 리소스](#10-일정-및-리소스)
11. [유지보수 및 확장 계획](#11-유지보수-및-확장-계획)

---

## 1. 제안서 종합 분석

### 1.1 두 제안서의 핵심 내용 비교

| 항목 | 기본 제안서 (MD) | 개선 제안서 (PDF) |
|------|------------------|-------------------|
| **핵심 접근법** | 동심원 r-프로파일 기반 색상 분석 | 동일 + 경쟁 기술과의 비교 분석 |
| **기술적 깊이** | 알고리즘 개념 및 구조 제시 | 구현 난이도, 성능, 비용 등 실용적 분석 |
| **개발 로드맵** | 4단계 개념적 제시 | 구체적 일정, 산출물, 마일스톤 명시 |
| **리소스 계획** | 언급 없음 | 필요 인력(3-4명), 역할, PoC 요건 상세 |
| **경쟁 기술 분석** | 향후 확장으로 언급만 | PatchCore, PaDiM, Diffusion 등 상세 비교 |
| **현장 도입** | 장비 가이드 초안 언급 | MES 연동, SKU 관리, 유지보수 구체화 |

### 1.2 종합 평가

**강점:**
- **도메인 특화 접근**: 동심원 패턴이라는 렌즈 특성을 적극 활용
- **설명 가능성**: ΔE 수치 기반으로 결과 해석이 명확
- **실용성**: 딥러닝 없이도 구현 가능, 빠른 ROI 기대
- **유연성**: SKU 추가 시 재학습 불필요, 설정값만 등록

**보완 필요 사항:**
- 복잡한 패턴(꽃무늬 등)에 대한 확장성 검증 필요
- 형태적 결함(패턴 밀림, 찌그러짐) 검출 한계
- 조명 변화, 렌즈 위치 변동에 대한 강건성 검증 필요

---

## 2. 기술 선택 및 타당성

### 2.1 채택 기술: 동심원 r-프로파일 기반 접근법

**선택 이유:**

1. **구현 난이도**: OpenCV 기반으로 2명의 개발자로 4-5개월 내 구현 가능
2. **설명 가능성**: 의료기기 규제 대응 가능 (XAI 요구사항 충족)
3. **속도**: 수십 ms/장, 실시간 처리 가능
4. **다품종 대응**: 새 SKU 추가 시 샘플 몇 장으로 즉시 적용
5. **비용 효율성**: GPU 불필요, 일반 산업용 PC로 동작

### 2.2 경쟁 기술 분석 요약

| 기술 | AUROC | 구현 난이도 | 속도 | SKU 대응 | 권장 사항 |
|------|-------|-------------|------|----------|-----------|
| **r-프로파일** | 85-90% (추정) | 낮음 | 매우 빠름 | 우수 | **1차 채택** |
| **PatchCore** | 95-99% | 중간 | 양호 (100-200ms) | 중간 | 2단계 검토 |
| **PaDiM** | 90-95% | 중간 | 양호 | 중간 | 대안 |
| **AutoEncoder** | 80-90% | 중간-높음 | 양호 | 낮음 | 비추천 |
| **Diffusion** | 99%+ | 높음 | 낮음 | 낮음 | 연구용만 |

**결론**: 초기 시스템은 r-프로파일 방식으로 구축하고, 안정화 후 PatchCore를 보조 모듈로 추가하는 2단계 전략 채택

### 2.3 기술 스택 선정

#### 2.3.1 프로그래밍 언어
- **Python 3.10+**: 주 개발 언어
  - 이유: 풍부한 CV/ML 라이브러리, 빠른 프로토타이핑, 유지보수 용이

#### 2.3.2 핵심 라이브러리

```python
# 영상 처리
opencv-python==4.8.1          # 렌즈 검출, 기하 변환
opencv-contrib-python==4.8.1  # 추가 알고리즘
scikit-image==0.22.0          # 고급 영상 처리
pillow==10.1.0                # 이미지 I/O

# 색공간 변환 및 색차 계산
colormath==3.0.0              # LAB 변환, ΔE 계산 (CIEDE2000)
colour-science==0.4.3         # 고급 색 과학 계산

# 수치 계산
numpy==1.26.2                 # 행렬 연산
scipy==1.11.4                 # 신호 처리, 최적화

# 데이터 처리
pandas==2.1.4                 # 검사 결과 관리
h5py==3.10.0                  # 대용량 데이터 저장

# 시각화
matplotlib==3.8.2             # 그래프, 프로파일 플롯
seaborn==0.13.0               # 통계 시각화
plotly==5.18.0                # 인터랙티브 시각화

# GUI (옵션 A: 데스크탑 앱)
PyQt6==6.6.1                  # 크로스 플랫폼 GUI
pyqtgraph==0.13.3             # 실시간 그래프

# GUI (옵션 B: 웹 기반)
fastapi==0.108.0              # 백엔드 API
uvicorn==0.25.0               # ASGI 서버
streamlit==1.29.0             # 프로토타입 대시보드

# 데이터베이스
sqlalchemy==2.0.25            # ORM
sqlite3 (내장)                # 경량 DB (초기)
# postgresql (선택)           # 프로덕션용

# 설정 관리
pydantic==2.5.3               # 설정 검증
python-dotenv==1.0.0          # 환경 변수

# 로깅
loguru==0.7.2                 # 향상된 로깅

# 테스트
pytest==7.4.3                 # 단위 테스트
pytest-cov==4.1.0             # 커버리지

# 카메라 인터페이스
pypylon==3.0.1                # Basler 카메라 (예시)
# vimba (Alliance Vision)     # AVT 카메라
# harvesters==1.4.2           # GenICam 표준
```

#### 2.3.3 개발 도구

```bash
# 버전 관리
git==2.43.0

# 가상 환경
conda or venv

# 코드 품질
black==23.12.1                # 코드 포매터
flake8==7.0.0                 # 린터
mypy==1.8.0                   # 타입 체커
pre-commit==3.6.0             # Git 훅

# 문서화
sphinx==7.2.6                 # API 문서
mkdocs==1.5.3                 # 프로젝트 문서

# 성능 프로파일링
line_profiler==4.1.1
memory_profiler==0.61.0
```

---

## 3. 시스템 아키텍처 상세 설계

### 3.1 전체 시스템 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    검사 시스템 (Inspection System)             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │  카메라     │───▶│ 영상 획득부   │───▶│ 전처리 모듈  │    │
│  │  (Camera)  │    │ (Acquisition)│    │(Preprocessing)│    │
│  └────────────┘    └──────────────┘    └──────┬───────┘    │
│                                                │             │
│                    ┌───────────────────────────▼───────┐    │
│                    │   렌즈 검출 모듈 (LensDetector)   │    │
│                    │  - 중심점 (x₀, y₀) 검출           │    │
│                    │  - 반경 R 추정                    │    │
│                    └───────────────┬───────────────────┘    │
│                                    │                         │
│                    ┌───────────────▼────────────────────┐   │
│                    │ r-프로파일 추출 (RadialProfiler)   │   │
│                    │  - 극좌표 변환                     │   │
│                    │  - 링별 LAB 색상 계산             │   │
│                    │  - L*(r), a*(r), b*(r) 곡선 생성  │   │
│                    └───────────────┬────────────────────┘   │
│                                    │                         │
│       ┌────────────────────────────┼────────────────┐       │
│       │                            │                │       │
│  ┌────▼─────────┐     ┌───────────▼──────┐  ┌─────▼─────┐ │
│  │ Zone 분할     │     │  색상 평가        │  │ 보조 모듈  │ │
│  │(ZoneSegmenter)│     │(ColorEvaluator)  │  │(Optional) │ │
│  │- 변곡점 검출  │     │- ΔE 계산         │  │-PatchCore │ │
│  │- 경계 r값 결정│     │- OK/NG 판정      │  │-기포 검출 │ │
│  └────┬─────────┘     └───────────┬──────┘  └───────────┘ │
│       │                            │                         │
│       └────────────┬───────────────┘                         │
│                    │                                         │
│       ┌────────────▼──────────────────────┐                 │
│       │  결과 출력 & 시각화 (Visualizer)   │                 │
│       │  - 오버레이 이미지                │                 │
│       │  - 프로파일 그래프                │                 │
│       │  - ΔE 히트맵                      │                 │
│       └────────────┬──────────────────────┘                 │
│                    │                                         │
│       ┌────────────▼──────────────┐                         │
│       │  데이터 저장 (Logger)      │                         │
│       │  - 검사 결과 DB 저장       │                         │
│       │  - NG 샘플 이미지 저장     │                         │
│       │  - 통계 리포트 생성        │                         │
│       └───────────────────────────┘                         │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                  설정 관리 (SkuConfigManager)                │
│  - SKU별 기준값 (목표 LAB, ΔE 허용치, zone 경계)            │
│  - 장비 보정 파라미터 (카메라, 조명)                         │
└─────────────────────────────────────────────────────────────┘
         ▲                                      │
         │                                      ▼
    ┌────┴─────┐                        ┌──────────┐
    │   GUI    │                        │   MES    │
    │ 사용자 UI │                        │  연동    │
    └──────────┘                        └──────────┘
```

### 3.2 모듈별 상세 설계

#### 3.2.1 ImageLoader (영상 로더)

**책임**: 이미지 입력 및 기본 전처리

**인터페이스**:
```python
class ImageLoader:
    """이미지 로드 및 기본 전처리"""

    def __init__(self, config: ImageConfig):
        """
        Args:
            config: 이미지 설정 (해상도, ROI 등)
        """
        self.config = config
        self.roi = config.roi  # Region of Interest

    def load_from_file(self, filepath: Path) -> np.ndarray:
        """
        파일에서 이미지 로드

        Args:
            filepath: 이미지 파일 경로

        Returns:
            BGR 이미지 (H, W, 3)
        """
        pass

    def load_from_camera(self, camera: Camera) -> np.ndarray:
        """
        카메라에서 실시간 획득

        Args:
            camera: 카메라 객체

        Returns:
            BGR 이미지
        """
        pass

    def preprocess(self, image: np.ndarray) -> dict:
        """
        전처리 수행

        Args:
            image: 원본 이미지

        Returns:
            {
                'original': 원본 이미지,
                'gray': 그레이스케일,
                'denoised': 노이즈 제거,
                'roi': ROI 크롭 이미지 (선택)
            }
        """
        pass
```

**상세 구현 사항**:

1. **노이즈 제거**
   ```python
   def denoise(self, image: np.ndarray) -> np.ndarray:
       """
       가우시안 블러 + bilateral 필터 조합
       - 가우시안: 고주파 노이즈 제거
       - Bilateral: 경계는 보존하면서 노이즈 제거
       """
       # 가우시안 블러 (kernel size: 3x3 or 5x5)
       blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=0.5)

       # Bilateral 필터 (색상 차이는 보존)
       denoised = cv2.bilateralFilter(
           blurred,
           d=5,              # 필터 크기
           sigmaColor=50,    # 색상 공간 표준편차
           sigmaSpace=50     # 좌표 공간 표준편차
       )
       return denoised
   ```

2. **ROI 자동 검출** (렌즈 영역만 크롭)
   ```python
   def detect_roi(self, image: np.ndarray) -> tuple:
       """
       렌즈 대략적 위치 검출로 ROI 설정
       Returns: (x, y, w, h)
       """
       gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

       # Otsu 이진화로 렌즈 영역 분리
       _, binary = cv2.threshold(
           gray, 0, 255,
           cv2.THRESH_BINARY + cv2.THRESH_OTSU
       )

       # 컨투어 검출
       contours, _ = cv2.findContours(
           binary,
           cv2.RETR_EXTERNAL,
           cv2.CHAIN_APPROX_SIMPLE
       )

       # 가장 큰 컨투어를 렌즈로 가정
       if contours:
           largest = max(contours, key=cv2.contourArea)
           x, y, w, h = cv2.boundingRect(largest)

           # 여유 마진 추가 (20%)
           margin = int(max(w, h) * 0.2)
           x = max(0, x - margin)
           y = max(0, y - margin)
           w = w + 2 * margin
           h = h + 2 * margin

           return (x, y, w, h)
       else:
           # ROI 검출 실패 시 전체 이미지 사용
           return (0, 0, image.shape[1], image.shape[0])
   ```

3. **화이트 밸런스 보정**
   ```python
   def white_balance(self, image: np.ndarray,
                     method='gray_world') -> np.ndarray:
       """
       화이트 밸런스 보정 (색상 일관성 확보)

       Methods:
           - gray_world: RGB 평균을 회색으로 조정
           - color_chart: X-Rite 차트 기반 보정
       """
       if method == 'gray_world':
           # Gray World 가정: 평균 색상 = 회색
           avg_b = np.mean(image[:, :, 0])
           avg_g = np.mean(image[:, :, 1])
           avg_r = np.mean(image[:, :, 2])

           avg_gray = (avg_b + avg_g + avg_r) / 3

           # 스케일 팩터 계산
           scale_b = avg_gray / avg_b
           scale_g = avg_gray / avg_g
           scale_r = avg_gray / avg_r

           # 보정 적용
           balanced = image.copy().astype(np.float32)
           balanced[:, :, 0] *= scale_b
           balanced[:, :, 1] *= scale_g
           balanced[:, :, 2] *= scale_r

           return np.clip(balanced, 0, 255).astype(np.uint8)

       elif method == 'color_chart':
           # ColorChecker 기반 보정 (별도 구현)
           pass
   ```

#### 3.2.2 LensDetector (렌즈 검출)

**책임**: 렌즈의 중심점과 반경 검출

**인터페이스**:
```python
@dataclass
class LensDetection:
    """렌즈 검출 결과"""
    center_x: float        # 중심 x 좌표
    center_y: float        # 중심 y 좌표
    radius: float          # 반경 (픽셀)
    confidence: float      # 신뢰도 (0~1)
    method: str            # 사용된 검출 방법
    contour: np.ndarray    # 검출된 외곽선 (선택)


class LensDetector:
    """렌즈 중심 및 반경 검출"""

    def __init__(self, config: DetectorConfig):
        """
        Args:
            config: 검출 설정
                - method: 'hough' or 'contour' or 'hybrid'
                - min_radius: 최소 반경
                - max_radius: 최대 반경
        """
        self.config = config

    def detect(self, image: np.ndarray) -> LensDetection:
        """
        렌즈 검출 메인 함수

        Args:
            image: 입력 이미지 (BGR 또는 그레이)

        Returns:
            LensDetection 객체

        Raises:
            LensNotFoundError: 렌즈 검출 실패
        """
        pass

    def _detect_hough(self, gray: np.ndarray) -> List[tuple]:
        """Hough 원 검출"""
        pass

    def _detect_contour(self, gray: np.ndarray) -> LensDetection:
        """컨투어 기반 검출"""
        pass

    def _refine_center(self, image: np.ndarray,
                       initial: tuple) -> tuple:
        """중심점 정밀화 (sub-pixel 정확도)"""
        pass
```

**상세 구현**:

1. **Hough 원 검출**
   ```python
   def _detect_hough(self, gray: np.ndarray) -> List[tuple]:
       """
       HoughCircles를 이용한 원 검출

       Returns:
           [(x, y, r), ...] 검출된 원들
       """
       # 엣지 강화 (Canny)
       edges = cv2.Canny(gray,
                         threshold1=50,
                         threshold2=150,
                         apertureSize=3)

       # Hough 원 검출
       circles = cv2.HoughCircles(
           edges,
           cv2.HOUGH_GRADIENT,
           dp=1.2,                    # 해상도 비율
           minDist=100,               # 원 간 최소 거리
           param1=50,                 # Canny 상위 임계값
           param2=30,                 # 누산기 임계값
           minRadius=self.config.min_radius,
           maxRadius=self.config.max_radius
       )

       if circles is not None:
           circles = circles[0]  # 첫 번째 배열 선택
           # (x, y, r) 튜플 리스트로 변환
           return [(int(c[0]), int(c[1]), int(c[2]))
                   for c in circles]
       else:
           return []
   ```

2. **컨투어 기반 검출** (더 견고함)
   ```python
   def _detect_contour(self, gray: np.ndarray) -> LensDetection:
       """
       컨투어 검출 + 최소외접원 방법
       - Hough보다 부분 가림에 강함
       """
       # 이진화 (Otsu)
       _, binary = cv2.threshold(
           gray, 0, 255,
           cv2.THRESH_BINARY + cv2.THRESH_OTSU
       )

       # 모폴로지 연산으로 구멍 메우기
       kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
       closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

       # 컨투어 검출
       contours, _ = cv2.findContours(
           closed,
           cv2.RETR_EXTERNAL,
           cv2.CHAIN_APPROX_NONE
       )

       if not contours:
           raise LensNotFoundError("컨투어 검출 실패")

       # 가장 큰 컨투어를 렌즈로 가정
       lens_contour = max(contours, key=cv2.contourArea)

       # 최소외접원 계산
       (x, y), radius = cv2.minEnclosingCircle(lens_contour)

       # 신뢰도 계산: 컨투어가 얼마나 원에 가까운가?
       area_contour = cv2.contourArea(lens_contour)
       area_circle = np.pi * radius ** 2
       confidence = area_contour / area_circle  # 이상적: 1.0

       return LensDetection(
           center_x=x,
           center_y=y,
           radius=radius,
           confidence=confidence,
           method='contour',
           contour=lens_contour
       )
   ```

3. **하이브리드 방법** (권장)
   ```python
   def detect(self, image: np.ndarray) -> LensDetection:
       """
       Hough + Contour 조합으로 강건성 향상
       """
       gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
              if len(image.shape) == 3 else image

       # 1차: Hough 시도
       hough_circles = self._detect_hough(gray)

       # 2차: Contour 시도
       contour_result = self._detect_contour(gray)

       if len(hough_circles) > 0:
           # Hough와 Contour 결과 비교
           hx, hy, hr = hough_circles[0]  # 가장 확실한 원

           # 거리 계산
           dist = np.sqrt((hx - contour_result.center_x)**2 +
                          (hy - contour_result.center_y)**2)

           # 두 결과가 유사하면 평균 사용 (더 정확)
           if dist < 10:  # 10픽셀 이내
               final_x = (hx + contour_result.center_x) / 2
               final_y = (hy + contour_result.center_y) / 2
               final_r = (hr + contour_result.radius) / 2

               return LensDetection(
                   center_x=final_x,
                   center_y=final_y,
                   radius=final_r,
                   confidence=0.95,
                   method='hybrid'
               )
           else:
               # 불일치 시 신뢰도 높은 쪽 선택
               # (보통 contour가 더 안정적)
               return contour_result
       else:
           # Hough 실패 시 contour 결과 사용
           return contour_result
   ```

4. **Sub-pixel 정밀화** (선택사항, 고정밀도 필요 시)
   ```python
   def _refine_center(self, gray: np.ndarray,
                      initial_x: float, initial_y: float,
                      radius: float) -> tuple:
       """
       Circular Hough Transform의 중심을 sub-pixel 정밀도로 개선
       방법: 반경 방향 그래디언트의 교차점 계산
       """
       # 관심 영역 (중심 주변 작은 윈도우)
       window_size = 20
       x, y = int(initial_x), int(initial_y)
       roi = gray[
           max(0, y-window_size):min(gray.shape[0], y+window_size),
           max(0, x-window_size):min(gray.shape[1], x+window_size)
       ]

       # 그래디언트 계산
       gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
       gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)

       # 중심점 재계산 (weighted centroid)
       # (상세 수식 생략 - moment 기반 계산)

       refined_x = initial_x + dx  # sub-pixel 보정
       refined_y = initial_y + dy

       return refined_x, refined_y
   ```

#### 3.2.3 RadialProfiler (r-프로파일 추출)

**책임**: 극좌표 변환 및 반경별 색상 프로파일 생성

**인터페이스**:
```python
@dataclass
class RadialProfile:
    """반경 방향 색상 프로파일"""
    r_normalized: np.ndarray   # 정규화된 반경 (0~1)
    L_profile: np.ndarray      # L* 프로파일
    a_profile: np.ndarray      # a* 프로파일
    b_profile: np.ndarray      # b* 프로파일
    std_L: np.ndarray          # 각 r에서 L* 표준편차
    std_a: np.ndarray          # a* 표준편차
    std_b: np.ndarray          # b* 표준편차
    pixel_count: np.ndarray    # 각 링의 픽셀 수


class RadialProfiler:
    """극좌표 기반 색상 프로파일 추출"""

    def __init__(self, config: ProfilerConfig):
        """
        Args:
            config:
                - r_start: 시작 반경 (정규화, 예: 0.3)
                - r_end: 종료 반경 (정규화, 예: 1.0)
                - r_step: 반경 간격 (예: 0.01)
                - theta_samples: 각도 샘플 수 (예: 360)
        """
        self.config = config
        self.r_bins = np.arange(
            config.r_start,
            config.r_end,
            config.r_step
        )

    def extract_profile(self,
                        image: np.ndarray,
                        lens: LensDetection) -> RadialProfile:
        """
        r-프로파일 추출

        Args:
            image: BGR 이미지
            lens: 렌즈 검출 결과

        Returns:
            RadialProfile 객체
        """
        pass

    def _cartesian_to_polar(self,
                            image: np.ndarray,
                            center: tuple,
                            radius: float) -> np.ndarray:
        """직교좌표 → 극좌표 변환"""
        pass

    def _compute_ring_statistics(self,
                                 image_lab: np.ndarray,
                                 mask: np.ndarray) -> dict:
        """링 영역의 색상 통계 계산"""
        pass
```

**상세 구현**:

1. **극좌표 변환**
   ```python
   def _cartesian_to_polar(self,
                           image: np.ndarray,
                           cx: float, cy: float,
                           radius: float) -> dict:
       """
       직교좌표계 이미지를 극좌표계로 변환

       Returns:
           {
               'polar_image': (r_bins, theta_bins, 3) 극좌표 이미지,
               'r_map': 각 픽셀의 정규화 반경,
               'theta_map': 각 픽셀의 각도
           }
       """
       h, w = image.shape[:2]

       # 각 픽셀의 (x, y) 좌표 생성
       y_grid, x_grid = np.ogrid[0:h, 0:w]

       # 중심으로부터 거리 계산
       dx = x_grid - cx
       dy = y_grid - cy
       r_map = np.sqrt(dx**2 + dy**2)

       # 정규화 (0~1)
       r_normalized = r_map / radius

       # 각도 계산 (0~2π)
       theta_map = np.arctan2(dy, dx)
       theta_map[theta_map < 0] += 2 * np.pi  # 0~2π 범위로

       return {
           'r_normalized': r_normalized,
           'theta_map': theta_map
       }
   ```

2. **링별 색상 통계 계산**
   ```python
   def extract_profile(self,
                       image: np.ndarray,
                       lens: LensDetection) -> RadialProfile:
       """
       메인 프로파일 추출 함수
       """
       # BGR → LAB 변환
       image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

       # 극좌표 맵 생성
       polar_data = self._cartesian_to_polar(
           image, lens.center_x, lens.center_y, lens.radius
       )
       r_norm = polar_data['r_normalized']

       # 각 반경 bin에 대해 통계 계산
       n_bins = len(self.r_bins)
       L_profile = np.zeros(n_bins)
       a_profile = np.zeros(n_bins)
       b_profile = np.zeros(n_bins)
       std_L = np.zeros(n_bins)
       std_a = np.zeros(n_bins)
       std_b = np.zeros(n_bins)
       pixel_count = np.zeros(n_bins, dtype=int)

       for i, r in enumerate(self.r_bins):
           # 현재 반경 링의 마스크 생성
           # (r - Δr/2) ≤ r_norm < (r + Δr/2)
           r_min = r - self.config.r_step / 2
           r_max = r + self.config.r_step / 2

           mask = (r_norm >= r_min) & (r_norm < r_max)

           if np.sum(mask) == 0:
               # 해당 링에 픽셀이 없으면 NaN
               L_profile[i] = np.nan
               a_profile[i] = np.nan
               b_profile[i] = np.nan
               continue

           # 링 내 픽셀들의 LAB 값 추출
           L_pixels = image_lab[mask, 0]
           a_pixels = image_lab[mask, 1]
           b_pixels = image_lab[mask, 2]

           # 평균 및 표준편차 계산
           L_profile[i] = np.mean(L_pixels)
           a_profile[i] = np.mean(a_pixels)
           b_profile[i] = np.mean(b_pixels)

           std_L[i] = np.std(L_pixels)
           std_a[i] = np.std(a_pixels)
           std_b[i] = np.std(b_pixels)

           pixel_count[i] = np.sum(mask)

       return RadialProfile(
           r_normalized=self.r_bins,
           L_profile=L_profile,
           a_profile=a_profile,
           b_profile=b_profile,
           std_L=std_L,
           std_a=std_a,
           std_b=std_b,
           pixel_count=pixel_count
       )
   ```

3. **프로파일 스무딩** (노이즈 제거)
   ```python
   def smooth_profile(self,
                      profile: RadialProfile,
                      window_size: int = 5) -> RadialProfile:
       """
       이동 평균 또는 Savitzky-Golay 필터로 스무딩

       Args:
           profile: 원본 프로파일
           window_size: 윈도우 크기 (홀수)

       Returns:
           스무딩된 프로파일
       """
       from scipy.signal import savgol_filter

       # NaN 제거 (선형 보간)
       L_clean = self._interpolate_nans(profile.L_profile)
       a_clean = self._interpolate_nans(profile.a_profile)
       b_clean = self._interpolate_nans(profile.b_profile)

       # Savitzky-Golay 필터 적용
       # (다항식 차수 3, 윈도우 크기 지정)
       L_smooth = savgol_filter(L_clean, window_size, polyorder=3)
       a_smooth = savgol_filter(a_clean, window_size, polyorder=3)
       b_smooth = savgol_filter(b_clean, window_size, polyorder=3)

       return RadialProfile(
           r_normalized=profile.r_normalized,
           L_profile=L_smooth,
           a_profile=a_smooth,
           b_profile=b_smooth,
           std_L=profile.std_L,
           std_a=profile.std_a,
           std_b=profile.std_b,
           pixel_count=profile.pixel_count
       )

   def _interpolate_nans(self, arr: np.ndarray) -> np.ndarray:
       """NaN을 선형 보간으로 채움"""
       mask = np.isnan(arr)
       if not np.any(mask):
           return arr

       x = np.arange(len(arr))
       arr_interp = np.interp(x, x[~mask], arr[~mask])
       return arr_interp
   ```

#### 3.2.4 ZoneSegmenter (Zone 분할)

**책임**: 프로파일의 변곡점 검출 및 잉크 영역 분할

**인터페이스**:
```python
@dataclass
class Zone:
    """색상 영역"""
    name: str              # 'A', 'A-B', 'B', 'B-C', 'C'
    r_start: float         # 시작 반경 (정규화)
    r_end: float           # 종료 반경
    mean_L: float          # 평균 L*
    mean_a: float          # 평균 a*
    mean_b: float          # 평균 b*
    std_L: float           # 표준편차
    std_a: float
    std_b: float
    zone_type: str         # 'pure' or 'mix'


class ZoneSegmenter:
    """변곡점 기반 Zone 자동 분할"""

    def __init__(self, config: SegmenterConfig):
        """
        Args:
            config:
                - min_zone_width: 최소 zone 폭 (정규화)
                - smoothing_window: 프로파일 스무딩 윈도우
                - polyorder: Savitzky-Golay 필터 다항식 차수
                - min_gradient_threshold: 그래디언트 피크 검출 최소 임계값
                - min_delta_e_threshold: ΔE 피크 검출 최소 임계값
                - transition_buffer_px: 혼합 구간 버퍼 (픽셀)
        """
        self.config = config

    def segment(self,
                profile: RadialProfile,
                expected_zones: int = None) -> List[Zone]:
        """
        프로파일을 Zone으로 분할

        Args:
            profile: 색상 프로파일
            expected_zones: 기대하는 Zone 개수 (SKU 설정으로부터의 힌트)

        Returns:
            Zone 리스트 (바깥→안쪽 순서)
        """
        pass

    def _detect_boundaries_gradient(self, profile: RadialProfile) -> List[float]:
        """그래디언트 기반 경계 검출"""
        pass

    def _detect_boundaries_delta_e(self, profile: RadialProfile) -> List[float]:
        """ΔE 기반 경계 검출"""
        pass

    def _detect_boundaries_kmeans(self, profile: RadialProfile, k: int) -> List[float]:
        """K-means 클러스터링 기반 경계 검출"""
        pass

    def _fallback_uniform_split(self, profile: RadialProfile, n_zones: int) -> List[float]:
        """균등 분할 Fallback"""
        pass

    def _merge_narrow_zones(self, zones: List[Zone]) -> List[Zone]:
        """최소 폭 미만 Zone 병합"""
        pass

    def _apply_transition_buffer(self,
                                 zones: List[Zone],
                                 profile_r_length: int) -> List[Zone]:
        """Transition buffer 적용"""
        pass
```

**상세 구현 (개선된 ZoneSegmenter)**:

1.  **전처리 (Smoothing)**
    ```python
    def segment(self,
                profile: RadialProfile,
                expected_zones: int = None) -> List[Zone]:
        """
        메인 Zone 분할 함수 (개선된 로직)
        """
        # 1. 프로파일 스무딩 (노이즈 제거)
        from scipy.signal import savgol_filter
        
        # NaN 값 처리: 선형 보간 또는 NaN 포함 값은 Zone 분할에서 제외
        L_profile_clean = np.nan_to_num(profile.L_profile, nan=np.nanmean(profile.L_profile))
        a_profile_clean = np.nan_to_num(profile.a_profile, nan=np.nanmean(profile.a_profile))
        b_profile_clean = np.nan_to_num(profile.b_profile, nan=np.nanmean(profile.b_profile))

        profile_L_smooth = savgol_filter(L_profile_clean, self.config.smoothing_window, self.config.polyorder)
        profile_a_smooth = savgol_filter(a_profile_clean, self.config.smoothing_window, self.config.polyorder)
        profile_b_smooth = savgol_filter(b_profile_clean, self.config.smoothing_window, self.config.polyorder)

        # 스무딩된 프로파일로 임시 RadialProfile 객체 생성 (경계 검출용)
        smoothed_profile_for_detection = RadialProfile(
            r_normalized=profile.r_normalized,
            L_profile=profile_L_smooth,
            a_profile=profile_a_smooth,
            b_profile=profile_b_smooth,
            std_L=profile.std_L, # 표준편차는 원본 사용
            std_a=profile.std_a,
            std_b=profile.std_b,
            pixel_count=profile.pixel_count
        )

        boundaries = []

        # 2. 경계 검출 전략 (우선순위 기반)
        # 2.1. Gradient 및 Delta E 기반 (Hybrid)
        gradient_boundaries = self._detect_boundaries_gradient(smoothed_profile_for_detection)
        delta_e_boundaries = self._detect_boundaries_delta_e(smoothed_profile_for_detection)

        # 두 방법의 경계점을 통합하고 중복 제거
        combined_boundaries_r_norm = np.unique(np.concatenate((gradient_boundaries, delta_e_boundaries)))
        
        # 3. K-means 클러스터링 기반 Zone 분할 (fallback 혹은 refine)
        if expected_zones and len(combined_boundaries_r_norm) != expected_zones - 1:
            # 예상 Zone 개수와 다르면 K-means로 재시도
            kmeans_boundaries = self._detect_boundaries_kmeans(profile, expected_zones)
            if kmeans_boundaries:
                boundaries = kmeans_boundaries
            else:
                # 4. Fallback 1: 예상 Zone 개수 기반 균등 분할
                boundaries = self._fallback_uniform_split(profile, expected_zones)
        elif len(combined_boundaries_r_norm) > 0:
            boundaries = combined_boundaries_r_norm
        else:
            # 5. Fallback 2: 기대 Zone 개수(힌트)가 없거나 검출된 경계가 없는 경우
            # (기본 3분할 혹은 expected_zones 기반 균등 분할)
            n_fallback_zones = expected_zones if expected_zones else 3
            boundaries = self._fallback_uniform_split(profile, n_fallback_zones)

        # 6. 경계점 정렬 (바깥 → 안쪽)
        boundaries = sorted(list(set(boundaries)), reverse=True) # 중복 제거 후 정렬

        # 7. Zone 생성 및 레이블링 (기존 로직 활용)
        zones = []
        # ... (기존 Zone 생성 및 평균 색상 계산 로직)
        # 이 부분은 변경이 없으므로, 기존 코드를 재사용하거나 새롭게 정의
        # 편의상, 이 문서에서는 Zone 생성 및 평균 색상 계산 로직을 간략히 생략하고
        # _create_zones_from_boundaries 헬퍼 함수를 가정합니다.
        zones = self._create_zones_from_boundaries(boundaries, profile)


        # 8. 최소 폭 미만 Zone 병합 (기존 로직 유지)
        zones = self._merge_narrow_zones(zones)

        # 9. Transition Buffer 적용
        zones = self._apply_transition_buffer(zones, len(profile.r_normalized))

        return zones

    def _create_zones_from_boundaries(self, boundaries: List[float], profile: RadialProfile) -> List[Zone]:
        # (기존 Zone 생성 및 평균 색상 계산 로직)
        # 이 헬퍼 함수는 위에 정의된 `segment` 함수의 기존 Zone 생성 로직을 캡슐화합니다.
        # 실제 구현에서는 `segment` 함수 내부에 이 로직이 직접 포함되거나
        # 이 헬퍼 함수가 RadialProfile, boundaries를 받아 List[Zone]을 반환하도록 구현됩니다.
        
        # 임시 구현 (실제 로직으로 대체 필요)
        all_zones = []
        if not boundaries: # 경계가 없으면 전체를 하나의 Zone으로
            mean_L, mean_a, mean_b, std_L, std_a, std_b = self._calculate_zone_stats(profile, 0, len(profile.r_normalized))
            all_zones.append(Zone(
                name="Full", r_start=profile.r_normalized[0], r_end=profile.r_normalized[-1],
                mean_L=mean_L, mean_a=mean_a, mean_b=mean_b, std_L=std_L, std_a=std_a, std_b=std_b,
                zone_type='pure'
            ))
            return all_zones

        
        # 경계점을 기준으로 Zone 생성
        # boundaries 리스트에는 r_normalized의 값이 들어있다.
        # 이 값은 프로파일의 인덱스와 매칭시켜야 한다.
        r_indices = np.searchsorted(profile.r_normalized, boundaries)
        
        # 시작과 끝 인덱스 추가
        all_indices = np.unique(np.concatenate(([0], r_indices, [len(profile.r_normalized) - 1])))
        all_indices = sorted(all_indices)

        for i in range(len(all_indices) - 1):
            start_idx = all_indices[i]
            end_idx = all_indices[i+1]
            
            # Zone의 반경 범위는 r_normalized의 실제 값을 사용
            r_start_norm = profile.r_normalized[start_idx]
            r_end_norm = profile.r_normalized[end_idx]

            mean_L, mean_a, mean_b, std_L, std_a, std_b = self._calculate_zone_stats(profile, start_idx, end_idx)
            
            # 레이블 생성 로직은 나중에 추가
            zone_name = f"Zone{i+1}"
            
            all_zones.append(Zone(
                name=zone_name,
                r_start=r_start_norm,
                r_end=r_end_norm,
                mean_L=mean_L,
                mean_a=mean_a,
                mean_b=mean_b,
                std_L=std_L,
                std_a=std_a,
                std_b=std_b,
                zone_type='pure' # 초기화는 pure, 나중에 mix 판단
            ))
            
        return all_zones
    
    def _calculate_zone_stats(self, profile: RadialProfile, start_idx: int, end_idx: int):
        # 주어진 인덱스 범위에 해당하는 프로파일 통계 계산
        L_slice = profile.L_profile[start_idx:end_idx]
        a_slice = profile.a_profile[start_idx:end_idx]
        b_slice = profile.b_profile[start_idx:end_idx]
        
        # NaN이 포함될 수 있으므로 nanmean 사용
        mean_L = np.nanmean(L_slice)
        mean_a = np.nanmean(a_slice)
        mean_b = np.nanmean(b_slice)
        
        std_L = np.nanstd(L_slice)
        std_a = np.nanstd(a_slice)
        std_b = np.nanstd(b_slice)
        
        return mean_L, mean_a, mean_b, std_L, std_a, std_b


    def _detect_boundaries_gradient(self,
                                  profile: RadialProfile) -> List[float]:
        """
        a* 프로파일의 그래디언트 급변 지점 검출 (스무딩된 프로파일 사용)
        """
        from scipy.signal import find_peaks

        gradient_a = np.gradient(profile.a_profile)
        
        # 노이즈가 이미 스무딩되어 있으므로 추가 스무딩은 생략
        # 그래디언트 절댓값에서 피크 검출
        peaks, properties = find_peaks(
            np.abs(gradient_a),
            height=self.config.min_gradient_threshold,     # 최소 기울기 임계값
            distance=int(self.config.min_zone_width * len(profile.r_normalized)) # 최소 Zone 폭
        )

        inflection_r = profile.r_normalized[peaks]
        return inflection_r.tolist()

    def _detect_boundaries_delta_e(self,
                                  profile: RadialProfile) -> List[float]:
        """
        연속된 두 반경 간 ΔE 계산, 급변 지점 검출 (스무딩된 프로파일 사용)
        """
        from colormath.color_objects import LabColor
        from colormath.color_diff import delta_e_cie2000

        r = profile.r_normalized
        L = profile.L_profile
        a = profile.a_profile
        b = profile.b_profile

        delta_e_profile = []
        for i in range(len(r) - 1):
            color1 = LabColor(L[i], a[i], b[i])
            color2 = LabColor(L[i+1], a[i+1], b[i+1])
            de = delta_e_cie2000(color1, color2)
            delta_e_profile.append(de)

        delta_e_profile = np.array(delta_e_profile)

        from scipy.signal import find_peaks
        peaks, _ = find_peaks(
            delta_e_profile,
            height=self.config.min_delta_e_threshold,  # 최소 ΔE 임계값
            distance=int(self.config.min_zone_width * len(r))
        )

        inflection_r = r[peaks]
        return inflection_r.tolist()

    def _detect_boundaries_kmeans(self, profile: RadialProfile, k: int) -> List[float]:
        """
        K-means 클러스터링 기반 경계 검출
        프로파일 데이터를 K개의 색상 그룹으로 군집화하여 경계를 찾음.
        """
        if k <= 1: # Zone이 1개 이하면 경계가 없음
            return []

        from sklearn.cluster import KMeans

        # LAB 프로파일 데이터를 클러스터링에 사용
        # L, a, b 값을 함께 사용하여 K-means를 수행
        lab_data = np.stack([profile.L_profile, profile.a_profile, profile.b_profile], axis=-1)
        
        # NaN 값은 클러스터링에 방해가 되므로 제거하거나 보간
        # 여기서는 간단하게 NaN을 제외하고 클러스터링을 수행
        valid_indices = ~np.isnan(lab_data).any(axis=1)
        valid_lab_data = lab_data[valid_indices]
        valid_r_normalized = profile.r_normalized[valid_indices]

        if len(valid_lab_data) < k: # 유효한 데이터가 클러스터 수보다 적으면 클러스터링 불가
            return []

        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(valid_lab_data)
        labels = kmeans.labels_

        # 레이블이 변경되는 지점을 경계로 간주
        boundaries_r = []
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                # 경계는 두 레이블의 중간 지점으로 (반경 정규화 값)
                boundaries_r.append((valid_r_normalized[i] + valid_r_normalized[i-1]) / 2)
        
        return sorted(list(set(boundaries_r)))


    def _fallback_uniform_split(self, profile: RadialProfile, n_zones: int) -> List[float]:
        """
        주어진 Zone 개수에 따라 프로파일을 균등 분할
        """
        if n_zones <= 1:
            return []
        
        # 0 (중심)부터 1 (외곽)까지 균등하게 분할
        # n_zones 개수만큼 영역을 만들려면 n_zones - 1 개의 경계가 필요
        boundaries = np.linspace(profile.r_normalized[0], profile.r_normalized[-1], n_zones + 1)
        
        # 시작과 끝을 제외한 경계점 반환
        return boundaries[1:-1].tolist()
    
    def _merge_narrow_zones(self, zones: List[Zone]) -> List[Zone]:
        """
        최소 폭 미만 Zone을 인접 Zone과 병합 (구현 예정)
        """
        merged_zones = []
        # (TODO: 구현 필요)
        # self.config.min_zone_width를 활용하여 Zone 폭이 너무 좁은 경우 병합
        return zones

    def _apply_transition_buffer(self,
                                 zones: List[Zone],
                                 profile_r_length: int) -> List[Zone]:
        """
        Transition buffer (혼합 구간) 적용
        경계 주변 픽셀을 색상 평가에서 제외하거나 가중치를 낮춤
        """
        if not self.config.transition_buffer_px or self.config.transition_buffer_px <= 0:
            return zones # 버퍼 설정이 없으면 변경 없음

        buffered_zones = []
        # 각 Zone의 r_start, r_end를 픽셀 인덱스로 변환해야함
        # 프로파일의 총 길이(profile_r_length)를 활용
        
        # 이 로직은 Zone의 r_start, r_end가 정규화된 값임을 가정하고
        # 이를 다시 픽셀 인덱스로 변환하여 버퍼를 적용해야 함
        
        # (TODO: 구현 필요)
        # for zone in zones:
        #     # r_start, r_end를 픽셀 인덱스로 변환
        #     start_px = int(zone.r_start * profile_r_length)
        #     end_px = int(zone.r_end * profile_r_length)
            
        #     # 버퍼 적용 (예: start_px + buffer_px, end_px - buffer_px)
        #     # 버퍼가 Zone의 폭보다 커지지 않도록 주의
        #     new_start_px = start_px + self.config.transition_buffer_px
        #     new_end_px = end_px - self.config.transition_buffer_px
            
        #     # ... 새로운 Zone 객체 생성 및 반환
            
        return zones


#### 3.2.5 ColorEvaluator (색상 평가 및 판정)

**책임**: ΔE 계산 및 OK/NG 판정

**인터페이스**:
```python
@dataclass
class InspectionResult:
    """검사 결과"""
    sku: str                        # SKU 코드
    timestamp: datetime             # 검사 시간
    judgment: str                   # 'OK' or 'NG'
    overall_delta_e: float          # 전체 평균 ΔE
    zone_results: List[ZoneResult]  # Zone별 결과
    ng_reasons: List[str]           # NG 이유 목록
    confidence: float               # 판정 신뢰도


@dataclass
class ZoneResult:
    """Zone별 평가 결과"""
    zone_name: str
    measured_lab: tuple             # (L*, a*, b*)
    target_lab: tuple               # 기준값
    delta_e: float                  # ΔE2000
    threshold: float                # 허용 ΔE
    is_ok: bool                     # 이 zone의 판정


class ColorEvaluator:
    """색상 평가 및 OK/NG 판정"""

    def __init__(self, config_manager: SkuConfigManager):
        """
        Args:
            config_manager: SKU별 기준값 관리자
        """
        self.config_manager = config_manager

    def evaluate(self,
                 zones: List[Zone],
                 sku: str) -> InspectionResult:
        """
        Zone 리스트를 평가하여 OK/NG 판정

        Args:
            zones: 측정된 Zone 리스트
            sku: 제품 SKU 코드

        Returns:
            InspectionResult
        """
        pass

    def calculate_delta_e(self,
                          lab1: tuple,
                          lab2: tuple,
                          method='cie2000') -> float:
        """ΔE 계산"""
        pass
```

**상세 구현**:

1. **ΔE 계산** (CIEDE2000)
   ```python
   def calculate_delta_e(self,
                         lab1: tuple,
                         lab2: tuple,
                         method='cie2000') -> float:
       """
       색차 계산 (CIE ΔE2000 권장)

       Args:
           lab1: (L1, a1, b1)
           lab2: (L2, a2, b2)
           method: 'cie1976', 'cie1994', or 'cie2000'

       Returns:
           ΔE 값
       """
       from colormath.color_objects import LabColor
       from colormath.color_diff import (
           delta_e_cie1976,
           delta_e_cie1994,
           delta_e_cie2000
       )

       color1 = LabColor(lab1[0], lab1[1], lab1[2])
       color2 = LabColor(lab2[0], lab2[1], lab2[2])

       if method == 'cie2000':
           return delta_e_cie2000(color1, color2)
       elif method == 'cie1994':
           return delta_e_cie1994(color1, color2)
       else:  # cie1976
           return delta_e_cie1976(color1, color2)
   ```

2. **Zone별 평가**
   ```python
   def evaluate(self,
                zones: List[Zone],
                sku: str) -> InspectionResult:
       """
       메인 평가 함수
       """
       # SKU 기준값 로드
       sku_config = self.config_manager.get_config(sku)

       if sku_config is None:
           raise ValueError(f"SKU {sku}의 기준값이 등록되지 않음")

       # 각 Zone 평가
       zone_results = []
       ng_reasons = []
       delta_e_list = []

       for zone in zones:
           # 해당 zone의 기준값 찾기
           target = sku_config.get_zone_target(zone.name)

           if target is None:
               # 기준값 없으면 스킵 (경고 로그)
               logger.warning(f"Zone {zone.name}의 기준값 없음")
               continue

           # 측정값
           measured_lab = (zone.mean_L, zone.mean_a, zone.mean_b)
           target_lab = (target['L'], target['a'], target['b'])

           # ΔE 계산
           de = self.calculate_delta_e(measured_lab, target_lab)
           delta_e_list.append(de)

           # 허용치 비교
           threshold = target.get('delta_e_threshold',
                                  sku_config.default_threshold)
           is_ok = de <= threshold

           zone_result = ZoneResult(
               zone_name=zone.name,
               measured_lab=measured_lab,
               target_lab=target_lab,
               delta_e=de,
               threshold=threshold,
               is_ok=is_ok
           )
           zone_results.append(zone_result)

           # NG 사유 수집
           if not is_ok:
               ng_reasons.append(
                   f"Zone {zone.name}: ΔE={de:.2f} > {threshold:.2f}"
               )

       # 전체 판정
       overall_ok = all(zr.is_ok for zr in zone_results)
       overall_delta_e = np.mean(delta_e_list) if delta_e_list else 0.0

       return InspectionResult(
           sku=sku,
           timestamp=datetime.now(),
           judgment='OK' if overall_ok else 'NG',
           overall_delta_e=overall_delta_e,
           zone_results=zone_results,
           ng_reasons=ng_reasons,
           confidence=self._calculate_confidence(zone_results)
       )

   def _calculate_confidence(self,
                             zone_results: List[ZoneResult]) -> float:
       """
       판정 신뢰도 계산

       신뢰도 = 1 - (ΔE / threshold)의 가중 평균
       """
       if not zone_results:
           return 0.0

       ratios = [zr.delta_e / zr.threshold for zr in zone_results]
       avg_ratio = np.mean(ratios)

       # 0~1 범위로 정규화
       confidence = max(0.0, 1.0 - avg_ratio)
       return confidence
   ```

3. **Mix Zone 추가 검증**
   ```python
   def evaluate_with_mix_check(self,
                               zones: List[Zone],
                               sku: str) -> InspectionResult:
       """
       Mix zone의 혼합 품질도 검증하는 확장 평가
       """
       # 기본 평가
       result = self.evaluate(zones, sku)

       # Mix zone 추가 검증
       for i, zone in enumerate(zones):
           if zone.zone_type == 'mix':
               # 앞뒤 pure zone 찾기
               before_zone = zones[i-1] if i > 0 else None
               after_zone = zones[i+1] if i < len(zones)-1 else None

               if before_zone and after_zone:
                   # Mix 품질 평가
                   segmenter = ZoneSegmenter(self.config)
                   mix_eval = segmenter.evaluate_mix_zone(
                       zone, before_zone, after_zone
                   )

                   if not mix_eval['is_valid']:
                       result.ng_reasons.append(
                           f"Mix zone {zone.name}: "
                           f"혼합 편차 {mix_eval['distance_from_line']:.2f}"
                       )
                       result.judgment = 'NG'

       return result
   ```

#### 3.2.6 Visualizer (시각화)

**책임**: 검사 결과 시각화 및 리포트 생성

**인터페이스**:
```python
class Visualizer:
    """결과 시각화 및 리포트 생성"""

    def __init__(self, config: VisualizerConfig):
        """
        Args:
            config:
                - output_dir: 출력 디렉토리
                - dpi: 이미지 해상도
                - colormap: 색상 맵
        """
        self.config = config

    def create_overlay_image(self,
                             original: np.ndarray,
                             lens: LensDetection,
                             zones: List[Zone],
                             result: InspectionResult) -> np.ndarray:
        """
        원본 이미지에 zone 경계 및 판정 결과 오버레이

        Returns:
            오버레이된 이미지
        """
        pass

    def plot_radial_profile(self,
                            profile: RadialProfile,
                            zones: List[Zone],
                            save_path: Path = None) -> Figure:
        """
        r-프로파일 그래프 생성 (L*, a*, b*)

        Returns:
            matplotlib Figure
        """
        pass

    def create_delta_e_heatmap(self,
                               image: np.ndarray,
                               lens: LensDetection,
                               zones: List[Zone],
                               sku_config: SkuConfig) -> np.ndarray:
        """
        픽셀별 ΔE 히트맵 생성

        Returns:
            히트맵 이미지
        """
        pass

    def generate_report(self,
                        result: InspectionResult,
                        images: dict) -> Path:
        """
        종합 리포트 생성 (HTML 또는 PDF)

        Args:
            result: 검사 결과
            images: {'overlay': ..., 'profile': ..., 'heatmap': ...}

        Returns:
            생성된 리포트 파일 경로
        """
        pass
```

**상세 구현**:

1. **오버레이 이미지 생성**
   ```python
   def create_overlay_image(self,
                            original: np.ndarray,
                            lens: LensDetection,
                            zones: List[Zone],
                            result: InspectionResult) -> np.ndarray:
       """
       원본 이미지 위에 zone 경계와 판정 결과 표시
       """
       overlay = original.copy()
       cx, cy = int(lens.center_x), int(lens.center_y)
       R = lens.radius

       # 1. 렌즈 외곽선 그리기
       cv2.circle(overlay, (cx, cy), int(R), (255, 255, 255), 2)

       # 2. Zone 경계 그리기
       for zone in zones:
           r_start = zone.r_start * R
           r_end = zone.r_end * R

           # 경계원 그리기
           color = (0, 255, 0) if zone.zone_type == 'pure' else (255, 255, 0)
           cv2.circle(overlay, (cx, cy), int(r_end), color, 1)

           # Zone 레이블 표시
           label_r = (r_start + r_end) / 2
           label_x = cx + int(label_r * 0.7)  # 45도 방향
           label_y = cy - int(label_r * 0.7)

           # 배경 박스
           text = zone.name
           font = cv2.FONT_HERSHEY_SIMPLEX
           font_scale = 0.5
           thickness = 1
           (text_w, text_h), baseline = cv2.getTextSize(
               text, font, font_scale, thickness
           )

           cv2.rectangle(
               overlay,
               (label_x - 2, label_y - text_h - 2),
               (label_x + text_w + 2, label_y + baseline + 2),
               (0, 0, 0),
               -1
           )

           cv2.putText(
               overlay,
               text,
               (label_x, label_y),
               font,
               font_scale,
               (255, 255, 255),
               thickness
           )

       # 3. 판정 결과 표시
       judgment_color = (0, 255, 0) if result.judgment == 'OK' else (0, 0, 255)
       judgment_text = f"{result.judgment} (ΔE={result.overall_delta_e:.2f})"

       cv2.rectangle(
           overlay,
           (10, 10),
           (300, 80),
           (0, 0, 0),
           -1
       )

       cv2.putText(
           overlay,
           judgment_text,
           (20, 50),
           cv2.FONT_HERSHEY_SIMPLEX,
           1.0,
           judgment_color,
           2
       )

       # 4. NG 사유 표시 (NG인 경우)
       if result.judgment == 'NG' and result.ng_reasons:
           y_offset = 100
           for reason in result.ng_reasons[:3]:  # 최대 3개만
               cv2.putText(
                   overlay,
                   reason,
                   (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5,
                   (0, 0, 255),
                   1
               )
               y_offset += 25

       return overlay
   ```

2. **프로파일 그래프**
   ```python
   def plot_radial_profile(self,
                           profile: RadialProfile,
                           zones: List[Zone],
                           save_path: Path = None) -> Figure:
       """
       L*, a*, b* vs r 그래프
       """
       import matplotlib.pyplot as plt

       fig, axes = plt.subplots(3, 1, figsize=(10, 12))

       r = profile.r_normalized
       profiles = [
           (profile.L_profile, 'L*', 'black'),
           (profile.a_profile, 'a*', 'red'),
           (profile.b_profile, 'b*', 'blue')
       ]

       for ax, (data, label, color) in zip(axes, profiles):
           # 프로파일 플롯
           ax.plot(r, data, color=color, linewidth=2, label=label)

           # 표준편차 영역 (신뢰 구간)
           if label == 'L*':
               std = profile.std_L
           elif label == 'a*':
               std = profile.std_a
           else:
               std = profile.std_b

           ax.fill_between(
               r,
               data - std,
               data + std,
               color=color,
               alpha=0.2,
               label=f'{label} ± σ'
           )

           # Zone 경계 표시
           for zone in zones:
               ax.axvline(zone.r_start, color='gray',
                          linestyle='--', alpha=0.5)
               ax.axvline(zone.r_end, color='gray',
                          linestyle='--', alpha=0.5)

               # Zone 레이블
               mid_r = (zone.r_start + zone.r_end) / 2
               ax.text(mid_r, ax.get_ylim()[1] * 0.95,
                       zone.name,
                       ha='center',
                       fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

           ax.set_xlabel('Normalized Radius (r)')
           ax.set_ylabel(label)
           ax.grid(True, alpha=0.3)
           ax.legend()
           ax.set_title(f'{label} Radial Profile')

       plt.tight_layout()

       if save_path:
           plt.savefig(save_path, dpi=self.config.dpi)

       return fig
   ```

3. **ΔE 히트맵**
   ```python
   def create_delta_e_heatmap(self,
                              image: np.ndarray,
                              lens: LensDetection,
                              zones: List[Zone],
                              sku_config: SkuConfig) -> np.ndarray:
       """
       각 픽셀의 ΔE를 색상으로 표현한 히트맵
       """
       # LAB 변환
       image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

       # 극좌표 맵 생성
       h, w = image.shape[:2]
       y_grid, x_grid = np.ogrid[0:h, 0:w]
       dx = x_grid - lens.center_x
       dy = y_grid - lens.center_y
       r_map = np.sqrt(dx**2 + dy**2) / lens.radius

       # 각 픽셀의 ΔE 계산
       delta_e_map = np.zeros((h, w), dtype=np.float32)

       for zone in zones:
           # 해당 zone의 마스크
           mask = (r_map >= zone.r_end) & (r_map < zone.r_start)

           if np.sum(mask) == 0:
               continue

           # 기준값
           target = sku_config.get_zone_target(zone.name)
           if target is None:
               continue

           target_lab = np.array([target['L'], target['a'], target['b']])

           # 픽셀별 ΔE 계산 (벡터화)
           pixels_lab = image_lab[mask].astype(np.float32)
           # CIE76 근사 (빠른 계산)
           delta_e_pixels = np.sqrt(
               ((pixels_lab[:, 0] - target_lab[0]) ** 2) +
               ((pixels_lab[:, 1] - target_lab[1]) ** 2) +
               ((pixels_lab[:, 2] - target_lab[2]) ** 2)
           )

           delta_e_map[mask] = delta_e_pixels

       # 히트맵 색상 적용
       # ΔE 범위: 0 (blue) → threshold (green) → 2*threshold (red)
       normalized = np.clip(delta_e_map / (sku_config.default_threshold * 2), 0, 1)
       heatmap = cv2.applyColorMap(
           (normalized * 255).astype(np.uint8),
           cv2.COLORMAP_JET
       )

       # 렌즈 외부는 검은색
       lens_mask = r_map <= 1.0
       heatmap[~lens_mask] = 0

       return heatmap
   ```

#### 3.2.7 Logger (데이터 저장 및 로깅)

**책임**: 검사 결과 저장, NG 샘플 관리, 통계 생성

**인터페이스**:
```python
class Logger:
    """검사 결과 로깅 및 데이터베이스 관리"""

    def __init__(self, config: LoggerConfig, db_path: Path):
        """
        Args:
            config: 로깅 설정
            db_path: SQLite DB 경로
        """
        self.config = config
        self.db = self._init_database(db_path)

    def _init_database(self, db_path: Path) -> sqlalchemy.Engine:
        """데이터베이스 초기화 및 테이블 생성"""
        pass

    def log_result(self,
                   result: InspectionResult,
                   image_path: Path = None,
                   overlay_path: Path = None) -> int:
        """
        검사 결과 저장

        Returns:
            생성된 레코드 ID
        """
        pass

    def save_ng_sample(self,
                       result: InspectionResult,
                       original_image: np.ndarray,
                       overlay_image: np.ndarray) -> Path:
        """
        NG 샘플 이미지 저장

        Returns:
            저장된 파일 경로
        """
        pass

    def get_statistics(self,
                       sku: str = None,
                       start_date: datetime = None,
                       end_date: datetime = None) -> dict:
        """
        통계 조회 (OK/NG 비율, 평균 ΔE 등)

        Returns:
            {
                'total_count': int,
                'ok_count': int,
                'ng_count': int,
                'ok_rate': float,
                'avg_delta_e': float,
                'zone_statistics': {...}
            }
        """
        pass

    def export_to_csv(self,
                      output_path: Path,
                      filters: dict = None) -> Path:
        """
        검사 결과를 CSV로 내보내기
        """
        pass
```

**상세 구현**:

1. **데이터베이스 스키마**
   ```python
   from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey
   from sqlalchemy.ext.declarative import declarative_base
   from sqlalchemy.orm import relationship

   Base = declarative_base()

   class InspectionRecord(Base):
       """검사 결과 레코드"""
       __tablename__ = 'inspection_records'

       id = Column(Integer, primary_key=True, autoincrement=True)
       timestamp = Column(DateTime, nullable=False)
       sku = Column(String(50), nullable=False, index=True)
       judgment = Column(String(10), nullable=False)  # 'OK' or 'NG'
       overall_delta_e = Column(Float)
       confidence = Column(Float)
       image_path = Column(String(500))
       overlay_path = Column(String(500))
       ng_reasons = Column(String(1000))  # JSON string

       # 관계
       zone_results = relationship("ZoneRecord", back_populates="inspection")

   class ZoneRecord(Base):
       """Zone별 결과"""
       __tablename__ = 'zone_records'

       id = Column(Integer, primary_key=True, autoincrement=True)
       inspection_id = Column(Integer, ForeignKey('inspection_records.id'))
       zone_name = Column(String(20), nullable=False)
       measured_L = Column(Float)
       measured_a = Column(Float)
       measured_b = Column(Float)
       target_L = Column(Float)
       target_a = Column(Float)
       target_b = Column(Float)
       delta_e = Column(Float)
       threshold = Column(Float)
       is_ok = Column(Boolean)

       # 관계
       inspection = relationship("InspectionRecord", back_populates="zone_results")

   def _init_database(self, db_path: Path) -> sqlalchemy.Engine:
       """DB 초기화"""
       from sqlalchemy import create_engine
       from sqlalchemy.orm import sessionmaker

       engine = create_engine(f'sqlite:///{db_path}')
       Base.metadata.create_all(engine)

       self.SessionMaker = sessionmaker(bind=engine)

       return engine
   ```

2. **결과 저장**
   ```python
   def log_result(self,
                  result: InspectionResult,
                  image_path: Path = None,
                  overlay_path: Path = None) -> int:
       """
       검사 결과를 DB에 저장
       """
       import json

       session = self.SessionMaker()

       try:
           # 메인 레코드 생성
           record = InspectionRecord(
               timestamp=result.timestamp,
               sku=result.sku,
               judgment=result.judgment,
               overall_delta_e=result.overall_delta_e,
               confidence=result.confidence,
               image_path=str(image_path) if image_path else None,
               overlay_path=str(overlay_path) if overlay_path else None,
               ng_reasons=json.dumps(result.ng_reasons, ensure_ascii=False)
           )
           session.add(record)
           session.flush()  # ID 생성

           # Zone 결과 저장
           for zr in result.zone_results:
               zone_record = ZoneRecord(
                   inspection_id=record.id,
                   zone_name=zr.zone_name,
                   measured_L=zr.measured_lab[0],
                   measured_a=zr.measured_lab[1],
                   measured_b=zr.measured_lab[2],
                   target_L=zr.target_lab[0],
                   target_a=zr.target_lab[1],
                   target_b=zr.target_lab[2],
                   delta_e=zr.delta_e,
                   threshold=zr.threshold,
                   is_ok=zr.is_ok
               )
               session.add(zone_record)

           session.commit()
           return record.id

       except Exception as e:
           session.rollback()
           raise
       finally:
           session.close()
   ```

3. **통계 조회**
   ```python
   def get_statistics(self,
                      sku: str = None,
                      start_date: datetime = None,
                      end_date: datetime = None) -> dict:
       """
       지정 기간/SKU의 검사 통계
       """
       from sqlalchemy import func

       session = self.SessionMaker()

       try:
           # 쿼리 빌드
           query = session.query(InspectionRecord)

           if sku:
               query = query.filter(InspectionRecord.sku == sku)
           if start_date:
               query = query.filter(InspectionRecord.timestamp >= start_date)
           if end_date:
               query = query.filter(InspectionRecord.timestamp <= end_date)

           # 전체 카운트
           total_count = query.count()

           if total_count == 0:
               return {
                   'total_count': 0,
                   'ok_count': 0,
                   'ng_count': 0,
                   'ok_rate': 0.0,
                   'avg_delta_e': 0.0
               }

           # OK/NG 카운트
           ok_count = query.filter(InspectionRecord.judgment == 'OK').count()
           ng_count = total_count - ok_count

           # 평균 ΔE
           avg_delta_e = session.query(
               func.avg(InspectionRecord.overall_delta_e)
           ).filter(
               InspectionRecord.sku == sku if sku else True,
               InspectionRecord.timestamp >= start_date if start_date else True,
               InspectionRecord.timestamp <= end_date if end_date else True
           ).scalar()

           # Zone별 통계
           zone_stats = session.query(
               ZoneRecord.zone_name,
               func.avg(ZoneRecord.delta_e).label('avg_delta_e'),
               func.count(ZoneRecord.id).label('count'),
               func.sum(ZoneRecord.is_ok.cast(Integer)).label('ok_count')
           ).join(InspectionRecord).filter(
               InspectionRecord.sku == sku if sku else True,
               InspectionRecord.timestamp >= start_date if start_date else True,
               InspectionRecord.timestamp <= end_date if end_date else True
           ).group_by(ZoneRecord.zone_name).all()

           zone_statistics = {}
           for zone_name, avg_de, count, ok_cnt in zone_stats:
               zone_statistics[zone_name] = {
                   'avg_delta_e': avg_de,
                   'total_count': count,
                   'ok_count': ok_cnt,
                   'ng_count': count - ok_cnt,
                   'ok_rate': ok_cnt / count if count > 0 else 0.0
               }

           return {
               'total_count': total_count,
               'ok_count': ok_count,
               'ng_count': ng_count,
               'ok_rate': ok_count / total_count,
               'avg_delta_e': avg_delta_e or 0.0,
               'zone_statistics': zone_statistics
           }

       finally:
           session.close()
   ```

#### 3.2.8 SkuConfigManager (SKU 설정 관리)

**책임**: SKU별 기준값 저장, 로드, 관리

**인터페이스**:
```python
@dataclass
class SkuConfig:
    """SKU별 검사 기준"""
    sku: str
    default_threshold: float  # 기본 ΔE 허용치
    zones: dict               # zone_name → target LAB, threshold
    metadata: dict            # 기타 정보 (등록일, 설명 등)

class SkuConfigManager:
    """SKU 설정 관리"""

    def __init__(self, config_dir: Path):
        """
        Args:
            config_dir: SKU 설정 파일 저장 디렉토리
        """
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}  # 메모리 캐시

    def register_sku(self,
                     sku: str,
                     golden_samples: List[np.ndarray],
                     default_threshold: float = 3.0) -> SkuConfig:
        """
        새 SKU 등록 (골든 샘플로부터 기준값 자동 생성)

        Args:
            sku: SKU 코드
            golden_samples: 양품 이미지 리스트
            default_threshold: 기본 ΔE 허용치

        Returns:
            생성된 SkuConfig
        """
        pass

    def get_config(self, sku: str) -> SkuConfig:
        """SKU 설정 로드"""
        pass

    def update_config(self, config: SkuConfig) -> None:
        """SKU 설정 업데이트"""
        pass

    def list_skus(self) -> List[str]:
        """등록된 모든 SKU 목록"""
        pass
```

**상세 구현**:

1. **골든 샘플로부터 기준값 생성**
   ```python
   def register_sku(self,
                    sku: str,
                    golden_samples: List[np.ndarray],
                    default_threshold: float = 3.0) -> SkuConfig:
       """
       여러 장의 양품 샘플을 분석하여 기준값 자동 생성
       """
       from collections import defaultdict

       # 각 샘플을 처리하여 zone별 LAB 수집
       zone_lab_samples = defaultdict(list)  # zone_name → [(L,a,b), ...]

       for sample_image in golden_samples:
           # 전체 파이프라인 실행
           loader = ImageLoader(...)
           preprocessed = loader.preprocess(sample_image)

           detector = LensDetector(...)
           lens = detector.detect(preprocessed['gray'])

           profiler = RadialProfiler(...)
           profile = profiler.extract_profile(sample_image, lens)

           segmenter = ZoneSegmenter(...)
           zones = segmenter.segment(profile)

           # Zone별 LAB 값 수집
           for zone in zones:
               zone_lab_samples[zone.name].append(
                   (zone.mean_L, zone.mean_a, zone.mean_b)
               )

       # 각 zone의 평균 및 표준편차 계산
       zone_configs = {}

       for zone_name, lab_list in zone_lab_samples.items():
           lab_array = np.array(lab_list)  # (N, 3)

           mean_L = np.mean(lab_array[:, 0])
           mean_a = np.mean(lab_array[:, 1])
           mean_b = np.mean(lab_array[:, 2])

           std_L = np.std(lab_array[:, 0])
           std_a = np.std(lab_array[:, 1])
           std_b = np.std(lab_array[:, 2])

           # ΔE 허용치: 표준편차의 2배 또는 기본값 중 큰 값
           # (샘플 간 변동을 허용)
           zone_threshold = max(
               2.0 * np.sqrt(std_L**2 + std_a**2 + std_b**2),
               default_threshold
           )

           zone_configs[zone_name] = {
               'L': float(mean_L),
               'a': float(mean_a),
               'b': float(mean_b),
               'std_L': float(std_L),
               'std_a': float(std_a),
               'std_b': float(std_b),
               'delta_e_threshold': float(zone_threshold)
           }

       # SkuConfig 객체 생성
       config = SkuConfig(
           sku=sku,
           default_threshold=default_threshold,
           zones=zone_configs,
           metadata={
               'registered_at': datetime.now().isoformat(),
               'sample_count': len(golden_samples),
               'description': ''
           }
       )

       # 저장
       self.update_config(config)

       return config
   ```

2. **설정 파일 저장/로드** (JSON 형식)
   ```python
   def _get_config_path(self, sku: str) -> Path:
       """SKU 설정 파일 경로"""
       return self.config_dir / f"{sku}.json"

   def get_config(self, sku: str) -> SkuConfig:
       """설정 로드 (캐시 활용)"""
       # 캐시 확인
       if sku in self.cache:
           return self.cache[sku]

       # 파일 로드
       config_path = self._get_config_path(sku)

       if not config_path.exists():
           return None

       with open(config_path, 'r', encoding='utf-8') as f:
           data = json.load(f)

       config = SkuConfig(
           sku=data['sku'],
           default_threshold=data['default_threshold'],
           zones=data['zones'],
           metadata=data['metadata']
       )

       # 캐시 저장
       self.cache[sku] = config

       return config

   def update_config(self, config: SkuConfig) -> None:
       """설정 저장"""
       config_path = self._get_config_path(config.sku)

       data = {
           'sku': config.sku,
           'default_threshold': config.default_threshold,
           'zones': config.zones,
           'metadata': config.metadata
       }

       with open(config_path, 'w', encoding='utf-8') as f:
           json.dump(data, f, ensure_ascii=False, indent=2)

       # 캐시 업데이트
       self.cache[config.sku] = config

       logger.info(f"SKU {config.sku} 설정 저장 완료: {config_path}")
   ```

3. **설정 검증 및 백업**
   ```python
   def validate_config(self, config: SkuConfig) -> List[str]:
       """
       설정 유효성 검증

       Returns:
           경고 메시지 리스트 (빈 리스트면 정상)
       """
       warnings = []

       # 필수 zone 존재 확인
       if not config.zones:
           warnings.append("Zone 정보가 없습니다.")

       # 각 zone 검증
       for zone_name, zone_data in config.zones.items():
           # 필수 필드 확인
           required_fields = ['L', 'a', 'b', 'delta_e_threshold']
           for field in required_fields:
               if field not in zone_data:
                   warnings.append(f"Zone {zone_name}: {field} 필드 없음")

           # LAB 범위 확인
           L = zone_data.get('L', 0)
           a = zone_data.get('a', 0)
           b = zone_data.get('b', 0)

           if not (0 <= L <= 100):
               warnings.append(f"Zone {zone_name}: L* 값 범위 초과 ({L})")
           if not (-128 <= a <= 127):
               warnings.append(f"Zone {zone_name}: a* 값 범위 초과 ({a})")
           if not (-128 <= b <= 127):
               warnings.append(f"Zone {zone_name}: b* 값 범위 초과 ({b})")

       return warnings

   def backup_configs(self, backup_dir: Path) -> Path:
       """
       모든 SKU 설정 백업

       Returns:
           백업 아카이브 경로
       """
       import shutil
       from datetime import datetime

       timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
       backup_path = backup_dir / f"sku_configs_backup_{timestamp}.zip"

       shutil.make_archive(
           str(backup_path.with_suffix('')),
           'zip',
           self.config_dir
       )

       logger.info(f"SKU 설정 백업 완료: {backup_path}")
       return backup_path
   ```

---

## 4. 상세 구현 계획 (단계별)

### 4.1 1단계: 알고리즘 프로토타입 (4주)

**목표**: 핵심 알고리즘의 기능 검증

#### 주차별 세부 계획

**Week 1: 개발 환경 구축 및 기본 모듈**
- **Day 1-2**: 개발 환경 설정
  - Python 3.10 가상환경 생성
  - 필수 라이브러리 설치 (requirements.txt 작성)
  - Git 저장소 초기화
  - 프로젝트 디렉토리 구조 생성
    ```
    color_meter/
    ├── src/
    │   ├── core/              # 핵심 모듈
    │   │   ├── __init__.py
    │   │   ├── image_loader.py
    │   │   ├── lens_detector.py
    │   │   ├── radial_profiler.py
    │   │   ├── zone_segmenter.py
    │   │   ├── color_evaluator.py
    │   │   └── visualizer.py
    │   ├── config/            # 설정 관리
    │   │   ├── __init__.py
    │   │   └── sku_manager.py
    │   ├── data/              # 데이터 관리
    │   │   ├── __init__.py
    │   │   └── logger.py
    │   └── utils/             # 유틸리티
    │       └── __init__.py
    ├── tests/                 # 테스트
    ├── notebooks/             # Jupyter 프로토타입
    ├── data/                  # 데이터
    │   ├── samples/           # 샘플 이미지
    │   ├── configs/           # SKU 설정
    │   └── results/           # 검사 결과
    ├── docs/                  # 문서
    ├── requirements.txt
    └── README.md
    ```

- **Day 3-4**: ImageLoader 구현
  - 이미지 로드 (파일, NumPy 배열)
  - 기본 전처리 (노이즈 제거, ROI 검출)
  - 화이트 밸런스 보정
  - 단위 테스트 작성

- **Day 5**: 샘플 데이터 수집
  - 테스트용 렌즈 이미지 5-10장 촬영
  - 다양한 조명 조건 테스트
  - 데이터 정리 및 레이블링

**Week 2: 렌즈 검출 및 프로파일 추출**
- **Day 1-2**: LensDetector 구현
  - Hough 원 검출
  - 컨투어 기반 검출
  - 하이브리드 방법
  - 검출 신뢰도 계산
  - 테스트: 다양한 위치/각도에서 검출 정확도 확인

- **Day 3-5**: RadialProfiler 구현
  - 극좌표 변환
  - 링별 LAB 색상 추출
  - 프로파일 스무딩
  - 시각화 (r vs L*, a*, b* 그래프)
  - 테스트: 프로파일이 직관적으로 zone 구조를 보이는지 확인

**Week 3: Zone 분할 및 색상 평가**
- **Day 1-3**: ZoneSegmenter 구현
  - a* 그래디언트 기반 변곡점 검출
  - ΔE 기반 변곡점 검출
  - Zone 분류 (pure/mix)
  - Mix zone 품질 평가
  - 파라미터 튜닝 (스무딩 윈도우, 최소 기울기 등)

- **Day 4-5**: ColorEvaluator 구현
  - ΔE2000 계산 (colormath 활용)
  - Zone별 평가 로직
  - OK/NG 판정
  - 신뢰도 계산

**Week 4: 통합 및 프로토타입 완성**
- **Day 1-2**: 전체 파이프라인 통합
  - 단일 이미지 검사 함수 작성
  - Jupyter Notebook으로 시연
  - 중간 결과 시각화 (각 단계별)

- **Day 3-4**: 프로토타입 검증
  - 10장 이상의 샘플로 테스트
  - Zone 분할이 의도대로 되는지 검증
  - 색상 차이가 ΔE로 잘 표현되는지 확인
  - 문제점 파악 및 개선 항목 리스트업

- **Day 5**: 1단계 리포트 작성
  - 알고리즘 동작 확인 문서
  - 시각화 결과 (이미지, 그래프)
  - 한계점 및 개선 방향
  - 2단계 진입 조건 평가

**산출물**:
- 동작하는 프로토타입 코드 (Jupyter Notebook)
- 샘플 이미지 10장에 대한 검사 결과
- r-프로파일 그래프 (L*, a*, b*)
- Zone 분할 시각화
- 1단계 완료 보고서

---

### 4.2 2단계: Zone 자동 분할 및 SKU 기준 설정 (4주)

**목표**: SKU별 기준값 자동 생성 및 관리 시스템 구축

#### 주차별 세부 계획

**Week 1: SkuConfigManager 구현**
- **Day 1-2**: 기본 구조
  - SkuConfig 데이터 클래스 정의
  - JSON 기반 설정 저장/로드
  - 설정 검증 로직

- **Day 3-5**: 골든 샘플 기반 기준값 생성
  - 여러 장의 양품 이미지 → 평균 LAB 계산
  - Zone별 표준편차 → ΔE 허용치 자동 설정
  - 기준값 등록 인터페이스 (CLI 스크립트)

**Week 2: 변곡점 검출 고도화**
- **Day 1-2**: 알고리즘 개선
  - 다양한 샘플에서 변곡점 검출 안정성 테스트
  - False positive 제거 (노이즈로 인한 오탐)
  - 파라미터 자동 조정 로직

- **Day 3-4**: 다양한 패턴 대응
  - 3-zone, 5-zone, 7-zone 등 자동 구분
  - Zone 레이블 자동 생성
  - Mix zone 폭 분석

- **Day 5**: 테스트
  - 최소 3개 SKU에 대해 테스트
  - 각 SKU당 양품 30장씩 수집
  - 기준값 생성 및 검증

**Week 3: SKU 관리 도구 개발**
- **Day 1-2**: CLI 도구
  ```bash
  # SKU 등록
  python manage_sku.py register --sku SKU001 --samples ./golden_samples/SKU001/*.jpg

  # SKU 목록 조회
  python manage_sku.py list

  # SKU 상세 정보
  python manage_sku.py show --sku SKU001

  # 기준값 수정
  python manage_sku.py update --sku SKU001 --zone A --threshold 4.0

  # 백업
  python manage_sku.py backup --output ./backups/
  ```

- **Day 3-4**: 간단한 GUI (PyQt 또는 Streamlit)
  - SKU 목록 표시
  - 기준값 시각화 (zone별 LAB 값, 허용 범위)
  - 기준값 수동 조정 인터페이스
  - 골든 샘플 이미지 표시

- **Day 5**: 문서화
  - 사용자 매뉴얼 작성
  - SKU 등록 절차 문서화
  - 파라미터 튜닝 가이드

**Week 4: 다양한 SKU 테스트**
- **Day 1-3**: 실제 제품 테스트
  - 최소 5개 SKU 등록
  - 각 SKU당 양품 30장, 불량 10장 수집
  - 기준값의 적절성 평가
  - 임계값 조정

- **Day 4**: 교차 검증
  - 양품을 다른 SKU 기준으로 검사 시 NG 나오는지 확인
  - 불량품이 제대로 걸러지는지 확인
  - False positive/negative 비율 측정

- **Day 5**: 2단계 리포트
  - SKU별 검출 성능 정리
  - 권장 파라미터 세트
  - 3단계 진입 조건 평가

**산출물**:
- SkuConfigManager 모듈
- SKU 관리 CLI/GUI 도구
- 5개 SKU 기준값 세트
- SKU 등록 매뉴얼
- 2단계 완료 보고서

---

### 4.3 3단계: 품질 판정 로직 고도화 (4주)

**목표**: 실제 불량 검출률 향상 및 오탐률 감소

#### 주차별 세부 계획

**Week 1: 불량 샘플 수집 및 분석**
- **Day 1-2**: 불량 유형 정의
  - 색상 편차 (잉크 농도 과다/부족)
  - 경계 번짐 (mix zone 과도)
  - 인쇄 누락 (도트 빠짐)
  - 색상 치우침 (특정 zone만 이상)
  - 기타 (기포, 이물질 등)

- **Day 3-5**: 불량 샘플 확보
  - 각 유형별 10장 이상
  - 의도적 불량 생성 (가능하면)
  - 불량 이미지에 대해 현재 시스템 반응 테스트
  - 미탐지 사례 분석

**Week 2: 판정 로직 개선**
- **Day 1-2**: Zone별 가중치 도입
  - 중요 zone(예: 인너)에 더 엄격한 기준
  - Zone별 ΔE 허용치 개별 설정
  - 가중 평균 ΔE 계산

- **Day 3-4**: Mix zone 품질 검증 강화
  - 혼합 편차 허용 범위 설정
  - Mix zone 폭 이상 검출
  - 색상 경로 이탈 검출

- **Day 5**: 다중 판정 기준
  - ΔE뿐 아니라 L*, a*, b* 개별 편차도 확인
  - 표준편차 기반 이상치 검출
  - 앙상블 판정 (여러 기준 조합)

**Week 3: 성능 평가 및 튜닝**
- **Day 1-2**: 성능 메트릭 계산
  - Confusion Matrix (TP, TN, FP, FN)
  - 민감도 (Sensitivity) = TP / (TP + FN)
  - 특이도 (Specificity) = TN / (TN + FP)
  - 정확도 (Accuracy) = (TP + TN) / Total
  - F1 Score

- **Day 3-4**: ROC 곡선 분석
  - ΔE 임계값 변화에 따른 성능 변화
  - 최적 임계값 탐색
  - SKU별 최적 임계값 다를 수 있음

- **Day 5**: 하이퍼파라미터 튜닝
  - Grid Search 또는 Bayesian Optimization
  - 튜닝 대상 파라미터:
    - r_step (프로파일 해상도)
    - smoothing_window (스무딩 윈도우)
    - min_gradient (변곡점 검출 임계값)
    - delta_e_threshold (zone별)

**Week 4: 엣지 케이스 처리**
- **Day 1-2**: 조명 변화 대응
  - 밝기 변화 시 화이트 밸런스 보정 효과 검증
  - 그림자 영향 최소화
  - 필요 시 조명 정규화 추가

- **Day 3**: 렌즈 위치 변동 대응
  - 이미지 중심에서 벗어난 경우 테스트
  - 회전 불변성 검증
  - 부분 가림 대응 (가능한 범위)

- **Day 4**: 오류 처리
  - 렌즈 검출 실패 시 재시도 로직
  - 부적합 이미지 (초점 불량 등) 거부
  - 예외 상황 로깅

- **Day 5**: 3단계 리포트
  - 최종 성능 메트릭
  - 불량 유형별 검출률
  - 권장 운영 파라미터
  - 알려진 한계점 및 주의사항

**산출물**:
- 개선된 ColorEvaluator
- 성능 평가 보고서 (민감도, 특이도, F1)
- 불량 유형별 검출 결과
- 최적 파라미터 세트
- 3단계 완료 보고서

---

### 4.4 4단계: UI 구현 및 현장 도입 준비 (4주)

**목표**: 현장에서 사용 가능한 완성된 시스템

#### 주차별 세부 계획

**Week 1: 데이터베이스 및 로깅**
- **Day 1-2**: Logger 구현
  - SQLite 데이터베이스 초기화
  - 검사 결과 저장
  - NG 샘플 자동 저장
  - CSV 내보내기

- **Day 3-4**: 통계 기능
  - 일별/주별/월별 OK/NG 비율
  - Zone별 평균 ΔE 트렌드
  - SKU별 불량률
  - 차트 생성 (matplotlib)

- **Day 5**: 테스트
  - 대량 데이터 저장/조회 성능 테스트
  - 100장 이상 검사 결과 저장
  - 통계 쿼리 속도 확인

**Week 2: GUI 개발**
- **Day 1-2**: 메인 화면 (PyQt6 또는 Streamlit)
  - 레이아웃 설계
    ```
    ┌─────────────────────────────────────────┐
    │  [메뉴바]  파일 | 검사 | 설정 | 통계 | 도움말 │
    ├──────────────┬──────────────────────────┤
    │              │                          │
    │  이미지 표시  │   검사 정보              │
    │  (원본/오버  │   - SKU: SKU001          │
    │   레이)      │   - 판정: OK             │
    │              │   - ΔE: 2.34             │
    │              │   - 신뢰도: 95%          │
    │              │                          │
    │              │   Zone 결과:             │
    │              │   ┌───────────────────┐  │
    │              │   │ Zone │ ΔE │ 판정 │  │
    │              │   ├───────────────────┤  │
    │              │   │  A   │2.1 │ OK  │  │
    │              │   │ A-B  │1.8 │ OK  │  │
    │              │   │  B   │2.9 │ OK  │  │
    │              │   └───────────────────┘  │
    ├──────────────┴──────────────────────────┤
    │  [검사 시작] [이미지 로드] [설정] [통계]  │
    └─────────────────────────────────────────┘
    ```
  - 기본 위젯 배치

- **Day 3-4**: 기능 구현
  - 이미지 로드 버튼
  - 검사 시작 버튼 → 진행 표시
  - 결과 표시 (판정, ΔE, zone별 상세)
  - 이미지 토글 (원본 ↔ 오버레이)

- **Day 5**: 프로파일 그래프 표시
  - Matplotlib을 PyQt에 임베딩
  - 실시간 그래프 업데이트

**Week 3: 카메라 연동 및 실시간 검사**
- **Day 1-2**: 카메라 인터페이스
  - 카메라 SDK 연동 (예: Basler Pylon, Allied Vision Vimba)
  - 실시간 프리뷰
  - 트리거 신호 수신 (외부 센서 연동)
