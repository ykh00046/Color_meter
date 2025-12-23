# 측정 지표 확장 UI 요구사항

## 목적
Inspection에서 측정된 지표를 Comparison에서 STD 대비 비교로 재사용할 수 있도록 UI와 JSON 출력 규격을 정리한다.

## 적용 범위
- Inspection: 측정/추출 결과 표시 및 JSON 저장
- Comparison: STD 대비 차이와 경고 표시, 해석 가이드 제공

## 현 상태 이슈
- 일부 UI 텍스트가 깨져 표시됨(인코딩 문제). 아래 파일의 텍스트 정상화 필요:
  - `src/web/templates/index.html`
  - `src/web/templates/compare.html`

## 공통 UI 요구사항
- 모든 신규 지표는 **"측정값"과 "STD 대비 차이"**를 분리 표시한다.
- 지표별 **상태 배지**(OK/WARN/FAIL)를 표시한다.
- 값이 없으면 `N/A`로 표시하고 경고 문구는 출력하지 않는다.
- JSON 다운로드에는 측정값/비교값/판정 근거를 모두 포함한다.

## Comparison UI 요구사항 (`/compare`)
### 요약 카드(상단)
- 기존: ΔE, ΔL/a/b, action guide
- 추가:
  - Blur(선명도) 요약: `blur_score`, `blur_delta`
  - Histogram 요약: `hist_diff` (HSV/Lab)
  - Dot 통계 요약: `dot_count`, `dot_coverage`, `dot_delta`
  - 정합 품질(옵션): `alignment_score` 또는 `alignment_confidence`

### 상세 탭(하단)
- Summary / Zone / Ink / Quality(신규) 탭 구성
- Quality 탭:
  - Blur: Laplacian variance + 기준 비교
  - Histogram: 채널별 거리/상관도
  - Dot stats: 개수/면적/coverage
  - 2D SSIM/PSNR(옵션)

### Action Guide
- 기존 ΔL/a/b 기반 문구 유지
- 신규 지표가 임계치 초과 시 경고 추가
  - 예: "Blur 낮음: 초점/노광 점검"
  - 예: "Histogram 차이 큼: 색 분포 이탈"
  - 예: "Dot coverage 급변: 잉크 도트 불균일"

## Inspection UI 요구사항
### 결과 요약 영역
- 측정값 표시: `blur_score`, `histogram`, `dot_stats`
- Quality 요약 배지 표시

### 상세 로그/JSON
- 측정값과 파라미터(윈도우 크기, 히스토그램 bin 등) 기록

## JSON 출력 요구사항
### 공통 필드
- `metrics.blur`
- `metrics.histogram`
- `metrics.dot_stats`

### Comparison 전용
- `comparison.blur_delta`
- `comparison.hist_diff`
- `comparison.dot_delta`

## 기준/임계치 정의
- 기본값: SKU 설정 또는 STD 통계 기반
- 초기에는 `warning`만 사용, 안정화 후 `fail` 기준 도입
