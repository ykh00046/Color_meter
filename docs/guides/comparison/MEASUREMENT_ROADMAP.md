# 비교/검사 공통 지표 확장 로드맵

## 전제
- Inspection: 지표 **측정/추출** 중심
- Comparison: STD 대비 **비교/판정** 중심
- 동일 지표라도 **Inspection에서 측정한 값을 Comparison이 재사용**하는 구조가 바람직

## 추천 우선순위 (높음 → 낮음)

### 1) 선명도/블러 (Laplacian Variance)
- 이유: 촬영 품질 문제를 빠르게 걸러내는 핵심 QC 지표
- Inspection: `blur_score` 저장
- Comparison: STD 대비 상대 변화 감지(옵션)

### 2) 히스토그램 비교 (HSV/Lab)
- 이유: 전체 색 분포 변화 감지에 비용 대비 효과가 큼
- Inspection: 채널별 히스토그램 저장
- Comparison: STD 히스토그램과 거리/상관 비교

### 3) 점 패턴 통계 (개수/면적/밀도)
- 이유: 도트 품질 문제를 직접 잡는 제조 핵심 지표
- Inspection: `dot_count`, `dot_area_mean`, `dot_area_std`, `dot_coverage` 저장
- Comparison: STD 대비 변화량 비교

### 4) 구조 유사도 (2D SSIM/PSNR)
- 이유: 정합 전제가 강해 초기 도입 리스크 존재
- Inspection: 필요 시 계산(옵션)
- Comparison: STD와 정합된 후 SSIM 계산

### 5) 색상 코드/Pantone 기반 비교
- 이유: 장기 확장용(데이터베이스/캘리브레이션 필요)
- Inspection: 위치별 대표 색상 추출 저장
- Comparison: 기준 코드 대비 ΔE 평가

## 권장 설계 원칙
- Inspection에서 측정 → Comparison에서 비교 구조 고정
- 지표는 스키마에 저장하고 JSON/로그에 포함
- 초기엔 설명용(Soft) 지표로 노출, 운영 안정 후 판정 지표(Hard)로 승격
