# STD vs Sample 비교 (빠른 시작)
- 페이지: `/compare`
- 입력: STD 이미지 1장 + Sample 이미지 1장 업로드
- 처리: `POST /compare` (reference_file + test_files)
- 출력: mean/max Delta E, Delta L/Delta a/Delta b 방향, overall shift, zone별 Delta E
- 결과 다운로드: JSON 다운로드 버튼 (AI 검토/재현용)
- 현재 방식: Zone 기반 비교 + ink 매핑(있으면) + ink 기준 경고 표시

## 용어/범위 정합
- Quick Start의 STD는 임시 Reference Image이며, M2 Comparison System의 STD Model과 다름.
- Quick Start 결과는 M2 ComparisonResult의 부분집합.
- overall shift는 설명용 지표이며 판정에는 사용하지 않음.

## ink 매핑
- SKU 설정 파일의 `params.ink_mapping`으로 Zone → Ink 매핑을 정의한다.
- 매핑이 없으면 모든 Zone은 `ink1`로 묶인다.

## ink 임계치
- SKU 설정 파일의 `params.ink_thresholds`로 잉크별 임계치를 정의한다.
- `default.max_delta_e`가 기본값이며, 잉크별 값으로 덮어쓴다.
- 잉크 기준은 해당 잉크에 속한 Zone들의 `max_delta_e`를 사용한다.

## 관련 문서
- 비교 UI 가이드 + 정형 텍스트 템플릿: `docs/guides/comparison/COMPARE_UI_GUIDE.md`
- Quick Start ↔ M2 매핑: `docs/guides/comparison/QUICK_START_M2_MAPPING.md`
- Comparison System 1page: `docs/guides/comparison/COMPARISON_SYSTEM_OVERVIEW.md`
- 샘플 이미지 위치: `data/samples/`
