# Comparison System 개념 정의 (1 Page)

## 1. 시스템 정체성
### Inspection System
- 절대 품질 검사
- SKU 기준
- 단일 이미지 판정
- 운영 시스템

### Comparison System
- 상대 품질 비교
- STD 기준
- STD vs Sample
- 분석/판정/이력 시스템

## 2. STD 정의 (중요)
- Reference Image: Quick Start에서 임시 비교용
- STD Model: DB에 등록된 공식 기준
- Active STD: 판정에 사용되는 STD Model

Comparison System의 모든 판정은 STD Model만 사용한다.

## 3. 점수/지표 책임 분리
### 판정에 사용되는 지표 (Hard)
- zone_score
- total_score
- confidence_score
- threshold 기반 rule

### 설명용 지표 (Soft)
- overall_shift
- Delta L/Delta a/Delta b 방향
- mean/max Delta E

Soft 지표는 판정에 사용하지 않는다.

## 4. Judgment 상태 의미
- PASS: STD와 충분히 유사
- RETAKE: 불확실 / 재측정 필요
- FAIL: 명확한 편차
- MANUAL_REVIEW: 시스템 신뢰 불가

## 5. Failure Reason 역할
- 판정 근거
- ML 학습용 라벨
- 현장 조치 판단용 설명

## 6. 이 시스템이 하지 않는 것
- 공정 조정량 계산
- 불량 유형 단정
- 잉크 조합 최적화

## 7. 진화 경로 (요약)
- M2: STD vs Sample 비교 + 설명
- P1: 불확실 케이스 탐지 (ML-lite)
- P2: 불량 유형 분류 + 조치 추천

## 참고
- 정형 텍스트 템플릿: `docs/guides/comparison/COMPARE_UI_GUIDE.md`
