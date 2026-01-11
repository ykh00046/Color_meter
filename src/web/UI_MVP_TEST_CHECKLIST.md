# v7 UI MVP Test Checklist

## 1) STD 등록/검증
- 역할: Operator
- SKU/INK 입력
- LOW/MID/HIGH 이미지 업로드
- 등록 + 검증 실행
- 결과 요약에 `activation_allowed` 표시 확인

## 2) 활성화
- 역할: Approver
- SKU/INK 입력
- 후보 버전 불러오기
- LOW/MID/HIGH 선택
- 승인자/사유 입력 후 활성화
- 활성화 결과에 ACTIVE 경로 표시 확인

## 3) 검사
- 역할: Operator
- SKU/INK 입력
- 이미지 업로드
- 검사 실행
- 결과 요약에 label_counts + ACTIVE 버전 표시 확인

## 4) 오류 방지 확인
- STD_ACCEPTABLE이 아니면 활성화 버튼 비활성
- ACTIVE 없는 SKU/INK로 검사 시 RETAKE 반환 확인
