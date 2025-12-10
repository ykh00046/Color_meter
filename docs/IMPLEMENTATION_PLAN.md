# Color Lens Color QA System ? Implementation Plan

## 0. Assumptions & Targets
- Scope: r-profile rule-based 색상 검사 우선, 차후 비지도 이상탐지(PatchCore/PaDiM) 옵션 연구.
- Timeline 목표: 4~5개월 내 현장 적용(2~3인 팀 기준), PoC 통과 후 점진 고도화.
- Throughput: 실시간 검사 ≤200 ms/장(@IPC), 렌즈 지름 ≥1000 px 촬영, SKU 혼류 대응.

## 1. Workstreams & Owners (예시)
- Vision/ML: 렌즈 검출, r-profile, ΔE 판정, 변곡점 기반 zone 분할.
- SW/UI: 설정/로그/리포트, SKU 관리, UI/카메라 연동.
- PM/QA: 요구수집, 샘플/불량 데이터 확보, 현장 테스트 조율.

## 2. Data & HW Readiness
- 정상 샘플: SKU별 ≥30장(동일 조명/촬영), 불량 유형별 ≥20장(번짐/누락/색치우침/패턴밀림 등).
- 카메라: ≥5MP, 렌즈 지름 이미지상 ≥1000 px, 매크로/적절 배율 렌즈.
- 조명: 링/돔, 5000K, 균일도 측정; 주 1회 컬러차트로 화밸/감마 교정 SOP.
- 지그/트리거: 컨베이어/트레이 정렬, 촬영 트리거 연동.

## 3. Modules & Interfaces
- ImageLoader: 파일/카메라 입력, 전처리(그레이, 노이즈 억제).
- LensDetector: 허프/컨투어 최소외접원 → 중심/반경 추정.
- RadialProfiler: 0.3R~1.0R, Δr 설정 후 링별 평균 RGB→LAB, L*a*b vs r 곡선.
- ZoneSegmenter: ΔE(r) 변곡점+평활화→A/A-B/B/B-C/C 구간, 혼합 폭 추출.
- ColorEvaluator: SKU 기준(LAB, ΔE 한계, 혼합 거리/폭) 로드→OK/NG.
- SkuConfigManager: 설정 JSON/YAML (버전/이력/백업/롤백), 바코드/코드 매핑.
- Visualizer: 원본+zone/NG 오버레이, r-profile 그래프.
- Logger/Reporter: CSV+이미지+r-profile 저장, PDF/화면 요약, MES API 훅 포인트.

## 4. Milestones & Deliverables
### 0단계 준비 (1주)
- 리포/CI 기본, ΔE2000 util, 샘플 수집 계획/체크리스트.

### 1단계 프로토타입 (3주)
- 기능: 렌즈 검출, r-profile, LAB 변환, 변곡점 가시화.
- 산출: 노트북/CLI 데모, 원본+zone 오버레이, L*a*b vs r 그래프.
- 지표: 처리지연(ms), 검출 실패율, 시각적 분리 여부.

### 2단계 SKU 기준 자동화 (4주)
- 기능: 다장 샘플→경계 통계→자동 zone/혼합 폭 산출, SKU JSON 생성.
- 설정 UI/CLI: 샘플 등록, 버전/백업, 롤백.
- 지표: ΔE 분포(평균±2σ), 경계 재현성, 설정 적용 시간.

### 3단계 판정 고도화 (4주)
- 기능: ΔE 상·하한 튜닝, 혼합 경로 거리/폭 기준, 조명 변화 대응 전처리, NG 저장/태깅.
- 데이터: 정상≥30, 불량 유형별≥20 확보/라벨.
- 지표: 민감도/특이도, FPR/TPR, 처리지연 95p, 반복 측정 분산.

### 4단계 UI·통합 (4주)
- 기능: 실시간 뷰어, 배치 처리, SKU 선택/바코드, 리포트(PDF/CSV), 로그 관리, 설정 롤백.
- 연동: 카메라 트리거, 조명 제어, IPC 배포 스크립트.
- 지표: UI 반응, 스루풋(렌즈/분), 알림 정확성.

### 5단계 현장 PoC (4주)
- 설치/캘리브레이션 SOP 실행(주 1회 컬러차트), 하루≥200ea 시험, LOT별 ΔE 트렌드 리포트.
- 오탐/미탐 리뷰→파라미터 미세조정.

### 6단계 고도화 옵션 (병행 4주 R&D)
- PatchCore/PaDiM 벤치마크(정상50/불량50)→r-profile 룰과 late-fusion 비교.
- Diffusion/AE는 연구 트랙으로 데이터/시간/XAI 요구 검증.

## 5. Governance & Ops
- 설정 관리: SKU 기준 버전/이력, 변경 승인 플로우, 롤백 버튼.
- 로그/보관: 원본/전처리/오버레이/r-profile/판정 CSV, 보관 기간·익명화 규정.
- 캘리브레이션: 조명/카메라 주기적 점검 체크리스트, 실패 시 알림.
- 품질 튜닝: 월간 리뷰(오탐/미탐 사례), 파라미터 업데이트 절차 문서화.

## 6. Risks & Mitigation
- 조명 드리프트 → 주기적 캘리브레이션+광량 로그, 실패 시 운영 알람.
- 렌즈 미검출 → ROI 여유, 예외 처리 후 리트라이/버퍼 배출.
- 다품종 혼류 → SKU 미선택 시 촬영 차단, 바코드/QR 연동.
- 데이터 부족 → 불량 합성/공정 테스트로 보충, 추후 실불량 반영.
- 성능 병목 → Δr/ROI 튜닝, 벡터화, 필요 시 GPU 옵션(PatchCore 등) 분리.
