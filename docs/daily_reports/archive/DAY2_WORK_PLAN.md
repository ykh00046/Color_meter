# Day 2 작업 분담 계획 (상세)

> **목표**: 엔드투엔드 통합 파이프라인 완성 및 Jupyter Notebook 프로토타입 검증
>
> **예상 소요 시간**: 2-3시간
>
> **최종 산출물**:
> - ✅ 실행 가능한 CLI 프로그램 (`python src/main.py --image ... --sku ...`)
> - ✅ 시각화된 Jupyter Notebook 데모 (`notebooks/01_prototype.ipynb`)
> - ✅ 10장 샘플 이미지 검증 완료

---

## 📋 작업 의존성 그래프

```
[작업 흐름도]

Phase 1: 준비 (병렬 가능)
├─ 🤖 Claude: Task C1 (파이프라인 설계 문서)    [15분]
└─ 🔴 개발자 A: Task A1 (더미 데이터 생성)      [20분]

↓ (모두 완료 후)

Phase 2: 핵심 구현 (순차)
└─ 🤖 Claude: Task C2 (파이프라인 구현)         [60분]
                     └─ InspectionPipeline 클래스
                     └─ 하드코딩 SKU 기준값
                     └─ main.py CLI

↓ (Claude Task C2 완료 후)

Phase 3: 검증 (병렬 가능)
├─ 🤖 Claude: Task C3 (파이프라인 테스트)        [20분]
│                    └─ 10장 샘플 실행
│                    └─ 결과 CSV 저장
│
└─ 🔵 개발자 B: Task B1 (Jupyter Notebook)      [60분]
                       └─ 파이프라인 임포트
                       └─ 중간 결과 시각화
                       └─ Zone 색상 표시

↓ (모두 완료 후)

Phase 4: 최종화
└─ 전체: Task ALL (최종 검증 및 문서 업데이트)   [15분]
```

---

## 🤖 Claude의 작업 (핵심 파이프라인)

### ✅ **Task C1: 파이프라인 설계 문서** (15분)
**시작 조건**: 즉시 시작 가능
**목표**: 파이프라인 인터페이스 설계 및 SKU 기준값 포맷 정의

**산출물**:
1. `docs/PIPELINE_DESIGN.md` - 파이프라인 설계 문서
2. SKU 기준값 JSON 포맷 정의

**다음 작업**: Claude Task C2

---

### ✅ **Task C2: InspectionPipeline 구현** (60분)
**시작 조건**: Task C1 완료
**목표**: 5개 모듈을 연결하는 통합 파이프라인 구현

**구현 파일**:
1. `src/pipeline.py` - `InspectionPipeline` 클래스
   ```python
   class InspectionPipeline:
       def __init__(self, sku_config: dict)
       def process(self, image_path: str) -> InspectionResult
       def process_batch(self, image_paths: list) -> list[InspectionResult]
   ```

2. `src/main.py` - CLI 엔트리포인트
   ```bash
   python src/main.py --image data/raw_images/OK_001.jpg --sku SKU001
   python src/main.py --batch data/raw_images/ --output results.csv
   ```

3. `config/sku_db/SKU001.json` - 하드코딩된 SKU 기준값 (임시)
   ```json
   {
     "sku_code": "SKU001",
     "default_threshold": 3.0,
     "zones": {
       "A": {"L": 45.0, "a": 10.0, "b": -40.0, "threshold": 4.0},
       "B": {"L": 70.0, "a": -5.0, "b": 60.0, "threshold": 3.5}
     }
   }
   ```

**핵심 기능**:
- ImageLoader → LensDetector → RadialProfiler → ZoneSegmenter → ColorEvaluator 연결
- 각 단계별 중간 결과 저장 (디버깅용)
- 에러 처리 및 로깅
- 결과를 JSON/CSV로 저장

**검증**:
```bash
# 단일 이미지 테스트
python src/main.py --image data/raw_images/OK_001.jpg --sku SKU001

# 배치 테스트
python src/main.py --batch data/raw_images/ --output results.csv
```

**다음 작업**:
- Claude Task C3 (파이프라인 테스트)
- 개발자 B Task B1 (Jupyter Notebook) - **병렬 시작 가능**

---

### ✅ **Task C3: 파이프라인 통합 테스트** (20분)
**시작 조건**: Task C2 완료
**목표**: 10장 샘플 이미지 전체 검증

**작업 내용**:
1. 10장 샘플 전체 실행
   ```bash
   python src/main.py --batch data/raw_images/ --output results/day2_validation.csv
   ```

2. 결과 분석
   - OK 5장: 모두 OK 판정 확인
   - NG 5장: 모두 NG 판정 확인
   - 평균 처리 시간 측정

3. 통합 테스트 케이스 작성 (`tests/test_pipeline.py`)

**산출물**:
- `results/day2_validation.csv` - 검증 결과
- `tests/test_pipeline.py` - 통합 테스트

**다음 작업**: Task ALL (최종 검증)

---

## 🔴 개발자 A의 작업 (데이터 준비)

### ✅ **Task A1: 더미 이미지 생성 및 검증** (20분)
**시작 조건**: 즉시 시작 가능 (Claude Task C1과 병렬)
**목표**: 테스트용 이미지 및 SKU 기준값 준비

**작업 내용**:
1. 더미 데이터 생성 실행
   ```bash
   python tools/generate_dummy_data.py
   ```

2. 생성된 이미지 확인
   - `data/raw_images/OK_001.jpg ~ OK_005.jpg` (양품 5장)
   - `data/raw_images/NG_001.jpg ~ NG_005.jpg` (불량 5장)
   - `data/raw_images/metadata.csv`

3. SKU 기준값 검토
   - Claude가 작성한 `config/sku_db/SKU001.json` 확인
   - 필요시 값 조정 제안

**검증**:
```bash
# 이미지 개수 확인
ls data/raw_images/*.jpg | wc -l  # 10개 확인

# metadata.csv 확인
cat data/raw_images/metadata.csv
```

**산출물**:
- ✅ 10장 렌즈 이미지 (data/raw_images/)
- ✅ metadata.csv
- ✅ SKU 기준값 검토 완료

**다음 작업**:
- **대기**: Claude Task C2 완료 대기
- Claude Task C2 완료 후 Task C3 협업 (결과 검증)

---

## 🔵 개발자 B의 작업 (Jupyter Notebook 프로토타입)

### ✅ **Task B1: Jupyter Notebook 프로토타입 작성** (60분)
**시작 조건**: **Claude Task C2 완료 대기** ⏸️
**목표**: 파이프라인의 각 단계를 시각화하는 Notebook 작성

**⚠️ 중요**:
- Claude가 `src/pipeline.py` 완성할 때까지 **대기**
- 대기 중 Notebook 구조 설계 가능 (아래 템플릿 참고)

**작업 내용**:
1. `notebooks/01_prototype.ipynb` 생성

2. Notebook 구조 (7개 섹션):
   ```
   # 1. 환경 설정
   - 모듈 임포트
   - 경로 설정

   # 2. 이미지 로드 및 전처리 (ImageLoader)
   - 원본 이미지 표시
   - 전처리 결과 비교 (ROI, 화이트밸런스)

   # 3. 렌즈 검출 (LensDetector)
   - 검출된 중심/반경 오버레이
   - 신뢰도 점수 표시

   # 4. 극좌표 변환 및 프로파일 추출 (RadialProfiler)
   - 극좌표 이미지 표시
   - L*a*b* 프로파일 그래프

   # 5. Zone 분할 (ZoneSegmenter)
   - Zone 경계선 오버레이
   - Zone별 평균 색상 표시

   # 6. 색상 평가 (ColorEvaluator)
   - Zone별 ΔE 바 차트
   - OK/NG 판정 결과

   # 7. 전체 파이프라인 실행
   - 10장 배치 처리
   - 결과 요약 테이블
   ```

3. 시각화 코드 작성
   - Matplotlib/Seaborn 사용
   - Zone별 색상을 원본 이미지에 오버레이
   - ΔE 그래프 (허용치 대비)

**핵심 코드 예시**:
```python
# Section 7: 전체 파이프라인 실행
from src.pipeline import InspectionPipeline
from src.utils.file_io import read_json

# SKU 설정 로드
sku_config = read_json('config/sku_db/SKU001.json')

# 파이프라인 초기화
pipeline = InspectionPipeline(sku_config)

# 단일 이미지 처리
result = pipeline.process('data/raw_images/OK_001.jpg')

# 결과 시각화
print(f"판정: {result.judgment}")
print(f"평균 ΔE: {result.overall_delta_e:.2f}")

# Zone별 결과 표시
for zone_result in result.zone_results:
    print(f"Zone {zone_result.zone_name}: ΔE={zone_result.delta_e:.2f} ({'OK' if zone_result.is_ok else 'NG'})")
```

**산출물**:
- ✅ `notebooks/01_prototype.ipynb` (실행 가능한 Notebook)
- ✅ 각 단계별 시각화 포함

**다음 작업**: Task ALL (최종 검증)

---

## 👥 전체 작업 (최종 검증)

### ✅ **Task ALL: 최종 검증 및 문서 업데이트** (15분)
**시작 조건**: Claude Task C3 + 개발자 B Task B1 완료
**목표**: Day 2 완료 확인 및 문서 업데이트

**작업 내용** (전체 협업):
1. **결과 확인**
   - CLI: 10장 전체 정상 동작 확인
   - Notebook: 시각화 정상 표시 확인

2. **문서 업데이트**
   - `WORK_ASSIGNMENT.md`: Day 2 완료 체크
   - `docs/DEVELOPMENT_GUIDE.md`: "오늘 해야 할 일" 업데이트
   - `README.md`: "개발 로드맵" Stage 1 완료 표시

3. **Git 커밋**
   ```bash
   git add -A
   git commit -m "feat: Complete Day 2 - Integrate pipeline and Jupyter prototype"
   git push
   ```

**산출물**:
- ✅ Day 2 완료 보고서
- ✅ Git 커밋 이력

---

## ⏱️ 타임라인 (총 2-3시간)

| 시간 | Claude 🤖 | 개발자 A 🔴 | 개발자 B 🔵 |
|------|-----------|-------------|-------------|
| **0:00-0:15** | Task C1 (설계) | Task A1 (데이터 생성) | 대기 (Notebook 구조 설계) |
| **0:15-1:15** | Task C2 (파이프라인 구현) | **대기** (C2 완료 대기) | **대기** (C2 완료 대기) |
| **1:15-1:35** | Task C3 (테스트) | Task C3 협업 (결과 검증) | Task B1 시작 (Notebook) |
| **1:35-2:35** | 휴식 또는 버그 수정 | 휴식 또는 버그 수정 | Task B1 계속 (Notebook) |
| **2:35-2:50** | Task ALL (최종 검증) | Task ALL (최종 검증) | Task ALL (최종 검증) |

---

## 🚨 대기 시점 명확화

### **개발자 A의 대기 시점**
```
Task A1 완료 (0:15)
    ↓
⏸️  Claude Task C2 완료 대기 (1:15까지 대기)
    ↓
Task C3 협업 시작 (1:15)
```

**대기 중 할 일**:
- 생성된 이미지 품질 검토
- metadata.csv 정확성 확인
- SKU 기준값 사전 검토
- 문서 미리 읽기 (DETAILED_IMPLEMENTATION_PLAN.md)

---

### **개발자 B의 대기 시점**
```
Notebook 구조 설계 (0:00-0:15)
    ↓
⏸️  Claude Task C2 완료 대기 (1:15까지 대기)
    ↓
Task B1 시작 (1:15) - 파이프라인 임포트 가능
    ↓
Task B1 계속 (2:35까지)
```

**대기 중 할 일**:
- Notebook 섹션 구조 작성 (Markdown 셀)
- 시각화 레이아웃 설계
- Matplotlib 코드 템플릿 준비
- 샘플 이미지 미리 확인

---

## 📤 산출물 체크리스트

### Claude 산출물
- [ ] `docs/PIPELINE_DESIGN.md`
- [ ] `src/pipeline.py` (InspectionPipeline 클래스)
- [ ] `src/main.py` (CLI 엔트리포인트)
- [ ] `config/sku_db/SKU001.json`
- [ ] `tests/test_pipeline.py`
- [ ] `results/day2_validation.csv`

### 개발자 A 산출물
- [ ] 10장 이미지 (`data/raw_images/*.jpg`)
- [ ] `data/raw_images/metadata.csv`
- [ ] SKU 기준값 검토 완료

### 개발자 B 산출물
- [ ] `notebooks/01_prototype.ipynb`
- [ ] 7개 섹션 모두 실행 가능
- [ ] 시각화 완성

---

## 🔧 블로킹 해결 방안

### **블로킹 시나리오 1**: Claude Task C2가 지연될 경우
**해결책**:
- 개발자 A: 더미 이미지 품질 개선 (색상, 크기 조정)
- 개발자 B: 모의(Mock) 파이프라인 작성 후 Notebook 구조 먼저 완성

### **블로킹 시나리오 2**: 파이프라인 실행 시 에러
**해결책**:
- Claude: 즉시 디버깅 및 수정
- 개발자 A, B: 로그 확인 및 입력 데이터 검증 지원

### **블로킹 시나리오 3**: SKU 기준값이 부적절
**해결책**:
- 개발자 A: 이미지 실측값 기반 기준값 제안
- Claude: SKU JSON 즉시 수정 및 재실행

---

## 💬 커뮤니케이션 체크포인트

### **체크포인트 1** (0:15 - Phase 1 완료)
- Claude: "Task C1 완료, 설계 문서 공유"
- 개발자 A: "Task A1 완료, 이미지 10장 생성 완료"
- 개발자 B: "Notebook 구조 설계 완료, C2 대기 중"

### **체크포인트 2** (1:15 - Phase 2 완료)
- Claude: "Task C2 완료, 파이프라인 실행 가능"
- 개발자 A: "Task C3 시작, 10장 실행 중"
- 개발자 B: "Task B1 시작, 파이프라인 임포트 성공"

### **체크포인트 3** (2:35 - Phase 3 완료)
- 전체: "Phase 3 완료, 최종 검증 시작"

---

## ✅ 성공 기준

Day 2 완료 조건:
1. ✅ CLI로 단일 이미지 처리 가능
2. ✅ CLI로 10장 배치 처리 가능
3. ✅ 결과 CSV 파일 생성
4. ✅ Jupyter Notebook 전체 실행 가능
5. ✅ OK 5장 모두 OK 판정
6. ✅ NG 5장 모두 NG 판정 (또는 ΔE 값 차이 확인)
7. ✅ 평균 처리 시간 <500ms/장

---

**작성일**: 2025-12-11
**작성자**: Claude (AI Assistant)
**버전**: 1.0
