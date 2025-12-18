# Day 3 작업 계획서

> **목표**: SKU 관리 시스템 구현 및 다중 SKU 지원
> **예상 소요 시간**: 3시간
> **작업 방식**: 병렬 작업 + 명확한 핸드오프 포인트

---

## 📋 3가지 옵션 비교

| 옵션 | 설명 | 우선순위 | 난이도 | 소요 시간 |
|------|------|----------|---------|-----------|
| **Option 1** | **SKU 관리 시스템** | ⭐⭐⭐ 추천 | 중 | 3시간 |
| Option 2 | Visualizer 구현 | ⭐⭐ | 중상 | 4시간 |
| Option 3 | Logger & DB | ⭐ | 하 | 2.5시간 |

**추천 이유 (Option 1):**
- 현재 시스템은 SKU001만 지원 (실제 운영에는 수십~수백 개 SKU 필요)
- 베이스라인 측정을 자동화하면 운영 효율성 대폭 향상
- 다중 SKU 지원은 실제 생산 환경 필수 기능

---

## 🎯 Option 1: SKU 관리 시스템 (추천)

### 목표
1. **SkuConfigManager** 클래스 구현 (CRUD 기능)
2. **베이스라인 자동 생성 도구** (OK 샘플 이미지 → SKU JSON)
3. **다중 SKU 데이터베이스** 지원 및 검증
4. **CLI 확장** (SKU 등록/조회/수정/삭제)

---

## 👥 작업 분담

### Phase 1: 준비 및 설계 (병렬) - 20분

#### 👤 Claude Task C1: 설계 문서 작성 (15분)
**산출물:**
- `docs/SKU_MANAGEMENT_DESIGN.md`
  - SkuConfigManager 클래스 설계
  - SKU JSON 스키마 확장
  - 베이스라인 생성 알고리즘
  - CLI 명령어 인터페이스

**시작:** 즉시
**완료 조건:** 설계 문서 커밋

---

#### 👤 개발자 A Task A1: 다중 SKU 더미 데이터 생성 (20분)
**작업 내용:**
- **SKU002 데이터** (파란색 렌즈):
  - `data/raw_images/SKU002_OK_001~005.jpg` (5장)
  - `data/raw_images/SKU002_NG_001~005.jpg` (5장)
  - 예상 LAB: L=70, a=-10, b=-30 (파란색 계열)

- **SKU003 데이터** (갈색 렌즈):
  - `data/raw_images/SKU003_OK_001~005.jpg` (5장)
  - `data/raw_images/SKU003_NG_001~005.jpg` (5장)
  - 예상 LAB: L=50, a=15, b=25 (갈색 계열)

- **metadata.csv 업데이트:**
  ```csv
  filename,sku,expected_judgment,color_type,notes
  SKU002_OK_001.jpg,SKU002,OK,blue,Baseline sample
  SKU002_NG_001.jpg,SKU002,NG,blue,Delta_E defect
  SKU003_OK_001.jpg,SKU003,OK,brown,Baseline sample
  ...
  ```

**도구:** 기존 더미 생성 스크립트 재사용 (색상만 변경)
**시작:** 즉시
**완료 조건:** 20장 이미지 + metadata.csv 업데이트

---

### Phase 2: 핵심 구현 (순차) - 90분

#### 👤 Claude Task C2: SkuConfigManager + CLI 구현 (90분)

**⏸️ 대기 조건:**
- Phase 1 완료 후 시작 (Task A1 완료 필요)
- 실제 이미지로 베이스라인 생성 테스트 필요

**작업 내용:**

**2-1. SkuConfigManager 클래스** (`src/sku_manager.py`, 300+ lines)
```python
class SkuConfigManager:
    def __init__(self, db_path: Path = Path("config/sku_db")):
        pass

    def create_sku(self, sku_code: str, description: str, ...) -> Dict
    def get_sku(self, sku_code: str) -> Dict
    def update_sku(self, sku_code: str, updates: Dict) -> Dict
    def delete_sku(self, sku_code: str) -> bool
    def list_all_skus(self) -> List[Dict]
    def generate_baseline(
        self,
        sku_code: str,
        ok_images: List[Path],
        output_path: Path
    ) -> Dict
```

**2-2. 베이스라인 자동 생성 도구** (`tools/generate_sku_baseline.py`, 150+ lines)
```bash
# 사용 예시
python -m tools.generate_sku_baseline \
  --sku SKU002 \
  --images data/raw_images/SKU002_OK_*.jpg \
  --output config/sku_db/SKU002.json \
  --description "Blue colored lens"
```

**기능:**
- OK 샘플 5~10장 로드
- 각 이미지에서 Zone LAB 측정
- 평균 + 표준편차 계산
- threshold = mean + 2*std 자동 설정
- SKU JSON 생성 및 저장

**2-3. CLI 확장** (`src/main.py` 수정, +150 lines)
```bash
# SKU 관리 명령어
python -m src.main sku list
python -m src.main sku show SKU001
python -m src.main sku create --code SKU002 --desc "Blue lens"
python -m src.main sku generate-baseline --sku SKU002 --images data/raw_images/SKU002_OK_*.jpg
python -m src.main sku delete SKU002

# 기존 검사 명령어 (변경 없음)
python -m src.main inspect --image data/raw_images/SKU002_OK_001.jpg --sku SKU002
python -m src.main batch --dir data/raw_images --pattern "SKU002_*.jpg" --sku SKU002
```

**2-4. SKU002, SKU003 베이스라인 생성**
- `config/sku_db/SKU002.json` (자동 생성)
- `config/sku_db/SKU003.json` (자동 생성)

**완료 조건:**
- SkuConfigManager 클래스 완성
- 베이스라인 생성 도구 완성
- CLI 명령어 동작 확인
- SKU002, SKU003 베이스라인 생성 완료

---

### Phase 3: 검증 및 확장 (병렬) - 60분

#### 👤 Claude Task C3: 통합 테스트 (30분)

**⏸️ 대기 조건:** Task C2 완료 후

**작업 내용:**
- `tests/test_sku_manager.py` (200+ lines, 15개 테스트)
  - test_create_sku()
  - test_get_sku()
  - test_update_sku()
  - test_delete_sku()
  - test_list_all_skus()
  - test_generate_baseline_single_zone()
  - test_generate_baseline_multi_zone()
  - test_baseline_threshold_calculation()
  - test_multi_sku_batch_processing()
  - test_invalid_sku_handling()
  - test_sku_json_schema_validation()
  - test_cli_sku_commands()
  - ...

**검증:**
- 전체 테스트 통과 (113개 → 128개)
- SKU002, SKU003 검사 정확도 확인
- 성능 테스트 (<200ms/장 유지)

**완료 조건:** 전체 테스트 통과

---

#### 👤 개발자 B Task B1: Jupyter Notebook 확장 (60분)

**⏸️ 대기 조건:** Task C2 완료 후 (SkuConfigManager 사용 필요)

**작업 내용:**
- `notebooks/02_multi_sku_analysis.ipynb` (신규 생성)

**섹션 구성 (8개):**
1. **환경 설정** - SkuConfigManager 임포트
2. **SKU 목록 조회** - 전체 SKU 리스트 표시
3. **SKU 비교 시각화** - SKU001 vs SKU002 vs SKU003 LAB 비교
4. **베이스라인 생성 데모** - OK 샘플로 자동 생성 과정 시각화
5. **다중 SKU 배치 처리** - 3개 SKU 동시 검사
6. **SKU별 통계** - OK/NG 비율, 평균 ΔE
7. **Zone 패턴 분석** - SKU별 Zone 개수 및 분포
8. **대시보드** - SKU 관리 현황 요약

**완료 조건:** Notebook 실행 가능 + 시각화 완성

---

### Phase 4: 최종 검증 및 문서화 (순차) - 30분

#### 👥 전체 작업자 (Claude + 개발자 A + 개발자 B)

**⏸️ 대기 조건:** Task C3, B1 모두 완료

**검증 항목:**
1. ✅ SKU002, SKU003 베이스라인 자동 생성 성공
2. ✅ 3개 SKU 동시 배치 처리 성공
3. ✅ CLI SKU 관리 명령어 동작 확인
4. ✅ 전체 테스트 통과 (128개)
5. ✅ Jupyter Notebook 실행 가능
6. ✅ 성능 기준 충족 (<200ms/장)
7. ✅ 문서화 완료

**문서 작성 (Claude):**
- `DAY3_COMPLETION_REPORT.md`
- `README.md` 업데이트 (SKU 관리 섹션 추가)
- `docs/USER_GUIDE.md` (SKU 등록 가이드)

**Git 커밋:**
```bash
git add -A
git commit -m "feat: Day 3 - Implement SKU management system with multi-SKU support

- Add SkuConfigManager for CRUD operations
- Implement automatic baseline generation from OK samples
- Extend CLI with SKU management commands
- Add SKU002 (blue) and SKU003 (brown) test data
- Create multi-SKU analysis Jupyter notebook
- Add 15 integration tests (113 → 128 total)
- Update documentation with SKU management guide

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## 📊 성공 기준 (7개)

| # | 기준 | 목표 | 검증 방법 |
|---|------|------|-----------|
| 1 | SKU CRUD 동작 | create/read/update/delete 성공 | CLI 명령어 실행 |
| 2 | 베이스라인 자동 생성 | SKU002, SKU003 JSON 생성 | 파일 존재 + 스키마 검증 |
| 3 | 다중 SKU 배치 처리 | 3개 SKU 동시 검사 성공 | 30장 처리 (각 SKU 10장) |
| 4 | 테스트 통과 | 128개 테스트 통과 | pytest 실행 |
| 5 | CLI 확장 | `sku` 서브커맨드 동작 | help 출력 + 실행 확인 |
| 6 | Jupyter Notebook | 8개 섹션 실행 가능 | 전체 셀 실행 |
| 7 | 성능 유지 | <200ms/장 (다중 SKU) | 성능 테스트 |

---

## ⏱️ 타임라인

```
00:00 ━━━━━━━━━━━━━━━━━ Phase 1 시작 (병렬)
       ├─ Claude C1 (설계 문서)
       └─ 개발자 A (SKU002/003 데이터 생성)

00:20 ━━━━━━━━━━━━━━━━━ Phase 2 시작 (순차)
       └─ Claude C2 (SkuConfigManager + CLI)
       ⏸️ 개발자 A, B 대기

01:50 ━━━━━━━━━━━━━━━━━ Phase 3 시작 (병렬)
       ├─ Claude C3 (통합 테스트)
       └─ 개발자 B (Jupyter Notebook)

02:20 ━━━━━━━━━━━━━━━━━ Phase 4 시작 (전체)
       └─ 최종 검증 + 문서화

03:00 ━━━━━━━━━━━━━━━━━ 완료 🎉
```

---

## 📦 예상 산출물

### 코드
- `src/sku_manager.py` (300+ lines) - SkuConfigManager 클래스
- `tools/generate_sku_baseline.py` (150+ lines) - 베이스라인 생성 도구
- `src/main.py` (+150 lines) - CLI SKU 서브커맨드
- `tests/test_sku_manager.py` (200+ lines, 15개 테스트)

### 데이터
- `config/sku_db/SKU002.json` (자동 생성)
- `config/sku_db/SKU003.json` (자동 생성)
- `data/raw_images/SKU002_*.jpg` (10장, by 개발자 A)
- `data/raw_images/SKU003_*.jpg` (10장, by 개발자 A)
- `data/raw_images/metadata.csv` (업데이트)

### 문서
- `docs/SKU_MANAGEMENT_DESIGN.md` (설계)
- `docs/USER_GUIDE.md` (사용자 가이드)
- `DAY3_COMPLETION_REPORT.md` (완료 보고서)
- `notebooks/02_multi_sku_analysis.ipynb` (by 개발자 B)

**총 신규 코드:** ~800 lines
**총 테스트:** 128개 (113 → +15)

---

## 🔄 핸드오프 포인트 요약

### 🚦 누가 누구를 기다리는가?

**Phase 1 → Phase 2:**
- ✋ **Claude Task C2**는 **개발자 A Task A1** 완료 대기
  - 이유: SKU002/003 실제 이미지로 베이스라인 생성 테스트 필요

**Phase 2 → Phase 3:**
- ✋ **Claude Task C3**는 **Claude Task C2** 완료 대기
  - 이유: SkuConfigManager 구현 완료 후 테스트 가능
- ✋ **개발자 B Task B1**은 **Claude Task C2** 완료 대기
  - 이유: SkuConfigManager API 사용 필요

**Phase 3 → Phase 4:**
- ✋ **전체 작업자**는 **Task C3, B1 모두** 완료 대기
  - 이유: 통합 검증 및 최종 문서화

---

## 💡 주요 설계 결정

### 1. SKU JSON 스키마 확장
```json
{
  "sku_code": "SKU002",
  "description": "Blue colored contact lens",
  "default_threshold": 3.5,
  "zones": {
    "A": {"L": 70, "a": -10, "b": -30, "threshold": 4.0},
    "B": {"L": 65, "a": -8, "b": -25, "threshold": 3.5}
  },
  "metadata": {
    "created_at": "2025-12-11T14:00:00",
    "baseline_samples": 5,
    "last_updated": "2025-12-11T14:00:00",
    "calibration_method": "auto_generated",
    "statistics": {
      "zone_A_std": {"L": 0.5, "a": 0.3, "b": 0.4}
    }
  }
}
```

### 2. 베이스라인 생성 알고리즘
```
1. OK 샘플 5~10장 로드
2. 각 이미지 파이프라인 처리 → Zone LAB 추출
3. Zone별 LAB 평균 계산: mean(L), mean(a), mean(b)
4. Zone별 표준편차 계산: std(L), std(a), std(b)
5. Threshold 설정: default + 2*max(std)
6. SKU JSON 생성 및 저장
```

### 3. CLI 계층 구조
```bash
python -m src.main
  ├─ inspect         # 단일 이미지 검사 (기존)
  ├─ batch           # 배치 처리 (기존)
  └─ sku             # SKU 관리 (신규)
      ├─ list
      ├─ show <sku_code>
      ├─ create --code --desc
      ├─ generate-baseline --sku --images
      ├─ update <sku_code> --field value
      └─ delete <sku_code>
```

---

## ❓ 리스크 및 대응

| 리스크 | 확률 | 영향 | 대응 방안 |
|--------|------|------|-----------|
| 더미 데이터 품질 불량 | 중 | 중 | 개발자 A에게 색상 범위 명확히 전달 |
| 베이스라인 생성 실패 | 낮 | 고 | 수동 fallback 옵션 제공 |
| 다중 SKU 성능 저하 | 낮 | 중 | 각 SKU별 독립 처리로 격리 |
| 테스트 시간 초과 | 중 | 낮 | 성능 테스트 timeout 완화 |

---

## ✅ 시작 전 체크리스트

- [ ] Day 2 완료 확인 (Git 커밋 완료)
- [ ] 113개 테스트 통과 확인
- [ ] 개발자 A, B 준비 상태 확인
- [ ] Option 1 (SKU 관리) 최종 승인

---

**작성자**: Claude (AI Assistant)
**검토 필요**: 개발자 A, 개발자 B
**작성일**: 2025-12-11
