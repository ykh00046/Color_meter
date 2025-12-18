# 검토 의견 및 개선 계획

> **검토일**: 2025-12-17
> **검토자**: 사용자 (품질 관리 전문가)
> **대상 문서**: STD_BASED_QC_SYSTEM_PLAN.md
> **상태**: 🟡 개선 작업 진행 중

---

## 📋 검토 결과 요약

### 종합 평가

| 항목 | 평점 | 코멘트 |
|------|------|--------|
| **기술적 타당성** | ★★★★☆ (4/5) | 기존 파이프라인 재사용, 아키텍처 명확 |
| **일정 현실성** | ★★★☆☆ (3/5) | MVP 우선순위 타당하나 10주는 도전적 |
| **우선순위 적절성** | ★★★★★ (5/5) | P0/P1/P2 구분 명확, 단계별 계획 구체적 |
| **전반적 평가** | ★★★★☆ (4/5) | **탄탄한 기초, 성능 검증 및 운영 강화 필요** |

**한 줄 요약**:
> "탄탁한 기초 위에 설계된 STD 기반 품질 검사 시스템이지만, 고성능 알고리즘 검증과 운영 안전성 확보가 과제로 남아있음."

---

## ✅ 긍정적 측면

### 1. 기술 기반의 견고함
- ✅ 단일 이미지 분석 파이프라인 완성 (94.7% 테스트 커버리지)
- ✅ 극좌표 변환, CIEDE2000, Zone 분할 등 핵심 기능 구현 완료
- ✅ 기존 기능을 재활용하여 STD 프로파일 생성 가능

### 2. 명확한 아키텍처
- ✅ 3-tier 구조 (Frontend/Backend/DB) 명확
- ✅ 서비스별 모듈화 (STD Profile/Comparison/Judgment)
- ✅ FastAPI 기반 REST API로 유연한 확장 가능

### 3. 구체적인 워크플로우
- ✅ STD 등록 → 양산 비교 → 조치 권장 → 판정 전 단계 정의
- ✅ 사용자 시나리오 3가지 명확 (엔지니어/검사자/관리자)

### 4. 체계적인 단계별 계획
- ✅ Phase 0~4 (10주) 구체적 마일스톤
- ✅ MVP (Week 6) 범위 명확: STD 등록 + 비교 분석 기본
- ✅ 우선순위 (P0/P1/P2) 잘 구분됨

---

## ⚠️ 우려 사항

### 1. 알고리즘 계산 비용
**문제**: DTW O(n²) 복잡도로 3초 목표 달성 어려울 수 있음

**상세**:
- Radial Profile 길이: ~500 포인트
- DTW 계산: 500 × 500 = 250,000 연산
- 3개 채널 (L, a, b) → 750,000 연산
- 추가 알고리즘 (상관계수, KS test) 포함 시 더 증가

**영향**:
- 비교 분석 > 3초 → 사용자 경험 저하
- 배치 처리 시 병목 발생

### 2. 데이터베이스 설계 문제
**문제**: JSON 칼럼 과다 사용 → 검색/분석 비효율

**상세**:
```sql
-- 현재 설계
profile_data JSON  -- 프로파일 전체를 JSON에
color_data JSON    -- 색상 통계 전체를 JSON에

-- 문제점
SELECT * FROM std_profiles
WHERE color_data->>'zones'->0->'mean_lab'->0 > 70.0  -- 복잡, 인덱스 불가
```

**영향**:
- Zone별 평균값 검색 어려움
- 통계 분석 (평균 ΔE, 트렌드) 비효율
- PostgreSQL JSON 함수 의존

### 3. 운영 요구사항 누락
**문제**: 보안, 백업, 감사 로그 미계획

**누락 항목**:
- [ ] 사용자 인증/권한 관리
- [ ] STD 승인 워크플로우
- [ ] 감사 로그 (누가, 언제, 무엇을)
- [ ] 백업 및 복구 전략
- [ ] 이미지 파일 관리 (DB vs 파일시스템)

**영향**:
- 데이터 무결성 위협
- 규정 준수 (Compliance) 문제
- 장애 복구 지연

### 4. 일정 리스크
**문제**: 10주에 모든 기능 완성은 도전적

**세부 리스크**:
- Phase 2 (비교 엔진 3주): 알고리즘 개발 + 검증 + UI
- Phase 3 (권장 2주): 실측 데이터 기반 튜닝 필요
- 예외 처리 (불완전 이미지, 복수 렌즈 등) 미고려

**영향**:
- 일정 지연 가능성 50% 이상
- 품질 희생 또는 범위 축소 필요

---

## 🔧 개선 제안 (반영 계획)

### 1. 알고리즘 성능 강화 🔴 P0

#### 1.1 DTW 최적화
```python
# 현재 계획
from dtaidistance import dtw
distance = dtw.distance(test, std)  # O(n²)

# 개선안
from dtaidistance import dtw_ndim
# 1) FastDTW 사용 (근사 알고리즘, O(n))
distance = dtw_ndim.distance_fast(test, std, window=10)

# 2) 프로파일 다운샘플링
test_downsampled = test[::2]  # 500 → 250 포인트
std_downsampled = std[::2]

# 3) C 라이브러리 활용
import dtw_c  # Custom C extension
```

#### 1.2 대안 알고리즘 추가
```python
# Cross-correlation (상관계수보다 이동 허용)
from scipy.signal import correlate
corr = correlate(test, std, mode='valid')
max_corr = np.max(corr) / (len(test) * np.std(test) * np.std(std))

# PCA 기반 형태 비교
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
test_pca = pca.fit_transform(test.reshape(-1, 1))
std_pca = pca.transform(std.reshape(-1, 1))
distance_pca = np.linalg.norm(test_pca - std_pca)
```

#### 1.3 성능 벤치마크
```python
# Phase 0에 포함할 프로토타입
def benchmark_algorithms():
    profiles = load_sample_profiles(n=100)

    algorithms = {
        "pearson": profile_correlation,
        "dtw_full": profile_dtw_distance,
        "dtw_fast": profile_dtw_fast,
        "cross_corr": profile_cross_correlation,
    }

    for name, func in algorithms.items():
        times = []
        for i in range(100):
            start = time.time()
            func(profiles[i], profiles[0])
            times.append(time.time() - start)

        print(f"{name}: avg={np.mean(times):.3f}s, max={np.max(times):.3f}s")

    # 목표: avg < 1.0s, max < 3.0s
```

**일정**: Phase 0 (Week 1) - 벤치마크 및 알고리즘 선정

---

### 2. 데이터베이스 설계 최적화 🔴 P0

#### 2.1 JSON 최소화 + 주요 필드 분리

**개선된 스키마**:
```sql
CREATE TABLE std_profiles (
    id SERIAL PRIMARY KEY,
    sku_code VARCHAR(50) NOT NULL,
    version INTEGER NOT NULL,
    image_path VARCHAR(500) NOT NULL,

    -- 🔧 주요 필드 분리 (검색/인덱스 가능)
    zone_count INTEGER,
    overall_mean_L FLOAT,
    overall_mean_a FLOAT,
    overall_mean_b FLOAT,
    overall_delta_e_std FLOAT,  -- 표준편차

    -- JSON (상세 데이터만)
    profile_data JSONB,  -- Radial profile raw data
    meta_data JSONB,     -- 촬영 조건, 승인 정보

    -- 범위 설정
    upper_limit JSONB,
    lower_limit JSONB,

    -- 감사 정보
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(100),
    approved_by VARCHAR(100),
    approved_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,

    UNIQUE(sku_code, version)
);

CREATE TABLE std_profile_zones (
    id SERIAL PRIMARY KEY,
    std_profile_id INTEGER REFERENCES std_profiles(id) ON DELETE CASCADE,
    zone_name VARCHAR(10) NOT NULL,

    -- 🔧 Zone별 색상 정보 (인덱스 가능)
    mean_L FLOAT NOT NULL,
    mean_a FLOAT NOT NULL,
    mean_b FLOAT NOT NULL,
    std_L FLOAT,
    std_a FLOAT,
    std_b FLOAT,

    -- 경계 정보
    r_start FLOAT,
    r_end FLOAT,

    -- 상세 통계 (JSON)
    percentiles JSONB,

    pixel_count INTEGER,
    ink_pixel_count INTEGER
);

-- 인덱스
CREATE INDEX idx_std_sku ON std_profiles(sku_code, is_active);
CREATE INDEX idx_std_zone_lab ON std_profile_zones(mean_L, mean_a, mean_b);
```

**장점**:
- ✅ Zone별 색상 검색: `SELECT * FROM std_profile_zones WHERE mean_L > 70`
- ✅ 통계 분석: `SELECT AVG(mean_L) FROM std_profile_zones WHERE zone_name='A'`
- ✅ 인덱스 활용 가능

#### 2.2 이미지 파일 저장 전략

**현재 계획**: DB blob vs 파일시스템 (미정)

**개선안**: **파일시스템 (추천)**
```python
# 저장 경로 구조
data/
  std_images/
    SKU001/
      v1_20251217_150030.jpg
      v2_20251220_093015.jpg
    SKU002/
      v1_20251218_114522.jpg
  test_samples/
    SKU001/
      2025-12-17/
        sample_001.jpg
        sample_002.jpg

# DB에는 경로만 저장
image_path = "data/std_images/SKU001/v1_20251217_150030.jpg"
```

**장점**:
- ✅ DB 크기 관리 용이
- ✅ 백업 별도 관리 (이미지 파일 vs DB)
- ✅ 파일 접근 속도 빠름 (CDN 연동 가능)

**단점**:
- ⚠️ 파일-DB 동기화 필요
- ⚠️ 파일 삭제 시 고아 레코드 방지 로직 필요

**일정**: Phase 0 (Week 1) - 스키마 확정 및 마이그레이션

---

### 3. 운영 요구사항 추가 🟡 P1

#### 3.1 사용자 인증 및 권한 관리

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL,  -- admin, engineer, inspector
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    action VARCHAR(50) NOT NULL,  -- std_register, std_approve, test_compare
    target_type VARCHAR(50),      -- std_profile, test_sample
    target_id INTEGER,
    details JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**권한 매트릭스**:
| 역할 | STD 등록 | STD 승인 | 양산 검사 | 결과 조회 | 범위 설정 |
|------|---------|---------|----------|----------|----------|
| **Admin** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Engineer** | ✅ | ❌ | ✅ | ✅ | ❌ |
| **Inspector** | ❌ | ❌ | ✅ | ✅ | ❌ |

#### 3.2 STD 승인 워크플로우

```python
class STDApprovalWorkflow:
    def register_std(self, image_path: str, sku: str, user: User) -> STDProfile:
        """STD 등록 (상태: PENDING)"""
        profile = self.manager.create_std_profile(image_path, sku)
        profile.status = "PENDING"
        profile.created_by = user.username
        self.db.save(profile)

        # 감사 로그
        self.audit_log(user, "std_register", profile.id)

        return profile

    def approve_std(self, profile_id: int, approver: User) -> bool:
        """STD 승인 (관리자만 가능)"""
        if approver.role != "admin":
            raise PermissionError("Only admin can approve STD")

        profile = self.db.get_std_profile(profile_id)
        profile.status = "ACTIVE"
        profile.approved_by = approver.username
        profile.approved_at = datetime.now()

        # 기존 active STD를 비활성화
        self.db.deactivate_other_versions(profile.sku_code)

        self.db.save(profile)
        self.audit_log(approver, "std_approve", profile_id)

        return True
```

#### 3.3 백업 및 복구 전략

**자동 백업**:
```bash
# 일일 백업 (cron)
0 2 * * * /scripts/backup_db.sh   # DB 덤프
0 3 * * * /scripts/backup_images.sh  # 이미지 rsync

# backup_db.sh
pg_dump lens_inspection > /backups/db_$(date +%Y%m%d).sql
# 7일치 보관, 이후 삭제
find /backups -name "db_*.sql" -mtime +7 -delete
```

**복구 절차**:
```markdown
1. DB 복구
   psql lens_inspection < /backups/db_20251217.sql

2. 이미지 복구
   rsync -av /backups/images/ /data/std_images/

3. 검증
   - STD 프로파일 개수 확인
   - 이미지 파일 무결성 체크 (md5sum)
   - 최신 테스트 샘플 재분석
```

**일정**: Phase 1 (Week 2-3) - 인증/권한 구현, Phase 2 (Week 4) - 백업 자동화

---

### 4. AI/ML 도입 고려 🟢 P2 (장기)

#### 4.1 조치 권장 학습

```python
class InkAdjustmentRecommender:
    def __init__(self):
        self.model = None  # 추후 ML 모델
        self.history = []  # 엔지니어 피드백

    def recommend(self, test_lab: np.ndarray, std_lab: np.ndarray) -> dict:
        """색상 조정 권장"""
        # 초기: Rule-based
        dL, da, db = test_lab - std_lab

        # 나중: ML-based (피드백 학습)
        if self.model:
            adjustment = self.model.predict([[dL, da, db]])
        else:
            adjustment = self._rule_based(dL, da, db)

        return {
            "dL": float(adjustment[0]),
            "da": float(adjustment[1]),
            "db": float(adjustment[2]),
            "confidence": 0.8  # 모델 신뢰도
        }

    def collect_feedback(self, recommendation: dict, actual: dict, effective: bool):
        """엔지니어 피드백 수집"""
        self.history.append({
            "recommended": recommendation,
            "actual": actual,
            "effective": effective,
            "timestamp": datetime.now()
        })

        # 100건 이상 수집 시 모델 학습
        if len(self.history) >= 100:
            self._train_model()
```

#### 4.2 이상 감지 (Anomaly Detection)

```python
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.05)
        self.fitted = False

    def fit_std_samples(self, std_samples: List[STDProfile]):
        """정상 STD 샘플로 학습"""
        features = []
        for std in std_samples:
            # 특징 추출: 프로파일 형태, 색상 분포 등
            feat = self.extract_features(std)
            features.append(feat)

        self.model.fit(features)
        self.fitted = True

    def detect_anomaly(self, test_sample: dict) -> bool:
        """양산 샘플이 비정상인지 감지"""
        if not self.fitted:
            return False

        feat = self.extract_features(test_sample)
        prediction = self.model.predict([feat])

        return prediction[0] == -1  # -1 = anomaly
```

**일정**: v2.0 (Phase 4 이후, Week 11+)

---

## 🚨 리스크 완화 방안

### 1. 오진 위험 완화

**문제**: 자동 판정이 수동 검사와 불일치

**완화 방안**:
```python
# Phase 2에 포함할 검증 프로세스
class ComparisonValidator:
    def validate_accuracy(self, n_samples: int = 100):
        """자동 비교 vs 수동 검사 정확도 검증"""
        results = []

        for i in range(n_samples):
            # 1. 자동 비교
            auto_result = self.engine.compare(test[i], std)

            # 2. 수동 검사 (엔지니어)
            manual_result = input(f"Sample {i}: PASS or FAIL? ")

            # 3. 일치율 계산
            match = (auto_result.pass_fail == manual_result)
            results.append(match)

        accuracy = np.mean(results)
        print(f"Accuracy: {accuracy:.2%}")

        if accuracy < 0.95:
            print("⚠️ Accuracy too low! Adjust thresholds.")
            self.tune_thresholds()

        return accuracy
```

**검증 단계**:
1. Week 6 (MVP): 100개 샘플로 정확도 측정 (목표 > 90%)
2. Week 8: 1000개 샘플로 재검증 (목표 > 95%)
3. 운영 초기 3개월: 이중 검증 (자동 + 수동 병행)

---

### 2. 데이터 무결성 보장

**문제**: STD 프로파일 손상 또는 오용

**완화 방안**:
```python
class STDIntegrityChecker:
    def validate_profile(self, profile: STDProfile) -> bool:
        """STD 프로파일 무결성 검증"""
        checks = [
            self._check_profile_length(profile),
            self._check_color_range(profile),
            self._check_zone_boundaries(profile),
            self._check_image_exists(profile),
        ]

        return all(checks)

    def _check_profile_length(self, profile: STDProfile) -> bool:
        """프로파일 길이 검증"""
        expected = 500  # ±10%
        actual = len(profile.structure["radial_profile"]["L"])
        return 450 <= actual <= 550

    def _check_color_range(self, profile: STDProfile) -> bool:
        """색상 값 범위 검증"""
        for zone in profile.color["zones"]:
            L, a, b = zone["mean_lab"]
            if not (0 <= L <= 100 and -128 <= a <= 127 and -128 <= b <= 127):
                return False
        return True
```

**버전 관리**:
```python
def update_std(self, sku: str, new_profile: STDProfile):
    """STD 업데이트 (새 버전 생성)"""
    # 기존 버전 조회
    current = self.db.get_active_std(sku)

    # 새 버전 생성
    new_profile.version = current.version + 1
    new_profile.is_active = False  # 승인 전까지 비활성

    # 저장
    self.db.save(new_profile)

    # 롤백 가능하도록 기존 버전 유지
    # (is_active=False로 변경만)
```

---

### 3. 일정 리스크 완화

**문제**: 10주 일정 달성 어려움

**완화 방안**:

#### 3.1 일정 재조정 (버퍼 추가)

| Phase | 원래 | 조정 후 | 버퍼 |
|-------|------|---------|------|
| Phase 0: 기반 | 1주 | **1.5주** | +0.5주 (성능 벤치마크) |
| Phase 1: STD 프로파일 | 2주 | **2.5주** | +0.5주 (인증/권한) |
| Phase 2: 비교 엔진 | 3주 | **4주** | +1주 (알고리즘 검증) |
| Phase 3: 조치 권장 | 2주 | **2주** | - |
| Phase 4: 범위 관리 | 2주 | **v2.0으로 연기** | - |
| **합계** | 10주 | **10주 (MVP 범위 축소)** | - |

**MVP 범위 재정의** (Week 10):
- ✅ STD 등록 및 승인 (인증 포함)
- ✅ 양산 샘플 비교 (구조 + 색상 유사도)
- ✅ 비교 리포트 UI
- ✅ 조치 권장 (Rule-based)
- ❌ 범위 관리 (v2.0으로 연기)
- ❌ AI/ML 기능 (v2.0으로 연기)

#### 3.2 주간 체크포인트

```
Week 1.5: ✅ DB 스키마, 알고리즘 벤치마크
Week 4: ✅ STD 등록/승인 완료, 인증 구현
Week 8: ✅ 비교 엔진 완료, 정확도 검증 (95% 이상)
Week 10: ✅ MVP 릴리스, 조치 권장 포함
```

---

### 4. 운영 중단 위험 완화

**문제**: 시스템 장애 시 복구 지연

**완화 방안**:

#### 4.1 헬스 체크

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/health")
def health_check():
    """시스템 헬스 체크"""
    checks = {
        "database": check_db_connection(),
        "storage": check_image_storage(),
        "std_profiles": check_std_availability(),
    }

    healthy = all(checks.values())
    status_code = 200 if healthy else 503

    return JSONResponse(content=checks, status_code=status_code)

def check_std_availability():
    """활성 STD가 있는지 확인"""
    skus = ["SKU001", "SKU002", "SKU003"]
    for sku in skus:
        std = db.get_active_std(sku)
        if not std:
            return False
    return True
```

#### 4.2 장애 복구 절차 문서화

```markdown
# 장애 복구 절차 (Disaster Recovery)

## 1. DB 장애
1. 서비스 중단 공지
2. 최신 백업 확인: `ls -lh /backups/db_*.sql`
3. DB 복구: `psql lens_inspection < /backups/db_YYYYMMDD.sql`
4. 무결성 검증: `SELECT COUNT(*) FROM std_profiles WHERE is_active=true`
5. 서비스 재시작

## 2. 이미지 파일 손상
1. 영향 범위 확인: `md5sum -c /data/checksums.txt`
2. 백업에서 복구: `rsync -av /backups/images/ /data/std_images/`
3. STD 프로파일 재검증

## 3. 애플리케이션 장애
1. 로그 확인: `tail -f /var/log/lens_inspection/error.log`
2. 재시작: `systemctl restart lens-inspection`
3. 헬스 체크: `curl http://localhost:8000/health`
```

---

## 📊 개선 계획 우선순위

### P0 (필수, Week 1-6 MVP)
- [x] ~~검토 의견 문서화~~ (완료)
- [ ] 알고리즘 성능 벤치마크 (Week 1.5)
- [ ] DB 스키마 개선 (Week 1.5)
- [ ] STD 등록 및 승인 (Week 2-4)
- [ ] 비교 엔진 구현 (Week 5-8)
- [ ] 정확도 검증 (Week 8)
- [ ] 조치 권장 Rule-based (Week 9-10)

### P1 (중요, Week 7-10)
- [ ] 사용자 인증/권한 (Week 2-4)
- [ ] 감사 로그 (Week 4)
- [ ] 백업 자동화 (Week 4)
- [ ] 헬스 체크 (Week 10)

### P2 (추후, v2.0)
- [ ] 범위 관리 시스템
- [ ] AI/ML 조치 권장
- [ ] 이상 감지
- [ ] 실시간 모니터링

---

## 🎯 다음 단계 (즉시 착수)

### Week 1 (현재)
1. ✅ 검토 의견 반영 문서 작성 ← **현재 여기**
2. ⏭️ 알고리즘 성능 벤치마크 스크립트 작성
3. ⏭️ DB 스키마 확정 (JSON 최소화)
4. ⏭️ SQLAlchemy 모델 정의

### Week 1.5 (다음 주)
1. 알고리즘 벤치마크 실행 (100개 샘플)
2. 최적 알고리즘 선정 (DTW vs FastDTW vs Cross-corr)
3. DB 마이그레이션 스크립트 작성
4. Phase 1 착수 (STD 등록 API)

---

## 📝 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2025-12-17 | 초기 작성 (검토 의견 반영) |
| 2025-12-17 | 알고리즘 최적화, DB 개선, 운영 요구사항 추가 |
| 2025-12-17 | 리스크 완화 방안, 일정 재조정 |

---

**작성자**: Claude Sonnet 4.5 (사용자 검토 기반)
**문서 위치**: `docs/planning/REVIEW_FEEDBACK_AND_IMPROVEMENTS.md`
