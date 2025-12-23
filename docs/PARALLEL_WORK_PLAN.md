# 병렬 작업 계획서

**작성일**: 2025-12-19
**목적**: 여러 작업자가 동시에 작업할 수 있도록 독립적인 작업 분배

---

## 📊 현재 상황

### ✅ 완료된 작업
- Priority 1-3: 100% 완료
- Task 4.1: Auto-Detect Ink Config ✅
- Task 4.2: Inspection History System ✅

### 🎯 남은 작업
- Task 4.3: 통계 대시보드
- 추가 개선 작업들

---

## 🚀 병렬 작업 할당

### 작업자 A: 통계 대시보드 구현 (고우선순위)
### 작업자 B: Web UI 개선 및 통합 (중우선순위)
### 작업자 C: 리팩토링 및 최적화 (저우선순위)

---

## 👤 작업자 A: 통계 대시보드 구현

### 📋 작업 개요
- **목표**: 검사 결과 통계 시각화 대시보드 구축
- **예상 소요**: 6-8시간
- **우선순위**: 높음
- **의존성**: Task 4.2 완료 (✅ 이미 완료됨)

### 🎯 세부 작업

#### A-1: 차트 라이브러리 선택 및 설정 (1시간)
**파일**: `src/web/templates/dashboard.html` (이미 존재)

**작업 내용**:
1. Chart.js 또는 Plotly.js 선택
2. CDN 추가 또는 npm 설치
3. 기본 차트 템플릿 작성

**샘플 코드**:
```html
<!-- Chart.js 사용 예시 -->
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<canvas id="passRateChart"></canvas>
```

---

#### A-2: 일별/주별/월별 통계 차트 (2-3시간)
**파일**: `src/web/templates/dashboard.html`

**구현할 차트**:
1. **Pass Rate 트렌드** (Line Chart)
   - X축: 날짜
   - Y축: Pass Rate (%)
   - 데이터 소스: `/api/inspection/history` with date grouping

2. **Judgment 분포** (Pie Chart)
   - OK, OK_WITH_WARNING, NG, RETAKE 비율
   - 데이터 소스: `/api/inspection/history/stats/summary`

3. **일별 검사 건수** (Bar Chart)
   - X축: 날짜
   - Y축: 검사 건수
   - 색상: Judgment별 구분

**API 호출 예시**:
```javascript
// 통계 데이터 가져오기
fetch('/api/inspection/history/stats/summary?days=30')
  .then(res => res.json())
  .then(data => {
    // Chart.js로 차트 그리기
    new Chart(ctx, {
      type: 'line',
      data: { ... },
      options: { ... }
    });
  });
```

---

#### A-3: SKU별 통계 테이블 (1-2시간)
**파일**: `src/web/templates/dashboard.html`

**구현 내용**:
1. SKU별 Pass Rate, 평균 ΔE, 총 검사 건수 테이블
2. 정렬 기능 (Pass Rate 낮은 순 등)
3. 상세 보기 링크 (/history?sku_code=XXX)

**데이터 소스**:
```javascript
fetch('/api/inspection/history/stats/by-sku?days=30')
  .then(res => res.json())
  .then(data => {
    // 테이블 렌더링
  });
```

---

#### A-4: RETAKE 사유 분석 (1-2시간)
**파일**: `src/web/routers/inspection.py` (새 엔드포인트 추가)

**새 API 엔드포인트**:
```python
@router.get("/history/stats/retake-reasons")
def get_retake_reasons_stats(days: int = 7, db: Session = Depends(get_session)):
    """
    RETAKE 사유별 통계

    Returns:
        {
            "R1": 15,  # Low confidence
            "R2": 8,   # Lens detection failed
            ...
        }
    """
    # SQL query to count retake reasons
    pass
```

**차트**: Bar Chart (RETAKE 사유별 건수)

---

#### A-5: 필터링 및 날짜 범위 선택 (1시간)
**파일**: `src/web/templates/dashboard.html`

**기능**:
- 날짜 범위 선택 (7일, 30일, 90일, Custom)
- SKU 필터
- Refresh 버튼

---

### 📦 전달 자료

**컨텍스트 문서**:
1. `docs/INSPECTION_HISTORY_GUIDE.md` - API 레퍼런스
2. `src/web/templates/history.html` - 참고용 UI 템플릿
3. `src/models/inspection_models.py` - DB 스키마

**테스트 방법**:
```bash
# 서버 실행
cd C:/X/Color_total/Color_meter
python -m uvicorn src.web.app:app --host 0.0.0.0 --port 8000 --reload

# 브라우저에서 확인
http://127.0.0.1:8000/dashboard
```

**Git 브랜치**:
```bash
git checkout -b feature/statistics-dashboard
```

---

## 👤 작업자 B: Web UI 개선 및 통합

### 📋 작업 개요
- **목표**: 사용자 경험 개선 및 UI 일관성 확보
- **예상 소요**: 4-6시간
- **우선순위**: 중간
- **의존성**: 없음 (독립적)

### 🎯 세부 작업

#### B-1: 메인 페이지에 History 링크 추가 (30분)
**파일**: `src/web/templates/index.html` 또는 `index_modern.html`

**작업 내용**:
1. 네비게이션 메뉴에 "Inspection History" 링크 추가
2. 카드 형태로 "View History" 버튼 추가

**샘플 코드**:
```html
<nav class="...">
  <a href="/">Home</a>
  <a href="/history">History</a>
  <a href="/dashboard">Dashboard</a>
</nav>
```

---

#### B-2: 검사 결과 페이지에서 History 이동 링크 (1시간)
**파일**: `src/web/templates/index.html` (결과 표시 부분)

**작업 내용**:
1. 검사 완료 후 "View in History" 버튼 추가
2. session_id를 사용해 해당 검사 결과로 바로 이동
3. URL: `/history?session_id=xxx`

**구현 예시**:
```javascript
// 검사 완료 후
const sessionId = response.session_id;
const historyLink = `/history?session_id=${sessionId}`;
// 버튼 추가
```

---

#### B-3: History 페이지에 세션 ID 필터 기능 (1-2시간)
**파일**: `src/web/templates/history.html`

**작업 내용**:
1. URL 파라미터에서 session_id 읽기
2. 자동으로 해당 검사 결과 하이라이트
3. 스크롤 자동 이동

**구현**:
```javascript
// URL 파라미터 읽기
const params = new URLSearchParams(window.location.search);
const sessionId = params.get('session_id');

if (sessionId) {
  // 해당 검사 결과 찾아서 하이라이트
  viewDetail(sessionId);
}
```

---

#### B-4: 반응형 디자인 개선 (1-2시간)
**파일**: `src/web/templates/history.html`, `dashboard.html`

**작업 내용**:
1. 모바일 화면 대응 (Tailwind CSS breakpoints)
2. 테이블 가로 스크롤 개선
3. 차트 크기 자동 조정

---

#### B-5: 로딩 인디케이터 추가 (1시간)
**파일**: 모든 템플릿 파일

**작업 내용**:
1. API 호출 중 로딩 스피너 표시
2. 에러 메시지 개선 (Toast 알림)

**샘플 코드**:
```html
<div id="loading" class="hidden">
  <i class="fas fa-spinner fa-spin"></i> Loading...
</div>

<script>
async function loadData() {
  document.getElementById('loading').classList.remove('hidden');
  try {
    const data = await fetch('/api/...');
    // ...
  } finally {
    document.getElementById('loading').classList.add('hidden');
  }
}
</script>
```

---

### 📦 전달 자료

**컨텍스트 문서**:
1. `src/web/templates/index_modern.html` - 현재 메인 페이지
2. `src/web/templates/history.html` - 히스토리 페이지
3. Tailwind CSS 문서

**테스트 방법**:
```bash
# 서버 실행 후 브라우저 테스트
http://127.0.0.1:8000/
http://127.0.0.1:8000/history
```

**Git 브랜치**:
```bash
git checkout -b feature/ui-improvements
```

---

## 👤 작업자 C: 리팩토링 및 최적화

### 📋 작업 개요
- **목표**: 코드 품질 향상 및 성능 최적화
- **예상 소요**: 6-8시간
- **우선순위**: 낮음
- **의존성**: 없음 (독립적)

### 🎯 세부 작업

#### C-1: zone_analyzer_2d.py 리팩토링 (4-5시간)
**파일**: `src/core/zone_analyzer_2d.py`

**대상 함수**:
1. `find_transition_ranges` (155 lines) → 3-4개 함수로 분할
2. `auto_define_zone_B` (147 lines) → 2-3개 함수로 분할
3. `compute_zone_results_2d` (145 lines) → 2-3개 함수로 분할

**리팩토링 원칙**:
- 각 함수는 50줄 이하
- 단일 책임 원칙 (Single Responsibility)
- 명확한 함수명

**예시** (find_transition_ranges):
```python
# Before: 155 lines
def find_transition_ranges(...):
    # 복잡한 로직 155줄

# After: 4개 함수로 분할
def find_transition_ranges(...):
    gradient = _compute_gradient(radial_profile)
    peaks = _detect_peaks(gradient, threshold)
    ranges = _convert_peaks_to_ranges(peaks)
    return _apply_fallback_if_needed(ranges, radial_profile)

def _compute_gradient(...):
    # 30줄

def _detect_peaks(...):
    # 40줄

def _convert_peaks_to_ranges(...):
    # 30줄

def _apply_fallback_if_needed(...):
    # 40줄
```

**검증**:
```bash
pytest tests/test_zone_analyzer_2d.py -v
# 40개 테스트 모두 통과해야 함
```

---

#### C-2: DB 쿼리 최적화 (1-2시간)
**파일**: `src/web/routers/inspection.py`

**작업 내용**:
1. N+1 쿼리 문제 확인
2. 필요 시 `joinedload()` 사용
3. 인덱스 추가 검토

**예시**:
```python
# Before
results = db.query(InspectionHistory).all()
for r in results:
    # 각 레코드마다 추가 쿼리 발생

# After
results = db.query(InspectionHistory).options(
    joinedload(InspectionHistory.related_model)
).all()
```

---

#### C-3: 캐싱 추가 (1-2시간)
**파일**: 새 파일 `src/utils/cache.py`

**작업 내용**:
1. 통계 API 결과 캐싱 (5분)
2. SKU 설정 캐싱 (10분)

**구현**:
```python
from functools import lru_cache
from datetime import datetime, timedelta

# 간단한 메모리 캐시
cache = {}

def get_cached_stats(days: int):
    cache_key = f"stats_{days}"
    if cache_key in cache:
        cached_at, data = cache[cache_key]
        if datetime.utcnow() - cached_at < timedelta(minutes=5):
            return data

    # 캐시 미스 - 새로 계산
    data = compute_stats(days)
    cache[cache_key] = (datetime.utcnow(), data)
    return data
```

---

#### C-4: 에러 핸들링 개선 (1시간)
**파일**: `src/web/app.py`, `src/web/routers/*.py`

**작업 내용**:
1. 모든 API 엔드포인트에 try-except 추가
2. 명확한 에러 메시지 반환
3. 로깅 추가

**예시**:
```python
@router.get("/history")
def list_history(...):
    try:
        results = db.query(...).all()
        return {"results": results}
    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve inspection history"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
```

---

### 📦 전달 자료

**컨텍스트 문서**:
1. `docs/planning/IMPROVEMENT_PLAN.md` - Task 3.4 리팩토링 계획
2. `tests/test_zone_analyzer_2d.py` - 테스트 케이스
3. `src/core/zone_analyzer_2d.py` - 대상 파일

**테스트 방법**:
```bash
# 리팩토링 후 전체 테스트 실행
pytest tests/ -v

# 특정 모듈 테스트
pytest tests/test_zone_analyzer_2d.py -v
```

**Git 브랜치**:
```bash
git checkout -b refactor/zone-analyzer-cleanup
```

---

## 🔄 작업 조율

### 코드 충돌 방지

**파일 분리**:
- 작업자 A: `src/web/templates/dashboard.html`, `src/web/routers/inspection.py` (새 엔드포인트만)
- 작업자 B: `src/web/templates/index*.html`, `src/web/templates/history.html`
- 작업자 C: `src/core/zone_analyzer_2d.py`, `src/utils/cache.py` (새 파일)

**중복 최소화**:
- 각자 독립적인 Git 브랜치 사용
- 완료 후 순차적으로 merge

---

## 📅 타임라인

### Week 1 (현재)
- **Day 1-2**: 작업자 A (대시보드 핵심 기능)
- **Day 1-2**: 작업자 B (UI 개선)
- **Day 1-2**: 작업자 C (리팩토링 계획 및 시작)

### Week 2
- **Day 3-4**: 작업자 A (고급 차트 및 필터)
- **Day 3-4**: 작업자 B (반응형 및 UX 개선)
- **Day 3-4**: 작업자 C (리팩토링 완료 및 테스트)

### Week 3
- **통합 테스트**
- **버그 수정**
- **문서 업데이트**

---

## ✅ 완료 기준

### 작업자 A
- [ ] Dashboard 페이지 접속 가능
- [ ] 3개 이상의 차트 정상 동작
- [ ] SKU별 통계 테이블 표시
- [ ] 날짜 필터 동작

### 작업자 B
- [ ] 메인 페이지에서 History 이동 가능
- [ ] 검사 후 History에서 결과 확인 가능
- [ ] 모바일 화면에서 정상 표시
- [ ] 로딩 인디케이터 동작

### 작업자 C
- [ ] zone_analyzer_2d.py 리팩토링 완료
- [ ] 40개 테스트 모두 통과
- [ ] 캐싱 구현 및 테스트
- [ ] 에러 핸들링 개선

---

## 📞 커뮤니케이션

### 일일 체크인
- 매일 오전: 진행 상황 공유
- 매일 오후: 블로커 확인

### 블로커 해결
- 즉시 Slack/이메일로 공유
- 필요 시 화상 회의

---

**작성자**: 프로젝트 매니저
**최종 업데이트**: 2025-12-19
