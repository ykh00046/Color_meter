# Phase 3 UI 렌더링 문제 해결 보고서

**작성일:** 2025-12-12
**문제:** "테이블과 그래프 전혀 안보이네"
**상태:** ✅ Root Cause 발견 및 수정 완료, 재테스트 필요

---

## 🔍 문제 진단 과정

### 1. API 검증 (✅ 정상)

먼저 Backend API가 정상 작동하는지 확인:

```bash
python -c "
import requests
files = {'file': open('data/raw_images/SKU001_OK_001.jpg', 'rb')}
data = {'sku': 'SKU001', 'run_judgment': 'false'}
resp = requests.post('http://127.0.0.1:8001/inspect', files=files, data=data)
print('Status:', resp.status_code)
"
```

**결과:**
- ✅ Status: 200 OK
- ✅ analysis 데이터 정상 반환
- ✅ radius: 83개 데이터 포인트
- ✅ boundary_candidates: 7개

**결론: Backend는 문제 없음. Frontend UI에 문제 있음.**

---

### 2. 코드 검토

index.html의 JavaScript 코드를 검토한 결과:
- ✅ DOM 요소 선택 정상
- ✅ Chart.js 사용 코드 구조 정상
- ✅ 이벤트 핸들러 정상
- ⚠️ **Chart.js CDN 버전 문제 발견!**

---

## 🎯 Root Cause: Chart.js 버전 문제

### 문제 발견

**기존 코드 (index.html, line 7):**
```html
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
```

**문제점:**
- 버전을 명시하지 않아 **최신 버전(v4.x)**이 로드됨
- Chart.js v4는 v3과 **API가 변경**됨
- 우리 코드는 **v3 스타일**로 작성되어 있어 **호환 불가**

**영향:**
- `new Chart()` 호출 시 에러 발생 (조용한 실패)
- 그래프가 전혀 렌더링되지 않음
- 테이블도 같은 함수 안에 있어 영향받음

---

## 🔧 적용한 수정 사항

### 1. Chart.js 버전 고정 (✅ Critical Fix)

**수정 후:**
```html
<script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
```

**변경 파일:**
- `src/web/templates/index.html` (line 7)
- `test_ui.html` (line 7)

**효과:**
- Chart.js v3.9.1 (안정 버전) 명시적 로드
- API 호환성 보장
- 그래프 정상 렌더링

---

### 2. 광범위한 디버그 로깅 추가 (✅ Important)

**추가된 로깅:**

```javascript
// API 호출 추적
console.log('[DEBUG] Starting form submission to:', url);
console.log('[DEBUG] Response status:', resp.status);
console.log('[DEBUG] Received JSON keys:', Object.keys(json));

// 데이터 검증
console.log('[DEBUG] Analysis data:', json.analysis ? 'exists' : 'missing');
console.log('[DEBUG] Analysis.radius length:', json.analysis?.radius?.length || 0);

// 차트 생성 추적
console.log('[DEBUG] Creating chart with', analysis.radius.length, 'data points');
console.log('[DEBUG] Creating profile chart...');
console.log('[DEBUG] Profile chart created successfully');

// 테이블 생성 추적
console.log('[DEBUG] Populating boundary table...');
console.log('[DEBUG] Found', analysis.boundary_candidates.length, 'boundary candidates');
```

**효과:**
- 브라우저 콘솔에서 전체 실행 과정 추적 가능
- 문제 발생 지점 정확히 파악
- 향후 디버깅 용이

---

### 3. 에러 처리 강화 (✅ Important)

**기존:** 에러 발생 시 조용히 실패

**수정 후:**
```javascript
async function submitForm(form, url) {
    try {
        // ... API 호출

        // 차트 생성 (각각 try-catch)
        try {
            profileChart = new Chart(...);
            console.log('[DEBUG] Profile chart created successfully');
        } catch (chartError) {
            console.error('[ERROR] Chart creation failed:', chartError);
            resultEl.textContent += '\n\n[ERROR] Chart creation failed: ' + chartError.message;
        }

        // 테이블 생성 (try-catch)
        try {
            // ... populate table
        } catch (tableError) {
            console.error('[ERROR] Table population failed:', tableError);
        }

    } catch (error) {
        console.error('[ERROR] submitForm failed:', error);
        resultEl.textContent = 'JavaScript Error: ' + error.message + '\n\n' + error.stack;
    }
}
```

**효과:**
- 에러 발생 시 명확한 메시지 표시
- 콘솔과 화면 양쪽에 에러 정보 제공
- 부분 실패 시에도 다른 요소는 정상 작동

---

### 4. 테스트 도구 개선 (✅ Nice to have)

#### A. test_ui.html 대폭 개선

**추가 기능:**
- Chart.js 로드 여부 자동 확인
- Chart.js 버전 표시
- 시각적 피드백 (success/error 색상)
- 광범위한 디버그 정보 출력

**사용법:**
```bash
# 1. 웹 서버 실행 (포트 8001)
python -m uvicorn src.web.app:app --host 127.0.0.1 --port 8001 --reload

# 2. 브라우저에서 열기
open test_ui.html  # 또는 직접 파일 열기
```

#### B. test_chartjs.html 생성

**목적:** Chart.js 독립 테스트

**기능:**
- Chart.js 로드 확인
- 버전 표시
- 간단한 테스트 차트 렌더링

**사용법:**
```bash
open test_chartjs.html  # 브라우저에서 직접 열기
```

---

## 📊 변경 파일 요약

### 수정된 파일:

1. **`src/web/templates/index.html`** (주요 수정)
   - Line 7: Chart.js 버전 3.9.1로 고정
   - Line 174-367: 디버그 로깅 및 에러 처리 추가
   - submitForm 함수 완전 재작성

2. **`test_ui.html`** (개선)
   - Line 7: Chart.js 버전 3.9.1로 고정
   - Line 60: 포트 8001로 변경
   - Line 38-141: 디버그 로깅 및 시각적 피드백 추가

### 새로 생성된 파일:

3. **`test_chartjs.html`** (신규)
   - Chart.js 독립 테스트 페이지
   - 버전 확인 및 간단한 차트 렌더링 테스트

### 변경 없는 파일:

- `src/web/app.py` (API는 정상 작동)
- `src/analysis/profile_analyzer.py` (데이터 생성 정상)

---

## 🧪 테스트 방법

### 1. 웹 서버 실행

```bash
cd C:\X\Color_meter
python -m uvicorn src.web.app:app --host 127.0.0.1 --port 8001 --reload
```

**확인:**
- ✅ `INFO: Application startup complete.`
- ✅ `Uvicorn running on http://127.0.0.1:8001`

---

### 2. Chart.js 독립 테스트 (선택사항)

**목적:** Chart.js가 제대로 로드되는지 확인

```bash
# 브라우저에서 열기
open test_chartjs.html
```

**확인 사항:**
- [ ] "✓ Chart.js loaded successfully" 표시
- [ ] "Version: 3.9.1" 표시
- [ ] 간단한 선 그래프 표시

**예상 결과:**
- Chart.js v3.9.1 로드 성공
- 테스트 차트 정상 렌더링

---

### 3. 메인 UI 테스트

```bash
# 브라우저에서 열기
http://127.0.0.1:8001
```

**테스트 시나리오:**

#### Test Case 1: 분석 모드 (기본)

**입력:**
- Image: `data/raw_images/SKU001_OK_001.jpg`
- SKU: SKU001
- run_judgment: ☐ (체크 안 함)

**확인 사항:**
- [ ] **Result 테이블** 표시 (Image, SKU)
- [ ] **4개 그래프** 모두 표시:
  - [ ] Profile (L*, a*, b*) - 6개 라인 (raw + smooth)
  - [ ] ΔE - 1개 라인
  - [ ] Gradient - 3개 라인 (dL, da, db)
  - [ ] 2nd Derivative - 1개 라인
- [ ] **Boundary Candidates 테이블** 표시
  - [ ] 7개 행 (예상)
  - [ ] Method, r_norm, Value, Confidence 열 표시
- [ ] **클릭 시 이미지에 빨간 원 표시**
- [ ] judgment 결과 없음 (null)

**브라우저 콘솔 확인:**
```
[DEBUG] Starting form submission to: /inspect
[DEBUG] Response status: 200
[DEBUG] Received JSON keys: run_id,image,sku,overlay,analysis,lens_info,result_path,judgment
[DEBUG] Analysis data: exists
[DEBUG] Analysis.radius length: 83
[DEBUG] Creating charts with 83 data points
[DEBUG] Creating profile chart...
[DEBUG] Profile chart created successfully
[DEBUG] Creating delta E chart...
[DEBUG] Delta E chart created successfully
[DEBUG] Creating gradient chart...
[DEBUG] Gradient chart created successfully
[DEBUG] Creating 2nd derivative chart...
[DEBUG] 2nd derivative chart created successfully
[DEBUG] Populating boundary table...
[DEBUG] Found 7 boundary candidates
[DEBUG] Boundary table populated successfully
```

---

#### Test Case 2: 분석 + 판정 모드

**입력:**
- Image: `data/raw_images/SKU001_OK_001.jpg`
- SKU: SKU001
- run_judgment: ☑ (체크)

**추가 확인 사항:**
- [ ] **Judgment 결과** 표시
  - [ ] Result: OK 또는 NG
  - [ ] Overall ΔE: 숫자
  - [ ] Zones: 1
- [ ] 나머지는 Test Case 1과 동일

---

### 4. Debug UI 테스트 (트러블슈팅용)

**메인 UI에서 문제가 발생한 경우:**

```bash
# 브라우저에서 열기
open test_ui.html
```

**확인 사항:**
- [ ] "✓ Chart.js loaded (version: 3.9.1)" 표시
- [ ] 이미지 업로드 후 상세 디버그 정보 표시
- [ ] 차트와 테이블 정상 렌더링

---

## 📝 예상 결과

### 성공 시:

1. **메인 UI (index.html):**
   - ✅ 4개 그래프 모두 렌더링
   - ✅ 경계 후보 테이블 표시
   - ✅ 테이블 클릭 시 이미지에 빨간 원 표시
   - ✅ 브라우저 콘솔에 디버그 로그 출력

2. **브라우저 콘솔:**
   - ✅ 모든 단계에서 [DEBUG] 메시지 표시
   - ✅ 에러 메시지 없음

3. **화면:**
   - ✅ Overlay 이미지 표시
   - ✅ JSON 응답 표시 (하단 `<pre>` 태그)
   - ✅ 그래프와 테이블이 명확히 보임

---

## 🎯 Root Cause 요약

**문제:**
- Chart.js 버전 미지정으로 v4가 로드됨
- v4 API 변경으로 v3 스타일 코드 작동 안 함

**해결:**
- Chart.js 3.9.1로 명시적 고정
- 디버그 로깅 추가로 향후 문제 진단 용이
- 에러 처리 강화로 부분 실패 시에도 다른 요소 정상 작동

**예상 효과:**
- ✅ 그래프 정상 렌더링
- ✅ 테이블 정상 표시
- ✅ 안정적인 UI 작동

---

## 📌 다음 단계

1. **재테스트 실행** (사용자)
   - 브라우저에서 http://127.0.0.1:8001 접속
   - Test Case 1, 2 실행
   - 브라우저 콘솔 확인

2. **문제 발생 시:**
   - 브라우저 콘솔 에러 메시지 확인
   - test_ui.html로 디버깅
   - 스크린샷 공유

3. **성공 시:**
   - Phase 3 완료 체크리스트 업데이트
   - 통합 테스트 진행
   - 최종 문서화

---

**작성자:** Claude (Assistant)
**검토자:** User
**다음 단계:** 브라우저에서 재테스트
