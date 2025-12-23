# 작업 할당서 B: Web UI 개선

**담당자**: 작업자 B
**예상 소요**: 4-6시간
**우선순위**: 중간
**시작일**: 2025-12-19

---

## 🎯 목표

사용자 경험 개선 및 페이지 간 연결성 강화

---

## 📋 할 일 체크리스트

### 1. 환경 설정 (15분)
- [ ] 프로젝트 클론 및 서버 실행
  ```bash
  cd C:/X/Color_total/Color_meter
  python -m uvicorn src.web.app:app --host 0.0.0.0 --port 8000 --reload
  ```
- [ ] 주요 페이지 확인
  - http://127.0.0.1:8000/ (메인)
  - http://127.0.0.1:8000/history (히스토리)
  - http://127.0.0.1:8000/stats (통계)

### 2. 메인 페이지 네비게이션 추가 (1시간)
**파일**: `src/web/templates/index.html`

- [ ] 상단에 네비게이션 바 추가
  ```html
  <nav class="bg-white shadow-md mb-6">
    <div class="max-w-7xl mx-auto px-4">
      <div class="flex items-center justify-between h-16">
        <div class="flex space-x-4">
          <a href="/" class="text-purple-600 font-semibold">
            <i class="fas fa-home mr-2"></i>Inspection
          </a>
          <a href="/history" class="text-gray-600 hover:text-purple-600">
            <i class="fas fa-history mr-2"></i>History
          </a>
          <a href="/stats" class="text-gray-600 hover:text-purple-600">
            <i class="fas fa-chart-bar mr-2"></i>Statistics
          </a>
        </div>
      </div>
    </div>
  </nav>
  ```

### 3. History 페이지 네비게이션 (30분)
**파일**: `src/web/templates/history.html`

- [ ] "Back to Home" 버튼 → 네비게이션 바로 교체
- [ ] 동일한 네비게이션 구조 적용

### 4. Stats 페이지 네비게이션 (30분)
**파일**: `src/web/templates/stats.html`

- [ ] 동일한 네비게이션 바 추가

### 5. 검사 완료 후 History 링크 추가 (1.5시간)
**파일**: `src/web/templates/index.html` (결과 표시 섹션)

- [ ] 검사 완료 시 "View in History" 버튼 추가
  ```javascript
  // 검사 완료 후
  function showResultActions(sessionId) {
    const actionsHtml = `
      <div class="mt-4 flex gap-2">
        <a href="/history?session=${sessionId}"
           class="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700">
          <i class="fas fa-history mr-2"></i>View in History
        </a>
        <button onclick="runAnotherInspection()"
                class="bg-white text-purple-600 border border-purple-600 px-4 py-2 rounded-lg hover:bg-purple-50">
          <i class="fas fa-redo mr-2"></i>New Inspection
        </button>
      </div>
    `;
    document.getElementById('result-actions').innerHTML = actionsHtml;
  }
  ```

### 6. History 페이지 세션 하이라이트 (2시간)
**파일**: `src/web/templates/history.html`

- [ ] URL 파라미터에서 `session` 읽기
  ```javascript
  const urlParams = new URLSearchParams(window.location.search);
  const highlightSession = urlParams.get('session');

  if (highlightSession) {
    // 자동으로 상세 모달 열기
    viewDetail(highlightSession);

    // 해당 행 하이라이트
    const row = document.querySelector(`tr[data-session="${highlightSession}"]`);
    if (row) {
      row.classList.add('bg-yellow-100');
      row.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }
  ```

- [ ] 테이블 행에 `data-session` 속성 추가

### 7. 반응형 디자인 개선 (1.5시간)
**파일**: 모든 템플릿 파일

- [ ] 테이블 가로 스크롤
  ```html
  <div class="overflow-x-auto">
    <table class="min-w-full">
      <!-- 테이블 내용 -->
    </table>
  </div>
  ```

- [ ] 모바일 화면 대응 (Tailwind breakpoints)
  ```html
  <!-- 데스크톱: 4열, 태블릿: 2열, 모바일: 1열 -->
  <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
  ```

### 8. 로딩 인디케이터 추가 (1시간)
**파일**: 모든 템플릿 파일

- [ ] 공통 로딩 스피너 추가
  ```html
  <div id="loading-overlay" class="hidden fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
    <div class="bg-white p-6 rounded-lg">
      <i class="fas fa-spinner fa-spin text-4xl text-purple-600"></i>
      <p class="mt-4 text-gray-700">Loading...</p>
    </div>
  </div>
  ```

- [ ] API 호출 시 표시/숨김
  ```javascript
  function showLoading() {
    document.getElementById('loading-overlay').classList.remove('hidden');
  }

  function hideLoading() {
    document.getElementById('loading-overlay').classList.add('hidden');
  }
  ```

---

## 📦 수정할 파일

1. `src/web/templates/index.html` - 메인 페이지
2. `src/web/templates/history.html` - 히스토리 페이지
3. `src/web/templates/stats.html` - 통계 페이지

---

## 🧪 테스트 방법

### 시나리오 1: 검사 → History 이동
1. 메인 페이지에서 이미지 검사 실행
2. "View in History" 버튼 클릭
3. History 페이지에서 해당 결과가 하이라이트되는지 확인

### 시나리오 2: 네비게이션
1. 메인 페이지에서 History 링크 클릭
2. History에서 Stats 링크 클릭
3. Stats에서 Home 링크 클릭
4. 모든 페이지 이동이 원활한지 확인

### 시나리오 3: 모바일 화면
1. 브라우저를 모바일 크기로 조정 (F12 → Device Toolbar)
2. 모든 페이지가 깨지지 않는지 확인
3. 테이블 가로 스크롤 확인

---

## 📝 완료 기준

- [ ] 모든 페이지에 통일된 네비게이션
- [ ] 검사 후 History로 이동 가능
- [ ] History에서 세션 하이라이트 동작
- [ ] 모바일 화면에서 정상 표시
- [ ] 로딩 인디케이터 동작

---

## 🎨 디자인 가이드

**색상**:
- Primary: `#667eea` (보라색)
- Success: `#10b981` (초록)
- Warning: `#f59e0b` (노랑)
- Danger: `#ef4444` (빨강)

**폰트**:
- Font Family: 'Inter', sans-serif
- 헤더: 600-700 weight
- 본문: 400-500 weight

**간격**:
- 섹션 간: `mb-6` (1.5rem)
- 카드 padding: `p-6` (1.5rem)
- 버튼 gap: `gap-4` (1rem)

---

## 🚫 주의사항

1. **Tailwind CSS 사용**: 별도 CSS 파일 작성 금지
2. **Git 브랜치**: `feature/ui-improvements`
3. **파일 충돌 방지**: 템플릿 파일만 수정

---

## 💬 질문/도움

- UI/UX 디자인 조언 필요
- Tailwind CSS 문법 질문
- 반응형 레이아웃 구현 도움

---

**시작 시간**: ___________
**예상 완료**: ___________
**실제 완료**: ___________
