# 웹 UI 현대화 및 통합 계획 (Web UI Modernization Plan)

> **문서 버전**: 1.0
> **작성일**: 2026-01-17
> **상태**: ✅ 승인됨 (Approved)
> **소유자**: 개발팀

## 문서 목적

Color Meter 프로젝트의 웹 UI를 통합하고 현대화하여 사용성, 유지보수성, 확장성을 향상시키기 위한 종합 실행 계획을 제시합니다.

---

## 목차

1. [현황 분석](#1-현황-분석)
2. [문제점 식별](#2-문제점-식별)
3. [개선 전략](#3-개선-전략)
4. [실행 계획](#4-실행-계획)
5. [검증 계획](#5-검증-계획)
6. [리스크 관리](#6-리스크-관리)
7. [타임라인](#7-타임라인)

---

## 1. 현황 분석

### 1.1 웹 UI 구조

#### HTML 템플릿 (6개)

```
src/web/templates/
├── index.html              # 메인 랜딩 페이지
├── v7_mvp.html            # v7 엔진 통합 콘솔 (4개 탭)
├── single_analysis.html    # 단독 샘플 분석 (935줄)
├── calibration.html       # 색상 정확도 교정
├── history.html           # 검사 이력 조회
└── stats.html             # 통계 대시보드
```

#### JavaScript 모듈 (17개)

```
src/web/static/js/v7/
├── api.js                 # API 호출 래퍼
├── state.js               # 전역 상태 관리
├── inspection.js          # 검사 로직
├── registration.js        # STD 등록
├── single_analysis.js     # 단독 분석
├── visuals.js             # 시각화 공통
├── ink_visuals.js         # 잉크 시각화
├── diagnostics_visuals.js # 진단 시각화
└── ... (8개 추가 모듈)
```

#### API 라우터

- `src/web/routers/v7.py` (85KB) - v7 엔진 API
- `src/web/routers/inspection.py` (25KB) - 검사 이력 API

### 1.2 기술 스택

**현재 사용 중**:

- **Frontend**: HTML, Vanilla JavaScript, CSS, Tailwind CSS (CDN)
- **Backend**: FastAPI, Jinja2 Templates
- **Visualization**: Chart.js (CDN)
- **Fonts**: Inter, JetBrains Mono (Google Fonts)

---

## 2. 문제점 식별

### 2.1 디자인 시스템 부재

**증상**:

- 각 페이지가 독립적인 스타일 정의 (중복 코드)
- 일관성 없는 컬러 팔레트 및 타이포그래피
- 반응형 브레이크포인트 불일치

**예시**:

```html
<!-- v7_mvp.html -->
<style>
  :root {
    --base-black: #0f172a;
    --amber-primary: #3b82f6;
  }
</style>

<!-- index.html -->
<style>
  :root {
    --bg-dark: #0a0f1e;
    --primary: #f59e0b; /* 다른 색상! */
  }
</style>
```

### 2.2 컴포넌트 재사용 불가

**증상**:

- 동일한 UI 패턴이 템플릿마다 재정의됨
- 버튼, 카드, 입력 필드 등의 스타일이 파일마다 상이

**중복 예시**:

- `.btn-primary` 정의가 3곳에 존재 (v7_mvp.html, index.html, single_analysis.html)
- `.metric-card` 구조가 각 페이지마다 다름

### 2.3 네비게이션 파편화

**증상**:

- 페이지 간 일관된 네비게이션 바 없음
- `index.html`에서만 전체 페이지 링크 제공
- `v7_mvp.html`은 독립적인 사이드바 사용

**사용자 경험 문제**:

- Single Analysis → History로 직접 이동 불가
- 매번 메인 페이지로 돌아가야 함

### 2.4 JavaScript 아키텍처 문제

**증상**:

1. **모듈 의존성 불명확**: `window.v7 = {}` 전역 객체 패턴
2. **로딩 순서 의존성**: 스크립트 로드 순서 오류 시 런타임 에러
3. **상태 관리 분산**: 각 페이지별 독립 상태, 동기화 불가

**예시**:

```javascript
// state.js
window.v7 = window.v7 || {};
window.v7.state = { currentSku: null };

// inspection.js (의존성 불명확)
function runInspection() {
  const sku = window.v7.state.currentSku; // 에러 가능성
}
```

### 2.5 접근성 (a11y) 미흡

**문제점**:

- ARIA 레이블 누락 (스크린 리더 지원 부족)
- 키보드 네비게이션 불완전
- 색상 대비 비율 미달 영역 존재

---

## 3. 개선 전략

### 3.1 설계 원칙

1. **점진적 개선 (Progressive Enhancement)**
   - 기존 기능 보존
   - 단계적 마이그레이션
   - 롤백 가능한 구조

2. **컴포넌트 기반 설계**
   - 재사용 가능한 UI 블록
   - 단일 책임 원칙 (SRP)

3. **성능 우선**
   - 코드 스플리팅
   - 레이지 로딩
   - 캐싱 전략

4. **접근성 보장**
   - WCAG 2.1 AA 준수
   - 시맨틱 HTML
   - 키보드 네비게이션

### 3.2 기술 선택

**선택한 접근법**: 하이브리드 모던 스택

| 항목       | 기술                     | 이유               |
| ---------- | ------------------------ | ------------------ |
| 템플릿     | Jinja2 (유지)            | 백엔드 통합 용이   |
| CSS        | Design System + Tailwind | 일관성 + 개발 속도 |
| JavaScript | ES6 Modules              | 의존성 명시화      |
| 빌드       | Vite (선택)              | 빠른 HMR, 번들링   |
| 테스트     | Playwright + Vitest      | E2E + 단위 테스트  |

**거부한 옵션**:

- ❌ React/Vue 전환: 과도한 개발 시간, 높은 리스크
- ❌ 완전 SPA: 백엔드 렌더링 장점 상실

---

## 4. 실행 계획

### Phase 1: 기초 인프라 (Week 1-2)

#### 1.1 디자인 시스템 구축

**파일**: `src/web/static/css/design_system.css`

**내용**:

```css
/* CSS 변수 정의 */
:root {
  /* Color Palette */
  --color-primary: #3b82f6;
  --color-secondary: #2563eb;
  --color-success: #10b981;
  --color-warning: #f59e0b;
  --color-error: #ef4444;

  /* Backgrounds */
  --bg-base: #0f172a;
  --bg-surface: #151e32;
  --bg-elevated: #1e293b;

  /* Typography */
  --font-sans: "Inter", -apple-system, sans-serif;
  --font-mono: "JetBrains Mono", monospace;
  --font-size-xs: 0.75rem;
  --font-size-sm: 0.875rem;
  --font-size-base: 1rem;
  --font-size-lg: 1.125rem;
  --font-size-xl: 1.25rem;

  /* Spacing Scale */
  --spacing-xs: 4px;
  --spacing-sm: 8px;
  --spacing-md: 16px;
  --spacing-lg: 24px;
  --spacing-xl: 32px;

  /* Border Radius */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
  --radius-xl: 16px;
}

/* Component Classes */
.btn {
  padding: var(--spacing-sm) var(--spacing-md);
  border-radius: var(--radius-md);
  font-weight: 600;
  transition: all 0.2s;
  cursor: pointer;
}

.btn-primary {
  background: var(--color-primary);
  color: white;
}

.btn-primary:hover {
  background: var(--color-secondary);
  transform: translateY(-1px);
}

.card {
  background: var(--bg-surface);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: var(--radius-lg);
  padding: var(--spacing-lg);
}

.metric-card {
  background: var(--bg-elevated);
  padding: var(--spacing-md);
  border-radius: var(--radius-md);
}

/* ... 더 많은 컴포넌트 */
```

#### 1.2 공통 레이아웃 템플릿

**파일**: `src/web/templates/_base_layout.html`

**내용**:

```html
<!DOCTYPE html>
<html lang="ko">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{% block title %}Color Meter{% endblock %}</title>

    <!-- Design System -->
    <link rel="stylesheet" href="/static/css/design_system.css" />

    <!-- Tailwind (CDN) -->
    <script src="/static/vendor/tailwind.min.js"></script>

    <!-- Fonts -->
    <link rel="stylesheet" href="https://rsms.me/inter/inter.css" />

    {% block extra_head %}{% endblock %}
  </head>
  <body class="bg-base text-primary">
    <!-- 통합 네비게이션 -->
    <nav class="main-nav">
      <div
        class="container mx-auto px-6 py-4 flex items-center justify-between"
      >
        <a href="/" class="nav-brand flex items-center gap-3">
          <div
            class="w-10 h-10 rounded-lg bg-gradient-to-br from-primary to-secondary flex items-center justify-center text-white font-bold"
          >
            CM
          </div>
          <span class="font-bold text-lg">Color Meter</span>
        </a>

        <div class="nav-menu flex gap-2">
          <a href="/" class="nav-link">Home</a>
          <a href="/v7" class="nav-link">Inspection</a>
          <a href="/single_analysis" class="nav-link">Single Analysis</a>
          <a href="/calibration" class="nav-link">Calibration</a>
          <a href="/history" class="nav-link">History</a>
          <a href="/stats" class="nav-link">Stats</a>
        </div>
      </div>
    </nav>

    <!-- 메인 컨텐츠 -->
    <main class="main-content">{% block content %}{% endblock %}</main>

    <!-- 공통 스크립트 -->
    <script type="module" src="/static/js/core/state.js"></script>
    <script type="module" src="/static/js/core/api.js"></script>
    {% block extra_scripts %}{% endblock %}
  </body>
</html>
```

#### 1.3 컴포넌트 라이브러리

**파일**: `src/web/static/js/components/base.js`

**내용**:

```javascript
/**
 * 재사용 가능한 UI 컴포넌트 라이브러리
 */

export const Components = {
  /**
   * 버튼 생성
   * @param {string} text - 버튼 텍스트
   * @param {string} variant - 'primary' | 'secondary' | 'success' | 'warning' | 'error'
   * @param {Function} onClick - 클릭 핸들러
   * @returns {HTMLButtonElement}
   */
  createButton(text, variant = "primary", onClick = null) {
    const btn = document.createElement("button");
    btn.className = `btn btn-${variant}`;
    btn.textContent = text;
    if (onClick) btn.addEventListener("click", onClick);
    return btn;
  },

  /**
   * 카드 생성
   * @param {string} title - 카드 제목
   * @param {HTMLElement|string} content - 카드 내용
   * @returns {HTMLDivElement}
   */
  createCard(title, content) {
    const card = document.createElement("div");
    card.className = "card";

    if (title) {
      const header = document.createElement("h3");
      header.className = "text-lg font-bold mb-4";
      header.textContent = title;
      card.appendChild(header);
    }

    if (typeof content === "string") {
      card.innerHTML += content;
    } else {
      card.appendChild(content);
    }

    return card;
  },

  /**
   * 메트릭 카드 생성
   * @param {string} label - 메트릭 레이블
   * @param {string|number} value - 메트릭 값
   * @param {string} unit - 단위
   * @param {string} status - 'success' | 'warning' | 'error'
   * @returns {HTMLDivElement}
   */
  createMetricCard(label, value, unit = "", status = null) {
    const card = document.createElement("div");
    card.className = "metric-card";

    const labelEl = document.createElement("div");
    labelEl.className = "text-xs text-dim uppercase mb-2";
    labelEl.textContent = label;

    const valueEl = document.createElement("div");
    valueEl.className = "text-2xl font-mono font-bold";
    if (status) valueEl.classList.add(`text-${status}`);
    valueEl.innerHTML = `${value}${unit ? `<span class="text-sm ml-1">${unit}</span>` : ""}`;

    card.appendChild(labelEl);
    card.appendChild(valueEl);

    return card;
  },

  /**
   * 탭 시스템 생성
   * @param {Array<{id: string, label: string, content: HTMLElement}>} tabs
   * @param {Function} onTabChange - 탭 변경 콜백
   * @returns {HTMLDivElement}
   */
  createTabs(tabs, onTabChange = null) {
    const container = document.createElement("div");
    container.className = "tabs-container";

    // 탭 헤더
    const tabHeaders = document.createElement("div");
    tabHeaders.className = "flex gap-2 border-b border-white/10 mb-6";

    // 탭 컨텐츠 컨테이너
    const tabContents = document.createElement("div");
    tabContents.className = "tab-contents";

    tabs.forEach((tab, index) => {
      // 헤더 버튼
      const btn = document.createElement("button");
      btn.className = `tab-btn ${index === 0 ? "active" : ""}`;
      btn.textContent = tab.label;
      btn.dataset.tabId = tab.id;

      btn.addEventListener("click", () => {
        // 모든 탭 비활성화
        tabHeaders
          .querySelectorAll(".tab-btn")
          .forEach((b) => b.classList.remove("active"));
        tabContents
          .querySelectorAll(".tab-content")
          .forEach((c) => c.classList.add("hidden"));

        // 현재 탭 활성화
        btn.classList.add("active");
        document
          .getElementById(`tab-content-${tab.id}`)
          .classList.remove("hidden");

        if (onTabChange) onTabChange(tab.id);
      });

      tabHeaders.appendChild(btn);

      // 컨텐츠
      const content = document.createElement("div");
      content.id = `tab-content-${tab.id}`;
      content.className = `tab-content ${index !== 0 ? "hidden" : ""}`;
      content.appendChild(tab.content);

      tabContents.appendChild(content);
    });

    container.appendChild(tabHeaders);
    container.appendChild(tabContents);

    return container;
  },
};
```

---

### Phase 2: 템플릿 통합 (Week 3-4)

#### 2.1 index.html 리팩토링

**변경 사항**:

```diff
- <!DOCTYPE html>
- <html lang="ko">
- <head>
-   <meta charset="UTF-8">
-   ...독립 헤더...
- </head>
+ {% extends "_base_layout.html" %}
+
+ {% block title %}Color Meter | 렌즈 품질 분석{% endblock %}
+
+ {% block content %}
```

#### 2.2 v7_mvp.html 리팩토링

**결정**: 사이드바 → 서브 네비게이션으로 변경

**Before**:

```html
<aside class="w-16 md:w-64 sidebar">
  <button>Inspection</button>
  <button>Registration</button>
  ...
</aside>
```

**After**:

```html
{% extends "_base_layout.html" %} {% block content %}
<div class="container mx-auto px-6 py-8">
  <!-- 서브 네비게이션 -->
  <div class="sub-nav mb-8">
    <button class="sub-nav-btn active" data-view="inspection">
      Inspection
    </button>
    <button class="sub-nav-btn" data-view="registration">Registration</button>
    <button class="sub-nav-btn" data-view="history">History</button>
    <button class="sub-nav-btn" data-view="test">Test Lab</button>
  </div>

  <!-- 컨텐츠 -->
  <div id="view-inspection" class="view active">...</div>
  <div id="view-registration" class="view hidden">...</div>
  ...
</div>
{% endblock %}
```

---

### Phase 3: JavaScript 리팩토링 (Week 5-6)

#### 3.1 모듈 구조 재설계

**새로운 구조**:

```
src/web/static/js/
├── core/
│   ├── api.js           # API 호출 레이어
│   ├── state.js         # 중앙 상태 관리
│   ├── router.js        # 클라이언트 라우팅
│   └── eventBus.js      # 이벤트 시스템
├── components/
│   ├── base.js          # 기본 컴포넌트
│   ├── charts.js        # Chart.js 래퍼
│   └── dataGrid.js      # 데이터 그리드
├── features/
│   ├── inspection/
│   │   ├── inspection.js
│   │   └── visuals.js
│   ├── registration/
│   │   └── registration.js
│   ├── analysis/
│   │   ├── single.js
│   │   └── ink.js
│   └── calibration/
│       └── calibration.js
└── utils/
    ├── formatters.js
    └── validators.js
```

#### 3.2 ES6 모듈 전환

**Before**:

```javascript
// window.v7 전역 패턴
window.v7 = window.v7 || {};
window.v7.api = {
  /* ... */
};
```

**After**:

```javascript
// ES6 import/export
// core/api.js
export class ApiClient {
  async post(endpoint, data) {
    const response = await fetch(`/api${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return response.json();
  }
}

export const apiClient = new ApiClient();

// features/inspection/inspection.js
import { apiClient } from "../../core/api.js";

export async function runInspection(data) {
  return await apiClient.post("/v7/inspect", data);
}
```

#### 3.3 상태 관리 통합

**파일**: `src/web/static/js/core/state.js`

```javascript
/**
 * 중앙화된 애플리케이션 상태 관리
 */
class AppState {
  constructor() {
    this.state = {
      user: null,
      currentSku: null,
      currentInk: null,
      inspection: {
        selectedProduct: null,
        lastResult: null,
        isProcessing: false,
      },
      analysis: {
        uploadedFile: null,
        result: null,
      },
    };

    this.listeners = new Map();
  }

  /**
   * 상태 구독
   */
  subscribe(path, callback) {
    if (!this.listeners.has(path)) {
      this.listeners.set(path, []);
    }
    this.listeners.get(path).push(callback);

    // 언구독 함수 반환
    return () => {
      const callbacks = this.listeners.get(path);
      const index = callbacks.indexOf(callback);
      if (index > -1) callbacks.splice(index, 1);
    };
  }

  /**
   * 상태 업데이트
   */
  setState(path, value) {
    const keys = path.split(".");
    let current = this.state;

    for (let i = 0; i < keys.length - 1; i++) {
      current = current[keys[i]];
    }

    current[keys[keys.length - 1]] = value;

    // 리스너 알림
    if (this.listeners.has(path)) {
      this.listeners.get(path).forEach((cb) => cb(value));
    }
  }

  /**
   * 상태 조회
   */
  getState(path) {
    const keys = path.split(".");
    let current = this.state;

    for (const key of keys) {
      current = current[key];
    }

    return current;
  }
}

export const appState = new AppState();
```

---

### Phase 4: 기능 개선 (Week 7-8)

#### 4.1 Inspection 페이지 개선

**개선 항목**:

1. 드래그 앤 드롭 파일 업로드
2. 실시간 분석 진행률 표시
3. 결과 차트 인터랙션 개선
4. 반응형 레이아웃

#### 4.2 Single Analysis 페이지 개선

**개선 항목**:

1. 935줄 HTML → 컴포넌트 분할
2. 탭 네비게이션 UX 개선
3. 접근성 강화 (ARIA)

#### 4.3 History & Stats 페이지 개선

**개선 항목**:

1. 데이터 테이블 정렬/필터링
2. 날짜 범위 선택기
3. Excel 내보내기

---

## 5. 검증 계획

### 5.1 수동 테스트

**테스트 매트릭스**:

| 브라우저 | 버전   | Desktop | Mobile |
| -------- | ------ | ------- | ------ |
| Chrome   | Latest | ✅      | ✅     |
| Firefox  | Latest | ✅      | ✅     |
| Safari   | Latest | ✅      | ✅     |
| Edge     | Latest | ✅      | ✅     |

**테스트 시나리오**:

1. 메인 페이지 네비게이션
2. Inspection 워크플로우
3. Single Analysis 워크플로우
4. 반응형 레이아웃

### 5.2 자동화 테스트

#### E2E 테스트 (Playwright)

**파일**: `tests/e2e/test_ui_workflows.py`

```python
import pytest
from playwright.sync_api import Page, expect

def test_navigation(page: Page):
    """메인 네비게이션 흐름 테스트"""
    page.goto("http://localhost:8000")

    page.click("text=Inspection")
    expect(page).to_have_url("http://localhost:8000/v7")

    page.click("text=Single Analysis")
    expect(page).to_have_url("http://localhost:8000/single_analysis")

def test_inspection_workflow(page: Page):
    """검사 워크플로우 테스트"""
    page.goto("http://localhost:8000/v7")
    page.select_option("#inspProductSelect", "SKU_TEMP_INK2_LOW")
    page.set_input_files("#inspFiles", "tests/fixtures/test_lens.jpg")
    page.click("#btnInspect")
    page.wait_for_selector("#inspResultArea", state="visible", timeout=30000)
    expect(page.locator("#inspLabel")).to_be_visible()
```

#### JavaScript 단위 테스트 (Vitest)

**파일**: `tests/js/components.test.js`

```javascript
import { describe, it, expect } from "vitest";
import { Components } from "../../src/web/static/js/components/base.js";

describe("Components", () => {
  it("버튼 생성 테스트", () => {
    const btn = Components.createButton("Test", "primary");
    expect(btn.classList.contains("btn-primary")).toBe(true);
  });

  it("메트릭 카드 생성 테스트", () => {
    const card = Components.createMetricCard("Label", "42", "px");
    expect(card.querySelector(".metric-value").textContent).toContain("42");
  });
});
```

### 5.3 성능 테스트

**Lighthouse CI 설정**:

```yaml
# .lighthouserc.yml
ci:
  collect:
    url:
      - http://localhost:8000
      - http://localhost:8000/v7
      - http://localhost:8000/single_analysis
    numberOfRuns: 3
  assert:
    assertions:
      categories:performance:
        - minScore: 0.9
      categories:accessibility:
        - minScore: 0.95
      categories:best-practices:
        - minScore: 0.9
```

---

## 6. 리스크 관리

### 6.1 식별된 리스크

| 리스크               | 확률 | 영향 | 완화 전략                          |
| -------------------- | ---- | ---- | ---------------------------------- |
| 기존 워크플로우 중단 | 중   | 고   | 기존 URL 유지, 점진적 마이그레이션 |
| 브라우저 호환성      | 중   | 중   | Polyfill, 광범위 테스트            |
| 성능 저하            | 저   | 중   | 코드 스플리팅, 최적화              |
| 사용자 반발          | 저   | 중   | A/B 테스트, 피드백 수집            |

### 6.2 완화 전략 상세

#### 기존 워크플로우 보호

```python
# FastAPI에서 기능 플래그 사용
USE_NEW_UI = os.getenv("USE_NEW_UI", "false") == "true"

@app.get("/v7")
async def v7_page(request: Request):
    if USE_NEW_UI:
        return templates.TemplateResponse("v7_mvp_new.html", {"request": request})
    else:
        return templates.TemplateResponse("v7_mvp.html", {"request": request})
```

#### 브라우저 호환성

```html
<!-- Polyfill for older browsers -->
<script
  crossorigin
  src="https://polyfill.io/v3/polyfill.min.js?features=es2015%2Ces2016%2Ces2017"
></script>
```

---

## 7. 타임라인

### Week 1-2: 기초 인프라

- [ ] 디자인 시스템 CSS 작성
- [ ] `_base_layout.html` 생성
- [ ] 컴포넌트 라이브러리 구축

### Week 3-4: 템플릿 통합

- [ ] `index.html` 리팩토링
- [ ] `v7_mvp.html` 리팩토링
- [ ] `single_analysis.html` 리팩토링

### Week 5-6: JavaScript 리팩토링

- [ ] ES6 모듈 전환
- [ ] 상태 관리 통합
- [ ] API 레이어 개선

### Week 7-8: 기능 개선

- [ ] Inspection UI 개선
- [ ] Single Analysis UI 개선
- [ ] History/Stats 개선

### Week 9-10: 테스트 및 배포

- [ ] E2E 테스트 작성
- [ ] 브라우저 호환성 테스트
- [ ] 성능 최적화
- [ ] 프로덕션 배포

---

## 8. 성공 지표 (KPI)

| 지표                     | 현재        | 목표    |
| ------------------------ | ----------- | ------- |
| Lighthouse Performance   | -           | > 90    |
| Lighthouse Accessibility | -           | > 95    |
| 코드 중복률              | ~40% (추정) | < 15%   |
| 평균 페이지 로드 시간    | -           | < 2초   |
| JavaScript 번들 크기     | ~300KB      | < 200KB |

---

## 9. 참고 문서

- [ENGINE_UNIFICATION_STATUS.md](engine_v7/ENGINE_UNIFICATION_STATUS.md) - 엔진 통합 현황
- [UI_FLOW_DIAGRAM.md](engine_v7/UI_FLOW_DIAGRAM.md) - UI 플로우
- [Longterm_Roadmap.md](Longterm_Roadmap.md) - 장기 로드맵

---

---

## 10. Phase 6 엔진 리팩토링 연동 고려사항

> Phase 6에서 완료된 엔진 리팩토링 작업과의 연동을 위한 추가 고려사항입니다.

### 10.1 신규 엔진 모듈 구조

Phase 6에서 다음과 같은 모듈 분리가 완료되었습니다:

```
src/engine_v7/core/
├── plate/
│   ├── __init__.py
│   ├── _helpers.py          # 공유 헬퍼 함수 (신규)
│   ├── plate_engine.py      # 전체 분석
│   └── plate_gate.py        # 경량 Gate 추출 (신규)
├── simulation/
│   ├── __init__.py
│   ├── color_simulator.py   # Area-ratio + Mask-based 시뮬레이션
│   └── mask_compositor.py   # 마스크 기반 합성 (신규)
└── ...
```

**UI 연동 포인트:**

| 모듈                 | UI 연동                 | API 엔드포인트                    |
| -------------------- | ----------------------- | --------------------------------- |
| `plate_gate.py`      | Hard Gate 시각화        | `/api/v7/plate_gate` (신규 필요)  |
| `mask_compositor.py` | 디지털 프루핑 결과 표시 | `/api/v7/simulation` 확장         |
| `color_simulator.py` | 시뮬레이션 방법 선택 UI | 기존 API에 `method` 파라미터 추가 |

### 10.2 API 라우터 업데이트 필요사항

**`src/web/routers/v7.py` 수정 사항:**

#### 시뮬레이션 방법 선택 지원

```python
from pydantic import BaseModel
from typing import Literal

class SimulationRequest(BaseModel):
    sku: str
    ink: str
    method: Literal["area_ratio", "mask_based"] = "area_ratio"
    # ... other fields

@router.post("/simulation")
async def run_simulation(request: SimulationRequest):
    """디지털 프루핑 시뮬레이션 (방법 선택 가능)"""

    if request.method == "mask_based":
        # mask_compositor 사용 (정밀)
        from ...engine_v7.core.simulation.mask_compositor import composite_from_masks

        # 마스크 기반 합성 파라미터
        result = composite_from_masks(
            masks=request.masks,  # List[np.ndarray]
            ink_labs=request.ink_labs,  # List[Tuple[float, float, float]]
            bg_type=request.bg_type,  # "white" | "black"
            composite_resolution=512
        )
    else:
        # area_ratio 방식 (빠름)
        from ...engine_v7.core.simulation.color_simulator import simulate_perceived_color

        result = simulate_perceived_color(
            ink_lab=request.ink_lab,
            area_ratio=request.area_ratio,
            bg="white"  # or "black"
        )

    return {
        "method": request.method,
        "perceived_lab": result.get("perceived_lab"),
        "perceived_rgb": result.get("perceived_rgb"),
        "composite_image": result.get("composite_image"),  # Base64 or URL
    }
```

#### Plate Gate 추출 엔드포인트 (신규)

```python
@router.post("/plate_gate")
async def extract_plate_gate_api(
    white_file: UploadFile,
    black_file: UploadFile,
    sku: str
):
    """경량 Plate Gate 추출 (시각화용)"""
    from ...engine_v7.core.plate.plate_gate import extract_plate_gate

    # 이미지 로드
    white_bgr = await load_image_from_upload(white_file)
    black_bgr = await load_image_from_upload(black_file)

    # SKU config 로드
    cfg = load_sku_config(sku)

    # Gate 추출
    gate_result = extract_plate_gate(white_bgr, black_bgr, cfg)

    return {
        "usable": gate_result["gate_quality"]["usable"],
        "artifact_ratio": gate_result["gate_quality"]["artifact_ratio"],
        "registration": gate_result["registration"],
        # Polar mask를 이미지로 변환하여 반환
        "ink_mask_polar_image": polar_mask_to_base64(
            gate_result["ink_mask_core_polar"]
        ),
        "geom": {
            "cx": gate_result["geom"].cx,
            "cy": gate_result["geom"].cy,
            "r": gate_result["geom"].r,
        }
    }
```

**신규 엔드포인트 우선순위:**

| 엔드포인트                                  | 용도                             | 우선순위 | Phase 연계   |
| ------------------------------------------- | -------------------------------- | -------- | ------------ |
| `POST /api/v7/plate_gate`                   | Gate 마스크 추출 (시각화용)      | **높음** | Phase 6 연동 |
| `POST /api/v7/simulation` (method 파라미터) | 시뮬레이션 방법 선택             | **중**   | Phase 6 연동 |
| `GET /api/v7/simulation/methods`            | 사용 가능한 시뮬레이션 방법 목록 | 저       | 정보성       |
| `POST /api/v7/compare_simulation`           | Area-ratio vs Mask-based 비교    | 저       | 고급 기능    |

### 10.3 UI 컴포넌트 추가 제안

#### 시뮬레이션 방법 선택 UI

**HTML 컴포넌트 (`single_analysis.html` 또는 `v7_mvp.html`):**

```html
<!-- Inspection 탭 또는 Single Analysis 탭에 추가 -->
<div
  class="simulation-method-selector bg-surface-elevated p-4 rounded-lg border border-white/10 mb-4"
>
  <div class="flex items-center justify-between mb-3">
    <label class="text-sm font-bold text-dim uppercase">시뮬레이션 방법</label>
    <button id="simMethodInfo" class="btn-icon" title="시뮬레이션 방법 도움말">
      <i class="fa-solid fa-circle-info"></i>
    </button>
  </div>

  <select id="simMethodSelect" class="terminal-input">
    <option value="area_ratio">Area Ratio (빠름, 권장)</option>
    <option value="mask_based">Mask-based (정밀, 실험적)</option>
  </select>

  <!-- 설명 툴팁 -->
  <div
    id="simMethodTooltip"
    class="hidden mt-2 p-2 bg-brand-500/10 border border-brand-500/30 rounded text-xs"
  >
    <p><b>Area Ratio:</b> 잉크 면적 비율 기반 근사 (빠름)</p>
    <p class="mt-1"><b>Mask-based:</b> 픽셀 단위 정밀 합성 (느림)</p>
  </div>
</div>
```

**JavaScript 핸들러:**

```javascript
// features/analysis/simulation_selector.js
import { appState } from "../../core/state.js";

export function initSimulationSelector() {
  const select = document.getElementById("simMethodSelect");
  const tooltip = document.getElementById("simMethodTooltip");
  const infoBtn = document.getElementById("simMethodInfo");

  // 방법 변경 이벤트
  select.addEventListener("change", (e) => {
    const method = e.target.value;
    appState.setState("analysis.simulationMethod", method);

    // 사용자에게 알림
    if (method === "mask_based") {
      showNotification(
        "정밀 모드 선택됨",
        "시뮬레이션 시간이 길어질 수 있습니다.",
        "info",
      );
    }
  });

  // 도움말 토글
  infoBtn.addEventListener("click", () => {
    tooltip.classList.toggle("hidden");
  });
}
```

#### Gate 마스크 시각화 컴포넌트

**파일**: `features/inspection/gate_visual.js`

```javascript
import { apiClient } from "../../core/api.js";

/**
 * Plate Gate 결과를 Canvas에 렌더링
 * @param {Object} plateGateResult - /api/v7/plate_gate 응답
 */
export async function renderGateMask(plateGateResult) {
  const { ink_mask_polar_image, gate_quality, geom, registration } =
    plateGateResult;

  // Gate 품질 표시
  const qualityEl = document.getElementById("gateQuality");
  qualityEl.innerHTML = `
    <div class="metric-card ${
      gate_quality.usable ? "border-success" : "border-error"
    }">
      <div class="text-xs text-dim">Gate Quality</div>
      <div class="text-2xl font-bold ${
        gate_quality.usable ? "text-success" : "text-error"
      }">
        ${gate_quality.usable ? "✓ USABLE" : "✗ UNUSABLE"}
      </div>
      <div class="text-xs mt-1">Artifact Ratio: ${(gate_quality.artifact_ratio * 100).toFixed(1)}%</div>
    </div>
  `;

  // Polar mask 이미지 표시
  const canvas = document.getElementById("gateMaskCanvas");
  const ctx = canvas.getContext("2d");

  const img = new Image();
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    // 중심점 표시
    ctx.strokeStyle = "#3b82f6";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(geom.cx, geom.cy, geom.r, 0, Math.PI * 2);
    ctx.stroke();
  };
  img.src = `data:image/png;base64,${ink_mask_polar_image}`;

  // Registration 정보 표시
  if (registration.swapped) {
    showNotification(
      "Image Swapped",
      "White/Black 이미지가 자동으로 교환되었습니다.",
      "warning",
    );
  }
}

/**
 * Plate Gate 추출 실행
 */
export async function runPlateGateExtraction(whiteFile, blackFile, sku) {
  const formData = new FormData();
  formData.append("white_file", whiteFile);
  formData.append("black_file", blackFile);
  formData.append("sku", sku);

  try {
    const result = await apiClient.post(
      "/v7/plate_gate",
      formData,
      "multipart",
    );
    await renderGateMask(result);
    return result;
  } catch (error) {
    console.error("Plate Gate extraction failed:", error);
    showNotification("Gate 추출 실패", error.message, "error");
    throw error;
  }
}
```

**HTML 템플릿 추가 (`single_analysis.html` Plate 탭):**

````html
<!-- Plate Tab에 추가 -->
<div class="gate-visualization mb-6">
  <h4 class="text-lg font-bold mb-4">Gate 마스크 시각화</h4>

  <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
    <!-- Gate Quality -->
    <div id="gateQuality"></div>

    <!-- Canvas -->
    <div class="terminal-panel p-4">
      <canvas
        id="gateMaskCanvas"
        class="w-full border border-white/10 rounded"
      ></canvas>
    </div>
  </div>
</div>

### 10.4 Lazy Import 고려 Phase 6에서 도입된 lazy import 패턴을 UI에서도
활용하여 초기 로딩 속도를 개선합니다. #### 동적 모듈 로딩 ```javascript //
features/inspection/lazy_loader.js /** * 필요 시점에 잉크 시각화 모듈 로드 */
export async function loadInkVisuals() { const { renderInkChart,
renderInkPalette } = await import('./ink_visuals.js'); return { renderInkChart,
renderInkPalette }; } /** * 차트 라이브러리 동적 로드 */ export async function
loadChartJs() { if (!window.Chart) { await
import('https://cdn.jsdelivr.net/npm/chart.js'); } return window.Chart; } /** *
대용량 진단 시각화 모듈 lazy 로드 */ export async function
loadDiagnosticsVisuals() { const module = await
import('./diagnostics_visuals.js'); return module; } // 사용 예시 export async
function showInkAnalysis(data) { const { renderInkChart } = await
loadInkVisuals(); const Chart = await loadChartJs(); await renderInkChart(data,
Chart); }
````

#### 페이지별 번들 분할 (Vite 사용 시)

```javascript
// vite.config.js (빌드 시스템 도입 시)
export default {
  build: {
    rollupOptions: {
      input: {
        main: "src/web/static/js/main.js",
        inspection: "src/web/static/js/features/inspection/index.js",
        analysis: "src/web/static/js/features/analysis/index.js",
      },
      output: {
        manualChunks: {
          vendor: ["chart.js"],
          core: ["./core/api.js", "./core/state.js"],
        },
      },
    },
  },
};
```

---

### 10.5 헬퍼 함수 및 유틸리티

#### Polar mask → Base64 변환

**파일**: `src/web/routers/helpers.py`

```python
import base64
import io
import numpy as np
from PIL import Image

def polar_mask_to_base64(polar_mask: np.ndarray) -> str:
    """
    Polar boolean mask를 Base64 PNG 이미지로 변환

    Args:
        polar_mask: (T, R) boolean array

    Returns:
        Base64 encoded PNG string
    """
    if polar_mask is None:
        return ""

    # Boolean mask를 uint8 이미지로 변환 (0 or 255)
    img_array = (polar_mask.astype(np.uint8) * 255)

    # PIL Image로 변환
    img = Image.fromarray(img_array, mode='L')  # Grayscale

    # PNG로 인코딩
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    # Base64 인코딩
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return img_base64

async def load_image_from_upload(file: UploadFile) -> np.ndarray:
    """
    UploadFile을 OpenCV BGR 이미지로 로드
    """
    import cv2
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img
```

#### NotificationHelper (JavaScript)

**파일**: `src/web/static/js/utils/notifications.js`

```javascript
/**
 * 사용자 알림 표시 (토스트 스타일)
 * @param {string} title - 알림 제목
 * @param {string} message - 알림 메시지
 * @param {string} type - 'success' | 'error' | 'warning' | 'info'
 */
export function showNotification(title, message, type = "info") {
  const container =
    document.getElementById("notificationContainer") ||
    createNotificationContainer();

  const notification = document.createElement("div");
  notification.className = `notification notification-${type} animate-slide-in`;
  notification.innerHTML = `
    <div class="flex items-start gap-3">
      <div class="notification-icon">${getIcon(type)}</div>
      <div class="flex-1">
        <div class="notification-title">${title}</div>
        <div class="notification-message">${message}</div>
      </div>
      <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
        <i class="fa-solid fa-times"></i>
      </button>
    </div>
  `;

  container.appendChild(notification);

  // 5초 후 자동 제거
  setTimeout(() => {
    notification.classList.add("animate-slide-out");
    setTimeout(() => notification.remove(), 300);
  }, 5000);
}

function createNotificationContainer() {
  const container = document.createElement("div");
  container.id = "notificationContainer";
  container.className = "fixed top-4 right-4 z-50 space-y-2";
  document.body.appendChild(container);
  return container;
}

function getIcon(type) {
  const icons = {
    success: '<i class="fa-solid fa-circle-check text-green-ok"></i>',
    error: '<i class="fa-solid fa-circle-xmark text-red-error"></i>',
    warning:
      '<i class="fa-solid fa-triangle-exclamation text-yellow-warning"></i>',
    info: '<i class="fa-solid fa-circle-info text-blue-brand"></i>',
  };
  return icons[type] || icons.info;
}
```

---

## 11. 추가 개선 고려사항

### 11.1 실시간 업데이트 (WebSocket)

장시간 검사 작업의 진행 상태를 실시간으로 표시:

```python
# FastAPI WebSocket 엔드포인트
@app.websocket("/ws/inspection/{job_id}")
async def inspection_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    while True:
        progress = get_job_progress(job_id)
        await websocket.send_json(progress)
        if progress["status"] in ("completed", "failed"):
            break
        await asyncio.sleep(0.5)
```

### 11.2 다크/라이트 테마 전환

디자인 시스템에 테마 전환 지원 추가:

```css
/* design_system.css */
:root {
  --bg-base: #0f172a;
  /* ... dark theme defaults ... */
}

[data-theme="light"] {
  --bg-base: #f8fafc;
  --bg-surface: #ffffff;
  --text-primary: #0f172a;
  /* ... */
}
```

```javascript
// core/theme.js
export function toggleTheme() {
  const current = document.documentElement.dataset.theme || "dark";
  document.documentElement.dataset.theme =
    current === "dark" ? "light" : "dark";
  localStorage.setItem("theme", document.documentElement.dataset.theme);
}
```

### 11.3 오프라인 지원 (PWA)

검사 현장에서 네트워크 불안정 시 대비:

```javascript
// service-worker.js
const CACHE_NAME = "color-meter-v1";
const STATIC_ASSETS = [
  "/",
  "/static/css/design_system.css",
  "/static/js/core/api.js",
  // ...
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS)),
  );
});
```

### 11.4 국제화 (i18n) 준비

향후 다국어 지원을 위한 구조 준비:

```javascript
// core/i18n.js
const translations = {
  ko: {
    "inspection.title": "검사",
    "inspection.result.pass": "합격",
    "inspection.result.fail": "불합격",
  },
  en: {
    "inspection.title": "Inspection",
    "inspection.result.pass": "PASS",
    "inspection.result.fail": "FAIL",
  },
};

export function t(key) {
  const lang = localStorage.getItem("lang") || "ko";
  return translations[lang]?.[key] || key;
}
```

---

## 12. 우선순위 정리

| 순위 | 항목                     | Phase | 비고            |
| ---- | ------------------------ | ----- | --------------- |
| 1    | 디자인 시스템 구축       | 1     | 기반 작업       |
| 2    | `_base_layout.html` 생성 | 1     | 네비게이션 통합 |
| 3    | ES6 모듈 전환            | 3     | 코드 품질       |
| 4    | simulation API 연동      | -     | Phase 6 연계    |
| 5    | 다크/라이트 테마         | 4     | 사용자 경험     |
| 6    | WebSocket 진행률         | 4     | 사용자 경험     |
| 7    | PWA 지원                 | -     | 선택 사항       |
| 8    | i18n                     | -     | 선택 사항       |

---

## 변경 이력

| 버전 | 날짜       | 변경 내용                                  | 작성자 |
| ---- | ---------- | ------------------------------------------ | ------ |
| 1.0  | 2026-01-17 | 초기 문서 작성                             | 개발팀 |
| 1.1  | 2026-01-17 | Phase 6 연동 고려사항, 추가 개선 항목 추가 | 개발팀 |
