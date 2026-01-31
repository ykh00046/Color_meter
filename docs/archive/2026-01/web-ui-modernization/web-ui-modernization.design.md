# Design: web-ui-modernization

> **Phase**: Design
> **Created**: 2026-01-31
> **Plan Reference**: `docs/01-plan/features/web-ui-modernization.plan.md`
> **Existing Plan Reference**: `docs/WEB_UI_MODERNIZATION_PLAN.md` (v1.0, Approved)

---

## 1. Architecture Overview

### 1.1 Current Stack (Phase 1~3 Complete)

```
Browser
  ├─ HTML: Jinja2 SSR (8 templates, _base_layout.html 공통)
  ├─ CSS:  design_system.css (710L, CSS vars) + Tailwind CDN
  ├─ JS:   ES6 Modules (23 files, ~4,166L)
  │   ├─ core/      state.js (Observer), api.js (REST client)
  │   ├─ components/ base.js (factories), visuals.js, tabs.js
  │   ├─ features/   inspection/, analysis/, registration/
  │   └─ utils/      helpers.js, notifications.js, i18n.js
  └─ Libs: Chart.js (CDN), Font Awesome (CDN)

FastAPI Backend
  ├─ Routers: v7_registration.py, v7_inspection.py, v7_plate.py, ...
  └─ Engine:  src/engine_v7/
```

### 1.2 Target Architecture (Phase 4~6 After)

```
Browser
  ├─ HTML: Jinja2 SSR + Partials ({% include %} for component splitting)
  ├─ CSS:  design_system.css (a11y 보강) + Tailwind CDN (현행 유지)
  ├─ JS:   ES6 Modules (기존 구조 확장)
  │   ├─ core/      state.js, api.js (변경 없음)
  │   ├─ components/ base.js, tabs.js (ARIA 보강)
  │   ├─ shared/    ★ file_upload.js, data_table.js, progress.js (신규)
  │   ├─ features/  inspection/ (DnD 적용), analysis/ (분할 후)
  │   └─ utils/     helpers.js, notifications.js, i18n.js (확장)
  └─ Libs: Chart.js (CDN), Font Awesome (CDN)

FastAPI Backend
  ├─ Routers: (변경 없음, SSE 엔드포인트 1개 추가 가능)
  └─ Engine:  (변경 없음)
```

---

## 2. Detailed Design

### 2.1 Phase 4-A: Drag & Drop File Upload Component

#### 2.1.1 Module: `src/web/static/js/shared/file_upload.js`

**API:**
```javascript
/**
 * DragDropUpload - 재사용 가능한 드래그앤드롭 파일 업로드 컴포넌트
 *
 * @param {Object} options
 * @param {string} options.dropZoneId   - 드롭 영역 요소 ID
 * @param {string} options.fileInputId  - hidden file input ID
 * @param {string} options.displayId    - 파일명 표시 요소 ID
 * @param {string[]} options.accept     - 허용 확장자 ['.png','.jpg','.jpeg','.bmp','.tiff']
 * @param {number} options.maxSizeMB    - 최대 파일 크기 (기본 50)
 * @param {Function} options.onFile     - 파일 선택 콜백 (file) => void
 * @param {Function} options.onError    - 에러 콜백 (message) => void
 */
export function createDragDropUpload(options) { ... }
```

**동작 사양:**
1. 드롭 영역에 `dragover` → 시각적 하이라이트 (border 색상 변경)
2. `dragleave` → 하이라이트 해제
3. `drop` → 파일 확장자/크기 검증 → `onFile` 콜백
4. 클릭 → 기존 file input 트리거 (하위 호환)
5. 파일명 표시 (truncate), 미리보기 썸네일 (이미지인 경우)

**적용 대상:**

| 페이지 | dropZoneId | fileInputId | displayId |
|--------|-----------|-------------|-----------|
| v7_mvp.html (Inspection) | `inspDropZone` | `inspFiles` | `inspFileName` |
| single_analysis.html (White) | `saWhiteDropZone` | `fileWhite` | `fileWhiteName` |
| single_analysis.html (Black) | `saBlackDropZone` | `fileBlack` | `fileBlackName` |

**HTML 변경 (v7_mvp.html 예시):**
```html
<!-- Before -->
<div class="relative">
  <input id="inspFiles" type="file" class="absolute inset-0 opacity-0 cursor-pointer">
  <div class="terminal-input truncate ...">
    <span id="inspFileName" class="text-dim">Upload Image...</span>
    <i class="fa-solid fa-upload"></i>
  </div>
</div>

<!-- After -->
<div id="inspDropZone" class="drop-zone" role="button" tabindex="0"
     aria-label="Upload inspection image. Click or drag and drop.">
  <input id="inspFiles" type="file" class="sr-only"
         accept=".png,.jpg,.jpeg,.bmp,.tiff,.tif" aria-hidden="true">
  <div class="drop-zone-content">
    <i class="fa-solid fa-cloud-arrow-up drop-zone-icon"></i>
    <span id="inspFileName" class="drop-zone-text">
      Click or drag image here
    </span>
  </div>
  <div id="inspFilePreview" class="drop-zone-preview hidden"></div>
</div>
```

**CSS 추가 (design_system.css):**
```css
.drop-zone {
  border: 2px dashed var(--border-dim);
  border-radius: var(--radius-lg);
  padding: var(--spacing-lg);
  text-align: center;
  cursor: pointer;
  transition: border-color 0.2s, background-color 0.2s;
}
.drop-zone:hover,
.drop-zone.drag-over {
  border-color: var(--color-primary);
  background-color: rgba(59, 130, 246, 0.05);
}
.drop-zone:focus-visible {
  outline: 2px solid var(--color-primary);
  outline-offset: 2px;
}
.drop-zone-icon {
  font-size: var(--font-size-2xl);
  color: var(--text-dim);
  margin-bottom: var(--spacing-sm);
}
.drop-zone-text {
  color: var(--text-dim);
  font-size: var(--font-size-sm);
}
.drop-zone-preview {
  margin-top: var(--spacing-sm);
}
.drop-zone-preview img {
  max-height: 80px;
  border-radius: var(--radius-sm);
  object-fit: contain;
}
```

---

### 2.2 Phase 4-B: Single Analysis Template Splitting

#### 2.2.1 분할 구조

**현재:** `single_analysis.html` (843줄, 단일 파일)

**분할 후:**
```
src/web/templates/
├── single_analysis.html           (~120줄, 골격 + includes)
└── partials/
    ├── _sa_upload.html            (~80줄, 업로드 폼 + 옵션)
    ├── _sa_palette_tab.html       (~200줄, 잉크 팔레트 + 시뮬레이터)
    ├── _sa_radial_tab.html        (~40줄, 레이디얼 차트)
    ├── _sa_plate_tab.html         (~100줄, Plate 분석 결과)
    └── _sa_details_tab.html       (~80줄, 상세 메트릭)
```

**single_analysis.html 골격:**
```html
{% extends "_base_layout.html" %}
{% block content %}
<div class="sa-container">
  {% include "partials/_sa_upload.html" %}

  <section id="resultsSection" class="hidden" aria-label="Analysis Results">
    <div class="result-tabs" role="tablist" aria-label="Result tabs">
      <button role="tab" data-tab-button="palette" aria-selected="true"
              aria-controls="panel-palette" id="tab-palette">
        <i class="fa-solid fa-palette"></i> Palette
      </button>
      <button role="tab" data-tab-button="radial" aria-selected="false"
              aria-controls="panel-radial" id="tab-radial">
        <i class="fa-solid fa-chart-pie"></i> Radial
      </button>
      <button role="tab" data-tab-button="plate" aria-selected="false"
              aria-controls="panel-plate" id="tab-plate">
        <i class="fa-solid fa-layer-group"></i> Plate
      </button>
      <button role="tab" data-tab-button="details" aria-selected="false"
              aria-controls="panel-details" id="tab-details">
        <i class="fa-solid fa-list"></i> Details
      </button>
    </div>

    <div id="panel-palette" role="tabpanel" aria-labelledby="tab-palette"
         data-result-tab="palette">
      {% include "partials/_sa_palette_tab.html" %}
    </div>
    <div id="panel-radial" role="tabpanel" aria-labelledby="tab-radial"
         data-result-tab="radial" class="hidden">
      {% include "partials/_sa_radial_tab.html" %}
    </div>
    <div id="panel-plate" role="tabpanel" aria-labelledby="tab-plate"
         data-result-tab="plate" class="hidden">
      {% include "partials/_sa_plate_tab.html" %}
    </div>
    <div id="panel-details" role="tabpanel" aria-labelledby="tab-details"
         data-result-tab="details" class="hidden">
      {% include "partials/_sa_details_tab.html" %}
    </div>
  </section>
</div>
{% endblock %}
```

**ID 보존 규칙:** 기존 JS가 참조하는 모든 `#id`는 분할 후에도 동일하게 유지한다.
- `#fileWhite`, `#fileBlack`, `#inkCountInput`, `#analysisScope`
- `#btnAnalyze`, `#resultsSection`, `#loadingOverlay`
- 탭 패널 내부 ID들 (팔레트 카드, 차트 캔버스 등)

---

### 2.3 Phase 4-C: Data Table Component

#### 2.3.1 Module: `src/web/static/js/shared/data_table.js`

**API:**
```javascript
/**
 * DataTable - 정렬/필터/내보내기 지원 테이블 컴포넌트
 *
 * @param {Object} options
 * @param {string} options.containerId  - 테이블 컨테이너 ID
 * @param {Array} options.columns       - [{key, label, sortable, filterable, formatter}]
 * @param {Function} options.fetchData  - (params) => Promise<{rows, total}>
 * @param {number} options.pageSize     - 페이지 크기 (기본 20)
 */
export class DataTable {
  constructor(options) { ... }
  async load(page = 1) { ... }
  sort(columnKey, direction) { ... }
  filter(filters) { ... }
  async exportCsv(filename) { ... }
  destroy() { ... }
}
```

**기능 상세:**

| 기능 | 구현 방식 |
|------|-----------|
| 정렬 | 컬럼 헤더 클릭 → ASC/DESC 토글, `aria-sort` 속성 업데이트 |
| 필터 | 상단 필터 바 → `fetchData(params)` 호출 |
| 페이지네이션 | 하단 prev/next + 페이지 번호, `aria-label="Page N"` |
| CSV 내보내기 | 현재 필터 기준 전체 데이터 → Blob → download |
| 빈 상태 | 데이터 없을 때 `Components.createEmptyState()` 표시 |

**접근성:**
```html
<table role="grid" aria-label="Inspection History">
  <thead>
    <tr role="row">
      <th role="columnheader" aria-sort="ascending" tabindex="0">
        Date <i class="fa-solid fa-sort-up"></i>
      </th>
    </tr>
  </thead>
  <tbody role="rowgroup" aria-live="polite">
    <!-- 동적 행 -->
  </tbody>
</table>
```

**적용 대상:**
- `history.html` - 기존 수동 테이블 → `DataTable` 교체
- `stats.html` - 통계 상세 테이블 (해당 시)

---

### 2.4 Phase 4-D: Progress Indicator (SSE)

#### 2.4.1 Module: `src/web/static/js/shared/progress.js`

**API:**
```javascript
/**
 * ProgressTracker - SSE 기반 실시간 진행률 표시
 *
 * @param {Object} options
 * @param {string} options.containerId - 프로그레스 바 컨테이너 ID
 * @param {string} options.endpoint    - SSE 엔드포인트 URL
 * @param {Function} options.onComplete - 완료 콜백
 * @param {Function} options.onError   - 에러 콜백
 */
export class ProgressTracker {
  start(params) { ... }   // EventSource 연결
  stop() { ... }          // 연결 해제
}
```

**SSE 메시지 형식 (백엔드):**
```json
{"stage": "gate", "progress": 30, "message": "Running gate analysis..."}
{"stage": "ink", "progress": 60, "message": "Extracting ink masks..."}
{"stage": "signature", "progress": 90, "message": "Computing signature..."}
{"stage": "done", "progress": 100, "message": "Complete"}
```

**UI 렌더링:**
```html
<div id="inspProgress" class="progress-container hidden"
     role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"
     aria-label="Analysis progress">
  <div class="progress-bar">
    <div class="progress-fill" style="width: 0%"></div>
  </div>
  <span class="progress-text">Preparing...</span>
</div>
```

**백엔드 엔드포인트 (선택, 구현 시):**
```python
# src/web/routers/v7_inspection.py
@router.get("/stream_progress/{task_id}")
async def stream_progress(task_id: str):
    async def event_generator():
        # ... yield SSE events
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

> **Note:** 백엔드 SSE 엔드포인트는 현재 구현 범위 밖이다. Phase 4에서는 프론트엔드 컴포넌트만 구현하고, 실제 SSE 연동은 백엔드 준비 후 진행한다. 초기 구현에서는 단계별 스피너(Gate → Ink → Signature → Complete)를 시뮬레이션한다.

---

### 2.5 Phase 5-A: Accessibility (WCAG 2.1 AA)

#### 2.5.1 탭 컴포넌트 ARIA 보강

**수정 파일:** `src/web/static/js/components/tabs.js` (48줄)

**변경 사항:**
```javascript
// switchResultTab 함수 수정
export function switchResultTab(tabName) {
  document.querySelectorAll('[data-tab-button]').forEach(btn => {
    const isActive = btn.dataset.tabButton === tabName;
    btn.classList.toggle('active', isActive);
    btn.setAttribute('aria-selected', String(isActive));
    btn.setAttribute('tabindex', isActive ? '0' : '-1');
  });

  document.querySelectorAll('[data-result-tab]').forEach(panel => {
    const isVisible = panel.dataset.resultTab === tabName;
    panel.classList.toggle('hidden', !isVisible);
    panel.setAttribute('aria-hidden', String(!isVisible));
  });
}

// 키보드 네비게이션 추가
export function initResultTabs() {
  const tabList = document.querySelector('[role="tablist"]');
  if (!tabList) return;

  tabList.addEventListener('keydown', (e) => {
    const tabs = [...tabList.querySelectorAll('[role="tab"]')];
    const idx = tabs.indexOf(e.target);
    let next = -1;

    if (e.key === 'ArrowRight') next = (idx + 1) % tabs.length;
    else if (e.key === 'ArrowLeft') next = (idx - 1 + tabs.length) % tabs.length;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = tabs.length - 1;

    if (next >= 0) {
      e.preventDefault();
      tabs[next].focus();
      tabs[next].click();
    }
  });
}
```

#### 2.5.2 폼 접근성 보강

**패턴 (모든 input에 적용):**
```html
<!-- Before -->
<input id="inkCountInput" type="number" class="terminal-input" value="3">

<!-- After -->
<label for="inkCountInput" class="form-label">Expected Ink Count</label>
<input id="inkCountInput" type="number" class="terminal-input" value="3"
       aria-describedby="inkCountHint" min="1" max="8">
<span id="inkCountHint" class="form-hint">Number of inks expected (1-8)</span>
```

#### 2.5.3 로딩 상태 접근성

```html
<!-- 분석 중 상태 -->
<div id="loadingOverlay" class="hidden" role="status" aria-live="polite">
  <div class="spinner" aria-hidden="true"></div>
  <span class="sr-only">Analysis in progress, please wait.</span>
</div>
```

#### 2.5.4 색상 대비 검증

**현재 문제 영역 (design_system.css):**

| 요소 | 현재 | 대비 비율 | 수정 후 | 대비 비율 |
|------|------|-----------|---------|-----------|
| `--text-dim` on `--bg-base` | #94a3b8 on #0f172a | ~4.2:1 | #a1b0c4 | ~4.6:1 |
| `--text-dim` on `--bg-surface` | #94a3b8 on #151e32 | ~3.9:1 | #a1b0c4 | ~4.5:1 |
| `.badge` warning text | #f59e0b on transparent | 가변 | 배경 추가 | >= 4.5:1 |

**검증 도구:** Lighthouse Accessibility audit, axe-core

---

### 2.6 Phase 5-B: Performance Optimization

#### 2.6.1 JS 모듈 Lazy Loading

**현재:** `_base_layout.html`에서 모든 core 모듈을 즉시 로드
**변경:** 페이지별 모듈만 로드, 나머지는 `import()` 동적 로드

```javascript
// v7_mvp.html - 필요한 모듈만 로드
import { initInspection } from '/static/js/features/inspection/inspection.js';

// 탭 전환 시 동적 로드
async function onTabSwitch(tabName) {
  if (tabName === 'registration') {
    const { initRegistration } = await import('/static/js/features/registration/registration.js');
    initRegistration();
  }
}
```

#### 2.6.2 Chart.js Optimization

**현재:** CDN에서 전체 Chart.js 로드 (~200KB)
**변경:** ESM 빌드 사용 또는 CDN 유지 + `defer` 속성

```html
<!-- defer로 렌더링 차단 방지 -->
<script defer src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
```

#### 2.6.3 이미지 최적화

- 결과 이미지: `loading="lazy"` 속성 추가
- 오버레이/히트맵: 뷰포트 진입 시만 로드
- favicon, 로고: 캐시 헤더 설정 (FastAPI `StaticFiles`)

---

### 2.7 Phase 6: Advanced Features (Optional)

#### 2.7.1 i18n 시스템

**기존 파일:** `src/web/static/js/utils/i18n.js` (이미 존재)

**확장 설계:**
```javascript
// i18n.js 확장
const translations = {};
let currentLocale = 'ko';

export async function loadLocale(locale) {
  if (!translations[locale]) {
    const res = await fetch(`/static/locales/${locale}.json`);
    translations[locale] = await res.json();
  }
  currentLocale = locale;
  document.documentElement.lang = locale;
  updateAllTranslations();
}

export function t(key, params = {}) {
  const template = translations[currentLocale]?.[key] || key;
  return template.replace(/\{(\w+)\}/g, (_, k) => params[k] ?? `{${k}}`);
}

function updateAllTranslations() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    el.textContent = t(el.dataset.i18n);
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  });
}
```

**번역 파일 구조 (`locales/ko.json`):**
```json
{
  "nav.home": "홈",
  "nav.inspection": "검사",
  "nav.single_analysis": "단독 분석",
  "nav.calibration": "교정",
  "nav.history": "이력",
  "nav.stats": "통계",
  "upload.drag_hint": "이미지를 여기에 드래그하거나 클릭하세요",
  "upload.file_too_large": "파일이 너무 큽니다 (최대 {max}MB)",
  "analysis.running": "분석 중...",
  "analysis.complete": "분석 완료"
}
```

#### 2.7.2 PWA (선택)

- `manifest.json`: 앱 이름, 아이콘, 테마 색상
- Service Worker: 정적 에셋 캐싱 (CSS, JS, 폰트)
- 오프라인 페이지: 네트워크 없을 때 안내 메시지

> Phase 6는 선택 사항이며, Phase 4~5 완료 후 필요에 따라 진행한다.

---

## 3. Implementation Order

```
Phase 4-A: Drag & Drop Upload Component
  ├─ 1. design_system.css에 .drop-zone 스타일 추가
  ├─ 2. shared/file_upload.js 모듈 구현
  ├─ 3. v7_mvp.html 업로드 영역 교체
  └─ 4. single_analysis.html 업로드 영역 교체

Phase 4-B: Single Analysis Splitting
  ├─ 5. partials/ 디렉토리 생성
  ├─ 6. _sa_upload.html 분리
  ├─ 7. _sa_palette_tab.html 분리
  ├─ 8. _sa_radial_tab.html 분리
  ├─ 9. _sa_plate_tab.html 분리
  ├─ 10. _sa_details_tab.html 분리
  └─ 11. single_analysis.html 골격으로 교체 (ID 보존 검증)

Phase 4-C: Data Table Component
  ├─ 12. shared/data_table.js 모듈 구현
  ├─ 13. history.html에 DataTable 적용
  └─ 14. CSV 내보내기 구현

Phase 4-D: Progress Indicator
  ├─ 15. shared/progress.js 모듈 구현 (프론트엔드만)
  └─ 16. v7_mvp.html + single_analysis.html에 진행률 표시 추가

Phase 5-A: Accessibility
  ├─ 17. tabs.js ARIA 속성 + 키보드 네비게이션
  ├─ 18. 모든 form input에 label 연결
  ├─ 19. 로딩 상태 aria-live 추가
  ├─ 20. 색상 대비 비율 수정 (design_system.css)
  └─ 21. Lighthouse Accessibility 95+ 검증

Phase 5-B: Performance
  ├─ 22. JS 동적 import() 적용 (탭 전환 시)
  ├─ 23. 이미지 lazy loading
  └─ 24. Lighthouse Performance 90+ 검증
```

---

## 4. File Change Matrix

### 수정 파일

| # | 파일 | 변경 내용 | Phase |
|---|------|-----------|-------|
| 1 | `src/web/static/css/design_system.css` | .drop-zone 스타일 + a11y 색상 보정 | 4-A, 5-A |
| 2 | `src/web/templates/v7_mvp.html` | 업로드 영역 DnD 교체 + 진행률 UI | 4-A, 4-D |
| 3 | `src/web/templates/single_analysis.html` | 골격으로 교체 (include) + DnD | 4-A, 4-B |
| 4 | `src/web/templates/history.html` | DataTable 적용 | 4-C |
| 5 | `src/web/static/js/components/tabs.js` | ARIA + 키보드 네비게이션 | 5-A |
| 6 | `src/web/static/js/features/inspection/inspection.js` | DnD 연동 + progress | 4-A, 4-D |
| 7 | `src/web/templates/_base_layout.html` | sr-only 클래스, lang 속성 | 5-A |

### 신규 파일

| # | 파일 | 용도 | Phase |
|---|------|------|-------|
| 8 | `src/web/static/js/shared/file_upload.js` | DnD 업로드 컴포넌트 | 4-A |
| 9 | `src/web/static/js/shared/data_table.js` | 정렬/필터 테이블 | 4-C |
| 10 | `src/web/static/js/shared/progress.js` | 진행률 표시 | 4-D |
| 11 | `src/web/templates/partials/_sa_upload.html` | SA 업로드 파셜 | 4-B |
| 12 | `src/web/templates/partials/_sa_palette_tab.html` | SA 팔레트 파셜 | 4-B |
| 13 | `src/web/templates/partials/_sa_radial_tab.html` | SA 레이디얼 파셜 | 4-B |
| 14 | `src/web/templates/partials/_sa_plate_tab.html` | SA 플레이트 파셜 | 4-B |
| 15 | `src/web/templates/partials/_sa_details_tab.html` | SA 상세 파셜 | 4-B |

---

## 5. State Management Impact

**기존 appState 구조 변경 없음.** 신규 컴포넌트는 기존 state 경로를 사용한다.

| 컴포넌트 | 읽는 State | 쓰는 State |
|----------|-----------|-----------|
| DnD Upload | - | `inspection.uploadedFile`, `analysis.uploadedFile` |
| DataTable | `history.filters`, `history.currentPage` | `history.currentPage` |
| Progress | `inspection.isProcessing` | - (UI only) |

---

## 6. Testing Strategy

### 6.1 E2E (Playwright)

```
tests/e2e/
├── test_drag_drop_upload.py     - DnD 파일 업로드 동작 검증
├── test_single_analysis_tabs.py - 탭 전환, 키보드 네비게이션
├── test_history_table.py        - 정렬, 필터, 페이지네이션
└── test_accessibility.py        - axe-core 기반 a11y 스캔
```

### 6.2 Lighthouse CI Targets

| Category | Target | Current (est.) |
|----------|--------|----------------|
| Performance | >= 90 | ~75 |
| Accessibility | >= 95 | ~70 |
| Best Practices | >= 90 | ~85 |

---

## 7. Constraints & Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Tailwind CDN 유지 | Yes | Phase 4~5에서 제거하면 리스크 높음, Phase 6에서 검토 |
| 번들러 미도입 | Phase 6 이후 | 현재 ES6 modules + CDN으로 충분 |
| SSE 백엔드 미구현 | Phase 4에서 프론트만 | 백엔드 변경 최소화 원칙 |
| React/Vue 미도입 | 확정 | Plan에서 거부됨, Jinja2 SSR 유지 |
| 기존 ID 전수 보존 | 필수 | JS 바인딩 깨짐 방지 |
