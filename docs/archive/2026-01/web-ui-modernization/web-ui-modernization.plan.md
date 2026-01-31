# Plan: web-ui-modernization

> **Phase**: Plan
> **Created**: 2026-01-31
> **Status**: Draft

## 1. Overview

### 1.1 Feature Description
Color Meter 웹 UI의 Phase 4~6 현대화를 완료한다. Phase 1~3(디자인 시스템, 베이스 레이아웃, ES6 모듈 전환)은 이미 완료되었으며, 남은 기능 개선/접근성/성능 최적화를 수행한다.

### 1.2 Goal
- Phase 4: 주요 페이지 UX 개선 (드래그앤드롭, 반응형, 컴포넌트 분할)
- Phase 5: 접근성(a11y) WCAG 2.1 AA 준수 및 성능 최적화
- Phase 6: PWA 지원, i18n 확장, Vite 번들러 도입 (선택)

### 1.3 Scope

**In Scope**:
- Inspection 페이지 드래그앤드롭 파일 업로드
- 실시간 분석 진행률 표시 (WebSocket 또는 SSE)
- Single Analysis 935줄 HTML 컴포넌트 분할
- History/Stats 데이터 테이블 정렬/필터링/내보내기
- ARIA 레이블 추가 및 키보드 네비게이션
- Lighthouse 성능 점수 90+ 달성
- 다크/라이트 테마 전환 안정화

**Out of Scope**:
- React/Vue 프레임워크 전환 (거부됨)
- SPA 전환 (Jinja2 SSR 유지)
- 백엔드 API 로직 변경
- 모바일 네이티브 앱

## 2. Current State Analysis

### 2.1 완료된 작업 (Phase 1~3)

| Phase | 내용 | 상태 |
|-------|------|------|
| Phase 1 | 디자인 시스템 CSS (710줄), CSS 변수, 다크/라이트 테마 | ✅ 완료 |
| Phase 2 | `_base_layout.html` 베이스 템플릿, 공통 네비게이션 | ✅ 완료 |
| Phase 3 | ES6 모듈 전환, Observer 패턴 상태관리 (`state.js`) | ✅ 완료 |

### 2.2 현재 아키텍처

**Templates** (8개, ~120KB):
- `_base_layout.html` - 공통 레이아웃 (nav, footer, 테마)
- `index.html` - 메인 랜딩
- `v7_mvp.html` - v7 엔진 콘솔 (4탭: Registration/Inspection/Activation/Metrics)
- `single_analysis.html` - 단독 분석 (935줄, 분할 필요)
- `calibration.html` - 색상 교정
- `history.html` - 검사 이력
- `stats.html` - 통계 대시보드
- `plate_lite.html` - Plate Lite 도구

**JavaScript** (23개 파일, ~4,166줄):
```
src/web/static/js/
├── core/           # state.js, api.js, components.js, theme.js
├── v7/             # inspection.js, registration.js, single_analysis.js, ...
├── shared/         # chart_helpers.js, format_utils.js
└── pages/          # calibration_page.js, history_page.js, stats_page.js
```

**CSS** (2개):
- `design_system.css` (710줄) - CSS 변수, 컴포넌트 클래스
- `tailwind` (CDN) - 유틸리티 클래스

**기술 스택**:
- Backend: FastAPI + Jinja2
- Frontend: Vanilla JS (ES6 Modules), Tailwind CSS (CDN), Chart.js (CDN)
- Fonts: Inter, JetBrains Mono (Google Fonts)

### 2.3 남은 문제점

1. **single_analysis.html 935줄**: 단일 파일에 4개 탭의 모든 마크업 포함
2. **파일 업로드 UX**: 기본 `<input type="file">`, 드래그앤드롭 미지원
3. **접근성 미흡**: ARIA 레이블 누락, 키보드 네비게이션 불완전
4. **성능**: Tailwind CDN 전체 로드 (불필요한 유틸리티 포함)
5. **i18n**: 한국어/영어 하드코딩, 체계적 번역 시스템 없음
6. **진행률 표시 없음**: 분석 중 사용자 피드백 부재

## 3. Implementation Strategy

### 3.1 Phase 4: 기능 개선

#### 4.1 Inspection 페이지 개선
- 드래그앤드롭 파일 업로드 컴포넌트 (`file_upload.js`)
- 실시간 분석 진행률 (SSE 기반 `/v7/stream_progress`)
- 결과 영역 차트 인터랙션 개선
- 반응형 그리드 레이아웃

#### 4.2 Single Analysis 컴포넌트 분할
- 935줄 HTML → Jinja2 `{% include %}` 매크로로 분할
  - `_sa_upload_tab.html`
  - `_sa_results_tab.html`
  - `_sa_comparison_tab.html`
  - `_sa_plate_tab.html`
- 각 탭별 JS 모듈 분리

#### 4.3 History/Stats 페이지 개선
- 데이터 테이블 정렬/필터링 (vanilla JS, 외부 라이브러리 없음)
- 날짜 범위 선택기
- CSV/Excel 내보내기 기능

### 3.2 Phase 5: 접근성 & 성능

#### 5.1 접근성 (WCAG 2.1 AA)
- 모든 인터랙티브 요소에 `aria-label`, `role` 추가
- 키보드 네비게이션 (Tab, Enter, Escape)
- 색상 대비 비율 4.5:1 이상 검증
- `<main>`, `<nav>`, `<section>` 시맨틱 태그 정비

#### 5.2 성능 최적화
- Tailwind CDN → 빌드 시 purge된 CSS (또는 필요 유틸리티만 수동 추출)
- JS 모듈 lazy loading (`import()`)
- Chart.js tree-shaking (ESM 빌드 사용)
- 이미지 최적화 (WebP, lazy loading)
- Lighthouse Performance 90+, Accessibility 95+ 목표

### 3.3 Phase 6: 고급 기능 (선택)

#### 6.1 PWA 기본 지원
- `manifest.json`, Service Worker 캐싱
- 오프라인 기본 페이지

#### 6.2 i18n 시스템
- JSON 기반 번역 파일 (`locales/ko.json`, `locales/en.json`)
- `i18n.js` 모듈로 동적 언어 전환

#### 6.3 Vite 번들러 (선택)
- HMR 개발 경험 개선
- CSS/JS 번들링 및 최적화
- 기존 Jinja2 템플릿과 연동

## 4. Key Files

### 수정 대상
| 파일 | 변경 내용 |
|------|-----------|
| `src/web/templates/single_analysis.html` | 컴포넌트 분할 (935줄 → include 매크로) |
| `src/web/templates/v7_mvp.html` | 드래그앤드롭, 진행률 표시 |
| `src/web/templates/history.html` | 테이블 정렬/필터/내보내기 |
| `src/web/templates/stats.html` | 차트 인터랙션 개선 |
| `src/web/static/css/design_system.css` | a11y 색상 대비 보정 |

### 신규 생성
| 파일 | 용도 |
|------|------|
| `src/web/static/js/shared/file_upload.js` | 드래그앤드롭 업로드 컴포넌트 |
| `src/web/static/js/shared/data_table.js` | 정렬/필터링 테이블 컴포넌트 |
| `src/web/static/js/shared/progress.js` | SSE 진행률 표시 |
| `src/web/templates/partials/_sa_*.html` | Single Analysis 분할 파셜 |
| `src/web/static/js/shared/i18n.js` | 국제화 모듈 (Phase 6) |
| `src/web/static/locales/ko.json` | 한국어 번역 (Phase 6) |
| `src/web/static/locales/en.json` | 영어 번역 (Phase 6) |

## 5. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| 기존 워크플로우 중단 | High | 기존 URL 유지, feature flag로 점진적 전환 |
| Single Analysis 분할 시 JS 바인딩 깨짐 | Medium | 분할 전 E2E 테스트 추가 후 리팩토링 |
| SSE 진행률 백엔드 부하 | Low | 타임아웃 + 연결 수 제한 |
| Tailwind CDN 제거 시 스타일 누락 | Medium | 제거 전 사용 중인 클래스 전수 조사 |

## 6. Verification Criteria

- [ ] Lighthouse Performance >= 90
- [ ] Lighthouse Accessibility >= 95
- [ ] 모든 페이지 키보드만으로 조작 가능
- [ ] 드래그앤드롭 업로드 Chrome/Firefox/Edge 동작 확인
- [ ] Single Analysis 분할 후 기존 기능 100% 유지
- [ ] History 테이블 1000건 이상 정렬/필터 성능 1초 미만
- [ ] 다크/라이트 테마 전환 시 깨짐 없음

## 7. References

- `docs/WEB_UI_MODERNIZATION_PLAN.md` - 기존 승인된 현대화 계획 (v1.0)
- `src/web/static/css/design_system.css` - 현재 디자인 시스템
- `src/web/templates/_base_layout.html` - 베이스 레이아웃
- `tests/e2e/` - 기존 Playwright E2E 테스트
