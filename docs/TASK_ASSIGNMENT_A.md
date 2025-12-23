# 작업 할당서 A: 통계 대시보드 구현

**담당자**: 작업자 A
**예상 소요**: 6-8시간
**우선순위**: 높음
**시작일**: 2025-12-19

---

## 🎯 목표

검사 결과 데이터를 시각화하는 통계 대시보드 페이지 구현

---

## 📋 할 일 체크리스트

### 1. 환경 설정 (15분)
- [ ] 프로젝트 클론 및 서버 실행 확인
  ```bash
  cd C:/X/Color_total/Color_meter
  python -m uvicorn src.web.app:app --host 0.0.0.0 --port 8000 --reload
  ```
- [ ] http://127.0.0.1:8000/stats 접속 확인

### 2. 차트 라이브러리 설정 (30분)
- [ ] `src/web/templates/stats.html` 파일 열기
- [ ] Chart.js CDN 추가
  ```html
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
  ```

### 3. Pass Rate 트렌드 차트 (2시간)
**파일**: `src/web/templates/stats.html`

- [ ] API 호출: `GET /api/inspection/history/stats/daily?days=30`
- [ ] Line Chart 구현
  ```javascript
  const ctx = document.getElementById('passRateChart').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates, // ['2025-12-01', '2025-12-02', ...]
      datasets: [{
        label: 'Pass Rate (%)',
        data: passRates, // [92.5, 88.0, ...]
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      }]
    }
  });
  ```

### 4. Judgment 분포 Pie Chart (1.5시간)
- [ ] API 호출: `GET /api/inspection/history/stats/summary?days=30`
- [ ] Pie Chart 구현
  ```javascript
  new Chart(ctx, {
    type: 'pie',
    data: {
      labels: ['OK', 'OK with Warning', 'NG', 'RETAKE'],
      datasets: [{
        data: [100, 10, 5, 8], // judgment_counts에서 추출
        backgroundColor: [
          'rgb(34, 197, 94)',  // green
          'rgb(251, 191, 36)', // yellow
          'rgb(239, 68, 68)',  // red
          'rgb(99, 102, 241)'  // indigo
        ]
      }]
    }
  });
  ```

### 5. 일별 검사 건수 Bar Chart (1.5시간)
- [ ] API 호출: `GET /api/inspection/history/stats/daily?days=30`
- [ ] Stacked Bar Chart 구현
  ```javascript
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: dates,
      datasets: [
        { label: 'OK', data: okCounts, backgroundColor: 'rgb(34, 197, 94)' },
        { label: 'NG', data: ngCounts, backgroundColor: 'rgb(239, 68, 68)' },
        { label: 'RETAKE', data: retakeCounts, backgroundColor: 'rgb(99, 102, 241)' }
      ]
    },
    options: { scales: { x: { stacked: true }, y: { stacked: true } } }
  });
  ```

### 6. SKU별 통계 테이블 (1.5시간)
- [ ] API 호출: `GET /api/inspection/history/stats/by-sku?days=30`
- [ ] 테이블 렌더링
  ```html
  <table>
    <thead>
      <tr>
        <th>SKU</th>
        <th>Total</th>
        <th>Pass Rate</th>
        <th>Avg ΔE</th>
      </tr>
    </thead>
    <tbody id="sku-table">
      <!-- JavaScript로 채움 -->
    </tbody>
  </table>
  ```

### 7. RETAKE 사유 분석 Bar Chart (1.5시간)
- [ ] API 호출: `GET /api/inspection/history/stats/retake-reasons?days=30`
- [ ] Horizontal Bar Chart 구현

### 8. 날짜 필터 UI (1시간)
- [ ] 날짜 범위 선택 드롭다운 (7일, 30일, 90일)
- [ ] Refresh 버튼
- [ ] 선택 시 모든 차트 업데이트

---

## 📦 필요한 파일 및 API

### 기존 파일
- `src/web/templates/stats.html` (이미 존재, 편집 필요)
- `src/web/app.py` (이미 `/stats` 라우트 추가됨)

### 사용할 API 엔드포인트
1. `GET /api/inspection/history/stats/summary?days=30` - 전체 통계
2. `GET /api/inspection/history/stats/daily?days=30` - 일별 통계
3. `GET /api/inspection/history/stats/by-sku?days=30` - SKU별 통계
4. `GET /api/inspection/history/stats/retake-reasons?days=30` - RETAKE 사유

### 참고 문서
- `docs/INSPECTION_HISTORY_GUIDE.md` - API 레퍼런스
- `src/web/templates/history.html` - UI 스타일 참고

---

## 🧪 테스트 방법

### 1. API 테스트
```bash
# 일별 통계
curl "http://127.0.0.1:8000/api/inspection/history/stats/daily?days=30"

# SKU별 통계
curl "http://127.0.0.1:8000/api/inspection/history/stats/by-sku?days=30"

# RETAKE 사유
curl "http://127.0.0.1:8000/api/inspection/history/stats/retake-reasons?days=30"
```

### 2. 브라우저 테스트
- http://127.0.0.1:8000/stats 접속
- 모든 차트가 로드되는지 확인
- 날짜 필터 동작 확인

---

## 📝 완료 기준

- [ ] 5개 차트 모두 정상 표시
- [ ] SKU별 통계 테이블 동작
- [ ] 날짜 필터 정상 작동
- [ ] 모바일 화면에서도 정상 표시
- [ ] 로딩 인디케이터 추가

---

## 🚫 주의사항

1. **DB에 데이터가 없으면** 차트가 비어 보일 수 있음
   - 테스트 데이터 필요 시 요청하세요
2. **Git 브랜치** 사용: `feature/statistics-dashboard`
3. **파일 충돌 방지**: `stats.html`만 수정하세요

---

## 💬 질문/도움

막히는 부분이 있으면 언제든 연락 주세요:
- API 응답 형식 확인 필요 시
- 차트 구현 어려움
- 테스트 데이터 필요

---

**시작 시간**: ___________
**예상 완료**: ___________
**실제 완료**: ___________
