# 작업 할당서 C: 코드 리팩토링

**담당자**: 작업자 C
**예상 소요**: 6-8시간
**우선순위**: 낮음
**시작일**: 2025-12-19

---

## 🎯 목표

코드 품질 향상 및 유지보수성 개선

---

## 📋 할 일 체크리스트

### 1. 환경 설정 (15분)
- [ ] 프로젝트 클론
  ```bash
  cd C:/X/Color_total/Color_meter
  git checkout -b refactor/zone-analyzer-cleanup
  ```
- [ ] 테스트 실행 확인
  ```bash
  pytest tests/test_zone_analyzer_2d.py -v
  # 40개 테스트 모두 PASS 확인
  ```

### 2. find_transition_ranges 리팩토링 (3시간)
**파일**: `src/core/zone_analyzer_2d.py:XXX` (155 lines)

**현재 문제**:
- 155줄의 긴 함수
- 복잡한 중첩 로직
- 주석 없는 매직 넘버

**리팩토링 목표**:
- [ ] 4개 함수로 분할
  ```python
  def find_transition_ranges(...) -> Tuple[float, float]:
      """Main orchestrator"""
      gradient = _compute_delta_e_gradient(radial_profile)
      peaks = _detect_transition_peaks(gradient, smoothing_window)
      ranges = _convert_peaks_to_ranges(peaks)
      return _apply_fallback_if_needed(ranges, radial_profile)

  def _compute_delta_e_gradient(profile: Dict) -> np.ndarray:
      """Compute ΔE gradient along radial direction"""
      # 30줄 이내
      pass

  def _detect_transition_peaks(gradient: np.ndarray, window: int) -> List[int]:
      """Detect peaks in gradient using scipy.signal"""
      # 40줄 이내
      pass

  def _convert_peaks_to_ranges(peaks: List[int]) -> Tuple[float, float]:
      """Convert peak indices to normalized radial ranges"""
      # 30줄 이내
      pass

  def _apply_fallback_if_needed(ranges: Tuple, profile: Dict) -> Tuple[float, float]:
      """Apply fallback logic if detection fails"""
      # 40줄 이내
      pass
  ```

**검증**:
- [ ] `pytest tests/test_zone_analyzer_2d.py::test_find_transition_ranges* -v`
- [ ] 모든 테스트 통과 확인

### 3. auto_define_zone_B 리팩토링 (2시간)
**파일**: `src/core/zone_analyzer_2d.py:XXX` (147 lines)

**리팩토링 목표**:
- [ ] 3개 함수로 분할
  ```python
  def auto_define_zone_B(...) -> Tuple[float, float]:
      """Main orchestrator"""
      r_a_outer, r_c_inner = find_transition_ranges(...)

      if has_sufficient_gap(r_a_outer, r_c_inner):
          return _calculate_zone_b_with_gap(r_a_outer, r_c_inner)
      else:
          return _calculate_zone_b_narrow(r_a_outer, r_c_inner)

  def _calculate_zone_b_with_gap(r_a: float, r_c: float) -> Tuple[float, float]:
      """Calculate Zone B when A-C gap is sufficient"""
      # 40줄 이내
      pass

  def _calculate_zone_b_narrow(r_a: float, r_c: float) -> Tuple[float, float]:
      """Calculate Zone B when A-C gap is narrow"""
      # 40줄 이내
      pass
  ```

**검증**:
- [ ] `pytest tests/test_zone_analyzer_2d.py::test_auto_define* -v`

### 4. compute_zone_results_2d 리팩토링 (2시간)
**파일**: `src/core/zone_analyzer_2d.py:XXX` (145 lines)

**리팩토링 목표**:
- [ ] 3개 함수로 분할
  ```python
  def compute_zone_results_2d(...) -> List[ZoneResult]:
      """Main orchestrator"""
      zone_masks = _create_zone_masks(zones, img_shape)
      zone_colors = _extract_zone_colors(img_lab, zone_masks)
      return _evaluate_zones(zone_colors, target_labs, thresholds)

  def _create_zone_masks(...) -> Dict[str, np.ndarray]:
      """Create binary masks for each zone"""
      # 40줄 이내
      pass

  def _extract_zone_colors(...) -> Dict[str, Tuple[float, float, float]]:
      """Extract average LAB color from each zone"""
      # 40줄 이내
      pass

  def _evaluate_zones(...) -> List[ZoneResult]:
      """Compare measured vs target colors"""
      # 40줄 이내
      pass
  ```

**검증**:
- [ ] `pytest tests/test_zone_analyzer_2d.py::test_compute_zone* -v`

### 5. 전체 테스트 (1시간)
- [ ] 전체 테스트 실행
  ```bash
  pytest tests/test_zone_analyzer_2d.py -v
  # 40개 테스트 모두 PASS 확인
  ```

- [ ] 통합 테스트 실행
  ```bash
  pytest tests/ -v --tb=short
  ```

- [ ] 코드 품질 체크
  ```bash
  # 라인 길이 확인
  flake8 src/core/zone_analyzer_2d.py --max-line-length=120

  # Type hints 확인
  mypy src/core/zone_analyzer_2d.py
  ```

### 6. 문서화 (30분)
- [ ] 각 함수에 docstring 추가
  ```python
  def _compute_delta_e_gradient(profile: Dict) -> np.ndarray:
      """
      Compute ΔE gradient along radial direction.

      Args:
          profile: Radial profile dict with 'r', 'L', 'a', 'b' keys

      Returns:
          1D array of ΔE values between adjacent points

      Notes:
          Uses CIE76 formula: sqrt((ΔL)² + (Δa)² + (Δb)²)
      """
      pass
  ```

---

## 📦 수정할 파일

**주요 파일**:
- `src/core/zone_analyzer_2d.py` - 리팩토링 대상

**테스트 파일**:
- `tests/test_zone_analyzer_2d.py` - 검증용 (수정 금지)

---

## 🧪 테스트 방법

### 단계별 검증
```bash
# Step 1: 리팩토링 전 테스트 (baseline)
pytest tests/test_zone_analyzer_2d.py -v > before.txt

# Step 2: 리팩토링 후 테스트
pytest tests/test_zone_analyzer_2d.py -v > after.txt

# Step 3: 비교
diff before.txt after.txt
# 차이가 없어야 함 (모두 PASS)
```

### 성능 비교
```bash
# 리팩토링 전
pytest tests/test_zone_analyzer_2d.py --durations=10

# 리팩토링 후
pytest tests/test_zone_analyzer_2d.py --durations=10

# 성능 저하 없는지 확인
```

---

## 📝 완료 기준

- [ ] find_transition_ranges 함수 < 50줄
- [ ] auto_define_zone_B 함수 < 50줄
- [ ] compute_zone_results_2d 함수 < 50줄
- [ ] 40개 테스트 모두 통과
- [ ] 성능 저하 없음 (±5% 이내)
- [ ] 모든 함수에 docstring 추가
- [ ] Type hints 완벽

---

## 🎯 리팩토링 원칙

1. **단일 책임 원칙 (SRP)**
   - 각 함수는 하나의 일만 수행
   - 50줄 이내로 제한

2. **명확한 함수명**
   - `_compute_*`: 계산 수행
   - `_detect_*`: 패턴 검출
   - `_convert_*`: 데이터 변환
   - `_apply_*`: 로직 적용

3. **Private 함수 사용**
   - 내부 헬퍼는 `_`로 시작
   - Public API는 변경 금지

4. **테스트 유지**
   - 기존 테스트 모두 통과
   - 새 테스트 추가 불필요 (리팩토링만)

5. **성능 유지**
   - 알고리즘 변경 금지
   - 함수 호출 오버헤드 최소화

---

## 🚫 주의사항

1. **알고리즘 변경 금지**
   - 로직은 동일하게 유지
   - 코드 구조만 개선

2. **테스트 수정 금지**
   - `test_zone_analyzer_2d.py` 수정 금지
   - 모든 기존 테스트 통과 필수

3. **Public API 유지**
   - `analyze_lens_zones_2d()` 시그니처 변경 금지
   - 외부에서 호출하는 함수 보호

4. **Git 커밋 전략**
   - 각 함수 리팩토링마다 커밋
   - 커밋 메시지: `refactor: Extract _compute_delta_e_gradient from find_transition_ranges`

---

## 💬 질문/도움

- 함수 분할 전략 조언
- 테스트 실패 시 디버깅 지원
- 성능 최적화 제안

---

## 📚 참고 자료

**코드 위치**:
- `src/core/zone_analyzer_2d.py:XXX` - find_transition_ranges
- Line 수 확인: `wc -l src/core/zone_analyzer_2d.py`

**테스트 케이스**:
- `tests/test_zone_analyzer_2d.py` - 40개 테스트

**문서**:
- `docs/planning/IMPROVEMENT_PLAN.md` - Task 3.4 참고

---

**시작 시간**: ___________
**예상 완료**: ___________
**실제 완료**: ___________
