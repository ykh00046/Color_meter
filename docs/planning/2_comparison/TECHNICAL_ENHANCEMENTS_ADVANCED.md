# STD-Based QC System: Advanced Technical Enhancements

**작성일**: 2025-12-17
**상태**: 기술적 보완 계획 (Technical Enhancement Specification)
**목적**: STD 기반 품질 관리 시스템의 핵심 알고리즘 및 아키텍처 고도화

---

## 📋 Executive Summary

기존 `STD_BASED_QC_SYSTEM_PLAN.md`는 **기술적으로 충분히 가능한 상태**입니다. 본 문서는 해당 계획을 한 단계 더 발전시키기 위한 **7가지 핵심 기술 보강 사항**을 상세히 정의합니다.

### 핵심 개선 영역
1. **STD Statistical Model**: 단일 이미지 → 통계적 분포 모델 (mean ± σ)
2. **Elastic Alignment**: 구조 비교 전 정렬 단계 (Anchor Zone 기반)
3. **Worst-Case Color Metrics**: 평균 ΔE 외 극값 지표 도입
4. **Ink-Aware Comparison**: Zone과 Ink 이중 관점 비교 레이어
5. **Explainability Layer**: NG 원인 자동 설명 생성
6. **Performance & Stability**: 캐싱, Fail-Safe 설계, Confidence Score 분리
7. **Phenomenological Classification**: 결함 유형 분류 체계

---

## 1. STD Statistical Model (통계적 표준 모델)

### 1.1 현재 설계의 한계
- **문제**: STD를 "좋은 한 장의 이미지"로 취급
- **리스크**:
  - STD 자체의 자연 편차(natural variance)를 무시
  - 양산 샘플 비교 시 False Negative 발생 (실제로는 OK인데 NG로 판정)
  - 상한/중간/하한 설정 시 근거 부족

### 1.2 개선 방안: 통계적 분포 모델

#### 데이터 수집
```
STD Samples: n = 5~10장 (동일 SKU, 동일 생산 배치)
각 샘플에 대해:
  - Radial Profile (500 points)
  - Zone Results (A, B, C zones)
  - Ink Colors (Lab 색상)
  - Zone Boundaries (transitions)
```

#### 통계 모델 구성
```python
class STDStatisticalModel:
    def __init__(self, sku_code: str, samples: List[AnalysisResult]):
        self.sku_code = sku_code
        self.n_samples = len(samples)

        # 1. Radial Profile 통계
        self.profile_mean = np.mean([s.radial_profile for s in samples], axis=0)
        self.profile_std = np.std([s.radial_profile for s in samples], axis=0)
        self.profile_covariance = np.cov([s.radial_profile for s in samples])

        # 2. Zone Color 통계 (각 Zone별)
        for zone_name in ['A', 'B', 'C']:
            colors = [s.zones[zone_name].color_lab for s in samples]
            self.zones[zone_name] = {
                'mean_L': np.mean([c[0] for c in colors]),
                'std_L': np.std([c[0] for c in colors]),
                'mean_a': np.mean([c[1] for c in colors]),
                'std_a': np.std([c[1] for c in colors]),
                'mean_b': np.mean([c[2] for c in colors]),
                'std_b': np.std([c[2] for c in colors]),
                'covariance': np.cov(colors)
            }

        # 3. Zone Boundary 통계
        for transition in ['inner_to_A', 'A_to_B', 'B_to_C']:
            positions = [s.boundaries[transition] for s in samples]
            self.boundaries[transition] = {
                'mean': np.mean(positions),
                'std': np.std(positions),
                'percentile_5': np.percentile(positions, 5),
                'percentile_95': np.percentile(positions, 95)
            }

        # 4. Ink Count 통계 (Image-Based)
        ink_counts = [s.ink_analysis.n_inks for s in samples]
        self.ink_stats = {
            'mode': stats.mode(ink_counts)[0],  # 최빈값
            'allowed_values': set(ink_counts)  # 허용 범위
        }
```

#### 판정 기준 자동 도출
```python
def derive_acceptance_criteria(self, confidence_level=0.95):
    """STD 분포로부터 합격 기준 자동 도출"""
    z_score = stats.norm.ppf((1 + confidence_level) / 2)  # 95% → z=1.96

    for zone_name in ['A', 'B', 'C']:
        zone_stats = self.zones[zone_name]

        # ΔE 허용 범위 = 3 × std (99.7% 포함)
        self.acceptance_criteria[zone_name] = {
            'max_delta_E': 3.0 * np.sqrt(
                zone_stats['std_L']**2 +
                zone_stats['std_a']**2 +
                zone_stats['std_b']**2
            ),
            'L_range': (
                zone_stats['mean_L'] - z_score * zone_stats['std_L'],
                zone_stats['mean_L'] + z_score * zone_stats['std_L']
            )
        }

    # Boundary 허용 오차 = 2 × std
    for transition, stats_dict in self.boundaries.items():
        self.acceptance_criteria[transition] = {
            'tolerance': 2.0 * stats_dict['std'],
            'range': (stats_dict['percentile_5'], stats_dict['percentile_95'])
        }
```

### 1.3 DB Schema 변경

#### 기존 (Single STD)
```sql
std_profiles (
    id, sku_code, version, profile_data JSON, created_at
)
```

#### 개선 (Statistical STD)
```sql
-- STD 메타데이터
std_models (
    id SERIAL PRIMARY KEY,
    sku_code VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    n_samples INTEGER NOT NULL,  -- 5~10
    created_at TIMESTAMP,
    approved_by VARCHAR(100),
    approved_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- STD 구성 샘플들 (개별 이미지)
std_samples (
    id SERIAL PRIMARY KEY,
    std_model_id INTEGER REFERENCES std_models(id),
    sample_index INTEGER,  -- 1~n
    image_path VARCHAR(500),
    analysis_result JSONB,  -- 전체 분석 결과
    created_at TIMESTAMP
);

-- STD 통계 요약 (쿼리 성능 최적화)
std_statistics (
    id SERIAL PRIMARY KEY,
    std_model_id INTEGER REFERENCES std_models(id),
    zone_name VARCHAR(10),  -- 'A', 'B', 'C', 'profile', 'boundary_AB'

    -- 색상 통계
    mean_L FLOAT,
    std_L FLOAT,
    mean_a FLOAT,
    std_a FLOAT,
    mean_b FLOAT,
    std_b FLOAT,

    -- 구조 통계
    mean_position FLOAT,  -- Boundary 위치
    std_position FLOAT,
    percentile_5 FLOAT,
    percentile_95 FLOAT,

    -- 상세 통계 (covariance matrix 등)
    detailed_stats JSONB
);

CREATE INDEX idx_std_stats_model ON std_statistics(std_model_id, zone_name);
```

---

## 2. Elastic Alignment (탄성 정렬)

### 2.1 문제 정의
- **구조적 변동**: 내측 링 위치 변화 (±5%), 외측 링 이동, 중심 치우침
- **현재 방식**: 고정 위치 비교 → 패턴 shift로 인한 False Negative
- **예시**: Zone B가 2mm 외곽으로 이동했지만 구조 자체는 동일한 경우

### 2.2 Anchor Zone 기반 정렬 전략

#### Phase 1: Anchor Zone 식별
```python
def identify_anchor_zone(test_profile, std_profile):
    """가장 안정적인 Zone을 Anchor로 선택"""
    # C Zone (최외곽)을 기본 Anchor로 사용
    # 이유: 렌즈 가장자리는 물리적으로 가장 안정적
    anchor_zone = 'C'

    # Anchor Zone의 중심 위치 추출
    test_anchor_center = test_profile.zones[anchor_zone].center_position
    std_anchor_center = std_profile.zones[anchor_zone].center_position

    # 위치 차이 계산
    shift = test_anchor_center - std_anchor_center
    return shift, anchor_zone
```

#### Phase 2: Circular Shift Alignment
```python
def align_radial_profiles(test_profile, std_profile, shift):
    """Radial Profile을 shift만큼 순환 이동"""
    # shift > 0: 외곽 방향 이동
    # shift < 0: 내측 방향 이동

    shift_pixels = int(shift / pixel_size)  # mm → pixel

    if shift_pixels > 0:
        # 우측 순환 이동
        aligned_test = np.roll(test_profile, shift_pixels)
    else:
        # 좌측 순환 이동
        aligned_test = np.roll(test_profile, shift_pixels)

    return aligned_test
```

#### Phase 3: DTW Pre-Alignment (Fine-Tuning)
```python
from dtaidistance import dtw

def fine_tune_alignment(test_profile, std_profile, max_warp=10):
    """DTW로 미세 정렬 (±10 pixel 범위 내)"""
    # DTW path 계산
    path = dtw.warping_path(test_profile, std_profile)

    # Warp 제약: 최대 ±10 pixel 이내만 허용
    # (과도한 warping 방지)
    constrained_path = constrain_warping(path, max_warp=10)

    # Path 적용하여 정렬된 프로파일 생성
    aligned_test = apply_warping_path(test_profile, constrained_path)

    return aligned_test, constrained_path
```

#### Phase 4: Zone Boundary Re-Detection
```python
def realign_zone_boundaries(aligned_profile, std_boundaries):
    """정렬된 프로파일에서 Zone 경계 재탐지"""
    # STD 경계 위치 주변 ±20 pixel 범위에서 탐색
    detected_boundaries = {}

    for boundary_name, std_position in std_boundaries.items():
        search_window = (std_position - 20, std_position + 20)

        # 경계 탐지 (Gradient 기반)
        gradient = np.gradient(aligned_profile)
        peak_idx = np.argmax(np.abs(gradient[search_window[0]:search_window[1]]))

        detected_boundaries[boundary_name] = search_window[0] + peak_idx

    return detected_boundaries
```

### 2.3 Alignment Quality Metrics
```python
def evaluate_alignment_quality(aligned_test, std_profile):
    """정렬 품질 평가"""
    # 1. Correlation Coefficient
    corr = np.corrcoef(aligned_test, std_profile)[0, 1]

    # 2. RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((aligned_test - std_profile)**2))

    # 3. Structural Similarity Index (SSIM)
    from skimage.metrics import structural_similarity
    ssim = structural_similarity(aligned_test, std_profile)

    alignment_quality = {
        'correlation': corr,  # 0.95 이상 양호
        'rmse': rmse,  # 5.0 이하 양호
        'ssim': ssim,  # 0.90 이상 양호
        'is_good': corr > 0.95 and rmse < 5.0 and ssim > 0.90
    }

    return alignment_quality
```

---

## 3. Worst-Case Color Metrics (극값 색상 지표)

### 3.1 평균 ΔE의 한계
- **문제**: 평균 ΔE = 2.5 (OK)인데 시각적으로 NG
- **원인**: 국소적 결함 (5% 영역이 ΔE=15.0)이 평균에 묻힘

### 3.2 보완 지표 체계

#### 3.2.1 Percentile-Based ΔE
```python
def calculate_percentile_delta_e(test_zone, std_zone):
    """픽셀별 ΔE 분포의 백분위수 계산"""
    from colormath.color_diff import delta_e_cie2000

    # 각 픽셀별 ΔE 계산
    pixel_delta_e = []
    for test_lab, std_lab in zip(test_zone.pixels_lab, std_zone.pixels_lab):
        de = delta_e_cie2000(test_lab, std_lab)
        pixel_delta_e.append(de)

    pixel_delta_e = np.array(pixel_delta_e)

    return {
        'mean': np.mean(pixel_delta_e),
        'median': np.median(pixel_delta_e),
        'p75': np.percentile(pixel_delta_e, 75),
        'p90': np.percentile(pixel_delta_e, 90),
        'p95': np.percentile(pixel_delta_e, 95),  # ⭐ 핵심 지표
        'p99': np.percentile(pixel_delta_e, 99),
        'max': np.max(pixel_delta_e)
    }
```

#### 3.2.2 Spatial Hotspot Detection
```python
def detect_color_hotspots(test_image_lab, std_image_lab, threshold=10.0, min_area=100):
    """ΔE 이상 영역(Hotspot) 탐지"""
    from skimage.measure import label, regionprops

    # 픽셀별 ΔE 맵 생성
    delta_e_map = np.zeros(test_image_lab.shape[:2])
    for i in range(test_image_lab.shape[0]):
        for j in range(test_image_lab.shape[1]):
            delta_e_map[i, j] = delta_e_cie2000(
                test_image_lab[i, j],
                std_image_lab[i, j]
            )

    # Hotspot 마스크 (ΔE > 10.0)
    hotspot_mask = delta_e_map > threshold

    # Connected Component 분석
    labeled_mask = label(hotspot_mask)
    regions = regionprops(labeled_mask)

    hotspots = []
    for region in regions:
        if region.area >= min_area:  # 최소 100 픽셀 이상
            hotspots.append({
                'area': region.area,
                'centroid': region.centroid,
                'bbox': region.bbox,
                'mean_delta_e': np.mean(delta_e_map[labeled_mask == region.label]),
                'max_delta_e': np.max(delta_e_map[labeled_mask == region.label])
            })

    # Severity 순으로 정렬
    hotspots.sort(key=lambda x: x['mean_delta_e'] * x['area'], reverse=True)

    return hotspots, delta_e_map
```

#### 3.2.3 Spatial Clustering (DBSCAN)
```python
from sklearn.cluster import DBSCAN

def cluster_abnormal_pixels(delta_e_map, threshold=8.0):
    """ΔE 이상 픽셀의 공간적 군집 분석"""
    # 이상 픽셀 좌표 추출
    abnormal_coords = np.argwhere(delta_e_map > threshold)

    if len(abnormal_coords) < 10:
        return []  # 이상 픽셀 너무 적음

    # DBSCAN 클러스터링 (eps=5 픽셀)
    clustering = DBSCAN(eps=5, min_samples=10).fit(abnormal_coords)

    clusters = []
    for cluster_id in set(clustering.labels_):
        if cluster_id == -1:  # Noise
            continue

        cluster_mask = clustering.labels_ == cluster_id
        cluster_coords = abnormal_coords[cluster_mask]

        clusters.append({
            'n_pixels': len(cluster_coords),
            'center': np.mean(cluster_coords, axis=0),
            'radius': np.std(cluster_coords, axis=0),
            'mean_delta_e': np.mean(delta_e_map[cluster_coords[:, 0], cluster_coords[:, 1]])
        })

    return clusters
```

### 3.3 판정 기준 통합
```python
def evaluate_color_quality(test_zone, std_zone, criteria):
    """색상 품질 종합 평가"""
    percentiles = calculate_percentile_delta_e(test_zone, std_zone)
    hotspots, delta_e_map = detect_color_hotspots(test_zone.image, std_zone.image)
    clusters = cluster_abnormal_pixels(delta_e_map)

    # 판정 로직
    is_pass = True
    fail_reasons = []

    # 1. 평균 ΔE 체크
    if percentiles['mean'] > criteria['max_mean_delta_e']:
        is_pass = False
        fail_reasons.append(f"평균 ΔE={percentiles['mean']:.2f} (기준: {criteria['max_mean_delta_e']})")

    # 2. 95 백분위수 체크 (⭐ 핵심)
    if percentiles['p95'] > criteria['max_p95_delta_e']:
        is_pass = False
        fail_reasons.append(f"95% ΔE={percentiles['p95']:.2f} (기준: {criteria['max_p95_delta_e']})")

    # 3. Hotspot 체크
    if len(hotspots) > 0:
        largest_hotspot = hotspots[0]
        if largest_hotspot['area'] > criteria['max_hotspot_area']:
            is_pass = False
            fail_reasons.append(
                f"Hotspot 발견: 면적={largest_hotspot['area']}px, "
                f"평균 ΔE={largest_hotspot['mean_delta_e']:.2f}"
            )

    # 4. 클러스터 체크
    if len(clusters) > criteria['max_allowed_clusters']:
        is_pass = False
        fail_reasons.append(f"이상 영역 {len(clusters)}개 발견 (기준: {criteria['max_allowed_clusters']})")

    return {
        'is_pass': is_pass,
        'fail_reasons': fail_reasons,
        'percentiles': percentiles,
        'hotspots': hotspots,
        'clusters': clusters,
        'delta_e_map': delta_e_map  # 시각화용
    }
```

---

## 4. Ink-Aware Comparison Layer (잉크 인식 비교)

### 4.1 문제 정의
- **Zone ≠ Ink**: Zone B에 잉크 2개 (Blue, Sky Blue) 존재
- **현재 한계**: Zone 평균 색상만 비교 → 잉크 혼합/누락 감지 불가

### 4.2 Dual Scoring Architecture

#### 4.2.1 Zone-Based Score (구조적 유사도)
```python
def calculate_zone_based_score(test_zones, std_zones):
    """Zone 단위 구조 비교"""
    zone_scores = {}

    for zone_name in ['A', 'B', 'C']:
        test_zone = test_zones[zone_name]
        std_zone = std_zones[zone_name]

        # 1. Zone 색상 유사도 (평균 Lab)
        color_score = 100 - delta_e_cie2000(
            test_zone.mean_lab,
            std_zone.mean_lab
        ) * 10  # ΔE=1 → -10점

        # 2. Zone 경계 위치 유사도
        boundary_diff = abs(test_zone.boundary - std_zone.boundary)
        boundary_score = 100 - boundary_diff * 20  # 1px 차이 → -20점

        # 3. Zone 면적 비율 유사도
        area_ratio_diff = abs(test_zone.area_ratio - std_zone.area_ratio)
        area_score = 100 - area_ratio_diff * 100  # 1% 차이 → -100점

        zone_scores[zone_name] = {
            'color': max(0, color_score),
            'boundary': max(0, boundary_score),
            'area': max(0, area_score),
            'total': (color_score + boundary_score + area_score) / 3
        }

    overall = np.mean([z['total'] for z in zone_scores.values()])
    return overall, zone_scores
```

#### 4.2.2 Ink-Based Score (잉크 유사도)
```python
def calculate_ink_based_score(test_inks, std_inks):
    """Ink 단위 색상 비교"""
    # 1. Ink 개수 일치 확인
    if test_inks.n_inks != std_inks.n_inks:
        count_penalty = abs(test_inks.n_inks - std_inks.n_inks) * 30
        # 1개 차이 → -30점
    else:
        count_penalty = 0

    # 2. Ink 색상 매칭 (Hungarian Algorithm)
    from scipy.optimize import linear_sum_assignment

    # Cost Matrix: 각 Ink 쌍의 ΔE
    cost_matrix = np.zeros((test_inks.n_inks, std_inks.n_inks))
    for i, test_ink in enumerate(test_inks.colors_lab):
        for j, std_ink in enumerate(std_inks.colors_lab):
            cost_matrix[i, j] = delta_e_cie2000(test_ink, std_ink)

    # 최적 매칭
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_delta_e = cost_matrix[row_ind, col_ind]

    # 3. 색상 점수 계산
    color_scores = [100 - de * 10 for de in matched_delta_e]
    avg_color_score = np.mean(color_scores)

    # 4. 종합 점수
    final_score = max(0, avg_color_score - count_penalty)

    return final_score, {
        'count_penalty': count_penalty,
        'matched_delta_e': matched_delta_e.tolist(),
        'color_scores': color_scores,
        'matching': list(zip(row_ind, col_ind))
    }
```

#### 4.2.3 Combined Scoring
```python
def calculate_combined_score(test_result, std_result, weights=None):
    """Zone-Based + Ink-Based 통합 점수"""
    if weights is None:
        weights = {'zone': 0.5, 'ink': 0.5}  # 기본 50:50

    # Zone-Based Score
    zone_score, zone_details = calculate_zone_based_score(
        test_result.zones,
        std_result.zones
    )

    # Ink-Based Score
    ink_score, ink_details = calculate_ink_based_score(
        test_result.ink_analysis,
        std_result.ink_analysis
    )

    # 가중 평균
    combined_score = (
        zone_score * weights['zone'] +
        ink_score * weights['ink']
    )

    return {
        'total_score': combined_score,
        'zone_score': zone_score,
        'ink_score': ink_score,
        'zone_details': zone_details,
        'ink_details': ink_details,
        'weights': weights
    }
```

### 4.3 Diagnosis Matrix
```python
def diagnose_failure_type(zone_score, ink_score, threshold=70):
    """실패 유형 진단"""
    if zone_score >= threshold and ink_score >= threshold:
        return 'PASS', 'OK'

    if zone_score < threshold and ink_score < threshold:
        return 'FAIL', '구조 및 잉크 모두 이상'

    if zone_score < threshold and ink_score >= threshold:
        return 'FAIL', '구조 이상 (Zone 경계/면적 문제), 잉크 색상은 OK'

    if zone_score >= threshold and ink_score < threshold:
        return 'FAIL', '구조는 OK, 잉크 색상/개수 이상'

    return 'UNKNOWN', '판정 불가'
```

---

## 5. Explainability Layer (설명 가능성)

### 5.1 목표
- **What**: "FAIL 점수 65점" → "왜 실패했는가?"
- **Why**: 현장 작업자가 조치 가능하도록 구체적 근거 제시
- **How**: Top 3 실패 원인 자동 추출

### 5.2 Failure Reason Extraction

#### 5.2.1 원인 후보 수집
```python
class FailureReasonExtractor:
    def __init__(self):
        self.reasons = []

    def extract_all_reasons(self, test_result, std_result, comparison_result):
        """모든 실패 원인 후보 추출"""
        self.reasons = []

        # 1. Zone 경계 이탈
        for zone_name, zone_detail in comparison_result['zone_details'].items():
            if zone_detail['boundary'] < 70:
                boundary_diff = test_result.zones[zone_name].boundary - std_result.zones[zone_name].boundary
                self.reasons.append({
                    'category': 'BOUNDARY',
                    'severity': 100 - zone_detail['boundary'],
                    'zone': zone_name,
                    'message': f"Zone {zone_name} 경계 위치 {boundary_diff:+.1f}px 이탈 (기준 대비 {abs(boundary_diff)/std_result.zones[zone_name].boundary*100:.1f}%)",
                    'action': f"{'외곽' if boundary_diff > 0 else '내측'} 방향 조정 필요"
                })

        # 2. Zone 색상 차이
        for zone_name, zone_detail in comparison_result['zone_details'].items():
            if zone_detail['color'] < 70:
                delta_e = delta_e_cie2000(
                    test_result.zones[zone_name].mean_lab,
                    std_result.zones[zone_name].mean_lab
                )
                self.reasons.append({
                    'category': 'ZONE_COLOR',
                    'severity': delta_e * 10,
                    'zone': zone_name,
                    'message': f"Zone {zone_name} 색상 ΔE={delta_e:.2f} (기준: <3.0)",
                    'color_diff': self._format_color_diff(test_result.zones[zone_name].mean_lab, std_result.zones[zone_name].mean_lab),
                    'action': "색상 농도 조정 필요"
                })

        # 3. Ink 개수 불일치
        if comparison_result['ink_details']['count_penalty'] > 0:
            test_n = test_result.ink_analysis.n_inks
            std_n = std_result.ink_analysis.n_inks
            self.reasons.append({
                'category': 'INK_COUNT',
                'severity': comparison_result['ink_details']['count_penalty'],
                'message': f"잉크 개수 불일치: 검출={test_n}개, 기준={std_n}개",
                'action': f"{'잉크 추가' if test_n < std_n else '잉크 제거/혼합 확인'} 필요"
            })

        # 4. Ink 색상 차이
        for idx, (test_idx, std_idx) in enumerate(comparison_result['ink_details']['matching']):
            delta_e = comparison_result['ink_details']['matched_delta_e'][idx]
            if delta_e > 3.0:
                self.reasons.append({
                    'category': 'INK_COLOR',
                    'severity': delta_e * 10,
                    'ink_index': idx,
                    'message': f"Ink #{idx+1} 색상 ΔE={delta_e:.2f} (기준: <3.0)",
                    'color_diff': self._format_color_diff(
                        test_result.ink_analysis.colors_lab[test_idx],
                        std_result.ink_analysis.colors_lab[std_idx]
                    ),
                    'action': "해당 잉크 농도 조정"
                })

        # 5. Hotspot (국소 결함)
        if 'hotspots' in comparison_result and len(comparison_result['hotspots']) > 0:
            for hotspot in comparison_result['hotspots'][:2]:  # 최대 2개
                self.reasons.append({
                    'category': 'HOTSPOT',
                    'severity': hotspot['mean_delta_e'] * hotspot['area'] / 100,
                    'message': f"국소 색상 이상: 위치=({hotspot['centroid'][0]:.0f}, {hotspot['centroid'][1]:.0f}), 면적={hotspot['area']}px, 평균 ΔE={hotspot['mean_delta_e']:.2f}",
                    'action': "해당 영역 육안 점검 및 재작업"
                })

        # 6. 95% ΔE 초과
        if 'percentiles' in comparison_result:
            p95 = comparison_result['percentiles']['p95']
            if p95 > 8.0:
                self.reasons.append({
                    'category': 'P95_DELTA_E',
                    'severity': p95 * 5,
                    'message': f"95% ΔE={p95:.2f} (기준: <8.0) - 상위 5% 픽셀 색상 이상",
                    'action': "전반적 색상 품질 개선 필요"
                })

        return self.reasons

    def _format_color_diff(self, test_lab, std_lab):
        """Lab 차이를 사람이 읽기 쉬운 형태로"""
        diff_L = test_lab[0] - std_lab[0]
        diff_a = test_lab[1] - std_lab[1]
        diff_b = test_lab[2] - std_lab[2]

        parts = []
        if abs(diff_L) > 2.0:
            parts.append(f"{'밝기' if diff_L > 0 else '어두움'} {abs(diff_L):.1f}")
        if abs(diff_a) > 2.0:
            parts.append(f"{'빨강' if diff_a > 0 else '초록'} {abs(diff_a):.1f}")
        if abs(diff_b) > 2.0:
            parts.append(f"{'노랑' if diff_b > 0 else '파랑'} {abs(diff_b):.1f}")

        return ', '.join(parts) if parts else '미세 차이'
```

#### 5.2.2 Top 3 Ranking
```python
def get_top_failure_reasons(self, n=3):
    """Severity 기준 Top N 추출"""
    sorted_reasons = sorted(self.reasons, key=lambda x: x['severity'], reverse=True)
    return sorted_reasons[:n]
```

### 5.3 UI/Report 통합
```python
def generate_failure_report(test_result, std_result, comparison_result):
    """실패 보고서 자동 생성"""
    extractor = FailureReasonExtractor()
    all_reasons = extractor.extract_all_reasons(test_result, std_result, comparison_result)
    top_reasons = extractor.get_top_failure_reasons(n=3)

    report = {
        'judgment': 'FAIL',
        'total_score': comparison_result['total_score'],
        'top_3_reasons': [
            {
                'rank': idx + 1,
                'category': reason['category'],
                'message': reason['message'],
                'action': reason['action'],
                'severity': reason['severity']
            }
            for idx, reason in enumerate(top_reasons)
        ],
        'all_reasons_count': len(all_reasons),
        'detailed_reasons': all_reasons
    }

    return report
```

### 5.4 Output Example
```json
{
  "judgment": "FAIL",
  "total_score": 65.3,
  "top_3_reasons": [
    {
      "rank": 1,
      "category": "ZONE_COLOR",
      "message": "Zone B 색상 ΔE=8.5 (기준: <3.0)",
      "color_diff": "빨강 4.2, 노랑 3.8",
      "action": "색상 농도 조정 필요",
      "severity": 85
    },
    {
      "rank": 2,
      "category": "BOUNDARY",
      "message": "Zone A 경계 위치 +12.3px 이탈 (기준 대비 4.8%)",
      "action": "외곽 방향 조정 필요",
      "severity": 61
    },
    {
      "rank": 3,
      "category": "INK_COUNT",
      "message": "잉크 개수 불일치: 검출=2개, 기준=3개",
      "action": "잉크 추가 필요",
      "severity": 30
    }
  ]
}
```

---

## 6. Performance & Stability (성능 및 안정성)

### 6.1 Profile Caching
```python
from functools import lru_cache
import hashlib

class STDProfileCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}

    def get_cache_key(self, sku_code, version):
        """캐시 키 생성"""
        return f"{sku_code}_{version}"

    @lru_cache(maxsize=100)
    def load_std_profile(self, sku_code, version):
        """STD 프로파일 로드 (캐싱)"""
        key = self.get_cache_key(sku_code, version)

        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]

        # DB에서 로드
        profile = self._load_from_db(sku_code, version)

        # 캐시 저장
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        self.cache[key] = profile
        self.access_count[key] = 1

        return profile

    def _evict_lru(self):
        """LRU 제거"""
        lru_key = min(self.access_count, key=self.access_count.get)
        del self.cache[lru_key]
        del self.access_count[lru_key]
```

### 6.2 Profile Normalization
```python
def normalize_radial_profile(profile, method='zscore'):
    """프로파일 정규화"""
    if method == 'zscore':
        # Z-score normalization
        mean = np.mean(profile)
        std = np.std(profile)
        return (profile - mean) / std

    elif method == 'minmax':
        # Min-Max normalization
        min_val = np.min(profile)
        max_val = np.max(profile)
        return (profile - min_val) / (max_val - min_val)

    elif method == 'robust':
        # Robust normalization (IQR)
        q1 = np.percentile(profile, 25)
        q3 = np.percentile(profile, 75)
        iqr = q3 - q1
        median = np.median(profile)
        return (profile - median) / iqr
```

### 6.3 Fail-Safe Design
```python
class RobustComparisonEngine:
    def compare_with_failsafe(self, test_result, std_result):
        """실패 방지 설계"""
        try:
            # 1. 입력 검증
            self._validate_inputs(test_result, std_result)

            # 2. 비교 수행
            comparison = self._perform_comparison(test_result, std_result)

            # 3. 결과 검증
            self._validate_results(comparison)

            return comparison

        except LensDetectionFailure as e:
            # 렌즈 미검출 → RETAKE
            return {
                'status': 'RETAKE',
                'reason': '렌즈 검출 실패',
                'detail': str(e),
                'action': '이미지 재촬영 필요'
            }

        except InsufficientDataError as e:
            # 데이터 부족 → RETAKE
            return {
                'status': 'RETAKE',
                'reason': '분석 데이터 부족',
                'detail': str(e),
                'action': '촬영 조건 개선 후 재촬영'
            }

        except Exception as e:
            # 알 수 없는 오류 → 수동 점검
            logger.error(f"Unexpected error: {e}")
            return {
                'status': 'MANUAL_REVIEW',
                'reason': '시스템 오류',
                'detail': str(e),
                'action': '담당자 확인 필요'
            }

    def _validate_inputs(self, test_result, std_result):
        """입력 데이터 검증"""
        # 렌즈 검출 확인
        if not test_result.lens_detected:
            raise LensDetectionFailure("Test sample: Lens not detected")

        # Zone 개수 확인
        if len(test_result.zones) != len(std_result.zones):
            raise InsufficientDataError(f"Zone count mismatch: {len(test_result.zones)} vs {len(std_result.zones)}")

        # Radial Profile 길이 확인
        if len(test_result.radial_profile) < 100:
            raise InsufficientDataError("Radial profile too short")
```

### 6.4 Confidence Score
```python
def calculate_confidence_score(test_result, comparison_result):
    """신뢰도 점수 계산 (판정과 분리)"""
    confidence_factors = []

    # 1. Lens Detection Quality
    if test_result.lens_detection_score > 0.95:
        confidence_factors.append(100)
    elif test_result.lens_detection_score > 0.80:
        confidence_factors.append(70)
    else:
        confidence_factors.append(30)

    # 2. Zone Segmentation Quality
    zone_confidence = np.mean([z.segmentation_confidence for z in test_result.zones.values()])
    confidence_factors.append(zone_confidence * 100)

    # 3. Alignment Quality
    if 'alignment_quality' in comparison_result:
        alignment_conf = comparison_result['alignment_quality']['correlation'] * 100
        confidence_factors.append(alignment_conf)

    # 4. Data Completeness
    completeness = (
        (1.0 if test_result.ink_analysis else 0.5) * 50 +
        (1.0 if len(test_result.zones) == 3 else 0.5) * 50
    )
    confidence_factors.append(completeness)

    overall_confidence = np.mean(confidence_factors)

    return {
        'confidence': overall_confidence,
        'factors': confidence_factors,
        'recommendation': 'HIGH' if overall_confidence > 80 else 'MEDIUM' if overall_confidence > 60 else 'RETAKE'
    }
```

---

## 7. Phenomenological Classification (현상학적 분류)

### 7.1 결함 유형 분류 체계

#### 7.1.1 Taxonomy
```python
DEFECT_TAXONOMY = {
    'COLOR_DEFECTS': {
        'UNDERDOSE': '색상 농도 부족형',  # ΔE > 0, L↑
        'OVERDOSE': '색상 농도 과다형',   # ΔE > 0, L↓
        'HUE_SHIFT': '색조 변화형',       # Δa or Δb > ΔL
        'FADING': '전반적 탈색형',        # All zones ΔE↑
        'LOCALIZED': '국소 변색형'        # Hotspot 존재
    },
    'STRUCTURE_DEFECTS': {
        'BOUNDARY_BLUR': '경계 흐림형',   # Transition width > std
        'MISALIGNMENT': '중심 치우침형',  # Shift > 5%
        'ZONE_EXPANSION': '영역 확장형',  # Zone area > std
        'ZONE_SHRINKAGE': '영역 축소형'   # Zone area < std
    },
    'INK_DEFECTS': {
        'INK_MISSING': '잉크 누락형',     # n_inks < std
        'INK_EXTRA': '잉크 추가형',       # n_inks > std
        'INK_MIXING': '잉크 혼합 이상형'  # Mixing detected
    },
    'COMPOSITE': {
        'MULTI_DEFECT': '복합 결함형'     # 2개 이상 동시 발생
    }
}
```

#### 7.1.2 Classification Algorithm
```python
class DefectClassifier:
    def classify(self, test_result, std_result, comparison_result):
        """결함 유형 분류"""
        defects = []

        # 1. 색상 결함 분류
        for zone_name, zone_detail in comparison_result['zone_details'].items():
            if zone_detail['color'] < 70:
                defect_type = self._classify_color_defect(
                    test_result.zones[zone_name],
                    std_result.zones[zone_name]
                )
                defects.append({
                    'category': 'COLOR_DEFECTS',
                    'type': defect_type,
                    'zone': zone_name,
                    'severity': 100 - zone_detail['color']
                })

        # 2. 구조 결함 분류
        if comparison_result.get('alignment_quality', {}).get('shift', 0) > 5.0:
            defects.append({
                'category': 'STRUCTURE_DEFECTS',
                'type': 'MISALIGNMENT',
                'severity': comparison_result['alignment_quality']['shift'] * 10
            })

        for zone_name, zone_detail in comparison_result['zone_details'].items():
            if zone_detail['boundary'] < 70:
                defects.append({
                    'category': 'STRUCTURE_DEFECTS',
                    'type': 'BOUNDARY_BLUR',
                    'zone': zone_name,
                    'severity': 100 - zone_detail['boundary']
                })

        # 3. 잉크 결함 분류
        if comparison_result['ink_details']['count_penalty'] > 0:
            test_n = test_result.ink_analysis.n_inks
            std_n = std_result.ink_analysis.n_inks

            if test_n < std_n:
                defect_type = 'INK_MISSING'
            else:
                defect_type = 'INK_EXTRA'

            defects.append({
                'category': 'INK_DEFECTS',
                'type': defect_type,
                'severity': comparison_result['ink_details']['count_penalty']
            })

        # 4. 복합 결함 판단
        if len(defects) >= 2:
            defects.append({
                'category': 'COMPOSITE',
                'type': 'MULTI_DEFECT',
                'component_count': len(defects),
                'severity': np.mean([d['severity'] for d in defects])
            })

        return defects

    def _classify_color_defect(self, test_zone, std_zone):
        """색상 결함 세부 분류"""
        diff_L = test_zone.mean_lab[0] - std_zone.mean_lab[0]
        diff_a = test_zone.mean_lab[1] - std_zone.mean_lab[1]
        diff_b = test_zone.mean_lab[2] - std_zone.mean_lab[2]

        # Lightness 변화가 주 원인
        if abs(diff_L) > abs(diff_a) and abs(diff_L) > abs(diff_b):
            if diff_L > 0:
                return 'UNDERDOSE'  # 밝아짐 = 농도 부족
            else:
                return 'OVERDOSE'   # 어두워짐 = 농도 과다

        # Hue 변화가 주 원인
        else:
            return 'HUE_SHIFT'
```

### 7.2 ML Training Data Preparation
```python
def export_for_ml_training(comparison_results, defect_classifications):
    """ML 학습용 데이터 준비"""
    training_data = []

    for result, classification in zip(comparison_results, defect_classifications):
        # Feature Vector 구성
        features = {
            # Structure features
            'boundary_A_diff': result['zone_details']['A']['boundary'],
            'boundary_B_diff': result['zone_details']['B']['boundary'],
            'boundary_C_diff': result['zone_details']['C']['boundary'],
            'alignment_shift': result.get('alignment_quality', {}).get('shift', 0),

            # Color features
            'zone_A_delta_e': result['zone_details']['A']['color'],
            'zone_B_delta_e': result['zone_details']['B']['color'],
            'zone_C_delta_e': result['zone_details']['C']['color'],
            'p95_delta_e': result.get('percentiles', {}).get('p95', 0),

            # Ink features
            'ink_count_diff': result['ink_details']['count_penalty'],
            'avg_ink_delta_e': np.mean(result['ink_details']['matched_delta_e']),

            # Hotspot features
            'hotspot_count': len(result.get('hotspots', [])),
            'max_hotspot_area': max([h['area'] for h in result.get('hotspots', [])], default=0)
        }

        # Label: Primary defect type
        primary_defect = max(classification, key=lambda x: x['severity'])
        label = f"{primary_defect['category']}_{primary_defect['type']}"

        training_data.append({
            'features': features,
            'label': label,
            'severity': primary_defect['severity']
        })

    return training_data
```

---

## 8. Updated System Architecture

### 8.1 Enhanced Comparison Pipeline
```
Input: Test Image + STD Model
    ↓
[Phase 1] Analysis
    - Lens Detection
    - Zone Segmentation
    - Ink Estimation
    - Radial Profiling
    ↓
[Phase 2] Statistical Comparison
    - Load STD Statistical Model (cached)
    - Profile Normalization
    ↓
[Phase 3] Elastic Alignment
    - Anchor Zone Identification
    - Circular Shift Alignment
    - DTW Fine-Tuning
    - Alignment Quality Check
    ↓
[Phase 4] Dual Scoring
    - Zone-Based Score (structure)
    - Ink-Based Score (color)
    - Worst-Case Metrics (p95, hotspots)
    ↓
[Phase 5] Judgment & Explainability
    - Combined Score Calculation
    - Pass/Fail Decision
    - Failure Reason Extraction (Top 3)
    - Defect Classification
    - Confidence Score
    ↓
[Phase 6] Output
    - Judgment: PASS / FAIL / RETAKE / MANUAL_REVIEW
    - Score: 0-100
    - Confidence: 0-100
    - Reasons: Top 3 with actions
    - Defect Types: Phenomenological classification
    - Visualizations: Delta E map, Hotspot overlay
```

### 8.2 Updated DB Schema
```sql
-- STD 통계 모델 (Multiple samples)
CREATE TABLE std_models (
    id SERIAL PRIMARY KEY,
    sku_code VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    n_samples INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    approved_by VARCHAR(100),
    approved_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(sku_code, version)
);

CREATE TABLE std_samples (
    id SERIAL PRIMARY KEY,
    std_model_id INTEGER REFERENCES std_models(id) ON DELETE CASCADE,
    sample_index INTEGER,
    image_path VARCHAR(500),
    analysis_result JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE std_statistics (
    id SERIAL PRIMARY KEY,
    std_model_id INTEGER REFERENCES std_models(id) ON DELETE CASCADE,
    zone_name VARCHAR(10),

    -- Color statistics
    mean_L FLOAT,
    std_L FLOAT,
    mean_a FLOAT,
    std_a FLOAT,
    mean_b FLOAT,
    std_b FLOAT,

    -- Structure statistics
    mean_position FLOAT,
    std_position FLOAT,
    percentile_5 FLOAT,
    percentile_95 FLOAT,

    -- Detailed stats
    detailed_stats JSONB
);

-- Comparison results (enhanced)
CREATE TABLE comparison_results (
    id SERIAL PRIMARY KEY,
    test_sample_id INTEGER REFERENCES test_samples(id),
    std_model_id INTEGER REFERENCES std_models(id),

    -- Scores
    total_score FLOAT,
    zone_score FLOAT,
    ink_score FLOAT,
    confidence_score FLOAT,

    -- Judgment
    judgment VARCHAR(20),  -- PASS, FAIL, RETAKE, MANUAL_REVIEW

    -- Explainability
    top_failure_reasons JSONB,  -- Top 3 reasons
    defect_classifications JSONB,  -- Phenomenological types

    -- Detailed results
    zone_details JSONB,
    ink_details JSONB,
    alignment_details JSONB,
    worst_case_metrics JSONB,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_time_ms INTEGER
);

CREATE INDEX idx_comparison_judgment ON comparison_results(judgment);
CREATE INDEX idx_comparison_score ON comparison_results(total_score);
CREATE INDEX idx_std_statistics_model ON std_statistics(std_model_id, zone_name);
```

---

## 9. Implementation Roadmap (Updated)

### Phase 0: Technical Foundation (Week 1-2)
- [ ] **Week 1.0**: Algorithm Benchmarking
  - DTW vs FastDTW performance test (100 samples)
  - Alignment algorithm prototyping
  - Worst-case metrics validation
- [ ] **Week 1.5**: DB Schema Migration
  - Implement `std_models`, `std_samples`, `std_statistics`
  - Migration scripts
  - SQLAlchemy models

### Phase 1: STD Statistical Model (Week 3-4)
- [ ] **Week 3**: Multi-Sample Registration
  - UI for uploading 5-10 STD samples
  - Batch analysis pipeline
  - Statistical calculation engine
- [ ] **Week 4**: Acceptance Criteria Derivation
  - Auto-calculate thresholds from distribution
  - Upper/Middle/Lower limit configuration

### Phase 2: Enhanced Comparison Engine (Week 5-8)
- [ ] **Week 5-6**: Elastic Alignment
  - Anchor zone detection
  - Circular shift + DTW implementation
  - Alignment quality metrics
- [ ] **Week 7**: Dual Scoring + Worst-Case Metrics
  - Zone-based + Ink-based scoring
  - Percentile ΔE calculation
  - Hotspot detection
- [ ] **Week 8**: Explainability Layer
  - Failure reason extraction
  - Top 3 ranking
  - Defect classification

### Phase 3: Production Features (Week 9-10)
- [ ] **Week 9**: Fail-Safe & Confidence
  - Profile caching
  - RETAKE logic
  - Confidence score separation
- [ ] **Week 10**: UI/Dashboard
  - Comparison results visualization
  - Delta E heatmap
  - Explainability report UI

### Phase 4: Advanced Analytics (v2.0)
- [ ] ML model training (defect classification)
- [ ] Process control charts
- [ ] Automated adjustment recommendations

---

## 10. Success Metrics

### Technical Metrics
- **Performance**: avg < 1.0s, p99 < 3.0s
- **Accuracy**: 95% agreement with manual inspection
- **False Negative Rate**: < 2% (NG를 OK로 판정)
- **False Positive Rate**: < 5% (OK를 NG로 판정)

### Operational Metrics
- **Explainability**: 100% FAIL cases have Top 3 reasons
- **Actionability**: 90% reasons include specific actions
- **Confidence**: 80% results have confidence > 80%

### Business Metrics
- **Inspection Time**: 3분 → 30초 (6배 향상)
- **Manual Review Rate**: < 3% (MANUAL_REVIEW + RETAKE)
- **Defect Detection Recall**: > 98%

---

## 11. Risk Mitigation

### Technical Risks
1. **Alignment 실패**: Correlation < 0.8
   - Mitigation: Fail-safe → RETAKE
2. **Hotspot 과검출**: False alarm
   - Mitigation: min_area threshold tuning
3. **DB 성능 저하**: 통계 테이블 조인 비용
   - Mitigation: 인덱스 최적화, 캐싱

### Operational Risks
1. **STD 샘플 품질 저하**: 불량 샘플 포함
   - Mitigation: Approval workflow, outlier detection
2. **설명 과잉**: Top 3 reasons 해석 어려움
   - Mitigation: 사용자 교육, UI 개선

---

## 12. Next Steps

1. **알고리즘 벤치마크 스크립트 작성** (즉시 착수)
   - `tools/benchmark_alignment.py`
   - DTW vs FastDTW 비교
   - 100 샘플 테스트

2. **DB 스키마 개선 구현**
   - `src/models/std_models.py` (SQLAlchemy)
   - Alembic migration scripts

3. **통계 모델 엔진 구현**
   - `src/services/std_statistical_model.py`
   - Multi-sample aggregation
   - Acceptance criteria derivation

4. **Elastic Alignment 프로토타입**
   - `src/core/alignment.py`
   - Anchor zone detection
   - DTW fine-tuning

---

**작성자**: Claude Sonnet 4.5
**검토 필요**: 알고리즘 타당성, 성능 목표, 일정 현실성
**승인 후 작업**: Phase 0 (Week 1) 착수
