"""
Zone Analyzer 2D - AI Template Integration

2D 이미지 기반 Zone 분석 (AI 피드백 반영)
- print_inner/outer 자동 추정
- ink_mask로 도트 픽셀만 평균
- 정확한 pixel_count
- mean_all vs mean_ink 비교
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.color_evaluator import InspectionResult, ZoneResult
from src.core.ink_estimator import InkEstimator
from src.core.lens_detector import LensDetection

logger = logging.getLogger(__name__)


# ================================
# Color Space Utilities
# ================================


def lab_to_rgb_hex(lab: Tuple[float, float, float]) -> Tuple[Tuple[int, int, int], str]:
    """Convert CIELAB (L:0-100, a,b:-128-127) to sRGB and HEX."""
    L, a, b = lab
    # OpenCV Lab uses L in [0,255] scale; a,b shifted by +128.
    # Use uint8 to get 0~255 output range (not normalized 0~1)
    lab_cv = np.array([L * 255.0 / 100.0, a + 128.0, b + 128.0], dtype=np.uint8)
    lab_cv = lab_cv.reshape(1, 1, 3)
    bgr = cv2.cvtColor(lab_cv, cv2.COLOR_Lab2BGR)[0, 0]
    rgb = (int(np.clip(bgr[2], 0, 255)), int(np.clip(bgr[1], 0, 255)), int(np.clip(bgr[0], 0, 255)))
    hexv = "#{:02X}{:02X}{:02X}".format(*rgb)
    return rgb, hexv


def bgr_to_lab_float(bgr: np.ndarray) -> np.ndarray:
    """
    OpenCV Lab (uint8) -> Standard CIE Lab (float)

    OpenCV Lab: L in [0,255], a,b in [0,255] with 128 offset
    Standard Lab: L* in [0,100], a* in [-128,127], b* in [-128,127]
    """
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    return np.stack([L, a, b], axis=-1)


def delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Simple Euclidean distance in Lab space"""
    d = lab1 - lab2
    return float(np.sqrt(np.sum(d * d)))


def safe_mean_lab(lab_float: np.ndarray, mask: np.ndarray) -> Tuple[Optional[List[float]], int]:
    """Calculate mean Lab from masked region"""
    idx = mask.astype(bool)
    n = int(np.sum(idx))
    if n == 0:
        return None, 0

    pixels = lab_float[idx]
    mean = pixels.mean(axis=0)
    return mean.tolist(), n


# ================================
# Geometry Utilities
# ================================


def radial_map(shape_hw: Tuple[int, int], cx: float, cy: float) -> np.ndarray:
    """Create radial distance map"""
    h, w = shape_hw
    yy, xx = np.indices((h, w))
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.asarray(rr, dtype=np.float32)


def circle_mask(shape_hw: Tuple[int, int], cx: float, cy: float, r: float) -> np.ndarray:
    """Create circular mask"""
    rr = radial_map(shape_hw, cx, cy)
    return (rr <= r).astype(np.uint8) * 255


# ================================
# Ink Mask (Dot Pixels)
# ================================


@dataclass
class InkMaskConfig:
    method: str = "sat_val"  # "sat_val" or "gray_otsu"
    saturation_min: int = 40  # HSV S 하한 (잉크 검출)
    value_max: int = 200  # HSV V 상한 (하이라이트 제거)
    morph_kernel_size: int = 3
    morph_iterations: int = 1


def build_ink_mask(img_bgr: np.ndarray, lens_mask: np.ndarray, config: InkMaskConfig = InkMaskConfig()) -> np.ndarray:
    """
    잉크 도트 픽셀만 분리

    Args:
        img_bgr: OpenCV BGR 이미지
        lens_mask: 렌즈 영역 마스크
        config: ink mask 설정

    Returns:
        ink_mask (uint8, 0 or 255)
    """
    h, w = img_bgr.shape[:2]

    if config.method == "sat_val":
        # HSV 기반: 채도 높고 명도 낮은 영역 = 잉크
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)

        # 잉크는 보통 배경/하이라이트보다 "채도↑, 명도↓"
        ink = ((S > config.saturation_min) & (V < config.value_max)).astype(np.uint8) * 255

    elif config.method == "gray_otsu":
        # 그레이 Otsu 기반
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_roi = gray.copy()
        gray_roi[lens_mask == 0] = 255
        _, th = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink = np.asarray(th, dtype=np.uint8)

    else:
        raise ValueError(f"Unknown ink_mask method: {config.method}")

    # 렌즈 영역 밖 제거
    ink = np.asarray(cv2.bitwise_and(ink, ink, mask=lens_mask), dtype=np.uint8)

    # 점(dot) 잡음 정리 (가볍게)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.morph_kernel_size, config.morph_kernel_size))
    ink = np.asarray(cv2.morphologyEx(ink, cv2.MORPH_OPEN, k, iterations=config.morph_iterations), dtype=np.uint8)

    return ink


# ================================
# Print Boundaries Detection
# ================================


@dataclass
class PrintBoundaryResult:
    print_inner: float  # pixel 단위
    print_outer: float  # pixel 단위
    r_inner_norm: float  # 정규화 (0~1)
    r_outer_norm: float  # 정규화 (0~1)
    confidence: float
    radial_profile: List[float]


def estimate_print_boundaries(
    img_bgr: np.ndarray,
    cx: float,
    cy: float,
    lens_radius: float,
    lens_mask: np.ndarray,
    ink_mask: np.ndarray,
    n_bins: int = 220,
) -> PrintBoundaryResult:
    """
    잉크 밀도(반경별 ink 픽셀 비율)로 print_inner/outer 추정

    Returns:
        PrintBoundaryResult with boundaries and confidence
    """
    h, w = img_bgr.shape[:2]

    yy, xx = np.indices((h, w))
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rr_norm = rr / float(lens_radius + 1e-6)

    valid = (lens_mask > 0) & (rr <= lens_radius)
    ink = (ink_mask > 0) & valid

    # 반경 binning
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    radial_ink_ratio = np.zeros(n_bins, np.float32)
    radial_count = np.zeros(n_bins, np.int32)

    rflat = rr_norm[valid].ravel()
    inkflat = ink[valid].ravel()

    inds = np.clip(np.digitize(rflat, bins) - 1, 0, n_bins - 1)
    for i in range(n_bins):
        m = inds == i
        radial_count[i] = int(m.sum())
        if radial_count[i] > 0:
            radial_ink_ratio[i] = float(inkflat[m].mean())

    # Smoothing
    radial_s = cv2.GaussianBlur(radial_ink_ratio.reshape(-1, 1), (1, 21), 0).ravel()

    # 잉크가 "뚜렷이 존재"하는 구간 찾기
    peak = float(radial_s.max())
    if peak < 0.02:
        # 잉크가 거의 안 잡힘 - fallback
        print(f"Ink peak too low ({peak:.3f}), using fallback boundaries")
        return PrintBoundaryResult(
            print_inner=lens_radius * 0.15,
            print_outer=lens_radius * 0.95,
            r_inner_norm=0.15,
            r_outer_norm=0.95,
            confidence=0.2,
            radial_profile=radial_s.tolist(),
        )

    thr = max(0.12 * peak, 0.03)
    active = radial_s > thr

    # 가장 큰 연속 구간 선택
    runs = []
    start = None
    for i, v in enumerate(active):
        if v and start is None:
            start = i
        if (not v) and (start is not None):
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, n_bins - 1))

    if not runs:
        print("No ink runs found, using fallback boundaries")
        return PrintBoundaryResult(
            print_inner=lens_radius * 0.15,
            print_outer=lens_radius * 0.95,
            r_inner_norm=0.15,
            r_outer_norm=0.95,
            confidence=0.2,
            radial_profile=radial_s.tolist(),
        )

    # Run 길이 최대 선택
    runs.sort(key=lambda x: (x[1] - x[0]), reverse=True)
    a, b = runs[0]

    r_in_norm = float(bins[a])
    r_out_norm = float(bins[b + 1])

    print_inner = r_in_norm * lens_radius
    print_outer = r_out_norm * lens_radius

    # Confidence from peak strength
    conf = float(np.clip(peak / 0.5, 0, 1))

    print(
        f"[PRINT BOUNDARIES] inner={print_inner:.1f}px ({r_in_norm:.3f}), "
        f"outer={print_outer:.1f}px ({r_out_norm:.3f}), "
        f"confidence={conf:.2f}"
    )

    return PrintBoundaryResult(
        print_inner=print_inner,
        print_outer=print_outer,
        r_inner_norm=r_in_norm,
        r_outer_norm=r_out_norm,
        confidence=conf,
        radial_profile=radial_s.tolist(),
    )


# ================================
# Zone Spec (needed by production functions)
# ================================


@dataclass
class ZoneSpec:
    name: str
    r_start_norm: float  # 0..1 within print band
    r_end_norm: float  # 0..1 within print band


# ================================
# Production-Level Advanced Functions
# ================================


@dataclass
class TransitionRange:
    """전이 구간 (gradient가 높은 불안정 영역)"""

    r_start: float  # 정규화 좌표 (0~1)
    r_end: float  # 정규화 좌표 (0~1)
    max_gradient: float  # 최대 gradient 값


def find_transition_ranges(
    img_bgr: np.ndarray,
    cx: float,
    cy: float,
    print_inner: float,
    print_outer: float,
    lens_mask: np.ndarray,
    target_labs: Dict[str, List[float]],
    bins: int = 400,
    sigma_bins: int = 3,
    k_mad: float = 4.0,
    max_exclude_frac: float = 0.30,
) -> List[TransitionRange]:
    """
    Gradient 기반 전이 구간 자동 탐지

    Returns:
        List[TransitionRange]: 전이 구간 리스트
    """
    print(f"[TRANSITION] Finding transition ranges (bins={bins}, sigma={sigma_bins}, k_mad={k_mad})...")

    # 1. Calculate radial Lab profiles
    radial_lab, radial_count, bin_centers, bin_width = _calculate_radial_lab_profiles(
        img_bgr, cx, cy, print_inner, print_outer, lens_mask, bins, sigma_bins
    )

    # 2. Calculate gradients and threshold
    gradients, valid_bins = _calculate_gradients(radial_lab, radial_count)

    median_grad = float(np.median(gradients[gradients > 0])) if np.any(gradients > 0) else 0.0
    mad = float(np.median(np.abs(gradients - median_grad))) if np.any(gradients > 0) else 0.0
    threshold = median_grad + k_mad * mad

    print(f"[TRANSITION] Gradient stats: median={median_grad:.2f}, MAD={mad:.2f}, threshold={threshold:.2f}")

    # 3. Extract, merge and filter ranges
    merged = _extract_and_merge_ranges(gradients, threshold, bin_centers, bin_width, bins, max_exclude_frac)

    print(f"[TRANSITION] Found {len(merged)} transition ranges:")
    for tr in merged:
        print(f"  - r=[{tr.r_start:.3f}, {tr.r_end:.3f}], max_grad={tr.max_gradient:.2f}")

    return merged


def _calculate_radial_lab_profiles(
    img_bgr: np.ndarray,
    cx: float,
    cy: float,
    print_inner: float,
    print_outer: float,
    lens_mask: np.ndarray,
    bins: int,
    sigma_bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Calculate average Lab values per radius bin."""
    h, w = img_bgr.shape[:2]
    yy, xx = np.indices((h, w))
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Print band 내부만
    band = (rr >= print_inner) & (rr <= print_outer) & (lens_mask > 0)
    r_norm = (rr - print_inner) / max(print_outer - print_inner, 1e-6)

    # Lab 이미지
    lab = bgr_to_lab_float(img_bgr)

    # 반경별 평균 Lab 계산
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = 1.0 / bins

    radial_lab = np.zeros((bins, 3), np.float32)
    radial_count = np.zeros(bins, np.int32)

    rflat = r_norm[band].ravel()
    labflat = lab[band].reshape(-1, 3)

    if rflat.size > 0:
        inds = np.clip(np.digitize(rflat, bin_edges) - 1, 0, bins - 1)
        for i in range(bins):
            m = inds == i
            cnt = int(m.sum())
            radial_count[i] = cnt
            if cnt > 0:
                radial_lab[i] = labflat[m].mean(axis=0)

    # Smoothing (Gaussian)
    if sigma_bins > 0:
        for ch in range(3):
            radial_lab[:, ch] = cv2.GaussianBlur(radial_lab[:, ch].reshape(-1, 1), (1, 2 * sigma_bins + 1), 0).ravel()

    return radial_lab, radial_count, bin_centers, bin_width


def _calculate_gradients(radial_lab: np.ndarray, radial_count: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate CIEDE76 gradients between adjacent bins."""
    bins = len(radial_lab)
    valid_bins = radial_count > max(radial_count.max() * 0.01, 10)

    gradients = np.zeros(bins - 1, np.float32)
    for i in range(bins - 1):
        if valid_bins[i] and valid_bins[i + 1]:
            lab1 = radial_lab[i]
            lab2 = radial_lab[i + 1]
            gradients[i] = delta_e_cie76(lab1, lab2)

    return gradients, valid_bins


def _extract_and_merge_ranges(
    gradients: np.ndarray,
    threshold: float,
    bin_centers: np.ndarray,
    bin_width: float,
    bins: int,
    max_exclude_frac: float,
) -> List[TransitionRange]:
    """Extract continuous transition ranges and merge overlapping ones."""
    is_transition = gradients > threshold

    # 연속 구간 추출
    ranges = []
    start = None
    for i, flag in enumerate(is_transition):
        if flag and start is None:
            start = i
        if (not flag) and (start is not None):
            r_start = float(bin_centers[start])
            r_end = float(bin_centers[i])
            max_grad = float(gradients[start : i + 1].max())
            ranges.append(TransitionRange(r_start, r_end, max_grad))
            start = None

    if start is not None:
        r_start = float(bin_centers[start])
        r_end = 1.0
        max_grad = float(gradients[start:].max())
        ranges.append(TransitionRange(r_start, r_end, max_grad))

    # 전이 구간 확장 (dilation, ±1 bin) 및 병합
    dilated_ranges = []
    for tr in ranges:
        dilated_ranges.append(
            TransitionRange(max(0.0, tr.r_start - bin_width), min(1.0, tr.r_end + bin_width), tr.max_gradient)
        )

    merged = []
    if dilated_ranges:
        dilated_ranges.sort(key=lambda x: x.r_start)
        current = dilated_ranges[0]
        for tr in dilated_ranges[1:]:
            if tr.r_start <= current.r_end:
                current = TransitionRange(
                    current.r_start, max(current.r_end, tr.r_end), max(current.max_gradient, tr.max_gradient)
                )
            else:
                merged.append(current)
                current = tr
        merged.append(current)

    # 최대 제외 비율 체크
    total_excluded = sum(tr.r_end - tr.r_start for tr in merged)
    if total_excluded > max_exclude_frac:
        print(f"[TRANSITION] WARNING: Excluded fraction {total_excluded:.2%} > {max_exclude_frac:.0%}")
        merged.sort(key=lambda x: x.max_gradient, reverse=True)
        cumulative = 0.0
        filtered = []
        for tr in merged:
            if cumulative + (tr.r_end - tr.r_start) <= max_exclude_frac:
                filtered.append(tr)
                cumulative += tr.r_end - tr.r_start
            else:
                break
        filtered.sort(key=lambda x: x.r_start)
        merged = filtered

    return merged


def auto_define_zone_B(
    transition_ranges: List[TransitionRange],
    min_width: float = 0.15,
    max_width: float = 0.25,
    min_pixels: int = 30000,
    max_pixels: int = 80000,
    expected_pixel_ratio: float = 0.25,
    print_band_area: int = 300000,
) -> Optional[ZoneSpec]:
    """
    Zone B 경계 자동 정의 (전이 구간 제거 + 최적 구간 선택)

    Returns:
        ZoneSpec or None (조건 미충족 시 None)
    """
    print(f"[AUTO ZONE B] Defining Zone B (min_width={min_width}, max_width={max_width}, min_pixels={min_pixels})...")

    # B 후보 영역: 전체 범위에서 C(0~0.33), A(0.66~1.0) 제외
    B_candidate_start = 0.33
    B_candidate_end = 0.66
    B_candidate_width = B_candidate_end - B_candidate_start

    print(
        f"[AUTO ZONE B] B_candidate_range: [{B_candidate_start:.3f}, {B_candidate_end:.3f}], "
        f"width={B_candidate_width:.3f}"
    )

    # 가용 구간 계산 (전이 구간 제외)
    available_segments = _split_segment_by_exclusion(B_candidate_start, B_candidate_end, transition_ranges)

    print(f"[AUTO ZONE B] Available segments after transition removal: {available_segments}")

    if not available_segments:
        print("[AUTO ZONE B] No available segments after transition removal, using SAFE FALLBACK")
        return ZoneSpec("B", 0.38, 0.61)

    # 다중 후보 검증
    candidates = []
    for seg_start, seg_end in available_segments:
        width = seg_end - seg_start

        # 가드레일 체크
        if width < min_width:
            continue
        if width > max_width:
            continue

        estimated_pixels = int(print_band_area * width)
        if estimated_pixels < min_pixels or estimated_pixels > max_pixels:
            continue

        candidates.append((seg_start, seg_end, width, estimated_pixels))

    if not candidates:
        print("[AUTO ZONE B] No candidates met constraints, using SAFE FALLBACK")
        return ZoneSpec("B", 0.38, 0.61)

    # 최적 후보 선택 (폭 우선)
    best = max(candidates, key=lambda x: x[2])
    seg_start, seg_end, width, _ = best

    # 추가 안전 가드
    SAFE_MAX_WIDTH = 0.23
    if width > SAFE_MAX_WIDTH:
        print(f"[AUTO ZONE B] width={width:.3f} > {SAFE_MAX_WIDTH}, using SAFE FALLBACK")
        return ZoneSpec("B", 0.38, 0.61)

    print(f"[AUTO ZONE B] B_selected_range: [{seg_start:.3f}, {seg_end:.3f}]")
    return ZoneSpec("B", seg_start, seg_end)


def _split_segment_by_exclusion(
    start: float, end: float, exclusions: List[TransitionRange]
) -> List[Tuple[float, float]]:
    """Split a range [start, end] by excluding specific segments."""
    segments = [(start, end)]

    for tr in exclusions:
        ex_start, ex_end = tr.r_start, tr.r_end
        new_segments = []
        for seg_start, seg_end in segments:
            # No overlap
            if ex_end <= seg_start or ex_start >= seg_end:
                new_segments.append((seg_start, seg_end))
            else:
                # Overlap - split
                if seg_start < ex_start:
                    new_segments.append((seg_start, ex_start))
                if ex_end < seg_end:
                    new_segments.append((ex_end, seg_end))
        segments = new_segments

    return segments


def robust_mean_lab(
    lab_float: np.ndarray,
    mask: np.ndarray,
    target_lab: Optional[np.ndarray] = None,
    mode: str = "trimmed",
    trim_fraction: float = 0.10,
    min_pixels: int = 20000,
) -> Tuple[Optional[List[float]], int]:
    """
    Robust 평균 Lab 계산 (이상치 제거)

    Args:
        lab_float: Lab 이미지
        mask: 영역 마스크
        target_lab: target Lab (ΔE 계산용, trimmed 모드에서 사용)
        mode: "trimmed" (상하위 제거), "winsorize" (클램핑), "cluster" (k-means)
        trim_fraction: trimmed/winsorize 시 제거 비율 (0.10 = 상하위 10%씩)
        min_pixels: 최소 픽셀 수

    Returns:
        (mean_lab, pixel_count)
    """
    idx = mask.astype(bool)
    n = int(np.sum(idx))

    if n < min_pixels:
        print(f"[ROBUST MEAN] Insufficient pixels: {n} < {min_pixels}")
        return None, n

    pixels = lab_float[idx]  # shape: (n, 3)

    if mode == "trimmed":
        # ΔE 기준 trimmed mean
        if target_lab is not None:
            # ΔE 계산
            delta_e_values = np.array([delta_e_cie76(p, target_lab) for p in pixels])
            # 정렬
            sorted_indices = np.argsort(delta_e_values)
            # 상하위 trim_fraction 제거
            trim_count = int(n * trim_fraction)
            keep_indices = sorted_indices[trim_count : n - trim_count]
            trimmed_pixels = pixels[keep_indices]
            mean = trimmed_pixels.mean(axis=0)
            print(f"[ROBUST MEAN] Trimmed {trim_count*2}/{n} pixels ({trim_fraction:.0%} each side)")
        else:
            # target 없으면 일반 평균
            mean = pixels.mean(axis=0)
            print("[ROBUST MEAN] No target, using simple mean")

    elif mode == "winsorize":
        # Winsorize: 상하위 trim_fraction을 중앙값으로 클램핑
        percentile_low = trim_fraction * 100
        percentile_high = (1 - trim_fraction) * 100

        winsorized = pixels.copy()
        for ch in range(3):
            p_low = np.percentile(pixels[:, ch], percentile_low)
            p_high = np.percentile(pixels[:, ch], percentile_high)
            winsorized[:, ch] = np.clip(pixels[:, ch], p_low, p_high)

        mean = winsorized.mean(axis=0)
        print(f"[ROBUST MEAN] Winsorized at p{percentile_low:.0f}/p{percentile_high:.0f}")

    elif mode == "cluster":
        # k-means clustering (2 clusters: 잉크 vs 배경)
        from sklearn.cluster import KMeans

        # k=2로 클러스터링
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)

        # target에 가까운 클러스터 선택
        if target_lab is not None:
            de0 = delta_e_cie76(kmeans.cluster_centers_[0], target_lab)
            de1 = delta_e_cie76(kmeans.cluster_centers_[1], target_lab)
            best_cluster = 0 if de0 < de1 else 1
        else:
            # L 값이 낮은 클러스터 선택 (잉크)
            best_cluster = 0 if kmeans.cluster_centers_[0, 0] < kmeans.cluster_centers_[1, 0] else 1

        cluster_pixels = pixels[labels == best_cluster]
        mean = cluster_pixels.mean(axis=0)
        print(f"[ROBUST MEAN] Cluster {best_cluster} selected: {len(cluster_pixels)}/{n} pixels")

    else:
        # 기본: 단순 평균
        mean = pixels.mean(axis=0)
        print(f"[ROBUST MEAN] Simple mean (mode={mode})")

    return mean.tolist(), n


@dataclass
class ConfidenceFactors:
    """Confidence 점수 구성 요소"""

    pixel_count_score: float  # 픽셀 수 비율
    transition_score: float  # 전이 구간 제거 비율
    std_score: float  # 표준편차 점수
    sector_uniformity: float  # 섹터 균일도 (옵션)
    lens_detection: float  # 렌즈 검출 신뢰도
    overall: float  # 전체 점수


def compute_sector_statistics(
    img_lab: np.ndarray,
    zone_masks: Dict[str, np.ndarray],
    lens_center: Tuple[int, int],
    num_sectors: int = 8,
) -> Dict:
    """
    각도별 섹터 통계를 계산하여 국부 불량을 감지합니다.

    Args:
        img_lab: LAB 이미지 (float32, CIELAB 스케일)
        zone_masks: Zone별 마스크 딕셔너리
        lens_center: 렌즈 중심 좌표 (cx, cy)
        num_sectors: 섹터 개수 (기본 8개, 45도씩)

    Returns:
        섹터 통계 딕셔너리
    """
    cx, cy = lens_center
    h, w = img_lab.shape[:2]

    # 각 픽셀의 각도 계산
    yy, xx = np.ogrid[:h, :w]
    dx = xx - cx
    dy = yy - cy
    angles = np.arctan2(dy, dx) * 180 / np.pi  # -180 ~ 180
    angles = (angles + 360) % 360  # 0 ~ 360

    sector_stats = {}
    max_sector_std_L = 0.0
    worst_zone = None

    for zone_name, mask in zone_masks.items():
        zone_pixels = mask > 0

        # 각 섹터별 L 값 수집
        sector_L_values = []
        for i in range(num_sectors):
            angle_start = i * (360 / num_sectors)
            angle_end = (i + 1) * (360 / num_sectors)

            sector_mask = zone_pixels & (angles >= angle_start) & (angles < angle_end)
            L_values = img_lab[sector_mask, 0]

            if len(L_values) > 0:
                sector_L_values.append(np.mean(L_values))

        if len(sector_L_values) >= 3:  # 최소 3개 섹터는 있어야 통계 의미 있음
            std_L = float(np.std(sector_L_values))
            sector_stats[zone_name] = {
                "sector_count": len(sector_L_values),
                "std_L": std_L,
                "L_mean": float(np.mean(sector_L_values)),
                "L_range": float(np.max(sector_L_values) - np.min(sector_L_values)),
            }

            if std_L > max_sector_std_L:
                max_sector_std_L = std_L
                worst_zone = zone_name

    return {
        "enabled": len(sector_stats) > 0,
        "num_sectors": num_sectors,
        "zone_stats": sector_stats,
        "max_sector_std_L": max_sector_std_L,
        "worst_zone": worst_zone,
        "uniform": max_sector_std_L < 5.0,  # 임계값: std_L < 5.0이면 균일
    }


def compute_confidence(
    zone_results_raw: List[Dict[str, Any]],
    transition_ranges: List[TransitionRange],
    lens_confidence: float = 1.0,
    sector_uniformity: Optional[float] = None,
    expected_pixel_counts: Optional[Dict[str, int]] = None,
) -> ConfidenceFactors:
    """
    Multi-factor confidence 점수 계산

    Args:
        zone_results_raw: Zone 결과 (pixel_count, std_lab 포함)
        transition_ranges: 전이 구간 리스트
        lens_confidence: 렌즈 검출 신뢰도
        sector_uniformity: 섹터 균일도 (있으면 사용)
        expected_pixel_counts: Zone별 예상 픽셀 수

    Returns:
        ConfidenceFactors
    """
    print("[CONFIDENCE] Computing multi-factor confidence...")

    # 1. Pixel count score
    pixel_scores = []
    if expected_pixel_counts:
        for zr in zone_results_raw:
            zn = zr["zone_name"]
            actual = zr["pixel_count"]
            expected = expected_pixel_counts.get(zn, actual)
            if expected > 0:
                ratio = actual / expected
                # 0.8 ~ 1.2 범위: score = 1.0
                # 그 외: 선형 감소
                if 0.8 <= ratio <= 1.2:
                    score = 1.0
                elif ratio < 0.8:
                    score = max(0.0, ratio / 0.8)
                else:
                    score = max(0.0, 2.0 - ratio)
                pixel_scores.append(score)
    pixel_count_score = float(np.mean(pixel_scores)) if pixel_scores else 1.0

    # 2. Transition removal score
    total_excluded = sum(tr.r_end - tr.r_start for tr in transition_ranges)
    # 제외 비율이 적을수록 좋음 (신뢰도 높음)
    # 0~10%: 1.0, 10~30%: 선형 감소, 30%~: 0.0
    if total_excluded <= 0.10:
        transition_score = 1.0
    elif total_excluded <= 0.30:
        transition_score = 1.0 - (total_excluded - 0.10) / 0.20
    else:
        transition_score = 0.0

    # 3. std_L score
    std_l_values = []
    for zr in zone_results_raw:
        std_lab = zr.get("std_lab")
        if std_lab:
            std_l_values.append(std_lab[0])

    if std_l_values:
        avg_std_l = float(np.mean(std_l_values))
        # std_L < 5: 1.0, 5~15: 선형 감소, 15~: 0.0
        if avg_std_l < 5.0:
            std_score = 1.0
        elif avg_std_l < 15.0:
            std_score = 1.0 - (avg_std_l - 5.0) / 10.0
        else:
            std_score = 0.0
    else:
        std_score = 1.0

    # 4. Sector uniformity (옵션)
    if sector_uniformity is not None:
        sector_score = float(sector_uniformity)
    else:
        sector_score = 1.0  # 데이터 없으면 중립

    # 5. Lens detection
    lens_score = float(lens_confidence)

    # Overall: 가중 평균
    # pixel_count: 30%, transition: 25%, std: 25%, sector: 10%, lens: 10%
    overall = (
        0.30 * pixel_count_score + 0.25 * transition_score + 0.25 * std_score + 0.10 * sector_score + 0.10 * lens_score
    )

    factors = ConfidenceFactors(
        pixel_count_score=pixel_count_score,
        transition_score=transition_score,
        std_score=std_score,
        sector_uniformity=sector_score,
        lens_detection=lens_score,
        overall=overall,
    )

    print(
        f"[CONFIDENCE] Scores: pixel={pixel_count_score:.2f}, transition={transition_score:.2f}, "
        f"std={std_score:.2f}, sector={sector_score:.2f}, lens={lens_score:.2f}"
    )
    print(f"[CONFIDENCE] Overall={overall:.2f}")

    # 3-tier 분류
    if overall >= 0.70:
        tier = "OK"
    elif overall >= 0.40:
        tier = "REVIEW"
    else:
        tier = "NG"
    print(f"[CONFIDENCE] Tier: {tier}")

    return factors


# ================================
# Zone Masks (Print Band Basis)
# ================================


def build_zone_masks_from_printband(
    h: int,
    w: int,
    cx: float,
    cy: float,
    print_inner: float,
    print_outer: float,
    lens_mask: np.ndarray,
    zone_specs: List[ZoneSpec],
) -> Dict[str, np.ndarray]:
    """
    print_inner~print_outer 기준으로 정규화된 Zone 마스크 생성

    Returns:
        Dict[zone_name, mask (uint8)]
    """
    yy, xx = np.indices((h, w))
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    band = (rr >= print_inner) & (rr <= print_outer) & (lens_mask > 0)

    # r_norm within print band
    r_norm = (rr - print_inner) / (max(print_outer - print_inner, 1e-6))

    masks = {}
    for z in zone_specs:
        r0, r1 = sorted([z.r_start_norm, z.r_end_norm])
        m = band & (r_norm >= r0) & (r_norm < r1)
        masks[z.name] = m.astype(np.uint8) * 255

    return masks


# ================================
# Zone Results Computation
# ================================


def compute_zone_results_2d(
    img_bgr: np.ndarray,
    zone_masks: Dict[str, np.ndarray],
    target_labs: Dict[str, List[float]],
    thresholds: Dict[str, float],
    lens_mask: np.ndarray,
    ink_mask: np.ndarray,
    min_ink_ratio: float,
    min_ink_pixels: int,
) -> Tuple[str, float, List[Dict[str, Any]], List[str]]:
    """
    Zone별 결과 계산 (mean_all, mean_ink, pixel_count, ΔE)

    Returns:
        judgment, overall_delta_e, zone_results, ng_reasons
    """
    lab = bgr_to_lab_float(img_bgr)

    results = []
    ng_reasons = []
    delta_e_list = []

    for zn, zmask in zone_masks.items():
        # Calculate result for a single zone
        res = _calculate_single_zone_result(
            zn,
            zmask,
            lab,
            target_labs[zn],
            thresholds[zn],
            lens_mask,
            ink_mask,
            min_ink_ratio,
            min_ink_pixels,
        )

        results.append(res)

        # Collect ΔE and NG reasons
        if res["delta_e"] is not None:
            delta_e_list.append(res["delta_e"])

        if not res["is_ok"] and not res.get("ink_sufficient", True):
            continue
        if not res["is_ok"]:
            if res["measured_lab"] is None:
                ng_reasons.append(f"Zone {zn}: no pixels")
            else:
                ng_reasons.append(f"Zone {zn}: ΔE={res['delta_e']:.2f} > {res['threshold']:.2f}")

    # Overall ΔE
    overall_de = float(np.mean(delta_e_list)) if delta_e_list else 0.0

    # Judgment
    judgment = "OK" if len(ng_reasons) == 0 else "NG"

    return judgment, overall_de, results, ng_reasons


def _calculate_single_zone_result(
    zone_name: str,
    zone_mask: np.ndarray,
    lab_float: np.ndarray,
    target_lab: List[float],
    threshold: float,
    lens_mask: np.ndarray,
    ink_mask: np.ndarray,
    min_ink_ratio: float,
    min_ink_pixels: int,
) -> Dict[str, Any]:
    """Calculate statistics and comparison for a single zone."""
    z = (zone_mask > 0) & (lens_mask > 0)
    z_ink = z & (ink_mask > 0)

    # mean_all (전체)
    mean_all, n_all = safe_mean_lab(lab_float, z)

    # mean_ink (잉크만)
    mean_ink, n_ink = safe_mean_lab(lab_float, z_ink)

    ink_ratio = (n_ink / n_all) if n_all > 0 else 0.0

    # 상세 통계 (AI 진단용)
    std_all = None
    percentiles_all = None
    if n_all > 0:
        pixels_all = lab_float[z]
        std_all = [float(np.std(pixels_all[:, i])) for i in range(3)]
        percentiles_all = {
            "L": {f"p{p}": float(np.percentile(pixels_all[:, 0], p)) for p in [5, 25, 50, 75, 95]},
            "a": {f"p{p}": float(np.percentile(pixels_all[:, 1], p)) for p in [5, 25, 50, 75, 95]},
            "b": {f"p{p}": float(np.percentile(pixels_all[:, 2], p)) for p in [5, 25, 50, 75, 95]},
        }

    # Target Lab
    tgt = np.array(target_lab, np.float32)
    thr = float(threshold)

    # ΔE 계산: ink_ratio에 따라 선택
    ink_sufficient = n_ink >= min_ink_pixels and ink_ratio >= min_ink_ratio
    if mean_ink is not None and n_ink > 0 and ink_sufficient:
        used = np.array(mean_ink, np.float32)
        de = delta_e_cie76(used, tgt)
        used_basis = "mean_ink"
        measured_lab = mean_ink
    elif n_all > 0:
        de = None
        used_basis = "insufficient_ink"
        measured_lab = None
    else:
        de = None
        used_basis = "none"
        measured_lab = None

    is_ok = (de is not None) and (de <= thr)

    # 디버깅 로그
    de_str = f"{de:.2f}" if de is not None else "N/A"
    std_str = f"std=[{std_all[0]:.1f}, {std_all[1]:.1f}, {std_all[2]:.1f}]" if std_all else "std=N/A"
    print(
        f"  Zone {zone_name}: "
        f"pixels_all={n_all}, pixels_ink={n_ink}, ink_ratio={ink_ratio:.2%}, "
        f"Lab_all={mean_all}, Lab_ink={mean_ink}, {std_str}, "
        f"ΔE={de_str} (basis={used_basis})"
    )

    # Diff 계산 (측정 - 기준)
    diff_info = None
    if measured_lab is not None:
        dL = measured_lab[0] - target_lab[0]
        da = measured_lab[1] - target_lab[1]
        db = measured_lab[2] - target_lab[2]

        from src.core.color_evaluator import describe_color_shift

        diff_info = {
            "dL": float(dL),
            "da": float(da),
            "db": float(db),
            "direction": describe_color_shift(dL, da, db),
        }

    return {
        "zone_name": zone_name,
        "measured_lab": measured_lab,
        "target_lab": target_lab,
        "delta_e": de,
        "threshold": thr,
        "is_ok": bool(is_ok),
        "pixel_count": n_all,
        "pixel_count_ink": n_ink,
        "ink_pixel_ratio": float(ink_ratio),
        "ink_sufficient": bool(ink_sufficient),
        "measured_lab_all": mean_all,
        "measured_lab_ink": mean_ink,
        "delta_e_basis": used_basis,
        "std_lab": std_all,
        "percentiles": percentiles_all,
        "diff": diff_info,
    }


# ================================
# Main Pipeline
# ================================


# ================================
# Helper Functions for analyze_lens_zones_2d
# ================================


def _buffer_mask(mask: np.ndarray, sku_config: dict) -> np.ndarray:
    params = sku_config.get("params", {})
    mask_policy = params.get("mask_policy", {})
    buffer_iter = int(mask_policy.get("std_mask_buffer_iter", 1))
    if buffer_iter <= 0:
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(mask, kernel, iterations=buffer_iter)


def _warp_std_ink_mask_to_test(
    ref_image: Optional[np.ndarray],
    ref_lens: Optional[Any],
    test_image: Optional[np.ndarray],
    test_lens: Optional[Any],
    sku_config: dict,
    ink_mask_config: InkMaskConfig,
) -> Optional[np.ndarray]:
    if ref_image is None or ref_lens is None or test_image is None or test_lens is None:
        return None

    ref_h, ref_w = ref_image.shape[:2]
    ref_cx = float(ref_lens.center_x)
    ref_cy = float(ref_lens.center_y)
    ref_radius = float(ref_lens.radius)
    ref_lens_mask = circle_mask((ref_h, ref_w), ref_cx, ref_cy, ref_radius)
    ref_ink_mask = build_ink_mask(ref_image, ref_lens_mask, ink_mask_config)
    ref_ink_mask = _buffer_mask(ref_ink_mask, sku_config)
    ref_ink_mask = cv2.bitwise_and(ref_ink_mask, ref_ink_mask, mask=ref_lens_mask)

    test_h, test_w = test_image.shape[:2]
    test_cx = float(test_lens.center_x)
    test_cy = float(test_lens.center_y)
    test_radius = float(test_lens.radius)

    shared_radius = min(ref_radius, test_radius)
    r_bins = int(max(64, min(256, shared_radius)))
    theta_bins = 360

    polar = cv2.warpPolar(
        ref_ink_mask,
        (r_bins, theta_bins),
        (ref_cx, ref_cy),
        ref_radius,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS,
    )

    warped = cv2.warpPolar(
        polar,
        (test_w, test_h),
        (test_cx, test_cy),
        test_radius,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP,
    )

    return (warped > 0).astype(np.uint8)


def _setup_masks_and_boundaries(
    img_bgr: np.ndarray,
    lens_detection: LensDetection,
    sku_config: dict,
    ink_mask_config: Optional[InkMaskConfig],
    std_ref_image: Optional[np.ndarray] = None,
    std_ref_lens: Optional[Any] = None,
    mask_source: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float, float]:
    """
    초기화, 마스크 생성, 프린트 경계 설정

    Returns:
        lens_mask, ink_mask, cx, cy, r_lens, print_inner, print_outer
    """
    h, w = img_bgr.shape[:2]
    cx = lens_detection.center_x
    cy = lens_detection.center_y
    r_lens = lens_detection.radius

    if ink_mask_config is None:
        ink_mask_config = InkMaskConfig()

    # 1. Lens mask
    lens_mask = circle_mask((h, w), cx, cy, r_lens)

    # 2. Ink mask
    print("[2D ZONE ANALYSIS] Building ink mask...")
    if mask_source is None:
        if std_ref_image is not None and std_ref_lens is not None:
            mask_source = "std_warped"
        else:
            mask_source = "sample"

    if mask_source == "std_warped":
        ink_mask = _warp_std_ink_mask_to_test(
            ref_image=std_ref_image,
            ref_lens=std_ref_lens,
            test_image=img_bgr,
            test_lens=lens_detection,
            sku_config=sku_config,
            ink_mask_config=ink_mask_config,
        )
        if ink_mask is None:
            mask_source = "sample_fallback"
            ink_mask = build_ink_mask(img_bgr, lens_mask, ink_mask_config)
    else:
        ink_mask = build_ink_mask(img_bgr, lens_mask, ink_mask_config)
        if mask_source == "std":
            ink_mask = _buffer_mask(ink_mask, sku_config)
    ink_mask = cv2.bitwise_and(ink_mask, ink_mask, mask=lens_mask)

    # 3. Print boundaries - SKU config 우선, 자동 추정은 fallback
    print("[2D ZONE ANALYSIS] Getting print boundaries from SKU config...")
    optical_clear_ratio = sku_config.get("params", {}).get("optical_clear_ratio", 0.15)
    r_inner_norm = max(0.0, optical_clear_ratio)
    r_outer_norm = 0.95

    print_inner = r_inner_norm * r_lens
    print_outer = r_outer_norm * r_lens
    print(
        f"[PRINT BOUNDARIES] Using SKU config: inner={print_inner:.1f}px ({r_inner_norm:.3f}), "
        f"outer={print_outer:.1f}px ({r_outer_norm:.3f})"
    )

    return lens_mask, ink_mask, cx, cy, r_lens, print_inner, print_outer


def _detect_transitions_and_zones(
    img_bgr: np.ndarray,
    cx: float,
    cy: float,
    print_inner: float,
    print_outer: float,
    lens_mask: np.ndarray,
    sku_config: dict,
) -> Tuple[Dict[str, List[float]], Dict[str, float], List, List, int]:
    """
    Zone 스펙 로드, 전이 구간 자동 탐지, Zone B 자동 정의

    Returns:
        target_labs, thresholds, transition_ranges, zone_specs, print_band_area
    """
    # Zone specs (SKU config 기반)
    zone_targets = sku_config.get("zones", {})
    default_threshold = sku_config.get("default_threshold", 8.0)

    # Target Labs & Thresholds
    target_labs = {}
    thresholds = {}
    for zn in ["A", "B", "C"]:
        if zn in zone_targets:
            zt = zone_targets[zn]
            target_labs[zn] = [zt["L"], zt["a"], zt["b"]]
            thresholds[zn] = zt.get("threshold", default_threshold)

    # 전이 구간 자동 탐지 (Production-level enhancement)
    print("[2D ZONE ANALYSIS] Finding transition ranges...")
    transition_ranges = find_transition_ranges(
        img_bgr=img_bgr,
        cx=cx,
        cy=cy,
        print_inner=print_inner,
        print_outer=print_outer,
        lens_mask=lens_mask,
        target_labs=target_labs,
        bins=400,
        sigma_bins=1,  # 3 → 1: 스무딩 약하게 (gradient peak 살리기)
        k_mad=2.5,  # 4.0 → 2.5: 더 민감하게 전이 검출
        max_exclude_frac=0.30,
    )

    # Zone B 자동 정의 (전이 구간 제거)
    print("[2D ZONE ANALYSIS] Auto-defining Zone B...")
    print_band_area = int(np.pi * (print_outer**2 - print_inner**2))
    zone_b_spec = auto_define_zone_B(
        transition_ranges=transition_ranges,
        min_width=0.15,
        max_width=0.25,  # 🚨 NEW: 최대 폭 제한 (0.33 방지)
        min_pixels=30000,
        max_pixels=120000,
        expected_pixel_ratio=0.25,
        print_band_area=print_band_area,
    )

    # Zone 순서: C (inner) -> B (middle) -> A (outer)
    zone_specs = [
        ZoneSpec("C", 0.0, 0.33),
        zone_b_spec if zone_b_spec else ZoneSpec("B", 0.38, 0.61),  # Fallback
        ZoneSpec("A", 0.66, 1.0),
    ]

    print(
        f"[2D ZONE ANALYSIS] Final zone specs: C=[0.00, 0.33], "
        f"B=[{zone_specs[1].r_start_norm:.3f}, {zone_specs[1].r_end_norm:.3f}], A=[0.66, 1.00]"
    )

    return target_labs, thresholds, transition_ranges, zone_specs, print_band_area


def _compute_zone_results_wrapper(
    img_bgr: np.ndarray,
    h: int,
    w: int,
    cx: float,
    cy: float,
    print_inner: float,
    print_outer: float,
    lens_mask: np.ndarray,
    ink_mask: np.ndarray,
    zone_specs: List,
    target_labs: Dict[str, List[float]],
    thresholds: Dict[str, float],
    min_ink_ratio: float,
    min_ink_pixels: int,
) -> Tuple[str, float, List, List[Dict], List[str], Dict[str, np.ndarray], Dict]:
    """
    Zone masks 생성, Zone 결과 계산, ZoneResult 변환, Sector statistics 계산

    Returns:
        judgment, overall_de, zone_results (List[ZoneResult]), zone_results_raw, ng_reasons, zone_masks, sector_stats
    """
    # Zone masks 생성
    print("[2D ZONE ANALYSIS] Building zone masks...")
    zone_masks = build_zone_masks_from_printband(h, w, cx, cy, print_inner, print_outer, lens_mask, zone_specs)

    # Zone 결과 계산
    print("[2D ZONE ANALYSIS] Computing zone results...")
    judgment, overall_de, zone_results_raw, ng_reasons = compute_zone_results_2d(
        img_bgr,
        zone_masks,
        target_labs,
        thresholds,
        lens_mask,
        ink_mask,
        min_ink_ratio,
        min_ink_pixels,
    )

    # ZoneResult 변환
    zone_results = []
    for zr in zone_results_raw:
        if zr["measured_lab"] is not None:
            measured_lab = tuple(zr["measured_lab"])
        else:
            measured_lab = (0.0, 0.0, 0.0)

        zone_result = ZoneResult(
            zone_name=zr["zone_name"],
            measured_lab=measured_lab,
            target_lab=tuple(zr["target_lab"]),
            delta_e=zr["delta_e"] if zr["delta_e"] is not None else 0.0,
            threshold=zr["threshold"],
            is_ok=zr["is_ok"],
            pixel_count=zr["pixel_count"],
            diff=zr.get("diff"),  # 운영 UX: 색상 변화 상세
        )
        zone_results.append(zone_result)

    # Sector 통계 계산 (국부 불량 감지)
    print("[2D ZONE ANALYSIS] Computing sector statistics...")
    img_lab = bgr_to_lab_float(img_bgr)
    sector_stats = compute_sector_statistics(
        img_lab=img_lab, zone_masks=zone_masks, lens_center=(int(cx), int(cy)), num_sectors=8
    )

    if sector_stats["enabled"]:
        print(
            f"[SECTOR] max_std_L={sector_stats['max_sector_std_L']:.2f} "
            f"(worst={sector_stats['worst_zone']}), uniform={sector_stats['uniform']}"
        )

    return judgment, overall_de, zone_results, zone_results_raw, ng_reasons, zone_masks, sector_stats


def _calculate_confidence_wrapper(
    zone_results_raw: List[Dict],
    transition_ranges: List,
    lens_detection: LensDetection,
    sector_stats: Dict,
    zone_specs: List,
    print_band_area: int,
) -> Tuple[float, Any, float, Optional[float]]:
    """
    Multi-factor confidence 계산

    Returns:
        confidence, confidence_factors, lens_conf, sector_uniformity_score
    """
    print("[2D ZONE ANALYSIS] Computing confidence...")

    # Expected pixel counts (근사)
    expected_pixel_counts = {
        "C": int(print_band_area * 0.33),
        "B": int(print_band_area * (zone_specs[1].r_end_norm - zone_specs[1].r_start_norm)),
        "A": int(print_band_area * 0.34),
    }

    # Lens detection confidence (기본값 1.0, lens_detection에서 가져올 수 있으면 사용)
    lens_conf = getattr(lens_detection, "confidence", 1.0)

    # Sector uniformity score (1.0 = 균일, 0.0 = 불균일)
    sector_uniformity_score = None
    if sector_stats["enabled"]:
        # std_L < 3: 1.0, std_L > 8: 0.0, 선형 보간
        max_std = sector_stats["max_sector_std_L"]
        sector_uniformity_score = np.clip(1.0 - (max_std - 3.0) / 5.0, 0.0, 1.0)

    confidence_factors = compute_confidence(
        zone_results_raw=zone_results_raw,
        transition_ranges=transition_ranges,
        lens_confidence=lens_conf,
        sector_uniformity=sector_uniformity_score,
        expected_pixel_counts=expected_pixel_counts,
    )

    confidence = confidence_factors.overall

    return confidence, confidence_factors, lens_conf, sector_uniformity_score


def _check_retake_conditions(
    lens_conf: float,
    zone_results_raw: List[Dict],
    print_band_area: int,
    used_fallback_B: bool,
    confidence: float,
    min_ink_ratio: float,
    min_ink_pixels: int,
) -> List[Dict]:
    """
    RETAKE 조건을 체크하여 retake_reasons 리스트 반환

    Args:
        lens_conf: 렌즈 검출 신뢰도
        zone_results_raw: Zone 분석 결과
        print_band_area: 프린트 밴드 면적
        used_fallback_B: B zone fallback 사용 여부
        confidence: 전체 신뢰도

    Returns:
        retake_reasons 리스트
    """
    retake_reasons = []

    # R1_DetectionLow: 렌즈 검출 신뢰도 낮음
    if lens_conf < 0.7:
        retake_reasons.append(
            {
                "code": "R1_DetectionLow",
                "reason": "렌즈 검출 신뢰도 낮음",
                "actions": ["렌즈 중앙 정렬 확인", "반사 감소 (조명 각도 조정)", "초점 재조정"],
                "lever": "촬영",
            }
        )

    # R2_CoverageLow: 픽셀 커버리지 부족
    total_pixels = sum(zr["pixel_count"] for zr in zone_results_raw if zr["pixel_count"] is not None)
    min_total_pixels = int(print_band_area * 0.5)  # 50% 이상 필요
    if total_pixels < min_total_pixels:
        retake_reasons.append(
            {
                "code": "R2_CoverageLow",
                "reason": f"픽셀 커버리지 부족 ({total_pixels}/{min_total_pixels})",
                "actions": ["초점 재조정", "렌즈 위치 확인", "ROI 설정 검증"],
                "lever": "촬영",
            }
        )

    # R3_BoundaryUncertain: B zone fallback + 낮은 confidence
    insufficient_zones = [zr.get("zone_name") for zr in zone_results_raw if not zr.get("ink_sufficient", True)]
    if insufficient_zones:
        zones_label = ", ".join([z for z in insufficient_zones if z])
        retake_reasons.append(
            {
                "code": "R5_InkInsufficient",
                "reason": (
                    "ink pixels below threshold "
                    f"(min_ratio={min_ink_ratio:.2%}, min_pixels={min_ink_pixels}, zones={zones_label})"
                ),
                "actions": ["Check illumination and exposure", "Verify ink coverage in ROI", "Re-shoot image"],
                "lever": "capture",
            }
        )

    if used_fallback_B and confidence < 0.8:
        retake_reasons.append(
            {
                "code": "R3_BoundaryUncertain",
                "reason": "경계 탐지 불확실 (fallback 사용)",
                "actions": ["이미지 품질 향상", "조명 균일성 확인"],
                "lever": "촬영",
            }
        )

    # R4_UniformityLow: 각도 불균일 (촬영/반사 의심)
    max_std_l = max([zr.get("std_lab", [0])[0] for zr in zone_results_raw if zr.get("std_lab")], default=0.0)
    if max_std_l > 12.0:
        retake_reasons.append(
            {
                "code": "R4_UniformityLow",
                "reason": f"각도 불균일 높음 (std_L={max_std_l:.1f})",
                "actions": ["반사 감소 (조명 각도)", "렌즈 표면 이물질 제거", "재촬영"],
                "lever": "촬영",
            }
        )

    return retake_reasons


def _check_warning_conditions(
    zone_results_raw: List[Dict],
    used_fallback_B: bool,
    confidence: float,
) -> List[str]:
    """
    OK 상태에서 경고 조건을 체크하여 warning_reasons 리스트 반환

    Args:
        zone_results_raw: Zone 분석 결과
        used_fallback_B: B zone fallback 사용 여부
        confidence: 전체 신뢰도

    Returns:
        warning_reasons 리스트
    """
    warning_reasons = []

    max_std_l = max([zr.get("std_lab", [0])[0] for zr in zone_results_raw if zr.get("std_lab")], default=0.0)

    # 1. 각도 균일성 경고 (10.0~12.0)
    if 10.0 < max_std_l <= 12.0:
        warning_reasons.append(f"각도 균일성 경계값 근접 (std_L={max_std_l:.1f})")

    # 2. B zone fallback 경고
    if used_fallback_B and confidence < 0.85:
        warning_reasons.append("Zone B 경계 탐지 불확실 (fallback 사용)")

    # 3. Confidence 경계값 경고
    if 0.6 < confidence < 0.7:
        warning_reasons.append(f"신뢰도 경계값 근접 (confidence={confidence:.2f})")

    return warning_reasons


def _build_ok_context(zone_results_raw: List[Dict], confidence: float) -> List[str]:
    """
    OK/OK_WITH_WARNING 상태의 컨텍스트 정보 생성

    Args:
        zone_results_raw: Zone 분석 결과
        confidence: 전체 신뢰도

    Returns:
        ok_context 리스트
    """
    ok_context = []

    # 1. Zone별 여유도
    margins = []
    for zr in zone_results_raw:
        if zr["delta_e"] is not None and zr["threshold"] is not None:
            margin = zr["threshold"] - zr["delta_e"]
            margin_pct = (margin / zr["threshold"]) * 100
            margins.append(f"{zr['zone_name']}:ΔE={zr['delta_e']:.1f}(여유 {margin_pct:.0f}%)")

    if margins:
        ok_context.append(f"Zone 여유도: {', '.join(margins)}")

    # 2. 신뢰도
    ok_context.append(f"신뢰도: {confidence:.2f} ({'재촬영 불필요' if confidence >= 0.7 else '경계값'})")

    return ok_context


def _build_decision_trace_and_actions(
    judgment: str,
    ng_reasons: List[str],
    retake_reasons: List[Dict],
    warning_reasons: List[str],
    zones_all_ok: bool,
) -> Tuple[Dict, List[str]]:
    """
    decision_trace와 next_actions 생성

    Args:
        judgment: 최종 판정
        ng_reasons: NG 이유 리스트
        retake_reasons: RETAKE 이유 리스트
        warning_reasons: 경고 이유 리스트
        zones_all_ok: Zone 판정이 모두 OK인지 여부

    Returns:
        (decision_trace, next_actions) 튜플
    """
    decision_trace: Dict[str, Any] = {"final": judgment, "because": [], "overrides": None}
    next_actions: List[str] = []

    if judgment == "RETAKE":
        # RETAKE 이유 코드 수집
        decision_trace["because"] = [r["code"] for r in retake_reasons]
        if zones_all_ok:
            decision_trace["overrides"] = "zones_all_ok"
        # next_actions: RETAKE reasons의 actions 통합
        for r in retake_reasons:
            next_actions.extend(r["actions"])

    elif judgment == "NG":
        # NG 이유 수집
        decision_trace["because"] = ng_reasons
        # next_actions: 공정 조정 권장
        next_actions.append("공정 파라미터 확인 및 조정")
        next_actions.append("ΔE 초과 Zone의 색상 변화 방향 확인 (diff)")

    elif judgment == "OK_WITH_WARNING":
        # 경고 이유 수집
        decision_trace["because"] = warning_reasons
        decision_trace["overrides"] = None
        # next_actions: 예방 조치
        next_actions.append("경고 사항 모니터링")
        if "균일성" in ", ".join(warning_reasons):
            next_actions.append("조명 균일성 확인")
        if "fallback" in ", ".join(warning_reasons):
            next_actions.append("샘플 재촬영 권장")

    elif judgment == "OK":
        decision_trace["because"] = ["모든 Zone이 허용 범위 내"]
        next_actions.append("정상 판정 - 추가 조치 불필요")

    return decision_trace, next_actions


def _determine_judgment_with_retake(
    judgment: str,
    ng_reasons: List[str],
    zone_results_raw: List[Dict],
    zone_specs: List,
    confidence: float,
    lens_conf: float,
    print_band_area: int,
    min_ink_ratio: float,
    min_ink_pixels: int,
) -> Tuple[str, List[Dict], Dict, List[str], List[str], List[str], bool, bool]:
    """
    RETAKE 조건 체크, 판정 결정, decision_trace 생성

    Returns:
        judgment (updated), retake_reasons, decision_trace, next_actions, ok_context, warning_reasons,
        zones_all_ok, used_fallback_B
    """
    # used_fallback_B 계산
    B_selected_range = [zone_specs[1].r_start_norm, zone_specs[1].r_end_norm]
    used_fallback_B = abs(B_selected_range[0] - 0.38) < 0.01 and abs(B_selected_range[1] - 0.61) < 0.01

    # RETAKE 조건 체크
    retake_reasons = _check_retake_conditions(
        lens_conf=lens_conf,
        zone_results_raw=zone_results_raw,
        print_band_area=print_band_area,
        used_fallback_B=used_fallback_B,
        confidence=confidence,
        min_ink_ratio=min_ink_ratio,
        min_ink_pixels=min_ink_pixels,
    )

    # 판정 순서: RETAKE > NG > OK_WITH_WARNING > OK
    ok_context = []
    warning_reasons = []
    zones_all_ok = judgment == "OK"  # Zone 판정 결과 저장

    if retake_reasons:
        # RETAKE 우선
        judgment = "RETAKE"
        ng_reasons = [str(r["code"]) + ": " + str(r["reason"]) for r in retake_reasons]
    elif judgment == "NG":
        # NG 유지 (Zone 불량)
        pass
    elif judgment == "OK":
        # OK 상태에서 경고 조건 체크
        warning_reasons = _check_warning_conditions(
            zone_results_raw=zone_results_raw,
            used_fallback_B=used_fallback_B,
            confidence=confidence,
        )

        # 경고가 있으면 OK_WITH_WARNING
        if warning_reasons:
            judgment = "OK_WITH_WARNING"
            ok_context.append(f"경고: {', '.join(warning_reasons)}")

        # OK / OK_WITH_WARNING 공통 컨텍스트
        ok_context.extend(_build_ok_context(zone_results_raw=zone_results_raw, confidence=confidence))

    # decision_trace 및 next_actions 생성
    decision_trace, next_actions = _build_decision_trace_and_actions(
        judgment=judgment,
        ng_reasons=ng_reasons,
        retake_reasons=retake_reasons,
        warning_reasons=warning_reasons,
        zones_all_ok=zones_all_ok,
    )

    return (
        judgment,
        retake_reasons,
        decision_trace,
        next_actions,
        ok_context,
        warning_reasons,
        zones_all_ok,
        used_fallback_B,
    )


def _build_confidence_breakdown(confidence: float, confidence_factors: Any) -> Dict:
    """
    Confidence breakdown 생성

    Returns:
        confidence_breakdown dict
    """
    return {
        "overall": float(confidence),
        "factors": [
            {
                "name": "pixel_count",
                "weight": 0.30,
                "score": float(confidence_factors.pixel_count_score),
                "contribution": float(0.30 * confidence_factors.pixel_count_score),
                "status": (
                    "good"
                    if confidence_factors.pixel_count_score >= 0.9
                    else ("warning" if confidence_factors.pixel_count_score >= 0.7 else "poor")
                ),
                "description": "Zone별 픽셀 수 충분도",
            },
            {
                "name": "transition",
                "weight": 0.25,
                "score": float(confidence_factors.transition_score),
                "contribution": float(0.25 * confidence_factors.transition_score),
                "status": (
                    "good"
                    if confidence_factors.transition_score >= 0.9
                    else ("warning" if confidence_factors.transition_score >= 0.7 else "poor")
                ),
                "description": "전이 구간 제거 정도 (적을수록 좋음)",
            },
            {
                "name": "uniformity",
                "weight": 0.25,
                "score": float(confidence_factors.std_score),
                "contribution": float(0.25 * confidence_factors.std_score),
                "status": (
                    "good"
                    if confidence_factors.std_score >= 0.9
                    else ("warning" if confidence_factors.std_score >= 0.7 else "poor")
                ),
                "description": "각도 균일성 (std_L 낮을수록 좋음)",
            },
            {
                "name": "sector_uniformity",
                "weight": 0.10,
                "score": float(confidence_factors.sector_uniformity),
                "contribution": float(0.10 * confidence_factors.sector_uniformity),
                "status": (
                    "good"
                    if confidence_factors.sector_uniformity >= 0.9
                    else ("warning" if confidence_factors.sector_uniformity >= 0.7 else "poor")
                ),
                "description": "섹터 간 균일성",
            },
            {
                "name": "lens_detection",
                "weight": 0.10,
                "score": float(confidence_factors.lens_detection),
                "contribution": float(0.10 * confidence_factors.lens_detection),
                "status": (
                    "good"
                    if confidence_factors.lens_detection >= 0.9
                    else ("warning" if confidence_factors.lens_detection >= 0.7 else "poor")
                ),
                "description": "렌즈 검출 신뢰도",
            },
        ],
    }


def _build_analysis_summary(
    max_std_l: float,
    used_fallback_B: bool,
    confidence_factors: Any,
    confidence: float,
    zone_results_raw: List[Dict],
    print_band_area: int,
) -> Dict:
    """
    Analysis summary 생성

    Returns:
        analysis_summary dict
    """
    return {
        "uniformity": {
            "max_std_L": float(max_std_l),
            "threshold_retake": 12.0,
            "threshold_warning": 10.0,
            "status": "good" if max_std_l < 10.0 else ("warning" if max_std_l <= 12.0 else "poor"),
            "impact": (
                "정상" if max_std_l < 10.0 else ("OK_WITH_WARNING 트리거" if max_std_l <= 12.0 else "RETAKE 트리거")
            ),
        },
        "boundary_quality": {
            "B_zone_method": "fallback" if used_fallback_B else "auto_detected",
            "confidence_contribution": float(confidence_factors.transition_score),
            "status": "good" if not used_fallback_B else ("warning" if confidence >= 0.8 else "poor"),
            "impact": "정상" if not used_fallback_B else ("Confidence 페널티" if confidence < 0.85 else "경고"),
        },
        "coverage": {
            "total_pixels": int(sum(zr["pixel_count"] for zr in zone_results_raw if zr["pixel_count"] is not None)),
            "expected_min": int(print_band_area * 0.5),
            "status": (
                "good"
                if sum(zr["pixel_count"] for zr in zone_results_raw if zr["pixel_count"] is not None)
                >= print_band_area * 0.5
                else "poor"
            ),
            "impact": (
                "정상"
                if sum(zr["pixel_count"] for zr in zone_results_raw if zr["pixel_count"] is not None)
                >= print_band_area * 0.5
                else "RETAKE (R2_CoverageLow)"
            ),
        },
    }


def _build_risk_factors(
    max_std_l: float,
    sector_stats: Dict,
    used_fallback_B: bool,
    confidence: float,
    lens_conf: float,
    zone_results_raw: List[Dict],
    print_band_area: int,
) -> List[Dict[str, Any]]:
    """
    Risk factors 생성

    Returns:
        risk_factors list
    """
    risk_factors: List[Dict[str, Any]] = []

    # Uniformity 위험 요소
    if max_std_l > 12.0:
        risk_factors.append(
            {
                "category": "uniformity",
                "severity": "high",
                "message": f"각도 불균일 높음 (std_L={max_std_l:.1f}, 임계값=12.0)",
                "source": "R4_UniformityLow",
            }
        )
    elif max_std_l > 10.0:
        risk_factors.append(
            {
                "category": "uniformity",
                "severity": "medium",
                "message": f"각도 불균일 경계값 근접 (std_L={max_std_l:.1f}, 경고=10.0)",
                "source": "OK_WITH_WARNING 조건",
            }
        )

    # Sector Uniformity 위험 요소 (국부 불량)
    if sector_stats["enabled"]:
        max_sector_std = sector_stats["max_sector_std_L"]
        worst_zone = sector_stats["worst_zone"]

        if max_sector_std > 8.0:
            risk_factors.append(
                {
                    "category": "sector_uniformity",
                    "severity": "high",
                    "message": f"Zone {worst_zone} 섹터 간 편차 높음 (std_L={max_sector_std:.1f}, 임계값=8.0)",
                    "source": "sector_analysis",
                    "details": {
                        "zone": worst_zone,
                        "max_sector_std_L": float(max_sector_std),
                        "zone_stats": sector_stats["zone_stats"],
                    },
                }
            )
        elif max_sector_std > 5.0:
            risk_factors.append(
                {
                    "category": "sector_uniformity",
                    "severity": "medium",
                    "message": f"Zone {worst_zone} 섹터 간 편차 경계값 근접 (std_L={max_sector_std:.1f}, 경고=5.0)",
                    "source": "sector_analysis",
                    "details": {
                        "zone": worst_zone,
                        "max_sector_std_L": float(max_sector_std),
                        "zone_stats": sector_stats["zone_stats"],
                    },
                }
            )

    # Boundary 위험 요소
    if used_fallback_B:
        severity = "high" if confidence < 0.8 else "medium"
        risk_factors.append(
            {
                "category": "boundary",
                "severity": severity,
                "message": "Zone B 경계 자동 탐지 실패 (fallback 사용)",
                "source": "R3_BoundaryUncertain" if confidence < 0.8 else "경고",
            }
        )

    # Lens detection 위험 요소
    if lens_conf < 0.7:
        risk_factors.append(
            {
                "category": "lens_detection",
                "severity": "high",
                "message": f"렌즈 검출 신뢰도 낮음 (confidence={lens_conf:.2f})",
                "source": "R1_DetectionLow",
            }
        )

    # Coverage 위험 요소
    total_pixels = sum(zr["pixel_count"] for zr in zone_results_raw if zr["pixel_count"] is not None)
    min_total_pixels = int(print_band_area * 0.5)
    if total_pixels < min_total_pixels:
        risk_factors.append(
            {
                "category": "coverage",
                "severity": "high",
                "message": f"픽셀 커버리지 부족 ({total_pixels}/{min_total_pixels})",
                "source": "R2_CoverageLow",
            }
        )

    # Zone ΔE 위험 요소
    for zr in zone_results_raw:
        if not zr["is_ok"] and zr["delta_e"] is not None:
            severity = "high" if zr["delta_e"] > zr["threshold"] * 1.5 else "medium"
            risk_factors.append(
                {
                    "category": "zone_quality",
                    "severity": severity,
                    "message": f"Zone {zr['zone_name']} ΔE 초과 (ΔE={zr['delta_e']:.1f} > {zr['threshold']:.1f})",
                    "source": "NG 판정",
                }
            )

    return risk_factors


def _generate_analysis_summaries(
    confidence_factors: Any,
    zone_results_raw: List[Dict],
    used_fallback_B: bool,
    confidence: float,
    print_band_area: int,
    sector_stats: Dict,
    lens_conf: float,
) -> Tuple[Dict, Dict, List[Dict]]:
    """
    confidence_breakdown, analysis_summary, risk_factors 생성

    Returns:
        confidence_breakdown, analysis_summary, risk_factors
    """
    # max_std_l 계산
    max_std_l = max([zr.get("std_lab", [0])[0] for zr in zone_results_raw if zr.get("std_lab")], default=0.0)

    # 3개의 sub-함수 호출 (리팩토링됨)
    confidence_breakdown = _build_confidence_breakdown(confidence, confidence_factors)

    analysis_summary = _build_analysis_summary(
        max_std_l, used_fallback_B, confidence_factors, confidence, zone_results_raw, print_band_area
    )

    risk_factors = _build_risk_factors(
        max_std_l, sector_stats, used_fallback_B, confidence, lens_conf, zone_results_raw, print_band_area
    )

    return confidence_breakdown, analysis_summary, risk_factors


def _perform_ink_analysis(
    img_bgr: np.ndarray, zone_results_raw: List[Dict], zone_specs: List, sku_config: dict, used_fallback_B: bool
) -> Dict:
    """
    ink_analysis 생성 (zone_based + image_based)

    Returns:
        ink_analysis (dict)
    """
    print("=" * 80)
    print("[INK_ANALYSIS_START] Starting ink count analysis with pixel ratio filtering")
    print("=" * 80)

    position_map = {"C": "inner", "B": "middle", "A": "outer"}

    # 🔧 FIX: Zone ≠ Ink. 충분한 잉크 픽셀이 있는 Zone만 잉크로 인정
    MIN_INK_PIXEL_RATIO = 0.05  # 전체 잉크 픽셀의 5% 이상이어야 잉크로 간주

    # 전체 잉크 픽셀 수 계산 (pixels_ink 기준)
    total_ink_pixels = sum(zr["pixel_count_ink"] for zr in zone_results_raw if zr["pixel_count_ink"])
    print(f"[INK_ANALYSIS] Total ink pixels across all zones: {total_ink_pixels}")

    # Zone 순서: C (inner) → B (middle) → A (outer)
    # Ink 번호: 실제 잉크가 있는 Zone만 카운트
    inks_zone = []
    all_zones = []  # 모든 Zone 정보 (잉크 여부 무관)

    ink_num = 1
    for zr, zspec in zip(zone_results_raw, zone_specs):
        # 잉크 픽셀 비율 계산 (전체 잉크 픽셀 대비)
        ink_pixel_ratio = zr["pixel_count_ink"] / total_ink_pixels if total_ink_pixels > 0 else 0

        # LAB → RGB/HEX 변환 (색상 표시용)
        measured_rgb = None
        measured_hex = None
        if zr["measured_lab"] is not None:
            try:
                measured_lab_tuple = (
                    float(zr["measured_lab"][0]),
                    float(zr["measured_lab"][1]),
                    float(zr["measured_lab"][2]),
                )
                measured_rgb, measured_hex = lab_to_rgb_hex(measured_lab_tuple)
            except Exception as e:
                print(f"[ZONE_COLOR] Failed to convert LAB to RGB for {zr['zone_name']}: {e}")

        # Zone 정보 구성
        zone_info = {
            "zone_name": zr["zone_name"],
            "position": position_map.get(zr["zone_name"], "unknown"),
            "radial_range": [float(zspec.r_start_norm), float(zspec.r_end_norm)],
            "measured_color": {
                "L": float(zr["measured_lab"][0]) if zr["measured_lab"] else None,
                "a": float(zr["measured_lab"][1]) if zr["measured_lab"] else None,
                "b": float(zr["measured_lab"][2]) if zr["measured_lab"] else None,
                "rgb": list(measured_rgb) if measured_rgb else None,
                "hex": measured_hex,
            },
            "reference_color": {
                "L": float(zr["target_lab"][0]),
                "a": float(zr["target_lab"][1]),
                "b": float(zr["target_lab"][2]),
            },
            "delta_e": float(zr["delta_e"]) if zr["delta_e"] is not None else None,
            "is_within_spec": zr["is_ok"],
            "pixel_count": zr["pixel_count"],
            "pixel_count_ink": zr["pixel_count_ink"],
            "ink_pixel_ratio": float(ink_pixel_ratio),
        }

        # 모든 Zone 저장
        all_zones.append(zone_info)

        # 충분한 잉크 픽셀이 있는 Zone만 잉크로 카운트
        if ink_pixel_ratio >= MIN_INK_PIXEL_RATIO:
            ink_info = zone_info.copy()
            ink_info["ink_number"] = ink_num
            inks_zone.append(ink_info)
            ink_num += 1
            print(
                f"[INK_ZONE] Zone {zr['zone_name']} counted as ink "
                f"(ink_ratio={ink_pixel_ratio:.2%}, ink_pixels={zr['pixel_count_ink']})"
            )
        else:
            print(
                f"[INK_ZONE] Zone {zr['zone_name']} excluded "
                f"(ink_ratio={ink_pixel_ratio:.4%} < {MIN_INK_PIXEL_RATIO:.0%}, ink_pixels={zr['pixel_count_ink']})"
            )

    # Image-based 분석 (InkEstimator - GMM + BIC)
    inks_image = []
    image_based_meta = {}
    try:
        estimator = InkEstimator(random_seed=42)
        result = estimator.estimate_from_array(
            img_bgr, k_max=3, chroma_thresh=6.0, L_max=98.0, merge_de_thresh=5.0, linearity_thresh=3.0
        )
        inks_image = result.get("inks", [])
        image_based_meta = result.get("meta", {})

        print(f"[INK_ESTIMATOR] Detected {result.get('ink_count', 0)} inks (GMM-based)")
        if image_based_meta.get("correction_applied"):
            print("[INK_ESTIMATOR] Mixing correction applied (3→2 inks)")
    except Exception as e:
        print(f"[INK_ESTIMATOR] Failed to run image-based analysis: {e}")
        inks_image = []
        image_based_meta = {"error": str(e)}

    # 통합 ink_analysis 구조
    ink_analysis = {
        "zone_based": {
            "detected_ink_count": len(inks_zone),  # 🔧 FIX: 실제 잉크만 카운트 (pixel_ratio >= 5%)
            "detection_method": "fallback" if used_fallback_B else "transition_based",
            "expected_ink_count": sku_config.get("params", {}).get("expected_zones"),
            "inks": inks_zone,  # 잉크로 인정된 Zone만
            "all_zones": all_zones,  # 모든 Zone 정보 (참고용)
            "filter_threshold": MIN_INK_PIXEL_RATIO,  # 필터링 기준값
        },
        "image_based": {
            "detected_ink_count": len(inks_image),
            "detection_method": "gmm_bic",
            "inks": inks_image,
            "meta": image_based_meta,
        },
    }

    print(
        f"[INK_ANALYSIS] Zone-based: {len(inks_zone)} inks (from {len(all_zones)} zones), "
        f"Image-based: {len(inks_image)} inks"
    )

    return ink_analysis


def _assemble_result(
    judgment: str,
    overall_de: float,
    zone_results: List,
    ng_reasons: List[str],
    confidence: float,
    decision_trace: Dict,
    next_actions: List[str],
    retake_reasons: List[Dict],
    analysis_summary: Dict,
    confidence_breakdown: Dict,
    risk_factors: List[Dict],
    ink_analysis: Dict,
    radial_profile_dict: Optional[Dict],  # P1-2: Radial profile data
    sku_config: dict,
    transition_ranges: List,
    zone_specs: List,
    used_fallback_B: bool,
    ok_context: List[str],
    print_outer: float,
) -> Tuple[InspectionResult, Dict]:
    """
    InspectionResult 생성 및 debug_info 구성

    Returns:
        result, debug_info
    """
    # InspectionResult 생성
    result = InspectionResult(
        sku=sku_config.get("sku_code", "unknown"),
        timestamp=datetime.now(),
        judgment=judgment,
        overall_delta_e=overall_de if overall_de is not None else 0.0,
        zone_results=zone_results,
        ng_reasons=ng_reasons,
        confidence=max(0.0, min(1.0, confidence)),
        decision_trace=decision_trace,  # 운영 UX
        next_actions=next_actions if next_actions else None,  # 운영 UX
        retake_reasons=retake_reasons if retake_reasons else None,  # 운영 UX
        analysis_summary=analysis_summary,  # 운영 UX: 프로파일 요약
        confidence_breakdown=confidence_breakdown,  # 운영 UX: Confidence 분해
        risk_factors=risk_factors if risk_factors else None,  # 운영 UX: 위험 요소
        ink_analysis=ink_analysis,  # 사용자 목표: 잉크 색 도출
        radial_profile=radial_profile_dict,  # P1-2: Radial profile for comparison
    )

    # 디버그 정보 저장 (JSON 출력용)
    # B_candidate_range
    B_candidate_range = [0.33, 0.66]

    # excluded_ratio 계산
    total_excluded = sum(
        tr.r_end - tr.r_start
        for tr in transition_ranges
        if tr.r_start >= B_candidate_range[0] and tr.r_end <= B_candidate_range[1]
    )
    B_candidate_width = B_candidate_range[1] - B_candidate_range[0]
    excluded_ratio_B = total_excluded / B_candidate_width if B_candidate_width > 0 else 0.0

    # B_selected_range
    B_selected_range = [zone_specs[1].r_start_norm, zone_specs[1].r_end_norm]

    debug_info = {
        "B_candidate_range": B_candidate_range,
        "transition_ranges": [[tr.r_start, tr.r_end, tr.max_gradient] for tr in transition_ranges],
        "B_selected_range": B_selected_range,
        "excluded_ratio_B": float(excluded_ratio_B),
        "used_fallback_B": used_fallback_B,
        "retake_reasons": retake_reasons if retake_reasons else None,  # 운영 UX
        "ok_context": ok_context if ok_context else None,  # OK 상태 컨텍스트
    }

    # InspectionResult에 debug_info 저장 (임시 속성)
    result._debug_info = debug_info  # type: ignore
    print(f"[2D ZONE ANALYSIS] Debug info attached: {list(debug_info.keys())}")
    print(
        f"[2D ZONE ANALYSIS] used_fallback_B={debug_info['used_fallback_B']}, B_range={debug_info['B_selected_range']}"
    )

    overall_de_val = overall_de if overall_de is not None else 0.0
    print(f"[2D ZONE ANALYSIS] Complete: {judgment}, ΔE={overall_de_val:.2f}, confidence={confidence:.2f}")

    return result, debug_info


def analyze_lens_zones_2d(
    img_bgr: np.ndarray,
    lens_detection: LensDetection,
    sku_config: Dict[str, Any],
    ink_mask_config: Optional[InkMaskConfig] = None,
    std_ref_image: Optional[np.ndarray] = None,
    std_ref_lens: Optional[Any] = None,
    mask_source: Optional[str] = None,
    save_debug: bool = False,
    save_debug_on: Optional[List[str]] = None,
    debug_low_confidence: float = 0.75,
    debug_prefix: str = "debug_2d",
) -> Tuple[InspectionResult, Dict[str, Any]]:
    """
    2D 이미지 기반 Zone 분석 (AI 템플릿)

    Args:
        img_bgr: OpenCV BGR 이미지
        lens_detection: 렌즈 검출 결과
        sku_config: SKU 설정 (zones, thresholds, params)
        ink_mask_config: ink mask 설정
        save_debug: 디버그 이미지 저장 여부
        debug_prefix: 디버그 파일 prefix

    Returns:
        InspectionResult, debug_info
    """
    print("[2D ZONE ANALYSIS] Starting...")

    h, w = img_bgr.shape[:2]
    params = sku_config.get("params", {})
    mask_policy = params.get("mask_policy", {})
    min_ink_ratio = float(mask_policy.get("min_ink_ratio_2d", 0.05))
    min_ink_pixels = int(mask_policy.get("min_ink_pixels_2d", 500))
    if mask_source is None:
        mask_source = mask_policy.get("zone_2d_mask_source")
    debug_low_confidence = float(params.get("debug_low_confidence_threshold", debug_low_confidence))

    # 1. Setup masks and boundaries
    lens_mask, ink_mask, cx, cy, r_lens, print_inner, print_outer = _setup_masks_and_boundaries(
        img_bgr,
        lens_detection,
        sku_config,
        ink_mask_config,
        std_ref_image=std_ref_image,
        std_ref_lens=std_ref_lens,
        mask_source=mask_source,
    )

    # 2. Detect transitions and zones
    target_labs, thresholds, transition_ranges, zone_specs, print_band_area = _detect_transitions_and_zones(
        img_bgr, cx, cy, print_inner, print_outer, lens_mask, sku_config
    )

    # 3. Compute zone results
    judgment, overall_de, zone_results, zone_results_raw, ng_reasons, zone_masks, sector_stats = (
        _compute_zone_results_wrapper(
            img_bgr,
            h,
            w,
            cx,
            cy,
            print_inner,
            print_outer,
            lens_mask,
            ink_mask,
            zone_specs,
            target_labs,
            thresholds,
            min_ink_ratio,
            min_ink_pixels,
        )
    )

    # 4. Calculate confidence
    confidence, confidence_factors, lens_conf, sector_uniformity_score = _calculate_confidence_wrapper(
        zone_results_raw, transition_ranges, lens_detection, sector_stats, zone_specs, print_band_area
    )

    # 5. Determine judgment with RETAKE logic
    (
        judgment,
        retake_reasons,
        decision_trace,
        next_actions,
        ok_context,
        warning_reasons,
        zones_all_ok,
        used_fallback_B,
    ) = _determine_judgment_with_retake(
        judgment,
        ng_reasons,
        zone_results_raw,
        zone_specs,
        confidence,
        lens_conf,
        print_band_area,
        min_ink_ratio,
        min_ink_pixels,
    )

    # 6. Generate analysis summaries
    confidence_breakdown, analysis_summary, risk_factors = _generate_analysis_summaries(
        confidence_factors, zone_results_raw, used_fallback_B, confidence, print_band_area, sector_stats, lens_conf
    )

    # 7. Perform ink analysis
    ink_analysis = _perform_ink_analysis(img_bgr, zone_results_raw, zone_specs, sku_config, used_fallback_B)

    # 7.5. Extract radial profile (P1-2)
    radial_profile_dict = None
    try:
        from src.core.radial_profiler import ProfilerConfig, RadialProfiler

        profiler = RadialProfiler(ProfilerConfig())
        radial_profile = profiler.extract_profile(img_bgr, lens_detection)

        # Convert to dict for JSON serialization
        radial_profile_dict = {
            "r_normalized": radial_profile.r_normalized.tolist(),
            "L": radial_profile.L.tolist(),
            "a": radial_profile.a.tolist(),
            "b": radial_profile.b.tolist(),
            "std_L": radial_profile.std_L.tolist(),
            "std_a": radial_profile.std_a.tolist(),
            "std_b": radial_profile.std_b.tolist(),
            "pixel_count": radial_profile.pixel_count.tolist(),
        }
        print(f"[RADIAL_PROFILE] Extracted profile with {len(radial_profile.L)} points")
    except Exception as e:
        print(f"[RADIAL_PROFILE] Failed to extract profile: {e}")

    # 8. Assemble result
    result, debug_info = _assemble_result(
        judgment,
        overall_de,
        zone_results,
        ng_reasons,
        confidence,
        decision_trace,
        next_actions,
        retake_reasons,
        analysis_summary,
        confidence_breakdown,
        risk_factors,
        ink_analysis,
        radial_profile_dict,  # P1-2: Add radial profile
        sku_config,
        transition_ranges,
        zone_specs,
        used_fallback_B,
        ok_context,
        print_outer,
    )

    # 9. Save debug images (optional)
    should_save_debug = save_debug
    if not should_save_debug and save_debug_on:
        if "retake" in save_debug_on and judgment == "RETAKE":
            should_save_debug = True
        if "ng" in save_debug_on and judgment == "NG":
            should_save_debug = True
        if "low_confidence" in save_debug_on and confidence < debug_low_confidence:
            should_save_debug = True

    if should_save_debug:
        try:
            _save_debug_images(img_bgr, zone_masks, ink_mask, lens_mask, debug_prefix, ink_analysis)
        except Exception as e:
            print(f"Failed to save debug images: {e}")

    return result, debug_info


def _save_debug_images(
    img_bgr: np.ndarray,
    zone_masks: Dict[str, np.ndarray],
    ink_mask: np.ndarray,
    lens_mask: np.ndarray,
    prefix: str,
    ink_analysis: Optional[Dict] = None,
) -> None:
    """디버그 이미지 저장 (색상 팔레트 포함)"""
    # Zone overlay
    colors = {"A": (0, 0, 255), "B": (0, 255, 255), "C": (255, 0, 0)}
    out = img_bgr.copy()

    for zn, m in zone_masks.items():
        overlay = img_bgr.copy()
        overlay[m > 0] = colors.get(zn, (255, 255, 255))
        out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)

    cv2.imwrite(f"{prefix}_zones.png", out)

    # Ink overlay (with color palette)
    out_ink = img_bgr.copy()
    overlay_ink = img_bgr.copy()
    overlay_ink[ink_mask > 0] = (0, 255, 0)
    out_ink = cv2.addWeighted(overlay_ink, 0.3, out_ink, 0.7, 0)

    # 🎨 D) 색상 팔레트를 이미지 상단에 추가
    if ink_analysis and ink_analysis.get("image_based", {}).get("inks"):
        inks = ink_analysis["image_based"]["inks"]
        h, w = out_ink.shape[:2]

        # 팔레트 영역 생성 (상단 80px)
        palette_height = 100
        palette_img = np.ones((palette_height, w, 3), dtype=np.uint8) * 240  # 밝은 회색 배경

        # 각 잉크 색상 칩 그리기
        num_inks = len(inks)
        chip_width = min(150, w // (num_inks + 1))
        chip_height = 60
        margin = 10
        start_x = (w - (chip_width * num_inks + margin * (num_inks - 1))) // 2

        for i, ink in enumerate(inks):
            rgb = ink.get("rgb", [128, 128, 128])
            hex_val = ink.get("hex", "#808080")
            weight = ink.get("weight", 0.0)
            lab = ink.get("lab", [0, 0, 0])

            # BGR로 변환 (OpenCV는 BGR 사용)
            bgr_color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))

            # 색상 칩 위치
            x1 = start_x + i * (chip_width + margin)
            y1 = 20
            x2 = x1 + chip_width
            y2 = y1 + chip_height

            # 색상 박스 그리기
            cv2.rectangle(palette_img, (x1, y1), (x2, y2), bgr_color, -1)
            cv2.rectangle(palette_img, (x1, y1), (x2, y2), (100, 100, 100), 2)

            # 텍스트 색상 결정 (밝기 기준)
            text_color = (0, 0, 0) if lab[0] > 50 else (255, 255, 255)

            # 텍스트 추가
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1

            # Ink 번호
            cv2.putText(
                palette_img,
                f"Ink {i+1}",
                (x1 + 5, y1 + 15),
                font,
                font_scale,
                text_color,
                thickness,
            )

            # HEX 코드
            cv2.putText(
                palette_img,
                hex_val,
                (x1 + 5, y1 + 35),
                font,
                font_scale,
                text_color,
                thickness,
            )

            # 비중
            cv2.putText(
                palette_img,
                f"{weight*100:.1f}%",
                (x1 + 5, y1 + 55),
                font,
                font_scale,
                text_color,
                thickness,
            )

        # 팔레트와 원본 이미지 합치기
        out_ink = np.vstack([palette_img, out_ink])

    cv2.imwrite(f"{prefix}_ink.png", out_ink)

    print(f"Debug images saved: {prefix}_zones.png, {prefix}_ink.png")
