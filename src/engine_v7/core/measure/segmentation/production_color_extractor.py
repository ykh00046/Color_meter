"""
콘택트렌즈 색상 추출 시스템 - Production v1.0.7

v1.0.7 → v1.0.7 배경 ΔE 기반 필터 (최종 안전화):
- 🔴 bg_leak = (ΔE<8) & (chroma<15) AND 조건
- 🔴 ROI 주변 링 기반 bg_lab 추정 (1.02~1.08R)
- 🔴 specular vs bg_leak 경고 분리
- 🟡 디자인 색 보호 + 배경 누출 정확 제거

Production Ready - Final Safeguards
"""

import json
import os
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from skimage import color as skcolor
from skimage import morphology
from sklearn.cluster import KMeans

# ========================================
# 색상 데이터 정의
# ========================================

COLOR_NAME_KR = {
    "Sky Blue": "스카이블루",
    "Blue": "블루",
    "Navy": "네이비",
    "Cyan/Turquoise": "청록/터콰이즈",
    "Beige/Cream": "베이지/크림",
    "Beige/Gold": "베이지골드",
    "Orange/Beige": "오렌지베이지",
    "Brown": "브라운",
    "Orange/Brown": "오렌지브라운",
    "Dark Brown": "다크브라운",
    "Green": "그린",
    "Olive": "올리브",
    "Green/Yellow": "연두",
    "Gold/Yellow": "골드/옐로우",
    "Purple/Violet": "퍼플/바이올렛",
    "Pink/Magenta": "핑크/마젠타",
    "Red/Wine": "레드/와인",
    "Gray": "그레이",
    "Light Gray": "라이트그레이",
    "Black/Dark": "블랙/다크",
    "Other": "기타",
}

COLOR_GROUPS = {
    "Cool Tone": ["Sky Blue", "Blue", "Navy", "Cyan/Turquoise", "Purple/Violet", "Gray", "Light Gray"],
    "Warm Tone": [
        "Beige/Cream",
        "Beige/Gold",
        "Orange/Beige",
        "Brown",
        "Orange/Brown",
        "Dark Brown",
        "Gold/Yellow",
        "Red/Wine",
    ],
    "Green Tone": ["Green", "Olive", "Green/Yellow"],
    "Neutral": ["Black/Dark", "Gray", "Light Gray"],
    "Natural": ["Beige/Cream", "Brown", "Dark Brown", "Orange/Beige"],
    "Vivid": ["Sky Blue", "Blue", "Green", "Purple/Violet", "Pink/Magenta", "Gold/Yellow"],
}


def load_image(path):
    """이미지 로드 및 RGB 변환"""
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def trimmed_mean_lab(lab_pixels, trim_percent=10):
    """
    C) ΔE 거리 기반 trimmed_mean (개선)

    채널별 정렬이 아닌 픽셀 단위 outlier 제거:
    - median Lab 기준 거리(≈ΔE) 큰 순서로 trim
    - Lab 벡터 상관관계 유지

    Parameters:
    - trim_percent: 제거할 비율 (기본 10%)

    Returns:
    - trimmed_mean_lab: outlier 제거 후 평균
    """
    n = len(lab_pixels)
    if n < 10:
        return lab_pixels.mean(axis=0)

    k = int(n * trim_percent / 100)
    if k == 0 or 2 * k >= n:
        return lab_pixels.mean(axis=0)

    # median Lab 기준 거리 계산 (ΔE 근사)
    med = np.median(lab_pixels, axis=0)
    distances = np.linalg.norm(lab_pixels - med, axis=1)

    # 거리가 먼 k개씩 제거
    keep_indices = np.argsort(distances)[k : n - k]

    return lab_pixels[keep_indices].mean(axis=0)


def get_dynamic_background_threshold(gray, percentile_level=95, delta=5):
    """P2: 동적 배경 임계값 계산"""
    bg_level = np.percentile(gray, percentile_level)
    bg_threshold = max(bg_level - delta, 235)
    return bg_threshold


def extract_lens_roi(image, fg_mask, method="largest_component"):
    """Step 1: 렌즈 ROI 추출 (먼지/그림자 제거)"""
    h, w = image.shape[:2]

    if method == "largest_component":
        fg_mask_uint8 = fg_mask.astype(np.uint8) * 255

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask_uint8, connectivity=8)

        if num_labels < 2:
            return fg_mask, {"method": method, "status": "no_component"}

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        roi_mask = labels == largest_label

        roi_info = {
            "method": method,
            "num_components": num_labels - 1,
            "largest_area": stats[largest_label, cv2.CC_STAT_AREA],
            "center": centroids[largest_label].tolist(),
            "bbox": stats[largest_label, :4].tolist(),
        }

    elif method == "circular":
        fg_mask_uint8 = fg_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(fg_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return fg_mask, {"method": method, "status": "no_contour"}

        largest_contour = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)

        canvas = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(canvas, (int(cx), int(cy)), int(radius * 0.95), 1, -1)
        roi_mask = canvas.astype(bool)

        roi_info = {
            "method": method,
            "center": (float(cx), float(cy)),
            "radius": float(radius),
            "area": float(np.pi * radius**2),
        }
    else:
        roi_mask = fg_mask
        roi_info = {"method": "none"}

    return roi_mask, roi_info


def rgb_to_lab_scaled(rgb_pixels, l_weight=0.3):
    """Step 2: RGB를 Lab로 변환하고 L 채널 가중치 조정"""
    l_weight = max(l_weight, 0.05)
    rgb_normalized = rgb_pixels / 255.0
    lab = skcolor.rgb2lab(rgb_normalized.reshape(-1, 1, 3)).reshape(-1, 3)

    lab_scaled = lab.copy()
    lab_scaled[:, 0] = lab[:, 0] * l_weight

    return lab_scaled, lab


def lab_to_rgb(lab_values):
    """Lab 값을 RGB로 역변환"""
    lab_reshaped = lab_values.reshape(-1, 1, 3)
    rgb_normalized = skcolor.lab2rgb(lab_reshaped).reshape(-1, 3)
    rgb = (rgb_normalized * 255).clip(0, 255)
    return rgb


def calculate_lab_chroma(lab_values):
    """Lab chroma 계산: sqrt(a^2 + b^2)"""
    a = lab_values[:, 1]
    b = lab_values[:, 2]
    chroma = np.sqrt(a**2 + b**2)
    return chroma


def filter_highlights_spatial(
    lab_img,  # (H,W,3) Lab 이미지
    roi_mask,  # (H,W) bool 마스크
    core_L=97,  # 완전 흰 핵 기준
    core_chroma=12,  # 핵은 chroma가 낮음
    halo_L=70,  # 번짐(halo)까지 제거할 밝기
    halo_dilate_ratio=0.06,  # 렌즈 반경 대비 확장 비율
    max_dilate_px=40,
    min_dilate_px=6,
    bg_delta_e_threshold=8.0,  # v1.0.7: 배경 유사도 ΔE 임계값
    bg_chroma_threshold=15.0,  # v1.0.7: 배경 누출 chroma 임계값
):
    """
    v1.0.7: 배경 ΔE 기반 하이라이트 제거 (최종 안전화)

    코어 + 헤일로 + 배경 유사도(ΔE & chroma) AND 조건

    Parameters:
    - lab_img: (H,W,3) Lab 이미지
    - roi_mask: (H,W) ROI 마스크
    - core_L: 코어 밝기 임계값 (기본 97)
    - core_chroma: 코어 채도 임계값 (기본 12)
    - halo_L: 헤일로 밝기 임계값 (기본 70)
    - halo_dilate_ratio: 팽창 비율
    - bg_delta_e_threshold: 배경 유사도 ΔE 임계값 (기본 8.0)
    - bg_chroma_threshold: 배경 누출 chroma 임계값 (기본 15.0)

    Returns:
    - highlight_2d: (H,W) bool 하이라이트 마스크
    - info: 코어/헤일로/배경누출 정보
    """
    L = lab_img[..., 0]
    a = lab_img[..., 1]
    b = lab_img[..., 2]
    chroma = np.sqrt(a * a + b * b)

    # 렌즈 반경 추정
    area = float(np.sum(roi_mask))
    radius = np.sqrt(max(area, 1.0) / np.pi)
    dilate_px = int(np.clip(radius * halo_dilate_ratio, min_dilate_px, max_dilate_px))

    # 1) 코어(완전 흰 핵)
    core = roi_mask & (L >= core_L) & (chroma <= core_chroma)

    # 2) 헤일로(코어 주변 번짐)
    core_dilated = morphology.binary_dilation(core, morphology.disk(dilate_px))
    halo = core_dilated & roi_mask & (L >= halo_L)

    # 3) v1.0.7: ROI 주변 링 기반 bg_lab 추정
    h, w = lab_img.shape[:2]
    cy, cx = h // 2, w // 2

    # ROI 중심 찾기
    roi_coords = np.argwhere(roi_mask)
    if len(roi_coords) > 0:
        cy, cx = roi_coords.mean(axis=0).astype(int)

    # 거리 맵 생성
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)

    # ROI 주변 링 (1.02R ~ 1.08R)
    bg_band = (dist_from_center > 1.02 * radius) & (dist_from_center < 1.08 * radius) & ~roi_mask

    # 배경 Lab 추정
    if np.sum(bg_band) > 100:
        bg_lab = np.median(lab_img[bg_band], axis=0)
    else:
        # 링이 충분하지 않으면 전체 배경 사용
        bg_mask = ~roi_mask
        if np.sum(bg_mask) > 100:
            bg_lab = np.median(lab_img[bg_mask], axis=0)
        else:
            bg_lab = np.array([100, 0, 0])  # 기본값 (흰색)

    # 4) v1.0.7: bg_leak = (ΔE < threshold) & (chroma < threshold)
    delta_e = np.sqrt((L - bg_lab[0]) ** 2 + (a - bg_lab[1]) ** 2 + (b - bg_lab[2]) ** 2)

    bg_leak = roi_mask & (delta_e < bg_delta_e_threshold) & (chroma < bg_chroma_threshold)

    highlight_2d = core | halo | bg_leak

    info = {
        "core_pixels": int(core.sum()),  # Core 영역 크기 (중복 포함)
        "halo_pixels": int(halo.sum()),  # Halo 영역 크기 (core 포함)
        "bg_leak_pixels": int(bg_leak.sum()),  # BG leak 영역 크기 (독립)
        "dilate_px": int(dilate_px),
        "core_L": float(core_L),
        "halo_L": float(halo_L),
        "core_chroma": float(core_chroma),
        "bg_delta_e_threshold": float(bg_delta_e_threshold),
        "bg_chroma_threshold": float(bg_chroma_threshold),
        "bg_lab": bg_lab.tolist(),  # 배경 Lab 기록
        "bg_band_pixels": int(np.sum(bg_band)),  # 링 픽셀 수
    }

    return highlight_2d, info


def filter_highlights(lab_pixels, l_threshold=92, chroma_threshold=6, use_dynamic_chroma=False):
    """
    [DEPRECATED] Legacy 1D 하이라이트 필터 - v1.0.7에서 미사용

    v1.0.7부터 filter_highlights_spatial()로 완전 대체됨
    이 함수는 하위 호환성을 위해 유지되나 실제로는 사용되지 않음

    Parameters:
    - use_dynamic_chroma: True면 더 공격적으로 제거 (threshold 상승)

    Returns:
    - non_highlight_mask, highlight_mask, actual_chroma_threshold, dynamic_info
    """
    L = lab_pixels[:, 0]
    chroma = calculate_lab_chroma(lab_pixels)

    dynamic_info = {}

    # 성능 3: 동적 chroma 로그 보강
    if use_dynamic_chroma:
        dynamic_percentile_value = np.percentile(chroma, 5)
        actual_chroma_threshold = max(chroma_threshold, dynamic_percentile_value)
        dynamic_info["dynamic_chroma_percentile_value"] = float(dynamic_percentile_value)
        dynamic_info["threshold_increased"] = actual_chroma_threshold > chroma_threshold
    else:
        actual_chroma_threshold = chroma_threshold

    highlight_mask = (L > l_threshold) & (chroma < actual_chroma_threshold)
    non_highlight_mask = ~highlight_mask

    return non_highlight_mask, highlight_mask, actual_chroma_threshold, dynamic_info


def calculate_dynamic_low_chroma_samples(roi_filtered_pixels, base_samples=2000, ratio=0.05):
    """
    성능 2: ROI 크기 기반 동적 low_chroma_samples 계산

    Parameters:
    - base_samples: 기본값 (최대)
    - ratio: ROI 대비 비율 (5%)

    Returns:
    - samples: 동적으로 계산된 샘플 수
    """
    dynamic_samples = int(ratio * roi_filtered_pixels)
    samples = min(base_samples, dynamic_samples)
    return max(samples, 500)  # 최소 500


def sample_pixels_two_band(rgb_pixels, lab_pixels, high_chroma_percentile=40, low_chroma_samples=2000, rng=None):
    """
    Risk 1: 2-밴드 샘플링 (연한 색 보존)

    Returns:
    - sampled_rgb, sampled_lab, combined_mask, threshold, band_info
    """
    if rng is None:
        rng = np.random.default_rng(42)

    chroma = calculate_lab_chroma(lab_pixels)
    threshold = np.percentile(chroma, high_chroma_percentile)

    # Band A: 상위 chroma
    high_chroma_mask = chroma >= threshold
    high_count = np.sum(high_chroma_mask)

    # Band B: 하위 chroma 샘플링
    low_chroma_mask = chroma < threshold
    low_chroma_indices = np.where(low_chroma_mask)[0]

    if len(low_chroma_indices) > low_chroma_samples:
        selected_low = rng.choice(low_chroma_indices, low_chroma_samples, replace=False)
        low_chroma_sample_mask = np.zeros(len(lab_pixels), dtype=bool)
        low_chroma_sample_mask[selected_low] = True
        low_count = low_chroma_samples
    else:
        low_chroma_sample_mask = low_chroma_mask
        low_count = len(low_chroma_indices)

    # 통합
    combined_mask = high_chroma_mask | low_chroma_sample_mask

    sampled_rgb = rgb_pixels[combined_mask]
    sampled_lab = lab_pixels[combined_mask]

    band_info = {
        "high_band_count": int(high_count),
        "low_band_count": int(low_count),
        "total_sampled": int(np.sum(combined_mask)),
    }

    return sampled_rgb, sampled_lab, combined_mask, threshold, band_info


def calculate_delta_e(lab1, lab2):
    """ΔE (CIE76) 계산"""
    return np.sqrt(np.sum((lab1 - lab2) ** 2))


def merge_similar_clusters(cluster_centers_lab, labels, delta_e_threshold=4.0, min_percentage=1.5):
    """
    D) ΔE 기반 클러스터 병합 (개선: 병합 후 크기 재계산)

    Returns:
    - merged_labels, merge_map, merge_info
    """
    n_clusters = len(cluster_centers_lab)
    total_pixels = len(labels)

    cluster_sizes = np.array([np.sum(labels == i) for i in range(n_clusters)])
    cluster_percentages = (cluster_sizes / total_pixels) * 100

    merge_map = {i: i for i in range(n_clusters)}
    delta_e_merges = 0

    # ΔE 기반 병합
    for i in range(n_clusters):
        if merge_map[i] != i:
            continue

        for j in range(i + 1, n_clusters):
            if merge_map[j] != j:
                continue

            delta_e = calculate_delta_e(cluster_centers_lab[i], cluster_centers_lab[j])

            if delta_e < delta_e_threshold:
                merge_map[j] = i
                delta_e_merges += 1

    # D) ΔE 병합 후 크기 재계산 (중요!)
    tmp_labels = np.array([merge_map[label] for label in labels])
    roots, counts = np.unique(tmp_labels, return_counts=True)
    root_sizes = dict(zip(roots, counts))
    root_percentages = {r: (c / total_pixels) * 100 for r, c in root_sizes.items()}

    # 작은 클러스터 병합 (재계산된 root_percentages 사용)
    small_merges = 0
    for i in range(n_clusters):
        root_i = merge_map[i]

        # 이미 다른 곳으로 병합됨
        if root_i != i:
            continue

        # root 기준 비율 확인
        if root_percentages.get(root_i, 0) < min_percentage:
            min_delta = float("inf")
            best_target = root_i

            # 큰 root 찾기
            for j in range(n_clusters):
                root_j = merge_map[j]

                if root_i == root_j:
                    continue

                if root_percentages.get(root_j, 0) < min_percentage:
                    continue

                delta_e = calculate_delta_e(cluster_centers_lab[root_i], cluster_centers_lab[root_j])

                if delta_e < min_delta:
                    min_delta = delta_e
                    best_target = root_j

            if best_target != root_i:
                # root_i를 best_target으로 병합
                for k in range(n_clusters):
                    if merge_map[k] == root_i:
                        merge_map[k] = best_target
                small_merges += 1

    merged_labels = np.array([merge_map[label] for label in labels])

    # 병합 정보
    merge_info = {
        "delta_e_merges": delta_e_merges,
        "small_merges": small_merges,
        "total_merges": delta_e_merges + small_merges,
    }

    return merged_labels, merge_map, merge_info


def check_quality_guardrails(coverage_info):
    """
    가드레일: ROI 품질 모니터링 (성능 1: ROI 대비 비율 추가)

    Returns:
    - warnings: 경고 메시지 리스트
    """
    warnings_list = []

    # Guardrail 1: ROI 커버리지 너무 낮음
    if coverage_info.get("roi_mask_coverage", 0) < 20:
        warnings_list.append(
            {
                "level": "ERROR",
                "code": "LOW_ROI",
                "message": f"ROI 커버리지 매우 낮음: {coverage_info.get('roi_mask_coverage', 0):.1f}% < 20%",
                "suggestion": "렌즈 크롭 실패 가능성",
            }
        )

    # Guardrail 2: 연결 요소 수 과다
    roi_info = coverage_info.get("roi_info", {})
    num_components = roi_info.get("num_components", 0)

    if num_components > 150:
        warnings_list.append(
            {
                "level": "WARNING",
                "code": "NOISE_HIGH",
                "message": f"연결 요소 과다: {num_components}개 > 150",
                "suggestion": "먼지/배경 누출 심함",
            }
        )

    # Guardrail 3: 필터링 후 픽셀 부족 (성능 1: ROI 대비 비율)
    roi_filtered_ratio_vs_roi = coverage_info.get("roi_filtered_ratio_vs_roi", 0)

    if roi_filtered_ratio_vs_roi < 40:
        warnings_list.append(
            {
                "level": "WARNING",
                "code": "OVER_FILTERED",
                "message": f"ROI 대비 필터링 후 {roi_filtered_ratio_vs_roi:.1f}% < 40%",
                "suggestion": "하이라이트/필터링 과도",
            }
        )

    # v1.0.7: Guardrail 4 - 하이라이트 분리 (specular vs bg_leak)
    specular_ratio = coverage_info.get("specular_ratio", 0)
    bg_leak_ratio = coverage_info.get("bg_leak_ratio", 0)

    # 반사(specular) 경고
    if specular_ratio > 20:
        warnings_list.append(
            {
                "level": "WARNING",
                "code": "HIGHLIGHT_VERY_HIGH",
                "message": f"반사 과다: {specular_ratio:.1f}% > 20%",
                "suggestion": "조명/반사 영향 큼",
            }
        )
    elif specular_ratio > 10:
        warnings_list.append(
            {
                "level": "INFO",
                "code": "HIGHLIGHT_HIGH",
                "message": f"반사: {specular_ratio:.1f}% > 10%",
                "suggestion": "광택 (정상 범위)",
            }
        )

    # 배경 누출 제거 정보
    if bg_leak_ratio > 10:
        warnings_list.append(
            {
                "level": "INFO",
                "code": "BG_LEAK_REMOVED",
                "message": f"배경 누출 제거: {bg_leak_ratio:.1f}%",
                "suggestion": "배경 정리 완료",
            }
        )

    return warnings_list


def generate_coverage_summary_line(coverage_info):
    """
    운영: 한 줄 요약 로그 생성

    Example:
    "ROI 66% | filtered 52%(vs ROI 79%) | highlight 16% | components 88 | clusters 8→5 | warn: INFO highlight"
    """
    roi_cov = coverage_info.get("roi_mask_coverage", 0)
    filtered_cov = coverage_info.get("roi_filtered_coverage", 0)
    filtered_vs_roi = coverage_info.get("roi_filtered_ratio_vs_roi", 0)
    highlight = coverage_info.get("highlight_ratio", 0)
    components = coverage_info.get("roi_info", {}).get("num_components", 0)

    n_req = coverage_info.get("n_clusters_requested", 0)
    n_after = coverage_info.get("n_clusters_after_merge", 0)

    # D: merge 정보
    merge_info = coverage_info.get("merge_info", {})
    total_merges = merge_info.get("total_merges", 0)

    # 경고 요약 (전체 우선순위 정렬 후 상위 2개)
    warnings = coverage_info.get("quality_warnings", [])

    if warnings:
        # 우선순위: ERROR > WARNING > INFO
        level_priority = {"ERROR": 0, "WARNING": 1, "INFO": 2}

        # 전체 warnings를 우선순위로 정렬
        warnings_sorted = sorted(
            warnings, key=lambda w: (level_priority.get(w.get("level", "INFO"), 9), w.get("code", ""))
        )

        # 상위 2개만 출력
        top = warnings_sorted[:2]
        warn_levels_top = [w["level"] for w in top]
        warn_codes_top = [w["code"] for w in top]

        warn_str = f"{','.join(warn_levels_top)}: {','.join(warn_codes_top)}"
    else:
        warn_str = "OK"

    summary = (
        f"ROI {roi_cov:.0f}% | "
        f"filtered {filtered_cov:.0f}%(vs ROI {filtered_vs_roi:.0f}%) | "
        f"highlight {highlight:.1f}% | "
        f"components {components} | "
        f"clusters {n_req}→{n_after}(merged {total_merges}) | "  # D: merge 카운트
        f"warn: {warn_str}"
    )

    return summary


def classify_color_category_refined(rgb, hsv):
    """
    HSV 기반 색상 분류

    Note: 저채도에서 hue 불안정 (장기: Lab hue angle로 전환)
    """
    h, s, v = hsv

    if s < 20:
        if v < 50:
            return "Black/Dark"
        elif v < 100:
            return "Gray"
        elif v < 180:
            return "Light Gray"
        else:
            return "Light Gray"

    if v < 60:
        if 10 <= h <= 30:
            return "Dark Brown"
        else:
            return "Black/Dark"

    # 블루 계열
    if 85 <= h <= 115:
        if s >= 25:
            if v >= 120:
                return "Sky Blue"
            elif v >= 80:
                return "Blue"
            else:
                return "Navy"
        else:
            return "Light Gray"

    if 80 <= h < 95:
        if s >= 30 and v >= 100:
            return "Sky Blue"
        else:
            return "Cyan/Turquoise"

    if 95 <= h <= 130:
        if v < 80:
            return "Navy"
        elif s > 60:
            return "Blue"
        else:
            return "Sky Blue"

    # 브라운/베이지 계열
    if 15 <= h <= 35:
        if s < 60:
            if v >= 140:
                return "Beige/Cream"
            elif v >= 80:
                return "Brown"
            else:
                return "Dark Brown"
        else:
            if v >= 120:
                return "Orange/Beige"
            else:
                return "Orange/Brown"

    if 5 <= h <= 25:
        if s >= 60:
            return "Orange/Brown"
        else:
            return "Brown"

    # 황금/노랑 계열
    if 25 <= h <= 45:
        if s > 70:
            return "Gold/Yellow"
        elif s > 40:
            return "Beige/Gold"
        else:
            return "Beige/Cream"

    # 그린 계열
    if 40 <= h <= 75:
        if s < 50:
            return "Olive"
        elif h < 60:
            return "Green/Yellow"
        else:
            return "Green"

    if 75 <= h <= 85:
        return "Green"

    # 보라/핑크 계열
    if 130 <= h <= 155:
        return "Purple/Violet"

    if 155 <= h <= 175:
        return "Pink/Magenta"

    if h >= 175 or h <= 5:
        return "Red/Wine"

    return "Other"


def extract_colors_production_v107(
    image,
    n_clusters=8,
    seed=42,
    use_dynamic_bg=True,
    use_roi_extraction=True,
    roi_method="largest_component",
    use_lab_clustering=True,
    l_weight=0.3,
    use_two_band_sampling=True,
    high_chroma_percentile=40,
    low_chroma_samples_base=2000,
    use_dynamic_low_samples=True,
    dynamic_low_ratio=0.05,  # A: ratio 파라미터
    use_cluster_merging=True,
    merge_threshold=4.0,
    min_cluster_percentage=1.5,
    highlight_l_threshold=92,
    highlight_chroma_threshold=6,
    use_dynamic_highlight_chroma=False,
    enable_auto_retry=True,
    min_clusters_threshold=2,
    use_trimmed_mean=True,  # C: trimmed_mean 옵션
    trim_percent=10,
):
    """
    Production v1.0.7 색상 추출

    A) 자동 복구: dynamic_low_ratio 증가 (5% → 10%)
    C) trimmed_mean: outlier 제거 (상하 10%)
    """
    h, w = image.shape[:2]
    total_pixels = h * w

    rng = np.random.default_rng(seed)

    # 1. 전경 추출
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if use_dynamic_bg:
        bg_threshold = get_dynamic_background_threshold(gray)
        fg_mask = gray < bg_threshold
    else:
        fg_mask = gray < 240

    # Step 1: ROI 추출
    if use_roi_extraction:
        roi_mask, roi_info = extract_lens_roi(image, fg_mask, method=roi_method)
    else:
        roi_mask = fg_mask
        roi_info = {"method": "none"}
    # v1.0.7: 전체 이미지 Lab 변환 (spatial 필터용)
    rgb_normalized = image / 255.0
    lab_img = skcolor.rgb2lab(rgb_normalized)

    # v1.0.7: Spatial 하이라이트 제거 (코어 + 헤일로)
    highlight_2d, highlight_info = filter_highlights_spatial(
        lab_img, roi_mask, core_L=97, core_chroma=12, halo_L=70, halo_dilate_ratio=0.06  # v1.0.7: halo_L=70 (최종)
    )

    # ROI 픽셀만 추출
    roi_pixels_rgb = image[roi_mask]

    if len(roi_pixels_rgb) == 0:
        coverage_info = {"total_pixels": total_pixels, "error": "No ROI pixels"}
        return {}, coverage_info

    # Step 2: Lab 변환 (1D)
    roi_pixels_lab_scaled, roi_pixels_lab_original = rgb_to_lab_scaled(roi_pixels_rgb, l_weight=l_weight)

    # v1.0.7: 2D → 1D 마스크 변환
    # v1.0.7: 2D → 1D 마스크 변환
    highlight_mask = highlight_2d[roi_mask]
    non_highlight_mask = ~highlight_mask

    roi_pixels_rgb_filtered = roi_pixels_rgb[non_highlight_mask]
    roi_pixels_lab_scaled_filtered = roi_pixels_lab_scaled[non_highlight_mask]
    roi_pixels_lab_original_filtered = roi_pixels_lab_original[non_highlight_mask]

    highlight_ratio = float(np.sum(highlight_mask) / len(roi_pixels_rgb) * 100)

    # ROI 대비 비율
    roi_filtered_ratio_vs_roi = (
        float(len(roi_pixels_rgb_filtered) / len(roi_pixels_rgb) * 100) if len(roi_pixels_rgb) > 0 else 0
    )
    highlight_ratio_vs_roi = float(np.sum(highlight_mask) / len(roi_pixels_rgb) * 100) if len(roi_pixels_rgb) > 0 else 0

    # 커버리지 정보
    # 커버리지 정보
    coverage_info = {
        "total_pixels": total_pixels,
        "fg_mask_pixels": int(np.sum(fg_mask)),
        "fg_mask_coverage": float(np.sum(fg_mask) / total_pixels * 100),
        "roi_mask_pixels": int(np.sum(roi_mask)),
        "roi_mask_coverage": float(np.sum(roi_mask) / total_pixels * 100),
        "highlight_pixels": int(np.sum(highlight_mask)),
        "highlight_ratio": highlight_ratio,
        "highlight_ratio_vs_roi": highlight_ratio_vs_roi,
        "highlight_core_pixels": highlight_info["core_pixels"],
        "highlight_halo_pixels": highlight_info["halo_pixels"],
        "highlight_bg_leak_pixels": highlight_info["bg_leak_pixels"],  # v1.0.7
        "highlight_dilate_px": highlight_info["dilate_px"],
        "bg_lab": highlight_info["bg_lab"],  # v1.0.7: 배경 Lab
        "bg_band_pixels": highlight_info["bg_band_pixels"],  # v1.0.7: 링 픽셀 수
        # v1.0.7: ratio 분리
        "specular_ratio": (
            float((highlight_info["core_pixels"] + highlight_info["halo_pixels"]) / len(roi_pixels_rgb) * 100)
            if len(roi_pixels_rgb) > 0
            else 0
        ),
        "bg_leak_ratio": (
            float(highlight_info["bg_leak_pixels"] / len(roi_pixels_rgb) * 100) if len(roi_pixels_rgb) > 0 else 0
        ),
        "roi_filtered_pixels": len(roi_pixels_rgb_filtered),
        "roi_filtered_coverage": float(len(roi_pixels_rgb_filtered) / total_pixels * 100),
        "roi_filtered_ratio_vs_roi": roi_filtered_ratio_vs_roi,
        "bg_threshold": bg_threshold if use_dynamic_bg else 240,
        "roi_info": roi_info,
        "use_lab": use_lab_clustering,
        "l_weight": l_weight,
        "n_clusters_requested": n_clusters,
        "percentage_basis": "roi_filtered",
        "use_two_band_sampling": use_two_band_sampling,
        "use_dynamic_low_samples": use_dynamic_low_samples,
        "use_trimmed_mean": use_trimmed_mean,
        "trim_percent": trim_percent if use_trimmed_mean else None,
        "dynamic_low_ratio": dynamic_low_ratio,
    }

    if len(roi_pixels_rgb_filtered) == 0:
        coverage_info["error"] = "No pixels after filtering"
        return {}, coverage_info

    # 성능 2: 동적 low_chroma_samples
    if use_dynamic_low_samples:
        low_chroma_samples = calculate_dynamic_low_chroma_samples(
            len(roi_pixels_rgb_filtered), base_samples=low_chroma_samples_base, ratio=dynamic_low_ratio  # A: ratio 전달
        )
        coverage_info["low_chroma_samples_dynamic"] = low_chroma_samples
        coverage_info["dynamic_low_ratio"] = dynamic_low_ratio  # A
    else:
        low_chroma_samples = low_chroma_samples_base
        coverage_info["low_chroma_samples_static"] = low_chroma_samples

    # 2-밴드 샘플링
    if use_two_band_sampling:
        sampled_rgb, sampled_lab_original, sample_mask, chroma_threshold_value, band_info = sample_pixels_two_band(
            roi_pixels_rgb_filtered,
            roi_pixels_lab_original_filtered,
            high_chroma_percentile=high_chroma_percentile,
            low_chroma_samples=low_chroma_samples,
            rng=rng,
        )

        sampled_lab_scaled = roi_pixels_lab_scaled_filtered[sample_mask]
        clustering_data = sampled_lab_scaled

        coverage_info["chroma_threshold_value"] = float(chroma_threshold_value)
        coverage_info["band_info"] = band_info
    else:
        chroma = calculate_lab_chroma(roi_pixels_lab_original_filtered)
        threshold = np.percentile(chroma, high_chroma_percentile)
        sample_mask = chroma >= threshold

        clustering_data = roi_pixels_lab_scaled_filtered[sample_mask]
        coverage_info["chroma_threshold_value"] = float(threshold)

    # K-means 샘플링
    max_samples = 15000
    if len(clustering_data) > max_samples:
        indices = rng.choice(len(clustering_data), max_samples, replace=False)
        kmeans_data = clustering_data[indices]
    else:
        kmeans_data = clustering_data

    # 6. kmeans 샘플링 정보 추가
    coverage_info["kmeans_data_size"] = len(kmeans_data)
    coverage_info["kmeans_sample_ratio"] = float(len(kmeans_data) / len(roi_pixels_rgb_filtered) * 100)

    # K-means 클러스터링
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(kmeans_data)

    # ROI 전체 픽셀에 라벨 할당
    labels_roi = kmeans.predict(roi_pixels_lab_scaled_filtered)

    cluster_centers = kmeans.cluster_centers_
    cluster_centers_lab = cluster_centers.copy()
    cluster_centers_lab[:, 0] = cluster_centers[:, 0] / max(l_weight, 0.05)

    # 클러스터 병합
    if use_cluster_merging:
        labels_roi, merge_map, merge_info = merge_similar_clusters(
            cluster_centers_lab, labels_roi, delta_e_threshold=merge_threshold, min_percentage=min_cluster_percentage
        )
        coverage_info["merge_map"] = merge_map

        coverage_info["merge_info"] = merge_info  # D: 병합 정보 저장
    # 최종 센터 재계산
    unique_labels = np.unique(labels_roi)
    n_clusters_after_merge = len(unique_labels)

    # 안정성: 자동 복구 재시도
    if enable_auto_retry and n_clusters_after_merge <= min_clusters_threshold:
        coverage_info["auto_retry_triggered"] = True
        coverage_info["retry_reason"] = f"n_clusters_after_merge={n_clusters_after_merge} <= {min_clusters_threshold}"

        # 재시도: dynamic_chroma 끄고 재실행
        retry_colors, retry_coverage = extract_colors_production_v107(
            image,
            n_clusters=n_clusters,
            seed=seed,
            use_dynamic_bg=use_dynamic_bg,
            use_roi_extraction=use_roi_extraction,
            roi_method=roi_method,
            use_lab_clustering=use_lab_clustering,
            l_weight=l_weight,
            use_two_band_sampling=use_two_band_sampling,
            dynamic_low_ratio=dynamic_low_ratio * 2,  # A: 5% → 10%
            high_chroma_percentile=high_chroma_percentile,
            low_chroma_samples_base=low_chroma_samples_base * 2,  # 2배 증가
            use_dynamic_low_samples=use_dynamic_low_samples,
            use_cluster_merging=use_cluster_merging,
            merge_threshold=merge_threshold,
            min_cluster_percentage=min_cluster_percentage,
            highlight_l_threshold=highlight_l_threshold,
            highlight_chroma_threshold=highlight_chroma_threshold,
            use_trimmed_mean=use_trimmed_mean,  # C
            trim_percent=trim_percent,  # C
            use_dynamic_highlight_chroma=False,  # 끄기
            enable_auto_retry=False,  # 재귀 방지
            min_clusters_threshold=min_clusters_threshold,
        )

        retry_coverage["is_retry"] = True
        retry_coverage["original_n_clusters"] = n_clusters_after_merge

        return retry_colors, retry_coverage

    final_centers_lab = []

    # C) trimmed_mean으로 최종 센터 계산
    for label in unique_labels:
        mask = labels_roi == label

        if use_trimmed_mean:
            center_lab = trimmed_mean_lab(roi_pixels_lab_original_filtered[mask], trim_percent=trim_percent)
        else:
            center_lab = roi_pixels_lab_original_filtered[mask].mean(axis=0)

        final_centers_lab.append(center_lab)

    final_centers_lab = np.array(final_centers_lab)
    colors_rgb = lab_to_rgb(final_centers_lab)

    coverage_info["n_clusters_after_merge"] = n_clusters_after_merge

    # 클러스터 정보 수집
    cluster_info = []
    total_roi_filtered = len(roi_pixels_rgb_filtered)

    for i, label in enumerate(unique_labels):
        count_roi = np.sum(labels_roi == label)
        pct_roi = (count_roi / total_roi_filtered) * 100

        color_rgb = colors_rgb[i]
        color_hsv = cv2.cvtColor(np.uint8([[color_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        category = classify_color_category_refined(color_rgb, color_hsv)

        cluster_info.append(
            {
                "index": label,
                "color_rgb": color_rgb,
                "color_hsv": color_hsv,
                "category": category,
                "count_roi": count_roi,
                "pct_roi": pct_roi,
            }
        )

    # 카테고리별 그룹화
    category_groups = defaultdict(list)
    for info in cluster_info:
        category_groups[info["category"]].append(info)

    # 대표 색상
    final_colors = {}

    for category, clusters in category_groups.items():
        total_count = sum(c["count_roi"] for c in clusters)
        total_pct_roi = sum(c["pct_roi"] for c in clusters)

        weighted_rgb = np.zeros(3)
        for c in clusters:
            weight = c["count_roi"] / total_count
            weighted_rgb += c["color_rgb"] * weight

        if total_pct_roi >= min_cluster_percentage:
            final_colors[category] = {
                "color_rgb": weighted_rgb,
                "pct_roi": total_pct_roi,
                "cluster_count": len(clusters),
                "color_name_kr": COLOR_NAME_KR.get(category, category),
            }

    coverage_info["n_categories_final"] = len(final_colors)

    # 가드레일 체크
    coverage_info["quality_warnings"] = check_quality_guardrails(coverage_info)

    # 운영: 한 줄 요약
    coverage_info["summary_line"] = generate_coverage_summary_line(coverage_info)

    return final_colors, coverage_info


def get_color_groups(colors_dict):
    """색상을 그룹으로 분류"""
    grouped_colors = defaultdict(lambda: {"colors": [], "total_pct_roi": 0})

    for category, info in colors_dict.items():
        for group_name, group_categories in COLOR_GROUPS.items():
            if category in group_categories:
                grouped_colors[group_name]["colors"].append(
                    {
                        "category": category,
                        "category_kr": info["color_name_kr"],
                        "pct_roi": info["pct_roi"],
                        "color_rgb": info["color_rgb"],
                    }
                )
                grouped_colors[group_name]["total_pct_roi"] += info["pct_roi"]

    for group in grouped_colors.values():
        group["colors"].sort(key=lambda x: x["pct_roi"], reverse=True)

    return dict(grouped_colors)


def analyze_lens_colors(image_path, image_name=None, seed=42, **kwargs):
    """콘택트렌즈 색상 분석 - Production v1.0.7"""
    if image_name is None:
        image_name = os.path.basename(image_path)

    image = load_image(image_path)

    colors, coverage_info = extract_colors_production_v107(image, seed=seed, **kwargs)

    if len(colors) == 0:
        return {"success": False, "error": "색상을 추출하지 못했습니다.", "coverage_info": coverage_info}

    sorted_colors = sorted(colors.items(), key=lambda x: x[1]["pct_roi"], reverse=True)
    main_colors = sorted_colors[:3]

    color_groups = get_color_groups(colors)

    summary = {
        "image_name": image_name,
        "total_colors": len(colors),
        "main_color": main_colors[0][1]["color_name_kr"] if len(main_colors) > 0 else None,
        "dominant_group": max(color_groups.items(), key=lambda x: x[1]["total_pct_roi"])[0] if color_groups else None,
    }

    result = {
        "success": True,
        "colors": colors,
        "main_colors": main_colors,
        "color_groups": color_groups,
        "summary": summary,
        "coverage_info": coverage_info,
        "image": image,
    }

    return result


def print_coverage_info(coverage_info):
    """커버리지 정보 및 경고 출력"""
    # 운영: 한 줄 요약 먼저
    print(f"\n[한 줄 요약]")
    print(f"  {coverage_info.get('summary_line', 'N/A')}")

    # 자동 재시도 정보
    if coverage_info.get("auto_retry_triggered"):
        print(f"\n[자동 복구]")
        print(f"  ⚠️  재시도 발생: {coverage_info.get('retry_reason', 'Unknown')}")
        print(f"  원본 클러스터: {coverage_info.get('original_n_clusters', 0)}개")
        print(f"  재시도 후: {coverage_info.get('n_clusters_after_merge', 0)}개")

    print(f"\n[상세 정보]")
    print(f"  ROI: {coverage_info['roi_mask_pixels']:,} pixels ({coverage_info['roi_mask_coverage']:.1f}%)")

    # 성능 1: ROI 대비 비율
    print(
        f"  하이라이트: {coverage_info.get('highlight_pixels', 0):,} pixels "
        f"({coverage_info.get('highlight_ratio', 0):.1f}% 전체, "
        f"{coverage_info.get('highlight_ratio_vs_roi', 0):.1f}% vs ROI)"
    )

    print(
        f"  분석 대상: {coverage_info.get('roi_filtered_pixels', 0):,} pixels "
        f"({coverage_info.get('roi_filtered_coverage', 0):.1f}% 전체, "
        f"{coverage_info.get('roi_filtered_ratio_vs_roi', 0):.1f}% vs ROI)"
    )

    # 성능 2: 동적 샘플링
    if coverage_info.get("use_dynamic_low_samples"):
        print(f"  동적 샘플링: {coverage_info.get('low_chroma_samples_dynamic', 0):,} 픽셀")

    # 가드레일 경고
    warnings_list = coverage_info.get("quality_warnings", [])
    if warnings_list:
        print(f"\n[품질 경고]")
        for warning in warnings_list:
            level = warning["level"]
            symbol = "🔴" if level == "ERROR" else "⚠️" if level == "WARNING" else "ℹ️"
            print(f"  {symbol} [{warning['code']}] {warning['message']}")


def print_analysis_result(result):
    """분석 결과 출력"""
    if not result["success"]:
        print(f"분석 실패: {result.get('error', 'Unknown')}")
        if "coverage_info" in result:
            print_coverage_info(result["coverage_info"])
        return

    summary = result["summary"]

    print(f"\n{'='*70}")
    print(f"분석 결과: {summary['image_name']}")
    print(f"{'='*70}")

    print_coverage_info(result["coverage_info"])

    print(f"\n[메인 3색]")
    for idx, (category, info) in enumerate(result["main_colors"], 1):
        rgb = info["color_rgb"]
        hex_code = "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        print(f"  {idx}. {info['color_name_kr']:15s}: {hex_code} ({info['pct_roi']:.1f}%)")


def main():
    """메인 실행"""

    test_images = [
        "/mnt/user-data/uploads/1962.jpg",
        "/mnt/user-data/uploads/1836.jpg",
        "/mnt/user-data/uploads/1940.jpg",
    ]

    print("=" * 70)
    print("콘택트렌즈 색상 추출 시스템 - Production v1.0.7")
    print("=" * 70)
    print("🔴 C) trimmed_mean: ΔE 거리 기반 (치명적 버그 수정)")
    print("🔴 D) merge: ΔE 병합 후 크기 재계산 (치명적 버그 수정)")
    print("🟡 경고 우선순위 정렬 + kmeans 정보 추가")
    print("=" * 70)

    for img_path in test_images:
        img_name = os.path.basename(img_path)

        result = analyze_lens_colors(
            img_path,
            img_name,
            seed=42,
            use_dynamic_bg=True,
            use_roi_extraction=True,
            roi_method="largest_component",
            use_lab_clustering=True,
            l_weight=0.3,
            use_two_band_sampling=True,
            high_chroma_percentile=40,
            low_chroma_samples_base=2000,
            use_dynamic_low_samples=True,
            dynamic_low_ratio=0.05,  # A
            use_cluster_merging=True,
            merge_threshold=4.0,
            min_cluster_percentage=1.5,
            highlight_l_threshold=92,
            highlight_chroma_threshold=6,
            use_dynamic_highlight_chroma=False,
            enable_auto_retry=True,
            min_clusters_threshold=2,
            use_trimmed_mean=True,  # C
            trim_percent=10,  # C
        )

        if result["success"]:
            print_analysis_result(result)
        print()

    print("=" * 70)
    print("Production v1.0.7 완료 - 치명적 버그 수정!")
    print("=" * 70)


if __name__ == "__main__":
    main()
