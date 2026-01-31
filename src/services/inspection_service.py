"""
Inspection Pipeline Module

5개 핵심 모듈을 연결하는 엔드투엔드 검사 파이프라인.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from scipy.signal import savgol_filter

from src.config.v7_paths import V7_MODELS, V7_ROOT
from src.converters import build_inspection_result_from_v7
from src.engine_v7.core.types import LensGeometry
from src.schemas.inspection import InspectionResult
from src.utils.file_io import FileIO

logger = logging.getLogger(__name__)


@dataclass
class ImageConfig:
    resolution_w: int = 1920
    resolution_h: int = 1080
    roi: Optional[tuple[int, int, int, int]] = None
    denoise_enabled: bool = True
    denoise_method: str = "bilateral"
    denoise_kernel_size: int = 5
    denoise_sigma_color: int = 75
    denoise_sigma_space: int = 75
    white_balance_enabled: bool = True
    white_balance_method: str = "gray_world"
    auto_roi_detection: bool = True
    auto_roi_margin_ratio: float = 0.2


class PipelineError(Exception):
    """파이프라인 실행 중 발생하는 예외"""

    pass


class LensDetectionError(Exception):
    pass


@dataclass
class DetectorConfig:
    method: str = "hybrid"
    hough_dp: float = 1.2
    hough_min_dist_ratio: float = 0.5
    hough_param1: float = 50
    hough_param2: float = 30
    hough_min_radius_ratio: float = 0.3
    hough_max_radius_ratio: float = 0.8
    contour_threshold_method: str = "otsu"
    contour_morph_kernel_size: int = 5
    contour_min_area_ratio: float = 0.01
    hybrid_merge_dist_threshold: float = 10.0
    subpixel_refinement_enabled: bool = False
    subpixel_window_size: int = 21
    background_fallback_enabled: bool = True
    background_color_distance_threshold: float = 30.0
    background_min_area_ratio: float = 0.05


@dataclass
class RadialProfile:
    r_normalized: np.ndarray
    L: np.ndarray
    a: np.ndarray
    b: np.ndarray
    std_L: np.ndarray
    std_a: np.ndarray
    std_b: np.ndarray
    pixel_count: np.ndarray


@dataclass
class ProfilerConfig:
    r_start_ratio: float = 0.0
    r_end_ratio: float = 1.0
    r_step_pixels: int = 1
    theta_samples: int = 360
    smoothing_enabled: bool = True
    smoothing_method: str = "savgol"
    savgol_window_length: int = 11
    savgol_polyorder: int = 3
    moving_average_window: int = 5
    sample_percentile: Optional[float] = None


class InspectionPipeline:
    """
    엔드투엔드 렌즈 색상 검사 파이프라인.

    ImageLoader → LensDetector → RadialProfiler → ZoneSegmenter → ColorEvaluator
    순서로 5개 모듈을 연결하여 최종 판정 결과 생성.
    """

    def __init__(
        self,
        sku_config: Dict[str, Any],
        image_config: Optional[ImageConfig] = None,
        detector_config: Optional[DetectorConfig] = None,
        profiler_config: Optional[ProfilerConfig] = None,
        segmenter_config: Optional[Any] = None,
        save_intermediates: bool = False,
    ):
        """
        파이프라인 초기화.

        Args:
            sku_config: SKU별 기준값 설정
            image_config: ImageLoader 설정 (기본값 사용 시 None)
            detector_config: LensDetector 설정 (기본값 사용 시 None)
            profiler_config: RadialProfiler 설정 (기본값 사용 시 None)
            segmenter_config: ZoneSegmenter 설정 (기본값 사용 시 None)
            save_intermediates: 중간 결과 저장 여부
        """
        self.sku_config = sku_config
        self.save_intermediates = save_intermediates

        self._image_config = image_config or ImageConfig()
        self._detector_config = detector_config or DetectorConfig()
        self._profiler_config = profiler_config or ProfilerConfig()
        self._file_io = FileIO()
        self.zone_segmenter = None
        self.color_evaluator = None

        logger.info("InspectionPipeline initialized")

    def _load_image(self, path: Path) -> Optional[np.ndarray]:
        image = self._file_io.load_image(path)
        if image is None:
            logger.warning("Failed to load image from %s", path)
        return image

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            return None
        if image.size == 0:
            return image

        processed = image.copy()
        cfg = self._image_config

        current_roi = cfg.roi
        if cfg.auto_roi_detection:
            x, y, w, h = self._detect_roi_from_image(processed)
            if w > 0 and h > 0:
                current_roi = (x, y, w, h)

        if current_roi:
            x, y, w, h = current_roi
            if x >= 0 and y >= 0 and x + w <= processed.shape[1] and y + h <= processed.shape[0]:
                processed = processed[y : y + h, x : x + w]

        if cfg.denoise_enabled:
            processed = self._denoise_image(processed, cfg)

        if cfg.white_balance_enabled:
            processed = self._apply_white_balance(processed, cfg)

        return processed

    def _denoise_image(self, image: np.ndarray, cfg: ImageConfig) -> np.ndarray:
        if image.size == 0:
            return image
        if cfg.denoise_method == "gaussian":
            return cv2.GaussianBlur(image, (cfg.denoise_kernel_size, cfg.denoise_kernel_size), 0)
        if cfg.denoise_method == "bilateral":
            return cv2.bilateralFilter(image, cfg.denoise_kernel_size, cfg.denoise_sigma_color, cfg.denoise_sigma_space)
        return image

    def _apply_white_balance(self, image: np.ndarray, cfg: ImageConfig) -> np.ndarray:
        if image.size == 0:
            return image
        if cfg.white_balance_method == "gray_world":
            return self._gray_world_white_balance(image)
        return image

    def _gray_world_white_balance(self, image: np.ndarray) -> np.ndarray:
        if image.size == 0:
            return image
        b, g, r = cv2.split(image)
        avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)

        avg_b = 1.0 if avg_b == 0 else avg_b
        avg_g = 1.0 if avg_g == 0 else avg_g
        avg_r = 1.0 if avg_r == 0 else avg_r

        avg_all = (avg_b + avg_g + avg_r) / 3.0

        scale_b = avg_all / avg_b
        scale_g = avg_all / avg_g
        scale_r = avg_all / avg_r

        b = cv2.convertScaleAbs(b * scale_b)
        g = cv2.convertScaleAbs(g * scale_g)
        r = cv2.convertScaleAbs(r * scale_r)
        return cv2.merge([b, g, r])

    def _detect_roi_from_image(self, image: np.ndarray) -> tuple[int, int, int, int]:
        if image is None or image.size == 0:
            return (0, 0, 0, 0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return (0, 0, image.shape[1], image.shape[0])

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        margin_x = int(w * self._image_config.auto_roi_margin_ratio)
        margin_y = int(h * self._image_config.auto_roi_margin_ratio)

        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(image.shape[1] - x, w + 2 * margin_x)
        h = min(image.shape[0] - y, h + 2 * margin_y)

        return (x, y, w, h)

    def _configure_profiler(self, params: Dict[str, Any]) -> None:
        cfg = self._profiler_config
        optical_clear = params.get("optical_clear_ratio")
        center_exclude = params.get("center_exclude_ratio")
        if isinstance(optical_clear, (int, float)) and 0 <= optical_clear < 1:
            cfg.r_start_ratio = float(optical_clear)
        if isinstance(center_exclude, (int, float)) and 0 <= center_exclude < 1:
            cfg.r_start_ratio = max(cfg.r_start_ratio, float(center_exclude))
        if cfg.r_start_ratio > 0:
            logger.info(
                "Applying r_start_ratio=%.3f (optical_clear_ratio=%s, center_exclude_ratio=%s)",
                cfg.r_start_ratio,
                f"{optical_clear:.3f}" if isinstance(optical_clear, (int, float)) else "None",
                f"{center_exclude:.3f}" if isinstance(center_exclude, (int, float)) else "None",
            )

    def _build_radial_profile_v7(self, image: np.ndarray, detection: LensGeometry) -> RadialProfile:
        if image is None:
            raise ValueError("Input image cannot be None.")
        if detection is None:
            raise ValueError("LensGeometry object cannot be None.")

        cfg = self._profiler_config
        r_samples = int(detection.r / cfg.r_step_pixels) if cfg.r_step_pixels > 0 else int(detection.r)
        if r_samples < 1:
            r_samples = 1
        theta_samples = int(cfg.theta_samples)

        from src.engine_v7.core.signature.radial_signature import to_polar
        from src.engine_v7.core.utils import to_cie_lab

        geom = LensGeometry(
            cx=float(detection.cx),
            cy=float(detection.cy),
            r=float(detection.r),
            confidence=float(detection.confidence),
            source="pipeline",
        )
        polar_bgr = to_polar(image, geom, R=r_samples, T=theta_samples)
        if polar_bgr.size == 0:
            dummy_r = np.linspace(0, 1, 10)
            zeros = np.zeros_like(dummy_r)
            return RadialProfile(dummy_r, zeros, zeros, zeros, zeros, zeros, zeros, np.zeros_like(dummy_r, dtype=int))

        lab = to_cie_lab(polar_bgr, source="bgr")
        _, R, _ = lab.shape
        r0 = int(R * cfg.r_start_ratio)
        r1 = int(R * cfg.r_end_ratio)
        r0 = max(0, min(R - 1, r0))
        r1 = max(r0 + 1, min(R, r1))

        lab_roi = lab[:, r0:r1, :]
        if lab_roi.size == 0:
            dummy_r = np.linspace(0, 1, 10)
            zeros = np.zeros_like(dummy_r)
            return RadialProfile(dummy_r, zeros, zeros, zeros, zeros, zeros, zeros, np.zeros_like(dummy_r, dtype=int))

        L_plane = lab_roi[..., 0]
        a_plane = lab_roi[..., 1]
        b_plane = lab_roi[..., 2]

        if cfg.sample_percentile is not None:
            L_profile = np.percentile(L_plane, cfg.sample_percentile, axis=0)
            a_profile = np.percentile(a_plane, cfg.sample_percentile, axis=0)
            b_profile = np.percentile(b_plane, cfg.sample_percentile, axis=0)
        else:
            L_profile = L_plane.mean(axis=0)
            a_profile = a_plane.mean(axis=0)
            b_profile = b_plane.mean(axis=0)

        std_L = L_plane.std(axis=0)
        std_a = a_plane.std(axis=0)
        std_b = b_plane.std(axis=0)

        r_normalized = np.linspace(0.0, 1.0, R)[r0:r1]
        pixel_count = np.full_like(L_profile, lab_roi.shape[0])

        profile = RadialProfile(
            r_normalized=r_normalized,
            L=L_profile,
            a=a_profile,
            b=b_profile,
            std_L=std_L,
            std_a=std_a,
            std_b=std_b,
            pixel_count=pixel_count,
        )

        if cfg.smoothing_enabled:
            profile = self._smooth_profile(profile, cfg)

        return profile

    def _smooth_profile(self, profile: RadialProfile, cfg: ProfilerConfig) -> RadialProfile:
        L, a, b = profile.L, profile.a, profile.b
        if cfg.smoothing_method == "savgol" and len(L) >= cfg.savgol_window_length:
            L = savgol_filter(L, cfg.savgol_window_length, cfg.savgol_polyorder)
            a = savgol_filter(a, cfg.savgol_window_length, cfg.savgol_polyorder)
            b = savgol_filter(b, cfg.savgol_window_length, cfg.savgol_polyorder)
        elif cfg.smoothing_method == "moving_average" and len(L) >= cfg.moving_average_window:
            weights = np.ones(cfg.moving_average_window) / cfg.moving_average_window
            L_valid = np.convolve(L, weights, mode="valid")
            a_valid = np.convolve(a, weights, mode="valid")
            b_valid = np.convolve(b, weights, mode="valid")
            pad_len = (cfg.moving_average_window - 1) // 2
            L = np.pad(L_valid, (pad_len, len(L) - len(L_valid) - pad_len), mode="edge")
            a = np.pad(a_valid, (pad_len, len(a) - len(a_valid) - pad_len), mode="edge")
            b = np.pad(b_valid, (pad_len, len(b) - len(b_valid) - pad_len), mode="edge")

        return RadialProfile(
            profile.r_normalized, L, a, b, profile.std_L, profile.std_a, profile.std_b, profile.pixel_count
        )

    def _compute_quality_metrics(
        self,
        image_bgr: np.ndarray,
        lens_detection: Optional[Any] = None,
        include_dot_stats: bool = True,
    ) -> Dict[str, Any]:
        if image_bgr is None or image_bgr.size == 0:
            return {}

        lens_mask = None
        if lens_detection is not None:
            lens_mask = self._circle_mask(
                image_bgr.shape[:2],
                float(lens_detection.cx),
                float(lens_detection.cy),
                float(lens_detection.r),
            )

        metrics = {
            "blur": {"score": self._compute_blur_score(image_bgr)},
            "histogram": self._compute_histograms(image_bgr, lens_mask=lens_mask),
        }

        if include_dot_stats:
            dot_stats = self._compute_dot_stats(image_bgr, lens_mask=lens_mask)
            if dot_stats is not None:
                metrics["dot_stats"] = dot_stats

        return metrics

    def _compute_blur_score(self, image_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _compute_histograms(
        self,
        image_bgr: np.ndarray,
        lens_mask: Optional[np.ndarray] = None,
        bins: int = 32,
    ) -> Dict[str, Any]:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        mask = lens_mask
        if mask is not None and mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        return {
            "bins": bins,
            "lab": {
                "L": self._calc_hist(lab, 0, bins, (0, 256), mask),
                "a": self._calc_hist(lab, 1, bins, (0, 256), mask),
                "b": self._calc_hist(lab, 2, bins, (0, 256), mask),
            },
            "hsv": {
                "H": self._calc_hist(hsv, 0, bins, (0, 180), mask),
                "S": self._calc_hist(hsv, 1, bins, (0, 256), mask),
                "V": self._calc_hist(hsv, 2, bins, (0, 256), mask),
            },
        }

    def _calc_hist(
        self,
        image: np.ndarray,
        channel: int,
        bins: int,
        value_range: tuple,
        mask: Optional[np.ndarray],
    ) -> list:
        hist = cv2.calcHist([image], [channel], mask, [bins], value_range).astype(np.float32)
        total = float(np.sum(hist))
        if total > 0:
            hist /= total
        return hist.flatten().tolist()

    def _compute_dot_stats(
        self,
        image_bgr: np.ndarray,
        lens_mask: Optional[np.ndarray] = None,
        min_area: int = 5,
    ) -> Optional[Dict[str, Any]]:
        if lens_mask is None:
            return None

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]

        mask = lens_mask
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        s_vals = s_channel[mask > 0]
        if s_vals.size == 0:
            return None

        # cv2.threshold returns (threshold_value, thresholded_image)
        # We only need the threshold_value here to apply it to the full s_channel
        s_thresh_val, _ = cv2.threshold(s_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        s_mask = (s_channel >= s_thresh_val).astype(np.uint8) * 255
        s_mask = cv2.bitwise_and(s_mask, s_mask, mask=mask)

        lens_pixels = int(np.sum(mask > 0))
        if lens_pixels <= 0:
            return None

        ink_pixels = int(np.sum(s_mask > 0))
        if ink_pixels <= 0:
            return None

        dot_coverage = ink_pixels / lens_pixels

        contours, _ = cv2.findContours(s_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        areas = [a for a in areas if a >= min_area]

        if not areas:
            return {
                "dot_count": 0,
                "dot_coverage": float(dot_coverage),
                "dot_area_mean": 0.0,
                "dot_area_std": 0.0,
                "dot_area_min": 0.0,
                "dot_area_max": 0.0,
            }

        areas_arr = np.asarray(areas, dtype=np.float32)
        return {
            "dot_count": int(len(areas)),
            "dot_coverage": float(dot_coverage),
            "dot_area_mean": float(np.mean(areas_arr)),
            "dot_area_std": float(np.std(areas_arr)),
            "dot_area_min": float(np.min(areas_arr)),
            "dot_area_max": float(np.max(areas_arr)),
        }

    def _circle_mask(
        self,
        shape: tuple,
        center_x: float,
        center_y: float,
        radius: float,
    ) -> np.ndarray:
        h, w = shape
        yy, xx = np.indices((h, w))
        rr = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        mask = (rr <= radius).astype(np.uint8) * 255
        return mask

    def _evaluate_v7_decision(
        self, image: np.ndarray, sku: str, ink: str, params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        V7 엔진을 사용하여 판정을 수행합니다.
        새로운 Facade API(src.engine_v7.api)를 사용하여 로직을 단순화합니다.
        """
        try:
            from src.engine_v7.api import inspect_single
        except ImportError as exc:
            logger.warning("V7 API import failed: %s", exc)
            return None

        # API를 통한 단일 검사 실행
        # inspect_single 내부에서 config, model, baseline 로딩을 모두 처리함
        result = inspect_single(
            image_bgr=image,
            sku=sku,
            ink=ink,
            cfg_override=params.get("v7_config_override"),
            expected_ink_count=params.get("expected_ink_count"),
            run_id=params.get("run_id", ""),
        )

        return result

    def _attach_lens_roi(self, detection: LensGeometry, image_shape) -> LensGeometry:
        h, w = image_shape[:2]
        roi_x = int(detection.cx - detection.r * 1.1)
        roi_y = int(detection.cy - detection.r * 1.1)
        roi_w = int(detection.r * 2.2)
        roi_h = int(detection.r * 2.2)

        roi_x_start = max(0, roi_x)
        roi_y_start = max(0, roi_y)
        roi_x_end = min(w, roi_x + roi_w)
        roi_y_end = min(h, roi_y + roi_h)

        detection.roi = (roi_x_start, roi_y_start, roi_x_end - roi_x_start, roi_y_end - roi_y_start)
        return detection

    def _detect_lens_v7(self, image, params: Dict[str, Any]) -> Optional[LensGeometry]:
        try:
            from src.engine_v7.core.geometry.lens_geometry import detect_lens_circle
        except Exception as exc:
            logger.warning("V7 geometry import failed: %s", exc)
            return None

        cfg = params.get("v7_geometry", {}) if isinstance(params, dict) else {}
        try:
            geom = detect_lens_circle(image, cfg=cfg)
        except Exception as exc:
            logger.warning("V7 geometry detection failed: %s", exc)
            return None

        if geom is None or geom.r <= 0 or geom.confidence <= 0:
            logger.warning("V7 geometry returned low confidence; falling back to secondary detector")
            return None

        detection = LensGeometry(
            cx=float(geom.cx),
            cy=float(geom.cy),
            r=float(geom.r),
            confidence=float(geom.confidence),
            source=f"v7:{geom.source}",
        )
        return self._attach_lens_roi(detection, image.shape)

    def _detect_hough(self, gray_image: np.ndarray, cfg: DetectorConfig) -> Optional[LensGeometry]:
        min_radius = int(min(gray_image.shape) * cfg.hough_min_radius_ratio)
        max_radius = int(max(gray_image.shape) * cfg.hough_max_radius_ratio)

        circles = cv2.HoughCircles(
            gray_image,
            cv2.HOUGH_GRADIENT,
            dp=cfg.hough_dp,
            minDist=min_radius * cfg.hough_min_dist_ratio,
            param1=cfg.hough_param1,
            param2=cfg.hough_param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        if circles is not None:
            cx, cy, r = circles[0][0]
            return LensGeometry(cx=cx, cy=cy, r=r, confidence=0.9, source="hough")
        return None

    def _detect_contour(self, gray_image: np.ndarray, cfg: DetectorConfig) -> Optional[LensGeometry]:
        if cfg.contour_threshold_method == "otsu":
            _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.contour_morph_kernel_size, cfg.contour_morph_kernel_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        min_area = gray_image.size * cfg.contour_min_area_ratio
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        if not valid_contours:
            return None

        largest = max(valid_contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest)

        area_contour = cv2.contourArea(largest)
        area_circle = np.pi * radius**2
        confidence = area_contour / area_circle if area_circle > 0 else 0

        return LensGeometry(cx=x, cy=y, r=radius, confidence=confidence, source="contour")

    def _detect_hybrid(self, gray_image: np.ndarray, cfg: DetectorConfig) -> Optional[LensGeometry]:
        hough = self._detect_hough(gray_image, cfg)
        contour = self._detect_contour(gray_image, cfg)

        result = None
        if hough and contour:
            dist = np.sqrt((hough.cx - contour.cx) ** 2 + (hough.cy - contour.cy) ** 2)
            if dist < cfg.hybrid_merge_dist_threshold:
                result = LensGeometry(
                    cx=(hough.cx + contour.cx) / 2,
                    cy=(hough.cy + contour.cy) / 2,
                    r=(hough.r + contour.r) / 2,
                    confidence=1.0,
                    source="hybrid",
                )
            else:
                result = hough if hough.confidence > contour.confidence else contour
        else:
            result = hough or contour

        if result:
            result.source = "hybrid"
        return result

    def _detect_background_based(self, image: np.ndarray, cfg: DetectorConfig) -> Optional[LensGeometry]:
        if len(image.shape) == 2:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr_image = image

        h, w = bgr_image.shape[:2]

        bg_color = self._sample_background_color(bgr_image)
        logger.debug("Background color sampled: %s", bg_color)

        color_dist = np.linalg.norm(bgr_image.astype(np.float32) - bg_color, axis=2)

        threshold = cfg.background_color_distance_threshold
        _, foreground_mask = cv2.threshold(color_dist.astype(np.uint8), int(threshold), 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.warning("Background-based detection failed: No contours found")
            return None

        min_area = (h * w) * cfg.background_min_area_ratio
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        if not valid_contours:
            logger.warning("Background-based detection failed: No contours larger than %.0f pixels", min_area)
            return None

        largest_contour = max(valid_contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)

        area_contour = cv2.contourArea(largest_contour)
        area_circle = np.pi * radius**2
        circularity = area_contour / area_circle if area_circle > 0 else 0

        confidence = 0.3 + (circularity * 0.3)

        logger.info(
            "Background-based detection succeeded: center=(%.1f, %.1f), radius=%.1f, confidence=%.2f",
            x,
            y,
            radius,
            confidence,
        )

        return LensGeometry(cx=x, cy=y, r=radius, confidence=confidence, source="background")

    def _sample_background_color(self, bgr_image: np.ndarray) -> np.ndarray:
        h, w = bgr_image.shape[:2]
        edge_width = 10

        top_edge = bgr_image[0:edge_width, :]
        bottom_edge = bgr_image[h - edge_width : h, :]
        left_edge = bgr_image[:, 0:edge_width]
        right_edge = bgr_image[:, w - edge_width : w]

        edge_samples = np.vstack(
            [top_edge.reshape(-1, 3), bottom_edge.reshape(-1, 3), left_edge.reshape(-1, 3), right_edge.reshape(-1, 3)]
        )

        return np.asarray(np.median(edge_samples, axis=0), dtype=np.float32)

    def _refine_center(
        self, gray_image: np.ndarray, cx: float, cy: float, radius: float, cfg: DetectorConfig
    ) -> tuple[float, float]:
        r_int = int(radius)
        x_min = max(0, int(cx - r_int))
        y_min = max(0, int(cy - r_int))
        x_max = min(gray_image.shape[1], int(cx + r_int))
        y_max = min(gray_image.shape[0], int(cy + r_int))

        roi = gray_image[y_min:y_max, x_min:x_max]

        if roi.size == 0:
            logger.warning("ROI is empty, skipping refinement")
            return cx, cy

        edges_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        edges_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(edges_x**2 + edges_y**2)

        threshold = np.percentile(edge_mag, 90)
        edge_mask = edge_mag > threshold

        if edge_mask.sum() == 0:
            logger.warning("No strong edges found, skipping refinement")
            return cx, cy

        y_coords, x_coords = np.where(edge_mask)
        weights = edge_mag[edge_mask]

        cx_refined = x_min + np.average(x_coords, weights=weights)
        cy_refined = y_min + np.average(y_coords, weights=weights)

        logger.debug("Center refined: (%.2f, %.2f) -> (%.2f, %.2f)", cx, cy, cx_refined, cy_refined)

        return cx_refined, cy_refined

    def _resolve_v7_sku(self, sku: str, ink: str) -> str:
        if not sku:
            return sku
        if ink and ink.upper() == "INK3" and sku.upper() == "DEFAULT":
            return "DEFAULT_INK3"
        return sku

    def _attach_detection_context(
        self,
        result: InspectionResult,
        lens_detection: LensGeometry,
        image: np.ndarray,
        radial_profile: RadialProfile,
        quality_metrics: Dict[str, Any],
    ) -> InspectionResult:
        """Attach common detection context to an InspectionResult."""
        result.lens_detection = lens_detection
        result.image = image
        result.radial_profile = radial_profile
        result.metrics = quality_metrics
        return result

    def _build_retake_result(
        self,
        sku: str,
        code: str,
        reason: str,
        actions: List[str],
        lever: str,
        ng_reasons: Optional[List[str]] = None,
    ) -> InspectionResult:
        """Build a standardized RETAKE InspectionResult."""
        return InspectionResult(
            sku=sku,
            timestamp=datetime.now(),
            judgment="RETAKE",
            overall_delta_e=0.0,
            ng_reasons=ng_reasons or [],
            confidence=0.0,
            retake_reasons=[
                {
                    "code": code,
                    "reason": reason,
                    "actions": actions,
                    "lever": lever,
                }
            ],
        )

    def process(
        self,
        image_path: str,
        sku: str,
        ink: str = "INK_DEFAULT",
        save_dir: Optional[Path] = None,
        run_1d_judgment: bool = True,
        include_dot_stats: bool = True,
    ) -> InspectionResult:
        """
        단일 이미지 처리.

        Args:
            image_path: 입력 이미지 경로
            sku: SKU 코드
            save_dir: 중간 결과 저장 디렉토리 (옵션)

        Returns:
            InspectionResult: 검사 결과

        Raises:
            PipelineError: 파이프라인 실행 중 오류 발생 시
        """
        start_time = datetime.now()
        image_path = Path(image_path)
        params = self.sku_config.get("params", {})
        sku_used = self._resolve_v7_sku(sku, ink)

        if sku_used != sku:
            logger.info("Processing image: %s, SKU: %s (v7 override: %s)", image_path, sku, sku_used)
        else:
            logger.info(f"Processing image: {image_path}, SKU: {sku}")

        # PHASE7 Priority 5: 진단 정보 수집
        diagnostics = []
        warnings = []
        suggestions = []

        try:
            # 1. 이미지 로드 및 전처리 (retry 로직 포함)
            logger.debug("Step 1: Loading and preprocessing image")
            max_retries = 3
            image = None

            for attempt in range(max_retries):
                try:
                    image = self._load_image(image_path)

                    # None 체크
                    if image is None:
                        if attempt < max_retries - 1:
                            logger.warning(f"Image load attempt {attempt+1}/{max_retries} returned None, retrying...")
                            continue
                        else:
                            raise ValueError("Image loader returned None")

                    processed_image = self._preprocess_image(image)

                    # Preprocess 결과도 체크
                    if processed_image is None:
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Image preprocess attempt {attempt+1}/{max_retries} returned None, retrying..."
                            )
                            continue
                        else:
                            raise ValueError("Image preprocess returned None")

                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Image load attempt {attempt+1}/{max_retries} failed: {e}, retrying...")
                        continue
                    else:
                        logger.error(f"Failed to load image after {max_retries} attempts: {image_path}")
                        # 파일 존재 여부 확인
                        if not image_path.exists():
                            raise PipelineError(
                                f"Image file not found\n"
                                f"  File: {image_path}\n"
                                f"  Suggestion: Check if the file path is correct"
                            )
                        else:
                            raise PipelineError(
                                f"Image load failed: {e}\n"
                                f"  File: {image_path}\n"
                                f"  Suggestion: Check if file is readable and is a valid image format (JPG, PNG, etc.)"
                            )

            # 2. 렌즈 검출 (상세 에러 메시지)
            logger.debug("Step 2: Detecting lens")
            lens_detection = None
            lens_detection = self._detect_lens_v7(processed_image, params)

            if lens_detection is None:
                img_h, img_w = processed_image.shape[:2]
                diagnostics.append("✗ Lens detection failed")
                suggestions.append("→ Check if image contains a clear circular lens")
                suggestions.append("→ Try adjusting detector parameters (min_radius, max_radius)")
                raise PipelineError(
                    f"Lens detection failed\n"
                    f"  File: {image_path}\n"
                    f"  Image size: {img_w}x{img_h}\n"
                    f"  Suggestion: Check if image contains a clear circular lens. "
                    f"Try adjusting detector parameters (min_radius, max_radius) in config."
                )

            # 렌즈 검출 성공
            diagnostics.append(
                f"✓ Lens detected: center=({lens_detection.cx:.1f}, {lens_detection.cy:.1f}), "
                f"radius={lens_detection.r:.1f}, confidence={lens_detection.confidence:.2f}"
            )

            if lens_detection.confidence < 0.5:
                logger.warning(f"Low lens detection confidence: {lens_detection.confidence:.2f}")
                warnings.append(f"⚠ Low lens detection confidence: {lens_detection.confidence:.2f}")
                suggestions.append("→ Verify image quality or adjust detector parameters")

            # 3. 극좌표 변환 및 프로파일 추출
            self._configure_profiler(params)
            num_samples = params.get("num_samples")
            if isinstance(num_samples, (int, float)) and num_samples > 0:
                self._profiler_config.theta_samples = int(num_samples)
            num_points = params.get("num_points")
            if isinstance(num_points, (int, float)) and num_points > 0 and lens_detection.r > 0:
                r_step = max(1, int(lens_detection.r / float(num_points)))
                self._profiler_config.r_step_pixels = r_step
            sample_percentile = params.get("sample_percentile")
            if isinstance(sample_percentile, (int, float)) and 0 <= sample_percentile <= 100:
                self._profiler_config.sample_percentile = float(sample_percentile)

            quality_metrics = self._compute_quality_metrics(
                processed_image,
                lens_detection,
                include_dot_stats=include_dot_stats,
            )
            logger.debug("Step 3: Extracting radial profile")
            radial_profile = self._build_radial_profile_v7(processed_image, lens_detection)

            logger.debug("Step 4: Preparing decision inputs")

            if not run_1d_judgment:
                suggestions.append("Run 2D analysis for final judgment.")
                inspection_result = self._build_retake_result(
                    sku=sku_used,
                    code="1d_judgment_skipped",
                    reason="1D judgment was skipped by configuration.",
                    actions=["Run 2D analysis for final judgment."],
                    lever="use_2d_analysis",
                )
                self._attach_detection_context(
                    inspection_result, lens_detection, image, radial_profile, quality_metrics
                )
                inspection_result.diagnostics = diagnostics if diagnostics else None
                inspection_result.warnings = warnings if warnings else None
                inspection_result.suggestions = suggestions if suggestions else None

                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                logger.info(
                    "Processing complete (1D judgment skipped): "
                    f"time={processing_time:.1f}ms, "
                    f"diagnostics={len(diagnostics)}, warnings={len(warnings)}, suggestions={len(suggestions)}"
                )

                if self.save_intermediates and save_dir:
                    self._save_intermediates(
                        save_dir,
                        image_path.stem,
                        {
                            "processed_image": processed_image,
                            "lens_detection": lens_detection,
                            "radial_profile": radial_profile,
                            "zones": [],
                            "inspection_result": inspection_result,
                            "processing_time_ms": processing_time,
                        },
                    )

                return inspection_result

            # Print area range (SKU config)
            v7_payload = self._evaluate_v7_decision(
                processed_image, sku_used, ink, {**params, "run_id": (save_dir.name if save_dir else "")}
            )
            if v7_payload and "decision" in v7_payload:
                decision = v7_payload["decision"]
                v2_diag = getattr(decision, "diagnostics", {}).get("v2_diagnostics") or {}
                v7_warnings = []
                if isinstance(v2_diag, dict):
                    v7_warnings = list(v2_diag.get("warnings") or [])

                inspection_result = build_inspection_result_from_v7(
                    decision,
                    sku_used,
                    v7_warnings,
                )
                if sku_used != sku:
                    inspection_result.decision_trace = inspection_result.decision_trace or {}
                    inspection_result.decision_trace["sku_override"] = {
                        "input": sku,
                        "used": sku_used,
                        "ink": ink,
                    }
                self._attach_detection_context(
                    inspection_result, lens_detection, image, radial_profile, quality_metrics
                )

                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                logger.info(
                    "Processing complete (v7 decision): "
                    f"time={processing_time:.1f}ms, "
                    f"warnings={len(v7_warnings)}"
                )

                if self.save_intermediates and save_dir:
                    self._save_intermediates(
                        save_dir,
                        image_path.stem,
                        {
                            "processed_image": processed_image,
                            "lens_detection": lens_detection,
                            "radial_profile": radial_profile,
                            "zones": [],
                            "inspection_result": inspection_result,
                            "processing_time_ms": processing_time,
                        },
                    )

                return inspection_result

            # V7 decision unavailable fallback
            inspection_result = self._build_retake_result(
                sku=sku_used,
                code="v7_decision_unavailable",
                reason="V7 decision pipeline was unavailable.",
                actions=["Verify v7 config/models availability."],
                lever="v7_unavailable",
                ng_reasons=["V7_DECISION_UNAVAILABLE"],
            )
            self._attach_detection_context(inspection_result, lens_detection, image, radial_profile, quality_metrics)
            return inspection_result

        except LensDetectionError as e:
            logger.error(f"Lens detection failed: {e}")
            raise PipelineError(
                f"Pipeline failed at lens detection\n"
                f"  Image: {image_path}\n"
                f"  Error: {e}\n"
                f"  Suggestion: Check image quality, adjust detector config, or verify lens is visible"
            )

        except Exception as e:
            logger.error(f"Unexpected error in pipeline: {e}", exc_info=True)
            raise PipelineError(
                f"Pipeline failed with unexpected error\n"
                f"  Image: {image_path}\n"
                f"  Error: {e}\n"
                f"  Suggestion: Check logs for detailed traceback"
            )

    def process_batch(
        self,
        image_paths: List[str],
        sku: str,
        ink: str = "INK_DEFAULT",
        output_csv: Optional[Path] = None,
        continue_on_error: bool = True,
        parallel: bool = False,
        max_workers: int = 4,
    ) -> List[InspectionResult]:
        """
        배치 처리 (옵션으로 병렬 처리 지원).

        Args:
            image_paths: 입력 이미지 경로 리스트
            sku: SKU 코드
            output_csv: 결과 CSV 저장 경로 (옵션)
            continue_on_error: 오류 발생 시 계속 진행 여부
            parallel: 병렬 처리 사용 여부 (기본값: False)
            max_workers: 병렬 처리 시 최대 워커 수 (기본값: 4)

        Returns:
            List[InspectionResult]: 검사 결과 리스트
        """
        logger.info(f"Batch processing {len(image_paths)} images (parallel={parallel})")

        results = []
        errors = []

        if parallel and len(image_paths) > 1:
            # Parallel processing using ThreadPoolExecutor
            import gc
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_path = {executor.submit(self.process, path, sku, ink): path for path in image_paths}

                # Collect results as they complete
                for i, future in enumerate(as_completed(future_to_path)):
                    image_path = future_to_path[future]
                    logger.info(f"Processing {i+1}/{len(image_paths)}: {image_path}")

                    try:
                        result = future.result()
                        results.append(result)

                    except Exception as e:
                        logger.error(f"Error processing {image_path}: {e}")
                        errors.append((image_path, str(e)))

                        if not continue_on_error:
                            raise PipelineError(f"Batch processing failed: {e}")

                    # Release memory periodically
                    if i % 10 == 0:
                        gc.collect()

        else:
            # Sequential processing (original behavior)
            for i, image_path in enumerate(image_paths):
                logger.info(f"Processing {i+1}/{len(image_paths)}: {image_path}")

                try:
                    result = self.process(image_path, sku, ink)
                    results.append(result)

                except PipelineError as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    errors.append((image_path, str(e)))

                    if not continue_on_error:
                        raise

        logger.info(f"Batch processing complete: {len(results)} succeeded, {len(errors)} failed")

        if errors:
            logger.error(f"Failed images ({len(errors)}):")
            for path, err in errors:
                logger.error(f"  - {path}: {err}")

        # CSV 저장 (옵션)
        if output_csv and results:
            self._save_results_csv(results, output_csv)

        return results

    def _save_intermediates(self, save_dir: Path, image_name: str, intermediates: Dict[str, Any]):
        """
        중간 결과 저장.

        Args:
            save_dir: 저장 디렉토리
            image_name: 이미지 이름
            intermediates: 중간 결과 딕셔너리
        """
        import cv2

        output_dir = Path(save_dir) / image_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 처리된 이미지 저장
        if "processed_image" in intermediates:
            cv2.imwrite(str(output_dir / "02_preprocessed.jpg"), intermediates["processed_image"])

        # 렌즈 검출 결과 저장
        if "lens_detection" in intermediates and "processed_image" in intermediates:
            img_with_circle = intermediates["processed_image"].copy()
            detection = intermediates["lens_detection"]
            cv2.circle(
                img_with_circle,
                (int(detection.cx), int(detection.cy)),
                int(detection.r),
                (0, 255, 0),
                2,
            )
            cv2.circle(img_with_circle, (int(detection.cx), int(detection.cy)), 3, (0, 0, 255), -1)
            cv2.imwrite(str(output_dir / "03_lens_detection.jpg"), img_with_circle)

        # 메타데이터 저장
        metadata = {
            "lens_detection": (
                {
                    "center_x": float(intermediates["lens_detection"].cx),
                    "center_y": float(intermediates["lens_detection"].cy),
                    "radius": float(intermediates["lens_detection"].r),
                    "confidence": float(intermediates["lens_detection"].confidence),
                    "method": intermediates["lens_detection"].source,
                }
                if "lens_detection" in intermediates
                else None
            ),
            "zones": [
                {
                    "name": z.name,
                    "r_start": float(z.r_start),
                    "r_end": float(z.r_end),
                    "mean_lab": [float(z.mean_L), float(z.mean_a), float(z.mean_b)],
                    "zone_type": z.zone_type,
                }
                for z in intermediates.get("zones", [])
            ],
            "processing_time_ms": intermediates.get("processing_time_ms"),
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Intermediates saved to {output_dir}")

    def _save_results_csv(self, results: List[InspectionResult], output_path: Path):
        """
        결과를 CSV 파일로 저장.

        Args:
            results: 검사 결과 리스트
            output_path: 출력 CSV 경로
        """
        import csv

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # 헤더
            writer.writerow(["sku", "timestamp", "judgment", "overall_delta_e", "confidence", "ng_reasons"])

            # 데이터
            for result in results:
                writer.writerow(
                    [
                        result.sku,
                        result.timestamp.isoformat(),
                        result.judgment,
                        f"{result.overall_delta_e:.2f}",
                        f"{result.confidence:.2f}",
                        "; ".join(result.ng_reasons) if result.ng_reasons else "",
                    ]
                )

        logger.info(f"Results saved to {output_path}")
