"""
Sector Segmenter Module - PHASE7 Priority 0 (Critical)

Ring × Sector 2D 분할로 렌즈를 분석하고, 각 셀의 Lab 평균 및 ΔE를 계산합니다.
각도별 색상 불균일 검출이 가능하여 방사형 분석의 한계를 극복합니다.

ANALYSIS_IMPROVEMENTS 핵심 기능 - AI 템플릿 분석 방식 기반
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SectorConfig:
    """Sector 분할 설정"""

    sector_count: int = 12  # Sector 개수 (기본 12 = 30도씩)
    ring_boundaries: List[float] = None  # Ring 경계 (0~1 정규화)

    def __post_init__(self):
        if self.ring_boundaries is None:
            self.ring_boundaries = [0.0, 0.33, 0.67, 1.0]  # 3 rings by default


@dataclass
class SectorSegmentationResult:
    """Sector 분할 결과"""

    cells: List  # SectorCell 리스트 (angular_profiler에서 정의)
    ring_boundaries: List[float]
    sector_count: int
    r_inner: float
    r_outer: float
    total_cells: int
    valid_pixel_ratio: float


class SectorSegmenter:
    """
    Ring × Sector 2D 분할 및 분석

    PHASE7 Priority 0 (Critical): 각도별 불균일 검출 필수

    통합 기능:
    1. Lab 색공간 변환
    2. 조명 편차 보정 (optional)
    3. 배경 마스킹
    4. 경계 자동 검출 (r_inner/r_outer)
    5. Angular profiling (Ring × Sector)
    6. 균일성 분석
    """

    def __init__(self, config: Optional[SectorConfig] = None):
        """
        Args:
            config: Sector 분할 설정 (None이면 기본값)
        """
        self.config = config or SectorConfig()
        logger.info(
            f"SectorSegmenter initialized: {len(self.config.ring_boundaries)-1} rings, "
            f"{self.config.sector_count} sectors"
        )

    def segment_and_analyze(
        self,
        image_bgr: np.ndarray,
        center_x: float,
        center_y: float,
        radius: float,
        radial_profile=None,
        enable_illumination_correction: bool = False,
    ) -> Tuple[SectorSegmentationResult, Optional[dict]]:
        """
        Ring × Sector 2D 분할 및 균일성 분석 (통합 파이프라인)

        Args:
            image_bgr: BGR 이미지 (H × W × 3)
            center_x: 렌즈 중심 x 좌표 (픽셀)
            center_y: 렌즈 중심 y 좌표 (픽셀)
            radius: 렌즈 반경 (픽셀)
            radial_profile: 방사형 프로파일 (경계 검출용, optional)
            enable_illumination_correction: 조명 보정 사용 여부

        Returns:
            tuple: (SectorSegmentationResult, uniformity_data)
                - SectorSegmentationResult: 분할 결과
                - uniformity_data: 균일성 분석 결과 dict (None if failed)
        """
        # 1. Lab 색공간 변환
        if len(image_bgr.shape) == 3:
            image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
        else:
            image_lab = image_bgr  # Already Lab
            logger.warning("Input image already in Lab format")

        # 2. 조명 편차 보정 (optional)
        if enable_illumination_correction:
            image_lab = self._apply_illumination_correction(image_lab, center_x, center_y, radius)

        # 3. 경계 자동 검출 (r_inner/r_outer)
        r_inner_detected, r_outer_detected = self._detect_boundaries(
            radial_profile, default_inner=0.0, default_outer=1.0
        )

        # 4. 배경 마스킹
        background_mask = self._create_background_mask(image_lab, center_x, center_y, radius)

        # 5. Angular profiling (Ring × Sector)
        cells = self._extract_2d_profile(
            image_lab,
            center_x,
            center_y,
            radius,
            r_inner_detected,
            r_outer_detected,
            background_mask,
        )

        # 6. 결과 구성
        valid_pixels = sum(cell.pixel_count for cell in cells)
        total_area = np.pi * radius**2
        valid_ratio = valid_pixels / total_area if total_area > 0 else 0.0

        segmentation_result = SectorSegmentationResult(
            cells=cells,
            ring_boundaries=self.config.ring_boundaries,
            sector_count=self.config.sector_count,
            r_inner=r_inner_detected,
            r_outer=r_outer_detected,
            total_cells=len(cells),
            valid_pixel_ratio=valid_ratio,
        )

        logger.info(
            f"Segmentation complete: {len(cells)} cells, "
            f"{valid_pixels:,} pixels ({valid_ratio*100:.1f}% of lens area)"
        )

        # 7. 균일성 분석
        uniformity_data = self._analyze_uniformity(cells)

        return segmentation_result, uniformity_data

    def _apply_illumination_correction(
        self, image_lab: np.ndarray, center_x: float, center_y: float, radius: float
    ) -> np.ndarray:
        """조명 편차 보정 적용"""
        try:
            from src.core.illumination_corrector import CorrectorConfig, IlluminationCorrector

            corrector = IlluminationCorrector(CorrectorConfig(enabled=True))
            correction_result = corrector.correct(
                image_lab=image_lab,
                center_x=center_x,
                center_y=center_y,
                radius=radius,
            )
            if correction_result.correction_applied:
                logger.info("Illumination correction applied")
                return correction_result.corrected_image
            else:
                logger.warning("Illumination correction skipped")
                return image_lab
        except ImportError:
            logger.warning("IlluminationCorrector not available, skipping correction")
            return image_lab
        except Exception as e:
            logger.warning(f"Illumination correction failed: {e}")
            return image_lab

    def _detect_boundaries(self, radial_profile, default_inner: float, default_outer: float) -> Tuple[float, float]:
        """경계 자동 검출 (r_inner/r_outer)"""
        if radial_profile is None:
            logger.info(
                f"No radial profile provided, using defaults: " f"r_inner={default_inner}, r_outer={default_outer}"
            )
            return default_inner, default_outer

        try:
            from src.core.boundary_detector import BoundaryConfig, BoundaryDetector

            boundary_detector = BoundaryDetector(BoundaryConfig())
            boundaries = boundary_detector.detect_boundaries(radial_profile)
            r_inner = boundaries.r_inner
            r_outer = boundaries.r_outer

            # Safety: Ring 0 보호 (0.0~0.33)
            if r_inner > 0.25:
                logger.warning(f"r_inner={r_inner:.3f} too large (>0.25), forcing to 0.0")
                r_inner = 0.0

            logger.info(
                f"Auto-detected boundaries: r_inner={r_inner:.3f}, "
                f"r_outer={r_outer:.3f}, confidence={boundaries.confidence:.2f}"
            )

            return r_inner, r_outer
        except ImportError:
            logger.warning("BoundaryDetector not available, using defaults")
            return default_inner, default_outer
        except Exception as e:
            logger.warning(f"Boundary detection failed: {e}, using defaults")
            return default_inner, default_outer

    def _create_background_mask(
        self, image_lab: np.ndarray, center_x: float, center_y: float, radius: float
    ) -> np.ndarray:
        """배경 마스크 생성"""
        try:
            from src.core.background_masker import BackgroundMasker, MaskConfig

            masker = BackgroundMasker(MaskConfig())
            mask_result = masker.create_mask(
                image_lab=image_lab,
                center_x=center_x,
                center_y=center_y,
                radius=radius,
            )
            logger.info(f"Background mask created: {mask_result.valid_pixel_ratio * 100:.1f}% valid pixels")
            return mask_result.mask
        except ImportError:
            logger.warning("BackgroundMasker not available, using full mask")
            h, w = image_lab.shape[:2]
            return np.ones((h, w), dtype=bool)
        except Exception as e:
            logger.warning(f"Background masking failed: {e}, using full mask")
            h, w = image_lab.shape[:2]
            return np.ones((h, w), dtype=bool)

    def _extract_2d_profile(
        self,
        image_lab: np.ndarray,
        center_x: float,
        center_y: float,
        radius: float,
        r_inner: float,
        r_outer: float,
        mask: np.ndarray,
    ) -> List:
        """Angular profiling (Ring × Sector)"""
        try:
            from src.core.angular_profiler import AngularProfiler, SectorConfig

            angular_profiler = AngularProfiler(SectorConfig(sector_count=self.config.sector_count))
            cells = angular_profiler.extract_2d_profile(
                image_lab=image_lab,
                center_x=center_x,
                center_y=center_y,
                radius=radius,
                ring_boundaries=self.config.ring_boundaries,
                r_inner=r_inner,
                r_outer=r_outer,
                mask=mask,
            )
            logger.info(f"Extracted {len(cells)} cells from 2D profile")
            return cells
        except ImportError as e:
            logger.error(f"AngularProfiler not available: {e}")
            return []
        except Exception as e:
            logger.error(f"2D profile extraction failed: {e}")
            return []

    def _analyze_uniformity(self, cells: List) -> Optional[dict]:
        """균일성 분석"""
        if not cells:
            logger.warning("No cells to analyze for uniformity")
            return None

        try:
            from src.analysis.uniformity_analyzer import UniformityAnalyzer, UniformityConfig

            uniformity_analyzer = UniformityAnalyzer(UniformityConfig())
            uniformity_report = uniformity_analyzer.analyze(cells)

            uniformity_data = {
                "is_uniform": uniformity_report.is_uniform,
                "global_mean_lab": list(uniformity_report.global_mean_lab),
                "global_std_lab": list(uniformity_report.global_std_lab),
                "max_delta_e": uniformity_report.max_delta_e,
                "mean_delta_e": uniformity_report.mean_delta_e,
                "outlier_cells": uniformity_report.outlier_cells,
                "ring_uniformity": uniformity_report.ring_uniformity,
                "sector_uniformity": uniformity_report.sector_uniformity,
                "confidence": uniformity_report.confidence,
            }

            logger.info(
                f"Uniformity: {'UNIFORM' if uniformity_report.is_uniform else 'NON-UNIFORM'}, "
                f"max ΔE={uniformity_report.max_delta_e:.2f}"
            )

            return uniformity_data
        except ImportError:
            logger.warning("UniformityAnalyzer not available")
            return None
        except Exception as e:
            logger.warning(f"Uniformity analysis failed: {e}")
            return None

    def format_response_data(self, cells: List) -> List[dict]:
        """
        API 응답 형식으로 변환

        Returns:
            List of cell dictionaries for JSON serialization
        """
        response_data = [
            {
                "ring": cell.ring_index,
                "sector": cell.sector_index,
                "angle": f"{cell.angle_start:.0f}-{cell.angle_end:.0f}°",
                "r_range": [float(cell.r_start), float(cell.r_end)],
                "lab": [float(cell.mean_L), float(cell.mean_a), float(cell.mean_b)],
                "std": [float(cell.std_L), float(cell.std_a), float(cell.std_b)],
                "pixel_count": cell.pixel_count,
            }
            for cell in cells
        ]

        return response_data
