"""
Uniformity Analyzer Module

자기 참조 균일성 분석 (Self-Referenced Uniformity Analysis)
SKU 기준값 없이 Ring × Sector 셀 간 색상 편차를 분석하여 불균일 검출.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from src.core.angular_profiler import RingSectorCell
from src.utils.color_delta import delta_e_cie2000

logger = logging.getLogger(__name__)


@dataclass
class UniformityConfig:
    """
    균일성 분석 설정

    Attributes:
        delta_e_threshold: 균일성 임계값 (ΔE, 이 값 이상이면 불균일)
        outlier_z_score_threshold: 이상치 검출 Z-score 임계값
        min_cell_count: 최소 셀 개수 (이보다 적으면 분석 불가)
        analyze_by_ring: Ring별 균일성 분석 여부
        analyze_by_sector: Sector별 균일성 분석 여부
        analyze_global: 전체 균일성 분석 여부
    """

    delta_e_threshold: float = 5.0
    outlier_z_score_threshold: float = 2.5
    min_cell_count: int = 12
    analyze_by_ring: bool = True
    analyze_by_sector: bool = True
    analyze_global: bool = True


@dataclass
class UniformityReport:
    """
    균일성 분석 결과

    Attributes:
        is_uniform: 전체 균일 여부 (True=균일)
        global_mean_lab: 전체 평균 Lab 값
        global_std_lab: 전체 표준편차 Lab 값
        max_delta_e: 최대 셀 간 색차 (ΔE)
        mean_delta_e: 평균 셀 간 색차
        outlier_cells: 이상치 셀 리스트 [(ring, sector, delta_e), ...]
        ring_uniformity: Ring별 균일성 {ring_idx: {'mean_de': float, 'is_uniform': bool}, ...}
        sector_uniformity: Sector별 균일성 {sector_idx: {'mean_de': float, 'is_uniform': bool}, ...}
        confidence: 분석 신뢰도 (0.0~1.0)
    """

    is_uniform: bool
    global_mean_lab: Tuple[float, float, float]
    global_std_lab: Tuple[float, float, float]
    max_delta_e: float
    mean_delta_e: float
    outlier_cells: List[Dict[str, Any]]
    ring_uniformity: Dict[int, Dict[str, Any]]
    sector_uniformity: Dict[int, Dict[str, Any]]
    confidence: float


class UniformityAnalyzerError(Exception):
    """UniformityAnalyzer 처리 중 발생하는 예외"""

    pass


class UniformityAnalyzer:
    """
    자기 참조 균일성 분석기

    Ring × Sector 셀 간 색상 편차를 분석하여:
    - 전체 균일성 평가
    - Ring별 균일성 (반경 방향)
    - Sector별 균일성 (각도 방향)
    - 이상치 셀 검출

    알고리즘:
    1. 전체 평균 Lab 계산
    2. 각 셀과 평균 간 ΔE 계산
    3. Z-score로 이상치 검출
    4. Ring별/Sector별 그룹 통계
    """

    def __init__(self, config: UniformityConfig = None):
        """
        UniformityAnalyzer 초기화

        Args:
            config: 균일성 분석 설정 (None이면 기본값 사용)
        """
        self.config = config or UniformityConfig()
        logger.info(f"UniformityAnalyzer initialized: ΔE threshold={self.config.delta_e_threshold}")

    def analyze(self, cells: List[RingSectorCell]) -> UniformityReport:
        """
        Ring × Sector 셀 균일성 분석

        Args:
            cells: RingSectorCell 리스트

        Returns:
            UniformityReport: 균일성 분석 결과

        Raises:
            UniformityAnalyzerError: 셀 개수 부족 등

        Example:
            >>> analyzer = UniformityAnalyzer()
            >>> report = analyzer.analyze(cells)
            >>> print(f"Uniform: {report.is_uniform}, Max ΔE: {report.max_delta_e:.2f}")
        """
        # 입력 검증
        if cells is None or len(cells) == 0:
            raise UniformityAnalyzerError("Cell list is empty")

        if len(cells) < self.config.min_cell_count:
            raise UniformityAnalyzerError(f"Insufficient cells: {len(cells)} < {self.config.min_cell_count}")

        # 1. 전체 평균 Lab 계산
        global_mean_lab, global_std_lab = self._calculate_global_stats(cells)

        # 2. 각 셀과 평균 간 ΔE 계산
        delta_e_list = []
        for cell in cells:
            cell_lab = (cell.mean_L, cell.mean_a, cell.mean_b)
            de = delta_e_cie2000(cell_lab, global_mean_lab)
            delta_e_list.append(de)

        delta_e_array = np.array(delta_e_list)

        # 3. 이상치 검출
        outlier_cells = self._detect_outliers(cells, delta_e_array)

        # 4. Ring별 균일성
        ring_uniformity = {}
        if self.config.analyze_by_ring:
            ring_uniformity = self._analyze_by_ring(cells, global_mean_lab)

        # 5. Sector별 균일성
        sector_uniformity = {}
        if self.config.analyze_by_sector:
            sector_uniformity = self._analyze_by_sector(cells, global_mean_lab)

        # 6. 전체 판정
        max_delta_e = float(np.max(delta_e_array))
        mean_delta_e = float(np.mean(delta_e_array))
        is_uniform = max_delta_e <= self.config.delta_e_threshold and len(outlier_cells) == 0

        # 7. 신뢰도 계산
        confidence = self._calculate_confidence(delta_e_array)

        report = UniformityReport(
            is_uniform=is_uniform,
            global_mean_lab=global_mean_lab,
            global_std_lab=global_std_lab,
            max_delta_e=max_delta_e,
            mean_delta_e=mean_delta_e,
            outlier_cells=outlier_cells,
            ring_uniformity=ring_uniformity,
            sector_uniformity=sector_uniformity,
            confidence=confidence,
        )

        logger.info(
            f"Uniformity analysis: {'UNIFORM' if is_uniform else 'NON-UNIFORM'}, "
            f"max ΔE={max_delta_e:.2f}, outliers={len(outlier_cells)}"
        )

        return report

    def _calculate_global_stats(
        self, cells: List[RingSectorCell]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        전체 평균 및 표준편차 계산

        Args:
            cells: RingSectorCell 리스트

        Returns:
            (평균 Lab, 표준편차 Lab)
        """
        # 픽셀 가중 평균 (픽셀 수가 많은 셀의 영향을 크게)
        total_pixels = sum(cell.pixel_count for cell in cells)

        if total_pixels == 0:
            # Fallback: 단순 평균
            mean_L = np.mean([cell.mean_L for cell in cells])
            mean_a = np.mean([cell.mean_a for cell in cells])
            mean_b = np.mean([cell.mean_b for cell in cells])

            std_L = np.std([cell.mean_L for cell in cells])
            std_a = np.std([cell.mean_a for cell in cells])
            std_b = np.std([cell.mean_b for cell in cells])
        else:
            # 가중 평균
            mean_L = sum(cell.mean_L * cell.pixel_count for cell in cells) / total_pixels
            mean_a = sum(cell.mean_a * cell.pixel_count for cell in cells) / total_pixels
            mean_b = sum(cell.mean_b * cell.pixel_count for cell in cells) / total_pixels

            # 가중 표준편차
            var_L = sum(((cell.mean_L - mean_L) ** 2) * cell.pixel_count for cell in cells) / total_pixels
            var_a = sum(((cell.mean_a - mean_a) ** 2) * cell.pixel_count for cell in cells) / total_pixels
            var_b = sum(((cell.mean_b - mean_b) ** 2) * cell.pixel_count for cell in cells) / total_pixels

            std_L = np.sqrt(var_L)
            std_a = np.sqrt(var_a)
            std_b = np.sqrt(var_b)

        mean_lab = (float(mean_L), float(mean_a), float(mean_b))
        std_lab = (float(std_L), float(std_a), float(std_b))

        return mean_lab, std_lab

    def _detect_outliers(self, cells: List[RingSectorCell], delta_e_array: np.ndarray) -> List[Dict[str, Any]]:
        """
        Z-score 기반 이상치 검출

        Args:
            cells: RingSectorCell 리스트
            delta_e_array: ΔE 배열

        Returns:
            이상치 셀 리스트
        """
        outliers = []

        mean_de = np.mean(delta_e_array)
        std_de = np.std(delta_e_array)

        if std_de == 0:
            # 모든 셀이 동일 → 이상치 없음
            return outliers

        # Z-score 계산
        z_scores = (delta_e_array - mean_de) / std_de

        for i, (cell, z_score) in enumerate(zip(cells, z_scores)):
            if abs(z_score) > self.config.outlier_z_score_threshold:
                outliers.append(
                    {
                        "ring": cell.ring_index,
                        "sector": cell.sector_index,
                        "delta_e": float(delta_e_array[i]),
                        "z_score": float(z_score),
                        "lab": (float(cell.mean_L), float(cell.mean_a), float(cell.mean_b)),
                    }
                )

        return outliers

    def _analyze_by_ring(
        self, cells: List[RingSectorCell], global_mean_lab: Tuple[float, float, float]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Ring별 균일성 분석

        Args:
            cells: RingSectorCell 리스트
            global_mean_lab: 전체 평균 Lab

        Returns:
            Ring별 통계
        """
        ring_stats = {}

        # Ring별 그룹화
        rings = {}
        for cell in cells:
            if cell.ring_index not in rings:
                rings[cell.ring_index] = []
            rings[cell.ring_index].append(cell)

        # Ring별 ΔE 계산
        for ring_idx, ring_cells in rings.items():
            delta_e_list = []
            for cell in ring_cells:
                cell_lab = (cell.mean_L, cell.mean_a, cell.mean_b)
                de = delta_e_cie2000(cell_lab, global_mean_lab)
                delta_e_list.append(de)

            mean_de = float(np.mean(delta_e_list))
            max_de = float(np.max(delta_e_list))
            is_uniform = max_de <= self.config.delta_e_threshold

            ring_stats[ring_idx] = {
                "mean_de": mean_de,
                "max_de": max_de,
                "is_uniform": is_uniform,
                "cell_count": len(ring_cells),
            }

        return ring_stats

    def _analyze_by_sector(
        self, cells: List[RingSectorCell], global_mean_lab: Tuple[float, float, float]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Sector별 균일성 분석

        Args:
            cells: RingSectorCell 리스트
            global_mean_lab: 전체 평균 Lab

        Returns:
            Sector별 통계
        """
        sector_stats = {}

        # Sector별 그룹화
        sectors = {}
        for cell in cells:
            if cell.sector_index not in sectors:
                sectors[cell.sector_index] = []
            sectors[cell.sector_index].append(cell)

        # Sector별 ΔE 계산
        for sector_idx, sector_cells in sectors.items():
            delta_e_list = []
            for cell in sector_cells:
                cell_lab = (cell.mean_L, cell.mean_a, cell.mean_b)
                de = delta_e_cie2000(cell_lab, global_mean_lab)
                delta_e_list.append(de)

            mean_de = float(np.mean(delta_e_list))
            max_de = float(np.max(delta_e_list))
            is_uniform = max_de <= self.config.delta_e_threshold

            sector_stats[sector_idx] = {
                "mean_de": mean_de,
                "max_de": max_de,
                "is_uniform": is_uniform,
                "cell_count": len(sector_cells),
            }

        return sector_stats

    def _calculate_confidence(self, delta_e_array: np.ndarray) -> float:
        """
        분석 신뢰도 계산

        Args:
            delta_e_array: ΔE 배열

        Returns:
            신뢰도 (0.0~1.0)
        """
        # 신뢰도 = 1 - (평균 ΔE / 임계값)
        mean_de = np.mean(delta_e_array)
        confidence = 1.0 - (mean_de / self.config.delta_e_threshold)

        # 0.0~1.0 범위로 clamp
        confidence = max(0.0, min(1.0, confidence))

        return float(confidence)
