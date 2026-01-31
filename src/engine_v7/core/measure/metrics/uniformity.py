"""
Uniformity Analyzer Module

Self-Referenced Uniformity Analysis for v7 Engine.
Analyzes color deviation between Ring x Sector cells without SKU reference.
Ported from src/analysis/uniformity_analyzer.py.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from ...types import RingSectorCell
from ...utils import delta_e_cie2000

logger = logging.getLogger(__name__)


@dataclass
class UniformityConfig:
    """
    Uniformity Analysis Configuration

    Attributes:
        delta_e_threshold: Uniformity threshold (ΔE)
        outlier_z_score_threshold: Z-score threshold for outlier detection
        min_cell_count: Minimum number of cells required
        analyze_by_ring: Whether to analyze by ring
        analyze_by_sector: Whether to analyze by sector
        analyze_global: Whether to analyze global uniformity
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
    Uniformity Analysis Report

    Attributes:
        is_uniform: Global uniformity verdict (True=Uniform)
        global_mean_lab: Global mean Lab
        global_std_lab: Global std Lab
        max_delta_e: Max ΔE between cells and global mean
        mean_delta_e: Mean ΔE
        outlier_cells: List of outlier cells
        ring_uniformity: Per-ring uniformity stats
        sector_uniformity: Per-sector uniformity stats
        confidence: Analysis confidence (0.0~1.0)
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
    """Exception during UniformityAnalyzer processing"""

    pass


class UniformityAnalyzer:
    """
    Self-Referenced Uniformity Analyzer

    Analyzes Ring x Sector cells to detect:
    - Global uniformity
    - Ring-wise uniformity (Radial)
    - Sector-wise uniformity (Angular)
    - Outlier cells

    Algorithm:
    1. Calculate global mean Lab
    2. Calculate ΔE for each cell against global mean
    3. Detect outliers using Z-score
    4. Compute Ring/Sector group statistics
    """

    def __init__(self, config: UniformityConfig = None):
        """
        Initialize UniformityAnalyzer

        Args:
            config: Configuration (uses default if None)
        """
        self.config = config or UniformityConfig()
        # logger.info(f"UniformityAnalyzer initialized: ΔE threshold={self.config.delta_e_threshold}")

    def analyze(self, cells: List[RingSectorCell]) -> UniformityReport:
        """
        Analyze uniformity of Ring x Sector cells

        Args:
            cells: List of RingSectorCell

        Returns:
            UniformityReport

        Raises:
            UniformityAnalyzerError: If insufficient cells
        """
        # Input validation
        if cells is None or len(cells) == 0:
            raise UniformityAnalyzerError("Cell list is empty")

        if len(cells) < self.config.min_cell_count:
            raise UniformityAnalyzerError(f"Insufficient cells: {len(cells)} < {self.config.min_cell_count}")

        # 1. Global stats
        global_mean_lab, global_std_lab = self._calculate_global_stats(cells)

        # 2. Calculate ΔE per cell
        delta_e_list = []
        for cell in cells:
            cell_lab = (cell.mean_L, cell.mean_a, cell.mean_b)
            de = delta_e_cie2000(cell_lab, global_mean_lab)
            delta_e_list.append(de)

        delta_e_array = np.array(delta_e_list)

        # 3. Detect outliers
        outlier_cells = self._detect_outliers(cells, delta_e_array)

        # 4. Ring stats
        ring_uniformity = {}
        if self.config.analyze_by_ring:
            ring_uniformity = self._analyze_by_ring(cells, global_mean_lab)

        # 5. Sector stats
        sector_uniformity = {}
        if self.config.analyze_by_sector:
            sector_uniformity = self._analyze_by_sector(cells, global_mean_lab)

        # 6. Global verdict
        max_delta_e = float(np.max(delta_e_array))
        mean_delta_e = float(np.mean(delta_e_array))
        is_uniform = max_delta_e <= self.config.delta_e_threshold and len(outlier_cells) == 0

        # 7. Confidence
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

        return report

    def _calculate_global_stats(
        self, cells: List[RingSectorCell]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Calculate global weighted mean and std deviation"""
        total_pixels = sum(cell.pixel_count for cell in cells)

        if total_pixels == 0:
            # Fallback: simple mean
            mean_L = np.mean([cell.mean_L for cell in cells])
            mean_a = np.mean([cell.mean_a for cell in cells])
            mean_b = np.mean([cell.mean_b for cell in cells])

            std_L = np.std([cell.mean_L for cell in cells])
            std_a = np.std([cell.mean_a for cell in cells])
            std_b = np.std([cell.mean_b for cell in cells])
        else:
            # Weighted mean
            mean_L = sum(cell.mean_L * cell.pixel_count for cell in cells) / total_pixels
            mean_a = sum(cell.mean_a * cell.pixel_count for cell in cells) / total_pixels
            mean_b = sum(cell.mean_b * cell.pixel_count for cell in cells) / total_pixels

            # Weighted variance
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
        """Detect outliers using Z-score"""
        outliers = []

        mean_de = np.mean(delta_e_array)
        std_de = np.std(delta_e_array)

        if std_de == 0:
            return outliers

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
        """Analyze uniformity per Ring"""
        ring_stats = {}
        rings = {}
        for cell in cells:
            if cell.ring_index not in rings:
                rings[cell.ring_index] = []
            rings[cell.ring_index].append(cell)

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
        """Analyze uniformity per Sector"""
        sector_stats = {}
        sectors = {}
        for cell in cells:
            if cell.sector_index not in sectors:
                sectors[cell.sector_index] = []
            sectors[cell.sector_index].append(cell)

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
        """Calculate analysis confidence (0.0~1.0)"""
        mean_de = np.mean(delta_e_array)
        confidence = 1.0 - (mean_de / self.config.delta_e_threshold)
        return float(max(0.0, min(1.0, confidence)))
