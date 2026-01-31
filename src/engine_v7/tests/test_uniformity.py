"""
Test for Uniformity Analyzer (v7 Port)
"""

import pytest

from src.engine_v7.core.measure.metrics.uniformity import UniformityAnalyzer, UniformityConfig, UniformityReport
from src.engine_v7.core.types import RingSectorCell


@pytest.fixture
def dummy_cells():
    """Create dummy RingSectorCells for testing"""
    cells = []
    # Create 3 rings x 12 sectors = 36 cells
    # Ring 0: Uniform
    # Ring 1: One outlier
    # Ring 2: Uniform
    for r in range(3):
        for s in range(12):
            # Base color (Gray-ish)
            L, a, b = 50.0, 0.0, 0.0

            # Inject outlier in Ring 1, Sector 0
            if r == 1 and s == 0:
                L = 80.0  # Significant difference

            cell = RingSectorCell(
                ring_index=r,
                sector_index=s,
                r_start=r * 0.33,
                r_end=(r + 1) * 0.33,
                angle_start=s * 30.0,
                angle_end=(s + 1) * 30.0,
                mean_L=L,
                mean_a=a,
                mean_b=b,
                std_L=1.0,
                std_a=1.0,
                std_b=1.0,
                pixel_count=100,
            )
            cells.append(cell)
    return cells


def test_uniformity_analyzer_init():
    config = UniformityConfig(delta_e_threshold=3.0)
    analyzer = UniformityAnalyzer(config)
    assert analyzer.config.delta_e_threshold == 3.0


def test_analyze_outlier_detection(dummy_cells):
    analyzer = UniformityAnalyzer()
    report = analyzer.analyze(dummy_cells)

    assert isinstance(report, UniformityReport)
    assert report.is_uniform is False  # Should fail due to outlier
    assert len(report.outlier_cells) >= 1

    outlier = report.outlier_cells[0]
    assert outlier["ring"] == 1
    assert outlier["sector"] == 0
    assert outlier["delta_e"] > 10.0


def test_analyze_perfect_uniformity():
    # Create perfectly uniform cells
    cells = []
    for r in range(2):
        for s in range(12):
            cell = RingSectorCell(
                ring_index=r,
                sector_index=s,
                r_start=0,
                r_end=1,
                angle_start=0,
                angle_end=1,
                mean_L=50.0,
                mean_a=0.0,
                mean_b=0.0,
                std_L=0,
                std_a=0,
                std_b=0,
                pixel_count=100,
            )
            cells.append(cell)

    analyzer = UniformityAnalyzer()
    report = analyzer.analyze(cells)

    assert report.is_uniform is True
    assert report.max_delta_e < 0.001
    assert len(report.outlier_cells) == 0
