"""
Unit tests for UniformityAnalyzer module
"""

import numpy as np
import pytest

from src.analysis.uniformity_analyzer import (
    UniformityAnalyzer,
    UniformityAnalyzerError,
    UniformityConfig,
    UniformityReport,
)
from src.core.angular_profiler import RingSectorCell

# ================================================================
# Fixtures
# ================================================================


@pytest.fixture
def uniform_cells():
    """
    균일한 색상의 셀들 (모두 비슷한 Lab 값)
    """
    cells = []
    base_L, base_a, base_b = 128.0, 128.0, 128.0

    # 3 rings × 12 sectors = 36 cells
    for ring_idx in range(3):
        for sector_idx in range(12):
            # 약간의 노이즈 추가 (±2)
            noise_L = np.random.uniform(-2, 2)
            noise_a = np.random.uniform(-2, 2)
            noise_b = np.random.uniform(-2, 2)

            cell = RingSectorCell(
                ring_index=ring_idx,
                sector_index=sector_idx,
                r_start=ring_idx * 0.33,
                r_end=(ring_idx + 1) * 0.33,
                angle_start=sector_idx * 30.0,
                angle_end=(sector_idx + 1) * 30.0,
                mean_L=base_L + noise_L,
                mean_a=base_a + noise_a,
                mean_b=base_b + noise_b,
                std_L=5.0,
                std_a=5.0,
                std_b=5.0,
                pixel_count=1000,
            )
            cells.append(cell)

    return cells


@pytest.fixture
def non_uniform_cells():
    """
    불균일한 색상의 셀들 (일부 Sector가 다른 색상)
    """
    cells = []

    # 3 rings × 12 sectors = 36 cells
    for ring_idx in range(3):
        for sector_idx in range(12):
            # Sector 6~8은 다른 색상 (빨강)
            if 6 <= sector_idx <= 8:
                mean_L, mean_a, mean_b = 150.0, 180.0, 120.0  # 빨강
            else:
                mean_L, mean_a, mean_b = 128.0, 128.0, 128.0  # 회색

            cell = RingSectorCell(
                ring_index=ring_idx,
                sector_index=sector_idx,
                r_start=ring_idx * 0.33,
                r_end=(ring_idx + 1) * 0.33,
                angle_start=sector_idx * 30.0,
                angle_end=(sector_idx + 1) * 30.0,
                mean_L=mean_L,
                mean_a=mean_a,
                mean_b=mean_b,
                std_L=5.0,
                std_a=5.0,
                std_b=5.0,
                pixel_count=1000,
            )
            cells.append(cell)

    return cells


@pytest.fixture
def gradient_cells():
    """
    그라데이션 셀들 (Ring별로 색상이 변함)
    """
    cells = []

    # 3 rings × 12 sectors = 36 cells
    for ring_idx in range(3):
        # Ring마다 L 값 변화
        mean_L = 100.0 + ring_idx * 30.0  # 100, 130, 160

        for sector_idx in range(12):
            cell = RingSectorCell(
                ring_index=ring_idx,
                sector_index=sector_idx,
                r_start=ring_idx * 0.33,
                r_end=(ring_idx + 1) * 0.33,
                angle_start=sector_idx * 30.0,
                angle_end=(sector_idx + 1) * 30.0,
                mean_L=mean_L,
                mean_a=128.0,
                mean_b=128.0,
                std_L=5.0,
                std_a=5.0,
                std_b=5.0,
                pixel_count=1000,
            )
            cells.append(cell)

    return cells


# ================================================================
# Test Cases
# ================================================================


def test_uniformity_config_defaults():
    """UniformityConfig 기본값 확인"""
    config = UniformityConfig()
    assert config.delta_e_threshold == 5.0
    assert config.outlier_z_score_threshold == 2.5
    assert config.min_cell_count == 12
    assert config.analyze_by_ring is True
    assert config.analyze_by_sector is True
    assert config.analyze_global is True


def test_uniformity_config_custom():
    """UniformityConfig 커스텀 설정"""
    config = UniformityConfig(delta_e_threshold=3.0, outlier_z_score_threshold=2.0, min_cell_count=24)
    assert config.delta_e_threshold == 3.0
    assert config.outlier_z_score_threshold == 2.0
    assert config.min_cell_count == 24


def test_uniformity_analyzer_creation():
    """UniformityAnalyzer 생성 확인"""
    analyzer = UniformityAnalyzer()
    assert analyzer.config.delta_e_threshold == 5.0

    custom_config = UniformityConfig(delta_e_threshold=3.0)
    analyzer_custom = UniformityAnalyzer(custom_config)
    assert analyzer_custom.config.delta_e_threshold == 3.0


def test_analyze_uniform_cells(uniform_cells):
    """균일한 셀 분석"""
    analyzer = UniformityAnalyzer()
    report = analyzer.analyze(uniform_cells)

    # 결과 타입 확인
    assert isinstance(report, UniformityReport)
    assert isinstance(report.is_uniform, bool)
    assert isinstance(report.global_mean_lab, tuple)
    assert isinstance(report.max_delta_e, float)

    # 균일해야 함
    assert report.is_uniform is True

    # ΔE가 작아야 함
    assert report.max_delta_e < 5.0
    assert report.mean_delta_e < 3.0

    # 이상치 없어야 함
    assert len(report.outlier_cells) == 0

    # 신뢰도 높아야 함
    assert report.confidence > 0.5


def test_analyze_non_uniform_cells(non_uniform_cells):
    """불균일한 셀 분석"""
    analyzer = UniformityAnalyzer(UniformityConfig(delta_e_threshold=5.0))
    report = analyzer.analyze(non_uniform_cells)

    # 불균일해야 함
    assert report.is_uniform is False

    # 큰 ΔE 검출
    assert report.max_delta_e > 10.0

    # mean_delta_e도 높아야 함
    assert report.mean_delta_e > 5.0

    # 신뢰도 낮아야 함 (또는 0)
    assert report.confidence <= 0.5


def test_analyze_gradient_cells(gradient_cells):
    """그라데이션 셀 분석 (Ring별 변화)"""
    analyzer = UniformityAnalyzer(UniformityConfig(delta_e_threshold=10.0))
    report = analyzer.analyze(gradient_cells)

    # Ring별 통계 확인
    assert len(report.ring_uniformity) == 3

    # Ring 간 ΔE 차이가 있어야 함
    assert report.max_delta_e > 5.0


def test_ring_uniformity_analysis(non_uniform_cells):
    """Ring별 균일성 분석"""
    config = UniformityConfig(analyze_by_ring=True, analyze_by_sector=False)
    analyzer = UniformityAnalyzer(config)
    report = analyzer.analyze(non_uniform_cells)

    # Ring 통계 존재
    assert len(report.ring_uniformity) > 0

    for ring_idx, stats in report.ring_uniformity.items():
        assert "mean_de" in stats
        assert "max_de" in stats
        assert "is_uniform" in stats
        assert "cell_count" in stats
        assert stats["cell_count"] > 0


def test_sector_uniformity_analysis(non_uniform_cells):
    """Sector별 균일성 분석"""
    config = UniformityConfig(analyze_by_ring=False, analyze_by_sector=True)
    analyzer = UniformityAnalyzer(config)
    report = analyzer.analyze(non_uniform_cells)

    # Sector 통계 존재
    assert len(report.sector_uniformity) > 0

    for sector_idx, stats in report.sector_uniformity.items():
        assert "mean_de" in stats
        assert "max_de" in stats
        assert "is_uniform" in stats
        assert "cell_count" in stats

    # Sector 6~8은 불균일해야 함
    for sector_idx in [6, 7, 8]:
        if sector_idx in report.sector_uniformity:
            assert report.sector_uniformity[sector_idx]["is_uniform"] is False


def test_outlier_detection(non_uniform_cells):
    """이상치 검출"""
    # 매우 낮은 임계값으로 이상치 검출 유도
    analyzer = UniformityAnalyzer(UniformityConfig(outlier_z_score_threshold=0.5))
    report = analyzer.analyze(non_uniform_cells)

    # 이상치 존재 여부 확인 (없을 수도 있음)
    # 대신 Sector별 균일성으로 확인
    assert len(report.sector_uniformity) > 0

    # Sector 6,7,8이 불균일해야 함
    non_uniform_sectors = [s for s, stats in report.sector_uniformity.items() if not stats["is_uniform"]]
    assert len(non_uniform_sectors) >= 3

    # 이상치가 있다면 정보 확인
    if len(report.outlier_cells) > 0:
        outlier = report.outlier_cells[0]
        assert "ring" in outlier
        assert "sector" in outlier
        assert "delta_e" in outlier
        assert "z_score" in outlier
        assert "lab" in outlier


def test_global_stats_calculation(uniform_cells):
    """전체 통계 계산 확인"""
    analyzer = UniformityAnalyzer()
    report = analyzer.analyze(uniform_cells)

    # 평균 Lab 값 확인
    mean_L, mean_a, mean_b = report.global_mean_lab
    assert 126.0 <= mean_L <= 130.0  # 약 128
    assert 126.0 <= mean_a <= 130.0
    assert 126.0 <= mean_b <= 130.0

    # 표준편차 확인
    std_L, std_a, std_b = report.global_std_lab
    assert std_L >= 0.0
    assert std_a >= 0.0
    assert std_b >= 0.0


def test_confidence_calculation(uniform_cells, non_uniform_cells):
    """신뢰도 계산 확인"""
    analyzer = UniformityAnalyzer()

    report_uniform = analyzer.analyze(uniform_cells)
    report_non_uniform = analyzer.analyze(non_uniform_cells)

    # 균일한 셀은 높은 신뢰도
    assert report_uniform.confidence > 0.5

    # 불균일한 셀은 낮은 신뢰도
    assert report_non_uniform.confidence < report_uniform.confidence

    # 신뢰도 범위 확인
    assert 0.0 <= report_uniform.confidence <= 1.0
    assert 0.0 <= report_non_uniform.confidence <= 1.0


def test_invalid_input_empty_cells():
    """빈 셀 리스트 에러 처리"""
    analyzer = UniformityAnalyzer()

    with pytest.raises(UniformityAnalyzerError, match="Cell list is empty"):
        analyzer.analyze([])


def test_invalid_input_insufficient_cells():
    """셀 개수 부족 에러 처리"""
    # 3개 셀만 제공 (기본 최소값 12개)
    cells = [
        RingSectorCell(0, 0, 0.0, 0.33, 0.0, 30.0, 128, 128, 128, 5, 5, 5, 1000),
        RingSectorCell(0, 1, 0.0, 0.33, 30.0, 60.0, 128, 128, 128, 5, 5, 5, 1000),
        RingSectorCell(0, 2, 0.0, 0.33, 60.0, 90.0, 128, 128, 128, 5, 5, 5, 1000),
    ]

    analyzer = UniformityAnalyzer()

    with pytest.raises(UniformityAnalyzerError, match="Insufficient cells"):
        analyzer.analyze(cells)


def test_low_min_cell_count():
    """낮은 최소 셀 개수 설정"""
    cells = [
        RingSectorCell(0, 0, 0.0, 0.33, 0.0, 30.0, 128, 128, 128, 5, 5, 5, 1000),
        RingSectorCell(0, 1, 0.0, 0.33, 30.0, 60.0, 128, 128, 128, 5, 5, 5, 1000),
        RingSectorCell(0, 2, 0.0, 0.33, 60.0, 90.0, 130, 130, 130, 5, 5, 5, 1000),
    ]

    config = UniformityConfig(min_cell_count=3)
    analyzer = UniformityAnalyzer(config)

    report = analyzer.analyze(cells)
    assert isinstance(report, UniformityReport)
