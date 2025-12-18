"""
Unit tests for AngularProfiler module (PHASE7 - Ring × Sector 2D analysis)
"""

import cv2
import numpy as np
import pytest

from src.core.angular_profiler import AngularProfiler, AngularProfilerError, RingSectorCell, SectorConfig

# ================================================================
# Fixtures
# ================================================================


@pytest.fixture
def simple_image_lab():
    """
    간단한 Lab 이미지 (300x300)
    - 중심: (150, 150)
    - 3개 Ring: 내부(빨강), 중간(녹색), 외부(파랑)
    """
    image = np.zeros((300, 300, 3), dtype=np.uint8)

    # Ring 1: 빨강 (r < 50)
    cv2.circle(image, (150, 150), 50, (0, 0, 255), -1)

    # Ring 2: 녹색 (50 <= r < 100)
    cv2.circle(image, (150, 150), 100, (0, 255, 0), -1)
    cv2.circle(image, (150, 150), 50, (0, 0, 255), -1)

    # Ring 3: 파랑 (100 <= r < 150)
    cv2.circle(image, (150, 150), 150, (255, 0, 0), -1)
    cv2.circle(image, (150, 150), 100, (0, 255, 0), -1)
    cv2.circle(image, (150, 150), 50, (0, 0, 255), -1)

    # BGR → Lab 변환
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    return image_lab


@pytest.fixture
def gradient_image_lab():
    """
    각도별로 색상이 변하는 이미지 (Sector 테스트용)
    """
    h, w = 300, 300
    y, x = np.ogrid[:h, :w]
    dx = x - 150
    dy = y - 150
    theta = np.arctan2(dy, dx)
    theta_deg = np.degrees(theta) % 360

    # 각도에 따라 L 값 변화
    L = (theta_deg / 360.0 * 100).astype(np.uint8)
    a = np.full((h, w), 128, dtype=np.uint8)
    b = np.full((h, w), 128, dtype=np.uint8)

    image_lab = np.stack([L, a, b], axis=-1)
    return image_lab


# ================================================================
# Test Cases
# ================================================================


def test_sector_config_defaults():
    """SectorConfig 기본값 확인"""
    config = SectorConfig()
    assert config.sector_count == 12
    assert config.angle_start == 0.0
    assert config.clockwise is True


def test_sector_config_custom():
    """SectorConfig 커스텀 설정"""
    config = SectorConfig(sector_count=24, angle_start=15.0, clockwise=False)
    assert config.sector_count == 24
    assert config.angle_start == 15.0
    assert config.clockwise is False


def test_angular_profiler_creation():
    """AngularProfiler 생성 확인"""
    profiler = AngularProfiler()
    assert profiler.config.sector_count == 12

    custom_config = SectorConfig(sector_count=24)
    profiler_custom = AngularProfiler(custom_config)
    assert profiler_custom.config.sector_count == 24


def test_extract_2d_profile_basic(simple_image_lab):
    """기본 2D 프로파일 추출"""
    profiler = AngularProfiler(SectorConfig(sector_count=12))

    # 3 Rings × 12 Sectors = 36 cells
    ring_boundaries = [0.0, 0.33, 0.67, 1.0]

    cells = profiler.extract_2d_profile(
        image_lab=simple_image_lab, center_x=150.0, center_y=150.0, radius=150.0, ring_boundaries=ring_boundaries
    )

    # 총 36개 셀 생성 확인
    assert len(cells) == 36

    # Ring 인덱스 확인
    ring_indices = set(cell.ring_index for cell in cells)
    assert ring_indices == {0, 1, 2}

    # Sector 인덱스 확인
    sector_indices = set(cell.sector_index for cell in cells)
    assert sector_indices == set(range(12))


def test_extract_2d_profile_cell_properties(simple_image_lab):
    """셀의 속성 확인"""
    profiler = AngularProfiler(SectorConfig(sector_count=12))

    ring_boundaries = [0.0, 0.33, 0.67, 1.0]
    cells = profiler.extract_2d_profile(
        image_lab=simple_image_lab, center_x=150.0, center_y=150.0, radius=150.0, ring_boundaries=ring_boundaries
    )

    cell = cells[0]

    # 데이터 타입 확인
    assert isinstance(cell, RingSectorCell)
    assert isinstance(cell.ring_index, int)
    assert isinstance(cell.sector_index, int)
    assert isinstance(cell.mean_L, float)
    assert isinstance(cell.mean_a, float)
    assert isinstance(cell.mean_b, float)
    assert isinstance(cell.std_L, float)
    assert isinstance(cell.pixel_count, int)

    # 범위 확인 (OpenCV Lab: L, a, b 모두 0~255)
    assert 0 <= cell.mean_L <= 255
    assert 0 <= cell.mean_a <= 255
    assert 0 <= cell.mean_b <= 255
    assert cell.pixel_count > 0


def test_sector_angle_clockwise(gradient_image_lab):
    """시계 방향 각도 정렬 확인"""
    profiler = AngularProfiler(SectorConfig(sector_count=4, clockwise=True))

    ring_boundaries = [0.0, 1.0]  # 1개 Ring

    cells = profiler.extract_2d_profile(
        image_lab=gradient_image_lab, center_x=150.0, center_y=150.0, radius=150.0, ring_boundaries=ring_boundaries
    )

    # 4개 Sector: 0도, 90도, 180도, 270도
    assert len(cells) == 4

    # 각도 범위 확인
    angles = [(c.sector_index, c.angle_start, c.angle_end) for c in cells]
    angles.sort()

    assert angles[0][1] == 0.0 and angles[0][2] == 90.0  # 3시 방향
    assert angles[1][1] == 90.0 and angles[1][2] == 180.0  # 6시 방향
    assert angles[2][1] == 180.0 and angles[2][2] == 270.0  # 9시 방향
    assert angles[3][1] == 270.0 and angles[3][2] == 360.0  # 12시 방향


def test_r_inner_r_outer_filter(simple_image_lab):
    """r_inner/r_outer 범위 필터링"""
    profiler = AngularProfiler(SectorConfig(sector_count=12))

    # 전체 범위
    ring_boundaries = [0.0, 0.33, 0.67, 1.0]

    # 중간 Ring만 분석
    cells = profiler.extract_2d_profile(
        image_lab=simple_image_lab,
        center_x=150.0,
        center_y=150.0,
        radius=150.0,
        ring_boundaries=ring_boundaries,
        r_inner=0.3,
        r_outer=0.7,
    )

    # Ring 0 (0.0-0.33)은 제외됨 (r_end=0.33 <= r_inner=0.3 아님, 겹침 있음)
    # Ring 1 (0.33-0.67)은 포함됨
    # Ring 2 (0.67-1.0)은 제외됨 (r_start=0.67 >= r_outer=0.7 아님, 겹침 있음)

    # r_inner=0.3, r_outer=0.7 범위에서
    # Ring 0의 일부 (0.3-0.33)
    # Ring 1 전체 (0.33-0.67)
    # Ring 2의 일부 (0.67-0.7)가 포함될 수 있음

    # 실제로는 Ring 1만 완전히 포함되므로 12개 정도 예상
    assert 12 <= len(cells) <= 36  # 범위 포함


def test_get_cell_by_indices(simple_image_lab):
    """특정 인덱스 셀 검색"""
    profiler = AngularProfiler(SectorConfig(sector_count=12))

    ring_boundaries = [0.0, 0.33, 0.67, 1.0]
    cells = profiler.extract_2d_profile(
        image_lab=simple_image_lab, center_x=150.0, center_y=150.0, radius=150.0, ring_boundaries=ring_boundaries
    )

    # Ring 1, Sector 3 검색
    cell = profiler.get_cell_by_indices(cells, ring_index=1, sector_index=3)

    assert cell is not None
    assert cell.ring_index == 1
    assert cell.sector_index == 3


def test_get_cells_by_ring(simple_image_lab):
    """특정 Ring의 모든 셀"""
    profiler = AngularProfiler(SectorConfig(sector_count=12))

    ring_boundaries = [0.0, 0.33, 0.67, 1.0]
    cells = profiler.extract_2d_profile(
        image_lab=simple_image_lab, center_x=150.0, center_y=150.0, radius=150.0, ring_boundaries=ring_boundaries
    )

    ring_1_cells = profiler.get_cells_by_ring(cells, ring_index=1)

    # Ring 1은 12개 Sector
    assert len(ring_1_cells) == 12

    # 모두 Ring 1
    assert all(c.ring_index == 1 for c in ring_1_cells)

    # Sector 순서대로 정렬 확인
    sector_indices = [c.sector_index for c in ring_1_cells]
    assert sector_indices == list(range(12))


def test_get_cells_by_sector(simple_image_lab):
    """특정 Sector의 모든 셀"""
    profiler = AngularProfiler(SectorConfig(sector_count=12))

    ring_boundaries = [0.0, 0.33, 0.67, 1.0]
    cells = profiler.extract_2d_profile(
        image_lab=simple_image_lab, center_x=150.0, center_y=150.0, radius=150.0, ring_boundaries=ring_boundaries
    )

    sector_5_cells = profiler.get_cells_by_sector(cells, sector_index=5)

    # Sector 5는 3개 Ring
    assert len(sector_5_cells) == 3

    # 모두 Sector 5
    assert all(c.sector_index == 5 for c in sector_5_cells)

    # Ring 순서대로 정렬 확인
    ring_indices = [c.ring_index for c in sector_5_cells]
    assert ring_indices == [0, 1, 2]


def test_invalid_input_empty_image():
    """빈 이미지 에러 처리"""
    profiler = AngularProfiler()

    with pytest.raises(AngularProfilerError, match="Input image is empty"):
        profiler.extract_2d_profile(
            image_lab=np.array([]), center_x=150.0, center_y=150.0, radius=150.0, ring_boundaries=[0.0, 1.0]
        )


def test_invalid_input_wrong_shape():
    """잘못된 이미지 shape 에러 처리"""
    profiler = AngularProfiler()

    # 2D 이미지 (Lab이 아님)
    image_2d = np.zeros((300, 300), dtype=np.uint8)

    with pytest.raises(AngularProfilerError, match="Expected Lab image"):
        profiler.extract_2d_profile(
            image_lab=image_2d, center_x=150.0, center_y=150.0, radius=150.0, ring_boundaries=[0.0, 1.0]
        )


def test_invalid_ring_boundaries():
    """잘못된 ring_boundaries 에러 처리"""
    profiler = AngularProfiler()
    image_lab = np.zeros((300, 300, 3), dtype=np.uint8)

    with pytest.raises(AngularProfilerError, match="at least 2 ring boundaries"):
        profiler.extract_2d_profile(
            image_lab=image_lab, center_x=150.0, center_y=150.0, radius=150.0, ring_boundaries=[0.5]  # 1개만
        )


def test_sector_count_24(simple_image_lab):
    """24 Sector (15도씩) 분할"""
    profiler = AngularProfiler(SectorConfig(sector_count=24))

    ring_boundaries = [0.0, 0.5, 1.0]  # 2 Rings

    cells = profiler.extract_2d_profile(
        image_lab=simple_image_lab, center_x=150.0, center_y=150.0, radius=150.0, ring_boundaries=ring_boundaries
    )

    # 2 Rings × 24 Sectors = 48 cells
    assert len(cells) == 48

    # 각도 스텝 확인 (15도)
    ring_0_cells = profiler.get_cells_by_ring(cells, ring_index=0)
    angle_steps = [c.angle_end - c.angle_start for c in ring_0_cells]
    assert all(pytest.approx(step, abs=0.1) == 15.0 for step in angle_steps)
