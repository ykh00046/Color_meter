"""
Angular Profiler Module

Ring × Sector 2D 분할을 통해 각도별 색상 분포 분석.
PHASE7 핵심 개선사항 - 각도별 불균일 검출 가능.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SectorConfig:
    """
    Sector 분할 설정

    Attributes:
        sector_count: Sector 개수 (12 = 30도씩, 24 = 15도씩)
        angle_start: 시작 각도 (도 단위, 3시 방향 = 0도)
        clockwise: 시계 방향 여부 (PHASE4 표준)
    """

    sector_count: int = 12
    angle_start: float = 0.0
    clockwise: bool = True


@dataclass
class RingSectorCell:
    """
    Ring × Sector 하나의 셀 데이터

    2D 분할의 기본 단위. 하나의 Ring과 하나의 Sector의 교집합 영역.

    Attributes:
        ring_index: Ring 번호 (0부터 시작)
        sector_index: Sector 번호 (0부터 시작, 3시=0, 시계방향)
        r_start: 정규화된 시작 반경 (0.0~1.0)
        r_end: 정규화된 끝 반경 (0.0~1.0)
        angle_start: 시작 각도 (도, 0~360)
        angle_end: 끝 각도 (도, 0~360)
        mean_L: 평균 L* 값
        mean_a: 평균 a* 값
        mean_b: 평균 b* 값
        std_L: 표준편차 L*
        std_a: 표준편차 a*
        std_b: 표준편차 b*
        pixel_count: 이 셀에 포함된 픽셀 개수
    """

    ring_index: int
    sector_index: int
    r_start: float
    r_end: float
    angle_start: float
    angle_end: float
    mean_L: float
    mean_a: float
    mean_b: float
    std_L: float
    std_a: float
    std_b: float
    pixel_count: int


class AngularProfilerError(Exception):
    """AngularProfiler 처리 중 발생하는 예외"""

    pass


class AngularProfiler:
    """
    2D Angular Profiler

    렌즈 이미지를 Ring × Sector로 분할하여 각도별 색상 분포 분석.

    주요 기능:
    - 극좌표 변환 (Cartesian → Polar)
    - Ring별, Sector별 LAB 통계 계산
    - 각도별 불균일 검출 가능

    좌표계 표준 (PHASE4):
    - 3시 방향 = 0도
    - 시계 방향 증가 (CW)
    """

    def __init__(self, config: SectorConfig = None):
        """
        AngularProfiler 초기화

        Args:
            config: Sector 분할 설정 (None이면 기본값 사용)
        """
        self.config = config or SectorConfig()
        logger.info(f"AngularProfiler initialized: {self.config.sector_count} sectors")

    def extract_2d_profile(
        self,
        image_lab: np.ndarray,
        center_x: float,
        center_y: float,
        radius: float,
        ring_boundaries: List[float],
        r_inner: float = 0.0,
        r_outer: float = 1.0,
        mask: Optional[np.ndarray] = None,
    ) -> List[RingSectorCell]:
        """
        Ring × Sector 2D 프로파일 추출

        Args:
            image_lab: Lab 색공간 이미지 (H × W × 3)
            center_x: 렌즈 중심 x 좌표 (픽셀)
            center_y: 렌즈 중심 y 좌표 (픽셀)
            radius: 렌즈 반경 (픽셀)
            ring_boundaries: Ring 경계 리스트 (정규화된 반경, 0.0~1.0)
                           예: [0.0, 0.33, 0.67, 1.0] → 3개 Ring
            r_inner: 분석 시작 반경 (정규화, 0.0~1.0)
            r_outer: 분석 끝 반경 (정규화, 0.0~1.0)
            mask: 배경 마스크 (H × W, bool, True=유효 픽셀, None=마스크 없음)

        Returns:
            RingSectorCell 리스트 (총 (Rings × Sectors)개)

        Raises:
            AngularProfilerError: 이미지 또는 파라미터 오류

        Example:
            >>> profiler = AngularProfiler(SectorConfig(sector_count=12))
            >>> cells = profiler.extract_2d_profile(
            ...     image_lab=image,
            ...     center_x=400, center_y=400, radius=350,
            ...     ring_boundaries=[0.0, 0.33, 0.67, 1.0]
            ... )
            >>> len(cells)  # 3 rings × 12 sectors = 36 cells
            36
        """
        # 입력 검증
        if image_lab is None or image_lab.size == 0:
            raise AngularProfilerError("Input image is empty")

        if len(image_lab.shape) != 3 or image_lab.shape[2] != 3:
            raise AngularProfilerError(f"Expected Lab image (H×W×3), got {image_lab.shape}")

        if len(ring_boundaries) < 2:
            raise AngularProfilerError("Need at least 2 ring boundaries")

        h, w = image_lab.shape[:2]
        cells = []

        logger.debug(
            f"Extracting 2D profile: center=({center_x:.1f}, {center_y:.1f}), "
            f"radius={radius:.1f}, rings={len(ring_boundaries)-1}, "
            f"sectors={self.config.sector_count}"
        )

        # 1. 좌표 그리드 생성
        y, x = np.ogrid[:h, :w]

        # 2. 극좌표 변환
        dx = x - center_x
        dy = y - center_y
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)  # -π ~ π

        # 3. 각도 정규화 (3시 방향 0도, 시계 방향)
        if self.config.clockwise:
            theta = -theta  # 반시계 → 시계 방향 변환

        theta = np.degrees(theta)  # 라디안 → 도
        theta = (theta + self.config.angle_start) % 360  # 시작점 조정 및 0~360 범위

        # 4. 정규화된 반경 계산
        r_norm = r / radius

        # 5. Ring × Sector 셀 추출
        angle_step = 360.0 / self.config.sector_count

        for ring_idx in range(len(ring_boundaries) - 1):
            r_start = ring_boundaries[ring_idx]
            r_end = ring_boundaries[ring_idx + 1]

            # r_inner/r_outer 범위 체크
            if r_end <= r_inner or r_start >= r_outer:
                logger.debug(f"Ring {ring_idx} skipped: outside analysis range")
                continue

            # 실제 분석 범위 조정
            r_start_actual = max(r_start, r_inner)
            r_end_actual = min(r_end, r_outer)

            for sector_idx in range(self.config.sector_count):
                angle_start = sector_idx * angle_step
                angle_end = (sector_idx + 1) * angle_step

                # 마스크 생성
                mask_ring = (r_norm >= r_start_actual) & (r_norm < r_end_actual)
                mask_sector = (theta >= angle_start) & (theta < angle_end)
                cell_mask = mask_ring & mask_sector

                # 배경 마스크 적용 (PHASE7 개선)
                # 단, Ring 0 (중심부)는 마스크를 완화하여 적용 (반사광 영역 보존)
                if mask is not None:
                    if ring_idx == 0:
                        # Ring 0: 원형 마스크만 적용 (휘도/채도 필터 제외)
                        # 중심부는 반사광이 많아 필터에 걸릴 가능성 높음
                        y, x = np.ogrid[: image_lab.shape[0], : image_lab.shape[1]]
                        dx_grid = x - center_x
                        dy_grid = y - center_y
                        dist = np.sqrt(dx_grid**2 + dy_grid**2)
                        circular_mask_only = dist <= radius
                        cell_mask = cell_mask & circular_mask_only
                    else:
                        # Ring 1, 2: 전체 배경 마스크 적용
                        cell_mask = cell_mask & mask

                pixel_count = int(np.sum(cell_mask))

                if pixel_count == 0:
                    logger.warning(f"Cell [Ring {ring_idx}, Sector {sector_idx}] has 0 pixels")
                    continue

                # LAB 통계 계산 - OpenCV Lab → Standard Lab 변환
                from src.utils.color_space import opencv_lab_to_standard

                L_vals_cv = image_lab[cell_mask, 0]
                a_vals_cv = image_lab[cell_mask, 1]
                b_vals_cv = image_lab[cell_mask, 2]

                # OpenCV → Standard Lab 변환
                L_vals, a_vals, b_vals = opencv_lab_to_standard(L_vals_cv, a_vals_cv, b_vals_cv)

                cell = RingSectorCell(
                    ring_index=ring_idx,
                    sector_index=sector_idx,
                    r_start=r_start_actual,
                    r_end=r_end_actual,
                    angle_start=angle_start,
                    angle_end=angle_end,
                    mean_L=float(np.mean(L_vals)),
                    mean_a=float(np.mean(a_vals)),
                    mean_b=float(np.mean(b_vals)),
                    std_L=float(np.std(L_vals)),
                    std_a=float(np.std(a_vals)),
                    std_b=float(np.std(b_vals)),
                    pixel_count=pixel_count,
                )

                cells.append(cell)

        logger.info(
            f"Extracted {len(cells)} cells: " f"{len(ring_boundaries)-1} rings × {self.config.sector_count} sectors"
        )

        if len(cells) == 0:
            raise AngularProfilerError("No cells extracted - check parameters")

        return cells

    def get_cell_by_indices(
        self, cells: List[RingSectorCell], ring_index: int, sector_index: int
    ) -> Optional[RingSectorCell]:
        """
        특정 Ring, Sector 인덱스의 셀 검색

        Args:
            cells: 셀 리스트
            ring_index: Ring 번호
            sector_index: Sector 번호

        Returns:
            해당하는 셀 또는 None
        """
        for cell in cells:
            if cell.ring_index == ring_index and cell.sector_index == sector_index:
                return cell
        return None

    def get_cells_by_ring(self, cells: List[RingSectorCell], ring_index: int) -> List[RingSectorCell]:
        """
        특정 Ring의 모든 셀 반환

        Args:
            cells: 셀 리스트
            ring_index: Ring 번호

        Returns:
            해당 Ring의 셀 리스트 (Sector 순서대로)
        """
        ring_cells = [c for c in cells if c.ring_index == ring_index]
        return sorted(ring_cells, key=lambda c: c.sector_index)

    def get_cells_by_sector(self, cells: List[RingSectorCell], sector_index: int) -> List[RingSectorCell]:
        """
        특정 Sector의 모든 셀 반환 (모든 Ring)

        Args:
            cells: 셀 리스트
            sector_index: Sector 번호

        Returns:
            해당 Sector의 셀 리스트 (Ring 순서대로)
        """
        sector_cells = [c for c in cells if c.sector_index == sector_index]
        return sorted(sector_cells, key=lambda c: c.ring_index)
