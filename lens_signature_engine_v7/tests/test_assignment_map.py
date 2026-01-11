"""
Unit tests for assignment_map.py
"""

import numpy as np
import pytest

from core.measure.assignment_map import build_assignment_map, get_cluster_spatial_stats, validate_assignment_map


class TestBuildAssignmentMap:
    """build_assignment_map 함수 테스트"""

    def test_basic_assignment(self):
        """기본 할당 테스트"""
        # 100x360 ROI, 2개 클러스터
        R, T = 100, 360
        roi_mask = np.ones((R, T), dtype=bool)

        # Lab 맵 생성 (상/하 절반을 다른 색으로)
        lab_map = np.zeros((R, T, 3), dtype=np.float32)
        lab_map[:50, :, :] = [30, 120, 130]  # 상반부
        lab_map[50:, :, :] = [70, 140, 140]  # 하반부

        # 클러스터 중심
        centers = np.array([[30, 120, 130], [70, 140, 140]], dtype=np.float32)

        state_map = build_assignment_map(roi_mask, lab_map, centers)

        # 검증
        assert state_map.shape == (R, T)
        assert state_map[:50, :].max() <= 1  # 상반부는 클러스터 0 또는 1
        assert state_map[50:, :].max() <= 1  # 하반부도 0 또는 1

        # 상반부는 주로 클러스터 0
        assert np.mean(state_map[:50, :] == 0) > 0.9
        # 하반부는 주로 클러스터 1
        assert np.mean(state_map[50:, :] == 1) > 0.9

    def test_roi_outside_is_minus_one(self):
        """ROI 외부는 -1로 할당"""
        R, T = 50, 180

        # ROI: 중간 영역만
        roi_mask = np.zeros((R, T), dtype=bool)
        roi_mask[10:40, :] = True

        lab_map = np.random.rand(R, T, 3).astype(np.float32) * 100
        centers = np.array([[50, 128, 128]], dtype=np.float32)

        state_map = build_assignment_map(roi_mask, lab_map, centers)

        # ROI 외부는 -1
        assert np.all(state_map[:10, :] == -1)
        assert np.all(state_map[40:, :] == -1)

        # ROI 내부는 0 (클러스터 1개)
        assert np.all(state_map[10:40, :] == 0)

    def test_empty_roi(self):
        """빈 ROI 처리"""
        R, T = 50, 180
        roi_mask = np.zeros((R, T), dtype=bool)  # 전부 False
        lab_map = np.random.rand(R, T, 3).astype(np.float32)
        centers = np.array([[50, 128, 128]], dtype=np.float32)

        state_map = build_assignment_map(roi_mask, lab_map, centers)

        # 전부 -1
        assert np.all(state_map == -1)

    def test_zero_clusters(self):
        """클러스터 0개"""
        R, T = 50, 180
        roi_mask = np.ones((R, T), dtype=bool)
        lab_map = np.random.rand(R, T, 3).astype(np.float32)
        centers = np.zeros((0, 3), dtype=np.float32)  # 빈 배열

        state_map = build_assignment_map(roi_mask, lab_map, centers)

        # 전부 -1
        assert np.all(state_map == -1)

    def test_invalid_lab_shape(self):
        """잘못된 lab_map 차원"""
        roi_mask = np.ones((50, 180), dtype=bool)
        lab_map = np.random.rand(50, 180).astype(np.float32)  # 2D (잘못됨)
        centers = np.array([[50, 128, 128]], dtype=np.float32)

        with pytest.raises(ValueError, match="lab_map must be"):
            build_assignment_map(roi_mask, lab_map, centers)

    def test_invalid_centers_shape(self):
        """잘못된 centers 차원"""
        roi_mask = np.ones((50, 180), dtype=bool)
        lab_map = np.random.rand(50, 180, 3).astype(np.float32)
        centers = np.array([50, 128, 128], dtype=np.float32)  # 1D (잘못됨)

        with pytest.raises(ValueError, match="centers must be"):
            build_assignment_map(roi_mask, lab_map, centers)


class TestValidateAssignmentMap:
    """validate_assignment_map 함수 테스트"""

    def test_valid_map(self):
        """정상 맵 검증"""
        R, T = 100, 360
        roi_mask = np.ones((R, T), dtype=bool)

        # 2개 클러스터로 균등 분할
        state_map = np.full((R, T), -1, dtype=np.int32)
        state_map[roi_mask] = np.random.randint(0, 2, size=np.sum(roi_mask))

        valid, msg = validate_assignment_map(state_map, roi_mask, expected_k=2)

        assert valid
        assert "OK" in msg

    def test_roi_outside_not_minus_one(self):
        """ROI 외부가 -1이 아님"""
        R, T = 50, 180
        roi_mask = np.zeros((R, T), dtype=bool)
        roi_mask[10:40, :] = True

        state_map = np.zeros((R, T), dtype=np.int32)  # 전부 0 (잘못됨)

        valid, msg = validate_assignment_map(state_map, roi_mask, expected_k=1)

        assert not valid
        assert "ROI 외부" in msg

    def test_missing_cluster(self):
        """일부 클러스터만 할당됨"""
        R, T = 100, 360
        roi_mask = np.ones((R, T), dtype=bool)

        # 클러스터 3개 기대, but 0과 1만 할당
        state_map = np.full((R, T), -1, dtype=np.int32)
        state_map[roi_mask] = np.random.randint(0, 2, size=np.sum(roi_mask))

        valid, msg = validate_assignment_map(state_map, roi_mask, expected_k=3)

        assert not valid
        assert "2개만 할당" in msg

    def test_out_of_range(self):
        """범위 벗어난 클러스터 ID"""
        R, T = 50, 180
        roi_mask = np.ones((R, T), dtype=bool)

        state_map = np.full((R, T), -1, dtype=np.int32)
        state_map[roi_mask] = 5  # expected_k=2인데 5 할당 (잘못됨)

        valid, msg = validate_assignment_map(state_map, roi_mask, expected_k=2)

        assert not valid
        assert "범위 벗어난" in msg

    def test_cluster_imbalance_warning(self):
        """클러스터 불균형 경고"""
        R, T = 100, 360
        roi_mask = np.ones((R, T), dtype=bool)

        # 클러스터 0: 99.5%, 클러스터 1: 0.5% (극단적 불균형)
        state_map = np.full((R, T), -1, dtype=np.int32)
        roi_pixels = np.sum(roi_mask)
        assignments = np.zeros(roi_pixels, dtype=np.int32)
        assignments[: int(roi_pixels * 0.005)] = 1  # 0.5%만 클러스터 1
        state_map[roi_mask] = assignments

        valid, msg = validate_assignment_map(state_map, roi_mask, expected_k=2)

        assert valid  # 여전히 valid
        assert "불균형" in msg  # 하지만 경고


class TestGetClusterSpatialStats:
    """get_cluster_spatial_stats 함수 테스트"""

    def test_full_coverage_cluster(self):
        """전체 영역 커버하는 클러스터"""
        R, T = 100, 360
        state_map = np.zeros((R, T), dtype=np.int32)  # 전부 클러스터 0

        stats = get_cluster_spatial_stats(state_map, cluster_id=0)

        assert stats["pixel_count"] == R * T
        assert stats["r_min"] == 0.0
        assert stats["r_max"] >= 0.99  # 거의 1.0
        assert stats["theta_coverage"] == 1.0  # 전체 각도
        assert stats["fragmentation"] < 0.1  # 연속적

    def test_empty_cluster(self):
        """빈 클러스터"""
        R, T = 50, 180
        state_map = np.zeros((R, T), dtype=np.int32)  # 전부 0

        stats = get_cluster_spatial_stats(state_map, cluster_id=1)  # 클러스터 1 없음

        assert stats["pixel_count"] == 0
        assert stats["fragmentation"] == 1.0

    def test_radial_band_cluster(self):
        """특정 반경 영역만"""
        R, T = 100, 360
        state_map = np.full((R, T), -1, dtype=np.int32)

        # 클러스터 0: r=30~40 영역만
        state_map[30:40, :] = 0

        stats = get_cluster_spatial_stats(state_map, cluster_id=0)

        assert stats["r_min"] >= 0.29
        assert stats["r_max"] <= 0.41
        assert 0.34 <= stats["r_mean"] <= 0.36  # 중간값
        assert stats["theta_coverage"] == 1.0  # 전 각도

    def test_scattered_cluster(self):
        """흩어진 클러스터"""
        R, T = 100, 360
        state_map = np.full((R, T), -1, dtype=np.int32)

        # 클러스터 0: 체크보드 패턴 (fragmentation 높음)
        state_map[::2, ::2] = 0

        stats = get_cluster_spatial_stats(state_map, cluster_id=0)

        assert stats["pixel_count"] > 0
        assert stats["fragmentation"] > 0.4  # 단절적


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
