"""
Unit tests for angular_metrics.py
"""

import numpy as np
import pytest
from core.measure.metrics.angular_metrics import (
    _calculate_runs,
    build_angular_features,
    calculate_angular_continuity,
    calculate_angular_coverage,
    calculate_angular_uniformity,
)


class TestCalculateRuns:
    """_calculate_runs 함수 테스트"""

    def test_single_run(self):
        """단일 연속 구간"""
        arr = np.array([True, True, True, True])
        runs = _calculate_runs(arr)
        assert runs == [4]

    def test_multiple_runs(self):
        """여러 연속 구간"""
        arr = np.array([True, True, False, True, True, True, False, True])
        runs = _calculate_runs(arr)
        assert runs == [2, 3, 1]

    def test_all_false(self):
        """모두 False"""
        arr = np.array([False, False, False])
        runs = _calculate_runs(arr)
        assert runs == []

    def test_all_true(self):
        """모두 True"""
        arr = np.array([True] * 10)
        runs = _calculate_runs(arr)
        assert runs == [10]

    def test_alternating(self):
        """교대로"""
        arr = np.array([True, False, True, False, True])
        runs = _calculate_runs(arr)
        assert runs == [1, 1, 1]


class TestCalculateAngularContinuity:
    """calculate_angular_continuity 함수 테스트"""

    def test_full_continuous_cluster(self):
        """완전 연속적인 클러스터"""
        R, T = 100, 360
        state_map = np.zeros((R, T), dtype=np.int32)  # 전부 클러스터 0

        continuity = calculate_angular_continuity(state_map, cluster_id=0, r_bins=10)

        # 전체가 연결되어 있으므로 높은 점수
        assert continuity > 0.9

    def test_scattered_cluster(self):
        """흩어진 클러스터"""
        R, T = 100, 360
        state_map = np.full((R, T), -1, dtype=np.int32)

        # 클러스터 0: 체크보드 패턴 (theta 방향으로 단절)
        state_map[::2, ::2] = 0

        continuity = calculate_angular_continuity(state_map, cluster_id=0, r_bins=10)

        # 단절적이므로 낮은 점수
        assert continuity < 0.5

    def test_empty_cluster(self):
        """빈 클러스터"""
        R, T = 50, 180
        state_map = np.zeros((R, T), dtype=np.int32)  # 전부 0

        continuity = calculate_angular_continuity(state_map, cluster_id=1)  # 클러스터 1 없음

        assert continuity == 0.0

    def test_partial_band_continuous(self):
        """일부 radial band만, 하지만 연속적"""
        R, T = 100, 360
        state_map = np.full((R, T), -1, dtype=np.int32)

        # 클러스터 0: r=30~40 영역, theta 전체
        state_map[30:40, :] = 0

        continuity = calculate_angular_continuity(state_map, cluster_id=0, r_bins=10)

        # 일부 영역이지만 연속적
        assert continuity > 0.8

    def test_partial_band_fragmented(self):
        """일부 radial band, 하지만 단절적"""
        R, T = 100, 360
        state_map = np.full((R, T), -1, dtype=np.int32)

        # 클러스터 0: r=30~40 영역, theta는 일부만 (0, 90, 180, 270도)
        state_map[30:40, 0] = 0
        state_map[30:40, 90] = 0
        state_map[30:40, 180] = 0
        state_map[30:40, 270] = 0

        continuity = calculate_angular_continuity(state_map, cluster_id=0, r_bins=10)

        # 단절적
        assert continuity < 0.3


class TestCalculateAngularUniformity:
    """calculate_angular_uniformity 함수 테스트"""

    def test_perfect_uniform(self):
        """완벽히 균일"""
        R, T = 100, 360
        state_map = np.zeros((R, T), dtype=np.int32)  # 전부 동일

        uniformity = calculate_angular_uniformity(state_map, cluster_id=0)

        # CV=0 → uniformity=1.0
        assert uniformity == 1.0

    def test_concentrated_cluster(self):
        """일부 각도에만 집중"""
        R, T = 100, 360
        state_map = np.full((R, T), -1, dtype=np.int32)

        # 클러스터 0: theta=0~10도만
        state_map[:, :10] = 0

        uniformity = calculate_angular_uniformity(state_map, cluster_id=0)

        # 극도로 불균일
        assert uniformity < 0.3

    def test_empty_cluster(self):
        """빈 클러스터"""
        R, T = 50, 180
        state_map = np.zeros((R, T), dtype=np.int32)

        uniformity = calculate_angular_uniformity(state_map, cluster_id=1)

        assert uniformity == 0.0


class TestCalculateAngularCoverage:
    """calculate_angular_coverage 함수 테스트"""

    def test_full_coverage(self):
        """전체 각도 커버"""
        R, T = 100, 360
        state_map = np.zeros((R, T), dtype=np.int32)

        coverage = calculate_angular_coverage(state_map, cluster_id=0)

        assert coverage == 1.0

    def test_half_coverage(self):
        """절반 각도 커버"""
        R, T = 100, 360
        state_map = np.full((R, T), -1, dtype=np.int32)

        # 클러스터 0: theta 0~180도만
        state_map[:, :180] = 0

        coverage = calculate_angular_coverage(state_map, cluster_id=0)

        assert coverage == 0.5

    def test_sparse_coverage(self):
        """드문드문 커버"""
        R, T = 100, 360
        state_map = np.full((R, T), -1, dtype=np.int32)

        # 클러스터 0: 10도 간격으로만
        state_map[:, ::10] = 0

        coverage = calculate_angular_coverage(state_map, cluster_id=0)

        assert coverage == 0.1  # 36 / 360

    def test_empty_cluster(self):
        """빈 클러스터"""
        R, T = 50, 180
        state_map = np.zeros((R, T), dtype=np.int32)

        coverage = calculate_angular_coverage(state_map, cluster_id=1)

        assert coverage == 0.0


class TestBuildAngularFeatures:
    """build_angular_features 함수 테스트"""

    def test_ink_like_pattern(self):
        """잉크 같은 패턴 (연속적, 균일, 전체 커버)"""
        R, T = 100, 360
        state_map = np.zeros((R, T), dtype=np.int32)

        features = build_angular_features(state_map, cluster_id=0, r_bins=10)

        assert "angular_continuity" in features
        assert "angular_uniformity" in features
        assert "angular_coverage" in features
        assert "angular_score" in features

        # 모두 높아야 함
        assert features["angular_continuity"] > 0.9
        assert features["angular_uniformity"] == 1.0
        assert features["angular_coverage"] == 1.0
        assert features["angular_score"] > 0.9

    def test_noise_like_pattern(self):
        """노이즈 같은 패턴 (단절적, 불균일, 일부만)"""
        R, T = 100, 360
        state_map = np.full((R, T), -1, dtype=np.int32)

        # 클러스터 0: 랜덤 위치에만
        np.random.seed(42)
        random_positions = np.random.choice(R * T, size=1000, replace=False)
        state_map.flat[random_positions] = 0

        features = build_angular_features(state_map, cluster_id=0, r_bins=10)

        # 모두 낮아야 함
        assert features["angular_continuity"] < 0.6
        assert features["angular_uniformity"] < 0.8
        assert features["angular_score"] < 0.7

    def test_radial_band_pattern(self):
        """Radial band 패턴 (일부 반경만, 하지만 연속적)"""
        R, T = 100, 360
        state_map = np.full((R, T), -1, dtype=np.int32)

        # 클러스터 0: r=40~60 영역, theta 전체
        state_map[40:60, :] = 0

        features = build_angular_features(state_map, cluster_id=0, r_bins=10)

        # continuity와 uniformity, coverage는 높아야 함
        assert features["angular_continuity"] > 0.7
        assert features["angular_uniformity"] == 1.0
        assert features["angular_coverage"] == 1.0
        assert features["angular_score"] > 0.8

    def test_empty_cluster(self):
        """빈 클러스터"""
        R, T = 50, 180
        state_map = np.zeros((R, T), dtype=np.int32)

        features = build_angular_features(state_map, cluster_id=1, r_bins=10)

        assert features["angular_continuity"] == 0.0
        assert features["angular_uniformity"] == 0.0
        assert features["angular_coverage"] == 0.0
        assert features["angular_score"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
