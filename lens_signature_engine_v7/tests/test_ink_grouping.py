"""
Unit tests for ink_grouping.py
"""

import numpy as np
import pytest

from core.measure.ink_grouping import (
    _calculate_correlation,
    _find_connected_components,
    _should_merge,
    get_group_statistics,
    group_ink_states,
    validate_groups,
)


class TestCalculateCorrelation:
    """_calculate_correlation 함수 테스트"""

    def test_perfect_correlation(self):
        """완전 상관 (동일 곡선)"""
        curve_a = [0.1, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
        curve_b = [0.1, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]

        corr = _calculate_correlation(curve_a, curve_b)
        assert corr == pytest.approx(1.0, abs=0.01)

    def test_high_correlation(self):
        """높은 상관 (유사한 패턴)"""
        curve_a = [0.1, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
        curve_b = [0.15, 0.35, 0.55, 0.75, 0.85, 0.75, 0.55, 0.35, 0.15, 0.05]

        corr = _calculate_correlation(curve_a, curve_b)
        assert corr > 0.9

    def test_low_correlation(self):
        """낮은 상관 (다른 패턴)"""
        curve_a = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        curve_b = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        corr = _calculate_correlation(curve_a, curve_b)
        assert corr < -0.9  # 반대 패턴 (negative correlation)

    def test_zero_variance(self):
        """분산 0 (상수 곡선)"""
        curve_a = [0.5] * 10
        curve_b = [0.3] * 10

        corr = _calculate_correlation(curve_a, curve_b)
        assert corr == 0.0  # 분산 0이면 상관계수 계산 불가


class TestFindConnectedComponents:
    """_find_connected_components 함수 테스트"""

    def test_no_connections(self):
        """연결 없음 (모두 개별 그룹)"""
        adjacency = np.array([[False, False, False], [False, False, False], [False, False, False]])

        groups = _find_connected_components(adjacency)
        assert groups == [[0], [1], [2]]

    def test_all_connected(self):
        """전체 연결 (단일 그룹)"""
        adjacency = np.array([[False, True, True], [True, False, True], [True, True, False]])

        groups = _find_connected_components(adjacency)
        assert groups == [[0, 1, 2]]

    def test_partial_connections(self):
        """부분 연결 (여러 그룹)"""
        adjacency = np.array(
            [
                [False, True, False, False],
                [True, False, False, False],
                [False, False, False, True],
                [False, False, True, False],
            ]
        )

        groups = _find_connected_components(adjacency)
        assert groups == [[0, 1], [2, 3]]

    def test_chain_connection(self):
        """체인 연결 (0-1-2-3)"""
        adjacency = np.array(
            [
                [False, True, False, False],
                [True, False, True, False],
                [False, True, False, True],
                [False, False, True, False],
            ]
        )

        groups = _find_connected_components(adjacency)
        # 0-1, 1-2, 2-3 연결 → 모두 하나의 그룹
        assert groups == [[0, 1, 2, 3]]


class TestShouldMerge:
    """_should_merge 함수 테스트"""

    def test_merge_gradient_states(self):
        """그라데이션 state (병합 조건 만족)"""
        state_i = {
            "state_id": 0,
            "compactness": 0.75,
            "radial_presence_curve": [0.1, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0],
        }
        state_j = {
            "state_id": 1,
            "compactness": 0.78,  # 차이 0.03 < 0.15
            "radial_presence_curve": [0.15, 0.35, 0.55, 0.75, 0.85, 0.75, 0.55, 0.35, 0.15, 0.05],
        }

        should_merge = _should_merge(
            state_i,
            state_j,
            deltaE=12.0,  # < 15.0
            threshold_deltaE=15.0,
            threshold_compactness=0.15,
            threshold_radial_corr=0.7,
        )

        assert should_merge is True

    def test_reject_large_deltaE(self):
        """색차 크면 병합 거부"""
        state_i = {"compactness": 0.75, "radial_presence_curve": [0.5] * 10}
        state_j = {"compactness": 0.78, "radial_presence_curve": [0.5] * 10}

        should_merge = _should_merge(
            state_i,
            state_j,
            deltaE=25.0,  # > 15.0
            threshold_deltaE=15.0,
            threshold_compactness=0.15,
            threshold_radial_corr=0.7,
        )

        assert should_merge is False

    def test_reject_different_compactness(self):
        """Compactness 차이 크면 병합 거부"""
        state_i = {"compactness": 0.50, "radial_presence_curve": [0.5] * 10}
        state_j = {"compactness": 0.80, "radial_presence_curve": [0.5] * 10}  # 차이 0.30 > 0.15

        should_merge = _should_merge(
            state_i, state_j, deltaE=10.0, threshold_deltaE=15.0, threshold_compactness=0.15, threshold_radial_corr=0.7
        )

        assert should_merge is False

    def test_reject_different_radial_pattern(self):
        """Radial 패턴 다르면 병합 거부"""
        state_i = {"compactness": 0.75, "radial_presence_curve": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]}
        state_j = {"compactness": 0.78, "radial_presence_curve": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}

        should_merge = _should_merge(
            state_i, state_j, deltaE=10.0, threshold_deltaE=15.0, threshold_compactness=0.15, threshold_radial_corr=0.7
        )

        assert should_merge is False  # 반대 패턴 (corr < 0)


class TestGroupInkStates:
    """group_ink_states 함수 테스트"""

    def test_gradient_2_states_merge(self):
        """그라데이션 2개 state → 1개 그룹으로 병합"""
        state_features = [
            {
                "state_id": 0,
                "inkness_score": 0.85,
                "compactness": 0.75,
                "radial_presence_curve": [0.1, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0],
            },
            {
                "state_id": 1,
                "inkness_score": 0.82,
                "compactness": 0.78,
                "radial_presence_curve": [0.15, 0.35, 0.55, 0.75, 0.85, 0.75, 0.55, 0.35, 0.15, 0.05],
            },
        ]

        pairwise_deltaE = np.array([[0.0, 12.0], [12.0, 0.0]])

        groups = group_ink_states(state_features, pairwise_deltaE)

        assert len(groups) == 1
        assert groups[0] == [0, 1]

    def test_different_inks_no_merge(self):
        """서로 다른 잉크 → 병합 안 됨"""
        state_features = [
            {
                "state_id": 0,
                "compactness": 0.75,
                "radial_presence_curve": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            },
            {
                "state_id": 1,
                "compactness": 0.50,  # 차이 0.25 > 0.15
                "radial_presence_curve": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            },
        ]

        pairwise_deltaE = np.array([[0.0, 30.0], [30.0, 0.0]])  # 색차 큼

        groups = group_ink_states(state_features, pairwise_deltaE)

        assert len(groups) == 2
        assert groups == [[0], [1]]

    def test_3_states_partial_merge(self):
        """3개 state → 일부만 병합 (0-1 병합, 2 단독)"""
        state_features = [
            {"state_id": 0, "compactness": 0.75, "radial_presence_curve": [0.5] * 10},
            {"state_id": 1, "compactness": 0.78, "radial_presence_curve": [0.5] * 10},
            {"state_id": 2, "compactness": 0.30, "radial_presence_curve": [0.2] * 10},  # 다른 밀도
        ]

        pairwise_deltaE = np.array([[0.0, 10.0, 35.0], [10.0, 0.0, 35.0], [35.0, 35.0, 0.0]])

        groups = group_ink_states(state_features, pairwise_deltaE)

        assert len(groups) == 2
        assert [0, 1] in groups
        assert [2] in groups

    def test_chain_merge(self):
        """체인 병합 (0-1, 1-2 → 0-1-2 전체 병합)"""
        state_features = [
            {"state_id": 0, "compactness": 0.70, "radial_presence_curve": [0.5] * 10},
            {"state_id": 1, "compactness": 0.75, "radial_presence_curve": [0.5] * 10},
            {"state_id": 2, "compactness": 0.80, "radial_presence_curve": [0.5] * 10},
        ]

        pairwise_deltaE = np.array(
            [
                [0.0, 10.0, 20.0],  # 0-1 병합 가능, 0-2 색차 크지만 체인으로 연결
                [10.0, 0.0, 12.0],  # 1-2 병합 가능
                [20.0, 12.0, 0.0],
            ]
        )

        groups = group_ink_states(state_features, pairwise_deltaE, merge_threshold_deltaE=15.0)

        # 0-1, 1-2 연결 → 전체 하나의 그룹
        assert len(groups) == 1
        assert groups[0] == [0, 1, 2]

    def test_empty_input(self):
        """빈 입력"""
        groups = group_ink_states([], np.zeros((0, 0)))
        assert groups == []

    def test_single_state(self):
        """단일 state"""
        state_features = [{"state_id": 0, "compactness": 0.75, "radial_presence_curve": [0.5] * 10}]

        pairwise_deltaE = np.array([[0.0]])

        groups = group_ink_states(state_features, pairwise_deltaE)

        assert len(groups) == 1
        assert groups[0] == [0]


class TestValidateGroups:
    """validate_groups 함수 테스트"""

    def test_valid_groups(self):
        """정상 그룹"""
        state_features = [{}, {}, {}]
        groups = [[0, 1], [2]]

        valid, msg = validate_groups(groups, state_features)

        assert valid
        assert "OK" in msg

    def test_missing_state(self):
        """일부 state 누락"""
        state_features = [{}, {}, {}]
        groups = [[0, 1]]  # state 2 누락

        valid, msg = validate_groups(groups, state_features)

        assert not valid
        assert "Expected 3" in msg

    def test_duplicate_state(self):
        """중복 state"""
        state_features = [{}, {}]
        groups = [[0, 1], [1]]  # state 1 중복

        valid, msg = validate_groups(groups, state_features)

        assert not valid
        assert "Duplicate" in msg

    def test_out_of_range(self):
        """범위 벗어난 state ID"""
        state_features = [{}, {}]
        groups = [[0, 1, 5]]  # 5는 범위 벗어남

        valid, msg = validate_groups(groups, state_features)

        assert not valid
        assert "out of range" in msg


class TestGetGroupStatistics:
    """get_group_statistics 함수 테스트"""

    def test_no_merge(self):
        """병합 없음"""
        state_features = [{}, {}, {}]
        groups = [[0], [1], [2]]

        stats = get_group_statistics(groups, state_features)

        assert stats["num_groups"] == 3
        assert stats["num_states"] == 3
        assert stats["merge_ratio"] == 0.0
        assert stats["largest_group_size"] == 1

    def test_partial_merge(self):
        """일부 병합"""
        state_features = [{}, {}, {}, {}]
        groups = [[0, 1], [2], [3]]

        stats = get_group_statistics(groups, state_features)

        assert stats["num_groups"] == 3
        assert stats["num_states"] == 4
        assert stats["merge_ratio"] == pytest.approx(0.333, abs=0.01)
        assert stats["largest_group_size"] == 2

    def test_full_merge(self):
        """전체 병합"""
        state_features = [{}, {}, {}, {}]
        groups = [[0, 1, 2, 3]]

        stats = get_group_statistics(groups, state_features)

        assert stats["num_groups"] == 1
        assert stats["num_states"] == 4
        assert stats["merge_ratio"] == 1.0
        assert stats["largest_group_size"] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
