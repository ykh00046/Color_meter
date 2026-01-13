"""
Unit tests for ZoneSegmenter (adaptive + expected_zones fallback).
"""

import numpy as np
import pytest

from src.core.radial_profiler import RadialProfile
from src.core.zone_segmenter import SegmenterConfig, Zone, ZoneSegmenter


def make_profile(a_values):
    n = len(a_values)
    r = np.linspace(0, 1, n)
    L = np.full(n, 50.0)
    b = np.full(n, 20.0)
    std = np.full(n, 0.5)
    pixel_count = np.full(n, 360.0)
    return RadialProfile(r, L, np.array(a_values), b, std, std, std, pixel_count)


def test_config_defaults():
    cfg = SegmenterConfig()
    assert cfg.detection_method == "variable_width"
    assert cfg.min_zone_width == pytest.approx(0.03)
    assert cfg.smoothing_window == 11
    assert cfg.polyorder == 3
    assert cfg.min_gradient == pytest.approx(0.25)
    assert cfg.min_delta_e == pytest.approx(2.0)
    assert cfg.expected_zones is None


def test_expected_zones_uniform_split():
    """평탄 프로파일 + expected_zones 힌트가 있으면 균등 분할한다."""
    profile = make_profile([10.0] * 60)
    cfg = SegmenterConfig(expected_zones=2, min_gradient=10.0, min_delta_e=10.0)
    seg = ZoneSegmenter(cfg)

    zones = seg.segment(profile)

    assert len(zones) == 2
    spans = sum(z.r_start - z.r_end for z in zones)
    assert pytest.approx(spans, rel=1e-3) == 1.0


def test_gradient_detects_step_change():
    """단계형 a* 변화에서 2개 이상 zone을 생성한다."""
    a_values = [5.0] * 30 + [25.0] * 30
    profile = make_profile(a_values)
    seg = ZoneSegmenter(SegmenterConfig(expected_zones=2, min_gradient=0.1))

    zones = seg.segment(profile)

    assert len(zones) >= 2
    for z in zones:
        assert z.r_start > z.r_end


def test_default_fallback_three_zones():
    """힌트/변곡이 없으면 기본 3분할 fallback."""
    profile = make_profile([10.0] * 40)
    seg = ZoneSegmenter(SegmenterConfig(expected_zones=None, min_gradient=10.0, min_delta_e=10.0))

    zones = seg.segment(profile)

    assert len(zones) == 3
    spans = sum(z.r_start - z.r_end for z in zones)
    assert pytest.approx(spans, rel=1e-3) == 1.0


def test_generate_zone_labels():
    seg = ZoneSegmenter()
    assert seg._generate_zone_labels(4) == ["A", "B", "C", "D"]


def test_zone_dataclass_properties():
    zone = Zone(
        name="A",
        r_start=1.0,
        r_end=0.5,
        mean_L=50.0,
        mean_a=10.0,
        mean_b=20.0,
        std_L=1.0,
        std_a=0.5,
        std_b=0.5,
        zone_type="pure",
    )
    assert zone.name == "A"
    assert zone.r_start == pytest.approx(1.0)
    assert zone.r_end == pytest.approx(0.5)
    assert zone.zone_type == "pure"
