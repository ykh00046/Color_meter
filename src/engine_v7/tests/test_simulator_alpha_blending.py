"""
Test P0-1 & P0-2: Simulator alpha blending integration.

Verifies that:
1. Simulator uses effective_density when alpha is available
2. Model metadata correctly reflects alpha blending usage
3. Coverage structure includes alpha details
"""

import numpy as np
import pytest
from core.simulation.color_simulator import build_simulation_result


class TestSimulatorAlphaBlending:
    """Test alpha blending in color simulator."""

    def test_no_alpha_uses_area_ratio(self):
        """Without alpha fields, model_input should be area_ratio (opaque assumption)."""
        ink_clusters = [
            {
                "id": 0,
                "role": "ink",
                "lab_centroid_cie": [30.0, 10.0, -20.0],
                "area_ratio": 0.25,
                # No effective_density, alpha_used, alpha_fallback_level
            },
        ]

        result = build_simulation_result(ink_clusters)

        # Check model_input equals area_ratio
        sim = result["simulations"][0]
        assert sim["coverage"]["model_input"] == 0.25
        assert sim["coverage"]["use_alpha_blending"] is False
        assert sim["coverage"]["alpha"]["source"] == "default_opaque"

        # Check model metadata
        assert result["model"]["alpha_blending"]["enabled"] is False
        assert "no_alpha_transparency" in result["model"]["limitations"]

    def test_with_alpha_uses_effective_density(self):
        """With alpha fields, model_input should be effective_density."""
        ink_clusters = [
            {
                "id": 0,
                "role": "ink",
                "lab_centroid_cie": [30.0, 10.0, -20.0],
                "area_ratio": 0.25,
                "effective_density": 0.175,  # 0.25 * 0.7
                "alpha_used": 0.7,
                "alpha_fallback_level": "L1_radial",
            },
        ]

        result = build_simulation_result(ink_clusters)

        # Check model_input equals effective_density
        sim = result["simulations"][0]
        assert sim["coverage"]["model_input"] == 0.175
        assert sim["coverage"]["use_alpha_blending"] is True
        assert sim["coverage"]["effective_density"] == 0.175
        assert sim["coverage"]["alpha"]["used"] == 0.7
        assert sim["coverage"]["alpha"]["fallback_level"] == "L1_radial"

        # Check model metadata
        assert result["model"]["alpha_blending"]["enabled"] is True
        assert result["model"]["alpha_blending"]["inks_with_alpha"] == 1
        assert "no_alpha_transparency" not in result["model"]["limitations"]
        assert result["model"]["version"] == "v2.1"

    def test_mixed_alpha_availability(self):
        """Test with some inks having alpha and others not."""
        ink_clusters = [
            {
                "id": 0,
                "role": "ink",
                "lab_centroid_cie": [30.0, 10.0, -20.0],
                "area_ratio": 0.20,
                "effective_density": 0.14,  # 0.20 * 0.7
                "alpha_used": 0.7,
                "alpha_fallback_level": "L1_radial",
            },
            {
                "id": 1,
                "role": "ink",
                "lab_centroid_cie": [50.0, -5.0, 15.0],
                "area_ratio": 0.15,
                # No alpha fields - should use area_ratio
            },
        ]

        result = build_simulation_result(ink_clusters)

        # First ink uses effective_density
        sim0 = result["simulations"][0]
        assert sim0["coverage"]["model_input"] == 0.14
        assert sim0["coverage"]["use_alpha_blending"] is True

        # Second ink uses area_ratio
        sim1 = result["simulations"][1]
        assert sim1["coverage"]["model_input"] == 0.15
        assert sim1["coverage"]["use_alpha_blending"] is False

        # Model metadata shows alpha was used for some
        assert result["model"]["alpha_blending"]["enabled"] is True
        assert result["model"]["alpha_blending"]["inks_with_alpha"] == 1
        assert result["model"]["alpha_blending"]["total_inks"] == 2

    def test_l2_zone_fallback(self):
        """Test L2_zone fallback level."""
        ink_clusters = [
            {
                "id": 0,
                "role": "ink",
                "lab_centroid_cie": [30.0, 10.0, -20.0],
                "area_ratio": 0.25,
                "effective_density": 0.20,
                "alpha_used": 0.8,
                "alpha_fallback_level": "L2_zone",
            },
        ]

        result = build_simulation_result(ink_clusters)

        sim = result["simulations"][0]
        assert sim["coverage"]["model_input"] == 0.20
        assert sim["coverage"]["use_alpha_blending"] is True
        assert sim["coverage"]["alpha"]["fallback_level"] == "L2_zone"

    def test_l3_global_fallback(self):
        """Test L3_global fallback level."""
        ink_clusters = [
            {
                "id": 0,
                "role": "ink",
                "lab_centroid_cie": [30.0, 10.0, -20.0],
                "area_ratio": 0.25,
                "effective_density": 0.225,
                "alpha_used": 0.9,
                "alpha_fallback_level": "L3_global",
            },
        ]

        result = build_simulation_result(ink_clusters)

        sim = result["simulations"][0]
        assert sim["coverage"]["model_input"] == 0.225
        assert sim["coverage"]["use_alpha_blending"] is True
        assert sim["coverage"]["alpha"]["fallback_level"] == "L3_global"

    def test_composite_uses_model_input(self):
        """Verify composite calculation uses model_input (effective_density when available)."""
        ink_clusters = [
            {
                "id": 0,
                "role": "ink",
                "lab_centroid_cie": [30.0, 10.0, -20.0],
                "area_ratio": 0.30,
                "effective_density": 0.21,  # 0.30 * 0.7
                "alpha_used": 0.7,
                "alpha_fallback_level": "L1_radial",
            },
            {
                "id": 1,
                "role": "ink",
                "lab_centroid_cie": [50.0, -5.0, 15.0],
                "area_ratio": 0.20,
                "effective_density": 0.16,  # 0.20 * 0.8
                "alpha_used": 0.8,
                "alpha_fallback_level": "L1_radial",
            },
        ]

        result = build_simulation_result(ink_clusters)

        # Total coverage should be sum of effective_densities, not area_ratios
        composite = result["composite"]["on_white"]
        expected_total = 0.21 + 0.16  # 0.37, not 0.50
        assert abs(composite["total_ink_coverage"] - expected_total) < 0.01

    def test_alpha_blending_affects_perceived_color(self):
        """Verify that using effective_density changes the perceived color."""
        # Same ink with different alpha
        ink_opaque = {
            "id": 0,
            "role": "ink",
            "lab_centroid_cie": [30.0, 10.0, -20.0],
            "area_ratio": 0.50,
            # No alpha - uses area_ratio as model_input
        }
        ink_transparent = {
            "id": 0,
            "role": "ink",
            "lab_centroid_cie": [30.0, 10.0, -20.0],
            "area_ratio": 0.50,
            "effective_density": 0.25,  # 50% transparent
            "alpha_used": 0.5,
            "alpha_fallback_level": "L1_radial",
        }

        result_opaque = build_simulation_result([ink_opaque])
        result_transparent = build_simulation_result([ink_transparent])

        lab_opaque = result_opaque["simulations"][0]["predicted"]["on_white"]["lab"]
        lab_transparent = result_transparent["simulations"][0]["predicted"]["on_white"]["lab"]

        # Transparent version should be lighter (higher L) because less ink coverage
        assert lab_transparent[0] > lab_opaque[0], (
            f"Transparent ink L*={lab_transparent[0]} should be higher than " f"opaque ink L*={lab_opaque[0]}"
        )


class TestCoverageStructureEnhancement:
    """Test enhanced coverage structure from P0-1."""

    def test_coverage_has_all_fields(self):
        """Verify coverage structure has all required fields."""
        ink_clusters = [
            {
                "id": 0,
                "role": "ink",
                "lab_centroid_cie": [30.0, 10.0, -20.0],
                "area_ratio": 0.25,
                "effective_density": 0.175,
                "alpha_used": 0.7,
                "alpha_fallback_level": "L1_radial",
            },
        ]

        result = build_simulation_result(ink_clusters)
        coverage = result["simulations"][0]["coverage"]

        # Required fields
        assert "area_ratio" in coverage
        assert "effective_density" in coverage
        assert "model_input" in coverage
        assert "use_alpha_blending" in coverage

        # Alpha substructure
        assert "alpha" in coverage
        assert "mean" in coverage["alpha"]
        assert "used" in coverage["alpha"]
        assert "source" in coverage["alpha"]
        assert "fallback_level" in coverage["alpha"]

        # Legacy fields
        assert "ratio" in coverage
        assert "percent" in coverage


class TestModelMetadataEnhancement:
    """Test enhanced model metadata from P0-1."""

    def test_model_has_alpha_blending_section(self):
        """Verify model metadata has alpha_blending section."""
        ink_clusters = [
            {
                "id": 0,
                "role": "ink",
                "lab_centroid_cie": [30.0, 10.0, -20.0],
                "area_ratio": 0.25,
            },
        ]

        result = build_simulation_result(ink_clusters)
        model = result["model"]

        assert "alpha_blending" in model
        assert "enabled" in model["alpha_blending"]
        assert "inks_with_alpha" in model["alpha_blending"]
        assert "total_inks" in model["alpha_blending"]

    def test_version_reflects_alpha_usage(self):
        """Verify model version changes based on alpha usage."""
        ink_no_alpha = [{"id": 0, "role": "ink", "lab_centroid_cie": [30.0, 0, 0], "area_ratio": 0.2}]
        ink_with_alpha = [
            {
                "id": 0,
                "role": "ink",
                "lab_centroid_cie": [30.0, 0, 0],
                "area_ratio": 0.2,
                "effective_density": 0.14,
                "alpha_used": 0.7,
                "alpha_fallback_level": "L1_radial",
            }
        ]

        result_no = build_simulation_result(ink_no_alpha)
        result_yes = build_simulation_result(ink_with_alpha)

        assert result_no["model"]["version"] == "v1.1"
        assert result_yes["model"]["version"] == "v2.1"


class TestAlphaQualityGate:
    """Test P0-3: Alpha quality gate functionality."""

    def test_quality_gate_passes_good_alpha(self):
        """When alpha quality is good, alpha blending should be used."""
        from core.measure.metrics.alpha_density import compute_effective_density

        # Create good quality alpha map (no NaN, no excessive clipping)
        T, R = 360, 200
        polar_alpha = np.random.uniform(0.3, 0.8, size=(T, R)).astype(np.float32)

        masks = {"ink0": np.ones((T, R), dtype=bool)}
        area_ratios = {"ink0": 0.25}

        result = compute_effective_density(polar_alpha, masks, area_ratios)

        # Should not have quality gate warning
        warnings_str = " ".join(result.warnings)
        assert "ALPHA_QUALITY_GATE_FAILED" not in warnings_str

        # Should use L1/L2/L3 fallback, not default_opaque
        cluster = result.clusters["ink0"]
        assert cluster.fallback_reason != "alpha_quality_gate"

    def test_quality_gate_fails_high_nan(self):
        """When NaN ratio > 10%, alpha should be skipped."""
        from core.measure.metrics.alpha_density import compute_effective_density

        T, R = 360, 200
        polar_alpha = np.random.uniform(0.3, 0.8, size=(T, R)).astype(np.float32)

        # Set 15% to NaN
        n_nan = int(T * R * 0.15)
        nan_indices = np.random.choice(T * R, n_nan, replace=False)
        polar_alpha.flat[nan_indices] = np.nan

        masks = {"ink0": np.ones((T, R), dtype=bool)}
        area_ratios = {"ink0": 0.25}

        result = compute_effective_density(polar_alpha, masks, area_ratios)

        # Should have quality gate warning
        warnings_str = " ".join(result.warnings)
        assert "ALPHA_QUALITY_GATE_FAILED" in warnings_str

        # Should fall back to opaque (alpha=1.0)
        cluster = result.clusters["ink0"]
        assert cluster.alpha_used == 1.0
        assert cluster.effective_density == 0.25  # Same as area_ratio
        # P2-2: Updated fallback reason format
        assert "alpha_quality_fail" in (cluster.fallback_reason or "")

    def test_quality_gate_fails_high_clip(self):
        """When clip ratio > 30%, alpha should be skipped."""
        from core.measure.metrics.alpha_density import compute_effective_density

        T, R = 360, 200
        polar_alpha = np.random.uniform(0.3, 0.8, size=(T, R)).astype(np.float32)

        # Set 35% to clipped values (at boundaries)
        n_clip = int(T * R * 0.35)
        clip_indices = np.random.choice(T * R, n_clip, replace=False)
        polar_alpha.flat[clip_indices[: n_clip // 2]] = 0.02  # at min clip
        polar_alpha.flat[clip_indices[n_clip // 2 :]] = 0.98  # at max clip

        masks = {"ink0": np.ones((T, R), dtype=bool)}
        area_ratios = {"ink0": 0.25}

        result = compute_effective_density(polar_alpha, masks, area_ratios)

        # Should have quality gate warning
        warnings_str = " ".join(result.warnings)
        assert "ALPHA_QUALITY_GATE_FAILED" in warnings_str

        # Should fall back to opaque
        cluster = result.clusters["ink0"]
        assert cluster.alpha_used == 1.0
        # P2-2: Updated fallback reason format
        assert "alpha_quality_fail" in (cluster.fallback_reason or "")

    def test_quality_gate_configurable_thresholds(self):
        """Quality gate thresholds should be configurable."""
        from core.measure.metrics.alpha_density import compute_effective_density

        T, R = 360, 200
        polar_alpha = np.random.uniform(0.3, 0.8, size=(T, R)).astype(np.float32)

        # Set 8% to NaN (below default 10% threshold)
        n_nan = int(T * R * 0.08)
        nan_indices = np.random.choice(T * R, n_nan, replace=False)
        polar_alpha.flat[nan_indices] = np.nan

        masks = {"ink0": np.ones((T, R), dtype=bool)}
        area_ratios = {"ink0": 0.25}

        # With default threshold (10%), should pass
        result_default = compute_effective_density(polar_alpha, masks, area_ratios)
        warnings_default = " ".join(result_default.warnings)
        assert "ALPHA_QUALITY_GATE_FAILED" not in warnings_default

        # With stricter threshold (5%), should fail
        cfg_strict = {"quality_gate_nan_threshold": 0.05}
        result_strict = compute_effective_density(polar_alpha, masks, area_ratios, cfg=cfg_strict)
        warnings_strict = " ".join(result_strict.warnings)
        assert "ALPHA_QUALITY_GATE_FAILED" in warnings_strict

    def test_quality_fail_nested_config(self):
        """P2-2: Quality fail thresholds can be configured via nested structure."""
        from core.measure.metrics.alpha_density import compute_effective_density

        T, R = 360, 200
        polar_alpha = np.random.uniform(0.3, 0.8, size=(T, R)).astype(np.float32)

        # Set 8% to NaN (below default 10% threshold)
        n_nan = int(T * R * 0.08)
        nan_indices = np.random.choice(T * R, n_nan, replace=False)
        polar_alpha.flat[nan_indices] = np.nan

        masks = {"ink0": np.ones((T, R), dtype=bool)}
        area_ratios = {"ink0": 0.25}

        # With nested config structure
        cfg_nested = {
            "quality_fail": {
                "nan_ratio": 0.05,  # 5% - should fail
                "clip_ratio": 0.30,
                "moire_severity": 0.20,
            }
        }
        result = compute_effective_density(polar_alpha, masks, area_ratios, cfg=cfg_nested)
        warnings_str = " ".join(result.warnings)
        assert "ALPHA_QUALITY_GATE_FAILED" in warnings_str

    def test_quality_fail_moire_severity(self):
        """P2-2: Moire severity check in quality fail."""
        from core.measure.metrics.alpha_density import compute_effective_density

        T, R = 360, 200
        polar_alpha = np.random.uniform(0.3, 0.8, size=(T, R)).astype(np.float32)

        masks = {"ink0": np.ones((T, R), dtype=bool)}
        area_ratios = {"ink0": 0.25}

        # Pass moire severity via _moire_severity config
        cfg_high_moire = {
            "_moire_severity": 0.25,  # Above default 0.20 threshold
            "quality_fail": {
                "moire_severity": 0.20,
            },
        }
        result = compute_effective_density(polar_alpha, masks, area_ratios, cfg=cfg_high_moire)
        warnings_str = " ".join(result.warnings)
        assert "ALPHA_QUALITY_GATE_FAILED" in warnings_str
        assert "moire" in warnings_str.lower()
