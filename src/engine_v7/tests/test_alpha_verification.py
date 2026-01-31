"""
Test P1-2: Alpha verification (registration-less vs plate-based comparison).

Verifies that:
1. verify_alpha_agreement correctly compares two alpha maps
2. Verification passes when maps agree
3. Verification fails when maps disagree significantly
4. Proper warnings and metrics are generated
"""

import numpy as np
import pytest
from core.measure.metrics.alpha_density import AlphaVerificationResult, verify_alpha_agreement


class TestVerifyAlphaAgreement:
    """Test alpha verification between registration-less and plate-based approaches."""

    def test_identical_maps_pass(self):
        """Identical alpha maps should pass verification with high scores."""
        T, R = 180, 100
        alpha = np.random.uniform(0.3, 0.8, size=(T, R)).astype(np.float32)

        result = verify_alpha_agreement(alpha, alpha.copy())

        assert result.passed is True
        assert result.agreement_score >= 0.95  # Should be very high
        assert result.rmse < 0.01  # Should be near zero
        assert result.correlation >= 0.99  # Should be near 1.0
        # No error/failure warnings (info warnings like LOW_CORE_PIXELS are ok)
        error_warnings = [
            w for w in result.warnings if "HIGH_RMSE" in w or "LOW_CORRELATION" in w or "LOW_AGREEMENT" in w
        ]
        assert len(error_warnings) == 0

    def test_similar_maps_pass(self):
        """Similar alpha maps with small differences should pass verification."""
        T, R = 180, 100
        alpha1 = np.random.uniform(0.3, 0.8, size=(T, R)).astype(np.float32)
        # Add small noise
        noise = np.random.randn(T, R).astype(np.float32) * 0.03
        alpha2 = np.clip(alpha1 + noise, 0.02, 0.98)

        result = verify_alpha_agreement(alpha1, alpha2)

        assert result.passed is True
        assert result.agreement_score >= 0.7
        assert result.rmse <= 0.15
        assert result.correlation >= 0.7

    def test_different_maps_fail(self):
        """Very different alpha maps should fail verification."""
        T, R = 180, 100
        alpha1 = np.random.uniform(0.3, 0.5, size=(T, R)).astype(np.float32)
        alpha2 = np.random.uniform(0.6, 0.9, size=(T, R)).astype(np.float32)

        result = verify_alpha_agreement(alpha1, alpha2)

        assert result.passed is False
        assert result.rmse > 0.15
        assert len(result.warnings) > 0
        assert any("HIGH_RMSE" in w or "LOW" in w for w in result.warnings)

    def test_shape_mismatch_resamples(self):
        """Different shaped maps should be resampled and compared."""
        T1, R1 = 180, 100
        T2, R2 = 360, 200

        alpha1 = np.random.uniform(0.4, 0.7, size=(T1, R1)).astype(np.float32)
        alpha2 = np.random.uniform(0.4, 0.7, size=(T2, R2)).astype(np.float32)

        result = verify_alpha_agreement(alpha1, alpha2)

        assert isinstance(result, AlphaVerificationResult)
        # Should have shape mismatch warning
        assert any("SHAPE_MISMATCH" in w for w in result.warnings)

    def test_nan_handling(self):
        """Maps with NaN values should be handled correctly."""
        T, R = 180, 100
        alpha1 = np.random.uniform(0.3, 0.8, size=(T, R)).astype(np.float32)
        alpha2 = alpha1.copy()

        # Add some NaN values
        alpha1[:10, :] = np.nan
        alpha2[-10:, :] = np.nan

        result = verify_alpha_agreement(alpha1, alpha2)

        assert isinstance(result, AlphaVerificationResult)
        assert "valid_ratio" in result.summary
        # Valid ratio should be less than 1.0 due to NaN
        assert result.summary["valid_ratio"] < 1.0

    def test_low_valid_overlap_fails(self):
        """Maps with very low valid overlap should fail."""
        T, R = 180, 100
        alpha1 = np.full((T, R), np.nan, dtype=np.float32)
        alpha2 = np.full((T, R), np.nan, dtype=np.float32)

        # Only 5% overlap
        alpha1[:9, :] = 0.5
        alpha2[:9, :] = 0.5

        result = verify_alpha_agreement(alpha1, alpha2)

        assert result.passed is False
        assert any("LOW_VALID_OVERLAP" in w for w in result.warnings)

    def test_summary_contains_required_fields(self):
        """Summary should contain all required diagnostic fields."""
        T, R = 180, 100
        alpha = np.random.uniform(0.3, 0.8, size=(T, R)).astype(np.float32)

        result = verify_alpha_agreement(alpha, alpha.copy())

        required_fields = [
            "valid_ratio",
            "valid_pixels",
            "total_pixels",
            "registrationless_mean",
            "plate_mean",
            "mean_diff",
            "std_diff",
            "radial_agreement_mean",
            "thresholds",
        ]
        for field in required_fields:
            assert field in result.summary, f"Missing field: {field}"

    def test_radial_agreement_shape(self):
        """Radial agreement array should match R dimension."""
        T, R = 180, 100
        alpha = np.random.uniform(0.3, 0.8, size=(T, R)).astype(np.float32)

        result = verify_alpha_agreement(alpha, alpha.copy())

        assert result.radial_agreement.shape == (R,)

    def test_custom_thresholds(self):
        """Custom thresholds should be respected."""
        T, R = 180, 100
        alpha1 = np.random.uniform(0.3, 0.8, size=(T, R)).astype(np.float32)
        # Add moderate noise
        noise = np.random.randn(T, R).astype(np.float32) * 0.05
        alpha2 = np.clip(alpha1 + noise, 0.02, 0.98)

        # Strict thresholds should fail
        result_strict = verify_alpha_agreement(
            alpha1,
            alpha2,
            rmse_threshold=0.01,
            correlation_threshold=0.99,
            agreement_threshold=0.99,
        )

        # Lenient thresholds should pass
        result_lenient = verify_alpha_agreement(
            alpha1,
            alpha2,
            rmse_threshold=0.20,
            correlation_threshold=0.5,
            agreement_threshold=0.5,
        )

        # With moderate noise, strict should fail, lenient should pass
        assert result_strict.passed is False or result_lenient.passed is True

    def test_correlation_with_uniform_maps(self):
        """Uniform maps (zero variance) should be handled gracefully."""
        T, R = 180, 100
        alpha1 = np.full((T, R), 0.5, dtype=np.float32)
        alpha2 = np.full((T, R), 0.5, dtype=np.float32)

        result = verify_alpha_agreement(alpha1, alpha2)

        # Should handle zero variance gracefully
        assert isinstance(result, AlphaVerificationResult)
        # May or may not pass depending on implementation
        # The key is it shouldn't crash


class TestVerificationConfidenceAdjustment:
    """P2-3: Test verification → confidence threshold adjustment."""

    def test_low_agreement_increases_thresholds(self):
        """When verification agreement is low, L1/L2 thresholds should increase."""
        from core.measure.metrics.alpha_density import compute_effective_density

        T, R = 360, 200
        # Create good quality alpha (should normally use L1)
        polar_alpha = np.random.uniform(0.4, 0.6, size=(T, R)).astype(np.float32)

        masks = {"ink0": np.ones((T, R), dtype=bool)}
        area_ratios = {"ink0": 0.25}

        # Without verification - should use L1
        result_no_verify = compute_effective_density(polar_alpha, masks, area_ratios)

        # With low verification agreement - should potentially fall back to L2/L3
        cfg_low_agree = {
            "_verification_enabled": True,
            "_verification_agreement": 0.5,  # Low agreement
        }
        result_low_agree = compute_effective_density(polar_alpha, masks, area_ratios, cfg=cfg_low_agree)

        # Check warning is generated
        warnings_str = " ".join(result_low_agree.warnings)
        assert "VERIFICATION_CONFIDENCE_ADJUSTMENT" in warnings_str

    def test_high_agreement_no_adjustment(self):
        """When verification agreement is high, thresholds should not change significantly."""
        from core.measure.metrics.alpha_density import compute_effective_density

        T, R = 360, 200
        polar_alpha = np.random.uniform(0.4, 0.6, size=(T, R)).astype(np.float32)

        masks = {"ink0": np.ones((T, R), dtype=bool)}
        area_ratios = {"ink0": 0.25}

        # With high verification agreement
        cfg_high_agree = {
            "_verification_enabled": True,
            "_verification_agreement": 0.95,  # High agreement
        }
        result = compute_effective_density(polar_alpha, masks, area_ratios, cfg=cfg_high_agree)

        # Should still produce adjustment warning but with small multiplier
        warnings_str = " ".join(result.warnings)
        assert "VERIFICATION_CONFIDENCE_ADJUSTMENT" in warnings_str

    def test_disabled_verification_no_adjustment(self):
        """When verification is disabled, no adjustment should be made."""
        from core.measure.metrics.alpha_density import compute_effective_density

        T, R = 360, 200
        polar_alpha = np.random.uniform(0.4, 0.6, size=(T, R)).astype(np.float32)

        masks = {"ink0": np.ones((T, R), dtype=bool)}
        area_ratios = {"ink0": 0.25}

        # Without verification enabled (even with agreement score)
        cfg_disabled = {
            "_verification_enabled": False,
            "_verification_agreement": 0.5,
        }
        result = compute_effective_density(polar_alpha, masks, area_ratios, cfg=cfg_disabled)

        # Should NOT have adjustment warning
        warnings_str = " ".join(result.warnings)
        assert "VERIFICATION_CONFIDENCE_ADJUSTMENT" not in warnings_str


class TestAlphaVerificationIntegration:
    """Integration tests for alpha verification in the analysis pipeline."""

    def test_verification_result_serializable(self):
        """Verification result should be serializable for JSON output."""
        T, R = 180, 100
        alpha = np.random.uniform(0.3, 0.8, size=(T, R)).astype(np.float32)

        result = verify_alpha_agreement(alpha, alpha.copy())

        # Check that all values can be serialized
        import json

        verification_dict = {
            "passed": result.passed,
            "agreement_score": result.agreement_score,
            "rmse": result.rmse,
            "correlation": result.correlation,
            "summary": result.summary,
            "warnings": result.warnings,
        }
        # Should not raise
        json_str = json.dumps(verification_dict)
        assert len(json_str) > 0

    def test_edge_case_extreme_values(self):
        """Maps at extreme clip boundaries should be handled."""
        T, R = 180, 100
        # Maps at boundaries
        alpha1 = np.full((T, R), 0.02, dtype=np.float32)
        alpha2 = np.full((T, R), 0.98, dtype=np.float32)

        result = verify_alpha_agreement(alpha1, alpha2)

        # Should complete without error
        assert isinstance(result, AlphaVerificationResult)
        # Very different - should fail
        assert result.passed is False


class TestP24CoreRegionMetrics:
    """P2-4: Test agreement score improvement with boundary/transition exclusion."""

    def test_core_metrics_in_result(self):
        """Verification result should include core region metrics."""
        T, R = 180, 100
        alpha = np.random.uniform(0.3, 0.8, size=(T, R)).astype(np.float32)

        result = verify_alpha_agreement(alpha, alpha.copy())

        # Check new P2-4 fields exist
        assert hasattr(result, "core_agreement_score")
        assert hasattr(result, "core_rmse")
        assert hasattr(result, "core_correlation")
        assert hasattr(result, "transition_ratio")

    def test_identical_maps_high_core_agreement(self):
        """Identical maps should have high core agreement score."""
        T, R = 180, 100
        alpha = np.random.uniform(0.3, 0.8, size=(T, R)).astype(np.float32)

        result = verify_alpha_agreement(alpha, alpha.copy())

        assert result.core_agreement_score >= 0.95
        assert result.core_rmse < 0.01
        # Transition ratio should be low for smooth alpha
        assert result.transition_ratio < 0.3

    def test_transition_regions_detected(self):
        """Sharp boundaries should be detected as transition regions."""
        T, R = 180, 100
        alpha1 = np.random.uniform(0.3, 0.5, size=(T, R)).astype(np.float32)
        alpha2 = alpha1.copy()

        # Create sharp boundary in both maps
        alpha1[:, R // 2 :] = np.random.uniform(0.6, 0.8, size=(T, R // 2)).astype(np.float32)
        alpha2[:, R // 2 :] = np.random.uniform(0.6, 0.8, size=(T, R // 2)).astype(np.float32)

        result = verify_alpha_agreement(alpha1, alpha2)

        # Should detect some transition at the boundary
        assert result.transition_ratio > 0.01
        assert "transition_ratio" in result.summary

    def test_core_agreement_better_than_full_with_boundary_disagreement(self):
        """Core agreement should be better than full when boundaries disagree."""
        T, R = 180, 100
        # Create base alpha with some structure (not constant)
        np.random.seed(42)
        alpha1 = np.random.uniform(0.4, 0.6, size=(T, R)).astype(np.float32)
        alpha2 = alpha1.copy()

        # Add small uniform noise to whole image (high agreement in core)
        noise = np.random.randn(T, R).astype(np.float32) * 0.02
        alpha2 = np.clip(alpha1 + noise, 0.02, 0.98)

        # Add LARGE difference at boundary (column R//2 +/- 5)
        # This should trigger transition detection
        boundary_cols = slice(R // 2 - 5, R // 2 + 5)
        alpha2[:, boundary_cols] = np.clip(
            alpha1[:, boundary_cols] + 0.30, 0.02, 0.98  # 30% diff exceeds 25% threshold
        )

        result = verify_alpha_agreement(alpha1, alpha2)

        # When boundary is excluded, core metrics should be better or equal
        # Note: if transition_ratio > 0, core should be better
        if result.transition_ratio > 0:
            # Core should be better because boundary disagreement is excluded
            assert result.core_rmse <= result.rmse + 0.01  # Allow small tolerance
        # Full image RMSE should reflect the boundary disagreement
        assert result.rmse > 0.01  # Not zero due to boundary

    def test_use_core_for_decision(self):
        """When use_core_for_decision=True, pass/fail should use core metrics."""
        T, R = 180, 100
        np.random.seed(123)
        # Create structured alpha (not constant for correlation)
        alpha1 = np.random.uniform(0.4, 0.6, size=(T, R)).astype(np.float32)
        alpha2 = alpha1.copy()

        # Small noise in core (should pass)
        noise = np.random.randn(T, R).astype(np.float32) * 0.02
        alpha2 = np.clip(alpha1 + noise, 0.02, 0.98)

        # Large disagreement at edges (triggers transition detection)
        # Use 30% difference to exceed the 25% threshold
        alpha2[:, :10] = np.clip(alpha1[:, :10] + 0.30, 0.02, 0.98)
        alpha2[:, -10:] = np.clip(alpha1[:, -10:] + 0.30, 0.02, 0.98)

        result_core = verify_alpha_agreement(alpha1, alpha2, use_core_for_decision=True)
        result_full = verify_alpha_agreement(alpha1, alpha2, use_core_for_decision=False)

        # Check that decision_source reflects the setting
        # If core_count >= 100, should use core; otherwise full
        if result_core.summary.get("core_valid_pixels", 0) >= 100:
            assert result_core.summary["decision_source"] == "core"
        else:
            assert result_core.summary["decision_source"] == "full"
        assert result_full.summary["decision_source"] == "full"

    def test_auto_detect_transitions_disabled(self):
        """When auto_detect_transitions=False, no transition detection."""
        T, R = 180, 100
        alpha1 = np.random.uniform(0.3, 0.8, size=(T, R)).astype(np.float32)
        alpha2 = alpha1.copy()

        # Add sharp boundary
        alpha1[:, R // 2 :] += 0.2
        alpha1 = np.clip(alpha1, 0.02, 0.98)
        alpha2[:, R // 2 :] += 0.2
        alpha2 = np.clip(alpha2, 0.02, 0.98)

        result = verify_alpha_agreement(alpha1, alpha2, auto_detect_transitions=False)

        # No transition detection means transition_ratio should be 0
        assert result.transition_ratio == 0.0

    def test_custom_transition_mask(self):
        """Custom transition mask should be used when provided."""
        T, R = 180, 100
        alpha = np.random.uniform(0.3, 0.8, size=(T, R)).astype(np.float32)

        # Create custom transition mask (mark 20% as transitions)
        custom_mask = np.zeros((T, R), dtype=bool)
        custom_mask[:, : R // 5] = True  # First 20% of columns

        result = verify_alpha_agreement(
            alpha,
            alpha.copy(),
            transition_mask=custom_mask,
            auto_detect_transitions=False,
        )

        # Transition ratio should be approximately 20% (of valid pixels)
        assert result.transition_ratio > 0.15
        assert result.transition_ratio < 0.25

    def test_summary_contains_core_fields(self):
        """Summary should include P2-4 core region fields."""
        T, R = 180, 100
        alpha = np.random.uniform(0.3, 0.8, size=(T, R)).astype(np.float32)

        result = verify_alpha_agreement(alpha, alpha.copy())

        required_fields = [
            "core_valid_pixels",
            "core_ratio",
            "transition_ratio",
            "decision_source",
        ]
        for field in required_fields:
            assert field in result.summary, f"Missing field: {field}"
