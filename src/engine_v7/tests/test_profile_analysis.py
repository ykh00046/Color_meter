"""
Test for Profile Analysis (v7 Port)
"""

import numpy as np
import pytest

from src.engine_v7.core.signature.profile_analysis import (
    analyze_profile,
    compute_gradient,
    detect_peaks,
    detect_print_boundaries,
    smooth_profile,
)


@pytest.fixture
def dummy_profile():
    """Create a dummy profile with a step (edge)"""
    x = np.linspace(0, 1, 100)
    # Step function at 0.5
    y = np.where(x < 0.5, 10.0, 90.0)
    # Add some noise
    np.random.seed(42)
    noise = np.random.normal(0, 1.0, 100)
    return x, y + noise


def test_smooth_profile(dummy_profile):
    _, y = dummy_profile
    smoothed = smooth_profile(y, window_length=11, polyorder=2)
    assert len(smoothed) == len(y)
    # Noise should be reduced
    assert np.std(smoothed[:40]) < np.std(y[:40])


def test_compute_gradient(dummy_profile):
    x, y = dummy_profile
    smoothed = smooth_profile(y, window_length=11)
    grad = compute_gradient(smoothed, x)

    # Gradient should peak around 0.5
    peak_idx = np.argmax(np.abs(grad))
    assert 0.45 < x[peak_idx] < 0.55


def test_detect_peaks(dummy_profile):
    x, y = dummy_profile
    smoothed = smooth_profile(y, window_length=11)
    grad = np.abs(compute_gradient(smoothed, x))

    peaks = detect_peaks(grad, x, height=10.0)
    assert len(peaks) > 0
    best_peak = max(peaks, key=lambda p: p["value"])
    assert 0.45 < best_peak["radius"] < 0.55


def test_detect_print_boundaries():
    # Simulate a printed area from 0.3 to 0.8
    r_norm = np.linspace(0, 1, 100)
    # Chroma is high in print area
    chroma = np.zeros_like(r_norm)
    chroma[(r_norm >= 0.3) & (r_norm <= 0.8)] = 50.0

    # Create dummy a, b channels
    a = chroma.copy()
    b = np.zeros_like(chroma)

    r_inner, r_outer, conf = detect_print_boundaries(r_norm, a, b, method="chroma")

    assert 0.28 < r_inner < 0.32
    assert 0.78 < r_outer < 0.82
    assert conf > 0.5


def test_detect_print_boundaries_empty():
    r_norm = np.linspace(0, 1, 50)
    a = np.zeros_like(r_norm)
    b = np.zeros_like(r_norm)
    r_inner, r_outer, conf = detect_print_boundaries(r_norm, a, b, method="chroma")

    assert np.isclose(r_inner, 0.0)
    assert np.isclose(r_outer, 1.0)
    assert conf == 0.0


def test_analyze_profile(dummy_profile):
    x, y = dummy_profile
    # Use same data for L, a, b for simplicity
    result = analyze_profile(x, y, y, y)

    assert "profile" in result
    assert "derivatives" in result
    assert "boundary_candidates" in result
    assert len(result["boundary_candidates"]) > 0


def test_analyze_profile_flat_profile():
    r = np.linspace(0, 1, 60)
    l_vals = np.full_like(r, 50.0)
    a = np.zeros_like(r)
    b = np.zeros_like(r)
    result = analyze_profile(r, l_vals, a, b, smoothing_window=7, gradient_threshold=0.5)

    assert "profile" in result
    assert "derivatives" in result
    assert isinstance(result["boundary_candidates"], list)


def test_smooth_profile_short_data():
    data = np.array([1.0, 2.0, 3.0])
    smoothed = smooth_profile(data, window_length=7, polyorder=2)

    assert np.allclose(smoothed, data)
