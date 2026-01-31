import io
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.engine_v7.core.plate import plate_engine as pe
from src.web.app import app

client = TestClient(app)


def _create_dummy_image_bytes(value: int = 128) -> io.BytesIO:
    img = np.full((100, 100, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", img)
    assert ok
    return io.BytesIO(encoded.tobytes())


def test_compute_alpha_map_lite_uniform(white_black_pair):
    white, black = white_black_pair
    alpha_map, _ = pe._compute_alpha_map_lite(white, black, blur_ksize=5, backlight=255.0)
    expected = 1.0 - ((200.0 - 50.0) / 255.0)
    assert np.isclose(alpha_map.mean(), expected, atol=0.01)
    assert alpha_map.std() < 0.01


def test_compute_alpha_map_lite_gradient(gradient_pair):
    white, black = gradient_pair
    alpha_map, _ = pe._compute_alpha_map_lite(white, black, blur_ksize=5, backlight=255.0)
    assert alpha_map.min() >= 0.02
    assert alpha_map.max() <= 0.98


def test_compute_alpha_map_lite_blur_effect():
    rng = np.random.default_rng(0)
    white = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
    black = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
    alpha_no_blur, _ = pe._compute_alpha_map_lite(white, black, blur_ksize=1, backlight=255.0)
    alpha_blur, _ = pe._compute_alpha_map_lite(white, black, blur_ksize=7, backlight=255.0)
    assert alpha_blur.std() < alpha_no_blur.std()


def test_resolve_paper_color_static():
    lite_cfg = {"paper_color": {"lab": [95.0, 0.0, 0.0], "source": "static"}}
    paper_lab, meta, warnings = pe._resolve_paper_color(lite_cfg, None, None)
    assert np.allclose(paper_lab, [95.0, 0.0, 0.0])
    assert meta["source"] == "static"
    assert warnings == []


def test_resolve_paper_color_auto_valid():
    lite_cfg = {"paper_color": {"lab": [95.0, 0.0, 0.0], "source": "auto"}}
    lab = np.array([[[10.0, 1.0, 2.0], [20.0, 2.0, 3.0]], [[30.0, 3.0, 4.0], [40.0, 4.0, 5.0]]], dtype=np.float32)
    clear_mask = np.array([[True, False], [True, False]])
    paper_lab, meta, warnings = pe._resolve_paper_color(lite_cfg, clear_mask, lab)
    assert np.allclose(paper_lab, [20.0, 2.0, 3.0])
    assert meta["source"] == "auto"
    assert warnings == []


def test_resolve_paper_color_auto_fallback():
    lite_cfg = {"paper_color": {"lab": [95.0, 0.0, 0.0], "source": "auto"}}
    paper_lab, meta, warnings = pe._resolve_paper_color(lite_cfg, None, None)
    assert np.allclose(paper_lab, [95.0, 0.0, 0.0])
    assert meta["source"] == "static"
    assert "paper_color_auto_fallback_static" in warnings


def test_resolve_paper_color_calibration():
    lite_cfg = {"paper_color": {"lab": [92.0, 1.0, -1.0], "source": "calibration"}}
    paper_lab, meta, warnings = pe._resolve_paper_color(lite_cfg, None, None)
    assert np.allclose(paper_lab, [92.0, 1.0, -1.0])
    assert meta["source"] == "calibration"
    assert warnings == []


def _run_lite_with_patches(monkeypatch, obs_lab, alpha, paper_lab, alpha_threshold=0.1):
    lab = np.tile(obs_lab, (2, 2, 1)).astype(np.float32)
    alpha_map = np.full((2, 2), alpha, dtype=np.float32)
    dummy_img = np.zeros((2, 2, 3), dtype=np.uint8)
    dummy_geom = SimpleNamespace(cx=1.0, cy=1.0, r=1.0)
    reg = {"method": "test"}

    monkeypatch.setattr(
        pe,
        "_prepare_pair",
        lambda white_bgr, black_bgr, cfg, geom_hint=None: (dummy_img, dummy_img, dummy_geom, reg),
    )
    monkeypatch.setattr(
        pe,
        "_compute_alpha_map_lite",
        lambda w, b, blur_ksize, backlight, alpha_clip: (alpha_map, {"method": "test"}),
    )
    monkeypatch.setattr(
        pe,
        "_make_plate_masks",
        lambda alpha, geom, cfg, l_map=None: {
            "ring": np.ones((2, 2), dtype=bool),
            "dot": np.zeros((2, 2), dtype=bool),
            "clear": np.zeros((2, 2), dtype=bool),
            "valid": np.ones((2, 2), dtype=bool),
        },
    )
    monkeypatch.setattr(pe, "_split_core_transition", lambda mask, geom, cfg: (mask, np.zeros_like(mask)))
    monkeypatch.setattr(pe, "to_cie_lab", lambda img: lab)

    lite_cfg = {
        "alpha_threshold": alpha_threshold,
        "paper_color": {"lab": paper_lab.tolist(), "source": "static"},
        "blur_ksize": 5,
        "backlight": 255.0,
    }
    return pe.analyze_plate_lite_pair(dummy_img, dummy_img, lite_cfg, plate_cfg={})


def test_ink_restoration_formula(monkeypatch, synthetic_ink_data):
    out = _run_lite_with_patches(
        monkeypatch,
        synthetic_ink_data["obs_lab"],
        synthetic_ink_data["alpha"],
        synthetic_ink_data["paper_lab"],
    )
    ink_lab = np.array(out["zones"]["ring_core"]["ink_lab"])
    assert np.allclose(ink_lab, synthetic_ink_data["ink_lab"], atol=1.0)


def test_safety_clamp_low_alpha(monkeypatch, synthetic_ink_data):
    out = _run_lite_with_patches(
        monkeypatch,
        synthetic_ink_data["obs_lab"],
        0.05,
        synthetic_ink_data["paper_lab"],
    )
    zone = out["zones"]["ring_core"]
    assert np.allclose(zone["ink_lab"], synthetic_ink_data["obs_lab"].tolist(), atol=1e-6)
    assert "ring_core_alpha_too_low_using_observed" in out["warnings"]


def test_boundary_alpha_zero(monkeypatch, synthetic_ink_data):
    out = _run_lite_with_patches(
        monkeypatch,
        synthetic_ink_data["obs_lab"],
        0.02,
        synthetic_ink_data["paper_lab"],
        alpha_threshold=0.0,
    )
    ink_lab = np.array(out["zones"]["ring_core"]["ink_lab"])
    assert np.isfinite(ink_lab).all()


def test_boundary_alpha_one(monkeypatch, synthetic_ink_data):
    out = _run_lite_with_patches(
        monkeypatch,
        synthetic_ink_data["obs_lab"],
        0.98,
        synthetic_ink_data["paper_lab"],
        alpha_threshold=0.0,
    )
    ink_lab = np.array(out["zones"]["ring_core"]["ink_lab"])
    assert np.isfinite(ink_lab).all()


@pytest.mark.parametrize("run", range(5))
def test_alpha_stability(white_black_pair, run):
    white, black = white_black_pair
    alpha1, _ = pe._compute_alpha_map_lite(white, black, blur_ksize=5, backlight=255.0)
    alpha2, _ = pe._compute_alpha_map_lite(white, black, blur_ksize=5, backlight=255.0)
    assert np.allclose(alpha1, alpha2, atol=1e-6)


# --- Plan 5: Error handling and calibration tests ---


def test_white_black_swap_detection():
    """Detect when white/black images are swapped."""
    # Normal: white=200, black=50
    white = np.full((100, 100, 3), 200, dtype=np.uint8)
    black = np.full((100, 100, 3), 50, dtype=np.uint8)
    _, meta_normal = pe._compute_alpha_map_lite(white, black, blur_ksize=5, backlight=255.0)
    assert "possible_white_black_swap" not in meta_normal.get("warnings", [])

    # Swapped: white=50, black=200
    _, meta_swapped = pe._compute_alpha_map_lite(black, white, blur_ksize=5, backlight=255.0)
    assert "possible_white_black_swap" in meta_swapped.get("warnings", [])


def test_alpha_outlier_warning_low():
    """Warn when alpha mean is too low (< 0.1)."""
    # Both images nearly identical -> alpha near 1.0 (fully opaque)
    white = np.full((100, 100, 3), 100, dtype=np.uint8)
    black = np.full((100, 100, 3), 95, dtype=np.uint8)
    _, meta = pe._compute_alpha_map_lite(white, black, blur_ksize=5, backlight=255.0)
    # diff = 5, alpha = 1 - 5/255 ≈ 0.98 -> not low
    assert "alpha_mean_too_low" not in meta.get("warnings", [])

    # Create scenario where alpha is very low (large diff)
    white_bright = np.full((100, 100, 3), 250, dtype=np.uint8)
    black_dark = np.full((100, 100, 3), 10, dtype=np.uint8)
    _, meta_low = pe._compute_alpha_map_lite(white_bright, black_dark, blur_ksize=5, backlight=255.0)
    # diff = 240, alpha = 1 - 240/255 ≈ 0.06 -> low warning
    assert "alpha_mean_too_low" in meta_low.get("warnings", [])


def test_alpha_outlier_warning_high():
    """Warn when alpha mean is too high (> 0.9)."""
    # Nearly identical images -> alpha very high
    white = np.full((100, 100, 3), 100, dtype=np.uint8)
    black = np.full((100, 100, 3), 98, dtype=np.uint8)
    _, meta = pe._compute_alpha_map_lite(white, black, blur_ksize=5, backlight=255.0)
    # diff = 2, alpha = 1 - 2/255 ≈ 0.99 -> high warning
    assert "alpha_mean_too_high" in meta.get("warnings", [])


def test_paper_color_auto_clear_contamination():
    """Auto mode with partially valid clear mask."""
    lite_cfg = {"paper_color": {"lab": [95.0, 0.0, 0.0], "source": "auto"}}
    # Lab with contamination in some pixels
    lab = np.array(
        [
            [[90.0, 1.0, 1.0], [50.0, 10.0, 10.0]],  # first row: clean, contaminated
            [[92.0, -1.0, 0.0], [48.0, 12.0, 8.0]],  # second row: clean, contaminated
        ],
        dtype=np.float32,
    )
    # Only select clean pixels
    clear_mask = np.array([[True, False], [True, False]])
    paper_lab, meta, warnings = pe._resolve_paper_color(lite_cfg, clear_mask, lab)
    # Should average clean pixels: (90+92)/2=91, (1-1)/2=0, (1+0)/2=0.5
    assert np.isclose(paper_lab[0], 91.0, atol=0.1)
    assert meta["source"] == "auto"
    assert warnings == []


def test_api_plate_lite_response_structure(monkeypatch):
    def _mock_analyze_single_sample(*args, **kwargs):
        return {
            "plate": {"schema_version": "plate_v1.2"},
            "plate_lite": {"schema_version": "plate_lite_v1.0"},
        }

    monkeypatch.setattr("src.web.routers.v7._load_cfg", lambda sku=None: {"plate_lite": {"enabled": True}})
    monkeypatch.setattr(
        "src.engine_v7.core.pipeline.single_analyzer.analyze_single_sample",
        _mock_analyze_single_sample,
    )
    monkeypatch.setattr("src.web.routers.v7._generate_single_analysis_artifacts", lambda *args, **kwargs: {})
    monkeypatch.setattr("src.web.routers.v7._generate_plate_pair_artifacts", lambda *args, **kwargs: {})

    white = _create_dummy_image_bytes(200)
    black = _create_dummy_image_bytes(50)
    response = client.post(
        "/api/v7/analyze_single",
        headers={"X-User-Role": "operator"},
        files=[
            ("files", ("white.jpg", white, "image/jpeg")),
            ("black_files", ("black.jpg", black, "image/jpeg")),
        ],
    )
    assert response.status_code == 200
    data = response.json()
    analysis = data["results"][0]["analysis"]
    assert analysis["plate_lite"]["schema_version"] == "plate_lite_v1.0"


def test_api_plate_lite_disabled(monkeypatch):
    def _mock_analyze_single_sample(*args, **kwargs):
        return {"plate": {"schema_version": "plate_v1.2"}}

    monkeypatch.setattr("src.web.routers.v7._load_cfg", lambda sku=None: {"plate_lite": {"enabled": False}})
    monkeypatch.setattr(
        "src.engine_v7.core.pipeline.single_analyzer.analyze_single_sample",
        _mock_analyze_single_sample,
    )
    monkeypatch.setattr("src.web.routers.v7._generate_single_analysis_artifacts", lambda *args, **kwargs: {})
    monkeypatch.setattr("src.web.routers.v7._generate_plate_pair_artifacts", lambda *args, **kwargs: {})

    white = _create_dummy_image_bytes(200)
    black = _create_dummy_image_bytes(50)
    response = client.post(
        "/api/v7/analyze_single",
        headers={"X-User-Role": "operator"},
        files=[
            ("files", ("white.jpg", white, "image/jpeg")),
            ("black_files", ("black.jpg", black, "image/jpeg")),
        ],
    )
    assert response.status_code == 200
    analysis = response.json()["results"][0]["analysis"]
    assert "plate_lite" not in analysis


def test_plate_and_plate_lite_parallel(monkeypatch):
    def _mock_analyze_single_sample(*args, **kwargs):
        return {
            "plate": {"schema_version": "plate_v1.2"},
            "plate_lite": {"schema_version": "plate_lite_v1.0"},
        }

    monkeypatch.setattr(
        "src.web.routers.v7._load_cfg",
        lambda sku=None: {"plate_lite": {"enabled": True, "override_plate": False}},
    )
    monkeypatch.setattr(
        "src.engine_v7.core.pipeline.single_analyzer.analyze_single_sample",
        _mock_analyze_single_sample,
    )
    monkeypatch.setattr("src.web.routers.v7._generate_single_analysis_artifacts", lambda *args, **kwargs: {})
    monkeypatch.setattr("src.web.routers.v7._generate_plate_pair_artifacts", lambda *args, **kwargs: {})

    white = _create_dummy_image_bytes(200)
    black = _create_dummy_image_bytes(50)
    response = client.post(
        "/api/v7/analyze_single",
        headers={"X-User-Role": "operator"},
        files=[
            ("files", ("white.jpg", white, "image/jpeg")),
            ("black_files", ("black.jpg", black, "image/jpeg")),
        ],
    )
    assert response.status_code == 200
    analysis = response.json()["results"][0]["analysis"]
    assert "plate" in analysis
    assert "plate_lite" in analysis


# --- Integration tests for plate_lite → color_comparison ---


def test_plate_lite_to_color_comparison_mapping():
    """Test that plate_lite zones map correctly to color_comparison."""
    from src.engine_v7.core.simulation.color_simulator import build_simulation_result

    ink_clusters = [
        {"id": 0, "role": "ink", "centroid_lab": [50.0, 10.0, 20.0], "mean_hex": "#654321", "area_ratio": 0.3},
        {"id": 1, "role": "ink", "centroid_lab": [70.0, 5.0, 10.0], "mean_hex": "#987654", "area_ratio": 0.2},
    ]
    plate_lite_info = {
        "schema_version": "plate_lite_v1.0",
        "zones": {
            "ring_core": {"alpha_mean": 0.8, "obs_lab": [45.0, 8.0, 18.0], "ink_hex": "#553311"},
            "dot_core": {"alpha_mean": 0.6, "obs_lab": [75.0, 3.0, 8.0], "ink_hex": "#aa8866"},
        },
    }

    result = build_simulation_result(
        ink_clusters=ink_clusters,
        plate_info=None,
        plate_lite_info=plate_lite_info,
    )

    color_comparison = result.get("color_comparison", [])
    assert len(color_comparison) >= 2

    # Ink 0 should have plate_measurement from ring_core
    assert color_comparison[0]["plate_measurement"] is not None
    assert color_comparison[0]["plate_measurement"]["matched_plate_ink"] == "ring_core"
    assert color_comparison[0]["plate_measurement"]["lab"] == [45.0, 8.0, 18.0]

    # Ink 1 should have plate_measurement from dot_core
    assert color_comparison[1]["plate_measurement"] is not None
    assert color_comparison[1]["plate_measurement"]["matched_plate_ink"] == "dot_core"

    # Model metadata should indicate plate_lite source
    assert result["model"]["observed_source"] == "plate_lite"
    assert result["model"]["observed_on_white_available"] is True


def test_plate_lite_fallback_when_plate_missing():
    """Test that plate_lite data is used when plate_info is None."""
    from src.engine_v7.core.simulation.color_simulator import build_simulation_result

    ink_clusters = [
        {"id": 0, "role": "ink", "centroid_lab": [50.0, 10.0, 20.0], "mean_hex": "#654321", "area_ratio": 0.3},
    ]
    plate_lite_info = {
        "zones": {
            "ring_core": {"alpha_mean": 0.8, "obs_lab": [45.0, 8.0, 18.0], "ink_hex": "#553311"},
            "dot_core": {"empty": True},
        },
    }

    # With plate_lite but no plate
    result = build_simulation_result(
        ink_clusters=ink_clusters,
        plate_info=None,
        plate_lite_info=plate_lite_info,
    )
    assert result["color_comparison"][0]["plate_measurement"] is not None
    assert result["model"]["observed_source"] == "plate_lite"

    # Without both (should have no plate_measurement)
    result_empty = build_simulation_result(
        ink_clusters=ink_clusters,
        plate_info=None,
        plate_lite_info=None,
    )
    assert result_empty["color_comparison"][0]["plate_measurement"] is None
    assert result_empty["model"]["observed_source"] is None


def test_plate_lite_hex_from_obs_lab():
    """Test that plate_measurement hex is computed from obs_lab, not ink_hex."""
    from src.engine_v7.core.simulation.color_simulator import _lab_to_hex, build_simulation_result

    ink_clusters = [
        {"id": 0, "role": "ink", "centroid_lab": [50.0, 10.0, 20.0], "mean_hex": "#654321", "area_ratio": 0.3},
    ]
    obs_lab = [45.0, 8.0, 18.0]
    plate_lite_info = {
        "zones": {
            "ring_core": {
                "alpha_mean": 0.8,
                "obs_lab": obs_lab,
                "ink_hex": "#DIFFERENT",  # This should NOT be used
            },
        },
    }

    result = build_simulation_result(
        ink_clusters=ink_clusters,
        plate_info=None,
        plate_lite_info=plate_lite_info,
    )

    pm = result["color_comparison"][0]["plate_measurement"]
    expected_hex = _lab_to_hex(obs_lab)
    assert pm["hex"] == expected_hex
    assert pm["hex"] != "#DIFFERENT"
