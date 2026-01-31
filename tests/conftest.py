import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def white_black_pair():
    """Uniform white/black image pair."""
    white = np.full((100, 100, 3), 200, dtype=np.uint8)
    black = np.full((100, 100, 3), 50, dtype=np.uint8)
    return white, black


@pytest.fixture
def gradient_pair():
    """Left-to-right gradient image pair."""
    grad = np.linspace(0, 255, 100).astype(np.uint8)
    white = np.tile(grad, (100, 1))[:, :, None].repeat(3, axis=2)
    black = np.full((100, 100, 3), 20, dtype=np.uint8)
    return white, black


@pytest.fixture
def synthetic_ink_data():
    """Synthetic data for ink restoration validation."""
    ink_lab = np.array([40.0, 20.0, 30.0], dtype=np.float32)
    paper_lab = np.array([95.0, 0.0, 0.0], dtype=np.float32)
    alpha = 0.6
    obs_lab = alpha * ink_lab + (1.0 - alpha) * paper_lab
    return {"ink_lab": ink_lab, "paper_lab": paper_lab, "alpha": alpha, "obs_lab": obs_lab}


@pytest.fixture
def plate_lite_config():
    """Plate-Lite test config."""
    return {
        "enabled": True,
        "blur_ksize": 5,
        "backlight": 255.0,
        "alpha_threshold": 0.1,
        "paper_color": {"lab": [95.0, 0.0, 0.0], "source": "static"},
    }


@pytest.fixture
def tmp_json(tmp_path: Path):
    def _make(data, name="sample.json"):
        path = tmp_path / name
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    return _make


@pytest.fixture
def sample_image():
    # 100x100 RGB 검정 바탕
    return np.zeros((100, 100, 3), dtype=np.uint8)
