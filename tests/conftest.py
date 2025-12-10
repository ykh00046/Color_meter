import json
from pathlib import Path

import numpy as np
import pytest


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

