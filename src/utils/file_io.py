import json
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np


class FileIO:
    def load_image(self, filepath: Path) -> np.ndarray:
        filepath = Path(filepath)
        if not filepath.exists():
            return None
        try:
            # np.fromfile + imdecode로 비ASCII 경로에서도 로딩 안정화
            data = np.fromfile(str(filepath), dtype=np.uint8)
            if data.size == 0:
                return None
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    def save_image(self, filepath: Path, image: np.ndarray) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(filepath), image)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(path: Path, pattern: str = "*.*") -> List[Path]:
    return list(path.glob(pattern))


def read_json(filepath: Path) -> dict:
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def write_json(data: Any, filepath: Path):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
