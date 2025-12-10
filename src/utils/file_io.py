from pathlib import Path
import numpy as np
import cv2
import json
from typing import List, Any

class FileIO:
    def load_image(self, filepath: Path) -> np.ndarray:
        if filepath.exists():
            return cv2.imread(str(filepath))
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
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def write_json(data: Any, filepath: Path):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
