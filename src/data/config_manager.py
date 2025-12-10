from pathlib import Path
from typing import Any, Dict, Optional
import json
from src.utils.file_io import read_json, write_json

class ConfigManager:
    def __init__(self, path: Optional[Path] = None, data: Optional[Dict] = None):
        self.config_path = path
        self._config = {}
        if data:
            self._config = data
        elif path and path.exists():
            self._config = read_json(path)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        val = self._config
        try:
            for k in keys:
                val = val[k]
            return val
        except:
            return default

    def set(self, key: str, value: Any):
        keys = key.split('.')
        val = self._config
        for k in keys[:-1]:
            val = val.setdefault(k, {})
        val[keys[-1]] = value

    def save(self, path: Optional[Path] = None):
        target = path or self.config_path
        if target:
            write_json(self._config, target)
