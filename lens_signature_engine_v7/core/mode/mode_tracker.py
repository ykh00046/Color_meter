"""
Mode Tracker - Mode Shift Detection for Sampling QC

Tracks per-SKU decision mode changes (LOW/MID/HIGH) for sampling quality control.

Operational Stability Improvements:
1. history default_factory - Safe dataclass field initialization
2. new_mode normalization - Strips whitespace, prevents spurious shifts
3. Atomic write - Temp file + replace prevents corrupted JSON
4. Concurrent access - File lock prevents race conditions
5. SKU filename sanitization - Prevents path traversal issues
6. JSON corruption recovery - Handles decode errors gracefully

State storage: {state_dir}/{sanitized_sku}.json
Lock mechanism: {state_dir}/.{sanitized_sku}.lock
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_name(s: str) -> str:
    """
    5) Sanitize SKU for safe filename usage.

    Prevents path traversal (../, /) and filesystem-unsafe characters.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", s or "unknown")


@dataclass
class ModeState:
    sku: str
    last_mode: str = ""
    # 1) history default_factory - Safe initialization
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _state_path(state_dir: str, sku: str) -> Path:
    """Get state file path with sanitized SKU"""
    safe_sku = _safe_name(sku)
    return Path(state_dir) / f"{safe_sku}.json"


def _lock_path(state_dir: str, sku: str) -> Path:
    """Get lock file path for concurrent access control"""
    safe_sku = _safe_name(sku)
    return Path(state_dir) / f".{safe_sku}.lock"


class _FileLock:
    """
    4) Simple file-based lock for concurrent access.

    Uses exclusive lock on lock file to prevent race conditions.
    Cross-platform compatible (fcntl on Unix, msvcrt on Windows).
    """

    def __init__(self, lock_file: Path, timeout: float = 5.0):
        self.lock_file = lock_file
        self.timeout = timeout
        self.fp = None

    def __enter__(self):
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(self.lock_file, "w")

        start = time.time()
        while True:
            try:
                # Platform-specific locking
                if sys.platform == "win32":
                    import msvcrt

                    msvcrt.locking(self.fp.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (IOError, OSError):
                if time.time() - start > self.timeout:
                    self.fp.close()
                    raise TimeoutError(f"Could not acquire lock on {self.lock_file}")
                time.sleep(0.01)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fp:
            # Platform-specific unlock
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self.fp.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
            self.fp.close()


def load_state(state_dir: str, sku: str) -> ModeState:
    """
    Load SKU mode state with JSON corruption recovery.

    6) Handles JSONDecodeError gracefully - returns fresh state if corrupted.
    """
    p = _state_path(state_dir, sku)

    if not p.exists():
        return ModeState(sku=sku, last_mode="", history=[])

    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return ModeState(sku=sku, last_mode=d.get("last_mode", ""), history=d.get("history", []))
    except (json.JSONDecodeError, ValueError, OSError) as e:
        # 6) JSON corruption recovery - log and return fresh state
        # In production, log this to monitoring system
        print(f"Warning: Corrupted state file {p}: {e}. Starting fresh.", file=sys.stderr)
        return ModeState(sku=sku, last_mode="", history=[])


def save_state(state_dir: str, state: ModeState, window: int = 10) -> None:
    """
    Save SKU mode state with atomic write and concurrency protection.

    3) Atomic write - Writes to temp file first, then replaces
    4) File lock - Prevents concurrent write conflicts
    """
    p = _state_path(state_dir, state.sku)
    lock_file = _lock_path(state_dir, state.sku)

    p.parent.mkdir(parents=True, exist_ok=True)

    # Trim history to window size
    hist = state.history or []
    if window and len(hist) > window:
        hist = hist[-window:]

    payload = {"sku": state.sku, "last_mode": state.last_mode, "history": hist}

    # 4) Acquire lock for atomic operation
    with _FileLock(lock_file):
        # 3) Atomic write via temp file
        tmp = p.with_suffix(".json.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            # Atomic replace (overwrites existing file)
            tmp.replace(p)
        finally:
            # Clean up temp file if replace failed
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass


def update_and_check_shift(state: ModeState, new_mode: str, *, window: int = 10) -> Dict[str, Any]:
    """
    Update mode and detect shift (Sampling QC).

    2) new_mode normalization - Strips whitespace to prevent spurious shifts

    We don't override decisions. We only emit a shift flag when mode changes vs last.
    """
    # 2) Normalize new_mode - strip whitespace
    new_mode = (new_mode or "").strip()

    now = int(time.time())
    prev = state.last_mode
    state.last_mode = new_mode
    state.history = (state.history or []) + [{"ts": now, "mode": new_mode}]

    # Trim history to window size
    if window and len(state.history) > window:
        state.history = state.history[-window:]

    shifted = (prev != "") and (prev != new_mode)
    return {
        "shifted": shifted,
        "from": prev,
        "to": new_mode,
        "ts": now,
        "history_len": len(state.history),
    }
