"""
Config Normalization Module for V7 Engine

This module provides centralized configuration access functions to eliminate
inconsistencies between modules that read config in different ways.

Problem addressed:
- Some modules use cfg.get("polar_R") (flat structure)
- Others use cfg.get("polar", {}).get("R") (nested structure)
- Some use cfg["polar"]["R"] (direct access, can raise KeyError)

This normalization layer supports both old and new config formats with
proper fallbacks, preventing "Polar Sum=0" and coordinate mismatches.
"""

from typing import Any, Dict, Tuple


def get_polar_dims(cfg: Dict[str, Any]) -> Tuple[int, int]:
    """
    Get polar coordinate dimensions (R, T) from config.

    Supports both flat and nested config structures:
    - Nested: cfg["polar"]["R"], cfg["polar"]["T"]
    - Flat (legacy): cfg["polar_R"], cfg["polar_T"]

    Args:
        cfg: Configuration dictionary

    Returns:
        Tuple of (R, T) where:
        - R: Radial dimension (default: 260)
        - T: Theta/angular dimension (default: 720)

    Examples:
        >>> cfg = {"polar": {"R": 300, "T": 800}}
        >>> get_polar_dims(cfg)
        (300, 800)

        >>> cfg = {"polar_R": 300, "polar_T": 800}  # legacy
        >>> get_polar_dims(cfg)
        (300, 800)

        >>> cfg = {}  # empty config
        >>> get_polar_dims(cfg)
        (260, 720)
    """
    if not isinstance(cfg, dict):
        return 260, 720

    # Try nested structure first (preferred)
    polar = cfg.get("polar", {}) or {}
    R = polar.get("R") if isinstance(polar, dict) else None
    T = polar.get("T") if isinstance(polar, dict) else None

    # Fallback to flat structure (legacy)
    if R is None:
        R = cfg.get("polar_R", 260)
    if T is None:
        T = cfg.get("polar_T", 720)

    return int(R), int(T)


def get_plate_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get plate-specific configuration subcfg.

    Args:
        cfg: Configuration dictionary

    Returns:
        Plate configuration dictionary (empty dict if not found)

    Examples:
        >>> cfg = {"plate": {"r_clear": 0.4, "r_ring0": 0.7}}
        >>> get_plate_cfg(cfg)
        {'r_clear': 0.4, 'r_ring0': 0.7}
    """
    if not isinstance(cfg, dict):
        return {}

    plate = cfg.get("plate", {})
    return plate if isinstance(plate, dict) else {}


def get_ink_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get ink-specific configuration subcfg.

    Args:
        cfg: Configuration dictionary

    Returns:
        Ink configuration dictionary (empty dict if not found)

    Examples:
        >>> cfg = {"ink": {"k": 3, "dist_weight": 1.0}}
        >>> get_ink_cfg(cfg)
        {'k': 3, 'dist_weight': 1.0}
    """
    if not isinstance(cfg, dict):
        return {}

    ink = cfg.get("ink", {})
    return ink if isinstance(ink, dict) else {}


def get_roi_params(cfg: Dict[str, Any]) -> Tuple[float, float, bool]:
    """
    Get ROI (Region of Interest) parameters for signature analysis.

    Args:
        cfg: Configuration dictionary

    Returns:
        Tuple of (r_start, r_end, center_excluded) where:
        - r_start: Inner radius ratio (default: 0.0)
        - r_end: Outer radius ratio (default: 1.0)
        - center_excluded: Whether center is excluded (default: False)

    Examples:
        >>> cfg = {"signature": {"r_start": 0.2, "r_end": 0.9}}
        >>> get_roi_params(cfg)
        (0.2, 0.9, False)
    """
    if not isinstance(cfg, dict):
        return 0.0, 1.0, False

    # Try signature subcfg first
    sig = cfg.get("signature", {}) or {}
    if isinstance(sig, dict):
        r_start = sig.get("r_start", 0.0)
        r_end = sig.get("r_end", 1.0)
        center_excluded = sig.get("center_excluded", False)
    else:
        # Fallback to top-level keys
        r_start = cfg.get("r_start", 0.0)
        r_end = cfg.get("r_end", 1.0)
        center_excluded = cfg.get("center_excluded", False)

    return float(r_start), float(r_end), bool(center_excluded)


def get_v2_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get v2-specific ink configuration subcfg.

    Args:
        cfg: Configuration dictionary

    Returns:
        V2 configuration dictionary (empty dict if not found)
    """
    if not isinstance(cfg, dict):
        return {}

    v2 = cfg.get("v2_ink", {})
    return v2 if isinstance(v2, dict) else {}
