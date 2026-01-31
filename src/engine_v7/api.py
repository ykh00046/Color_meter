"""
Engine V7 Facade API.

This module provides a stable public interface for the V7 inspection engine.
It currently bridges calls to the implementation in `src.engine_v7`,
facilitating future refactoring where the implementation moves here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from src.config.v7_paths import REPO_ROOT, V7_MODELS, V7_ROOT
from src.engine_v7.core.config_loader import load_cfg_with_sku

# Implementation imports
from src.engine_v7.core.model_registry import (
    compute_cfg_hash,
    load_expected_ink_count,
    load_pattern_baseline,
    load_std_models_auto,
)
from src.engine_v7.core.pipeline.analyzer import evaluate_multi, evaluate_per_color
from src.engine_v7.core.pipeline.single_analyzer import analyze_single_sample
from src.engine_v7.core.types import Decision, GateResult

logger = logging.getLogger(__name__)


def load_config(sku: str, cfg_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load engine configuration for a specific SKU.

    1. Loads base engine defaults from src/engine_v7/configs/default.json
    2. Overlays SKU-specific settings from config/sku_db/{sku}.json (if exists)
    3. Applies runtime overrides (if provided)
    """
    base_cfg_path = V7_ROOT / "configs" / "default.json"
    sku_db_dir = REPO_ROOT / "config" / "sku_db"

    try:
        # Load base and SKU overlay using the core config_loader
        cfg, sources, warnings = load_cfg_with_sku(base_path=str(base_cfg_path), sku=sku, sku_dir=str(sku_db_dir))

        if warnings:
            logger.warning(f"SKU {sku} config has unknown keys: {', '.join(warnings)}")

        # 3. Apply runtime overrides
        if cfg_override:
            from copy import deepcopy

            from src.engine_v7.core.config_loader import deep_merge

            cfg = deep_merge(deepcopy(cfg), cfg_override)

        return cfg
    except Exception as exc:
        logger.error(f"Failed to load config for SKU {sku}: {exc}")
        # Fallback to defaults only if everything fails
        try:
            from src.engine_v7.core.config_loader import load_cfg

            cfg, _, _ = load_cfg(str(base_cfg_path))
            return cfg
        except Exception as e:
            logger.error(f"Fallback config load also failed: {e}")
            return {}


def load_models(sku: str, ink: str, cfg: Optional[Dict[str, Any]] = None) -> Tuple[Any, str, Any, list]:
    """
    Load standard models and baselines for inspection.

    Args:
        sku: SKU identifier
        ink: Ink identifier
        cfg: Configuration dictionary (used for hash verification)

    Returns:
        Tuple containing:
        - models: Loaded models structure
        - color_mode: 'aggregate' or 'per_color'
        - color_metadata: Metadata for colors (if per_color)
        - reasons: List of failure reasons
    """
    cfg_hash = compute_cfg_hash(cfg) if cfg else None

    try:
        return load_std_models_auto(str(V7_MODELS), sku, ink, cfg_hash=cfg_hash)
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return None, "error", None, [str(e)]


def inspect_single(
    image_bgr: np.ndarray,
    sku: str,
    ink: str = "INK_DEFAULT",
    cfg_override: Optional[Dict[str, Any]] = None,
    # Context options
    expected_ink_count: Optional[int] = None,
    run_id: str = "",
) -> Dict[str, Any]:
    """
    Run single sample analysis (Feature Extraction).
    Does NOT require STD models.
    """
    # 1. Load Configuration
    v7_cfg = load_config(sku, cfg_override)
    if not v7_cfg:
        return {"error": "Configuration could not be loaded"}

    # 2. Determine Ink Count
    final_ink_count = expected_ink_count
    if final_ink_count is None:
        final_ink_count = v7_cfg.get("expected_ink_count")

    # Override config with runtime ink count
    if final_ink_count is not None:
        v7_cfg["expected_ink_count"] = int(final_ink_count)

    # 3. Execute Analysis
    try:
        # Use single_analyzer which produces full feature extraction (ink, radial, etc.)
        results = analyze_single_sample(
            test_bgr=image_bgr,
            cfg=v7_cfg,
            analysis_modes=["gate", "color", "radial", "ink", "pattern", "zones", "plate"],
            match_id=run_id,
        )

        # 4. Wrap for API response
        # The web UI expects: { results: [ { analysis: ... } ] } or direct result
        # app.py wraps this return value into response['v7']

        # We return a structure that mimics the 'result' object expected by the UI
        return {
            "results": [
                {"filename": "image", "analysis": results, "metadata": {"sku": sku, "ink": ink, "run_id": run_id}}
            ]
        }

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return {"error": str(e)}


def _create_error_result(
    label: str, reasons: list, cfg: Dict[str, Any] = None, baseline_reasons: list = None, ink_count: int = None
) -> Dict[str, Any]:
    """Helper to create a standardized error result."""
    decision = Decision(
        label=label,
        reasons=reasons,
        reason_codes=reasons,
        reason_messages=reasons,
        gate=GateResult(passed=False, reasons=reasons, scores={}),
        signature=None,
        anomaly=None,
        phase="INSPECTION",
    )
    if reasons:
        decision.debug = {"error_context": {"reasons": reasons}}

    return {
        "decision": decision,
        "v7_cfg": cfg or {},
        "pattern_baseline_reasons": baseline_reasons or [],
        "expected_ink_count_input": ink_count,
    }
