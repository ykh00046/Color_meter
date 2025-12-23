# Force reload v8: Added sampling ROI info + sector uniformity analysis
import asyncio
import logging
import shutil
import uuid
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.pipeline import InspectionPipeline, PipelineError
from src.services.analysis_service import AnalysisService
from src.utils.file_io import read_json
from src.visualizer import InspectionVisualizer, VisualizerConfig

logger = logging.getLogger("web")

# PHASE7: Image caching for parameter recomputation
IMAGE_CACHE_MAX = 100
IMAGE_CACHE_TTL_SEC = 60 * 30  # 30 minutes
image_cache: Dict[str, Tuple[np.ndarray, float]] = {}
cache_lock = asyncio.Lock()


def _evict_old_cache() -> None:
    now = time()
    expired = [k for k, (_, ts) in image_cache.items() if now - ts > IMAGE_CACHE_TTL_SEC]
    for k in expired:
        image_cache.pop(k, None)

    if len(image_cache) <= IMAGE_CACHE_MAX:
        return

    # Evict oldest entries by timestamp
    to_evict = sorted(image_cache.items(), key=lambda item: item[1][1])[: len(image_cache) - IMAGE_CACHE_MAX]
    for k, _ in to_evict:
        image_cache.pop(k, None)


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR.parent / "results" / "web"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Color Meter Web UI", version="0.2")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server error")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
analysis_service = AnalysisService()


# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and cleanup old results on application startup"""
    from src.models.database import create_tables, init_database

    logger.info("Initializing database...")
    init_database(database_url="sqlite:///./color_meter.db", echo=False)
    # Tables are already created by Alembic migration
    # create_tables() is not needed if using Alembic
    logger.info("Database initialized")

    # Run result directory cleanup
    logger.info("Running result directory cleanup on startup...")
    cleanup_old_results(max_age_hours=24, max_results=100)


# Include API routers
from src.web.routers import comparison, inspection, sku, std, test

app.include_router(std.router)
app.include_router(test.router)
app.include_router(comparison.router)
app.include_router(sku.router)
app.include_router(inspection.router)


def load_sku_config(sku: str, config_dir: Path = Path("config/sku_db")) -> dict:
    from src.utils.security import SecurityError, safe_sku_path

    try:
        cfg_path = safe_sku_path(sku, config_dir)
    except SecurityError as e:
        raise HTTPException(status_code=400, detail=f"Invalid SKU: {str(e)}")

    if not cfg_path.exists():
        raise HTTPException(status_code=404, detail=f"SKU config not found: {sku}")
    return read_json(cfg_path)


def _safe_result_path(run_id: str, filename: Optional[str] = None) -> Path:
    run_dir = (RESULTS_DIR / run_id).resolve()
    try:
        run_dir.relative_to(RESULTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run_id")

    if filename is None:
        return run_dir

    target = (run_dir / filename).resolve()
    try:
        target.relative_to(run_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return target


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    import zipfile

    dest_dir = dest_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = (dest_dir / member.filename).resolve()
            try:
                member_path.relative_to(dest_dir)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid zip entry path")
        zf.extractall(dest_dir)


def cleanup_old_results(max_age_hours: int = 24, max_results: int = 100):
    """
    Clean up old result files to prevent disk exhaustion.

    Args:
        max_age_hours: Delete results older than this (hours)
        max_results: Keep only this many most recent results
    """
    import time

    try:
        if not RESULTS_DIR.exists():
            return

        # Get all result directories (each run_id is a directory)
        result_dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir()]

        if not result_dirs:
            return

        # Sort by modification time (newest first)
        result_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

        # Strategy 1: Keep only max_results newest
        dirs_to_delete_by_count = result_dirs[max_results:]

        # Strategy 2: Delete anything older than max_age_hours
        cutoff_time = time.time() - (max_age_hours * 3600)
        dirs_to_delete_by_age = [d for d in result_dirs if d.stat().st_mtime < cutoff_time]

        # Union of both strategies
        dirs_to_delete = set(dirs_to_delete_by_count) | set(dirs_to_delete_by_age)

        deleted_count = 0
        for dir_path in dirs_to_delete:
            try:
                shutil.rmtree(dir_path)
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {dir_path}: {e}")

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old result directories")

    except Exception as e:
        logger.error(f"Error during result cleanup: {e}")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    """Inspection history viewer page"""
    return templates.TemplateResponse("history.html", {"request": request})


@app.get("/stats", response_class=HTMLResponse)
async def stats_page(request: Request):
    """Inspection statistics dashboard page"""
    return templates.TemplateResponse("stats.html", {"request": request})


@app.get("/compare", response_class=HTMLResponse)
async def compare_page(request: Request):
    """STD vs Sample one-off comparison page"""
    return templates.TemplateResponse("compare.html", {"request": request})


# ================================
# Helper Functions for /inspect endpoint
# ================================


async def validate_and_save_file(file: UploadFile, run_id: str, run_dir: Path) -> tuple[bytes, Path, str]:
    """Validate uploaded file and save to disk.

    Returns:
        tuple: (file_content, input_path, original_name)
    """
    from src.utils.security import validate_file_extension, validate_file_size

    # File validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    if not validate_file_extension(file.filename, [".jpg", ".jpeg", ".png", ".bmp"]):
        raise HTTPException(status_code=400, detail="Invalid file type. Allowed: .jpg, .jpeg, .png, .bmp")

    # Read and validate size
    file_content = await file.read()
    file_size = len(file_content)

    if not validate_file_size(file_size, max_size_mb=10):
        raise HTTPException(status_code=413, detail=f"File too large: {file_size / 1024 / 1024:.1f}MB (max 10MB)")

    # Save file
    original_name = file.filename
    safe_ext = Path(file.filename).suffix or ".jpg"
    input_path = run_dir / f"upload_{run_id}{safe_ext}"

    with input_path.open("wb") as f:
        f.write(file_content)

    return file_content, input_path, original_name


def parse_inspection_options(options: Optional[str]) -> bool:
    """Parse inspection options JSON string.

    Returns:
        bool: enable_illumination_correction
    """
    if not options:
        return False

    try:
        import json

        opts = json.loads(options)
        logger.info(f"Received options: {opts}")
        return opts.get("illumination_correction", False)
    except Exception as e:
        logger.warning(f"Failed to parse options: {e}")
        return False


def run_2d_zone_analysis(img_bgr, lens_detection, sku_config: dict, run_dir: Path):
    """Run 2D zone analysis using zone_analyzer_2d.

    Returns:
        tuple: (result_2d, debug_info_2d) or (None, None) on failure
    """
    try:
        print("[INSPECT] Using 2D zone analysis (AI template)")
        from src.core.zone_analyzer_2d import InkMaskConfig, analyze_lens_zones_2d

        if img_bgr is None or lens_detection is None:
            raise ValueError("Invalid inputs for 2D analysis")

        print(
            f"[INSPECT] lens_detection: center=({lens_detection.center_x}, {lens_detection.center_y}), "
            f"radius={lens_detection.radius}"
        )

        # Execute 2D analysis
        result_2d, debug_info_2d = analyze_lens_zones_2d(
            img_bgr=img_bgr,
            lens_detection=lens_detection,
            sku_config=sku_config,
            ink_mask_config=InkMaskConfig(),
            save_debug=True,
            debug_prefix=str(run_dir / "debug_2d"),
        )

        print(f"[INSPECT] 2D analysis complete: {result_2d.judgment}, ΔE={result_2d.overall_delta_e:.2f}")
        return result_2d, debug_info_2d

    except Exception as e2d:
        print(f"[INSPECT] 2D analysis FAILED: {e2d}")
        import traceback

        traceback.print_exc()
        return None, None


def apply_2d_results_to_inspection(result, result_2d):
    """Apply 2D analysis results to inspection result object."""
    if result_2d is None:
        return

    result.judgment = result_2d.judgment
    result.overall_delta_e = result_2d.overall_delta_e
    result.zone_results = result_2d.zone_results
    result.ng_reasons = result_2d.ng_reasons
    result.confidence = result_2d.confidence

    # Remove 1D data to avoid confusion
    result.zones = []
    result.uniformity_analysis = None
    print("[INSPECT] Applied 2D results and removed 1D data")


def generate_radial_analysis(result, sku_config: dict) -> tuple:
    """Generate radial profile analysis data.

    Returns:
        tuple: (analysis_payload, lens_info)
    """
    rp = getattr(result, "radial_profile", None)
    lens_detection = getattr(result, "lens_detection", None)

    if rp is None or lens_detection is None:
        return None, None

    zones_cfg = sku_config.get("zones", {})
    analysis_payload = analysis_service.analyze_radial_profile(
        profile=rp, lens_radius=float(lens_detection.radius), zones_config=zones_cfg
    )

    lens_info = {
        "center_x": float(lens_detection.center_x),
        "center_y": float(lens_detection.center_y),
        "radius": float(lens_detection.radius),
        "confidence": float(lens_detection.confidence),
    }

    logger.warning(
        f"[DEBUG] Lens detected: center=({lens_info['center_x']:.1f}, {lens_info['center_y']:.1f}), "
        f"radius={lens_info['radius']:.1f}"
    )

    return analysis_payload, lens_info


def run_ring_sector_analysis(result, enable_illumination_correction: bool):
    """
    Run Ring × Sector 2D analysis (PHASE7).

    Refactored to use SectorSegmenter module.

    Returns:
        dict or None: ring_sector_data
    """
    lens_detection = getattr(result, "lens_detection", None)
    if lens_detection is None or not hasattr(result, "image") or result.image is None:
        return None

    try:
        from src.core.sector_segmenter import SectorConfig, SectorSegmenter

        # Initialize segmenter
        segmenter = SectorSegmenter(SectorConfig(sector_count=12, ring_boundaries=[0.0, 0.33, 0.67, 1.0]))  # 3 rings

        # Run segmentation and analysis
        segmentation_result, uniformity_data = segmenter.segment_and_analyze(
            image_bgr=result.image,
            center_x=float(lens_detection.center_x),
            center_y=float(lens_detection.center_y),
            radius=float(lens_detection.radius),
            radial_profile=getattr(result, "radial_profile", None),
            enable_illumination_correction=enable_illumination_correction,
        )

        # Store cells in result
        result.ring_sector_cells = segmentation_result.cells

        # Store uniformity analysis
        if uniformity_data is not None:
            result.uniformity_analysis = uniformity_data

        # Convert to API response format
        ring_sector_data = segmenter.format_response_data(segmentation_result.cells)

        # Log ring statistics
        ring_stats = {}
        for cell in segmentation_result.cells:
            if cell.ring_index not in ring_stats:
                ring_stats[cell.ring_index] = {"count": 0, "total_pixels": 0}
            ring_stats[cell.ring_index]["count"] += 1
            ring_stats[cell.ring_index]["total_pixels"] += cell.pixel_count

        for ring_idx, stats in sorted(ring_stats.items()):
            logger.warning(
                f"[DEBUG] Ring {ring_idx}: {stats['count']} cells, "
                f"{stats['total_pixels']:,} pixels (avg {stats['total_pixels'] // stats['count']:,} per cell)"
            )

        logger.info(
            f"2D analysis completed: {segmentation_result.total_cells} cells, "
            f"{segmentation_result.valid_pixel_ratio*100:.1f}% valid pixels"
        )

        return ring_sector_data

    except Exception as e:
        logger.warning(f"2D analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def create_overlay_visualization(result, run_dir: Path) -> Optional[Path]:
    """Create overlay visualization image.

    Returns:
        Path or None: overlay_path
    """
    overlay_path = run_dir / "overlay.png"
    try:
        visualizer = InspectionVisualizer(VisualizerConfig())
        overlay = visualizer.visualize_zone_overlay(
            getattr(result, "image", None),
            getattr(result, "lens_detection", None),
            getattr(result, "zones", None),
            result,
            show_result=True,
        )
        visualizer.save_visualization(overlay, overlay_path)
        return overlay_path
    except Exception as viz_err:
        logger.warning(f"Failed to create overlay: {viz_err}")
        return None


def save_result_json(result, debug_info_2d, use_2d_analysis: bool, run_dir: Path) -> Path:
    """Save inspection result as JSON.

    Returns:
        Path: output JSON file path
    """
    import json
    from dataclasses import asdict

    output = run_dir / "result.json"
    result_dict = asdict(result)

    # Remove 1D data if using 2D analysis
    if use_2d_analysis and hasattr(result, "zone_results") and result.zone_results:
        result_dict.pop("zones", None)
        result_dict.pop("uniformity_analysis", None)
        print("[INSPECT] Removed zones/uniformity from JSON output")

        # Add 2D debug info
        if debug_info_2d is not None:
            result_dict["debug"] = debug_info_2d
            print(f"[INSPECT] Added debug info to JSON output: {list(debug_info_2d.keys())}")

    output.write_text(json.dumps(result_dict, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def save_inspection_to_db(
    session_id: str,
    sku_code: str,
    image_filename: str,
    image_path: str,
    result,
    operator: Optional[str] = None,
) -> None:
    """
    Save inspection result to database.

    Args:
        session_id: Unique session ID (run_id)
        sku_code: SKU code
        image_filename: Original image filename
        image_path: Full path to saved image
        result: InspectionResult object
        operator: Operator name (optional)
    """
    try:
        from src.models.database import get_db
        from src.web.routers.inspection import save_inspection_to_history

        db = get_db()
        try:
            save_inspection_to_history(
                db=db,
                session_id=session_id,
                sku_code=sku_code,
                image_filename=image_filename,
                result=result,
                image_path=image_path,
                operator=operator,
                processing_time_ms=None,  # Could be measured if needed
            )
        finally:
            db.close()

        logger.info(f"Saved inspection to database: session_id={session_id}")
    except Exception as e:
        # Don't fail the inspection if DB save fails
        logger.error(f"Failed to save inspection to database: {e}", exc_info=True)


def build_inspection_response(
    run_id: str,
    original_name: str,
    sku: str,
    overlay_path: Optional[Path],
    analysis_payload,
    lens_info,
    ring_sector_data,
    uniformity_data,
    output: Path,
    result,
    run_judgment: bool,
    image_id: Optional[str] = None,
    applied_params: Optional[Dict[str, Any]] = None,
) -> dict:
    """Build final inspection API response."""
    response = {
        "run_id": run_id,
        "image": original_name,
        "sku": sku,
        "image_id": image_id,  # PHASE7: For parameter recomputation
        "overlay": f"/results/{run_id}/overlay.png" if overlay_path and overlay_path.exists() else None,
        "analysis": analysis_payload,
        "metrics": getattr(result, "metrics", None),
        "lens_info": lens_info,
        "ring_sector_cells": ring_sector_data,
        "uniformity": uniformity_data,
        "result_path": str(output),
    }
    if applied_params:
        response["applied_params"] = applied_params

    # Add judgment results if requested
    if run_judgment:
        zone_results_data = []
        if hasattr(result, "zone_results") and result.zone_results:
            for zr in result.zone_results:
                zone_results_data.append(
                    {
                        "zone_name": zr.zone_name,
                        "measured_lab": [float(v) for v in zr.measured_lab],
                        "target_lab": [float(v) for v in zr.target_lab] if zr.target_lab else None,
                        "delta_e": float(zr.delta_e),
                        "threshold": float(zr.threshold),
                        "is_ok": zr.is_ok,
                    }
                )

        response["judgment"] = {
            "result": result.judgment,
            "overall_delta_e": float(result.overall_delta_e),
            "confidence": float(result.confidence) if hasattr(result, "confidence") else 1.0,
            "zones_count": len(getattr(result, "zone_results", [])),
            "ng_reasons": result.ng_reasons if hasattr(result, "ng_reasons") else [],
        }
        response["zone_results"] = zone_results_data
    else:
        response["judgment"] = None
        response["zone_results"] = []

    return response


# ================================
# Main /inspect endpoint (refactored)
# ================================


@app.post("/inspect")
async def inspect_image(
    file: UploadFile = File(...),
    sku: str = Form(...),
    expected_zones: Optional[int] = Form(None),
    run_judgment: bool = Form(False),
    options: Optional[str] = Form(None),
    use_2d_analysis: bool = Form(True),
):
    """
    Main inspection endpoint (refactored for clarity).

    Handles file upload, validation, pipeline execution, 2D analysis,
    visualization, and response generation.
    """
    # 1. Setup
    run_id = uuid.uuid4().hex[:8]
    run_dir = _safe_result_path(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 2. File validation and save
    file_content, input_path, original_name = await validate_and_save_file(file, run_id, run_dir)

    # 3. Parse options
    enable_illumination_correction = parse_inspection_options(options)

    # 4. Load SKU config
    sku_config = load_sku_config(sku)
    if expected_zones is not None:
        sku_config.setdefault("params", {})["expected_zones"] = expected_zones

    # 5. Run pipeline
    pipeline = InspectionPipeline(sku_config, save_intermediates=True)
    try:
        result = pipeline.process(str(input_path), sku, save_dir=run_dir)
    except PipelineError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 6. Run 2D zone analysis (optional)
    debug_info_2d = None
    image_id = None

    if use_2d_analysis:
        import cv2

        img_bgr = cv2.imread(str(input_path))
        lens_detection = getattr(result, "lens_detection", None)

        if img_bgr is not None and lens_detection is not None:
            # PHASE7: Cache image for parameter recomputation
            image_id = str(uuid.uuid4())
            async with cache_lock:
                image_cache[image_id] = (img_bgr.copy(), time())
                _evict_old_cache()
            logger.info(f"Cached image with ID: {image_id}")

            result_2d, debug_info_2d = run_2d_zone_analysis(img_bgr, lens_detection, sku_config, run_dir)
            apply_2d_results_to_inspection(result, result_2d)

    # 7. Generate radial analysis data
    analysis_payload, lens_info = generate_radial_analysis(result, sku_config)

    # 8. Run ring-sector 2D analysis (PHASE7)
    ring_sector_data = run_ring_sector_analysis(result, enable_illumination_correction)
    uniformity_data = getattr(result, "uniformity_analysis", None)

    # 9. Create overlay visualization
    overlay_path = create_overlay_visualization(result, run_dir)

    # 10. Save result JSON
    output = save_result_json(result, debug_info_2d, use_2d_analysis, run_dir)

    # 10.5. Save to database history
    save_inspection_to_db(
        session_id=run_id,
        sku_code=sku,
        image_filename=original_name,
        image_path=str(input_path),
        result=result,
    )

    # 11. Build and return response
    return build_inspection_response(
        run_id=run_id,
        original_name=original_name,
        sku=sku,
        overlay_path=overlay_path,
        analysis_payload=analysis_payload,
        lens_info=lens_info,
        ring_sector_data=ring_sector_data,
        uniformity_data=uniformity_data,
        output=output,
        result=result,
        run_judgment=run_judgment,
        image_id=image_id,
    )


# ================================
# PHASE7: Parameter Recomputation API
# ================================


def validate_recompute_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize recomputation parameters.

    Allowed parameter groups:
    - segmenter_config: detection_method, smoothing_window, min_gradient, min_delta_e,
                       expected_zones, uniform_split_priority
    - profiler_config: num_samples, num_points, sample_percentile
    - corrector_config: method (gray_world, white_patch, auto, polynomial, gaussian)
    - sector_config: sector_count, ring_boundaries

    Returns:
        dict: Validated parameters

    Raises:
        HTTPException: If parameters are invalid
    """
    allowed_params = {
        # Zone segmentation
        "detection_method": ["gradient", "delta_e", "hybrid", "variable_width"],
        "smoothing_window": (1, 100),  # (min, max)
        "min_gradient": (0.0, 10.0),
        "min_delta_e": (0.0, 20.0),
        "expected_zones": (1, 20),
        "uniform_split_priority": [True, False],
        # Radial profiling
        "num_samples": (100, 10000),
        "num_points": (50, 1000),
        "sample_percentile": (0, 100),
        # Illumination correction
        "correction_method": ["gray_world", "white_patch", "auto", "polynomial", "gaussian", "none"],
        # Ring × Sector
        "sector_count": (4, 36),
        "ring_count": (1, 10),
    }

    validated = {}

    for key, value in params.items():
        if key not in allowed_params:
            raise HTTPException(
                status_code=400, detail=f"Unknown parameter: {key}. Allowed: {list(allowed_params.keys())}"
            )

        constraint = allowed_params[key]

        # Check enum values
        if isinstance(constraint, list):
            if value not in constraint:
                raise HTTPException(status_code=400, detail=f"Invalid value for {key}: {value}. Allowed: {constraint}")

        # Check numeric ranges
        elif isinstance(constraint, tuple):
            min_val, max_val = constraint
            if not isinstance(value, (int, float)):
                raise HTTPException(status_code=400, detail=f"Parameter {key} must be numeric")
            if not (min_val <= value <= max_val):
                raise HTTPException(
                    status_code=400, detail=f"Parameter {key}={value} out of range [{min_val}, {max_val}]"
                )

        validated[key] = value

    return validated


def apply_params_to_config(sku_config: dict, params: Dict[str, Any]) -> dict:
    """
    Apply validated parameters to SKU config.

    Maps flat parameter names to nested config structure:
    - detection_method → params.detection_method
    - smoothing_window → params.smoothing_window
    - correction_method → corrector.method
    - sector_count → params.sector_count

    Args:
        sku_config: Base SKU configuration
        params: Validated parameters

    Returns:
        dict: Updated config
    """
    config = sku_config.copy()

    # Ensure params section exists
    if "params" not in config:
        config["params"] = {}

    # Ensure corrector section exists
    if "corrector" not in config:
        config["corrector"] = {}

    # Map parameters to config structure
    param_mapping = {
        # Zone segmentation
        "detection_method": ("params", "detection_method"),
        "smoothing_window": ("params", "smoothing_window"),
        "min_gradient": ("params", "min_gradient"),
        "min_delta_e": ("params", "min_delta_e"),
        "expected_zones": ("params", "expected_zones"),
        "uniform_split_priority": ("params", "uniform_split_priority"),
        # Radial profiling
        "num_samples": ("params", "num_samples"),
        "num_points": ("params", "num_points"),
        "sample_percentile": ("params", "sample_percentile"),
        # Illumination correction
        "correction_method": ("corrector", "method"),
        # Ring × Sector
        "sector_count": ("params", "sector_count"),
        "ring_count": ("params", "ring_count"),
    }

    for param_name, value in params.items():
        if param_name in param_mapping:
            section, key = param_mapping[param_name]
            config[section][key] = value
            logger.info(f"Applied parameter: {section}.{key} = {value}")

    return config


@app.post("/recompute")
async def recompute_analysis(
    image_id: str = Form(...),
    sku: str = Form(...),
    params: Optional[str] = Form(None),
    run_judgment: bool = Form(False),
):
    """
    PHASE7: Reanalyze cached image with new parameters.

    This endpoint allows users to adjust analysis parameters without re-uploading
    the image, enabling rapid parameter tuning and experimentation.

    Args:
        image_id: UUID from previous /inspect call
        sku: SKU identifier
        params: JSON string of parameter overrides
        run_judgment: Whether to run judgment logic

    Returns:
        dict: Same format as /inspect endpoint

    Example params:
        {
            "detection_method": "variable_width",
            "smoothing_window": 7,
            "min_gradient": 0.8,
            "expected_zones": 5,
            "correction_method": "auto"
        }
    """
    # 1. Retrieve image from cache
    async with cache_lock:
        entry = image_cache.get(image_id)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail=f"Image ID not found in cache: {image_id}. "
                f"Please re-upload the image or use a valid image_id from /inspect",
            )
        img_bgr, cached_at = entry
        if time() - cached_at > IMAGE_CACHE_TTL_SEC:
            image_cache.pop(image_id, None)
            raise HTTPException(
                status_code=404,
                detail=f"Image ID expired: {image_id}. Please re-upload the image.",
            )
        img_bgr = img_bgr.copy()

    logger.info(f"Retrieved cached image {image_id}: shape={img_bgr.shape}")

    # 2. Parse and validate parameters
    param_overrides = {}
    if params:
        try:
            import json

            param_overrides = json.loads(params)
            param_overrides = validate_recompute_params(param_overrides)
            logger.info(f"Validated parameters: {param_overrides}")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in params: {e}")

    # 3. Load SKU config and apply parameter overrides
    sku_config = load_sku_config(sku)
    if param_overrides:
        sku_config = apply_params_to_config(sku_config, param_overrides)

    # 4. Setup run directory
    run_id = uuid.uuid4().hex[:8]
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save image to run_dir for pipeline processing
    import cv2

    temp_image_path = run_dir / "recompute_input.jpg"
    cv2.imwrite(str(temp_image_path), img_bgr)

    # 5. Run pipeline with new parameters
    pipeline = InspectionPipeline(sku_config, save_intermediates=True)
    try:
        result = pipeline.process(str(temp_image_path), sku, save_dir=run_dir)
    except PipelineError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 6. Run 2D zone analysis
    debug_info_2d = None
    lens_detection = getattr(result, "lens_detection", None)

    if lens_detection is not None:
        result_2d, debug_info_2d = run_2d_zone_analysis(img_bgr, lens_detection, sku_config, run_dir)
        apply_2d_results_to_inspection(result, result_2d)

    # 7. Generate radial analysis data
    analysis_payload, lens_info = generate_radial_analysis(result, sku_config)

    # 8. Run ring-sector 2D analysis
    enable_illumination_correction = param_overrides.get("correction_method") not in [None, "none"]
    ring_sector_data = run_ring_sector_analysis(result, enable_illumination_correction)
    uniformity_data = getattr(result, "uniformity_analysis", None)

    # 9. Create overlay visualization
    overlay_path = create_overlay_visualization(result, run_dir)

    # 10. Save result JSON
    output = save_result_json(result, debug_info_2d, True, run_dir)

    # 11. Build and return response (same format as /inspect)
    response = build_inspection_response(
        run_id=run_id,
        original_name=f"recompute_{image_id}",
        sku=sku,
        overlay_path=overlay_path,
        analysis_payload=analysis_payload,
        lens_info=lens_info,
        ring_sector_data=ring_sector_data,
        uniformity_data=uniformity_data,
        output=output,
        result=result,
        run_judgment=run_judgment,
        image_id=image_id,  # Return original image_id for further recomputation
        applied_params=param_overrides if param_overrides else None,
    )

    return response


@app.post("/batch")
async def batch_inspect(
    sku: str = Form(...),
    batch_dir: Optional[str] = Form(None),
    batch_zip: Optional[UploadFile] = File(None),
    expected_zones: Optional[int] = Form(None),
):
    """
    배치 검사: 서버 경로 또는 ZIP 업로드 중 하나를 선택.
    """
    if not batch_dir and not batch_zip:
        raise HTTPException(status_code=400, detail="batch_dir or batch_zip must be provided")

    run_id = uuid.uuid4().hex[:8]
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    if batch_dir:
        image_paths = list(Path(batch_dir).glob("*.jpg"))
    elif batch_zip:
        # ZIP 업로드를 inputs 폴더에 저장 후 해제
        zip_path = input_dir / batch_zip.filename
        with zip_path.open("wb") as f:
            shutil.copyfileobj(batch_zip.file, f)
        _safe_extract_zip(zip_path, input_dir)
        image_paths = list(input_dir.glob("**/*.jpg"))

    if not image_paths:
        raise HTTPException(status_code=404, detail="No jpg images found for batch processing")

    sku_config = load_sku_config(sku)
    if expected_zones is not None:
        sku_config.setdefault("params", {})["expected_zones"] = expected_zones

    pipeline = InspectionPipeline(sku_config)
    output_csv = run_dir / "batch.csv"

    results = pipeline.process_batch([str(p) for p in image_paths], sku, output_csv=output_csv)

    # 요약 통계
    judgments = [r.judgment for r in results]
    overall_de = [r.overall_delta_e for r in results]
    summary = {
        "OK": judgments.count("OK"),
        "NG": judgments.count("NG"),
        "mean_delta_e": float(np.mean(overall_de)) if overall_de else 0.0,
        "max_delta_e": float(np.max(overall_de)) if overall_de else 0.0,
        "min_delta_e": float(np.min(overall_de)) if overall_de else 0.0,
    }

    return {"run_id": run_id, "count": len(results), "csv": str(output_csv), "summary": summary}


@app.get("/results/{run_id}")
async def get_result(run_id: str):
    run_dir = _safe_result_path(run_id)
    output = run_dir / "result.json"
    if output.exists():
        return FileResponse(output)
    batch = run_dir / "batch.csv"
    if batch.exists():
        return FileResponse(batch)
    raise HTTPException(status_code=404, detail="Result not found")


@app.get("/results/{run_id}/{filename}")
async def get_result_file(run_id: str, filename: str):
    target = _safe_result_path(run_id, filename)
    if target.exists():
        return FileResponse(target)
    raise HTTPException(status_code=404, detail="File not found")


@app.post("/inspect_v2")
async def inspect_v2(
    file: UploadFile = File(...),
    sku: str = Form(...),
    smoothing_window: int = Form(5),
    gradient_threshold: float = Form(0.5),
):
    """
    Dashboard용 endpoint - pipeline.process() 통짜 호출 방식.
    기존 판정 로직을 100% 재사용하고, 추가로 ProfileAnalyzer로 그래프 데이터만 생성.
    """
    from src.analysis.profile_analyzer import ProfileAnalyzer
    from src.utils.security import validate_file_extension, validate_file_size

    # 1. File Validation
    if not validate_file_extension(file.filename, [".jpg", ".jpeg", ".png", ".bmp"]):
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_content = await file.read()
    if not validate_file_size(len(file_content), max_size_mb=10):
        raise HTTPException(status_code=413, detail="File too large")

    run_id = uuid.uuid4().hex[:8]
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 한글 파일명 문제 방지: 원본명 저장하되 실제 파일은 안전한 이름으로
    original_name = file.filename
    safe_ext = Path(file.filename).suffix or ".jpg"
    input_path = run_dir / f"upload_{run_id}{safe_ext}"

    with input_path.open("wb") as f:
        f.write(file_content)

    # 2. Pipeline 통짜 호출 (기존 로직 100% 재사용)
    sku_config = load_sku_config(sku)
    pipeline = InspectionPipeline(sku_config)

    try:
        logger.info(f"[INSPECT_V2] Calling pipeline.process() for {file.filename}")

        inspection_result = pipeline.process(image_path=str(input_path), sku=sku, save_dir=run_dir)

        logger.info(f"[INSPECT_V2] Pipeline completed: {inspection_result.judgment}")

        # 2D zone analysis 실행 (기존 /inspect와 동일)
        logger.info("[INSPECT_V2] Running 2D zone analysis...")
        import cv2

        from src.core.zone_analyzer_2d import InkMaskConfig, analyze_lens_zones_2d

        # 이미지 로드
        img_bgr = cv2.imread(str(input_path))
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {input_path}")

        # lens_detection 확인
        lens_detection = getattr(inspection_result, "lens_detection", None)
        if lens_detection is None:
            raise ValueError("lens_detection not found in pipeline result")

        # 2D 분석 실행
        result_2d, debug_info_2d = analyze_lens_zones_2d(
            img_bgr=img_bgr,
            lens_detection=lens_detection,
            sku_config=sku_config,
            ink_mask_config=InkMaskConfig(),
            save_debug=True,
            debug_prefix=str(run_dir / "debug_2d"),
        )

        # Zone 결과를 2D로 교체 (기존 /inspect와 동일)
        inspection_result.judgment = result_2d.judgment
        inspection_result.overall_delta_e = result_2d.overall_delta_e
        inspection_result.zone_results = result_2d.zone_results
        inspection_result.ng_reasons = result_2d.ng_reasons
        inspection_result.confidence = result_2d.confidence

        # 운영 UX 필드들도 복사 (P1 + Step 1 + 잉크 분석)
        inspection_result.decision_trace = result_2d.decision_trace
        inspection_result.next_actions = result_2d.next_actions
        inspection_result.retake_reasons = result_2d.retake_reasons
        inspection_result.analysis_summary = result_2d.analysis_summary
        inspection_result.confidence_breakdown = result_2d.confidence_breakdown
        inspection_result.risk_factors = result_2d.risk_factors
        inspection_result.ink_analysis = result_2d.ink_analysis  # 잉크 정보!

        logger.info(f"[INSPECT_V2] 2D analysis completed: {inspection_result.judgment}")

        # 3. Judgment 데이터 추출 (2D 결과 사용, 운영 UX 개선)
        judgment_result = {
            "result": inspection_result.judgment,
            "overall_delta_e": float(inspection_result.overall_delta_e),
            "confidence": float(inspection_result.confidence) if hasattr(inspection_result, "confidence") else 1.0,
            "zones": [
                {
                    "name": z.zone_name,
                    "delta_e": float(z.delta_e),
                    "threshold": float(z.threshold) if hasattr(z, "threshold") else 0.0,
                    "is_ok": z.is_ok,
                    # Diff 정보 추가 (운영 UX): 색상 변화 방향
                    "diff": z.diff,
                }
                for z in inspection_result.zone_results
            ],
            # 운영 UX: decision_trace, next_actions, retake_reasons 승격
            "decision_trace": (
                inspection_result.decision_trace if hasattr(inspection_result, "decision_trace") else None
            ),
            "next_actions": inspection_result.next_actions if hasattr(inspection_result, "next_actions") else None,
            "retake_reasons": (
                inspection_result.retake_reasons if hasattr(inspection_result, "retake_reasons") else None
            ),
            # 운영 UX: 프로파일 요약, Confidence 분해, 위험 요소
            "analysis_summary": (
                inspection_result.analysis_summary if hasattr(inspection_result, "analysis_summary") else None
            ),
            "confidence_breakdown": (
                inspection_result.confidence_breakdown if hasattr(inspection_result, "confidence_breakdown") else None
            ),
            "risk_factors": inspection_result.risk_factors if hasattr(inspection_result, "risk_factors") else None,
            # 사용자 목표: 잉크 수와 각 잉크 색 도출
            "ink_analysis": inspection_result.ink_analysis if hasattr(inspection_result, "ink_analysis") else None,
        }

        # 4. ProfileAnalyzer로 그래프 데이터 생성
        analyzer = ProfileAnalyzer()
        rp = inspection_result.radial_profile

        if rp is None:
            raise ValueError("Radial profile not available in inspection result")

        analysis_result = analyzer.analyze_profile(
            r_norm=rp.r_normalized,
            l_data=rp.L,
            a_data=rp.a,
            b_data=rp.b,
            smoothing_window=smoothing_window,
            gradient_threshold=gradient_threshold,
        )

        # 5. lens_info 추가 (overlay용)
        ld = inspection_result.lens_detection
        analysis_result["lens_info"] = {
            "center_x": float(ld.center_x),
            "center_y": float(ld.center_y),
            "radius_px": float(ld.radius),
            "confidence": float(ld.confidence),
        }

        # 6. debug 정보 추가 (2D zone analysis 디버그)
        if debug_info_2d is not None:
            analysis_result["debug"] = debug_info_2d
            logger.info(f"[INSPECT_V2] Added debug info: {list(debug_info_2d.keys())}")
        else:
            logger.warning("[INSPECT_V2] No debug_info_2d available")

        # 7. 응답 반환
        return {
            "run_id": run_id,
            "image": original_name,
            "sku": sku,
            "analysis": analysis_result,
            "judgment": judgment_result,
        }

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ================================
# PHASE7: Lot Comparison API (Priority 9)
# ================================


def _describe_shift(dL: float, da: float, db: float) -> str:
    """
    Describe color shift based on Lab delta values.

    Args:
        dL: Delta L* (lightness)
        da: Delta a* (red-green)
        db: Delta b* (blue-yellow)

    Returns:
        str: Human-readable description of the shift

    Examples:
        >>> _describe_shift(-5, 2, 3)
        'Darker and more red and more yellow'
        >>> _describe_shift(1, 0.5, -0.3)
        'No significant shift'
    """
    parts = []

    # Lightness shift
    if abs(dL) > 3:
        parts.append("Darker" if dL < 0 else "Brighter")

    # Red-green shift
    if abs(da) > 2:
        parts.append("more green" if da < 0 else "more red")

    # Blue-yellow shift
    if abs(db) > 2:
        parts.append("more blue" if db < 0 else "more yellow")

    return " and ".join(parts) if parts else "No significant shift"


def _calculate_stability_score(test_results: list) -> float:
    """
    Calculate stability score for batch consistency.

    Score ranges from 0 to 1:
    - 1.0 = All images identical to reference
    - 0.5 = Moderate variation
    - 0.0 = High variation (mean ΔE ≥ 10)

    Args:
        test_results: List of test result dicts with 'max_delta_e'

    Returns:
        float: Stability score (0~1)
    """
    if not test_results:
        return 1.0

    all_max_des = [t["max_delta_e"] for t in test_results]
    mean_de = np.mean(all_max_des)

    # Linear scale: 0 ΔE = 1.0, 10+ ΔE = 0.0
    return 1.0 - min(mean_de / 10.0, 1.0)


def _detect_outliers(test_results: list, threshold: float = 2.0) -> list:
    """
    Detect outlier images based on ΔE statistics.

    An image is considered an outlier if its max ΔE exceeds:
        mean(max_ΔE) + threshold * std(max_ΔE)

    Args:
        test_results: List of test result dicts
        threshold: Z-score threshold for outlier detection (default: 2.0)

    Returns:
        list: Filenames of outlier images
    """
    if len(test_results) < 3:
        return []  # Need at least 3 samples for meaningful outlier detection

    all_max_des = [t["max_delta_e"] for t in test_results]
    mean = np.mean(all_max_des)
    std = np.std(all_max_des)

    if std < 0.1:  # No variation
        return []

    outliers = []
    for t in test_results:
        if t["max_delta_e"] > mean + threshold * std:
            outliers.append(t["filename"])

    return outliers


def _resolve_ink_mapping(sku_config: dict, zone_names: list[str]) -> dict[str, str]:
    mapping = sku_config.get("params", {}).get("ink_mapping") or sku_config.get("ink_mapping") or {}
    if not mapping:
        return {name: "ink1" for name in zone_names}

    normalized = {str(k).upper(): str(v) for k, v in mapping.items()}
    return {name: normalized.get(name.upper(), "ink1") for name in zone_names}


def _summarize_ink_deltas(zone_deltas: list[dict], ink_mapping: dict[str, str]) -> dict[str, dict]:
    buckets: dict[str, dict[str, list]] = {}
    for item in zone_deltas:
        zone_name = item.get("zone")
        if not zone_name:
            continue
        ink_name = ink_mapping.get(zone_name, "ink1")
        bucket = buckets.setdefault(
            ink_name,
            {"zones": [], "delta_L": [], "delta_a": [], "delta_b": [], "delta_e": []},
        )
        bucket["zones"].append(zone_name)
        bucket["delta_L"].append(float(item.get("delta_L", 0.0)))
        bucket["delta_a"].append(float(item.get("delta_a", 0.0)))
        bucket["delta_b"].append(float(item.get("delta_b", 0.0)))
        bucket["delta_e"].append(float(item.get("delta_e", 0.0)))

    summary: dict[str, dict] = {}
    for ink_name, values in buckets.items():
        deltas = values["delta_e"]
        if not deltas:
            continue
        summary[ink_name] = {
            "zones": values["zones"],
            "mean_delta_e": float(np.mean(deltas)),
            "max_delta_e": float(np.max(deltas)),
            "mean_delta_L": float(np.mean(values["delta_L"])),
            "mean_delta_a": float(np.mean(values["delta_a"])),
            "mean_delta_b": float(np.mean(values["delta_b"])),
        }

    return summary


def _resolve_ink_thresholds(sku_config: dict, ink_names: list[str]) -> dict[str, dict]:
    thresholds = sku_config.get("params", {}).get("ink_thresholds") or {}
    default_threshold = sku_config.get("default_threshold", None)
    default_entry = thresholds.get("default")
    if default_entry is None and default_threshold is not None:
        default_entry = {"max_delta_e": float(default_threshold)}

    resolved: dict[str, dict] = {}
    for name in ink_names:
        entry = thresholds.get(name) or thresholds.get(name.lower()) or thresholds.get(name.upper())
        if entry is None and default_entry is not None:
            entry = dict(default_entry)
        if entry:
            resolved[name] = entry
    return resolved


def _resolve_quality_thresholds(sku_config: dict) -> dict[str, dict]:
    defaults = {
        "blur": {"delta_warn": -50.0},
        "histogram": {"lab_mean": 0.2, "hsv_mean": 0.2},
        "dot_stats": {"dot_count_delta": 50, "dot_coverage_delta": 0.05},
    }

    raw = sku_config.get("params", {}).get("quality_thresholds") or {}
    resolved = {}
    for key, default in defaults.items():
        entry = raw.get(key) or {}
        merged = dict(default)
        merged.update(entry)
        resolved[key] = merged
    return resolved


def _evaluate_ink_flags(ink_deltas: dict[str, dict], ink_thresholds: dict[str, dict]) -> list[dict]:
    flags: list[dict] = []
    for ink_name, deltas in ink_deltas.items():
        threshold = ink_thresholds.get(ink_name) or {}
        max_limit = threshold.get("max_delta_e")
        mean_limit = threshold.get("mean_delta_e")
        if max_limit is not None and deltas.get("max_delta_e", 0.0) > float(max_limit):
            flags.append(
                {
                    "ink": ink_name,
                    "metric": "max_delta_e",
                    "value": deltas.get("max_delta_e"),
                    "threshold": float(max_limit),
                }
            )
        if mean_limit is not None and deltas.get("mean_delta_e", 0.0) > float(mean_limit):
            flags.append(
                {
                    "ink": ink_name,
                    "metric": "mean_delta_e",
                    "value": deltas.get("mean_delta_e"),
                    "threshold": float(mean_limit),
                }
            )
    return flags


def _compare_quality_metrics(ref_metrics: Optional[dict], test_metrics: Optional[dict]) -> Optional[dict]:
    if not ref_metrics or not test_metrics:
        return None

    blur_delta = None
    ref_blur = (ref_metrics.get("blur") or {}).get("score")
    test_blur = (test_metrics.get("blur") or {}).get("score")
    if ref_blur is not None and test_blur is not None:
        blur_delta = float(test_blur) - float(ref_blur)

    hist_diff = _compare_histograms(ref_metrics.get("histogram"), test_metrics.get("histogram"))
    dot_delta = _compare_dot_stats(ref_metrics.get("dot_stats"), test_metrics.get("dot_stats"))

    return {
        "blur_delta": blur_delta,
        "hist_diff": hist_diff,
        "dot_delta": dot_delta,
    }


def _compare_histograms(ref_hist: Optional[dict], test_hist: Optional[dict]) -> Optional[dict]:
    if not ref_hist or not test_hist:
        return None

    if ref_hist.get("bins") != test_hist.get("bins"):
        return None

    def _l1(a: list, b: list) -> float:
        arr_a = np.asarray(a, dtype=np.float32)
        arr_b = np.asarray(b, dtype=np.float32)
        if arr_a.shape != arr_b.shape:
            return float("nan")
        return float(np.sum(np.abs(arr_a - arr_b)))

    def _space_diff(space: str, channels: list[str]) -> dict:
        diffs = {}
        values = []
        for ch in channels:
            a = (ref_hist.get(space) or {}).get(ch)
            b = (test_hist.get(space) or {}).get(ch)
            if a is None or b is None:
                continue
            diff = _l1(a, b)
            diffs[ch] = diff
            values.append(diff)
        diffs["mean"] = float(np.mean(values)) if values else None
        return diffs

    return {
        "method": "l1",
        "lab": _space_diff("lab", ["L", "a", "b"]),
        "hsv": _space_diff("hsv", ["H", "S", "V"]),
    }


def _compare_dot_stats(ref_stats: Optional[dict], test_stats: Optional[dict]) -> Optional[dict]:
    if not ref_stats or not test_stats:
        return None

    keys = ["dot_count", "dot_coverage", "dot_area_mean", "dot_area_std"]
    deltas = {}
    for key in keys:
        if ref_stats.get(key) is None or test_stats.get(key) is None:
            continue
        deltas[key] = float(test_stats[key]) - float(ref_stats[key])

    return deltas if deltas else None


def _lens_info_from_detection(lens_detection: Optional[Any]) -> Optional[dict]:
    if lens_detection is None:
        return None

    return {
        "center_x": float(lens_detection.center_x),
        "center_y": float(lens_detection.center_y),
        "radius": float(lens_detection.radius),
        "confidence": float(getattr(lens_detection, "confidence", 0.0)),
    }


def _center_offset_info(image_bgr: Optional[np.ndarray], lens_detection: Optional[Any]) -> Optional[dict]:
    if image_bgr is None or lens_detection is None:
        return None

    img_h, img_w = image_bgr.shape[:2]
    img_center_x = img_w / 2.0
    img_center_y = img_h / 2.0
    dx = float(lens_detection.center_x) - img_center_x
    dy = float(lens_detection.center_y) - img_center_y
    offset_px = float(np.hypot(dx, dy))
    radius = float(lens_detection.radius) if lens_detection.radius else 0.0
    offset_ratio = offset_px / radius if radius > 0 else None

    return {
        "image_width": int(img_w),
        "image_height": int(img_h),
        "image_center_x": img_center_x,
        "image_center_y": img_center_y,
        "offset_x": dx,
        "offset_y": dy,
        "offset_px": offset_px,
        "offset_ratio": offset_ratio,
    }


def _compute_cluster_results(
    image_bgr: Optional[np.ndarray],
    lens_detection: Optional[Any],
    sku_config: dict,
    mask_source: str = "sample",
    std_ref_image: Optional[np.ndarray] = None,
    std_ref_lens: Optional[Any] = None,
) -> Optional[dict]:
    if image_bgr is None or lens_detection is None:
        return None

    from itertools import permutations

    from src.core.zone_analyzer_2d import InkMaskConfig, bgr_to_lab_float, build_ink_mask, circle_mask
    from src.utils.color_delta import delta_e_cie2000

    params = sku_config.get("params", {})
    ink_profile = params.get("ink_profile", {})
    if ink_profile.get("mode") != "cluster":
        return None

    k = int(ink_profile.get("k", 0))
    if k <= 0:
        return None

    h, w = image_bgr.shape[:2]
    cx = float(lens_detection.center_x)
    cy = float(lens_detection.center_y)
    radius = float(lens_detection.radius)

    lens_mask = circle_mask((h, w), cx, cy, radius)

    if mask_source == "std_warped":
        ink_mask = _warp_std_ink_mask_to_test(
            std_ref_image,
            std_ref_lens,
            image_bgr,
            lens_detection,
            sku_config,
        )
        if ink_mask is None:
            mask_source = "sample_fallback"
            ink_mask = build_ink_mask(image_bgr, lens_mask, InkMaskConfig())
    else:
        ink_mask = build_ink_mask(image_bgr, lens_mask, InkMaskConfig())
        if mask_source == "std":
            ink_mask = _buffer_mask(ink_mask, sku_config)

    ink_mask = cv2.bitwise_and(ink_mask, ink_mask, mask=lens_mask)

    optical_clear_ratio = params.get("optical_clear_ratio", 0.15)
    center_exclude_ratio = params.get("center_exclude_ratio", 0.0)
    r_inner = max(0.0, optical_clear_ratio, center_exclude_ratio)
    r_outer = 0.95

    yy, xx = np.indices((h, w))
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    valid = (rr >= radius * r_inner) & (rr <= radius * r_outer) & (lens_mask > 0)
    mask = valid & (ink_mask > 0)

    if np.sum(mask) < max(100, k * 50):
        return {"clusters": [], "match": [], "match_confidence": None}

    lab = bgr_to_lab_float(image_bgr)
    ab = lab[:, :, 1:3]
    samples = ab[mask].astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    flags = cv2.KMEANS_PP_CENTERS
    compactness, labels, centers = cv2.kmeans(samples, k, None, criteria, 5, flags)

    labels = labels.flatten()
    total_pixels = int(samples.shape[0])

    clusters = []
    for idx in range(k):
        idx_mask = labels == idx
        count = int(np.sum(idx_mask))
        if count == 0:
            continue
        mean_lab = lab[mask][idx_mask].mean(axis=0)
        clusters.append(
            {
                "cluster_id": int(idx),
                "pixels": count,
                "coverage": float(count / total_pixels) if total_pixels > 0 else 0.0,
                "mean_lab": [float(v) for v in mean_lab],
                "mean_ab": [float(mean_lab[1]), float(mean_lab[2])],
            }
        )

    return {
        "clusters": clusters,
        "algorithm": "kmeans",
        "mask_source": mask_source,
        "match": [],
        "match_confidence": None,
    }


def _warp_std_ink_mask_to_test(
    ref_image: Optional[np.ndarray],
    ref_lens: Optional[Any],
    test_image: Optional[np.ndarray],
    test_lens: Optional[Any],
    sku_config: dict,
) -> Optional[np.ndarray]:
    if ref_image is None or ref_lens is None or test_image is None or test_lens is None:
        return None

    from src.core.zone_analyzer_2d import InkMaskConfig, build_ink_mask, circle_mask

    ref_h, ref_w = ref_image.shape[:2]
    ref_cx = float(ref_lens.center_x)
    ref_cy = float(ref_lens.center_y)
    ref_radius = float(ref_lens.radius)
    ref_lens_mask = circle_mask((ref_h, ref_w), ref_cx, ref_cy, ref_radius)
    ref_ink_mask = build_ink_mask(ref_image, ref_lens_mask, InkMaskConfig())
    ref_ink_mask = _buffer_mask(ref_ink_mask, sku_config)
    ref_ink_mask = cv2.bitwise_and(ref_ink_mask, ref_ink_mask, mask=ref_lens_mask)

    test_h, test_w = test_image.shape[:2]
    test_cx = float(test_lens.center_x)
    test_cy = float(test_lens.center_y)
    test_radius = float(test_lens.radius)

    shared_radius = min(ref_radius, test_radius)
    r_bins = int(max(64, min(256, shared_radius)))
    theta_bins = 360

    polar = cv2.warpPolar(
        ref_ink_mask,
        (r_bins, theta_bins),
        (ref_cx, ref_cy),
        ref_radius,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS,
    )

    warped = cv2.warpPolar(
        polar,
        (test_w, test_h),
        (test_cx, test_cy),
        test_radius,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP,
    )

    return (warped > 0).astype(np.uint8)


def _match_cluster_sets(ref_clusters: list, test_clusters: list) -> dict:
    if not ref_clusters or not test_clusters:
        return {"matches": [], "match_confidence": None}

    from itertools import permutations

    from src.utils.color_delta import delta_e_cie2000

    k = min(len(ref_clusters), len(test_clusters))
    ref_subset = ref_clusters[:k]
    test_subset = test_clusters[:k]

    cost_matrix = []
    for r in ref_subset:
        row = []
        for t in test_subset:
            row.append(delta_e_cie2000(r["mean_lab"], t["mean_lab"]))
        cost_matrix.append(row)

    best = None
    second = None
    best_perm = None

    for perm in permutations(range(k)):
        cost = sum(cost_matrix[i][perm[i]] for i in range(k))
        if best is None or cost < best:
            second = best
            best = cost
            best_perm = perm
        elif second is None or cost < second:
            second = cost

    matches = []
    if best_perm is not None:
        for i, j in enumerate(best_perm):
            matches.append(
                {
                    "ref_cluster_id": ref_subset[i].get("cluster_id"),
                    "ref_std_id": ref_subset[i].get("std_cluster_id"),
                    "ref_role": ref_subset[i].get("role"),
                    "test_cluster_id": test_subset[j].get("cluster_id"),
                    "delta_e": float(cost_matrix[i][j]),
                }
            )

    match_conf = None
    if best is not None and second is not None:
        match_conf = float(second - best)

    return {"matches": matches, "match_confidence": match_conf}


def _compute_sample_outside_std_ratio(
    ref_image: Optional[np.ndarray],
    ref_lens: Optional[Any],
    test_image: Optional[np.ndarray],
    test_lens: Optional[Any],
    sku_config: Optional[dict] = None,
) -> Optional[float]:
    if ref_image is None or ref_lens is None or test_image is None or test_lens is None:
        return None

    from src.core.zone_analyzer_2d import InkMaskConfig, build_ink_mask, circle_mask

    shared_radius = min(float(ref_lens.radius), float(test_lens.radius))
    r_bins = int(max(64, min(256, shared_radius)))
    theta_bins = 360

    def _polar_mask(img_bgr, lens, buffer_std: bool):
        h, w = img_bgr.shape[:2]
        cx = float(lens.center_x)
        cy = float(lens.center_y)
        radius = float(lens.radius)
        lens_mask = circle_mask((h, w), cx, cy, radius)
        ink_mask = build_ink_mask(img_bgr, lens_mask, InkMaskConfig())
        if buffer_std and sku_config:
            ink_mask = _buffer_mask(ink_mask, sku_config)
        ink_mask = cv2.bitwise_and(ink_mask, ink_mask, mask=lens_mask)
        polar = cv2.warpPolar(
            ink_mask,
            (r_bins, theta_bins),
            (cx, cy),
            radius,
            cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS,
        )
        return (polar > 0).astype(np.uint8)

    ref_polar = _polar_mask(ref_image, ref_lens, True)
    test_polar = _polar_mask(test_image, test_lens, False)

    test_total = int(test_polar.sum())
    if test_total == 0:
        return 0.0

    outside = int(((test_polar > 0) & (ref_polar == 0)).sum())
    return float(outside / test_total)


def _assign_roles_to_clusters(clusters: Optional[dict], std_clusters: Optional[list]) -> Optional[dict]:
    if not clusters or not std_clusters:
        return clusters

    from itertools import permutations

    cluster_list = clusters.get("clusters", [])
    if not cluster_list:
        return clusters

    std_defs = []
    for item in std_clusters:
        sig = item.get("signature") or {}
        mean_ab = sig.get("mean_ab")
        if not mean_ab or len(mean_ab) != 2:
            continue
        std_defs.append(
            {
                "id": item.get("id"),
                "role": item.get("role"),
                "mean_ab": [float(mean_ab[0]), float(mean_ab[1])],
            }
        )

    if not std_defs:
        return clusters

    k = min(len(std_defs), len(cluster_list))
    std_subset = std_defs[:k]
    cl_subset = cluster_list[:k]

    cost_matrix = []
    for s in std_subset:
        row = []
        for c in cl_subset:
            cab = c.get("mean_ab")
            if not cab or len(cab) != 2:
                row.append(float("inf"))
                continue
            da = cab[0] - s["mean_ab"][0]
            db = cab[1] - s["mean_ab"][1]
            row.append(float(np.hypot(da, db)))
        cost_matrix.append(row)

    best = None
    best_perm = None
    for perm in permutations(range(k)):
        cost = sum(cost_matrix[i][perm[i]] for i in range(k))
        if best is None or cost < best:
            best = cost
            best_perm = perm

    if best_perm is None:
        return clusters

    for i, j in enumerate(best_perm):
        cl_subset[j]["std_cluster_id"] = std_subset[i]["id"]
        cl_subset[j]["role"] = std_subset[i]["role"]

    return clusters


def _compute_zone_diagnostics(
    radial_profile: Optional[Any],
    sku_config: dict,
    expected_zones: Optional[int],
) -> Optional[dict]:
    if radial_profile is None:
        return None

    from src.core.zone_segmenter import SegmenterConfig, ZoneSegmenter

    params = sku_config.get("params", {})
    optical_clear_ratio = params.get("optical_clear_ratio", 0.15)
    center_exclude_ratio = params.get("center_exclude_ratio", 0.0)
    r_inner = max(0.0, optical_clear_ratio, center_exclude_ratio)
    r_outer = 0.95

    segmenter = ZoneSegmenter(SegmenterConfig())
    return segmenter.diagnostics(
        radial_profile,
        expected_zones=expected_zones,
        r_inner=r_inner,
        r_outer=r_outer,
    )


def _compute_zone_ink_ratios(
    image_bgr: Optional[np.ndarray],
    lens_detection: Optional[Any],
    zones: list,
    sku_config: dict,
) -> Optional[dict]:
    if image_bgr is None or lens_detection is None or not zones:
        return None

    from src.core.zone_analyzer_2d import (
        InkMaskConfig,
        ZoneSpec,
        build_ink_mask,
        build_zone_masks_from_printband,
        circle_mask,
    )

    h, w = image_bgr.shape[:2]
    cx = float(lens_detection.center_x)
    cy = float(lens_detection.center_y)
    radius = float(lens_detection.radius)
    lens_mask = circle_mask((h, w), cx, cy, radius)

    ink_mask = build_ink_mask(image_bgr, lens_mask, InkMaskConfig())
    ink_mask = _buffer_mask(ink_mask, sku_config)
    ink_mask = cv2.bitwise_and(ink_mask, ink_mask, mask=lens_mask)

    params = sku_config.get("params", {})
    optical_clear_ratio = params.get("optical_clear_ratio", 0.15)
    center_exclude_ratio = params.get("center_exclude_ratio", 0.0)
    r_inner = max(0.0, optical_clear_ratio, center_exclude_ratio)
    r_outer = 0.95

    print_inner = radius * r_inner
    print_outer = radius * r_outer

    zone_specs = [ZoneSpec(name=z.name, r_start_norm=z.r_start, r_end_norm=z.r_end) for z in zones]
    zone_masks = build_zone_masks_from_printband(h, w, cx, cy, print_inner, print_outer, lens_mask, zone_specs)

    ratios = {}
    for name, zmask in zone_masks.items():
        z = (zmask > 0) & (lens_mask > 0)
        total_pixels = int(np.sum(z))
        ink_pixels = int(np.sum((ink_mask > 0) & z))
        ink_ratio = (ink_pixels / total_pixels) if total_pixels > 0 else 0.0
        ratios[name] = {
            "ink_pixels": ink_pixels,
            "total_pixels": total_pixels,
            "ink_ratio": ink_ratio,
        }

    return ratios


def _buffer_mask(mask: np.ndarray, sku_config: dict) -> np.ndarray:
    params = sku_config.get("params", {})
    mask_policy = params.get("mask_policy", {})
    buffer_iter = int(mask_policy.get("std_mask_buffer_iter", 1))
    if buffer_iter <= 0:
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(mask, kernel, iterations=buffer_iter)


def _zone_stats_from_profile(profile: Optional[Any], r_start: float, r_end: float) -> Optional[dict]:
    if profile is None:
        return None

    r0, r1 = sorted([r_start, r_end])
    mask = (profile.r_normalized >= r0) & (profile.r_normalized < r1)
    if np.sum(mask) == 0:
        return None

    mean_L = float(np.mean(profile.L[mask]))
    mean_a = float(np.mean(profile.a[mask]))
    mean_b = float(np.mean(profile.b[mask]))
    return {
        "mean_L": mean_L,
        "mean_a": mean_a,
        "mean_b": mean_b,
    }


def _build_compare_overlay(
    image_bgr: Optional[np.ndarray],
    lens_detection: Optional[Any],
    zones: list,
    zone_deltas: list[dict],
    sku_config: dict,
) -> Optional[np.ndarray]:
    if image_bgr is None or lens_detection is None or not zones or not zone_deltas:
        return None

    delta_map = {z.get("zone"): z for z in zone_deltas if z.get("zone")}
    default_threshold = float(sku_config.get("default_threshold", 0.0))

    class _OverlayZoneResult:
        def __init__(self, zone_name: str, delta_e: float, threshold: float) -> None:
            self.zone_name = zone_name
            self.delta_e = float(delta_e)
            self.threshold = float(threshold)
            self.is_ok = self.delta_e <= self.threshold if self.threshold > 0 else True

    zone_results = []
    for zone in zones:
        delta = delta_map.get(zone.name)
        if not delta:
            continue
        cfg = (sku_config.get("zones") or {}).get(zone.name) or {}
        threshold = cfg.get("delta_e_threshold", cfg.get("threshold", default_threshold))
        zone_results.append(_OverlayZoneResult(zone.name, delta.get("delta_e", 0.0), threshold))

    if not zone_results:
        return None

    class _OverlayResult:
        def __init__(self, zone_results_list: list[_OverlayZoneResult]) -> None:
            self.zone_results = zone_results_list
            self.overall_delta_e = float(np.mean([zr.delta_e for zr in zone_results_list]))
            self.judgment = "OK" if all(zr.is_ok for zr in zone_results_list) else "NG"

    overlay_result = _OverlayResult(zone_results)
    visualizer = InspectionVisualizer(VisualizerConfig())
    overlay = visualizer.visualize_zone_overlay(image_bgr, lens_detection, zones, overlay_result, show_result=False)

    img_h, img_w = overlay.shape[:2]
    img_center = (int(img_w / 2), int(img_h / 2))
    cv2.drawMarker(
        overlay,
        img_center,
        (255, 255, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=2,
        line_type=cv2.LINE_AA,
    )

    center_exclude_ratio = float(sku_config.get("params", {}).get("center_exclude_ratio", 0.0))
    if center_exclude_ratio > 0:
        center = (int(lens_detection.center_x), int(lens_detection.center_y))
        radius = int(lens_detection.radius * center_exclude_ratio)
        if radius > 0:
            overlay_mask = overlay.copy()
            cv2.circle(overlay_mask, center, radius, (80, 80, 80), -1, cv2.LINE_AA)
            overlay = cv2.addWeighted(overlay_mask, 0.35, overlay, 0.65, 0)

    return overlay


@app.post("/compare")
async def compare_lots(
    reference_file: UploadFile = File(...),
    test_files: list[UploadFile] = File(...),
    sku: str = Form(...),
):
    """
    PHASE7: Compare reference image against multiple test images (Lot comparison).

    This endpoint performs batch analysis to detect color drift and consistency
    issues across multiple lenses from the same lot.

    Args:
        reference_file: Reference image (golden sample)
        test_files: List of test images to compare against reference
        sku: SKU identifier

    Returns:
        dict: Comparison results with zone-level deltas and batch summary

    Example Response:
        {
            "reference": {
                "filename": "ref.jpg",
                "zones": [...]
            },
            "tests": [
                {
                    "filename": "lot_002_001.jpg",
                    "zone_deltas": [
                        {
                            "zone": "A",
                            "delta_L": -2.3,
                            "delta_a": 0.5,
                            "delta_b": 1.2,
                            "delta_e": 2.7
                        }
                    ],
                    "overall_shift": "Darker and more yellow",
                    "max_delta_e": 3.5
                }
            ],
            "batch_summary": {
                "mean_delta_e_per_zone": {"A": 2.3, "B": 1.8},
                "max_delta_e_per_zone": {"A": 4.5, "B": 3.2},
                "std_delta_e_per_zone": {"A": 0.8, "B": 0.5},
                "stability_score": 0.82,
                "outliers": ["lot_002_005.jpg"]
            }
        }
    """
    import cv2

    from src.utils.file_io import read_json
    from src.utils.security import validate_file_extension, validate_file_size

    # Load SKU config
    sku_config = load_sku_config(sku)

    # 1. Process reference image
    logger.info(f"[COMPARE] Processing reference: {reference_file.filename}")

    if not validate_file_extension(reference_file.filename, [".jpg", ".jpeg", ".png", ".bmp"]):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {reference_file.filename}")

    ref_content = await reference_file.read()
    if not validate_file_size(len(ref_content), max_size_mb=10):
        raise HTTPException(status_code=413, detail=f"File too large: {reference_file.filename}")
    ref_run_id = uuid.uuid4().hex[:8]
    ref_run_dir = RESULTS_DIR / f"ref_{ref_run_id}"
    ref_run_dir.mkdir(parents=True, exist_ok=True)

    ref_path = ref_run_dir / f"reference{Path(reference_file.filename).suffix}"
    ref_path.write_bytes(ref_content)

    # Run pipeline on reference
    ref_pipeline = InspectionPipeline(sku_config, save_intermediates=False)
    try:
        ref_result = ref_pipeline.process(str(ref_path), sku, save_dir=ref_run_dir)
    except PipelineError as e:
        raise HTTPException(status_code=400, detail=f"Reference image processing failed: {str(e)}")

    # Get reference zones
    ref_zones = ref_result.zones
    if not ref_zones:
        raise HTTPException(status_code=400, detail="Reference image has no zones detected")

    logger.info(f"[COMPARE] Reference has {len(ref_zones)} zones")
    ref_metrics = getattr(ref_result, "metrics", None)
    params = sku_config.get("params", {})
    version_stamp = params.get("version_stamp")
    expected_zones = params.get("expected_zones")
    ref_ink_ratios = _compute_zone_ink_ratios(
        ref_result.image, getattr(ref_result, "lens_detection", None), ref_zones, sku_config
    )
    ref_clusters = _compute_cluster_results(
        ref_result.image,
        getattr(ref_result, "lens_detection", None),
        sku_config,
        mask_source="std",
    )
    std_clusters = (params.get("ink_profile") or {}).get("std_clusters")
    if std_clusters:
        ref_clusters = _assign_roles_to_clusters(ref_clusters, std_clusters)
    ink_mapping = _resolve_ink_mapping(sku_config, [z.name for z in ref_zones])
    ink_thresholds = _resolve_ink_thresholds(sku_config, list(set(ink_mapping.values())))
    quality_thresholds = _resolve_quality_thresholds(sku_config)
    mask_policy = params.get("mask_policy", {})
    std_ink_gate_min_ratio = float(mask_policy.get("min_ink_ratio_std", params.get("std_ink_gate_min_ratio", 0.15)))
    effective_min_ratio = float(mask_policy.get("min_ink_ratio_sample", params.get("effective_min_ratio", 0.05)))
    outside_warn = float(mask_policy.get("sample_outside_std_ratio_warn", 0.0))
    outside_fail = float(mask_policy.get("sample_outside_std_ratio_fail", 0.0))
    alignment_policy = params.get("alignment_policy", {})
    warn_offset = float(alignment_policy.get("warn_center_offset_ratio", 0.0))
    stop_offset = float(alignment_policy.get("stop_center_offset_ratio", 0.0))
    require_ok = bool(alignment_policy.get("require_ok", False))

    # 2. Process test images
    test_results = []
    all_deltas_per_zone = {zone.name: [] for zone in ref_zones}

    for test_file in test_files:
        logger.info(f"[COMPARE] Processing test: {test_file.filename}")

        if not validate_file_extension(test_file.filename, [".jpg", ".jpeg", ".png", ".bmp"]):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {test_file.filename}")

        test_content = await test_file.read()
        if not validate_file_size(len(test_content), max_size_mb=10):
            raise HTTPException(status_code=413, detail=f"File too large: {test_file.filename}")
        test_run_id = uuid.uuid4().hex[:8]
        test_run_dir = RESULTS_DIR / f"test_{test_run_id}"
        test_run_dir.mkdir(parents=True, exist_ok=True)

        test_path = test_run_dir / f"test{Path(test_file.filename).suffix}"
        test_path.write_bytes(test_content)

        # Run pipeline on test image
        test_pipeline = InspectionPipeline(sku_config, save_intermediates=False)
        try:
            test_result = test_pipeline.process(str(test_path), sku, save_dir=test_run_dir)
        except PipelineError as e:
            logger.warning(f"[COMPARE] Test image {test_file.filename} failed: {e}")
            continue

        test_metrics = getattr(test_result, "metrics", None)
        test_diagnostics = _compute_zone_diagnostics(
            getattr(test_result, "radial_profile", None), sku_config, expected_zones
        )
        sample_outside_std_ratio = _compute_sample_outside_std_ratio(
            ref_result.image,
            getattr(ref_result, "lens_detection", None),
            getattr(test_result, "image", None),
            getattr(test_result, "lens_detection", None),
            sku_config,
        )
        test_mask_source = (params.get("ink_profile") or {}).get("test_mask_source", "sample")
        test_clusters = _compute_cluster_results(
            getattr(test_result, "image", None),
            getattr(test_result, "lens_detection", None),
            sku_config,
            mask_source=test_mask_source,
            std_ref_image=ref_result.image,
            std_ref_lens=getattr(ref_result, "lens_detection", None),
        )
        cluster_match = _match_cluster_sets(
            (ref_clusters or {}).get("clusters", []),
            (test_clusters or {}).get("clusters", []),
        )
        quality_deltas = _compare_quality_metrics(ref_metrics, test_metrics)
        test_ink_ratios = _compute_zone_ink_ratios(
            getattr(test_result, "image", None),
            getattr(test_result, "lens_detection", None),
            test_result.zones or [],
            sku_config,
        )
        center_offset = _center_offset_info(
            getattr(test_result, "image", None), getattr(test_result, "lens_detection", None)
        )
        alignment_status = "ok"
        if center_offset and center_offset.get("offset_ratio") is not None:
            ratio = float(center_offset["offset_ratio"])
            if stop_offset and ratio >= stop_offset:
                alignment_status = "stop"
            elif warn_offset and ratio >= warn_offset:
                alignment_status = "warn"

        # Calculate zone deltas
        zone_deltas = []
        max_de = 0.0
        delta_Ls = []
        delta_as = []
        delta_bs = []
        zone_gate = {}
        excluded_zones = []

        test_zones = test_result.zones
        test_profile = getattr(test_result, "radial_profile", None)
        ref_profile = getattr(ref_result, "radial_profile", None)

        # 4.6. Compare radial profiles (P1-2)
        profile_details = None
        if test_profile and ref_profile:
            from src.core.profile_comparison import compare_radial_profiles

            # Convert Profile objects to dict if they aren't already
            def _to_dict(p):
                if hasattr(p, "L"):
                    return {"L": p.L.tolist(), "a": p.a.tolist(), "b": p.b.tolist()}
                return p

            profile_details = compare_radial_profiles(
                test_profile=_to_dict(test_profile),
                std_profile=_to_dict(ref_profile),
            )
            logger.info(f"[COMPARE] Profile similarity: {profile_details.get('profile_score', 0.0):.1f}")

        if not (alignment_status == "stop" and require_ok):
            # Match zones by name
            for ref_zone in ref_zones:
                # Recompute test zone stats using reference boundaries when possible
                test_zone = next((z for z in test_zones if z.name == ref_zone.name), None)
                test_stats = _zone_stats_from_profile(test_profile, ref_zone.r_start, ref_zone.r_end)

                if test_zone is None:
                    logger.warning(f"[COMPARE] Zone {ref_zone.name} not found in test image {test_file.filename}")
                    continue

                ref_ratio = None
                test_ratio = None
                if ref_ink_ratios and ref_zone.name in ref_ink_ratios:
                    ref_ratio = ref_ink_ratios[ref_zone.name]["ink_ratio"]
                if test_ink_ratios and test_zone and test_zone.name in test_ink_ratios:
                    test_ratio = test_ink_ratios[test_zone.name]["ink_ratio"]

                include = True
                reason = None
                if ref_ratio is not None and ref_ratio < std_ink_gate_min_ratio:
                    include = False
                    reason = "std_ink_ratio_below_threshold"
                elif test_ratio is not None and test_ratio < effective_min_ratio:
                    include = False
                    reason = "test_ink_ratio_below_threshold"

                zone_gate[ref_zone.name] = {
                    "ref_ink_ratio": ref_ratio,
                    "test_ink_ratio": test_ratio,
                    "included": include,
                    "reason": reason,
                }

                if not include:
                    excluded_zones.append(ref_zone.name)
                    continue

                # Calculate deltas (reference boundaries preferred)
                if test_stats is not None:
                    delta_L = test_stats["mean_L"] - ref_zone.mean_L
                    delta_a = test_stats["mean_a"] - ref_zone.mean_a
                    delta_b = test_stats["mean_b"] - ref_zone.mean_b
                else:
                    delta_L = test_zone.mean_L - ref_zone.mean_L
                    delta_a = test_zone.mean_a - ref_zone.mean_a
                    delta_b = test_zone.mean_b - ref_zone.mean_b
                delta_e = np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)

                max_de = max(max_de, delta_e)

                delta_Ls.append(delta_L)
                delta_as.append(delta_a)
                delta_bs.append(delta_b)

                zone_deltas.append(
                    {
                        "zone": ref_zone.name,
                        "delta_L": round(float(delta_L), 2),
                        "delta_a": round(float(delta_a), 2),
                        "delta_b": round(float(delta_b), 2),
                        "delta_e": round(float(delta_e), 2),
                    }
                )

                all_deltas_per_zone[ref_zone.name].append(delta_e)

        # Calculate overall shift description
        avg_dL = np.mean(delta_Ls) if delta_Ls else 0.0
        avg_da = np.mean(delta_as) if delta_as else 0.0
        avg_db = np.mean(delta_bs) if delta_bs else 0.0
        overall_shift = _describe_shift(avg_dL, avg_da, avg_db)
        if alignment_status == "stop" and require_ok:
            overall_shift = "alignment_stop"

        diagnostic_flags = []
        decision_blocked = False
        decision_block_reasons = []
        if alignment_status == "stop":
            diagnostic_flags.append("alignment_stop")
        elif alignment_status == "warn":
            diagnostic_flags.append("alignment_warn")

        if alignment_status == "stop":
            decision_blocked = True
            decision_block_reasons.append("alignment_stop")

        if sample_outside_std_ratio is not None and outside_fail > 0:
            if sample_outside_std_ratio >= outside_fail:
                diagnostic_flags.append("outside_std_ratio_fail")
                decision_blocked = True
                decision_block_reasons.append("outside_std_ratio_fail")

        ink_deltas = _summarize_ink_deltas(zone_deltas, ink_mapping)
        ink_flags = _evaluate_ink_flags(ink_deltas, ink_thresholds)
        cluster_max_de = max((m.get("delta_e", 0.0) for m in (cluster_match.get("matches", []) or [])), default=0.0)
        if cluster_max_de > 0:
            max_de = cluster_max_de

        overlay_url = None
        overlay = _build_compare_overlay(
            image_bgr=test_result.image,
            lens_detection=getattr(test_result, "lens_detection", None),
            zones=test_result.zones or [],
            zone_deltas=zone_deltas,
            sku_config=sku_config,
        )
        if overlay is not None:
            overlay_path = test_run_dir / "overlay_compare.png"
            InspectionVisualizer(VisualizerConfig()).save_visualization(overlay, overlay_path)
            overlay_url = f"/results/test_{test_run_id}/overlay_compare.png?v={int(time())}"

        test_results.append(
            {
                "filename": test_file.filename,
                "zone_deltas": zone_deltas,
                "ink_deltas": ink_deltas,
                "ink_flags": ink_flags,
                "zone_diagnostics": test_diagnostics,
                "zone_gate": zone_gate,
                "excluded_zones": excluded_zones,
                "decision_blocked": decision_blocked,
                "decision_block_reasons": decision_block_reasons,
                "alignment": {
                    "status": alignment_status,
                    "offset_ratio": (center_offset or {}).get("offset_ratio"),
                    "warn_threshold": warn_offset,
                    "stop_threshold": stop_offset,
                },
                "sample_outside_std_ratio": sample_outside_std_ratio,
                "cluster_results": test_clusters,
                "cluster_match": cluster_match.get("matches", []),
                "match_confidence": cluster_match.get("match_confidence"),
                "diagnostic_flags": diagnostic_flags,
                "metrics": test_metrics,
                "comparison": quality_deltas,
                "overall_shift": overall_shift,
                "max_delta_e": round(float(max_de), 2),
                "profile_score": profile_details.get("profile_score", 0.0) if profile_details else 0.0,
                "profile_details": profile_details,
                "radial_profile": _to_dict(test_profile) if test_profile else None,
                "overlay": overlay_url,
                "lens_info": _lens_info_from_detection(getattr(test_result, "lens_detection", None)),
                "center_offset": center_offset,
            }
        )

    # 3. Calculate batch summary
    batch_summary = {
        "mean_delta_e_per_zone": {
            zone: round(float(np.mean(deltas)), 2) if deltas else 0.0 for zone, deltas in all_deltas_per_zone.items()
        },
        "max_delta_e_per_zone": {
            zone: round(float(np.max(deltas)), 2) if deltas else 0.0 for zone, deltas in all_deltas_per_zone.items()
        },
        "std_delta_e_per_zone": {
            zone: round(float(np.std(deltas)), 2) if deltas else 0.0 for zone, deltas in all_deltas_per_zone.items()
        },
        "stability_score": round(_calculate_stability_score(test_results), 3),
        "outliers": _detect_outliers(test_results),
    }

    # 4. Build response
    response = {
        "reference": {
            "filename": reference_file.filename,
            "metrics": ref_metrics,
            "lens_info": _lens_info_from_detection(getattr(ref_result, "lens_detection", None)),
            "center_offset": _center_offset_info(
                getattr(ref_result, "image", None), getattr(ref_result, "lens_detection", None)
            ),
            "zone_diagnostics": _compute_zone_diagnostics(
                getattr(ref_result, "radial_profile", None), sku_config, expected_zones
            ),
            "radial_profile": (
                _to_dict(ref_result.radial_profile) if getattr(ref_result, "radial_profile", None) else None
            ),
            "zone_ink_ratios": ref_ink_ratios,
            "cluster_results": ref_clusters,
            "zones": [
                {
                    "name": z.name,
                    "mean_L": round(float(z.mean_L), 2),
                    "mean_a": round(float(z.mean_a), 2),
                    "mean_b": round(float(z.mean_b), 2),
                }
                for z in ref_zones
            ],
        },
        "tests": test_results,
        "batch_summary": batch_summary,
        "test_count": len(test_results),
        "ink_mapping": ink_mapping,
        "ink_thresholds": ink_thresholds,
        "quality_thresholds": quality_thresholds,
        "gate_thresholds": {
            "std_ink_gate_min_ratio": std_ink_gate_min_ratio,
            "effective_min_ratio": effective_min_ratio,
            "sample_outside_std_ratio_warn": outside_warn,
            "sample_outside_std_ratio_fail": outside_fail,
            "alignment_warn_offset_ratio": warn_offset,
            "alignment_stop_offset_ratio": stop_offset,
        },
        "decision_policy": params.get("decision_policy"),
        "alignment_policy": alignment_policy,
        "version_stamp": version_stamp,
    }

    logger.info(
        f"[COMPARE] Completed: {len(test_results)}/{len(test_files)} images, "
        f"stability={batch_summary['stability_score']:.2f}"
    )

    return response
