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

# Mount v7 results for visualization artifacts
V7_RESULTS_DIR = BASE_DIR.parent / "results" / "v7" / "web"
V7_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/v7_results", StaticFiles(directory=str(V7_RESULTS_DIR)), name="v7_results")

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
from src.web.routers import inspection, sku, std, test, v7

app.include_router(std.router)
app.include_router(test.router)
app.include_router(sku.router)
app.include_router(inspection.router)
app.include_router(v7.router)


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


@app.get("/v7", response_class=HTMLResponse)
async def v7_page(request: Request):
    """Lens Signature Engine v7 MVP page"""
    return templates.TemplateResponse("v7_mvp.html", {"request": request})


@app.get("/demo_3d", response_class=HTMLResponse)
async def demo_3d_viz(request: Request):
    """3D Visualization Demo - Showcase Plotly.js capabilities"""
    return templates.TemplateResponse("demo_3d_viz.html", {"request": request})


@app.get("/demo_ui_improvements", response_class=HTMLResponse)
async def demo_ui_improvements(request: Request):
    """UI Improvements Demo - 8 key enhancements"""
    return templates.TemplateResponse("demo_ui_improvements.html", {"request": request})


@app.get("/demo_heatmap", response_class=HTMLResponse)
async def demo_heatmap(request: Request):
    """Interactive Heatmap Demo - Canvas-based drill-down"""
    return templates.TemplateResponse("demo_heatmap.html", {"request": request})


@app.get("/v7_core", response_class=HTMLResponse)
async def v7_core_page(request: Request):
    """Core V7 UI (same as /v7)"""
    return templates.TemplateResponse("v7_mvp.html", {"request": request})


@app.get("/single_analysis", response_class=HTMLResponse)
async def single_analysis_page(request: Request):
    """Single Sample Analysis - Quality assessment without STD comparison"""
    return templates.TemplateResponse("single_analysis.html", {"request": request})


@app.get("/calibration", response_class=HTMLResponse)
async def calibration_page(request: Request):
    """Color Calibration - ColorChecker-based color accuracy verification"""
    return templates.TemplateResponse("calibration.html", {"request": request})


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


def _load_std_reference(sku_config: dict) -> Tuple[Optional[np.ndarray], Optional[Any]]:
    params = sku_config.get("params", {})
    std_ref_path = params.get("std_ref_image_path") or params.get("std_ref_path")
    if not std_ref_path:
        return None, None

    path = Path(std_ref_path)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    else:
        path = path.resolve()

    try:
        path.relative_to(BASE_DIR.resolve())
    except ValueError:
        logger.warning(f"STD reference path outside base dir: {path}")
        return None, None

    if not path.exists():
        logger.warning(f"STD reference image not found: {path}")
        return None, None

    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        logger.warning(f"STD reference image load failed: {path}")
        return None, None

    try:
        from src.core.lens_detector import DetectorConfig, LensDetector

        lens_detector = LensDetector(DetectorConfig())
        lens_detection = lens_detector.detect(img_bgr)
        if lens_detection is None:
            logger.warning("STD reference lens detection failed")
            return None, None
        return img_bgr, lens_detection
    except Exception as exc:
        logger.warning(f"STD reference lens detection error: {exc}")
        return None, None


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

        params = sku_config.get("params", {})
        debug_save_on = params.get("debug_save_on_2d")
        if not debug_save_on:
            debug_save_on = ["retake", "low_confidence"]
        elif isinstance(debug_save_on, str):
            debug_save_on = [debug_save_on]
        debug_low_confidence = float(params.get("debug_low_confidence_threshold", 0.75))
        std_ref_image, std_ref_lens = _load_std_reference(sku_config)

        result_2d, debug_info_2d = analyze_lens_zones_2d(
            img_bgr=img_bgr,
            lens_detection=lens_detection,
            sku_config=sku_config,
            ink_mask_config=InkMaskConfig(),
            std_ref_image=std_ref_image,
            std_ref_lens=std_ref_lens,
            save_debug=False,
            save_debug_on=debug_save_on,
            debug_low_confidence=debug_low_confidence,
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
        result = pipeline.process(
            str(input_path),
            sku,
            save_dir=run_dir,
            run_1d_judgment=not use_2d_analysis,
        )
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
        result = pipeline.process(
            str(temp_image_path),
            sku,
            save_dir=run_dir,
            run_1d_judgment=False,
        )
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

        inspection_result = pipeline.process(
            image_path=str(input_path),
            sku=sku,
            save_dir=run_dir,
            run_1d_judgment=False,
        )

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

        params = sku_config.get("params", {})
        debug_save_on = params.get("debug_save_on_2d")
        if not debug_save_on:
            debug_save_on = ["retake", "low_confidence"]
        elif isinstance(debug_save_on, str):
            debug_save_on = [debug_save_on]
        debug_low_confidence = float(params.get("debug_low_confidence_threshold", 0.75))
        std_ref_image, std_ref_lens = _load_std_reference(sku_config)

        # 2D 분석 실행
        result_2d, debug_info_2d = analyze_lens_zones_2d(
            img_bgr=img_bgr,
            lens_detection=lens_detection,
            sku_config=sku_config,
            ink_mask_config=InkMaskConfig(),
            std_ref_image=std_ref_image,
            std_ref_lens=std_ref_lens,
            save_debug=False,
            save_debug_on=debug_save_on,
            debug_low_confidence=debug_low_confidence,
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
