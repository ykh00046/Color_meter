# Force reload v8: Added sampling ROI info + sector uniformity analysis
import asyncio
import logging
import os
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

# Rate limiting (Quick Win: DoS prevention)
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    RATE_LIMIT_ENABLED = True
except ImportError:
    RATE_LIMIT_ENABLED = False
    Limiter = None

from src.config.v7_paths import V7_RESULTS
from src.services.analysis_service import AnalysisService
from src.services.inspection_service import InspectionPipeline, PipelineError
from src.utils.file_io import read_json

logger = logging.getLogger("web")

# ── In-process image cache for parameter recomputation ──────────────────
# Assumption: single-worker deployment (uvicorn --workers 1).
# In multi-worker or multi-node setups this dict is NOT shared; consider
# migrating to Redis (store as compressed bytes) or a shared-memory store.
IMAGE_CACHE_MAX_ENTRIES = 20  # Max number of cached images
IMAGE_CACHE_MAX_BYTES = 500 * 1024 * 1024  # 500 MB max total cache size
IMAGE_CACHE_TTL_SEC = 60 * 15  # 15-minute TTL
image_cache: Dict[str, Tuple[np.ndarray, float]] = {}
cache_lock = asyncio.Lock()
_cache_total_bytes = 0  # Track total memory usage


def _get_image_bytes(img: np.ndarray) -> int:
    """Get memory size of numpy image array"""
    return img.nbytes if img is not None else 0


def _evict_old_cache() -> None:
    """Evict expired and excess cache entries to prevent memory leaks"""
    global _cache_total_bytes
    now = time()

    # 1. Remove expired entries first
    expired = [k for k, (_, ts) in image_cache.items() if now - ts > IMAGE_CACHE_TTL_SEC]
    for k in expired:
        entry = image_cache.pop(k, None)
        if entry:
            _cache_total_bytes -= _get_image_bytes(entry[0])

    # 2. Remove oldest entries if over count limit
    while len(image_cache) > IMAGE_CACHE_MAX_ENTRIES:
        oldest_key = min(image_cache.keys(), key=lambda k: image_cache[k][1])
        entry = image_cache.pop(oldest_key, None)
        if entry:
            _cache_total_bytes -= _get_image_bytes(entry[0])
        logger.debug(f"Cache evicted (count limit): {oldest_key}")

    # 3. Remove oldest entries if over memory limit
    while _cache_total_bytes > IMAGE_CACHE_MAX_BYTES and image_cache:
        oldest_key = min(image_cache.keys(), key=lambda k: image_cache[k][1])
        entry = image_cache.pop(oldest_key, None)
        if entry:
            _cache_total_bytes -= _get_image_bytes(entry[0])
        logger.debug(f"Cache evicted (memory limit): {oldest_key}")

    # Ensure counter doesn't go negative
    _cache_total_bytes = max(0, _cache_total_bytes)


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR.parent / "results" / "web"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
BATCH_BASE_DIR = (BASE_DIR.parent / "data").resolve()
BATCH_ZIP_MAX_SIZE_MB = 200
BATCH_ZIP_MAX_UNCOMPRESSED_MB = 500
BATCH_ZIP_MAX_FILES = 2000

app = FastAPI(title="Color Meter Web UI", version="0.2")

# Rate Limiting Configuration (Quick Win: DoS prevention)
if RATE_LIMIT_ENABLED:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info("Rate limiting enabled")
else:
    limiter = None
    logger.warning("Rate limiting disabled (slowapi not installed)")


def rate_limit(limit_string: str):
    """Conditional rate limit decorator - only applies if slowapi is installed."""

    def decorator(func):
        if RATE_LIMIT_ENABLED and limiter:
            return limiter.limit(limit_string)(func)
        return func

    return decorator


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server error")
    # Security: Don't expose internal exception details to client
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount v7 results for visualization artifacts
V7_RESULTS.mkdir(parents=True, exist_ok=True)
app.mount("/v7_results", StaticFiles(directory=str(V7_RESULTS)), name="v7_results")

analysis_service = AnalysisService()


# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and cleanup old results on application startup"""
    from src.models.database import create_tables, init_database

    logger.info("Initializing database...")
    init_database(
        database_url=os.getenv("DATABASE_URL", "sqlite:///./color_meter.db"),
        echo=False,
    )
    # Tables are already created by Alembic migration
    # create_tables() is not needed if using Alembic
    logger.info("Database initialized")

    # Run result directory cleanup
    logger.info("Running result directory cleanup on startup...")
    cleanup_old_results(max_age_hours=24, max_results=100)


# Include API routers
from src.web.routers import inspection, v7

app.include_router(inspection.router)
app.include_router(v7.router)


def load_sku_config(sku: str, config_dir: Path = Path("config/sku_db")) -> dict:
    from src.utils.security import SecurityError, safe_sku_path

    try:
        cfg_path = safe_sku_path(sku, config_dir)
    except SecurityError as e:
        # Security: Log the actual error, but don't expose details to client
        logger.warning(f"SKU validation failed for '{sku}': {e}")
        raise HTTPException(status_code=400, detail="Invalid SKU format")

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
        if len(zf.infolist()) > BATCH_ZIP_MAX_FILES:
            raise HTTPException(status_code=400, detail="ZIP contains too many files")
        total_uncompressed = 0
        for member in zf.infolist():
            member_path = (dest_dir / member.filename).resolve()
            try:
                member_path.relative_to(dest_dir)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid zip entry path")
            total_uncompressed += member.file_size
            if total_uncompressed > BATCH_ZIP_MAX_UNCOMPRESSED_MB * 1024 * 1024:
                raise HTTPException(status_code=400, detail="ZIP contents too large to extract")
        zf.extractall(dest_dir)


def _safe_batch_dir(batch_dir: str) -> Path:
    candidate = Path(batch_dir).resolve()
    try:
        candidate.relative_to(BATCH_BASE_DIR)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"batch_dir must be under {BATCH_BASE_DIR}",
        )
    if not candidate.exists() or not candidate.is_dir():
        raise HTTPException(status_code=404, detail="batch_dir not found or not a directory")
    return candidate


def _get_upload_size(upload: UploadFile) -> Optional[int]:
    try:
        upload.file.seek(0, 2)
        size = upload.file.tell()
        upload.file.seek(0)
        return int(size)
    except Exception:
        return None


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


@app.get("/design_system_demo", response_class=HTMLResponse)
async def design_system_demo_page(request: Request):
    """Design System Demo - Phase 1 verification page"""
    return templates.TemplateResponse("design_system_demo.html", {"request": request})


# ================================
# Helper Functions for /inspect endpoint
# ================================


async def validate_and_save_file(file: UploadFile, run_id: str, run_dir: Path) -> tuple[bytes, Path, str]:
    """Validate uploaded file and save to disk.

    Returns:
        tuple: (file_content, input_path, original_name)
    """
    from src.utils.security import validate_file_extension, validate_file_size

    # Constants for image validation
    MAX_IMAGE_DIMENSION = 8192  # 8K max to prevent DoS

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

    # Validate image dimensions (Quick Win: DoS prevention)
    try:
        img_array = np.frombuffer(file_content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file: could not decode")
        h, w = img.shape[:2]
        if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
            raise HTTPException(
                status_code=400, detail=f"Image too large: {w}x{h} (max {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

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


def generate_radial_analysis(result, sku_config: dict) -> tuple:
    """Generate radial profile analysis data.

    Returns:
        tuple: (analysis_payload, lens_info)
    """
    rp = getattr(result, "radial_profile", None)
    lens_detection = getattr(result, "lens_detection", None)

    if rp is None or lens_detection is None:
        return None, None

    analysis_payload = analysis_service.analyze_radial_profile(profile=rp, lens_radius=float(lens_detection.r))

    lens_info = {
        "center_x": float(lens_detection.cx),
        "center_y": float(lens_detection.cy),
        "radius": float(lens_detection.r),
        "confidence": float(lens_detection.confidence),
    }

    logger.warning(
        f"[DEBUG] Lens detected: center=({lens_info['center_x']:.1f}, {lens_info['center_y']:.1f}), "
        f"radius={lens_info['radius']:.1f}"
    )

    return analysis_payload, lens_info


def save_result_json(result, run_dir: Path) -> Path:
    """Save inspection result as JSON.

    Returns:
        Path: output JSON file path
    """
    import json
    from dataclasses import asdict, fields, is_dataclass

    output = run_dir / "result.json"
    if is_dataclass(result):
        result_dict = {f.name: getattr(result, f.name) for f in fields(result) if f.name != "image"}
    else:
        result_dict = asdict(result)

    def _to_jsonable(value):
        if is_dataclass(value):
            return {k: _to_jsonable(v) for k, v in asdict(value).items()}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, dict):
            return {k: _to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_jsonable(v) for v in value]
        return value

    result_dict = _to_jsonable(result_dict)

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
    ink: str,
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
        "ink": ink,
        "image_id": image_id,  # PHASE7: For parameter recomputation
        "overlay": f"/results/{run_id}/overlay.png" if overlay_path and overlay_path.exists() else None,
        "analysis": analysis_payload,
        "metrics": getattr(result, "metrics", None),
        "lens_info": lens_info,
        "ring_sector_cells": ring_sector_data,
        "uniformity": uniformity_data,
        "result_path": f"/results/{run_id}",
    }
    if applied_params:
        response["applied_params"] = applied_params

    # Add judgment results if requested
    if run_judgment:
        response["judgment"] = {
            "result": result.judgment,
            "overall_delta_e": float(result.overall_delta_e),
            "confidence": float(result.confidence) if hasattr(result, "confidence") else 1.0,
            "ng_reasons": result.ng_reasons if hasattr(result, "ng_reasons") else [],
        }
    else:
        response["judgment"] = None

    return response


# ================================
# Main /inspect endpoint (refactored)
# ================================


@app.post("/inspect")
@rate_limit("30/minute")  # Quick Win: DoS prevention
async def inspect_image(
    request: Request,
    file: UploadFile = File(...),
    sku: str = Form(...),
    ink: str = Form("INK_DEFAULT"),
    run_judgment: bool = Form(False),
    options: Optional[str] = Form(None),
):
    """
    Main inspection endpoint (refactored for clarity).

    Handles file upload, validation, pipeline execution, visualization,
    and response generation.
    """
    v7_payload = None
    # 1. Setup
    run_id = uuid.uuid4().hex[:8]
    run_dir = _safe_result_path(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 2. File validation and save
    file_content, input_path, original_name = await validate_and_save_file(file, run_id, run_dir)

    # Load image once for multiple uses
    import cv2

    img_bgr = cv2.imread(str(input_path))

    # 2.5. Run V7 Engine API
    from src.engine_v7.api import inspect_single

    try:
        if img_bgr is not None:
            v7_payload = inspect_single(image_bgr=img_bgr, sku=sku, ink=ink, run_id=run_id)
        else:
            logger.warning("Failed to load image for V7 API")
    except Exception as exc:
        logger.warning("V7 API call failed; returning pipeline response only: %s", exc)

    # 3. Parse options
    enable_illumination_correction = parse_inspection_options(options)

    # 4. Load SKU config
    sku_config = load_sku_config(sku)

    # 5. Run pipeline
    pipeline = InspectionPipeline(sku_config, save_intermediates=True)
    try:
        result = pipeline.process(
            str(input_path),
            sku,
            ink=ink,
            save_dir=run_dir,
            run_1d_judgment=True,
        )
    except PipelineError as e:
        # Security: Log full error, but give generic message to client
        logger.error(f"Pipeline processing failed: {e}")
        raise HTTPException(status_code=400, detail="Image processing failed")

    # 6. Cache image for parameter recomputation
    image_id = None
    if img_bgr is not None:
        global _cache_total_bytes
        image_id = str(uuid.uuid4())
        img_copy = img_bgr.copy()
        async with cache_lock:
            image_cache[image_id] = (img_copy, time())
            _cache_total_bytes += _get_image_bytes(img_copy)
            _evict_old_cache()
        logger.info(
            f"Cached image with ID: {image_id}, cache entries: {len(image_cache)}, "
            f"cache MB: {_cache_total_bytes / 1024 / 1024:.1f}"
        )

    # 7. Generate radial analysis data
    analysis_payload, lens_info = generate_radial_analysis(result, sku_config)

    # 8. Ring-sector analysis removed (legacy dependency)
    ring_sector_data = None
    uniformity_data = None

    # 9. Create overlay visualization
    overlay_path = None

    # 10. Save result JSON
    output = save_result_json(result, run_dir)

    # 10.5. Save to database history
    save_inspection_to_db(
        session_id=run_id,
        sku_code=sku,
        image_filename=original_name,
        image_path=str(input_path),
        result=result,
    )

    # 11. Build and return response
    response = build_inspection_response(
        run_id=run_id,
        original_name=original_name,
        sku=sku,
        ink=ink,
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
    if v7_payload is not None:
        from fastapi.encoders import jsonable_encoder

        response["v7"] = jsonable_encoder(v7_payload)
    return response


# ================================
# PHASE7: Parameter Recomputation API
# ================================


def validate_recompute_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize recomputation parameters.

    Allowed parameter groups:
    - profiler_config: num_samples, num_points, sample_percentile
    - corrector_config: method (gray_world, white_patch, auto, polynomial, gaussian)
    - sector_config: sector_count, ring_boundaries

    Returns:
        dict: Validated parameters

    Raises:
        HTTPException: If parameters are invalid
    """
    allowed_params = {
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
@rate_limit("60/minute")  # Quick Win: DoS prevention
async def recompute_analysis(
    request: Request,
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
            global _cache_total_bytes
            removed_entry = image_cache.pop(image_id, None)
            if removed_entry:
                _cache_total_bytes -= _get_image_bytes(removed_entry[0])
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
            run_1d_judgment=True,
        )
    except PipelineError as e:
        # Security: Log full error, but give generic message to client
        logger.error(f"Pipeline recompute failed: {e}")
        raise HTTPException(status_code=400, detail="Image reprocessing failed")

    # 6. Generate radial analysis data
    analysis_payload, lens_info = generate_radial_analysis(result, sku_config)

    # 7. Ring-sector analysis removed (legacy dependency)
    ring_sector_data = None
    uniformity_data = None

    # 8. Create overlay visualization
    overlay_path = None

    # 9. Save result JSON
    output = save_result_json(result, run_dir)

    # 10. Build and return response (same format as /inspect)
    response = build_inspection_response(
        run_id=run_id,
        original_name=f"recompute_{image_id}",
        sku=sku,
        ink="INK_DEFAULT",
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
@rate_limit("10/minute")  # Quick Win: DoS prevention (heavy operation)
async def batch_inspect(
    request: Request,
    sku: str = Form(...),
    batch_dir: Optional[str] = Form(None),
    batch_zip: Optional[UploadFile] = File(None),
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
        safe_dir = _safe_batch_dir(batch_dir)
        image_paths = list(safe_dir.glob("*.jpg"))
    elif batch_zip:
        # ZIP 업로드를 inputs 폴더에 저장 후 해제
        from src.utils.security import sanitize_filename

        upload_size = _get_upload_size(batch_zip)
        if upload_size is not None and upload_size > BATCH_ZIP_MAX_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail="ZIP file too large")

        zip_name = sanitize_filename(batch_zip.filename or "batch.zip")
        zip_path = input_dir / zip_name
        with zip_path.open("wb") as f:
            shutil.copyfileobj(batch_zip.file, f)
        _safe_extract_zip(zip_path, input_dir)
        image_paths = list(input_dir.glob("**/*.jpg"))

    if not image_paths:
        raise HTTPException(status_code=404, detail="No jpg images found for batch processing")

    sku_config = load_sku_config(sku)

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
@rate_limit("30/minute")  # Quick Win: DoS prevention
async def inspect_v2(
    request: Request,
    file: UploadFile = File(...),
    sku: str = Form(...),
    smoothing_window: int = Form(5),
    gradient_threshold: float = Form(0.5),
):
    """
    Dashboard용 endpoint - pipeline.process() 통짜 호출 방식.
    기존 판정 로직을 100% 재사용하고, 추가로 ProfileAnalyzer로 그래프 데이터만 생성.
    """
    from src.services.analysis_service import AnalysisService
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
            run_1d_judgment=True,
        )

        logger.info(f"[INSPECT_V2] Pipeline completed: {inspection_result.judgment}")

        # 3. Judgment 데이터 추출 (2D 결과 사용, 운영 UX 개선)
        judgment_result = {
            "result": inspection_result.judgment,
            "overall_delta_e": float(inspection_result.overall_delta_e),
            "confidence": float(inspection_result.confidence) if hasattr(inspection_result, "confidence") else 1.0,
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
        # Refactored to use AnalysisService which wraps v7 engine
        analysis_service = AnalysisService()
        rp = inspection_result.radial_profile

        if rp is None:
            raise ValueError("Radial profile not available in inspection result")

        # analyze_radial_profile 내부에서 v7 analyze_profile 호출
        analysis_result = analysis_service.analyze_radial_profile(
            profile=rp,
            lens_radius=float(inspection_result.lens_detection.r),
        )

        # 5. lens_info 추가 (overlay용)
        ld = inspection_result.lens_detection
        analysis_result["lens_info"] = {
            "center_x": float(ld.cx),
            "center_y": float(ld.cy),
            "radius_px": float(ld.r),
            "confidence": float(ld.confidence),
        }

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
        # Security: Don't expose internal error details to client
        raise HTTPException(status_code=500, detail="Analysis failed")


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
