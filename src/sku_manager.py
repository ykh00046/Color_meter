"""
SKU Configuration Manager

Manages SKU configurations including CRUD operations and automatic baseline generation.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.image_loader import ImageConfig, ImageLoader
from src.core.lens_detector import DetectorConfig, LensDetector
from src.core.radial_profiler import ProfilerConfig, RadialProfiler
from src.core.zone_segmenter import SegmenterConfig, ZoneSegmenter
from src.utils.file_io import read_json, write_json


class SkuManagerError(Exception):
    """Base exception for SKU manager errors"""

    pass


class SkuNotFoundError(SkuManagerError):
    """SKU not found in database"""

    pass


class SkuAlreadyExistsError(SkuManagerError):
    """SKU already exists"""

    pass


class InvalidSkuDataError(SkuManagerError):
    """Invalid SKU data or schema validation failed"""

    pass


class InsufficientSamplesError(SkuManagerError):
    """Insufficient samples for baseline generation"""

    pass


class SkuConfigManager:
    """
    SKU Configuration Manager

    Provides CRUD operations for SKU configurations and automatic baseline generation
    from OK sample images.
    """

    def __init__(self, db_path: Path = Path("config/sku_db")):
        """
        Initialize SKU manager

        Args:
            db_path: Path to SKU database directory
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

    def create_sku(
        self,
        sku_code: str,
        description: str,
        default_threshold: float = 3.5,
        zones: Optional[Dict[str, Dict[str, float]]] = None,
        author: str = "user",
    ) -> Dict[str, Any]:
        """
        Create new SKU configuration

        Args:
            sku_code: SKU code (e.g., "SKU002")
            description: SKU description
            default_threshold: Default ΔE threshold
            zones: Zone LAB values and thresholds
                   Example: {"A": {"L": 70, "a": -10, "b": -30, "threshold": 4.0}}
            author: Author name

        Returns:
            Created SKU data

        Raises:
            SkuAlreadyExistsError: If SKU already exists
            InvalidSkuDataError: If SKU code is invalid
        """
        # Validate SKU code format
        if not re.match(r"^SKU[A-Z0-9_]+$", sku_code):
            raise InvalidSkuDataError(
                f"Invalid SKU code format: {sku_code}. Must start with 'SKU' and contain A-Z/0-9/_"
            )

        # Check if already exists
        sku_path = self._get_sku_path(sku_code)
        if sku_path.exists():
            raise SkuAlreadyExistsError(f"SKU {sku_code} already exists")

        # Create SKU data
        now = datetime.now().isoformat()
        sku_data = {
            "sku_code": sku_code,
            "description": description,
            "default_threshold": default_threshold,
            "zones": zones or {},
            "metadata": {
                "created_at": now,
                "last_updated": now,
                "baseline_samples": 0,
                "calibration_method": "manual",
                "author": author,
            },
        }

        # Validate and save
        self._validate_sku(sku_data)
        self._save_sku(sku_code, sku_data)

        return sku_data

    def get_sku(self, sku_code: str) -> Dict[str, Any]:
        """
        Get SKU configuration

        Args:
            sku_code: SKU code

        Returns:
            SKU data

        Raises:
            SkuNotFoundError: If SKU doesn't exist
        """
        sku_path = self._get_sku_path(sku_code)
        if not sku_path.exists():
            raise SkuNotFoundError(f"SKU {sku_code} not found")

        return self._load_sku(sku_code)

    def update_sku(self, sku_code: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update SKU configuration

        Args:
            sku_code: SKU code
            updates: Fields to update
                     Example: {"description": "Updated", "default_threshold": 4.0}

        Returns:
            Updated SKU data

        Raises:
            SkuNotFoundError: If SKU doesn't exist
        """
        # Load existing SKU
        sku_data = self.get_sku(sku_code)

        # Apply updates
        for key, value in updates.items():
            if key == "metadata":
                # Merge metadata
                sku_data["metadata"].update(value)
            elif "." in key:
                # Nested update (e.g., "zones.A.threshold")
                parts = key.split(".")
                target = sku_data
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = value
            else:
                sku_data[key] = value

        # Update timestamp
        sku_data["metadata"]["last_updated"] = datetime.now().isoformat()

        # Validate and save
        self._validate_sku(sku_data)
        self._save_sku(sku_code, sku_data)

        return sku_data

    def delete_sku(self, sku_code: str) -> bool:
        """
        Delete SKU configuration

        Args:
            sku_code: SKU code

        Returns:
            True if deleted successfully

        Raises:
            SkuNotFoundError: If SKU doesn't exist
        """
        sku_path = self._get_sku_path(sku_code)
        if not sku_path.exists():
            raise SkuNotFoundError(f"SKU {sku_code} not found")

        sku_path.unlink()
        return True

    def list_all_skus(self) -> List[Dict[str, Any]]:
        """
        List all SKU configurations

        Returns:
            List of SKU summary information
        """
        skus = []
        for sku_path in sorted(self.db_path.glob("SKU*.json")):
            try:
                sku_data = read_json(sku_path)
                summary = {
                    "sku_code": sku_data["sku_code"],
                    "description": sku_data.get("description", ""),
                    "zones_count": len(sku_data.get("zones", {})),
                    "created_at": sku_data.get("metadata", {}).get("created_at", ""),
                    "baseline_samples": sku_data.get("metadata", {}).get("baseline_samples", 0),
                }
                skus.append(summary)
            except Exception:
                # Skip invalid files
                continue

        return skus

    def list_skus(self) -> List[str]:
        """
        List all SKU codes

        Returns:
            List of SKU codes (e.g., ["SKU001", "SKU002", "SKU003"])
        """
        sku_codes = []
        for sku_path in sorted(self.db_path.glob("SKU*.json")):
            try:
                sku_data = read_json(sku_path)
                sku_codes.append(sku_data["sku_code"])
            except Exception:
                # Skip invalid files
                continue

        return sku_codes

    def generate_baseline(
        self,
        sku_code: str,
        ok_images: List[Path],
        description: str = "",
        default_threshold: float = 3.5,
        threshold_method: str = "mean_plus_2std",
        image_config: Optional[ImageConfig] = None,
        detector_config: Optional[DetectorConfig] = None,
        profiler_config: Optional[ProfilerConfig] = None,
        segmenter_config: Optional[SegmenterConfig] = None,
    ) -> Dict[str, Any]:
        """
        Generate baseline from OK sample images

        Args:
            sku_code: SKU code
            ok_images: List of OK sample image paths (minimum 3, recommended 5-10)
            description: SKU description
            default_threshold: Default ΔE threshold
            threshold_method: Threshold calculation method
                - "mean_plus_2std": mean + 2*std (95% confidence, default)
                - "mean_plus_3std": mean + 3*std (99.7% confidence)
                - "fixed": Use default_threshold
            image_config: Image loader configuration
            detector_config: Lens detector configuration
            profiler_config: Radial profiler configuration
            segmenter_config: Zone segmenter configuration

        Returns:
            Generated SKU data

        Raises:
            InsufficientSamplesError: If less than 3 samples
            ValueError: If processing fails
        """
        # Validate sample count
        if len(ok_images) < 3:
            raise InsufficientSamplesError(f"Minimum 3 samples required, got {len(ok_images)}")

        # Initialize pipeline components
        image_loader = ImageLoader(image_config or ImageConfig())
        lens_detector = LensDetector(detector_config or DetectorConfig())
        radial_profiler = RadialProfiler(profiler_config or ProfilerConfig())
        zone_segmenter = ZoneSegmenter(segmenter_config or SegmenterConfig())

        # Process each image and collect zone data
        zone_data: Dict[str, List[Tuple[float, float, float]]] = {}
        successful_samples = 0
        expected_zone_names = None  # Enforce consistent zone structure

        for i, image_path in enumerate(ok_images):
            try:
                # Process image
                image = image_loader.load_from_file(image_path)
                processed = image_loader.preprocess(image)
                detection = lens_detector.detect(processed)
                profile = radial_profiler.extract_profile(processed, detection)
                zones = zone_segmenter.segment(profile)

                # Validate zone consistency (critical for accurate baseline)
                current_zone_names = set(zone.name for zone in zones)
                if expected_zone_names is None:
                    # First successful sample defines expected structure
                    expected_zone_names = current_zone_names
                    print(f"Baseline zone structure: {sorted(expected_zone_names)}")
                elif current_zone_names != expected_zone_names:
                    # Zone structure mismatch - skip this sample
                    print(f"Warning: Zone structure mismatch in {image_path}")
                    print(f"  Expected: {sorted(expected_zone_names)}")
                    print(f"  Got: {sorted(current_zone_names)}")
                    print("  Skipping sample to ensure baseline consistency")
                    continue

                # Collect zone LAB values
                for zone in zones:
                    if zone.name not in zone_data:
                        zone_data[zone.name] = []
                    zone_data[zone.name].append((zone.mean_L, zone.mean_a, zone.mean_b))

                successful_samples += 1

            except Exception as e:
                print(f"Warning: Failed to process {image_path}: {e}")
                continue

        # Check if we have enough successful samples
        if successful_samples < 3:
            raise InsufficientSamplesError(
                f"Only {successful_samples} samples processed successfully (minimum 3 required)"
            )

        # Calculate statistics for each zone
        zones_config = {}
        statistics = {}

        for zone_name, lab_values in zone_data.items():
            # Convert to numpy arrays
            L_values = np.array([lab[0] for lab in lab_values])
            a_values = np.array([lab[1] for lab in lab_values])
            b_values = np.array([lab[2] for lab in lab_values])

            # Calculate mean and std
            mean_L = float(np.mean(L_values))
            mean_a = float(np.mean(a_values))
            mean_b = float(np.mean(b_values))
            std_L = float(np.std(L_values))
            std_a = float(np.std(a_values))
            std_b = float(np.std(b_values))

            # Calculate threshold
            if threshold_method == "mean_plus_2std":
                max_std = max(std_L, std_a, std_b)
                threshold = default_threshold + 2.0 * max_std
            elif threshold_method == "mean_plus_3std":
                max_std = max(std_L, std_a, std_b)
                threshold = default_threshold + 3.0 * max_std
            else:  # "fixed"
                threshold = default_threshold

            # Round to reasonable precision
            zones_config[zone_name] = {
                "L": round(mean_L, 1),
                "a": round(mean_a, 1),
                "b": round(mean_b, 1),
                "threshold": round(threshold, 1),
                "description": f"Zone {zone_name} (auto-generated)",
            }

            statistics[f"zone_{zone_name}"] = {
                "L_std": round(std_L, 2),
                "a_std": round(std_a, 2),
                "b_std": round(std_b, 2),
                "samples": len(lab_values),
            }

        # Create SKU data
        now = datetime.now().isoformat()
        sku_data = {
            "sku_code": sku_code,
            "description": description or f"Auto-generated SKU {sku_code}",
            "default_threshold": default_threshold,
            "zones": zones_config,
            "metadata": {
                "created_at": now,
                "last_updated": now,
                "baseline_samples": successful_samples,
                "calibration_method": "auto_generated",
                "threshold_method": threshold_method,
                "author": "system",
                "statistics": statistics,
                "notes": f"Generated from {successful_samples} OK samples",
            },
        }

        # Validate and save
        self._validate_sku(sku_data)
        self._save_sku(sku_code, sku_data)

        return sku_data

    def _validate_sku(self, sku_data: Dict) -> bool:
        """
        Validate SKU data against schema

        Args:
            sku_data: SKU data to validate

        Returns:
            True if valid

        Raises:
            InvalidSkuDataError: If validation fails
        """
        # Check required fields
        required_fields = ["sku_code", "description", "default_threshold", "zones", "metadata"]
        for field in required_fields:
            if field not in sku_data:
                raise InvalidSkuDataError(f"Missing required field: {field}")

        # Validate SKU code format
        if not re.match(r"^SKU[A-Z0-9_]+$", sku_data["sku_code"]):
            raise InvalidSkuDataError(f"Invalid SKU code: {sku_data['sku_code']}")

        # Validate zones
        if not isinstance(sku_data["zones"], dict):
            raise InvalidSkuDataError("zones must be a dictionary")

        for zone_name, zone_config in sku_data["zones"].items():
            # Check zone name format (single uppercase letter)
            if not re.match(r"^[A-Z]$", zone_name):
                raise InvalidSkuDataError(f"Invalid zone name: {zone_name}. Must be A-Z")

            # Check required zone fields
            required_zone_fields = ["L", "a", "b", "threshold"]
            for field in required_zone_fields:
                if field not in zone_config:
                    raise InvalidSkuDataError(f"Missing field in zone {zone_name}: {field}")

            # Validate LAB ranges (relaxed for practical implementations)
            if not (0 <= zone_config["L"] <= 100):
                raise InvalidSkuDataError(f"L value out of range [0-100]: {zone_config['L']}")
            if not (-200 <= zone_config["a"] <= 200):
                raise InvalidSkuDataError(f"a value out of range [-200,200]: {zone_config['a']}")
            if not (-200 <= zone_config["b"] <= 200):
                raise InvalidSkuDataError(f"b value out of range [-200,200]: {zone_config['b']}")
            if zone_config["threshold"] < 0:
                raise InvalidSkuDataError(f"threshold must be >= 0: {zone_config['threshold']}")

        # Validate metadata
        required_metadata = ["created_at", "last_updated", "baseline_samples", "calibration_method"]
        for field in required_metadata:
            if field not in sku_data["metadata"]:
                raise InvalidSkuDataError(f"Missing metadata field: {field}")

        if sku_data["metadata"]["calibration_method"] not in ["manual", "auto_generated"]:
            raise InvalidSkuDataError(f"Invalid calibration_method: {sku_data['metadata']['calibration_method']}")

        return True

    def _save_sku(self, sku_code: str, data: Dict) -> Path:
        """
        Save SKU data to JSON file

        Args:
            sku_code: SKU code
            data: SKU data

        Returns:
            Path to saved file
        """
        sku_path = self._get_sku_path(sku_code)
        write_json(data, sku_path)
        return sku_path

    def _load_sku(self, sku_code: str) -> Dict:
        """
        Load SKU data from JSON file

        Args:
            sku_code: SKU code

        Returns:
            SKU data
        """
        sku_path = self._get_sku_path(sku_code)
        return read_json(sku_path)

    def _get_sku_path(self, sku_code: str) -> Path:
        """
        Get SKU file path

        Args:
            sku_code: SKU code

        Returns:
            Path to SKU JSON file

        Raises:
            ValueError: If SKU code is invalid or path traversal detected
        """
        from src.utils.security import SecurityError, safe_sku_path

        # Use centralized security validation
        try:
            return safe_sku_path(sku_code, self.db_path)
        except SecurityError as e:
            raise ValueError(str(e))
