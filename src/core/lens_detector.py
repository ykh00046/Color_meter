import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class LensDetectionError(Exception):
    pass


@dataclass
class LensDetection:
    center_x: float
    center_y: float
    radius: float
    confidence: float = 1.0
    method: str = "unknown"
    roi: Optional[Tuple[int, int, int, int]] = None


@dataclass
class DetectorConfig:
    method: str = "hybrid"
    hough_dp: float = 1.2
    hough_min_dist_ratio: float = 0.5
    hough_param1: float = 50
    hough_param2: float = 30
    hough_min_radius_ratio: float = 0.3
    hough_max_radius_ratio: float = 0.8
    contour_threshold_method: str = "otsu"
    contour_morph_kernel_size: int = 5
    contour_min_area_ratio: float = 0.01
    hybrid_merge_dist_threshold: float = 10.0
    subpixel_refinement_enabled: bool = False
    subpixel_window_size: int = 21
    background_fallback_enabled: bool = True
    background_color_distance_threshold: float = 30.0
    background_min_area_ratio: float = 0.05


class LensDetector:
    def __init__(self, config: DetectorConfig = DetectorConfig()) -> None:
        self.config: DetectorConfig = config

    def detect(self, image: np.ndarray) -> LensDetection:
        if image is None:
            raise ValueError("Input image cannot be None.")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        detection = None
        if self.config.method == "hough":
            detection = self._detect_hough(gray_image)
        elif self.config.method == "contour":
            detection = self._detect_contour(gray_image)
        elif self.config.method == "hybrid":
            detection = self._detect_hybrid(gray_image)

        # Fallback to background-based detection if primary methods fail
        if not detection and self.config.background_fallback_enabled:
            logger.warning("Primary detection methods failed, trying background-based fallback")
            detection = self._detect_background_based(image)

        if detection and self.config.subpixel_refinement_enabled:
            detection.center_x, detection.center_y = self._refine_center(
                gray_image, detection.center_x, detection.center_y, detection.radius
            )
            detection.method += "+subpixel"

        if not detection:
            raise LensDetectionError("Lens detection failed.")

        # ROI 설정 (Fixed: Proper boundary clamping)
        h, w = image.shape[:2]
        roi_x = int(detection.center_x - detection.radius * 1.1)
        roi_y = int(detection.center_y - detection.radius * 1.1)
        roi_w = int(detection.radius * 2.2)
        roi_h = int(detection.radius * 2.2)

        # Clamp start points to image boundaries
        roi_x_start = max(0, roi_x)
        roi_y_start = max(0, roi_y)

        # Clamp end points to image boundaries
        roi_x_end = min(w, roi_x + roi_w)
        roi_y_end = min(h, roi_y + roi_h)

        # Calculate clamped width and height
        roi_w_clamped = roi_x_end - roi_x_start
        roi_h_clamped = roi_y_end - roi_y_start

        detection.roi = (roi_x_start, roi_y_start, roi_w_clamped, roi_h_clamped)

        return detection

    def _detect_hough(self, gray_image: np.ndarray) -> Optional[LensDetection]:
        min_radius = int(min(gray_image.shape) * self.config.hough_min_radius_ratio)
        max_radius = int(max(gray_image.shape) * self.config.hough_max_radius_ratio)

        circles = cv2.HoughCircles(
            gray_image,
            cv2.HOUGH_GRADIENT,
            dp=self.config.hough_dp,
            minDist=min_radius * self.config.hough_min_dist_ratio,
            param1=self.config.hough_param1,
            param2=self.config.hough_param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        if circles is not None:
            cx, cy, r = circles[0][0]
            return LensDetection(center_x=cx, center_y=cy, radius=r, confidence=0.9, method="hough")
        return None  # type: ignore[unreachable]

    def _detect_contour(self, gray_image: np.ndarray) -> Optional[LensDetection]:
        if self.config.contour_threshold_method == "otsu":
            _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.config.contour_morph_kernel_size, self.config.contour_morph_kernel_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        min_area = gray_image.size * self.config.contour_min_area_ratio
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        if not valid_contours:
            return None

        largest = max(valid_contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest)

        area_contour = cv2.contourArea(largest)
        area_circle = np.pi * radius**2
        confidence = area_contour / area_circle if area_circle > 0 else 0

        return LensDetection(center_x=x, center_y=y, radius=radius, confidence=confidence, method="contour")

    def _detect_hybrid(self, gray_image: np.ndarray) -> Optional[LensDetection]:
        hough = self._detect_hough(gray_image)
        contour = self._detect_contour(gray_image)

        result = None
        if hough and contour:
            dist = np.sqrt((hough.center_x - contour.center_x) ** 2 + (hough.center_y - contour.center_y) ** 2)
            if dist < self.config.hybrid_merge_dist_threshold:
                result = LensDetection(
                    center_x=(hough.center_x + contour.center_x) / 2,
                    center_y=(hough.center_y + contour.center_y) / 2,
                    radius=(hough.radius + contour.radius) / 2,
                    confidence=1.0,
                    method="hybrid",
                )
            else:
                result = hough if hough.confidence > contour.confidence else contour
        else:
            result = hough or contour

        if result:
            result.method = "hybrid"
        return result

    def _detect_background_based(self, image: np.ndarray) -> Optional[LensDetection]:
        """
        Background-color-based lens detection (fallback method).

        This method is used when Hough Circle and Contour detection fail.
        It works by:
        1. Sampling background color from image edges/corners
        2. Creating a mask where pixels differ from background
        3. Finding the largest connected component (the lens)
        4. Calculating centroid and minimum enclosing circle

        Args:
            image: BGR or grayscale image

        Returns:
            LensDetection with lower confidence, or None if failed
        """
        # Convert to BGR if grayscale
        if len(image.shape) == 2:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr_image = image

        h, w = bgr_image.shape[:2]

        # Step 1: Sample background color from image edges
        bg_color = self._sample_background_color(bgr_image)
        logger.debug(f"Background color sampled: {bg_color}")

        # Step 2: Calculate color distance for each pixel
        color_dist = np.linalg.norm(bgr_image.astype(np.float32) - bg_color, axis=2)

        # Step 3: Create binary mask (foreground = different from background)
        threshold = self.config.background_color_distance_threshold
        _, foreground_mask = cv2.threshold(color_dist.astype(np.uint8), int(threshold), 255, cv2.THRESH_BINARY)

        # Step 4: Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Step 5: Find largest connected component
        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.warning("Background-based detection failed: No contours found")
            return None

        # Filter by minimum area
        min_area = (h * w) * self.config.background_min_area_ratio
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        if not valid_contours:
            logger.warning(f"Background-based detection failed: No contours larger than {min_area:.0f} pixels")
            return None

        # Get largest contour
        largest_contour = max(valid_contours, key=cv2.contourArea)

        # Step 6: Calculate centroid and minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)

        # Calculate confidence based on circularity
        area_contour = cv2.contourArea(largest_contour)
        area_circle = np.pi * radius**2
        circularity = area_contour / area_circle if area_circle > 0 else 0

        # Lower confidence for background-based method (0.3-0.6 range)
        confidence = 0.3 + (circularity * 0.3)

        logger.info(
            f"Background-based detection succeeded: center=({x:.1f}, {y:.1f}), "
            f"radius={radius:.1f}, confidence={confidence:.2f}"
        )

        return LensDetection(center_x=x, center_y=y, radius=radius, confidence=confidence, method="background")

    def _sample_background_color(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Sample background color from image edges and corners.

        Args:
            bgr_image: BGR image

        Returns:
            Background color as numpy array [B, G, R]
        """
        h, w = bgr_image.shape[:2]

        # Sample from edges and corners (10 pixel strips)
        edge_width = 10

        top_edge = bgr_image[0:edge_width, :]
        bottom_edge = bgr_image[h - edge_width : h, :]
        left_edge = bgr_image[:, 0:edge_width]
        right_edge = bgr_image[:, w - edge_width : w]

        # Combine all edge samples
        edge_samples = np.vstack(
            [top_edge.reshape(-1, 3), bottom_edge.reshape(-1, 3), left_edge.reshape(-1, 3), right_edge.reshape(-1, 3)]
        )

        # Use median to be robust against outliers
        bg_color = np.asarray(np.median(edge_samples, axis=0), dtype=np.float32)

        return bg_color

    def _refine_center(self, gray_image: np.ndarray, cx: float, cy: float, radius: float) -> Tuple[float, float]:
        """
        Sub-pixel refinement using edge-based weighted centroid.

        Args:
            gray_image: Grayscale image
            cx: Initial center x
            cy: Initial center y
            radius: Lens radius

        Returns:
            Refined (cx, cy) coordinates
        """
        # Extract ROI around detected center
        r_int = int(radius)
        x_min = max(0, int(cx - r_int))
        y_min = max(0, int(cy - r_int))
        x_max = min(gray_image.shape[1], int(cx + r_int))
        y_max = min(gray_image.shape[0], int(cy + r_int))

        roi = gray_image[y_min:y_max, x_min:x_max]

        if roi.size == 0:
            logger.warning("ROI is empty, skipping refinement")
            return cx, cy

        # Sobel edge detection
        edges_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        edges_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(edges_x**2 + edges_y**2)

        # Threshold to get strong edges (top 10%)
        threshold = np.percentile(edge_mag, 90)
        edge_mask = edge_mag > threshold

        if edge_mask.sum() == 0:
            logger.warning("No strong edges found, skipping refinement")
            return cx, cy

        # Weighted centroid calculation
        y_coords, x_coords = np.where(edge_mask)
        weights = edge_mag[edge_mask]

        cx_refined = x_min + np.average(x_coords, weights=weights)
        cy_refined = y_min + np.average(y_coords, weights=weights)

        logger.debug(f"Center refined: ({cx:.2f}, {cy:.2f}) -> ({cx_refined:.2f}, {cy_refined:.2f})")

        return cx_refined, cy_refined
