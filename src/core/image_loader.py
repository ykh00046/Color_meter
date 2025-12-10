import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List

from src.utils.camera import Camera 
from src.utils.file_io import FileIO 

import logging
logger = logging.getLogger(__name__)

@dataclass
class ImageConfig:
    resolution_w: int = 1920
    resolution_h: int = 1080
    roi: Optional[Tuple[int, int, int, int]] = None
    
    denoise_enabled: bool = True
    denoise_method: str = "bilateral"
    denoise_kernel_size: int = 5
    denoise_sigma_color: int = 75
    denoise_sigma_space: int = 75

    white_balance_enabled: bool = True
    white_balance_method: str = "gray_world"
    
    auto_roi_detection: bool = True
    auto_roi_margin_ratio: float = 0.2

class ImageLoader:
    def __init__(self, config: ImageConfig = ImageConfig()):
        self.config = config
        self.file_io = FileIO()

    def load_from_file(self, filepath: Path) -> Optional[np.ndarray]:
        try:
            image = self.file_io.load_image(filepath)
            if image is None:
                logger.warning(f"Failed to load image from {filepath}")
            return image
        except Exception as e:
            logger.error(f"Error loading image from {filepath}: {e}")
            return None

    def load_from_camera(self, camera: Camera) -> Optional[np.ndarray]:
        try:
            image = camera.capture_frame()
            if image is None:
                logger.warning("Failed to capture frame from camera.")
            return image
        except Exception as e:
            logger.error(f"Error capturing from camera: {e}")
            return None

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            return None
        if image.size == 0:
            return image

        processed_image = image.copy()

        # 1. ROI 검출 및 크롭
        current_roi = self.config.roi
        if self.config.auto_roi_detection:
            x, y, w, h = self._detect_roi_from_image(processed_image)
            if w > 0 and h > 0:
                current_roi = (x, y, w, h)
        
        if current_roi:
            x, y, w, h = current_roi
            if x >= 0 and y >= 0 and x+w <= processed_image.shape[1] and y+h <= processed_image.shape[0]:
                processed_image = processed_image[y:y+h, x:x+w]

        # 2. 노이즈 제거
        if self.config.denoise_enabled:
            processed_image = self._denoise_image(processed_image)

        # 3. 화이트 밸런스 보정
        if self.config.white_balance_enabled:
            processed_image = self._apply_white_balance(processed_image)

        return processed_image

    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        if image.size == 0: return image
        if self.config.denoise_method == "gaussian":
            return cv2.GaussianBlur(image, (self.config.denoise_kernel_size, self.config.denoise_kernel_size), 0)
        elif self.config.denoise_method == "bilateral":
            return cv2.bilateralFilter(image, self.config.denoise_kernel_size, 
                                     self.config.denoise_sigma_color, self.config.denoise_sigma_space)
        return image

    def _apply_white_balance(self, image: np.ndarray) -> np.ndarray:
        if image.size == 0: return image
        if self.config.white_balance_method == "gray_world":
            return self._gray_world_white_balance(image)
        return image

    def _gray_world_white_balance(self, image: np.ndarray) -> np.ndarray:
        if image.size == 0: return image
        b, g, r = cv2.split(image)
        avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
        
        avg_b = 1.0 if avg_b == 0 else avg_b
        avg_g = 1.0 if avg_g == 0 else avg_g
        avg_r = 1.0 if avg_r == 0 else avg_r
        
        avg_all = (avg_b + avg_g + avg_r) / 3.0
        
        scale_b = avg_all / avg_b
        scale_g = avg_all / avg_g
        scale_r = avg_all / avg_r
        
        b = cv2.convertScaleAbs(b * scale_b)
        g = cv2.convertScaleAbs(g * scale_g)
        r = cv2.convertScaleAbs(r * scale_r)
        return cv2.merge([b, g, r])

    def _detect_roi_from_image(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        if image is None or image.size == 0:
            return (0, 0, 0, 0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return (0, 0, image.shape[1], image.shape[0])
            
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        margin_x = int(w * self.config.auto_roi_margin_ratio)
        margin_y = int(h * self.config.auto_roi_margin_ratio)
        
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(image.shape[1] - x, w + 2 * margin_x)
        h = min(image.shape[0] - y, h + 2 * margin_y)
        
        return (x, y, w, h)
