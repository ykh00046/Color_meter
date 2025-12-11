import cv2
import numpy as np
from dataclasses import dataclass
from scipy.signal import savgol_filter
from src.core.lens_detector import LensDetection

@dataclass
class RadialProfile:
    r_normalized: np.ndarray
    L: np.ndarray
    a: np.ndarray
    b: np.ndarray
    std_L: np.ndarray
    std_a: np.ndarray
    std_b: np.ndarray
    pixel_count: np.ndarray

@dataclass
class ProfilerConfig:
    r_start_ratio: float = 0.0
    r_end_ratio: float = 1.0
    r_step_pixels: int = 1
    theta_samples: int = 360
    smoothing_enabled: bool = True
    smoothing_method: str = "savgol"
    savgol_window_length: int = 11
    savgol_polyorder: int = 3
    moving_average_window: int = 5

class RadialProfiler:
    def __init__(self, config: ProfilerConfig = ProfilerConfig()):
        self.config = config

    def extract_profile(self, image: np.ndarray, lens: LensDetection) -> RadialProfile:
        if image is None: raise ValueError("Input image cannot be None.")
        if not lens: raise ValueError("LensDetection object cannot be None.")

        cx, cy, radius = lens.center_x, lens.center_y, lens.radius
        r_samples = int(radius / self.config.r_step_pixels)
        if r_samples < 1: r_samples = 1

        polar_image = cv2.warpPolar(
            image, (r_samples, self.config.theta_samples),
            (cx, cy), radius,
            cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS
        )

        if polar_image.size == 0:
            dummy_r = np.linspace(0, 1, 10)
            zeros = np.zeros_like(dummy_r)
            return RadialProfile(dummy_r, zeros, zeros, zeros, zeros, zeros, zeros, np.zeros_like(dummy_r, dtype=int))

        polar_lab = cv2.cvtColor(polar_image, cv2.COLOR_BGR2LAB)
        
        # axis=1 (theta 방향) 평균
        L_profile = polar_lab[:, :, 0].mean(axis=1)
        a_profile = polar_lab[:, :, 1].mean(axis=1)
        b_profile = polar_lab[:, :, 2].mean(axis=1)
        
        std_L = polar_lab[:, :, 0].std(axis=1)
        std_a = polar_lab[:, :, 1].std(axis=1)
        std_b = polar_lab[:, :, 2].std(axis=1)

        pixel_count = np.full_like(L_profile, self.config.theta_samples)
        r_normalized = np.linspace(0.0, 1.0, r_samples)
        
        # Crop
        start_idx = int(r_samples * self.config.r_start_ratio)
        end_idx = int(r_samples * self.config.r_end_ratio)
        
        profile = RadialProfile(
            r_normalized=r_normalized[start_idx:end_idx],
            L=L_profile[start_idx:end_idx],
            a=a_profile[start_idx:end_idx],
            b=b_profile[start_idx:end_idx],
            std_L=std_L[start_idx:end_idx],
            std_a=std_a[start_idx:end_idx],
            std_b=std_b[start_idx:end_idx],
            pixel_count=pixel_count[start_idx:end_idx]
        )

        if self.config.smoothing_enabled:
            profile = self._smooth_profile(profile)
            
        return profile

    def _smooth_profile(self, profile: RadialProfile) -> RadialProfile:
        L, a, b = profile.L, profile.a, profile.b
        
        if self.config.smoothing_method == "savgol" and len(L) >= self.config.savgol_window_length:
            L = savgol_filter(L, self.config.savgol_window_length, self.config.savgol_polyorder)
            a = savgol_filter(a, self.config.savgol_window_length, self.config.savgol_polyorder)
            b = savgol_filter(b, self.config.savgol_window_length, self.config.savgol_polyorder)
        elif self.config.smoothing_method == "moving_average" and len(L) >= self.config.moving_average_window:
            weights = np.ones(self.config.moving_average_window) / self.config.moving_average_window
            # mode='valid' 사용 후 패딩
            L_valid = np.convolve(L, weights, mode='valid')
            a_valid = np.convolve(a, weights, mode='valid')
            b_valid = np.convolve(b, weights, mode='valid')
            
            # 앞뒤 패딩으로 길이 맞춤
            pad_len = (self.config.moving_average_window - 1) // 2
            L = np.pad(L_valid, (pad_len, len(L) - len(L_valid) - pad_len), mode='edge')
            a = np.pad(a_valid, (pad_len, len(a) - len(a_valid) - pad_len), mode='edge')
            b = np.pad(b_valid, (pad_len, len(b) - len(b_valid) - pad_len), mode='edge')
            
        return RadialProfile(
            profile.r_normalized, L, a, b,
            profile.std_L, profile.std_a, profile.std_b, profile.pixel_count
        )
