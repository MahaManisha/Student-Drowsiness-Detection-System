"""
Image Preprocessing and Enhancement Module.
Handles automatic low-light detection, CLAHE contrast enhancement,
histogram equalization, and noise reduction for low-visibility environments.
"""
import cv2
import numpy as np
from src.config import SETTINGS
from utils.logger import logger


class ImageEnhancer:
    """Provides adaptive image enhancement for low-light webcam/video inputs."""

    def __init__(self):
        self.lum_threshold = SETTINGS["low_light"]["luminance_threshold"]
        self.clip_limit = SETTINGS["low_light"]["clahe_clip_limit"]
        self.grid_size = tuple(SETTINGS["low_light"]["clahe_grid_size"])
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.grid_size
        )

    def calculate_luminance(self, frame: np.ndarray) -> float:
        """Calculate mean luminance using YUV/LAB L-channel."""
        if frame is None or frame.size == 0:
            return 0.0
        # Convert BGR to Grayscale for fast mean luminance check
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def enhance_frame(self, frame: np.ndarray) -> tuple[np.ndarray, bool, float]:
        """
        Detect low light and apply CLAHE + Denoising if below threshold.

        Returns:
            enhanced_frame (np.ndarray): Processed image
            is_low_light (bool): True if enhancement was applied
            luminance (float): Measured luminance value
        """
        if frame is None or frame.size == 0:
            return frame, False, 0.0

        luminance = self.calculate_luminance(frame)
        is_low_light = luminance < self.lum_threshold

        if not is_low_light:
            return frame, False, luminance

        try:
            # Convert to LAB color space to process Luminance channel independently
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            # Apply CLAHE to L-channel
            cl_enhanced = self.clahe.apply(l_channel)

            # Merge back and convert to BGR
            enhanced_lab = cv2.merge((cl_enhanced, a_channel, b_channel))
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            # Optional subtle noise reduction for low light grain
            enhanced_bgr = cv2.fastNlMeansDenoisingColored(
                enhanced_bgr, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21
            )

            return enhanced_bgr, True, luminance
        except Exception as e:
            logger.error(f"Image enhancement error: {e}")
            return frame, True, luminance


enhancer = ImageEnhancer()
