"""
Expression Analyzer Module.
Computes geometric-based facial expression signals from 3D FaceMesh landmarks.
Tracks deviations from calibrated neutral baselines to detect Happy, Surprised, Sad, Angry, and Drowsy states.
"""
import numpy as np
from utils.logger import logger


def euclidean_distance(pt1: np.ndarray, pt2: np.ndarray) -> float:
    """Calculate 2D Euclidean distance between two points."""
    return float(np.linalg.norm(pt1[:2] - pt2[:2]))


class ExpressionAnalyzer:
    """Classifies facial expressions using scale-normalized landmark distances and calibrated baselines."""

    def __init__(self):
        # Calibrated neutral baselines
        self.baseline_smile = -0.02
        self.baseline_brow_raise = 0.20
        self.baseline_brow_knit = 0.21

        # Smooth output expression
        self.current_expression = "Neutral"

    def set_baselines(self, smile: float, brow_raise: float, brow_knit: float):
        """Set personalized baselines computed during calibration."""
        self.baseline_smile = float(smile)
        self.baseline_brow_raise = float(brow_raise)
        self.baseline_brow_knit = float(brow_knit)
        logger.info(
            f"ExpressionAnalyzer calibrated baselines set: "
            f"smile={self.baseline_smile:.4f}, raise={self.baseline_brow_raise:.4f}, knit={self.baseline_brow_knit:.4f}"
        )

    def extract_expression_metrics(self, landmarks: np.ndarray) -> dict:
        """
        Extract scale-normalized metrics from 3D landmark array.

        Returns:
            dict containing:
                - smile_score (float)
                - brow_raise_score (float)
                - brow_knit_score (float)
        """
        if landmarks is None or len(landmarks) == 0:
            return {"smile_score": 0.0, "brow_raise_score": 0.0, "brow_knit_score": 0.0}

        try:
            # Face scale anchors
            face_height = euclidean_distance(landmarks[10], landmarks[152])  # Forehead to chin
            face_width = euclidean_distance(landmarks[234], landmarks[454])   # Left to right cheek

            if face_height < 1e-5 or face_width < 1e-5:
                return {"smile_score": 0.0, "brow_raise_score": 0.0, "brow_knit_score": 0.0}

            # 1. Smile Score (elevation of mouth corners relative to mouth center)
            left_corner = landmarks[61]
            right_corner = landmarks[291]
            mouth_center = (landmarks[13] + landmarks[14]) / 2.0

            # Y axis increases downwards in image coordinates.
            # Elevation = mouth_center_y - corner_y (positive when corners are higher than center)
            smile_l = mouth_center[1] - left_corner[1]
            smile_r = mouth_center[1] - right_corner[1]
            smile_score = (smile_l + smile_r) / (2.0 * face_height)

            # 2. Eyebrow Raise Score (average eye-to-brow distance)
            left_eye_top = landmarks[159]
            left_brow_peak = landmarks[70]
            right_eye_top = landmarks[386]
            right_brow_peak = landmarks[300]

            dist_l = euclidean_distance(left_brow_peak, left_eye_top)
            dist_r = euclidean_distance(right_brow_peak, right_eye_top)
            brow_raise_score = (dist_l + dist_r) / (2.0 * face_height)

            # 3. Eyebrow Knit Score (distance between inner brows)
            inner_brow_left = landmarks[55]
            inner_brow_right = landmarks[285]
            brow_knit_dist = euclidean_distance(inner_brow_left, inner_brow_right)
            brow_knit_score = brow_knit_dist / face_width

            return {
                "smile_score": round(smile_score, 4),
                "brow_raise_score": round(brow_raise_score, 4),
                "brow_knit_score": round(brow_knit_score, 4),
            }

        except Exception as e:
            logger.error(f"Error extracting expression metrics: {e}")
            return {"smile_score": 0.0, "brow_raise_score": 0.0, "brow_knit_score": 0.0}

    def classify_expression(
        self, landmarks: np.ndarray, avg_ear: float, ear_threshold: float
    ) -> str:
        """
        Classifies current facial expression based on metrics and calibrated baselines.
        """
        if landmarks is None or len(landmarks) == 0:
            return "Neutral"

        metrics = self.extract_expression_metrics(landmarks)
        smile = metrics["smile_score"]
        brow_raise = metrics["brow_raise_score"]
        brow_knit = metrics["brow_knit_score"]

        # Calculate deviations from baselines
        smile_dev = smile - self.baseline_smile
        brow_raise_dev = brow_raise - self.baseline_brow_raise
        brow_knit_dev = self.baseline_brow_knit - brow_knit  # Positive when knit (narrower than normal)

        # Classification rule engine
        if smile_dev > 0.022:
            expression = "Happy"
        elif brow_raise_dev > 0.026 and avg_ear > ear_threshold * 1.08:
            expression = "Surprised"
        elif brow_knit_dev > 0.035:
            expression = "Angry"
        elif smile_dev < -0.015 and brow_knit_dev > 0.012:
            expression = "Sad"
        elif avg_ear < ear_threshold * 1.05 and avg_ear >= (ear_threshold * 0.90):
            expression = "Drowsy"
        else:
            expression = "Neutral"

        self.current_expression = expression
        return expression


expression_analyzer = ExpressionAnalyzer()
