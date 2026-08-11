"""
Feature Extraction Module.
Computes Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) using
vectorized Euclidean distance calculations from 2D/3D MediaPipe landmarks.
"""
import numpy as np
from src.config import LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_INDICES
from utils.logger import logger


def euclidean_distance(pt1: np.ndarray, pt2: np.ndarray) -> float:
    """Calculate 2D Euclidean distance between two points."""
    return float(np.linalg.norm(pt1[:2] - pt2[:2]))


def calculate_ear(landmarks: np.ndarray, eye_indices: list[int]) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) for a single eye given facial landmarks.

    Formula:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2.0 * ||p1 - p4||)
    """
    if landmarks is None or len(landmarks) == 0:
        return 0.0

    try:
        p1 = landmarks[eye_indices[0]]  # Outer corner
        p2 = landmarks[eye_indices[1]]  # Top right
        p3 = landmarks[eye_indices[2]]  # Top left
        p4 = landmarks[eye_indices[3]]  # Inner corner
        p5 = landmarks[eye_indices[4]]  # Bottom left
        p6 = landmarks[eye_indices[5]]  # Bottom right

        # Vertical distances
        vert1 = euclidean_distance(p2, p6)
        vert2 = euclidean_distance(p3, p5)

        # Horizontal distance
        horiz = euclidean_distance(p1, p4)

        if horiz < 1e-6:
            return 0.0

        ear = (vert1 + vert2) / (2.0 * horiz)
        return float(np.clip(ear, 0.0, 1.0))
    except Exception as e:
        logger.error(f"EAR calculation error: {e}")
        return 0.0


def calculate_mar(landmarks: np.ndarray) -> float:
    """
    Calculate Mouth Aspect Ratio (MAR) for mouth opening / yawning detection.

    Formula:
        MAR = (||t1 - b1|| + ||t2 - b2|| + ||t3 - b3||) / (2.0 * ||left - right||)
    """
    if landmarks is None or len(landmarks) == 0:
        return 0.0

    try:
        p_left = landmarks[MOUTH_INDICES["left"]]
        p_right = landmarks[MOUTH_INDICES["right"]]

        p_top1 = landmarks[MOUTH_INDICES["top_1"]]
        p_bot1 = landmarks[MOUTH_INDICES["bottom_1"]]

        p_top2 = landmarks[MOUTH_INDICES["top_2"]]
        p_bot2 = landmarks[MOUTH_INDICES["bottom_2"]]

        p_top3 = landmarks[MOUTH_INDICES["top_3"]]
        p_bot3 = landmarks[MOUTH_INDICES["bottom_3"]]

        vert1 = euclidean_distance(p_top1, p_bot1)
        vert2 = euclidean_distance(p_top2, p_bot2)
        vert3 = euclidean_distance(p_top3, p_bot3)

        horiz = euclidean_distance(p_left, p_right)

        if horiz < 1e-6:
            return 0.0

        mar = (vert1 + vert2 + vert3) / (2.0 * horiz)
        return float(np.clip(mar, 0.0, 2.0))
    except Exception as e:
        logger.error(f"MAR calculation error: {e}")
        return 0.0


def extract_facial_features(landmarks: np.ndarray) -> dict:
    """
    Extract EAR (Left, Right, Avg) and MAR from landmark array.

    Returns:
        dict: {
            'left_ear': float,
            'right_ear': float,
            'avg_ear': float,
            'mar': float
        }
    """
    if landmarks is None:
        return {"left_ear": 0.0, "right_ear": 0.0, "avg_ear": 0.0, "mar": 0.0}

    left_ear = calculate_ear(landmarks, LEFT_EYE_INDICES)
    right_ear = calculate_ear(landmarks, RIGHT_EYE_INDICES)
    avg_ear = (left_ear + right_ear) / 2.0
    mar = calculate_mar(landmarks)

    return {
        "left_ear": round(left_ear, 4),
        "right_ear": round(right_ear, 4),
        "avg_ear": round(avg_ear, 4),
        "mar": round(mar, 4),
    }
