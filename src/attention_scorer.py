"""
Attention Scorer Module.
Computes real-time Attention Score (0 to 100) based on head pose,
EAR, MAR, drowsiness state, and distraction penalties.
"""
import numpy as np


class AttentionScorer:
    """Calculates real-time composite Attention Score (0-100)."""

    def __init__(self):
        self.current_score = 100.0
        self.ema_alpha = 0.2  # Smooth transition factor

    def calculate_score(
        self,
        is_drowsy: bool,
        is_distracted: bool,
        is_yawning: bool,
        eyes_closed: bool,
        pose_state: str,
        pitch: float,
        yaw: float,
        avg_ear: float,
        ear_threshold: float,
    ) -> float:
        """
        Calculate instantaneous attention score (0-100).

        Deductions:
            - Drowsy State: -65 pts
            - Eyes Closed (non-drowsy): -35 pts
            - Distracted State: -40 pts
            - Yawning: -15 pts
            - Off-center Head Angle: up to -25 pts proportional to angle
        """
        raw_score = 100.0

        if is_drowsy:
            raw_score -= 65.0
        elif eyes_closed:
            raw_score -= 30.0

        if is_distracted:
            raw_score -= 40.0
        elif pose_state != "Normal":
            angle_dev = abs(yaw) + abs(pitch)
            penalty = min(25.0, (angle_dev / 45.0) * 25.0)
            raw_score -= penalty

        if is_yawning:
            raw_score -= 15.0

        raw_score = float(np.clip(raw_score, 0.0, 100.0))

        # Apply EMA smoothing
        self.current_score = (self.ema_alpha * raw_score) + ((1 - self.ema_alpha) * self.current_score)
        return round(float(self.current_score), 1)


attention_scorer = AttentionScorer()
