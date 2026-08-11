"""
Analytics & Session Summary Module.
Aggregates timeline metrics, computes session Focus % and Alertness %,
determines Risk Level, and generates personalized recommendations.
"""
import time
import numpy as np


class SessionAnalytics:
    """Aggregates real-time metrics across session duration."""

    def __init__(self):
        self.start_time = time.time()
        self.history_timestamps = []
        self.history_ear = []
        self.history_mar = []
        self.history_attention = []
        self.history_state = []

    def record_frame(
        self,
        timestamp: float,
        ear: float,
        mar: float,
        attention_score: float,
        state: str,
    ):
        """Record metric data point for dynamic plotting and summary."""
        self.history_timestamps.append(timestamp - self.start_time)
        self.history_ear.append(ear)
        self.history_mar.append(mar)
        self.history_attention.append(attention_score)
        self.history_state.append(state)

    def get_session_duration(self) -> float:
        """Return elapsed session duration in seconds."""
        return time.time() - self.start_time

    def compute_summary(
        self,
        blink_count: int,
        yawn_count: int,
        total_drowsy_duration: float,
        total_distraction_duration: float,
        longest_distraction: float,
        drowsy_event_count: int,
        distraction_count: int,
    ) -> dict:
        """
        Generate comprehensive session summary report.

        Returns:
            dict containing:
                - focus_percentage (float)
                - alert_percentage (float)
                - avg_attention_score (float)
                - risk_level (str): 'LOW', 'MODERATE', 'HIGH', 'CRITICAL'
                - recommendations (list[str])
        """
        duration = max(1.0, self.get_session_duration())
        avg_attention = (
            float(np.mean(self.history_attention))
            if self.history_attention
            else 100.0
        )

        # Focus Percentage = (Time in Normal Alert State / Total Session Time) * 100
        unfocused_time = total_drowsy_duration + total_distraction_duration
        focused_time = max(0.0, duration - unfocused_time)
        focus_percentage = (focused_time / duration) * 100.0

        alert_percentage = avg_attention

        # Risk Level Assessment
        if total_drowsy_duration > 10.0 or drowsy_event_count >= 3:
            risk_level = "CRITICAL"
        elif total_drowsy_duration > 3.0 or drowsy_event_count >= 1 or yawn_count >= 4:
            risk_level = "HIGH"
        elif total_distraction_duration > 15.0 or distraction_count >= 4 or yawn_count >= 2:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        # Dynamic Recommendations
        recommendations = []
        if risk_level in ["HIGH", "CRITICAL"]:
            recommendations.append(
                "[ALERT] High Drowsiness Detected: Take an immediate 15-minute rest or walk."
            )
        if yawn_count >= 3:
            recommendations.append(
                "[YAWN] Frequent Yawning: Hydrate and ensure proper ventilation in your study area."
            )
        if distraction_count >= 3 or total_distraction_duration > 15.0:
            recommendations.append(
                "[DISTRACTION] High Distraction: Move mobile phone away and keep screen aligned at eye level."
            )
        if blink_count / (duration / 60.0) < 10:
            recommendations.append(
                "[EYE STRAIN] Low Blink Rate Detected: Perform eye strain exercises (20-20-20 rule)."
            )
        if not recommendations:
            recommendations.append(
                "[OK] Excellent Focus! Maintain current posture and study interval habits."
            )

        return {
            "session_duration_sec": round(duration, 1),
            "focus_percentage": round(focus_percentage, 1),
            "alert_percentage": round(alert_percentage, 1),
            "avg_attention_score": round(avg_attention, 1),
            "total_blinks": blink_count,
            "total_yawns": yawn_count,
            "total_drowsy_duration": round(total_drowsy_duration, 1),
            "total_distraction_duration": round(total_distraction_duration, 1),
            "longest_distraction": round(longest_distraction, 1),
            "drowsy_event_count": drowsy_event_count,
            "distraction_count": distraction_count,
            "risk_level": risk_level,
            "recommendations": recommendations,
        }

    def reset(self):
        """Reset session tracking."""
        self.start_time = time.time()
        self.history_timestamps.clear()
        self.history_ear.clear()
        self.history_mar.clear()
        self.history_attention.clear()
        self.history_state.clear()
