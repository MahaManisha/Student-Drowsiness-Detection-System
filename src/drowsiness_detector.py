"""
Drowsiness Detection & Temporal Processing Engine.
Uses a 90-frame sliding window, Exponential Moving Average (EMA),
majority voting, and continuous low-EAR threshold monitoring to detect drowsiness.
"""
from collections import deque
import time
import numpy as np
from src.config import SETTINGS
from utils.logger import logger


class DrowsinessDetector:
    """Detects eye blinks, closed eyes, and sustained drowsiness using temporal buffers."""

    def __init__(self, buffer_size: int = 300):
        self.buffer_size = buffer_size
        self.ear_buffer = deque(maxlen=buffer_size)
        self.timestamp_buffer = deque(maxlen=buffer_size)

        self.ema_alpha = 0.25  # Exponential smoothing factor
        self.smoothed_ear = None

        # Blink tracking
        self.blink_count = 0
        self.eye_closed_frames = 0
        self.eye_open_frames = 0
        self.in_blink = False
        self.blink_start_time = None
        self.last_blink_time = None

        # Drowsiness tracking
        self.is_drowsy = False
        self.drowsy_start_time = None
        self.total_drowsy_duration = 0.0
        self.drowsy_event_count = 0

    def update(
        self, avg_ear: float, ear_threshold: float, fps: float = 30.0
    ) -> dict:
        """
        Process current frame EAR value.

        Args:
            avg_ear (float): Current Eye Aspect Ratio
            ear_threshold (float): Active EAR threshold for eyes closed
            fps (float): Measured frames per second

        Returns:
            dict containing:
                - is_drowsy (bool)
                - eyes_closed (bool)
                - smoothed_ear (float)
                - blink_count (int)
                - drowsy_duration (float)
                - event_triggered (bool)
        """
        curr_time = time.time()
        self.ear_buffer.append(avg_ear)
        self.timestamp_buffer.append(curr_time)

        # Exponential Moving Average (EMA) smoothing with dynamic initialization
        if self.smoothed_ear is None:
            self.smoothed_ear = avg_ear
        else:
            self.smoothed_ear = (self.ema_alpha * avg_ear) + ((1 - self.ema_alpha) * self.smoothed_ear)

        # Blink and Eyes Closed Logic
        is_eyes_closed = (self.smoothed_ear < ear_threshold) or (avg_ear < ear_threshold)
        adaptive_blink_threshold = ear_threshold * 1.0

        if is_eyes_closed:
            self.eye_closed_frames += 1
            self.eye_open_frames = 0
        else:
            self.eye_open_frames += 1
            if self.eye_open_frames >= 3:
                self.eye_closed_frames = 0

        # Time-based Blink Detection
        if avg_ear < adaptive_blink_threshold:
            if not self.in_blink:
                self.in_blink = True
                self.blink_start_time = curr_time
        else:
            if self.in_blink:
                if self.blink_start_time is not None:
                    blink_duration = curr_time - self.blink_start_time
                    # Broadened blink duration logic (from 20ms up to 800ms) to capture fast and slow blinks
                    if 0.02 <= blink_duration <= 0.80:
                        # Prevent double counting with a 250ms cooldown
                        if self.last_blink_time is None or (curr_time - self.last_blink_time) >= 0.25:
                            self.blink_count += 1
                            self.last_blink_time = curr_time
                self.in_blink = False
                self.blink_start_time = None

        # Temporal Drowsiness Evaluation
        required_drowsy_sec = SETTINGS["temporal"]["drowsy_duration_sec"]
        fps_clamped = max(10.0, min(60.0, fps))
        required_drowsy_frames = max(10, int(required_drowsy_sec * fps_clamped))

        event_triggered = False
        if len(self.ear_buffer) >= 1:
            window_size = min(len(self.ear_buffer), required_drowsy_frames)
            recent_frames = list(self.ear_buffer)[-window_size:]
            recent_low_count = sum(1 for e in recent_frames if e < ear_threshold)
            low_ratio = recent_low_count / float(window_size)

            drowsy_condition = (
                (self.eye_closed_frames >= required_drowsy_frames) or
                (window_size >= int(required_drowsy_frames * 0.75) and low_ratio >= 0.70)
            )

            if drowsy_condition:
                if not self.is_drowsy:
                    self.is_drowsy = True
                    self.drowsy_start_time = curr_time
                    self.drowsy_event_count += 1
                    event_triggered = True
                    logger.warning("DROWSINESS DETECTED!")
            else:
                if self.is_drowsy and self.eye_open_frames >= 5:
                    self.is_drowsy = False
                    if self.drowsy_start_time:
                        self.total_drowsy_duration += curr_time - self.drowsy_start_time
                        self.drowsy_start_time = None

        drowsy_duration = 0.0
        if self.is_drowsy and self.drowsy_start_time:
            drowsy_duration = curr_time - self.drowsy_start_time

        return {
            "is_drowsy": self.is_drowsy,
            "eyes_closed": is_eyes_closed,
            "smoothed_ear": round(float(self.smoothed_ear), 4),
            "blink_count": self.blink_count,
            "drowsy_duration": round(float(drowsy_duration), 2),
            "total_drowsy_duration": round(float(self.total_drowsy_duration + drowsy_duration), 2),
            "event_triggered": event_triggered,
        }

    def update_face_lost(self):
        """Handle face lost event to stop active drowsiness accumulation."""
        if self.is_drowsy:
            self.is_drowsy = False
            if self.drowsy_start_time:
                curr_time = time.time()
                self.total_drowsy_duration += curr_time - self.drowsy_start_time
                self.drowsy_start_time = None
        self.in_blink = False
        self.blink_start_time = None

    def reset(self):
        """Reset state tracking."""
        self.ear_buffer.clear()
        self.timestamp_buffer.clear()
        self.blink_count = 0
        self.eye_closed_frames = 0
        self.in_blink = False
        self.blink_start_time = None
        self.last_blink_time = None
        self.is_drowsy = False
        self.drowsy_start_time = None
        self.total_drowsy_duration = 0.0
        self.drowsy_event_count = 0
