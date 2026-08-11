"""
CSV Exporter and Screenshot Generator Module.
Exports frame-by-frame telemetry logs to CSV and captures automated screenshots on alert events.
"""
import csv
import datetime
import os
import cv2
import pandas as pd
from src.config import REPORTS_DIR, SCREENSHOTS_DIR
from utils.logger import logger


class SessionExporter:
    """Manages CSV session report creation and frame screenshot saving."""

    def __init__(self):
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"session_report_{timestamp_str}.csv"
        self.csv_filepath = os.path.join(REPORTS_DIR, self.csv_filename)
        self.file_initialized = False

        self.last_log_time = 0.0
        self.last_state = None
        self.last_blink = 0
        self.last_yawn = 0

        self._init_csv()

    def _init_csv(self):
        """Initialize CSV file with required header columns."""
        try:
            headers = [
                "Timestamp",
                "EAR",
                "MAR",
                "Yaw",
                "Pitch",
                "Roll",
                "State",
                "Expression",
                "Blink",
                "Yawn",
                "Attention Score",
            ]
            with open(self.csv_filepath, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            self.file_initialized = True
            logger.info(f"Initialized CSV Report: {self.csv_filepath}")
        except Exception as e:
            logger.error(f"Failed to initialize CSV exporter: {e}")

    def log_frame(
        self,
        timestamp_str: str,
        ear: float,
        mar: float,
        yaw: float,
        pitch: float,
        roll: float,
        state: str,
        expression: str,
        blink: int,
        yawn: int,
        attention_score: float,
    ):
        """Append frame telemetry record to CSV, rate-limited to 1 Hz or on state/event changes."""
        if not self.file_initialized:
            return

        import time
        curr_time = time.time()

        # Check if 1.0 second has elapsed or if critical event/state has changed
        time_elapsed = (curr_time - self.last_log_time) >= 1.0
        state_changed = (state != self.last_state)
        blink_incremented = (blink != self.last_blink)
        yawn_incremented = (yawn != self.last_yawn)

        if not (time_elapsed or state_changed or blink_incremented or yawn_incremented):
            return

        self.last_log_time = curr_time
        self.last_state = state
        self.last_blink = blink
        self.last_yawn = yawn

        try:
            row = [
                timestamp_str,
                f"{ear:.4f}",
                f"{mar:.4f}",
                f"{yaw:.2f}",
                f"{pitch:.2f}",
                f"{roll:.2f}",
                state,
                expression,
                blink,
                yawn,
                f"{attention_score:.1f}",
            ]
            with open(self.csv_filepath, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            logger.error(f"Error logging frame to CSV: {e}")

    def capture_drowsy_screenshot(self, frame) -> str:
        """
        Save frame image to reports/screenshots/ when drowsiness is triggered.

        Returns:
            filepath (str): Saved image path
        """
        if frame is None:
            return ""

        try:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]
            filename = f"drowsy_alert_{timestamp_str}.jpg"
            filepath = os.path.join(SCREENSHOTS_DIR, filename)
            cv2.imwrite(filepath, frame)
            logger.warning(f"Saved Drowsy Screenshot: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error capturing screenshot: {e}")
            return ""

    def get_csv_dataframe(self) -> pd.DataFrame:
        """Read current session CSV into a pandas DataFrame."""
        if os.path.exists(self.csv_filepath):
            try:
                return pd.read_csv(self.csv_filepath)
            except Exception:
                pass
        return pd.DataFrame()
