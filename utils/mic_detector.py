"""
Microphone Audio Activity Detector.
Runs a background thread using sounddevice to read mic inputs,
compute RMS, and detect vocal activity via dynamic noise-floor calculations.
"""
import time
import threading
import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except (ImportError, OSError):
    sd = None
    HAS_SOUNDDEVICE = False

from utils.logger import logger



class MicDetector:
    """Monitors local microphone level in a background thread to detect speaking activity."""

    def __init__(self, sample_rate: int = 16000, block_size: int = 1024):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.current_volume = 0.0
        self.current_db = -90.0
        self.is_talking = False
        self.is_running = False
        self.thread = None
        self.stream = None
        self.noise_floor = 0.005
        self.volume_history = []

    def _audio_callback(self, indata, frames, time_info, status):
        """Processes audio input buffers from sounddevice stream."""
        if status:
            pass
        
        # Calculate RMS volume
        rms = np.sqrt(np.mean(indata**2))
        self.current_volume = float(rms)
        
        # Convert to Decibels (dB FS)
        if rms > 1e-6:
            self.current_db = float(20 * np.log10(rms))
        else:
            self.current_db = -90.0

        # Maintain volume history for dynamic noise floor calculation
        self.volume_history.append(rms)
        if len(self.volume_history) > 100:
            self.volume_history.pop(0)
        
        # Determine dynamic noise floor (10th percentile to ignore speech spikes)
        self.noise_floor = np.percentile(self.volume_history, 10)
        
        # Speech detection logic: voice must be significantly above noise floor and minimum volume
        # Minimum threshold of 0.015 prevents faint fan noise from registering
        self.is_talking = rms > max(0.018, self.noise_floor * 3.0)

    def start(self):
        """Start background recording thread."""
        if not HAS_SOUNDDEVICE or sd is None:
            logger.warning("PortAudio or sounddevice not available. Mic detection disabled.")
            return

        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._run_stream, daemon=True)
        self.thread.start()
        logger.info("Microphone voice detector started.")

    def _run_stream(self):
        """Active recording loop within background thread."""
        if not HAS_SOUNDDEVICE or sd is None:
            self.is_running = False
            return
        try:
            # Query standard input device
            device_info = sd.query_devices(kind='input')

            if not device_info:
                logger.error("No input audio devices found.")
                self.is_running = False
                return

            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self._audio_callback,
                blocksize=self.block_size
            )
            with self.stream:
                while self.is_running:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Microphone recording failed: {e}")
            self.is_running = False

    def stop(self):
        """Stop mic monitoring and cleanup resource streams."""
        self.is_running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        
        if self.thread is not None:
            self.thread.join(timeout=0.5)
            self.thread = None
        
        self.is_talking = False
        self.current_volume = 0.0
        self.current_db = -90.0
        logger.info("Microphone voice detector stopped.")
