"""
Logger utility for Student Drowsiness & Attention Monitoring System.
"""
import logging
import os
import sys
from src.config import LOGS_DIR


def setup_logger(name: str = "DrowsinessMonitor") -> logging.Logger:
    """Setup structured logger writing to console and file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    # Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    c_handler.setFormatter(c_format)

    # File Handler
    log_file = os.path.join(LOGS_DIR, "app.log")
    f_handler = logging.FileHandler(log_file, encoding="utf-8")
    f_handler.setLevel(logging.DEBUG)
    f_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
    )
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


logger = setup_logger()
