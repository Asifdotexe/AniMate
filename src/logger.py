"""
This module provides utility functions for RecommendationHaki.
"""

import gc
import logging
import sys

import psutil

from src import config as config_module

def setup_logging(name: str) -> logging.Logger:
    """
    Setup a logger with a standard console handler.

    :param name: Name of the logger.
    :return: Configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        try:
            # Lazy load config to avoid import-time side effects
            debug_mode = config_module.config.app.debug
        except (AttributeError, ImportError, FileNotFoundError) as e:
            # Fallback if config fails to load
            print(f"Warning: Could not load config for logging ({e}). Defaulting to INFO.")
            debug_mode = False

        level = logging.DEBUG if debug_mode else logging.INFO
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def get_memory_usage() -> float:
    """
    Get the current system memory usage percentage.

    :returns: Memory usage percentage.
    """
    gc.collect()
    return psutil.virtual_memory().percent
