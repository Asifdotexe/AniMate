"""
This module provides utility functions for AniMate.
"""

import gc
import logging
import sys

import psutil

from src.config import config


def setup_logging(name: str) -> logging.Logger:
    """
    Setup a logger with a standard console handler.

    :param name: Name of the logger.
    :return: Configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = logging.DEBUG if config.app.debug else logging.INFO
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
