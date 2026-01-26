"""
This module provides utility functions for AniMate.
"""

import gc

import psutil


def get_memory_usage() -> float:
    """
    Get the current system memory usage percentage.

    :returns: Memory usage percentage.
    """
    gc.collect()
    return psutil.virtual_memory().percent
