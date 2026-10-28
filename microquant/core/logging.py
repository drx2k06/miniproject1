"""
MicroQuant Structured Logging Configuration.
"""

import logging
import sys
from typing import Optional


def get_logger(name: str = "microquant", level: Optional[str] = "INFO") -> logging.Logger:
    """Configures and returns a standardized structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper() if level else "INFO"))
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
