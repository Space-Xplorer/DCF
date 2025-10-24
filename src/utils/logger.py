#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging module: centralized logging configuration for all modules.
"""

import logging
from .config import LOG_LEVEL, LOG_FORMAT

def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance for a module.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only add handler if not already present (avoid duplicates)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    return logger

# Create default loggers for common modules
dcf_logger = get_logger("dcf")
data_logger = get_logger("data")
api_logger = get_logger("api")
