"""
Logging configuration for the application
"""
import sys
from pathlib import Path
from loguru import logger
from config.settings import settings


def setup_logger():
    """Configure loguru logger with custom settings"""
    
    # Remove default logger
    logger.remove()
    
    # Add console handler with custom format
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL,
        colorize=True
    )
    
    # Add file handler if log file is specified
    if settings.LOG_FILE:
        log_path = Path(settings.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            settings.LOG_FILE,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=settings.LOG_LEVEL,
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )
    
    return logger


# Initialize logger
log = setup_logger()