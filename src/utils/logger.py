"""
Logging Configuration
Setup and configure application logging
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

from config import LOGS_DIR


def setup_logger(name: str = 'modelflow', 
                level: int = logging.INFO,
                log_to_file: bool = True) -> logging.Logger:
    """
    Setup and configure logger
    
    Args:
        name: Logger name
        level: Logging level
        log_to_file: Whether to log to file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        log_file = LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"Logger '{name}' initialized")
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger or create new one
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerContextManager:
    """
    Context manager for temporary logging level changes
    """
    
    def __init__(self, logger: logging.Logger, level: int):
        """
        Initialize context manager
        
        Args:
            logger: Logger instance
            level: Temporary logging level
        """
        self.logger = logger
        self.new_level = level
        self.old_level = logger.level
    
    def __enter__(self):
        """Enter context"""
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context"""
        self.logger.setLevel(self.old_level)
