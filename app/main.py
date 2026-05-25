"""
Main Application Logic
Handles core application functionality
"""
import logging
from flask import Flask
from config import create_directories

logger = logging.getLogger(__name__)

def create_app():
    """
    Create and configure the Flask application
    """
    from app import create_app as app_factory
    
    # Initialize directories
    try:
        create_directories()
        logger.info("Application directories initialized")
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
    
    # Create Flask app
    app = app_factory()
    
    # Additional initialization
    with app.app_context():
        # Initialize database (if needed)
        # Initialize model cache
        logger.info("Application initialized successfully")
    
    return app

def initialize_models():
    """
    Pre-load frequently used models for faster inference
    """
    from src.models.pretrained_loader import PretrainedLoader
    
    loader = PretrainedLoader()
    # Pre-load common models
    logger.info("Pre-loading common models...")
    # This can be expanded based on usage patterns
