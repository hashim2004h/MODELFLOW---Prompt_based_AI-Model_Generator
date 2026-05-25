"""
Input Validation
Validate user inputs and data
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

from config import (
    ALLOWED_IMAGE_EXTENSIONS, 
    ALLOWED_TEXT_EXTENSIONS,
    ALLOWED_CSV_EXTENSIONS,
    MAX_UPLOAD_SIZE
)

logger = logging.getLogger(__name__)


class Validator:
    """
    Input validation utilities
    """
    
    @staticmethod
    def validate_file_upload(file_path: Path, expected_type: str) -> Tuple[bool, str]:
        """
        Validate uploaded file
        
        Args:
            file_path: Path to uploaded file
            expected_type: Expected file type ('image', 'text', 'csv')
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if file exists
        if not file_path.exists():
            return False, "File does not exist"
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > MAX_UPLOAD_SIZE:
            return False, f"File size exceeds maximum allowed size ({MAX_UPLOAD_SIZE / (1024*1024):.1f} MB)"
        
        # Check file extension
        extension = file_path.suffix.lower()
        
        if expected_type == 'image':
            if extension not in ALLOWED_IMAGE_EXTENSIONS:
                return False, f"Invalid image format. Allowed: {ALLOWED_IMAGE_EXTENSIONS}"
        elif expected_type == 'text':
            if extension not in ALLOWED_TEXT_EXTENSIONS:
                return False, f"Invalid text format. Allowed: {ALLOWED_TEXT_EXTENSIONS}"
        elif expected_type == 'csv':
            if extension not in ALLOWED_CSV_EXTENSIONS:
                return False, f"Invalid CSV format. Allowed: {ALLOWED_CSV_EXTENSIONS}"
        
        return True, ""
    
    @staticmethod
    def validate_image(image: np.ndarray) -> Tuple[bool, str]:
        """
        Validate image data
        
        Args:
            image: Image array
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if array is not empty
        if image is None or image.size == 0:
            return False, "Image is empty"
        
        # Check dimensions
        if len(image.shape) not in [2, 3]:
            return False, f"Invalid image dimensions: {image.shape}"
        
        # Check if grayscale or RGB
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            return False, f"Invalid number of channels: {image.shape[2]}"
        
        # Check data type
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            return False, f"Invalid data type: {image.dtype}"
        
        return True, ""
    
    @staticmethod
    def validate_text(text: str, min_length: int = 1, max_length: int = 10000) -> Tuple[bool, str]:
        """
        Validate text input
        
        Args:
            text: Input text
            min_length: Minimum text length
            max_length: Maximum text length
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not text:
            return False, "Text is empty"
        
        text_length = len(text)
        
        if text_length < min_length:
            return False, f"Text too short (minimum {min_length} characters)"
        
        if text_length > max_length:
            return False, f"Text too long (maximum {max_length} characters)"
        
        return True, ""
    
    @staticmethod
    def validate_model_config(config: dict) -> Tuple[bool, str]:
        """
        Validate model configuration
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ['task_type', 'input_type']
        
        for field in required_fields:
            if field not in config:
                return False, f"Missing required field: {field}"
        
        # Validate task type
        from config import TASK_TYPES
        if config['task_type'] not in TASK_TYPES:
            return False, f"Invalid task type: {config['task_type']}"
        
        return True, ""
    
    @staticmethod
    def validate_training_config(config: dict) -> Tuple[bool, str]:
        """
        Validate training configuration
        
        Args:
            config: Training configuration
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check epochs
        epochs = config.get('epochs', 0)
        if epochs <= 0 or epochs > 1000:
            return False, f"Invalid epochs value: {epochs} (must be 1-1000)"
        
        # Check batch size
        batch_size = config.get('batch_size', 0)
        if batch_size <= 0 or batch_size > 256:
            return False, f"Invalid batch size: {batch_size} (must be 1-256)"
        
        # Check learning rate
        learning_rate = config.get('learning_rate', 0)
        if learning_rate <= 0 or learning_rate > 1:
            return False, f"Invalid learning rate: {learning_rate} (must be 0-1)"
        
        return True, ""
    
    @staticmethod
    def validate_dataset_path(dataset_path: Path) -> Tuple[bool, str]:
        """
        Validate dataset path
        
        Args:
            dataset_path: Path to dataset
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not dataset_path.exists():
            return False, f"Dataset path does not exist: {dataset_path}"
        
        if not dataset_path.is_dir():
            # Check if it's a valid file
            if not dataset_path.is_file():
                return False, f"Invalid dataset path: {dataset_path}"
        
        return True, ""
