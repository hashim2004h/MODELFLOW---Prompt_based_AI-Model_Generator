"""
Data Preprocessing
Preprocessing utilities for different data types
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import cv2

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Data preprocessing utilities
    """
    
    def __init__(self):
        """Initialize preprocessor"""
        self.scalers = {}
        self.encoders = {}
    
    def preprocess_image(self, image: np.ndarray, 
                        target_size: Tuple[int, int] = (224, 224),
                        normalize: bool = True) -> np.ndarray:
        """
        Preprocess a single image
        
        Args:
            image: Input image
            target_size: Target size (height, width)
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            Preprocessed image
        """
        # Resize
        if image.shape[:2] != target_size:
            image = cv2.resize(image, target_size)
        
        # Normalize
        if normalize:
            image = image.astype(np.float32) / 255.0
        
        return image
    
    def preprocess_image_batch(self, images: List[np.ndarray],
                               target_size: Tuple[int, int] = (224, 224),
                               normalize: bool = True) -> np.ndarray:
        """
        Preprocess a batch of images
        
        Args:
            images: List of images
            target_size: Target size
            normalize: Whether to normalize
            
        Returns:
            Batch of preprocessed images
        """
        preprocessed = []
        
        for image in images:
            processed = self.preprocess_image(image, target_size, normalize)
            preprocessed.append(processed)
        
        return np.array(preprocessed)
    
    def preprocess_text(self, text: str, 
                       lowercase: bool = True,
                       remove_punctuation: bool = False) -> str:
        """
        Preprocess text
        
        Args:
            text: Input text
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
            
        Returns:
            Preprocessed text
        """
        if lowercase:
            text = text.lower()
        
        if remove_punctuation:
            import string
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess_tabular(self, data: pd.DataFrame,
                          target_column: Optional[str] = None,
                          scaling_method: str = 'standard') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess tabular data
        
        Args:
            data: Input dataframe
            target_column: Name of target column (if any)
            scaling_method: 'standard' or 'minmax'
            
        Returns:
            Tuple of (scaled_features, target)
        """
        # Separate features and target
        if target_column:
            X = data.drop(columns=[target_column])
            y = data[target_column].values
        else:
            X = data
            y = None
        
        # Handle categorical variables
        X = self._encode_categorical(X)
        
        # Scale numerical features
        X_scaled = self._scale_features(X, method=scaling_method)
        
        logger.info(f"Preprocessed tabular data: {X_scaled.shape}")
        return X_scaled, y
    
    def _encode_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        encoded_data = data.copy()
        
        for column in data.columns:
            if data[column].dtype == 'object':
                if column not in self.encoders:
                    self.encoders[column] = LabelEncoder()
                    encoded_data[column] = self.encoders[column].fit_transform(data[column])
                else:
                    encoded_data[column] = self.encoders[column].transform(data[column])
        
        return encoded_data
    
    def _scale_features(self, data: pd.DataFrame, method: str = 'standard') -> np.ndarray:
        """Scale numerical features"""
        if method == 'standard':
            if 'standard' not in self.scalers:
                self.scalers['standard'] = StandardScaler()
                scaled = self.scalers['standard'].fit_transform(data)
            else:
                scaled = self.scalers['standard'].transform(data)
        elif method == 'minmax':
            if 'minmax' not in self.scalers:
                self.scalers['minmax'] = MinMaxScaler()
                scaled = self.scalers['minmax'].fit_transform(data)
            else:
                scaled = self.scalers['minmax'].transform(data)
        else:
            scaled = data.values
        
        return scaled
    
    def normalize_labels(self, labels: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Convert labels to one-hot encoding
        
        Args:
            labels: Input labels
            num_classes: Number of classes
            
        Returns:
            One-hot encoded labels
        """
        from tensorflow.keras.utils import to_categorical
        return to_categorical(labels, num_classes)
    
    def denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Convert normalized image back to [0, 255]
        
        Args:
            image: Normalized image [0, 1]
            
        Returns:
            Denormalized image [0, 255]
        """
        return (image * 255).astype(np.uint8)
