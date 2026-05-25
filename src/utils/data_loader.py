"""
Data Loader
Load and prepare datasets for training and inference
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List
import cv2
import tensorflow as tf

from config import UPLOADED_DATA_DIR, SAMPLES_DATA_DIR

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load data from various sources and formats
    """
    
    def __init__(self):
        """Initialize data loader"""
        pass
    
    def load_dataset(self, dataset_path: str, config: dict) -> Tuple:
        """
        Load dataset based on configuration
        
        Args:
            dataset_path: Path to dataset
            config: Configuration dictionary
            
        Returns:
            Tuple of (train_data, val_data)
        """
        data_type = config.get('data_type', 'image')
        
        if data_type == 'image':
            return self.load_image_dataset(dataset_path, config)
        elif data_type == 'text':
            return self.load_text_dataset(dataset_path, config)
        elif data_type == 'csv' or data_type == 'tabular':
            return self.load_tabular_dataset(dataset_path, config)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    def load_image_dataset(self, dataset_path: str, config: dict) -> Tuple:
        """
        Load image dataset
        
        Args:
            dataset_path: Path to image directory
            config: Configuration
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        image_size = config.get('image_size', (224, 224))
        batch_size = config.get('batch_size', 32)
        validation_split = config.get('validation_split', 0.2)
        
        # Load using tf.keras.utils.image_dataset_from_directory
        train_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            validation_split=validation_split,
            subset='training',
            seed=123,
            image_size=image_size,
            batch_size=batch_size
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            validation_split=validation_split,
            subset='validation',
            seed=123,
            image_size=image_size,
            batch_size=batch_size
        )
        
        # Normalize images
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
        
        # Optimize performance
        train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        
        logger.info(f"Loaded image dataset from {dataset_path}")
        return train_ds, val_ds
    
    def load_text_dataset(self, dataset_path: str, config: dict) -> Tuple:
        """
        Load text dataset
        
        Args:
            dataset_path: Path to text data (CSV/JSON)
            config: Configuration
            
        Returns:
            Tuple of (train_data, val_data)
        """
        # Load from CSV
        df = pd.read_csv(dataset_path)
        
        text_column = config.get('text_column', 'text')
        label_column = config.get('label_column', 'label')
        
        texts = df[text_column].tolist()
        labels = df[label_column].tolist()
        
        # Split into train/val
        validation_split = config.get('validation_split', 0.2)
        split_idx = int(len(texts) * (1 - validation_split))
        
        train_texts = texts[:split_idx]
        train_labels = labels[:split_idx]
        val_texts = texts[split_idx:]
        val_labels = labels[split_idx:]
        
        logger.info(f"Loaded {len(train_texts)} training and {len(val_texts)} validation texts")
        return (train_texts, train_labels), (val_texts, val_labels)
    
    def load_tabular_dataset(self, dataset_path: str, config: dict) -> Tuple:
        """
        Load tabular dataset from CSV
        
        Args:
            dataset_path: Path to CSV file
            config: Configuration
            
        Returns:
            Tuple of (train_data, val_data)
        """
        df = pd.read_csv(dataset_path)
        
        target_column = config.get('target_column', 'target')
        
        # Separate features and target
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values
        
        # Split
        from sklearn.model_selection import train_test_split
        validation_split = config.get('validation_split', 0.2)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        logger.info(f"Loaded tabular dataset: {X_train.shape[0]} train, {X_val.shape[0]} val samples")
        return (X_train, y_train), (X_val, y_val)
    
    def load_from_file(self, file_path: str, data_type: str) -> List:
        """
        Load data from a single file
        
        Args:
            file_path: Path to file
            data_type: Type of data
            
        Returns:
            List of loaded data
        """
        path = Path(file_path)
        
        if data_type == 'image':
            return [self.load_image(file_path)]
        elif data_type == 'text':
            with open(path, 'r') as f:
                return [f.read()]
        elif data_type == 'csv':
            df = pd.read_csv(path)
            return df.values.tolist()
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load a single image
        
        Args:
            image_path: Path to image
            
        Returns:
            Image as numpy array
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def load_images_from_directory(self, directory: str) -> List[np.ndarray]:
        """
        Load all images from a directory
        
        Args:
            directory: Path to directory
            
        Returns:
            List of image arrays
        """
        images = []
        path = Path(directory)
        
        for img_file in path.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image = self.load_image(str(img_file))
                images.append(image)
        
        logger.info(f"Loaded {len(images)} images from {directory}")
        return images
