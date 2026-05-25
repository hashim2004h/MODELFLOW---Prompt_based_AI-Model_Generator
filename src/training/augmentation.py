"""
Data Augmentation
Augmentation strategies for different data types
"""

import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class DataAugmentation:
    """
    Data augmentation utilities
    """
    
    @staticmethod
    def get_image_augmentation(input_shape: Tuple[int, int, int],
                               augmentation_level: str = 'medium') -> keras.Sequential:
        """
        Create image augmentation pipeline
        
        Args:
            input_shape: Input image shape
            augmentation_level: 'light', 'medium', or 'heavy'
            
        Returns:
            Keras Sequential model with augmentation layers
        """
        if augmentation_level == 'light':
            augmentation = keras.Sequential([
                keras.layers.RandomFlip('horizontal'),
                keras.layers.RandomRotation(0.1),
            ])
        elif augmentation_level == 'medium':
            augmentation = keras.Sequential([
                keras.layers.RandomFlip('horizontal'),
                keras.layers.RandomRotation(0.2),
                keras.layers.RandomZoom(0.1),
                keras.layers.RandomContrast(0.1),
            ])
        else:  # heavy
            augmentation = keras.Sequential([
                keras.layers.RandomFlip('horizontal'),
                keras.layers.RandomFlip('vertical'),
                keras.layers.RandomRotation(0.3),
                keras.layers.RandomZoom(0.2),
                keras.layers.RandomContrast(0.2),
                keras.layers.RandomBrightness(0.2),
            ])
        
        logger.info(f"Created {augmentation_level} image augmentation pipeline")
        return augmentation
    
    @staticmethod
    def augment_image_batch(images: np.ndarray, labels: np.ndarray,
                           augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment a batch of images
        
        Args:
            images: Input images
            labels: Corresponding labels
            augmentation_factor: How many augmented versions to create
            
        Returns:
            Tuple of (augmented_images, augmented_labels)
        """
        augmented_images = [images]
        augmented_labels = [labels]
        
        for _ in range(augmentation_factor - 1):
            # Random horizontal flip
            flipped = tf.image.random_flip_left_right(images)
            
            # Random rotation
            rotated = tf.image.rot90(flipped, k=np.random.randint(0, 4))
            
            # Random brightness
            brightened = tf.image.random_brightness(rotated, max_delta=0.2)
            
            # Random contrast
            contrasted = tf.image.random_contrast(brightened, lower=0.8, upper=1.2)
            
            augmented_images.append(contrasted.numpy())
            augmented_labels.append(labels)
        
        # Concatenate all augmented data
        final_images = np.concatenate(augmented_images, axis=0)
        final_labels = np.concatenate(augmented_labels, axis=0)
        
        logger.info(f"Augmented dataset from {len(images)} to {len(final_images)} samples")
        return final_images, final_labels
    
    @staticmethod
    def mixup(images: np.ndarray, labels: np.ndarray, alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup augmentation
        
        Args:
            images: Input images
            labels: One-hot encoded labels
            alpha: Mixup parameter
            
        Returns:
            Mixed images and labels
        """
        batch_size = len(images)
        
        # Sample lambda from beta distribution
        lam = np.random.beta(alpha, alpha, batch_size)
        
        # Reshape for broadcasting
        lam = lam.reshape(batch_size, 1, 1, 1)
        
        # Shuffle indices
        indices = np.random.permutation(batch_size)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[indices]
        
        # Mix labels
        lam_labels = lam.reshape(batch_size, 1)
        mixed_labels = lam_labels * labels + (1 - lam_labels) * labels[indices]
        
        return mixed_images, mixed_labels
    
    @staticmethod
    def cutout(images: np.ndarray, mask_size: int = 16, n_holes: int = 1) -> np.ndarray:
        """
        Apply cutout augmentation
        
        Args:
            images: Input images
            mask_size: Size of cutout mask
            n_holes: Number of cutout holes
            
        Returns:
            Images with cutout applied
        """
        h, w = images.shape[1:3]
        augmented_images = images.copy()
        
        for img_idx in range(len(images)):
            for _ in range(n_holes):
                # Random position
                y = np.random.randint(h)
                x = np.random.randint(w)
                
                # Calculate cutout box
                y1 = np.clip(y - mask_size // 2, 0, h)
                y2 = np.clip(y + mask_size // 2, 0, h)
                x1 = np.clip(x - mask_size // 2, 0, w)
                x2 = np.clip(x + mask_size // 2, 0, w)
                
                # Apply cutout
                augmented_images[img_idx, y1:y2, x1:x2, :] = 0
        
        return augmented_images
    
    @staticmethod
    def text_augmentation(texts: list, method: str = 'synonym') -> list:
        """
        Augment text data
        
        Args:
            texts: List of text strings
            method: Augmentation method ('synonym', 'backtranslation', 'insertion')
            
        Returns:
            Augmented texts
        """
        # Placeholder for text augmentation
        # In production, use libraries like nlpaug or textaugment
        
        augmented = []
        for text in texts:
            if method == 'synonym':
                # Simple word replacement (placeholder)
                augmented.append(text)
            elif method == 'insertion':
                # Random word insertion (placeholder)
                augmented.append(text)
            elif method == 'deletion':
                # Random word deletion (placeholder)
                words = text.split()
                if len(words) > 3:
                    del_idx = np.random.randint(0, len(words))
                    words.pop(del_idx)
                    augmented.append(' '.join(words))
                else:
                    augmented.append(text)
            else:
                augmented.append(text)
        
        return augmented
    
    @staticmethod
    def create_augmentation_pipeline(task_type: str, level: str = 'medium'):
        """
        Create task-specific augmentation pipeline
        
        Args:
            task_type: Type of task
            level: Augmentation level
            
        Returns:
            Augmentation function
        """
        if 'image' in task_type:
            return DataAugmentation.get_image_augmentation((224, 224, 3), level)
        elif 'text' in task_type:
            return lambda x: DataAugmentation.text_augmentation(x, method='deletion')
        else:
            return None
