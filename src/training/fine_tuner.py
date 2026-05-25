"""
Fine-Tuning Module
Transfer learning and model fine-tuning utilities
"""

import logging
from typing import Optional, List
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


class FineTuner:
    """
    Fine-tune pretrained models on custom datasets
    """
    
    def __init__(self, base_model: keras.Model):
        """
        Initialize fine-tuner
        
        Args:
            base_model: Pretrained base model
        """
        self.base_model = base_model
        self.model = None
        self.trainable_layers = []
    
    def freeze_base_model(self):
        """Freeze all layers in the base model"""
        self.base_model.trainable = False
        logger.info("Base model frozen")
    
    def unfreeze_base_model(self):
        """Unfreeze all layers in the base model"""
        self.base_model.trainable = True
        logger.info("Base model unfrozen")
    
    def unfreeze_top_layers(self, num_layers: int):
        """
        Unfreeze only the top N layers
        
        Args:
            num_layers: Number of top layers to unfreeze
        """
        # Freeze all layers first
        self.base_model.trainable = True
        
        # Then freeze all except top layers
        total_layers = len(self.base_model.layers)
        for layer in self.base_model.layers[:-num_layers]:
            layer.trainable = False
        
        logger.info(f"Unfroze top {num_layers} layers")
    
    def add_classification_head(self, num_classes: int, 
                               hidden_units: Optional[List[int]] = None):
        """
        Add classification head on top of base model
        
        Args:
            num_classes: Number of output classes
            hidden_units: Optional list of hidden layer sizes
        """
        if hidden_units is None:
            hidden_units = [256]
        
        # Create new model with classification head
        inputs = self.base_model.input
        x = self.base_model.output
        
        # Add global pooling if needed
        if len(x.shape) > 2:
            x = keras.layers.GlobalAveragePooling2D()(x)
        
        # Add hidden layers
        for units in hidden_units:
            x = keras.layers.Dense(units, activation='relu')(x)
            x = keras.layers.Dropout(0.5)(x)
        
        # Output layer
        outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        logger.info(f"Added classification head with {num_classes} classes")
    
    def compile_for_transfer_learning(self, learning_rate: float = 0.001):
        """
        Compile model for transfer learning (frozen base)
        
        Args:
            learning_rate: Learning rate
        """
        if self.model is None:
            raise ValueError("Model not created. Call add_classification_head() first.")
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info(f"Compiled for transfer learning (lr={learning_rate})")
    
    def compile_for_fine_tuning(self, learning_rate: float = 0.0001):
        """
        Compile model for fine-tuning (unfrozen base)
        
        Args:
            learning_rate: Lower learning rate for fine-tuning
        """
        if self.model is None:
            raise ValueError("Model not created.")
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info(f"Compiled for fine-tuning (lr={learning_rate})")
    
    def get_model(self) -> keras.Model:
        """Get the complete model"""
        return self.model
    
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
        else:
            logger.warning("Model not created yet")
    
    def count_trainable_params(self) -> dict:
        """
        Count trainable and non-trainable parameters
        
        Returns:
            Dictionary with parameter counts
        """
        if self.model is None:
            return {'trainable': 0, 'non_trainable': 0, 'total': 0}
        
        trainable = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        non_trainable = sum([tf.size(w).numpy() for w in self.model.non_trainable_weights])
        
        return {
            'trainable': int(trainable),
            'non_trainable': int(non_trainable),
            'total': int(trainable + non_trainable)
        }
