"""
Custom Training Callbacks
Specialized callbacks for model training
"""

import logging
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import json

logger = logging.getLogger(__name__)


class CustomCallbacks:
    """
    Factory for creating custom training callbacks
    """
    
    @staticmethod
    def create_callbacks(training_id: str, checkpoint_dir: Path, log_dir: Path) -> list:
        """
        Create a list of training callbacks
        
        Args:
            training_id: Training session ID
            checkpoint_dir: Directory for checkpoints
            log_dir: Directory for logs
            
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Model checkpoint callback
        checkpoint_path = checkpoint_dir / f"{training_id}_{{epoch:02d}}.h5"
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # TensorBoard callback
        tensorboard_dir = log_dir / 'tensorboard' / training_id
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard_callback)
        
        # Learning rate scheduler
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # Custom progress callback
        progress_callback = TrainingProgressCallback(training_id, log_dir)
        callbacks.append(progress_callback)
        
        logger.info(f"Created {len(callbacks)} callbacks for training")
        return callbacks


class TrainingProgressCallback(keras.callbacks.Callback):
    """
    Custom callback to track and log training progress
    """
    
    def __init__(self, training_id: str, log_dir: Path):
        """
        Initialize progress callback
        
        Args:
            training_id: Training session ID
            log_dir: Directory for logs
        """
        super().__init__()
        self.training_id = training_id
        self.log_file = log_dir / f"{training_id}_progress.json"
        self.epoch_logs = []
        self.start_time = None
    
    def on_train_begin(self, logs=None):
        """Called at the start of training"""
        self.start_time = datetime.now()
        logger.info(f"Training started: {self.training_id}")
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of each epoch"""
        self.epoch_start_time = datetime.now()
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        epoch_duration = (datetime.now() - self.epoch_start_time).total_seconds()
        
        epoch_log = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            'duration': epoch_duration,
            'metrics': {k: float(v) for k, v in logs.items()}
        }
        
        self.epoch_logs.append(epoch_log)
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump({
                'training_id': self.training_id,
                'epochs': self.epoch_logs
            }, f, indent=2)
        
        # Log to console
        logger.info(f"Epoch {epoch+1} completed in {epoch_duration:.2f}s - "
                   f"loss: {logs.get('loss', 0):.4f} - "
                   f"val_loss: {logs.get('val_loss', 0):.4f}")
    
    def on_train_end(self, logs=None):
        """Called at the end of training"""
        total_duration = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"Training completed in {total_duration:.2f}s")


class MetricsLogger(keras.callbacks.Callback):
    """
    Log metrics to custom file format
    """
    
    def __init__(self, log_file: Path):
        """
        Initialize metrics logger
        
        Args:
            log_file: Path to log file
        """
        super().__init__()
        self.log_file = log_file
        self.metrics_history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
    
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at end of each epoch"""
        for metric, value in logs.items():
            if metric in self.metrics_history:
                self.metrics_history[metric].append(float(value))
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)


class GradientLogger(keras.callbacks.Callback):
    """
    Log gradient information during training
    """
    
    def __init__(self, log_frequency: int = 10):
        """
        Initialize gradient logger
        
        Args:
            log_frequency: How often to log (in batches)
        """
        super().__init__()
        self.log_frequency = log_frequency
        self.batch_count = 0
    
    def on_batch_end(self, batch, logs=None):
        """Log gradients periodically"""
        self.batch_count += 1
        
        if self.batch_count % self.log_frequency == 0:
            # Get gradients (if available)
            if hasattr(self.model, 'optimizer'):
                logger.debug(f"Batch {self.batch_count}: loss={logs.get('loss', 0):.4f}")
