"""
Keras Tuner Wrapper
Hyperparameter tuning using Keras Tuner
"""

import logging
from typing import Callable, Optional
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


class KerasTunerWrapper:
    """
    Wrapper for Keras Tuner hyperparameter optimization
    """
    
    def __init__(self, model_builder: Callable, max_trials: int = 10):
        """
        Initialize Keras Tuner wrapper
        
        Args:
            model_builder: Function to build model with hyperparameters
            max_trials: Maximum trials for search
        """
        self.model_builder = model_builder
        self.max_trials = max_trials
        self.tuner = None
        self.best_model = None
    
    def search_hyperparameters(self, X_train, y_train,
                              X_val=None, y_val=None,
                              search_method: str = 'random',
                              epochs: int = 10) -> None:
        """
        Search for best hyperparameters
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            search_method: 'random', 'bayesian', or 'hyperband'
            epochs: Epochs per trial
        """
        try:
            import keras_tuner as kt
            
            # Create tuner based on method
            if search_method == 'random':
                self.tuner = kt.RandomSearch(
                    self.model_builder,
                    objective='val_accuracy',
                    max_trials=self.max_trials,
                    directory='keras_tuner',
                    project_name='hyperparameter_search'
                )
            elif search_method == 'bayesian':
                self.tuner = kt.BayesianOptimization(
                    self.model_builder,
                    objective='val_accuracy',
                    max_trials=self.max_trials,
                    directory='keras_tuner',
                    project_name='hyperparameter_search'
                )
            elif search_method == 'hyperband':
                self.tuner = kt.Hyperband(
                    self.model_builder,
                    objective='val_accuracy',
                    max_epochs=epochs,
                    directory='keras_tuner',
                    project_name='hyperparameter_search'
                )
            else:
                raise ValueError(f"Unknown search method: {search_method}")
            
            logger.info(f"Starting {search_method} search with {self.max_trials} trials")
            
            # Run search
            self.tuner.search(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=epochs,
                verbose=1
            )
            
            # Get best model
            self.best_model = self.tuner.get_best_models(num_models=1)[0]
            
            logger.info("Hyperparameter search completed")
            
        except ImportError:
            logger.error("Keras Tuner not installed. Install with: pip install keras-tuner")
            raise
        except Exception as e:
            logger.error(f"Error in hyperparameter search: {e}")
            raise
    
    def get_best_hyperparameters(self):
        """
        Get the best hyperparameters
        
        Returns:
            Best hyperparameters
        """
        if self.tuner is None:
            raise ValueError("No tuner found. Run search first.")
        
        best_hp = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        return best_hp.values
    
    def get_best_model(self):
        """
        Get the best model
        
        Returns:
            Best Keras model
        """
        if self.best_model is None:
            raise ValueError("No best model found. Run search first.")
        
        return self.best_model
    
    def results_summary(self, num_trials: int = 5):
        """
        Print summary of search results
        
        Args:
            num_trials: Number of top trials to show
        """
        if self.tuner is None:
            raise ValueError("No tuner found. Run search first.")
        
        self.tuner.results_summary(num_trials=num_trials)
