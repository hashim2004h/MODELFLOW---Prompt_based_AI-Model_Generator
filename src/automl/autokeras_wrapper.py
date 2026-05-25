"""
AutoKeras Wrapper
Automated model architecture search using AutoKeras
"""

import logging
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class AutoKerasWrapper:
    """
    Wrapper for AutoKeras functionality
    """
    
    def __init__(self, task_type: str, max_trials: int = 10):
        """
        Initialize AutoKeras wrapper
        
        Args:
            task_type: Type of task
            max_trials: Maximum number of trials for search
        """
        self.task_type = task_type
        self.max_trials = max_trials
        self.model = None
    
    def search_image_classifier(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: Optional[np.ndarray] = None,
                                y_val: Optional[np.ndarray] = None,
                                epochs: int = 10) -> None:
        """
        Search for best image classification model
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Epochs per trial
        """
        try:
            import autokeras as ak
            
            # Create image classifier
            self.model = ak.ImageClassifier(
                max_trials=self.max_trials,
                overwrite=True,
                directory='autokeras_models',
                project_name='image_classifier'
            )
            
            # Search for best model
            logger.info(f"Starting AutoKeras search with {self.max_trials} trials")
            
            self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=epochs,
                verbose=1
            )
            
            logger.info("AutoKeras search completed")
            
        except ImportError:
            logger.error("AutoKeras not installed. Install with: pip install autokeras")
            raise
        except Exception as e:
            logger.error(f"Error in AutoKeras search: {e}")
            raise
    
    def search_text_classifier(self, X_train, y_train,
                              X_val=None, y_val=None,
                              epochs: int = 10) -> None:
        """
        Search for best text classification model
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            epochs: Epochs per trial
        """
        try:
            import autokeras as ak
            
            self.model = ak.TextClassifier(
                max_trials=self.max_trials,
                overwrite=True,
                directory='autokeras_models',
                project_name='text_classifier'
            )
            
            logger.info(f"Starting AutoKeras text search with {self.max_trials} trials")
            
            self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=epochs,
                verbose=1
            )
            
            logger.info("AutoKeras text search completed")
            
        except ImportError:
            logger.error("AutoKeras not installed")
            raise
        except Exception as e:
            logger.error(f"Error in AutoKeras text search: {e}")
            raise
    
    def search_structured_data_classifier(self, X_train: np.ndarray, y_train: np.ndarray,
                                         X_val: Optional[np.ndarray] = None,
                                         y_val: Optional[np.ndarray] = None,
                                         epochs: int = 10) -> None:
        """
        Search for best structured data classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Epochs per trial
        """
        try:
            import autokeras as ak
            
            self.model = ak.StructuredDataClassifier(
                max_trials=self.max_trials,
                overwrite=True,
                directory='autokeras_models',
                project_name='structured_data_classifier'
            )
            
            logger.info(f"Starting AutoKeras structured data search with {self.max_trials} trials")
            
            self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=epochs,
                verbose=1
            )
            
            logger.info("AutoKeras structured data search completed")
            
        except ImportError:
            logger.error("AutoKeras not installed")
            raise
        except Exception as e:
            logger.error(f"Error in AutoKeras search: {e}")
            raise
    
    def get_best_model(self):
        """
        Get the best model from search
        
        Returns:
            Best Keras model
        """
        if self.model is None:
            raise ValueError("No model found. Run search first.")
        
        return self.model.export_model()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Evaluate the best model
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model found. Run search first.")
        
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("No model found. Run search first.")
        
        return self.model.predict(X)
