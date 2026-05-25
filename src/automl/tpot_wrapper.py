"""
TPOT Wrapper
Automated machine learning pipeline optimization for tabular data
"""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class TPOTWrapper:
    """
    Wrapper for TPOT AutoML
    """
    
    def __init__(self, task_type: str = 'classification', 
                 generations: int = 5,
                 population_size: int = 20):
        """
        Initialize TPOT wrapper
        
        Args:
            task_type: 'classification' or 'regression'
            generations: Number of generations for evolution
            population_size: Population size per generation
        """
        self.task_type = task_type
        self.generations = generations
        self.population_size = population_size
        self.model = None
    
    def optimize_pipeline(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: Optional[np.ndarray] = None,
                         y_val: Optional[np.ndarray] = None) -> None:
        """
        Optimize ML pipeline using genetic programming
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        try:
            from tpot import TPOTClassifier, TPOTRegressor
            
            logger.info(f"Starting TPOT optimization: {self.task_type}")
            
            if self.task_type == 'classification':
                self.model = TPOTClassifier(
                    generations=self.generations,
                    population_size=self.population_size,
                    verbosity=2,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                self.model = TPOTRegressor(
                    generations=self.generations,
                    population_size=self.population_size,
                    verbosity=2,
                    random_state=42,
                    n_jobs=-1
                )
            
            # Fit TPOT
            self.model.fit(X_train, y_train)
            
            # Evaluate if validation data provided
            if X_val is not None and y_val is not None:
                score = self.model.score(X_val, y_val)
                logger.info(f"Validation score: {score:.4f}")
            
            logger.info("TPOT optimization completed")
            
        except ImportError:
            logger.error("TPOT not installed. Install with: pip install tpot")
            raise
        except Exception as e:
            logger.error(f"Error in TPOT optimization: {e}")
            raise
    
    def get_best_pipeline(self):
        """
        Get the best pipeline
        
        Returns:
            Best fitted pipeline
        """
        if self.model is None:
            raise ValueError("No model found. Run optimization first.")
        
        return self.model.fitted_pipeline_
    
    def export_pipeline(self, output_file: str = 'tpot_pipeline.py'):
        """
        Export the best pipeline as Python code
        
        Args:
            output_file: Output file path
        """
        if self.model is None:
            raise ValueError("No model found. Run optimization first.")
        
        self.model.export(output_file)
        logger.info(f"Pipeline exported to {output_file}")
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("No model found. Run optimization first.")
        
        return self.model.predict(X)
    
    def score(self, X, y):
        """
        Score the model
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Score
        """
        if self.model is None:
            raise ValueError("No model found. Run optimization first.")
        
        return self.model.score(X, y)
