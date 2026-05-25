"""
Metrics Calculator
Calculate and track model performance metrics
"""

import logging
import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate performance metrics for different tasks
    """
    
    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, 
                                        y_pred: np.ndarray,
                                        average: str = 'weighted') -> Dict:
        """
        Calculate classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging method for multi-class
            
        Returns:
            Dictionary with metrics
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average=average, zero_division=0))
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        logger.info(f"Classification metrics: Accuracy={metrics['accuracy']:.4f}, "
                   f"F1={metrics['f1_score']:.4f}")
        
        return metrics
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, 
                                    y_pred: np.ndarray) -> Dict:
        """
        Calculate regression metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2_score': float(r2)
        }
        
        logger.info(f"Regression metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        
        return metrics
    
    @staticmethod
    def calculate_detection_metrics(true_boxes: List, pred_boxes: List,
                                   iou_threshold: float = 0.5) -> Dict:
        """
        Calculate object detection metrics
        
        Args:
            true_boxes: List of true bounding boxes
            pred_boxes: List of predicted bounding boxes
            iou_threshold: IoU threshold for matching
            
        Returns:
            Dictionary with metrics
        """
        # Simplified detection metrics
        # In production, use libraries like pycocotools
        
        metrics = {
            'num_true_boxes': len(true_boxes),
            'num_pred_boxes': len(pred_boxes),
            'iou_threshold': iou_threshold
        }
        
        return metrics
    
    @staticmethod
    def calculate_segmentation_metrics(y_true_mask: np.ndarray,
                                      y_pred_mask: np.ndarray) -> Dict:
        """
        Calculate segmentation metrics
        
        Args:
            y_true_mask: True segmentation mask
            y_pred_mask: Predicted segmentation mask
            
        Returns:
            Dictionary with metrics
        """
        # Flatten masks
        y_true_flat = y_true_mask.flatten()
        y_pred_flat = y_pred_mask.flatten()
        
        # Calculate IoU (Intersection over Union)
        intersection = np.sum(y_true_flat == y_pred_flat)
        union = len(y_true_flat)
        iou = intersection / union if union > 0 else 0
        
        # Pixel accuracy
        pixel_accuracy = np.mean(y_true_flat == y_pred_flat)
        
        metrics = {
            'iou': float(iou),
            'pixel_accuracy': float(pixel_accuracy)
        }
        
        logger.info(f"Segmentation metrics: IoU={iou:.4f}, Pixel Accuracy={pixel_accuracy:.4f}")
        
        return metrics
    
    @staticmethod
    def get_classification_report(y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 target_names: List[str] = None) -> str:
        """
        Get detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Class names
            
        Returns:
            Classification report string
        """
        report = classification_report(y_true, y_pred, target_names=target_names)
        return report
    
    @staticmethod
    def calculate_top_k_accuracy(y_true: np.ndarray, 
                                y_pred_proba: np.ndarray,
                                k: int = 5) -> float:
        """
        Calculate top-k accuracy
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            k: Top k predictions to consider
            
        Returns:
            Top-k accuracy
        """
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
        
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
        
        top_k_acc = correct / len(y_true)
        logger.info(f"Top-{k} accuracy: {top_k_acc:.4f}")
        
        return float(top_k_acc)
