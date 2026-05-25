"""
Model Predictor
Single prediction inference
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
import tensorflow as tf
import torch

logger = logging.getLogger(__name__)


class Predictor:
    """
    Handle single predictions for any model type
    """
    
    def __init__(self, model_manager=None):
        """Initialize predictor"""
        self.model_cache = {}
        self.model_manager = model_manager
        
        if not self.model_manager:
            try:
                from src.models.model_manager import ModelManager
                self.model_manager = ModelManager()
            except ImportError:
                 self.model_manager = None
    
    def predict(self, model_id: str, input_data: Any) -> Dict:
        """
        Make a prediction
        
        Args:
            model_id: ID of the model to use
            input_data: Input data for prediction
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Get model from manager
            if self.model_manager:
                model = self.model_manager.get_model(model_id)
                metadata = self.model_manager.get_metadata(model_id)
            else:
                 raise ValueError("Model Manager not initialized")
            
            if model is None:
                raise ValueError(f"Model {model_id} not found")
            
            # Route to appropriate prediction method based on task type
            task_type = metadata.get('task_type', 'unknown')
            
            if 'image' in task_type:
                result = self._predict_image(model, input_data, task_type)
            elif 'text' in task_type or 'sentiment' in task_type:
                result = self._predict_text(model, input_data, metadata)
            elif 'tabular' in task_type:
                result = self._predict_tabular(model, input_data)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
            logger.info(f"Prediction completed for model {model_id}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def _predict_image(self, model, image: np.ndarray, task_type: str) -> Dict:
        """
        Predict for image input
        
        Args:
            model: Image model
            image: Input image array
            task_type: Specific image task type
            
        Returns:
            Prediction results
        """
        # Preprocess image
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Resize if needed
        target_size = (224, 224)
        if image.shape[1:3] != target_size:
            image = tf.image.resize(image, target_size).numpy()
        
        # Normalize
        image = image / 255.0
        
        # Predict
        predictions = model.predict(image, verbose=0)
        
        if 'classification' in task_type:
            # Get class names from metadata if available
            class_names = getattr(model, 'class_names', None)
            
            predicted_class_idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_class_idx])
            
            # Default label
            label = f"Class {predicted_class_idx}"
            
            # Try to get better label
            if class_names and predicted_class_idx < len(class_names):
                label = class_names[predicted_class_idx]
            else:
                # Try ImageNet decoding for pretrained models
                try:
                    from tensorflow.keras.applications.imagenet_utils import decode_predictions
                    # decode_predictions expects (batch, classes)
                    # We need to reshape predictions to (1, 1000) if it isn't already
                    # ResNet50 output is (1, 1000)
                    decoded = decode_predictions(predictions, top=1)[0][0]
                    # decoded is (class_id, label, score)
                    label = decoded[1]
                except Exception:
                    pass
            
            # Anomaly Detection / OOD Check
            # If confidence is too low, it's likely not any of the known classes
            if confidence < 0.4:
                return {
                    'predicted_class': -1,
                    'label': "Unexpected Data",
                    'confidence': confidence,
                    'all_predictions': predictions[0].tolist(),
                    'is_anomaly': True
                }

            return {
                'predicted_class': predicted_class_idx,
                'label': label,
                'confidence': confidence,
                'all_predictions': predictions[0].tolist(),
                'is_anomaly': False
            }
        elif 'detection' in task_type:
            return self._parse_detection_output(predictions)
        elif 'segmentation' in task_type:
            return self._parse_segmentation_output(predictions)
        else:
            return {'predictions': predictions.tolist()}
    
    def _predict_text(self, model, text: str, metadata: Dict) -> Dict:
        """
        Predict for text input
        
        Args:
            model: Text model
            text: Input text
            metadata: Model metadata
            
        Returns:
            Prediction results
        """
        # Get tokenizer from metadata
        tokenizer = metadata.get('tokenizer')
        
        if tokenizer is None:
            raise ValueError("Tokenizer not found in model metadata")
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move to appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
        
        predicted_class = int(torch.argmax(probabilities, dim=1).item())
        confidence = float(probabilities[0][predicted_class].item())
        
        return {
            'text': text,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': probabilities[0].cpu().numpy().tolist()
        }
    
    def _predict_tabular(self, model, data: np.ndarray) -> Dict:
        """
        Predict for tabular input
        
        Args:
            model: Tabular model
            data: Input features
            
        Returns:
            Prediction results
        """
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=0)
        
        # Predict
        predictions = model.predict(data, verbose=0)
        
        return {
            'predictions': predictions.tolist(),
            'shape': predictions.shape
        }
    
    def _parse_detection_output(self, predictions: np.ndarray) -> Dict:
        """Parse object detection output"""
        # Simplified detection output parsing
        return {
            'detections': [],
            'num_detections': 0,
            'raw_output': predictions.tolist()
        }
    
    def _parse_segmentation_output(self, predictions: np.ndarray) -> Dict:
        """Parse segmentation output"""
        mask = np.argmax(predictions[0], axis=-1)
        
        return {
            'mask': mask.tolist(),
            'shape': mask.shape,
            'num_classes': predictions.shape[-1]
        }
