"""
Batch Predictor
Handle batch predictions efficiently
"""

import logging
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm

from src.inference.predictor import Predictor

logger = logging.getLogger(__name__)


class BatchPredictor:
    """
    Efficient batch prediction handling
    """
    
    def __init__(self, batch_size: int = 32):
        """
        Initialize batch predictor
        
        Args:
            batch_size: Batch size for predictions
        """
        self.batch_size = batch_size
        self.predictor = Predictor()
    
    def predict_batch(self, model_id: str, input_data_list: List[Any]) -> List[Dict]:
        """
        Predict for multiple inputs
        
        Args:
            model_id: Model ID
            input_data_list: List of input data
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Process in batches
        num_batches = (len(input_data_list) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Processing {len(input_data_list)} items in {num_batches} batches")
        
        for i in tqdm(range(0, len(input_data_list), self.batch_size)):
            batch = input_data_list[i:i + self.batch_size]
            
            # Predict for each item in batch
            for item in batch:
                result = self.predictor.predict(model_id, item)
                results.append(result)
        
        logger.info(f"Batch prediction completed: {len(results)} results")
        return results
    
    def predict_batch_images(self, model_id: str, images: np.ndarray) -> List[Dict]:
        """
        Optimized batch prediction for images
        
        Args:
            model_id: Model ID
            images: Array of images [N, H, W, C]
            
        Returns:
            List of prediction results
        """
        from src.models.model_manager import ModelManager
        
        model_manager = ModelManager()
        model = model_manager.get_model(model_id)
        
        if model is None:
            raise ValueError(f"Model {model_id} not found")
        
        # Preprocess all images at once
        images_normalized = images / 255.0
        
        # Predict in batches
        results = []
        for i in range(0, len(images), self.batch_size):
            batch = images_normalized[i:i + self.batch_size]
            predictions = model.predict(batch, verbose=0)
            
            # Parse predictions
            for j, pred in enumerate(predictions):
                predicted_class = int(np.argmax(pred))
                confidence = float(pred[predicted_class])
                
                results.append({
                    'index': i + j,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'all_predictions': pred.tolist()
                })
        
        return results
    
    def predict_batch_texts(self, model_id: str, texts: List[str]) -> List[Dict]:
        """
        Optimized batch prediction for texts
        
        Args:
            model_id: Model ID
            texts: List of text strings
            
        Returns:
            List of prediction results
        """
        from src.models.model_manager import ModelManager
        import torch
        
        model_manager = ModelManager()
        model = model_manager.get_model(model_id)
        metadata = model_manager.get_metadata(model_id)
        
        tokenizer = metadata.get('tokenizer')
        if tokenizer is None:
            raise ValueError("Tokenizer not found")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
            
            # Parse results
            for j, (text, probs) in enumerate(zip(batch_texts, probabilities)):
                predicted_class = int(torch.argmax(probs).item())
                confidence = float(probs[predicted_class].item())
                
                results.append({
                    'index': i + j,
                    'text': text,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'all_probabilities': probs.cpu().numpy().tolist()
                })
        
        return results
    
    def predict_from_file(self, model_id: str, file_path: str, 
                         data_type: str = 'image') -> List[Dict]:
        """
        Predict from a file containing multiple samples
        
        Args:
            model_id: Model ID
            file_path: Path to data file
            data_type: Type of data ('image', 'text', 'csv')
            
        Returns:
            List of prediction results
        """
        from src.utils.data_loader import DataLoader
        
        data_loader = DataLoader()
        data = data_loader.load_from_file(file_path, data_type)
        
        return self.predict_batch(model_id, data)
