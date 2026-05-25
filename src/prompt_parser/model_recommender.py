"""
Model Recommender
Recommends appropriate model architectures based on task type
"""

import logging
from typing import List, Dict, Optional

from config import IMAGE_MODELS, TEXT_MODELS

logger = logging.getLogger(__name__)


class ModelRecommender:
    """
    Recommend suitable model architectures for different ML tasks
    """
    
    def __init__(self, model_manager=None):
        """Initialize model recommender with model database"""
        self.model_database = self._build_model_database()
        
        # Initialize ModelManager if not provided
        if model_manager:
            self.model_manager = model_manager
        else:
            try:
                from src.models.model_manager import ModelManager
                self.model_manager = ModelManager()
            except ImportError:
                logger.warning("Could not import ModelManager, dynamic models will not be available")
                self.model_manager = None
    
    def _build_model_database(self) -> Dict:
        """
        Build comprehensive model database
        
        Returns:
            Dictionary mapping task types to model recommendations
        """
        return {
            'image_classification': [
                {
                    'name': 'MobileNetV2',
                    'architecture': 'mobilenet_v2',
                    'size': 'Small (14 MB)',
                    'speed': 'Fast',
                    'accuracy': 'Good',
                    'description': 'Lightweight model optimized for mobile devices',
                    'best_for': 'Real-time applications, mobile deployment',
                    'pretrained': True
                },
                {
                    'name': 'ResNet50',
                    'architecture': 'resnet50',
                    'size': 'Medium (98 MB)',
                    'speed': 'Medium',
                    'accuracy': 'Excellent',
                    'description': 'Deep residual network with strong performance',
                    'best_for': 'High accuracy requirements',
                    'pretrained': True
                },
                {
                    'name': 'EfficientNetB0',
                    'architecture': 'efficientnet_b0',
                    'size': 'Small (29 MB)',
                    'speed': 'Fast',
                    'accuracy': 'Excellent',
                    'description': 'Optimized architecture balancing accuracy and efficiency',
                    'best_for': 'Best overall performance',
                    'pretrained': True
                },
                {
                    'name': 'VGG16',
                    'architecture': 'vgg16',
                    'size': 'Large (528 MB)',
                    'speed': 'Slow',
                    'accuracy': 'Good',
                    'description': 'Classic deep CNN architecture',
                    'best_for': 'Transfer learning, feature extraction',
                    'pretrained': True
                }
            ],
            'object_detection': [
                {
                    'name': 'YOLOv8n',
                    'architecture': 'yolov8n',
                    'size': 'Small (6 MB)',
                    'speed': 'Very Fast',
                    'accuracy': 'Excellent',
                    'description': 'State-of-the-art real-time object detection (nano variant)',
                    'best_for': 'Real-time detection with limited resources',
                    'pretrained': True
                },
                {
                    'name': 'YOLOv8s',
                    'architecture': 'yolov8s',
                    'size': 'Medium (21 MB)',
                    'speed': 'Fast',
                    'accuracy': 'Excellent',
                    'description': 'State-of-the-art real-time object detection (small variant)',
                    'best_for': 'Real-time detection balancing speed and accuracy',
                    'pretrained': True
                },
                {
                    'name': 'YOLOv5',
                    'architecture': 'yolov5',
                    'size': 'Medium (27 MB)',
                    'speed': 'Very Fast',
                    'accuracy': 'Good',
                    'description': 'Real-time object detection',
                    'best_for': 'Real-time video processing',
                    'pretrained': True
                },
                {
                    'name': 'SSD MobileNet',
                    'architecture': 'ssd_mobilenet',
                    'size': 'Small (19 MB)',
                    'speed': 'Fast',
                    'accuracy': 'Good',
                    'description': 'Single Shot Detector with MobileNet backbone',
                    'best_for': 'Mobile object detection',
                    'pretrained': True
                },
                {
                    'name': 'Faster R-CNN',
                    'architecture': 'faster_rcnn',
                    'size': 'Large (167 MB)',
                    'speed': 'Slow',
                    'accuracy': 'Excellent',
                    'description': 'Two-stage detector with high accuracy',
                    'best_for': 'High precision requirements',
                    'pretrained': True
                }
            ],
            'image_segmentation': [
                {
                    'name': 'U-Net',
                    'architecture': 'unet',
                    'size': 'Medium (31 MB)',
                    'speed': 'Medium',
                    'accuracy': 'Excellent',
                    'description': 'Popular architecture for medical image segmentation',
                    'best_for': 'Biomedical images, general segmentation',
                    'pretrained': False
                },
                {
                    'name': 'DeepLabV3+',
                    'architecture': 'deeplabv3plus',
                    'size': 'Large (210 MB)',
                    'speed': 'Slow',
                    'accuracy': 'Excellent',
                    'description': 'State-of-the-art semantic segmentation',
                    'best_for': 'High-quality segmentation',
                    'pretrained': True
                }
            ],
            'image_to_text': [
                {
                    'name': 'EasyOCR',
                    'architecture': 'easyocr',
                    'size': 'Medium',
                    'speed': 'Variable',
                    'accuracy': 'Good',
                    'description': 'Extracts text from images using optical character recognition',
                    'best_for': 'Document scanning, text reading',
                    'pretrained': True
                }
            ],
            'audio_to_text': [
                {
                    'name': 'Whisper',
                    'architecture': 'whisper',
                    'size': 'Various',
                    'speed': 'Variable',
                    'accuracy': 'Excellent',
                    'description': 'OpenAI\'s Whisper for robust speech recognition and transcription',
                    'best_for': 'Transcribing audio files to text',
                    'pretrained': True
                }
            ],
            'audio_classification': [
                {
                    'name': 'Simple CNN',
                    'architecture': 'simple_cnn',
                    'size': 'Small',
                    'speed': 'Fast',
                    'accuracy': 'Good',
                    'description': 'Custom CNN processing Mel Spectrograms',
                    'best_for': 'Simple audio classification tasks',
                    'pretrained': False
                },
                {
                    'name': 'MobileNetV2',
                    'architecture': 'mobilenet_v2',
                    'size': 'Medium',
                    'speed': 'Fast',
                    'accuracy': 'Excellent',
                    'description': 'Transfer learning on spectrograms',
                    'best_for': 'Complex environmental sounds or speech',
                    'pretrained': True
                }
            ],
            'text_classification': [
                {
                    'name': 'DistilBERT',
                    'architecture': 'distilbert',
                    'size': 'Medium (268 MB)',
                    'speed': 'Fast',
                    'accuracy': 'Good',
                    'description': 'Distilled version of BERT, faster and smaller',
                    'best_for': 'Fast inference, production deployment',
                    'pretrained': True
                },
                {
                    'name': 'BERT Base',
                    'architecture': 'bert',
                    'size': 'Large (438 MB)',
                    'speed': 'Medium',
                    'accuracy': 'Excellent',
                    'description': 'Bidirectional transformer for NLP',
                    'best_for': 'High accuracy NLP tasks',
                    'pretrained': True
                },
                {
                    'name': 'RoBERTa',
                    'architecture': 'roberta',
                    'size': 'Large (498 MB)',
                    'speed': 'Medium',
                    'accuracy': 'Excellent',
                    'description': 'Robustly optimized BERT',
                    'best_for': 'State-of-the-art text classification',
                    'pretrained': True
                }
            ],
            'sentiment_analysis': [
                {
                    'name': 'DistilBERT (Sentiment)',
                    'architecture': 'distilbert',
                    'size': 'Medium (268 MB)',
                    'speed': 'Fast',
                    'accuracy': 'Excellent',
                    'description': 'Fine-tuned for sentiment analysis',
                    'best_for': 'Fast sentiment classification',
                    'pretrained': True
                },
                {
                    'name': 'RoBERTa (Sentiment)',
                    'architecture': 'roberta',
                    'size': 'Large (498 MB)',
                    'speed': 'Medium',
                    'accuracy': 'Excellent',
                    'description': 'State-of-the-art sentiment model',
                    'best_for': 'Highest accuracy sentiment analysis',
                    'pretrained': True
                }
            ],
            'tabular_classification': [
                {
                    'name': 'AutoML (AutoKeras)',
                    'architecture': 'autokeras',
                    'size': 'Variable',
                    'speed': 'Variable',
                    'accuracy': 'Good',
                    'description': 'Automated architecture search',
                    'best_for': 'Automatic model selection',
                    'pretrained': False
                },
                {
                    'name': 'Neural Network',
                    'architecture': 'feedforward_nn',
                    'size': 'Small',
                    'speed': 'Fast',
                    'accuracy': 'Good',
                    'description': 'Simple feedforward neural network',
                    'best_for': 'Quick prototyping',
                    'pretrained': False
                }
            ],
            'tabular_regression': [
                {
                    'name': 'AutoML (TPOT)',
                    'architecture': 'tpot',
                    'size': 'Variable',
                    'speed': 'Variable',
                    'accuracy': 'Good',
                    'description': 'Genetic programming for model selection',
                    'best_for': 'Automatic pipeline optimization',
                    'pretrained': False
                },
                {
                    'name': 'Neural Network',
                    'architecture': 'feedforward_nn',
                    'size': 'Small',
                    'speed': 'Fast',
                    'accuracy': 'Good',
                    'description': 'Simple regression neural network',
                    'best_for': 'Quick prototyping',
                    'pretrained': False
                }
            ]
        }
    
    def recommend(self, task_type: str, input_type: Optional[str] = None, 
                  use_case: Optional[str] = None) -> List[Dict]:
        """
        Recommend models for a given task type
        
        Args:
            task_type: Type of ML task
            input_type: Type of input data (optional)
            use_case: Specific use case description (optional)
            
        Returns:
            List of recommended model dictionaries
        """
        # Get static models for task type
        static_models = self.model_database.get(task_type, [])
        
        # Get dynamic models from manager
        dynamic_models = []
        if self.model_manager:
            try:
                all_models = self.model_manager.list_available_models()
                for m in all_models:
                    # Check if model matches task type
                    if m.get('task_type') == task_type:
                        # Convert to recommendation format
                        dynamic_model = {
                            'name': m.get('name', 'Unknown Model'),
                            'architecture': m.get('architecture', 'custom'),
                            'size': 'Unknown', # Could calculate from file size
                            'speed': 'Unknown',
                            'accuracy': 'Unknown', # Could use validation accuracy if available
                            'description': f"User trained model: {m.get('name')}",
                            'best_for': 'Custom task',
                            'pretrained': False,
                            'is_dynamic': True, # Flag to indicate this is a user model
                            'id': m.get('id')
                        }
                        dynamic_models.append(dynamic_model)
            except Exception as e:
                logger.error(f"Error fetching dynamic models: {e}")

        models = static_models + dynamic_models
        
        if not models:
            logger.warning(f"No models found for task type: {task_type}")
            return self._get_default_recommendation(input_type)
        
        # Sort models by recommendation score
        scored_models = []
        for model in models:
            score = self._calculate_recommendation_score(
                model, task_type, use_case
            )
            scored_models.append({**model, 'recommendation_score': score})
        
        # Sort by score (descending)
        scored_models.sort(key=lambda x: x['recommendation_score'], reverse=True)
        
        logger.info(f"Recommended {len(scored_models)} models for {task_type}")
        return scored_models
    
    def _calculate_recommendation_score(self, model: Dict, task_type: str, 
                                       use_case: Optional[str]) -> float:
        """
        Calculate recommendation score for a model
        
        Args:
            model: Model dictionary
            task_type: Task type
            use_case: Use case description
            
        Returns:
            Recommendation score (0-1)
        """
        score = 0.5  # Base score
        
        # Boost score for pretrained models
        if model.get('pretrained'):
            score += 0.2
        
        # Boost based on speed requirements
        if use_case and any(word in use_case.lower() for word in ['real-time', 'fast', 'mobile']):
            if model.get('speed') in ['Fast', 'Very Fast']:
                score += 0.15
        
        # Boost based on accuracy requirements
        if use_case and any(word in use_case.lower() for word in ['accurate', 'precision', 'high quality']):
            if model.get('accuracy') == 'Excellent':
                score += 0.15
        
        return min(score, 1.0)
    
    def _get_default_recommendation(self, input_type: Optional[str]) -> List[Dict]:
        """
        Get default recommendation when no specific models found
        
        Args:
            input_type: Type of input data
            
        Returns:
            List with default model recommendation
        """
        default_model = {
            'name': 'AutoML',
            'architecture': 'autokeras',
            'size': 'Variable',
            'speed': 'Variable',
            'accuracy': 'Good',
            'description': 'Automated machine learning will find the best architecture',
            'best_for': 'Automatic model selection',
            'pretrained': False,
            'recommendation_score': 0.7
        }
        
        return [default_model]
    
    def get_model_details(self, architecture: str) -> Optional[Dict]:
        """
        Get detailed information about a specific model architecture
        
        Args:
            architecture: Model architecture name
            
        Returns:
            Model details dictionary or None
        """
        for task_models in self.model_database.values():
            for model in task_models:
                if model['architecture'] == architecture:
                    return model
        
        return None
