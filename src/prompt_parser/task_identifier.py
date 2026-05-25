"""
Task Identifier
Identifies specific ML task types from extracted information
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TaskIdentifier:
    """
    Identify and classify ML task types
    """
    
    # Task type mappings
    TASK_CATEGORIES = {
        'image_classification': ['image', 'picture', 'photo', 'classify', 'categorize'],
        'image_to_text': ['ocr', 'extract text', 'read text', 'text from image', 'scan text', 'text extraction', 'extract'],
        'audio_to_text': ['speech to text', 'transcribe', 'text from audio', 'extract text from audio'],
        'audio_classification': ['audio', 'sound', 'noise', 'voice', 'speech', 'music', 'classify', 'categorize', 'recognize'],
        'object_detection': ['detect', 'find', 'locate', 'bounding box', 'bbox'],
        'image_segmentation': ['segment', 'mask', 'pixel', 'region'],
        'text_classification': ['text', 'document', 'classify', 'categorize'],
        'sentiment_analysis': ['sentiment', 'emotion', 'feeling', 'opinion', 'review'],
        'tabular_classification': ['table', 'csv', 'data', 'classify', 'categorical'],
        'tabular_regression': ['predict', 'forecast', 'regression', 'continuous', 'value']
    }
    
    def __init__(self):
        """Initialize task identifier"""
        pass
    
    def identify(self, task_info: Dict, original_prompt: str = "") -> str:
        """
        Identify specific task type from extracted information
        
        Args:
            task_info: Extracted task information from LLM
            original_prompt: Original user prompt for fallback
            
        Returns:
            Identified task type string
        """
        # First, check if task_type is already specified
        if 'task_type' in task_info:
            specified_task = task_info['task_type'].lower()
            input_type = task_info.get('input_type', '').lower()
            
            # Guard against LLM hallucinating image_classification when input is audio
            if input_type == 'audio' and 'image' in specified_task:
                specified_task = 'audio_classification'
            # Guard against LLM hallucinating image_classification when input is text
            elif input_type == 'text' and 'image' in specified_task:
                specified_task = 'text_classification'
                
            if self._is_valid_task_type(specified_task):
                return specified_task
        
        # Extract input and output types
        input_type = task_info.get('input_type', '').lower()
        output_type = task_info.get('output_type', '').lower()
        keywords = task_info.get('keywords', [])
        
        # Combine all text for analysis
        analysis_text = ' '.join([
            input_type,
            output_type,
            ' '.join(keywords),
            original_prompt.lower()
        ])
        
        # Score each task type
        scores = {}
        for task_type, task_keywords in self.TASK_CATEGORIES.items():
            score = sum(1 for keyword in task_keywords if keyword in analysis_text)
            scores[task_type] = score
        
        # Get task type with highest score
        if scores:
            best_task = max(scores.items(), key=lambda x: x[1])
            if best_task[1] > 0:
                logger.info(f"Identified task type: {best_task[0]} (score: {best_task[1]})")
                return best_task[0]
        
        # Fallback logic based on input type
        return self._fallback_identification(input_type)
    
    def _is_valid_task_type(self, task_type: str) -> bool:
        """
        Check if a task type is valid
        
        Args:
            task_type: Task type string
            
        Returns:
            True if valid, False otherwise
        """
        valid_tasks = list(self.TASK_CATEGORIES.keys())
        return task_type in valid_tasks
    
    def _fallback_identification(self, input_type: str) -> str:
        """
        Fallback identification based on input type
        
        Args:
            input_type: Type of input data
            
        Returns:
            Default task type for the input type
        """
        fallback_map = {
            'image': 'image_classification',
            'text': 'text_classification',
            'tabular': 'tabular_classification',
            'audio': 'audio_classification',
            'video': 'video_classification'
        }
        
        return fallback_map.get(input_type, 'image_classification')
    
    def get_task_requirements(self, task_type: str) -> Dict:
        """
        Get specific requirements for a task type
        
        Args:
            task_type: ML task type
            
        Returns:
            Dictionary with task requirements
        """
        requirements = {
            'image_classification': {
                'input_format': 'Images (JPG, PNG)',
                'output_format': 'Class labels',
                'min_samples': 100,
                'recommended_samples': 1000,
                'labels_required': True
            },
            'object_detection': {
                'input_format': 'Images with bounding box annotations',
                'output_format': 'Bounding boxes + class labels',
                'min_samples': 200,
                'recommended_samples': 2000,
                'labels_required': True
            },
            'image_segmentation': {
                'input_format': 'Images with pixel masks',
                'output_format': 'Segmentation masks',
                'min_samples': 150,
                'recommended_samples': 1500,
                'labels_required': True
            },
            'image_to_text': {
                'input_format': 'Images containing text',
                'output_format': 'Extracted text',
                'min_samples': 0,
                'recommended_samples': 0,
                'labels_required': False
            },
            'audio_to_text': {
                'input_format': 'Audio files (MP3, WAV, etc.)',
                'output_format': 'Transcribed text',
                'min_samples': 0,
                'recommended_samples': 0,
                'labels_required': False
            },
            'audio_classification': {
                'input_format': 'Audio files (MP3, WAV, etc.)',
                'output_format': 'Class labels',
                'min_samples': 20,
                'recommended_samples': 200,
                'labels_required': True
            },
            'text_classification': {
                'input_format': 'Text documents',
                'output_format': 'Class labels',
                'min_samples': 100,
                'recommended_samples': 1000,
                'labels_required': True
            },
            'sentiment_analysis': {
                'input_format': 'Text (reviews, comments, tweets)',
                'output_format': 'Sentiment labels (positive/negative/neutral)',
                'min_samples': 200,
                'recommended_samples': 2000,
                'labels_required': True
            },
            'tabular_classification': {
                'input_format': 'CSV with features and target column',
                'output_format': 'Class labels',
                'min_samples': 100,
                'recommended_samples': 1000,
                'labels_required': True
            },
            'tabular_regression': {
                'input_format': 'CSV with features and target values',
                'output_format': 'Continuous values',
                'min_samples': 100,
                'recommended_samples': 1000,
                'labels_required': True
            }
        }
        
        return requirements.get(task_type, {
            'input_format': 'Unknown',
            'output_format': 'Unknown',
            'min_samples': 100,
            'recommended_samples': 1000,
            'labels_required': True
        })
