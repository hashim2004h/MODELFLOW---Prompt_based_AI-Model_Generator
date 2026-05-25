"""
Main Prompt Parser
Uses LLM to interpret natural language prompts and extract task information
"""

import os
import json
import logging
from typing import Dict, List, Optional
from openai import OpenAI

from config import OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_BASE_URL
from src.prompt_parser.task_identifier import TaskIdentifier
from src.prompt_parser.model_recommender import ModelRecommender

logger = logging.getLogger(__name__)


class PromptParser:
    """
    Parse natural language prompts to identify ML tasks and recommend models
    """
    
    def __init__(self):
        """Initialize the prompt parser with OpenRouter client"""
        if OPENROUTER_API_KEY:
            try:
                self.client = OpenAI(
                    api_key=OPENROUTER_API_KEY,
                    base_url=OPENROUTER_BASE_URL,
                    http_client=None  # Let it create default client
                )
            except Exception as e:
                print(f"Warning: Could not initialize OpenRouter client: {e}")
                self.client = None
        else:
            self.client = None
        
        self.task_identifier = TaskIdentifier()
        self.model_recommender = ModelRecommender()

        
        # System prompt for the LLM
        self.system_prompt = """You are an AI assistant specialized in machine learning task analysis.
Given a user's natural language description of an ML task, extract and return the following information in JSON format:

1. task_type: The type of ML task (image_classification, audio_classification, text_classification, object_detection, image_segmentation, tabular_regression, image_to_text, audio_to_text, etc.)
2. input_type: Type of input data (image, text, tabular, audio, video)
3. output_type: Type of output (labels, bounding_boxes, masks, values, text)
4. num_classes: Estimated number of classes (if classification)
5. dataset_requirements: Description of required dataset
6. use_case: Brief description of the use case
7. keywords: List of relevant keywords from the prompt

Return only valid JSON, no additional text."""
    
    def parse(self, prompt: str) -> Dict:
        """
        Parse a natural language prompt
        
        Args:
            prompt: User's natural language description of ML task
            
        Returns:
            Dictionary containing task information and recommendations
        """
        try:
            # Step 1: Extract task information using LLM
            task_info = self._extract_task_info(prompt)
            
            # Step 2: Identify specific task type
            task_type = self.task_identifier.identify(task_info, prompt)
            
            # Step 3: Get model recommendations
            recommended_models = self.model_recommender.recommend(
                task_type=task_type,
                input_type=task_info.get('input_type'),
                use_case=task_info.get('use_case')
            )
            
            # Step 4: Compile complete result
            result = {
                'task_type': task_type,
                'input_type': task_info.get('input_type', 'unknown'),
                'output_type': task_info.get('output_type', 'unknown'),
                'num_classes': task_info.get('num_classes'),
                'dataset_requirements': task_info.get('dataset_requirements', ''),
                'use_case': task_info.get('use_case', ''),
                'keywords': task_info.get('keywords', []),
                'recommended_models': recommended_models,
                'confidence': self._calculate_confidence(task_info)
            }
            
            logger.info(f"Successfully parsed prompt: {task_type}")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing prompt: {e}")
            # Fallback to rule-based parsing
            return self._fallback_parse(prompt)
    
    def _extract_task_info(self, prompt: str) -> Dict:
        """
        Extract task information using OpenAI API
        
        Args:
            prompt: User prompt
            
        Returns:
            Dictionary with extracted task information
        """
        if not self.client:
            logger.warning("OpenAI client not initialized, using fallback")
            return self._fallback_parse(prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse JSON response
            content = response.choices[0].message.content
            task_info = json.loads(content)
            
            return task_info
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return self._fallback_parse(prompt)
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return self._fallback_parse(prompt)
    
    def _fallback_parse(self, prompt: str) -> Dict:
        """
        Rule-based fallback parser when LLM is unavailable
        
        Args:
            prompt: User prompt
            
        Returns:
            Dictionary with basic task information
        """
        prompt_lower = prompt.lower()
        
        # Detect input type (prioritize audio over text if both present)
        if any(word in prompt_lower for word in ['audio', 'voice', 'speech', 'sound']):
            input_type = 'audio'
        elif any(word in prompt_lower for word in ['image', 'picture', 'photo', 'visual']):
            input_type = 'image'
        elif any(word in prompt_lower for word in ['text', 'sentence', 'document', 'review']):
            input_type = 'text'
        elif any(word in prompt_lower for word in ['table', 'csv', 'data', 'spreadsheet']):
            input_type = 'tabular'
        else:
            input_type = 'unknown'
        
        # Detect task type
        if any(word in prompt_lower for word in ['transcribe', 'speech to text']) or (input_type == 'audio' and any(word in prompt_lower for word in ['text', 'extraction', 'extract'])):
            task_type = 'audio_to_text'
        elif any(word in prompt_lower for word in ['extract', 'extraction', 'ocr', 'read text', 'scan text']) and input_type != 'audio':
            task_type = 'image_to_text'
        elif any(word in prompt_lower for word in ['classify', 'classification', 'categorize']):
            task_type = f'{input_type}_classification'
        elif any(word in prompt_lower for word in ['detect', 'detection', 'find', 'locate']):
            task_type = 'object_detection'
        elif any(word in prompt_lower for word in ['segment', 'segmentation', 'mask']):
            task_type = 'image_segmentation'
        elif any(word in prompt_lower for word in ['sentiment', 'emotion', 'feeling']):
            task_type = 'sentiment_analysis'
        elif any(word in prompt_lower for word in ['predict', 'forecast', 'regression']):
            task_type = 'tabular_regression'
        else:
            task_type = f'{input_type}_classification'  # Default
        
        # Guard against weird types
        if task_type == 'unknown_classification':
            task_type = 'image_classification'
        
        return {
            'task_type': task_type,
            'input_type': input_type,
            'output_type': 'labels',
            'dataset_requirements': f'Dataset of {input_type} data with labels',
            'use_case': prompt[:100],
            'keywords': prompt_lower.split()[:10]
        }
    
    def _calculate_confidence(self, task_info: Dict) -> float:
        """
        Calculate confidence score for the parsed information
        
        Args:
            task_info: Extracted task information
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.5  # Base confidence
        
        # Increase confidence if key fields are present
        if task_info.get('task_type'):
            confidence += 0.2
        if task_info.get('input_type'):
            confidence += 0.15
        if task_info.get('output_type'):
            confidence += 0.1
        if task_info.get('keywords'):
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def validate_prompt(self, prompt: str) -> tuple[bool, str]:
        """
        Validate if the prompt is suitable for ML task extraction
        
        Args:
            prompt: User prompt
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not prompt or len(prompt.strip()) < 10:
            return False, "Prompt is too short. Please provide more details."
        
        if len(prompt) > 1000:
            return False, "Prompt is too long. Please keep it under 1000 characters."
        
        # Check for ML-related keywords
        ml_keywords = [
            'classify', 'detect', 'predict', 'recognize', 'identify',
            'segment', 'analyze', 'model', 'train', 'learn'
        ]
        
        if not any(keyword in prompt.lower() for keyword in ml_keywords):
            return False, "Prompt doesn't seem to describe an ML task. Please include words like 'classify', 'detect', 'predict', etc."
        
        return True, ""
