"""
Download Pretrained Models
Script to download and cache pretrained models
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PRETRAINED_MODELS_DIR
from src.models.pretrained_loader import PretrainedLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_image_models():
    """Download pretrained image models"""
    logger.info("Downloading pretrained image models...")
    
    loader = PretrainedLoader()
    
    models = ['mobilenet_v2', 'resnet50', 'efficientnet_b0']
    
    for model_name in models:
        try:
            logger.info(f"Downloading {model_name}...")
            loader.load(model_name, 'image_classification')
            logger.info(f"✓ {model_name} downloaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to download {model_name}: {e}")


def download_text_models():
    """Download pretrained text models"""
    logger.info("Downloading pretrained text models...")
    
    loader = PretrainedLoader()
    
    models = ['distilbert', 'bert']
    
    for model_name in models:
        try:
            logger.info(f"Downloading {model_name}...")
            loader.load(model_name, 'text_classification')
            logger.info(f"✓ {model_name} downloaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to download {model_name}: {e}")


def main():
    """Main function"""
    logger.info("Starting pretrained model download...")
    logger.info(f"Models will be cached in: {PRETRAINED_MODELS_DIR}")
    
    # Create directory
    PRETRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download models
    download_image_models()
    download_text_models()
    
    logger.info("All downloads completed!")


if __name__ == '__main__':
    main()
