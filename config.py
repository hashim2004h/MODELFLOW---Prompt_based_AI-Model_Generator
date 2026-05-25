
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# API Configuration
# OpenRouter API Configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'openai/gpt-oss-20b:free')
OPENROUTER_BASE_URL = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')

# Application Configuration
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 5000))
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Directory Paths
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
UPLOADED_DATA_DIR = DATA_DIR / 'uploaded'
SAMPLES_DATA_DIR = DATA_DIR / 'samples'

# Uploaded subdirectories
UPLOADED_IMAGES_DIR = UPLOADED_DATA_DIR / 'images'
UPLOADED_TEXT_DIR = UPLOADED_DATA_DIR / 'text'
UPLOADED_CSV_DIR = UPLOADED_DATA_DIR / 'csv'
UPLOADED_AUDIO_DIR = UPLOADED_DATA_DIR / 'audio'

# Model Storage Paths
MODELS_STORAGE_DIR = BASE_DIR / 'models_storage'
PRETRAINED_MODELS_DIR = MODELS_STORAGE_DIR / 'pretrained'
CHECKPOINTS_DIR = MODELS_STORAGE_DIR / 'checkpoints'
TRAINED_MODELS_DIR = MODELS_STORAGE_DIR / 'trained'
EXPORTED_MODELS_DIR = MODELS_STORAGE_DIR / 'exported'

# Exported model subdirectories
ONNX_DIR = EXPORTED_MODELS_DIR / 'onnx'
TFLITE_DIR = EXPORTED_MODELS_DIR / 'tflite'
H5_DIR = EXPORTED_MODELS_DIR / 'h5'
PT_DIR = EXPORTED_MODELS_DIR / 'pt'

# Logs
LOGS_DIR = BASE_DIR / 'logs'

# Model Configuration
MAX_UPLOAD_SIZE = 16 * 1024 * 1024  # 16MB
ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
ALLOWED_TEXT_EXTENSIONS = {'.txt', '.csv', '.json'}
ALLOWED_CSV_EXTENSIONS = {'.csv', '.xlsx'}
ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.flac'}

# Training Configuration
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_VALIDATION_SPLIT = 0.2

# Supported Task Types
TASK_TYPES = [
    'image_classification',
    'image_to_text',
    'audio_to_text',
    'audio_classification',
    'object_detection',
    'image_segmentation',
    'text_classification',
    'sentiment_analysis',
    'tabular_classification',
    'tabular_regression'
]

# Supported Model Architectures
IMAGE_MODELS = {
    'mobilenet_v2': 'MobileNetV2',
    'resnet50': 'ResNet50',
    'efficientnet_b0': 'EfficientNetB0',
    'vgg16': 'VGG16',
    'inception_v3': 'InceptionV3'
}

TEXT_MODELS = {
    'distilbert': 'distilbert-base-uncased',
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base'
}

# Create directories if they don't exist
def create_directories():
    """Create all necessary directories"""
    dirs = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, UPLOADED_DATA_DIR,
        SAMPLES_DATA_DIR, UPLOADED_IMAGES_DIR, UPLOADED_TEXT_DIR,
        UPLOADED_CSV_DIR, UPLOADED_AUDIO_DIR, MODELS_STORAGE_DIR, PRETRAINED_MODELS_DIR,
        CHECKPOINTS_DIR, TRAINED_MODELS_DIR, EXPORTED_MODELS_DIR,
        ONNX_DIR, TFLITE_DIR, H5_DIR, PT_DIR, LOGS_DIR
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    create_directories()
    print("All directories created successfully!")
