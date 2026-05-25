"""
Environment Setup Script
Setup and verify the development environment
"""

import sys
from pathlib import Path
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import create_directories
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is compatible"""
    logger.info("Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        logger.error("Python 3.10 or higher is required")
        return False
    
    logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'flask',
        'tensorflow',
        'torch',
        'numpy',
        'pandas',
        'opencv-python',
        'pillow',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✓ {package}")
        except ImportError:
            logger.warning(f"✗ {package} not found")
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Install with: pip install -r requirements.txt")
        return False
    
    return True


def setup_directories():
    """Create necessary directories"""
    logger.info("Setting up directories...")
    
    try:
        create_directories()
        logger.info("✓ All directories created")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to create directories: {e}")
        return False


def create_env_file():
    """Create .env file template"""
    logger.info("Creating .env file template...")
    
    env_file = Path(__file__).resolve().parent.parent / '.env'
    
    if env_file.exists():
        logger.info("✓ .env file already exists")
        return True
    
    env_template = """# MODELFLOW Environment Variables

# OpenAI API (for prompt parsing)
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4

# Application Settings
DEBUG=True
HOST=0.0.0.0
PORT=5000
SECRET_KEY=change-this-in-production

# Database (if needed)
# DATABASE_URL=sqlite:///modelflow.db

# Storage Paths (optional overrides)
# DATA_DIR=data
# MODELS_DIR=models_storage
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_template)
        logger.info(f"✓ .env file created at {env_file}")
        logger.info("  Please update with your API keys")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to create .env file: {e}")
        return False


def verify_gpu():
    """Check if GPU is available"""
    logger.info("Checking GPU availability...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            logger.info(f"✓ {len(gpus)} GPU(s) detected")
            for gpu in gpus:
                logger.info(f"  - {gpu.name}")
        else:
            logger.info("  No GPU detected, will use CPU")
        
        return True
    except Exception as e:
        logger.warning(f"Could not check GPU: {e}")
        return True


def main():
    """Main setup function"""
    logger.info("=" * 60)
    logger.info("MODELFLOW Environment Setup")
    logger.info("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directories", setup_directories),
        ("Environment File", create_env_file),
        ("GPU Support", verify_gpu)
    ]
    
    results = []
    for name, check_func in checks:
        logger.info(f"\n{name}:")
        results.append(check_func())
    
    logger.info("\n" + "=" * 60)
    if all(results):
        logger.info("✓ Environment setup complete!")
        logger.info("\nNext steps:")
        logger.info("1. Update .env file with your API keys")
        logger.info("2. Run: python scripts/download_pretrained.py")
        logger.info("3. Start the app: python run.py")
    else:
        logger.warning("⚠ Setup completed with warnings")
        logger.info("Please resolve the issues above before running the app")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
