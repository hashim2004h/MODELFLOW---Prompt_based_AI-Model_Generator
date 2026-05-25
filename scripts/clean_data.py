"""
Data Cleaning Script
Clean up temporary and uploaded data
"""

import sys
from pathlib import Path
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import UPLOADED_DATA_DIR, CHECKPOINTS_DIR, LOGS_DIR
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_uploaded_data():
    """Remove uploaded data files"""
    logger.info("Cleaning uploaded data...")
    
    try:
        if UPLOADED_DATA_DIR.exists():
            for item in UPLOADED_DATA_DIR.rglob('*'):
                if item.is_file() and item.name != '.gitkeep':
                    item.unlink()
                    logger.info(f"Deleted: {item}")
        
        logger.info("✓ Uploaded data cleaned")
        return True
    except Exception as e:
        logger.error(f"Error cleaning uploaded data: {e}")
        return False


def clean_checkpoints():
    """Remove training checkpoints"""
    logger.info("Cleaning checkpoints...")
    
    try:
        if CHECKPOINTS_DIR.exists():
            for item in CHECKPOINTS_DIR.glob('*'):
                if item.is_file():
                    item.unlink()
                    logger.info(f"Deleted: {item}")
        
        logger.info("✓ Checkpoints cleaned")
        return True
    except Exception as e:
        logger.error(f"Error cleaning checkpoints: {e}")
        return False


def clean_logs():
    """Remove old log files"""
    logger.info("Cleaning logs...")
    
    try:
        if LOGS_DIR.exists():
            for item in LOGS_DIR.glob('*.log*'):
                item.unlink()
                logger.info(f"Deleted: {item}")
        
        logger.info("✓ Logs cleaned")
        return True
    except Exception as e:
        logger.error(f"Error cleaning logs: {e}")
        return False


def main():
    """Main cleaning function"""
    logger.info("=" * 60)
    logger.info("MODELFLOW Data Cleanup")
    logger.info("=" * 60)
    
    response = input("\nThis will delete uploaded data, checkpoints, and logs.\nContinue? (y/n): ")
    
    if response.lower() != 'y':
        logger.info("Cleanup cancelled")
        return
    
    logger.info("\nStarting cleanup...\n")
    
    clean_uploaded_data()
    clean_checkpoints()
    clean_logs()
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Cleanup complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
