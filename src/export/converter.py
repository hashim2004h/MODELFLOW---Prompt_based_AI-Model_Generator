"""
Model Converter
Main orchestrator for model format conversion
"""

import logging
from pathlib import Path
from typing import Optional
import shutil

from config import EXPORTED_MODELS_DIR, ONNX_DIR, TFLITE_DIR, H5_DIR, PT_DIR
from src.export.onnx_exporter import ONNXExporter
from src.export.tflite_exporter import TFLiteExporter
from src.export.pytorch_exporter import PyTorchExporter

logger = logging.getLogger(__name__)


class ModelConverter:
    """
    Convert models to different formats for deployment
    """
    
    def __init__(self):
        """Initialize model converter"""
        self.onnx_exporter = ONNXExporter()
        self.tflite_exporter = TFLiteExporter()
        self.pytorch_exporter = PyTorchExporter()
        
        # Ensure export directories exist
        for directory in [ONNX_DIR, TFLITE_DIR, H5_DIR, PT_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def convert(self, model_id: str, export_format: str, 
                model_name: Optional[str] = None) -> Path:
        """
        Convert model to specified format
        
        Args:
            model_id: ID of the model to convert
            export_format: Target format ('onnx', 'tflite', 'h5', 'pt')
            model_name: Optional custom name for exported model
            
        Returns:
            Path to exported model file
        """
        from src.models.model_manager import ModelManager
        
        model_manager = ModelManager()
        model = model_manager.get_model(model_id)
        metadata = model_manager.get_metadata(model_id)
        
        if model is None:
            raise ValueError(f"Model {model_id} not found")
        
        if model_name is None:
            model_name = metadata.get('name', 'model').replace(' ', '_')
        
        export_format = export_format.lower()
        
        logger.info(f"Converting model {model_id} to {export_format} format")
        
        try:
            if export_format == 'onnx':
                export_path = self.onnx_exporter.export(model, model_name, metadata)
            elif export_format == 'tflite':
                export_path = self.tflite_exporter.export(model, model_name, metadata)
            elif export_format == 'h5':
                export_path = self._export_h5(model, model_name)
            elif export_format == 'pt' or export_format == 'pytorch':
                export_path = self.pytorch_exporter.export(model, model_name, metadata)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            logger.info(f"Model exported successfully to: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise
    
    def _export_h5(self, model, model_name: str) -> Path:
        """
        Export TensorFlow/Keras model to H5 format
        
        Args:
            model: Keras model
            model_name: Model name
            
        Returns:
            Path to exported H5 file
        """
        export_path = H5_DIR / f"{model_name}.h5"
        model.save(str(export_path))
        
        logger.info(f"Exported H5 model to {export_path}")
        return export_path
    
    def convert_multiple_formats(self, model_id: str, 
                                 formats: list = ['onnx', 'tflite', 'h5']) -> dict:
        """
        Convert model to multiple formats at once
        
        Args:
            model_id: Model ID
            formats: List of target formats
            
        Returns:
            Dictionary mapping format to export path
        """
        results = {}
        
        for fmt in formats:
            try:
                export_path = self.convert(model_id, fmt)
                results[fmt] = str(export_path)
            except Exception as e:
                logger.error(f"Failed to export to {fmt}: {e}")
                results[fmt] = None
        
        return results
    
    def get_model_size(self, model_path: Path) -> dict:
        """
        Get size information for exported model
        
        Args:
            model_path: Path to model file
            
        Returns:
            Dictionary with size information
        """
        if not model_path.exists():
            return {'error': 'File not found'}
        
        size_bytes = model_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        
        return {
            'path': str(model_path),
            'size_bytes': size_bytes,
            'size_mb': round(size_mb, 2),
            'size_human': self._human_readable_size(size_bytes)
        }
    
    def _human_readable_size(self, size_bytes: int) -> str:
        """Convert bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def cleanup_exports(self, model_name: str):
        """
        Clean up all exported versions of a model
        
        Args:
            model_name: Name of the model
        """
        for directory in [ONNX_DIR, TFLITE_DIR, H5_DIR, PT_DIR]:
            for file in directory.glob(f"{model_name}.*"):
                file.unlink()
                logger.info(f"Deleted {file}")
