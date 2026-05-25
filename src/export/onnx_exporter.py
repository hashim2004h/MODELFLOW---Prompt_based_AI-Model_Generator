"""
ONNX Exporter
Export models to ONNX format for cross-platform deployment
"""

import logging
from pathlib import Path
from typing import Dict
import numpy as np

from config import ONNX_DIR

logger = logging.getLogger(__name__)


class ONNXExporter:
    """
    Export models to ONNX format
    """
    
    def export(self, model, model_name: str, metadata: Dict) -> Path:
        """
        Export model to ONNX format
        
        Args:
            model: Model to export
            model_name: Name for exported model
            metadata: Model metadata
            
        Returns:
            Path to exported ONNX file
        """
        framework = metadata.get('framework', 'tensorflow')
        
        if framework == 'tensorflow':
            return self._export_tensorflow_to_onnx(model, model_name, metadata)
        elif framework == 'pytorch':
            return self._export_pytorch_to_onnx(model, model_name, metadata)
        else:
            raise ValueError(f"Unsupported framework for ONNX export: {framework}")
    
    def _export_tensorflow_to_onnx(self, model, model_name: str, metadata: Dict) -> Path:
        """Export TensorFlow model to ONNX"""
        import tf2onnx
        import tensorflow as tf
        
        export_path = ONNX_DIR / f"{model_name}.onnx"
        
        try:
            # Get input signature
            input_shape = metadata.get('input_shape', (224, 224, 3))
            
            # Create input signature
            input_signature = [tf.TensorSpec(
                shape=[None] + list(input_shape),
                dtype=tf.float32,
                name='input'
            )]
            
            # Convert to ONNX
            onnx_model, _ = tf2onnx.convert.from_keras(
                model,
                input_signature=input_signature,
                opset=13,
                output_path=str(export_path)
            )
            
            logger.info(f"TensorFlow model exported to ONNX: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Error exporting TensorFlow to ONNX: {e}")
            raise
    
    def _export_pytorch_to_onnx(self, model, model_name: str, metadata: Dict) -> Path:
        """Export PyTorch model to ONNX"""
        import torch
        
        export_path = ONNX_DIR / f"{model_name}.onnx"
        
        try:
            # Set model to eval mode
            model.eval()
            
            # Create dummy input
            input_shape = metadata.get('input_shape', (1, 3, 224, 224))
            dummy_input = torch.randn(*input_shape)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                str(export_path),
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            logger.info(f"PyTorch model exported to ONNX: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Error exporting PyTorch to ONNX: {e}")
            raise
    
    def verify_onnx_model(self, onnx_path: Path) -> bool:
        """
        Verify ONNX model is valid
        
        Args:
            onnx_path: Path to ONNX file
            
        Returns:
            True if valid, False otherwise
        """
        import onnx
        
        try:
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            logger.info(f"ONNX model verified: {onnx_path}")
            return True
        except Exception as e:
            logger.error(f"ONNX model verification failed: {e}")
            return False
    
    def test_onnx_inference(self, onnx_path: Path, sample_input: np.ndarray) -> np.ndarray:
        """
        Test ONNX model inference
        
        Args:
            onnx_path: Path to ONNX model
            sample_input: Sample input data
            
        Returns:
            Model output
        """
        import onnxruntime as ort
        
        try:
            # Create inference session
            session = ort.InferenceSession(str(onnx_path))
            
            # Get input name
            input_name = session.get_inputs()[0].name
            
            # Run inference
            outputs = session.run(None, {input_name: sample_input})
            
            logger.info("ONNX inference test successful")
            return outputs[0]
            
        except Exception as e:
            logger.error(f"ONNX inference test failed: {e}")
            raise
