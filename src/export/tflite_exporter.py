"""
TensorFlow Lite Exporter
Export models to TFLite format for mobile and edge deployment
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import tensorflow as tf

from config import TFLITE_DIR

logger = logging.getLogger(__name__)


class TFLiteExporter:
    """
    Export models to TensorFlow Lite format
    """
    
    def export(self, model, model_name: str, metadata: Dict,
              quantize: bool = False) -> Path:
        """
        Export model to TFLite format
        
        Args:
            model: TensorFlow/Keras model
            model_name: Name for exported model
            metadata: Model metadata
            quantize: Whether to apply quantization
            
        Returns:
            Path to exported TFLite file
        """
        framework = metadata.get('framework', 'tensorflow')
        
        if framework != 'tensorflow':
            raise ValueError("TFLite export only supports TensorFlow models")
        
        export_path = TFLITE_DIR / f"{model_name}.tflite"
        
        try:
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Apply optimizations
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                logger.info("Applying quantization for smaller model size")
            
            # Convert
            tflite_model = converter.convert()
            
            # Save to file
            with open(export_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"Model exported to TFLite: {export_path}")
            
            # Log size reduction if quantized
            if quantize:
                size_mb = len(tflite_model) / (1024 * 1024)
                logger.info(f"Quantized model size: {size_mb:.2f} MB")
            
            return export_path
            
        except Exception as e:
            logger.error(f"Error exporting to TFLite: {e}")
            raise
    
    def export_with_metadata(self, model, model_name: str, metadata: Dict,
                            class_labels: Optional[list] = None) -> Path:
        """
        Export TFLite model with metadata
        
        Args:
            model: TensorFlow model
            model_name: Model name
            metadata: Model metadata
            class_labels: Optional list of class labels
            
        Returns:
            Path to exported model
        """
        from tflite_support import metadata as _metadata
        from tflite_support import metadata_schema_py_generated as _metadata_fb
        
        # First, export the model
        export_path = self.export(model, model_name, metadata)
        
        # TODO: Add metadata to TFLite model
        # This requires tflite_support package
        
        return export_path
    
    def quantize_model(self, model, model_name: str, 
                      representative_dataset=None) -> Path:
        """
        Export with full integer quantization
        
        Args:
            model: TensorFlow model
            model_name: Model name
            representative_dataset: Dataset for calibration
            
        Returns:
            Path to quantized model
        """
        export_path = TFLITE_DIR / f"{model_name}_quantized.tflite"
        
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Full integer quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if representative_dataset:
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
            
            tflite_model = converter.convert()
            
            with open(export_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"Quantized model exported: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Error in quantization: {e}")
            raise
    
    def test_tflite_inference(self, tflite_path: Path, sample_input):
        """
        Test TFLite model inference
        
        Args:
            tflite_path: Path to TFLite model
            sample_input: Sample input data
            
        Returns:
            Model output
        """
        try:
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Set input
            interpreter.set_tensor(input_details[0]['index'], sample_input)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output = interpreter.get_tensor(output_details[0]['index'])
            
            logger.info("TFLite inference test successful")
            return output
            
        except Exception as e:
            logger.error(f"TFLite inference test failed: {e}")
            raise
