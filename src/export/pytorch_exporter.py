"""
PyTorch Exporter
Export PyTorch models in .pt format
"""

import logging
from pathlib import Path
from typing import Dict
import torch

from config import PT_DIR

logger = logging.getLogger(__name__)


class PyTorchExporter:
    """
    Export PyTorch models
    """
    
    def export(self, model, model_name: str, metadata: Dict) -> Path:
        """
        Export PyTorch model
        
        Args:
            model: PyTorch model
            model_name: Name for exported model
            metadata: Model metadata
            
        Returns:
            Path to exported .pt file
        """
        framework = metadata.get('framework', 'unknown')
        
        if framework != 'pytorch':
            raise ValueError("This exporter only supports PyTorch models")
        
        export_path = PT_DIR / f"{model_name}.pt"
        
        try:
            # Save model state dict
            torch.save({
                'model_state_dict': model.state_dict(),
                'metadata': metadata
            }, str(export_path))
            
            logger.info(f"PyTorch model exported: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Error exporting PyTorch model: {e}")
            raise
    
    def export_traced(self, model, model_name: str, 
                     sample_input: torch.Tensor) -> Path:
        """
        Export traced PyTorch model (TorchScript)
        
        Args:
            model: PyTorch model
            model_name: Model name
            sample_input: Sample input for tracing
            
        Returns:
            Path to traced model
        """
        export_path = PT_DIR / f"{model_name}_traced.pt"
        
        try:
            # Set to eval mode
            model.eval()
            
            # Trace the model
            traced_model = torch.jit.trace(model, sample_input)
            
            # Save traced model
            traced_model.save(str(export_path))
            
            logger.info(f"Traced PyTorch model exported: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Error tracing PyTorch model: {e}")
            raise
    
    def export_scripted(self, model, model_name: str) -> Path:
        """
        Export scripted PyTorch model (TorchScript)
        
        Args:
            model: PyTorch model
            model_name: Model name
            
        Returns:
            Path to scripted model
        """
        export_path = PT_DIR / f"{model_name}_scripted.pt"
        
        try:
            # Set to eval mode
            model.eval()
            
            # Script the model
            scripted_model = torch.jit.script(model)
            
            # Save scripted model
            scripted_model.save(str(export_path))
            
            logger.info(f"Scripted PyTorch model exported: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Error scripting PyTorch model: {e}")
            raise
    
    def load_exported_model(self, model_path: Path, model_class=None):
        """
        Load an exported PyTorch model
        
        Args:
            model_path: Path to .pt file
            model_class: Model class (if loading state dict)
            
        Returns:
            Loaded model
        """
        try:
            if 'traced' in model_path.name or 'scripted' in model_path.name:
                # Load TorchScript model
                model = torch.jit.load(str(model_path))
            else:
                # Load state dict
                checkpoint = torch.load(str(model_path))
                
                if model_class is None:
                    raise ValueError("model_class required to load state dict")
                
                model = model_class()
                model.load_state_dict(checkpoint['model_state_dict'])
            
            model.eval()
            logger.info(f"Model loaded from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
