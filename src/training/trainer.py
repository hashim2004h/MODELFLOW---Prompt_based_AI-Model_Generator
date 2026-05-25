"""
Model Trainer
Manages training workflows and progress tracking
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path
import json
import threading

from config import CHECKPOINTS_DIR, LOGS_DIR

logger = logging.getLogger(__name__)


class Trainer:
    """
    Centralized training manager
    """
    
    def __init__(self):
        """Initialize trainer"""
        self.active_trainings = {}  # Store active training sessions
        self.training_history = {}  # Store training history
    
    def start_training(self, model_id: str, dataset_path: str, config: Dict) -> str:
        """
        Start a new training session
        
        Args:
            model_id: ID of the model to train
            dataset_path: Path to training dataset
            config: Training configuration dictionary
            
        Returns:
            Training session ID
        """
        training_id = str(uuid.uuid4())
        
        # Initialize training session
        session = {
            'id': training_id,
            'model_id': model_id,
            'dataset_path': dataset_path,
            'config': config,
            'status': 'initialized',
            'started_at': datetime.now().isoformat(),
            'current_epoch': 0,
            'total_epochs': config.get('epochs', 10),
            'metrics': {},
            'logs': []
        }
        
        self.active_trainings[training_id] = session
        
        # Start training in background thread
        training_thread = threading.Thread(
            target=self._run_training,
            args=(training_id,)
        )
        training_thread.start()
        
        logger.info(f"Started training session: {training_id}")
        return training_id
    
    def _run_training(self, training_id: str):
        """
        Run training process in background
        
        Args:
            training_id: Training session ID
        """
        session = self.active_trainings[training_id]
        
        try:
            # Update status
            session['status'] = 'running'
            
            # Load model
            from src.models.model_manager import ModelManager
            model_manager = ModelManager()
            model = model_manager.get_model(session['model_id'])
            
            if model is None:
                raise ValueError(f"Model {session['model_id']} not found")
            
            # Load dataset
            from src.utils.data_loader import DataLoader
            data_loader = DataLoader()
            train_data, val_data = data_loader.load_dataset(
                session['dataset_path'],
                session['config']
            )
            
            # Setup callbacks
            from src.training.callbacks import CustomCallbacks
            callbacks = CustomCallbacks.create_callbacks(
                training_id=training_id,
                checkpoint_dir=CHECKPOINTS_DIR,
                log_dir=LOGS_DIR
            )
            
            # Train model
            config = session['config']
            history = model.fit(
                train_data,
                validation_data=val_data,
                epochs=config.get('epochs', 10),
                batch_size=config.get('batch_size', 32),
                callbacks=callbacks,
                verbose=1
            )
            
            # Update session with results
            session['status'] = 'completed'
            session['completed_at'] = datetime.now().isoformat()
            session['metrics'] = {
                'final_loss': float(history.history['loss'][-1]),
                'final_accuracy': float(history.history.get('accuracy', [0])[-1]),
                'history': history.history
            }
            
            # Save model
            model_path = model_manager.save_model(session['model_id'])
            session['model_path'] = str(model_path)
            
            logger.info(f"Training completed: {training_id}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            session['status'] = 'failed'
            session['error'] = str(e)
            session['failed_at'] = datetime.now().isoformat()
    
    def get_status(self, training_id: str) -> Dict:
        """
        Get training status
        
        Args:
            training_id: Training session ID
            
        Returns:
            Dictionary with training status
        """
        session = self.active_trainings.get(training_id)
        
        if session is None:
            return {'error': 'Training session not found'}
        
        return {
            'id': training_id,
            'status': session['status'],
            'current_epoch': session['current_epoch'],
            'total_epochs': session['total_epochs'],
            'metrics': session.get('metrics', {}),
            'started_at': session['started_at'],
            'elapsed_time': self._calculate_elapsed_time(session)
        }
    
    def stop_training(self, training_id: str) -> bool:
        """
        Stop an active training session
        
        Args:
            training_id: Training session ID
            
        Returns:
            True if stopped, False otherwise
        """
        session = self.active_trainings.get(training_id)
        
        if session and session['status'] == 'running':
            session['status'] = 'stopped'
            session['stopped_at'] = datetime.now().isoformat()
            logger.info(f"Stopped training: {training_id}")
            return True
        
        return False
    
    def _calculate_elapsed_time(self, session: Dict) -> float:
        """Calculate elapsed training time in seconds"""
        if 'started_at' not in session:
            return 0.0
        
        start_time = datetime.fromisoformat(session['started_at'])
        
        if session['status'] == 'completed':
            end_time = datetime.fromisoformat(session['completed_at'])
        elif session['status'] == 'failed':
            end_time = datetime.fromisoformat(session['failed_at'])
        else:
            end_time = datetime.now()
        
        elapsed = (end_time - start_time).total_seconds()
        return elapsed
    
    def list_trainings(self) -> list:
        """List all training sessions"""
        return list(self.active_trainings.values())
