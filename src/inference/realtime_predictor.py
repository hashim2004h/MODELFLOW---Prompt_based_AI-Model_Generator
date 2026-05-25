"""
Real-time Predictor
Handle real-time predictions (e.g., webcam streaming)
"""

import logging
import numpy as np
from typing import Dict, Callable, Optional
import cv2
import time
from threading import Thread, Lock

logger = logging.getLogger(__name__)


class RealtimePredictor:
    """
    Real-time prediction for streaming data (webcam, video)
    """
    
    def __init__(self, model_id: str):
        """
        Initialize real-time predictor
        
        Args:
            model_id: Model ID to use for predictions
        """
        self.model_id = model_id
        self.is_running = False
        self.current_prediction = None
        self.prediction_lock = Lock()
        self.fps = 0
        self.frame_count = 0
        self.start_time = None
    
    def start_webcam_prediction(self, callback: Optional[Callable] = None,
                               camera_id: int = 0, display: bool = True):
        """
        Start real-time prediction from webcam
        
        Args:
            callback: Optional callback function for predictions
            camera_id: Webcam device ID
            display: Whether to display video feed
        """
        from src.inference.predictor import Predictor
        
        predictor = Predictor()
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")
        
        self.is_running = True
        self.start_time = time.time()
        
        logger.info("Starting webcam prediction")
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning("Failed to capture frame")
                    break
                
                # Resize frame for model
                frame_resized = cv2.resize(frame, (224, 224))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                # Predict
                try:
                    prediction = predictor.predict(self.model_id, frame_rgb)
                    
                    with self.prediction_lock:
                        self.current_prediction = prediction
                    
                    # Callback
                    if callback:
                        callback(prediction)
                    
                    # Update FPS
                    self.frame_count += 1
                    elapsed = time.time() - self.start_time
                    self.fps = self.frame_count / elapsed
                    
                    # Display
                    if display:
                        self._display_prediction(frame, prediction)
                
                except Exception as e:
                    logger.error(f"Prediction error: {e}")
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.is_running = False
            logger.info("Webcam prediction stopped")
    
    def _display_prediction(self, frame: np.ndarray, prediction: Dict):
        """Display prediction on frame"""
        # Add prediction text to frame
        text = f"Class: {prediction.get('predicted_class', 'Unknown')}"
        confidence = prediction.get('confidence', 0)
        text2 = f"Confidence: {confidence:.2f}"
        fps_text = f"FPS: {self.fps:.1f}"
        
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        cv2.putText(frame, text2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        cv2.putText(frame, fps_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 0, 0), 2)
        
        cv2.imshow('Real-time Prediction', frame)
    
    def stop(self):
        """Stop real-time prediction"""
        self.is_running = False
        logger.info("Stopping real-time prediction")
    
    def get_current_prediction(self) -> Optional[Dict]:
        """Get the most recent prediction"""
        with self.prediction_lock:
            return self.current_prediction
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.fps
    
    def predict_video_file(self, video_path: str, output_path: Optional[str] = None):
        """
        Process a video file with predictions
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save annotated video
        """
        from src.inference.predictor import Predictor
        
        predictor = Predictor()
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        predictions_log = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize and predict
                frame_resized = cv2.resize(frame, (224, 224))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                prediction = predictor.predict(self.model_id, frame_rgb)
                predictions_log.append(prediction)
                
                # Annotate frame
                if writer:
                    annotated_frame = self._annotate_frame(frame, prediction)
                    writer.write(annotated_frame)
        
        finally:
            cap.release()
            if writer:
                writer.release()
        
        logger.info(f"Processed video: {len(predictions_log)} frames")
        return predictions_log
    
    def _annotate_frame(self, frame: np.ndarray, prediction: Dict) -> np.ndarray:
        """Annotate frame with prediction"""
        annotated = frame.copy()
        
        text = f"Class: {prediction.get('predicted_class', 'Unknown')}"
        confidence = prediction.get('confidence', 0)
        text2 = f"Conf: {confidence:.2f}"
        
        cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 2)
        cv2.putText(annotated, text2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 2)
        
        return annotated
