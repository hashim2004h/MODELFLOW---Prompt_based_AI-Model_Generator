
import sys
import os
from pathlib import Path
import logging

# Add project root to Python path
sys.path.insert(0, str(Path.cwd()))

# Mock config BEFORE importing anything that uses it
import os
os.environ['OPENROUTER_API_KEY'] = 'dummy'

from src.models.model_manager import ModelManager
from src.prompt_parser.model_recommender import ModelRecommender
from src.inference.predictor import Predictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_reuse():
    print("--- Verifying Model Reuse ---")
    manager = ModelManager()
    
    # Register a dummy model
    model_info = {
         'name': 'Verification_Model_Reuse',
         'architecture': 'custom',
         'task_type': 'image_classification',
         'framework': 'tensorflow',
         'input_shape': (224, 224, 3),
         'num_classes': 2
    }
    # Pass None as model object for this test (manager allows it for registration, but saving might fail if we tried)
    model_id = manager.register_model("dummy_model_object", model_info)
    print(f"Registered model {model_id}")
    
    # Initialize recommender
    recommender = ModelRecommender(model_manager=manager)
    
    # Ask for recommendation
    recs = recommender.recommend('image_classification')
    
    found = False
    for r in recs:
        if r['name'] == 'Verification_Model_Reuse':
            found = True
            print("SUCCESS: Found user reused model in recommendations!")
            break
            
    if not found:
        print("FAILURE: Did not find user model.")
        
    # Clean up
    manager.delete_model(model_id)

def verify_real_inference_load():
    print("\n--- Verifying Real Model Loading ---")
    manager = ModelManager()
    
    try:
        # Try loading a small pretrained model
        # This might trigger download
        print("Loading MobileNetV2 (this may take time)...")
        data = manager.load_pretrained('mobilenet_v2', 'image_classification')
        model = data['model']
        print(f"SUCCESS: Loaded real model: {model}")
        
    except Exception as e:
        print(f"FAILURE: Could not load real model: {e}")

if __name__ == "__main__":
    verify_reuse()
    # verify_real_inference_load() # Skip potentially long download in this script, user can try in UI
