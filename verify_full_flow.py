import requests
import numpy as np
from PIL import Image
import io
import time

def create_dummy_image(color):
    """Create a 224x224 RGB image of a solid color."""
    img = Image.new('RGB', (224, 224), color)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf

def verify_full_flow():
    base_url = "http://127.0.0.1:5000/api"
    print(f"Testing against {base_url}")
    
    # 1. Prepare Training Data (Red vs Blue)
    # We need at least 5 images total, 2 classes.
    files = []
    data = {}
    
    print("1. Preparing dummy training data...")
    # 3 Red images (Class 'Red')
    for i in range(3):
        img_buf = create_dummy_image((255, 0, 0))
        files.append(('image_r'+str(i), ('red.png', img_buf, 'image/png')))
        data[f'class_r{i}'] = 'Red'
        
    # 3 Blue images (Class 'Blue')
    for i in range(3):
        img_buf = create_dummy_image((0, 0, 255))
        files.append(('image_b'+str(i), ('blue.png', img_buf, 'image/png')))
        data[f'class_b{i}'] = 'Blue'
        
    # 2. Train Model
    print("2. Sending training request (this may take a minute)...")
    try:
        start_time = time.time()
        resp = requests.post(f"{base_url}/train-model", files=files, data=data, timeout=120)
        print(f"Response: {resp.status_code}")
        
        if resp.status_code != 200:
            print(f"Training Failed: {resp.text}")
            return
            
        result = resp.json()
        if not result.get('success'):
            print(f"Training Failed: {result}")
            return
            
        print(f"Training Success! ID: {result['training_id']}, Accuracy: {result['accuracy']:.2%}")
        
        # 3. Verify Prediction
        print("3. Verifying prediction with a Red image...")
        # Create a new Red image
        test_img = create_dummy_image((255, 0, 0))
        
        # Predict
        pred_resp = requests.post(
            f"{base_url}/predict-trained-model",
            files={'file': ('test.png', test_img, 'image/png')}
        )
        
        if pred_resp.status_code != 200:
            print(f"Prediction Request Failed: {pred_resp.text}")
            return
            
        pred_result = pred_resp.json()
        print(f"Prediction Result: {pred_result}")
        
        predictions = pred_result.get('predictions', [])
        if not predictions:
            print("No predictions returned!")
            return
            
        top_class = predictions[0]['class']
        confidence = predictions[0]['confidence']
        
        print(f"Top Class: {top_class}, Confidence: {confidence:.4f}")
        
        if top_class == 'Red' and confidence > 0.8:
            print("✅ VERIFICATION PASSED: Correctly identified Red image with high confidence.")
        else:
            print("❌ VERIFICATION FAILED: Prediction Incorrect or Low Confidence.")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    verify_full_flow()
