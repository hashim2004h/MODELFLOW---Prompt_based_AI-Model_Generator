import requests
import io
import time
from PIL import Image

def create_image(color):
    img = Image.new('RGB', (224, 224), color)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return ('image.png', buf, 'image/png')

def verify_model(backbone_name):
    print(f"\n--- Testing Backbone: {backbone_name} ---")
    files = []
    data = {'backbone': backbone_name}
    
    # 5 Red, 5 Blue
    for i in range(5):
        files.append(('image_r'+str(i), create_image('red')))
        data[f'class_r{i}'] = 'Red'
        files.append(('image_b'+str(i), create_image('blue')))
        data[f'class_b{i}'] = 'Blue'
        
    start = time.time()
    resp = requests.post("http://127.0.0.1:5000/api/train-model", files=files, data=data, timeout=300)
    duration = time.time() - start
    
    if resp.status_code == 200:
        res = resp.json()
        if res.get('success'):
            print(f"✅ Success! Trained in {duration:.1f}s. Accuracy: {res['accuracy']:.2%}")
        else:
            print(f"❌ Failed: {res.get('error')}")
    else:
        print(f"❌ HTTP Error: {resp.status_code} - {resp.text}")

if __name__ == "__main__":
    # 1. Test Custom NN
    verify_model('custom_cnn')
    
    # 2. Test AutoML
    verify_model('automl')
