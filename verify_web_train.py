
import requests
import json
import time

def verify_web_train():
    print("--- Verifying Web Search Training ---")
    base_url = "http://127.0.0.1:5000/api"
    
    # 1. Search for images (Class 1: Pizza)
    print("1. Searching for 'pizza'...")
    try:
        resp = requests.post(f"{base_url}/search-images", json={'query': 'pizza', 'max_results': 5})
        if resp.status_code != 200:
            print(f"Search failed: {resp.text}")
            return
        
        pizza_urls = resp.json().get('urls', [])
        print(f"Found {len(pizza_urls)} pizza images.")
        
        # 2. Search for images (Class 2: Burger)
        print("2. Searching for 'burger'...")
        resp = requests.post(f"{base_url}/search-images", json={'query': 'burger', 'max_results': 5})
        burger_urls = resp.json().get('urls', [])
        print(f"Found {len(burger_urls)} burger images.")
        
        if not pizza_urls or not burger_urls:
            print("Failed to get images for both classes.")
            return

        # 3. Train Model
        print("3. Training model with these URLs...")
        
        # Construct form data
        data = {}
        
        # Add Pizza URLs
        for i, url in enumerate(pizza_urls):
            data[f'image_url_{i}'] = url
            data[f'class_url_{i}'] = 'pizza'
            
        # Add Burger URLs (offset index)
        offset = len(pizza_urls)
        for i, url in enumerate(burger_urls):
            data[f'image_url_{offset+i}'] = url
            data[f'class_url_{offset+i}'] = 'burger'
            
        start = time.time()
        # requests.post(..., data=data) sends form-encoded data
        resp = requests.post(f"{base_url}/train-model", data=data, timeout=60)
        duration = time.time() - start
        
        print(f"Training finished in {duration:.2f}s")
        print(f"Status: {resp.status_code}")
        
        if resp.status_code == 200:
            res_json = resp.json()
            if res_json.get('success'):
                print(f"SUCCESS! Model trained.")
                print(f"Accuracy: {res_json.get('accuracy', 0):.2%}")
                print(f"Classes: {res_json.get('classes')}")
            else:
                print(f"FAILURE: {res_json.get('error')}")
        else:
            print(f"FAILURE: HTTP {resp.status_code} - {resp.text}")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    verify_web_train()
