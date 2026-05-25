
import requests
import json
import time

def verify_auto_analyze():
    print("--- Verifying Auto-Analysis Feature ---")
    url = "http://127.0.0.1:5000/api/auto-analyze"
    
    # Payload
    payload = {
        "query": "golden retriever",
        "model_name": "MobileNetV2", # UI sends this, backend should now handle it
        "max_images": 2
    }
    
    try:
        print(f"Sending request to {url}...")
        start = time.time()
        response = requests.post(url, json=payload, timeout=30)
        duration = time.time() - start
        
        print(f"Response status: {response.status_code}")
        print(f"Duration: {duration:.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"SUCCESS: Found {len(data.get('results', []))} images.")
                for res in data['results']:
                    if res.get('status') == 'success':
                        print(f" - Image: {res['url'][:30]}... | Prediction: {res.get('label')} ({res.get('confidence', 0):.2f})")
                    else:
                        print(f" - ERROR for image {res.get('url')}...: {res.get('error')}")
            else:
                print(f"FAILURE: API returned success=False. Error: {data.get('error')}")
        else:
            print(f"FAILURE: HTTP {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"FAILURE: Request failed: {e}")

if __name__ == "__main__":
    verify_auto_analyze()
