import requests
import re

def search(q):
    url = f"https://freesound.org/search/?q={q}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(url, headers=headers)
    
    # Try to find mp3 previews in the HTML
    # freesound audio elements usually look like <a data-mp3="https://cdn.freesound.org/previews/123/123456_1234-hq.mp3" ...>
    urls = re.findall(r'(https://cdn\.freesound\.org/previews/[0-9]+/[0-9a-zA-Z_-]+\.mp3)', r.text)
    print("Found urls:", set(urls[:5]))

search("cat")
