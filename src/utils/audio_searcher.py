import logging
import requests
import re
from typing import List

logger = logging.getLogger(__name__)

class AudioSearcher:
    """
    Search for audio files on the internet and return preview URLs.
    Uses FreeSound.org public search HTML scraping (no API key required).
    """
    
    def search(self, query: str, max_results: int = 5) -> List[str]:
        try:
            logger.info(f"Searching audio for: {query} via FreeSound")
            url = f"https://freesound.org/search/?q={query}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/91.0.4472.124 Safari/537.36'
            }
            resp = requests.get(url, headers=headers, timeout=10)
            
            if resp.status_code != 200:
                logger.error(f"FreeSound requested failed with status {resp.status_code}")
                return []
                
            # FreeSound HTML contains preview mp3 links matching this pattern
            pattern = r'(https://cdn\.freesound\.org/previews/[0-9]+/[0-9a-zA-Z_-]+\.mp3)'
            matches = re.findall(pattern, resp.text)
            
            # Deduplicate while preserving order and limit to max_results
            unique_urls = []
            for m in matches:
                if m not in unique_urls:
                    unique_urls.append(m)
                    if len(unique_urls) >= max_results:
                        break
                        
            return unique_urls
            
        except Exception as e:
            logger.error(f"Error searching audio: {e}")
            return []
