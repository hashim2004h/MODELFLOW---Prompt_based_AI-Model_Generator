
import logging
from typing import List
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

logger = logging.getLogger(__name__)

class ImageSearcher:
    """
    Search for images on the internet and return URLs.
    Uses duckduckgo_search library for robust access.
    """

    def __init__(self):
        if DDGS is None:
            logger.error("duckduckgo_search library not installed. Please run: pip install duckduckgo-search")

    def search(self, query: str, max_results: int = 5) -> List[str]:
        """
        Search for images using DuckDuckGo.
        """
        if DDGS is None:
            return []

        try:
            logger.info(f"Searching images for: {query} via DuckDuckGo (Lib)")
            
            image_urls = []
            with DDGS() as ddgs:
                # max_results kwarg might be 'max_results' or just iterate
                # version 6+ uses max_results
                results = ddgs.images(
                    keywords=query,
                    region="wt-wt",
                    safesearch="off",
                    max_results=max_results
                )
                
                for r in results:
                    url = r.get('image')
                    if url:
                        image_urls.append(url)
            
            return image_urls

        except Exception as e:
            logger.error(f"Error searching images: {e}")
            return []
