
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.main import create_app
from config import HOST, PORT, DEBUG, create_directories

def main():
    """Initialize and run the application"""
    # Create necessary directories
    create_directories()
    
    # Create Flask app
    app = create_app()
    
    # Run the application
    print(f"Starting MODELFLOW on http://{HOST}:{PORT}")
    print(f"Debug mode: {DEBUG}")
    
    app.run(
        host=HOST,
        port=PORT,
        debug=DEBUG,
        threaded=True
    )

if __name__ == '__main__':
    main()
