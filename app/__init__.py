"""
Flask Application Initialization
"""
from flask import Flask
from flask_cors import CORS
import logging
from pathlib import Path

def create_app(config=None):
    """
    Application factory pattern
    Creates and configures the Flask application
    """
    app = Flask(__name__,
                template_folder='templates',
                static_folder='static')
    
    # Load configuration
    app.config.from_object('config')
    if config:
        app.config.update(config)
    
    # Enable CORS
    CORS(app)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Register blueprints
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {'error': 'Internal server error'}, 500
    
    return app
