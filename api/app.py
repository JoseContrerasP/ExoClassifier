import os
import sys
from flask import Flask
from flask_cors import CORS

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PROJECT_ROOT)

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Configuration
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['UPLOAD_FOLDER'] = os.path.join(PROJECT_ROOT, 'data', 'uploads')
    app.config['TEMP_DATA_FOLDER'] = os.path.join(PROJECT_ROOT, 'data', 'temp')
    
    # Ensure directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['TEMP_DATA_FOLDER'], exist_ok=True)
    
    # Enable CORS for frontend communication
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Register blueprints
    from api.routes.upload import upload_bp
    from api.routes.predict import predict_bp
    from api.routes.data import data_bp
    from api.routes.train import train_bp
    
    app.register_blueprint(upload_bp, url_prefix='/api/upload')
    app.register_blueprint(predict_bp, url_prefix='/api/predict')
    app.register_blueprint(data_bp, url_prefix='/api/data')
    app.register_blueprint(train_bp, url_prefix='/api/train')
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return {
            'status': 'healthy',
            'message': 'Exoplanet Detection API is running'
        }, 200
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)

