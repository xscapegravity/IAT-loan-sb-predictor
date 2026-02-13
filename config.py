import os

class Config:
    """Base configuration."""
    DEBUG = os.environ.get('FLASK_DEBUG', 'True') == 'True'
    PORT = int(os.environ.get('FLASK_PORT', 8000))
    MODEL_PATH = os.environ.get('MODEL_PATH', 'model.joblib')
    PREPROCESSOR_PATH = os.environ.get('PREPROCESSOR_PATH', 'preprocessor.joblib')
