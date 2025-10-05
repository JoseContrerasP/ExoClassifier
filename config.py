import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

RAW_DATA_PATH = os.path.join(DATA_DIR, 'cumulative_koi.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.pkl')

XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
NN_MODEL_PATH = os.path.join(MODEL_DIR, 'neural_network_model.h5')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, 'feature_names.json')

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Feature engineering
CORRELATION_THRESHOLD = 0.80  

# Model hyperparameters
# Tuned for better performance on combined Kepler + TESS data
XGBOOST_PARAMS = {
    'max_depth': 7,              # Increased from 6 for more complex patterns
    'learning_rate': 0.05,       # Lower learning rate for better generalization
    'n_estimators': 300,         # More trees for better performance
    'subsample': 0.85,           # Slightly increased
    'colsample_bytree': 0.85,    # Slightly increased
    'min_child_weight': 3,       # Add regularization
    'gamma': 0.1,                # Add regularization
    'reg_alpha': 0.01,           # L1 regularization
    'reg_lambda': 2,             # L2 regularization
}

RF_PARAMS = {
    'n_estimators': 300,         # More trees
    'max_depth': 25,             # Deeper trees
    'min_samples_split': 3,      # Lower threshold
    'min_samples_leaf': 2,       # Add regularization
    'max_features': 'sqrt',      # Better feature sampling
    'class_weight': 'balanced',
}

NN_PARAMS = {
    'layer_sizes': [512, 256, 128, 64],  # Deeper network
    'dropout_rate': 0.4,         # More dropout to prevent overfitting
    'learning_rate': 0.0008,     # Lower learning rate
    'epochs': 150,               # More epochs
    'batch_size': 32,            # Larger batch size
}

# API settings
API_HOST = '0.0.0.0'
API_PORT = 5000
DEBUG_MODE = True