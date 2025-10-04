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

# Model hyperparameters
XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}

RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'class_weight': 'balanced',
}

NN_PARAMS = {
    'layer_sizes': [256, 128, 64],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 32,
}

# API settings
API_HOST = '0.0.0.0'
API_PORT = 5000
DEBUG_MODE = True