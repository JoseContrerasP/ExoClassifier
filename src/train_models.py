import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.data_pipeline import DataPipeline
from src.model_training import ModelTrainer
import config

def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # ------------------ Data Pipeline ------------------
    pipeline = DataPipeline()
    df = pipeline.load_data()
    X, y, df_clean = pipeline.prepare_dataset(df)
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, X_train, X_val, X_test = pipeline.split_and_scale(X, y)
    pipeline.save_artifacts()
    
    # ------------------ Model Training ------------------
    trainer = ModelTrainer()
    trainer.train_xgboost(X_train, y_train, X_val, y_val)
    trainer.train_random_forest(X_train, y_train)
    trainer.train_neural_network(X_train_scaled, y_train, X_val_scaled, y_val, input_dim=X_train_scaled.shape[1])
    
    # ------------------ Save Models ------------------
    trainer.save_models(save_dir=config.MODEL_DIR)
    
    # ------------------ Evaluate Models ------------------
    trainer.evaluate_models(X_test_scaled=X_test_scaled, X_test=X_test, y_test=y_test)
    
    print(f"\n✓ Models saved in: {config.MODEL_DIR}")

if __name__ == "__main__":
    main()
