import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.data_pipeline import DataPipeline
from src.model_training import ModelTrainer
import config

# ============================================================
# TRAINING CONFIGURATION
# ============================================================
# Options: 'both' (Kepler + TESS), 'kepler', or 'tess'
DATASET_TYPE = 'both'  # DEFAULT: Train on combined dataset
# ============================================================


def train_models(dataset_type='both'):

    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    print("="*70)
    print("EXOPLANET DETECTION MODEL TRAINING")
    print("="*70)
    print(f"Dataset type: {dataset_type or 'auto-detect'}")
    print("="*70)
    
    # ------------------ Data Pipeline ------------------
    print("\n[1/4] Loading and preparing data...")
    pipeline = DataPipeline(dataset_type=dataset_type)
    df = pipeline.load_data()
    X, y, df_clean = pipeline.prepare_dataset(df)
    
    print(f"\n[2/4] Splitting and scaling data...")
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
    print(f"\n[4/4] Evaluating models on test set...")
    trainer.evaluate_models(X_test_scaled=X_test_scaled, X_test=X_test, y_test=y_test)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"✓ Models saved in: {config.MODEL_DIR}")
    print(f"✓ Dataset: {pipeline.dataset_type.upper()}")
    print(f"✓ Features: {len(pipeline.feature_names)}")
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Test samples: {len(X_test)}")
    print("="*70)
    
    # Return results for frontend integration
    return {
        'success': True,
        'dataset_type': pipeline.dataset_type,
        'feature_count': len(pipeline.feature_names),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'planet_count': int(y.sum()),
        'false_positive_count': int(len(y) - y.sum()),
        'model_dir': config.MODEL_DIR,
    }


def main():
    """
    Main function - uses configuration variables at top of file
    """
    result = train_models(
        dataset_type=DATASET_TYPE
    )
    return result

if __name__ == "__main__":
    main()
