"""
Fine-tuning API routes
Allows users to train custom models with their uploaded datasets
"""

from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
import joblib

# Add parent directory to path to import from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.model_training import ModelTrainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

train_bp = Blueprint('train', __name__)

@train_bp.route('/upload_labeled', methods=['POST'])
def upload_labeled_data():
    """
    Upload raw labeled data for fine-tuning (keeps disposition column)
    
    Form Data:
    - file: CSV file with disposition column
    - dataset_type: 'tess' or 'kepler'
    - keep_labels: 'true' (flag to preserve disposition)
    
    Returns:
    {
        "success": true,
        "upload_id": "abc123",
        "filename": "training_data.csv",
        "summary": {
            "total_rows": 5000,
            "disposition_column": "koi_disposition",
            "planet_count": 3000,
            "fp_count": 2000
        }
    }
    """
    try:
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        dataset_type = request.form.get('dataset_type', 'tess')
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({
                'success': False,
                'error': 'File must be a CSV'
            }), 400
        
        # Generate unique ID for this upload
        import uuid
        upload_id = str(uuid.uuid4())[:12]
        
        # Save original file temporarily
        temp_dir = current_app.config['TEMP_DATA_FOLDER']
        original_path = os.path.join(temp_dir, f"{upload_id}_labeled_raw.csv")
        file.save(original_path)
        
        print(f"\n{'='*70}")
        print(f"UPLOADING LABELED DATA FOR TRAINING")
        print(f"{'='*70}")
        print(f"Upload ID: {upload_id}")
        print(f"Filename: {file.filename}")
        print(f"Dataset Type: {dataset_type}")
        
        # Load and check for disposition column
        df = pd.read_csv(original_path)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Find disposition column
        disposition_col = None
        for col in ['koi_disposition', 'tfopwg_disp', 'disposition']:
            if col in df.columns:
                disposition_col = col
                break
        
        if disposition_col is None:
            # Try case-insensitive search
            for col in df.columns:
                if 'disposition' in col.lower():
                    disposition_col = col
                    break
        
        if disposition_col is None:
            return jsonify({
                'success': False,
                'error': f'No disposition column found. Available columns: {", ".join(df.columns[:10])}...'
            }), 400
        
        print(f"Found disposition column: {disposition_col}")
        
        # Count labels
        disposition_counts = df[disposition_col].value_counts()
        print(f"\nLabel distribution:")
        for label, count in disposition_counts.items():
            print(f"  {label}: {count}")
        
        # Count planets vs false positives
        planet_values = ['CONFIRMED', 'CANDIDATE', 'CP', 'PC', 'KP', 'APC']
        fp_values = ['FALSE POSITIVE', 'FP', 'FA']
        
        planet_count = df[disposition_col].isin(planet_values).sum()
        fp_count = df[disposition_col].isin(fp_values).sum()
        
        print(f"\nBinary classification:")
        print(f"  Planets: {planet_count}")
        print(f"  False Positives: {fp_count}")
        
        # Save to training data folder
        training_path = os.path.join(temp_dir, f"{upload_id}_training.csv")
        df.to_csv(training_path, index=False)
        
        print(f"\nSaved training data to: {training_path}")
        print(f"{'='*70}")
        
        return jsonify({
            'success': True,
            'upload_id': upload_id,
            'filename': file.filename,
            'summary': {
                'total_rows': len(df),
                'disposition_column': disposition_col,
                'planet_count': int(planet_count),
                'fp_count': int(fp_count),
            }
        })
        
    except Exception as e:
        print(f"Error uploading labeled data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@train_bp.route('/finetune', methods=['POST'])
def finetune_models():
    """
    Fine-tune models with user's uploaded dataset
    
    Request JSON:
    {
        "upload_id": "abc123",
        "models": ["xgboost", "random_forest", "neural_network"],
        "xgboost_params": {...},
        "random_forest_params": {...},
        "neural_network_params": {...},
        "validation_split": 0.15,
        "test_split": 0.15
    }
    
    Returns:
    {
        "success": true,
        "results": {
            "xgboost": {"accuracy": 0.85, "precision": 0.84, ...},
            "random_forest": {...},
            "neural_network": {...}
        },
        "model_dir": "models/finetuned/timestamp_id/",
        "training_info": {...}
    }
    """
    try:
        data = request.json
        upload_id = data.get('upload_id')
        models_to_train = data.get('models', [])
        xgb_params = data.get('xgboost_params')
        rf_params = data.get('random_forest_params')
        nn_params = data.get('neural_network_params')
        validation_split = data.get('validation_split', 0.15)
        test_split = data.get('test_split', 0.15)
        
        if not upload_id:
            return jsonify({
                'success': False,
                'error': 'upload_id is required'
            }), 400
        
        if not models_to_train:
            return jsonify({
                'success': False,
                'error': 'At least one model must be selected'
            }), 400
        
        # Load training data (raw labeled data)
        temp_dir = current_app.config['TEMP_DATA_FOLDER']
        data_path = os.path.join(temp_dir, f"{upload_id}_training.csv")
        
        if not os.path.exists(data_path):
            return jsonify({
                'success': False,
                'error': 'Training data not found. Please upload labeled data first from the Fine-Tune page.'
            }), 404
        
        print(f"\n{'='*70}")
        print(f"FINE-TUNING MODELS")
        print(f"{'='*70}")
        print(f"Upload ID: {upload_id}")
        print(f"Models: {', '.join(models_to_train)}")
        
        # Load raw data
        df_raw = pd.read_csv(data_path)
        print(f"Loaded {len(df_raw)} samples (raw)")
        
        # Check for disposition column (labels)
        disposition_col = None
        for col in ['koi_disposition', 'tfopwg_disp', 'disposition']:
            if col in df_raw.columns:
                disposition_col = col
                break
        
        if disposition_col is None:
            return jsonify({
                'success': False,
                'error': 'No disposition column found. Fine-tuning requires labeled data.'
            }), 400
        
        print(f"Disposition column: {disposition_col}")
        
        # Preprocess data (column mapping, feature engineering, etc.)
        from api.utils.preprocessor import preprocess_data
        
        # Extract disposition before preprocessing
        disposition_series = df_raw[disposition_col].copy()
        
        # Preprocess (this will add engineered features)
        df_processed = preprocess_data(df_raw, dataset_type='tess')  # Auto-detect or use metadata
        
        print(f"Preprocessed to {len(df_processed)} rows, {len(df_processed.columns)} features")
        
        # Prepare features (exclude disposition if it somehow remained)
        X = df_processed.drop(columns=[disposition_col], errors='ignore')
        
        # Convert disposition to binary (1 = planet, 0 = false positive)
        y = disposition_series.copy()
        # Align y with X after preprocessing (in case rows were dropped)
        y = y.loc[X.index]
        
        planet_values = ['CONFIRMED', 'CANDIDATE', 'CP', 'PC', 'KP', 'APC']
        fp_values = ['FALSE POSITIVE', 'FP', 'FA']
        
        y = y.replace(planet_values, 1)
        y = y.replace(fp_values, 0)
        
        # Remove any rows with invalid labels
        valid_mask = y.isin([0, 1])
        X = X[valid_mask]
        y = y[valid_mask].astype(int)
        
        print(f"\nLabel distribution:")
        print(f"  Planets: {(y == 1).sum()}")
        print(f"  False Positives: {(y == 0).sum()}")
        
        # Split data
        train_size = 1.0 - validation_split - test_split
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42, stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = validation_split / (train_size + validation_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"\nData split:")
        print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Scale data for neural network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Create directory for fine-tuned models
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{timestamp}_{upload_id[:8]}"
        model_dir = os.path.join('models', 'finetuned', model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"\nSaving models to: {model_dir}")
        
        # Initialize trainer
        trainer = ModelTrainer()
        results = {}
        
        # Train selected models
        if 'xgboost' in models_to_train:
            print(f"\n{'='*50}")
            print("Training XGBoost...")
            print(f"{'='*50}")
            
            # Use custom params or defaults
            params = xgb_params if xgb_params else None
            trainer.train_xgboost(X_train, y_train, X_val, y_val, params=params)
            
            # Evaluate
            y_pred = trainer.xgb_model.predict(X_test)
            y_pred_proba = trainer.xgb_model.predict_proba(X_test)[:, 1]
            
            results['xgboost'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_pred_proba),
            }
            
            # Save model
            model_path = os.path.join(model_dir, 'xgboost_model.pkl')
            joblib.dump(trainer.xgb_model, model_path)
            print(f"✓ Saved XGBoost to {model_path}")
            print(f"  Test Accuracy: {results['xgboost']['accuracy']:.4f}")
            print(f"  Test AUC: {results['xgboost']['auc']:.4f}")
        
        if 'random_forest' in models_to_train:
            print(f"\n{'='*50}")
            print("Training Random Forest...")
            print(f"{'='*50}")
            
            # Use custom params or defaults
            params = rf_params if rf_params else None
            trainer.train_random_forest(X_train, y_train, params=params)
            
            # Evaluate
            y_pred = trainer.rf_model.predict(X_test)
            y_pred_proba = trainer.rf_model.predict_proba(X_test)[:, 1]
            
            results['random_forest'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_pred_proba),
            }
            
            # Save model
            model_path = os.path.join(model_dir, 'random_forest_model.pkl')
            joblib.dump(trainer.rf_model, model_path)
            print(f"✓ Saved Random Forest to {model_path}")
            print(f"  Test Accuracy: {results['random_forest']['accuracy']:.4f}")
            print(f"  Test AUC: {results['random_forest']['auc']:.4f}")
        
        if 'neural_network' in models_to_train:
            print(f"\n{'='*50}")
            print("Training Neural Network...")
            print(f"{'='*50}")
            
            # Use custom params or defaults
            params = nn_params if nn_params else None
            trainer.train_neural_network(
                X_train_scaled, y_train, 
                X_val_scaled, y_val, 
                input_dim=X_train.shape[1],
                params=params
            )
            
            # Evaluate
            y_pred_proba = trainer.nn_model.predict(X_test_scaled, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            results['neural_network'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_pred_proba),
            }
            
            # Save model
            model_path = os.path.join(model_dir, 'neural_network_model.h5')
            trainer.nn_model.save(model_path)
            print(f"✓ Saved Neural Network to {model_path}")
            print(f"  Test Accuracy: {results['neural_network']['accuracy']:.4f}")
            print(f"  Test AUC: {results['neural_network']['auc']:.4f}")
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        
        # Save feature names
        feature_names_path = os.path.join(model_dir, 'feature_names.json')
        with open(feature_names_path, 'w') as f:
            json.dump(list(X.columns), f, indent=2)
        
        # Save training metadata
        metadata = {
            'timestamp': timestamp,
            'upload_id': upload_id,
            'models_trained': models_to_train,
            'total_samples': len(X) + len(X_val) + len(X_test),  # Total before split
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'num_features': X.shape[1],
            'feature_names': list(X.columns),
            'hyperparameters': {
                'xgboost': xgb_params,
                'random_forest': rf_params,
                'neural_network': nn_params,
            },
            'data_splits': {
                'train': train_size,
                'validation': validation_split,
                'test': test_split,
            },
            'results': results,
        }
        
        metadata_path = os.path.join(model_dir, 'training_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"FINE-TUNING COMPLETE!")
        print(f"{'='*70}")
        print(f"Models saved to: {model_dir}")
        print(f"Metadata saved to: {metadata_path}")
        
        return jsonify({
            'success': True,
            'results': results,
            'model_dir': model_dir,
            'model_id': model_id,
            'training_info': {
                'total_samples': len(X) + len(X_val) + len(X_test),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'num_features': X.shape[1],
            },
            'timestamp': timestamp,
        })
        
    except Exception as e:
        print(f"Error in fine-tuning: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@train_bp.route('/models/list', methods=['GET'])
def list_finetuned_models():
    """
    List all fine-tuned models
    
    Returns:
    {
        "success": true,
        "models": [
            {
                "model_id": "20250105_123456_abc12345",
                "timestamp": "2025-01-05 12:34:56",
                "models": ["xgboost", "random_forest"],
                "samples": 5000,
                "features": 27,
                "results": {...}
            }
        ]
    }
    """
    try:
        models_dir = os.path.join('models', 'finetuned')
        
        if not os.path.exists(models_dir):
            return jsonify({
                'success': True,
                'models': []
            })
        
        models_list = []
        
        for model_id in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_id)
            metadata_path = os.path.join(model_path, 'training_metadata.json')
            
            if os.path.isdir(model_path) and os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                models_list.append({
                    'model_id': model_id,
                    'timestamp': metadata.get('timestamp'),
                    'models': metadata.get('models_trained', []),
                    'samples': metadata.get('total_samples'),
                    'features': metadata.get('num_features'),
                    'results': metadata.get('results', {}),
                })
        
        # Sort by timestamp (newest first)
        models_list.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'success': True,
            'models': models_list
        })
        
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

