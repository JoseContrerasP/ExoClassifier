
import sys
import os
from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
sys.path.append(PROJECT_ROOT)

from src.column_mapper import ColumnMapper
from src.feature_engineering import FeatureEngineer
from src.ensemble import EnsemblePredictor
from api.utils.validators import validate_feature_ranges

predict_bp = Blueprint('predict', __name__)

# Initialize components
column_mapper = ColumnMapper()
feature_engineer = FeatureEngineer()

# Load models (lazy loading)
_ensemble_predictor = None

def get_ensemble_predictor():
    """Lazy load ensemble predictor"""
    global _ensemble_predictor
    if _ensemble_predictor is None:
        try:
            _ensemble_predictor = EnsemblePredictor()
            print("[PREDICTOR] Ensemble models loaded successfully")
        except Exception as e:
            print(f"[PREDICTOR ERROR] Failed to load models: {e}")
            raise
    return _ensemble_predictor

@predict_bp.route('/single', methods=['POST'])
def predict_single():
    """
    Predict single exoplanet candidate (manual entry)
    
    Expected JSON:
        - dataset_type: 'tess' or 'kepler'
        - features: Dictionary of feature values
        
    Returns:
        - prediction: Classification result
        - probability: Confidence score
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        dataset_type = data.get('dataset_type', 'tess')
        features = data.get('features', {})
        
        if not features:
            return jsonify({
                'success': False,
                'error': 'No features provided'
            }), 400
        
        # Validate feature ranges
        validation_result = validate_feature_ranges(features)
        if not validation_result['valid']:
            return jsonify({
                'success': False,
                'error': 'Input validation failed',
                'validation_errors': validation_result['errors']
            }), 400
        
        # Convert to DataFrame for processing
        df = pd.DataFrame([features])
        
        # Map columns to standard format
        df_mapped, _ = column_mapper.map_columns(df, dataset_type)
        
        # For TESS, validate and convert (auto-calculate missing fields)
        if dataset_type == 'tess':
            df_processed, _ = column_mapper.validate_and_convert(df_mapped)
        else:
            df_processed = df_mapped
        
        # Apply feature engineering
        df_engineered = feature_engineer.engineer_all_features(df_processed)
        
        # Get ensemble predictor
        predictor = get_ensemble_predictor()
        
        # Prepare features for prediction
        feature_names = predictor.feature_names
        
        # Debug: Check which features are available
        available_features = df_engineered.columns.tolist()
        missing_features = [f for f in feature_names if f not in available_features]
        if missing_features:
            print(f"[WARNING] Missing features: {missing_features}")
        
        X = df_engineered[feature_names].values
        
        # Debug: Check for NaN/Inf
        if np.isnan(X).any() or np.isinf(X).any():
            print(f"[WARNING] NaN or Inf detected in features!")
            print(f"  NaN count: {np.isnan(X).sum()}")
            print(f"  Inf count: {np.isinf(X).sum()}")
        
        X_scaled = predictor.scaler.transform(X)
        
        # Debug: Check scaled values
        print(f"[DEBUG] X shape: {X.shape}, X_scaled range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
        
        # Debug: Find outlier features
        outlier_threshold = 10.0
        outlier_mask = np.abs(X_scaled[0]) > outlier_threshold
        if outlier_mask.any():
            outlier_indices = np.where(outlier_mask)[0]
            print(f"[WARNING] Outlier features detected (|scaled| > {outlier_threshold}):")
            for idx in outlier_indices:
                feature_name = feature_names[idx]
                raw_value = X[0, idx]
                scaled_value = X_scaled[0, idx]
                print(f"  - {feature_name}: raw={raw_value:.4f}, scaled={scaled_value:.2f}")
        
        # Make prediction
        prediction_proba = predictor.predict_proba(X, X_scaled)[0]
        prediction_class = predictor.predict(X, X_scaled)[0]
        
        # Get individual model predictions
        xgb_proba = predictor.xgb_model.predict_proba(X)[:,1][0]
        rf_proba = predictor.rf_model.predict_proba(X)[:,1][0]
        nn_raw = predictor.nn_model.predict(X_scaled, verbose=0)
        nn_proba = float(nn_raw.flatten()[0])
        
        # Debug logging
        print(f"[DEBUG] XGB: {xgb_proba:.4f}, RF: {rf_proba:.4f}, NN: {nn_proba:.4f}, Ensemble: {prediction_proba:.4f}")
        
        # Determine classification
        classification = "CONFIRMED PLANET" if prediction_class == 1 else "FALSE POSITIVE"
        confidence = "HIGH" if abs(prediction_proba - 0.5) > 0.3 else "MEDIUM" if abs(prediction_proba - 0.5) > 0.15 else "LOW"
        
        return jsonify({
            'success': True,
            'prediction': {
                'classification': classification,
                'is_planet': bool(prediction_class == 1),
                'probability': float(prediction_proba),
                'confidence': confidence
            },
            'model_breakdown': {
                'xgboost': float(xgb_proba),
                'random_forest': float(rf_proba),
                'neural_network': float(nn_proba),
                'ensemble': float(prediction_proba)
            },
            'metadata': {
                'dataset_type': dataset_type,
                'features_used': len(feature_names),
                'auto_calculated': ['koi_model_snr', 'koi_smass', 'koi_sma'] if dataset_type == 'tess' else []
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@predict_bp.route('/batch', methods=['POST'])
def predict_batch():
    """
    Predict batch of candidates from uploaded CSV
    
    Expected JSON:
        - upload_id: ID from previous CSV upload
        - model_type: 'ensemble', 'xgboost', 'random_forest', or 'neural_network'
        
    Returns:
        - predictions: Summary of predictions
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        upload_id = data.get('upload_id')
        model_type = data.get('model_type', 'ensemble')
        
        if not upload_id:
            return jsonify({
                'success': False,
                'error': 'No upload_id provided'
            }), 400
        
        # Check if using fine-tuned model
        is_finetuned = model_type.startswith('finetuned:')
        if is_finetuned:
            finetuned_model_id = model_type.replace('finetuned:', '')
            print(f"[BATCH] Using fine-tuned model: {finetuned_model_id}")
        
        # Load the processed data from temp folder
        temp_data_folder = os.path.join(PROJECT_ROOT, 'data', 'temp')
        data_file = os.path.join(temp_data_folder, f"{upload_id}_processed.csv")
        
        if not os.path.exists(data_file):
            return jsonify({
                'success': False,
                'error': f'Dataset not found for upload_id: {upload_id}'
            }), 404
        
        # Read preprocessed data
        df = pd.read_csv(data_file)
        print(f"[BATCH] Loaded {len(df)} rows for prediction")
        
        # Load predictor (base or fine-tuned)
        if is_finetuned:
            # Load fine-tuned model
            finetuned_dir = os.path.join(PROJECT_ROOT, 'models', 'finetuned', finetuned_model_id)
            if not os.path.exists(finetuned_dir):
                return jsonify({
                    'success': False,
                    'error': f'Fine-tuned model not found: {finetuned_model_id}'
                }), 404
            
            # Load metadata to determine which models were trained
            import json
            with open(os.path.join(finetuned_dir, 'training_metadata.json'), 'r') as f:
                metadata = json.load(f)
            
            # Load the fine-tuned model
            import joblib
            from tensorflow import keras
            
            trained_models = metadata['models_trained']
            if 'xgboost' in trained_models:
                xgb_model = joblib.load(os.path.join(finetuned_dir, 'xgboost_model.pkl'))
            if 'random_forest' in trained_models:
                rf_model = joblib.load(os.path.join(finetuned_dir, 'random_forest_model.pkl'))
            if 'neural_network' in trained_models:
                nn_model = keras.models.load_model(os.path.join(finetuned_dir, 'neural_network_model.h5'))
            
            scaler = joblib.load(os.path.join(finetuned_dir, 'scaler.pkl'))
            with open(os.path.join(finetuned_dir, 'feature_names.json'), 'r') as f:
                feature_names = json.load(f)
            
            print(f"[BATCH] Loaded fine-tuned model with {len(feature_names)} features")
        else:
            # Get base ensemble predictor
            predictor = get_ensemble_predictor()
            feature_names = predictor.feature_names
        
        # Prepare features
        X = df[feature_names].values
        
        # Make predictions based on model type
        if is_finetuned:
            # Use fine-tuned model
            X_scaled = scaler.transform(X)
            model_name = trained_models[0]  # Use the first (and only) trained model
            
            if model_name == 'xgboost':
                predictions_proba = xgb_model.predict_proba(X)[:,1]
                predictions = (predictions_proba > 0.5).astype(int)
            elif model_name == 'random_forest':
                predictions_proba = rf_model.predict_proba(X)[:,1]
                predictions = (predictions_proba > 0.5).astype(int)
            elif model_name == 'neural_network':
                predictions_proba = nn_model.predict(X_scaled, verbose=0).flatten()
                predictions = (predictions_proba > 0.5).astype(int)
            
            print(f"[BATCH] Used fine-tuned {model_name}")
        elif model_type == 'ensemble':
            X_scaled = predictor.scaler.transform(X)
            predictions_proba = predictor.predict_proba(X, X_scaled)
            predictions = predictor.predict(X, X_scaled)
        elif model_type == 'xgboost':
            predictions_proba = predictor.xgb_model.predict_proba(X)[:,1]
            predictions = (predictions_proba >= 0.5).astype(int)
        elif model_type == 'random_forest':
            predictions_proba = predictor.rf_model.predict_proba(X)[:,1]
            predictions = (predictions_proba >= 0.5).astype(int)
        elif model_type == 'neural_network':
            X_scaled = predictor.scaler.transform(X)
            predictions_proba = predictor.nn_model.predict(X_scaled, verbose=0).flatten()
            predictions = (predictions_proba >= 0.5).astype(int)
        else:
            return jsonify({
                'success': False,
                'error': f'Invalid model_type: {model_type}'
            }), 400
        
        # Calculate summary statistics
        planet_count = int(np.sum(predictions == 1))
        false_positive_count = int(np.sum(predictions == 0))
        
        # Save predictions to file
        results_df = df.copy()
        results_df['prediction'] = predictions
        results_df['probability'] = predictions_proba
        results_df['classification'] = ['CONFIRMED PLANET' if p == 1 else 'FALSE POSITIVE' for p in predictions]
        
        results_file = os.path.join(temp_data_folder, f"{upload_id}_predictions.csv")
        results_df.to_csv(results_file, index=False)
        
        print(f"[BATCH] Predictions complete: {planet_count} planets, {false_positive_count} false positives")
        
        return jsonify({
            'success': True,
            'predictions': {
                'total_predictions': len(predictions),
                'planet_count': planet_count,
                'false_positive_count': false_positive_count,
                'model_used': model_type,
                'results_file': f"{upload_id}_predictions.csv",
                'upload_id': upload_id
            }
        }), 200
        
    except Exception as e:
        print(f"[BATCH ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Batch prediction failed: {str(e)}'
        }), 500

@predict_bp.route('/export/<upload_id>', methods=['GET'])
def export_predictions(upload_id):
    """
    Export prediction results as CSV
    
    Returns:
        CSV file with all predictions and probabilities
    """
    try:
        from flask import send_file
        
        # Find the predictions file
        temp_data_folder = os.path.join(PROJECT_ROOT, 'data', 'temp')
        results_file = os.path.join(temp_data_folder, f"{upload_id}_predictions.csv")
        
        if not os.path.exists(results_file):
            return jsonify({
                'success': False,
                'error': f'Predictions not found for upload_id: {upload_id}'
            }), 404
        
        # Send file as download
        return send_file(
            results_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'exoplanet_predictions_{upload_id}.csv'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Export failed: {str(e)}'
        }), 500

@predict_bp.route('/details/<upload_id>', methods=['GET'])
def get_prediction_details(upload_id):
    """
    Get detailed prediction statistics and breakdown
    
    Returns:
        Detailed statistics, probability distribution, etc.
    """
    try:
        # Find the predictions file
        temp_data_folder = os.path.join(PROJECT_ROOT, 'data', 'temp')
        results_file = os.path.join(temp_data_folder, f"{upload_id}_predictions.csv")
        
        if not os.path.exists(results_file):
            return jsonify({
                'success': False,
                'error': f'Predictions not found for upload_id: {upload_id}'
            }), 404
        
        # Read predictions
        df = pd.read_csv(results_file)
        
        # Calculate detailed statistics
        planet_predictions = df[df['prediction'] == 1]
        fp_predictions = df[df['prediction'] == 0]
        
        # Probability distribution bins
        prob_bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
        prob_labels = ['Very Low (0-30%)', 'Low (30-50%)', 'Medium (50-70%)', 'High (70-90%)', 'Very High (90-100%)']
        df['confidence_bin'] = pd.cut(df['probability'], bins=prob_bins, labels=prob_labels, include_lowest=True)
        confidence_distribution = df['confidence_bin'].value_counts().to_dict()
        
        # Top confident planets (high probability)
        top_planets = planet_predictions.nlargest(10, 'probability')[
            ['koi_period', 'koi_depth', 'koi_prad', 'koi_teq', 'probability', 'classification']
        ].to_dict('records') if len(planet_predictions) > 0 else []
        
        # Top confident false positives (low probability)
        top_fps = fp_predictions.nsmallest(10, 'probability')[
            ['koi_period', 'koi_depth', 'koi_prad', 'koi_teq', 'probability', 'classification']
        ].to_dict('records') if len(fp_predictions) > 0 else []
        
        # Sample predictions (first 20 for quick view)
        sample_predictions = df.head(20)[
            ['koi_period', 'koi_depth', 'koi_prad', 'koi_steff', 'probability', 'classification']
        ].to_dict('records')
        
        return jsonify({
            'success': True,
            'details': {
                'summary': {
                    'total': len(df),
                    'planets': int((df['prediction'] == 1).sum()),
                    'false_positives': int((df['prediction'] == 0).sum()),
                    'avg_planet_probability': float(planet_predictions['probability'].mean()) if len(planet_predictions) > 0 else 0,
                    'avg_fp_probability': float(fp_predictions['probability'].mean()) if len(fp_predictions) > 0 else 0,
                },
                'confidence_distribution': {str(k): int(v) for k, v in confidence_distribution.items()},
                'top_planets': top_planets,
                'top_false_positives': top_fps,
                'sample_predictions': sample_predictions,
            }
        }), 200
        
    except Exception as e:
        print(f"[DETAILS ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Failed to get details: {str(e)}'
        }), 500

