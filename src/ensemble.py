import numpy as np
import joblib
from tensorflow import keras
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

sys.path.append(PROJECT_ROOT)

import config


class EnsemblePredictor:
    def __init__(self, weights=None):
        print("Loading ensemble models...")
        
        self.xgb_model = joblib.load(config.XGBOOST_MODEL_PATH)
        self.rf_model = joblib.load(config.RF_MODEL_PATH)
        self.nn_model = keras.models.load_model(config.NN_MODEL_PATH)
        self.scaler = joblib.load(config.SCALER_PATH)
        
        # Load feature names
        with open(config.FEATURE_NAMES_PATH, 'r') as f:
            self.feature_names = json.load(f)
        
        # Set weights (default: equal weighting)
        if weights is None:
            self.weights = np.array([0.333, 0.333, 0.334])
        else:
            self.weights = np.array(weights)
            # Normalize to sum to 1
            self.weights = self.weights / self.weights.sum()
        
        print(f"✓ Loaded 3 models with {len(self.feature_names)} features")
        print(f"  Ensemble weights: XGB={self.weights[0]:.3f}, RF={self.weights[1]:.3f}, NN={self.weights[2]:.3f}")
    
    def predict_proba(self, X, X_scaled=None):
        # Get predictions from each model
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        
        # Scale for neural network if needed
        if X_scaled is None:
            X_scaled = self.scaler.transform(X)
        nn_proba = self.nn_model.predict(X_scaled, verbose=0).flatten()
        
        # Weighted average
        ensemble_proba = (
            self.weights[0] * xgb_proba +
            self.weights[1] * rf_proba +
            self.weights[2] * nn_proba
        )
        
        return ensemble_proba
    
    def predict(self, X, X_scaled=None, threshold=0.5):
        proba = self.predict_proba(X, X_scaled)
        return (proba >= threshold).astype(int)
    
    def evaluate(self, X, X_scaled, y_true, threshold=0.5):
        # Get predictions
        y_proba = self.predict_proba(X, X_scaled)
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        auc = roc_auc_score(y_true, y_proba)
        accuracy = (y_pred == y_true).mean()
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Print results
        print("\n" + "="*70)
        print("ENSEMBLE MODEL Evaluation")
        print("="*70)
        print(f"Weights: XGB={self.weights[0]:.3f}, RF={self.weights[1]:.3f}, NN={self.weights[2]:.3f}")
        print(f"Threshold: {threshold}")
        print()
        print(classification_report(y_true, y_pred, target_names=['False Positive', 'Planet']))
        print(f"AUC: {auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                 FP    Planet")
        print(f"Actual FP      {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"Actual Planet  {cm[1,0]:5d}  {cm[1,1]:5d}")
        
        return {
            'auc': auc,
            'accuracy': accuracy,
            'threshold': threshold,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'confusion_matrix': cm
        }
    
    def optimize_threshold(self, X, X_scaled, y_true, metric='f1'):
        from sklearn.metrics import f1_score
        
        y_proba = self.predict_proba(X, X_scaled)
        
        # Try thresholds from 0.3 to 0.7
        thresholds = np.linspace(0.3, 0.7, 41)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'accuracy':
                score = (y_pred == y_true).mean()
            elif metric == 'youden':
                # Youden's J statistic = sensitivity + specificity - 1
                cm = confusion_matrix(y_true, y_pred)
                sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
                specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
                score = sensitivity + specificity - 1
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores.append(score)
        
        # Find best threshold
        best_idx = np.argmax(scores)
        best_threshold = thresholds[best_idx]
        best_score = scores[best_idx]
        
        print(f"\nThreshold Optimization (metric={metric}):")
        print(f"  Best threshold: {best_threshold:.3f}")
        print(f"  Best {metric}: {best_score:.4f}")
        print(f"  Default (0.5) {metric}: {scores[20]:.4f}")
        print(f"  Improvement: {best_score - scores[20]:.4f}")
        
        return best_threshold
    
    def get_individual_predictions(self, X, X_scaled=None):
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        
        if X_scaled is None:
            X_scaled = self.scaler.transform(X)
        nn_proba = self.nn_model.predict(X_scaled, verbose=0).flatten()
        
        return {
            'xgboost': xgb_proba,
            'random_forest': rf_proba,
            'neural_network': nn_proba
        }


def optimize_ensemble_weights(X_train, X_train_scaled, y_train, X_val, X_val_scaled, y_val):
    from scipy.optimize import minimize
    
    print("\n" + "="*70)
    print("OPTIMIZING ENSEMBLE WEIGHTS")
    print("="*70)
    
    # Load models
    xgb_model = joblib.load(config.XGBOOST_MODEL_PATH)
    rf_model = joblib.load(config.RF_MODEL_PATH)
    nn_model = keras.models.load_model(config.NN_MODEL_PATH)
    
    # Get individual predictions on validation set
    xgb_proba = xgb_model.predict_proba(X_val)[:, 1]
    rf_proba = rf_model.predict_proba(X_val)[:, 1]
    nn_proba = nn_model.predict(X_val_scaled, verbose=0).flatten()
    
    # Objective: maximize AUC on validation set
    def objective(weights):
        # Ensure weights sum to 1
        weights = weights / weights.sum()
        ensemble_proba = weights[0]*xgb_proba + weights[1]*rf_proba + weights[2]*nn_proba
        auc = roc_auc_score(y_val, ensemble_proba)
        return -auc  # Negative because we minimize
    
    # Initial guess: equal weights
    initial_weights = np.array([0.333, 0.333, 0.334])
    
    # Constraints: weights must be non-negative and sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    bounds = [(0, 1), (0, 1), (0, 1)]
    
    # Optimize
    result = minimize(objective, initial_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x / result.x.sum()
    
    # Evaluate with optimal weights
    ensemble_proba = optimal_weights[0]*xgb_proba + optimal_weights[1]*rf_proba + optimal_weights[2]*nn_proba
    optimal_auc = roc_auc_score(y_val, ensemble_proba)
    
    # Compare with equal weights
    equal_proba = (xgb_proba + rf_proba + nn_proba) / 3
    equal_auc = roc_auc_score(y_val, equal_proba)
    
    print(f"\nEqual weights [0.333, 0.333, 0.334]:")
    print(f"  Validation AUC: {equal_auc:.4f}")
    print(f"\nOptimized weights [{optimal_weights[0]:.3f}, {optimal_weights[1]:.3f}, {optimal_weights[2]:.3f}]:")
    print(f"  Validation AUC: {optimal_auc:.4f}")
    print(f"  Improvement: {optimal_auc - equal_auc:.4f}")
    
    return optimal_weights


if __name__ == "__main__":
    """
    Test ensemble model
    """
    from data_pipeline import DataPipeline
    
    print("="*70)
    print("TESTING ENSEMBLE MODEL")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    pipeline = DataPipeline()
    df = pipeline.load_data()
    X, y, df_clean = pipeline.prepare_dataset(df)
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, X_train, X_val, X_test = pipeline.split_and_scale(X, y)
    
    # Test with equal weights
    print("\n" + "="*70)
    print("1. EQUAL WEIGHTS ENSEMBLE")
    print("="*70)
    ensemble_equal = EnsemblePredictor(weights=None)
    results_equal = ensemble_equal.evaluate(X_test, X_test_scaled, y_test)
    
    # Optimize threshold
    print("\n" + "="*70)
    print("2. THRESHOLD OPTIMIZATION")
    print("="*70)
    best_threshold = ensemble_equal.optimize_threshold(X_val, X_val_scaled, y_val, metric='f1')
    
    # Evaluate with optimized threshold
    print("\n" + "="*70)
    print("3. ENSEMBLE WITH OPTIMIZED THRESHOLD")
    print("="*70)
    results_optimized = ensemble_equal.evaluate(X_test, X_test_scaled, y_test, threshold=best_threshold)
    
    # Optimize weights
    print("\n" + "="*70)
    print("4. WEIGHT OPTIMIZATION")
    print("="*70)
    optimal_weights = optimize_ensemble_weights(
        X_train, X_train_scaled, y_train,
        X_val, X_val_scaled, y_val
    )
    
    # Test with optimized weights
    print("\n" + "="*70)
    print("5. OPTIMIZED WEIGHTS + THRESHOLD")
    print("="*70)
    ensemble_optimized = EnsemblePredictor(weights=optimal_weights)
    results_final = ensemble_optimized.evaluate(X_test, X_test_scaled, y_test, threshold=best_threshold)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Equal weights, threshold=0.5:     Accuracy={results_equal['accuracy']:.4f}, AUC={results_equal['auc']:.4f}")
    print(f"Equal weights, optimized threshold: Accuracy={results_optimized['accuracy']:.4f}, AUC={results_optimized['auc']:.4f}")
    print(f"Optimized weights + threshold:     Accuracy={results_final['accuracy']:.4f}, AUC={results_final['auc']:.4f}")

