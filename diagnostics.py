"""
Diagnostic Script for Exoplanet Detection Pipeline
Checks for data leakage, overfitting, and model validity
"""

import numpy as np
import pandas as pd
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


class ModelDiagnostics:
    def __init__(self):
        print("="*80)
        print("EXOPLANET MODEL DIAGNOSTICS")
        print("="*80)
        
        # Load models and artifacts
        try:
            self.xgb_model = joblib.load(config.XGBOOST_MODEL_PATH)
            self.rf_model = joblib.load(config.RF_MODEL_PATH)
            self.scaler = joblib.load(config.SCALER_PATH)
            
            with open(config.FEATURE_NAMES_PATH, 'r') as f:
                self.feature_names = json.load(f)
            
            print(f"✓ Loaded models and {len(self.feature_names)} features")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("Run 'python src/train_models.py' first!")
            sys.exit(1)
        
        self.issues_found = []
        self.warnings_found = []
    
    def check_feature_leakage(self):
        """Check for features that might leak the target variable"""
        print("\n" + "="*80)
        print("1. FEATURE LEAKAGE CHECK")
        print("="*80)
        
        # List of suspicious keywords
        SUSPICIOUS_KEYWORDS = [
            'score', 'pdisposition', 'disposition', 'pdisp',
            'probability', 'class', 'label', 'confirmed', 'candidate'
        ]
        
        suspicious_features = []
        for feature in self.feature_names:
            for keyword in SUSPICIOUS_KEYWORDS:
                if keyword in feature.lower():
                    suspicious_features.append((feature, keyword))
        
        if suspicious_features:
            print("⚠️  SUSPICIOUS FEATURES FOUND (potential data leakage):")
            for feat, keyword in suspicious_features:
                print(f"   - '{feat}' (contains '{keyword}')")
            self.issues_found.append("Potential data leakage in features")
        else:
            print("✓ No obviously suspicious feature names detected")
        
        # Check against known Kepler leakage features
        KNOWN_LEAKAGE = ['koi_pdisposition', 'koi_score', 'koi_disposition']
        leakage_found = [f for f in self.feature_names if f in KNOWN_LEAKAGE]
        
        if leakage_found:
            print(f"\n❌ CRITICAL: Known leakage features found: {leakage_found}")
            print("   These features directly encode the target variable!")
            self.issues_found.append(f"CRITICAL LEAKAGE: {leakage_found}")
        else:
            print("✓ No known Kepler leakage features found")
    
    def check_feature_importance(self):
        """Analyze feature importance for suspicious patterns"""
        print("\n" + "="*80)
        print("2. FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        importance = self.xgb_model.feature_importances_
        
        # Create importance dataframe
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        # Check for dominance
        top1_pct = importance.max() / importance.sum() * 100
        top3_pct = importance[np.argsort(importance)[-3:]].sum() / importance.sum() * 100
        
        print(f"\nImportance Distribution:")
        print(f"  Top 1 feature: {top1_pct:.1f}% of total importance")
        print(f"  Top 3 features: {top3_pct:.1f}% of total importance")
        
        if top1_pct > 50:
            print(f"❌ CRITICAL: One feature dominates with {top1_pct:.1f}% importance")
            print("   This strongly suggests data leakage!")
            self.issues_found.append(f"Feature dominance: {feature_importance.iloc[0]['feature']} ({top1_pct:.1f}%)")
        elif top1_pct > 30:
            print(f"⚠️  WARNING: Top feature has {top1_pct:.1f}% importance")
            self.warnings_found.append(f"High feature dominance: {top1_pct:.1f}%")
        else:
            print("✓ Feature importance is well distributed")
        
        # Save plot
        try:
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(20)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title('Top 20 Feature Importances (XGBoost)')
            plt.tight_layout()
            plt.savefig('diagnostic_feature_importance.png', dpi=150)
            print("\n✓ Saved feature importance plot: diagnostic_feature_importance.png")
            plt.close()
        except Exception as e:
            print(f"⚠️  Could not save plot: {e}")
    
    def load_data_for_testing(self):
        """Load data for further diagnostics"""
        print("\n" + "="*80)
        print("3. LOADING DATA FOR VALIDATION")
        print("="*80)
        
        try:
            from src.data_pipeline import DataPipeline
            
            pipeline = DataPipeline()
            df = pipeline.load_data()
            X, y, df_clean = pipeline.prepare_dataset(df)
            X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, X_train, X_val, X_test = pipeline.split_and_scale(X, y)
            
            print(f"✓ Data loaded successfully")
            print(f"  Train: {len(X_train)} samples")
            print(f"  Val: {len(X_val)} samples") 
            print(f"  Test: {len(X_test)} samples")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
        
        except Exception as e:
            print(f"⚠️  Could not load data: {e}")
            print("   Skipping data-dependent checks")
            return None, None, None, None, None, None
    
    def check_cross_validation(self, X, y):
        """Perform cross-validation to check for overfitting"""
        if X is None or y is None:
            print("\n⚠️  Skipping cross-validation (no data)")
            return
        
        print("\n" + "="*80)
        print("4. CROSS-VALIDATION ANALYSIS")
        print("="*80)
        
        print("Running 5-fold cross-validation (this may take a minute)...")
        
        # Create a fresh model without early stopping for CV
        from xgboost import XGBClassifier
        temp_model = XGBClassifier(
            max_depth=self.xgb_model.get_params()['max_depth'],
            learning_rate=self.xgb_model.get_params()['learning_rate'],
            n_estimators=self.xgb_model.get_params()['n_estimators'],
            random_state=42,
            n_jobs=-1
        )
        
        cv_scores = cross_val_score(
            temp_model, X, y,
            cv=5, scoring='roc_auc', n_jobs=-1
        )
        
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        print(f"\nCross-Validation Results:")
        print(f"  Individual fold scores: {cv_scores}")
        print(f"  Mean AUC: {mean_score:.4f}")
        print(f"  Std Dev: {std_score:.4f}")
        print(f"  Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
        
        # Interpretation
        if mean_score > 0.98 and std_score < 0.01:
            print("\n❌ CRITICAL: Suspiciously perfect scores with minimal variance")
            print("   This is a strong indicator of data leakage!")
            self.issues_found.append(f"CV too perfect: {mean_score:.4f} ± {std_score:.4f}")
        elif mean_score > 0.95:
            print("\n⚠️  WARNING: Very high CV scores - double-check for leakage")
            self.warnings_found.append(f"Very high CV: {mean_score:.4f}")
        else:
            print("\n✓ Cross-validation scores look reasonable")
    
    def check_confusion_matrix(self, X_test, y_test):
        """Analyze confusion matrix for perfect predictions"""
        if X_test is None or y_test is None:
            print("\n⚠️  Skipping confusion matrix (no data)")
            return
        
        print("\n" + "="*80)
        print("5. CONFUSION MATRIX ANALYSIS")
        print("="*80)
        
        # Get predictions
        y_pred = self.xgb_model.predict(X_test)
        y_proba = self.xgb_model.predict_proba(X_test)[:, 1]
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                 FP    Planet")
        print(f"Actual FP      {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"Actual Planet  {cm[1,0]:5d}  {cm[1,1]:5d}")
        
        # Calculate metrics
        total_errors = cm[0,1] + cm[1,0]
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        
        print(f"\nTotal errors: {total_errors} out of {len(y_test)} ({total_errors/len(y_test)*100:.2f}%)")
        print(f"Accuracy: {accuracy:.4f}")
        
        if total_errors == 0:
            print("\n❌ CRITICAL: ZERO prediction errors on test set!")
            print("   This is virtually impossible without data leakage")
            self.issues_found.append("Perfect test accuracy (0 errors)")
        elif total_errors < 5:
            print("\n⚠️  WARNING: Very few errors - check for data leakage")
            self.warnings_found.append(f"Only {total_errors} errors")
        else:
            print("\n✓ Error rate looks reasonable")
        
        # Save confusion matrix plot
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['False Positive', 'Planet'],
                       yticklabels=['False Positive', 'Planet'])
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title('Confusion Matrix - XGBoost')
            plt.tight_layout()
            plt.savefig('diagnostic_confusion_matrix.png', dpi=150)
            print("✓ Saved confusion matrix: diagnostic_confusion_matrix.png")
            plt.close()
        except Exception as e:
            print(f"⚠️  Could not save plot: {e}")
    
    def check_baseline_comparison(self, X_train, X_test, y_train, y_test):
        """Compare with dummy baseline classifier"""
        if X_train is None:
            print("\n⚠️  Skipping baseline comparison (no data)")
            return
        
        print("\n" + "="*80)
        print("6. BASELINE COMPARISON")
        print("="*80)
        
        # Train dummy classifier
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(X_train, y_train)
        dummy_acc = dummy.score(X_test, y_test)
        
        # Your model
        model_acc = self.xgb_model.score(X_test, y_test)
        
        improvement = model_acc - dummy_acc
        
        print(f"Dummy classifier (always predict most frequent): {dummy_acc:.4f}")
        print(f"Your XGBoost model:                              {model_acc:.4f}")
        print(f"Improvement over baseline:                       {improvement:.4f}")
        
        if model_acc >= 0.99:
            print("\n⚠️  Model is nearly perfect - suspicious for real-world data")
            self.warnings_found.append(f"Near-perfect accuracy: {model_acc:.4f}")
        elif improvement < 0.1:
            print("\n⚠️  Model barely beats baseline - features may not be informative")
            self.warnings_found.append("Weak improvement over baseline")
        else:
            print("\n✓ Model shows good improvement over baseline")
    
    def check_learning_curves(self, X_train, y_train):
        """Plot learning curves to check for overfitting"""
        if X_train is None:
            print("\n⚠️  Skipping learning curves (no data)")
            return
        
        print("\n" + "="*80)
        print("7. LEARNING CURVES")
        print("="*80)
        print("Computing learning curves (this may take several minutes)...")
        
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                self.xgb_model, X_train, y_train,
                cv=3, scoring='roc_auc',
                train_sizes=np.linspace(0.1, 1.0, 5),
                n_jobs=-1
            )
            
            train_mean = train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            val_mean = val_scores.mean(axis=1)
            val_std = val_scores.std(axis=1)
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_mean, 'o-', label='Training score', linewidth=2)
            plt.plot(train_sizes, val_mean, 'o-', label='Validation score', linewidth=2)
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
            plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
            plt.xlabel('Training Set Size')
            plt.ylabel('AUC Score')
            plt.title('Learning Curves - XGBoost')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('diagnostic_learning_curves.png', dpi=150)
            print("✓ Saved learning curves: diagnostic_learning_curves.png")
            plt.close()
            
            # Analysis
            gap = train_mean[-1] - val_mean[-1]
            print(f"\nFinal scores (largest training size):")
            print(f"  Training:   {train_mean[-1]:.4f} ± {train_std[-1]:.4f}")
            print(f"  Validation: {val_mean[-1]:.4f} ± {val_std[-1]:.4f}")
            print(f"  Gap:        {gap:.4f}")
            
            if gap > 0.1:
                print("\n⚠️  WARNING: Large gap between training and validation scores")
                print("   This indicates overfitting")
                self.warnings_found.append(f"Overfitting: train-val gap = {gap:.4f}")
            elif train_mean[-1] > 0.99 and val_mean[-1] > 0.99:
                print("\n❌ Both training and validation are near-perfect")
                print("   This strongly suggests data leakage!")
                self.issues_found.append("Both train and val near perfect")
            else:
                print("\n✓ Learning curves look reasonable")
        
        except Exception as e:
            print(f"⚠️  Could not compute learning curves: {e}")
    
    def check_permutation_importance(self, X_test, y_test):
        """Check permutation importance for leakage detection"""
        if X_test is None:
            print("\n⚠️  Skipping permutation importance (no data)")
            return
        
        print("\n" + "="*80)
        print("8. PERMUTATION IMPORTANCE")
        print("="*80)
        print("Computing permutation importance (this may take a minute)...")
        
        try:
            result = permutation_importance(
                self.xgb_model, X_test, y_test,
                n_repeats=10, random_state=42,
                scoring='roc_auc', n_jobs=-1
            )
            
            # Get top features
            sorted_idx = result.importances_mean.argsort()[::-1]
            
            print("\nTop 10 features by permutation importance:")
            print("(How much AUC drops when feature is shuffled)")
            print()
            
            critical_features = []
            for i in sorted_idx[:10]:
                mean_drop = result.importances_mean[i]
                std_drop = result.importances_std[i]
                print(f"  {self.feature_names[i]:30s}: {mean_drop:.4f} ± {std_drop:.4f}")
                
                if mean_drop > 0.15:  # Dropping this feature hurts a lot
                    critical_features.append((self.feature_names[i], mean_drop))
            
            if critical_features:
                print(f"\n⚠️  Features that cause large performance drops when removed:")
                for feat, drop in critical_features:
                    print(f"   - {feat}: {drop:.4f}")
                print("   These features might be too powerful (potential leakage)")
                self.warnings_found.append(f"{len(critical_features)} critical features")
            else:
                print("\n✓ No single feature is critically important")
        
        except Exception as e:
            print(f"⚠️  Could not compute permutation importance: {e}")
    
    def generate_report(self):
        """Generate final diagnostic report"""
        print("\n" + "="*80)
        print("DIAGNOSTIC SUMMARY")
        print("="*80)
        
        if self.issues_found:
            print(f"\n❌ CRITICAL ISSUES FOUND ({len(self.issues_found)}):")
            for i, issue in enumerate(self.issues_found, 1):
                print(f"   {i}. {issue}")
            print("\n   ACTION REQUIRED: These issues likely indicate data leakage!")
            print("   - Review your feature list carefully")
            print("   - Check for koi_pdisposition, koi_score, or similar")
            print("   - Retrain after removing leakage features")
        else:
            print("\n✓ No critical issues detected")
        
        if self.warnings_found:
            print(f"\n⚠️  WARNINGS ({len(self.warnings_found)}):")
            for i, warning in enumerate(self.warnings_found, 1):
                print(f"   {i}. {warning}")
            print("\n   These may indicate problems but could also be legitimate")
        else:
            print("✓ No warnings")
        
        if not self.issues_found and not self.warnings_found:
            print("\n🎉 All checks passed! Your model appears legitimate.")
        
        print("\n" + "="*80)
        print("Generated diagnostic plots:")
        print("  - diagnostic_feature_importance.png")
        print("  - diagnostic_confusion_matrix.png")
        print("  - diagnostic_learning_curves.png")
        print("="*80)
    
    def run_all_diagnostics(self):
        """Run all diagnostic checks"""
        # Checks that don't need data
        self.check_feature_leakage()
        self.check_feature_importance()
        
        # Load data for remaining checks
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data_for_testing()
        
        if X_train is not None:
            # Combine for cross-validation
            X_full = np.vstack([X_train, X_val, X_test])
            y_full = np.concatenate([y_train, y_val, y_test])
            
            self.check_cross_validation(X_full, y_full)
            self.check_confusion_matrix(X_test, y_test)
            self.check_baseline_comparison(X_train, X_test, y_train, y_test)
            self.check_learning_curves(X_train, y_train)
            self.check_permutation_importance(X_test, y_test)
        
        # Final report
        self.generate_report()


if __name__ == "__main__":
    diagnostics = ModelDiagnostics()
    diagnostics.run_all_diagnostics()

