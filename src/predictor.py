import numpy as np
import joblib
import json
from tensorflow import keras
import config

class ExoplanetPredictor:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = self.load_model(model_type)
        self.scaler = joblib.load(config.SCALER_PATH)
        with open(config.FEATURE_NAMES_PATH, 'r') as f:
            self.feature_names = json.load(f)
    
    def load_model(self, model_type):
        if model_type == 'xgboost':
            return joblib.load(config.XGBOOST_MODEL_PATH)
        elif model_type == 'random_forest':
            return joblib.load(config.RF_MODEL_PATH)
        elif model_type == 'neural_network':
            return keras.models.load_model(config.NN_MODEL_PATH)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def engineer_features(self, input_dict):
        engineered = {}
        
        # ====================================================================
        # CATEGORY 1: ERROR-BASED FEATURES (5 features)
        # ====================================================================
        
        # 1. Period Relative Error
        if 'koi_period' in input_dict and 'koi_period_err1' in input_dict:
            engineered['period_rel_err'] = abs(input_dict['koi_period_err1']) / (input_dict['koi_period'] + 1e-10)
        
        # 2. Depth Relative Error
        if 'koi_depth' in input_dict and 'koi_depth_err1' in input_dict:
            engineered['depth_rel_err'] = abs(input_dict['koi_depth_err1']) / (input_dict['koi_depth'] + 1e-10)
        
        # 3. Duration Relative Error
        if 'koi_duration' in input_dict and 'koi_duration_err1' in input_dict:
            engineered['duration_rel_err'] = abs(input_dict['koi_duration_err1']) / (input_dict['koi_duration'] + 1e-10)
        
        # 4. Depth Error Asymmetry
        if 'koi_depth_err1' in input_dict and 'koi_depth_err2' in input_dict:
            err1 = abs(input_dict['koi_depth_err1'])
            err2 = abs(input_dict['koi_depth_err2'])
            engineered['depth_err_asymmetry'] = abs(err1 - err2) / (err1 + err2 + 1e-10)
        
        # 5. Depth to Uncertainty
        if 'koi_depth' in input_dict and 'koi_depth_err1' in input_dict:
            engineered['depth_to_uncertainty'] = input_dict['koi_depth'] / (abs(input_dict['koi_depth_err1']) + 1e-10)
        
        # ====================================================================
        # CATEGORY 2: ASTROPHYSICAL FEATURES (7 features)
        # ====================================================================
        
        # 6. Radius Ratio
        if 'koi_prad' in input_dict and 'koi_srad' in input_dict:
            engineered['radius_ratio'] = input_dict['koi_prad'] / (input_dict['koi_srad'] * 109.1)
            
            # 7. Depth Ratio Mismatch
            if 'koi_depth' in input_dict:
                expected_depth = (engineered['radius_ratio'] ** 2) * 1e6
                engineered['depth_ratio_mismatch'] = abs(input_dict['koi_depth'] - expected_depth) / (input_dict['koi_depth'] + 1e-10)
        
        # 8. Scaled Semi-Major Axis
        if 'koi_sma' in input_dict and 'koi_srad' in input_dict:
            engineered['scaled_sma'] = (input_dict['koi_sma'] * 215) / input_dict['koi_srad']
        
        # 9. Transit Probability
        if 'koi_srad' in input_dict and 'koi_sma' in input_dict:
            engineered['transit_probability'] = input_dict['koi_srad'] / (input_dict['koi_sma'] * 215 + 1e-10)
        
        # 10. Insolation Ratio Earth
        if 'koi_insol' in input_dict:
            engineered['insol_ratio_earth'] = input_dict['koi_insol']
            
            # 11. In Habitable Zone
            engineered['in_habitable_zone'] = 1 if (0.36 <= input_dict['koi_insol'] <= 1.77) else 0
        
        # 12. Teq Mismatch
        if all(k in input_dict for k in ['koi_teq', 'koi_steff', 'koi_sma', 'koi_srad']):
            expected_teq = input_dict['koi_steff'] * np.sqrt(input_dict['koi_srad'] * 0.00465 / (2 * input_dict['koi_sma']))
            engineered['teq_mismatch'] = abs(input_dict['koi_teq'] - expected_teq) / (input_dict['koi_teq'] + 1e-10)
        
        # ====================================================================
        # CATEGORY 3: QUALITY DIAGNOSTICS (2 features)
        # ====================================================================
        
        # 13. Effective SNR
        if 'koi_depth' in input_dict and 'koi_depth_err1' in input_dict:
            engineered['effective_snr'] = input_dict['koi_depth'] / (abs(input_dict['koi_depth_err1']) + 1e-10)
        
        # 14. Signal Strength
        if 'koi_depth' in input_dict and 'koi_model_snr' in input_dict:
            engineered['signal_strength'] = input_dict['koi_depth'] * input_dict['koi_model_snr']
        
        # ====================================================================
        # CATEGORY 4: FLAG AGGREGATIONS (4 features)
        # ====================================================================
        
        # 15. Centroid Offset Total
        if 'koi_dicco_mra' in input_dict and 'koi_dicco_mdec' in input_dict:
            engineered['centroid_offset_total'] = np.sqrt(input_dict['koi_dicco_mra']**2 + input_dict['koi_dicco_mdec']**2)
        
        # 16. Number of Red Flags
        flag_cols = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
        flags_present = [input_dict.get(col, 0) for col in flag_cols]
        engineered['num_red_flags'] = sum(flags_present)
        
        # 17. Any Red Flag
        engineered['any_red_flag'] = 1 if engineered['num_red_flags'] > 0 else 0
        
        # 18. Warning Category
        engineered['warning_category'] = min(engineered['num_red_flags'], 2)
        
        return engineered
    
    def build_feature_vector(self, input_dict):
        engineered = self.engineer_features(input_dict)
        
        combined = {**input_dict, **engineered}
        
        feature_vector = []
        missing_features = []
        
        for fname in self.feature_names:
            if fname in combined:
                feature_vector.append(combined[fname])
            else:
                feature_vector.append(0.0)
                missing_features.append(fname)
        
        if len(missing_features) > 0:
            print(f"Warning: {len(missing_features)} features missing from input")
            print(f"Missing: {missing_features[:5]}...")
        
        return np.array([feature_vector]) 
    
    def predict(self, input_dict):
        X = self.build_feature_vector(input_dict)
        
        if self.model_type == 'neural_network':
            X = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X)[0, 1]
        else:
            y_proba = float(self.model.predict(X, verbose=0)[0, 0])
        
        y_pred = 1 if y_proba > 0.5 else 0
        
        return {
            'prediction': 'EXOPLANET' if y_pred == 1 else 'FALSE POSITIVE',
            'confidence': float(y_proba if y_pred == 1 else 1 - y_proba),
            'probability_planet': float(y_proba),
            'probability_false_positive': float(1 - y_proba)
        }
    
    def get_required_features(self):
        required = [
            'koi_period',
            'koi_depth',
            'koi_duration',
            'koi_model_snr',
        ]
        
        recommended = [
            'koi_depth_err1',
            'koi_period_err1',
            'koi_duration_err1',
            'koi_dicco_msky',
            'koi_fpflag_ss',
            'koi_fpflag_co',
            'koi_fpflag_nt',
            'koi_fpflag_ec',
        ]
        
        optional = [
            'koi_time0bk',
            'koi_period_err2',
            'koi_depth_err2',
            'koi_duration_err2',
            'koi_dicco_msky_err',
            'koi_dicco_mra',
            'koi_dicco_mdec',
            'koi_prad',
            'koi_sma',
            'koi_impact',
            'koi_teq',
            'koi_insol',
            'koi_incl',
            'koi_steff',
            'koi_slogg',
            'koi_srad',
            'koi_smass',
        ]
        
        return {
            'required': required,
            'recommended': recommended,
            'optional': optional,
            'total_features': len(self.feature_names)
        }
    
    def validate_input(self, input_dict):
        required_info = self.get_required_features()
        
        missing_required = [f for f in required_info['required'] if f not in input_dict]
        missing_recommended = [f for f in required_info['recommended'] if f not in input_dict]
        
        warnings = []
        
        if 'koi_period' in input_dict:
            if input_dict['koi_period'] < 0.1 or input_dict['koi_period'] > 10000:
                warnings.append(f"Period {input_dict['koi_period']} days is unusual")
        
        if 'koi_depth' in input_dict:
            if input_dict['koi_depth'] < 0 or input_dict['koi_depth'] > 100000:
                warnings.append(f"Depth {input_dict['koi_depth']} ppm is unusual")
        
        if len(missing_recommended) > 3:
            warnings.append(f"Missing {len(missing_recommended)} recommended features - accuracy may be reduced")
        
        return {
            'valid': len(missing_required) == 0,
            'missing_required': missing_required,
            'missing_recommended': missing_recommended,
            'warnings': warnings
        }


if __name__ == "__main__":    
    # Example 1: Confirmed planet (Earth-like)
    earth_like = {
        'koi_period': 365.25,
        'koi_depth': 84,  # Earth-like transit depth
        'koi_duration': 13.0,
        'koi_model_snr': 25.0,
        'koi_depth_err1': 5.0,
        'koi_period_err1': 0.1,
        'koi_duration_err1': 0.5,
        'koi_dicco_msky': 0.02,
        'koi_fpflag_ss': 0,
        'koi_fpflag_co': 0,
        'koi_fpflag_nt': 0,
        'koi_fpflag_ec': 0,
        'koi_prad': 1.0,
        'koi_srad': 1.0,
        'koi_sma': 1.0,
        'koi_teq': 288,
        'koi_insol': 1.0,
        'koi_steff': 5778,
    }
    
    # Example 2: Likely false positive (eclipsing binary)
    false_positive = {
        'koi_period': 2.5,
        'koi_depth': 8000,  # Very deep
        'koi_duration': 5.0,
        'koi_model_snr': 15.0,
        'koi_depth_err1': 500.0,  # Large error
        'koi_period_err1': 0.5,
        'koi_duration_err1': 1.0,
        'koi_dicco_msky': 2.5,  # Large centroid offset
        'koi_fpflag_ss': 1,  # Secondary eclipse detected!
        'koi_fpflag_co': 1,  # Centroid offset flagged!
        'koi_fpflag_nt': 0,
        'koi_fpflag_ec': 0,
        'koi_prad': 15.0,  # Too large
        'koi_srad': 1.5,
        'koi_sma': 0.05,
        'koi_teq': 2000,
        'koi_insol': 500.0,
        'koi_steff': 6500,
    }
    
    # Load predictor
    print("Loading XGBoost predictor...")
    predictor = ExoplanetPredictor(model_type='xgboost')
    
    # Test Earth-like planet
    print("\n" + "="*70)
    print("TEST 1: Earth-like Planet")
    print("="*70)
    validation = predictor.validate_input(earth_like)
    print(f"Input valid: {validation['valid']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    
    result = predictor.predict(earth_like)
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Planet probability: {result['probability_planet']:.2%}")
    
    # Test false positive
    print("\n" + "="*70)
    print("TEST 2: Likely False Positive (Eclipsing Binary)")
    print("="*70)
    validation = predictor.validate_input(false_positive)
    print(f"Input valid: {validation['valid']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    
    result = predictor.predict(false_positive)
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"False positive probability: {result['probability_false_positive']:.2%}")
    
    # Show required features
    print("\n" + "="*70)
    print("REQUIRED FEATURES INFO")
    print("="*70)
    features_info = predictor.get_required_features()
    print(f"Required features: {len(features_info['required'])}")
    print(f"  {features_info['required']}")
    print(f"\nRecommended features: {len(features_info['recommended'])}")
    print(f"  {features_info['recommended'][:5]}...")
    print(f"\nTotal features used by model: {features_info['total_features']}")