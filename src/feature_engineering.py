import numpy as np
import pandas as pd

class FeatureEngineer:
    
    def __init__(self):
        self.engineered_features = []
    
    def engineer_all_features(self, df):
        print("\n" + "="*70)
        print("ADVANCED FEATURE ENGINEERING")
        print("="*70)
        
        df_eng = df.copy()
        
        df_eng = self.add_error_features(df_eng)
        
        df_eng = self.add_astrophysical_features(df_eng)
        
        df_eng = self.add_quality_diagnostics(df_eng)
        
        df_eng = self.add_basic_engineered_features(df_eng)
        
        print(f"\nTotal engineered features: {len(self.engineered_features)}")
        
        return df_eng
    
    # ========================================================================
    # CATEGORY 1: ERROR-BASED FEATURES (Uncertainty Analysis)
    # ========================================================================
    
    def add_error_features(self, df):
        print("\n1. Error-Based Features:")
        
        # Feature 1: Period Relative Error
        if 'koi_period' in df.columns and 'koi_period_err1' in df.columns:
            df['period_rel_err'] = np.abs(df['koi_period_err1']) / df['koi_period']
            self.engineered_features.append('period_rel_err')
            print("  ✓ period_rel_err = |period_err| / period")
        
        # Feature 2: Depth Relative Error
        if 'koi_depth' in df.columns and 'koi_depth_err1' in df.columns:
            df['depth_rel_err'] = np.abs(df['koi_depth_err1']) / df['koi_depth']
            self.engineered_features.append('depth_rel_err')
            print("  ✓ depth_rel_err = |depth_err| / depth")
        
        # Feature 3: Duration Relative Error
        if 'koi_duration' in df.columns and 'koi_duration_err1' in df.columns:
            df['duration_rel_err'] = np.abs(df['koi_duration_err1']) / df['koi_duration']
            self.engineered_features.append('duration_rel_err')
            print("  ✓ duration_rel_err = |duration_err| / duration")
        
        # Feature 4: Error Asymmetry (for skewed distributions)
        if 'koi_depth_err1' in df.columns and 'koi_depth_err2' in df.columns:
            err1 = np.abs(df['koi_depth_err1'])
            err2 = np.abs(df['koi_depth_err2'])
            df['depth_err_asymmetry'] = np.abs(err1 - err2) / (err1 + err2 + 1e-10)
            self.engineered_features.append('depth_err_asymmetry')
            print("  ✓ depth_err_asymmetry = |err1 - err2| / (err1 + err2)")
        
        # Feature 5: Signal-to-Uncertainty Ratio
        if 'koi_depth' in df.columns and 'koi_depth_err1' in df.columns:
            df['depth_to_uncertainty'] = df['koi_depth'] / (np.abs(df['koi_depth_err1']) + 1e-10)
            self.engineered_features.append('depth_to_uncertainty')
            print("  ✓ depth_to_uncertainty = depth / depth_err")
        
        return df
    
    # ========================================================================
    # CATEGORY 2: ASTROPHYSICAL FEATURES (Physics Validation)
    # ========================================================================
    
    def add_astrophysical_features(self, df):
        print("\n2. Astrophysical Derived Features:")
        
        # Feature 6: Planet-Star Radius Ratio
        if 'koi_prad' in df.columns and 'koi_srad' in df.columns:
            # Convert stellar radius to Earth radii (1 R_sun = 109.1 R_earth)
            df['radius_ratio'] = df['koi_prad'] / (df['koi_srad'] * 109.1)
            self.engineered_features.append('radius_ratio')
            print(" radius_ratio = Rp / Rs")
            
            # Expected depth from radius ratio
            if 'koi_depth' in df.columns:
                expected_depth = (df['radius_ratio'] ** 2) * 1e6  # Convert to ppm
                df['depth_ratio_mismatch'] = np.abs(df['koi_depth'] - expected_depth) / (df['koi_depth'] + 1e-10)
                self.engineered_features.append('depth_ratio_mismatch')
                print("depth_ratio_mismatch = |observed_depth - expected_depth| / observed_depth")
        
        # Feature 7: Scaled Semi-Major Axis (a/Rs)
        if 'koi_sma' in df.columns and 'koi_srad' in df.columns:
            # Convert: sma is in AU, srad is in solar radii
            # 1 AU = 215 solar radii
            df['scaled_sma'] = (df['koi_sma'] * 215) / df['koi_srad']
            self.engineered_features.append('scaled_sma')
            print(" scaled_sma = a / Rs")
        
        # Feature 8: Transit Probability
        if 'koi_srad' in df.columns and 'koi_sma' in df.columns:
            # P_transit ≈ Rs / a
            df['transit_probability'] = df['koi_srad'] / (df['koi_sma'] * 215 + 1e-10)
            self.engineered_features.append('transit_probability')
            print("  ✓ transit_probability ≈ Rs / a")
        
        # Feature 9: Insolation Relative to Earth
        if 'koi_insol' in df.columns:
            df['insol_ratio_earth'] = df['koi_insol']  # Already relative to Earth
            df['in_habitable_zone'] = ((df['koi_insol'] >= 0.36) & (df['koi_insol'] <= 1.77)).astype(int)
            self.engineered_features.append('insol_ratio_earth')
            self.engineered_features.append('in_habitable_zone')
            print("  ✓ insol_ratio_earth = koi_insol")
            print("  ✓ in_habitable_zone = 1 if 0.36 ≤ insol ≤ 1.77")
        
        # Feature 10: Effective Temperature Validation
        if 'koi_teq' in df.columns and 'koi_steff' in df.columns and 'koi_sma' in df.columns:
            # Rough expected Teq = Tstar * sqrt(Rstar / 2a)
            expected_teq = df['koi_steff'] * np.sqrt(df['koi_srad'] * 0.00465 / (2 * df['koi_sma']))
            df['teq_mismatch'] = np.abs(df['koi_teq'] - expected_teq) / (df['koi_teq'] + 1e-10)
            self.engineered_features.append('teq_mismatch')
            print("teq_mismatch = |observed_Teq - expected_Teq| / observed_Teq")
        
        return df
    
    # ========================================================================
    # CATEGORY 3: QUALITY DIAGNOSTICS (Signal Quality)
    # ========================================================================
    
    def add_quality_diagnostics(self, df):

        print("\n3. Quality Diagnostic Features:")
        
        # Feature 11: Effective SNR (depth-based)
        if 'koi_depth' in df.columns and 'koi_depth_err1' in df.columns:
            df['effective_snr'] = df['koi_depth'] / (np.abs(df['koi_depth_err1']) + 1e-10)
            self.engineered_features.append('effective_snr')
            print("  ✓ effective_snr = depth / depth_err")
        
        # Feature 13: Signal Strength (composite)
        if 'koi_depth' in df.columns and 'koi_model_snr' in df.columns:
            df['signal_strength'] = df['koi_depth'] * df['koi_model_snr']
            self.engineered_features.append('signal_strength')
            print("  ✓ signal_strength = depth × SNR")
        
        
        return df
    
    # ========================================================================
    # CATEGORY 4: FLAG AGGREGATIONS (Multiple Warning Signals)
    # ========================================================================
    
    def add_basic_engineered_features(self, df):
        
        # NEW: 5 Additional Physics-Based Features
        print("\n Additional Physics Features:")
        
        # Feature 16: Planet Density Proxy (mass/radius relationship) ✓ TESS
        if 'koi_prad' in df.columns:
            # Assuming typical planet mass-radius relationship: M ∝ R^2.06
            # Density proxy: higher values = denser objects (more planet-like)
            df['planet_density_proxy'] = df['koi_prad'] ** (-1.06)
            self.engineered_features.append('planet_density_proxy')
            print("  ✓ planet_density_proxy = R^(-1.06) (mass-radius proxy)")
        
        # Feature 17: Stellar Insolation Factor (combining stellar and orbital properties) ✓ TESS
        if all(col in df.columns for col in ['koi_steff', 'koi_srad', 'koi_period']):
            # Estimated insolation based on stellar properties
            # Kepler's 3rd law approximation for semi-major axis
            a_approx = (df['koi_period'] / 365.25) ** (2/3)  # AU, assuming solar mass
            df['stellar_insolation_factor'] = (df['koi_steff'] / 5778) ** 4 * (df['koi_srad'] ** 2) / (a_approx ** 2 + 1e-10)
            self.engineered_features.append('stellar_insolation_factor')
            print("  ✓ stellar_insolation_factor = (Tstar/Tsun)^4 * (Rs^2/a^2)")
        
        # Feature 18: Transit Signal Strength (combining depth and SNR) ✓ TESS
        if 'koi_depth' in df.columns and 'koi_model_snr' in df.columns:
            # Normalized signal strength
            df['normalized_signal_strength'] = np.log10(df['koi_depth'] + 1) * np.log10(df['koi_model_snr'] + 1)
            self.engineered_features.append('normalized_signal_strength')
            print("  ✓ normalized_signal_strength = log(depth) * log(SNR)")
        
        # Feature 19: Orbital Period Harmonic Check ✓ TESS
        if 'koi_period' in df.columns:
            # Check if period is near common harmonics (false positives often at 1/2, 1/3 day periods)
            # Distance from nearest integer day (stellar rotation aliases)
            nearest_day = np.round(df['koi_period'])
            df['period_day_offset'] = np.abs(df['koi_period'] - nearest_day) / (df['koi_period'] + 1e-10)
            self.engineered_features.append('period_day_offset')
            print("  ✓ period_day_offset = |period - nearest_day| / period")
                
                
        return df
    
    def get_engineered_feature_names(self):
        """Return list of all engineered features"""
        return self.engineered_features