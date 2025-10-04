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
        
        print("\n4. Flag Aggregations & Basic Features:")
        
        # Feature 15: Total Centroid Offset
        if 'koi_dicco_mra' in df.columns and 'koi_dicco_mdec' in df.columns:
            df['centroid_offset_total'] = np.sqrt(df['koi_dicco_mra']**2 + df['koi_dicco_mdec']**2)
            self.engineered_features.append('centroid_offset_total')
            print("  ✓ centroid_offset_total = √(RA² + Dec²)")
        
        # Feature 16: Number of Red Flags
        flag_cols = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
        available_flags = [col for col in flag_cols if col in df.columns]
        if available_flags:
            df['num_red_flags'] = df[available_flags].fillna(0).sum(axis=1)
            self.engineered_features.append('num_red_flags')
            print(f"  ✓ num_red_flags = sum of {len(available_flags)} diagnostic flags")
            
            # Feature 17: Any Red Flag (binary)
            df['any_red_flag'] = (df['num_red_flags'] > 0).astype(int)
            self.engineered_features.append('any_red_flag')
            print("  ✓ any_red_flag = 1 if num_red_flags > 0")
        
        # Feature 18: Warning Category
        # 0 = clean, 1 = one warning, 2 = multiple warnings
        if 'num_red_flags' in df.columns:
            df['warning_category'] = np.clip(df['num_red_flags'], 0, 2)
            self.engineered_features.append('warning_category')
            print("  ✓ warning_category = 0 (clean), 1 (warning), 2+ (multiple)")
        
        return df
    
    def get_engineered_feature_names(self):
        """Return list of all engineered features"""
        return self.engineered_features