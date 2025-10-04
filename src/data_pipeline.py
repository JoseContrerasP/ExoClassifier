import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

sys.path.append(PROJECT_ROOT)

import config
from src.feature_engineering import FeatureEngineer
                                
class DataPipeline:
    def __init__(self):
        self.url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv"
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_engineer = FeatureEngineer()  
        
    def get_base_feature_list(self):
        features = [
            # Transit parameters (measurements)
            'koi_period',        # Orbital period (days)
            'koi_depth',         # Transit depth (ppm)
            'koi_duration',      # Transit duration (hours)
            'koi_time0bk',       # Transit epoch
            
            # NEW: Error columns (for uncertainty analysis)
            'koi_period_err1',   # Period upper error
            'koi_period_err2',   # Period lower error
            'koi_depth_err1',    # Depth upper error
            'koi_depth_err2',    # Depth lower error
            'koi_duration_err1', # Duration upper error
            'koi_duration_err2', # Duration lower error
            
            # Diagnostic tests - Signal quality
            'koi_model_snr',     # Signal-to-noise ratio
            
            # Diagnostic tests - Centroid (odd-even transit analysis)
            'koi_dicco_msky',    # Centroid offset magnitude
            'koi_dicco_msky_err',# Centroid offset error
            'koi_dicco_mra',     # RA component of centroid offset
            'koi_dicco_mdec',    # Dec component of centroid offset
            
            # Diagnostic flags
            'koi_fpflag_nt',     # Not transit-like flag
            'koi_fpflag_ss',     # Secondary eclipse (stellar eclipse)
            'koi_fpflag_co',     # Centroid offset flag
            'koi_fpflag_ec',     # Ephemeris contamination flag
            
            # Planet properties (derived)
            'koi_prad',          # Planet radius (Earth radii)
            'koi_sma',           # Semi-major axis (AU)
            'koi_impact',        # Impact parameter
            'koi_teq',           # Equilibrium temperature (K)
            'koi_insol',         # Insolation flux (Earth flux)
            'koi_incl',          # Inclination (degrees)
            
            # Stellar properties
            'koi_steff',         # Stellar effective temperature (K)
            'koi_slogg',         # Stellar surface gravity
            'koi_srad',          # Stellar radius (solar radii)
            'koi_smass',         # Stellar mass (solar masses)
        ]
        
        return features
    
    def load_data(self):
        print("LOADING NASA KEPLER KOI DATA...")        
        try:
            
            df = pd.read_csv(self.url)
            print(f"Loaded {len(df)} Kepler Objects of Interest")
            
            # Show data info
            print(f"\nDataset info:")
            print(f"  Confirmed planets: {(df['koi_disposition'] == 'CONFIRMED').sum()}")
            print(f"  Candidates: {(df['koi_disposition'] == 'CANDIDATE').sum()}")
            print(f"  False positives: {(df['koi_disposition'] == 'FALSE POSITIVE').sum()}")
            
        except Exception as e:
            print(e)
            raise
        
        return df
    def plot_correlation(self, df):
        plt.figure(figsize=(15, 12))
        corr = df.corr()
        sns.heatmap(corr, cmap="coolwarm", center=0, 
                    cbar=True, square=True, linewidths=0.5)
        plt.title("Correlation Heatmap of Features", fontsize=16)
        plt.show()

    def drop_highly_correlated(self, df, threshold=0.9):
        corr_matrix = df.corr().abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        print(f"Dropping {len(to_drop)} highly correlated features (>{threshold}):")
        print(to_drop)

        # Drop them
        df_reduced = df.drop(columns=to_drop)
        return df_reduced, to_drop
    
    def prepare_dataset(self, df):
        print("PREPARING DATASET WITH ADVANCED FEATURES")
        
        base_features = self.get_base_feature_list()

        print("\nApplying advanced feature engineering...")

        df_eng = self.feature_engineer.engineer_all_features(df)
        engineered_features = self.feature_engineer.get_engineered_feature_names()
        
        # Combine all features
        all_features = base_features + engineered_features
        
        available = [f for f in all_features if f in df_eng.columns]
        self.feature_names = available
        
        df_eng['target'] = df_eng['koi_disposition'].apply(
            lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0
        )
        
        print(f"\nTarget Distribution:")
        target_counts = df_eng['target'].value_counts()
        total = len(df_eng)
        planets = target_counts.get(1, 0)
        false_pos = target_counts.get(0, 0)
        
        print(f"  Planets (1):         {planets:5d} ({planets/total*100:5.1f}%)")
        print(f"  False Positives (0): {false_pos:5d} ({false_pos/total*100:5.1f}%)")
        print(f"  Class imbalance:     {false_pos/planets:.2f}:1 (FP:Planet)")
        
        # Handle missing values
        print(f"\nHandling Missing Values:")
        
        df_clean = df_eng[available + ['target']].copy()
        
        # For diagnostic flags, NaN means flag not set (0)
        flag_cols = [col for col in available if 'flag' in col.lower()]
        if flag_cols:
            df_clean[flag_cols] = df_clean[flag_cols].fillna(0)
            print(f"  ✓ Filled {len(flag_cols)} flag columns with 0 (no flag)")
        
        # Handling NaN values with median
        for col in df_clean.columns:
            if df_clean[col].isna().sum() > 0:
                # Check data type
                if df_clean[col].dtype in ['float64', 'int64']:
                    # Fill numeric columns with median
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)


        # Final dataset info
        print(f"\n{'='*70}")
        print(f"FINAL DATASET")
        print(f"{'='*70}")
        print(f"  Samples:  {len(df_clean)}")
        print(f"  Features: {len(available)}")
        print(f"  Planets:  {df_clean['target'].sum()} ({df_clean['target'].sum()/len(df_clean)*100:.1f}%)")
        print(f"  False Positives: {len(df_clean) - df_clean['target'].sum()} ({(len(df_clean) - df_clean['target'].sum())/len(df_clean)*100:.1f}%)")
        
        X = df_clean[available].values
        y = df_clean['target'].values

        df_reduced, dropped = self.drop_highly_correlated(df_clean[available], threshold=0.9)
        print(f"Remaining features: {len(df_reduced.columns)}")


        # self.plot_correlation(df_clean[available])
        
        return X, y, df_reduced
    
    def split_and_scale(self, X, y):
        print("\n" + "="*70)
        print("STEP 3: CREATING DATA SPLITS AND SCALING")
        print("="*70)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE, 
            stratify=y
        )
        
        # Second split: separate validation from training
        val_ratio = config.VAL_SIZE / (1 - config.TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_ratio, 
            random_state=config.RANDOM_STATE, 
            stratify=y_temp
        )
        
        print(f"\nDataset splits:")
        print(f"  Train:      {len(X_train):5d} samples ({len(X_train)/len(X)*100:5.1f}%)")
        print(f"    - Planets: {y_train.sum()}")
        print(f"    - False Positives: {len(y_train) - y_train.sum()}")
        print(f"  Validation: {len(X_val):5d} samples ({len(X_val)/len(X)*100:5.1f}%)")
        print(f"    - Planets: {y_val.sum()}")
        print(f"    - False Positives: {len(y_val) - y_val.sum()}")
        print(f"  Test:       {len(X_test):5d} samples ({len(X_test)/len(X)*100:5.1f}%)")
        print(f"    - Planets: {y_test.sum()}")
        print(f"    - False Positives: {len(y_test) - y_test.sum()}")
        
        # Scale features
        print(f"\nScaling features using StandardScaler...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Features scaled (mean=0, std=1)")
        
        # Return both scaled (for NN) and unscaled (for tree models)
        return (X_train_scaled, X_val_scaled, X_test_scaled,
                y_train, y_val, y_test,
                X_train, X_val, X_test)
    
    def save_artifacts(self):
        print("\n" + "="*70)
        print("SAVING PREPROCESSING ARTIFACTS")
        print("="*70)
        
        # Save scaler
        joblib.dump(self.scaler, config.SCALER_PATH)
        print(f"saved scaler to: {config.SCALER_PATH}")
        
        # Save feature names
        with open(config.FEATURE_NAMES_PATH, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print(f"Saved feature names to: {config.FEATURE_NAMES_PATH}")
        
        print(f"\nArtifacts saved for API use:")
        print(f"  - Scaler: {config.SCALER_PATH}")
        print(f"  - Feature names: {config.FEATURE_NAMES_PATH}")


# if __name__ == "__main__":
#     # Initialize pipeline
#     pipeline = DataPipeline()

#     # Step 1: Load data
#     df = pipeline.load_data()

#     # Step 2: Prepare dataset (feature engineering + cleaning + target)
#     X, y, df_clean = pipeline.prepare_dataset(df)

#     # Step 3: Split and scale
#     X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, X_train, X_val, X_test = pipeline.split_and_scale(X, y)

#     # Step 4: Save preprocessing artifacts
#     pipeline.save_artifacts()

#     print("\nPipeline executed successfully!")
