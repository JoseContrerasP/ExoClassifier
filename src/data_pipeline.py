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
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

sys.path.append(PROJECT_ROOT)

import config
from src.feature_engineering import FeatureEngineer
from src.column_mapper import ColumnMapper
                                
class DataPipeline:
    def __init__(self, dataset_type: Optional[str] = None):
        """
        Initialize DataPipeline
        
        Args:
            dataset_type: 'kepler', 'tess', 'both', or None for auto-detection
        """
        self.dataset_type = dataset_type
        
        # Dataset URLs - automatically selected based on dataset_type
        self.kepler_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv"
        self.tess_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv"
        
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_engineer = FeatureEngineer()
        self.column_mapper = ColumnMapper()  
        
    def get_base_feature_list(self):
        """
        Get list of base features for training.
        Features are compatible with both Kepler and TESS datasets.
        """
        features = [
            # Transit parameters (measurements) - 4 features
            'koi_period',        # Orbital period (days) ✓ TESS
            'koi_depth',         # Transit depth (ppm) ✓ TESS
            'koi_duration',      # Transit duration (hours) ✓ TESS
            'koi_time0bk',       # Transit epoch ✓ TESS
            
            # Error columns (for uncertainty analysis) - 6 features
            'koi_period_err1',   # Period upper error ✓ TESS
            'koi_period_err2',   # Period lower error ✓ TESS
            'koi_depth_err1',    # Depth upper error ✓ TESS
            'koi_depth_err2',    # Depth lower error ✓ TESS
            'koi_duration_err1', # Duration upper error ✓ TESS
            'koi_duration_err2', # Duration lower error ✓ TESS
            
            # Signal quality - 1 feature
            'koi_model_snr',     # Signal-to-noise ratio ✓ TESS (calculated)

            # Planet properties (derived) - 3 features
            'koi_prad',          # Planet radius (Earth radii) ✓ TESS
            'koi_sma',           # Semi-major axis (AU) ✓ TESS (calculated)
            # REMOVED: 'koi_impact' - Impact parameter ✗ TESS
            'koi_teq',           # Equilibrium temperature (K) ✓ TESS
            'koi_insol',         # Insolation flux (Earth flux) ✓ TESS
            # REMOVED: 'koi_incl' - Inclination (degrees) ✗ TESS
            
            # Stellar properties - 4 features
            'koi_steff',         # Stellar effective temperature (K) ✓ TESS
            'koi_slogg',         # Stellar surface gravity ✓ TESS
            'koi_srad',          # Stellar radius (solar radii) ✓ TESS
            'koi_smass',         # Stellar mass (solar masses) ✓ TESS (calculated)
        ]
        
        return features
    
    def load_data(self):
        """Load data from Kepler, TESS, or both"""
        if self.dataset_type == 'both':
            print("LOADING BOTH KEPLER AND TESS DATA...")
            return self._load_combined_data()
        elif self.dataset_type == 'tess':
            print("LOADING NASA TESS TOI DATA...")
            data_source = self.tess_url
        elif self.dataset_type == 'kepler':
            print("LOADING NASA KEPLER KOI DATA...")
            data_source = self.kepler_url
        else:
            # Default to both datasets
            print("LOADING BOTH KEPLER AND TESS DATA (default)...")
            self.dataset_type = 'both'
            return self._load_combined_data()
        
        try:
            df = pd.read_csv(data_source)
            print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            
            if self.dataset_type is None or self.dataset_type == 'auto':
                detected = self.column_mapper.detect_dataset_type(df)
                self.dataset_type = detected
                print(f"Auto-detected dataset type: {detected}")
            else:
                # Verify the dataset type matches the data
                detected = self.column_mapper.detect_dataset_type(df)
                if detected != self.dataset_type:
                    print(f"  Warning: Expected {self.dataset_type}, but detected {detected}")
                    print(f"  Using detected type: {detected}")
                    self.dataset_type = detected
            
            # Show data info if disposition column exists
            disp_col = 'koi_disposition'
            if disp_col in df.columns:
                print(f"\nDataset info ({disp_col}):")
                disposition_counts = df[disp_col].value_counts()
                for disp, count in disposition_counts.items():
                    print(f"  {disp}: {count}")
            elif 'tfopwg_disp' in df.columns:
                # TESS disposition column
                print(f"\nDataset info (TESS disposition):")
                disposition_counts = df['tfopwg_disp'].value_counts()
                for disp, count in disposition_counts.items():
                    print(f"  {disp}: {count}")
            else:
                print("\nNo disposition column found - this may be a feature-only dataset")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
        
        return df
    
    def _load_combined_data(self):
        """Load and combine both Kepler and TESS data"""
        print("\n" + "="*70)
        print("LOADING COMBINED DATASET (KEPLER + TESS)")
        print("="*70)
        
        # Load Kepler data
        print("\n[1/3] Loading Kepler data...")
        df_kepler = pd.read_csv(self.kepler_url)
        print(f"  Loaded {len(df_kepler)} Kepler KOIs")
        
        # Add source column
        df_kepler['data_source'] = 'kepler'
        
        # Load TESS data
        print("\n[2/3] Loading TESS data...")
        df_tess_raw = pd.read_csv(self.tess_url)
        print(f"  Loaded {len(df_tess_raw)} TESS TOIs")
        
        # Map TESS to Kepler format
        print("  Mapping TESS columns to standard format...")
        df_tess_mapped, mapping_info = self.column_mapper.map_columns(df_tess_raw, 'tess')
        print(f"    Mapped {mapping_info['mapped_columns']} columns")
        
        # Convert units and calculate missing fields
        df_tess, conversion_info = self.column_mapper.validate_and_convert(df_tess_mapped)
        print(f"    Applied {conversion_info['total_conversions']} conversions")
        
        # Add source column
        df_tess['data_source'] = 'tess'
        
        # Standardize disposition column names
        if 'tfopwg_disp' in df_tess.columns and 'koi_disposition' not in df_tess.columns:
            # Map TESS disposition to Kepler-style labels
            disp_mapping = {
                'CP': 'CONFIRMED',
                'PC': 'CANDIDATE', 
                'KP': 'CONFIRMED',
                'APC': 'CANDIDATE',
                'FP': 'FALSE POSITIVE',
                'FA': 'FALSE POSITIVE'
            }
            df_tess['koi_disposition'] = df_tess['tfopwg_disp'].map(disp_mapping)
            print(f"    Mapped TESS disposition to Kepler format")
        
        # Find common columns (features + disposition)
        common_cols = list(set(df_kepler.columns) & set(df_tess.columns))
        print(f"\n[3/3] Combining datasets...")
        print(f"  Common columns: {len(common_cols)}")
        
        # Combine datasets
        df_combined = pd.concat([
            df_kepler[common_cols],
            df_tess[common_cols]
        ], ignore_index=True)
        
        print(f"\n  Combined dataset:")
        print(f"    Total samples: {len(df_combined)}")
        print(f"    Kepler samples: {len(df_kepler)}")
        print(f"    TESS samples: {len(df_tess)}")
        print(f"    Total columns: {len(df_combined.columns)}")
        
        # Show disposition distribution
        if 'koi_disposition' in df_combined.columns:
            print(f"\n  Combined disposition distribution:")
            disp_counts = df_combined['koi_disposition'].value_counts()
            for disp, count in disp_counts.items():
                print(f"    {disp}: {count}")
        
        print("="*70)
        
        return df_combined
    
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
        print("\n" + "="*70)
        print("PREPARING DATASET WITH FEATURE ENGINEERING")
        print("="*70)
        
        # Step 1: Check if already mapped (combined dataset) or needs mapping
        if self.dataset_type == 'both':
            print("\nUsing pre-mapped combined dataset (Kepler + TESS)")
            df_working = df.copy()
        elif self.dataset_type == 'tess':
            print(f"\nMapping {self.dataset_type.upper()} columns to standard format...")
            df_mapped, mapping_info = self.column_mapper.map_columns(df, self.dataset_type)
            
            print(f"  Mapped {mapping_info['mapped_columns']} columns")
            print(f"  Coverage: {mapping_info['coverage']}")
            
            if mapping_info['required_missing']:
                print(f"  Warning: Missing required columns: {mapping_info['required_missing']}")

            df_converted, conversion_info = self.column_mapper.validate_and_convert(df_mapped)
            
            print(f"  Applied {conversion_info['total_conversions']} conversions:")
            for conv in conversion_info['conversions_applied']:
                print(f"    - {conv}")
            
            df_working = df_converted
        else:  # kepler or default
            print("\nUsing Kepler data directly (no mapping needed)")
            df_working = df.copy()
        
        # Step 3: Get base features
        base_features = self.get_base_feature_list()

        # Step 4: Apply feature engineering
        print("\nApplying advanced feature engineering...")
        df_eng = self.feature_engineer.engineer_all_features(df_working)
        engineered_features = self.feature_engineer.get_engineered_feature_names()
        
        # Combine all features
        all_features = base_features + engineered_features
        
        available = [f for f in all_features if f in df_eng.columns]
        self.feature_names = available
        
        print(f"\nFeature Summary:")
        print(f"  Base features: {len(base_features)}")
        print(f"  Engineered features: {len(engineered_features)}")
        print(f"  Total available: {len(available)}")
        
        # Step 5: Create target variable
        disp_col = 'koi_disposition'
        
        # Try multiple possible disposition column names
        if disp_col in df_eng.columns:
            df_eng['target'] = df_eng[disp_col].apply(
                lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0
            )
            print(f"Using disposition column: {disp_col}")
        elif 'tfopwg_disp' in df_eng.columns:
            df_eng['target'] = df_eng['tfopwg_disp'].apply(
                lambda x: 1 if str(x).upper() in ['CP', 'PC', 'KP', 'APC'] else 0
            )
            print(f"Using disposition column: tfopwg_disp")
        
        print(f"\nTarget Distribution:")
        target_counts = df_eng['target'].value_counts()
        total = len(df_eng)
        planets = target_counts.get(1, 0)
        false_pos = target_counts.get(0, 0)
        
        print(f"  Planets (1):         {planets:5d} ({planets/total*100:5.1f}%)")
        print(f"  False Positives (0): {false_pos:5d} ({false_pos/total*100:5.1f}%)")
        if planets > 0:
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
        
        df_reduced, dropped = self.drop_highly_correlated(df_clean[available], threshold=config.CORRELATION_THRESHOLD)
        print(f"Remaining features after correlation removal: {len(df_reduced.columns)}")
        if dropped:
            print(f"Dropped {len(dropped)} correlated features: {dropped[:5]}{'...' if len(dropped) > 5 else ''}")
        
        self.feature_names = list(df_reduced.columns)
        
        X = df_reduced.values
        y = df_clean['target'].values

        # self.plot_correlation(df_reduced)
        
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


if __name__ == "__main__":
    """
    Example usage for both Kepler and TESS datasets
    """
    
    # ========== EXAMPLE 1: KEPLER DATA ==========
    print("="*70)
    print("EXAMPLE 1: TRAINING WITH KEPLER DATA")
    print("="*70)
    
    # Initialize pipeline for Kepler
    pipeline_kepler = DataPipeline(dataset_type='kepler')
    
    # Load Kepler data from NASA archive
    df_kepler = pipeline_kepler.load_data()
    
    # Prepare dataset (no mapping needed for Kepler)
    X, y, df_clean = pipeline_kepler.prepare_dataset(df_kepler)
    
    # Split and scale
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, X_train, X_val, X_test = pipeline_kepler.split_and_scale(X, y)
    
    # Save artifacts
    pipeline_kepler.save_artifacts()
    
    print("\nKepler pipeline executed successfully!")
    
    # ========== EXAMPLE 2: TESS DATA ==========
    # Uncomment to use TESS data
    # print("\n" + "="*70)
    # print("EXAMPLE 2: TRAINING WITH TESS DATA")
    # print("="*70)
    # 
    # # Initialize pipeline for TESS (auto-detect)
    # pipeline_tess = DataPipeline(dataset_type=None)  # Will auto-detect
    # 
    # # Load TESS data from CSV file
    # tess_file_path = "C:/Users/74401/Downloads/raw_tess_toi.csv"
    # df_tess = pipeline_tess.load_data(data_source=tess_file_path)
    # 
    # # Prepare dataset (with automatic column mapping)
    # X_tess, y_tess, df_tess_clean = pipeline_tess.prepare_dataset(df_tess)
    # 
    # # Split and scale
    # X_train_scaled_tess, X_val_scaled_tess, X_test_scaled_tess, y_train_tess, y_val_tess, y_test_tess, X_train_tess, X_val_tess, X_test_tess = pipeline_tess.split_and_scale(X_tess, y_tess)
    # 
    # # Save artifacts
    # pipeline_tess.save_artifacts()
    # 
    # print("\nTESS pipeline executed successfully!")
