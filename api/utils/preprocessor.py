import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
sys.path.append(PROJECT_ROOT)

from src.column_mapper import ColumnMapper
from src.feature_engineering import FeatureEngineer

class DataPreprocessor:
    """
    Handles preprocessing of uploaded CSV data based on dataset type
    """
    
    def __init__(self, dataset_type='tess'):
        self.dataset_type = dataset_type
        self.column_mapper = ColumnMapper()
        self.feature_engineer = FeatureEngineer()
        
    def preprocess_csv(self, df_raw, upload_id):

        try:
            total_rows = len(df_raw)
            
            # Step 1: Map columns to standard format
            print(f"\n[PREPROCESSING] Upload ID: {upload_id}")
            print(f"[PREPROCESSING] Dataset type: {self.dataset_type}")
            print(f"[PREPROCESSING] Raw data: {total_rows} rows, {len(df_raw.columns)} columns")
            
            df_mapped, mapping_info = self.column_mapper.map_columns(df_raw, self.dataset_type)
            print(f"[PREPROCESSING] Mapped {mapping_info['mapped_columns']} columns")
            
            # Step 2: Convert units and calculate missing fields (for TESS)
            auto_calculated_fields = []
            if self.dataset_type == 'tess':
                df_converted, conversion_info = self.column_mapper.validate_and_convert(df_mapped)
                auto_calculated_fields = ['koi_model_snr', 'koi_smass', 'koi_sma']
                print(f"[PREPROCESSING] Applied {conversion_info['total_conversions']} conversions")
                print(f"[PREPROCESSING] Auto-calculated: {', '.join(auto_calculated_fields)}")
                df_processed = df_converted
            else:
                # For Kepler, just use mapped data
                df_processed = df_mapped
            
            # Step 3: Handle missing values in required columns
            missing_before = df_processed.isnull().sum().sum()
            
            # Get base features
            base_features = self._get_base_features()
            
            # Fill missing values with median for numeric columns
            missing_filled = 0
            for col in base_features:
                if col in df_processed.columns:
                    if df_processed[col].isnull().any():
                        median_val = df_processed[col].median()
                        df_processed[col].fillna(median_val, inplace=True)
                        missing_filled += 1
            
            print(f"[PREPROCESSING] Filled missing values in {missing_filled} columns")
            
            # Step 4: Remove rows with too many missing values
            # Keep rows that have at least 80% of required features
            threshold = int(0.8 * len(base_features))
            df_clean = df_processed.dropna(thresh=threshold, subset=base_features)
            removed_rows = total_rows - len(df_clean)
            
            if removed_rows > 0:
                print(f"[PREPROCESSING] Removed {removed_rows} rows with excessive missing values")
            
            # Step 5: Apply feature engineering
            print(f"[PREPROCESSING] Applying feature engineering...")
            df_engineered = self.feature_engineer.engineer_all_features(df_clean)
            
            engineered_features = self.feature_engineer.engineered_features
            print(f"[PREPROCESSING] Created {len(engineered_features)} engineered features")
            
            # Step 6: Final feature selection
            all_features = base_features + engineered_features
            available_features = [f for f in all_features if f in df_engineered.columns]
            
            df_final = df_engineered[available_features].copy()
            
            print(f"[PREPROCESSING] Final dataset: {len(df_final)} rows, {len(available_features)} features")
            print(f"[PREPROCESSING] Preprocessing complete!")
            
            return {
                'success': True,
                'processed_data': df_final,
                'total_rows': total_rows,
                'processed_rows': len(df_final),
                'removed_rows': removed_rows,
                'total_features': len(available_features),
                'base_features': len(base_features),
                'engineered_features': len(engineered_features),
                'missing_values_filled': missing_filled,
                'auto_calculated_fields': auto_calculated_fields
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Preprocessing failed: {str(e)}'
            }
    
    def _get_base_features(self):
        base_features = [
            # Transit parameters
            'koi_period', 'koi_depth', 'koi_duration', 'koi_time0bk',
            # Error columns
            'koi_period_err1', 'koi_period_err2',
            'koi_depth_err1', 'koi_depth_err2',
            'koi_duration_err1', 'koi_duration_err2',
            # Signal quality
            'koi_model_snr',
            # Planet properties
            'koi_prad', 'koi_sma', 'koi_teq', 'koi_insol',
            # Stellar properties
            'koi_steff', 'koi_slogg', 'koi_srad', 'koi_smass'
        ]
        
        return base_features


def preprocess_data(df, dataset_type='tess'):
    preprocessor = DataPreprocessor(dataset_type=dataset_type)
    result = preprocessor.preprocess_csv(df, upload_id='training')
    
    if result['success']:
        return result['processed_data']
    else:
        raise Exception(result['error'])
