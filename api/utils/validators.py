import pandas as pd
import numpy as np
from typing import Dict, List

# Value ranges for physical parameters (min, max)
VALUE_RANGES = {
    'pl_orbper': (0.1, 10000),           # Orbital period (days)
    'pl_trandep': (0.0001, 0.5),         # Transit depth (fraction)
    'pl_trandurh': (0.1, 48),            # Transit duration (hours)
    'pl_tranmid': (2450000, 2470000),    # BJD
    'pl_orbpererr1': (0, 1000),          # Errors
    'pl_orbpererr2': (0, 1000),
    'pl_trandeperr1': (0, 0.1),
    'pl_trandeperr2': (0, 0.1),
    'pl_trandurherr1': (0, 24),
    'pl_trandurherr2': (0, 24),
    'pl_rade': (0.1, 30),                # Planet radius (Earth radii)
    'pl_eqt': (0, 5000),                 # Equilibrium temp (K)
    'pl_insol': (0, 10000),              # Insolation flux
    'st_teff': (2500, 10000),            # Stellar temp (K) - KEY VALIDATION
    'st_logg': (3.0, 5.5),               # Surface gravity
    'st_rad': (0.1, 10),                 # Stellar radius (solar radii)
    'koi_model_snr': (0, 1000),          # SNR (capped)
    'koi_smass': (0.1, 10),              # Stellar mass (solar masses)
    'koi_sma': (0, 10),                  # Semi-major axis (AU)
}

# Required columns for each dataset type
REQUIRED_COLUMNS = {
    'tess': [
        'pl_orbper', 'pl_trandep', 'pl_trandurh', 'pl_tranmid',
        'pl_orbpererr1', 'pl_orbpererr2', 'pl_trandeperr1', 'pl_trandeperr2',
        'pl_trandurherr1', 'pl_trandurherr2',
        'pl_rade', 'pl_eqt', 'pl_insol',
        'st_teff', 'st_logg', 'st_rad'
    ],
    'kepler': [
        'pl_orbper', 'pl_trandep', 'pl_trandurh', 'pl_tranmid',
        'pl_orbpererr1', 'pl_orbpererr2', 'pl_trandeperr1', 'pl_trandeperr2',
        'pl_trandurherr1', 'pl_trandurherr2',
        'pl_rade', 'pl_eqt', 'pl_insol',
        'st_teff', 'st_logg', 'st_rad',
        'koi_model_snr', 'koi_smass', 'koi_sma'
    ]
}

def validate_dataset_type(dataset_type: str) -> Dict:
    """
    Validate dataset type
    
    Args:
        dataset_type: Dataset type string
        
    Returns:
        Dictionary with 'valid' boolean and optional 'error' message
    """
    if dataset_type not in ['tess', 'kepler']:
        return {
            'valid': False,
            'error': f'Invalid dataset type "{dataset_type}". Must be "tess" or "kepler"'
        }
    return {'valid': True}

def validate_feature_ranges(features: Dict) -> Dict:

    warnings = []
    errors = []
    
    for feature_name, value in features.items():
        if feature_name not in VALUE_RANGES:
            continue
            
        min_val, max_val = VALUE_RANGES[feature_name]
        
        try:
            value_float = float(value)
            
            # Check for NaN or Inf
            if np.isnan(value_float) or np.isinf(value_float):
                errors.append(f"{feature_name}: Invalid value (NaN or Inf)")
                continue
            
            # Check range
            if value_float < min_val or value_float > max_val:
                errors.append(
                    f"{feature_name}: Value {value_float:.4f} is outside valid range "
                    f"[{min_val}, {max_val}]. Please check your input."
                )
        except (ValueError, TypeError):
            errors.append(f"{feature_name}: Could not convert '{value}' to a number")
    
    return {
        'valid': len(errors) == 0,
        'warnings': warnings,
        'errors': errors
    }

def validate_csv_columns(df: pd.DataFrame, dataset_type: str) -> Dict:
 
    required_cols = REQUIRED_COLUMNS.get(dataset_type, [])
    
    # Check for missing columns
    df_columns = set(df.columns)
    required_set = set(required_cols)
    missing_columns = required_set - df_columns
    
    if missing_columns:
        return {
            'valid': False,
            'error': f'Missing required columns for {dataset_type.upper()} dataset',
            'missing_columns': list(missing_columns),
            'required_columns': required_cols,
            'found_columns': list(df_columns)
        }
    
    return {
        'valid': True,
        'found_columns': list(df_columns),
        'required_columns': required_cols
    }

def validate_numeric_columns(df: pd.DataFrame, columns: List[str]) -> Dict:

    non_numeric = []
    
    for col in columns:
        if col in df.columns:
            # Try converting to numeric
            try:
                pd.to_numeric(df[col], errors='raise')
            except (ValueError, TypeError):
                non_numeric.append(col)
    
    if non_numeric:
        return {
            'valid': False,
            'error': 'Some columns contain non-numeric data',
            'non_numeric_columns': non_numeric
        }
    
    return {'valid': True}

def validate_value_ranges(df: pd.DataFrame) -> Dict:

    warnings = []
    
    # Define reasonable ranges
    ranges = {
        'pl_orbper': (0.1, 10000),      # days
        'pl_trandep': (0.00001, 0.5),    # fraction
        'pl_trandurh': (0.1, 48),        # hours
        'st_teff': (2500, 10000),        # Kelvin
        'st_logg': (3.0, 5.5),           # log scale
        'st_rad': (0.1, 10),             # solar radii
        'pl_rade': (0.1, 30),            # Earth radii
        'pl_eqt': (0, 5000),             # Kelvin
        'pl_insol': (0, 10000),          # Earth flux
    }
    
    for col, (min_val, max_val) in ranges.items():
        if col in df.columns:
            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
            if len(out_of_range) > 0:
                warnings.append({
                    'column': col,
                    'message': f'{len(out_of_range)} values outside expected range [{min_val}, {max_val}]',
                    'count': len(out_of_range)
                })
    
    return {
        'valid': True,
        'warnings': warnings
    }

