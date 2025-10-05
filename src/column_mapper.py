"""
Column Mapper for Different Exoplanet Datasets
Maps TESS, Kepler, and custom datasets to standardized column names
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class ColumnMapper:
    """
    Maps column names from different exoplanet surveys to standardized names
    """
    
    # Standardized column names (what our model expects)
    STANDARD_COLUMNS = [
        # Transit parameters
        'koi_period',        # Orbital period (days)
        'koi_depth',         # Transit depth (ppm)
        'koi_duration',      # Transit duration (hours)
        'koi_time0bk',       # Transit epoch
        
        # Error columns
        'koi_period_err1',
        'koi_depth_err1',
        'koi_duration_err1',
        
        # Signal quality
        'koi_model_snr',     # Signal-to-noise ratio
        
        # Centroid measurements
        'koi_dicco_msky',
        'koi_dicco_msky_err',
        'koi_dicco_mra',
        'koi_dicco_mdec',
        
        # Planet properties
        'koi_prad',          # Planet radius (Earth radii)
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
    
    # Kepler column mapping (identity mapping)
    KEPLER_MAPPING = {
        'koi_period': 'koi_period',
        'koi_depth': 'koi_depth',
        'koi_duration': 'koi_duration',
        'koi_time0bk': 'koi_time0bk',
        'koi_period_err1': 'koi_period_err1',
        'koi_depth_err1': 'koi_depth_err1',
        'koi_duration_err1': 'koi_duration_err1',
        'koi_model_snr': 'koi_model_snr',
        'koi_dicco_msky': 'koi_dicco_msky',
        'koi_dicco_msky_err': 'koi_dicco_msky_err',
        'koi_dicco_mra': 'koi_dicco_mra',
        'koi_dicco_mdec': 'koi_dicco_mdec',
        'koi_prad': 'koi_prad',
        'koi_impact': 'koi_impact',
        'koi_teq': 'koi_teq',
        'koi_insol': 'koi_insol',
        'koi_incl': 'koi_incl',
        'koi_steff': 'koi_steff',
        'koi_slogg': 'koi_slogg',
        'koi_srad': 'koi_srad',
        'koi_smass': 'koi_smass',
    }
    
    # TESS column mapping (NASA Exoplanet Archive TOI format)
    TESS_MAPPING = {
        # Transit parameters (4 columns)
        'pl_orbper': 'koi_period',           # Orbital Period (days)
        'pl_trandep': 'koi_depth',           # Transit Depth (fraction) - need conversion to ppm
        'pl_trandurh': 'koi_duration',       # Transit Duration (hours)
        'pl_tranmid': 'koi_time0bk',         # Transit Midpoint (BJD)
        
        # Error columns (6 columns - err1 and err2)
        'pl_orbpererr1': 'koi_period_err1',
        'pl_orbpererr2': 'koi_period_err2',
        'pl_trandeperr1': 'koi_depth_err1',
        'pl_trandeperr2': 'koi_depth_err2',
        'pl_trandurherr1': 'koi_duration_err1',
        'pl_trandurherr2': 'koi_duration_err2',
        
        # Signal quality - koi_model_snr will be CALCULATED in validate_and_convert
        
        # Planet properties (3 columns available in TESS)
        'pl_rade': 'koi_prad',               # Planet Radius (Earth radii)
        'pl_eqt': 'koi_teq',                 # Equilibrium Temperature (K)
        'pl_insol': 'koi_insol',             # Insolation Flux (Earth flux)
        # NOT IN TESS: koi_impact, koi_incl (will be excluded from training)
        
        # Stellar properties (3 columns available, mass will be CALCULATED)
        'st_teff': 'koi_steff',              # Stellar Effective Temperature (K)
        'st_logg': 'koi_slogg',              # Stellar Surface Gravity
        'st_rad': 'koi_srad',                # Stellar Radius (solar radii)
        # koi_smass will be CALCULATED from radius and logg
        # koi_sma will be CALCULATED from period and stellar mass
        
        # Additional useful columns
        'st_dist': 'st_dist',                # Distance (pc)
        'toi': 'toi',                        # TESS Object of Interest number
        'tid': 'tid',                        # TESS Input Catalog ID
        'st_tmag': 'st_tmag',                # TESS magnitude
        'st_pmra': 'st_pmra',                # Proper motion RA
        'st_pmdec': 'st_pmdec',              # Proper motion Dec
        
        # IMPORTANT: Disposition column (for training labels)
        'tfopwg_disp': 'tfopwg_disp',        # TFOP Working Group Disposition (CP, PC, FP, etc.)
    }
    
    # Alternative TESS column names (TOI format)
    TESS_TOI_MAPPING = {
        'toi_period': 'koi_period',
        'toi_depth': 'koi_depth',
        'toi_duration': 'koi_duration',
        'toi_time0': 'koi_time0bk',
        'toi_period_err': 'koi_period_err1',
        'toi_depth_err': 'koi_depth_err1',
        'toi_duration_err': 'koi_duration_err1',
        'toi_snr': 'koi_model_snr',
    }
    
    # Common alternative names (user-friendly names)
    COMMON_MAPPING = {
        'period': 'koi_period',
        'orbital_period': 'koi_period',
        'depth': 'koi_depth',
        'transit_depth': 'koi_depth',
        'duration': 'koi_duration',
        'transit_duration': 'koi_duration',
        'epoch': 'koi_time0bk',
        'snr': 'koi_model_snr',
        'signal_to_noise': 'koi_model_snr',
        'planet_radius': 'koi_prad',
        'radius': 'koi_prad',
        'impact_parameter': 'koi_impact',
        'temperature': 'koi_teq',
        'equilibrium_temp': 'koi_teq',
        'insolation': 'koi_insol',
        'inclination': 'koi_incl',
        'stellar_temp': 'koi_steff',
        'stellar_radius': 'koi_srad',
        'stellar_mass': 'koi_smass',
    }
    
    def __init__(self):
        """Initialize the column mapper"""
        self.dataset_type = None
        self.mapping_used = None
    
    def detect_dataset_type(self, df: pd.DataFrame) -> str:

        columns = set(df.columns.str.lower())
        
        # Check for Kepler columns
        kepler_indicators = {'koi_period', 'koi_depth', 'koi_disposition', 'koi_score'}
        if len(kepler_indicators & columns) >= 2:
            return 'kepler'
        
        # Check for TESS TOI columns (has 'toi' identifier and planet/stellar parameters)
        # Updated to match actual TESS TOI column format
        toi_indicators = {'toi', 'tid', 'pl_orbper', 'pl_trandep', 'pl_trandurh', 'st_tmag', 'tfopwg_disp'}
        if len(toi_indicators & columns) >= 3:
            return 'tess'
        
        # Check for TESS planet archive columns (confirmed planets)
        tess_confirmed_indicators = {'pl_orbper', 'pl_trandep', 'pl_rade', 'st_teff', 'pl_name'}
        if len(tess_confirmed_indicators & columns) >= 3:
            return 'tess'
        
        # Legacy TESS TOI format check
        toi_legacy_indicators = {'toi_period', 'toi_depth', 'toi', 'tfopwg_disp'}
        if len(toi_legacy_indicators & columns) >= 2:
            return 'tess_toi'
        
        # Check for common column names
        common_indicators = {'period', 'depth', 'radius', 'temperature'}
        if len(common_indicators & columns) >= 2:
            return 'custom'
        
        return 'unknown'
    
    def map_columns(self, df: pd.DataFrame, dataset_type: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:

        if dataset_type is None:
            dataset_type = self.detect_dataset_type(df)
        
        self.dataset_type = dataset_type
        
        # Select appropriate mapping
        if dataset_type == 'kepler':
            mapping = self.KEPLER_MAPPING
        elif dataset_type == 'tess':
            mapping = self.TESS_MAPPING
        elif dataset_type == 'tess_toi':
            mapping = self.TESS_TOI_MAPPING
        elif dataset_type == 'custom':
            mapping = self.COMMON_MAPPING
        else:
            # Try to use common mapping as fallback
            mapping = self.COMMON_MAPPING
        
        self.mapping_used = mapping
        
        # Create mapped dataframe
        df_mapped = pd.DataFrame()
        mapped_columns = []
        unmapped_columns = []
        
        # Convert column names to lowercase for matching
        df_lower = df.copy()
        df_lower.columns = df_lower.columns.str.lower()
        
        # Apply mapping
        for source_col, target_col in mapping.items():
            if source_col in df_lower.columns:
                df_mapped[target_col] = df_lower[source_col]
                mapped_columns.append(f"{source_col} → {target_col}")
        
        # Check for unmapped columns that might be important
        all_source_cols = set(mapping.keys())
        available_cols = set(df_lower.columns)
        unmapped = available_cols - all_source_cols
        
        # Create mapping info
        mapping_info = {
            'dataset_type': dataset_type,
            'total_columns': len(df.columns),
            'mapped_columns': len(df_mapped.columns),
            'unmapped_columns': len(unmapped),
            'mapping_details': mapped_columns,
            'unmapped_list': list(unmapped)[:20],  # First 20 unmapped
            'required_missing': self._check_required_columns(df_mapped),
            'coverage': f"{len(df_mapped.columns)}/{len(self.STANDARD_COLUMNS)}"
        }
        
        return df_mapped, mapping_info
    
    def _check_required_columns(self, df: pd.DataFrame) -> List[str]:

        required = [
            'koi_period',
            'koi_depth',
            'koi_duration',
            'koi_model_snr',
        ]
        
        missing = [col for col in required if col not in df.columns]
        return missing
    
    def validate_and_convert(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        df_converted = df.copy()
        conversions = []
        
        # TESS transit depth is often in fraction or percentage, need to convert to ppm
        if self.dataset_type in ['tess', 'tess_toi']:
            if 'koi_depth' in df_converted.columns:
                # Check if values are < 1 (fraction or percentage)
                max_depth = df_converted['koi_depth'].max()
                if max_depth < 1:
                    # If very small (< 0.1), likely a fraction, convert to ppm
                    if max_depth < 0.1:
                        df_converted['koi_depth'] = df_converted['koi_depth'] * 1000000  # fraction to ppm
                        conversions.append("Transit depth converted from fraction to ppm (×1,000,000)")
                    else:
                        # Otherwise, likely percentage
                        df_converted['koi_depth'] = df_converted['koi_depth'] * 10000  # % to ppm
                        conversions.append("Transit depth converted from % to ppm (×10,000)")
            
            # Calculate SNR if not present (TESS TOI data often lacks direct SNR)
            if 'koi_model_snr' not in df_converted.columns:
                if 'koi_depth' in df_converted.columns and 'koi_depth_err1' in df_converted.columns:
                    # Approximate SNR as depth / depth_error
                    # IMPORTANT: Both must be in same units (use original fraction, not ppm)
                    depth_fraction = df_converted['koi_depth'] / 1e6 if df_converted['koi_depth'].median() > 100 else df_converted['koi_depth']
                    df_converted['koi_model_snr'] = (
                        depth_fraction.abs() / 
                        df_converted['koi_depth_err1'].abs()
                    ).replace([np.inf, -np.inf], np.nan).clip(upper=1000)  # Cap at reasonable max
                    conversions.append("SNR calculated from depth/error ratio (capped at 1000)")
                elif 'koi_depth' in df_converted.columns:
                    # If no error available, estimate based on TESS typical noise
                    # Use magnitude as proxy for noise level
                    if 'st_tmag' in df_converted.columns:
                        # Brighter stars (lower mag) = higher SNR
                        # This is a rough approximation
                        df_converted['koi_model_snr'] = (
                            df_converted['koi_depth'] / 
                            (10 ** ((df_converted['st_tmag'] - 10) / 5))
                        )
                        conversions.append("SNR estimated from depth and TESS magnitude")
                    else:
                        # Last resort: use typical value based on depth
                        df_converted['koi_model_snr'] = df_converted['koi_depth'] / 100
                        conversions.append("SNR roughly estimated from depth (depth/100)")
        
        # Duration might be in different units
        if 'koi_duration' in df_converted.columns:
            # Check typical values (should be 0.5 to 24 hours)
            median_duration = df_converted['koi_duration'].median()
            if median_duration < 0.1:
                df_converted['koi_duration'] = df_converted['koi_duration'] * 24  # days to hours
                conversions.append("Duration converted from days to hours (×24)")
        
        # Calculate stellar mass if not present (from radius and surface gravity)
        if 'koi_smass' not in df_converted.columns:
            if 'koi_srad' in df_converted.columns and 'koi_slogg' in df_converted.columns:
                # M = R^2 * g / G (in solar units)
                # log(g) is in cgs, convert to solar masses
                # M/M_sun ≈ (R/R_sun)^2 * 10^(log_g - 4.44)
                df_converted['koi_smass'] = (
                    df_converted['koi_srad'] ** 2 * 
                    10 ** (df_converted['koi_slogg'] - 4.44)
                )
                conversions.append("Stellar mass calculated from radius and surface gravity")
        
        # Calculate semi-major axis if not present (from period and stellar mass)
        if 'koi_sma' not in df_converted.columns:
            if 'koi_period' in df_converted.columns and 'koi_smass' in df_converted.columns:
                # Kepler's 3rd law: a^3 = (G * M * P^2) / (4 * pi^2)
                # With P in days, M in solar masses, result in AU:
                # a (AU) = (M_star * P^2_days / 365.25^2) ^ (1/3)
                df_converted['koi_sma'] = (
                    df_converted['koi_smass'] * 
                    (df_converted['koi_period'] / 365.25) ** 2
                ) ** (1/3)
                conversions.append("Semi-major axis calculated from period and stellar mass")
        
        conversion_info = {
            'conversions_applied': conversions,
            'total_conversions': len(conversions)
        }
        
        return df_converted, conversion_info
    
    def get_mapping_summary(self, df_original: pd.DataFrame, df_mapped: pd.DataFrame) -> Dict:
        """
        Get comprehensive mapping summary for user feedback
        
        Args:
            df_original: Original dataframe
            df_mapped: Mapped dataframe
            
        Returns:
            Detailed summary dictionary
        """
        return {
            'status': 'success' if len(df_mapped.columns) >= 4 else 'incomplete',
            'dataset_type': self.dataset_type,
            'original_columns': len(df_original.columns),
            'mapped_columns': len(df_mapped.columns),
            'coverage_percent': round(len(df_mapped.columns) / len(self.STANDARD_COLUMNS) * 100, 1),
            'has_required': len(self._check_required_columns(df_mapped)) == 0,
            'missing_required': self._check_required_columns(df_mapped),
            'row_count': len(df_original),
            'recommendations': self._get_recommendations(df_mapped)
        }
    
    def _get_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on mapped columns"""
        recommendations = []
        
        missing = self._check_required_columns(df)
        if missing:
            recommendations.append(f"Missing required columns: {', '.join(missing)}")
        
        if 'koi_prad' not in df.columns:
            recommendations.append("Planet radius not available - some features will be limited")
        
        if 'koi_steff' not in df.columns:
            recommendations.append("Stellar properties missing - astrophysical features unavailable")
        
        if len(df.columns) < 10:
            recommendations.append("Limited columns available - prediction accuracy may be reduced")
        
        if not recommendations:
            recommendations.append("Dataset looks good! All key columns mapped successfully")
        
        return recommendations


def process_uploaded_csv(file_path: str, dataset_type: Optional[str] = None) -> Dict:

    try:
        df = pd.read_csv(file_path)
        
        mapper = ColumnMapper()
        df_mapped, mapping_info = mapper.map_columns(df, dataset_type)
        
        df_final, conversion_info = mapper.validate_and_convert(df_mapped)
        summary = mapper.get_mapping_summary(df, df_final)
        
        result = {
            'success': True,
            'data': df_final,
            'summary': summary,
            'mapping_info': mapping_info,
            'conversion_info': conversion_info,
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f"Error processing CSV: {str(e)}"
        }


# if __name__ == "__main__":
#     """Test the column mapper"""
    
#     # Test with sample TESS TOI data (actual format)
#     print("="*70)
#     print("TESTING COLUMN MAPPER - TESS TOI FORMAT")
#     print("="*70)
    
#     # Create sample TESS TOI data matching the actual column format
#     tess_toi_data = pd.DataFrame({
#         'tid': [12345678, 23456789, 34567890],
#         'toi': [100.01, 200.01, 300.01],
#         'pl_orbper': [3.5, 12.3, 365.25],
#         'pl_orbpererr1': [0.001, 0.002, 0.01],
#         'pl_trandep': [0.01, 0.02, 0.008],  # In fraction
#         'pl_trandeperr1': [0.001, 0.002, 0.0005],
#         'pl_trandurh': [2.5, 4.1, 13.0],  # In hours
#         'pl_trandurherr1': [0.1, 0.2, 0.5],
#         'pl_tranmid': [2458800.5, 2458900.3, 2459000.1],
#         'pl_rade': [1.2, 2.5, 1.0],
#         'pl_eqt': [500, 800, 288],
#         'pl_insol': [50, 200, 1.0],
#         'st_teff': [5800, 6200, 5778],
#         'st_logg': [4.5, 4.3, 4.4],
#         'st_rad': [1.1, 1.3, 1.0],
#         'st_tmag': [10.5, 9.8, 11.2],
#         'st_dist': [50, 100, 150],
#     })
    
#     # Process
#     mapper = ColumnMapper()
#     detected_type = mapper.detect_dataset_type(tess_toi_data)
#     print(f"\nDetected dataset type: {detected_type}")
    
#     df_mapped, mapping_info = mapper.map_columns(tess_toi_data)
#     print(f"\nMapping Info:")
#     print(f"  Mapped: {mapping_info['mapped_columns']} columns")
#     print(f"  Coverage: {mapping_info['coverage']}")
#     print(f"  Missing required: {mapping_info['required_missing']}")
    
#     df_final, conversion_info = mapper.validate_and_convert(df_mapped)
#     print(f"\nConversions:")
#     for conv in conversion_info['conversions_applied']:
#         print(f"  - {conv}")
    
#     print(f"\nFinal columns:")
#     print(df_final.columns.tolist())
#     print(f"\nSample data:")
#     print(df_final.head())
    
#     summary = mapper.get_mapping_summary(tess_toi_data, df_final)
#     print(f"\nSummary:")
#     print(f"  Status: {summary['status']}")
#     print(f"  Coverage: {summary['coverage_percent']}%")
#     print(f"  Recommendations:")
#     for rec in summary['recommendations']:
#         print(f"    - {rec}")

