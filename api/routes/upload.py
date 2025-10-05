import os
import uuid
import pandas as pd
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import sys

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
sys.path.append(PROJECT_ROOT)

from src.column_mapper import ColumnMapper
from src.feature_engineering import FeatureEngineer
from api.utils.validators import validate_csv_columns, validate_dataset_type
from api.utils.preprocessor import DataPreprocessor

upload_bp = Blueprint('upload', __name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@upload_bp.route('/csv', methods=['POST'])
def upload_csv():
    """
    Upload CSV file with exoplanet data
    
    Expected Form Data:
        - file: CSV file
        - dataset_type: 'tess' or 'kepler'
        
    Returns:
        - upload_id: Unique identifier for this upload
        - summary: Data preprocessing summary
        - preview: First few rows of processed data
    """
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Only CSV files are allowed.'
            }), 400
        
        # Get dataset type from form data
        dataset_type = request.form.get('dataset_type', 'tess').lower()
        
        # Validate dataset type
        validation_result = validate_dataset_type(dataset_type)
        if not validation_result['valid']:
            return jsonify({
                'success': False,
                'error': validation_result['error']
            }), 400
        
        # Generate unique upload ID
        upload_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{upload_id}_{filename}")
        file.save(upload_path)
        
        # Read CSV file
        try:
            df_raw = pd.read_csv(upload_path)
        except Exception as e:
            os.remove(upload_path)  # Clean up
            return jsonify({
                'success': False,
                'error': f'Failed to read CSV file: {str(e)}'
            }), 400
        
        # Validate CSV columns
        column_validation = validate_csv_columns(df_raw, dataset_type)
        if not column_validation['valid']:
            os.remove(upload_path)  # Clean up
            return jsonify({
                'success': False,
                'error': column_validation['error'],
                'missing_columns': column_validation.get('missing_columns', [])
            }), 400
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(dataset_type)
        
        # Preprocess data
        preprocessing_result = preprocessor.preprocess_csv(df_raw, upload_id)
        
        if not preprocessing_result['success']:
            os.remove(upload_path)  # Clean up
            return jsonify({
                'success': False,
                'error': preprocessing_result['error']
            }), 400
        
        # Save processed data temporarily
        temp_data_path = os.path.join(
            current_app.config['TEMP_DATA_FOLDER'], 
            f"{upload_id}_processed.csv"
        )
        preprocessing_result['processed_data'].to_csv(temp_data_path, index=False)
        
        # Get preview data (first 5 rows)
        preview_data = preprocessing_result['processed_data'].head(5).to_dict('records')
        
        # Clean up raw upload file
        os.remove(upload_path)
        
        # Return success response
        return jsonify({
            'success': True,
            'upload_id': upload_id,
            'filename': filename,
            'dataset_type': dataset_type,
            'summary': {
                'total_rows': preprocessing_result['total_rows'],
                'processed_rows': preprocessing_result['processed_rows'],
                'removed_rows': preprocessing_result['removed_rows'],
                'total_features': preprocessing_result['total_features'],
                'base_features': preprocessing_result['base_features'],
                'engineered_features': preprocessing_result['engineered_features'],
                'missing_values_filled': preprocessing_result.get('missing_values_filled', 0),
                'auto_calculated_fields': preprocessing_result.get('auto_calculated_fields', [])
            },
            'preview': preview_data,
            'column_names': list(preprocessing_result['processed_data'].columns),
            'message': 'Data uploaded and preprocessed successfully'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        }), 500

@upload_bp.route('/template/<dataset_type>', methods=['GET'])
def download_template(dataset_type):
    """
    Download CSV template for specified dataset type
    
    Parameters:
        dataset_type: 'tess' or 'kepler'
    """
    from flask import send_file
    import io
    
    # Validate dataset type
    if dataset_type not in ['tess', 'kepler']:
        return jsonify({
            'success': False,
            'error': 'Invalid dataset type. Must be "tess" or "kepler"'
        }), 400
    
    # Define template headers and example values
    if dataset_type == 'tess':
        headers = [
            'pl_orbper', 'pl_trandep', 'pl_trandurh', 'pl_tranmid',
            'pl_orbpererr1', 'pl_orbpererr2', 'pl_trandeperr1', 'pl_trandeperr2',
            'pl_trandurherr1', 'pl_trandurherr2',
            'pl_rade', 'pl_eqt', 'pl_insol',
            'st_teff', 'st_logg', 'st_rad'
        ]
        example_values = [
            '3.5', '0.01', '2.5', '2458800.5',
            '0.001', '0.001', '0.001', '0.001', '0.1', '0.1',
            '1.2', '500', '50',
            '5800', '4.5', '1.1'
        ]
    else:  # kepler
        headers = [
            'pl_orbper', 'pl_trandep', 'pl_trandurh', 'pl_tranmid',
            'pl_orbpererr1', 'pl_orbpererr2', 'pl_trandeperr1', 'pl_trandeperr2',
            'pl_trandurherr1', 'pl_trandurherr2',
            'pl_rade', 'pl_eqt', 'pl_insol',
            'st_teff', 'st_logg', 'st_rad',
            'koi_model_snr', 'koi_smass', 'koi_sma'
        ]
        example_values = [
            '3.5', '0.01', '2.5', '2458800.5',
            '0.001', '0.001', '0.001', '0.001', '0.1', '0.1',
            '1.2', '500', '50',
            '5800', '4.5', '1.1',
            '15.5', '1.0', '0.05'
        ]
    
    # Create CSV content
    csv_content = ','.join(headers) + '\n' + ','.join(example_values)
    
    # Create in-memory file
    mem_file = io.BytesIO()
    mem_file.write(csv_content.encode('utf-8'))
    mem_file.seek(0)
    
    return send_file(
        mem_file,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'exoplanet_{dataset_type}_template.csv'
    )

