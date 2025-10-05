import os
import pandas as pd
from flask import Blueprint, request, jsonify, current_app

data_bp = Blueprint('data', __name__)

@data_bp.route('/preview/<upload_id>', methods=['GET'])
def get_data_preview(upload_id):
    try:
        limit = int(request.args.get('limit', 10))
        
        data_path = os.path.join(
            current_app.config['TEMP_DATA_FOLDER'],
            f"{upload_id}_processed.csv"
        )
        
        if not os.path.exists(data_path):
            return jsonify({
                'success': False,
                'error': 'Upload not found or expired'
            }), 404
        
        df = pd.read_csv(data_path)
        
        preview_df = df.head(limit)
        preview_data = preview_df.to_dict('records')
        
        return jsonify({
            'success': True,
            'upload_id': upload_id,
            'total_rows': len(df),
            'preview_rows': len(preview_df),
            'columns': list(df.columns),
            'data': preview_data
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to retrieve data: {str(e)}'
        }), 500

@data_bp.route('/stats/<upload_id>', methods=['GET'])
def get_data_stats(upload_id):
    try:
        data_path = os.path.join(
            current_app.config['TEMP_DATA_FOLDER'],
            f"{upload_id}_processed.csv"
        )
        
        if not os.path.exists(data_path):
            return jsonify({
                'success': False,
                'error': 'Upload not found or expired'
            }), 404
        
        df = pd.read_csv(data_path)
        
        stats = df.describe().to_dict()
        
        return jsonify({
            'success': True,
            'upload_id': upload_id,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'statistics': stats
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to retrieve statistics: {str(e)}'
        }), 500

@data_bp.route('/delete/<upload_id>', methods=['DELETE'])
def delete_upload(upload_id):

    try:
        data_path = os.path.join(
            current_app.config['TEMP_DATA_FOLDER'],
            f"{upload_id}_processed.csv"
        )
        
        if os.path.exists(data_path):
            os.remove(data_path)
            
        return jsonify({
            'success': True,
            'message': 'Upload data deleted successfully'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to delete data: {str(e)}'
        }), 500

