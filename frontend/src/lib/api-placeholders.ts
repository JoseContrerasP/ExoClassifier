
// API Base URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

export const uploadCSVData = async (file: File, datasetType: 'tess' | 'kepler' = 'tess') => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('dataset_type', datasetType);

    const response = await fetch(`${API_BASE_URL}/upload/csv`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Upload failed');
    }

    const result = await response.json();
    return result;
  } catch (error: any) {
    console.error('[API Error] uploadCSVData:', error);
    throw error;
  }
};

export const downloadCSVTemplate = async (datasetType: 'tess' | 'kepler' = 'tess') => {
  try {
    const response = await fetch(`${API_BASE_URL}/upload/template/${datasetType}`);
    
    if (!response.ok) {
      throw new Error('Failed to download template');
    }

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `exoplanet_${datasetType}_template.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  } catch (error: any) {
    console.error('[API Error] downloadCSVTemplate:', error);
    throw error;
  }
};


export const getDataPreview = async (uploadId: string, limit: number = 10) => {
  try {
    const response = await fetch(`${API_BASE_URL}/data/preview/${uploadId}?limit=${limit}`);
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to get preview');
    }

    const result = await response.json();
    return result;
  } catch (error: any) {
    console.error('[API Error] getDataPreview:', error);
    throw error;
  }
};


export const getDataStats = async (uploadId: string) => {
  try {
    const response = await fetch(`${API_BASE_URL}/data/stats/${uploadId}`);
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to get statistics');
    }

    const result = await response.json();
    return result;
  } catch (error: any) {
    console.error('[API Error] getDataStats:', error);
    throw error;
  }
};


export const deleteUpload = async (uploadId: string) => {
  try {
    const response = await fetch(`${API_BASE_URL}/data/delete/${uploadId}`, {
      method: 'DELETE',
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to delete upload');
    }

    const result = await response.json();
    return result;
  } catch (error: any) {
    console.error('[API Error] deleteUpload:', error);
    throw error;
  }
};


export const submitManualParameters = async (
  features: Record<string, number>, 
  datasetType: 'tess' | 'kepler' = 'tess'
) => {
  try {
    const response = await fetch(`${API_BASE_URL}/predict/single`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        dataset_type: datasetType,
        features: features,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      const errorObj = new Error(error.error || 'Prediction failed');
      // Attach validation errors if present
      if (error.validation_errors) {
        (errorObj as any).validation_errors = error.validation_errors;
      }
      throw errorObj;
    }

    const result = await response.json();
    return result;
  } catch (error: any) {
    console.error('[API Error] submitManualParameters:', error);
    throw error;
  }
};


export const predictBatch = async (uploadId: string, modelType: string = 'ensemble') => {
  try {
    const response = await fetch(`${API_BASE_URL}/predict/batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        upload_id: uploadId,
        model_type: modelType,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Batch prediction failed');
    }

    const result = await response.json();
    return result.predictions;
  } catch (error: any) {
    console.error('[API Error] predictBatch:', error);
    throw error;
  }
};

// ============= Dataset Management =============

/**
 * Upload custom dataset for fine-tuning
 * Backend integration point: POST /api/datasets/upload
 */
export const uploadFineTuneDataset = async (file: File) => {
  // TODO: Implement backend API call
  console.log('[API Placeholder] uploadFineTuneDataset:', file.name);
  return { success: true, datasetId: 'dataset-123' };
};

/**
 * Delete a dataset
 * Backend integration point: DELETE /api/datasets/:id
 */
export const deleteDataset = async (datasetId: string) => {
  // TODO: Implement backend API call
  console.log('[API Placeholder] deleteDataset:', datasetId);
  return { success: true };
};

/**
 * List all user datasets
 * Backend integration point: GET /api/datasets
 */
export const listDatasets = async () => {
  // TODO: Implement backend API call
  console.log('[API Placeholder] listDatasets');
  return { success: true, datasets: [] };
};

// ============= Prediction =============

/**
 * Run prediction using selected model
 * Backend integration point: POST /api/predict
 */
export const runPrediction = async (modelType: 'xgboost' | 'dnn', inputData: any) => {
  // TODO: Implement backend API call
  console.log('[API Placeholder] runPrediction:', modelType, inputData);
  return {
    success: true,
    prediction: {
      class: 'CONFIRMED',
      probability: 0.89,
      confidence: 'HIGH'
    }
  };
};

// ============= Visualization =============

/**
 * Render folded light curve
 * Backend integration point: GET /api/visualize/lightcurve
 */
export const renderLightCurve = async (objectId: string) => {
  // TODO: Implement backend API call
  console.log('[API Placeholder] renderLightCurve:', objectId);
  return { success: true, chartData: [] };
};

/**
 * Get feature importance values
 * Backend integration point: GET /api/visualize/feature-importance
 */
export const getFeatureImportance = async (modelType: string) => {
  // TODO: Implement backend API call
  console.log('[API Placeholder] getFeatureImportance:', modelType);
  return { success: true, features: [] };
};

/**
 * Render model fit comparison
 * Backend integration point: GET /api/visualize/model-fit
 */
export const renderModelFit = async (objectId: string) => {
  // TODO: Implement backend API call
  console.log('[API Placeholder] renderModelFit:', objectId);
  return { success: true, chartData: [] };
};

/**
 * Render correlation matrix
 * Backend integration point: GET /api/visualize/correlation
 */
export const renderCorrelationMatrix = async () => {
  // TODO: Implement backend API call
  console.log('[API Placeholder] renderCorrelationMatrix');
  return { success: true, matrixData: [] };
};

/**
 * Render feature distributions
 * Backend integration point: GET /api/visualize/distributions
 */
export const renderDistributions = async () => {
  // TODO: Implement backend API call
  console.log('[API Placeholder] renderDistributions');
  return { success: true, distributionData: [] };
};

/**
 * Render confusion matrix
 * Backend integration point: GET /api/visualize/confusion-matrix
 */
export const renderConfusionMatrix = async (modelType: string) => {
  // TODO: Implement backend API call
  console.log('[API Placeholder] renderConfusionMatrix:', modelType);
  return { success: true, matrixData: [] };
};

/**
 * Render diagnostic plots (centroid offset, odd-even depth, etc.)
 * Backend integration point: GET /api/visualize/diagnostics/:type
 */
export const renderDiagnostics = async (diagnosticType: string, objectId: string) => {
  // TODO: Implement backend API call
  console.log('[API Placeholder] renderDiagnostics:', diagnosticType, objectId);
  return { success: true, chartData: [] };
};

// ============= Model Fine-Tuning =============

/**
 * Start fine-tuning with custom hyperparameters
 * Backend integration point: POST /api/fine-tune/start
 */
export const startFineTuning = async (
  modelType: 'xgboost' | 'dnn',
  hyperparameters: Record<string, any>,
  datasetId: string
) => {
  // TODO: Implement backend API call
  console.log('[API Placeholder] startFineTuning:', { modelType, hyperparameters, datasetId });
  return { success: true, jobId: 'training-job-123' };
};

/**
 * Check training job status
 * Backend integration point: GET /api/fine-tune/status/:jobId
 */
export const getTrainingStatus = async (jobId: string) => {
  // TODO: Implement backend API call
  console.log('[API Placeholder] getTrainingStatus:', jobId);
  return {
    success: true,
    status: 'running',
    progress: 45,
    metrics: {}
  };
};

/**
 * Get model performance metrics
 * Backend integration point: GET /api/models/:modelId/metrics
 */
export const getModelMetrics = async (modelId: string) => {
  // TODO: Implement backend API call
  console.log('[API Placeholder] getModelMetrics:', modelId);
  return {
    success: true,
    metrics: {
      accuracy: 0.942,
      precision: 0.918,
      recall: 0.935,
      f1Score: 0.926
    }
  };
};
