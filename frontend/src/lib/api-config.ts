/**
 * API Configuration
 * 
 * Uses environment variable VITE_API_URL if available,
 * otherwise falls back to localhost for development or relative path for production
 */

// Detect environment and set appropriate API URL
const isDevelopment = import.meta.env.DEV;
const envApiUrl = import.meta.env.VITE_API_URL;

// Get API URL from environment or use sensible defaults
export const API_BASE_URL = envApiUrl || (isDevelopment ? 'http://localhost:5000/api' : '/api');

// For direct API calls (without /api prefix)
export const API_ROOT = import.meta.env.VITE_API_URL?.replace('/api', '') || '';

// Full API endpoints
export const API_ENDPOINTS = {
  // Upload
  uploadCSV: `${API_BASE_URL}/upload/csv`,
  uploadTemplate: `${API_BASE_URL}/upload/template`,
  
  // Data
  dataPreview: (uploadId: string) => `${API_BASE_URL}/data/preview/${uploadId}`,
  dataStats: (uploadId: string) => `${API_BASE_URL}/data/stats/${uploadId}`,
  dataDelete: (uploadId: string) => `${API_BASE_URL}/data/delete/${uploadId}`,
  
  // Predict
  predictSingle: `${API_BASE_URL}/predict/single`,
  predictBatch: `${API_BASE_URL}/predict/batch`,
  predictExport: (uploadId: string) => `${API_BASE_URL}/predict/export/${uploadId}`,
  predictDetails: (uploadId: string) => `${API_BASE_URL}/predict/details/${uploadId}`,
  
  // Train
  trainUploadLabeled: `${API_BASE_URL}/train/upload_labeled`,
  trainFinetune: `${API_BASE_URL}/train/finetune`,
  trainModelsList: `${API_BASE_URL}/train/models/list`,
  
  // Health
  health: `${API_BASE_URL}/health`,
};

export default API_BASE_URL;

