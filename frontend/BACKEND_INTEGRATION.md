# Backend Integration Guide

This document outlines all the backend integration points required for the Exoplanet WebApp.

## Overview

All backend API placeholders are located in `src/lib/api-placeholders.ts`. These functions should be replaced with actual API implementations that connect to your Python backend.

## Required Backend Endpoints

### 1. Data Upload

#### CSV Upload
- **Endpoint**: `POST /api/upload/csv`
- **Function**: `uploadCSVData(file: File)`
- **Description**: Upload CSV file with exoplanet parameters
- **Expected Response**:
  ```json
  {
    "success": true,
    "data": {
      "recordCount": 100,
      "objectId": "sample-id"
    }
  }
  ```

#### Manual Parameter Entry
- **Endpoint**: `POST /api/upload/manual`
- **Function**: `submitManualParameters(parameters: Record<string, number>)`
- **Description**: Submit manually entered parameters
- **Request Body**:
  ```json
  {
    "koi_period": 10.5,
    "koi_duration": 3.2,
    "koi_depth": 250.0,
    "koi_prad": 1.5,
    "koi_model_snr": 15.8,
    "koi_num_transits": 50,
    "koi_srad": 1.1,
    "koi_steff": 5778,
    "koi_slogg": 4.5,
    "koi_impact": 0.3
  }
  ```

### 2. Dataset Management

#### Upload Fine-Tuning Dataset
- **Endpoint**: `POST /api/datasets/upload`
- **Function**: `uploadFineTuneDataset(file: File)`
- **Description**: Upload custom dataset for model fine-tuning
- **Expected Response**:
  ```json
  {
    "success": true,
    "datasetId": "dataset-123",
    "samples": 5000
  }
  ```

#### List Datasets
- **Endpoint**: `GET /api/datasets`
- **Function**: `listDatasets()`
- **Description**: Retrieve all user-uploaded datasets

#### Delete Dataset
- **Endpoint**: `DELETE /api/datasets/:id`
- **Function**: `deleteDataset(datasetId: string)`
- **Description**: Remove a dataset from storage

### 3. Prediction

#### Run Classification
- **Endpoint**: `POST /api/predict`
- **Function**: `runPrediction(modelType: 'xgboost' | 'dnn', inputData: any)`
- **Description**: Classify exoplanet candidate using selected model
- **Request Body**:
  ```json
  {
    "modelType": "xgboost",
    "objectId": "sample-id"
  }
  ```
- **Expected Response**:
  ```json
  {
    "success": true,
    "prediction": {
      "class": "CONFIRMED",
      "probability": 0.89,
      "confidence": "HIGH"
    }
  }
  ```

### 4. Visualization

#### Light Curve
- **Endpoint**: `GET /api/visualize/lightcurve?objectId={id}`
- **Function**: `renderLightCurve(objectId: string)`
- **Description**: Generate folded light curve data

#### Feature Importance
- **Endpoint**: `GET /api/visualize/feature-importance?model={type}`
- **Function**: `getFeatureImportance(modelType: string)`
- **Description**: Get feature importance values for model interpretation

#### Model Fit
- **Endpoint**: `GET /api/visualize/model-fit?objectId={id}`
- **Function**: `renderModelFit(objectId: string)`
- **Description**: Compare observed vs. model transit

#### Correlation Matrix
- **Endpoint**: `GET /api/visualize/correlation`
- **Function**: `renderCorrelationMatrix()`
- **Description**: Feature correlation heatmap data

#### Feature Distributions
- **Endpoint**: `GET /api/visualize/distributions`
- **Function**: `renderDistributions()`
- **Description**: Statistical distributions of input features

#### Confusion Matrix
- **Endpoint**: `GET /api/visualize/confusion-matrix?model={type}`
- **Function**: `renderConfusionMatrix(modelType: string)`
- **Description**: Model performance confusion matrix

#### Diagnostic Plots
- **Endpoint**: `GET /api/visualize/diagnostics/{type}?objectId={id}`
- **Function**: `renderDiagnostics(diagnosticType: string, objectId: string)`
- **Description**: Various diagnostic plots (centroid offset, odd-even depth, secondary eclipse, MES)
- **Diagnostic Types**:
  - `centroid-offset`
  - `odd-even-depth`
  - `secondary-eclipse`
  - `mes-distribution`

### 5. Model Fine-Tuning

#### Start Training
- **Endpoint**: `POST /api/fine-tune/start`
- **Function**: `startFineTuning(modelType, hyperparameters, datasetId)`
- **Description**: Initiate model fine-tuning with custom hyperparameters
- **Request Body**:
  ```json
  {
    "modelType": "xgboost",
    "hyperparameters": {
      "learningRate": 0.1,
      "numTrees": 100,
      "maxDepth": 6
    },
    "datasetId": "dataset-123"
  }
  ```
- **Expected Response**:
  ```json
  {
    "success": true,
    "jobId": "training-job-123"
  }
  ```

#### Training Status
- **Endpoint**: `GET /api/fine-tune/status/:jobId`
- **Function**: `getTrainingStatus(jobId: string)`
- **Description**: Monitor training progress
- **Expected Response**:
  ```json
  {
    "success": true,
    "status": "running",
    "progress": 45,
    "metrics": {
      "currentLoss": 0.23,
      "currentAccuracy": 0.87
    }
  }
  ```

#### Model Metrics
- **Endpoint**: `GET /api/models/:modelId/metrics`
- **Function**: `getModelMetrics(modelId: string)`
- **Description**: Retrieve performance metrics for trained model
- **Expected Response**:
  ```json
  {
    "success": true,
    "metrics": {
      "accuracy": 0.942,
      "precision": 0.918,
      "recall": 0.935,
      "f1Score": 0.926
    }
  }
  ```

## Hyperparameter Specifications

### XGBoost Model
- `learningRate`: 0.01 - 0.3 (default: 0.1)
- `numTrees`: 50 - 500 (default: 100)
- `maxDepth`: 3 - 10 (default: 6)

### Deep Neural Network
- `numLayers`: 2 - 8 (default: 4)
- `neuronsPerLayer`: 32 - 256 (default: 128)
- `activationFunction`: 'relu' | 'tanh' | 'sigmoid' (default: 'relu')

## Feature List

Required input features for classification:
- `koi_period` - Orbital period (days)
- `koi_duration` - Transit duration (hours)
- `koi_depth` - Transit depth (ppm)
- `koi_prad` - Planetary radius (Earth radii)
- `koi_model_snr` - Transit SNR
- `koi_num_transits` - Number of transits
- `koi_srad` - Stellar radius (Solar radii)
- `koi_steff` - Stellar temperature (K)
- `koi_slogg` - Stellar log g
- `koi_impact` - Impact parameter

Optional diagnostic features:
- Odd-even depth ratio
- Secondary eclipse depth
- Centroid offsets
- Multiple Event Statistic (MES)

## Error Handling

All backend responses should include proper error handling:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {}
  }
}
```

## Notes for Backend Developers

1. All API endpoints should support CORS for frontend integration
2. Implement proper authentication if required
3. Use appropriate HTTP status codes
4. Provide detailed error messages for debugging
5. Consider implementing rate limiting for resource-intensive operations
6. Support streaming responses for long-running training jobs
7. Implement proper file upload validation (size limits, format checks)
8. Store datasets securely with user isolation
9. Cache visualization data when possible for better performance
10. Log all API calls for monitoring and debugging
