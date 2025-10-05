/**
 * Exoplanet Feature Definitions
 * 
 * These feature definitions match the backend model requirements.
 * The model expects TESS/common format field names (pl_*, st_*).
 */

export interface FeatureDefinition {
  name: string;
  label: string;
  unit: string;
  tooltip: string;
  placeholder: string;
  required?: boolean;
  min?: number;
  max?: number;
}

/**
 * DATASET TYPE: Determines which fields are required
 * - TESS: 16 fields (3 auto-calculated: SNR, stellar mass, semi-major axis)
 * - KEPLER: 19 fields (includes SNR, stellar mass, semi-major axis directly)
 */
export type DatasetType = 'tess' | 'kepler';

/**
 * COMMON REQUIRED FEATURES (16 total)
 * These are needed for BOTH Kepler and TESS datasets
 * 
 * Base features breakdown:
 * - 4 transit parameters
 * - 6 error measurements  
 * - 3 planet properties
 * - 3 stellar properties
 * Total: 16 common features
 */
export const COMMON_REQUIRED_FEATURES: FeatureDefinition[] = [
  // TRANSIT PARAMETERS (4 features)
  {
    name: "pl_orbper",
    label: "Orbital Period",
    unit: "days",
    tooltip: "Time between transits (orbital period of the planet)",
    placeholder: "3.5",
    required: true,
    min: 0.1,
    max: 1000,
  },
  {
    name: "pl_trandep",
    label: "Transit Depth",
    unit: "fraction (0-1)",
    tooltip: "Decrease in star brightness during transit (NOT ppm)",
    placeholder: "0.01",
    required: true,
    min: 0.0001,
    max: 0.5,
  },
  {
    name: "pl_trandurh",
    label: "Transit Duration",
    unit: "hours",
    tooltip: "Length of the transit event from first to fourth contact",
    placeholder: "2.5",
    required: true,
    min: 0.1,
    max: 48,
  },
  {
    name: "pl_tranmid",
    label: "Transit Midpoint",
    unit: "BJD",
    tooltip: "Barycentric Julian Date of the transit center",
    placeholder: "2458800.5",
    required: true,
    min: 2450000,
    max: 2470000,
  },

  // ERROR MEASUREMENTS (6 features)
  {
    name: "pl_orbpererr1",
    label: "Period Error (Upper)",
    unit: "days",
    tooltip: "Upper uncertainty in orbital period measurement",
    placeholder: "0.001",
    required: true,
    min: 0,
  },
  {
    name: "pl_orbpererr2",
    label: "Period Error (Lower)",
    unit: "days",
    tooltip: "Lower uncertainty in orbital period measurement",
    placeholder: "0.001",
    required: true,
    min: 0,
  },
  {
    name: "pl_trandeperr1",
    label: "Depth Error (Upper)",
    unit: "fraction",
    tooltip: "Upper uncertainty in transit depth measurement",
    placeholder: "0.001",
    required: true,
    min: 0,
  },
  {
    name: "pl_trandeperr2",
    label: "Depth Error (Lower)",
    unit: "fraction",
    tooltip: "Lower uncertainty in transit depth measurement",
    placeholder: "0.001",
    required: true,
    min: 0,
  },
  {
    name: "pl_trandurherr1",
    label: "Duration Error (Upper)",
    unit: "hours",
    tooltip: "Upper uncertainty in transit duration measurement",
    placeholder: "0.1",
    required: true,
    min: 0,
  },
  {
    name: "pl_trandurherr2",
    label: "Duration Error (Lower)",
    unit: "hours",
    tooltip: "Lower uncertainty in transit duration measurement",
    placeholder: "0.1",
    required: true,
    min: 0,
  },

  // PLANET PROPERTIES (3 features)
  {
    name: "pl_rade",
    label: "Planet Radius",
    unit: "R⊕",
    tooltip: "Planet radius in Earth radii",
    placeholder: "1.2",
    required: true,
    min: 0.1,
    max: 30,
  },
  {
    name: "pl_eqt",
    label: "Equilibrium Temperature",
    unit: "K",
    tooltip: "Equilibrium temperature of the planet",
    placeholder: "500",
    required: true,
    min: 0,
    max: 5000,
  },
  {
    name: "pl_insol",
    label: "Insolation Flux",
    unit: "S⊕",
    tooltip: "Stellar flux received by the planet (in Earth flux units)",
    placeholder: "50",
    required: true,
    min: 0,
  },

  // STELLAR PROPERTIES (3 features)
  {
    name: "st_teff",
    label: "Stellar Temperature",
    unit: "K",
    tooltip: "Effective temperature of the host star",
    placeholder: "5800",
    required: true,
    min: 2500,
    max: 10000,
  },
  {
    name: "st_logg",
    label: "Stellar Surface Gravity",
    unit: "log₁₀(cm/s²)",
    tooltip: "Surface gravity of the star (log scale)",
    placeholder: "4.5",
    required: true,
    min: 3.0,
    max: 5.5,
  },
  {
    name: "st_rad",
    label: "Stellar Radius",
    unit: "R☉",
    tooltip: "Radius of the star in solar radii",
    placeholder: "1.1",
    required: true,
    min: 0.1,
    max: 10,
  },
];

/**
 * KEPLER-ONLY REQUIRED FEATURES (3 additional)
 * These fields are directly available in Kepler data but need to be
 * calculated for TESS data
 */
export const KEPLER_ONLY_FEATURES: FeatureDefinition[] = [
  {
    name: "koi_model_snr",
    label: "Transit Signal-to-Noise Ratio",
    unit: "ratio",
    tooltip: "Signal-to-noise ratio of the transit signal",
    placeholder: "15.5",
    required: true,
    min: 0,
  },
  {
    name: "koi_smass",
    label: "Stellar Mass",
    unit: "M☉ (Solar masses)",
    tooltip: "Mass of the host star in solar masses",
    placeholder: "1.0",
    required: true,
    min: 0.1,
    max: 10,
  },
  {
    name: "koi_sma",
    label: "Semi-major Axis",
    unit: "AU",
    tooltip: "Semi-major axis of the planet's orbit",
    placeholder: "0.05",
    required: true,
    min: 0,
  },
];

/**
 * Get required features based on dataset type
 */
export function getRequiredFeatures(datasetType: DatasetType): FeatureDefinition[] {
  if (datasetType === 'kepler') {
    return [...COMMON_REQUIRED_FEATURES, ...KEPLER_ONLY_FEATURES];
  }
  return COMMON_REQUIRED_FEATURES;
}

/**
 * Features that are automatically calculated by the backend (for TESS only)
 */
export const AUTO_CALCULATED_FEATURES = [
  "koi_model_snr",   // Signal-to-noise ratio
  "koi_smass",       // Stellar mass
  "koi_sma",         // Semi-major axis
];

/**
 * NO OPTIONAL FEATURES - All features are required based on dataset type
 */
export const OPTIONAL_FEATURES: FeatureDefinition[] = [];

/**
 * CSV column headers based on dataset type
 */
export function getCSVHeaders(datasetType: DatasetType): string[] {
  return getRequiredFeatures(datasetType).map(f => f.name);
}

/**
 * CSV column headers (TESS - 18 fields)
 */
export const CSV_TESS_HEADERS = COMMON_REQUIRED_FEATURES.map(f => f.name);

/**
 * CSV column headers (KEPLER - 21 fields)
 */
export const CSV_KEPLER_HEADERS = [
  ...COMMON_REQUIRED_FEATURES.map(f => f.name),
  ...KEPLER_ONLY_FEATURES.map(f => f.name),
];

/**
 * Generate a CSV template string based on dataset type
 */
export function generateCSVTemplate(datasetType: DatasetType): string {
  const headers = getCSVHeaders(datasetType);
  
  const tessExampleValues = [
    '3.5', '0.01', '2.5', '2458800.5',                    // Transit params (4)
    '0.001', '0.001', '0.001', '0.001', '0.1', '0.1',    // Errors (6)
    '1.2', '500', '50',                                   // Planet props (3)
    '5800', '4.5', '1.1',                                // Stellar props (3)
  ]; // 16 values
  
  const keplerExampleValues = [
    ...tessExampleValues,                                 // All TESS fields (16)
    '15.5', '1.0', '0.05',                               // Kepler-only: SNR, Mass, SMA (3)
  ]; // 19 values
  
  const exampleValues = datasetType === 'kepler' ? keplerExampleValues : tessExampleValues;
  
  return headers.join(',') + '\n' + exampleValues.join(',');
}

/**
 * Validate a single feature value
 */
export function validateFeature(
  feature: FeatureDefinition, 
  value: string | number
): { valid: boolean; error?: string } {
  const numValue = typeof value === 'string' ? parseFloat(value) : value;
  
  if (isNaN(numValue)) {
    return { valid: false, error: `${feature.label} must be a valid number` };
  }
  
  if (feature.min !== undefined && numValue < feature.min) {
    return { valid: false, error: `${feature.label} must be at least ${feature.min}` };
  }
  
  if (feature.max !== undefined && numValue > feature.max) {
    return { valid: false, error: `${feature.label} must be at most ${feature.max}` };
  }
  
  return { valid: true };
}

/**
 * Validate all required features are present based on dataset type
 */
export function validateRequiredFeatures(
  data: Record<string, any>,
  datasetType: DatasetType
): { valid: boolean; missingFields?: string[] } {
  const requiredFeatures = getRequiredFeatures(datasetType);
  const missingFields = requiredFeatures
    .filter(feature => !(feature.name in data) || data[feature.name] === null || data[feature.name] === '')
    .map(feature => feature.label);
  
  return {
    valid: missingFields.length === 0,
    missingFields: missingFields.length > 0 ? missingFields : undefined,
  };
}

