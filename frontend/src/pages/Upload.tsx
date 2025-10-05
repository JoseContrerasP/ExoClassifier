import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Upload as UploadIcon, FileText, AlertCircle, Download, Satellite, Telescope } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";
import { 
  COMMON_REQUIRED_FEATURES, 
  KEPLER_ONLY_FEATURES,
  getRequiredFeatures,
  generateCSVTemplate, 
  getCSVHeaders,
  type DatasetType 
} from "@/lib/exoplanet-features";
import { uploadCSVData, submitManualParameters } from "@/lib/api-placeholders";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";

const Upload = () => {
  const { toast } = useToast();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [datasetType, setDatasetType] = useState<DatasetType>('tess');
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<any>(null);
  const [predicting, setPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [showPredictionDialog, setShowPredictionDialog] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setUploadResult(null); // Clear previous results
    }
  };

  const handleFileUpload = async () => {
    if (!selectedFile) return;
    
    setUploading(true);
    try {
      const result = await uploadCSVData(selectedFile, datasetType);
      
      // Save to localStorage for Datasets page
      const uploadedDataset = {
        upload_id: result.upload_id,
        filename: result.filename,
        dataset_type: result.dataset_type,
        upload_date: new Date().toISOString(),
        summary: result.summary
      };
      
      const existing = localStorage.getItem('uploaded_datasets');
      const datasets = existing ? JSON.parse(existing) : [];
      datasets.unshift(uploadedDataset); // Add to beginning
      localStorage.setItem('uploaded_datasets', JSON.stringify(datasets));
      
      setUploadResult(result);
      toast({
        title: "✅ Upload successful",
        description: `Processed ${result.summary.processed_rows} rows with ${result.summary.total_features} features`,
      });
    } catch (error: any) {
      toast({
        title: "❌ Upload failed",
        description: error.message || 'Failed to upload file',
        variant: "destructive",
      });
    } finally {
      setUploading(false);
    }
  };

  const handleManualSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setPredicting(true);
    
    try {
      // Extract form data
      const formData = new FormData(e.target as HTMLFormElement);
      const features: Record<string, number> = {};
      
      formData.forEach((value, key) => {
        if (value) {
          features[key] = parseFloat(value as string);
        }
      });
      
      // Submit for prediction
      const result = await submitManualParameters(features, datasetType);
      
      setPredictionResult(result);
      setShowPredictionDialog(true);
      
      toast({
        title: "✅ Prediction complete",
        description: `Classification: ${result.prediction.classification}`,
      });
    } catch (error: any) {
      // Check if it's a validation error with details
      const errorMessage = error.validation_errors 
        ? error.validation_errors.join('\n') 
        : error.message || "Could not process prediction";
      
      toast({
        title: "❌ Prediction failed",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setPredicting(false);
    }
  };

  const downloadCSVTemplate = () => {
    const csvContent = generateCSVTemplate(datasetType);
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `exoplanet_${datasetType}_template.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
    
    const fieldCount = datasetType === 'kepler' ? 21 : 18;
    toast({
      title: "Template downloaded",
      description: `${datasetType.toUpperCase()} template with ${fieldCount} fields saved successfully.`,
    });
  };

  const requiredFeatures = getRequiredFeatures(datasetType);
  const fieldCount = requiredFeatures.length;

  return (
    <div className="container mx-auto px-4 py-8 max-w-5xl">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Data Upload & Input</h1>
        <p className="text-muted-foreground">
          Upload CSV files or manually enter exoplanet parameters for classification
        </p>
      </div>

      {/* DATASET TYPE SELECTOR */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Telescope className="h-5 w-5" />
            Dataset Type
          </CardTitle>
          <CardDescription>
            Select the source of your exoplanet data to determine required fields
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <Button
              variant={datasetType === 'tess' ? 'default' : 'outline'}
              onClick={() => setDatasetType('tess')}
              className="h-auto py-4 flex flex-col items-center gap-2"
            >
              <Satellite className="h-8 w-8" />
              <div>
                <div className="font-bold">TESS</div>
                <div className="text-xs font-normal opacity-80">16 fields required</div>
                <div className="text-xs font-normal opacity-80">3 auto-calculated</div>
              </div>
            </Button>
            <Button
              variant={datasetType === 'kepler' ? 'default' : 'outline'}
              onClick={() => setDatasetType('kepler')}
              className="h-auto py-4 flex flex-col items-center gap-2"
            >
              <Telescope className="h-8 w-8" />
              <div>
                <div className="font-bold">Kepler</div>
                <div className="text-xs font-normal opacity-80">19 fields required</div>
                <div className="text-xs font-normal opacity-80">All fields provided</div>
              </div>
            </Button>
          </div>
          
          <div className="mt-4 p-3 bg-muted rounded-lg text-sm">
            <p className="font-medium mb-1">
              {datasetType === 'tess' ? '🛰️ TESS Dataset' : '🔭 Kepler Dataset'}
            </p>
            <p className="text-muted-foreground">
              {datasetType === 'tess' 
                ? 'SNR, Stellar Mass, and Semi-major Axis will be automatically calculated from your input.' 
                : 'All fields including SNR, Stellar Mass, and Semi-major Axis must be provided.'}
            </p>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="csv" className="space-y-6">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="csv">CSV Upload</TabsTrigger>
          <TabsTrigger value="manual">Manual Entry</TabsTrigger>
        </TabsList>

        <TabsContent value="csv">
          <Card>
            <CardHeader>
              <CardTitle>Upload CSV File</CardTitle>
              <CardDescription>
                Upload a CSV file containing exoplanet transit parameters. The file should include 
                columns for orbital period, transit duration, depth, and other key features.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* DOWNLOAD TEMPLATE BUTTON */}
              <div className="flex justify-end">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={downloadCSVTemplate}
                  className="gap-2"
                >
                  <Download className="h-4 w-4" />
                  Download Template
                </Button>
              </div>

              <div className="border-2 border-dashed border-border rounded-lg p-12 text-center transition-smooth hover:border-primary">
                <UploadIcon className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                <Label htmlFor="file-upload" className="cursor-pointer">
                  <span className="text-primary font-medium">Click to upload</span> or drag and drop
                </Label>
                <Input
                  id="file-upload"
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={handleFileChange}
                />
                {selectedFile && (
                  <div className="mt-4 flex items-center justify-center gap-2 text-sm">
                    <FileText className="h-4 w-4" />
                    <span>{selectedFile.name}</span>
                  </div>
                )}
              </div>

              <div className="bg-muted/50 rounded-lg p-4">
                <h3 className="font-medium mb-2 flex items-center gap-2">
                  <AlertCircle className="h-4 w-4" />
                  Required CSV Columns ({fieldCount} total)
                </h3>
                <div className="text-sm text-muted-foreground space-y-2">
                  <div>
                    <strong>Common fields (16):</strong>
                    <ul className="ml-6 list-disc mt-1">
                      <li>Transit: <code>pl_orbper, pl_trandep, pl_trandurh, pl_tranmid</code> (4)</li>
                      <li>Errors: <code>pl_orbpererr1/2, pl_trandeperr1/2, pl_trandurherr1/2</code> (6)</li>
                      <li>Planet: <code>pl_rade, pl_eqt, pl_insol</code> (3)</li>
                      <li>Stellar: <code>st_teff, st_logg, st_rad</code> (3)</li>
                    </ul>
                  </div>
                  {datasetType === 'kepler' && (
                    <div>
                      <strong>Kepler-only fields (+3):</strong>
                      <ul className="ml-6 list-disc mt-1">
                        <li><code>koi_model_snr, koi_smass, koi_sma</code></li>
                      </ul>
                    </div>
                  )}
                  {datasetType === 'tess' && (
                    <div className="text-xs text-blue-600 dark:text-blue-400 mt-2">
                      <strong>✓ Auto-calculated for TESS:</strong> SNR, Stellar Mass, Semi-major Axis
                    </div>
                  )}
                </div>
              </div>

              <Button 
                onClick={handleFileUpload} 
                disabled={!selectedFile || uploading}
                className="w-full"
              >
                {uploading ? 'Processing...' : 'Upload and Process'}
              </Button>

              {/* Upload Success Message */}
              {uploadResult && (
                <div className="mt-4 p-4 bg-green-50 dark:bg-green-950 border border-green-200 dark:border-green-800 rounded-lg">
                  <h3 className="font-semibold text-green-900 dark:text-green-100 mb-2">
                    ✅ Upload Successful!
                  </h3>
                  <div className="text-sm text-green-800 dark:text-green-200 space-y-1">
                    <p><strong>File:</strong> {uploadResult.filename}</p>
                    <p><strong>Rows Processed:</strong> {uploadResult.summary.processed_rows} / {uploadResult.summary.total_rows}</p>
                    <p><strong>Features:</strong> {uploadResult.summary.total_features} total ({uploadResult.summary.base_features} base + {uploadResult.summary.engineered_features} engineered)</p>
                    {uploadResult.summary.auto_calculated_fields?.length > 0 && (
                      <p><strong>Auto-calculated:</strong> {uploadResult.summary.auto_calculated_fields.join(', ')}</p>
                    )}
                  </div>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    className="mt-3"
                    onClick={() => window.location.href = '/datasets'}
                  >
                    View in Datasets →
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="manual">
          <Card>
            <CardHeader>
              <CardTitle>Manual Parameter Entry</CardTitle>
              <CardDescription>
                Enter exoplanet transit parameters manually for individual classification
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleManualSubmit} className="space-y-6">
                <TooltipProvider>
                  {/* TRANSIT PARAMETERS (4 fields) */}
                  <div>
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                      Transit Parameters (4 fields)
                      <span className="text-xs font-normal text-destructive">* All Required</span>
                    </h3>
                    <div className="grid md:grid-cols-2 gap-4">
                      {COMMON_REQUIRED_FEATURES.slice(0, 4).map((field) => (
                        <div key={field.name} className="space-y-2">
                          <div className="flex items-center gap-2">
                            <Label htmlFor={field.name}>
                              {field.label} <span className="text-destructive">*</span>
                            </Label>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <AlertCircle className="h-3 w-3 text-muted-foreground cursor-help" />
                              </TooltipTrigger>
                              <TooltipContent>
                                <p className="max-w-xs">{field.tooltip}</p>
                              </TooltipContent>
                            </Tooltip>
                          </div>
                          <Input
                            id={field.name}
                            name={field.name}
                            type="number"
                            step="any"
                            placeholder={field.placeholder}
                            required={field.required}
                            min={field.min}
                            max={field.max}
                          />
                          <p className="text-xs text-muted-foreground">{field.unit}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* ERROR MEASUREMENTS (6 fields) */}
                  <div>
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                      Error Measurements (6 fields)
                      <span className="text-xs font-normal text-destructive">* All Required</span>
                    </h3>
                    <div className="grid md:grid-cols-2 gap-4">
                      {COMMON_REQUIRED_FEATURES.slice(4, 10).map((field) => (
                        <div key={field.name} className="space-y-2">
                          <div className="flex items-center gap-2">
                            <Label htmlFor={field.name}>
                              {field.label} <span className="text-destructive">*</span>
                            </Label>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <AlertCircle className="h-3 w-3 text-muted-foreground cursor-help" />
                              </TooltipTrigger>
                              <TooltipContent>
                                <p className="max-w-xs">{field.tooltip}</p>
                              </TooltipContent>
                            </Tooltip>
                          </div>
                          <Input
                            id={field.name}
                            name={field.name}
                            type="number"
                            step="any"
                            placeholder={field.placeholder}
                            required={field.required}
                            min={field.min}
                            max={field.max}
                          />
                          <p className="text-xs text-muted-foreground">{field.unit}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* PLANET PROPERTIES (3 fields) */}
                  <div>
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                      Planet Properties (3 fields)
                      <span className="text-xs font-normal text-destructive">* All Required</span>
                    </h3>
                    <div className="grid md:grid-cols-2 gap-4">
                      {COMMON_REQUIRED_FEATURES.slice(10, 13).map((field) => (
                        <div key={field.name} className="space-y-2">
                          <div className="flex items-center gap-2">
                            <Label htmlFor={field.name}>
                              {field.label} <span className="text-destructive">*</span>
                            </Label>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <AlertCircle className="h-3 w-3 text-muted-foreground cursor-help" />
                              </TooltipTrigger>
                              <TooltipContent>
                                <p className="max-w-xs">{field.tooltip}</p>
                              </TooltipContent>
                            </Tooltip>
                          </div>
                          <Input
                            id={field.name}
                            name={field.name}
                            type="number"
                            step="any"
                            placeholder={field.placeholder}
                            required={field.required}
                            min={field.min}
                            max={field.max}
                          />
                          <p className="text-xs text-muted-foreground">{field.unit}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* STELLAR PROPERTIES (3 fields) */}
                  <div>
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                      Stellar Properties (3 fields)
                      <span className="text-xs font-normal text-destructive">* All Required</span>
                    </h3>
                    <div className="grid md:grid-cols-2 gap-4">
                      {COMMON_REQUIRED_FEATURES.slice(13, 16).map((field) => (
                        <div key={field.name} className="space-y-2">
                          <div className="flex items-center gap-2">
                            <Label htmlFor={field.name}>
                              {field.label} <span className="text-destructive">*</span>
                            </Label>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <AlertCircle className="h-3 w-3 text-muted-foreground cursor-help" />
                              </TooltipTrigger>
                              <TooltipContent>
                                <p className="max-w-xs">{field.tooltip}</p>
                              </TooltipContent>
                            </Tooltip>
                          </div>
                          <Input
                            id={field.name}
                            name={field.name}
                            type="number"
                            step="any"
                            placeholder={field.placeholder}
                            required={field.required}
                            min={field.min}
                            max={field.max}
                          />
                          <p className="text-xs text-muted-foreground">{field.unit}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* KEPLER-ONLY FIELDS (3 fields) - Only shown for Kepler */}
                  {datasetType === 'kepler' && (
                    <div>
                      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        Kepler-Only Fields (3 fields)
                        <Badge variant="default">Kepler</Badge>
                        <span className="text-xs font-normal text-destructive">* Required for Kepler</span>
                      </h3>
                      <div className="grid md:grid-cols-2 gap-4">
                        {KEPLER_ONLY_FEATURES.map((field) => (
                          <div key={field.name} className="space-y-2">
                            <div className="flex items-center gap-2">
                              <Label htmlFor={field.name}>
                                {field.label} <span className="text-destructive">*</span>
                              </Label>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <AlertCircle className="h-3 w-3 text-muted-foreground cursor-help" />
                                </TooltipTrigger>
                                <TooltipContent>
                                  <p className="max-w-xs">{field.tooltip}</p>
                                </TooltipContent>
                              </Tooltip>
                            </div>
                            <Input
                              id={field.name}
                              name={field.name}
                              type="number"
                              step="any"
                              placeholder={field.placeholder}
                              required={field.required}
                            />
                            <p className="text-xs text-muted-foreground">{field.unit}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* SUMMARY NOTICE */}
                  <div className={`${datasetType === 'tess' ? 'bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800' : 'bg-purple-50 dark:bg-purple-950 border-purple-200 dark:border-purple-800'} border rounded-lg p-4`}>
                    <p className={`text-sm ${datasetType === 'tess' ? 'text-blue-900 dark:text-blue-100' : 'text-purple-900 dark:text-purple-100'}`}>
                      <strong>✓ Total: {fieldCount} required fields</strong>
                      <br />
                      {datasetType === 'tess' ? (
                        <>
                          <strong>🛰️ TESS Mode:</strong> SNR, Stellar Mass, and Semi-major Axis will be computed automatically from your input.
                        </>
                      ) : (
                        <>
                          <strong>🔭 Kepler Mode:</strong> All fields including SNR, Stellar Mass, and Semi-major Axis must be provided.
                        </>
                      )}
                    </p>
                  </div>
                </TooltipProvider>

                <Button type="submit" className="w-full" size="lg" disabled={predicting}>
                  {predicting ? 'Analyzing...' : 'Submit & Predict'}
                </Button>
              </form>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Prediction Result Dialog */}
      <Dialog open={showPredictionDialog} onOpenChange={setShowPredictionDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Prediction Result</DialogTitle>
            <DialogDescription>
              Analysis of your exoplanet candidate
            </DialogDescription>
          </DialogHeader>
          {predictionResult && (
            <div className="space-y-6">
              {/* Main Classification */}
              <div className={`p-6 rounded-lg border-2 text-center ${
                predictionResult.prediction.is_planet 
                  ? 'bg-green-50 dark:bg-green-950 border-green-500 text-green-900 dark:text-green-100' 
                  : 'bg-red-50 dark:bg-red-950 border-red-500 text-red-900 dark:text-red-100'
              }`}>
                <div className="text-3xl font-bold mb-2">
                  {predictionResult.prediction.classification}
                </div>
                <div className="text-lg">
                  Confidence: {predictionResult.prediction.confidence}
                </div>
                <div className="text-sm mt-2 opacity-80">
                  Probability: {(predictionResult.prediction.probability * 100).toFixed(1)}%
                </div>
              </div>

              {/* Model Breakdown */}
              <div>
                <h3 className="font-semibold mb-3">Model Breakdown</h3>
                <div className="space-y-2">
                  <div className="flex items-center justify-between p-3 bg-muted rounded">
                    <span className="font-medium">XGBoost</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 h-2 bg-background rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-blue-500"
                          style={{ width: `${predictionResult.model_breakdown.xgboost * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-mono">
                        {(predictionResult.model_breakdown.xgboost * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-muted rounded">
                    <span className="font-medium">Random Forest</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 h-2 bg-background rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-green-500"
                          style={{ width: `${predictionResult.model_breakdown.random_forest * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-mono">
                        {(predictionResult.model_breakdown.random_forest * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-muted rounded">
                    <span className="font-medium">Neural Network</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 h-2 bg-background rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-purple-500"
                          style={{ width: `${predictionResult.model_breakdown.neural_network * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-mono">
                        {(predictionResult.model_breakdown.neural_network * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-primary/10 rounded border-2 border-primary">
                    <span className="font-bold">Ensemble</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 h-2 bg-background rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-primary"
                          style={{ width: `${predictionResult.model_breakdown.ensemble * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-mono font-bold">
                        {(predictionResult.model_breakdown.ensemble * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Metadata */}
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="p-3 bg-muted rounded">
                  <div className="text-muted-foreground mb-1">Dataset Type</div>
                  <div className="font-semibold uppercase">{predictionResult.metadata.dataset_type}</div>
                </div>
                <div className="p-3 bg-muted rounded">
                  <div className="text-muted-foreground mb-1">Features Used</div>
                  <div className="font-semibold">{predictionResult.metadata.features_used}</div>
                </div>
              </div>

              {predictionResult.metadata.auto_calculated?.length > 0 && (
                <div className="text-xs text-muted-foreground p-3 bg-muted rounded">
                  <strong>Auto-calculated fields:</strong> {predictionResult.metadata.auto_calculated.join(', ')}
                </div>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Upload;
