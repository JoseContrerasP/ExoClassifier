import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Sparkles, CheckCircle2, XCircle, Database, FileText } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { Progress } from "@/components/ui/progress";
import { predictBatch } from "@/lib/api-placeholders";
import { API_ENDPOINTS } from "@/lib/api-config";

interface UploadedDataset {
  upload_id: string;
  filename: string;
  dataset_type: string;
  upload_date: string;
  summary: {
    processed_rows: number;
    total_features: number;
    base_features: number;
    engineered_features: number;
  };
}

interface FineTunedModel {
  model_id: string;
  timestamp: string;
  models: string[];
  samples: number;
  features: number;
  results: any;
}

const Predict = () => {
  const { toast } = useToast();
  const navigate = useNavigate();
  const [selectedModel, setSelectedModel] = useState<string>("ensemble");
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [datasets, setDatasets] = useState<UploadedDataset[]>([]);
  const [fineTunedModels, setFineTunedModels] = useState<FineTunedModel[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [predictions, setPredictions] = useState<any>(null);
  const [currentUploadId, setCurrentUploadId] = useState<string>("");

  // Load datasets and fine-tuned models on mount
  useEffect(() => {
    const stored = localStorage.getItem('uploaded_datasets');
    if (stored) {
      setDatasets(JSON.parse(stored));
    }
    loadFineTunedModels();
  }, []);

  const loadFineTunedModels = async () => {
    try {
      const response = await fetch(API_ENDPOINTS.trainModelsList);
      const result = await response.json();
      if (result.success) {
        setFineTunedModels(result.models);
      }
    } catch (error) {
      console.error('Failed to load fine-tuned models:', error);
    }
  };

  const handlePredict = async () => {
    if (!selectedDataset) {
      toast({
        title: "No dataset selected",
        description: "Please select a dataset to predict on",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    try {
      // Call the real batch prediction API
      const result = await predictBatch(selectedDataset, selectedModel);
      
      setPredictions(result);
      setCurrentUploadId(result.upload_id); // Store for export/details
      toast({
        title: "✅ Prediction complete",
        description: `Classified ${result.total_predictions} samples`,
      });
    } catch (error: any) {
      toast({
        title: "❌ Prediction failed",
        description: error.message || "Could not process predictions",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleExportResults = () => {
    if (!currentUploadId) return;
    
    // Download CSV file
    const url = API_ENDPOINTS.predictExport(currentUploadId);
    window.open(url, '_blank');
    
    toast({
      title: "📥 Downloading results",
      description: "Your predictions CSV is being downloaded",
    });
  };

  const handleViewDetails = () => {
    if (!currentUploadId) return;
    
    // Navigate to Visualize page with upload_id parameter
    navigate(`/visualize?upload_id=${currentUploadId}`);
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-5xl">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Exoplanet Prediction</h1>
        <p className="text-muted-foreground">
          Classify exoplanet candidates using trained machine learning models
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <div>
          <Card>
            <CardHeader>
              <CardTitle>Model Selection</CardTitle>
              <CardDescription>
                Choose the classification model to use
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Dataset Selection */}
              <div className="space-y-2">
                <Label>Select Dataset</Label>
                <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                  <SelectTrigger>
                    <SelectValue placeholder="Choose a dataset..." />
                  </SelectTrigger>
                  <SelectContent>
                    {datasets.length === 0 ? (
                      <div className="p-2 text-sm text-muted-foreground text-center">
                        No datasets available. Upload data first.
                      </div>
                    ) : (
                      datasets.map((dataset) => (
                        <SelectItem key={dataset.upload_id} value={dataset.upload_id}>
                          <div className="flex items-center gap-2">
                            <Database className="h-4 w-4" />
                            {dataset.filename} ({dataset.summary.processed_rows} rows)
                          </div>
                        </SelectItem>
                      ))
                    )}
                  </SelectContent>
                </Select>
              </div>

              {/* Model Selection */}
              <div className="space-y-2">
                <Label>Select Model</Label>
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">Base Models</div>
                    <SelectItem value="ensemble">🏆 Ensemble (All 3 Models)</SelectItem>
                    <SelectItem value="xgboost">⚡ XGBoost</SelectItem>
                    <SelectItem value="random_forest">🌲 Random Forest</SelectItem>
                    <SelectItem value="neural_network">🧠 Neural Network</SelectItem>
                    
                    {fineTunedModels.length > 0 && (
                      <>
                        <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground mt-2 border-t">Fine-Tuned Models</div>
                        {fineTunedModels.map((model) => (
                          <SelectItem key={model.model_id} value={`finetuned:${model.model_id}`}>
                            <div className="flex items-center gap-2">
                              <span>🎯</span>
                              <span className="text-xs">
                                {model.models.join(', ')} - {model.samples} samples
                              </span>
                            </div>
                          </SelectItem>
                        ))}
                      </>
                    )}
                  </SelectContent>
                </Select>
                {selectedModel.startsWith('finetuned:') && (
                  <p className="text-xs text-muted-foreground">
                    Using custom fine-tuned model
                  </p>
                )}
              </div>

              {/* Model Information */}
              <div className="bg-muted/50 rounded-lg p-4 space-y-2 text-sm">
                <p className="font-medium">Model Information</p>
                {selectedModel === "ensemble" && (
                  <div className="text-muted-foreground space-y-1">
                    <p>Weighted ensemble of 3 models</p>
                    <p>Uses XGBoost + Random Forest + Neural Network</p>
                    <p>Best overall performance</p>
                  </div>
                )}
                {selectedModel === "xgboost" && (
                  <div className="text-muted-foreground space-y-1">
                    <p>Gradient boosting classifier</p>
                    <p>Test Accuracy: ~83%</p>
                    <p>AUC: ~0.916</p>
                  </div>
                )}
                {selectedModel === "random_forest" && (
                  <div className="text-muted-foreground space-y-1">
                    <p>Random forest ensemble</p>
                    <p>Test Accuracy: ~85%</p>
                    <p>AUC: ~0.916</p>
                  </div>
                )}
                {selectedModel === "neural_network" && (
                  <div className="text-muted-foreground space-y-1">
                    <p>Deep neural network (4 layers)</p>
                    <p>Test Accuracy: ~83%</p>
                    <p>AUC: ~0.907</p>
                  </div>
                )}
              </div>

              {/* Selected Dataset Info */}
              {selectedDataset && (
                <div className="bg-muted/50 rounded-lg p-4 space-y-2 text-sm">
                  <p className="font-medium">Selected Dataset</p>
                  {datasets.find(d => d.upload_id === selectedDataset) && (() => {
                    const dataset = datasets.find(d => d.upload_id === selectedDataset)!;
                    return (
                      <div className="space-y-1 text-muted-foreground">
                        <div className="flex items-center justify-between">
                          <span>Filename</span>
                          <Badge variant="outline">{dataset.filename}</Badge>
                        </div>
                        <div className="flex items-center justify-between">
                          <span>Rows</span>
                          <Badge variant="default">{dataset.summary.processed_rows}</Badge>
                        </div>
                        <div className="flex items-center justify-between">
                          <span>Type</span>
                          <Badge variant="secondary">{dataset.dataset_type.toUpperCase()}</Badge>
                        </div>
                      </div>
                    );
                  })()}
                </div>
              )}

              <Button 
                onClick={handlePredict} 
                disabled={isLoading || !selectedDataset}
                className="w-full"
              >
                {isLoading ? (
                  <>Processing...</>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4 mr-2" />
                    Run Batch Prediction
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </div>

        <div>
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Prediction Results</CardTitle>
                  <CardDescription>
                    Classification output and confidence metrics
                  </CardDescription>
                </div>
                <Badge variant="outline" className="text-xs font-mono">
                  #PREDICTION_API
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              {!predictions && !isLoading && (
                <div className="text-center py-12 text-muted-foreground">
                  <Sparkles className="h-16 w-16 mx-auto mb-4 opacity-50" />
                  <p>Select dataset and run prediction to see results</p>
                </div>
              )}

              {isLoading && (
                <div className="space-y-4 py-8">
                  <div className="text-center text-muted-foreground mb-4">
                    Processing batch predictions...
                  </div>
                  <Progress value={60} className="w-full" />
                </div>
              )}

              {predictions && !isLoading && (
                <div className="space-y-6">
                  {/* Summary Cards */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-green-50 dark:bg-green-950 border-2 border-green-500 rounded-lg p-6 text-center">
                      <div className="flex justify-center mb-2">
                        <CheckCircle2 className="h-8 w-8 text-green-600 dark:text-green-400" />
                      </div>
                      <p className="text-3xl font-bold text-green-900 dark:text-green-100">
                        {predictions.planet_count || 0}
                      </p>
                      <p className="text-sm text-green-700 dark:text-green-300">Confirmed Planets</p>
                    </div>

                    <div className="bg-red-50 dark:bg-red-950 border-2 border-red-500 rounded-lg p-6 text-center">
                      <div className="flex justify-center mb-2">
                        <XCircle className="h-8 w-8 text-red-600 dark:text-red-400" />
                      </div>
                      <p className="text-3xl font-bold text-red-900 dark:text-red-100">
                        {predictions.false_positive_count || 0}
                      </p>
                      <p className="text-sm text-red-700 dark:text-red-300">False Positives</p>
                    </div>
                  </div>

                  {/* Statistics */}
                  <div className="bg-muted/50 rounded-lg p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Total Predictions</span>
                      <Badge variant="default" className="text-lg">
                        {predictions.total_predictions}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Model Used</span>
                      <Badge variant="outline">
                        {predictions.model_used === 'ensemble' ? '🏆 Ensemble' :
                         predictions.model_used === 'xgboost' ? '⚡ XGBoost' :
                         predictions.model_used === 'random_forest' ? '🌲 Random Forest' :
                         '🧠 Neural Network'}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Planet Rate</span>
                      <span className="font-medium">
                        {((predictions.planet_count / predictions.total_predictions) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <Button 
                      variant="outline" 
                      className="flex-1"
                      onClick={handleExportResults}
                      disabled={!currentUploadId}
                    >
                      <FileText className="h-4 w-4 mr-2" />
                      Export Results
                    </Button>
                    <Button 
                      variant="outline" 
                      className="flex-1"
                      onClick={handleViewDetails}
                      disabled={!currentUploadId}
                    >
                      View Details
                    </Button>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Predict;
