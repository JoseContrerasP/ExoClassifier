import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Settings, Play, AlertCircle, Database, CheckCircle2, Loader2, Upload, FileText } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";

interface UploadedDataset {
  upload_id: string;
  filename: string;
  dataset_type: string;
  upload_date: string;
  summary: {
    processed_rows: number;
    total_features: number;
  };
}

const FineTune = () => {
  const { toast } = useToast();
  const navigate = useNavigate();
  
  // File upload for raw labeled data
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [datasetType, setDatasetType] = useState<'tess' | 'kepler'>('tess');
  const [uploading, setUploading] = useState(false);
  const [uploadedTrainingId, setUploadedTrainingId] = useState<string>("");
  
  // Model selection (radio, single choice)
  const [selectedModel, setSelectedModel] = useState<string>("xgboost");
  
  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingResults, setTrainingResults] = useState<any>(null);

  // XGBoost hyperparameters
  const [xgbParams, setXgbParams] = useState({
    max_depth: 7,
    learning_rate: 0.05,
    n_estimators: 300,
    subsample: 0.85,
    colsample_bytree: 0.85,
    min_child_weight: 3,
    gamma: 0.1,
    reg_alpha: 0.01,
    reg_lambda: 2,
  });

  // Random Forest hyperparameters
  const [rfParams, setRfParams] = useState({
    n_estimators: 300,
    max_depth: 25,
    min_samples_split: 3,
    min_samples_leaf: 2,
    max_features: 'sqrt',
  });

  // Neural Network hyperparameters
  const [nnParams, setNnParams] = useState({
    layer_sizes: '512,256,128,64',
    dropout_rate: 0.4,
    learning_rate: 0.0008,
    epochs: 150,
    batch_size: 32,
  });

  // Training configuration
  const [validationSplit, setValidationSplit] = useState(0.15);
  const [testSplit, setTestSplit] = useState(0.15);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleUploadRawData = async () => {
    if (!selectedFile) {
      toast({
        title: "No file selected",
        description: "Please select a CSV file to upload",
        variant: "destructive",
      });
      return;
    }

    setUploading(true);
    try {
      // Upload raw labeled data (without preprocessing to keep disposition column)
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('dataset_type', datasetType);
      formData.append('keep_labels', 'true'); // Special flag to keep disposition column

      const response = await fetch('http://localhost:5000/api/train/upload_labeled', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (result.success) {
        setUploadedTrainingId(result.upload_id);
        toast({
          title: "✅ Upload successful",
          description: `Loaded ${result.summary.total_rows} rows for training`,
        });
      } else {
        throw new Error(result.error || 'Upload failed');
      }
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

  const handleStartTraining = async () => {
    if (!uploadedTrainingId) {
      toast({
        title: "No training data uploaded",
        description: "Please upload labeled data first",
        variant: "destructive",
      });
      return;
    }

    if (!selectedModel) {
      toast({
        title: "No model selected",
        description: "Please select a model to train",
        variant: "destructive",
      });
      return;
    }

    setIsTraining(true);
    setTrainingProgress(0);
    setTrainingResults(null);

    try {
      // Prepare params based on selected model
      let modelParams = null;
      if (selectedModel === 'xgboost') {
        modelParams = { xgboost_params: xgbParams };
      } else if (selectedModel === 'random_forest') {
        modelParams = { random_forest_params: rfParams };
      } else if (selectedModel === 'neural_network') {
        modelParams = {
          neural_network_params: {
            ...nnParams,
            layer_sizes: nnParams.layer_sizes.split(',').map(Number),
          }
        };
      }

      const response = await fetch('http://localhost:5000/api/train/finetune', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          upload_id: uploadedTrainingId,
          models: [selectedModel],
          ...modelParams,
          validation_split: validationSplit,
          test_split: testSplit,
        }),
      });

      const result = await response.json();

      if (result.success) {
        setTrainingResults(result);
        toast({
          title: "✅ Fine-tuning complete!",
          description: `Trained ${selectedModel} successfully`,
        });
      } else {
        throw new Error(result.error || 'Training failed');
      }
    } catch (error: any) {
      toast({
        title: "❌ Training failed",
        description: error.message || "Could not complete fine-tuning",
        variant: "destructive",
      });
    } finally {
      setIsTraining(false);
      setTrainingProgress(100);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Model Fine-Tuning</h1>
        <p className="text-muted-foreground">
          Train custom models with your labeled datasets and hyperparameters
        </p>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Main Configuration */}
        <div className="lg:col-span-2 space-y-6">
          {/* Upload Labeled Data */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="h-5 w-5" />
                Upload Labeled Training Data
              </CardTitle>
              <CardDescription>
                Upload raw CSV with disposition column (koi_disposition, tfopwg_disp, or disposition)
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Dataset Type Selection */}
              <div className="space-y-2">
                <Label>Dataset Type</Label>
                <RadioGroup value={datasetType} onValueChange={(val: any) => setDatasetType(val)}>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="tess" id="tess" />
                    <Label htmlFor="tess" className="font-normal cursor-pointer">
                      🛰️ TESS (16 base features + 3 auto-calculated)
                    </Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="kepler" id="kepler" />
                    <Label htmlFor="kepler" className="font-normal cursor-pointer">
                      🔭 Kepler (19 features required)
                    </Label>
                  </div>
                </RadioGroup>
              </div>

              {/* File Input */}
              <div className="space-y-2">
                <Label>CSV File</Label>
                <Input
                  type="file"
                  accept=".csv"
                  onChange={handleFileSelect}
                  disabled={uploading}
                />
                {selectedFile && (
                  <p className="text-sm text-muted-foreground">
                    Selected: {selectedFile.name} ({(selectedFile.size / 1024).toFixed(2)} KB)
                  </p>
                )}
              </div>

              {/* Upload Button */}
              <Button
                onClick={handleUploadRawData}
                disabled={!selectedFile || uploading}
                className="w-full"
              >
                {uploading ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <FileText className="h-4 w-4 mr-2" />
                    Upload for Training
                  </>
                )}
              </Button>

              {/* Upload Status */}
              {uploadedTrainingId && (
                <div className="bg-green-50 dark:bg-green-950 border border-green-500 rounded-lg p-3">
                  <div className="flex items-center gap-2 text-green-700 dark:text-green-300">
                    <CheckCircle2 className="h-4 w-4" />
                    <span className="text-sm font-medium">Data uploaded successfully!</span>
                  </div>
                  <p className="text-xs text-green-600 dark:text-green-400 mt-1">
                    Ready to train. Configure model and hyperparameters below.
                  </p>
                </div>
              )}

              {/* Important Note */}
              <div className="bg-amber-50 dark:bg-amber-950/30 border border-amber-300 dark:border-amber-700 rounded-lg p-3">
                <p className="text-sm text-amber-800 dark:text-amber-200">
                  <strong>⚠️ Important:</strong> Your CSV must include a disposition column with labels:
                </p>
                <ul className="text-xs text-amber-700 dark:text-amber-300 mt-2 space-y-1 ml-4 list-disc">
                  <li>Planets: CONFIRMED, CANDIDATE, CP, PC, KP, APC</li>
                  <li>False Positives: FALSE POSITIVE, FP, FA</li>
                </ul>
              </div>
            </CardContent>
          </Card>

          {/* Model Selection */}
          <Card>
            <CardHeader>
              <CardTitle>Model Selection</CardTitle>
              <CardDescription>Choose one model to train</CardDescription>
            </CardHeader>
            <CardContent>
              <RadioGroup value={selectedModel} onValueChange={setSelectedModel}>
                <div className="space-y-3">
                  {[
                    { id: 'xgboost', label: '⚡ XGBoost', desc: 'Gradient Boosting Classifier (~2-5 min)' },
                    { id: 'random_forest', label: '🌲 Random Forest', desc: 'Ensemble Tree Classifier (~3-7 min)' },
                    { id: 'neural_network', label: '🧠 Neural Network', desc: 'Deep Learning Classifier (~10-20 min)' },
                  ].map((model) => (
                    <div key={model.id} className="flex items-center space-x-3 p-3 border rounded-lg hover:bg-muted/50 cursor-pointer">
                      <RadioGroupItem value={model.id} id={model.id} />
                      <Label htmlFor={model.id} className="flex-1 cursor-pointer">
                        <p className="font-medium">{model.label}</p>
                        <p className="text-xs text-muted-foreground">{model.desc}</p>
                      </Label>
                    </div>
                  ))}
                </div>
              </RadioGroup>
            </CardContent>
          </Card>

          {/* Hyperparameters */}
          <Card>
            <CardHeader>
              <CardTitle>Hyperparameter Configuration</CardTitle>
              <CardDescription>Customize model parameters for optimal performance</CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs value={selectedModel} onValueChange={setSelectedModel}>
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="xgboost">
                    XGBoost
                  </TabsTrigger>
                  <TabsTrigger value="random_forest">
                    Random Forest
                  </TabsTrigger>
                  <TabsTrigger value="neural_network">
                    Neural Network
                  </TabsTrigger>
                </TabsList>

                {/* XGBoost Parameters */}
                <TabsContent value="xgboost" className="space-y-4 mt-4">
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Max Depth</Label>
                      <Input
                        type="number"
                        value={xgbParams.max_depth}
                        onChange={(e) => setXgbParams({...xgbParams, max_depth: Number(e.target.value)})}
                        min={3}
                        max={15}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Learning Rate</Label>
                      <Input
                        type="number"
                        step="0.01"
                        value={xgbParams.learning_rate}
                        onChange={(e) => setXgbParams({...xgbParams, learning_rate: Number(e.target.value)})}
                        min={0.001}
                        max={0.3}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Number of Estimators</Label>
                      <Input
                        type="number"
                        value={xgbParams.n_estimators}
                        onChange={(e) => setXgbParams({...xgbParams, n_estimators: Number(e.target.value)})}
                        min={50}
                        max={1000}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Subsample</Label>
                      <Input
                        type="number"
                        step="0.05"
                        value={xgbParams.subsample}
                        onChange={(e) => setXgbParams({...xgbParams, subsample: Number(e.target.value)})}
                        min={0.5}
                        max={1}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Colsample by Tree</Label>
                      <Input
                        type="number"
                        step="0.05"
                        value={xgbParams.colsample_bytree}
                        onChange={(e) => setXgbParams({...xgbParams, colsample_bytree: Number(e.target.value)})}
                        min={0.5}
                        max={1}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Min Child Weight</Label>
                      <Input
                        type="number"
                        value={xgbParams.min_child_weight}
                        onChange={(e) => setXgbParams({...xgbParams, min_child_weight: Number(e.target.value)})}
                        min={1}
                        max={10}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Gamma</Label>
                      <Input
                        type="number"
                        step="0.1"
                        value={xgbParams.gamma}
                        onChange={(e) => setXgbParams({...xgbParams, gamma: Number(e.target.value)})}
                        min={0}
                        max={5}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Reg Alpha (L1)</Label>
                      <Input
                        type="number"
                        step="0.01"
                        value={xgbParams.reg_alpha}
                        onChange={(e) => setXgbParams({...xgbParams, reg_alpha: Number(e.target.value)})}
                        min={0}
                        max={1}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Reg Lambda (L2)</Label>
                      <Input
                        type="number"
                        step="0.1"
                        value={xgbParams.reg_lambda}
                        onChange={(e) => setXgbParams({...xgbParams, reg_lambda: Number(e.target.value)})}
                        min={0}
                        max={10}
                      />
                    </div>
                  </div>
                </TabsContent>

                {/* Random Forest Parameters */}
                <TabsContent value="random_forest" className="space-y-4 mt-4">
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Number of Estimators</Label>
                      <Input
                        type="number"
                        value={rfParams.n_estimators}
                        onChange={(e) => setRfParams({...rfParams, n_estimators: Number(e.target.value)})}
                        min={50}
                        max={1000}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Max Depth</Label>
                      <Input
                        type="number"
                        value={rfParams.max_depth}
                        onChange={(e) => setRfParams({...rfParams, max_depth: Number(e.target.value)})}
                        min={10}
                        max={50}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Min Samples Split</Label>
                      <Input
                        type="number"
                        value={rfParams.min_samples_split}
                        onChange={(e) => setRfParams({...rfParams, min_samples_split: Number(e.target.value)})}
                        min={2}
                        max={10}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Min Samples Leaf</Label>
                      <Input
                        type="number"
                        value={rfParams.min_samples_leaf}
                        onChange={(e) => setRfParams({...rfParams, min_samples_leaf: Number(e.target.value)})}
                        min={1}
                        max={10}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Max Features</Label>
                      <Select
                        value={rfParams.max_features}
                        onValueChange={(val) => setRfParams({...rfParams, max_features: val})}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="sqrt">sqrt</SelectItem>
                          <SelectItem value="log2">log2</SelectItem>
                          <SelectItem value="None">None (all)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </TabsContent>

                {/* Neural Network Parameters */}
                <TabsContent value="neural_network" className="space-y-4 mt-4">
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Layer Sizes (comma-separated)</Label>
                      <Input
                        type="text"
                        value={nnParams.layer_sizes}
                        onChange={(e) => setNnParams({...nnParams, layer_sizes: e.target.value})}
                        placeholder="512,256,128,64"
                      />
                      <p className="text-xs text-muted-foreground">e.g., 512,256,128,64</p>
                    </div>
                    <div className="space-y-2">
                      <Label>Dropout Rate</Label>
                      <Input
                        type="number"
                        step="0.1"
                        value={nnParams.dropout_rate}
                        onChange={(e) => setNnParams({...nnParams, dropout_rate: Number(e.target.value)})}
                        min={0}
                        max={0.8}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Learning Rate</Label>
                      <Input
                        type="number"
                        step="0.0001"
                        value={nnParams.learning_rate}
                        onChange={(e) => setNnParams({...nnParams, learning_rate: Number(e.target.value)})}
                        min={0.0001}
                        max={0.01}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Epochs</Label>
                      <Input
                        type="number"
                        value={nnParams.epochs}
                        onChange={(e) => setNnParams({...nnParams, epochs: Number(e.target.value)})}
                        min={10}
                        max={300}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Batch Size</Label>
                      <Input
                        type="number"
                        value={nnParams.batch_size}
                        onChange={(e) => setNnParams({...nnParams, batch_size: Number(e.target.value)})}
                        min={8}
                        max={128}
                      />
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          {/* Training Configuration */}
          <Card>
            <CardHeader>
              <CardTitle>Training Configuration</CardTitle>
              <CardDescription>Data split and training settings</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Validation Split</Label>
                  <Input
                    type="number"
                    step="0.05"
                    value={validationSplit}
                    onChange={(e) => setValidationSplit(Number(e.target.value))}
                    min={0.1}
                    max={0.3}
                  />
                  <p className="text-xs text-muted-foreground">Fraction of data for validation</p>
                </div>
                <div className="space-y-2">
                  <Label>Test Split</Label>
                  <Input
                    type="number"
                    step="0.05"
                    value={testSplit}
                    onChange={(e) => setTestSplit(Number(e.target.value))}
                    min={0.1}
                    max={0.3}
                  />
                  <p className="text-xs text-muted-foreground">Fraction of data for testing</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Training Control Panel */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Training Control</CardTitle>
              <CardDescription>Start and monitor training</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Training Data</span>
                  <Badge variant={uploadedTrainingId ? "default" : "secondary"}>
                    {uploadedTrainingId ? "Uploaded" : "Not Uploaded"}
                  </Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Model</span>
                  <Badge variant={selectedModel ? "default" : "secondary"}>
                    {selectedModel ? selectedModel.replace('_', ' ').toUpperCase() : 'None'}
                  </Badge>
                </div>
              </div>

              {isTraining && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Training...</span>
                    <span className="font-medium">{trainingProgress}%</span>
                  </div>
                  <Progress value={trainingProgress} />
                </div>
              )}

              <Button
                onClick={handleStartTraining}
                disabled={isTraining || !uploadedTrainingId || !selectedModel}
                className="w-full"
              >
                {isTraining ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Training...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Start Fine-Tuning
                  </>
                )}
              </Button>

              <div className="bg-muted/50 rounded-lg p-4 text-sm">
                <p className="font-medium flex items-center gap-2 mb-2">
                  <Settings className="h-4 w-4" />
                  Configuration Summary
                </p>
                <div className="text-muted-foreground space-y-1">
                  <p>Model: {selectedModel ? selectedModel.replace('_', ' ').toUpperCase() : 'Not selected'}</p>
                  <p>Validation: {(validationSplit * 100).toFixed(0)}%</p>
                  <p>Test: {(testSplit * 100).toFixed(0)}%</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Training Results */}
          {trainingResults && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="h-5 w-5 text-green-600" />
                  Training Complete
                </CardTitle>
                <CardDescription>Fine-tuned models saved successfully</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  {Object.entries(trainingResults.results || {}).map(([model, metrics]: [string, any]) => (
                    <div key={model} className="p-3 border rounded-lg">
                      <p className="font-medium capitalize mb-1">{model.replace('_', ' ')}</p>
                      <div className="text-sm text-muted-foreground space-y-1">
                        <div className="flex justify-between">
                          <span>Accuracy:</span>
                          <span className="font-medium">{(metrics.accuracy * 100).toFixed(2)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span>AUC:</span>
                          <span className="font-medium">{metrics.auc?.toFixed(4)}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="text-xs text-muted-foreground p-3 bg-muted rounded">
                  <p className="font-medium mb-1">Models saved to:</p>
                  <p className="font-mono">{trainingResults.model_dir}</p>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Warning */}
          <Card className="border-yellow-500/50">
            <CardContent className="pt-6">
              <div className="flex items-start gap-3">
                <AlertCircle className="h-5 w-5 text-yellow-600 mt-0.5" />
                <div className="text-sm space-y-1">
                  <p className="font-medium">Important Notes</p>
                  <ul className="text-muted-foreground space-y-1 list-disc list-inside">
                    <li>New models will be saved separately</li>
                    <li>Base models remain unchanged</li>
                    <li>Training requires labeled data (disposition column)</li>
                    <li>Training time varies by dataset size</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default FineTune;
