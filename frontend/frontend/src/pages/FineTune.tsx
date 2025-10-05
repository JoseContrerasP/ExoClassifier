import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Settings, Play, AlertCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const FineTune = () => {
  const { toast } = useToast();
  const [selectedModel, setSelectedModel] = useState<string>("xgboost");
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);

  // XGBoost hyperparameters
  const [learningRate, setLearningRate] = useState([0.1]);
  const [numTrees, setNumTrees] = useState([100]);
  const [maxDepth, setMaxDepth] = useState([6]);

  // DNN hyperparameters
  const [numLayers, setNumLayers] = useState([4]);
  const [neuronsPerLayer, setNeuronsPerLayer] = useState([128]);

  const handleStartTraining = () => {
    setIsTraining(true);
    setTrainingProgress(0);

    // Backend integration point: startFineTuning(selectedModel, hyperparameters, datasetId)
    // Simulate training progress
    const interval = setInterval(() => {
      setTrainingProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsTraining(false);
          toast({
            title: "Training complete",
            description: "Model has been fine-tuned successfully.",
          });
          return 100;
        }
        return prev + 10;
      });
    }, 500);
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Model Fine-Tuning</h1>
        <p className="text-muted-foreground">
          Customize and retrain models with your own datasets and hyperparameters
        </p>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Hyperparameter Configuration</CardTitle>
              <CardDescription>
                Adjust model parameters for optimal performance
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label>Select Model Architecture</Label>
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="xgboost">XGBoost Classifier</SelectItem>
                    <SelectItem value="dnn">Deep Neural Network</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Tabs value={selectedModel} onValueChange={setSelectedModel}>
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="xgboost">XGBoost</TabsTrigger>
                  <TabsTrigger value="dnn">DNN</TabsTrigger>
                </TabsList>

                <TabsContent value="xgboost" className="space-y-6 mt-6">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <Label>Learning Rate</Label>
                        <span className="text-sm text-muted-foreground">{learningRate[0]}</span>
                      </div>
                      <Slider
                        value={learningRate}
                        onValueChange={setLearningRate}
                        min={0.01}
                        max={0.3}
                        step={0.01}
                      />
                      <p className="text-xs text-muted-foreground">
                        Controls the step size at each iteration
                      </p>
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <Label>Number of Trees</Label>
                        <span className="text-sm text-muted-foreground">{numTrees[0]}</span>
                      </div>
                      <Slider
                        value={numTrees}
                        onValueChange={setNumTrees}
                        min={50}
                        max={500}
                        step={10}
                      />
                      <p className="text-xs text-muted-foreground">
                        Total number of boosting rounds
                      </p>
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <Label>Maximum Depth</Label>
                        <span className="text-sm text-muted-foreground">{maxDepth[0]}</span>
                      </div>
                      <Slider
                        value={maxDepth}
                        onValueChange={setMaxDepth}
                        min={3}
                        max={10}
                        step={1}
                      />
                      <p className="text-xs text-muted-foreground">
                        Maximum depth of each tree
                      </p>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="dnn" className="space-y-6 mt-6">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <Label>Number of Layers</Label>
                        <span className="text-sm text-muted-foreground">{numLayers[0]}</span>
                      </div>
                      <Slider
                        value={numLayers}
                        onValueChange={setNumLayers}
                        min={2}
                        max={8}
                        step={1}
                      />
                      <p className="text-xs text-muted-foreground">
                        Number of hidden layers in the network
                      </p>
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <Label>Neurons per Layer</Label>
                        <span className="text-sm text-muted-foreground">{neuronsPerLayer[0]}</span>
                      </div>
                      <Slider
                        value={neuronsPerLayer}
                        onValueChange={setNeuronsPerLayer}
                        min={32}
                        max={256}
                        step={32}
                      />
                      <p className="text-xs text-muted-foreground">
                        Number of neurons in each hidden layer
                      </p>
                    </div>

                    <div className="space-y-2">
                      <Label>Activation Function</Label>
                      <Select defaultValue="relu">
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="relu">ReLU</SelectItem>
                          <SelectItem value="tanh">Tanh</SelectItem>
                          <SelectItem value="sigmoid">Sigmoid</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>

              <div className="bg-muted/50 rounded-lg p-4 flex items-start gap-3">
                <AlertCircle className="h-5 w-5 text-warning mt-0.5" />
                <div className="text-sm space-y-1">
                  <p className="font-medium">Training Configuration</p>
                  <p className="text-muted-foreground">
                    Fine-tuning will use your uploaded datasets. Training time varies based 
                    on dataset size and selected hyperparameters.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <div>
          <Card>
            <CardHeader>
              <CardTitle>Training Control</CardTitle>
              <CardDescription>
                Start and monitor training progress
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Training Dataset</span>
                  <Badge variant="default">Ready</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Validation Split</span>
                  <span className="text-sm font-medium">20%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Model</span>
                  <span className="text-sm font-medium">
                    {selectedModel === "xgboost" ? "XGBoost" : "DNN"}
                  </span>
                </div>
              </div>

              {isTraining && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Progress</span>
                    <span className="font-medium">{trainingProgress}%</span>
                  </div>
                  <Progress value={trainingProgress} />
                </div>
              )}

              <Button
                onClick={handleStartTraining}
                disabled={isTraining}
                className="w-full"
              >
                {isTraining ? (
                  <>Training in Progress...</>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Start Fine-Tuning
                  </>
                )}
              </Button>

              <div className="bg-muted/50 rounded-lg p-4 text-sm space-y-2">
                <p className="font-medium flex items-center gap-2">
                  <Settings className="h-4 w-4" />
                  Current Configuration
                </p>
                <div className="text-muted-foreground space-y-1">
                  {selectedModel === "xgboost" ? (
                    <>
                      <p>Learning Rate: {learningRate[0]}</p>
                      <p>Trees: {numTrees[0]}</p>
                      <p>Max Depth: {maxDepth[0]}</p>
                    </>
                  ) : (
                    <>
                      <p>Layers: {numLayers[0]}</p>
                      <p>Neurons: {neuronsPerLayer[0]}</p>
                      <p>Activation: ReLU</p>
                    </>
                  )}
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
