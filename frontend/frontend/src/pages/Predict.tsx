import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Sparkles, CheckCircle2, AlertTriangle, XCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { Progress } from "@/components/ui/progress";

const Predict = () => {
  const { toast } = useToast();
  const [selectedModel, setSelectedModel] = useState<string>("xgboost");
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<{
    class: string;
    probability: number;
    confidence: string;
  } | null>(null);

  const handlePredict = () => {
    setIsLoading(true);
    
    // Backend integration point: runPrediction(selectedModel, inputData)
    // Simulate API call
    setTimeout(() => {
      const mockPrediction = {
        class: "CONFIRMED",
        probability: 0.89,
        confidence: "HIGH"
      };
      setPrediction(mockPrediction);
      setIsLoading(false);
      toast({
        title: "Prediction complete",
        description: `Classification: ${mockPrediction.class}`,
      });
    }, 2000);
  };

  const getClassIcon = (classType: string) => {
    switch (classType) {
      case "CONFIRMED":
        return <CheckCircle2 className="h-6 w-6 text-success" />;
      case "CANDIDATE":
        return <AlertTriangle className="h-6 w-6 text-warning" />;
      case "FALSE POSITIVE":
        return <XCircle className="h-6 w-6 text-destructive" />;
      default:
        return null;
    }
  };

  const getClassColor = (classType: string) => {
    switch (classType) {
      case "CONFIRMED":
        return "bg-success/20 text-success border-success";
      case "CANDIDATE":
        return "bg-warning/20 text-warning border-warning";
      case "FALSE POSITIVE":
        return "bg-destructive/20 text-destructive border-destructive";
      default:
        return "";
    }
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
              <div className="space-y-2">
                <Label>Select Model</Label>
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

              <div className="bg-muted/50 rounded-lg p-4 space-y-2 text-sm">
                <p className="font-medium">Model Information</p>
                {selectedModel === "xgboost" ? (
                  <div className="text-muted-foreground space-y-1">
                    <p>Gradient boosting ensemble</p>
                    <p>Accuracy: 94.2%</p>
                    <p>Training samples: 5,124</p>
                    <p>Last updated: 2025-03-15</p>
                  </div>
                ) : (
                  <div className="text-muted-foreground space-y-1">
                    <p>Deep neural network (4 layers)</p>
                    <p>Accuracy: 92.8%</p>
                    <p>Training samples: 5,124</p>
                    <p>Last updated: 2025-03-15</p>
                  </div>
                )}
              </div>

              <div className="bg-muted/50 rounded-lg p-4 space-y-2 text-sm">
                <p className="font-medium">Input Data Status</p>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Parameters loaded</span>
                  <Badge variant="default">Ready</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Features extracted</span>
                  <Badge variant="default">Complete</Badge>
                </div>
              </div>

              <Button 
                onClick={handlePredict} 
                disabled={isLoading}
                className="w-full"
              >
                {isLoading ? (
                  <>Processing...</>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4 mr-2" />
                    Run Prediction
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
              {!prediction && !isLoading && (
                <div className="text-center py-12 text-muted-foreground">
                  <Sparkles className="h-16 w-16 mx-auto mb-4 opacity-50" />
                  <p>Run prediction to see results</p>
                </div>
              )}

              {isLoading && (
                <div className="space-y-4 py-8">
                  <div className="text-center text-muted-foreground mb-4">
                    Processing classification...
                  </div>
                  <Progress value={60} className="w-full" />
                </div>
              )}

              {prediction && !isLoading && (
                <div className="space-y-6">
                  <div className={`rounded-lg border-2 p-6 text-center ${getClassColor(prediction.class)}`}>
                    <div className="flex justify-center mb-3">
                      {getClassIcon(prediction.class)}
                    </div>
                    <h3 className="text-2xl font-bold mb-2">{prediction.class}</h3>
                    <p className="text-sm opacity-80">Classification Result</p>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span className="text-muted-foreground">Probability</span>
                        <span className="font-medium">{(prediction.probability * 100).toFixed(1)}%</span>
                      </div>
                      <Progress value={prediction.probability * 100} />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-muted/50 rounded-lg p-4">
                        <p className="text-sm text-muted-foreground mb-1">Confidence</p>
                        <p className="text-xl font-bold">{prediction.confidence}</p>
                      </div>
                      <div className="bg-muted/50 rounded-lg p-4">
                        <p className="text-sm text-muted-foreground mb-1">Model</p>
                        <p className="text-xl font-bold">{selectedModel === "xgboost" ? "XGB" : "DNN"}</p>
                      </div>
                    </div>
                  </div>

                  <Button variant="outline" className="w-full">
                    View Detailed Analysis
                  </Button>
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
