import { useEffect, useState } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { BarChart3, LineChart, PieChart, TrendingUp, ArrowLeft, Sparkles } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const Visualize = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { toast } = useToast();
  const uploadId = searchParams.get('upload_id');
  const [details, setDetails] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (uploadId) {
      fetchPredictionDetails();
    }
  }, [uploadId]);

  const fetchPredictionDetails = async () => {
    if (!uploadId) return;
    
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:5000/api/predict/details/${uploadId}`);
      const result = await response.json();
      
      if (result.success) {
        setDetails(result.details);
      } else {
        toast({
          title: "Error loading details",
          description: result.error,
          variant: "destructive",
        });
      }
    } catch (error: any) {
      toast({
        title: "Failed to load details",
        description: error.message,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  // If upload_id is provided, show prediction details
  if (uploadId) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="mb-6">
          <Button variant="ghost" onClick={() => navigate('/predict')} className="mb-4">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Predictions
          </Button>
          <h1 className="text-4xl font-bold mb-2">Prediction Details</h1>
          <p className="text-muted-foreground">
            Detailed analysis of batch prediction results
          </p>
        </div>

        {loading && (
          <div className="text-center py-12">
            <Sparkles className="h-16 w-16 mx-auto mb-4 opacity-50 animate-pulse" />
            <p className="text-muted-foreground">Loading prediction details...</p>
          </div>
        )}

        {details && !loading && (
          <div className="space-y-6">
            {/* Summary Statistics */}
            <div className="grid md:grid-cols-4 gap-4">
              <Card>
                <CardContent className="pt-6">
                  <div className="text-2xl font-bold">{details.summary.total}</div>
                  <p className="text-xs text-muted-foreground">Total Predictions</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <div className="text-2xl font-bold text-green-600">{details.summary.planets}</div>
                  <p className="text-xs text-muted-foreground">Confirmed Planets</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <div className="text-2xl font-bold text-red-600">{details.summary.false_positives}</div>
                  <p className="text-xs text-muted-foreground">False Positives</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <div className="text-2xl font-bold">{(details.summary.avg_planet_probability * 100).toFixed(1)}%</div>
                  <p className="text-xs text-muted-foreground">Avg Planet Probability</p>
                </CardContent>
              </Card>
            </div>

            {/* Confidence Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Confidence Distribution</CardTitle>
                <CardDescription>How confident is the model in its predictions?</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {Object.entries(details.confidence_distribution).map(([label, count]: [string, any]) => (
                    <div key={label} className="flex items-center gap-3">
                      <div className="w-40 text-sm">{label}</div>
                      <div className="flex-1 h-8 bg-muted rounded overflow-hidden">
                        <div 
                          className="h-full bg-primary"
                          style={{ width: `${(count / details.summary.total) * 100}%` }}
                        />
                      </div>
                      <div className="w-16 text-right text-sm font-medium">{count}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Top Planets */}
            {details.top_planets.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Top 10 Most Confident Planet Candidates</CardTitle>
                  <CardDescription>Highest probability detections</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {details.top_planets.map((planet: any, idx: number) => (
                      <div key={idx} className="flex items-center justify-between p-3 bg-muted rounded">
                        <div className="flex-1">
                          <Badge variant="outline" className="mr-2">#{idx + 1}</Badge>
                          Period: {planet.koi_period?.toFixed(2)} days | 
                          Radius: {planet.koi_prad?.toFixed(2)} R⊕ | 
                          Temp: {planet.koi_teq?.toFixed(0)} K
                        </div>
                        <Badge variant="default">{(planet.probability * 100).toFixed(1)}%</Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </div>
    );
  }

  // Default visualization page (no upload_id)
  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Model Insights</h1>
        <p className="text-muted-foreground">
          Explore trained model performance, feature importance, and diagnostics
        </p>
      </div>

      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="overview">Model Overview</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="features">Features</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Model Architecture Cards */}
          <div className="grid md:grid-cols-3 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  ⚡ XGBoost
                </CardTitle>
                <CardDescription>Gradient Boosting Classifier</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Test Accuracy</span>
                  <Badge variant="default">~83%</Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">AUC Score</span>
                  <Badge variant="default">0.916</Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Trees</span>
                  <Badge variant="outline">300</Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  🌲 Random Forest
                </CardTitle>
                <CardDescription>Ensemble Tree Classifier</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Test Accuracy</span>
                  <Badge variant="default">~85%</Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">AUC Score</span>
                  <Badge variant="default">0.916</Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Estimators</span>
                  <Badge variant="outline">300</Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  🧠 Neural Network
                </CardTitle>
                <CardDescription>Deep Learning Classifier</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Test Accuracy</span>
                  <Badge variant="default">~83%</Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">AUC Score</span>
                  <Badge variant="default">0.907</Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Layers</span>
                  <Badge variant="outline">4 Dense</Badge>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Training Info */}
          <Card>
            <CardHeader>
              <CardTitle>Training Dataset</CardTitle>
              <CardDescription>Combined Kepler + TESS data</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-4 gap-6">
                <div>
                  <div className="text-2xl font-bold">17,267</div>
                  <p className="text-xs text-muted-foreground">Total Samples</p>
                </div>
                <div>
                  <div className="text-2xl font-bold">27</div>
                  <p className="text-xs text-muted-foreground">Features Used</p>
                </div>
                <div>
                  <div className="text-2xl font-bold text-green-600">11,133</div>
                  <p className="text-xs text-muted-foreground">Confirmed Planets</p>
                </div>
                <div>
                  <div className="text-2xl font-bold text-red-600">6,134</div>
                  <p className="text-xs text-muted-foreground">False Positives</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* How to Use */}
          <Card>
            <CardHeader>
              <CardTitle>Get Started with Predictions</CardTitle>
              <CardDescription>Follow these steps to analyze your exoplanet data</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-start gap-4">
                  <Badge className="mt-1">1</Badge>
                  <div>
                    <p className="font-medium">Upload Data</p>
                    <p className="text-sm text-muted-foreground">Go to the Upload page and submit your CSV file (TESS or Kepler format)</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <Badge className="mt-1">2</Badge>
                  <div>
                    <p className="font-medium">Run Predictions</p>
                    <p className="text-sm text-muted-foreground">Navigate to Predict page, select your dataset and model, then analyze</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <Badge className="mt-1">3</Badge>
                  <div>
                    <p className="font-medium">View Results</p>
                    <p className="text-sm text-muted-foreground">Export predictions as CSV or view detailed visualizations here</p>
                  </div>
                </div>
              </div>
              <div className="mt-6 flex gap-3">
                <Button onClick={() => navigate('/upload')}>
                  Start Upload
                </Button>
                <Button variant="outline" onClick={() => navigate('/predict')}>
                  View Predictions
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Model Performance Comparison</CardTitle>
              <CardDescription>Test set metrics (combined Kepler + TESS)</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* XGBoost */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">⚡ XGBoost</span>
                    <span className="text-sm text-muted-foreground">AUC: 0.916</span>
                  </div>
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <div className="w-24 text-sm">Accuracy</div>
                      <div className="flex-1 h-6 bg-muted rounded overflow-hidden">
                        <div className="h-full bg-blue-500" style={{ width: '83%' }} />
                      </div>
                      <span className="text-sm font-medium">83%</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-24 text-sm">Precision</div>
                      <div className="flex-1 h-6 bg-muted rounded overflow-hidden">
                        <div className="h-full bg-blue-500" style={{ width: '82%' }} />
                      </div>
                      <span className="text-sm font-medium">82%</span>
                    </div>
                  </div>
                </div>

                {/* Random Forest */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">🌲 Random Forest</span>
                    <span className="text-sm text-muted-foreground">AUC: 0.916</span>
                  </div>
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <div className="w-24 text-sm">Accuracy</div>
                      <div className="flex-1 h-6 bg-muted rounded overflow-hidden">
                        <div className="h-full bg-green-500" style={{ width: '85%' }} />
                      </div>
                      <span className="text-sm font-medium">85%</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-24 text-sm">Precision</div>
                      <div className="flex-1 h-6 bg-muted rounded overflow-hidden">
                        <div className="h-full bg-green-500" style={{ width: '86%' }} />
                      </div>
                      <span className="text-sm font-medium">86%</span>
                    </div>
                  </div>
                </div>

                {/* Neural Network */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">🧠 Neural Network</span>
                    <span className="text-sm text-muted-foreground">AUC: 0.907</span>
                  </div>
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <div className="w-24 text-sm">Accuracy</div>
                      <div className="flex-1 h-6 bg-muted rounded overflow-hidden">
                        <div className="h-full bg-purple-500" style={{ width: '83%' }} />
                      </div>
                      <span className="text-sm font-medium">83%</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-24 text-sm">Precision</div>
                      <div className="flex-1 h-6 bg-muted rounded overflow-hidden">
                        <div className="h-full bg-purple-500" style={{ width: '89%' }} />
                      </div>
                      <span className="text-sm font-medium">89%</span>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="features" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Feature Information</CardTitle>
              <CardDescription>27 features used for training (after correlation filtering)</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <h3 className="font-semibold mb-2">Base Features (19)</h3>
                  <div className="grid md:grid-cols-2 gap-2 text-sm">
                    <Badge variant="outline">Transit Period</Badge>
                    <Badge variant="outline">Transit Depth</Badge>
                    <Badge variant="outline">Transit Duration</Badge>
                    <Badge variant="outline">Transit Midpoint</Badge>
                    <Badge variant="outline">Period Errors (±)</Badge>
                    <Badge variant="outline">Depth Errors (±)</Badge>
                    <Badge variant="outline">Duration Errors (±)</Badge>
                    <Badge variant="outline">Signal-to-Noise Ratio</Badge>
                    <Badge variant="outline">Planet Radius</Badge>
                    <Badge variant="outline">Equilibrium Temp</Badge>
                    <Badge variant="outline">Insolation Flux</Badge>
                    <Badge variant="outline">Stellar Temp</Badge>
                    <Badge variant="outline">Stellar Gravity</Badge>
                    <Badge variant="outline">Stellar Radius</Badge>
                    <Badge variant="outline">Stellar Mass</Badge>
                  </div>
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Engineered Features (8 of 18)</h3>
                  <div className="grid md:grid-cols-2 gap-2 text-sm">
                    <Badge variant="secondary">Period Relative Error</Badge>
                    <Badge variant="secondary">Depth Relative Error</Badge>
                    <Badge variant="secondary">Depth/Uncertainty Ratio</Badge>
                    <Badge variant="secondary">Radius Ratio</Badge>
                    <Badge variant="secondary">Habitable Zone Flag</Badge>
                    <Badge variant="secondary">Signal Strength</Badge>
                    <Badge variant="secondary">Planet Density Proxy</Badge>
                    <Badge variant="secondary">Period Day Offset</Badge>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Visualize;
