import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { BarChart3, LineChart, PieChart, TrendingUp } from "lucide-react";

const Visualize = () => {
  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Data Visualization</h1>
        <p className="text-muted-foreground">
          Explore model predictions, feature importance, and diagnostic metrics
        </p>
      </div>

      <Tabs defaultValue="lightcurve" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="lightcurve">Light Curve</TabsTrigger>
          <TabsTrigger value="features">Features</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
          <TabsTrigger value="diagnostics">Diagnostics</TabsTrigger>
        </TabsList>

        <TabsContent value="lightcurve" className="space-y-6">
          <div className="grid lg:grid-cols-1 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Transit Model Fit</CardTitle>
                <CardDescription>
                  Observed vs. model transit shape
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                  <div className="text-center text-muted-foreground">
                    <LineChart className="h-16 w-16 mx-auto mb-3 opacity-50" />
                    <p className="text-sm">Model fit comparison</p>
                    <p className="text-xs mt-1">Backend integration: renderModelFit()</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="features" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Feature Importance</CardTitle>
              <CardDescription>
                Contribution of each feature to model predictions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="aspect-[2/1] bg-muted rounded-lg flex items-center justify-center">
                <div className="text-center text-muted-foreground">
                  <BarChart3 className="h-16 w-16 mx-auto mb-3 opacity-50" />
                  <p className="text-sm">Feature importance chart</p>
                  <p className="text-xs mt-1">Backend integration: getFeatureImportance()</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Feature Correlations</CardTitle>
                <CardDescription>
                  Correlation matrix between input features
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="aspect-square bg-muted rounded-lg flex items-center justify-center">
                  <div className="text-center text-muted-foreground">
                    <PieChart className="h-16 w-16 mx-auto mb-3 opacity-50" />
                    <p className="text-sm">Correlation heatmap</p>
                    <p className="text-xs mt-1">Backend integration: renderCorrelationMatrix()</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Feature Distributions</CardTitle>
                <CardDescription>
                  Distribution of key parameters
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="aspect-square bg-muted rounded-lg flex items-center justify-center">
                  <div className="text-center text-muted-foreground">
                    <BarChart3 className="h-16 w-16 mx-auto mb-3 opacity-50" />
                    <p className="text-sm">Parameter distributions</p>
                    <p className="text-xs mt-1">Backend integration: renderDistributions()</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="metrics" className="space-y-6">
          <div className="flex justify-end mb-4">
            <Badge variant="outline" className="text-xs font-mono">
              #METRICS_TABLE_API
            </Badge>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { label: "Accuracy", value: "94.2%", icon: TrendingUp },
              { label: "Precision", value: "91.8%", icon: TrendingUp },
              { label: "Recall", value: "93.5%", icon: TrendingUp },
              { label: "F1-Score", value: "92.6%", icon: TrendingUp }
            ].map((metric) => (
              <Card key={metric.label}>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    {metric.label}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <span className="text-3xl font-bold">{metric.value}</span>
                    <metric.icon className="h-8 w-8 text-primary" />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Confusion Matrix</CardTitle>
              <CardDescription>
                Model performance across classification categories
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="aspect-square max-w-md mx-auto bg-muted rounded-lg flex items-center justify-center">
                <div className="text-center text-muted-foreground">
                  <PieChart className="h-16 w-16 mx-auto mb-3 opacity-50" />
                  <p className="text-sm">Confusion matrix visualization</p>
                  <p className="text-xs mt-1">Backend integration: renderConfusionMatrix()</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="diagnostics" className="space-y-6">
          <div className="grid lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Centroid Offset</CardTitle>
                <CardDescription>
                  Position shift during transit events
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                  <div className="text-center text-muted-foreground">
                    <LineChart className="h-16 w-16 mx-auto mb-3 opacity-50" />
                    <p className="text-sm">Centroid offset plot</p>
                    <p className="text-xs mt-1">Backend integration: renderCentroidOffset()</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Odd-Even Depth</CardTitle>
                <CardDescription>
                  Comparison of odd and even transit depths
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                  <div className="text-center text-muted-foreground">
                    <BarChart3 className="h-16 w-16 mx-auto mb-3 opacity-50" />
                    <p className="text-sm">Odd-even comparison</p>
                    <p className="text-xs mt-1">Backend integration: renderOddEvenDepth()</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Secondary Eclipse</CardTitle>
                <CardDescription>
                  Detection of secondary eclipse signal
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                  <div className="text-center text-muted-foreground">
                    <LineChart className="h-16 w-16 mx-auto mb-3 opacity-50" />
                    <p className="text-sm">Secondary eclipse plot</p>
                    <p className="text-xs mt-1">Backend integration: renderSecondaryEclipse()</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>MES Distribution</CardTitle>
                <CardDescription>
                  Multiple Event Statistic distribution
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                  <div className="text-center text-muted-foreground">
                    <BarChart3 className="h-16 w-16 mx-auto mb-3 opacity-50" />
                    <p className="text-sm">MES distribution</p>
                    <p className="text-xs mt-1">Backend integration: renderMESDistribution()</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Visualize;
