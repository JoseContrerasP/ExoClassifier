import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Upload, Trash2, Download, Database } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { Badge } from "@/components/ui/badge";

interface Dataset {
  id: string;
  name: string;
  size: string;
  uploadDate: string;
  samples: number;
  status: "ready" | "processing" | "error";
}

const Datasets = () => {
  const { toast } = useToast();
  const [datasets, setDatasets] = useState<Dataset[]>([
    {
      id: "1",
      name: "kepler_training_set.csv",
      size: "2.4 MB",
      uploadDate: "2025-03-15",
      samples: 5124,
      status: "ready"
    },
    {
      id: "2",
      name: "tess_candidates.csv",
      size: "1.8 MB",
      uploadDate: "2025-03-14",
      samples: 3842,
      status: "ready"
    }
  ]);

  const handleUploadDataset = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      // Backend integration point: uploadFineTuneDataset(file)
      toast({
        title: "Dataset uploaded",
        description: `${file.name} is being processed for fine-tuning.`,
      });
    }
  };

  const handleDeleteDataset = (id: string) => {
    // Backend integration point: deleteDataset(id)
    setDatasets(datasets.filter(d => d.id !== id));
    toast({
      title: "Dataset deleted",
      description: "The dataset has been removed successfully.",
    });
  };

  const getStatusColor = (status: Dataset["status"]) => {
    switch (status) {
      case "ready": return "bg-success";
      case "processing": return "bg-warning";
      case "error": return "bg-destructive";
      default: return "bg-muted";
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Dataset Management</h1>
        <p className="text-muted-foreground">
          Upload and manage custom datasets for model fine-tuning
        </p>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Upload New Dataset</CardTitle>
              <CardDescription>
                Add training data for fine-tuning
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="border-2 border-dashed border-border rounded-lg p-8 text-center transition-smooth hover:border-primary">
                <Database className="h-12 w-12 text-muted-foreground mx-auto mb-3" />
                <Label htmlFor="dataset-upload" className="cursor-pointer">
                  <span className="text-primary font-medium">Upload Dataset</span>
                </Label>
                <Input
                  id="dataset-upload"
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={handleUploadDataset}
                />
              </div>

              <div className="bg-muted/50 rounded-lg p-4 text-sm space-y-2">
                <p className="font-medium">Dataset Requirements:</p>
                <ul className="list-disc list-inside text-muted-foreground space-y-1">
                  <li>CSV format with header row</li>
                  <li>Include target labels</li>
                  <li>Match feature columns</li>
                  <li>Minimum 100 samples</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Your Datasets</CardTitle>
                  <CardDescription>
                    Manage uploaded training datasets
                  </CardDescription>
                </div>
                <Badge variant="outline" className="text-xs font-mono">
                  #TABLE_RENDER_API
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {datasets.length === 0 ? (
                  <div className="text-center py-12 text-muted-foreground">
                    <Database className="h-16 w-16 mx-auto mb-4 opacity-50" />
                    <p>No datasets uploaded yet</p>
                  </div>
                ) : (
                  datasets.map((dataset) => (
                    <div
                      key={dataset.id}
                      className="flex items-center justify-between p-4 border border-border rounded-lg transition-smooth hover:border-primary"
                    >
                      <div className="flex items-center gap-4">
                        <div className={`w-2 h-2 rounded-full ${getStatusColor(dataset.status)}`} />
                        <div>
                          <h3 className="font-medium">{dataset.name}</h3>
                          <div className="flex items-center gap-3 text-sm text-muted-foreground mt-1">
                            <span>{dataset.size}</span>
                            <span>•</span>
                            <span>{dataset.samples} samples</span>
                            <span>•</span>
                            <span>{dataset.uploadDate}</span>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant={dataset.status === "ready" ? "default" : "secondary"}>
                          {dataset.status}
                        </Badge>
                        <Button size="sm" variant="ghost">
                          <Download className="h-4 w-4" />
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => handleDeleteDataset(dataset.id)}
                        >
                          <Trash2 className="h-4 w-4 text-destructive" />
                        </Button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Datasets;
