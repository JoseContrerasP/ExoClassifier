import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Trash2, Eye, BarChart3, Database, Satellite, Telescope, RefreshCw } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { getDataPreview, getDataStats, deleteUpload } from "@/lib/api-placeholders";

interface UploadedDataset {
  upload_id: string;
  filename: string;
  dataset_type: 'tess' | 'kepler';
  upload_date: string;
  summary: {
    total_rows: number;
    processed_rows: number;
    total_features: number;
    base_features: number;
    engineered_features: number;
  };
}

const Datasets = () => {
  const { toast } = useToast();
  const [datasets, setDatasets] = useState<UploadedDataset[]>([]);

  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [previewData, setPreviewData] = useState<any>(null);
  const [statsData, setStatsData] = useState<any>(null);
  const [showPreviewDialog, setShowPreviewDialog] = useState(false);
  const [showStatsDialog, setShowStatsDialog] = useState(false);
  const [loading, setLoading] = useState(false);

  // Load datasets from localStorage on mount
  useEffect(() => {
    loadDatasetsFromStorage();
  }, []);

  const loadDatasetsFromStorage = () => {
    const stored = localStorage.getItem('uploaded_datasets');
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setDatasets(parsed);
      } catch (e) {
        console.error('Failed to load datasets from storage', e);
      }
    }
  };

  const handleViewPreview = async (upload_id: string) => {
    setLoading(true);
    setSelectedDataset(upload_id);
    try {
      const result = await getDataPreview(upload_id, 10);
      setPreviewData(result);
      setShowPreviewDialog(true);
    } catch (error: any) {
      toast({
        title: "Failed to load preview",
        description: error.message || "Could not retrieve data preview",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleViewStats = async (upload_id: string) => {
    setLoading(true);
    setSelectedDataset(upload_id);
    try {
      const result = await getDataStats(upload_id);
      setStatsData(result);
      setShowStatsDialog(true);
    } catch (error: any) {
      toast({
        title: "Failed to load statistics",
        description: error.message || "Could not retrieve statistics",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteDataset = async (upload_id: string) => {
    if (!confirm('Are you sure you want to delete this dataset?')) {
      return;
    }

    setLoading(true);
    try {
      await deleteUpload(upload_id);
      
      // Remove from state and localStorage
      const updated = datasets.filter(d => d.upload_id !== upload_id);
      setDatasets(updated);
      localStorage.setItem('uploaded_datasets', JSON.stringify(updated));
      
      toast({
        title: "Dataset deleted",
        description: "The dataset has been removed successfully.",
      });
    } catch (error: any) {
      toast({
        title: "Failed to delete",
        description: error.message || "Could not delete dataset",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold mb-2">Uploaded Datasets</h1>
          <p className="text-muted-foreground">
            View and manage preprocessed exoplanet data ready for prediction
          </p>
        </div>
        <Button 
          variant="outline" 
          onClick={loadDatasetsFromStorage}
          disabled={loading}
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Your Datasets ({datasets.length})</CardTitle>
              <CardDescription>
                Datasets uploaded from the Upload page with preprocessing applied
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {datasets.length === 0 ? (
            <div className="text-center py-16 text-muted-foreground">
              <Database className="h-20 w-20 mx-auto mb-4 opacity-30" />
              <h3 className="text-lg font-semibold mb-2">No datasets uploaded yet</h3>
              <p className="mb-4">Upload CSV files from the Upload page to see them here</p>
              <Button onClick={() => window.location.href = '/upload'}>
                Go to Upload Page
              </Button>
            </div>
          ) : (
            <div className="space-y-4">
              {datasets.map((dataset) => (
                <div
                  key={dataset.upload_id}
                  className="flex items-center justify-between p-4 border border-border rounded-lg hover:border-primary transition-colors"
                >
                  <div className="flex items-center gap-4 flex-1">
                    {dataset.dataset_type === 'tess' ? (
                      <Satellite className="h-8 w-8 text-blue-500" />
                    ) : (
                      <Telescope className="h-8 w-8 text-purple-500" />
                    )}
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-1">
                        <h3 className="font-semibold text-lg">{dataset.filename}</h3>
                        <Badge variant={dataset.dataset_type === 'tess' ? 'default' : 'secondary'}>
                          {dataset.dataset_type.toUpperCase()}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-4 text-sm text-muted-foreground">
                        <span>{dataset.summary.processed_rows} rows</span>
                        <span>•</span>
                        <span>{dataset.summary.total_features} features</span>
                        <span>•</span>
                        <span>{formatDate(dataset.upload_date)}</span>
                      </div>
                      <div className="flex items-center gap-3 mt-2 text-xs">
                        <Badge variant="outline" className="font-mono">
                          Base: {dataset.summary.base_features}
                        </Badge>
                        <Badge variant="outline" className="font-mono">
                          Engineered: {dataset.summary.engineered_features}
                        </Badge>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleViewPreview(dataset.upload_id)}
                      disabled={loading}
                    >
                      <Eye className="h-4 w-4 mr-1" />
                      Preview
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleViewStats(dataset.upload_id)}
                      disabled={loading}
                    >
                      <BarChart3 className="h-4 w-4 mr-1" />
                      Stats
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => handleDeleteDataset(dataset.upload_id)}
                      disabled={loading}
                    >
                      <Trash2 className="h-4 w-4 text-destructive" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Preview Dialog */}
      <Dialog open={showPreviewDialog} onOpenChange={setShowPreviewDialog}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Data Preview</DialogTitle>
            <DialogDescription>
              First {previewData?.preview_rows || 0} rows of {previewData?.total_rows || 0} total rows
            </DialogDescription>
          </DialogHeader>
          {previewData && (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    {previewData.columns.slice(0, 8).map((col: string) => (
                      <TableHead key={col} className="font-mono text-xs">
                        {col}
                      </TableHead>
                    ))}
                    {previewData.columns.length > 8 && (
                      <TableHead className="text-xs">... +{previewData.columns.length - 8} more</TableHead>
                    )}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {previewData.data.map((row: any, idx: number) => (
                    <TableRow key={idx}>
                      {previewData.columns.slice(0, 8).map((col: string) => (
                        <TableCell key={col} className="font-mono text-xs">
                          {typeof row[col] === 'number' ? row[col].toFixed(4) : row[col]}
                        </TableCell>
                      ))}
                      {previewData.columns.length > 8 && (
                        <TableCell className="text-xs text-muted-foreground">...</TableCell>
                      )}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Statistics Dialog */}
      <Dialog open={showStatsDialog} onOpenChange={setShowStatsDialog}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Statistical Summary</DialogTitle>
            <DialogDescription>
              Descriptive statistics for all features
            </DialogDescription>
          </DialogHeader>
          {statsData && (
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-4">
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold">{statsData.total_rows}</div>
                    <div className="text-xs text-muted-foreground">Total Rows</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold">{statsData.total_columns}</div>
                    <div className="text-xs text-muted-foreground">Features</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold">Ready</div>
                    <div className="text-xs text-muted-foreground">Status</div>
                  </CardContent>
                </Card>
              </div>
              
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Feature</TableHead>
                      <TableHead>Mean</TableHead>
                      <TableHead>Std</TableHead>
                      <TableHead>Min</TableHead>
                      <TableHead>Max</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {Object.keys(statsData.statistics).slice(0, 15).map((feature) => (
                      <TableRow key={feature}>
                        <TableCell className="font-mono text-xs">{feature}</TableCell>
                        <TableCell className="font-mono text-xs">
                          {statsData.statistics[feature].mean?.toFixed(4) || 'N/A'}
                        </TableCell>
                        <TableCell className="font-mono text-xs">
                          {statsData.statistics[feature].std?.toFixed(4) || 'N/A'}
                        </TableCell>
                        <TableCell className="font-mono text-xs">
                          {statsData.statistics[feature].min?.toFixed(4) || 'N/A'}
                        </TableCell>
                        <TableCell className="font-mono text-xs">
                          {statsData.statistics[feature].max?.toFixed(4) || 'N/A'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Datasets;
