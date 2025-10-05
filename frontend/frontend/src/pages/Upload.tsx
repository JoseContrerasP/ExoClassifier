import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Upload as UploadIcon, FileText, AlertCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

const Upload = () => {
  const { toast } = useToast();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleFileUpload = () => {
    // Backend integration point: uploadCSVData()
    toast({
      title: "File uploaded successfully",
      description: `${selectedFile?.name} has been processed.`,
    });
  };

  const handleManualSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Backend integration point: submitManualParameters()
    toast({
      title: "Parameters submitted",
      description: "Manual entry has been processed successfully.",
    });
  };

  const parameterFields = [
    { name: "koi_period", label: "Orbital Period (days)", tooltip: "Time for one complete orbit around the star" },
    { name: "koi_duration", label: "Transit Duration (hours)", tooltip: "Duration of the transit event" },
    { name: "koi_depth", label: "Transit Depth (ppm)", tooltip: "Depth of the transit in parts per million" },
    { name: "koi_prad", label: "Planetary Radius (Earth radii)", tooltip: "Planet radius relative to Earth" },
    { name: "koi_model_snr", label: "Transit SNR", tooltip: "Signal-to-noise ratio of the transit" },
    { name: "koi_num_transits", label: "Number of Transits", tooltip: "Number of observed transits" },
    { name: "koi_srad", label: "Stellar Radius (Solar radii)", tooltip: "Star radius relative to the Sun" },
    { name: "koi_steff", label: "Stellar Temperature (K)", tooltip: "Effective temperature of the host star" },
    { name: "koi_slogg", label: "Stellar Log g", tooltip: "Surface gravity of the host star" },
    { name: "koi_impact", label: "Impact Parameter", tooltip: "Sky-projected distance at transit center" },
  ];

  return (
    <div className="container mx-auto px-4 py-8 max-w-5xl">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Data Upload & Input</h1>
        <p className="text-muted-foreground">
          Upload CSV files or manually enter exoplanet parameters for classification
        </p>
      </div>

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
                  Required CSV Columns
                </h3>
                <ul className="text-sm text-muted-foreground space-y-1 ml-6 list-disc">
                  <li>koi_period (Orbital period in days)</li>
                  <li>koi_duration (Transit duration in hours)</li>
                  <li>koi_depth (Transit depth in ppm)</li>
                  <li>koi_prad (Planetary radius in Earth radii)</li>
                  <li>koi_model_snr (Signal-to-noise ratio)</li>
                  <li>Additional optional diagnostic columns</li>
                </ul>
              </div>

              <Button 
                onClick={handleFileUpload} 
                disabled={!selectedFile}
                className="w-full"
              >
                Upload and Process
              </Button>
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
                  <div className="grid md:grid-cols-2 gap-4">
                    {parameterFields.map((field) => (
                      <div key={field.name} className="space-y-2">
                        <div className="flex items-center gap-2">
                          <Label htmlFor={field.name}>{field.label}</Label>
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
                          placeholder="Enter value"
                        />
                      </div>
                    ))}
                  </div>
                </TooltipProvider>

                <Button type="submit" className="w-full">
                  Submit Parameters
                </Button>
              </form>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Upload;
