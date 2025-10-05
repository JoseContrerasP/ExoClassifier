import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Upload, Database, Sparkles, BarChart3, Settings, Telescope } from "lucide-react";

const Home = () => {
  const features = [
    {
      icon: Upload,
      title: "Data Upload",
      description: "Upload CSV files or manually enter exoplanet parameters for classification.",
      link: "/upload"
    },
    {
      icon: Database,
      title: "Custom Datasets",
      description: "Manage and upload your own datasets for model fine-tuning.",
      link: "/datasets"
    },
    {
      icon: Sparkles,
      title: "Prediction",
      description: "Classify exoplanets using XGBoost or Deep Neural Network models.",
      link: "/predict"
    },
    {
      icon: BarChart3,
      title: "Visualization",
      description: "Explore folded light curves, feature importance, and diagnostic metrics.",
      link: "/visualize"
    },
    {
      icon: Settings,
      title: "Fine-Tuning",
      description: "Customize models with hyperparameter tuning and user-provided training data.",
      link: "/fine-tune"
    }
  ];

  return (
    <div className="min-h-screen">
      <section className="min-h-screen flex items-center justify-center py-20 px-4 hero-space-bg relative">
        <div className="absolute inset-0 space-overlay"></div>
        <div className="container mx-auto max-w-4xl text-center relative z-10">
          <div className="flex justify-center mb-8">
            <div className="relative">
              <Telescope className="h-24 w-24 text-primary animate-pulse" />
              <div className="absolute inset-0 h-24 w-24 rounded-full bg-primary/20 blur-xl"></div>
            </div>
          </div>
          <h1 className="text-6xl md:text-7xl font-bold mb-6 cosmic-text">
            ExoClassifier
          </h1>
          <h2 className="text-2xl md:text-3xl font-semibold mb-8 text-white/90">
            AI-Powered Exoplanet Discovery
          </h2>
          <p className="text-xl text-white/80 mb-10 max-w-3xl mx-auto leading-relaxed">
            Harness the power of advanced machine learning to classify exoplanet candidates from transit photometry.
            Built for researchers, astronomers, and space enthusiasts.
          </p>
          <div className="flex flex-wrap justify-center gap-6">
            <Link to="/upload">
              <Button size="lg" className="bg-primary hover:bg-primary/80 text-white font-semibold px-8 py-4 text-lg transition-smooth hover:scale-105 hover:shadow-2xl">
                🚀 Discover Worlds
              </Button>
            </Link>
            <Link to="/predict">
              <Button size="lg" variant="outline" className="border-white/30 text-white hover:bg-white/10 font-semibold px-8 py-4 text-lg transition-smooth hover:scale-105 backdrop-blur-sm">
                ⚡ Quick Predict
              </Button>
            </Link>
          </div>
        </div>
      </section>

      <section className="py-20 px-4 bg-gradient-to-b from-background to-background/50">
        <div className="container mx-auto max-w-6xl">
          <h2 className="text-4xl font-bold text-center mb-4 cosmic-text">Platform Features</h2>
          <p className="text-center text-muted-foreground mb-12 text-lg">Comprehensive tools for exoplanet research and discovery</p>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature) => (
              <Link key={feature.title} to={feature.link}>
                <Card className="h-full card-hover-glow glass-effect cursor-pointer group">
                  <CardHeader>
                    <div className="relative mb-4">
                      <feature.icon className="h-14 w-14 text-primary group-hover:scale-110 transition-transform" />
                      <div className="absolute inset-0 h-14 w-14 rounded-full bg-primary/20 blur-lg opacity-0 group-hover:opacity-100 transition-opacity"></div>
                    </div>
                    <CardTitle className="text-xl group-hover:text-primary transition-colors">{feature.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-base leading-relaxed">
                      {feature.description}
                    </CardDescription>
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        </div>
      </section>

      <section className="py-20 px-4 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-primary/5 to-transparent"></div>
        <div className="container mx-auto max-w-5xl relative z-10">
          <Card className="glass-effect border-primary/20">
            <CardHeader className="text-center">
              <CardTitle className="text-3xl mb-4 cosmic-text">About ExoClassifier</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6 text-lg">
              <div className="grid md:grid-cols-2 gap-8">
                <div className="space-y-4">
                  <p className="text-foreground/90 leading-relaxed">
                    ExoClassifier provides state-of-the-art machine learning tools for classifying
                    exoplanet candidates from transit photometry data. Built for researchers,
                    astronomers, and space enthusiasts.
                  </p>
                  <div className="flex items-start space-x-3">
                    <Sparkles className="h-6 w-6 text-primary mt-1 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-foreground mb-2">AI Models</h4>
                      <p className="text-muted-foreground">XGBoost and Deep Neural Network models with 94%+ accuracy</p>
                    </div>
                  </div>
                </div>
                <div className="space-y-4">
                  <div className="flex items-start space-x-3">
                    <BarChart3 className="h-6 w-6 text-primary mt-1 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-foreground mb-2">Advanced Analytics</h4>
                      <p className="text-muted-foreground">Feature extraction, model interpretation, and comprehensive visualizations</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <Settings className="h-6 w-6 text-primary mt-1 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-foreground mb-2">Customization</h4>
                      <p className="text-muted-foreground">Fine-tune models with your datasets and hyperparameter optimization</p>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>
    </div>
  );
};

export default Home;
