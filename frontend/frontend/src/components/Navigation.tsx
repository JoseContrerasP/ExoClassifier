import { NavLink } from "react-router-dom";
import { Upload, Database, Sparkles, BarChart3, Settings, Home } from "lucide-react";
import { Button } from "@/components/ui/button";

const Navigation = () => {
  const navItems = [
    { to: "/", icon: Home, label: "Home" },
    { to: "/upload", icon: Upload, label: "Upload" },
    { to: "/datasets", icon: Database, label: "Datasets" },
    { to: "/predict", icon: Sparkles, label: "Predict" },
    { to: "/visualize", icon: BarChart3, label: "Visualize" },
    { to: "/fine-tune", icon: Settings, label: "Fine-Tune" },
  ];

  return (
    <nav className="border-b border-border bg-card">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 rounded-full gradient-cosmic"></div>
            <span className="text-xl font-bold">ExoClassifier</span>
          </div>
          
          <div className="flex items-center space-x-1">
            {navItems.map((item) => (
              <NavLink key={item.to} to={item.to} end>
                {({ isActive }) => (
                  <Button
                    variant={isActive ? "default" : "ghost"}
                    size="sm"
                    className="flex items-center gap-2"
                  >
                    <item.icon className="h-4 w-4" />
                    <span className="hidden md:inline">{item.label}</span>
                  </Button>
                )}
              </NavLink>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
