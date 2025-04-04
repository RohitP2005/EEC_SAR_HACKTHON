
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import OptionSelection from "./pages/OptionSelection";
import ImageUpload from "./pages/ImageUpload";
import LiveFeed from "./pages/LiveFeed";
import NotFound from "./pages/NotFound";
import 'leaflet/dist/leaflet.css';


const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/options" element={<OptionSelection />} />
          <Route path="/image-upload" element={<ImageUpload />} />
          <Route path="/live-feed" element={<LiveFeed />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
