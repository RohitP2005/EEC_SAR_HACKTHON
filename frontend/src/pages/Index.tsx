import React from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  ArrowRight, 
  Radar, 
  Eye, 
  Image as ImageIcon, 
  Zap
} from 'lucide-react';
import Layout from '../components/Layout';

const Index = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: <ImageIcon className="w-8 h-8 text-radar-500" />,
      title: "SAR to Optical Conversion",
      description: "Transform complex SAR images into intuitive optical representations with our advanced algorithms."
    },
    {
      icon: <Eye className="w-8 h-8 text-radar-500" />,
      title: "Enhanced Interpretability",
      description: "Make SAR data accessible by converting it to familiar optical-like imagery."
    },
    {
      icon: <Radar className="w-8 h-8 text-radar-500" />,
      title: "Live Feed Processing",
      description: "Process real-time SAR data streams and convert them into optical visualizations."
    },
    {
      icon: <Zap className="w-8 h-8 text-radar-500" />,
      title: "Rapid Processing",
      description: "High-performance algorithms deliver quick conversions without compromising quality."
    }
  ];

  return (
    <Layout>
      <div className="flex flex-col items-center">
        {/* Hero section */}
        <section className="w-full max-w-4xl mx-auto text-center mb-16 animate-fade-in">
          <h1 className="radar-title mb-4">
            SAR to Optical Image Converter
          </h1>
          <p className="radar-subtitle max-w-2xl mx-auto">
            Transforming synthetic aperture radar data into intuitive optical representations for enhanced analysis
          </p>
          
          <div className="relative w-full h-64 md:h-80 mt-12 mb-8 rounded-xl overflow-hidden">
            <div className="absolute inset-0 bg-radar-pattern opacity-20"></div>
            <div className="absolute inset-0 bg-grid-pattern opacity-10"></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-32 h-32 border-2 border-radar-600 rounded-full flex items-center justify-center animate-pulse-slow">
                <div className="w-24 h-24 border border-radar-500 rounded-full flex items-center justify-center">
                  <div className="w-16 h-16 bg-radar-700/30 rounded-full flex items-center justify-center animate-glow">
                    <Radar className="w-8 h-8 text-radar-400 animate-radar-scan" />
                  </div>
                </div>
              </div>
            </div>
          </div>

          <button 
            onClick={() => navigate('/options')}
            className="radar-button mt-8 group"
          >
            Start Project
            <ArrowRight className="w-5 h-5 ml-1 group-hover:translate-x-1 transition-transform" />
          </button>
        </section>

        {/* Features section */}
        <section className="w-full max-w-4xl mx-auto mb-16">
          <h2 className="radar-section-title text-center mb-10">Key Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {features.map((feature, index) => (
              <div 
                key={index} 
                className="radar-card hover:border-radar-700 transition-colors"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="mb-4">{feature.icon}</div>
                <h3 className="text-xl font-bold text-white mb-2">{feature.title}</h3>
                <p className="text-night-200">{feature.description}</p>
              </div>
            ))}
          </div>
        </section>

        {/* About section */}
        <section className="w-full max-w-4xl mx-auto">
          <div className="radar-card">
            <h2 className="radar-section-title">About the Project</h2>
            <p className="text-night-200 mb-4">
              This project provides tools to convert synthetic aperture radar (SAR) imagery into optical-like
              representations, making complex radar data more accessible and interpretable for analysis.
            </p>
            <p className="text-night-200">
              SAR technology excels in all-weather conditions, and our conversion helps bridge the gap between
              radar specialists and optical imagery analysts.
            </p>
          </div>
        </section>
      </div>
    </Layout>
  );
};

export default Index;