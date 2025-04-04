import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, Radio, ArrowRight } from 'lucide-react';
import Layout from '../components/Layout';

const OptionSelection = () => {
  const navigate = useNavigate();

  return (
    <Layout>
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="text-center mb-12">
          <h1 className="radar-title mb-4">SAR to Optical Conversion</h1>
          <div className="w-24 h-1 bg-gradient-to-r from-radar-400 to-radar-600 mx-auto mb-6 rounded-full"></div>
          <p className="text-night-300 max-w-2xl mx-auto">
            Select your preferred method to convert SAR images to optical representations
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Image Upload Option */}
          <div 
            className="radar-card hover:border-radar-600 cursor-pointer transition-all duration-300 hover:shadow-[0_10px_25px_rgba(0,188,212,0.15)]"
            onClick={() => navigate('/image-upload')}
          >
            <div className="h-40 mb-6 flex items-center justify-center rounded-lg bg-night-800/50 overflow-hidden group">
              <div className="p-6 bg-night-700/50 rounded-full group-hover:bg-radar-500/10 transition-colors">
                <Upload className="w-12 h-12 text-radar-400 group-hover:scale-110 transition-transform" />
              </div>
            </div>
            <h2 className="text-xl font-bold text-white mb-3">Image Conversion</h2>
            <p className="text-night-200 mb-6">
              Upload SAR images from your device to convert them into optical representations using our AI algorithms.
            </p>
            <div className="flex justify-end">
              <button className="flex items-center text-radar-400 hover:text-radar-300 transition-colors group">
                <span>Get started</span>
                <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
              </button>
            </div>
          </div>

          {/* Live Feed Option */}
          <div 
            className="radar-card hover:border-radar-600 cursor-pointer transition-all duration-300 hover:shadow-[0_10px_25px_rgba(0,188,212,0.15)]"
            onClick={() => navigate('/live-feed')}
          >
            <div className="h-40 mb-6 flex items-center justify-center rounded-lg bg-night-800/50 overflow-hidden group">
              <div className="p-6 bg-night-700/50 rounded-full group-hover:bg-radar-500/10 transition-colors">
                <Radio className="w-12 h-12 text-radar-400 group-hover:scale-110 transition-transform" />
              </div>
            </div>
            <h2 className="text-xl font-bold text-white mb-3">Live Conversion</h2>
            <p className="text-night-200 mb-6">
              Process real-time SAR data streams and convert them to optical imagery for continuous monitoring.
            </p>
            <div className="flex justify-end">
              <button className="flex items-center text-radar-400 hover:text-radar-300 transition-colors group">
                <span>Get started</span>
                <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default OptionSelection;