import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, X, ArrowLeft, RefreshCw, Check } from 'lucide-react';
import Layout from '../components/Layout';
import { toast } from 'sonner';

const ImageUpload = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [convertedImage, setConvertedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [uploadedFileName, setUploadedFileName] = useState<string>("");
  const navigate = useNavigate();

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(URL.createObjectURL(file));
      setUploadedFileName(file.name);
      setSelectedFile(file);  // Save the actual file
      setConvertedImage(null);
      toast.success("SAR image uploaded successfully!");
    }
  };

  const handleConvert = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);


    toast.info("Uploading and converting SAR image...");

    try {
      const response = await fetch("http://127.0.0.1:8000/colorize", {
        method: "POST",
        body: formData, // let browser handle headers
      });

      if (!response.ok) {
        throw new Error("Conversion failed");
      }

      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setConvertedImage(imageUrl);
      toast.success("Optical image generated successfully!");
    } catch (error) {
      console.error(error);
      toast.error("Error during conversion.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleClearImage = () => {
    setSelectedImage(null);
    setConvertedImage(null);
    setUploadedFileName("");
    toast.info("Image cleared");
  };

  return (
    <Layout>
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex items-center mb-8">
          <button
            onClick={() => navigate('/options')}
            className="flex items-center text-night-300 hover:text-radar-400 transition-colors"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            <span>Back to options</span>
          </button>
        </div>

        <div className="text-center mb-12">
          <h1 className="radar-title mb-4">SAR to Optical Conversion</h1>
          <div className="w-24 h-1 bg-gradient-to-r from-radar-400 to-radar-600 mx-auto mb-6 rounded-full"></div>
          <p className="text-night-300 max-w-2xl mx-auto">
            Transform synthetic aperture radar (SAR) imagery into photorealistic optical representations
            using our advanced AI algorithms
          </p>
        </div>

        {!selectedImage ? (
          <div className="upload-zone flex flex-col items-center justify-center py-16 bg-gradient-to-b from-night-800/30 to-night-900/50 backdrop-blur-sm border border-night-700 hover:border-radar-700/50 hover:shadow-[0_0_15px_rgba(0,188,212,0.15)] transition-all duration-300">
            <input
              type="file"
              id="image-upload"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
            />
            <div className="mb-6 p-6 rounded-full bg-night-800/70 border border-night-700 group-hover:border-radar-500 transition-colors">
              <Upload className="w-12 h-12 text-radar-400" />
            </div>
            <label
              htmlFor="image-upload"
              className="radar-button cursor-pointer text-lg"
            >
              Select SAR Image
            </label>
            <p className="mt-6 text-night-400 text-sm">Supported formats: JPG, PNG or GIF (max 10MB)</p>
          </div>
        ) : (
          <div className="radar-card bg-gradient-to-b from-night-900/80 to-night-800/80 border border-night-700 shadow-[0_10px_30px_rgba(0,0,0,0.25)]">
            <div className="flex justify-between items-center mb-8 pb-4 border-b border-night-700">
              <h3 className="text-xl font-semibold text-white flex items-center">
                <span className="w-2 h-2 bg-radar-500 rounded-full mr-2"></span>
                {uploadedFileName}
              </h3>
              <button
                onClick={handleClearImage}
                className="text-night-400 hover:text-radar-400 transition-colors p-2 hover:bg-night-800/50 rounded-full"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="flex flex-col">
                <div className="flex items-center mb-3">
                  <div className="w-1 h-4 bg-radar-500 rounded-full mr-2"></div>
                  <div className="text-night-200 text-sm font-medium uppercase tracking-wider">SAR Image</div>
                </div>
                <div className="h-64 md:h-80 bg-night-800/50 rounded-lg overflow-hidden flex items-center justify-center border border-night-700 shadow-inner">
                  <img
                    src={selectedImage}
                    alt="SAR"
                    className="max-w-full max-h-full object-contain"
                  />
                </div>
              </div>

              <div className="flex flex-col">
                <div className="flex items-center mb-3">
                  <div className="w-1 h-4 bg-radar-500 rounded-full mr-2"></div>
                  <div className="text-night-200 text-sm font-medium uppercase tracking-wider">Optical Representation</div>
                </div>
                <div className="h-64 md:h-80 bg-night-800/50 rounded-lg overflow-hidden flex items-center justify-center border border-night-700 shadow-inner relative">
                  {isProcessing ? (
                    <div className="flex flex-col items-center">
                      <RefreshCw className="w-14 h-14 text-radar-500 animate-spin mb-4" />
                      <div className="text-night-300 bg-night-900/80 px-6 py-2 rounded-full border border-night-700">
                        Processing image...
                      </div>
                    </div>
                  ) : convertedImage ? (
                    <div className="h-64 md:h-80  overflow-hidden flex items-center justify-center ">
                      <img
                        src={convertedImage}
                        alt="Optical"
                        className="max-w-full max-h-full object-contain mx-auto"
                      />
                      <div className="absolute top-3 right-3 bg-radar-600 text-white rounded-full p-2 shadow-lg animate-pulse-slow">
                        <Check className="w-4 h-4" />
                      </div>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center text-night-400">
                      <RefreshCw className="w-10 h-10 mb-4 opacity-50" />
                      <p>Click "Convert to Optical" to generate</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="flex justify-center mt-10">
              <button
                onClick={handleConvert}
                disabled={isProcessing}
                className={`radar-button text-lg ${isProcessing ? 'opacity-70 cursor-not-allowed' : 'animate-pulse-slow'}`}
              >
                {isProcessing ? (
                  <>
                    <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : convertedImage ? (
                  <>
                    <RefreshCw className="w-5 h-5 mr-2" />
                    Generate New Optical Image
                  </>
                ) : (
                  <>
                    <RefreshCw className="w-5 h-5 mr-2" />
                    Convert to Optical
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        <div className="mt-16 bg-night-800/30 border border-night-700 rounded-lg p-6 max-w-3xl mx-auto">
          <h3 className="text-lg font-medium text-radar-400 mb-3">About SAR to Optical Conversion</h3>
          <p className="text-night-300 mb-4">
            Synthetic Aperture Radar (SAR) imagery provides valuable data in all weather conditions and at night,
            but can be difficult to interpret. Our AI transforms these complex radar images into natural-looking
            optical representations for easier analysis.
          </p>
        </div>
      </div>
    </Layout>
  );
};

export default ImageUpload;