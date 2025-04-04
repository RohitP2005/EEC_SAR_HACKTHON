import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import Layout from '../components/Layout'; // Ensure this path is correct
import { MapContainer, TileLayer, Marker, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { toast } from 'sonner';

interface Position {
  lat: number;
  lng: number;
}

const LocationSelector = ({ onLocationSelected }: { onLocationSelected: (lat: number, lng: number) => void }) => {
  const [position, setPosition] = useState<[number, number] | null>(null);

  useMapEvents({
    click(e) {
      const { lat, lng } = e.latlng;
      setPosition([lat, lng]);
      onLocationSelected(lat, lng);
    },
  });

  return position ? <Marker position={position} /> : null;
};

const LiveFeed: React.FC = () => {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedLatLng, setSelectedLatLng] = useState<Position | null>(null);
  const [sarImageUrl, setSarImageUrl] = useState<string | null>(null);
  const [convertedImage, setConvertedImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (selectedLatLng) {
      generateSARImage();
    }
  }, [selectedLatLng]);

  const handleLocationSelect = (lat: number, lng: number) => {
    setSelectedLatLng({ lat, lng });
    setSarImageUrl(null);
    setError(null);
  };

  const generateSARImage = async () => {
    if (!selectedLatLng) return;
    setLoading(true);
    setError(null);

    try {
      // const res = await fetch('http://localhost:5000/get-sar-image', {
      const res = await fetch('http://127.0.0.1:8000/get-sar-image', {

        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          latitude: selectedLatLng.lat,
          longitude: selectedLatLng.lng,
        }),
      });

      const data = await res.json();
      if (data.image_url) {
        setSarImageUrl(data.image_url);
        await convertSarToOptical(data.image_url);
      } else {
        setError("Image URL not returned by server.");
      }
    } catch (err) {
      setError("Something went wrong while generating the SAR image.");
    } finally {
      setLoading(false);
    }
  };

  const convertSarToOptical = async (imageUrl: string) => {
    setLoading(true);
    setConvertedImage(null);
    setError(null);
  
    try {
      // Fetch the SAR image blob from the imageUrl
      const imageResponse = await fetch(imageUrl);
      const imageBlob = await imageResponse.blob();
  
      // Prepare FormData with the SAR image blob
      const formData = new FormData();
      formData.append("file", imageBlob, "sar_image.png"); // use key "file"
  
      toast.info("Uploading and converting SAR image...");
  
      const response = await fetch("http://127.0.0.1:8000/colorize", {
        method: "POST",
        body: formData, // No need for custom headers
      });
  
      if (!response.ok) {
        throw new Error("Conversion failed");
      }
  
      const resultBlob = await response.blob();
      const convertedImageUrl = URL.createObjectURL(resultBlob);
      setConvertedImage(convertedImageUrl);
    } catch (err) {
      console.error(err);
      setError("Something went wrong during conversion.");
    } finally {
      setLoading(false);
    }
  };
  


  return (
    <Layout>
      <div className="max-w-[1800px] mx-auto text-white px-4">
        <div className="flex items-center mb-8">
          <button
            onClick={() => navigate('/options')}
            className="flex items-center text-night-300 hover:text-radar-400 transition-colors"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            <span>Back to options</span>
          </button>
        </div>

        <h1 className="radar-title text-center">Live SAR to Optical Conversion</h1>
        <p className="text-center text-night-300 mb-12">
          Convert real-time SAR data streams into optical representations
        </p>

        <div className="grid grid-cols-1 xl:grid-cols-[3fr_1fr] gap-8">
          {/* Map Section - Wider */}
          <div className="radar-card h-[600px] overflow-hidden">
            <div className="h-full">
              <h3 className="text-lg font-medium text-white mb-4">Select Location</h3>
              <MapContainer
                center={[20, 78]}
                zoom={4}
                className="h-[90%] w-full rounded-lg z-0"
              >
                <TileLayer
                  attribution='&copy; OpenStreetMap contributors'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                <LocationSelector onLocationSelected={handleLocationSelect} />
              </MapContainer>
              {selectedLatLng && (
                <div className="mt-2 text-center">
                  <p className="text-sm">📍 Selected: {selectedLatLng.lat.toFixed(5)}, {selectedLatLng.lng.toFixed(5)}</p>
                </div>
              )}
            </div>
          </div>

          {/* Images Section - Compact */}
          <div className="flex flex-col gap-4">
            {/* SAR Image Card */}
            <div className="radar-card p-4 flex flex-col h-[240px]">
              <h2 className="text-xl font-semibold mb-2">🛰️ SAR Image</h2>
              <div className="flex-1 flex items-center justify-center overflow-hidden">
                {loading ? (
                  <div className="text-yellow-300">⏳ Processing SAR image...</div>
                ) : error ? (
                  <div className="text-red-400">{error}</div>
                ) : sarImageUrl ? (
                  <img
                    src={sarImageUrl}
                    alt="SAR Output"
                    className="max-h-[180px] max-w-full object-contain"
                  />
                ) : (
                  <div className="text-night-300">Select a location on the map</div>
                )}
              </div>
            </div>

            {/* Convert Button - Moved between images
            {sarImageUrl && (
              <button
                onClick={convertSarToOptical}
                className="bg-radar-500 hover:bg-radar-600 px-4 py-3 text-white rounded-lg text-sm w-full transition-colors disabled:opacity-60"
                disabled={isProcessing}
              >
                {isProcessing ? "Converting..." : "Convert to Optical"}
              </button>
            )} */}


            {/* Optical Image Card */}
            <div className="radar-card p-4 flex flex-col h-[240px]">
              <h2 className="text-xl font-semibold mb-2">🌄 Optical Image</h2>
              <div className="flex-1 flex items-center justify-center overflow-hidden">
                {convertedImage ? (
                  <img
                    src={convertedImage}
                    alt="Optical Output"
                    className="max-h-[180px] max-w-full object-contain"
                  />
                ) : isProcessing ? (
                  <div className="text-yellow-300">⏳ Converting image...</div>
                ) : (
                  <div className="text-night-300">SAR image will appear here first</div>
                )}
              </div>
            </div>

          </div>
        </div>
      </div>
    </Layout>
  );
};

export default LiveFeed;