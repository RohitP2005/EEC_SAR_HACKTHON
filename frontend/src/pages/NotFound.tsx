
import React, { useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { AlertTriangle, ArrowLeft } from "lucide-react";
import Layout from "../components/Layout";

const NotFound = () => {
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <Layout>
      <div className="min-h-[60vh] flex flex-col items-center justify-center">
        <div className="w-20 h-20 rounded-full bg-night-800 flex items-center justify-center mb-6">
          <AlertTriangle className="w-10 h-10 text-radar-500" />
        </div>
        <h1 className="radar-title mb-2">404</h1>
        <p className="text-xl text-night-300 mb-8">The page you're looking for doesn't exist</p>
        <button
          onClick={() => navigate('/')}
          className="radar-button"
        >
          <ArrowLeft className="w-5 h-5 mr-2" />
          Return to Home
        </button>
      </div>
    </Layout>
  );
};

export default NotFound;
