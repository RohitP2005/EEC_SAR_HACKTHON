
import React from 'react';
import { Link } from 'react-router-dom';
import { HomeIcon, RadarIcon } from 'lucide-react';

const Layout = ({ children }) => {
  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden">
      {/* Background elements */}
      <div className="absolute -top-40 -right-40 w-96 h-96 bg-radar-900/20 rounded-full filter blur-3xl" />
      <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-radar-800/20 rounded-full filter blur-3xl" />
      
      {/* Top navigation */}
      <header className="relative z-10 px-6 py-4 border-b border-night-700 bg-night-900/80 backdrop-blur-md">
        <div className="container mx-auto flex justify-between items-center">
          <Link to="/" className="flex items-center gap-2 group">
            <div className="w-8 h-8 rounded-full bg-radar-700 flex items-center justify-center group-hover:animate-pulse-slow">
              <RadarIcon className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold text-white">SAR<span className="text-radar-500">Vision</span></span>
          </Link>
          <nav className="hidden md:flex items-center gap-6">
            <Link to="/" className="text-white/80 hover:text-radar-400 transition-colors flex items-center gap-1">
              <HomeIcon className="w-4 h-4" />
              <span>Home</span>
            </Link>
            <Link to="/options" className="text-white/80 hover:text-radar-400 transition-colors">
              Generate SAR
            </Link>
          </nav>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-grow container mx-auto py-8 px-4 relative z-10">
        {children}
      </main>
      
      {/* Footer */}
      <footer className="relative z-10 border-t border-night-700 bg-night-900/80 backdrop-blur-md px-6 py-4">
        <div className="container mx-auto text-center text-night-400">
          <p>© {new Date().getFullYear()} SAR<span className="text-radar-500">Vision</span> | Hackathon Project</p>
        </div>
      </footer>
    </div>
  );
};

export default Layout;
