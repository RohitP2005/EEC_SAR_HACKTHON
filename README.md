# SAR Image Generator - Documentation

## Project Overview

The SAR Image Generator is a web-based application that transforms optical images into Synthetic Aperture Radar (SAR) representations. This tool provides two main functionalities:

1. **Image Upload**: Convert uploaded optical images into SAR representations
2. **Live Feed**: Process real-time data feeds to continuously generate SAR images

## Table of Contents

1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Features](#features)
4. [Pages](#pages)
5. [Components](#components)
6. [Styling](#styling)
7. [Future Enhancements](#future-enhancements)

## Getting Started

### Prerequisites

- Node.js (v14.0 or higher)
- npm (v6.0 or higher)

### Installation

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd sar-image-generator
   ```

2. Install dependencies
   ```bash
   npm install
   ```

3. Start the development server
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to `http://localhost:5173` (or the port shown in your terminal)

## Project Structure

The project follows a standard React application structure:

```
src/
├── components/         # Reusable UI components
│   ├── Layout.jsx      # Main layout wrapper
│   └── ui/             # UI components from shadcn
├── pages/              # Page components
│   ├── Index.tsx       # Home page
│   ├── OptionSelection.tsx  # Selection between upload/live feed
│   ├── ImageUpload.tsx # Image upload functionality
│   ├── LiveFeed.tsx    # Live feed processing page
│   └── NotFound.tsx    # 404 page
├── App.tsx             # Main application component with routing
├── main.tsx            # Entry point
└── index.css           # Global styles
```

## Features

### 1. Image Upload

- Upload optical images (JPG, PNG, GIF)
- View the original image and its SAR representation side by side
- Process uploaded images with a simulated conversion algorithm
- Download processed SAR images

### 2. Live Feed

- Process real-time data streams
- Visualize SAR generation in real-time
- Configure processing settings (resolution, quality)
- View processing statistics (FPS, processing time)
- Save individual frames from the stream

## Pages

### Home Page (Index.tsx)

The landing page introduces the project with:
- Project title and description
- Animated radar visualization
- Key features section with icons and descriptions
- About section with project details
- "Start Project" button to begin using the tool

### Option Selection Page (OptionSelection.tsx)

Presents two main options for generating SAR images:
1. **Image Upload**: For processing individual images
2. **Live Feed**: For real-time data processing

Each option is presented as a card with an icon, description, and navigation button.

### Image Upload Page (ImageUpload.tsx)

Features:
- Drag-and-drop or click-to-upload functionality
- Side-by-side display of original and processed images
- Processing animation during conversion
- File type and size restrictions (max 10MB)
- Option to generate new SAR image or start over

### Live Feed Page (LiveFeed.tsx)

Features:
- Real-time visualization of SAR generation
- Start/pause stream controls
- Processing statistics panel showing FPS, processing time, and frame count
- Configuration options for stream settings
- Option to save the current frame

### Not Found Page (NotFound.tsx)

Custom 404 page that appears when users navigate to non-existent routes.

## Components

### Layout Component

The `Layout` component provides consistent structure across all pages with:
- Navigation
- Content area
- Responsive design for all screen sizes

### UI Components

The project uses shadcn UI components for:
- Buttons
- Cards
- Toast notifications
- Tooltips
- Form elements

## Styling

The project uses a custom design system with:

### Color Palette

- Primary: Radar-themed teal/blue gradient
- Background: Dark blue/black with subtle gradients
- Accents: Bright teal highlights

### CSS Features

- Custom radar-themed component classes in `index.css`
- Responsive design for all screen sizes
- Animations for interactive elements
- Grid patterns and radar visualizations

### Animation Effects

- Radar scanning animations
- Pulse effects
- Fade-in transitions
- Hover effects

## Future Enhancements

Potential features for future development:

1. **AI-Powered Processing**: Implement actual machine learning models for accurate SAR conversion
2. **Batch Processing**: Allow users to upload and process multiple images at once
3. **Advanced Analysis**: Add tools for analyzing SAR images and extracting data
4. **Export Options**: Support for various export formats and resolutions
5. **User Accounts**: Save processing history and preferences
6. **API Integration**: Connect to external data sources for live feed processing

## Technical Implementation Notes

- Built with React and TypeScript for type safety
- Uses React Router for navigation
- Responsive design with Tailwind CSS
- State management with React hooks
- Simulated processing with setTimeout (to be replaced with actual algorithms)

---

*This documentation will be updated as the project evolves.*
