# SAR Image Generator - Backend Integration Guide

## Overview

This document provides backend developers with the necessary information to integrate their services with the SAR Image Generator frontend. The frontend is built with React, TypeScript, and Tailwind CSS, and requires specific API endpoints to function correctly.

## Table of Contents

1. [API Endpoints](#api-endpoints)
2. [Data Models](#data-models)
3. [Authentication](#authentication)
4. [Image Processing Requirements](#image-processing-requirements)
5. [Live Feed Integration](#live-feed-integration)
6. [Error Handling](#error-handling)
7. [Example Implementations](#example-implementations)

## API Endpoints

The frontend expects the following API endpoints:

### Image Processing

- **Endpoint**: `/api/process-image`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Request Body**:
  ```
  {
    "image": [Binary File],
    "processingOptions": {
      "resolution": string,
      "quality": number,
      "algorithm": string
    }
  }
  ```
- **Response**:
  ```json
  {
    "id": "unique-id",
    "originalUrl": "url-to-original-image",
    "processedUrl": "url-to-processed-sar-image",
    "metadata": {
      "processingTime": "time-in-ms",
      "algorithm": "algorithm-used",
      "timestamp": "ISO-date-string"
    }
  }
  ```

### Live Feed Processing

- **Endpoint**: `/api/live-feed`
- **Method**: WebSocket
- **Connection Parameters**:
  ```json
  {
    "resolution": "string",
    "quality": "number",
    "frameRate": "number"
  }
  ```
- **WebSocket Messages**:
  - **Client to Server**: Configuration updates
  - **Server to Client**: Processed frame data
    ```json
    {
      "frameId": "unique-id",
      "frameData": "base64-encoded-image",
      "metadata": {
        "timestamp": "ISO-date-string",
        "processingTime": "time-in-ms",
        "frameNumber": "number"
      }
    }
    ```

### Status Checking

- **Endpoint**: `/api/status`
- **Method**: GET
- **Response**:
  ```json
  {
    "status": "online|processing|offline",
    "load": "percentage",
    "uptime": "time-in-seconds",
    "message": "optional-status-message"
  }
  ```

## Data Models

### Image Processing Request

```typescript
interface ProcessingOptions {
  resolution: string; // e.g., "high", "medium", "low"
  quality: number; // 1-100
  algorithm: string; // e.g., "standard", "enhanced", "detailed"
}

interface ImageProcessingRequest {
  image: File;
  processingOptions: ProcessingOptions;
}
```

### Image Processing Response

```typescript
interface ProcessingMetadata {
  processingTime: number; // milliseconds
  algorithm: string;
  timestamp: string; // ISO format date string
}

interface ImageProcessingResponse {
  id: string;
  originalUrl: string;
  processedUrl: string;
  metadata: ProcessingMetadata;
}
```

### Live Feed Message

```typescript
interface LiveFeedFrame {
  frameId: string;
  frameData: string; // base64 encoded image
  metadata: {
    timestamp: string;
    processingTime: number;
    frameNumber: number;
  }
}
```

## Authentication

The frontend expects a token-based authentication system:

- **Endpoint**: `/api/auth/login`
- **Method**: POST
- **Request Body**:
  ```json
  {
    "username": "string",
    "password": "string"
  }
  ```
- **Response**:
  ```json
  {
    "token": "jwt-token",
    "expiresIn": "seconds",
    "user": {
      "id": "user-id",
      "name": "user-name",
      "role": "role-name"
    }
  }
  ```

Authentication tokens should be included in the header of all API requests:
```
Authorization: Bearer <token>
```

## Image Processing Requirements

The frontend uploads images for SAR conversion with the following constraints:

- **Supported formats**: JPG, PNG, GIF
- **Maximum file size**: 10MB
- **Expected processing time**: The frontend displays a loading animation during processing, but ideally processing should complete within 30 seconds for optimal user experience.

## Live Feed Integration

The live feed feature uses WebSocket for real-time processing. Backend requirements:

1. **Connection handling**: Maintain persistent WebSocket connections
2. **Frame rate control**: Respect the `frameRate` parameter sent by the client
3. **Error recovery**: Auto-reconnect protocols in case of connection failures
4. **Processing statistics**: Include processing metadata with each frame

WebSocket endpoint: `/api/live-feed`

## Error Handling

The API should use standard HTTP status codes and include detailed error messages:

```json
{
  "error": true,
  "code": "ERROR_CODE",
  "message": "Human-readable error message",
  "details": {
    // Additional error details if applicable
  }
}
```

Common error codes that the frontend handles:
- `INVALID_FILE_TYPE`: File format not supported
- `FILE_TOO_LARGE`: File exceeds size limit
- `PROCESSING_FAILED`: Image processing error
- `AUTHENTICATION_FAILED`: Invalid credentials
- `SESSION_EXPIRED`: Authentication token expired

## Example Implementations

### Sample Python Flask API

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import json
import base64
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
CORS(app)

@app.route('/api/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({
            'error': True,
            'code': 'NO_FILE_UPLOADED',
            'message': 'No image file uploaded'
        }), 400
    
    file = request.files['image']
    options = json.loads(request.form.get('processingOptions', '{}'))
    
    # Process image logic here
    # ...
    
    # Simulate processing time
    time.sleep(2)
    
    # Return processed image information
    return jsonify({
        'id': 'img-12345',
        'originalUrl': '/static/uploads/original.jpg',
        'processedUrl': '/static/processed/sar-image.jpg',
        'metadata': {
            'processingTime': 2000,
            'algorithm': options.get('algorithm', 'standard'),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ')
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### Sample Node.js Express API

```javascript
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const WebSocket = require('ws');
const http = require('http');

const app = express();
app.use(cors());
app.use(express.json());

const upload = multer({ dest: 'uploads/' });

// Image processing endpoint
app.post('/api/process-image', upload.single('image'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({
            error: true,
            code: 'NO_FILE_UPLOADED',
            message: 'No image file uploaded'
        });
    }
    
    const options = req.body.processingOptions ? 
        JSON.parse(req.body.processingOptions) : {};
    
    // Process image logic here
    // ...
    
    // Return processed image data
    res.json({
        id: `img-${Date.now()}`,
        originalUrl: `/static/uploads/${req.file.filename}`,
        processedUrl: `/static/processed/sar-${req.file.filename}`,
        metadata: {
            processingTime: 1500,
            algorithm: options.algorithm || 'standard',
            timestamp: new Date().toISOString()
        }
    });
});

// Status endpoint
app.get('/api/status', (req, res) => {
    res.json({
        status: 'online',
        load: 42,
        uptime: process.uptime(),
        message: 'System operating normally'
    });
});

// WebSocket server for live feed
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

wss.on('connection', (ws) => {
    console.log('Client connected');
    
    ws.on('message', (message) => {
        const config = JSON.parse(message);
        console.log('Received configuration:', config);
        
        // Setup interval to send frames based on frameRate
        const frameRate = config.frameRate || 10;
        const interval = 1000 / frameRate;
        
        const intervalId = setInterval(() => {
            // Generate or process frame data here
            // ...
            
            const frameData = {
                frameId: `frame-${Date.now()}`,
                frameData: 'base64-encoded-image-data',
                metadata: {
                    timestamp: new Date().toISOString(),
                    processingTime: 50,
                    frameNumber: frameCount++
                }
            };
            
            ws.send(JSON.stringify(frameData));
        }, interval);
        
        ws.on('close', () => {
            clearInterval(intervalId);
            console.log('Client disconnected');
        });
    });
});

let frameCount = 0;

server.listen(5000, () => {
    console.log('Server listening on port 5000');
});
```

## Next Steps for Integration

1. **Set up a development environment** with both frontend and backend running
2. **Implement mock API endpoints** based on the specifications above
3. **Test authentication flows** between systems
4. **Iterate on image processing algorithms** while maintaining API contracts
5. **Document any API changes** and communicate them to the frontend team

## Contact

For questions regarding frontend integration, please contact:
- Frontend Lead: [Your Name]
- Email: [Your Email]
- Project Repository: [Repository URL]

---

*This documentation will be updated as the frontend and API requirements evolve.*
