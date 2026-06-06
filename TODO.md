# Stroke Detection AI Web Application - Implementation Plan

## Project Overview
Full-stack AI web application for detecting brain stroke using CT scan images and EEG time series data.

## Information Gathered
- **Frontend**: React + TailwindCSS with medical dashboard UI (light theme)
- **Backend**: FastAPI with PyTorch deep learning model
- **Model Architecture**: Hybrid ResNet18 (CT) + GCN (EEG) + Fusion Layer
- **Features**: File upload, stroke classification, lesion detection with visualization

## Plan

### Phase 1: Backend Setup
1. Create backend directory structure
2. Create requirements.txt with all Python dependencies
3. Create PyTorch hybrid model (ResNet18 + GCN)
4. Create lesion detection module using OpenCV
5. Create FastAPI main server with endpoints
6. Create uploads directory

### Phase 2: Frontend Setup
1. Create frontend directory structure
2. Create package.json with React + Vite + Tailwind
3. Configure Vite, Tailwind, PostCSS
4. Create main App.jsx component
5. Create UploadSection component
6. Create ResultDisplay component
7. Create MetricsCard component
8. Create index.html entry point
9. Create CSS styles

## Dependent Files to be Created

### Backend Files:
- backend/requirements.txt
- backend/main.py
- backend/model.py
- backend/lesion_detection.py
- backend/uploads/.gitkeep

### Frontend Files:
- frontend/package.json
- frontend/vite.config.js
- frontend/tailwind.config.js
- frontend/postcss.config.js
- frontend/index.html
- frontend/src/main.jsx
- frontend/src/App.jsx
- frontend/src/index.css
- frontend/src/components/UploadSection.jsx
- frontend/src/components/ResultDisplay.jsx
- frontend/src/components/MetricsCard.jsx

## Followup Steps
1. Install backend dependencies: `pip install -r backend/requirements.txt`
2. Install frontend dependencies: `cd frontend && npm install`
3. Run backend: `cd backend && uvicorn main --reload`
4. Run frontend: `cd frontend && npm run dev`

