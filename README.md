# Brain Stroke Detection AI System

A full-stack AI web application for detecting brain stroke using CT scan images and EEG time series data.

## Project Overview

This application uses a hybrid deep learning model combining:
- **ResNet18** for CT Image feature extraction
- **Graph Convolutional Network (GCN)** for EEG time series data
- **Fusion layer** for final classification

## Features

- Upload CT brain scan images (PNG, JPG, JPEG)
- Upload EEG data as Excel files (.xlsx)
- AI-powered stroke detection (No Stroke, Hemorrhagic Stroke, Ischemic Stroke)
- Lesion visualization with highlighted regions
- Model performance metrics (Precision, Recall, F1 Score)
- Modern medical dashboard UI with light theme

## Tech Stack

### Frontend
- React 18
- TailwindCSS
- Vite
- Lucide React (icons)

### Backend
- FastAPI
- PyTorch (ResNet18 + GCN)
- OpenCV (lesion detection)
- Pandas (EEG data processing)

## Project Structure

```
Stroke Detection/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── model.py             # Hybrid PyTorch model
│   ├── lesion_detection.py # OpenCV lesion detection
│   ├── requirements.txt    # Python dependencies
│   ├── uploads/            # Uploaded files
│   └── output/             # Processed images
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── UploadSection.jsx
│   │   │   ├── ResultDisplay.jsx
│   │   │   └── MetricsCard.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── index.html
├── TODO.md
└── README.md
```

## Installation & Running

### Prerequisites
- Python 3.10+
- Node.js 18+
- npm or yarn

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Start the backend server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will run at: http://localhost:8000

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install npm dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will run at: http://localhost:5173

## API Endpoints

- `GET /` - Root endpoint with API info
- `GET /health` - Health check endpoint
- `POST /analyze` - Upload CT and EEG files for stroke detection
  - Request: Multipart form data with `ct_image` and `eeg_file`
  - Response: JSON with stroke classification, confidence, and metrics
- `GET /uploads/{filename}` - Serve uploaded files
- `GET /output/{filename}` - Serve output images

## Response Format

```json
{
  "result": "Hemorrhagic Stroke",
  "confidence": 97.1,
  "precision": 93.1,
  "recall": 92.2,
  "f1_score": 0.94,
  "lesion_image": "path/to/lesion_image.png",
  "lesion_detected": true
}
```

## Usage

1. Start both backend and frontend servers
2. Open http://localhost:5173 in your browser
3. Upload a CT brain scan image
4. Upload an EEG Excel file
5. Click "Analyze & Detect Stroke"
6. View the results with classification and lesion visualization

## License

This project is for educational and research purposes. Always consult with healthcare professionals for medical diagnosis.

