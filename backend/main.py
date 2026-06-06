"""
FastAPI Backend Server for Stroke Detection
Handles file uploads, preprocessing, model inference, and lesion detection
"""

import os
import io
import uuid
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Import custom modules
from model import HybridStrokeModel, mock_predict
from lesion_detection import detect_lesion_ct, highlight_lesion_region, preprocess_ct_image

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Create output directory for processed images
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Model instance
model = None


def load_model():
    """Load the hybrid stroke detection model"""
    global model
    model = HybridStrokeModel(num_classes=3)
    model.eval()
    print("Model loaded successfully!")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler (replaces deprecated on_event)"""
    load_model()
    yield


# Create FastAPI app
app = FastAPI(
    title="Stroke Detection API",
    description="AI-powered brain stroke detection using CT scans and EEG data",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisResponse(BaseModel):
    result: str
    confidence: float
    precision: float
    recall: float
    f1_score: float
    lesion_image: str = None
    lesion_detected: bool = False


def preprocess_ct(file_content: bytes, target_size=(224, 224)) -> torch.Tensor:
    """Preprocess CT image for model input"""
    image = Image.open(io.BytesIO(file_content))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize(target_size, Image.Resampling.BILINEAR)

    image_array = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std

    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

    return image_tensor.unsqueeze(0)


def preprocess_eeg(file_content: bytes) -> torch.Tensor:
    """
    Preprocess EEG Excel file for model input.
    Handles various Excel layouts: plain numeric, header rows,
    metadata rows at top, text-encoded numbers, multi-sheet files.
    """

    def try_parse(raw_bytes, header_row):
        """Attempt to read Excel with a given header row index."""
        df = pd.read_excel(io.BytesIO(raw_bytes), header=header_row)
        df = df.dropna(how='all')       # drop rows that are entirely empty
        df = df.loc[:, ~df.columns.duplicated()]  # drop duplicate columns

        # Keep only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        # If nothing numeric, try coercing every column
        if numeric_df.empty:
            coerced = df.apply(pd.to_numeric, errors='coerce')
            coerced = coerced.dropna(axis=1, how='all')   # drop all-NaN columns
            coerced = coerced.dropna(how='all')            # drop all-NaN rows
            numeric_df = coerced

        # Drop columns that are still more than 50% NaN after coercion
        threshold = len(numeric_df) * 0.5
        numeric_df = numeric_df.dropna(axis=1, thresh=int(threshold))
        numeric_df = numeric_df.dropna()

        return numeric_df

    # --- Step 1: Try reading with standard header (row 0) ---
    numeric_df = try_parse(file_content, header_row=0)

    # --- Step 2: If empty, try skipping one metadata row ---
    if numeric_df.empty:
        print("EEG parse attempt 1 failed, trying header=1")
        numeric_df = try_parse(file_content, header_row=1)

    # --- Step 3: Try skipping two metadata rows ---
    if numeric_df.empty:
        print("EEG parse attempt 2 failed, trying header=2")
        numeric_df = try_parse(file_content, header_row=2)

    # --- Step 4: Try reading every sheet and pick the largest numeric one ---
    if numeric_df.empty:
        print("EEG parse attempt 3 failed, scanning all sheets")
        xl = pd.ExcelFile(io.BytesIO(file_content))
        best = pd.DataFrame()
        for sheet in xl.sheet_names:
            df = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet, header=0)
            df = df.dropna(how='all')
            candidate = df.select_dtypes(include=[np.number]).dropna()
            if candidate.empty:
                coerced = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').dropna()
                candidate = coerced
            if len(candidate) > len(best):
                best = candidate
        numeric_df = best

    # --- Final check ---
    if numeric_df.empty or numeric_df.size == 0:
        raise ValueError(
            "No valid EEG data found in file. "
            "Make sure the Excel file contains numeric EEG signal values. "
            f"Tried header rows 0, 1, 2 and all sheets."
        )

    print(f"EEG data parsed successfully: shape={numeric_df.shape}")

    eeg_data = numeric_df.values.astype(np.float32)

    # --- Normalize each channel (z-score) ---
    col_std = eeg_data.std(axis=0)
    col_std[col_std < 1e-8] = 1.0          # avoid divide-by-zero for flat channels
    eeg_normalized = (eeg_data - eeg_data.mean(axis=0)) / col_std

    # --- Resample to fixed length (1000 timepoints) ---
    target_length = 1000
    current_length = eeg_normalized.shape[0]

    if current_length < target_length:
        repeats = int(np.ceil(target_length / current_length))
        eeg_normalized = np.tile(eeg_normalized, (repeats, 1))

    if eeg_normalized.shape[0] > target_length:
        indices = np.linspace(0, eeg_normalized.shape[0] - 1, target_length, dtype=int)
        eeg_normalized = eeg_normalized[indices, :]

    # --- Ensure exactly 19 channels ---
    num_channels = eeg_normalized.shape[1]

    if num_channels < 19:
        padding = np.zeros((eeg_normalized.shape[0], 19 - num_channels), dtype=np.float32)
        eeg_normalized = np.hstack([eeg_normalized, padding])
    elif num_channels > 19:
        eeg_normalized = eeg_normalized[:, :19]

    # --- Convert to tensor (19, 1000) ---
    eeg_tensor = torch.from_numpy(eeg_normalized.T)   # (19, 1000)

    return eeg_tensor.unsqueeze(0)   # (1, 19, 1000)


def generate_mock_results(result_class: str):
    """Generate mock metrics for demonstration"""
    import random

    if result_class == "No Stroke":
        confidence = random.uniform(85, 98)
    else:
        confidence = random.uniform(88, 99)

    if result_class == "No Stroke":
        precision = random.uniform(92, 97)
        recall = random.uniform(90, 96)
        f1 = random.uniform(0.91, 0.96)
    elif result_class == "Hemorrhagic Stroke":
        precision = random.uniform(90, 96)
        recall = random.uniform(88, 95)
        f1 = random.uniform(0.89, 0.95)
    else:  # Ischemic Stroke
        precision = random.uniform(88, 95)
        recall = random.uniform(86, 94)
        f1 = random.uniform(0.87, 0.94)

    return {
        "precision": round(precision, 1),
        "recall": round(recall, 1),
        "f1_score": round(f1, 2)
    }


@app.get("/")
async def root():
    return {
        "message": "Stroke Detection API is running",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "POST /analyze - Upload CT and EEG files for stroke detection"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_stroke(
    ct_image: UploadFile = File(...),
    eeg_file: UploadFile = File(...)
):
    """
    Analyze CT scan and EEG data for stroke detection.
    """
    try:
        analysis_id = str(uuid.uuid4())

        ct_path = UPLOAD_DIR / f"ct_{analysis_id}.png"
        eeg_path = UPLOAD_DIR / f"eeg_{analysis_id}.xlsx"

        ct_content = await ct_image.read()
        with open(ct_path, "wb") as f:
            f.write(ct_content)

        eeg_content = await eeg_file.read()
        with open(eeg_path, "wb") as f:
            f.write(eeg_content)

        # Preprocess inputs
        ct_tensor = preprocess_ct(ct_content)
        eeg_tensor = preprocess_eeg(eeg_content)

        # Run mock prediction (swap for real model inference when ready)
        result_class, confidence, _ = mock_predict()

        # For real inference uncomment:
        # with torch.no_grad():
        #     result_class, confidence, probs = model.predict(ct_tensor, eeg_tensor)

        metrics = generate_mock_results(result_class)

        lesion_detected = result_class != "No Stroke"
        lesion_image_path = None

        if lesion_detected:
            output_path = OUTPUT_DIR / f"lesion_{analysis_id}.png"
            try:
                highlight_lesion_region(str(ct_path), result_class, str(output_path))
                lesion_image_path = str(output_path)
            except Exception as e:
                print(f"Lesion detection error: {e}")
                import shutil
                fallback_path = OUTPUT_DIR / f"fallback_{analysis_id}.png"
                shutil.copy(ct_path, fallback_path)
                lesion_image_path = str(fallback_path)

        return AnalysisResponse(
            result=result_class,
            confidence=round(confidence, 1),
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            lesion_image=lesion_image_path,
            lesion_detected=lesion_detected
        )

    except ValueError as e:
        # Surface EEG/CT parsing errors clearly to the frontend
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/output/{filename}")
async def get_output_file(filename: str):
    file_path = OUTPUT_DIR / filename
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)