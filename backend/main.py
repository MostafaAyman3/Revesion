"""
FastAPI Backend for Brain Tumor Segmentation Web App
"""
import os
import sys
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
import numpy as np
import base64
import gzip
import json

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from inference import (
    BraTSInferencePipeline, 
    generate_3d_visualization_data,
    generate_2d_visualization_data,
    generate_3d_visualization_threejs
)

# Configuration - check multiple possible locations
MODEL_PATH = os.environ.get("MODEL_PATH", None)
DEVICE = os.environ.get("DEVICE", "auto")

# Global pipeline instance
pipeline: Optional[BraTSInferencePipeline] = None


def find_model_path():
    """Find model file in possible locations."""
    possible_paths = [
        Path(MODEL_PATH) if MODEL_PATH else None,
        Path(__file__).parent.parent / "brats_unet2d_final (15).pth",
        Path(__file__).parent.parent.parent / "brats_unet2d_final (15).pth",
        Path("D:/Data_Projects/brats_unet2d_final (15).pth"),
        Path("brats_unet2d_final (15).pth"),
    ]
    
    for p in possible_paths:
        if p and p.exists():
            return p
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global pipeline
    try:
        model_path = find_model_path()
        
        if not model_path:
            print("Warning: Model not found in any expected location")
            print("API will start but inference will fail until model is available.")
        else:
            print(f"Found model at: {model_path}")
            pipeline = BraTSInferencePipeline(str(model_path), DEVICE)
            print("Pipeline initialized successfully!")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
    
    yield
    
    # Cleanup on shutdown
    pipeline = None


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="BraTS Tumor Segmentation API",
    description="Brain Tumor Segmentation using ResUNet2D for INSTANT-ODC AI Hackathon",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class PredictionResponse(BaseModel):
    success: bool
    message: str
    patient_id: Optional[str] = None
    visualization_data: Optional[dict] = None
    visualization_2d: Optional[dict] = None
    visualization_3d: Optional[dict] = None
    metrics: Optional[dict] = None
    rle: Optional[dict] = None
    volume_shape: Optional[list] = None
    tumor_volumes: Optional[dict] = None
    tumor_volumes: Optional[dict] = None


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=pipeline is not None,
        device=str(pipeline.device) if pipeline else "not loaded"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return await root()


def validate_nifti_file(file: UploadFile) -> None:
    """Validate uploaded file is NIfTI format."""
    filename = file.filename.lower()
    if not (filename.endswith('.nii') or filename.endswith('.nii.gz')):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format: {file.filename}. Expected .nii or .nii.gz"
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    flair: UploadFile = File(..., description="FLAIR modality NIfTI file"),
    t1: UploadFile = File(..., description="T1 modality NIfTI file"),
    t1ce: UploadFile = File(..., description="T1ce modality NIfTI file"),
    t2: UploadFile = File(..., description="T2 modality NIfTI file"),
    patient_id: str = Form(default="Patient", description="Patient identifier"),
    return_rle: bool = Form(default=True, description="Return RLE encodings")
):
    """
    Run brain tumor segmentation inference.
    
    Accepts 4 NIfTI files (FLAIR, T1, T1ce, T2) and returns:
    - 2D slice visualization data (base64 images)
    - 3D mesh visualization data for Three.js
    - Tumor volume statistics and metrics
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    # Validate all files
    for file in [flair, t1, t1ce, t2]:
        validate_nifti_file(file)
    
    try:
        import time
        start_time = time.time()
        
        # Read file contents
        flair_bytes = await flair.read()
        t1_bytes = await t1.read()
        t1ce_bytes = await t1ce.read()
        t2_bytes = await t2.read()
        
        # Run inference (with T1ce for brain shell)
        result = pipeline.predict_from_bytes(
            flair_bytes, t1_bytes, t1ce_bytes, t2_bytes,
            return_rle=return_rle,
            return_t1ce=True
        )
        
        inference_time = time.time() - start_time
        
        prediction = result["prediction"]
        modalities = result.get("modalities", {})
        t1ce_volume = result.get("t1ce_volume", None)
        
        # Generate visualization data (old format for backward compatibility)
        visualization_data = generate_3d_visualization_data(
            prediction, 
            t1ce_volume=t1ce_volume
        )
        
        # Generate 2D slice visualization (new frontend)
        visualization_2d = generate_2d_visualization_data(
            modalities,
            prediction,
            step=1  # Include all slices
        )
        
        # Generate 3D mesh data for Three.js (new frontend)
        visualization_3d = generate_3d_visualization_threejs(
            prediction,
            t1ce_volume=t1ce_volume
        )
        
        # Calculate tumor volumes (in voxels)
        tumor_volumes = {
            "necrotic_core": int(np.sum(prediction == 1)),
            "edema": int(np.sum(prediction == 2)),
            "enhancing_tumor": int(np.sum(prediction == 4)),
            "total_tumor": int(np.sum(prediction > 0))
        }
        
        # Calculate metrics for new frontend
        total = tumor_volumes["total_tumor"]
        enhancing_pct = (tumor_volumes["enhancing_tumor"] / total * 100) if total > 0 else 0
        
        metrics = {
            "volume": total,
            "enhancing_percentage": round(enhancing_pct, 1),
            "inference_time": f"{inference_time:.2f}s"
        }
        
        return PredictionResponse(
            success=True,
            message="Segmentation completed successfully",
            patient_id=patient_id,
            visualization_data=visualization_data,
            visualization_2d=visualization_2d,
            visualization_3d=visualization_3d,
            metrics=metrics,
            rle=result.get("rle"),
            volume_shape=list(result["original_shape"]),
            tumor_volumes=tumor_volumes
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )


@app.post("/predict/volume")
async def predict_with_volume(
    flair: UploadFile = File(...),
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...)
):
    """
    Run inference and return compressed prediction volume.
    
    Returns the full 3D prediction volume as compressed base64.
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    # Validate all files
    for file in [flair, t1, t1ce, t2]:
        validate_nifti_file(file)
    
    try:
        flair_bytes = await flair.read()
        t1_bytes = await t1.read()
        t1ce_bytes = await t1ce.read()
        t2_bytes = await t2.read()
        
        result = pipeline.predict_from_bytes(
            flair_bytes, t1_bytes, t1ce_bytes, t2_bytes,
            return_rle=False
        )
        
        # Compress prediction volume
        prediction = result["prediction"]
        compressed = gzip.compress(prediction.tobytes())
        encoded = base64.b64encode(compressed).decode('utf-8')
        
        return JSONResponse({
            "success": True,
            "prediction_base64": encoded,
            "shape": list(prediction.shape),
            "dtype": str(prediction.dtype)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
