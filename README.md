# BraTS Brain Tumor Segmentation Web App

## INSTANT-ODC AI Hackathon 2026

A production-ready web application for Brain Tumor Segmentation using deep learning. Upload 4 MRI modalities and get instant 3D visualization of tumor regions.

![BraTS Segmentation](https://img.shields.io/badge/BraTS-Segmentation-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)
![React](https://img.shields.io/badge/React-18.2-blue)

---

## 📋 Project Overview

This application provides automated brain tumor segmentation from multimodal MRI scans using a ResUNet2D deep learning model trained on BraTS (Brain Tumor Segmentation) challenge data.

### Features

- **4-Channel MRI Input**: Supports FLAIR, T1, T1ce, and T2 modalities
- **Real-time Inference**: Fast slice-wise segmentation using GPU/CPU
- **Interactive 3D Visualization**: Plotly-based 3D scatter plot with rotation, zoom, and pan
- **Tumor Statistics**: Volume quantification for each tumor region
- **Professional UI**: Modern medical-grade interface with drag & drop upload

### Tumor Classes

| Class | Label | Color | Description |
|-------|-------|-------|-------------|
| 1 | Necrotic Core | 🔴 Red | Dead tumor tissue |
| 2 | Edema | 🟢 Green | Swelling around tumor |
| 4 | Enhancing Tumor | 🟡 Gold | Active tumor with contrast enhancement |

---

## 🧠 Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. LOAD NIfTI FILES                                           │
│     ├── FLAIR (.nii / .nii.gz)                                 │
│     ├── T1                                                      │
│     ├── T1ce (contrast-enhanced)                               │
│     └── T2                                                      │
│                                                                 │
│  2. PREPROCESSING                                               │
│     ├── Smart NIfTI loading (handle orientations)              │
│     ├── Create brain mask (non-zero voxels)                    │
│     └── BraTS normalization (z-score on brain region)          │
│                                                                 │
│  3. SLICE-WISE INFERENCE                                        │
│     ├── For each axial slice:                                  │
│     │   ├── Resize to 240×240                                  │
│     │   ├── Forward pass through ResUNet2D                     │
│     │   ├── Argmax for class prediction                        │
│     │   └── Resize back to original dimensions                 │
│     └── Stack into 3D volume                                   │
│                                                                 │
│  4. POST-PROCESSING                                             │
│     ├── Apply brain mask                                       │
│     ├── Map class 3 → label 4 (BraTS convention)              │
│     └── Generate statistics                                    │
│                                                                 │
│  5. VISUALIZATION                                               │
│     ├── Extract voxel coordinates per class                    │
│     ├── Subsample for browser performance                      │
│     └── Return Plotly-compatible 3D scatter data               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

### Model: ResUNet2D

```
Input (4, 240, 240) → Encoder → Bridge → Decoder → Output (4, 240, 240)
      ↓                                              ↓
   4 channels                                    4 classes
   (FLAIR,T1,T1ce,T2)                     (BG, NC, ED, ET)
```

- **Encoder**: 4 residual blocks with max pooling
- **Bridge**: Deep residual block (512 channels)
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: 1×1 convolution for pixel-wise classification

### Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + PyTorch |
| Frontend | React + Tailwind CSS |
| Visualization | Plotly.js |
| Medical Imaging | NiBabel |

---

## 🚀 How to Run

### Prerequisites

- Python 3.9+
- Node.js 18+
- CUDA (optional, for GPU acceleration)

### 1. Clone & Setup

```bash
cd Revesion
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Ensure model file exists
# model_epoch_18.pth should be in the project root
```

### 3. Start Backend

```bash
cd backend
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend will be available at: `http://localhost:8000`

API docs at: `http://localhost:8000/docs`

### 4. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will be available at: `http://localhost:3000`

### 5. Production Build

```bash
# Frontend
cd frontend
npm run build

# Serve with any static server
npx serve dist
```

---

## 📡 API Endpoints

### Health Check
```http
GET /health
```

### Run Segmentation
```http
POST /predict
Content-Type: multipart/form-data

flair: <.nii/.nii.gz file>
t1: <.nii/.nii.gz file>
t1ce: <.nii/.nii.gz file>
t2: <.nii/.nii.gz file>
return_rle: true/false
```

**Response:**
```json
{
  "success": true,
  "message": "Segmentation completed successfully",
  "visualization_data": {
    "classes": [
      {
        "class_id": 1,
        "label": "Necrotic Core",
        "color": "red",
        "x": [...],
        "y": [...],
        "z": [...],
        "count": 12345
      }
    ]
  },
  "tumor_volumes": {
    "necrotic_core": 12345,
    "edema": 54321,
    "enhancing_tumor": 6789,
    "total_tumor": 73455
  },
  "volume_shape": [240, 240, 155]
}
```

---

## 📁 Project Structure

```
Revesion/
├── model_epoch_18.pth          # Trained model weights
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── backend/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── model.py                # ResUNet2D architecture
│   ├── preprocessing.py        # NIfTI loading & normalization
│   └── inference.py            # Inference pipeline
│
└── frontend/
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    ├── index.html
    │
    ├── public/
    │   └── brain-icon.svg
    │
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── index.css
        │
        ├── api/
        │   └── inference.js
        │
        └── components/
            ├── Header.jsx
            ├── FileUpload.jsx
            ├── LoadingState.jsx
            ├── Visualization3D.jsx
            └── TumorStats.jsx
```

---

## 🔧 Configuration

### Environment Variables

**Backend:**
```bash
MODEL_PATH=../model_epoch_18.pth  # Path to model weights
DEVICE=auto                        # "auto", "cuda", or "cpu"
```

**Frontend:**
```bash
VITE_API_URL=http://localhost:8000  # Backend API URL
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Inference Time (GPU) | ~5-10 seconds |
| Inference Time (CPU) | ~30-60 seconds |
| Memory (GPU) | ~2-4 GB |
| Memory (CPU) | ~4-8 GB |

---

## 🏆 INSTANT-ODC AI Hackathon

This project was developed for the INSTANT-ODC AI Hackathon 2026, focusing on medical AI applications for brain tumor segmentation and visualization.

---

## 📄 License

MIT License

---

## 👥 Authors

INSTANT-ODC AI Hackathon Team

---

## 🔗 References

- [BraTS Challenge](https://www.synapse.org/brats)
- [NiBabel Documentation](https://nipy.org/nibabel/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Plotly.js](https://plotly.com/javascript/)
