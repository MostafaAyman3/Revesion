@echo off
REM Backend startup script for Windows

echo Starting BraTS Segmentation Backend...

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Set environment variables
set MODEL_PATH=../model_epoch_18.pth
set DEVICE=auto

REM Start server
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
