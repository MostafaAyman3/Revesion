#!/bin/bash
# Backend startup script

echo "Starting BraTS Segmentation Backend..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set environment variables
export MODEL_PATH="../model_epoch_18.pth"
export DEVICE="auto"

# Start server
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
