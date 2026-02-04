@echo off
REM Frontend startup script for Windows

echo Starting BraTS Segmentation Frontend...

cd frontend

REM Install dependencies if node_modules doesn't exist
if not exist "node_modules" (
    echo Installing dependencies...
    npm install
)

REM Start development server
npm run dev
