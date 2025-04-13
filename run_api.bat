@echo off
echo ===== Fraud Detection API Setup =====
echo.

echo Checking Python installation...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python not found!
    echo Please install Python 3.7+ and try again.
    pause
    exit /b 1
)

echo.
echo Installing required packages...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Some packages might not have installed correctly.
    echo Will continue, but you might encounter errors.
    pause
)

echo.
echo Creating necessary directories...
mkdir cleaned_data 2>nul
mkdir anomaly_results 2>nul
mkdir models 2>nul
mkdir evaluation_results 2>nul
mkdir fraud_results 2>nul
mkdir logs 2>nul

echo.
echo Checking initial setup...
if not exist "cleaned_data" (
    echo Error: Could not create required directories.
    echo Please check permissions and try again.
    pause
    exit /b 1
)

echo.
echo ===== Starting the API Server =====
echo.
echo API will be available at http://localhost:8000
echo Documentation will be available at http://localhost:8000/docs
echo.
echo Quick test endpoint: http://localhost:8000/api/quick-fraud-analysis
echo.
echo Press Ctrl+C to stop the server
echo.

uvicorn app:app --host 0.0.0.0 --port 8000 --reload 