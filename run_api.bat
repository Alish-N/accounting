@echo off
echo Installing required packages...
pip install -r requirements.txt

echo Creating necessary directories...
mkdir cleaned_data 2>nul
mkdir anomaly_results 2>nul
mkdir models 2>nul
mkdir evaluation_results 2>nul

echo Starting the API server...
uvicorn app:app --host 0.0.0.0 --port 8000 --reload 