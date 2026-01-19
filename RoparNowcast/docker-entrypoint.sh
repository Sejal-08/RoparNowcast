#!/bin/bash
set -e

echo "🚀 Starting RoparNowcast Container..."

# 1. Start the Auto-Predictor in the background
echo "   -> Starting Auto-Predictor (auto_run.py)..."
python auto_run.py &

# 2. Start the Auto-Retrainer in the background
echo "   -> Starting Auto-Retrainer (auto_retrain.py)..."
python auto_retrain.py &

# 3. Start the API Server in the background
echo "   -> Starting FastAPI Server (api.py)..."
uvicorn api:app --host 0.0.0.0 --port 8000 &

# 4. Start the Dashboard (Foreground process)
echo "   -> Starting Dashboard..."
exec streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false