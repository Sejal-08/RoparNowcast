import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure we can import from the root folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import settings

app = FastAPI(title="RoparNowcast API")

# Allow CORS for Flutter Web/Mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "online", "service": "RoparNowcast API"}

@app.get("/forecast")
def get_forecast():
    """Returns the latest forecast data as a JSON list."""
    if not os.path.exists(settings.OUTPUT_FILE):
        raise HTTPException(status_code=404, detail="Forecast data not generated yet.")
    
    try:
        df = pd.read_csv(settings.OUTPUT_FILE)
        # Convert DataFrame to a list of dictionaries (JSON records)
        data = df.to_dict(orient="records")
        return {
            "updated_at": pd.Timestamp.now().isoformat(),
            "count": len(data),
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)