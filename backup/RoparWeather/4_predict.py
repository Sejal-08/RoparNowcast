import requests
import pandas as pd
import joblib
import os
from datetime import datetime

# CONFIG
LAT, LON = 30.96, 76.47
MODEL_DIR = "models"

def get_live_forecast():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT, "longitude": LON,
        # ✅ Fetch Solar & Wind Direction for future
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,rain,shortwave_radiation,wind_direction_10m",
        "timezone": "auto",
        "forecast_days": 3
    }
    try:
        data = requests.get(url, params=params).json()
        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        return df.rename(columns={
            'temperature_2m': 'om_temp', 'relative_humidity_2m': 'om_hum',
            'surface_pressure': 'om_press', 'wind_speed_10m': 'om_wind', 'rain': 'om_rain',
            'shortwave_radiation': 'om_solar', 'wind_direction_10m': 'om_wind_dir'
        })
    except Exception as e:
        print(f"❌ Forecast API Failed: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = get_live_forecast()
    if df.empty: exit()

    df['month'] = df['time'].dt.month
    df['hour'] = df['time'].dt.hour
    
    print("\n📍 ROPAR HYPER-LOCAL FORECAST (Next 72 Hours)")
    print(f"{'Time':<20} | {'Global':<7} | {'YOUR AI':<7} | {'Diff'}")
    print("-" * 55)

    for i, row in df.iterrows():
        month = int(row['month'])
        model_path = f"{MODEL_DIR}/rf_month_{month}.pkl"
        
        # ✅ Features must match training EXACTLY
        features = [[
            row['om_temp'], row['om_hum'], row['om_press'], 
            row['om_wind'], row['om_rain'], 
            row['om_solar'], row['om_wind_dir'], # New Inputs
            row['hour']
        ]]
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            ai_temp = model.predict(features)[0]
            source = "AI"
        else:
            ai_temp = row['om_temp']
            source = "Raw"

        diff = ai_temp - row['om_temp']
        color = "+" if diff > 0 else ""
        
        print(f"{row['time']} | {row['om_temp']:.1f}°C  | {ai_temp:.1f}°C  | {color}{diff:.1f} ({source})")
    
    df['ai_temp'] = [model.predict([[row['om_temp'], row['om_hum'], row['om_press'], row['om_wind'], row['om_rain'], row['om_solar'], row['om_wind_dir'], row['hour']]])[0] if os.path.exists(f"{MODEL_DIR}/rf_month_{int(row['month'])}.pkl") else row['om_temp'] for i, row in df.iterrows()]
    df['source'] = ["AI" if os.path.exists(f"{MODEL_DIR}/rf_month_{int(row['month'])}.pkl") else "Raw" for i, row in df.iterrows()]
    df.to_csv("latest_forecast.csv", index=False)
    print("\n💾 Saved forecast to 'latest_forecast.csv'")