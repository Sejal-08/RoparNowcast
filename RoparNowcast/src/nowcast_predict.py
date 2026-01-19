import sys
import os
import requests
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import urllib3

# --------------------------------------------------
# Fix Import Path
# --------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Suppress InsecureRequestWarning when bypassing SSL verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from src.physics_utils import (
    calculate_dew_point,
    calculate_dew_point_depression,
    classify_weather_state
)

# --------------------------------------------------
# Smoothing helper
# --------------------------------------------------
def smooth_series(x, window=5):
    if len(x) < window:
        return x
    return pd.Series(x).rolling(window, center=True, min_periods=1).mean().values


# --------------------------------------------------
# Live station history
# --------------------------------------------------
def get_live_history():
    end = datetime.now()
    start = end - timedelta(hours=12)

    params = {
        "deviceid": settings.DEVICE_ID,
        "startdate": start.strftime("%d-%m-%Y"),
        "enddate": end.strftime("%d-%m-%Y")
    }

    r = requests.get(settings.SOURCE_API_URL, params=params, timeout=10, verify=False)
    raw = r.json()
    if isinstance(raw, str):
        import json
        raw = json.loads(raw)

    df = pd.DataFrame(raw["items"])

    col_map = {
        "CurrentTemperature": "temp",
        "CurrentHumidity": "humidity",
        "AtmPressure": "pressure",
        "RainfallHourly": "rain",
        "WindSpeed": "wind_speed",
        "LightIntensity": "light"
    }

    df = df.rename(columns=col_map)
    df["time"] = pd.to_datetime(df["TimeStamp"])

    for c in col_map.values():
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 🧹 CLEANING: Treat 0.0 as NaN (Sensor Failure)
    df['pressure'] = df['pressure'].replace(0.0, np.nan)
    df['humidity'] = df['humidity'].replace(0.0, np.nan)

    df["wind_speed"] *= 3.6
    df = df.set_index("time").sort_index()

    latest = df.dropna().iloc[-1]
    
    # Resample to 5-min steps to match training lags (instead of hourly)
    history = df.resample(f"{settings.TIME_STEP_MINUTES}min").mean(numeric_only=True)
    history = history.interpolate(limit=3).ffill()

    return history, latest


# --------------------------------------------------
# Global forecast (Open-Meteo)
# --------------------------------------------------
def get_global_forecast():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": settings.LAT,
        "longitude": settings.LON,
        "hourly": (
            "temperature_2m,relative_humidity_2m,"
            "surface_pressure,wind_speed_10m,rain,shortwave_radiation"
        ),
        "timezone": "auto",
        "forecast_days": 1
    }

    r = requests.get(url, params=params).json()
    df = pd.DataFrame(r["hourly"])
    df["time"] = pd.to_datetime(df["time"])

    return df.rename(columns={
        "temperature_2m": "om_temp",
        "relative_humidity_2m": "om_hum",
        "surface_pressure": "om_press",
        "wind_speed_10m": "om_wind",
        "rain": "om_rain",
        "shortwave_radiation": "om_solar"
    })


# --------------------------------------------------
# Data Logging (For Continuous Learning)
# --------------------------------------------------
def save_training_data(latest_sensor, globals_df):
    """Appends the current data point to a log file for future retraining."""
    try:
        # 1. Prepare Row from Sensor Data
        row = latest_sensor.to_dict()
        sensor_time = latest_sensor.name  # Timestamp is the index
        
        # 2. Match with Open-Meteo Data (Find nearest hour)
        # Calculate time difference to find the closest forecast row
        diffs = (globals_df['time'] - sensor_time).abs()
        nearest_idx = diffs.idxmin()
        om_row = globals_df.loc[nearest_idx]
        
        # Add OM columns to the row
        for col in ['om_temp', 'om_hum', 'om_press', 'om_wind', 'om_rain', 'om_solar']:
            row[col] = om_row[col]
            
        row['merge_time'] = sensor_time
        
        # 3. Save to Log CSV
        log_file = os.path.join(settings.DATA_DIR, "training_log.csv")
        df_new = pd.DataFrame([row])
        
        # Append mode, header only if file doesn't exist
        header = not os.path.exists(log_file)
        df_new.to_csv(log_file, mode='a', header=header, index=False)
        
    except Exception as e:
        print(f"⚠️ Data Logging Error: {e}")


# --------------------------------------------------
# Main prediction
# --------------------------------------------------
def run_prediction():
    print("🚀 [ROPAR NOWCAST] 3-Hour Smoothed Nowcasting")

    history, latest = get_live_history()
    globals_df = get_global_forecast()

    # Calculate Pressure Trend (Observed over last 3 hours)
    # 3 hours = 180 mins. With 5 min steps, that is 36 steps back.
    pressure_trend = 0.0
    if len(history) >= 37:
        # Compare latest history point vs 3 hours ago
        pressure_trend = history['pressure'].iloc[-1] - history['pressure'].iloc[-37]

    # [NEW] Log this data point for future training
    save_training_data(latest, globals_df)

    model_dir = os.path.join(settings.MODEL_DIR, "nowcast")
    feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))

    current_state = {}
    obs_cols = ["temp", "humidity", "pressure", "rain", "wind_speed", "light"]
    for i in range(1, settings.LAGS + 1):
        row = history.iloc[-i]
        for c in obs_cols:
            current_state[f"{c}_lag_{i}"] = row[c]

    issue_time = pd.Timestamp.now().floor("5min")
    results = []

    # --------------------------------------------------
    # 1. Calculate Bias (Model Error at T+0)
    # --------------------------------------------------
    # We predict T+5 min just to see how far off the model is from the current sensor
    bias = {"temp": 0.0, "humidity": 0.0, "pressure": 0.0, "wind_speed": 0.0, "rain": 0.0}
    
    # Create input for T+5 min (Step 1)
    valid_time_0 = issue_time + timedelta(minutes=5)
    fc_0 = globals_df[globals_df["time"] >= valid_time_0].head(1)
    
    if not fc_0.empty:
        fc_0 = fc_0.iloc[0]
        X_0 = {
            "om_temp": fc_0["om_temp"], "om_hum": fc_0["om_hum"], "om_press": fc_0["om_press"],
            "om_wind": fc_0["om_wind"], "om_rain": fc_0["om_rain"], "om_solar": fc_0["om_solar"],
            "hour": valid_time_0.hour, "month": valid_time_0.month
        }
        X_0.update(current_state)
        Xdf_0 = pd.DataFrame([X_0]).reindex(columns=feature_names, fill_value=0)

        for var in ["temp", "humidity", "pressure"]:
            model_path = os.path.join(model_dir, f"xgb_{var}_h01.pkl")
            if os.path.exists(model_path):
                pred_0 = float(joblib.load(model_path).predict(Xdf_0)[0])
                # Bias = Prediction - Actual. We will SUBTRACT this from future forecasts.
                bias[var] = float(pred_0 - latest[var])
            else:
                # FALLBACK: If h01 model is missing, use Global Forecast deviation.
                # This anchors the forecast to the current sensor reading.
                # Map variable names to Open-Meteo columns
                om_map = {"temp": "om_temp", "humidity": "om_hum", "pressure": "om_press", "wind_speed": "om_wind"}
                if var in om_map:
                    bias[var] = float(fc_0[om_map[var]] - latest[var])

            # Clamp bias to prevent over-correction (Safety Net)
            if var == "humidity": bias[var] = max(-20, min(20, bias[var]))
            if var == "temp": bias[var] = max(-10, min(10, bias[var]))

    print(f"⚖️ Calculated Bias: {bias}")

    # --------------------------------------------------
    # 2. Generate Hourly Forecasts
    # --------------------------------------------------
    # [MODIFIED] Anchor forecasts to fixed hourly timestamps (12:00, 13:00...)
    # regardless of when the script runs (e.g. 11:42).
    base_time = issue_time.floor("1h")

    # Build input features relative to base_time (e.g. 11:00)
    # We need to find the index of base_time in history to get correct lags
    if base_time in history.index:
        base_idx = history.index.get_loc(base_time)
    else:
        # Fallback if exact hour missing (unlikely with 12h history)
        base_idx = len(history) - 1
        base_time = history.index[base_idx]

    current_state_hourly = {}
    for i in range(1, settings.LAGS + 1):
        # History is 5-min resampled. lag_i is index - i.
        idx = base_idx - i
        if idx >= 0:
            row_h = history.iloc[idx]
            for c in obs_cols:
                current_state_hourly[f"{c}_lag_{i}"] = row_h[c]
        else:
            for c in obs_cols:
                current_state_hourly[f"{c}_lag_{i}"] = 0.0

    steps_per_hour = 60 // settings.TIME_STEP_MINUTES
    for step in range(steps_per_hour, settings.N_HORIZONS + 1, steps_per_hour):
        valid_time = base_time + timedelta(minutes=5 * step)

        # Skip if the forecast time is in the past or is the current time
        if valid_time <= issue_time:
            continue

        fc = globals_df[globals_df["time"] >= valid_time].head(1)
        if fc.empty:
            continue

        fc = fc.iloc[0]

        X = {
            "om_temp": fc["om_temp"],
            "om_hum": fc["om_hum"],
            "om_press": fc["om_press"],
            "om_wind": fc["om_wind"],
            "om_rain": fc["om_rain"],
            "om_solar": fc["om_solar"],
            "hour": valid_time.hour,
            "month": valid_time.month
        }
        X.update(current_state_hourly)

        Xdf = pd.DataFrame([X]).reindex(columns=feature_names, fill_value=0)

        row = {
            "issue_time": issue_time,
            "time": valid_time,
            "lead_minutes": step * 5,
            "temp_global": fc["om_temp"],
            "humidity_global": fc["om_hum"],
            "pressure_global": fc["om_press"],
            "wind_speed_global": fc["om_wind"],
            "rain_global": fc["om_rain"]
        }

        for var in ["temp", "humidity", "wind_speed", "rain", "pressure"]:
            model_path = os.path.join(model_dir, f"xgb_{var}_h{step:02d}.pkl")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                raw_pred = float(model.predict(Xdf)[0])
                
                # 1. Apply Bias Correction (Align with current reality)
                ai_forecast = raw_pred - bias.get(var, 0)

                # [NEW] Safety Clamp: Prevent AI from hallucinating Summer in Winter
                # If AI deviates > 3.0 units from Global Forecast, pull it back.
                # This fixes the 12:00 PM jump issue.
                if var in ["temp", "humidity", "pressure"]:
                    global_val = row[f"{var}_global"]
                    max_dev = 3.0 if var == "temp" else (15.0 if var == "humidity" else 2.0)
                    ai_forecast = max(global_val - max_dev, min(global_val + max_dev, ai_forecast))

                # 2. Nudging (Blend with Persistence)
                # Since training data is limited (Sept-Oct), the AI might hallucinate in Winter.
                # We anchor the forecast to the current observation and slowly relax to the AI.
                lead_min = step * 5
                
                if var == "pressure":
                    # Pressure: Very slow change. Trust Persistence (24h ramp).
                    ai_weight = min(1.0, lead_min / 1440.0)
                    row[var] = (1 - ai_weight) * latest[var] + ai_weight * ai_forecast
                
                elif var == "wind_speed":
                    # Wind: AI has high skill at 3h. Trust AI faster (4h ramp).
                    ai_weight = min(1.0, lead_min / 240.0)
                    row[var] = (1 - ai_weight) * latest[var] + ai_weight * ai_forecast

                elif var in ["temp", "humidity"]:
                    # Temp/Hum: Conservative 12h ramp to fix seasonal drift.
                    ai_weight = min(1.0, lead_min / 720.0)
                    row[var] = (1 - ai_weight) * latest[var] + ai_weight * ai_forecast
                
                else:
                    # Rain: Trust AI fully
                    row[var] = ai_forecast
            else:
                row[var] = latest[var]

        dp = calculate_dew_point(row["temp"], row["humidity"])
        dpd = calculate_dew_point_depression(row["temp"], dp)

        row["dew_point"] = dp
        row["pressure_change_3h"] = pressure_trend
        row["dew_point_depression"] = dpd
        row["light"] = fc["om_solar"] * 100
        row["condition"] = classify_weather_state(row)

        results.append(row)

    df = pd.DataFrame(results)

    if df.empty:
        print("❌ No forecasts generated")
        return

    # ---------------- POST-PROCESSING ----------------
    # 1. Physics Rate Limiter (Time-Aware)
    # We limit the rate of change to ensure physical realism.
    # Temp: Max 2.4°C/hour (0.04°C/min)
    # Pressure: Max 1.2 hPa/hour (0.02 hPa/min)
    
    limits = {
        "temp": 0.04,
        "pressure": 0.02
    }

    for var, rate_per_min in limits.items():
        if var in df.columns and not pd.isna(latest.get(var)):
            vals = df[var].values
            times = df["time"].values
            
            last_val = latest[var]
            last_time = pd.to_datetime(latest.name)
            
            for i in range(len(vals)):
                current_time = pd.to_datetime(times[i])
                delta_minutes = (current_time - last_time).total_seconds() / 60.0
                if delta_minutes <= 0: delta_minutes = 5.0 # Safety
                
                allowed_change = rate_per_min * delta_minutes
                change = vals[i] - last_val
                
                # Clamp change
                clamped_change = max(-allowed_change, min(allowed_change, change))
                vals[i] = last_val + clamped_change
                
                last_val = vals[i]
                last_time = current_time
            
            df[var] = vals

    # 2. Smoothing (Rolling Mean)
    for v in ["temp", "humidity", "wind_speed", "pressure"]:
        if v in df.columns:
            df[v] = smooth_series(df[v].values, window=5)

    df.to_csv(settings.OUTPUT_FILE, index=False)
    print(f"💾 Forecast saved → {settings.OUTPUT_FILE}")
    
    print("\n🔮 Forecast Preview:")
    # Create a clean view for printing
    preview = df[['time', 'temp', 'rain', 'condition']].head(3).copy()
    preview[['temp', 'rain']] = preview[['temp', 'rain']].round(2)
    print(preview.to_string(index=False))

    # --- HISTORY LOGGING ---
    # Append this forecast to a history file for "Forecast vs Actuals" analysis
    history_file = os.path.join(settings.DATA_DIR, "forecast_history.csv")
    # Write header only if file does not exist
    header = not os.path.exists(history_file)
    df.to_csv(history_file, mode='a', header=header, index=False)
    print(f"📜 Forecast appended to history → {history_file}")


if __name__ == "__main__":
    run_prediction()
