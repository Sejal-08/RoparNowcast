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

# Suppress InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from src.physics_utils import (
    calculate_dew_point,
    calculate_dew_point_depression,
    classify_weather_state
)

def smooth_series(x, window=5):
    if len(x) < window: return x
    return pd.Series(x).rolling(window, center=True, min_periods=1).mean().values

# --------------------------------------------------
# Live station history (Uses MAX for Cumulative Rain)
# --------------------------------------------------
def get_live_history():
    end = datetime.now()
    start = end - timedelta(hours=12)

    params = {
        "deviceid": settings.DEVICE_ID,
        "startdate": start.strftime("%d-%m-%Y"),
        "enddate": end.strftime("%d-%m-%Y")
    }

    try:
        r = requests.get(settings.SOURCE_API_URL, params=params, timeout=10, verify=False)
        raw = r.json()
        if isinstance(raw, str):
            import json
            raw = json.loads(raw)
        df = pd.DataFrame(raw["items"])
    except Exception as e:
        print(f"❌ API Error: {e}")
        return pd.DataFrame(), pd.Series()

    col_map = {
        "CurrentTemperature": "temp", "CurrentHumidity": "humidity",
        "AtmPressure": "pressure", "RainfallHourly": "rain",
        "WindSpeed": "wind_speed", "LightIntensity": "light"
    }

    # Only map columns that actually exist
    valid_map = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=valid_map)
    df["time"] = pd.to_datetime(df["TimeStamp"])
    
    for c in valid_map.values():
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clean Data
    if 'pressure' in df.columns: df['pressure'] = df['pressure'].replace(0.0, np.nan)
    if 'humidity' in df.columns: df['humidity'] = df['humidity'].replace(0.0, np.nan)
    if 'wind_speed' in df.columns: df["wind_speed"] *= 3.6
    if 'rain' in df.columns: df['rain'] = df['rain'].fillna(0.0)

    df = df.set_index("time").sort_index()
    if df.empty: return pd.DataFrame(), pd.Series()

    latest = df.iloc[-1]
    
    # Aggregation Rules
    agg_rules = {}
    for col in df.columns:
        if col == "rain":
            agg_rules[col] = "max"
        elif pd.api.types.is_numeric_dtype(df[col]):
            agg_rules[col] = "mean"
            
    history = df.resample(f"{settings.TIME_STEP_MINUTES}min").agg(agg_rules)
    
    cols_to_interp = [c for c in history.columns if c != 'rain']
    if cols_to_interp:
        history[cols_to_interp] = history[cols_to_interp].interpolate(limit=3)
    history = history.ffill()

    return history, latest

# --------------------------------------------------
# Global forecast
# --------------------------------------------------
def get_global_forecast():
    url = "https://api.open-meteo.com/v1/forecast"
    
    today_obj = datetime.now()
    tomorrow_obj = today_obj + timedelta(days=1)
    
    start_str = today_obj.strftime("%Y-%m-%d")
    end_str = tomorrow_obj.strftime("%Y-%m-%d")

    params = {
        "latitude": settings.LAT,
        "longitude": settings.LON,
        "hourly": (
            "temperature_2m,relative_humidity_2m,"
            "surface_pressure,wind_speed_10m,rain,shortwave_radiation,"
            "precipitation_probability"
        ),
        "timezone": "auto",
        "start_date": start_str,
        "end_date": end_str
    }

    try:
        r = requests.get(url, params=params).json()
        df = pd.DataFrame(r["hourly"])
        df["time"] = pd.to_datetime(df["time"])

        return df.rename(columns={
            "temperature_2m": "om_temp",
            "relative_humidity_2m": "om_hum",
            "surface_pressure": "om_press",
            "wind_speed_10m": "om_wind",
            "rain": "om_rain",
            "shortwave_radiation": "om_solar",
            "precipitation_probability": "om_prob"
        })
    except Exception as e:
        print(f"❌ Global Forecast Error: {e}")
        return pd.DataFrame()

# --------------------------------------------------
# Main prediction
# --------------------------------------------------
def run_prediction():
    print("🚀 [ROPAR NOWCAST] Generating Forecast...")

    history, latest = get_live_history()
    globals_df = get_global_forecast()
    
    if history.empty or globals_df.empty:
        print("❌ Critical Data Missing. Aborting.")
        return

    # Pressure Trend
    pressure_trend = 0.0
    if len(history) >= 37:
        pressure_trend = history['pressure'].iloc[-1] - history['pressure'].iloc[-37]

    model_dir = os.path.join(settings.MODEL_DIR, "nowcast")
    feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))

    # Prepare Lag Features
    current_state = {}
    obs_cols = ["temp", "humidity", "pressure", "rain", "wind_speed", "light"]
    for i in range(1, settings.LAGS + 1):
        idx = len(history) - 1 - i
        if idx >= 0:
            row_h = history.iloc[idx]
            for c in obs_cols: current_state[f"{c}_lag_{i}"] = row_h[c]
        else:
            for c in obs_cols: current_state[f"{c}_lag_{i}"] = 0.0

    issue_time = pd.Timestamp.now().floor("5min")
    
    # --------------------------------------------------
    # 1. Calculate Bias
    # --------------------------------------------------
    bias = {c: 0.0 for c in ["temp", "humidity", "pressure", "wind_speed", "rain"]}
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
                bias[var] = float(pred_0 - latest[var])
            else:
                om_map = {"temp": "om_temp", "humidity": "om_hum", "pressure": "om_press"}
                if var in om_map:
                    bias[var] = float(fc_0[om_map[var]] - latest[var])

            if var == "humidity": bias[var] = max(-20, min(20, bias[var]))
            if var == "temp": bias[var] = max(-10, min(10, bias[var]))

    # --------------------------------------------------
    # 2. Generate Raw Forecasts
    # --------------------------------------------------
    base_time = issue_time.floor("1h")
    if base_time in history.index:
        base_idx = history.index.get_loc(base_time)
    else:
        base_idx = len(history) - 1
        base_time = history.index[base_idx]

    current_state_hourly = {}
    for i in range(1, settings.LAGS + 1):
        idx = base_idx - i
        if idx >= 0:
            row_h = history.iloc[idx]
            for c in obs_cols: current_state_hourly[f"{c}_lag_{i}"] = row_h[c]
        else:
            for c in obs_cols: current_state_hourly[f"{c}_lag_{i}"] = 0.0

    results = []
    steps_per_hour = 60 // settings.TIME_STEP_MINUTES
    
    for step in range(steps_per_hour, settings.N_HORIZONS + 1, steps_per_hour):
        valid_time = base_time + timedelta(minutes=5 * step)
        if valid_time <= issue_time: continue

        fc = globals_df[globals_df["time"] >= valid_time].head(1)
        if fc.empty: continue
        fc = fc.iloc[0]

        X = {
            "om_temp": fc["om_temp"], "om_hum": fc["om_hum"], 
            "om_press": fc["om_press"], "om_wind": fc["om_wind"], 
            "om_rain": fc["om_rain"], "om_solar": fc["om_solar"],
            "hour": valid_time.hour, "month": valid_time.month
        }
        X.update(current_state_hourly)
        Xdf = pd.DataFrame([X]).reindex(columns=feature_names, fill_value=0)

        # Build Row
        row = {
            "issue_time": issue_time,
            "time": valid_time,
            "lead_minutes": step * 5,
            "temp_global": fc["om_temp"],
            "humidity_global": fc["om_hum"],
            "pressure_global": fc["om_press"],
            "wind_speed_global": fc["om_wind"],
            "rain_global": fc["om_rain"],
            "om_prob": fc.get("om_prob", 0)
        }

        # Predict
        for var in ["temp", "humidity", "wind_speed", "rain", "pressure"]:
            model_path = os.path.join(model_dir, f"xgb_{var}_h{step:02d}.pkl")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                raw_pred = float(model.predict(Xdf)[0])
                ai_forecast = raw_pred - bias.get(var, 0)
                
                # Nudging
                lead_min = step * 5
                if var == "pressure":
                    ai_weight = min(1.0, lead_min / 1440.0)
                    row[var] = (1 - ai_weight) * latest[var] + ai_weight * ai_forecast
                elif var == "wind_speed":
                    ai_weight = min(1.0, lead_min / 240.0)
                    row[var] = (1 - ai_weight) * latest[var] + ai_weight * ai_forecast
                elif var in ["temp", "humidity"]:
                    ai_weight = min(1.0, lead_min / 720.0)
                    row[var] = (1 - ai_weight) * latest[var] + ai_weight * ai_forecast
                else:
                    # Raw rain prediction
                    row[var] = max(0.0, ai_forecast)
            else:
                row[var] = latest[var]

        results.append(row)

    df = pd.DataFrame(results)
    
    if df.empty:
        print("❌ No forecasts")
        return

    # --------------------------------------------------
    # 3. Post-Processing & CLAMP (The "Reality" Fix) 🚧
    # --------------------------------------------------
    # First, smooth the data to remove jagged noise
    for v in ["temp", "humidity", "wind_speed", "pressure"]:
        if v in df.columns:
            df[v] = smooth_series(df[v].values, window=5)

    # Then, FORCE physical limits (The Clamp)
    if "humidity" in df.columns:
        df["humidity"] = df["humidity"].clip(0.0, 100.0)  # Clamp max to 100%
    
    if "rain" in df.columns:
        df["rain"] = df["rain"].clip(lower=0.0) # No negative rain
        # Zero out tiny trace amounts (< 0.1mm) to stop "ghost" drizzles
        df.loc[df["rain"] < 0.1, "rain"] = 0.0

    # --------------------------------------------------
    # 4. Final Physics Calculations (Re-Run on Clean Data) 🌡️
    # --------------------------------------------------
    # We loop again to calc Dew Point & Condition using the CORRECTED numbers
    
    for i, row in df.iterrows():
        # A. Recalculate Dew Point
        dp = calculate_dew_point(row["temp"], row["humidity"])
        df.at[i, "dew_point"] = dp
        df.at[i, "dew_point_depression"] = calculate_dew_point_depression(row["temp"], dp)
        
        # B. Get Light for Condition Check
        # We grab the global solar value we saved earlier or fetch it again
        # Since 'om_solar' was not in 'row' explicitly above, we use globals_df lookup
        light_val = 0.0
        fc_match = globals_df[globals_df["time"] == row["time"]]
        if not fc_match.empty:
             light_val = fc_match.iloc[0]["om_solar"] * 100 # Convert W/m2 to approx Lux/Light proxy
        df.at[i, "light"] = light_val
        
        # C. Classify Condition (Now it sees 100% humidity, not 102%)
        # Note: We must pass the UPDATED row values to the classifier
        current_row_data = df.iloc[i].to_dict()
        current_row_data['pressure_change_3h'] = pressure_trend # Ensure trend is passed
        df.at[i, "condition"] = classify_weather_state(current_row_data)

        # D. Probability Logic
        global_conf = row.get("om_prob", 50)
        pred_rain = row["rain"] # This is now clean (>= 0)
        
        # Local Confidence: 0.5mm = 75%, 1.0mm = 100%
        local_conf = min(100, pred_rain * 150) 
        
        physics_boost = 0
        if pressure_trend < -1.0: physics_boost += 20
        elif pressure_trend < -0.5: physics_boost += 10
        
        # Humidity Boost
        if row["humidity"] > 98: physics_boost += 10
        elif row["humidity"] > 90: physics_boost += 5
        
        final_prob = (global_conf * 0.4) + (local_conf * 0.4) + physics_boost
        
        # Sanity Check: If Physics explicitly says CLEAR, kill the probability
        if "CLEAR" in df.at[i, "condition"]:
            final_prob = min(final_prob, 10) # Cap at 10%
            
        df.at[i, "rain_prob"] = int(max(0, min(100, final_prob)))
        df.at[i, "pressure_change_3h"] = pressure_trend

    # --------------------------------------------------
    # 5. Save Files
    # --------------------------------------------------
    # Save Latest
    temp_file = settings.OUTPUT_FILE + ".tmp"
    df.to_csv(temp_file, index=False)
    os.replace(temp_file, settings.OUTPUT_FILE)
    print(f"💾 Forecast saved → {settings.OUTPUT_FILE}")

    # Log History with Schema Protection
    history_file = os.path.join(settings.DATA_DIR, "forecast_history.csv")
    write_mode = 'a'
    write_header = False
    
    if os.path.exists(history_file):
        try:
            existing_cols = pd.read_csv(history_file, nrows=0).columns.tolist()
            if len(existing_cols) != len(df.columns):
                print(f"⚠️ Schema Change Detected. Backing up old history.")
                os.rename(history_file, history_file + ".bak")
                write_mode = 'w'
                write_header = True
        except Exception:
             write_mode = 'w'
             write_header = True
    else:
        write_mode = 'w'
        write_header = True

    df.to_csv(history_file, mode=write_mode, header=write_header, index=False)
    print(f"📜 History logged → {history_file}")

    print(df[['time', 'condition', 'rain', 'rain_prob', 'humidity']].head(3))

if __name__ == "__main__":
    run_prediction()