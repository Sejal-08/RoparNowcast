import time
import sys
import os
import pandas as pd
import requests
import urllib3
from datetime import datetime, timedelta

# Ensure we can import from the 'src' folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import settings
from src.nowcast_train import run_training

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------
# 1. Fetch Sensor Data (FIXED: Uses Chunking + Retries)
# ---------------------------------------------------------
def fetch_sensor_data(start_dt, end_dt):
    """Fetches sensor data in chunks to avoid timeouts."""
    print(f"   -> Fetching Sensor Data: {start_dt.date()} to {end_dt.date()}")
    
    CHUNK_DAYS = 20  # Download 20 days at a time to prevent timeout
    all_chunks = []
    
    current_start = start_dt
    
    while current_start < end_dt:
        current_end = min(current_start + timedelta(days=CHUNK_DAYS), end_dt)
        
        # print(f"      ... downloading chunk: {current_start.date()} -> {current_end.date()}")
        
        params = {
            "deviceid": settings.DEVICE_ID,
            "startdate": current_start.strftime("%d-%m-%Y"),
            "enddate": current_end.strftime("%d-%m-%Y")
        }
        
        try:
            r = requests.get(settings.SOURCE_API_URL, params=params, timeout=30, verify=False)
            r.raise_for_status()
            
            data = r.json()
            if isinstance(data, str):
                import json
                data = json.loads(data)
            
            items = data.get("items", [])
            if items:
                chunk_df = pd.DataFrame(items)
                all_chunks.append(chunk_df)
                
        except Exception as e:
            print(f"      ⚠️ Chunk failed ({current_start.date()}): {e}")
            # Continue to next chunk even if one fails
            
        # Move to next chunk
        current_start = current_end + timedelta(days=1)
        time.sleep(0.5)  # Brief pause to be nice to the API

    if not all_chunks:
        print("   ❌ No data received from any chunk.")
        return pd.DataFrame()

    # Combine all chunks
    df = pd.concat(all_chunks, ignore_index=True)
    
    # ---------------------------
    # Data Cleaning & Mapping
    # ---------------------------
    col_map = {
        "TimeStamp": "merge_time",
        "CurrentTemperature": "temp",
        "CurrentHumidity": "humidity",
        "AtmPressure": "pressure",
        "RainfallHourly": "rain",
        "WindSpeed": "wind_speed",
        "LightIntensity": "light"
    }
    
    valid_map = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=valid_map)
    
    df["merge_time"] = pd.to_datetime(df["merge_time"])
    
    numeric_cols = ["temp", "humidity", "pressure", "rain", "wind_speed", "light"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    if 'pressure' in df.columns: df['pressure'] = df['pressure'].replace(0.0, float('nan'))
    if 'humidity' in df.columns: df['humidity'] = df['humidity'].replace(0.0, float('nan'))
    if 'rain' in df.columns: df['rain'] = df['rain'].fillna(0.0)
    
    if "wind_speed" in df.columns:
        df["wind_speed"] *= 3.6

    return df

# ---------------------------------------------------------
# 2. Fetch Weather Data (Open-Meteo)
# ---------------------------------------------------------
def fetch_weather_data(start_dt, end_dt):
    """Fetches weather context from Open-Meteo."""
    print(f"   -> Fetching Weather Data: {start_dt.date()} to {end_dt.date()}")
    
    lat = getattr(settings, 'LAT', 30.97)
    lon = getattr(settings, 'LON', 76.53)
    
    # Try Archive first
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,rain,shortwave_radiation",
        "timezone": "auto"
    }
    
    df_weather = pd.DataFrame()
    
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200:
            data = r.json()
            if "hourly" in data:
                h = data["hourly"]
                df_weather = pd.DataFrame({
                    "time": pd.to_datetime(h["time"]),
                    "om_temp": h["temperature_2m"],
                    "om_hum": h["relative_humidity_2m"],
                    "om_press": h["surface_pressure"],
                    "om_wind": h["wind_speed_10m"],
                    "om_rain": h["rain"],
                    "om_solar": h["shortwave_radiation"]
                })
    except Exception as e:
        print(f"   ⚠️ Archive API failed: {e}")

    # Supplement with Forecast API if needed
    if df_weather.empty or df_weather["time"].max() < end_dt - timedelta(days=1):
        print("   -> Supplementing with Forecast API...")
        try:
            url_fc = "https://api.open-meteo.com/v1/forecast"
            params_fc = {
                "latitude": lat,
                "longitude": lon,
                "past_days": 7, 
                "forecast_days": 1,
                "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,rain,shortwave_radiation",
                "timezone": "auto"
            }
            r = requests.get(url_fc, params=params_fc, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if "hourly" in data:
                    h = data["hourly"]
                    df_fc = pd.DataFrame({
                        "time": pd.to_datetime(h["time"]),
                        "om_temp": h["temperature_2m"],
                        "om_hum": h["relative_humidity_2m"],
                        "om_press": h["surface_pressure"],
                        "om_wind": h["wind_speed_10m"],
                        "om_rain": h["rain"],
                        "om_solar": h["shortwave_radiation"]
                    })
                    if not df_weather.empty:
                        df_weather = pd.concat([df_weather, df_fc]).drop_duplicates(subset="time").sort_values("time")
                    else:
                        df_weather = df_fc
        except Exception as e:
            print(f"   ❌ Forecast API Error: {e}")

    return df_weather

# ---------------------------------------------------------
# 3. Update Dataset Logic
# ---------------------------------------------------------
def update_dataset():
    print("📥 Checking for new data...")
    
    # Check for existing data
    if os.path.exists(settings.DATA_FILE):
        try:
            df_existing = pd.read_csv(settings.DATA_FILE)
            df_existing["merge_time"] = pd.to_datetime(df_existing["merge_time"])
            last_date = df_existing["merge_time"].max()
            print(f"   📅 Last data point: {last_date}")
        except:
            print("   ⚠️ Error reading existing data. Starting fresh.")
            df_existing = pd.DataFrame()
            last_date = datetime(2024, 1, 1)
    else:
        print("   ⚠️ Data file not found. Starting fresh (Full History).")
        df_existing = pd.DataFrame()
        # Fallback: Start from a safe past date (e.g., Jan 1, 2024)
        # The fetcher will handle empty chunks until it hits real data
        last_date = datetime(2024, 1, 1)
    
    start_fetch = last_date - timedelta(hours=1) 
    end_fetch = datetime.now()
    
    if (end_fetch - start_fetch).total_seconds() < 3600:
        print("   ✅ Data is up to date.")
        return False

    # Fetch Sensor Data (Chunked)
    df_sensor = fetch_sensor_data(start_fetch, end_fetch)
    if df_sensor.empty:
        print("   ⚠️ No new sensor data found.")
        return False
    
    # --- CRITICAL: AGGREGATION RULES (Max for Rain) ---
    df_sensor = df_sensor.set_index("merge_time").sort_index()
    df_sensor = df_sensor[~df_sensor.index.duplicated(keep='first')]

    agg_rules = {}
    for col in df_sensor.columns:
        if col == "rain":
            agg_rules[col] = "max"  # Correct Logic for Cumulative Rain
        elif pd.api.types.is_numeric_dtype(df_sensor[col]):
            agg_rules[col] = "mean"

    df_sensor = df_sensor.resample("5min").agg(agg_rules)
    
    # Interpolate (Exclude rain)
    cols_to_interp = [c for c in df_sensor.columns if c != 'rain']
    if cols_to_interp:
        df_sensor[cols_to_interp] = df_sensor[cols_to_interp].interpolate(limit=3)
    
    df_sensor = df_sensor.reset_index()

    # Fetch Weather Data
    df_weather = fetch_weather_data(start_fetch, end_fetch)
    if df_weather.empty:
        print("   ❌ Could not fetch weather context. Skipping update.")
        return False
        
    df_weather = df_weather.set_index("time").sort_index()
    df_weather = df_weather[~df_weather.index.duplicated(keep='first')]
    df_weather = df_weather.resample("5min").interpolate().reset_index()

    # Merge
    df_new = pd.merge_asof(
        df_sensor.sort_values("merge_time"),
        df_weather.sort_values("time"),
        left_on="merge_time",
        right_on="time",
        direction="nearest",
        tolerance=pd.Timedelta("15min")
    )
    
    if "time" in df_new.columns:
        del df_new["time"]
    
    df_new = df_new.dropna(subset=["temp", "om_temp"])
    
    if df_new.empty:
        print("   ⚠️ No valid matched data found.")
        return False

    # Append
    if not df_existing.empty:
        df_new = df_new[df_new["merge_time"] > last_date]
        if df_new.empty:
            print("   ✅ No new rows to add.")
            return False
        df_combined = pd.concat([df_existing, df_new])
    else:
        df_combined = df_new
        
    df_combined = df_combined.drop_duplicates(subset=["merge_time"], keep="last")
    df_combined.to_csv(settings.DATA_FILE, index=False)
    print(f"   ✅ Dataset updated. Added {len(df_new)} rows. Total: {len(df_combined)}")
    return True

# ---------------------------------------------------------
# 4. Main Scheduler
# ---------------------------------------------------------
def start_weekly_retrain():
    print("🧠 [AUTO-RETRAIN] Weekly Model Update System")
    print("   -> Schedule: Runs immediately, then every 7 days")
    print("   -> Press 'Ctrl + C' to stop.")
    print("-" * 50)

    while True:
        try:
            print(f"\n🔄 Starting Update Cycle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            data_updated = update_dataset()
            
            if data_updated:
                print("   -> Data changed. Running training...")
                run_training()
            else:
                print("   -> Data unchanged. Skipping training.")
            
            seconds_in_week = 7 * 24 * 60 * 60
            next_run = datetime.now() + timedelta(seconds=seconds_in_week)
            
            print(f"✅ Cycle Complete.")
            print(f"📅 Next scheduled run: {next_run.strftime('%A, %d-%b %H:%M')}")
            print(f"💤 Sleeping for 7 days...")
            
            time.sleep(seconds_in_week)

        except KeyboardInterrupt:
            print("\n🛑 Scheduler stopped by user.")
            break
        except Exception as e:
            print(f"❌ Critical Error: {e}")
            print("   -> Retrying in 1 hour...")
            time.sleep(3600)

if __name__ == "__main__":
    start_weekly_retrain()