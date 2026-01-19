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

def fetch_sensor_data(start_dt, end_dt):
    """Fetches sensor data from the source API."""
    print(f"   -> Fetching Sensor Data: {start_dt.date()} to {end_dt.date()}")
    
    # API requires DD-MM-YYYY
    params = {
        "deviceid": settings.DEVICE_ID,
        "startdate": start_dt.strftime("%d-%m-%Y"),
        "enddate": end_dt.strftime("%d-%m-%Y")
    }
    
    try:
        r = requests.get(settings.SOURCE_API_URL, params=params, timeout=30, verify=False)
        try:
            data = r.json()
            if isinstance(data, str):
                import json
                data = json.loads(data)
        except:
            print("   ❌ Failed to decode Sensor JSON")
            return pd.DataFrame()
        
        items = data.get("items", [])
        if not items:
            return pd.DataFrame()

        df = pd.DataFrame(items)
        
        # Map columns
        col_map = {
            "TimeStamp": "merge_time",
            "CurrentTemperature": "temp",
            "CurrentHumidity": "humidity",
            "AtmPressure": "pressure",
            "RainfallHourly": "rain",
            "WindSpeed": "wind_speed",
            "LightIntensity": "light"
        }
        
        # Rename available columns
        df = df.rename(columns=col_map)
        
        # Ensure merge_time is datetime
        df["merge_time"] = pd.to_datetime(df["merge_time"])
        
        # Convert numeric columns
        numeric_cols = ["temp", "humidity", "pressure", "rain", "wind_speed", "light"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        
        # Convert Wind Speed m/s -> km/h
        if "wind_speed" in df.columns:
            df["wind_speed"] *= 3.6

        return df
    except Exception as e:
        print(f"   ❌ Sensor API Error: {e}")
        return pd.DataFrame()

def fetch_weather_data(start_dt, end_dt):
    """Fetches weather context from Open-Meteo (Archive or Forecast)."""
    print(f"   -> Fetching Weather Data: {start_dt.date()} to {end_dt.date()}")
    
    lat = getattr(settings, 'LAT', 30.97)
    lon = getattr(settings, 'LON', 76.53)
    
    # Try Archive first (Better for history)
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

    # If Archive returned nothing or incomplete (recent days might be missing), try Forecast API
    # Forecast API 'past_days' gives up to 92 days of history
    if df_weather.empty or df_weather["time"].max() < end_dt - timedelta(days=1):
        print("   -> Supplementing with Forecast API (Recent History)...")
        try:
            url_fc = "https://api.open-meteo.com/v1/forecast"
            params_fc = {
                "latitude": lat,
                "longitude": lon,
                "past_days": 7, # Get last week
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
                    # Combine if we had some archive data
                    if not df_weather.empty:
                        df_weather = pd.concat([df_weather, df_fc]).drop_duplicates(subset="time").sort_values("time")
                    else:
                        df_weather = df_fc
        except Exception as e:
            print(f"   ❌ Forecast API Error: {e}")

    return df_weather

def update_dataset():
    print("📥 Checking for new data...")
    
    # 1. Determine Time Range
    if os.path.exists(settings.DATA_FILE):
        try:
            df_existing = pd.read_csv(settings.DATA_FILE)
            df_existing["merge_time"] = pd.to_datetime(df_existing["merge_time"])
            last_date = df_existing["merge_time"].max()
            print(f"   📅 Last data point: {last_date}")
        except:
            print("   ⚠️ Error reading existing data. Starting fresh.")
            df_existing = pd.DataFrame()
            last_date = datetime.now() - timedelta(days=30)
    else:
        print("   ⚠️ Data file not found. Starting fresh (last 30 days).")
        df_existing = pd.DataFrame()
        last_date = datetime.now() - timedelta(days=30)
    
    start_fetch = last_date - timedelta(hours=1) # Overlap to be safe
    end_fetch = datetime.now()
    
    if (end_fetch - start_fetch).total_seconds() < 3600:
        print("   ✅ Data is up to date (less than 1 hour gap).")
        return False

    # 2. Fetch Sensor Data
    df_sensor = fetch_sensor_data(start_fetch, end_fetch)
    if df_sensor.empty:
        print("   ⚠️ No new sensor data found.")
        return False
    
    # Resample Sensor to 5min (Handle duplicates and gaps)
    df_sensor = df_sensor.set_index("merge_time").sort_index()
    df_sensor = df_sensor[~df_sensor.index.duplicated(keep='first')]
    # Resample and interpolate small gaps
    df_sensor = df_sensor.resample("5min").mean(numeric_only=True).interpolate(limit=3).reset_index()

    # 3. Fetch Weather Data
    df_weather = fetch_weather_data(start_fetch, end_fetch)
    if df_weather.empty:
        print("   ❌ Could not fetch weather context. Skipping update.")
        return False
        
    # Resample Weather to 5min
    df_weather = df_weather.set_index("time").sort_index()
    df_weather = df_weather[~df_weather.index.duplicated(keep='first')]
    df_weather = df_weather.resample("5min").interpolate().reset_index()

    # 4. Merge
    # Use merge_asof to align sensor data with nearest weather data
    df_new = pd.merge_asof(
        df_sensor.sort_values("merge_time"),
        df_weather.sort_values("time"),
        left_on="merge_time",
        right_on="time",
        direction="nearest",
        tolerance=pd.Timedelta("15min")
    )
    
    # Clean up
    if "time" in df_new.columns:
        del df_new["time"]
    
    # Drop rows where essential data is missing
    df_new = df_new.dropna(subset=["temp", "om_temp"])
    
    if df_new.empty:
        print("   ⚠️ No valid matched data found.")
        return False

    # 5. Append and Save
    if not df_existing.empty:
        # Filter new data to only include times after last_date
        df_new = df_new[df_new["merge_time"] > last_date]
        if df_new.empty:
            print("   ✅ No new rows to add.")
            return False
            
        df_combined = pd.concat([df_existing, df_new])
    else:
        df_combined = df_new
        
    # Final deduplication
    df_combined = df_combined.drop_duplicates(subset=["merge_time"], keep="last")
    df_combined.to_csv(settings.DATA_FILE, index=False)
    print(f"   ✅ Dataset updated. Added {len(df_new)} rows. Total: {len(df_combined)}")
    return True

def start_weekly_retrain():
    print("🧠 [AUTO-RETRAIN] Weekly Model Update System")
    print("   -> Schedule: Runs immediately, then every 7 days")
    print("   -> Press 'Ctrl + C' to stop.")
    print("-" * 50)

    while True:
        try:
            print(f"\n🔄 Starting Update Cycle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 1. Update Data
            data_updated = update_dataset()
            
            # 2. Run Training (only if data changed or forced)
            if data_updated:
                print("   -> Data changed. Running training...")
                run_training()
            else:
                print("   -> Data unchanged. Skipping training.")
            
            # 3. Calculate Next Run (7 Days)
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
