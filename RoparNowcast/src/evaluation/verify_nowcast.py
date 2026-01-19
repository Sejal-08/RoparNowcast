import sys
import os
import argparse
import requests
import pandas as pd
import numpy as np
import urllib3
from datetime import timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import settings

# Suppress InsecureRequestWarning when bypassing SSL verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -----------------------------
# FETCH OBSERVATIONS
# -----------------------------
def fetch_observations(start, end):
    params = {
        "deviceid": settings.DEVICE_ID,
        "startdate": start.strftime("%d-%m-%Y"),
        "enddate": end.strftime("%d-%m-%Y"),
    }
    r = requests.get(settings.SOURCE_API_URL, params=params, timeout=10, verify=False)
    raw = r.json()
    items = raw.get("items", [])
    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items)
    df["time"] = pd.to_datetime(df["TimeStamp"])
    df = df.rename(
        columns={
            "CurrentTemperature": "temp",
            "CurrentHumidity": "humidity",
            "AtmPressure": "pressure",
            "WindSpeed": "wind_speed",
            "RainfallHourly": "rain",
        }
    )

    # Ensure numeric types
    for col in ["temp", "humidity", "pressure", "wind_speed", "rain"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert Wind Speed from m/s to km/h to match forecast units
    df["wind_speed"] = df["wind_speed"] * 3.6

    return df


# -----------------------------
# MAIN
# -----------------------------
def run_verification(start_date=None, end_date=None):
    print("📊 Running nowcast verification...")

    # 1. Load Forecast Data (History preferred)
    history_file = os.path.join(settings.DATA_DIR, "forecast_history.csv")
    if os.path.exists(history_file):
        print(f"📂 Reading Forecast History: {history_file}")
        fc = pd.read_csv(history_file, parse_dates=["time", "issue_time"])
    else:
        print(f"⚠️ History file not found. Reading latest forecast: {settings.OUTPUT_FILE}")
        fc = pd.read_csv(settings.OUTPUT_FILE, parse_dates=["time", "issue_time"])

    if fc.empty:
        print("❌ No forecast data")
        return

    # Filter by date if requested
    if start_date:
        s_dt = pd.to_datetime(start_date)
        print(f"   -> Filtering Start: {s_dt}")
        fc = fc[fc["time"] >= s_dt]

    if end_date:
        e_dt = pd.to_datetime(end_date) if end_date.lower() != "now" else pd.Timestamp.now()
        print(f"   -> Filtering End: {e_dt}")
        fc = fc[fc["time"] <= e_dt]

    if fc.empty:
        print("❌ No forecast data in the specified range.")
        return

    # Fetch observations (include buffer for persistence lag)
    start_obs = fc["time"].min() - timedelta(hours=2)
    end_obs = fc["time"].max()
    print(f"   -> Fetching observations from {start_obs} to {end_obs}")
    obs = fetch_observations(start_obs, end_obs)
    if obs.empty:
        print("❌ No observation data")
        return

    merged = pd.merge_asof(
        fc.sort_values("time"),
        obs.sort_values("time"),
        on="time",
        tolerance=pd.Timedelta("10min"),
        suffixes=("_fc", "_obs"),
    )

    # [FIXED] True Persistence Baseline (Obs at Issue Time)
    # We compare Forecast(Target) vs Obs(Issue). This is the fair baseline.
    obs_persist = obs.copy().rename(columns={
        c: f"{c}_issue" for c in ["temp", "humidity", "pressure", "wind_speed", "rain"]
    })
    
    merged = pd.merge_asof(
        merged.sort_values("issue_time"),
        obs_persist.sort_values("time"),
        left_on="issue_time",
        right_on="time",
        tolerance=pd.Timedelta("15min"),
        direction="nearest"
    )

    merged = merged.dropna(subset=["temp_obs"])
    if merged.empty:
        print("❌ No matched forecast–obs pairs")
        return

    rows = []
    for var in ["temp", "humidity", "wind_speed", "rain", "pressure"]:
        for lead in sorted(merged["lead_minutes"].unique()):
            sub = merged[merged["lead_minutes"] == lead]
            
            # Calculate MAE using pandas to handle NaNs automatically
            mae = (sub[f"{var}_fc"] - sub[f"{var}_obs"]).abs().mean()
            persist = (sub[f"{var}_obs"] - sub[f"{var}_issue"]).abs().mean()

            skill = (
                (persist - mae) / persist if persist > 0 else np.nan
            )

            rows.append(
                {
                    "lead_minutes": lead,
                    "variable": var,
                    "MAE": round(mae, 3),
                    "Persistence_MAE": round(persist, 3),
                    "Skill_vs_Persistence": round(skill, 3)
                    if not np.isnan(skill)
                    else np.nan,
                }
            )

    summary = pd.DataFrame(rows)
    print("\n📈 VERIFICATION SUMMARY (Skill > 0 = better than persistence)\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, help="Start date (e.g. 'Jan 15, 2026 15:35')")
    parser.add_argument("--end", type=str, help="End date (e.g. 'now')")
    args = parser.parse_args()

    run_verification(args.start, args.end)
