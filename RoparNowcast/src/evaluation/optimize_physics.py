import pandas as pd
import numpy as np
import os
import sys

# Ensure we can import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings
from src.physics_utils import calculate_dew_point, calculate_dew_point_depression

def load_and_prep_data():
    """Loads historical data and calculates physics metrics."""
    if not os.path.exists(settings.DATA_FILE):
        print(f"❌ Data file not found: {settings.DATA_FILE}")
        return pd.DataFrame()

    print(f"📂 Loading data from {settings.DATA_FILE}...")
    df = pd.read_csv(settings.DATA_FILE)
    df['time'] = pd.to_datetime(df['merge_time'])
    df = df.sort_values('time')

    # --- 1. Unit Correction (Wind Speed) ---
    # Ensure wind is in km/h for consistent thresholds
    if df['wind_speed'].max() < 30:  # Heuristic: likely m/s
        print("⚠️ Converting Wind Speed from m/s to km/h")
        df['wind_speed'] = df['wind_speed'] * 3.6

    # --- 2. Calculate Physics Features ---
    # Dew Point & Depression
    df['dew_point'] = calculate_dew_point(df['temp'].values, df['humidity'].values)
    df['dpd'] = calculate_dew_point_depression(df['temp'].values, df['dew_point'].values)

    # Clean Pressure: Treat 0.0 as NaN so we don't get fake trends
    df['pressure'] = df['pressure'].replace(0.0, np.nan)

    # Pressure Trend (3 hours)
    # Assuming data is roughly hourly. diff(3) compares current vs 3 rows ago.
    df['pressure_trend'] = df['pressure'].diff(3)

    # Drop rows where we couldn't calc trend (first 3 rows)
    return df.dropna()

def analyze_rain_conditions(df):
    """Prints statistics for rows where it is actually raining."""
    rain_df = df[df['rain'] > 0.2]
    if rain_df.empty:
        print("\n⚠️ No rain events found in data (>0.2mm). Cannot analyze conditions.")
        return

    print("\n🧐 CONDITIONS DURING RAIN (Ground Truth):")
    print(rain_df[['humidity', 'dpd', 'pressure_trend', 'wind_speed', 'light']].describe().loc[['mean', '50%', 'min', 'max']])

def evaluate_scenario(df, name, thresh):
    """Applies thresholds and calculates accuracy metrics."""
    
    # 1. Apply Physics Logic (Rule B: Dynamic System)
    # Rain if: High Humidity AND Low DPD AND Falling Pressure AND Wind AND Low Light
    pred_rain = (
        (df['humidity'] >= thresh['rh']) &
        (df['dpd'] <= thresh['dpd']) &
        (df['pressure_trend'] <= thresh['press']) &
        (df['wind_speed'] >= thresh['wind']) &
        (df['light'] < thresh['lux'])
    )

    # 2. Ground Truth (Actual Rain > 0.2mm)
    actual_rain = df['rain'] > 0.2

    # 3. Calculate Metrics
    tp = ((pred_rain == True) & (actual_rain == True)).sum()
    fp = ((pred_rain == True) & (actual_rain == False)).sum()
    fn = ((pred_rain == False) & (actual_rain == True)).sum()
    tn = ((pred_rain == False) & (actual_rain == False)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n🔹 Scenario: {name}")
    print(f"   Settings:  RH>={thresh['rh']}% | DPD<={thresh['dpd']} | PressChange<={thresh['press']} | Wind>={thresh['wind']} | Light<{thresh['lux']}")
    print(f"   [Stats]    TP: {tp} (Caught) | FP: {fp} (False Alarm) | FN: {fn} (Missed)")
    print(f"   [Metrics]  Precision: {precision:.2f} | Recall: {recall:.2f} | F1 Score: {f1:.2f}")

def main():
    df = load_and_prep_data()
    if df.empty: return

    print(f"📊 Evaluating {len(df)} historical records for Rain Detection...")
    analyze_rain_conditions(df)

    # --- SCENARIO 1: STRICT (Original) ---
    # Very safe, but might miss light rain
    evaluate_scenario(df, "STRICT (Original)", {
        'rh': 85, 'dpd': 2.0, 'press': -1.0, 'wind': 8.0, 'lux': 10000
    })

    # --- SCENARIO 2: BALANCED (Recommended) ---
    # Relaxed thresholds to catch more events
    evaluate_scenario(df, "BALANCED (Recommended)", {
        'rh': 80, 'dpd': 3.0, 'press': -0.5, 'wind': 5.0, 'lux': 20000
    })

    # --- SCENARIO 3: HIGH SENSITIVITY ---
    # Will catch almost everything, but many false alarms
    evaluate_scenario(df, "HIGH SENSITIVITY", {
        'rh': 75, 'dpd': 4.0, 'press': -0.2, 'wind': 2.0, 'lux': 30000
    })

    # --- SCENARIO 4: SATURATION ONLY (New) ---
    # Ignores Wind/Pressure (Good for bad sensors or calm rain)
    evaluate_scenario(df, "SATURATION ONLY (Fallback)", {
        'rh': 90, 'dpd': 2.0, 'press': 9999, 'wind': -1.0, 'lux': 15000
    })

    # --- SCENARIO 5: OPTIMIZED (Data-Driven) ---
    # Based on Ground Truth: Median RH=99.5%, Mean Wind=2.9
    evaluate_scenario(df, "OPTIMIZED (Data-Driven)", {
        'rh': 96, 'dpd': 1.5, 'press': 9999, 'wind': -1.0, 'lux': 10000
    })

    # --- SCENARIO 6: BEST / CHOSEN (Wind Filtered) ---
    # Same as Optimized, but requires Wind > 2.0 km/h to filter out Fog
    evaluate_scenario(df, "BEST (Wind Filtered)", {
        'rh': 96, 'dpd': 1.5, 'press': 9999, 'wind': 2.0, 'lux': 10000
    })

if __name__ == "__main__":
    main()
