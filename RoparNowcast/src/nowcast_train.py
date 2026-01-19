import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error

# Fix Import Path for 'config'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

def create_lagged_features(df):
    """Creates history columns (T-1, T-2, T-3) for all variables"""
    df = df.sort_values('merge_time').copy()
    
    # --- UPDATE 1: ADD 'light' TO FEATURES ---
    # We now track 6 variables (including Light Intensity)
    obs_cols = ['temp', 'humidity', 'pressure', 'rain', 'wind_speed', 'light']
    
    # --- UPDATE 2: WIND SPEED UNIT FIX ---
    # Ensure training data is in km/h (Standardize)
    if 'wind_speed' in df.columns:
        # Heuristic: If max wind is small (< 20), it's likely m/s. Convert to km/h.
        if df['wind_speed'].max() < 20:
            print("⚠️ Converting Training Data Wind Speed from m/s to km/h")
            df['wind_speed'] = df['wind_speed'] * 3.6

    for lag in range(1, settings.LAGS + 1):
        for col in obs_cols:
            # Check if col exists (older data might miss 'light')
            if col in df.columns:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            else:
                # Fill missing columns with 0 to prevent crash
                df[f'{col}_lag_{lag}'] = 0
            
    return df.dropna()

def run_training():
    print("🔬 [ROPAR NOWCAST] Training 6-Feature Spectrum Models (Inc. Light)...")
    
    if not os.path.exists(settings.DATA_FILE):
        print(f"❌ Error: Training data not found at {settings.DATA_FILE}")
        return

    # 1. Load Data
    df = pd.read_csv(settings.DATA_FILE)
    df['merge_time'] = pd.to_datetime(df['merge_time'])
    
    # [NEW] Ensure derived time features exist (in case new dataset lacks them)
    if 'hour' not in df.columns:
        df['hour'] = df['merge_time'].dt.hour
    if 'month' not in df.columns:
        df['month'] = df['merge_time'].dt.month

    # 🧹 CLEANING: Treat 0.0 Pressure/Humidity as Sensor Failure (NaN)
    # This ensures we don't train on invalid data or lags derived from it.
    df['pressure'] = df['pressure'].replace(0.0, np.nan)
    df['humidity'] = df['humidity'].replace(0.0, np.nan)
    
    # 2. Prepare History Features
    df_lagged = create_lagged_features(df)
    
    # Define Input Features (X)
    # Note: 'om_solar' is the Satellite's version of Light
    # REMOVED 'month': We don't have a full year of data yet, so 'month' causes overfitting to Autumn.
    features_base = ['om_temp', 'om_hum', 'om_press', 'om_wind', 'om_rain', 'om_solar', 'hour']
    features_lags = [c for c in df_lagged.columns if '_lag_' in c]
    X_cols = features_base + features_lags
    
    # Setup Model Directory
    model_path = os.path.join(settings.MODEL_DIR, "nowcast")
    os.makedirs(model_path, exist_ok=True)
    
    # Save feature names (Crucial for prediction alignment)
    joblib.dump(X_cols, os.path.join(model_path, "feature_names.pkl"))

    # 3. Training Loop (5 Targets x 6 Horizons)
    # We predict 5 vars (Temp, Hum, Wind, Rain, Press). We use Light as input, but rarely predict it.
    targets = ['temp', 'humidity', 'wind_speed', 'rain', 'pressure']
    
    # [OPTIMIZATION] Calculate steps per hour (e.g. 60 / 5 = 12 steps)
    steps_per_hour = 60 // settings.TIME_STEP_MINUTES

    print(f"\n{'Variable':<10} | {'Horizon':<8} | {'MAE':<8}")
    print("-" * 35)

    for var in targets:
        for h in settings.HORIZONS:
            # OPTIMIZATION: Only train h=1 (for Bias) and Hourly steps (12, 24, 36...)
            # Skip intermediate steps (2, 3, 4...) to save time/space
            if h != 1 and h % steps_per_hour != 0:
                continue

            target_col_name = f'target_{var}_h{h:02d}'

            df_lagged[target_col_name] = df_lagged[var].shift(-h)
            valid = df_lagged.dropna(subset=[target_col_name])

            if len(valid) < 50:
                print(f"⚠️ Not enough data to train {var} at +{h * settings.TIME_STEP_MINUTES} min. Skipping.")
                continue

            split = int(len(valid) * 0.8)
            train = valid.iloc[:split]
            test = valid.iloc[split:]

            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                objective='reg:absoluteerror',
                n_jobs=-1
            )

            model.fit(train[X_cols], train[target_col_name])

            preds = model.predict(test[X_cols])
            mae = mean_absolute_error(test[target_col_name], preds)

            model_name = f"xgb_{var}_h{h:02d}.pkl"
            joblib.dump(model, os.path.join(model_path, model_name))

            lead_min = h * settings.TIME_STEP_MINUTES
            print(f"{var:<10} | +{lead_min:>3} min | {mae:.2f}")
    print("\n✅ All Models Trained Successfully.")

if __name__ == "__main__":
    run_training()