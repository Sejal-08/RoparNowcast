import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from config import settings

def prepare_features(df):
    """
    Creates Lag Features for the Nowcasting Model.
    """
    df_processed = df.copy()
    
    # 1. Target Variables
    targets = ["temp", "humidity", "pressure", "rain", "wind_speed"]
    
    # 2. Create Lag Features
    feature_cols = []
    
    # Check which columns actually exist/have data
    valid_data_cols = [c for c in ["temp", "humidity", "pressure", "rain", "wind_speed", "light"] if c in df.columns]
    
    for i in range(1, settings.LAGS + 1):
        for col in ["temp", "humidity", "pressure", "rain", "wind_speed", "light"]:
            # We create the feature name even if the column is missing (fill with 0)
            # This ensures the feature list is always consistent for the predictor
            feat_name = f"{col}_lag_{i}"
            if col in valid_data_cols:
                df_processed[feat_name] = df[col].shift(i)
            else:
                df_processed[feat_name] = 0.0
            feature_cols.append(feat_name)
    
    # 3. Add Time Features
    df_processed["hour"] = df_processed.index.hour
    df_processed["month"] = df_processed.index.month
    feature_cols.extend(["hour", "month"])
    
    # 4. Add Global Forecast Features (Placeholders)
    # --- FIX: Explicit Mapping to match nowcast_predict.py ---
    om_map = {
        "temp": "om_temp",
        "humidity": "om_hum",
        "pressure": "om_press",
        "wind_speed": "om_wind",
        "rain": "om_rain"
    }

    for target in targets:
        # Use the map to get the correct name (e.g., "om_temp" instead of "om_tem")
        om_col = om_map.get(target, f"om_{target}")
        
        # If column exists, use it as the "Perfect Forecast" for training
        if target in df_processed.columns:
            df_processed[om_col] = df_processed[target] 
        else:
            df_processed[om_col] = 0.0
        
        feature_cols.append(om_col)
        
    # Add Solar (Light converted to 0-1 range approx or scaled)
    df_processed["om_solar"] = df_processed.get("light", 0) / 100.0
    feature_cols.append("om_solar")

    return df_processed, feature_cols

def run_training():
    print("🔬 [ROPAR NOWCAST] Training 6-Feature Spectrum Models...")
    
    # 1. Load Data
    if not os.path.exists(settings.DATA_FILE):
        print(f"❌ Data file not found: {settings.DATA_FILE}")
        return

    df = pd.read_csv(settings.DATA_FILE)
    
    # Normalize Time Column
    if "merge_time" in df.columns:
        df = df.rename(columns={"merge_time": "time"})
        
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='last')]
    
    print(f"   -> Loaded {len(df)} rows of data.")
    
    # 2. Prepare Data
    df_processed, feature_cols = prepare_features(df)
    
    # Save Feature Names
    model_dir = os.path.join(settings.MODEL_DIR, "nowcast")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(feature_cols, os.path.join(model_dir, "feature_names.pkl"))
    
    # 3. Train Loop
    targets = ["temp", "humidity", "pressure", "rain", "wind_speed"]
    steps_per_hour = 60 // settings.TIME_STEP_MINUTES
    
    print(f"\n{'Variable':<10} | {'Horizon':<8} | {'MAE':<6}")
    print("-" * 35)

    trained_count = 0
    
    for target in targets:
        # Check if target actually has data
        if target not in df.columns or df[target].count() < 100:
            print(f"⚠️ Skipping {target} (Not enough data / All NaNs)")
            continue

        # Custom Horizons: Always include h01 (5 min) for Bias Correction, then hourly
        horizons = [1] + list(range(steps_per_hour, settings.N_HORIZONS + 1, steps_per_hour))

        for step in horizons:
            lead_min = step * 5
            
            # Define Target (Future Value)
            y = df_processed[target].shift(-step)
            X = df_processed[feature_cols]
            
            # --- ROBUST CLEANING ---
            valid_idx = y.dropna().index
            
            # Align X and fill missing Input Features with 0
            X_clean = X.loc[valid_idx].fillna(0)
            y_clean = y.loc[valid_idx]
            
            if len(X_clean) < 50:
                if step == steps_per_hour:
                    print(f"⚠️ {target} skipped: Only {len(X_clean)} valid rows.")
                continue

            # Train XGBoost
            model = XGBRegressor(
                n_estimators=100, 
                learning_rate=0.05, 
                max_depth=5, 
                n_jobs=1
            )
            model.fit(X_clean, y_clean)
            
            # Evaluate
            preds = model.predict(X_clean)
            mae = mean_absolute_error(y_clean, preds)
            
            # Save Model
            model_path = os.path.join(model_dir, f"xgb_{target}_h{step:02d}.pkl")
            joblib.dump(model, model_path)
            
            print(f"{target:<10} | +{lead_min:>3} min | {mae:.2f}")
            trained_count += 1

    if trained_count > 0:
        print("\n✅ All Models Trained Successfully.")
    else:
        print("\n❌ No models were trained. Check your data file for NaNs.")

if __name__ == "__main__":
    run_training()