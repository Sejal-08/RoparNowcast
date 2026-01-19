import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ================================
# CONFIG
# ================================
LAT, LON = 30.96, 76.47
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    'temperature_2m',
    'relative_humidity_2m',
    'hour',
    'AtmPressure',
    'WindSpeed',
    'RainfallHourly'
]

TARGET = 'CurrentTemperature'

# ================================
# 1. LOAD CLEAN STATION DATA
# ================================
station_df = pd.read_csv('station_data_clean.csv')
station_df['TimeStamp'] = pd.to_datetime(station_df['TimeStamp'])

# ================================
# 2. QUALITY FILTERS (VERY IMPORTANT)
# ================================
station_df = station_df[
    (station_df['CurrentTemperature'].between(-10, 55)) &
    (station_df['CurrentHumidity'].between(0, 100)) &
    (station_df['Latitude'] != 0) &
    (station_df['Longitude'] != 0) &
    (station_df['RainfallHourly'] <= 300)
]

# Optional wind filter if available
if 'WindSpeed' in station_df.columns:
    station_df = station_df[station_df['WindSpeed'] <= 60]

print(f"✅ Station data after QC: {station_df.shape}")

# ================================
# 3. HOURLY AGGREGATION
# ================================
station_df['hour_sync'] = station_df['TimeStamp'].dt.floor('h')

station_hourly = (
    station_df
    .groupby('hour_sync')
    .mean(numeric_only=True)
    .reset_index()
)

# ================================
# 4. LOAD OPEN-METEO HISTORICAL
# ================================
api_df = pd.read_csv('open_meteo.csv')
api_df['time'] = pd.to_datetime(api_df['time'])

# ================================
# 5. MERGE DATA
# ================================
merged = pd.merge(
    station_hourly,
    api_df,
    left_on='hour_sync',
    right_on='time',
    how='inner'
)

merged['hour'] = merged['hour_sync'].dt.hour
merged['month'] = merged['hour_sync'].dt.month
merged['temp_bias'] = merged['CurrentTemperature'] - merged['temperature_2m']

merged.to_csv('merged_training_data_clean.csv', index=False)

print("✅ Merged training data saved")

# ================================
# 6. MONTH-WISE RANDOM FOREST
# ================================
metrics = {}
monthly_models = {}

for month in sorted(merged['month'].unique()):
    month_df = merged[merged['month'] == month].dropna(
        subset=FEATURES + [TARGET]
    )

    if len(month_df) < 100:
        print(f"⚠️ Skipping month {month} (low data)")
        continue

    X = month_df[FEATURES]
    y = month_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    joblib.dump(model, f"{MODEL_DIR}/rf_month_{month}.pkl")
    monthly_models[month] = model
    metrics[month] = round(mae, 2)

    print(f"📅 Month {month}: MAE = {mae:.2f}°C")

pd.DataFrame.from_dict(
    metrics, orient='index', columns=['MAE']
).to_csv('model_metrics_clean.csv')

print("✅ Model metrics saved")

# ================================
# 7. FUTURE FORECAST + CORRECTION
# ================================
forecast_url = (
    f"https://api.open-meteo.com/v1/forecast?"
    f"latitude={LAT}&longitude={LON}&"
    f"hourly=temperature_2m,relative_humidity_2m,surface_pressure&"
    f"forecast_days=7&timezone=auto"
)

future_raw = pd.read_json(forecast_url)

future_df = pd.DataFrame({
    "time": pd.to_datetime(future_raw["hourly"]["time"]),
    "temperature_2m": future_raw["hourly"]["temperature_2m"],
    "relative_humidity_2m": future_raw["hourly"]["relative_humidity_2m"],
    "AtmPressure": future_raw["hourly"]["surface_pressure"]
})

future_df['hour'] = future_df['time'].dt.hour
future_df['month'] = future_df['time'].dt.month

def apply_correction(row):
    model = monthly_models.get(row['month'])
    if model is None:
        return None
    X = [[
        row['temperature_2m'],
        row['relative_humidity_2m'],
        row['hour'],
        row['AtmPressure'],
        0.0,   # WindSpeed unknown for future
        0.0    # RainfallHourly unknown
    ]]
    return model.predict(X)[0]

future_df['corrected_temperature'] = future_df.apply(
    apply_correction, axis=1
)

future_df.to_csv('open_meteo_corrected_forecast_clean.csv', index=False)

print("✅ Final hyper-local forecast saved")
print(future_df[['time', 'temperature_2m', 'corrected_temperature']].head())
