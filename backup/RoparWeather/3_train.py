import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# CONFIG
DATA_FILE = "clean_training_data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load Data
if not os.path.exists(DATA_FILE):
    print(f"❌ File {DATA_FILE} not found. Run Step 2 first!")
    exit()

df = pd.read_csv(DATA_FILE)
print(f"🧠 Loaded {len(df)} training samples.")

# Features (Include your new Fog/Wind sensors)
features = [
    'om_temp', 'om_hum', 'om_press', 'om_wind', 'om_rain', 
    'om_solar', 'om_wind_dir', 
    'hour'
]
target = 'temp'

print("\nTraining Monthly Models & Applying Safety Checks...")
print(f"{'Month':<6} | {'Samples':<8} | {'Raw Error':<10} | {'AI Error':<10} | {'Action'}")
print("-" * 65)

for month in sorted(df['month'].unique()):
    month_df = df[df['month'] == month]
    
    # 1. Check Data Quantity
    if len(month_df) < 50:
        print(f"{month:<6} | {len(month_df):<8} | N/A        | N/A        | ⚠️ Skip (Low Data)")
        continue

    # 2. Train/Test Split
    split_idx = int(len(month_df) * 0.8)
    train = month_df.iloc[:split_idx]
    test = month_df.iloc[split_idx:]

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    # 3. Train Model
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)

    # 4. Evaluate
    preds = rf.predict(X_test)
    ai_mae = mean_absolute_error(y_test, preds)
    raw_mae = mean_absolute_error(y_test, X_test['om_temp'])
    
    model_path = f"{MODEL_DIR}/rf_month_{int(month)}.pkl"

    # 5. DECISION LOGIC: Only save if AI is SMARTER than Raw
    # We add a small buffer (0.05) to prevent switching for tiny gains
    if ai_mae < (raw_mae - 0.05):
        joblib.dump(rf, model_path)
        status = "✅ SAVED"
    else:
        # If model exists from previous run, DELETE it because it's now bad
        if os.path.exists(model_path):
            os.remove(model_path)
            status = "🗑️ DELETED"
        else:
            status = "❌ DISCARDED"

    print(f"{month:<6} | {len(month_df):<8} | {raw_mae:.2f}°C     | {ai_mae:.2f}°C     | {status}")

print(f"\n🎉 Optimization Complete. Only high-performance models remain in '{MODEL_DIR}/'.")