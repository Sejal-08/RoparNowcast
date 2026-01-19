import pandas as pd

# ================================
# CONFIG
# ================================
INPUT_FILE = "results (30).csv"
OUTPUT_FILE = "station_data_clean.csv"

COLUMNS_TO_KEEP = [
    "DeviceId",
    "TimeStamp",
    "AtmPressure",
    "CurrentHumidity",
    "CurrentTemperature",
    "Latitude",
    "Longitude",
    "RainfallHourly",
    "WindDirection",
    "WindSpeed"
]

# ================================
# 1. LOAD RAW DATA
# ================================
df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"📥 Loaded raw data: {df.shape}")

# ================================
# 2. SELECT REQUIRED COLUMNS
# ================================
missing_cols = [c for c in COLUMNS_TO_KEEP if c not in df.columns]
if missing_cols:
    raise ValueError(f"❌ Missing required columns: {missing_cols}")

df = df[COLUMNS_TO_KEEP]

print(f"✂️ Columns filtered: {df.shape}")

# ================================
# 3. FIX DATA TYPES
# ================================
df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")

numeric_cols = [
    "AtmPressure",
    "CurrentHumidity",
    "CurrentTemperature",
    "Latitude",
    "Longitude",
    "RainfallHourly",
    "WindDirection",
    "WindSpeed"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ================================
# 4. DROP BAD ROWS
# ================================
before = len(df)

df = df.dropna(subset=["TimeStamp", "CurrentTemperature"])

after = len(df)

print(f"🧹 Dropped {before - after} invalid rows")

# ================================
# 5. SORT & SAVE
# ================================
df = df.sort_values("TimeStamp").reset_index(drop=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\n✅ CLEAN STATION DATA SAVED")
print(f"📄 File: {OUTPUT_FILE}")
print(f"📊 Final shape: {df.shape}")
print("\nSample:")
print(df.head())
