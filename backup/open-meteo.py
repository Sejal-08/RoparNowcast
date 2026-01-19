import pandas as pd

lat, lon = 30.96, 76.47
start_date = "2025-06-20"
end_date = "2026-01-10"

url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={lat}&longitude={lon}&"
    f"start_date={start_date}&end_date={end_date}&"
    f"hourly=temperature_2m,relative_humidity_2m,surface_pressure&"
    f"format=csv"
)

try:
    df = pd.read_csv(
        url,
        skiprows=9,
        names=["time", "temperature_2m", "relative_humidity_2m", "surface_pressure"]
    )

    df["time"] = pd.to_datetime(df["time"])

    if not df.empty:
        print("✅ Open-Meteo data downloaded successfully")
        print(f"📊 Total rows: {len(df)}")
        print(df.head())

        # ✅ SAVE DATA
        output_file = "open_meteo.csv"
        df.to_csv(output_file, index=False)
        print(f"💾 Data saved as '{output_file}'")

    else:
        print("⚠️ Data downloaded but DataFrame is empty")

except Exception as e:
    print("❌ Failed to download Open-Meteo data")
    print("Error:", e)
