import pandas as pd
import matplotlib
matplotlib.use("Agg")   # <-- IMPORTANT (non-GUI backend)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ================================
# LOAD DATA
# ================================
df = pd.read_csv("open_meteo_corrected_forecast_clean.csv")
df["time"] = pd.to_datetime(df["time"])

# First 48 hours
df = df.head(48)

# ================================
# CREATE IMD-STYLE PLOT
# ================================
plt.figure(figsize=(18, 5))

plt.plot(
    df["time"],
    df["corrected_temperature"],
    color="orange",
    marker="o",
    linewidth=2,
    markersize=5
)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%d/%m\n%I %p")
)

plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Hyper-Local Temperature Forecast (IMD-Style)")
plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()

# ================================
# SAVE ONLY (NO plt.show())
# ================================
plt.savefig("hyperlocal_imd_style_temperature.png", dpi=150)
plt.close()

print("✅ Plot saved as hyperlocal_imd_style_temperature.png")
