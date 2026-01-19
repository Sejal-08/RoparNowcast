import pandas as pd
import plotly.graph_objects as go

# ================================
# LOAD DATA
# ================================
df = pd.read_csv("open_meteo_corrected_forecast_clean.csv")
df["time"] = pd.to_datetime(df["time"])

# First 48 hours
df = df.head(48)

# ================================
# CREATE INTERACTIVE PLOT
# ================================
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["time"],
        y=df["corrected_temperature"],
        mode="lines+markers",
        name="Hyper-local Forecast",
        line=dict(color="orange", width=2),
        marker=dict(size=6),
        hovertemplate=
        "<b>Date:</b> %{x|%d-%m-%Y}<br>" +
        "<b>Time:</b> %{x|%I:%M %p}<br>" +
        "<b>Temperature:</b> %{y:.1f} °C<br>" +
        "<extra></extra>"
    )
)

# ================================
# LAYOUT (IMD STYLE)
# ================================
fig.update_layout(
    title="Hyper-Local Temperature Forecast (IMD-Style Interactive)",
    xaxis_title="Time",
    yaxis_title="Temperature (°C)",
    hovermode="x unified",
    template="plotly_white",
    width=1200,
    height=450
)

# ================================
# SHOW IN BROWSER
# ================================
fig.show()
