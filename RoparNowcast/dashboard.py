import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys
import time
import requests
import urllib3
from datetime import datetime, timedelta

# Ensure config is found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import settings

# Suppress InsecureRequestWarning when bypassing SSL verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="Ropar Nowcast Center", layout="wide", page_icon="⚡")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("⚙️ Controls")
    if st.button("🗑️ Clear Forecast History", type="primary"):
        history_path = os.path.join(settings.DATA_DIR, "forecast_history.csv")
        if os.path.exists(history_path):
            os.remove(history_path)
            st.success("✅ History cleared!")
            time.sleep(1)
            st.rerun()
        else:
            st.warning("ℹ️ No history file found.")

    # Download Button
    history_path = os.path.join(settings.DATA_DIR, "forecast_history.csv")
    if os.path.exists(history_path):
        with open(history_path, "rb") as f:
            st.download_button(
                label="📥 Download Report (CSV)",
                data=f,
                file_name="forecast_history.csv",
                mime="text/csv"
            )

# 1. Load Data
if not os.path.exists(settings.OUTPUT_FILE):
    st.error(f"❌ Waiting for data... (File not found: {settings.OUTPUT_FILE})")
    st.stop()

# Load and process data
df = pd.read_csv(settings.OUTPUT_FILE)
df['time'] = pd.to_datetime(df['time'])

# Sort and get current prediction (first row)
curr = df.iloc[0]

# Safe access for new columns (in case CSV is old)
condition = curr.get('condition', 'Processing...')
dew_point = curr.get('dew_point', 0.0)

# --- HEADER ---
c_head_1, c_head_2 = st.columns([3, 1])
with c_head_1:
    st.title("⚡ Ropar Hyper-Local Nowcast")
    st.markdown(f"**Forecast Generated:** {curr['time'].strftime('%d-%b %H:%M')} | **Window:** Next 3 Hours")
with c_head_2:
    # Big Condition Badge
    st.markdown(f"<h2 style='text-align: center; color: #4F8BF9;'>{condition}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>Dew Point: {dew_point:.1f}°C</p>", unsafe_allow_html=True)

st.markdown("---")

# --- ROW 1: LIVE METRICS ---
c1, c2, c3, c4, c5 = st.columns(5)

def render_metric(col, label, val_ai, val_global, unit):
    diff = val_ai - val_global
    col.metric(
        label=label,
        value=f"{val_ai:.1f} {unit}",
        delta=f"{diff:+.1f} {unit}",
        delta_color="off"
    )

with c1: render_metric(st, "Temperature", curr['temp'], curr['temp_global'], "°C")
with c2: render_metric(st, "Humidity", curr['humidity'], curr['humidity_global'], "%")
with c3: render_metric(st, "Pressure", curr['pressure'], curr['pressure_global'], "hPa")
with c4: render_metric(st, "Wind Speed", curr['wind_speed'], curr['wind_speed_global'], "km/h")
with c5: render_metric(st, "Rainfall", curr['rain'], curr['rain_global'], "mm")

st.markdown("---")

# --- ROW 2: DETAILED GRAPHS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🌡️ Temp", "💧 Humidity", "⏲️ Pressure", "🌬️ Wind", "🌧️ Rain", "🔬 Physics", "🆚 Actuals"])

def get_recent_actuals(hours=24):
    """Fetches recent actual data for comparison"""
    end = datetime.now()
    start = end - timedelta(hours=hours)
    params = {
        "deviceid": settings.DEVICE_ID,
        "startdate": start.strftime("%d-%m-%Y"),
        "enddate": end.strftime("%d-%m-%Y")
    }
    try:
        r = requests.get(settings.SOURCE_API_URL, params=params, timeout=5, verify=False)
        data = r.json()
        if isinstance(data, str): import json; data = json.loads(data)
        
        df_act = pd.DataFrame(data['items'])
        df_act['time'] = pd.to_datetime(df_act['TimeStamp'])
        
        # Map columns
        df_act = df_act.rename(columns={"CurrentTemperature": "temp", "CurrentHumidity": "humidity", "AtmPressure": "pressure", "WindSpeed": "wind_speed"})
        for c in ["temp", "humidity", "pressure", "wind_speed"]: df_act[c] = pd.to_numeric(df_act[c], errors='coerce')
        df_act["wind_speed"] *= 3.6 # m/s to km/h
        return df_act.sort_values('time')
    except:
        return pd.DataFrame()

# Error growth rates per hour (Heuristic based on MAE)
ERROR_RATES = {
    "temp": 0.5,        # +/- 0.5°C per hour
    "humidity": 5.0,    # +/- 5% per hour
    "pressure": 0.5,    # +/- 0.5 hPa per hour
    "wind_speed": 2.0   # +/- 2 km/h per hour
}

def plot_graph(tab, var_col, glob_col, title, color, is_bar=False):
    with tab:
        fig = go.Figure()
        
        if is_bar:
            fig.add_trace(go.Bar(
                x=df['time'], y=df[var_col],
                name='AI Nowcast', marker_color=color
            ))
        else:
            fig.add_trace(go.Scatter(
                x=df['time'], y=df[var_col],
                mode='lines+markers', name='AI Nowcast',
                line=dict(color=color, width=4)
            ))
        
        # Global Line (Baseline)
        if glob_col:
            fig.add_trace(go.Scatter(
                x=df['time'], y=df[glob_col],
                mode='lines', name='Global Model',
                line=dict(color='gray', dash='dot', width=2)
            ))
            
        # Add Confidence Interval (Shaded Area)
        if not is_bar and var_col in ERROR_RATES:
            rate = ERROR_RATES[var_col]
            hours_out = df['lead_minutes'] / 60.0
            margin = hours_out * rate
            
            upper = df[var_col] + margin
            lower = df[var_col] - margin
            
            # Add bounds (Upper transparent, Lower filled)
            fig.add_trace(go.Scatter(
                x=df['time'], y=upper, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            
            # Convert hex to rgba for transparency
            hex_c = color.lstrip('#')
            rgb = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
            fill_color = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.2)"

            fig.add_trace(go.Scatter(
                x=df['time'], y=lower, mode='lines', line=dict(width=0), fill='tonexty', 
                fillcolor=fill_color,
                name='Confidence (±)', hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=title, 
            height=350, 
            margin=dict(t=40, b=20, l=20, r=20),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

plot_graph(tab1, 'temp', 'temp_global', "Temperature Forecast", "#FF4B4B")
plot_graph(tab2, 'humidity', 'humidity_global', "Humidity Forecast", "#00CC96")
plot_graph(tab3, 'pressure', 'pressure_global', "Atmospheric Pressure Forecast", "#AB63FA")
plot_graph(tab4, 'wind_speed', 'wind_speed_global', "Wind Speed Forecast", "#636EFA")
plot_graph(tab5, 'rain', 'rain_global', "Rainfall Forecast", "#FFA15A", is_bar=True)

# TAB 6: NEW PHYSICS VIEW
with tab6:
    st.info("ℹ️ **Dew Point Depression:** When the Red line (Temp) touches the Green line (Dew Point), FOG or RAIN happens.")
    fig_phys = go.Figure()
    fig_phys.add_trace(go.Scatter(x=df['time'], y=df['temp'], name='Temperature', line=dict(color='#FF4B4B', width=3)))
    fig_phys.add_trace(go.Scatter(x=df['time'], y=df['dew_point'], name='Dew Point', line=dict(color='#00CC96', width=3, dash='dash')))
    fig_phys.update_layout(title="Thermodynamic State (Saturation Check)", height=350)
    st.plotly_chart(fig_phys, use_container_width=True)

# TAB 7: ACTUALS VS FORECAST
with tab7:
    st.markdown("### 🆚 Forecast vs Observed (Last 24 Hours)")
    actuals = get_recent_actuals(hours=24)
    
    # Load Forecast History
    history_path = os.path.join(settings.DATA_DIR, "forecast_history.csv")
    hist_1h = pd.DataFrame()
    
    if os.path.exists(history_path):
        try:
            hist_df = pd.read_csv(history_path)
            hist_df['time'] = pd.to_datetime(hist_df['time'])
            # Filter for 1-hour lead times (Lead=60 minutes) to see short-term accuracy
            hist_1h = hist_df[hist_df['lead_minutes'] == 60].sort_values('time')
            # Deduplicate in case of overlapping runs
            hist_1h = hist_1h.drop_duplicates(subset=['time'], keep='last')
        except Exception as e:
            st.warning(f"Error loading forecast history: {e}")

    if not actuals.empty:
        var_comp = st.selectbox("Select Variable", ["temp", "humidity", "pressure", "wind_speed"], index=0)
        fig_comp = go.Figure()
        # 1. Observed Data (Black Line)
        fig_comp.add_trace(go.Scatter(x=actuals['time'], y=actuals[var_comp], name="Observed", line=dict(color='black', width=3)))
        # 2. Historical Forecast (Blue Dotted) - What we predicted 1 hour ago
        if not hist_1h.empty:
            fig_comp.add_trace(go.Scatter(x=hist_1h['time'], y=hist_1h[var_comp], name="Forecast (1h Lead)", line=dict(color='blue', dash='dot', width=2)))
        
        fig_comp.update_layout(title=f"Observed vs Forecast (1h Lead): {var_comp.capitalize()}", height=350)
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.warning("Could not fetch recent actuals.")

# --- ROW 3: RAW DATA ---
# --- ROW 3: RAW DATA ---
with st.expander("📂 View Raw Data Table"):
    # Fix: Apply formatting ONLY to numeric columns, leave Time/Text alone
    st.dataframe(
        df.style.format(
            formatter="{:.2f}", 
            subset=df.select_dtypes(include="number").columns
        )
    )
# --- AUTO REFRESH ---
# Refreshes the page every 5 minutes (300 seconds) to match the prediction cycle
time.sleep(300)
st.rerun()