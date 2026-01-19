import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Ropar Weather AI", layout="wide")

# Load Data
try:
    df = pd.read_csv("latest_forecast.csv")
    df['time'] = pd.to_datetime(df['time'])
except:
    st.error("No forecast found. Run 4_predict.py first!")
    st.stop()

# Header
st.title("📍 Hyper-Local Weather: Ropar")
st.markdown(f"**Lat:** 30.96 | **Lon:** 76.47 | **Last Update:** {df['time'].iloc[0]}")

# Metrics
col1, col2, col3 = st.columns(3)
current_temp = df['ai_temp'].iloc[0]
source = df['source'].iloc[0]
is_ai = source == "AI"

col1.metric("Current Temp", f"{current_temp:.1f} °C")
col2.metric("Source", source, delta="Custom AI Active" if is_ai else "Global Fallback", delta_color="normal")
col3.metric("Next 24h High", f"{df['ai_temp'].iloc[:24].max():.1f} °C")

# Chart
fig = go.Figure()

# Raw Global Forecast
fig.add_trace(go.Scatter(
    x=df['time'], y=df['om_temp'],
    mode='lines', name='Global Model (Satellite)',
    line=dict(color='gray', dash='dot')
))

# Your AI Forecast
fig.add_trace(go.Scatter(
    x=df['time'], y=df['ai_temp'],
    mode='lines+markers', name='Hyper-Local AI',
    line=dict(color='#00CC96', width=3)
))

st.plotly_chart(fig, use_container_width=True)

# Data Table
st.dataframe(df[['time', 'om_temp', 'ai_temp', 'source']].style.highlight_max(axis=0))