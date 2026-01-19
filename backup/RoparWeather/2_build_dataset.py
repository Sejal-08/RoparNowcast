import pandas as pd
import numpy as np
import requests
import boto3
from boto3.dynamodb.conditions import Key

# CONFIG
LAT, LON = 30.96, 76.47
DYNAMO_TABLE = "RoparWeather_History"
DEVICE_ID = "1"

def get_clean_station_data():
    print("1️⃣ Downloading Ground Truth (with Light & Wind Dir)...")
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table(DYNAMO_TABLE)
    
    # Fetch Data
    response = table.query(KeyConditionExpression=Key('device_id').eq(DEVICE_ID))
    items = response['Items']
    while 'LastEvaluatedKey' in response:
        response = table.query(KeyConditionExpression=Key('device_id').eq(DEVICE_ID), ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response['Items'])

    df = pd.DataFrame(items)
    cols = ['temp', 'humidity', 'pressure', 'rain', 'wind_speed', 'light', 'wind_dir']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 🧹 QC LAYER
    df = df.dropna(subset=cols)
    df = df[(df['temp'].between(-5, 55)) & (df['humidity'].between(0, 100))]
    
    # 📐 VECTOR MATH FOR WIND DIRECTION
    # Convert Degrees to Radians -> Sin/Cos
    df['wd_sin'] = np.sin(np.deg2rad(df['wind_dir']))
    df['wd_cos'] = np.cos(np.deg2rad(df['wind_dir']))

    # Round to Hour
    df['merge_time'] = df['timestamp'].dt.round('h')
    
    # Aggregate (Average)
    agg_cols = ['temp', 'humidity', 'pressure', 'rain', 'wind_speed', 'light', 'wd_sin', 'wd_cos']
    df_hourly = df.groupby('merge_time')[agg_cols].mean().reset_index()
    
    return df_hourly

def get_meteo_history(start, end):
    print(f"2️⃣ Fetching Global Models (Solar & Wind Dir included)...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT, "longitude": LON, "start_date": start, "end_date": end,
        # ✅ Added shortwave_radiation (Light) and wind_direction_10m
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,rain,shortwave_radiation,wind_direction_10m",
        "timezone": "auto"
    }
    data = requests.get(url, params=params).json()
    df = pd.DataFrame(data['hourly'])
    df['time'] = pd.to_datetime(df['time'])
    
    return df.rename(columns={
        'temperature_2m': 'om_temp', 'relative_humidity_2m': 'om_hum',
        'surface_pressure': 'om_press', 'wind_speed_10m': 'om_wind', 'rain': 'om_rain',
        'shortwave_radiation': 'om_solar', 'wind_direction_10m': 'om_wind_dir'
    })

if __name__ == "__main__":
    station_df = get_clean_station_data()
    if station_df.empty: exit()

    min_date = station_df['merge_time'].min().strftime("%Y-%m-%d")
    max_date = station_df['merge_time'].max().strftime("%Y-%m-%d")
    meteo_df = get_meteo_history(min_date, max_date)
    
    # Merge
    final_df = pd.merge(station_df, meteo_df, left_on='merge_time', right_on='time', how='inner')
    
    # Add Features
    final_df['month'] = final_df['merge_time'].dt.month
    final_df['hour'] = final_df['merge_time'].dt.hour
    
    final_df.to_csv("clean_training_data.csv", index=False)
    print(f"🎉 SUCCESS: Saved {len(final_df)} samples with Light & Wind vectors.")