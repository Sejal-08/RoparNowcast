import numpy as np

def calculate_dew_point(temp, humidity):
    """
    Calculates Dew Point using the Magnus Formula.
    Reference: Alduchov and Eskridge (1996).
    Range: -40°C to 50°C. Accuracy: ±0.1°C.
    
    Args:
        temp (float or np.array): Temperature in Celsius.
        humidity (float or np.array): Relative Humidity in %.
        
    Returns:
        float or np.array: Dew Point in Celsius.
    """
    # Constants
    b = 17.625
    c = 243.04
    
    # Safety: Ensure humidity is never 0 or negative to avoid log errors
    safe_humidity = np.maximum(humidity, 0.001)
    
    gamma = np.log(safe_humidity / 100.0) + (b * temp) / (c + temp)
    dew_point = (c * gamma) / (b - gamma)
    
    return dew_point

def calculate_dew_point_depression(temp, dew_point):
    """Calculates the difference between Temp and Dew Point."""
    return temp - dew_point

def classify_weather_state(row):
    """
    Determines the weather condition label based on physical parameters.
    
    Args:
        row (dict): Contains keys 'rain', 'dew_point_depression', 'wind_speed', 'humidity'.
        
    Returns:
        str: 'RAIN', 'FOG', 'CLOUDY', 'CLEAR', 'WINDY'
    """
    rain = row.get('rain', 0)
    dpd = row.get('dew_point_depression', 10)
    wind = row.get('wind_speed', 0)
    hum = row.get('humidity', 0)

    if rain > 0.2:
        return "RAIN 🌧️"
    elif dpd < 2.0 and wind < 5.0:
        return "FOG 🌫️"
    elif hum > 80.0:
        return "CLOUDY ☁️"
    else:
        return "CLEAR ☀️"