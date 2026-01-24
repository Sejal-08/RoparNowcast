import numpy as np

def calculate_dew_point(temp, humidity):
    """
    Calculates Dew Point using the Magnus Formula.
    Reference: Alduchov and Eskridge (1996).
    """
    b = 17.625
    c = 243.04
    safe_humidity = np.maximum(humidity, 0.001)
    
    gamma = np.log(safe_humidity / 100.0) + (b * temp) / (c + temp)
    dew_point = (c * gamma) / (b - gamma)
    return dew_point

def calculate_dew_point_depression(temp, dew_point):
    return temp - dew_point

def classify_weather_state(row):
    """
    Determines weather condition with Day/Night distinction.
    """
    # Extract variables safely
    rain = row.get('rain', 0)
    dpd = row.get('dew_point_depression', 10)
    wind = row.get('wind_speed', 0)
    hum = row.get('humidity', 0)
    lux = row.get('light', 0)
    pressure_change = row.get('pressure_change_3h', 0)

    # ---------------------------------------------------------
    # 1️⃣ THE SUNLIGHT OVERRIDE (Day Only)
    # ---------------------------------------------------------
    # If Sun is blazing, it is definitely Clear Day.
    if lux > 25000:
        return "CLEAR DAY ☀️"

    # ---------------------------------------------------------
    # 2️⃣ SMART RAIN CHECK
    # ---------------------------------------------------------
    if rain >= 1.0:
        return "RAIN 🌧️"
    elif rain >= 0.1:
        return "DRIZZLE 🌦️"
    elif rain > 0.2 and pressure_change < -0.5:
        # Fallback for pressure drop indication even if rain is light
        return "DRIZZLE 🌦️"

    # ---------------------------------------------------------
    # 3️⃣ FOG PHYSICS
    # ---------------------------------------------------------
    # Fog requires low light. If it's night (lux < 50), Fog is possible.
    if hum >= 97 and dpd < 2.0 and wind < 2.5 and lux < 5000:
        return "FOG 🌫️"

    # ---------------------------------------------------------
    # 4️⃣ CLOUDY / MIST
    # ---------------------------------------------------------
    if hum > 80.0:
        return "CLOUDY ☁️"

    # ---------------------------------------------------------
    # 5️⃣ DEFAULT: CLEAR (Day vs Night Split)
    # ---------------------------------------------------------
    # If we reached here, the sky is clear.
    # Check Lux to decide if it's Day or Night.
    
    if lux < 50:
        return "CLEAR NIGHT 🌙"  # Dark = Night
    else:
        return "CLEAR DAY ☀️"    # Light = Day