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
    # Extract variables safely with defaults
    rain = row.get('rain', 0)
    dpd = row.get('dew_point_depression', 10)
    wind = row.get('wind_speed', 0)
    hum = row.get('humidity', 0)
    lux = row.get('light', 0)
    pressure_change = row.get('pressure_change_3h', 0)

    # ---------------------------------------------------------
    # 1️⃣ SUPER-SATURATION CHECK (The "Whiteout" Fix) 🌫️
    # ---------------------------------------------------------
    # If the air is fully saturated, it is PHYSICALLY IMPOSSIBLE to be Clear.
    # We check this FIRST to override the Sunlight sensor (White Fog reflects light).
    
    is_saturated = (hum >= 98) or (dpd < 0.5)
    
    if is_saturated:
        # If it's raining significantly, it's Rain.
        if rain > 0.5:
            return "RAIN 🌧️"
        # Otherwise, it MUST be Fog (even if it's bright).
        else:
            return "FOG 🌫️"

    # ---------------------------------------------------------
    # 2️⃣ SMART RAIN CHECK 🌧️
    # ---------------------------------------------------------
    if rain >= 1.0:
        return "RAIN 🌧️"
    elif rain >= 0.1:
        return "DRIZZLE 🌦️"
    elif rain > 0.2 and pressure_change < -0.5:
        # Fallback for pressure drop indication even if rain is light
        return "DRIZZLE 🌦️"

    # ---------------------------------------------------------
    # 3️⃣ SUNLIGHT OVERRIDE (Day Only) ☀️
    # ---------------------------------------------------------
    # If Sun is blazing AND we passed the saturation check above, 
    # then it is definitely a Clear Day.
    if lux > 50000:
        return "CLEAR DAY ☀️"

    # ---------------------------------------------------------
    # 4️⃣ STANDARD FOG PHYSICS (For lower light/morning fog) 🌫️
    # ---------------------------------------------------------
    # Standard fog check for non-saturated but misty conditions.
    if hum >= 97 and dpd < 2.0 and wind < 2.5:
        return "FOG 🌫️"

    # ---------------------------------------------------------
    # 5️⃣ CLOUDY / MIST ☁️
    # ---------------------------------------------------------
    # If humidity is high but not saturated.
    if hum > 75.0:
        return "CLOUDY ☁️"
    
    # Pressure Instability Check: Falling pressure usually means clouds.
    if pressure_change < -1.5 and hum > 50.0:
        return "CLOUDY ☁️"

    # ---------------------------------------------------------
    # 6️⃣ DEFAULT: CLEAR (Day vs Night Split) 🌙/☀️
    # ---------------------------------------------------------
    # If we reached here, the sky is clear.
    # Check Lux to decide if it's Day or Night.
    
    if lux < 50:
        return "CLEAR NIGHT 🌙"  # Dark = Night
    else:
        return "CLEAR DAY ☀️"    # Light = Day