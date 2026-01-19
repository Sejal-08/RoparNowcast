import os

# --- GEOGRAPHY ---
LAT = 30.96
LON = 76.47

# --- API & AWS ---
REGION = "us-east-1"
DYNAMO_TABLE = "WS_Campus_Data"
DEVICE_ID = "1"
SOURCE_API_URL = "https://gtk47vexob.execute-api.us-east-1.amazonaws.com/campusdata"

# --- AUTOMATIC PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

DATA_FILE = os.path.join(DATA_DIR, "clean_training_data.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "nowcast_latest.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- NOWCAST CORE SETTINGS ---

# Observation frequency (minutes)
TIME_STEP_MINUTES = 5

# Total forecast length
NOWCAST_HOURS = 3
N_HORIZONS = int((NOWCAST_HOURS * 60) / TIME_STEP_MINUTES)  # 36

# Lead times (1 → +5 min, 36 → +180 min)
HORIZONS = list(range(1, N_HORIZONS + 1))

# History window used for features
NOWCAST_WINDOW_HOURS = 6
WINDOW_STEPS = int((NOWCAST_WINDOW_HOURS * 60) / TIME_STEP_MINUTES)  # 72

# Lag features (short-term memory)
LAGS = 12  # last 1 hour (12 × 5 min)

# Physics strength scaling
MAX_PHYSICS_WEIGHT = 1.0
