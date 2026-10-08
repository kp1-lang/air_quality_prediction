from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"

RAW_FILE = RAW_DIR / "air_quality_raw.csv"
PROCESSED_FILE = PROCESSED_DIR / "air_quality_processed.csv"
MODEL_FILE = MODEL_DIR / "aqi_model.joblib"

LATITUDE = 19.0760
LONGITUDE = 72.8777
TIMEZONE = "Asia/Kolkata"

for directory in [RAW_DIR, PROCESSED_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
