import requests
import pandas as pd
from src.config import (
    LATITUDE, LONGITUDE, TIMEZONE, RAW_FILE
)

API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

HOURLY_VARIABLES = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]

def fetch_air_quality():
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": TIMEZONE,
        "past_days": 30,
        "forecast_days": 1,
    }

    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()

    payload = response.json()
    if "hourly" not in payload:
        raise ValueError("API response does not contain hourly data.")

    df = pd.DataFrame(payload["hourly"])

    required = ["time"] + HOURLY_VARIABLES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing API columns: {missing}")

    df.to_csv(RAW_FILE, index=False)
    return df
