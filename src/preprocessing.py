import numpy as np
import pandas as pd
from src.config import PROCESSED_FILE

FEATURES = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
    "hour",
    "day_of_week",
    "pm2_5_lag1",
    "pm10_lag1",
    "aqi_lag1",
]

TARGET = "target_aqi"

def clean_and_engineer(df):
    data = df.copy()

    data["time"] = pd.to_datetime(data["time"], errors="coerce")

    numeric_cols = [
        "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
        "sulphur_dioxide", "ozone", "us_aqi"
    ]

    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["time"])
    data = data.sort_values("time").drop_duplicates("time")

    # Remove impossible negative measurements.
    for col in numeric_cols:
        data.loc[data[col] < 0, col] = np.nan

    data[numeric_cols] = data[numeric_cols].interpolate(limit_direction="both")

    data["hour"] = data["time"].dt.hour
    data["day_of_week"] = data["time"].dt.dayofweek

    # Previous-hour information becomes the model input.
    data["pm2_5_lag1"] = data["pm2_5"].shift(1)
    data["pm10_lag1"] = data["pm10"].shift(1)
    data["aqi_lag1"] = data["us_aqi"].shift(1)

    # Target is the next hour's AQI.
    data[TARGET] = data["us_aqi"].shift(-1)

    data = data.dropna(subset=FEATURES + [TARGET])
    data.to_csv(PROCESSED_FILE, index=False)

    return data
