from src.data_fetch import fetch_air_quality
from src.preprocessing import clean_and_engineer
from src.model import train_model, predict_next_hour


def run_pipeline():

    # 1. Fetch fresh data from API
    raw = fetch_air_quality()

    # 2. Clean data and create features
    processed = clean_and_engineer(raw)

    # 3. Train and compare models
    (
        model,
        best_metrics,
        best_model_name,
        linear_metrics,
        rf_metrics
    ) = train_model(processed)

    # 4. Predict next-hour AQI
    prediction = predict_next_hour(
        model,
        processed
    )

    # 5. Get latest air-quality observation
    latest = processed.iloc[-1][
        [
            "time",
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "us_aqi"
        ]
    ].to_dict()

    return {

        "metrics": best_metrics,

        "prediction": prediction,

        "latest": latest,

        "history": raw.tail(48),

        "best_model": best_model_name,

        "linear_metrics": linear_metrics,

        "rf_metrics": rf_metrics
    }