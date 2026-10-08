import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import MODEL_FILE
from src.preprocessing import FEATURES, TARGET


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": float(r2_score(y_test, predictions))
    }

    return metrics


def train_model(data):

    # Features and target
    X = data[FEATURES]
    y = data[TARGET]

    # ------------------------------------------------
    # Time-based split
    # ------------------------------------------------
    # No random shuffle because this is time-series data.
    split = int(len(data) * 0.80)

    if split < 20 or len(data) - split < 5:
        raise ValueError(
            "Not enough data for training and testing."
        )

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # ------------------------------------------------
    # Model 1: Baseline - Linear Regression
    # ------------------------------------------------
    linear_model = LinearRegression()

    linear_model.fit(
        X_train,
        y_train
    )

    linear_metrics = evaluate_model(
        linear_model,
        X_test,
        y_test
    )

    # ------------------------------------------------
    # Model 2: Random Forest
    # ------------------------------------------------
    random_forest = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    random_forest.fit(
        X_train,
        y_train
    )

    rf_metrics = evaluate_model(
        random_forest,
        X_test,
        y_test
    )

    # ------------------------------------------------
    # Select best model using RMSE
    # ------------------------------------------------

    if rf_metrics["RMSE"] < linear_metrics["RMSE"]:

        best_model = random_forest
        best_metrics = rf_metrics
        best_model_name = "Random Forest"

    else:

        best_model = linear_model
        best_metrics = linear_metrics
        best_model_name = "Linear Regression"

    # Save the selected model
    joblib.dump(
        best_model,
        MODEL_FILE
    )

    return (
        best_model,
        best_metrics,
        best_model_name,
        linear_metrics,
        rf_metrics
    )


def predict_next_hour(model, data):

    latest = data.iloc[-1]

    X_latest = pd.DataFrame(
        [latest[FEATURES].to_dict()]
    )

    prediction = float(
        model.predict(X_latest)[0]
    )

    # AQI cannot be negative
    return max(0.0, prediction)