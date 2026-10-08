import streamlit as st
import pandas as pd
from src.pipeline import run_pipeline

st.set_page_config(
    page_title="Real-Time Air Quality Prediction",
    page_icon="🌫️",
    layout="wide"
)

st.title("🌫️ Real-Time Air Quality Prediction & Monitoring")
st.caption(
    "Automatic pipeline: API → Cleaning → Feature Engineering → "
    "Model Comparison → Prediction"
)


# -------------------------------------------------
# AUTOMATIC PIPELINE
# -------------------------------------------------

if (
    "result" not in st.session_state
    or "best_model" not in st.session_state.result
):

    with st.spinner(
        "Fetching latest data and running the ML pipeline..."
    ):

        try:
            st.session_state.result = run_pipeline()

        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()


result = st.session_state.result

metrics = result["metrics"]
latest = result["latest"]
prediction = result["prediction"]
history = result["history"]

best_model = result["best_model"]

linear_metrics = result["linear_metrics"]
rf_metrics = result["rf_metrics"]


# -------------------------------------------------
# MAIN AQI METRICS
# -------------------------------------------------

st.subheader("🌫️ Current Air Quality")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Current US AQI",
    f"{latest['us_aqi']:.0f}"
)

c2.metric(
    "PM2.5",
    f"{latest['pm2_5']:.1f} µg/m³"
)

c3.metric(
    "PM10",
    f"{latest['pm10']:.1f} µg/m³"
)

c4.metric(
    "Predicted Next-Hour AQI",
    f"{prediction:.0f}"
)


st.divider()


# -------------------------------------------------
# BEST MODEL
# -------------------------------------------------

st.subheader("🏆 Selected Model")

st.success(
    f"Best Model: **{best_model}**"
)


# -------------------------------------------------
# MODEL COMPARISON
# -------------------------------------------------

st.subheader("📊 Model Comparison")

comparison_df = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest"
    ],

    "MAE": [
        linear_metrics["MAE"],
        rf_metrics["MAE"]
    ],

    "RMSE": [
        linear_metrics["RMSE"],
        rf_metrics["RMSE"]
    ],

    "R²": [
        linear_metrics["R2"],
        rf_metrics["R2"]
    ]

})

st.dataframe(
    comparison_df.style.format({
        "MAE": "{:.2f}",
        "RMSE": "{:.2f}",
        "R²": "{:.3f}"
    }),
    hide_index=True,
    use_container_width=True
)


st.divider()


# -------------------------------------------------
# BEST MODEL PERFORMANCE
# -------------------------------------------------

st.subheader("📈 Selected Model Performance")

m1, m2, m3 = st.columns(3)

m1.metric(
    "MAE",
    f"{metrics['MAE']:.2f}"
)

m2.metric(
    "RMSE",
    f"{metrics['RMSE']:.2f}"
)

m3.metric(
    "R²",
    f"{metrics['R2']:.3f}"
)


st.divider()


# -------------------------------------------------
# LATEST AIR QUALITY DATA
# -------------------------------------------------

left, right = st.columns(2)


with left:

    st.subheader("🌫️ Latest Air Quality Data")

    latest_display = pd.DataFrame({

        "Parameter": [
            "Time",
            "PM2.5",
            "PM10",
            "Carbon Monoxide",
            "Nitrogen Dioxide",
            "Sulphur Dioxide",
            "Ozone",
            "US AQI"
        ],

        "Value": [
            str(latest["time"]),
            f"{float(latest['pm2_5']):.2f} µg/m³",
            f"{float(latest['pm10']):.2f} µg/m³",
            f"{float(latest['carbon_monoxide']):.2f}",
            f"{float(latest['nitrogen_dioxide']):.2f}",
            f"{float(latest['sulphur_dioxide']):.2f}",
            f"{float(latest['ozone']):.2f}",
            f"{float(latest['us_aqi']):.0f}"
        ]

    })

    st.dataframe(
        latest_display,
        hide_index=True,
        use_container_width=True
    )


# -------------------------------------------------
# AQI TREND
# -------------------------------------------------

with right:

    st.subheader("📈 Recent AQI Trend")

    chart_df = history[
        ["time", "us_aqi"]
    ].copy()

    chart_df["time"] = pd.to_datetime(
        chart_df["time"]
    )

    chart_df = chart_df.set_index("time")

    st.line_chart(chart_df)


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()

st.caption(
    "Data source: Open-Meteo Air Quality API. "
    "Fresh data is fetched automatically when the application starts. "
    "The model is selected using time-based evaluation."
)