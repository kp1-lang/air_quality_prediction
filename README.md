# Real-Time Air Quality Prediction & Monitoring

An end-to-end machine learning project that automatically:

1. Fetches fresh air-quality data from the Open-Meteo API.
2. Validates and cleans the data.
3. Performs feature engineering.
4. Trains a Random Forest regression model.
5. Evaluates the model using MAE, RMSE and R².
6. Predicts the next-hour US AQI.
7. Displays the result in a Streamlit dashboard.

## Project Architecture

API → Data Extraction → Validation/Cleaning → Feature Engineering → Model Training → Evaluation → Prediction → Streamlit Dashboard

## Run locally

### 1. Create/activate virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## Data source

Open-Meteo Air Quality API:
https://air-quality-api.open-meteo.com/

The application uses Mumbai coordinates and fetches 30 days of hourly historical data plus a short forecast window.

## GitHub

```bash
git init
git add .
git commit -m "Initial end-to-end air quality ML project"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

## Important note

The model is retrained from freshly fetched data when the pipeline is run. This is intentional for demonstrating a fully automated end-to-end ML workflow.
