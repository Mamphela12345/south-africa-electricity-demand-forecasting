 South African Electricity Demand Forecasting

  1.Problem Statement

Electricity demand forecasting is important for electricity generation planning, grid management and energy resource allocation in South Africa.

Historical demand patterns can help identify trends, seasonal variations and changes in electricity demand. Accurate forecasting can support better planning and decision-making within the energy sector.

 2.Project Objective

The objective is to develop a machine-learning model that forecasts daily electricity demand in South Africa using historical Eskom RSA Contracted Demand data and relevant external factors.

The project aims to:

-Analyse historical electricity demand patterns.
-Identify seasonal, monthly and weekly variations.
-Incorporate relevant external information, including weather data.
-Develop and evaluate machine-learning forecasting models.
-Generate seven-day-ahead electricity demand forecasts.
-Use AI to explain forecast results, summarise demand patterns and identify potential anomalies.

Machine-learning approach: Supervised learning – regression.

Target variable: Daily energy measurement, expressed in megawatt-hours (MWh).

 3.Technologies and Tools

Cloud and Data Infrastructure
-Google Cloud Platform (GCP): Planned cloud infrastructure for the project.
-Google BigQuery: Used for data storage, SQL analysis and data quality validation.
-Google Cloud Storage: Planned for cloud-based data storage.

  Programming and Data Processing
-Python: Main programming language.
-NumPy: Numerical computation.
-Pandas: Data manipulation and analysis.
-SQL: Data querying, filtering, aggregation and transformation.

  APIs and External Data
-APIs: Used to retrieve external information, including historical weather data.
-Open-Meteo API: Historical weather data for selected South African cities.

  Machine Learning and AI
-Scikit-learn: Machine-learning model development and evaluation.
-XGBoost: Gradient-boosting model development.
-AI: Planned natural-language explanations of forecasts, demand patterns and potential anomalies.

  Version Control
-Git: Version control.
-GitHub: Source-code hosting and project documentation.

   4.Dataset

The primary dataset is Eskom RSA Contracted Demand data.

Dataset details:
-Source: Eskom.
-Original frequency: Hourly.
-Coverage: 1 April 2022 to 10 November 2026 in the original source.
-Processed frequency: Daily.
-Processed coverage: 1 April 2022 to 9 November 2026.
-Target measurement: Daily energy in MWh.

The original hourly dataset contained 24 missing demand values on 10 November 2026. Therefore, that incomplete day was excluded from the processed daily dataset.

Additional information includes historical weather data retrieved through the Open-Meteo API.

  5.Project Architecture

The project follows a data-processing and forecasting workflow:

1.Data collection: Obtain historical electricity demand data and external weather information.
2.Data processing: Use Python and Pandas to clean, validate and aggregate hourly demand into daily energy measurements.
3.Data storage and analysis: Store the processed dataset in BigQuery and perform SQL-based analysis.
4.Feature engineering: Create calendar, lag, rolling-average, holiday and weather features.
5.Machine learning: Train and evaluate forecasting models.
6.Forecasting: Generate seven-day-ahead demand predictions.
7.AI integration: Generate natural-language explanations of forecasts and identify potential anomalies.
8.Visualisation: Develop a dashboard to communicate historical patterns and forecast results.

Google Cloud Storage and the wider GCP pipeline remain planned infrastructure components.

 6.Machine Learning Approach

The project uses supervised learning with regression models to predict daily energy measurements.

The machine-learning workflow includes:

-Preparing historical data for model training.
-Creating time-based features and lag variables.
-Splitting data into training and testing periods.
-Training and evaluating regression models.
-Comparing model performance using evaluation metrics.
-Generating future demand forecasts.

Evaluation metrics include:

-Mean Absolute Error (MAE).
-Root Mean Squared Error (RMSE).
-Coefficient of Determination (R²).

The forecasting process uses historical observations to predict future daily energy measurements.

 7.Model Predictions

The model is designed to forecast daily Eskom RSA Contracted Demand energy measurements for the next seven days.

Forecast output:
-Prediction frequency: Daily.
-Forecast horizon: Seven days.
-Measurement unit: MWh.

The project also explores how weather conditions, calendar patterns and historical demand measurements relate to forecast performance.

The model does not currently predict individual household electricity consumption in kWh.

 8.Expected Results

The expected project outcomes include:

-A processed and validated electricity-demand dataset.
-SQL-based analysis of historical demand patterns.
-A machine-learning model capable of generating daily demand forecasts.
-Evaluation of forecasting accuracy using appropriate metrics.
-Seven-day-ahead demand predictions.
-A dashboard presenting historical data and forecast results.
-AI-generated explanations of forecast patterns and potential anomalies.
-A documented and version-controlled GitHub portfolio project.

The overall goal is to demonstrate practical skills in data engineering, cloud data technologies, SQL, Python, machine learning and AI.