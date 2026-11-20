#importing panda
import pandas as pd
#importing matplotlib for plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Import machine learning libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
#importing the holidays library for the South African public holidays
import holidays
#importing GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
#importing xgboost
from xgboost import XGBRegressor
#creating a csv file from the raw data
df = pd.read_csv("data/raw/ESK20149.csv")

#creating a copy
df_clean = df.copy()

# Remove rows where demand is unavailable
df_clean = df_clean.dropna(subset=["RSA Contracted Demand"])
print(df_clean.shape)

#Convert the timestamp column
df_clean["Date Time Hour Beginning"] = pd.to_datetime(df_clean["Date Time Hour Beginning"], format="%Y-%m-%d %I:%M:%S %p")

#creating daily electricity demand data
df_clean["Date"] = (df_clean["Date Time Hour Beginning"].dt.date)

daily_demand = df_clean.groupby("Date")["RSA Contracted Demand"].sum().reset_index()
#renaming our column from RSA Contracted DEmand to Daily Energy MWh
daily_demand.rename(columns={"RSA Contracted Demand": "Daily_Energy_MWh"}, inplace = True)

print(daily_demand.head())
print(daily_demand.shape)
#count valid hourly readings per day
valid_hourly_readings = df_clean.groupby("Date").size()
print(valid_hourly_readings.head())
print("Incomplete days: ", (valid_hourly_readings != 24).sum())
daily_demand.to_csv("data/processed/daily_demand.csv", index=False)
#plotting the daily demand data
plt.figure(figsize=(14,6))
plt.plot(daily_demand["Date"],  daily_demand["Daily_Energy_MWh"])
plt.xlabel("Date")
plt.ylabel("Daily Energy (MWh)")
plt.title("South African Daily Electricity Demand")
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
plt.savefig("xgboost_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

#Convert date column to datetime format
daily_demand["Date"] = pd.to_datetime(daily_demand["Date"])
#Extract calender feautures
daily_demand["Day_of_the_week"] = daily_demand["Date"].dt.day_of_week
daily_demand["Month"] = daily_demand["Date"].dt.month
daily_demand["Year"] = daily_demand["Date"].dt.year
daily_demand["Is_weekend"] = (daily_demand["Day_of_the_week"] >= 5).astype(int)
print(daily_demand.head())

# Validate hourly records per day
hours_per_day = df_clean.groupby("Date").size()
print("Hourly records per day:")
print(hours_per_day.value_counts())
print("\nIncomplete days:")
print(hours_per_day[hours_per_day != 24])
daily_demand["Lag_1_Day"] = (
    daily_demand["Daily_Energy_MWh"].shift(1)
)

daily_demand["Lag_7_Days"] = (
    daily_demand["Daily_Energy_MWh"].shift(7)
)

daily_demand["Rolling_7_Day_Average"] = (
    daily_demand["Daily_Energy_MWh"]
    .shift(1)
    .rolling(window=7)
    .mean()
)

print("\nHistorical demand features:")
print(daily_demand.head(10))
# Remove rows with missing historical features

model_data = daily_demand.dropna().copy()

print("Original dataset:", daily_demand.shape)
print("Machine learning dataset:", model_data.shape)

print("\nMissing values:")
print(model_data.isnull().sum())

print("\nPrepared dataset:")
print(model_data.head())

# Sort data chronologically
model_data = model_data.sort_values("Date").reset_index(drop=True)

# Determine the training and testing split
split_index = int(len(model_data) * 0.8)

# Split the dataset
train_data = model_data.iloc[:split_index]
test_data = model_data.iloc[split_index:]

# Select model features
features = [
    "Day_of_the_week",
    "Month",
    "Year",
    "Is_weekend",
    "Lag_1_Day",
    "Lag_7_Days",
    "Rolling_7_Day_Average"
]

# Define input variables (X) and target variable (y)
X_train = train_data[features]
y_train = train_data["Daily_Energy_MWh"]

X_test = test_data[features]
y_test = test_data["Daily_Energy_MWh"]

# Display results
print("\nTraining and Testing Data")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("Training period:", train_data["Date"].min(), "to", train_data["Date"].max())
print("Testing period:", test_data["Date"].min(), "to", test_data["Date"].max())

# Initialise the Random Forest model

rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=2,
    n_jobs=-1
)

# Train the model

rf_model.fit(X_train, y_train)

# Generate predictions

y_pred = rf_model.predict(X_test)

# Evaluate model performance

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Display evaluation results

print("\nRandom Forest Model Performance")
print("MAE:", round(mae, 2), "MWh")
print("RMSE:", round(rmse, 2), "MWh")
print("R-squared:", round(r2, 4))
# Baseline: Same day of the previous week

baseline_predictions = test_data["Lag_7_Days"]

# Evaluate baseline performance

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)

# Compare the two models

print("\nModel Comparison")

print("Random Forest MAE:", round(mae, 2))
print("Baseline MAE:", round(baseline_mae, 2))

print("Random Forest RMSE:", round(rmse, 2))
print("Baseline RMSE:", round(baseline_rmse, 2))

# Feature importance analysis

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

# Visualise feature importance

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()
# Compare actual and predicted demand

results = pd.DataFrame({
    "Date": test_data["Date"].values,
    "Actual_Demand_MWh": y_test.to_numpy(),
    "Predicted_Demand_MWh": y_pred
})

# Calculate prediction errors

results["Error_MWh"] = (
    results["Actual_Demand_MWh"] -
    results["Predicted_Demand_MWh"]
)

print("\nActual vs Predicted Demand:")
print(results.head(10))

# Plot actual vs predicted demand

plt.figure(figsize=(15, 6))

plt.plot(
    results["Date"],
    results["Actual_Demand_MWh"],
    label="Actual Demand"
)

plt.plot(
    results["Date"],
    results["Predicted_Demand_MWh"],
    label="Predicted Demand"
)

plt.title("Actual vs Predicted South African Electricity Demand")
plt.xlabel("Date")
plt.ylabel("Daily Energy (MWh)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
import requests

# Selected South African locations
locations = {
    "Pretoria": {"latitude": -25.7479, "longitude": 28.2293},
    "Johannesburg": {"latitude": -26.2041, "longitude": 28.0473},
    "Durban": {"latitude": -29.8587, "longitude": 31.0218},
    "Cape Town": {"latitude": -33.9249, "longitude": 18.4241},
    "Bloemfontein": {"latitude": -29.0852, "longitude": 26.1596}
}

# Get the date range from our electricity dataset
start_date = pd.to_datetime(
    daily_demand["Date"]
).min().strftime("%Y-%m-%d")

end_date = pd.to_datetime(
    daily_demand["Date"]
).max().strftime("%Y-%m-%d")

# Open-Meteo historical weather API
url = "https://archive-api.open-meteo.com/v1/archive"

weather_frames = []

# Retrieve weather data for each location
for city, coordinates in locations.items():

    parameters = {
        "latitude": coordinates["latitude"],
        "longitude": coordinates["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum"
        ],
        "timezone": "Africa/Johannesburg"
    }

    response = requests.get(
        url,
        params=parameters,
        timeout=60
    )

    response.raise_for_status()

    weather_data = response.json()

    if "daily" not in weather_data:
        raise ValueError(
            f"No daily weather data returned for {city}"
        )

    city_weather = pd.DataFrame(weather_data["daily"])

    city_weather["City"] = city

    weather_frames.append(city_weather)

    print(f"Weather data retrieved for {city}")

# Combine weather data from all locations
weather_by_location = pd.concat(
    weather_frames,
    ignore_index=True
)

# Save individual location data
weather_by_location.to_csv(
    "data/raw/weather_by_location.csv",
    index=False
)

# Calculate average weather conditions across locations
weather_national = (
    weather_by_location
    .groupby("time")
    .agg({
        "temperature_2m_mean": "mean",
        "temperature_2m_max": "mean",
        "temperature_2m_min": "mean",
        "precipitation_sum": "mean"
    })
    .reset_index()
)

# Rename columns
weather_national = weather_national.rename(
    columns={
        "time": "Date",
        "temperature_2m_mean": "Temperature_Mean_C",
        "temperature_2m_max": "Temperature_Max_C",
        "temperature_2m_min": "Temperature_Min_C",
        "precipitation_sum": "Precipitation_Mean_mm"
    }
)

# Convert dates
weather_national["Date"] = pd.to_datetime(
    weather_national["Date"]
)

# Save aggregated weather data
weather_national.to_csv(
    "data/processed/weather_national_proxy.csv",
    index=False
)

# Display results
print("\nNational Weather Proxy:")
print(weather_national.head())

print("\nWeather dataset shape:")
print(weather_national.shape)

print("\nMissing values:")
print(weather_national.isnull().sum())
print("\nWeather missing values:")
print(weather_national.isnull().sum())

print("\nDuplicate weather dates:")
print(weather_national["Date"].duplicated().sum())

print("\nWeather date range:")
print(weather_national["Date"].min())
print(weather_national["Date"].max())
# Merge electricity demand and weather datasets

weather_demand = pd.merge(
    daily_demand,
    weather_national,
    on="Date",
    how="inner",
    validate="one_to_one"
)

# Display combined dataset

print("\nCombined Electricity and Weather Dataset:")
print(weather_demand.head())

print("\nCombined dataset shape:")
print(weather_demand.shape)

print("\nMissing values:")
print(weather_demand.isnull().sum())
# Prepare weather-enhanced machine learning dataset

weather_model_data = weather_demand.dropna().copy()

weather_model_data = (
    weather_model_data
    .sort_values("Date")
    .reset_index(drop=True)
)

print("\nWeather Model Dataset:")
print("Records:", len(weather_model_data))

# Define features including weather information

weather_features = [
    "Day_of_the_week",
    "Month",
    "Year",
    "Is_weekend",
    "Lag_1_Day",
    "Lag_7_Days",
    "Rolling_7_Day_Average",
    "Temperature_Mean_C",
    "Temperature_Max_C",
    "Temperature_Min_C",
    "Precipitation_Mean_mm"
]

# Chronological training and testing split

weather_split_index = int(len(weather_model_data) * 0.8)

weather_train = weather_model_data.iloc[:weather_split_index]
weather_test = weather_model_data.iloc[weather_split_index:]

# Define X and y

X_train_weather = weather_train[weather_features]
y_train_weather = weather_train["Daily_Energy_MWh"]

X_test_weather = weather_test[weather_features]
y_test_weather = weather_test["Daily_Energy_MWh"]

print("\nWeather Model Training and Testing")

print("Training records:", len(X_train_weather))
print("Testing records:", len(X_test_weather))

print(
    "Training period:",
    weather_train["Date"].min(),
    "to",
    weather_train["Date"].max()
)

print(
    "Testing period:",
    weather_test["Date"].min(),
    "to",
    weather_test["Date"].max()
)
# Train Weather-Enhanced Random Forest Model

rf_weather_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=2,
    n_jobs=-1
)

# Train the model
rf_weather_model.fit(X_train_weather, y_train_weather)

# Generate predictions
y_pred_weather = rf_weather_model.predict(X_test_weather)

# Evaluate model performance
mae_weather = mean_absolute_error(
    y_test_weather, y_pred_weather
)

rmse_weather = np.sqrt(
    mean_squared_error(y_test_weather, y_pred_weather)
)

r2_weather = r2_score(
    y_test_weather, y_pred_weather
)

# Display results
print("\nWeather-Enhanced Random Forest Performance")

print("MAE:", round(mae_weather, 2), "MWh")
print("RMSE:", round(rmse_weather, 2), "MWh")
print("R-squared:", round(r2_weather, 4))
weather_importance = pd.DataFrame({
    "Feature": weather_features,
    "Importance": rf_weather_model.feature_importances_
})

weather_importance = weather_importance.sort_values(
    by="Importance",
    ascending=False
)

print(weather_importance)
daily_demand["Day_of_year"] = (
    daily_demand["Date"].dt.day_of_year
)

za_holidays = holidays.country_holidays("ZA")

daily_demand["Day_of_year"] = (
    daily_demand["Date"].dt.day_of_year
)

za_holidays = holidays.country_holidays(
    "ZA",
    years=range(2022, 2027)
)

# Public holiday
daily_demand["Is_public_holiday"] = (
    daily_demand["Date"].dt.date.map(za_holidays).notna()
).astype(int)

# Day before a public holiday
daily_demand["Day_Before_Holiday"] = (
    (daily_demand["Date"] + pd.Timedelta(days=1))
    .dt.date.map(za_holidays).notna()
).astype(int)

# Day after a public holiday
daily_demand["Day_After_Holiday"] = (
    (daily_demand["Date"] - pd.Timedelta(days=1))
    .dt.date.map(za_holidays).notna()
).astype(int)

# Christmas and New Year period
daily_demand["Holiday_Season"] = (
    (
        (daily_demand["Date"].dt.month == 12)
        & (daily_demand["Date"].dt.day >= 20)
    )
    |
    (
        (daily_demand["Date"].dt.month == 1)
        & (daily_demand["Date"].dt.day <= 5)
    )
).astype(int)

features = [
    "Day_of_the_week",
    "Month",
    "Year",
    "Is_weekend",
    "Day_of_year",
    "Is_public_holiday",
    "Day_Before_Holiday",
    "Day_After_Holiday",
    "Holiday_Season",
    "Lag_1_Day",
    "Lag_7_Days",
    "Rolling_7_Day_Average"
]

model_data = daily_demand.dropna().copy()
model_data = model_data.sort_values("Date").reset_index(drop=True)

split_index = int(len(model_data) * 0.8)

train_data = model_data.iloc[:split_index]
test_data = model_data.iloc[split_index:]

X_train = train_data[features]
y_train = train_data["Daily_Energy_MWh"]

X_test = test_data[features]
y_test = test_data["Daily_Energy_MWh"]

print("Training records:", len(train_data))
print("Testing records:", len(test_data))

rf_calendar_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=2,
    n_jobs=-1
)

rf_calendar_model.fit(X_train, y_train)

y_pred_calendar = rf_calendar_model.predict(X_test)

mae_calendar = mean_absolute_error(y_test, y_pred_calendar)
rmse_calendar = np.sqrt(mean_squared_error(y_test, y_pred_calendar))
r2_calendar = r2_score(y_test, y_pred_calendar)

print("\nCalendar-Enhanced Random Forest Performance")
print("MAE:", round(mae_calendar, 2), "MWh")
print("RMSE:", round(rmse_calendar, 2), "MWh")
print("R-squared:", round(r2_calendar, 4))

daily_demand["Rolling_14_Day_Average"] = (
    daily_demand["Daily_Energy_MWh"]
    .shift(1)
    .rolling(window=14)
    .mean()
)

daily_demand["Rolling_28_Day_Average"] = (
    daily_demand["Daily_Energy_MWh"]
    .shift(1)
    .rolling(window=28)
    .mean()
)

features = [
    "Day_of_the_week",
    "Month",
    "Year",
    "Is_weekend",
    "Lag_1_Day",
    "Lag_7_Days",
    "Rolling_7_Day_Average",
    "Rolling_14_Day_Average",
    "Rolling_28_Day_Average"
]
daily_demand["Rolling_14_Day_Average"] = (
    daily_demand["Daily_Energy_MWh"]
    .shift(1)
    .rolling(window=14)
    .mean()
)

daily_demand["Rolling_28_Day_Average"] = (
    daily_demand["Daily_Energy_MWh"]
    .shift(1)
    .rolling(window=28)
    .mean()
)
model_data = daily_demand.dropna().copy()

model_data = model_data.sort_values("Date").reset_index(drop=True)

print("Model data shape:", model_data.shape)
print("\nMissing values:")
print(model_data.isnull().sum())

model_data = daily_demand.dropna().copy()

model_data = model_data.sort_values("Date").reset_index(drop=True)

print("Model data shape:", model_data.shape)
print("\nMissing values:")
print(model_data.isnull().sum())
features = [
    "Day_of_the_week",
    "Month",
    "Year",
    "Is_weekend",
    "Lag_1_Day",
    "Lag_7_Days",
    "Rolling_7_Day_Average",
    "Rolling_14_Day_Average",
    "Rolling_28_Day_Average"
]
model_data = daily_demand.dropna().copy()
model_data = model_data.sort_values("Date").reset_index(drop=True)

split_index = int(len(model_data) * 0.8)

train_data = model_data.iloc[:split_index]
test_data = model_data.iloc[split_index:]

X_train = train_data[features]
y_train = train_data["Daily_Energy_MWh"]

X_test = test_data[features]
y_test = test_data["Daily_Energy_MWh"]

print("Training records:", len(train_data))
print("Testing records:", len(test_data))
print("Testing period:", test_data["Date"].min(), "to", test_data["Date"].max())
rf_rolling_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=2,
    n_jobs=-1
)

rf_rolling_model.fit(X_train, y_train)

y_pred_rolling = rf_rolling_model.predict(X_test)

mae_rolling = mean_absolute_error(y_test, y_pred_rolling)
rmse_rolling = np.sqrt(mean_squared_error(y_test, y_pred_rolling))
r2_rolling = r2_score(y_test, y_pred_rolling)

print("\nExtended Rolling Average Model Performance")
print("MAE:", round(mae_rolling, 2), "MWh")
print("RMSE:", round(rmse_rolling, 2), "MWh")
print("R-squared:", round(r2_rolling, 4))

comparison = pd.DataFrame({
    "Date": test_data["Date"].values,
    "Actual": y_test.to_numpy(),
    "Model_4": y_pred_rolling
})

comparison = comparison.merge(
    results[["Date", "Predicted_Demand_MWh"]],
    on="Date",
    how="inner"
)

mae_model_1_common = mean_absolute_error(
    comparison["Actual"],
    comparison["Predicted_Demand_MWh"]
)

mae_model_4_common = mean_absolute_error(
    comparison["Actual"],
    comparison["Model_4"]
)

print("Common test records:", len(comparison))
print("Model 1 MAE:", round(mae_model_1_common, 2))
print("Model 4 MAE:", round(mae_model_4_common, 2))

model_data = daily_demand.dropna(
    subset=[
        "Lag_1_Day",
        "Lag_7_Days",
        "Rolling_7_Day_Average"
    ]
).copy()

model_data = model_data.sort_values("Date").reset_index(drop=True)
features = [
    "Day_of_the_week",
    "Month",
    "Year",
    "Is_weekend",
    "Lag_1_Day",
    "Lag_7_Days",
    "Rolling_7_Day_Average"
]
# MODEL 5: GRADIENT BOOSTING

gb_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=2,
    random_state=42
)

# Train the model
gb_model.fit(X_train, y_train)

# Generate predictions
y_pred_gb = gb_model.predict(X_test)

# Evaluate the model
mae_gb = mean_absolute_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
r2_gb = r2_score(y_test, y_pred_gb)

print("\nGradient Boosting Model Performance")
print("MAE:", round(mae_gb, 2), "MWh")
print("RMSE:", round(rmse_gb, 2), "MWh")
print("R-squared:", round(r2_gb, 4))
# MODEL 6: XGBOOST

model_data = daily_demand.dropna(
    subset=[
        "Lag_1_Day",
        "Lag_7_Days",
        "Rolling_7_Day_Average"
    ]
).copy()

model_data = model_data.sort_values("Date").reset_index(drop=True)

features = [
    "Day_of_the_week",
    "Month",
    "Year",
    "Is_weekend",
    "Lag_1_Day",
    "Lag_7_Days",
    "Rolling_7_Day_Average"
]

split_index = int(len(model_data) * 0.8)

train_data = model_data.iloc[:split_index]
test_data = model_data.iloc[split_index:]

X_train = train_data[features]
y_train = train_data["Daily_Energy_MWh"]

X_test = test_data[features]
y_test = test_data["Daily_Energy_MWh"]
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

rmse_xgb = np.sqrt(
    mean_squared_error(y_test, y_pred_xgb)
)

r2_xgb = r2_score(y_test, y_pred_xgb)

print("\nXGBoost Model Performance")
print("MAE:", round(mae_xgb, 2), "MWh")
print("RMSE:", round(rmse_xgb, 2), "MWh")
print("R-squared:", round(r2_xgb, 4))

# XGBoost Feature Importance

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": xgb_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nXGBoost Feature Importance")
print(feature_importance)
# XGBoost Actual vs Predicted Graph

plt.figure(figsize=(14, 6))

plt.plot(test_data["Date"], y_test, label="Actual Demand")
plt.plot(test_data["Date"], y_pred_xgb, label="XGBoost Predictions")

plt.title("Actual vs Predicted Electricity Demand")
plt.xlabel("Date")
plt.ylabel("Daily Energy (MWh)")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "data/processed/xgboost_actual_vs_predicted.png",
    dpi=300
)

plt.close()

print("XGBoost graph saved successfully.")
#prediction comparison for XGBoost model
xgb_results = pd.DataFrame({
    "Date": test_data["Date"].values,
    "Actual_Demand_MWh": y_test.to_numpy(),
    "Predicted_Demand_MWh": y_pred_xgb
})

xgb_results["Error_MWh"] = (
    xgb_results["Actual_Demand_MWh"]
    - xgb_results["Predicted_Demand_MWh"]
)

print("\nXGBoost Prediction Comparison")
print(xgb_results.head(10))

#Calculate the largest prediction errors for XGBoost model
xgb_results = pd.DataFrame({
    "Date": test_data["Date"].values,
    "Actual": y_test.to_numpy(),
    "Predicted": y_pred_xgb
})

xgb_results["Absolute_Error"] = abs(
    xgb_results["Actual"] - xgb_results["Predicted"]
)

print("\nLargest 10 Prediction Errors")

print(
    xgb_results.sort_values(
        by="Absolute_Error",
        ascending=False
    ).head(10)
)
#Identifying whether the largest prediction errors

xgb_results = pd.DataFrame({
    "Date": test_data["Date"].values,
    "Actual_Demand_MWh": y_test.to_numpy(),
    "Predicted_Demand_MWh": y_pred_xgb
})

xgb_results["Error_MWh"] = (
    xgb_results["Actual_Demand_MWh"]
    - xgb_results["Predicted_Demand_MWh"]
)

xgb_results["Date"] = pd.to_datetime(xgb_results["Date"])
xgb_results["Holiday_Name"] = xgb_results["Date"].dt.date.apply(
    lambda date: za_holidays.get(date)
)

xgb_results["Is_Public_Holiday"] = (
    xgb_results["Holiday_Name"].notna()
).astype(int)

print("\nXGBoost Results on Public Holidays")

print(
    xgb_results.loc[
        xgb_results["Is_Public_Holiday"] == 1,
        [
            "Date",
            "Holiday_Name",
            "Actual_Demand_MWh",
            "Predicted_Demand_MWh",
            "Error_MWh"
        ]
    ].to_string(index=False)
)
print("\n--- HOLIDAY VS NON-HOLIDAY PERFORMANCE ---")

holiday_data = xgb_results[
    xgb_results["Is_Public_Holiday"] == 1
]

normal_data = xgb_results[
    xgb_results["Is_Public_Holiday"] == 0
]

print("\nPublic Holidays:")
print("Number of days:", len(holiday_data))
print("MAE:", holiday_data["Error_MWh"].abs().mean())
print("Average Error:", holiday_data["Error_MWh"].mean())

print("\nNormal Days:")
print("Number of days:", len(normal_data))
print("MAE:", normal_data["Error_MWh"].abs().mean())
print("Average Error:", normal_data["Error_MWh"].mean())

# ============================================================
# MODEL 7: HOLIDAY-ENHANCED XGBOOST
# ============================================================

# Prepare holiday calendar
za_holidays = holidays.country_holidays(
    "ZA",
    years=range(2022, 2027)
)

# Create holiday-related features
daily_demand["Is_public_holiday"] = (
    daily_demand["Date"].dt.date.map(za_holidays).notna()
).astype(int)

daily_demand["Day_Before_Holiday"] = (
    (daily_demand["Date"] + pd.Timedelta(days=1))
    .dt.date.map(za_holidays).notna()
).astype(int)

daily_demand["Day_After_Holiday"] = (
    (daily_demand["Date"] - pd.Timedelta(days=1))
    .dt.date.map(za_holidays).notna()
).astype(int)

daily_demand["Holiday_Season"] = (
    (
        (daily_demand["Date"].dt.month == 12)
        & (daily_demand["Date"].dt.day >= 20)
    )
    |
    (
        (daily_demand["Date"].dt.month == 1)
        & (daily_demand["Date"].dt.day <= 5)
    )
).astype(int)

# Select features
holiday_features = [
    "Day_of_the_week",
    "Month",
    "Year",
    "Is_weekend",
    "Day_of_year",
    "Is_public_holiday",
    "Day_Before_Holiday",
    "Day_After_Holiday",
    "Holiday_Season",
    "Lag_1_Day",
    "Lag_7_Days",
    "Rolling_7_Day_Average"
]

# Prepare model dataset
holiday_model_data = daily_demand.dropna(
    subset=[
        "Lag_1_Day",
        "Lag_7_Days",
        "Rolling_7_Day_Average"
    ]
).copy()

holiday_model_data = holiday_model_data.sort_values(
    "Date"
).reset_index(drop=True)

# Chronological 80/20 split
holiday_split_index = int(len(holiday_model_data) * 0.8)

holiday_train_data = holiday_model_data.iloc[
    :holiday_split_index
]

holiday_test_data = holiday_model_data.iloc[
    holiday_split_index:
]

X_train_holiday = holiday_train_data[holiday_features]
y_train_holiday = holiday_train_data["Daily_Energy_MWh"]

X_test_holiday = holiday_test_data[holiday_features]
y_test_holiday = holiday_test_data["Daily_Energy_MWh"]

# Train XGBoost
xgb_holiday_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

xgb_holiday_model.fit(
    X_train_holiday,
    y_train_holiday
)

# Generate predictions
y_pred_holiday = xgb_holiday_model.predict(
    X_test_holiday
)

# Evaluate model
mae_holiday = mean_absolute_error(
    y_test_holiday,
    y_pred_holiday
)

rmse_holiday = np.sqrt(
    mean_squared_error(y_test_holiday, y_pred_holiday)
)

r2_holiday = r2_score(
    y_test_holiday,
    y_pred_holiday
)

print("\n======================================")
print("MODEL 7: HOLIDAY-ENHANCED XGBOOST")
print("======================================")

print("Training records:", len(holiday_train_data))
print("Testing records:", len(holiday_test_data))

print(
    "Testing period:",
    holiday_test_data["Date"].min(),
    "to",
    holiday_test_data["Date"].max()
)

print("MAE:", round(mae_holiday, 2), "MWh")
print("RMSE:", round(rmse_holiday, 2), "MWh")
print("R-squared:", round(r2_holiday, 4))

# Save predictions for comparison
holiday_xgb_results = pd.DataFrame({
    "Date": holiday_test_data["Date"].values,
    "Actual_Demand_MWh": y_test_holiday.to_numpy(),
    "Predicted_Demand_MWh": y_pred_holiday
})

holiday_xgb_results["Error_MWh"] = (
    holiday_xgb_results["Actual_Demand_MWh"]
    - holiday_xgb_results["Predicted_Demand_MWh"]
)

# ============================================================
# COMPARE MODEL 6 AND MODEL 7
# ============================================================

# Match predictions by date
model_comparison = pd.merge(
    xgb_results[["Date", "Actual_Demand_MWh", "Predicted_Demand_MWh"]],
    holiday_xgb_results[["Date", "Predicted_Demand_MWh"]],
    on="Date",
    suffixes=("_Original", "_Holiday"),
    validate="one_to_one"
)

# Identify public holidays
model_comparison["Is_Public_Holiday"] = (
    model_comparison["Date"].dt.date.map(za_holidays).notna()
)

# Calculate prediction errors
model_comparison["Error_Original"] = (
    model_comparison["Actual_Demand_MWh"]
    - model_comparison["Predicted_Demand_MWh_Original"]
)

model_comparison["Error_Holiday"] = (
    model_comparison["Actual_Demand_MWh"]
    - model_comparison["Predicted_Demand_MWh_Holiday"]
)

# Compare holiday and normal-day performance
for group_name, group_data in [
    ("PUBLIC HOLIDAYS", model_comparison[model_comparison["Is_Public_Holiday"]]),
    ("NORMAL DAYS", model_comparison[~model_comparison["Is_Public_Holiday"]])
]:
    print(f"\n--- {group_name} ---")

    print("Number of days:", len(group_data))

    print(
        "Original XGBoost MAE:",
        round(group_data["Error_Original"].abs().mean(), 2)
    )

    print(
        "Holiday-Enhanced XGBoost MAE:",
        round(group_data["Error_Holiday"].abs().mean(), 2)
    )

# Model 7 Feature Importance

holiday_importance = pd.DataFrame({
    "Feature": holiday_features,
    "Importance": xgb_holiday_model.feature_importances_
})

holiday_importance = holiday_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nModel 7 - Holiday-Enhanced XGBoost Feature Importance")
print(holiday_importance.to_string(index=False))

# ============================================================
# MODEL 8: XGBOOST HYPERPARAMETER TUNING
# ============================================================

print("\n======================================")
print("MODEL 8: XGBOOST HYPERPARAMETER TUNING")
print("======================================")

# 1. Split the original training data into training
#    and validation sets. Keep the final test set separate.

validation_split = int(len(X_train_holiday) * 0.8)

X_tune_train = X_train_holiday.iloc[:validation_split]
y_tune_train = y_train_holiday.iloc[:validation_split]

X_validation = X_train_holiday.iloc[validation_split:]
y_validation = y_train_holiday.iloc[validation_split:]

print("Tuning training records:", len(X_tune_train))
print("Validation records:", len(X_validation))

# 2. Define hyperparameter combinations to test

parameter_options = [
    {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_child_weight": 2
    },
    {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 2
    },
    {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 2
    },
    {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_child_weight": 5
    },
    {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_child_weight": 5
    },
    {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 5
    }
]

# 3. Train and evaluate each configuration

tuning_results = []

for i, params in enumerate(parameter_options, start=1):

    model = XGBRegressor(
        **params,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_tune_train, y_tune_train)

    validation_predictions = model.predict(X_validation)

    validation_mae = mean_absolute_error(
        y_validation,
        validation_predictions
    )

    validation_rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            validation_predictions
        )
    )

    tuning_results.append({
        "Configuration": i,
        **params,
        "Validation_MAE": validation_mae,
        "Validation_RMSE": validation_rmse
    })

    print(
        f"Configuration {i}: "
        f"MAE = {validation_mae:.2f}, "
        f"RMSE = {validation_rmse:.2f}"
    )

# 4. Display tuning results

tuning_results_df = pd.DataFrame(tuning_results)

tuning_results_df = tuning_results_df.sort_values(
    by="Validation_MAE"
).reset_index(drop=True)

print("\nHYPERPARAMETER TUNING RESULTS")
print(tuning_results_df.to_string(index=False))

# 5. Select the best configuration using validation MAE

best_configuration = tuning_results_df.iloc[0]

best_parameters = {
    "n_estimators": int(best_configuration["n_estimators"]),
    "learning_rate": float(best_configuration["learning_rate"]),
    "max_depth": int(best_configuration["max_depth"]),
    "min_child_weight": int(best_configuration["min_child_weight"])
}

print("\nBEST PARAMETERS")
print(best_parameters)

# 6. Retrain using all original training records

tuned_xgb_model = XGBRegressor(
    **best_parameters,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

tuned_xgb_model.fit(
    X_train_holiday,
    y_train_holiday
)

# 7. Evaluate once on the final test set

y_pred_tuned = tuned_xgb_model.predict(X_test_holiday)

tuned_mae = mean_absolute_error(
    y_test_holiday,
    y_pred_tuned
)

tuned_rmse = np.sqrt(
    mean_squared_error(
        y_test_holiday,
        y_pred_tuned
    )
)

tuned_r2 = r2_score(
    y_test_holiday,
    y_pred_tuned
)

print("\n======================================")
print("TUNED XGBOOST FINAL TEST PERFORMANCE")
print("======================================")

print("MAE:", round(tuned_mae, 2), "MWh")
print("RMSE:", round(tuned_rmse, 2), "MWh")
print("R-squared:", round(tuned_r2, 4))

# 8. Save final test predictions

tuned_xgb_results = pd.DataFrame({
    "Date": holiday_test_data["Date"].values,
    "Actual_Demand_MWh": y_test_holiday.to_numpy(),
    "Predicted_Demand_MWh": y_pred_tuned
})

tuned_xgb_results["Error_MWh"] = (
    tuned_xgb_results["Actual_Demand_MWh"]
    - tuned_xgb_results["Predicted_Demand_MWh"]
)

# ============================================================
# MODEL 8: HOLIDAY VS NORMAL-DAY PERFORMANCE
# ============================================================

tuned_xgb_results["Date"] = pd.to_datetime(
    tuned_xgb_results["Date"]
)

tuned_xgb_results["Holiday_Name"] = (
    tuned_xgb_results["Date"].dt.date.apply(
        lambda date: za_holidays.get(date)
    )
)

tuned_xgb_results["Is_Public_Holiday"] = (
    tuned_xgb_results["Holiday_Name"].notna()
)

holiday_data_8 = tuned_xgb_results[
    tuned_xgb_results["Is_Public_Holiday"]
]

normal_data_8 = tuned_xgb_results[
    ~tuned_xgb_results["Is_Public_Holiday"]
]

print("\n======================================")
print("MODEL 8: HOLIDAY PERFORMANCE")
print("======================================")

print("\nPUBLIC HOLIDAYS")
print("Number of days:", len(holiday_data_8))
print(
    "MAE:",
    round(holiday_data_8["Error_MWh"].abs().mean(), 2)
)
print(
    "Average Error:",
    round(holiday_data_8["Error_MWh"].mean(), 2)
)

print("\nNORMAL DAYS")
print("Number of days:", len(normal_data_8))
print(
    "MAE:",
    round(normal_data_8["Error_MWh"].abs().mean(), 2)
)
print(
    "Average Error:",
    round(normal_data_8["Error_MWh"].mean(), 2)
)

print("\nPUBLIC HOLIDAY PREDICTIONS")
print(
    holiday_data_8[
        [
            "Date",
            "Holiday_Name",
            "Actual_Demand_MWh",
            "Predicted_Demand_MWh",
            "Error_MWh"
        ]
    ].to_string(index=False)
)
# MODEL 8: FEATURE IMPORTANCE

importance_8 = pd.DataFrame({
    "Feature": holiday_features,
    "Importance": tuned_xgb_model.feature_importances_
})

importance_8 = importance_8.sort_values(
    by="Importance",
    ascending=False
)

print("\n======================================")
print("MODEL 8: FEATURE IMPORTANCE")
print("======================================")

print(importance_8.to_string(index=False))


# ==========================================
# MODEL 8: SEVEN-DAY RECURSIVE FORECAST
# ==========================================

import numpy as np

# Use the latest available historical demand data
forecast_history = daily_demand[
    ["Date", "Daily_Energy_MWh"]
].copy()

forecast_history["Date"] = pd.to_datetime(
    forecast_history["Date"]
)

forecast_history = forecast_history.sort_values(
    "Date"
).reset_index(drop=True)

# Store the latest demand date
last_date = forecast_history["Date"].max()

# Create the next seven dates
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=7,
    freq="D"
)

# Store future predictions here
future_predictions = []

# Make predictions one day at a time
for forecast_date in future_dates:

    # Retrieve the most recent available demand values
    recent_demand = forecast_history[
        "Daily_Energy_MWh"
    ].tolist()

    # Calculate demand features
    lag_1_day = recent_demand[-1]
    lag_7_days = recent_demand[-7]
    rolling_7_day_average = np.mean(
        recent_demand[-7:]
    )

    # Calendar features
    day_of_week = forecast_date.day_of_week
    month = forecast_date.month
    year = forecast_date.year
    is_weekend = int(day_of_week >= 5)
    day_of_year = forecast_date.day_of_year

    # Public holiday features
    holiday_name = za_holidays.get(
        forecast_date.date()
    )

    is_public_holiday = int(
        holiday_name is not None
    )

    next_day_holiday = za_holidays.get(
        (forecast_date + pd.Timedelta(days=1)).date()
    )

    previous_day_holiday = za_holidays.get(
        (forecast_date - pd.Timedelta(days=1)).date()
    )

    day_before_holiday = int(
        next_day_holiday is not None
    )

    day_after_holiday = int(
        previous_day_holiday is not None
    )

    holiday_season = int(
        (
            month == 12 and forecast_date.day >= 20
        )
        or (
            month == 1 and forecast_date.day <= 5
        )
    )

    # Prepare model input in the exact training feature order
    forecast_input = pd.DataFrame([{
        "Day_of_the_week": day_of_week,
        "Month": month,
        "Year": year,
        "Is_weekend": is_weekend,
        "Day_of_year": day_of_year,
        "Is_public_holiday": is_public_holiday,
        "Day_Before_Holiday": day_before_holiday,
        "Day_After_Holiday": day_after_holiday,
        "Holiday_Season": holiday_season,
        "Lag_1_Day": lag_1_day,
        "Lag_7_Days": lag_7_days,
        "Rolling_7_Day_Average": rolling_7_day_average
    }])

    forecast_input = forecast_input[
        holiday_features
    ]

    # Predict demand using the trained Model 8
    predicted_demand = tuned_xgb_model.predict(
        forecast_input
    )[0]

    # Save prediction
    future_predictions.append({
        "Date": forecast_date,
        "Predicted_Demand_MWh": predicted_demand,
        "Holiday_Name": holiday_name
    })

    # Add prediction to history so the next day
    # can use it as a lag value
    forecast_history = pd.concat([
        forecast_history,
        pd.DataFrame([{
            "Date": forecast_date,
            "Daily_Energy_MWh": predicted_demand
        }])
    ], ignore_index=True)


# Create final forecast table
seven_day_forecast = pd.DataFrame(
    future_predictions
)

print("\n======================================")
print("SOUTH AFRICA: SEVEN-DAY DEMAND FORECAST")
print("======================================")

print(
    seven_day_forecast.to_string(index=False)
)

print("\nForecast completed successfully.")
# ==========================================
# SAVE SEVEN-DAY FORECAST
# ==========================================

forecast_file = "data/processed/seven_day_demand_forecast.csv"

seven_day_forecast.to_csv(
    forecast_file,
    index=False
)

print("\nForecast saved successfully!")
print("File location:", forecast_file)
