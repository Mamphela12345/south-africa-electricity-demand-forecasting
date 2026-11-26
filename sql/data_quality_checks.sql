
-- =========================================================
-- Project: South Africa Electricity Demand Forecasting
-- Purpose: Validate the processed daily-demand dataset
-- Platform: Google BigQuery
-- Table: electricity_demand.daily_demand
-- =========================================================


-- 1. Check for duplicate dates and inspect the date range

SELECT
    COUNT(*) AS Total_Records,
    COUNT(DISTINCT Date) AS Unique_Dates,
    COUNT(*) - COUNT(DISTINCT Date) AS Duplicate_Records,
    MIN(Date) AS First_Date,
    MAX(Date) AS Last_Date
FROM
    `sa-electricity-demand-forecast.electricity_demand.daily_demand`;


-- 2. Check for missing calendar dates

SELECT
    DATE_DIFF(MAX(Date), MIN(Date), DAY) + 1 AS Expected_Days,
    COUNT(*) AS Actual_Records,
    (
        DATE_DIFF(MAX(Date), MIN(Date), DAY) + 1
    ) - COUNT(*) AS Missing_Days
FROM
    `sa-electricity-demand-forecast.electricity_demand.daily_demand`;


-- 3. Check for NULL dates, NULL energy values,
--    and non-positive energy measurements

SELECT
    COUNT(*) AS Total_Records,
    COUNTIF(Date IS NULL) AS Null_Dates,
    COUNTIF(Daily_Energy_MWh IS NULL) AS Null_Energy_Values,
    COUNTIF(Daily_Energy_MWh <= 0) AS Non_Positive_Energy_Values
FROM
    `sa-electricity-demand-forecast.electricity_demand.daily_demand`;
