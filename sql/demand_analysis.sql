-- Project: South Africa Electricity Demand Forecasting
-- Purpose: Explore historical electricity demand patterns
-- Platform: Google BigQuery


-- 1. YEARLY DEMAND ANALYSIS
SELECT
    EXTRACT(YEAR FROM Date) AS Year,
    COUNT(*) AS Number_of_Days,
    ROUND(AVG(Daily_Energy_MWh), 2) AS Average_Daily_Energy_MWh,
    ROUND(MAX(Daily_Energy_MWh), 2) AS Highest_Daily_Energy_MWh,
    ROUND(MIN(Daily_Energy_MWh), 2) AS Lowest_Daily_Energy_MWh
FROM
    `sa-electricity-demand-forecast.electricity_demand.daily_demand`
GROUP BY
    Year
ORDER BY
    Year;



-- 2. MONTHLY DEMAND ANALYSIS
SELECT
    EXTRACT(YEAR FROM Date) AS Year,
    EXTRACT(MONTH FROM Date) AS Month_Number,
    FORMAT_DATE('%B', Date) AS Month_Name,
    COUNT(*) AS Number_of_Days,
    ROUND(AVG(Daily_Energy_MWh), 2) AS Average_Daily_Energy_MWh
FROM
    `sa-electricity-demand-forecast.electricity_demand.daily_demand`
GROUP BY
    Year,
    Month_Number,
    Month_Name
ORDER BY
    Year,
    Month_Number;



-- 3. YEAR-ON-YEAR MONTHLY COMPARISON
WITH monthly_demand AS (
    SELECT
        EXTRACT(YEAR FROM Date) AS Year,
        EXTRACT(MONTH FROM Date) AS Month,
        COUNT(*) AS Number_of_Days,
        AVG(Daily_Energy_MWh) AS Average_Daily_Energy_MWh
    FROM
        `sa-electricity-demand-forecast.electricity_demand.daily_demand`
    GROUP BY
        Year,
        Month
),

yearly_comparison AS (
    SELECT
        Year,
        Month,
        Number_of_Days,
        Average_Daily_Energy_MWh,
        LAG(Average_Daily_Energy_MWh) OVER (
            PARTITION BY Month
            ORDER BY Year
        ) AS Previous_Year_Average
    FROM monthly_demand
)

SELECT
    Year,
    Month,
    Number_of_Days,
    ROUND(Average_Daily_Energy_MWh, 2) AS Current_Year_Average,
    ROUND(Previous_Year_Average, 2) AS Previous_Year_Average,
    ROUND(
        Average_Daily_Energy_MWh - Previous_Year_Average,
        2
    ) AS Change_MWh,
    ROUND(
        SAFE_DIVIDE(
            Average_Daily_Energy_MWh - Previous_Year_Average,
            Previous_Year_Average
        ) * 100,
        2
    ) AS Change_Percentage
FROM yearly_comparison
ORDER BY
    Year,
    Month;


-- 4. SEASONAL DEMAND ANALYSIS
SELECT
    CASE
        WHEN EXTRACT(MONTH FROM Date) = 12
            THEN EXTRACT(YEAR FROM Date) + 1
        ELSE EXTRACT(YEAR FROM Date)
    END AS Year,

    CASE
        WHEN EXTRACT(MONTH FROM Date) IN (12, 1, 2)
            THEN 'Summer'
        WHEN EXTRACT(MONTH FROM Date) IN (3, 4, 5)
            THEN 'Autumn'
        WHEN EXTRACT(MONTH FROM Date) IN (6, 7, 8)
            THEN 'Winter'
        WHEN EXTRACT(MONTH FROM Date) IN (9, 10, 11)
            THEN 'Spring'
    END AS Season,

    COUNT(*) AS Number_of_Days,

    ROUND(
        AVG(Daily_Energy_MWh), 2
    ) AS Average_Daily_Energy_MWh

FROM
    `sa-electricity-demand-forecast.electricity_demand.daily_demand`

GROUP BY
    Year,
    Season

ORDER BY
    Year,
    CASE Season
        WHEN 'Summer' THEN 1
        WHEN 'Autumn' THEN 2
        WHEN 'Winter' THEN 3
        WHEN 'Spring' THEN 4
    END;


-- 5. WEEKDAY VS WEEKEND ANALYSIS
SELECT
    CASE
        WHEN EXTRACT(DAYOFWEEK FROM Date) IN (1, 7)
            THEN 'Weekend'
        ELSE 'Weekday'
    END AS Day_Type,

    COUNT(*) AS Number_of_Days,

    ROUND(
        AVG(Daily_Energy_MWh), 2
    ) AS Average_Daily_Energy_MWh,

    ROUND(
        MAX(Daily_Energy_MWh), 2
    ) AS Highest_Daily_Energy_MWh,

    ROUND(
        MIN(Daily_Energy_MWh), 2
    ) AS Lowest_Daily_Energy_MWh

FROM
    `sa-electricity-demand-forecast.electricity_demand.daily_demand`

GROUP BY
    Day_Type

ORDER BY
    Average_Daily_Energy_MWh DESC;


-- 6. INDIVIDUAL WEEKDAY ANALYSIS

SELECT
    FORMAT_DATE('%A', Date) AS Day_Name,
    EXTRACT(DAYOFWEEK FROM Date) AS Day_Number,
    COUNT(*) AS Number_of_Days,

    ROUND(
        AVG(Daily_Energy_MWh), 2
    ) AS Average_Daily_Energy_MWh

FROM
    `sa-electricity-demand-forecast.electricity_demand.daily_demand`

GROUP BY
    Day_Name,
    Day_Number

ORDER BY
    Day_Number;
