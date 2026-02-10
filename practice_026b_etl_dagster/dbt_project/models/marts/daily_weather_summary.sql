-- daily_weather_summary.sql
-- Mart model: aggregate weather readings by city and date.
--
-- WHAT IS A MART MODEL?
--   In dbt's modeling pattern, mart models are the FINAL layer -- they produce
--   tables optimized for consumption by dashboards, reports, or downstream
--   applications. Marts:
--   1. SELECT from staging or intermediate models (via ref())
--   2. Aggregate, join, pivot, or reshape data
--   3. Are materialized as TABLES (for query performance)
--   4. Have clear business meaning ("daily weather summary per city")
--
-- HOW ref() WORKS:
--   dbt's ref('stg_weather_readings') resolves to the actual table/view name
--   at compile time. It ALSO creates a dependency: this model depends on
--   stg_weather_readings. Dagster reads this dependency from the manifest
--   and creates an edge in the asset graph:
--       stg_weather_readings (dbt) --> daily_weather_summary (dbt)
--
-- TODO(human): Implement the aggregation query below.
--
-- YOUR TASK:
--   Write a SELECT statement that aggregates stg_weather_readings by city and
--   reading_date. For each (city, date) combination, compute:
--
--   - city: the city name
--   - reading_date: the date
--   - avg_temperature: AVG(temperature_c), rounded to 1 decimal
--   - min_temperature: MIN(temperature_c)
--   - max_temperature: MAX(temperature_c)
--   - avg_humidity: AVG(humidity_pct), rounded to 1 decimal
--   - max_wind: MAX(wind_speed_kmh)
--   - reading_count: COUNT(*)
--   - dominant_category: the temperature_category that appears most often (MODE)
--     Hint: Use PostgreSQL's MODE() WITHIN GROUP (ORDER BY temperature_category)
--     If MODE() feels unfamiliar, you can skip it and use a simpler alternative:
--       MAX(temperature_category) or just leave it out
--
-- EXAMPLE OUTPUT:
--   | city     | reading_date | avg_temperature | min_temperature | max_temperature | avg_humidity | max_wind | reading_count | dominant_category |
--   |----------|-------------|-----------------|-----------------|-----------------|--------------|----------|---------------|-------------------|
--   | New York | 2024-01-15  | 2.3             | -1.5            | 6.1             | 72.4         | 35.0     | 24            | cold              |
--   | London   | 2024-01-15  | 5.8             | 3.2             | 8.4             | 85.1         | 28.0     | 24            | cold              |
--
-- HINT:
--   SELECT
--       city,
--       reading_date,
--       ROUND(AVG(temperature_c)::NUMERIC, 1) AS avg_temperature,
--       ...
--   FROM {{ ref('stg_weather_readings') }}
--   GROUP BY city, reading_date
--   ORDER BY city, reading_date
--
-- AFTER IMPLEMENTING:
--   1. Run `dbt run --select daily_weather_summary` in the container to test
--   2. Or materialize it from Dagit as part of the full dbt asset chain
--   3. Check the output: `SELECT * FROM daily_weather_summary LIMIT 10`

-- =========================================================================
-- STUB: Replace this entire SELECT with your implementation.
-- This stub ensures dbt can compile the model even before you implement it.
-- =========================================================================
SELECT
    city,
    reading_date,
    0.0::REAL AS avg_temperature,
    0.0::REAL AS min_temperature,
    0.0::REAL AS max_temperature,
    0.0::REAL AS avg_humidity,
    0.0::REAL AS max_wind,
    0::INTEGER AS reading_count
FROM {{ ref('stg_weather_readings') }}
WHERE FALSE  -- Returns no rows; replace with your GROUP BY query
