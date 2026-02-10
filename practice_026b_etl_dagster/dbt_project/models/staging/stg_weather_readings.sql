-- stg_weather_readings.sql
-- Staging model: clean and standardize raw weather readings.
--
-- WHAT IS A STAGING MODEL?
--   In dbt's modeling pattern (staging -> intermediate -> marts), staging models are
--   the first transformation layer. They:
--   1. SELECT from sources (raw tables)
--   2. Rename columns to a consistent naming convention
--   3. Cast types explicitly
--   4. Filter out obviously invalid data
--   5. Are materialized as VIEWS (cheap, always up-to-date)
--
-- HOW THIS BECOMES A DAGSTER ASSET:
--   When Dagster reads the dbt manifest, this model becomes an asset named
--   "stg_weather_readings". Its dependency on {{ source('raw', 'raw_weather_readings') }}
--   becomes an edge to the raw_weather_readings source, which Dagster maps to
--   the weather_source Python asset.
--
--   In Dagit's asset graph, you'll see:
--       weather_source (Python) --> stg_weather_readings (dbt view)
--
-- COMPARISON WITH AIRFLOW:
--   In Airflow, this SQL would be inside a PostgresOperator or a dbt BashOperator.
--   Airflow sees it as an opaque task. Dagster sees it as a DATA ASSET with
--   lineage, metadata, and materialization history.

SELECT
    id,
    -- Standardize city names: trim whitespace, title case
    INITCAP(TRIM(city)) AS city,
    -- Explicit casting for type safety
    temperature_c::REAL AS temperature_c,
    humidity_pct::REAL AS humidity_pct,
    wind_speed_kmh::REAL AS wind_speed_kmh,
    reading_date,
    recorded_at,
    -- Add computed columns for downstream use
    CASE
        WHEN temperature_c < 0 THEN 'freezing'
        WHEN temperature_c < 15 THEN 'cold'
        WHEN temperature_c < 25 THEN 'mild'
        WHEN temperature_c < 35 THEN 'warm'
        ELSE 'hot'
    END AS temperature_category
FROM {{ source('raw', 'raw_weather_readings') }}
WHERE
    -- Basic validation: filter out sensor errors
    temperature_c BETWEEN -60 AND 60
    AND humidity_pct BETWEEN 0 AND 100
    AND wind_speed_kmh >= 0
