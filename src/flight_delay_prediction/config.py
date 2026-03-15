"""Project constants extracted from the notebook."""
from pathlib import Path

#########################################
# DATA CLEANING CONFIGS
#########################################

SEED = 100
DELAY_THRESHOLD = 15
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_PATH = PROJECT_ROOT / "data/intermediate"

FLIGHT_DELAY_INDICATOR_COLUMNS = [
    "SecurityDelay",
    "LateAircraftDelay",
    "NASDelay",
    "WeatherDelay",
    "CarrierDelay",
]

FLIGHT_COLUMNS_TO_DROP = [
    "DepDelay",
    "TaxiOut",
    "WheelsOff",
    "WheelsOn",
    "TaxiIn",
    "ActualElapsedTime",
    "Tail_Number",
    "FlightId",
]

FINAL_FLIGHT_COLUMNS_TO_DROP = ["CRSDepTime", "CRSArrTime"]

DESTINATION_WEATHER_COLUMNS_TO_DROP = [
    "dest_apparent_temperature_mean",
    "dest_apparent_temperature_max",
    "dest_apparent_temperature_min",
    "dest_wind_direction_10m_dominant",
    "dest_shortwave_radiation_sum",
    "dest_et0_fao_evapotranspiration",
]

ORIGIN_WEATHER_COLUMNS_TO_DROP = [
    "origin_apparent_temperature_mean",
    "origin_apparent_temperature_max",
    "origin_apparent_temperature_min",
    "origin_wind_direction_10m_dominant",
    "origin_shortwave_radiation_sum",
    "origin_et0_fao_evapotranspiration",
]

SUPPLEMENTARY_COLUMNS_TO_DROP = [
    "DOT_ID_Reporting_Airline",
    "dest_time",
    "origin_time",
    "origin_code",
    "code",
    "DOT_Id",
]

MODELING_COLUMNS_TO_DROP = [
    "FlightDate",
    "Dest",
    "ArrDelay",
    "Month",
    "DayOfWeek",
    "origin_latitude",
    "origin_longitude",
]

OH_ENCODING_COLS = ["Year", "Season", "CRSArr_TimeOfDay", "CRSDep_TimeOfDay"]
TARGET_ENCODING_COLS = ["Origin", "dest_weather_code", "origin_weather_code", "airline_name"]
TARGET_COLUMN = "IsDelayed"

NUMERIC_COLS = [
    "CRSElapsedTime",
    "AirTime",
    "Distance",
    "dest_temperature_2m_mean",
    "dest_wind_speed_10m_max",
    "dest_wind_gusts_10m_max",
    "dest_precipitation_sum",
    "dest_rain_sum",
    "dest_snowfall_sum",
    "dest_precipitation_hours",
    "dest_temperature_2m_range",
    "origin_temperature_2m_mean",
    "origin_wind_speed_10m_max",
    "origin_wind_gusts_10m_max",
    "origin_precipitation_sum",
    "origin_rain_sum",
    "origin_snowfall_sum",
    "origin_precipitation_hours",
    "origin_temperature_2m_range",
    "SinMonth",
    "CosMonth",
    "SinDay",
    "CosDay",
]

#########################################
# BTS SCRAPER CONFIGS
#########################################

ZIP_FILE_DIR = Path.cwd().parent / "data/external/bts_zip_files/"
EXTERNAL_DATA_DIR = Path.cwd().parent / "data/external"

YEAR_START = 2020
YEAR_END = 2025
MONTH_CUTOFF = 7

BASE_URL = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"

