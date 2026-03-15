"""Cleaning and join logic extracted from the notebook."""

import numpy as np
import pandas as pd
import pandasql as ps

from flight_delay_prediction.config import (
    DELAY_THRESHOLD,
    DESTINATION_WEATHER_COLUMNS_TO_DROP,
    FINAL_FLIGHT_COLUMNS_TO_DROP,
    FLIGHT_COLUMNS_TO_DROP,
    FLIGHT_DELAY_INDICATOR_COLUMNS,
    ORIGIN_WEATHER_COLUMNS_TO_DROP,
    SUPPLEMENTARY_COLUMNS_TO_DROP,
)
from flight_delay_prediction.features.build_features import extract_time_of_day



def clean_flight_data(flights_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the flight cleaning and simple feature engineering from the notebook."""
    flights_df = flights_df.copy()
    flights_df["FlightDate"] = pd.to_datetime(flights_df["FlightDate"])

    completed_flights_df = (
        flights_df[(flights_df["Cancelled"] == 0) & (flights_df["Diverted"] == 0)]
        .copy()
        .drop(columns=["Cancelled", "Diverted"])
    )

    completed_flights_df = completed_flights_df.drop(columns=FLIGHT_DELAY_INDICATOR_COLUMNS)

    engineered_flights_df = completed_flights_df.copy()
    engineered_flights_df = engineered_flights_df.drop(columns=FLIGHT_COLUMNS_TO_DROP)

    engineered_flights_df["Year"] = engineered_flights_df["FlightDate"].dt.year
    engineered_flights_df["Month"] = engineered_flights_df["FlightDate"].dt.month
    engineered_flights_df["DayOfWeek"] = engineered_flights_df["FlightDate"].dt.dayofweek
    engineered_flights_df["IsWeekend"] = (
        engineered_flights_df["DayOfWeek"].isin([5, 6]).astype(int)
    )

    query = """
    SELECT
      *,
      CASE
        WHEN Month IN (9, 10, 11) THEN 'Fall'
        WHEN Month IN (12, 1, 2) THEN 'Winter'
        WHEN Month IN (3, 4, 5) THEN 'Spring'
        WHEN Month IN (6, 7, 8) THEN 'Summer'
      END as Season
    FROM engineered_flights_df
    """

    engineered_flights_df = ps.sqldf(query, locals())
    engineered_flights_df = pd.DataFrame(engineered_flights_df)
    engineered_flights_df["FlightDate"] = pd.to_datetime(engineered_flights_df["FlightDate"])

    engineered_flights_df["CRSArr_TimeOfDay"] = engineered_flights_df["CRSArrTime"].apply(
        extract_time_of_day
    )
    engineered_flights_df["CRSDep_TimeOfDay"] = engineered_flights_df["CRSDepTime"].apply(
        extract_time_of_day
    )
    engineered_flights_df["IsDelayed"] = np.where(
        engineered_flights_df["ArrDelay"] > DELAY_THRESHOLD,
        1,
        0,
    )

    cleaned_flights_df = engineered_flights_df.copy()
    cleaned_flights_df = cleaned_flights_df.drop(columns=FINAL_FLIGHT_COLUMNS_TO_DROP)

    return cleaned_flights_df



def clean_destination_weather_data(destination_weather_df: pd.DataFrame) -> pd.DataFrame:
    """Apply destination weather cleaning and feature engineering from the notebook."""
    destination_weather_df = destination_weather_df.copy()
    destination_weather_df = destination_weather_df.drop(
        columns=DESTINATION_WEATHER_COLUMNS_TO_DROP
    )
    destination_weather_df["dest_time"] = pd.to_datetime(destination_weather_df["dest_time"])
    destination_weather_df["dest_temperature_2m_range"] = (
        destination_weather_df["dest_temperature_2m_max"]
        - destination_weather_df["dest_temperature_2m_min"]
    )
    destination_weather_df = destination_weather_df.drop(
        columns=["dest_temperature_2m_max", "dest_temperature_2m_min"]
    )
    return destination_weather_df



def clean_origin_weather_data(origin_weather_df: pd.DataFrame) -> pd.DataFrame:
    """Apply origin weather cleaning and feature engineering from the notebook."""
    origin_weather_df = origin_weather_df.copy()
    origin_weather_df = origin_weather_df.drop(columns=ORIGIN_WEATHER_COLUMNS_TO_DROP)
    origin_weather_df = origin_weather_df[origin_weather_df["origin_code"].notna()].copy()
    origin_weather_df["origin_time"] = pd.to_datetime(origin_weather_df["origin_time"])
    origin_weather_df["origin_temperature_2m_range"] = (
        origin_weather_df["origin_temperature_2m_max"]
        - origin_weather_df["origin_temperature_2m_min"]
    )
    origin_weather_df = origin_weather_df.drop(
        columns=["origin_temperature_2m_max", "origin_temperature_2m_min"]
    )
    return origin_weather_df



def join_flights_and_weather(
    cleaned_flights_df: pd.DataFrame,
    destination_weather_df: pd.DataFrame,
    origin_weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join flights to destination and origin weather exactly as in the notebook."""
    intermediate_df = pd.merge(
        left=cleaned_flights_df,
        right=destination_weather_df,
        how="left",
        left_on="FlightDate",
        right_on="dest_time",
    )

    flights_weather_df = pd.merge(
        left=intermediate_df,
        right=origin_weather_df,
        how="left",
        left_on=["FlightDate", "Origin"],
        right_on=["origin_time", "origin_code"],
    )

    return flights_weather_df



def enrich_with_supplementary_data(
    flights_weather_df: pd.DataFrame,
    airports: pd.DataFrame,
    airlines: pd.DataFrame,
) -> pd.DataFrame:
    """Join supplementary airport and airline lookup data and apply the manual fixes."""
    intermediate_df2 = pd.merge(
        left=flights_weather_df,
        right=airports[["code", "latitude", "longitude"]],
        how="left",
        left_on="Origin",
        right_on="code",
    )

    final_cleaned_df = pd.merge(
        left=intermediate_df2,
        right=airlines,
        how="left",
        left_on="DOT_ID_Reporting_Airline",
        right_on="DOT_Id",
    )

    final_cleaned_df.loc[final_cleaned_df["Origin"] == "SJU", "latitude"] = 18.44
    final_cleaned_df.loc[final_cleaned_df["Origin"] == "SJU", "longitude"] = 66.00
    final_cleaned_df.loc[final_cleaned_df["Origin"] == "STT", "latitude"] = 18.34
    final_cleaned_df.loc[final_cleaned_df["Origin"] == "STT", "longitude"] = 64.97
    final_cleaned_df.loc[final_cleaned_df["Origin"] == "BQN", "latitude"] = 18.50
    final_cleaned_df.loc[final_cleaned_df["Origin"] == "BQN", "longitude"] = 67.14

    final_cleaned_df = final_cleaned_df.drop(columns=SUPPLEMENTARY_COLUMNS_TO_DROP)
    final_cleaned_df.rename(
        columns={
            "latitude": "origin_latitude",
            "longitude": "origin_longitude",
            "Description": "airline_name",
        },
        inplace=True,
    )

    return final_cleaned_df
