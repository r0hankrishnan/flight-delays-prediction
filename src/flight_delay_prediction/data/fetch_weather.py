"""Functions for fetching historical daily weather data from Open-Meteo."""

from __future__ import annotations

import time
from datetime import timedelta
from typing import Iterable, Sequence

import pandas as pd
import requests


API_URL = "https://archive-api.open-meteo.com/v1/archive"

WEATHER_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "weather_code",
]

# Open-Meteo free-tier-safe defaults
MAX_VARS_PER_REQUEST = 10
MAX_DAYS_PER_REQUEST = 14
PAUSE_BETWEEN_REQUESTS_SECONDS = 0.25
RETRY_SLEEP_SECONDS = 65
MAX_RETRIES = 3


def chunk_sequence(seq: Sequence, chunk_size: int) -> list[list]:
    """Split a sequence into fixed-size chunks."""
    return [list(seq[i : i + chunk_size]) for i in range(0, len(seq), chunk_size)]


def build_date_windows(
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    max_days_per_request: int = MAX_DAYS_PER_REQUEST,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split a full date range into inclusive windows no longer than max_days_per_request."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current_start = start

    while current_start <= end:
        current_end = min(current_start + timedelta(days=max_days_per_request - 1), end)
        windows.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)

    return windows


def build_variable_groups(
    weather_vars: Sequence[str],
    max_vars_per_request: int = MAX_VARS_PER_REQUEST,
) -> list[list[str]]:
    """Split weather variables into groups of at most max_vars_per_request."""
    return chunk_sequence(list(weather_vars), max_vars_per_request)


def get_flight_date_range(
    flights: pd.DataFrame,
    date_col: str = "FlightDate",
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the min and max flight dates from the flights dataframe."""
    dates = pd.to_datetime(flights[date_col])
    return dates.min().normalize(), dates.max().normalize()


def build_request_params(
    latitude: float,
    longitude: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    weather_vars: Sequence[str],
) -> dict:
    """Build Open-Meteo request parameters for one location, one time window, one variable group."""
    return {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": ",".join(weather_vars),
    }


def request_with_retry(
    session: requests.Session,
    params: dict,
    api_url: str = API_URL,
    max_retries: int = MAX_RETRIES,
    retry_sleep_seconds: int = RETRY_SLEEP_SECONDS,
) -> dict:
    """Make one Open-Meteo request with retry handling for rate limiting."""
    for attempt in range(max_retries + 1):
        response = session.get(api_url, params=params, timeout=120)

        if response.status_code == 200:
            return response.json()

        if response.status_code == 429 and attempt < max_retries:
            print(
                f"429 rate limit hit. Sleeping {retry_sleep_seconds} seconds "
                f"before retry {attempt + 1}/{max_retries}."
            )
            time.sleep(retry_sleep_seconds)
            continue

        print(response.text)
        raise Exception(f"Request failed with status code: {response.status_code}")

    raise RuntimeError("Unreachable retry state in request_with_retry().")


def unpack_single_location_response(
    response_json: dict,
    airport_code: str,
    latitude: float,
    longitude: float,
    airport_col_name: str,
) -> pd.DataFrame:
    """
    Convert one Open-Meteo response for one location into a dataframe.

    Adds airport code and coordinate columns so the result is easy to join later.
    """
    if "daily" not in response_json:
        raise KeyError("Response JSON missing 'daily' key.")

    weather_df = pd.DataFrame(response_json["daily"])
    weather_df[airport_col_name] = airport_code
    weather_df["latitude"] = latitude
    weather_df["longitude"] = longitude
    return weather_df


def combine_weather_chunks(
    weather_chunks: Iterable[pd.DataFrame],
    airport_col_name: str,
    date_col: str = "time",
) -> pd.DataFrame:
    """
    Combine chunked weather responses into one wide dataframe.

    Because requests are split by variable groups and date windows, the same
    airport/date key can appear multiple times with different weather columns.
    This function collapses those partial rows back into one row per airport/date.
    """
    chunks = list(weather_chunks)
    if not chunks:
        return pd.DataFrame()

    combined = pd.concat(chunks, ignore_index=True)

    combined = (
        combined.groupby([date_col, airport_col_name, "latitude", "longitude"], as_index=False)
        .first()
        .sort_values([airport_col_name, date_col])
        .reset_index(drop=True)
    )

    return combined


def fetch_weather_for_airports(
    airport_tuples: list[tuple[str, float, float]],
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    airport_col_name: str,
    weather_vars: Sequence[str] = WEATHER_VARS,
    max_days_per_request: int = MAX_DAYS_PER_REQUEST,
    max_vars_per_request: int = MAX_VARS_PER_REQUEST,
    pause_between_requests_seconds: float = PAUSE_BETWEEN_REQUESTS_SECONDS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch weather for a list of (airport_code, latitude, longitude) tuples.

    Requests are chunked so that each HTTP request stays within:
    - 1 location
    - <= 10 weather variables
    - <= 14 days
    """
    if not airport_tuples:
        return pd.DataFrame()

    date_windows = build_date_windows(start_date, end_date, max_days_per_request)
    variable_groups = build_variable_groups(weather_vars, max_vars_per_request)

    total_requests = len(airport_tuples) * len(date_windows) * len(variable_groups)

    if verbose:
        print(
            f"Planned weather requests: {total_requests} "
            f"({len(airport_tuples)} airports × {len(date_windows)} date windows × "
            f"{len(variable_groups)} variable groups)"
        )

    weather_chunks: list[pd.DataFrame] = []
    request_counter = 0

    with requests.Session() as session:
        for airport_code, latitude, longitude in airport_tuples:
            if pd.isna(latitude) or pd.isna(longitude):
                print(f"Skipping {airport_code}: missing latitude/longitude.")
                continue

            for window_start, window_end in date_windows:
                for var_group in variable_groups:
                    request_counter += 1

                    if verbose:
                        print(
                            f"Request {request_counter}/{total_requests} | "
                            f"{airport_code} | "
                            f"{window_start.date()} to {window_end.date()} | "
                            f"{len(var_group)} vars"
                        )

                    params = build_request_params(
                        latitude=latitude,
                        longitude=longitude,
                        start_date=window_start,
                        end_date=window_end,
                        weather_vars=var_group,
                    )

                    response_json = request_with_retry(session=session, params=params)

                    chunk_df = unpack_single_location_response(
                        response_json=response_json,
                        airport_code=airport_code,
                        latitude=latitude,
                        longitude=longitude,
                        airport_col_name=airport_col_name,
                    )
                    weather_chunks.append(chunk_df)

                    if pause_between_requests_seconds > 0:
                        time.sleep(pause_between_requests_seconds)

    return combine_weather_chunks(
        weather_chunks=weather_chunks,
        airport_col_name=airport_col_name,
        date_col="time",
    )


def fetch_destination_weather(
    flights: pd.DataFrame,
    destination_airport: tuple[str, float, float],
    date_col: str = "FlightDate",
    weather_vars: Sequence[str] = WEATHER_VARS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch destination weather for one airport across the full flight date range."""
    start_date, end_date = get_flight_date_range(flights, date_col=date_col)

    return fetch_weather_for_airports(
        airport_tuples=[destination_airport],
        start_date=start_date,
        end_date=end_date,
        airport_col_name="Dest",
        weather_vars=weather_vars,
        verbose=verbose,
    )


def fetch_origin_weather(
    flights: pd.DataFrame,
    origin_airports: list[tuple[str, float, float]],
    date_col: str = "FlightDate",
    weather_vars: Sequence[str] = WEATHER_VARS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch origin weather for all unique origin airports across the full flight date range."""
    start_date, end_date = get_flight_date_range(flights, date_col=date_col)

    return fetch_weather_for_airports(
        airport_tuples=origin_airports,
        start_date=start_date,
        end_date=end_date,
        airport_col_name="Origin",
        weather_vars=weather_vars,
        verbose=verbose,
    )