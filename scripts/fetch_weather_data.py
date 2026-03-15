"""Script to fetch destination and origin weather data from Open-Meteo."""

from pathlib import Path

from flight_delay_prediction.data.fetch_weather import (
    fetch_destination_weather,
    fetch_origin_weather,
)
from flight_delay_prediction.data.load_data import (
    create_origin_lat_long,
    load_airports,
    load_flights,
)


DESTINATION_WEATHER_OUTPATH = Path("../data/raw/weather/destination_weather_data.csv")
ORIGIN_WEATHER_OUTPATH = Path("../data/raw/weather/origin_weather_data.csv")

# PHL airport code and coordinates
PHL_AIRPORT = ("PHL", 39.8729, -75.2437)


def main() -> None:
    DESTINATION_WEATHER_OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    ORIGIN_WEATHER_OUTPATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading flights and airport lookup data...")
    flights = load_flights()
    airports = load_airports()

    print("Creating origin airport coordinate tuples...")
    origin_airports = create_origin_lat_long(flights=flights, airports=airports)

    print("Fetching destination weather for PHL...")
    destination_weather = fetch_destination_weather(
        flights=flights,
        destination_airport=PHL_AIRPORT,
        date_col="FlightDate",
        verbose=True,
    )
    destination_weather.to_csv(DESTINATION_WEATHER_OUTPATH, index=False)
    print(f"Saved destination weather data to {DESTINATION_WEATHER_OUTPATH}")

    print("Fetching origin weather for all unique origin airports...")
    origin_weather = fetch_origin_weather(
        flights=flights,
        origin_airports=origin_airports,
        date_col="FlightDate",
        verbose=True,
    )
    origin_weather.to_csv(ORIGIN_WEATHER_OUTPATH, index=False)
    print(f"Saved origin weather data to {ORIGIN_WEATHER_OUTPATH}")

    print("Finished fetching weather data.")


if __name__ == "__main__":
    main()