"""Dataset loading functions extracted from the notebook."""

from collections.abc import Callable
from pathlib import Path
import pandas as pd
from flight_delay_prediction.config import DATASETS_PATH

def load_flights(datasets_path: str | Path = DATASETS_PATH) -> pd.DataFrame:
    """Load the parquet flight dataset from the intermediate data directory."""
    datasets_path = Path(datasets_path)
    return pd.read_parquet(datasets_path / "flights.parquet")

def load_destination_weather(datasets_path: str | Path = DATASETS_PATH) -> pd.DataFrame:
    """Load the parquet destination weather dataset from the intermediate data directory."""
    datasets_path = Path(datasets_path)
    return pd.read_parquet(datasets_path / "destination_weather.parquet")

def load_origin_weather(datasets_path: str | Path = DATASETS_PATH) -> pd.DataFrame:
    """Load the parquet origin weather dataset from the intermediate data directory."""
    datasets_path = Path(datasets_path)
    return pd.read_parquet(datasets_path / "origin_weather.parquet")

def load_airports(datasets_path: str | Path = DATASETS_PATH) -> pd.DataFrame:
    """Load the parquet supplementary airport lookup dataset from the intermediate data directory."""
    datasets_path = Path(datasets_path)
    return pd.read_parquet(datasets_path / "us_airports.parquet")

def load_airlines(datasets_path: str | Path = DATASETS_PATH) -> pd.DataFrame:
    """Load the parquet supplementary airline lookup dataset from the intermediate data directory."""
    datasets_path = Path(datasets_path)
    return pd.read_parquet(datasets_path / "airline_names.parquet")

def create_origin_lat_long(flights: pd.DataFrame, airports: pd.DataFrame) -> list[tuple[str, float, float]]:
    """Merge the flights and airports datasets to create a list of origin, latitude, and longitude values."""
    joined = pd.merge(flights, airports, left_on = "Origin", right_on = "code", how = "left")
    
    # Manually fill in missing lat/long for Puerto Rico airports
    joined.loc[joined["Origin"] == "SJU", "latitude"] = 18.44
    joined.loc[joined["Origin"] == "SJU", "longitude"] = 66.00

    joined.loc[joined["Origin"] == "STT", "latitude"] = 18.34
    joined.loc[joined["Origin"] == "STT", "longitude"] = 64.97

    joined.loc[joined["Origin"] == "BQN", "latitude"] = 18.50
    joined.loc[joined["Origin"] == "BQN", "longitude"] = 67.14
    
    origin = joined["Origin"].unique()
    lat = [joined[joined["Origin"] == o]["latitude"].iloc[0] for o in origin]
    long = [joined[joined["Origin"] == o]["longitude"].iloc[0] for o in origin]
    
    output = [(o, lt, lg) for (o, lt, lg) in zip(origin, lat, long)]
    
    return output

def load_PCA_train_and_test(datasets_path: str | Path = "../data/processed/PCA") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the PCA-transformed train and test datasets from the processed data directory."""
    datasets_path = Path(datasets_path)
    train = pd.read_parquet(datasets_path / "train.parquet")
    test = pd.read_parquet(datasets_path / "test.parquet")
    
    return train, test