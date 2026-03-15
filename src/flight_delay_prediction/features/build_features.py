"""Feature-building logic extracted from the notebook."""

import math

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder

from flight_delay_prediction.config import (
    MODELING_COLUMNS_TO_DROP,
    NUMERIC_COLS,
    OH_ENCODING_COLS,
    SEED,
    TARGET_COLUMN,
    TARGET_ENCODING_COLS,
)



def extract_time_of_day(CRSTime: int) -> str:
    """Map HHMM scheduled times to a time-of-day bucket."""
    hour = CRSTime // 100

    if 5 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 21:
        return "Evening"
    return "Night"



def cycle_transform(value: int, day: bool = True, sin: bool = True) -> float:
    """Apply the notebook's cyclic transformation."""
    if day and sin:
        return math.sin(2 * math.pi * value / 31)
    if day and not sin:
        return math.cos(2 * math.pi * value / 31)
    if not day and sin:
        return math.sin(2 * math.pi * value / 12)
    return math.cos(2 * math.pi * value / 12)



def build_model_dataframe(final_cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Create the modeling dataframe used before train/test split."""
    model_df = final_cleaned_df.copy()

    model_df["SinMonth"] = model_df["Month"].apply(
        lambda x: cycle_transform(value=x, day=False, sin=True)
    )
    model_df["CosMonth"] = model_df["Month"].apply(
        lambda x: cycle_transform(value=x, day=False, sin=False)
    )
    model_df["SinDay"] = model_df["DayOfWeek"].apply(
        lambda x: cycle_transform(value=x, day=True, sin=True)
    )
    model_df["CosDay"] = model_df["DayOfWeek"].apply(
        lambda x: cycle_transform(value=x, day=True, sin=False)
    )

    cols_to_drop = [col for col in MODELING_COLUMNS_TO_DROP if col in model_df.columns]
    
    model_df = model_df.drop(columns=cols_to_drop)
    return model_df



def get_modeling_metadata() -> dict[str, object]:
    """Return the column groupings defined in the notebook."""
    return {
        "oh_encoding_cols": OH_ENCODING_COLS,
        "target_encoding_cols": TARGET_ENCODING_COLS,
        "target": TARGET_COLUMN,
        "numeric_cols": NUMERIC_COLS,
    }



def train_test_split_model_data(
    model_df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    test_size: float = 0.30,
    random_state: int = SEED,
):
    """Apply the notebook's train/test split configuration."""
    X_train, X_test, y_train, y_test = train_test_split(
        model_df.drop(columns=[target]),
        model_df[target],
        test_size=test_size,
        random_state=random_state,
        stratify=model_df[target],
    )
    return X_train, X_test, y_train, y_test



def build_preprocessor(oh_encoding_cols, target_encoding_cols):
    return ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(drop="first"), oh_encoding_cols),
            ("te", TargetEncoder(smoothing=10), target_encoding_cols),
        ],
        remainder="passthrough",
    )
