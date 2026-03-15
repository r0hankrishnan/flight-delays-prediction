"""Helper utilities lifted directly from the notebook."""

from typing import List

import pandas as pd


def display_NA_summary(df: pd.DataFrame, only_nonzero: bool = False) -> pd.DataFrame:
    """Count NAs per column and return a summary DataFrame."""
    cols = [col for col in df.columns]
    dtypes = [str(df[col].dtype) for col in cols]
    nas = [df[col].isna().sum() for col in cols]
    pct_nas = [round(na / df.shape[0], 4) * 100 for na in nas]

    nas_dict = {
        "Column": cols,
        "Data Type": dtypes,
        "Number of NA Values": nas,
        "Percent NA": pct_nas,
    }

    na_summary_df = pd.DataFrame(nas_dict)

    if only_nonzero:
        return (
            na_summary_df[na_summary_df["Percent NA"] > 0]
            .sort_values(by="Percent NA", ascending=False)
            .copy()
        )

    return na_summary_df.sort_values(by="Percent NA", ascending=False)



def create_year_stratified_sample(
    df: pd.DataFrame,
    date_col: str,
    years: List[int],
    sample_frac: float,
    drop_subset: List[str] | None = None,
) -> pd.DataFrame:
    """Create a stratified-by-year sample of a DataFrame."""
    sample_list = []

    if df[date_col].dtype != "datetime64[ns]":
        raise TypeError(
            f"Incorrect data type for {date_col}. Must be of type 'datetime64[ns]'\n"
            "Try converting to datetime64[ns] using .to_datetime() method and then try again."
        )

    for year in years:
        sample_df = df[df[date_col].dt.year == year].sample(frac=sample_frac).copy()
        sample_list.append(sample_df)

    if drop_subset:
        concat_df = pd.concat(sample_list)
        return concat_df.drop(columns=drop_subset)

    return pd.concat(sample_list)
