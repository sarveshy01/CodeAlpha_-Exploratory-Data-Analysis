"""Transparent, analysis-friendly cleaning for the Titanic data."""

import pandas as pd


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy, preserving rows and recording missing categories.

    Age and embarkation are imputed with their median/mode. Deck is mostly
    unrecorded, so its missing values are kept as an explicit ``Unknown`` group
    instead of being dropped. The input frame is never mutated.
    """
    cleaned = data.copy()
    if "age" in cleaned and cleaned["age"].isna().any():
        median_age = cleaned["age"].median()
        if pd.isna(median_age):
            raise ValueError("Cannot impute age: the age column contains no values.")
        cleaned["age"] = cleaned["age"].fillna(median_age)

    for column in ("embarked", "embark_town"):
        if column in cleaned and cleaned[column].isna().any():
            modes = cleaned[column].mode(dropna=True)
            if modes.empty:
                raise ValueError(
                    f"Cannot impute {column}: the column contains no values."
                )
            cleaned[column] = cleaned[column].fillna(modes.iloc[0])

    if "deck" in cleaned:
        cleaned["deck"] = cleaned["deck"].fillna("Unknown")

    if {"sibsp", "parch"}.issubset(cleaned.columns):
        cleaned["family_size"] = cleaned["sibsp"] + cleaned["parch"] + 1
    if "alone" not in cleaned and "family_size" in cleaned:
        cleaned["alone"] = cleaned["family_size"].eq(1)

    return cleaned
