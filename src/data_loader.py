"""Loading and basic validation for the Titanic dataset."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "titanic.csv"
REQUIRED_COLUMNS = {
    "survived",
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
}


def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the raw Titanic CSV and check for the fields used by this project."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Titanic CSV not found at {csv_path}. See README.md for the dataset source."
        )

    frame = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            "Dataset is missing required Titanic columns: "
            + ", ".join(sorted(missing))
        )
    if frame.empty:
        raise ValueError(f"Titanic CSV is empty: {csv_path}")
    return frame
