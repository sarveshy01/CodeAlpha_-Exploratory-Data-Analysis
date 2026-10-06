import pandas as pd
import pytest

from src.cleaning import clean_data
from src.data_loader import load_data
from src.stats_tests import run_hypothesis_tests


def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "survived": [0, 1, 0, 1, 0, 1, 0, 1],
            "pclass": [3, 1, 3, 1, 2, 1, 3, 2],
            "sex": ["male", "female", "male", "female", "male", "female", "male", "female"],
            "age": [22.0, 38.0, None, 35.0, 30.0, 25.0, 40.0, 18.0],
            "sibsp": [1, 1, 0, 1, 0, 0, 2, 0],
            "parch": [0, 0, 0, 0, 0, 1, 0, 0],
            "fare": [7.25, 71.28, 7.93, 53.1, 13.0, 30.0, 8.0, 15.0],
            "embarked": ["S", "C", "S", "S", "Q", "C", "S", "Q"],
            "deck": [None, "C", None, "B", None, "D", None, "E"],
        }
    )


def test_cleaning_preserves_input_and_rows() -> None:
    original = sample_data()
    cleaned = clean_data(original)

    assert len(cleaned) == len(original)
    assert original["age"].isna().sum() == 1
    assert cleaned["age"].isna().sum() == 0
    assert cleaned["deck"].eq("Unknown").sum() == 4
    assert cleaned["family_size"].tolist() == [2, 2, 1, 2, 1, 2, 3, 1]


def test_hypothesis_tests_return_four_valid_results() -> None:
    results = run_hypothesis_tests(sample_data())

    assert len(results) == 4
    assert results["p_value"].between(0, 1).all()
    assert results["effect_size"].notna().all()


def test_loader_rejects_missing_required_columns(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"survived": [0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required"):
        load_data(path)
