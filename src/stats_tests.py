"""Hypothesis tests with effect sizes and machine-readable results."""

import math

import pandas as pd
from scipy import stats


def cramers_v(table: pd.DataFrame) -> float:
    """Compute bias-corrected Cramer's V for a contingency table."""
    observed = table.to_numpy(dtype=float)
    chi2 = stats.chi2_contingency(observed, correction=False)[0]
    n = observed.sum()
    rows, columns = observed.shape
    if n <= 1 or min(rows - 1, columns - 1) <= 0:
        return float("nan")

    phi2 = chi2 / n
    correction = ((columns - 1) * (rows - 1)) / (n - 1)
    corrected_phi2 = max(0.0, phi2 - correction)
    corrected_rows = rows - ((rows - 1) ** 2) / (n - 1)
    corrected_columns = columns - ((columns - 1) ** 2) / (n - 1)
    denominator = min(corrected_columns - 1, corrected_rows - 1)
    return math.sqrt(corrected_phi2 / denominator) if denominator > 0 else float("nan")


def chi_square_test(
    data: pd.DataFrame, first: str, second: str, hypothesis: str
) -> dict[str, object]:
    """Test independence and return chi-square, p-value, and Cramer's V."""
    table = pd.crosstab(data[first], data[second])
    statistic, p_value, degrees_freedom, _ = stats.chi2_contingency(table)
    return {
        "hypothesis": hypothesis,
        "test": "Chi-square test of independence",
        "statistic": float(statistic),
        "degrees_freedom": int(degrees_freedom),
        "p_value": float(p_value),
        "effect_size_name": "Cramer's V",
        "effect_size": cramers_v(table),
        "sample_size": int(table.to_numpy().sum()),
    }


def mann_whitney_test(
    data: pd.DataFrame,
    outcome: str,
    group: str,
    first_group: object,
    second_group: object,
    hypothesis: str,
) -> dict[str, object]:
    """Compare two independent groups with Mann-Whitney U and rank-biserial r."""
    first = data.loc[data[group] == first_group, outcome].dropna()
    second = data.loc[data[group] == second_group, outcome].dropna()
    if first.empty or second.empty:
        raise ValueError(f"Both groups must contain observations for {group}.")
    result = stats.mannwhitneyu(first, second, alternative="two-sided")
    effect = (2 * result.statistic / (len(first) * len(second))) - 1
    return {
        "hypothesis": hypothesis,
        "test": "Mann-Whitney U (two-sided)",
        "statistic": float(result.statistic),
        "degrees_freedom": None,
        "p_value": float(result.pvalue),
        "effect_size_name": "Rank-biserial correlation",
        "effect_size": float(effect),
        "sample_size": int(len(first) + len(second)),
    }


def run_hypothesis_tests(data: pd.DataFrame) -> pd.DataFrame:
    """Run four pre-specified tests on the Titanic training dataset."""
    tests = [
        chi_square_test(
            data,
            "sex",
            "survived",
            "Survival outcome is associated with passenger sex.",
        ),
        chi_square_test(
            data,
            "pclass",
            "survived",
            "Survival outcome is associated with passenger class.",
        ),
        mann_whitney_test(
            data,
            "fare",
            "survived",
            1,
            0,
            "Survivors and non-survivors have different fare distributions.",
        ),
        mann_whitney_test(
            data.dropna(subset=["age"]),
            "age",
            "survived",
            1,
            0,
            "Survivors and non-survivors have different age distributions.",
        ),
    ]
    results = pd.DataFrame(tests)
    results["significant_at_0_05"] = results["p_value"] < 0.05
    return results
