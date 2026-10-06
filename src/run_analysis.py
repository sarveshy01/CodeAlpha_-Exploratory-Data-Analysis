"""Run the reproducible Titanic EDA and write its deliverables."""

import argparse
from pathlib import Path

import pandas as pd

from src.cleaning import clean_data
from src.data_loader import DEFAULT_DATA_PATH, PROJECT_ROOT, load_data
from src.plots import save_eda_plots
from src.stats_tests import run_hypothesis_tests


def _percentage(value: float) -> str:
    return f"{value:.1%}"


def build_report(raw: pd.DataFrame, cleaned: pd.DataFrame, tests: pd.DataFrame) -> str:
    """Create a concise report using the actual loaded dataset and test results."""
    survival = raw.groupby("sex", observed=True)["survived"].mean().sort_values(ascending=False)
    class_survival = raw.groupby("pclass", observed=True)["survived"].mean().sort_index()
    missing = raw.isna().sum()
    missing = missing[missing.gt(0)].sort_values(ascending=False)

    lines = [
        "# Titanic Exploratory Data Analysis",
        "",
        "## Objective",
        "",
        "Explore passenger survival patterns, describe data quality, and test "
        "pre-specified associations. This is descriptive analysis; associations "
        "do not establish causation.",
        "",
        "## Dataset summary",
        "",
        f"- Rows: {len(raw):,}; columns: {len(raw.columns):,}.",
        f"- Overall survival rate: {_percentage(raw['survived'].mean())}.",
        f"- Cleaning preserved all {len(cleaned):,} passenger rows.",
        "- Source: seaborn-data Titanic CSV (see README for source and attribution).",
        "",
        "## Key findings",
        "",
        f"- Survival rate by sex: {', '.join(f'{key}: {_percentage(value)}' for key, value in survival.items())}.",
        f"- Survival rate by passenger class: {', '.join(f'class {key}: {_percentage(value)}' for key, value in class_survival.items())}.",
        "- The plots below show survival counts, age distribution, survival by sex "
        "and class, and fare distributions by outcome.",
        "",
        "### Charts",
        "",
        "![Passenger survival counts](figures/survival_count.png)",
        "",
        "![Passenger age distribution](figures/age_distribution.png)",
        "",
        "![Survival rate by sex](figures/survival_by_sex.png)",
        "",
        "![Survival rate by class](figures/survival_by_class.png)",
        "",
        "![Fare by survival outcome](figures/fare_by_survival.png)",
        "",
        "## Hypothesis tests",
        "",
        "Two-sided tests use α = 0.05. Chi-square tests report Cramer's V; "
        "Mann-Whitney tests report rank-biserial correlation. p-values do not "
        "measure practical importance.",
        "",
        "| Hypothesis | Test | Statistic | p-value | Effect size | Result |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in tests.to_dict(orient="records"):
        effect = (
            f"{row['effect_size']:.3f}"
            if pd.notna(row["effect_size"])
            else "not estimable"
        )
        conclusion = "evidence of a difference/association" if row["significant_at_0_05"] else "insufficient evidence"
        lines.append(
            f"| {row['hypothesis']} | {row['test']} | {row['statistic']:.3f} | "
            f"{row['p_value']:.3g} | {row['effect_size_name']}: {effect} | {conclusion} |"
        )

    lines.extend(
        [
            "",
            "## Data quality and cleaning",
            "",
            "| Column | Missing values | Missing share |",
            "|---|---:|---:|",
        ]
    )
    if missing.empty:
        lines.append("| None | 0 | 0.0% |")
    else:
        for column, count in missing.items():
            lines.append(
                f"| {column} | {int(count)} | {_percentage(count / len(raw))} |"
            )
    duplicate_count = int(raw.duplicated().sum())
    lines.extend(
        [
            "",
            f"- Exact duplicate rows: {duplicate_count}. They are reported, not "
            "automatically dropped, because this dataset has no passenger ID "
            "suitable for proving row identity.",
            "- Age: median imputation for a complete analysis copy.",
            "- Embarkation fields: mode imputation where needed.",
            "- Deck: missing entries represented as `Unknown` because the field "
            "is substantially incomplete.",
            "- Added `family_size = sibsp + parch + 1`; existing variables are preserved.",
            "",
            "## Limitations and next steps",
            "",
            "- The dataset is a historical sample and findings should not be "
            "generalized to other populations.",
            "- Correlated fields (such as `survived`/`alive`, `pclass`/`class`, "
            "and `embarked`/`embark_town`) are redundant for some analyses.",
            "- No causal claims are supported by these observational comparisons.",
            "- For predictive modeling, use a held-out validation design and "
            "fit preprocessing only on training folds.",
            "",
        ]
    )
    return "\n".join(lines)


def run(data_path: str | Path = DEFAULT_DATA_PATH, profile: bool = False) -> None:
    raw = load_data(data_path)
    cleaned = clean_data(raw)
    processed_dir = PROJECT_ROOT / "data" / "processed"
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    report_dir = PROJECT_ROOT / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    cleaned.to_csv(processed_dir / "cleaned_data.csv", index=False)
    results = run_hypothesis_tests(raw)
    results.to_csv(report_dir / "hypothesis_test_results.csv", index=False)
    save_eda_plots(cleaned, figures_dir)
    (report_dir / "final_report.md").write_text(
        build_report(raw, cleaned, results), encoding="utf-8"
    )
    if profile:
        try:
            from ydata_profiling import ProfileReport
        except ImportError as exc:
            raise RuntimeError(
                "Optional profiling requires ydata-profiling; install it with "
                "Python 3.10-3.13 as described in README.md."
            ) from exc
        ProfileReport(raw, explorative=True).to_file(
            report_dir / "profiling_report.html"
        )
    print(f"Analysis complete. Report: {report_dir / 'final_report.md'}")
    print(f"Cleaned data: {processed_dir / 'cleaned_data.csv'}")
    print(f"Hypothesis tests: {report_dir / 'hypothesis_test_results.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to a Titanic CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Generate the optional ydata-profiling HTML report (Python 3.10-3.13).",
    )
    args = parser.parse_args()
    run(args.data, profile=args.profile)


if __name__ == "__main__":
    main()
