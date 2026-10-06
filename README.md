# Titanic Exploratory Data Analysis

A complete, reproducible beginner-level EDA project: six guided notebooks, reusable Python modules, hypothesis tests with effect sizes, generated visualizations, a cleaned analysis copy, automated tests, and a data-driven report.

## Dataset

This project uses the Titanic CSV from the [seaborn-data repository](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv). The downloaded source file is stored in `data/raw/titanic.csv`; it is never edited by the analysis. The source has 891 passenger records and 15 fields. Field definitions and data types are in [`data/data_dictionary.md`](data/data_dictionary.md).

The dataset is used for educational descriptive analysis. It is not a causal study, and the patterns should not be generalized beyond this historical dataset.

## Setup

Python 3.10 or newer is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run the commands with `.venv\Scripts\python.exe` instead.

## Run the complete analysis

From the project root:

```powershell
python -m src.run_analysis
```

This writes:

- `data/processed/cleaned_data.csv` — analysis-ready copy; raw data is unchanged.
- `reports/figures/*.png` — core EDA charts.
- `reports/hypothesis_test_results.csv` — four test results and effect sizes.
- `reports/final_report.md` — generated summary with actual statistics and charts.

Use another compatible Titanic CSV with `python -m src.run_analysis --data path\to\titanic.csv`.

To generate the optional automated profiling report, install `ydata-profiling` on Python 3.10-3.13 and run `python -m src.run_analysis --profile`. The profiler currently does not support Python 3.14; the core analysis and notebooks do not require it.

## Guided notebooks

Launch JupyterLab from the project root:

```powershell
jupyter lab
```

Run notebooks in order:

1. `notebooks/01_questions_and_overview.ipynb` — analysis goal, questions, hypotheses, dataset overview.
2. `notebooks/02_data_structure.ipynb` — dimensions, dtypes, missingness, duplicates, and schema.
3. `notebooks/03_univariate_bivariate_analysis.ipynb` — distributions and comparisons.
4. `notebooks/04_trends_patterns_anomalies.ipynb` — group patterns, outliers, and logical checks.
5. `notebooks/05_hypothesis_testing.ipynb` — four statistical tests, effect sizes, and interpretations.
6. `notebooks/06_data_quality_issues.ipynb` — quality log and explicit cleaning decisions.

The notebooks use project modules and relative project paths; launch Jupyter from this directory.

## Statistical plan

The analysis uses two chi-square tests (sex and passenger class against survival) and two two-sided Mann-Whitney U tests (fare and age distributions by survival). It reports p-values and effect sizes (bias-corrected Cramer's V or rank-biserial correlation) at α = 0.05. Statistical significance is not equivalent to practical significance. The notebook explains the test assumptions and limitations; plots accompany the comparisons.

## Cleaning decisions and limitations

- Median-impute age and mode-impute embarkation where needed for a complete analysis copy.
- Preserve unknown deck values as `Unknown` rather than discarding passengers.
- Add `family_size`; preserve all input rows and report exact duplicates rather than dropping records without a reliable passenger identifier.
- Fields such as `survived`/`alive`, `pclass`/`class`, and `embarked`/`embark_town` are redundant and should not be treated as independent predictors.
- No missing cabin/deck values are invented; using `Unknown` does not imply that those passengers shared a known deck.

## Tests

```powershell
python -m pytest
```

## Project layout

```text
data/             Raw input, processed output, and column dictionary
notebooks/        Six ordered, executable Jupyter notebooks
reports/          Generated report, statistical results, and chart files
src/              Data loading, cleaning, plots, tests, and CLI pipeline
tests/            Focused unit tests for cleaning, loading, and statistics
```
