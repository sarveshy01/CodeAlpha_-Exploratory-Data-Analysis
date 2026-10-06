# Titanic Exploratory Data Analysis

## Objective

Explore passenger survival patterns, describe data quality, and test pre-specified associations. This is descriptive analysis; associations do not establish causation.

## Dataset summary

- Rows: 891; columns: 15.
- Overall survival rate: 38.4%.
- Cleaning preserved all 891 passenger rows.
- Source: seaborn-data Titanic CSV (see README for source and attribution).

## Key findings

- Survival rate by sex: female: 74.2%, male: 18.9%.
- Survival rate by passenger class: class 1: 63.0%, class 2: 47.3%, class 3: 24.2%.
- The plots below show survival counts, age distribution, survival by sex and class, and fare distributions by outcome.

### Charts

![Passenger survival counts](figures/survival_count.png)

![Passenger age distribution](figures/age_distribution.png)

![Survival rate by sex](figures/survival_by_sex.png)

![Survival rate by class](figures/survival_by_class.png)

![Fare by survival outcome](figures/fare_by_survival.png)

## Hypothesis tests

Two-sided tests use α = 0.05. Chi-square tests report Cramer's V; Mann-Whitney tests report rank-biserial correlation. p-values do not measure practical importance.

| Hypothesis | Test | Statistic | p-value | Effect size | Result |
|---|---|---:|---:|---:|---|
| Survival outcome is associated with passenger sex. | Chi-square test of independence | 260.717 | 1.2e-58 | Cramer's V: 0.543 | evidence of a difference/association |
| Survival outcome is associated with passenger class. | Chi-square test of independence | 102.889 | 4.55e-23 | Cramer's V: 0.337 | evidence of a difference/association |
| Survivors and non-survivors have different fare distributions. | Mann-Whitney U (two-sided) | 129951.500 | 4.55e-22 | Rank-biserial correlation: 0.384 | evidence of a difference/association |
| Survivors and non-survivors have different age distributions. | Mann-Whitney U (two-sided) | 57682.000 | 0.16 | Rank-biserial correlation: -0.062 | insufficient evidence |

## Data quality and cleaning

| Column | Missing values | Missing share |
|---|---:|---:|
| deck | 688 | 77.2% |
| age | 177 | 19.9% |
| embarked | 2 | 0.2% |
| embark_town | 2 | 0.2% |

- Exact duplicate rows: 107. They are reported, not automatically dropped, because this dataset has no passenger ID suitable for proving row identity.
- Age: median imputation for a complete analysis copy.
- Embarkation fields: mode imputation where needed.
- Deck: missing entries represented as `Unknown` because the field is substantially incomplete.
- Added `family_size = sibsp + parch + 1`; existing variables are preserved.

## Limitations and next steps

- The dataset is a historical sample and findings should not be generalized to other populations.
- Correlated fields (such as `survived`/`alive`, `pclass`/`class`, and `embarked`/`embark_town`) are redundant for some analyses.
- No causal claims are supported by these observational comparisons.
- For predictive modeling, use a held-out validation design and fit preprocessing only on training folds.
