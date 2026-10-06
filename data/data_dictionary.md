# Titanic data dictionary

Source: [`data/raw/titanic.csv`](raw/titanic.csv), downloaded from the seaborn-data public repository (see [README.md](../README.md)). The source contains 891 passenger records and 15 columns. `data/raw/` is treated as immutable input.

| Column | Type | Description |
|---|---|---|
| `survived` | Binary categorical (0/1) | Survival outcome; 0 = did not survive, 1 = survived. |
| `pclass` | Ordinal categorical (1-3) | Passenger ticket class. |
| `sex` | Nominal categorical | Recorded passenger sex. |
| `age` | Continuous numeric | Age in years; some values are missing. |
| `sibsp` | Discrete numeric | Number of siblings/spouses aboard. |
| `parch` | Discrete numeric | Number of parents/children aboard. |
| `fare` | Continuous numeric | Passenger fare. |
| `embarked` | Nominal categorical | Port of embarkation code (C, Q, S). |
| `class` | Ordinal categorical | Text label corresponding to `pclass`. |
| `who` | Nominal categorical | Passenger grouping in the source data (man, woman, child). |
| `adult_male` | Binary categorical | Source-provided adult-male indicator. |
| `deck` | Nominal categorical | Cabin deck; mostly unavailable. |
| `embark_town` | Nominal categorical | Port name corresponding to `embarked`. |
| `alive` | Binary categorical | Text form of `survived`; redundant outcome field. |
| `alone` | Binary categorical | Whether the passenger traveled without recorded family. |

Derived during cleaning:

| Column | Type | Description |
|---|---|---|
| `family_size` | Discrete numeric | `sibsp + parch + 1`, including the passenger. |
