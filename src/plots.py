"""Reusable plots saved for the final report."""

from pathlib import Path
from collections.abc import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")


def save_eda_plots(data: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Create and save the core univariate and survival-comparison charts."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    specifications: Iterable[tuple[str, Callable[[Axes], Axes]]] = (
        (
            "survival_count.png",
            lambda ax: sns.countplot(data=data, x="survived", ax=ax),
        ),
        ("age_distribution.png", lambda ax: sns.histplot(data=data, x="age", kde=True, ax=ax)),
        (
            "survival_by_sex.png",
            lambda ax: sns.barplot(data=data, x="sex", y="survived", errorbar=("ci", 95), ax=ax),
        ),
        (
            "survival_by_class.png",
            lambda ax: sns.barplot(data=data, x="pclass", y="survived", errorbar=("ci", 95), ax=ax),
        ),
        (
            "fare_by_survival.png",
            lambda ax: sns.boxplot(data=data, x="survived", y="fare", showfliers=False, ax=ax),
        ),
    )

    titles = {
        "survival_count.png": "Passenger survival counts",
        "age_distribution.png": "Passenger age distribution",
        "survival_by_sex.png": "Survival rate by sex",
        "survival_by_class.png": "Survival rate by passenger class",
        "fare_by_survival.png": "Fare by survival outcome (outliers hidden)",
    }
    labels = {
        "survival_count.png": ("Survived (0 = no, 1 = yes)", "Passengers"),
        "age_distribution.png": ("Age (years)", "Passengers"),
        "survival_by_sex.png": ("Sex", "Survival rate"),
        "survival_by_class.png": ("Passenger class", "Survival rate"),
        "fare_by_survival.png": ("Survived (0 = no, 1 = yes)", "Fare"),
    }

    for filename, draw in specifications:
        fig, ax = plt.subplots(figsize=(8, 5))
        draw(ax)
        ax.set_title(titles[filename])
        ax.set_xlabel(labels[filename][0])
        ax.set_ylabel(labels[filename][1])
        fig.tight_layout()
        path = destination / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    return written
