import pandas as pd
from scipy.stats import pearsonr  # type: ignore
import numpy as np

from typing import Iterable, Tuple

# ------------------------------------------------------------------
# Correlation computation
# ------------------------------------------------------------------


def compute_pearson(
        datasets: Iterable[pd.DataFrame],
        col1: str,
        col2: str,
        group_by: Iterable[str] | None = ("item", "zone")
        ) -> Tuple[float, float]:

    corrs = []

    for df in datasets:
        if group_by is not None:
            means = (
                df.groupby(list(group_by))[[col1, col2]]
                .mean()
                .dropna()
            )
        else:
            means = df

        r, _ = pearsonr(means[col1], means[col2])
        corrs.append(r)

    return (np.mean(corrs).item(), np.std(corrs).item())
