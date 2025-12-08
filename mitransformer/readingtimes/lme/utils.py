import pandas as pd
import numpy as np

from typing import Tuple, Iterable

from ...utils.logmaker import getLogger, info
logger = getLogger(__name__)

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------


def scaling_var(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    return (x - x.mean()) / x.std()


def get_bounds(x: pd.Series) -> Tuple[float, float]:
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def remove_outliers(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    info(0, logger, f"nrows before outlier removal {len(df)}")
    bounds = {c: get_bounds(df[c]) for c in cols}
    mask = np.ones(len(df), dtype=bool)

    for col in cols:
        lo, hi = bounds[col]
        mask &= (df[col] > lo) & (df[col] < hi)

    df = df.loc[mask]
    info(0, logger, f"nrows after outlier removal {len(df)}")
    return df
