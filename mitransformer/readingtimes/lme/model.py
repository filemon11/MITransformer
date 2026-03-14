import numpy as np
import pandas as pd
import gpboost as gpb  # type: ignore

from . import model_debug

from typing import Iterable, Tuple, Mapping, Literal

from ...utils.logmaker import getLogger
logger = getLogger(__name__)


# ------------------------------------------------------------------
# GPBoost core functions (replaces lmer + anova)
# ------------------------------------------------------------------

def build_X(df: pd.DataFrame, predictors: Iterable[str]) -> np.ndarray:
    X = df[predictors].values
    intercept = np.ones((X.shape[0], 1))
    return np.hstack([intercept, X])


def fit_gpboost(
        df: pd.DataFrame, y_col: str, predictors: Iterable[str],
        random_effects: Mapping[
            str, Iterable[str | Literal[0] | Literal[1]]] | None = None,
        debug: bool = False,
        device_type: Literal["cpu", "cuda"] = "cpu",
        ) -> Tuple[gpb.GPModel, pd.DataFrame]:
    """
    'random_effects': is a mapping from grouping variables to covariates.
        A random intercept is added for each random effect, except if '0' is
        provided as a covariate. You can be explicit and provide '1' but
        '0' overrides '1'.
    """
    predictors = list(predictors)
    group_vars: list[str] | None = None
    if random_effects is not None and len(random_effects) > 0:
        group_vars = list(random_effects.keys())

    df = df[
        [y_col, *predictors, *(group_vars if group_vars is not None else [])]
        ].dropna().reset_index(drop=True)

    group: None | np.ndarray = None
    drop_rand_intr: list[bool] | None = None
    Z = None
    pointers: None | list[int] = None
    # pointers to the column in 'group'

    if random_effects is not None and group_vars is not None:

        pointers = []
        flat_names: list[str] = []
        drop_rand_intr = []
        for idx, covs in enumerate(random_effects.values(), start=1):
            covs_list = list(covs)
            drop_rand_intr.append(0 in covs_list)
            filtered_list: list[str] = [
                c for c in covs_list if not (c == 0 or c == 1)]
            pointers.extend([idx]*len(filtered_list))
            flat_names.extend(filtered_list)

        if len(flat_names) > 0:
            Z = df[flat_names].values

        group, _ = encode_groups(df, group_vars)

    X = build_X(df, predictors)
    y = df[y_col].values

    if group_vars is None:
        group = np.zeros((len(y), 1))
        drop_rand_intr = [True]

    if drop_rand_intr is not None and not any(drop_rand_intr):
        drop_rand_intr = None

    if debug:
        print("group_vars:", group_vars)
        if group is not None:
            print("group_data shape:", group.shape)
        print("Z shape:", None if Z is None else Z.shape)
        print("pointers:", pointers)
        print("drop_rand_intr:", drop_rand_intr)

        print("y dtype:", y.dtype, "finite:", np.isfinite(y).all())
        print(
            "X dtype:", X.dtype, "shape:",
            X.shape, "finite:", np.isfinite(X).all())
        if group is not None:
            print("group dtype:", group.dtype, "shape:", group.shape)
        if Z is not None:
            print(
                "Z dtype:", Z.dtype, "shape:",
                Z.shape, "finite:", np.isfinite(Z).all())

        model_debug.debug_gpboost_structure(
            df=df,
            group_vars=group_vars,
            Z=Z,
            pointers=pointers,
            random_effects=random_effects,
            print_report=True,
        )

    if Z is not None:
        Z = Z.astype(np.float64)

    print("device", device_type)
    model = gpb.GPModel(
        group_data=group,
        group_rand_coef_data=Z,
        ind_effect_group_rand_coef=pointers,
        drop_intercept_group_rand_effect=drop_rand_intr,
        likelihood="gaussian",
        gp_approx="vecchia",
        GPU_use=device_type == "cuda",
    )

    # Vecchia approximations tested:
    # only miniscule decreases in accuracy

    model.fit(y=y.astype(np.float64), X=X.astype(np.float64), params={
        "trace": debug})

    return model, df

# TODO: the coefficient names and random effect names are not
# written to the output. The current anonymous names are hard
# to keep track of.


def encode_groups(
        df: pd.DataFrame, group_vars: list[str]
        ) -> tuple[np.ndarray, dict[str, dict]]:
    encoders = {}
    cols = []
    for gv in group_vars:
        codes, uniques = pd.factorize(df[gv], sort=True)
        if (codes < 0).any():
            raise ValueError(f"Grouping var {gv} contains NaN after dropna")
        cols.append(codes.astype(np.int32))
        encoders[gv] = {"uniques": uniques}
    group = np.column_stack(cols)
    return group, encoders
