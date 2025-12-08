import pandas as pd

from . import correlation, spill, utils, statistics

from typing import (
    TypedDict, List, Literal)

from ...utils.logmaker import getLogger, info
logger = getLogger(__name__)


class ModelProps(TypedDict):
    negloglik: float
    negloglik_per_row: float
    aic: float
    aic_per_row: float
    coef: pd.DataFrame
    cov_pars: pd.DataFrame


class ComparisonProps(TypedDict):
    delta_negloglik: float
    delta_negloglik_per_row: float
    delta_aic: float
    delta_aic_per_row: float
    lr_stat: float
    dof: float
    p_value: float
    props0: ModelProps
    props1: ModelProps


class AggregateComparison(TypedDict):
    mean: ComparisonProps
    std: ComparisonProps


# ------------------------------------------------------------------
# GPBoost core functions (replaces lmer + anova)
# ------------------------------------------------------------------

def run(base_name, num_models, corpus, spillover, directory, additional_name):
    # ------------------------------------------------------------------
    # corpus type
    # ------------------------------------------------------------------

    corpus_type: Literal["ET", "SP"] = (
        "SP" if corpus in {"frank_SP", "naturalstories"} else "ET"
    )

    # ------------------------------------------------------------------
    # goals and predictors
    # ------------------------------------------------------------------

    if corpus_type == "SP":
        goals = ["RT"]
    else:
        goals = ["FFD", "GPT", "GD"]

    candidates = ["surprisal", "demberg", "first_dependent_distance"]
    baseline_predictors = ["frequency", "length"]

    data_dir = f"{directory}/{corpus}_{additional_name}_preprocessed_"

    if num_models == 0:
        num_models = 1

    datasets: List[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------

    excluded_vars = goals + ["WorkerId", "item"]

    for x in range(num_models):
        if num_models == 0:
            path = f"{data_dir}{base_name}.csv"
        else:
            path = f"{data_dir}{base_name}_{x}.csv"

        data = pd.read_csv(path)

        numeric_cols = data.select_dtypes(include="number").columns
        cols_to_scale = [c for c in numeric_cols if c not in excluded_vars]

        data = utils.remove_outliers(data, goals + baseline_predictors)

        for col in cols_to_scale:
            data[col] = utils.scaling_var(data[col])

        datasets.append(data)

    # ------------------------------------------------------------------
    # Main loops (translated 1:1)
    # ------------------------------------------------------------------

    for goal in goals:
        for candidate in candidates:

            info(0, logger, f"\nCorrelation between {goal} and {candidate}")
            info(0, logger, str(
                correlation.compute_pearson(datasets, goal, candidate)))

            # Spillover improvement
            if spillover > 0:
                predict_from = spill.get_spillover_upto(
                    [candidate] + baseline_predictors,
                    spillover - 1
                )

                info(
                    0, logger,
                    f"\nDeltaLogLik improvement – {goal} & {candidate}, "
                    f"spillover {spillover}")
                info(
                    0, logger,
                    str(statistics.compute_aggregate_comparison(
                        datasets,
                        goal,
                        spill.get_spillover(
                            [candidate] + baseline_predictors, spillover),
                        predict_from,
                        {"WorkerId": [1]}
                    ))
                )

            info(0, logger, f"\nDeltaLogLik – {goal} & {candidate} (overall)")
            info(
                0, logger,
                str(statistics.compute_aggregate_comparison(
                    datasets,
                    goal,
                    spill.get_spillover_upto([candidate], spillover),
                    spill.get_spillover_upto(baseline_predictors, spillover),
                    {"WorkerId": [1]}
                ))
            )

            for candidate2 in candidates:
                if candidate != candidate2:

                    if spillover > 0:
                        predict_from = spill.get_spillover_upto(
                            [candidate, candidate2] + baseline_predictors,
                            spillover - 1
                        )

                        info(
                            0, logger,
                            f"\nDeltaLogLik – {goal}: {candidate}"
                            f"+{candidate2}, "
                            "spillover {spillover}")
                        info(
                            0, logger,
                            str(statistics.compute_aggregate_comparison(
                                datasets,
                                goal,
                                spill.get_spillover(
                                    [candidate, candidate2]
                                    + baseline_predictors, spillover),
                                predict_from,
                                {"WorkerId": [1]}
                            ))
                        )

                    info(
                        0, logger,
                        f"\nDeltaLogLik – {goal}: {candidate2} "
                        f"over {candidate}")
                    info(
                        0, logger,
                        str(statistics.compute_aggregate_comparison(
                            datasets,
                            goal,
                            spill.get_spillover_upto([candidate2], spillover),
                            spill.get_spillover_upto(
                                baseline_predictors + [candidate], spillover),
                            {"WorkerId": [1]}
                        ))
                    )

        for candidate1 in candidates:
            for candidate2 in candidates:
                if candidate1 != candidate2:

                    if spillover > 0:
                        predict_from = spill.get_spillover_upto(
                            [candidate1, candidate2] + baseline_predictors,
                            spillover - 1
                        )

                        info(
                            0, logger,
                            f"\nDeltaLogLik – {goal}: "
                            f"{candidate1}+{candidate2}, "
                            f"spillover {spillover}")
                        info(
                            0, logger,
                            str(statistics.compute_aggregate_comparison(
                                datasets,
                                goal,
                                spill.get_spillover(
                                    [candidate1, candidate2]
                                    + baseline_predictors,
                                    spillover),
                                predict_from,
                                {"WorkerId": [1]}
                            ))
                        )

                    info(
                        0, logger,
                        f"\nDeltaLogLik – {goal}: {candidate1}"
                        f"+{candidate2} (overall)")
                    info(
                        0, logger,
                        str(statistics.compute_aggregate_comparison(
                            datasets,
                            goal,
                            spill.get_spillover_upto(
                                [candidate1, candidate2], spillover),
                            spill.get_spillover_upto(
                                baseline_predictors, spillover),
                            {"WorkerId": [1]}
                        ))
                    )
