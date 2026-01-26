import numpy as np
import pandas as pd
import gpboost as gpb  # type: ignore
import scipy.stats as stats  # type: ignore


from . import model

from typing import (
    Iterable, Sequence, TypedDict, Callable,
    Literal, Mapping)

from ...utils.logmaker import getLogger
logger = getLogger(__name__)


class ModelProps(TypedDict):
    negloglik: float
    negloglik_per_row: float
    aic: float
    aic_per_row: float
    bic: float
    bic_per_row: float
    coef: pd.DataFrame
    cov_pars: pd.DataFrame


class ComparisonProps(TypedDict):
    delta_negloglik: float
    delta_negloglik_per_row: float
    delta_aic: float
    delta_aic_per_row: float
    delta_bic: float
    delta_bic_per_row: float
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

def get_model_props(
        model: gpb.GPModel, n_observations: int) -> ModelProps:
    """Requires fitted model."""
    coef: pd.DataFrame = model.get_coef()  # type: ignore
    cov_pars: pd.DataFrame = model.get_cov_pars()  # type: ignore

    negloglik = model.get_current_neg_log_likelihood()
    print("negloglik", negloglik)
    assert negloglik is not None
    loglik = -float(negloglik)

    k_fixed = int(coef.shape[1])
    k_random = int(cov_pars.shape[1])
    k = k_fixed + k_random

    aic = 2 * k - 2 * loglik
    bic = np.log(n_observations) * k - 2 * loglik

    cov_pars.loc[1] = np.sqrt(cov_pars.iloc[0])
    cov_pars.index = ["Variance", "Std.Dev."]  # type: ignore
    return {
        "negloglik": negloglik,
        "negloglik_per_row": negloglik / n_observations,
        "aic": aic,
        "aic_per_row": aic / n_observations,
        "bic": bic,
        "bic_per_row": bic / n_observations,
        "coef": coef,
        "cov_pars": cov_pars
    }


def get_model_comparison(
        model0: gpb.GPModel, model1: gpb.GPModel,
        n_observations: int
        ) -> ComparisonProps:
    props0 = get_model_props(model0, n_observations)
    props1 = get_model_props(model1, n_observations)

    delta_negloglik = props1["negloglik"] - props0["negloglik"]
    delta_negloglik_per_row = (
        props1["negloglik_per_row"] - props0["negloglik_per_row"])
    delta_aic = props1["aic"] - props0["aic"]
    delta_aic_per_row = props1["aic_per_row"] - props0["aic_per_row"]
    delta_bic = props1["bic"] - props0["bic"]
    delta_bic_per_row = props1["bic_per_row"] - props0["bic_per_row"]

    lr_stat = 2 * (props1["negloglik"] - props0["negloglik"])

    dof = int(props1["coef"].shape[1] - props0["coef"].shape[1])
    # difference in number of parameters

    p_value = stats.chi2.sf(lr_stat, dof).item()

    return {
        "delta_negloglik": delta_negloglik,
        "delta_negloglik_per_row": delta_negloglik_per_row,
        "delta_aic": delta_aic,
        "delta_aic_per_row": delta_aic_per_row,
        "delta_bic": delta_bic,
        "delta_bic_per_row": delta_bic_per_row,
        "lr_stat": lr_stat,
        "dof": dof,
        "p_value": p_value,
        "props0": props0,
        "props1": props1,
    }


def _get_aggr[K, V](
        in_dicts: Sequence[dict[K, V]],
        aggr_func: Callable[[Sequence[V]], V]) -> dict[K, V]:
    out_dict: dict[K, V] = {}
    for key, first_item in in_dicts[0].items():
        if isinstance(first_item, dict):
            out_dict[key] = _get_aggr(  # type: ignore
                [d[key] for d in in_dicts],  # type: ignore
                aggr_func)
        else:
            out_dict[key] = (
                aggr_func([d[key] for d in in_dicts]))  # type: ignore
    return out_dict


def get_aggregate(
        comparisons: Sequence[ComparisonProps]) -> AggregateComparison:

    def mean[V](values: Sequence[V]) -> V:
        if isinstance(values[0], pd.DataFrame):
            return sum(values)/len(values)  # type: ignore
        else:
            return values[0].__class__(np.mean(values))  # type: ignore

    def std[V](values: Sequence[V]) -> V:
        if isinstance(values[0], pd.DataFrame):
            variance = (
                (sum([v-mean(values) for v in values])**2)  # type: ignore
                / len(values))  # type: ignore
            return variance**(1/2)
        else:
            return values[0].__class__(np.std(values))  # type: ignore

    return {
        "mean": _get_aggr(comparisons, mean),  # type: ignore
        "std": _get_aggr(comparisons, std),  # type: ignore
    }


def compute_aggregate_comparison(
        datasets: Iterable[pd.DataFrame],
        to_predict: str,
        predict_from: Iterable[str],
        baseline: Sequence[str],
        random_effects: Mapping[
            str, Sequence[str | Literal[0, 1]]] | None = None,
        random_effects_predict_from:  Mapping[
            str, Sequence[str | Literal[0, 1]]] | None = None,
        ) -> AggregateComparison:

    comparisons: list[ComparisonProps] = []

    for df in datasets:
        n = len(df)
        # baseline
        m0, d0 = model.fit_gpboost(
            df, to_predict, baseline, random_effects)

        if (
                random_effects_predict_from is not None
                and random_effects is not None):
            random_effects = {**random_effects, **random_effects_predict_from}
        # full
        m1, d1 = model.fit_gpboost(
            df, to_predict, list(baseline) + list(predict_from),
            random_effects)

        assert len(d1) == n

        comparisons.append(get_model_comparison(m0, m1, n))

    return get_aggregate(comparisons)


def model_props_to_str(
        model_props: ModelProps,
        coefficient_names: Iterable[str] | None = None,
        random_coefficient_names: Iterable[str] | None = None,
        ) -> str:
    string = "Measures:\n"
    df = pd.DataFrame({
        "negloglik": [model_props["negloglik"]],
        "aic": [model_props["aic"]],
        "bic": [model_props["bic"]]})
    string += df.to_string()

    string += "\nMeasures per row:\n"
    df = pd.DataFrame({
        "negloglik": [model_props["negloglik_per_row"]],
        "aic": [model_props["aic_per_row"]],
        "bic": [model_props["bic_per_row"]]})
    string += df.to_string()
    del df

    coef = model_props["coef"]
    if coefficient_names is not None:
        coef = coef.copy()
        coef = coef.set_axis(coefficient_names, axis=1)

    string += "\nFixed effects:\n"
    string += coef.to_string()

    cov_pars = model_props["cov_pars"]
    if random_coefficient_names is not None:
        cov_pars = cov_pars.copy()
        cov_pars = cov_pars.set_axis(random_coefficient_names, axis=1)

    string += "\nRandom effects:\n"
    string += cov_pars.to_string()
    return string
