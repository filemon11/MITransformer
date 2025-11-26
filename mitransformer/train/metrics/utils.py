from . import base

import torch
import math
import numbers
import numpy as np

from typing import Iterable, cast, Any


def sum_metrics(metrics: Iterable[base.M]) -> base.M:
    '''This function calculates the sum of metrics in an iterable.

    Parameters
    ----------
    metrics : Iterable[M]
        The `metrics` parameter in the `sum_metrics`
    function is expected to be an iterable containing
    elements of type `M`. The function computes the sum of
    all elements in the iterable and returns the
    result.

    Returns
    -------
        The function `sum_metrics` returns the sum of
    all elements in the input `metrics` iterable.

    '''
    s: None | base.M = None
    for m in metrics:
        if s is None:
            s = m
        else:
            s = cast(base.M, m + s)
    assert s is not None, "Iterable of metrics cannot be empty."
    return s


def sum_and_std_metrics(
        metrics: Iterable[base.Metric]
        ) -> dict[str, tuple[float, float]]:
    '''This function calculates the mean and standard deviation for each
    metric in a given list of Metric
    objects.

    Parameters
    ----------
    metrics : "Iterable[Metric]"
        The `metrics` parameter is expected to be an
        iterable containing objects of type `Metric`.

    Returns
    -------
        The function `sum_and_std_metrics` returns
    a dictionary where each key corresponds to a metric name
    and the value is a tuple containing the mean and standard deviation of
    that metric calculated from
    the input list of Metric objects.
    '''
    ms = list(metrics)
    n = len(ms)
    out_dict: dict[str, tuple[float, float]] = dict()
    means: dict[str, float] = sum_metrics(ms).to_dict()
    for key, mean_value in means.items():
        if check_numeral(mean_value):
            xs = [getattr(m, key) for m in ms]
            out_dict[key] = (
                mean_value,
                math.sqrt(sum([(x-mean_value)**2 for x in xs]) / n))
    return out_dict


def check_type(value: Any) -> bool:
    if (isinstance(value, float)
            or isinstance(value, int)
            or isinstance(value, torch.Tensor)
            or isinstance(value, bool)
            or isinstance(value, str)):
        return True
    return False


def check_numeral(value: Any) -> bool:
    if (isinstance(value, numbers.Number)
            or isinstance(value, torch.Tensor)
            or isinstance(value, np.ndarray)):
        return True
    return False


minimise = {"lm_loss": True,
            "loss": True,
            "arc_loss": True,
            "perplexity": True,
            "uas": False,
            "distance_loss": True,
            "attention_entropy_loss": True
            }


def to_t(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    try:
        return torch.tensor(float(x))
    except Exception:
        return torch.tensor(0.)
