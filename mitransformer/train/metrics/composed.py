from . import base, field

import torch

from typing import Sequence, Any

# ---------------------- concrete metric classes ---------------------------

_dynamic_weighted_metric_cache: dict[tuple[str, ...], Any] = {}
_dynamic_weighted_eval_metric_cache: dict[tuple[str, ...], Any] = {}
# NOTE: this is a stupid way of implementing it and should be
# changed at some point...


class LMMetric(base.Metric):
    fields = {
        **base.Metric.fields,
        "lm_loss": field.loss("num")
    }


class SupervisedMetric(LMMetric):
    fields = {
        **LMMetric.fields,
        "arc_num": field.num,
        "arc_loss": field.loss("arc_num"),
        "alpha": field.MetricField(None, static=True),
    }

    def _compute_loss(self) -> torch.Tensor:
        lm_loss = getattr(self, "lm_loss")
        arc_loss = getattr(self, "arc_loss")
        alpha = getattr(self, "alpha")

        return (alpha*lm_loss + (1-alpha)*arc_loss)


class EvalMetric(LMMetric):
    fields = {
        **LMMetric.fields,
        "perplexity": field.perplexity,
    }


class SupervisedEvalMetric(SupervisedMetric, EvalMetric):
    # merge field maps carefully: rightmost wins
    fields: dict[str, field.MetricField] = {
        **SupervisedMetric.fields, **EvalMetric.fields,
        "uas": field.uas,
        "att_entropy": field.att_entropy,
    }


class CostsMetric(LMMetric, base.WeightedMetric):
    fields = {
        **LMMetric.fields,
        **base.WeightedMetric.fields,
        "attention_entropy_loss": field.attention_entropy_loss,
        "distance_loss": field.distance_loss
    }


class CostsEvalMetric(CostsMetric, EvalMetric):
    fields: dict[str, field.MetricField] = {
        **CostsMetric.fields, **EvalMetric.fields,
    }


def DynamicWeightedMetric(loss_names: Sequence):
    key = tuple(loss_names)
    if key in _dynamic_weighted_metric_cache:
        return _dynamic_weighted_metric_cache[key]

    new_class = type(
        f"DynamicWeightedMetric_{'_'.join(key)}",
        (base.WeightedMetric,),
        {"fields": {
            **base.WeightedMetric.fields,
            **{name: field.loss("num") for name in key}
        }}
    )
    _dynamic_weighted_metric_cache[key] = new_class
    return new_class


def DynamicWeightedEvalMetric(loss_names: Sequence):
    key = tuple(loss_names)
    if key in _dynamic_weighted_eval_metric_cache:
        return _dynamic_weighted_eval_metric_cache[key]

    TrainMetric = DynamicWeightedMetric(loss_names)
    new_class = type(
        f"DynamicWeightedEvalMetric_{'_'.join(key)}",
        (TrainMetric, EvalMetric),
        {"fields": {
            **TrainMetric.fields,
            **EvalMetric.fields}}
    )
    _dynamic_weighted_eval_metric_cache[key] = new_class
    return new_class
